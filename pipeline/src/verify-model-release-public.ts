/** Recheck approved public bytes before release mutation; private evidence is reviewed separately. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { parseReleaseManifest, RELEASE_BUNDLE_LIMITS } from "../../web/src/artifacts.js";
import { verifyPriorPlanApproval } from "./core/model-evaluation-plan-approval.js";
import { MODEL_OUTPUT_FILES, MODEL_SOURCE_FILES,
  type ModelPromotionAuditV2 } from "./package-model-release.js";
import { getRepoRoot } from "./paths.js";
import { RELEASE_FILES, releaseSha256, verifyReleaseBundle } from "./release-manifest.js";

const repoRoot = getRepoRoot(import.meta.url);
const TAG = /^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/;
const ARTIFACT = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const DIGEST = /^[a-f0-9]{64}$/;
const HTTPS = /^https:\/\/[^\s/]+\/\S+$/;
const BASE_FILES = [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood,
  RELEASE_FILES.explorer, RELEASE_FILES.catalog, "publication-audit.json"] as const;
const AUDIT_FIELDS = ["format", "status", "promotionId", "tag", "bundleId",
  "manifestSha256", "base", "datasetSha256", "sourceName", "modelSha256",
  "numericArchiveSha256", "numericMetadataSha256", "modelCoverage", "evidence",
  "approvals", "assets"] as const;
const EVIDENCE_FIELDS = ["reviewSha256", "selectionFileSha256", "finalReportSha256",
  "refitRecordSha256", "datasetBridgeSha256", "qualityPlanSha256",
  "qualityPolicySha256", "servingCohortSha256", "servingFinalSha256",
  "servingFreezeSha256", "servingReportSha256"] as const;
const APPROVAL_FIELDS = ["owner", "ownerApprovalRef", "freezeApprovalRef",
  "trainingApprovalRef", "publicationApprovalRef", "deploymentApprovalRef"] as const;
const MODEL_ENTRY_FIELDS = ["promotionId", "tag", "bundleId", "manifestSha256",
  "auditSha256", "baseTag", "baseBundleId", "baseManifestSha256", "sourceRunId",
  "artifactName", "decisionRef", "datasetBridgeSha256", "datasetBridgeReviewRef",
  "qualityPlanSha256", "qualityPolicySha256", "servingFinalSha256",
  "servingFreezeSha256", "freezeRevision", "freezeApprovalRef",
  "numericMetadataSha256", "servingCohortSha256", "servingReportSha256",
  "owner", "ownerApprovalRef", "trainingApprovalRef", "publicationApprovalRef",
  "deploymentApprovalRef"] as const;

export interface ModelPublicReleaseOptions {
  packageDir: string;
  baseDir: string;
  dispatch: { tag: string; baseTag: string; sourceRunId: number;
    artifactName: string; ownerApprovalRef: string };
  providerApprovals: unknown;
  publicationApprovals: unknown;
  modelApprovals: unknown;
  planApprovals: unknown;
  /** Invented tests use isolated Git history. The executable always uses this repository. */
  approvalRepoDir?: string;
}

type RecordValue = Record<string, unknown>;

function fail(field: string, reason: string): never {
  throw new Error(`Model public release ${field}: ${reason}`);
}

function fields(value: unknown, keys: readonly string[], field: string): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  const entry = value as RecordValue;
  for (const key of keys) if (!Object.hasOwn(entry, key)) fail(`${field}.${key}`, "is required");
  for (const key of Object.keys(entry)) if (!keys.includes(key)) fail(`${field}.${key}`, "is unsupported");
  return entry;
}

function match(value: unknown, pattern: RegExp, field: string): string {
  if (typeof value !== "string" || !pattern.test(value)) fail(field, "is invalid");
  return value as string;
}

function read(directory: string, name: string): Buffer {
  return fs.readFileSync(path.join(directory, name));
}

function json(bytes: Buffer, field: string): RecordValue {
  let value: unknown;
  try { value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
  catch { fail(field, "must be valid JSON and UTF-8"); }
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as RecordValue;
}

function exactFiles(directory: string, names: readonly string[], field: string): void {
  if (!fs.existsSync(directory) || !fs.lstatSync(directory).isDirectory() ||
      fs.lstatSync(directory).isSymbolicLink()) fail(field, "must be a real directory");
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  if (JSON.stringify(entries.map((entry) => entry.name).sort()) !==
      JSON.stringify([...names].sort())) fail(field, "has an unsupported file inventory");
  let total = 0;
  for (const entry of entries) {
    const item = fs.lstatSync(path.join(directory, entry.name));
    const maximum = entry.name === RELEASE_FILES.manifest ? RELEASE_BUNDLE_LIMITS.manifestBytes
      : entry.name.endsWith("-audit.json") ? 1024 * 1024
      : RELEASE_BUNDLE_LIMITS.plainAssetBytes;
    if (!entry.isFile() || item.isSymbolicLink() || item.size < 1 || item.size > maximum) {
      fail(`${field}.${entry.name}`, "must be a bounded regular file");
    }
    if (entry.name !== RELEASE_FILES.manifest && !entry.name.endsWith("-audit.json")) {
      total += item.size;
    }
  }
  if (total > RELEASE_BUNDLE_LIMITS.totalPlainBytes) fail(field, "total public bytes exceed limit");
}

function providerUse(provider: RecordValue, audit: RecordValue,
  approvals: RecordValue, use: "training" | "publication" | "deployment"): void {
  const record = fields(provider[use], ["approved", "sources", "sourceBasis", "use",
    "owner", "approvedAt", "decisionRef", "approvalRef"], `provider.${use}`);
  if (record.approved !== true || !Array.isArray(record.sources) ||
      !record.sources.includes(audit.sourceName) ||
      record.owner !== approvals.owner ||
      record.approvalRef !== approvals[`${use}ApprovalRef`] ||
      typeof record.sourceBasis !== "string" || !record.sourceBasis.trim() ||
      typeof record.use !== "string" || !record.use.trim() ||
      typeof record.approvedAt !== "string" ||
      !/^\d{4}-\d{2}-\d{2}$/.test(record.approvedAt) ||
      !Number.isFinite(new Date(`${record.approvedAt}T00:00:00Z`).getTime()) ||
      new Date(`${record.approvedAt}T00:00:00Z`).toISOString().slice(0, 10) !== record.approvedAt) {
    fail(`provider.${use}`, "does not match reviewed source/use and owner");
  }
  const decision = match(record.decisionRef,
    /^docs\/decisions\/[0-9]{4}-[a-z0-9-]+\.md$/, `provider.${use}.decisionRef`);
  if (!fs.existsSync(path.join(repoRoot, decision))) fail(`provider.${use}.decisionRef`, "is absent");
}

/** A no-private-input check for the write-permission release job. */
export function verifyModelPublicRelease(options: ModelPublicReleaseOptions): ModelPromotionAuditV2 {
  const dispatch = options.dispatch;
  match(dispatch.tag, TAG, "dispatch.tag");
  match(dispatch.baseTag, TAG, "dispatch.baseTag");
  if (dispatch.tag === dispatch.baseTag) fail("dispatch.baseTag", "must differ from tag");
  if (!Number.isSafeInteger(dispatch.sourceRunId) || dispatch.sourceRunId < 1) {
    fail("dispatch.sourceRunId", "must be a positive safe integer");
  }
  match(dispatch.artifactName, ARTIFACT, "dispatch.artifactName");
  match(dispatch.ownerApprovalRef, HTTPS, "dispatch.ownerApprovalRef");
  exactFiles(options.packageDir, MODEL_OUTPUT_FILES, "packageDir");
  exactFiles(options.baseDir, BASE_FILES, "baseDir");
  const manifest = verifyReleaseBundle(options.packageDir, options.baseDir);
  const base = parseReleaseManifest(json(read(options.baseDir, RELEASE_FILES.manifest),
    "base.release-manifest.json"), "base.release-manifest.json");
  if (manifest.tag !== dispatch.tag || base.tag !== dispatch.baseTag ||
      manifest.neighborhood.format !== "graph-compact-v3" ||
      manifest.explorer.format !== "graph-compact-v3" || !manifest.model ||
      base.neighborhood.format !== "graph-compact-v3" ||
      base.explorer.format !== "graph-compact-v3" || base.model !== null ||
      manifest.dataset.source === "synthetic-fixture" ||
      base.dataset.source !== manifest.dataset.source) {
    fail("release-manifest.json", "requires reviewed v3 item model over the named data-only base");
  }
  for (const name of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    RELEASE_FILES.catalog]) {
    if (!read(options.packageDir, name).equals(read(options.baseDir, name))) {
      fail(name, "differs from the approved data base");
    }
  }
  const auditBytes = read(options.packageDir, "model-promotion-audit.json");
  const audit = fields(json(auditBytes, "model-promotion-audit.json"), AUDIT_FIELDS,
    "model-promotion-audit.json");
  const evidence = fields(audit.evidence, EVIDENCE_FIELDS, "audit.evidence");
  const approvals = fields(audit.approvals, APPROVAL_FIELDS, "audit.approvals");
  const auditBase = fields(audit.base, ["tag", "bundleId", "manifestSha256"], "audit.base");
  for (const key of EVIDENCE_FIELDS) match(evidence[key], DIGEST, `audit.evidence.${key}`);
  for (const key of ["numericArchiveSha256", "numericMetadataSha256"] as const) {
    match(audit[key], DIGEST, `audit.${key}`);
  }
  if (typeof audit.promotionId !== "string" || !audit.promotionId.trim() ||
      audit.promotionId.length > 300 || typeof audit.modelCoverage !== "number" ||
      !Number.isFinite(audit.modelCoverage) || audit.modelCoverage <= 0 ||
      audit.modelCoverage > 1) {
    fail("model-promotion-audit.json", "has invalid promotion ID or model coverage");
  }
  for (const key of APPROVAL_FIELDS) {
    if (key === "owner") {
      if (typeof approvals.owner !== "string" || !approvals.owner.trim()) {
        fail("audit.approvals.owner", "is required");
      }
    } else match(approvals[key], HTTPS, `audit.approvals.${key}`);
  }
  if ([approvals.trainingApprovalRef, approvals.publicationApprovalRef,
    approvals.deploymentApprovalRef].includes(approvals.ownerApprovalRef)) {
    fail("audit.approvals.ownerApprovalRef", "must be separate from provider source/use approvals");
  }
  const manifestSha256 = releaseSha256(read(options.packageDir, RELEASE_FILES.manifest));
  const baseManifestSha256 = releaseSha256(read(options.baseDir, RELEASE_FILES.manifest));
  if (audit.format !== "model-promotion-audit-v2" || audit.status !== "pending-approval" ||
      audit.tag !== manifest.tag || audit.bundleId !== manifest.bundleId ||
      audit.manifestSha256 !== manifestSha256 ||
      audit.datasetSha256 !== manifest.dataset.sha256 ||
      audit.sourceName !== manifest.dataset.source ||
      audit.modelSha256 !== manifest.model.sha256 ||
      auditBase.tag !== base.tag || auditBase.bundleId !== base.bundleId ||
      auditBase.manifestSha256 !== baseManifestSha256) {
    fail("model-promotion-audit.json", "differs from verified candidate or data base");
  }
  const assets = MODEL_SOURCE_FILES.map((name) => ({ path: name,
    bytes: read(options.packageDir, name).length,
    sha256: releaseSha256(read(options.packageDir, name)) }));
  if (JSON.stringify(audit.assets) !== JSON.stringify(assets)) {
    fail("audit.assets", "differs from exact public bytes");
  }
  const baseAuditBytes = read(options.baseDir, "publication-audit.json");
  const baseAudit = json(baseAuditBytes, "publication-audit.json");
  if (baseAudit.format !== "publication-audit-v1" || baseAudit.publishable !== true ||
      baseAudit.tag !== base.tag || baseAudit.bundleId !== base.bundleId ||
      baseAudit.manifestSha256 !== baseManifestSha256) {
    fail("publication-audit.json", "does not identify a publishable reviewed base");
  }
  const providerRoot = fields(options.providerApprovals, ["schemaVersion", "approvals"],
    "providerApprovals");
  if (providerRoot.schemaVersion !== 1) fail("providerApprovals.schemaVersion", "is unsupported");
  const provider = fields(providerRoot.approvals, ["training", "publication", "deployment"],
    "providerApprovals.approvals");
  for (const use of ["training", "publication", "deployment"] as const) {
    providerUse(provider, audit, approvals, use);
  }
  const publication = fields(options.publicationApprovals, ["schemaVersion", "packages"],
    "publicationApprovals");
  if (publication.schemaVersion !== 1 || !Array.isArray(publication.packages)) {
    fail("publicationApprovals", "is unsupported");
  }
  const bases = publication.packages.filter((value) =>
    value && typeof value === "object" && !Array.isArray(value) &&
    (value as RecordValue).tag === base.tag);
  if (bases.length !== 1) fail("publicationApprovals", "requires one exact approved data base");
  const baseApproval = fields(bases[0], ["tag", "bundleId", "manifestSha256",
    "auditSha256", "sourceRunId", "artifactName", "previousTag", "decisionRef",
    "approvalRef", "owner", "bootstrap"], "publicationApprovals.base");
  if (baseApproval.bundleId !== base.bundleId ||
      baseApproval.manifestSha256 !== baseManifestSha256 ||
      baseApproval.auditSha256 !== releaseSha256(baseAuditBytes) ||
      baseApproval.approvalRef !== approvals.publicationApprovalRef ||
      baseApproval.owner !== approvals.owner) {
    fail("publicationApprovals.base", "differs from reviewed data base bytes or owner");
  }
  const registry = fields(options.modelApprovals, ["schemaVersion", "promotions"],
    "modelApprovals");
  if (registry.schemaVersion !== 1 || !Array.isArray(registry.promotions)) {
    fail("modelApprovals", "is unsupported");
  }
  const entries = registry.promotions.filter((value) =>
    value && typeof value === "object" && !Array.isArray(value) &&
    (value as RecordValue).tag === dispatch.tag);
  if (entries.length !== 1) fail("modelApprovals", "requires one exact owner approval");
  const entry = fields(entries[0], MODEL_ENTRY_FIELDS, "modelApprovals.promotion");
  const expected = { promotionId: audit.promotionId, tag: manifest.tag,
    bundleId: manifest.bundleId, manifestSha256, auditSha256: releaseSha256(auditBytes),
    baseTag: base.tag, baseBundleId: base.bundleId, baseManifestSha256,
    sourceRunId: dispatch.sourceRunId, artifactName: dispatch.artifactName,
    datasetBridgeSha256: evidence.datasetBridgeSha256,
    qualityPlanSha256: evidence.qualityPlanSha256,
    qualityPolicySha256: evidence.qualityPolicySha256,
    servingFinalSha256: evidence.servingFinalSha256,
    servingFreezeSha256: evidence.servingFreezeSha256,
    numericMetadataSha256: audit.numericMetadataSha256,
    servingCohortSha256: evidence.servingCohortSha256,
    servingReportSha256: evidence.servingReportSha256,
    owner: approvals.owner, ownerApprovalRef: approvals.ownerApprovalRef,
    freezeApprovalRef: approvals.freezeApprovalRef,
    trainingApprovalRef: approvals.trainingApprovalRef,
    publicationApprovalRef: approvals.publicationApprovalRef,
    deploymentApprovalRef: approvals.deploymentApprovalRef };
  for (const [key, value] of Object.entries(expected)) {
    if (entry[key] !== value) fail(`modelApprovals.promotion.${key}`, "differs from exact package");
  }
  if (approvals.ownerApprovalRef !== dispatch.ownerApprovalRef) {
    fail("dispatch.ownerApprovalRef", "differs from exact owner approval");
  }
  match(entry.freezeRevision, /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/,
    "modelApprovals.promotion.freezeRevision");
  match(entry.datasetBridgeReviewRef, HTTPS,
    "modelApprovals.promotion.datasetBridgeReviewRef");
  const decisionRef = match(entry.decisionRef,
    /^docs\/decisions\/[0-9]{4}-[a-z0-9-]+\.md$/,
    "modelApprovals.promotion.decisionRef");
  if (!fs.existsSync(path.join(repoRoot, decisionRef))) {
    fail("modelApprovals.promotion.decisionRef", "is absent");
  }
  verifyPriorPlanApproval(options.approvalRepoDir ?? repoRoot, entry.freezeRevision,
    options.planApprovals, options.modelApprovals, dispatch.tag, {
      sourceName: audit.sourceName as string,
      graphDatasetSha256: audit.datasetSha256 as string,
      qualityPlanSha256: evidence.qualityPlanSha256 as string,
      servingCohortSha256: evidence.servingCohortSha256 as string,
      servingFinalSha256: evidence.servingFinalSha256 as string,
      servingFreezeSha256: evidence.servingFreezeSha256 as string,
      owner: approvals.owner as string,
      approvalRef: approvals.freezeApprovalRef as string,
      decisionRef: "docs/decisions/0035-serving-evaluation-freeze.md",
    });
  return audit as unknown as ModelPromotionAuditV2;
}

function main(): void {
  const command = new Command();
  command.requiredOption("--package <path>").requiredOption("--base <path>")
    .requiredOption("--tag <tag>").requiredOption("--base-tag <tag>")
    .requiredOption("--run-id <number>").requiredOption("--artifact-name <name>")
    .requiredOption("--owner-approval-ref <url>");
  command.parse(process.argv);
  const flags = command.opts();
  const audit = verifyModelPublicRelease({ packageDir: flags.package, baseDir: flags.base,
    dispatch: { tag: flags.tag, baseTag: flags.baseTag,
      sourceRunId: Number(flags.runId), artifactName: flags.artifactName,
      ownerApprovalRef: flags.ownerApprovalRef },
    providerApprovals: json(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/provider-data.json")), "provider-data.json"),
    publicationApprovals: json(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/publication-bundles.json")), "publication-bundles.json"),
    modelApprovals: json(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/model-release-bundles.json")), "model-release-bundles.json"),
    planApprovals: json(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/model-evaluation-plans.json")), "model-evaluation-plans.json") });
  process.stdout.write(`Verified approved public model package ${audit.tag} ${audit.bundleId}.\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
