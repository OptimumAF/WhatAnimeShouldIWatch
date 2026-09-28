import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { RELEASE_BUNDLE_LIMITS } from "../../web/src/artifacts.js";
import { getRepoRoot } from "./paths.js";
import { MODEL_OUTPUT_FILES, MODEL_SOURCE_FILES, packageModelRelease,
  type ModelPromotionAuditV1 } from "./package-model-release.js";
import { releaseSha256 } from "./release-manifest.js";
import { verifyPublicationPackage } from "./verify-publication-package.js";

const repoRoot = getRepoRoot(import.meta.url);
const TAG = /^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/;
const DIGEST = /^[a-f0-9]{64}$/;
const ARTIFACT = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

export interface ModelPromotionDispatch {
  tag: string;
  baseTag: string;
  sourceRunId: number;
  artifactName: string;
}

export interface ModelPackageVerificationOptions {
  packageDir: string;
  baseDir: string;
  basePreviousDir?: string;
  evidenceDir: string;
  dispatch: ModelPromotionDispatch;
  providerApprovals: unknown;
  publicationApprovals: unknown;
  modelApprovals: unknown;
}

function fail(field: string, reason: string): never {
  throw new Error(`Model promotion verification ${field}: ${reason}`);
}

function object(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as Record<string, unknown>;
}

function fields(value: unknown, required: readonly string[], field: string): Record<string, unknown> {
  const entry = object(value, field);
  for (const key of required) if (!Object.hasOwn(entry, key)) fail(`${field}.${key}`, "is required");
  for (const key of Object.keys(entry)) if (!required.includes(key)) fail(`${field}.${key}`, "is unsupported");
  return entry;
}

function match(value: unknown, pattern: RegExp, field: string): string {
  if (typeof value !== "string" || !pattern.test(value)) fail(field, "is invalid");
  return value as string;
}

function runId(value: unknown, field: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) fail(field, "must be a positive safe integer");
  return value as number;
}

function decisionRef(value: unknown, field: string): string {
  const ref = match(value, /^docs\/decisions\/[0-9]{4}-[a-z0-9-]+\.md$/, field);
  if (!fs.existsSync(path.join(repoRoot, ref))) fail(field, "must name an existing decision file");
  return ref;
}

function exactPackage(directory: string): void {
  if (!fs.existsSync(directory) || !fs.lstatSync(directory).isDirectory() ||
      fs.lstatSync(directory).isSymbolicLink()) fail("packageDir", "must be a real directory");
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  if (JSON.stringify(entries.map((entry) => entry.name).sort()) !==
      JSON.stringify([...MODEL_OUTPUT_FILES].sort())) {
    fail("packageDir", "must contain exactly the six public model package files");
  }
  let total = 0;
  for (const entry of entries) {
    const item = fs.lstatSync(path.join(directory, entry.name));
    const maximum = entry.name === "release-manifest.json" ? RELEASE_BUNDLE_LIMITS.manifestBytes
      : entry.name === "model-promotion-audit.json" ? 1024 * 1024
      : RELEASE_BUNDLE_LIMITS.plainAssetBytes;
    if (!entry.isFile() || item.isSymbolicLink() || item.size < 1 || item.size > maximum) {
      fail(entry.name, "must be a bounded regular file");
    }
    if (entry.name !== "model-promotion-audit.json" && entry.name !== "release-manifest.json") {
      total += item.size;
    }
  }
  if (total > RELEASE_BUNDLE_LIMITS.totalPlainBytes) fail("packageDir", "total public bytes exceed limit");
}

function providerUse(value: unknown, use: "training" | "publication" | "deployment",
  audit: ModelPromotionAuditV1): void {
  const entry = object(value, `providerApprovals.approvals.${use}`);
  if (entry.approved !== true || entry.approvalRef !== audit.approvals[`${use}ApprovalRef`] ||
      entry.owner !== audit.approvals.owner || !Array.isArray(entry.sources) ||
      !entry.sources.includes(audit.sourceName) ||
      typeof entry.sourceBasis !== "string" || !entry.sourceBasis.trim() ||
      typeof entry.use !== "string" || !entry.use.trim() ||
      typeof entry.approvedAt !== "string" ||
      !/^\d{4}-\d{2}-\d{2}$/.test(entry.approvedAt) ||
      !Number.isFinite(new Date(`${entry.approvedAt}T00:00:00Z`).getTime()) ||
      new Date(`${entry.approvedAt}T00:00:00Z`).toISOString().slice(0, 10) !== entry.approvedAt) {
    fail(`providerApprovals.approvals.${use}`, "does not match reviewed source/use and owner");
  }
  decisionRef(entry.decisionRef, `providerApprovals.approvals.${use}.decisionRef`);
}

function cleanTemporary(directory: string): void {
  const real = fs.realpathSync(directory);
  const temporary = fs.realpathSync(os.tmpdir());
  if (path.dirname(real) !== temporary || !path.basename(real).startsWith("model-verify-") ||
      fs.lstatSync(directory).isSymbolicLink()) {
    fail("temporary", "refusing cleanup outside the OS temporary directory");
  }
  fs.rmSync(directory, { recursive: true, force: true });
}

/** Recompute the six public bytes from a private review and require exact committed approvals. */
export function verifyModelReleasePackage(options: ModelPackageVerificationOptions): ModelPromotionAuditV1 {
  match(options.dispatch.tag, TAG, "dispatch.tag");
  match(options.dispatch.baseTag, TAG, "dispatch.baseTag");
  if (options.dispatch.tag === options.dispatch.baseTag) fail("dispatch.baseTag", "must differ from tag");
  runId(options.dispatch.sourceRunId, "dispatch.sourceRunId");
  match(options.dispatch.artifactName, ARTIFACT, "dispatch.artifactName");
  exactPackage(options.packageDir);
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "model-verify-"));
  try {
    const candidate = path.join(temporary, "candidate");
    fs.mkdirSync(candidate);
    for (const name of MODEL_SOURCE_FILES) {
      fs.copyFileSync(path.join(options.packageDir, name), path.join(candidate, name),
        fs.constants.COPYFILE_EXCL);
    }
    const audit = packageModelRelease({ candidateDir: candidate, baseDir: options.baseDir,
      evidenceDir: options.evidenceDir, outputDir: path.join(temporary, "recomputed") });
    for (const name of MODEL_OUTPUT_FILES) {
      if (!fs.readFileSync(path.join(options.packageDir, name)).equals(
        fs.readFileSync(path.join(temporary, "recomputed", name)))) {
        fail(name, "differs from the independently recomputed package");
      }
    }
    if (audit.status !== "pending-approval" || audit.sourceName === "synthetic-fixture" ||
        audit.tag !== options.dispatch.tag || audit.base.tag !== options.dispatch.baseTag) {
      fail("model-promotion-audit.json", "is synthetic, unapproved, or mismatched to dispatch");
    }
    const privateReview = object(JSON.parse(fs.readFileSync(path.join(options.evidenceDir,
      "model-promotion-review.json"), "utf8")), "model-promotion-review.json");
    const bridge = object(JSON.parse(fs.readFileSync(path.join(options.evidenceDir,
      "dataset-bridge.json"), "utf8")), "dataset-bridge.json");
    const qualityPolicy = object(JSON.parse(fs.readFileSync(path.join(options.evidenceDir,
      "quality-policy.json"), "utf8")), "quality-policy.json");
    const provider = object(options.providerApprovals, "providerApprovals");
    if (provider.schemaVersion !== 1) fail("providerApprovals.schemaVersion", "is unsupported");
    const uses = object(provider.approvals, "providerApprovals.approvals");
    for (const use of ["training", "publication", "deployment"] as const) {
      providerUse(uses[use], use, audit);
    }

    const publication = fields(options.publicationApprovals, ["schemaVersion", "packages"],
      "publicationApprovals");
    if (publication.schemaVersion !== 1 || !Array.isArray(publication.packages)) {
      fail("publicationApprovals", "requires schemaVersion 1 and packages array");
    }
    const baseEntries = publication.packages.map((value, index) => fields(value,
      ["tag", "bundleId", "manifestSha256", "auditSha256", "sourceRunId", "artifactName",
        "previousTag", "decisionRef", "approvalRef", "owner", "bootstrap"],
      `publicationApprovals.packages[${index}]`));
    const baseMatches = baseEntries.filter((entry) => entry.tag === audit.base.tag);
    if (baseMatches.length !== 1) fail("publicationApprovals", "requires one exact approved data base");
    const baseEntry = baseMatches[0];
    if (baseEntry.bundleId !== audit.base.bundleId ||
        baseEntry.manifestSha256 !== audit.base.manifestSha256 ||
        baseEntry.approvalRef !== audit.approvals.publicationApprovalRef ||
        baseEntry.owner !== audit.approvals.owner ||
        baseEntry.decisionRef !== privateReview.sourceDecisionRef) {
      fail("publicationApprovals", "data base differs from the model review and provider use");
    }
    verifyPublicationPackage({ packageDir: options.baseDir, previousDir: options.basePreviousDir,
      dispatch: { tag: audit.base.tag, sourceRunId: baseEntry.sourceRunId as number,
        artifactName: baseEntry.artifactName as string,
        previousTag: baseEntry.previousTag as string | null,
        approvalRef: baseEntry.approvalRef as string },
      providerApprovals: options.providerApprovals,
      packageApprovals: options.publicationApprovals });

    const registry = fields(options.modelApprovals, ["schemaVersion", "promotions"], "modelApprovals");
    if (registry.schemaVersion !== 1 || !Array.isArray(registry.promotions)) {
      fail("modelApprovals", "requires schemaVersion 1 and promotions array");
    }
    const entries = registry.promotions.map((value, index) => fields(value,
      ["promotionId", "tag", "bundleId", "manifestSha256", "auditSha256", "baseTag",
        "baseBundleId", "baseManifestSha256", "sourceRunId", "artifactName", "decisionRef",
        "datasetBridgeSha256", "datasetBridgeReviewRef", "qualityPolicySha256",
        "servingReportSha256", "owner", "ownerApprovalRef", "trainingApprovalRef",
        "publicationApprovalRef", "deploymentApprovalRef"],
      `modelApprovals.promotions[${index}]`));
    const seen = new Set<string>();
    for (const [index, entry] of entries.entries()) {
      match(entry.tag, TAG, `modelApprovals.promotions[${index}].tag`);
      for (const key of ["bundleId", "manifestSha256", "auditSha256", "baseBundleId",
        "baseManifestSha256", "datasetBridgeSha256", "qualityPolicySha256",
        "servingReportSha256"] as const) {
        match(entry[key], DIGEST, `modelApprovals.promotions[${index}].${key}`);
      }
      decisionRef(entry.decisionRef, `modelApprovals.promotions[${index}].decisionRef`);
      runId(entry.sourceRunId, `modelApprovals.promotions[${index}].sourceRunId`);
      match(entry.artifactName, ARTIFACT, `modelApprovals.promotions[${index}].artifactName`);
      if (seen.has(entry.tag as string)) fail("modelApprovals.promotions", "contains duplicate tags");
      seen.add(entry.tag as string);
    }
    const approved = entries.filter((entry) => entry.tag === options.dispatch.tag);
    if (approved.length !== 1) fail("modelApprovals", "requires one exact committed owner review");
    const entry = approved[0];
    if (entry.promotionId !== audit.promotionId || entry.bundleId !== audit.bundleId ||
        entry.manifestSha256 !== audit.manifestSha256 ||
        entry.auditSha256 !== releaseSha256(fs.readFileSync(path.join(options.packageDir,
          "model-promotion-audit.json"))) ||
        entry.baseTag !== audit.base.tag || entry.baseBundleId !== audit.base.bundleId ||
        entry.baseManifestSha256 !== audit.base.manifestSha256 ||
        entry.sourceRunId !== options.dispatch.sourceRunId ||
        entry.artifactName !== options.dispatch.artifactName ||
        entry.decisionRef !== qualityPolicy.decisionRef ||
        entry.datasetBridgeSha256 !== audit.evidence.datasetBridgeSha256 ||
        entry.datasetBridgeReviewRef !== bridge.reviewRef ||
        entry.qualityPolicySha256 !== audit.evidence.qualityPolicySha256 ||
        entry.servingReportSha256 !== audit.evidence.servingReportSha256 ||
        entry.owner !== audit.approvals.owner ||
        entry.ownerApprovalRef !== audit.approvals.ownerApprovalRef ||
        entry.trainingApprovalRef !== audit.approvals.trainingApprovalRef ||
        entry.publicationApprovalRef !== audit.approvals.publicationApprovalRef ||
        entry.deploymentApprovalRef !== audit.approvals.deploymentApprovalRef) {
      fail("modelApprovals", "no exact independently reviewed model package matches these bytes and inputs");
    }
    if ([audit.approvals.trainingApprovalRef, audit.approvals.publicationApprovalRef,
      audit.approvals.deploymentApprovalRef].includes(audit.approvals.ownerApprovalRef)) {
      fail("modelApprovals.ownerApprovalRef", "must be separate from source/use approvals");
    }
    return audit;
  } finally { cleanTemporary(temporary); }
}

function main(): void {
  const command = new Command();
  command.requiredOption("--package <path>").requiredOption("--base <path>")
    .option("--base-previous <path>").requiredOption("--evidence <path>")
    .requiredOption("--tag <tag>").requiredOption("--base-tag <tag>")
    .requiredOption("--run-id <number>").requiredOption("--artifact-name <name>");
  command.parse(process.argv);
  const flags = command.opts();
  const result = verifyModelReleasePackage({ packageDir: flags.package, baseDir: flags.base,
    basePreviousDir: flags.basePrevious, evidenceDir: flags.evidence,
    dispatch: { tag: flags.tag, baseTag: flags.baseTag,
      sourceRunId: Number(flags.runId), artifactName: flags.artifactName },
    providerApprovals: JSON.parse(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/provider-data.json"), "utf8")),
    publicationApprovals: JSON.parse(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/publication-bundles.json"), "utf8")),
    modelApprovals: JSON.parse(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/model-release-bundles.json"), "utf8")) });
  process.stdout.write(`Verified model package ${result.tag} ${result.bundleId}.\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
