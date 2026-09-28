import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { parseReleaseManifest, RELEASE_BUNDLE_LIMITS, type ReleaseManifestV1 } from
  "../../web/src/artifacts.js";
import { getRepoRoot } from "./paths.js";
import { OUTPUT_FILES, PUBLIC_FIELDS, SOURCE_FILES, packageDataRelease,
  type PublicationAuditV1, type PublicationReviewV1,
  type ValidatedFirstBundleApproval, type ValidatedPublicationApproval } from
  "./package-data-release.js";
import { RELEASE_FILES, releaseSha256 } from "./release-manifest.js";

const repoRoot = getRepoRoot(import.meta.url);
const AUDIT_LIMIT = 1024 * 1024;
const TAG = /^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/;
const DIGEST = /^[a-f0-9]{64}$/;
const ARTIFACT_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

export interface PublicationDispatch {
  tag: string;
  sourceRunId: number;
  artifactName: string;
  previousTag: string | null;
  approvalRef: string;
}

export interface VerifyPublicationPackageOptions {
  packageDir: string;
  previousDir?: string;
  dispatch: PublicationDispatch;
  providerApprovals: unknown;
  packageApprovals: unknown;
}

function fail(field: string, reason: string): never {
  throw new Error(`Publication verification ${field}: ${reason}`);
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(label, "must be an object");
  }
  return value as Record<string, unknown>;
}

function fields(value: unknown, expected: readonly string[], label: string): Record<string, unknown> {
  const entry = record(value, label);
  for (const field of expected) if (!Object.hasOwn(entry, field)) fail(`${label}.${field}`, "is required");
  for (const field of Object.keys(entry)) if (!expected.includes(field)) {
    fail(`${label}.${field}`, "is unsupported");
  }
  return entry;
}

function match(value: unknown, regex: RegExp, label: string): string {
  if (typeof value !== "string" || !regex.test(value)) fail(label, "is invalid");
  return value;
}

function positiveRunId(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    fail(label, "must be a positive safe integer");
  }
  return value as number;
}

function decisionRef(value: unknown, label: string): string {
  const ref = match(value, /^docs\/decisions\/[0-9]{4}-[a-z0-9-]+\.md$/, label);
  if (!fs.existsSync(path.join(repoRoot, ref))) fail(label, "must name an existing decision file");
  return ref;
}

function canonicalDate(value: unknown, label: string): void {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    fail(label, "must be an ISO date");
  }
  const parsed = new Date(`${value}T00:00:00Z`);
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== value) {
    fail(label, "must be a real ISO date");
  }
}

function exactFiles(directory: string): void {
  if (!fs.existsSync(directory)) fail("packageDir", "is missing");
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) fail("packageDir", "must be a real directory");
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  if (JSON.stringify(entries.map((entry) => entry.name).sort()) !==
      JSON.stringify([...OUTPUT_FILES].sort())) {
    fail("packageDir", "must contain exactly the five declared files");
  }
  let total = 0;
  for (const entry of entries) {
    const filepath = path.join(directory, entry.name);
    const item = fs.lstatSync(filepath);
    const maximum = entry.name === RELEASE_FILES.manifest
      ? RELEASE_BUNDLE_LIMITS.manifestBytes
      : entry.name === "publication-audit.json" ? AUDIT_LIMIT : RELEASE_BUNDLE_LIMITS.plainAssetBytes;
    if (!entry.isFile() || item.isSymbolicLink() || item.size < 1 || item.size > maximum) {
      fail(entry.name, "must be a bounded regular file");
    }
    if (entry.name !== RELEASE_FILES.manifest && entry.name !== "publication-audit.json") {
      total += item.size;
    }
  }
  if (total > RELEASE_BUNDLE_LIMITS.totalPlainBytes) fail("packageDir", "total bytes exceed limit");
}

function validateDispatch(value: PublicationDispatch): void {
  match(value.tag, TAG, "dispatch.tag");
  positiveRunId(value.sourceRunId, "dispatch.sourceRunId");
  match(value.artifactName, ARTIFACT_NAME, "dispatch.artifactName");
  if (value.previousTag !== null) {
    match(value.previousTag, TAG, "dispatch.previousTag");
    if (value.previousTag === value.tag) fail("dispatch.previousTag", "must differ from tag");
  }
  match(value.approvalRef, /^https:\/\/[^\s/]+\/\S+$/, "dispatch.approvalRef");
}

function validatedApproval(provider: unknown, packageRegistry: unknown, audit: Record<string, unknown>,
  dispatch: PublicationDispatch, auditSha256: string, manifest: ReleaseManifestV1):
  { publication: ValidatedPublicationApproval; bootstrap?: ValidatedFirstBundleApproval } {
  const providerRoot = record(provider, "providerApprovals");
  if (providerRoot.schemaVersion !== 1) fail("providerApprovals.schemaVersion", "is unsupported");
  const approvals = record(providerRoot.approvals, "providerApprovals.approvals");
  const use = record(approvals.publication, "providerApprovals.approvals.publication");
  if (use.approved !== true) fail("providerApprovals.approvals.publication", "is not approved");
  const source = record(audit.source, "publication-audit.json.source");
  const redistribution = record(audit.redistribution, "publication-audit.json.redistribution");
  const sourceName = source.name;
  if (typeof sourceName !== "string" || !sourceName || sourceName === "synthetic-fixture" ||
      !Array.isArray(use.sources) || !use.sources.includes(sourceName) ||
      use.approvalRef !== dispatch.approvalRef || use.approvalRef !== redistribution.approvalRef ||
      use.owner !== redistribution.owner || use.decisionRef !== source.decisionRef ||
      typeof use.sourceBasis !== "string" || !use.sourceBasis.trim() ||
      typeof use.use !== "string" || !use.use.trim() ||
      typeof use.approvedAt !== "string") {
    fail("providerApprovals.approvals.publication", "does not match the reviewed package source/use");
  }
  canonicalDate(use.approvedAt, "providerApprovals.approvals.publication.approvedAt");
  decisionRef(use.decisionRef, "providerApprovals.approvals.publication.decisionRef");

  const registry = fields(packageRegistry, ["schemaVersion", "packages"], "packageApprovals");
  if (registry.schemaVersion !== 1 || !Array.isArray(registry.packages)) {
    fail("packageApprovals", "requires schemaVersion 1 and packages array");
  }
  const entries = registry.packages.map((value, index) => fields(value,
    ["tag", "bundleId", "manifestSha256", "auditSha256", "sourceRunId",
      "artifactName", "previousTag", "decisionRef", "approvalRef", "owner", "bootstrap"],
    `packageApprovals.packages[${index}]`));
  const seen = new Set<string>();
  for (const [index, entry] of entries.entries()) {
    match(entry.tag, TAG, `packageApprovals.packages[${index}].tag`);
    match(entry.bundleId, DIGEST, `packageApprovals.packages[${index}].bundleId`);
    match(entry.manifestSha256, DIGEST, `packageApprovals.packages[${index}].manifestSha256`);
    match(entry.auditSha256, DIGEST, `packageApprovals.packages[${index}].auditSha256`);
    positiveRunId(entry.sourceRunId, `packageApprovals.packages[${index}].sourceRunId`);
    match(entry.artifactName, ARTIFACT_NAME, `packageApprovals.packages[${index}].artifactName`);
    if (entry.previousTag !== null) {
      match(entry.previousTag, TAG, `packageApprovals.packages[${index}].previousTag`);
    }
    decisionRef(entry.decisionRef, `packageApprovals.packages[${index}].decisionRef`);
    match(entry.approvalRef, /^https:\/\/[^\s/]+\/\S+$/,
      `packageApprovals.packages[${index}].approvalRef`);
    if (typeof entry.owner !== "string" || !entry.owner.trim()) {
      fail(`packageApprovals.packages[${index}].owner`, "is required");
    }
    if (entry.previousTag === null) {
      const bootstrap = fields(entry.bootstrap,
        ["decisionRef", "approvalRef", "owner"],
        `packageApprovals.packages[${index}].bootstrap`);
      decisionRef(bootstrap.decisionRef, `packageApprovals.packages[${index}].bootstrap.decisionRef`);
      match(bootstrap.approvalRef, /^https:\/\/[^\s/]+\/\S+$/,
        `packageApprovals.packages[${index}].bootstrap.approvalRef`);
      if (bootstrap.approvalRef === entry.approvalRef ||
          bootstrap.decisionRef === entry.decisionRef ||
          bootstrap.owner !== entry.owner) {
        fail(`packageApprovals.packages[${index}].bootstrap`,
          "requires a separate approval and decision with the same owner");
      }
    } else if (entry.bootstrap !== null) {
      fail(`packageApprovals.packages[${index}].bootstrap`,
        "must be null when a previous bundle is named");
    }
    if (seen.has(entry.tag as string)) fail("packageApprovals.packages", "contains duplicate tags");
    seen.add(entry.tag as string);
  }
  const approved = entries.find((entry) => entry.tag === dispatch.tag);
  if (!approved || approved.bundleId !== audit.bundleId ||
      approved.manifestSha256 !== audit.manifestSha256 ||
      approved.auditSha256 !== auditSha256 ||
      approved.sourceRunId !== dispatch.sourceRunId ||
      approved.artifactName !== dispatch.artifactName ||
      approved.previousTag !== dispatch.previousTag ||
      approved.decisionRef !== use.decisionRef ||
      approved.approvalRef !== dispatch.approvalRef ||
      approved.owner !== use.owner) {
    fail("packageApprovals", "no exact independently reviewed package matches these bytes and inputs");
  }
  if (dispatch.previousTag !== null) {
    const prior = manifest.lastKnownGood;
    const priorApproved = entries.find((entry) => entry.tag === dispatch.previousTag);
    if (!prior || prior.tag !== dispatch.previousTag || !priorApproved ||
        priorApproved.bundleId !== prior.bundleId ||
        priorApproved.manifestSha256 !== prior.manifestSha256) {
      fail("packageApprovals", "named predecessor needs an exact approved registry entry");
    }
  } else if (manifest.lastKnownGood !== null) {
    fail("release-manifest.json.lastKnownGood", "must be null for reviewed first bundle");
  }
  const publication: ValidatedPublicationApproval = {
    scope: "publication", approved: true, approvalRef: dispatch.approvalRef,
    decisionRef: use.decisionRef as string, owner: use.owner as string,
    sources: use.sources as string[],
  };
  if (dispatch.previousTag !== null) return { publication };
  const bootstrap = approved.bootstrap as Record<string, string>;
  return { publication, bootstrap: {
    scope: "first-real-bundle", approved: true, tag: dispatch.tag,
    bundleId: approved.bundleId as string,
    manifestSha256: approved.manifestSha256 as string,
    decisionRef: bootstrap.decisionRef,
    approvalRef: bootstrap.approvalRef, owner: bootstrap.owner,
  } };
}

function reviewFromAudit(audit: Record<string, unknown>): PublicationReviewV1 {
  const source = record(audit.source, "publication-audit.json.source");
  const redistribution = record(audit.redistribution, "publication-audit.json.redistribution");
  const changes = record(audit.changes, "publication-audit.json.changes");
  const quality = record(audit.quality, "publication-audit.json.quality");
  return {
    format: "publication-review-v1", tag: audit.tag as string,
    bundleId: audit.bundleId as string, manifestSha256: audit.manifestSha256 as string,
    source: source as unknown as PublicationReviewV1["source"],
    redistribution: redistribution as unknown as PublicationReviewV1["redistribution"],
    changes: changes as unknown as PublicationReviewV1["changes"],
    quality: { checks: quality.checks as PublicationReviewV1["quality"]["checks"] },
  };
}

function cleanup(root: string): void {
  const realRoot = fs.realpathSync(root);
  const realTmp = fs.realpathSync(os.tmpdir());
  if (path.dirname(realRoot) !== realTmp || !path.basename(realRoot).startsWith("publication-verify-") ||
      fs.lstatSync(root).isSymbolicLink()) {
    fail("temporary", "refusing cleanup outside OS temporary directory");
  }
  fs.rmSync(root, { recursive: true, force: true });
}

/** Verify a downloaded five-file package without modifying a release. */
export function verifyPublicationPackage(options: VerifyPublicationPackageOptions): PublicationAuditV1 {
  validateDispatch(options.dispatch);
  if ((options.dispatch.previousTag === null) !== (options.previousDir === undefined)) {
    fail("previousDir", "must be absent exactly for a reviewed first bundle");
  }
  const packageDir = path.resolve(options.packageDir);
  exactFiles(packageDir);
  const manifest = parseReleaseManifest(JSON.parse(fs.readFileSync(path.join(packageDir,
    RELEASE_FILES.manifest), "utf8")), RELEASE_FILES.manifest);
  const auditBytes = fs.readFileSync(path.join(packageDir, "publication-audit.json"));
  let audit: Record<string, unknown>;
  try {
    audit = record(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(auditBytes)),
      "publication-audit.json");
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("Publication verification")) throw error;
    fail("publication-audit.json", "is invalid JSON or UTF-8");
  }
  if (audit.format !== "publication-audit-v1" || audit.publishable !== true ||
      audit.tag !== options.dispatch.tag) {
    fail("publication-audit.json", "must be a publishable v1 audit for the requested tag");
  }
  const changes = record(audit.changes, "publication-audit.json.changes");
  if (changes.previousTag !== options.dispatch.previousTag) {
    fail("publication-audit.json.changes.previousTag", "does not match requested prior");
  }
  const approvals = validatedApproval(options.providerApprovals, options.packageApprovals,
    audit, options.dispatch, releaseSha256(auditBytes), manifest);
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "publication-verify-"));
  try {
    const sourceDir = path.join(temporary, "source");
    fs.mkdirSync(sourceDir);
    for (const filename of SOURCE_FILES) {
      fs.copyFileSync(path.join(packageDir, filename), path.join(sourceDir, filename),
        fs.constants.COPYFILE_EXCL);
    }
    const recomputed = packageDataRelease({ candidateDir: sourceDir,
      previousDir: options.previousDir, outputDir: path.join(temporary, "recomputed"),
      review: reviewFromAudit(audit), approval: approvals.publication,
      bootstrapApproval: approvals.bootstrap });
    const computedDir = path.join(temporary, "recomputed");
    for (const filename of OUTPUT_FILES) {
      if (!fs.readFileSync(path.join(computedDir, filename)).equals(
        fs.readFileSync(path.join(packageDir, filename)))) {
        fail(filename, "bytes differ from the independently recomputed package");
      }
    }
    if (JSON.stringify(recomputed.redistribution.allowedFields) !== JSON.stringify(PUBLIC_FIELDS)) {
      fail("publication-audit.json.redistribution.allowedFields", "is not the public contract");
    }
    return recomputed;
  } finally {
    cleanup(temporary);
  }
}

function main(): void {
  const command = new Command();
  command.requiredOption("--package <path>").option("--previous <path>")
    .requiredOption("--tag <tag>").requiredOption("--run-id <number>")
    .requiredOption("--artifact-name <name>").option("--prior-tag <tag>")
    .requiredOption("--approval-ref <url>");
  command.parse(process.argv);
  const flags = command.opts();
  if (Boolean(flags.previous) !== Boolean(flags.priorTag)) {
    fail("previousDir", "--previous and --prior-tag must be supplied together");
  }
  const runId = Number(flags.runId);
  const dispatch: PublicationDispatch = { tag: flags.tag, sourceRunId: runId,
    artifactName: flags.artifactName, previousTag: flags.priorTag ?? null,
    approvalRef: flags.approvalRef };
  const audit = verifyPublicationPackage({ packageDir: flags.package, previousDir: flags.previous,
    dispatch,
    providerApprovals: JSON.parse(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/provider-data.json"), "utf8")),
    packageApprovals: JSON.parse(fs.readFileSync(path.join(repoRoot,
      "docs/approvals/publication-bundles.json"), "utf8")) });
  process.stdout.write(`Verified publication package ${audit.tag} ${audit.bundleId}.\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
