import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseCompactModel, parseReleaseManifest, RELEASE_BUNDLE_LIMITS } from
  "../../web/src/artifacts.js";
import { RELEASE_FILES, releaseSha256, verifyReleaseBundle } from "./release-manifest.js";

export const MODEL_SOURCE_FILES = [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood,
  RELEASE_FILES.explorer, RELEASE_FILES.catalog, RELEASE_FILES.model] as const;
export const MODEL_PRIVATE_FILES = ["model-promotion-review.json", "selection.json",
  "final-report.json", "final-report.sha256.json", "selection.test-used.json",
  "refit-record.json", "model.npz", "dataset-bridge.json", "quality-policy.json",
  "serving-cohort.json", "serving-report.json"] as const;
export const MODEL_OUTPUT_FILES = [...MODEL_SOURCE_FILES, "model-promotion-audit.json"] as const;

const DIGEST = /^[a-f0-9]{64}$/;
const TAG = /^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/;
const PRIVATE_JSON_LIMIT = 1024 * 1024;
const NUMERIC_ARCHIVE_LIMIT = 256 * 1024 * 1024;

type ObjectValue = Record<string, unknown>;

export interface ModelPackageOptions {
  candidateDir: string;
  baseDir: string;
  evidenceDir: string;
  outputDir: string;
  /** Only an invented fixture may use this; its package is permanently unpublishable. */
  syntheticFixture?: boolean;
  beforeActivate?: () => void;
}

export interface ModelPromotionAuditV1 {
  format: "model-promotion-audit-v1";
  status: "synthetic-only" | "pending-approval";
  promotionId: string;
  tag: string;
  bundleId: string;
  manifestSha256: string;
  base: { tag: string; bundleId: string; manifestSha256: string };
  datasetSha256: string;
  sourceName: string;
  modelSha256: string;
  numericArchiveSha256: string;
  modelCoverage: number;
  evidence: { reviewSha256: string; selectionFileSha256: string;
    finalReportSha256: string; refitRecordSha256: string;
    datasetBridgeSha256: string; qualityPolicySha256: string; servingCohortSha256: string;
    servingReportSha256: string };
  approvals: { owner: string; ownerApprovalRef: string; trainingApprovalRef: string;
    publicationApprovalRef: string; deploymentApprovalRef: string };
  assets: { path: string; bytes: number; sha256: string }[];
}

function fail(field: string, reason: string): never {
  throw new Error(`Model package ${field}: ${reason}`);
}

function object(value: unknown, field: string): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as ObjectValue;
}

function fields(value: unknown, required: readonly string[], field: string): ObjectValue {
  const entry = object(value, field);
  for (const key of required) if (!Object.hasOwn(entry, key)) fail(`${field}.${key}`, "is required");
  for (const key of Object.keys(entry)) if (!required.includes(key)) fail(`${field}.${key}`, "is unsupported");
  return entry;
}

function same(actual: unknown, expected: unknown, field: string): void {
  if (actual !== expected) fail(field, "does not match the bound evidence or artifact");
}

function text(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim() || value !== value.trim() || value.length > 300) {
    fail(field, "must be bounded nonempty trimmed text");
  }
  return value as string;
}

function pathExistsNoFollow(filepath: string): boolean {
  try { fs.lstatSync(filepath); return true; }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function digest(value: unknown, field: string): string {
  if (typeof value !== "string" || !DIGEST.test(value)) fail(field, "must be a lowercase SHA-256 digest");
  return value as string;
}

function tag(value: unknown, field: string): string {
  if (typeof value !== "string" || !TAG.test(value)) fail(field, "must be a versioned data-v tag");
  return value as string;
}

function https(value: unknown, field: string): string {
  const ref = text(value, field);
  if (!/^https:\/\/[^\s/]+\/\S+$/.test(ref)) fail(field, "must be an HTTPS reference");
  return ref;
}

function fraction(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    fail(field, "must be a finite number in [0, 1]");
  }
  return value as number;
}

function positive(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    fail(field, "must be a finite positive number");
  }
  return value as number;
}

function positiveInt(value: unknown, field: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) fail(field, "must be a positive safe integer");
  return value as number;
}

function decisionRef(value: unknown, field: string): string {
  const ref = text(value, field);
  if (!/^docs\/decisions\/[0-9]{4}-[a-z0-9-]+\.md$/.test(ref) ||
      !fs.existsSync(path.resolve(import.meta.dirname, "../..", ref))) {
    fail(field, "must name an existing repository decision");
  }
  return ref;
}

function json(bytes: Buffer, field: string): ObjectValue {
  try { return object(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)), field); }
  catch (error) {
    if (error instanceof SyntaxError || error instanceof TypeError) fail(field, "is invalid JSON or UTF-8");
    throw error;
  }
}

function inventory(directory: string, names: readonly string[], field: string): Map<string, Buffer> {
  if (!fs.existsSync(directory) || !fs.lstatSync(directory).isDirectory() ||
      fs.lstatSync(directory).isSymbolicLink()) fail(field, "must be a real directory");
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  if (JSON.stringify(entries.map((entry) => entry.name).sort()) !== JSON.stringify([...names].sort())) {
    fail(field, `must contain exactly ${names.join(", ")}`);
  }
  const bytes = new Map<string, Buffer>();
  let total = 0;
  for (const entry of entries) {
    const filename = entry.name;
    const item = fs.lstatSync(path.join(directory, filename));
    const maximum = filename === RELEASE_FILES.manifest ? RELEASE_BUNDLE_LIMITS.manifestBytes
      : filename === "model.npz" ? NUMERIC_ARCHIVE_LIMIT
      : (MODEL_PRIVATE_FILES as readonly string[]).includes(filename) ? PRIVATE_JSON_LIMIT
      : RELEASE_BUNDLE_LIMITS.plainAssetBytes;
    if (!entry.isFile() || item.isSymbolicLink() || item.size < 1 || item.size > maximum) {
      fail(`${field}.${filename}`, "must be a bounded regular file");
    }
    total += item.size;
    bytes.set(filename, fs.readFileSync(path.join(directory, filename)));
  }
  if (field === "candidate" && total > RELEASE_BUNDLE_LIMITS.totalPlainBytes +
      RELEASE_BUNDLE_LIMITS.manifestBytes) fail(field, "total public bytes exceed the limit");
  return bytes;
}

function readEvidence(bytes: Map<string, Buffer>, filename: string): ObjectValue {
  return json(bytes.get(filename)!, filename);
}

function modelMetrics(value: unknown, field: string) {
  const row = fields(value, ["ndcgAtK", "coverage", "p95LatencyMs"], field);
  return { ndcgAtK: fraction(row.ndcgAtK, `${field}.ndcgAtK`),
    coverage: fraction(row.coverage, `${field}.coverage`),
    p95LatencyMs: positive(row.p95LatencyMs, `${field}.p95LatencyMs`) };
}

function evidenceAudit(privateBytes: Map<string, Buffer>, review: ObjectValue, manifest: ReturnType<typeof parseReleaseManifest>,
  modelBytes: Buffer, syntheticFixture: boolean): Pick<ModelPromotionAuditV1, "promotionId" | "sourceName" |
    "numericArchiveSha256" | "modelCoverage" | "evidence" | "approvals"> {
  const required = ["format", "promotionId", "tag", "bundleId", "manifestSha256", "baseTag",
    "baseBundleId", "baseManifestSha256", "sourceName", "sourceDecisionRef", "datasetBridgeDecisionRef",
    "graphDatasetSha256", "rawContentSha256", "modelSha256", "numericArchiveSha256",
    "selectionFileSha256", "selectionSha256", "finalReportSha256", "refitRecordSha256",
    "datasetBridgeSha256", "qualityPolicySha256", "servingCohortSha256", "servingReportSha256",
    "minimumModelCoverage", "owner", "ownerApprovalRef",
    "trainingApprovalRef", "publicationApprovalRef", "deploymentApprovalRef"];
  fields(review, required, "review");
  same(review.format, "model-promotion-review-v2", "review.format");
  const promotionId = text(review.promotionId, "review.promotionId");
  const sourceName = text(review.sourceName, "review.sourceName");
  tag(review.tag, "review.tag");
  tag(review.baseTag, "review.baseTag");
  for (const key of ["bundleId", "manifestSha256", "baseBundleId", "baseManifestSha256",
    "graphDatasetSha256", "rawContentSha256", "modelSha256", "numericArchiveSha256",
    "selectionFileSha256", "selectionSha256", "finalReportSha256", "refitRecordSha256",
    "datasetBridgeSha256", "qualityPolicySha256", "servingCohortSha256",
    "servingReportSha256"] as const) {
    digest(review[key], `review.${key}`);
  }
  for (const key of ["sourceDecisionRef", "datasetBridgeDecisionRef"] as const) {
    decisionRef(review[key], `review.${key}`);
  }
  const minimumModelCoverage = fraction(review.minimumModelCoverage, "review.minimumModelCoverage");
  if (minimumModelCoverage <= 0) fail("review.minimumModelCoverage", "must be positive");
  const owner = text(review.owner, "review.owner");
  for (const key of ["ownerApprovalRef", "trainingApprovalRef", "publicationApprovalRef",
    "deploymentApprovalRef"] as const) https(review[key], `review.${key}`);

  same(review.tag, manifest.tag, "review.tag");
  same(review.bundleId, manifest.bundleId, "review.bundleId");
  same(review.graphDatasetSha256, manifest.dataset.sha256, "review.graphDatasetSha256");
  same(review.modelSha256, releaseSha256(modelBytes), "review.modelSha256");
  const model = parseCompactModel(json(modelBytes, RELEASE_FILES.model), RELEASE_FILES.model);
  if (model.sourceModel !== undefined && model.sourceModel !== "model.npz") {
    fail(`${RELEASE_FILES.model}.sourceModel`, "must name the private numeric archive without a path");
  }
  same(review.numericArchiveSha256, model.sourceModelSha256, "review.numericArchiveSha256");
  same(review.numericArchiveSha256, releaseSha256(privateBytes.get("model.npz")!),
    "model.npz");
  const coverage = manifest.model!.coverage.mappedAnimeCount /
    manifest.model!.coverage.totalCatalogAnimeCount;
  if (coverage < minimumModelCoverage) fail("review.minimumModelCoverage", "model map falls below the floor");

  const selection = readEvidence(privateBytes, "selection.json");
  const finalReport = readEvidence(privateBytes, "final-report.json");
  const reportDigest = fields(readEvidence(privateBytes, "final-report.sha256.json"),
    ["format", "selectionSha256", "reportSha256"], "final-report.sha256.json");
  const marker = fields(readEvidence(privateBytes, "selection.test-used.json"),
    ["format", "selectionSha256"], "selection.test-used.json");
  const refit = fields(readEvidence(privateBytes, "refit-record.json"),
    ["format", "selectionSha256", "finalReportSha256", "selectedCandidateId",
      "candidateSpecSha256", "rawContentSha256", "splitIdentitySha256", "metadataSha256",
      "fitMembership", "trainRows", "validationRows", "testRowsExcluded", "refitRows",
      "originalTrainSha256", "refitTrainSha256", "refitFitSha256", "refitModelSha256",
      "numericArchiveSha256", "webModelSha256", "evaluation", "releaseStatus"],
    "refit-record.json");
  const bridge = fields(readEvidence(privateBytes, "dataset-bridge.json"),
    ["format", "sourceName", "rawContentSha256", "graphDatasetSha256", "decisionRef", "reviewRef"],
    "dataset-bridge.json");
  const policy = fields(readEvidence(privateBytes, "quality-policy.json"),
    ["format", "decisionRef", "candidateBundleId", "cohortSha256", "baselineName",
      "seed", "suppliedCount", "positiveRawScoreMin", "topK", "minimumEligibleUsers",
      "minimumPositiveLabels", "minimumServingCoverage", "minimumNdcgLift",
      "maximumP95LatencyMs", "latencyWarmups", "latencySamples"], "quality-policy.json");
  const serving = fields(readEvidence(privateBytes, "serving-report.json"),
    ["format", "evaluator", "policySha256", "cohortSha256", "graphSha256",
      "modelSha256", "tag", "bundleId", "rawContentSha256", "graphDatasetSha256",
      "selectionSha256", "finalReportSha256", "topK", "suppliedCount",
      "baselineValidation", "validationUsers", "finalUsers", "eligibleUsers",
      "positiveLabels", "baselineName", "baseline", "model"], "serving-report.json");

  same(review.selectionFileSha256, releaseSha256(privateBytes.get("selection.json")!),
    "review.selectionFileSha256");
  same(selection.format, "split-first-selection-v1", "selection.format");
  same(selection.selectionSha256, review.selectionSha256, "selection.selectionSha256");
  same(selection.rawContentSha256, review.rawContentSha256, "selection.rawContentSha256");
  const selected = object(selection.selectedCandidate, "selection.selectedCandidate");
  const selectedId = text(selected.id, "selection.selectedCandidate.id");

  same(review.finalReportSha256, releaseSha256(privateBytes.get("final-report.json")!),
    "review.finalReportSha256");
  same(finalReport.format, "split-first-final-test-v1", "final-report.format");
  const finalStatus = text(finalReport.status, "final-report.status");
  if (syntheticFixture !== (finalStatus === "single synthetic warm-user report; no release claim")) {
    fail("final-report.status", "synthetic warm-user report cannot support a reviewed model package");
  }
  same(finalReport.selectionSha256, review.selectionSha256, "final-report.selectionSha256");
  same(finalReport.selectedCandidateId, selectedId, "final-report.selectedCandidateId");
  same(reportDigest.format, "split-first-final-test-digest-v1", "final-report.sha256.json.format");
  same(reportDigest.selectionSha256, review.selectionSha256, "final-report.sha256.json.selectionSha256");
  same(reportDigest.reportSha256, review.finalReportSha256, "final-report.sha256.json.reportSha256");
  same(marker.format, "split-first-test-used-v1", "selection.test-used.json.format");
  same(marker.selectionSha256, review.selectionSha256, "selection.test-used.json.selectionSha256");

  same(review.refitRecordSha256, releaseSha256(privateBytes.get("refit-record.json")!),
    "review.refitRecordSha256");
  same(refit.format, "split-first-final-refit-v1", "refit-record.format");
  same(refit.releaseStatus, "unapproved experiment artifact", "refit-record.releaseStatus");
  same(refit.evaluation, "none; this record contains no held-out quality metric",
    "refit-record.evaluation");
  same(refit.fitMembership, "train-plus-validation", "refit-record.fitMembership");
  same(refit.selectionSha256, review.selectionSha256, "refit-record.selectionSha256");
  same(refit.finalReportSha256, review.finalReportSha256, "refit-record.finalReportSha256");
  same(refit.selectedCandidateId, selectedId, "refit-record.selectedCandidateId");
  same(refit.rawContentSha256, review.rawContentSha256, "refit-record.rawContentSha256");
  same(refit.numericArchiveSha256, review.numericArchiveSha256, "refit-record.numericArchiveSha256");
  same(refit.webModelSha256, review.modelSha256, "refit-record.webModelSha256");
  const train = positiveInt(refit.trainRows, "refit-record.trainRows");
  const validation = positiveInt(refit.validationRows, "refit-record.validationRows");
  positiveInt(refit.testRowsExcluded, "refit-record.testRowsExcluded");
  same(refit.refitRows, train + validation, "refit-record.refitRows");
  for (const key of ["candidateSpecSha256", "splitIdentitySha256", "metadataSha256",
    "originalTrainSha256", "refitTrainSha256", "refitFitSha256", "refitModelSha256"] as const) {
    digest(refit[key], `refit-record.${key}`);
  }

  same(review.datasetBridgeSha256, releaseSha256(privateBytes.get("dataset-bridge.json")!),
    "review.datasetBridgeSha256");
  same(bridge.format, "model-dataset-bridge-v1", "dataset-bridge.format");
  same(bridge.sourceName, sourceName, "dataset-bridge.sourceName");
  same(bridge.rawContentSha256, review.rawContentSha256, "dataset-bridge.rawContentSha256");
  same(bridge.graphDatasetSha256, review.graphDatasetSha256, "dataset-bridge.graphDatasetSha256");
  same(bridge.decisionRef, review.datasetBridgeDecisionRef, "dataset-bridge.decisionRef");
  https(bridge.reviewRef, "dataset-bridge.reviewRef");

  same(review.qualityPolicySha256, releaseSha256(privateBytes.get("quality-policy.json")!),
    "review.qualityPolicySha256");
  same(review.servingCohortSha256, releaseSha256(privateBytes.get("serving-cohort.json")!),
    "review.servingCohortSha256");
  same(policy.format, "model-promotion-quality-policy-v1", "quality-policy.format");
  decisionRef(policy.decisionRef, "quality-policy.decisionRef");
  text(policy.baselineName, "quality-policy.baselineName");
  same(policy.candidateBundleId, manifest.bundleId, "quality-policy.candidateBundleId");
  same(policy.cohortSha256, review.servingCohortSha256, "quality-policy.cohortSha256");
  positiveInt(policy.suppliedCount, "quality-policy.suppliedCount");
  positiveInt(policy.positiveRawScoreMin, "quality-policy.positiveRawScoreMin");
  positiveInt(policy.topK, "quality-policy.topK");
  positiveInt(policy.minimumEligibleUsers, "quality-policy.minimumEligibleUsers");
  positiveInt(policy.minimumPositiveLabels, "quality-policy.minimumPositiveLabels");
  const minimumNdcgLift = positive(policy.minimumNdcgLift, "quality-policy.minimumNdcgLift");
  const maximumP95LatencyMs = positive(policy.maximumP95LatencyMs,
    "quality-policy.maximumP95LatencyMs");
  positiveInt(policy.latencySamples, "quality-policy.latencySamples");
  fraction(policy.minimumServingCoverage, "quality-policy.minimumServingCoverage");

  same(review.servingReportSha256, releaseSha256(privateBytes.get("serving-report.json")!),
    "review.servingReportSha256");
  same(serving.format, "model-serving-evaluation-v1", "serving-report.format");
  same(serving.evaluator, "browser-preference-eligibility-selector-v1",
    "serving-report.evaluator");
  same(serving.policySha256, review.qualityPolicySha256, "serving-report.policySha256");
  same(serving.cohortSha256, review.servingCohortSha256, "serving-report.cohortSha256");
  same(serving.graphSha256, manifest.neighborhood.sha256, "serving-report.graphSha256");
  same(serving.modelSha256, review.modelSha256, "serving-report.modelSha256");
  for (const key of ["tag", "bundleId", "rawContentSha256", "graphDatasetSha256",
    "selectionSha256", "finalReportSha256"] as const) {
    same(serving[key], review[key], `serving-report.${key}`);
  }
  same(serving.topK, policy.topK, "serving-report.topK");
  same(serving.suppliedCount, policy.suppliedCount, "serving-report.suppliedCount");
  if (positiveInt(serving.eligibleUsers, "serving-report.eligibleUsers") <
      (policy.minimumEligibleUsers as number) ||
      positiveInt(serving.positiveLabels, "serving-report.positiveLabels") <
      (policy.minimumPositiveLabels as number)) {
    fail("serving-report", "has too few eligible users or positive labels");
  }
  same(serving.baselineName, policy.baselineName, "serving-report.baselineName");
  const baseline = modelMetrics(serving.baseline, "serving-report.baseline");
  const measured = modelMetrics(serving.model, "serving-report.model");
  if (measured.ndcgAtK < baseline.ndcgAtK + minimumNdcgLift ||
      measured.coverage < (policy.minimumServingCoverage as number) ||
      measured.p95LatencyMs > maximumP95LatencyMs) {
    fail("serving-report", "does not meet the reviewed quality, coverage, and latency floors");
  }

  return { promotionId, sourceName, numericArchiveSha256: review.numericArchiveSha256 as string,
    modelCoverage: coverage,
    evidence: { reviewSha256: releaseSha256(privateBytes.get("model-promotion-review.json")!),
      selectionFileSha256: review.selectionFileSha256 as string,
      finalReportSha256: review.finalReportSha256 as string,
      refitRecordSha256: review.refitRecordSha256 as string,
      datasetBridgeSha256: review.datasetBridgeSha256 as string,
      qualityPolicySha256: review.qualityPolicySha256 as string,
      servingCohortSha256: review.servingCohortSha256 as string,
      servingReportSha256: review.servingReportSha256 as string },
    approvals: { owner, ownerApprovalRef: review.ownerApprovalRef as string,
      trainingApprovalRef: review.trainingApprovalRef as string,
      publicationApprovalRef: review.publicationApprovalRef as string,
      deploymentApprovalRef: review.deploymentApprovalRef as string } };
}

function outputPath(options: ModelPackageOptions): string {
  const output = path.resolve(options.outputDir);
  const parent = path.dirname(output);
  if (!fs.existsSync(parent) || !fs.lstatSync(parent).isDirectory() ||
      fs.lstatSync(parent).isSymbolicLink() || pathExistsNoFollow(output)) {
    fail("outputDir", "requires an unused path under a real parent directory");
  }
  const realParent = fs.realpathSync(parent);
  for (const input of [options.candidateDir, options.baseDir, options.evidenceDir]) {
    const realInput = fs.realpathSync(input);
    const relative = path.relative(realInput, realParent);
    if (relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." &&
        !path.isAbsolute(relative))) fail("outputDir", "cannot be inside an input directory");
  }
  return output;
}

/** Build only a local package. An independent approval verifier must run before any publication. */
export function packageModelRelease(options: ModelPackageOptions): ModelPromotionAuditV1 {
  const candidateBytes = inventory(options.candidateDir, MODEL_SOURCE_FILES, "candidate");
  const baseNames = fs.existsSync(path.join(options.baseDir, "publication-audit.json"))
    ? [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
      RELEASE_FILES.catalog, "publication-audit.json"]
    : [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
      RELEASE_FILES.catalog];
  const baseBytes = inventory(options.baseDir, baseNames, "base");
  const privateBytes = inventory(options.evidenceDir, MODEL_PRIVATE_FILES, "evidence");
  const manifest = verifyReleaseBundle(options.candidateDir, options.baseDir);
  for (const [name, captured] of candidateBytes) {
    if (!captured.equals(fs.readFileSync(path.join(options.candidateDir, name)))) {
      fail(`candidate.${name}`, "changed during verification");
    }
  }
  for (const [name, captured] of baseBytes) {
    if (!captured.equals(fs.readFileSync(path.join(options.baseDir, name)))) {
      fail(`base.${name}`, "changed during verification");
    }
  }
  const base = parseReleaseManifest(json(baseBytes.get(RELEASE_FILES.manifest)!,
    "base.release-manifest.json"), "base.release-manifest.json");
  if (manifest.neighborhood.format !== "graph-compact-v3" ||
      manifest.explorer.format !== "graph-compact-v3" || !manifest.model ||
      base.neighborhood.format !== "graph-compact-v3" || base.explorer.format !== "graph-compact-v3" ||
      base.model !== null) fail("release-manifest.json", "requires a v3 item-model candidate and data-only v3 base");
  for (const name of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer, RELEASE_FILES.catalog]) {
    if (!candidateBytes.get(name)!.equals(baseBytes.get(name)!)) {
      fail(name, "must be byte-for-byte identical to the approved data base");
    }
  }
  const review = readEvidence(privateBytes, "model-promotion-review.json");
  same(review.manifestSha256, releaseSha256(candidateBytes.get(RELEASE_FILES.manifest)!),
    "review.manifestSha256");
  same(review.baseTag, base.tag, "review.baseTag");
  same(review.baseBundleId, base.bundleId, "review.baseBundleId");
  same(review.baseManifestSha256, releaseSha256(baseBytes.get(RELEASE_FILES.manifest)!),
    "review.baseManifestSha256");
  const evidence = evidenceAudit(privateBytes, review, manifest,
    candidateBytes.get(RELEASE_FILES.model)!, options.syntheticFixture === true);
  same(evidence.sourceName, manifest.dataset.source, "review.sourceName");
  try {
    execFileSync(process.execPath, ["--import", "tsx",
      fileURLToPath(new URL("../../web/bench/model-promotion-serving-evaluation.ts", import.meta.url)),
      path.resolve(options.candidateDir), path.resolve(options.evidenceDir)],
    { cwd: path.resolve(import.meta.dirname, "../.."), encoding: "utf8",
      maxBuffer: 1024 * 1024, timeout: 30_000, stdio: ["ignore", "pipe", "pipe"] });
  } catch {
    fail("serving-report.json", "browser scorer recomputation or frozen quality gate failed");
  }
  const unchanged = (directory: string, names: readonly string[], captured: Map<string, Buffer>,
    field: string) => {
    const current = inventory(directory, names, field);
    for (const name of names) {
      if (!current.get(name)!.equals(captured.get(name)!)) {
        fail(`${field}.${name}`, "changed during serving evaluation");
      }
    }
  };
  unchanged(options.candidateDir, MODEL_SOURCE_FILES, candidateBytes, "candidate");
  unchanged(options.baseDir, baseNames, baseBytes, "base");
  unchanged(options.evidenceDir, MODEL_PRIVATE_FILES, privateBytes, "evidence");
  const synthetic = options.syntheticFixture === true;
  if (synthetic !== (evidence.sourceName === "synthetic-fixture")) {
    fail("syntheticFixture", "must match the invented source and cannot mark it publishable");
  }
  if (synthetic !== (manifest.dataset.source === "synthetic-fixture")) {
    fail("release-manifest.json.dataset.source", "must match syntheticFixture");
  }
  const audit: ModelPromotionAuditV1 = {
    format: "model-promotion-audit-v1", status: synthetic ? "synthetic-only" : "pending-approval",
    promotionId: evidence.promotionId, tag: manifest.tag, bundleId: manifest.bundleId,
    manifestSha256: releaseSha256(candidateBytes.get(RELEASE_FILES.manifest)!),
    base: { tag: base.tag, bundleId: base.bundleId,
      manifestSha256: releaseSha256(baseBytes.get(RELEASE_FILES.manifest)!) },
    datasetSha256: manifest.dataset.sha256, sourceName: evidence.sourceName,
    modelSha256: manifest.model.sha256, numericArchiveSha256: evidence.numericArchiveSha256,
    modelCoverage: evidence.modelCoverage, evidence: evidence.evidence,
    approvals: evidence.approvals,
    assets: MODEL_SOURCE_FILES.map((name) => ({ path: name,
      bytes: candidateBytes.get(name)!.length, sha256: releaseSha256(candidateBytes.get(name)!) })),
  };

  const output = outputPath(options);
  const parent = path.dirname(output);
  let stage: string | null = fs.mkdtempSync(path.join(parent, ".model-package-"));
  try {
    for (const name of MODEL_SOURCE_FILES) {
      fs.writeFileSync(path.join(stage, name), candidateBytes.get(name)!, { flag: "wx" });
    }
    fs.writeFileSync(path.join(stage, "model-promotion-audit.json"),
      `${JSON.stringify(audit, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    inventory(stage, MODEL_OUTPUT_FILES, "staging");
    options.beforeActivate?.();
    if (pathExistsNoFollow(output)) fail("outputDir", "was created during packaging");
    fs.renameSync(stage, output);
    stage = null;
    return audit;
  } finally {
    if (stage) {
      const realStage = fs.realpathSync(stage);
      if (path.dirname(realStage) !== fs.realpathSync(parent) ||
          !path.basename(realStage).startsWith(".model-package-")) {
        fail("staging", "refusing cleanup outside the output parent");
      }
      fs.rmSync(stage, { recursive: true, force: true });
    }
  }
}
