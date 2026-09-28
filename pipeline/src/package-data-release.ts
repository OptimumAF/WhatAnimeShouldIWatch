import fs from "node:fs";
import path from "node:path";
import {
  parseCompactGraph, parseReleaseManifest, RELEASE_BUNDLE_LIMITS, type CompactGraphDataV3,
  type ReleaseManifestV1,
} from "../../web/src/artifacts.js";
import { getRepoRoot } from "./paths.js";
import { RELEASE_FILES, releaseSha256, verifyReleaseBundle } from "./release-manifest.js";

const repoRoot = getRepoRoot(import.meta.url);
const SOURCE_FILES = [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood,
  RELEASE_FILES.explorer, RELEASE_FILES.catalog] as const;
const OUTPUT_FILES = [...SOURCE_FILES, "publication-audit.json"] as const;
const PUBLIC_FIELDS = ["anime.id", "anime.title", "pair.weight", "pair.support",
  "graph.provenance"] as const;
const QUALITY_IDS = ["schema", "privacy", "coverage"] as const;

type QualityId = typeof QUALITY_IDS[number];

export interface PublicationReviewV1 {
  format: "publication-review-v1";
  tag: string;
  bundleId: string;
  manifestSha256: string;
  source: {
    name: string;
    datasetSha256: string;
    derivation: string;
    decisionRef: string;
  };
  redistribution: {
    status: "synthetic-only" | "reviewed-allowed";
    basis: string;
    approvalRef: string | null;
    owner: string;
    allowedFields: string[];
    attribution: string;
    deletionCorrection: string;
  };
  changes: { previousTag: string | null; summary: string };
  quality: { checks: { id: QualityId; evidenceRef: string }[] };
}

/** The caller must validate this independently against decision 0002 first. */
export interface ValidatedPublicationApproval {
  scope: "publication";
  approved: true;
  approvalRef: string;
  decisionRef: string;
  owner: string;
  sources: string[];
}

/** A separate, exact first-real-bundle approval; the workflow validates its registry source. */
export interface ValidatedFirstBundleApproval {
  scope: "first-real-bundle";
  approved: true;
  tag: string;
  bundleId: string;
  manifestSha256: string;
  decisionRef: string;
  approvalRef: string;
  owner: string;
}

export interface PackageDataReleaseOptions {
  candidateDir: string;
  previousDir?: string;
  outputDir: string;
  review: unknown;
  fixtureGenesis?: boolean;
  approval?: ValidatedPublicationApproval;
  bootstrapApproval?: ValidatedFirstBundleApproval;
  /** Test seam for failure after complete staging but before output activation. */
  beforeActivate?: () => void;
}

export interface PublicationAuditV1 {
  format: "publication-audit-v1";
  publishable: boolean;
  tag: string;
  bundleId: string;
  manifestSha256: string;
  source: PublicationReviewV1["source"];
  redistribution: PublicationReviewV1["redistribution"];
  changes: PublicationReviewV1["changes"];
  quality: PublicationReviewV1["quality"] & {
    computed: { animeCount: number; pairCount: number; positivePairCount: number;
      minimumPairSupport: number | null };
  };
  compatibility: { manifestFormat: "release-manifest-v1";
    recommendationGraphFormat: "graph-compact-v3"; explorerGraphFormat: "graph-compact-v3";
    model: null; datasetSha256: string; catalogItemMapSha256: string;
    neighborhoodGraphId: string; explorerGraphId: string };
  assets: { path: string; bytes: number; sha256: string }[];
}

function fail(field: string, reason: string): never {
  throw new Error(`Publication package ${field}: ${reason}`);
}

function object(value: unknown, fields: readonly string[], label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(label, "must be an object");
  const record = value as Record<string, unknown>;
  for (const field of fields) if (!Object.hasOwn(record, field)) fail(`${label}.${field}`, "is required");
  for (const field of Object.keys(record)) if (!fields.includes(field)) fail(`${label}.${field}`, "is unsupported");
  return record;
}

function text(value: unknown, label: string, maximum = 300): string {
  if (typeof value !== "string" || !value.trim() || value.length > maximum ||
      /[\u0000-\u001f\u007f]/.test(value)) fail(label, "must be bounded nonempty text");
  return value;
}

function digest(value: unknown, label: string): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {
    fail(label, "must be a lowercase SHA-256 digest");
  }
  return value;
}

function parseReview(value: unknown): PublicationReviewV1 {
  const review = object(value, ["format", "tag", "bundleId", "manifestSha256", "source",
    "redistribution", "changes", "quality"], "review");
  if (review.format !== "publication-review-v1") fail("review.format", "is unsupported");
  if (typeof review.tag !== "string" ||
      !/^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(review.tag)) {
    fail("review.tag", "must be a versioned data-v tag");
  }
  digest(review.bundleId, "review.bundleId");
  digest(review.manifestSha256, "review.manifestSha256");
  const source = object(review.source, ["name", "datasetSha256", "derivation", "decisionRef"],
    "review.source");
  text(source.name, "review.source.name", 120);
  digest(source.datasetSha256, "review.source.datasetSha256");
  text(source.derivation, "review.source.derivation");
  const decisionRef = text(source.decisionRef, "review.source.decisionRef", 180);
  if (!/^docs\/decisions\/[0-9]{4}-[a-z0-9-]+\.md$/.test(decisionRef) ||
      !fs.existsSync(path.join(repoRoot, decisionRef))) {
    fail("review.source.decisionRef", "must name an existing decision file");
  }
  const rights = object(review.redistribution, ["status", "basis", "approvalRef", "owner",
    "allowedFields", "attribution", "deletionCorrection"], "review.redistribution");
  if (rights.status !== "synthetic-only" && rights.status !== "reviewed-allowed") {
    fail("review.redistribution.status", "is unsupported");
  }
  for (const field of ["basis", "owner", "attribution", "deletionCorrection"]) {
    text(rights[field], `review.redistribution.${field}`);
  }
  if (!Array.isArray(rights.allowedFields) ||
      JSON.stringify(rights.allowedFields) !== JSON.stringify(PUBLIC_FIELDS)) {
    fail("review.redistribution.allowedFields", "must exactly list the public fields in contract order");
  }
  if (rights.status === "synthetic-only") {
    if (rights.approvalRef !== null) fail("review.redistribution.approvalRef", "must be null for synthetic-only");
  } else if (typeof rights.approvalRef !== "string" ||
             !/^https:\/\/[^\s/]+\/\S+$/.test(rights.approvalRef)) {
    fail("review.redistribution.approvalRef", "must be an HTTPS review URL");
  }
  const changes = object(review.changes, ["previousTag", "summary"], "review.changes");
  if (changes.previousTag !== null &&
      (typeof changes.previousTag !== "string" ||
       !/^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(changes.previousTag))) {
    fail("review.changes.previousTag", "must be a versioned tag or null");
  }
  text(changes.summary, "review.changes.summary");
  const quality = object(review.quality, ["checks"], "review.quality");
  if (!Array.isArray(quality.checks) || quality.checks.length !== QUALITY_IDS.length) {
    fail("review.quality.checks", "must contain schema, privacy, and coverage evidence");
  }
  const checks = quality.checks.map((value, index) => {
    const check = object(value, ["id", "evidenceRef"], `review.quality.checks[${index}]`);
    if (check.id !== QUALITY_IDS[index]) {
      fail(`review.quality.checks[${index}].id`, `must be ${QUALITY_IDS[index]}`);
    }
    text(check.evidenceRef, `review.quality.checks[${index}].evidenceRef`, 300);
    return check;
  });
  return { ...review, source, redistribution: rights, changes,
    quality: { checks } } as PublicationReviewV1;
}

function exactInventory(directory: string, expected: readonly string[], label: string): void {
  if (!fs.existsSync(directory)) fail(label, "directory is missing");
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) fail(label, "must be a real directory");
  const entries = fs.readdirSync(directory, { withFileTypes: true });
  const names = entries.map((entry) => entry.name).sort();
  if (JSON.stringify(names) !== JSON.stringify([...expected].sort())) {
    fail(label, `inventory differs from ${expected.join(", ")}`);
  }
  for (const entry of entries) {
    if (!entry.isFile() || entry.isSymbolicLink()) {
      fail(`${label}.${entry.name}`, "must be a regular file, not a link or directory");
    }
  }
}

function pathExistsNoFollow(filepath: string): boolean {
  try {
    fs.lstatSync(filepath);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

function assertBoundedInput(directory: string, includeModel: boolean): void {
  const limits = RELEASE_BUNDLE_LIMITS;
  let total = 0;
  for (const filename of [...SOURCE_FILES,
    ...(includeModel && fs.existsSync(path.join(directory, RELEASE_FILES.model))
      ? [RELEASE_FILES.model] : [])]) {
    const filepath = path.join(directory, filename);
    if (!fs.existsSync(filepath)) fail(filename, "is missing");
    const stat = fs.lstatSync(filepath);
    const maximum = filename === RELEASE_FILES.manifest ? limits.manifestBytes : limits.plainAssetBytes;
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > maximum) {
      fail(filename, "is empty, linked, or exceeds the byte limit");
    }
    if (filename !== RELEASE_FILES.manifest) total += stat.size;
  }
  if (total > limits.totalPlainBytes) fail("bundle", "total plain bytes exceed the limit");
}

function boundedBytes(directory: string, manifest: ReleaseManifestV1): Map<string, Buffer> {
  const limits = RELEASE_BUNDLE_LIMITS;
  const files = new Map<string, Buffer>();
  const expectedSizes = new Map<string, number>([
    [RELEASE_FILES.neighborhood, manifest.neighborhood.bytes],
    [RELEASE_FILES.explorer, manifest.explorer.bytes],
    [RELEASE_FILES.catalog, manifest.catalog.bytes],
  ]);
  const expectedHashes = new Map<string, string>([
    [RELEASE_FILES.neighborhood, manifest.neighborhood.sha256],
    [RELEASE_FILES.explorer, manifest.explorer.sha256],
    [RELEASE_FILES.catalog, manifest.catalog.sha256],
  ]);
  let total = 0;
  for (const filename of SOURCE_FILES) {
    const filepath = path.join(directory, filename);
    const stat = fs.lstatSync(filepath);
    const maximum = filename === RELEASE_FILES.manifest ? limits.manifestBytes : limits.plainAssetBytes;
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > maximum) {
      fail(filename, "is empty, linked, or exceeds the byte limit");
    }
    if (expectedSizes.has(filename) && stat.size !== expectedSizes.get(filename)) {
      fail(filename, "byte length differs from the manifest");
    }
    if (filename !== RELEASE_FILES.manifest) total += stat.size;
    const content = fs.readFileSync(filepath);
    if (expectedHashes.has(filename) && releaseSha256(content) !== expectedHashes.get(filename)) {
      fail(filename, "bytes changed after verification or differ from the manifest hash");
    }
    files.set(filename, content);
  }
  if (total > limits.totalPlainBytes) fail("bundle", "total plain bytes exceed the limit");
  return files;
}

function assertApproval(review: PublicationReviewV1, options: PackageDataReleaseOptions): boolean {
  if (review.redistribution.status === "synthetic-only") {
    if (!options.fixtureGenesis || options.approval || options.bootstrapApproval ||
        review.source.name !== "synthetic-fixture") {
      fail("review.redistribution", "synthetic-only requires an invented fixture without approval");
    }
    return false;
  }
  if (options.fixtureGenesis) fail("fixtureGenesis", "cannot create a publishable genesis bundle");
  if (review.source.name === "synthetic-fixture") {
    fail("review.source.name", "invented fixture cannot be marked publishable");
  }
  const approval = options.approval;
  if (!approval || approval.scope !== "publication" || approval.approved !== true ||
      approval.approvalRef !== review.redistribution.approvalRef ||
      approval.decisionRef !== review.source.decisionRef ||
      approval.owner !== review.redistribution.owner ||
      !Array.isArray(approval.sources) || !approval.sources.includes(review.source.name)) {
    fail("approval", "must match a separately validated publication approval");
  }
  if (options.previousDir) {
    if (options.bootstrapApproval) fail("bootstrapApproval", "cannot accompany a previous bundle");
  } else {
    const bootstrap = options.bootstrapApproval;
    if (!bootstrap || bootstrap.scope !== "first-real-bundle" ||
        bootstrap.approved !== true || bootstrap.tag !== review.tag ||
        bootstrap.bundleId !== review.bundleId ||
        bootstrap.manifestSha256 !== review.manifestSha256 ||
        bootstrap.owner !== review.redistribution.owner ||
        bootstrap.approvalRef === approval.approvalRef ||
        !/^https:\/\/[^\s/]+\/\S+$/.test(bootstrap.approvalRef) ||
        !/^docs\/decisions\/[0-9]{4}-[a-z0-9-]+\.md$/.test(bootstrap.decisionRef) ||
        !fs.existsSync(path.join(repoRoot, bootstrap.decisionRef)) ||
        review.changes.previousTag !== null) {
      fail("bootstrapApproval", "requires a separate exact first-real-bundle approval");
    }
  }
  return true;
}

function removeStage(stage: string, parent: string): void {
  if (!fs.existsSync(stage)) return;
  const resolvedParent = fs.realpathSync(parent);
  const resolvedStage = fs.realpathSync(stage);
  if (path.dirname(resolvedStage) !== resolvedParent ||
      !path.basename(resolvedStage).startsWith(".publication-") ||
      fs.lstatSync(stage).isSymbolicLink()) {
    fail("staging", "refusing cleanup outside the output parent");
  }
  fs.rmSync(stage, { recursive: true, force: true });
}

/** Build a local package only. This function never calls GitHub or publishes it. */
export function packageDataRelease(options: PackageDataReleaseOptions): PublicationAuditV1 {
  const review = parseReview(options.review);
  const publishable = assertApproval(review, options);
  const candidateDir = path.resolve(options.candidateDir);
  exactInventory(candidateDir, SOURCE_FILES, "candidate");
  assertBoundedInput(candidateDir, false);
  if (options.previousDir) {
    const previousDir = path.resolve(options.previousDir);
    if (!fs.existsSync(previousDir) || !fs.lstatSync(previousDir).isDirectory() ||
        fs.lstatSync(previousDir).isSymbolicLink()) {
      fail("previousDir", "must be a real directory");
    }
    assertBoundedInput(previousDir, true);
  }
  const manifest = verifyReleaseBundle(candidateDir, options.previousDir,
    options.fixtureGenesis === true, options.bootstrapApproval !== undefined);
  if (publishable && options.previousDir) {
    const prior = parseReleaseManifest(JSON.parse(fs.readFileSync(path.join(options.previousDir,
      RELEASE_FILES.manifest), "utf8")), "previous.release-manifest.json");
    if (prior.dataset.source === "synthetic-fixture" ||
        prior.neighborhood.format !== "graph-compact-v3" ||
        prior.explorer.format !== "graph-compact-v3" || prior.model !== null) {
      fail("previousDir", "requires a reviewed data-only v3 predecessor");
    }
  }
  if (manifest.neighborhood.format !== "graph-compact-v3" ||
      manifest.explorer.format !== "graph-compact-v3" || manifest.model !== null) {
    fail("release-manifest.json", "requires a data-only v3 recommendation and explorer pair");
  }
  const bytes = boundedBytes(candidateDir, manifest);
  const manifestSha256 = releaseSha256(bytes.get(RELEASE_FILES.manifest)!);
  if (review.tag !== manifest.tag || review.bundleId !== manifest.bundleId ||
      review.manifestSha256 !== manifestSha256) {
    fail("review", "tag, bundle ID, or manifest-byte hash differs from verified candidate");
  }
  if (review.source.name !== manifest.dataset.source ||
      review.source.datasetSha256 !== manifest.dataset.sha256) {
    fail("review.source", "does not match manifest dataset provenance");
  }
  if (review.changes.previousTag !== (manifest.lastKnownGood?.tag ?? null)) {
    fail("review.changes.previousTag", "does not match the verified prior bundle");
  }
  const graph = parseCompactGraph(JSON.parse(bytes.get(RELEASE_FILES.neighborhood)!.toString("utf8")),
    RELEASE_FILES.neighborhood, "recommendation") as CompactGraphDataV3;
  let minimumPairSupport: number | null = null;
  let positivePairCount = 0;
  for (const edge of graph.aa) {
    minimumPairSupport = minimumPairSupport === null ? edge[3] : Math.min(minimumPairSupport, edge[3]);
    if (edge[2] > 0) positivePairCount++;
  }
  const audit: PublicationAuditV1 = {
    format: "publication-audit-v1", publishable,
    tag: manifest.tag, bundleId: manifest.bundleId, manifestSha256,
    source: review.source, redistribution: review.redistribution, changes: review.changes,
    quality: { checks: review.quality.checks,
      computed: { animeCount: graph.anime.length, pairCount: graph.aa.length,
        positivePairCount, minimumPairSupport } },
    compatibility: { manifestFormat: manifest.format,
      recommendationGraphFormat: "graph-compact-v3", explorerGraphFormat: "graph-compact-v3",
      model: null, datasetSha256: manifest.dataset.sha256,
      catalogItemMapSha256: manifest.catalog.itemMapSha256,
      neighborhoodGraphId: manifest.neighborhood.graphId,
      explorerGraphId: manifest.explorer.graphId },
    assets: SOURCE_FILES.map((filename) => {
      const content = bytes.get(filename)!;
      return { path: filename, bytes: content.length, sha256: releaseSha256(content) };
    }),
  };

  const outputDir = path.resolve(options.outputDir);
  for (const sourceDir of [candidateDir, ...(options.previousDir ? [path.resolve(options.previousDir)] : [])]) {
    const relative = path.relative(sourceDir, outputDir);
    if (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative)) {
      fail("outputDir", "must be outside the candidate and previous bundle directories");
    }
  }
  const parent = path.dirname(outputDir);
  if (!pathExistsNoFollow(parent)) fail("outputDir", "parent directory must already exist");
  const parentStat = fs.lstatSync(parent);
  if (!parentStat.isDirectory() || parentStat.isSymbolicLink()) {
    fail("outputDir", "parent must be a real directory");
  }
  const parentReal = fs.realpathSync(parent);
  for (const sourceDir of [candidateDir, ...(options.previousDir ? [path.resolve(options.previousDir)] : [])]) {
    const relative = path.relative(fs.realpathSync(sourceDir), parentReal);
    if (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative)) {
      fail("outputDir", "parent resolves inside a source bundle directory");
    }
  }
  if (pathExistsNoFollow(outputDir)) {
    fail("outputDir", "requires an unused path under a real parent directory");
  }
  let stage: string | null = fs.mkdtempSync(path.join(parent, ".publication-"));
  try {
    for (const filename of SOURCE_FILES) {
      fs.writeFileSync(path.join(stage, filename), bytes.get(filename)!, { flag: "wx" });
    }
    fs.writeFileSync(path.join(stage, "publication-audit.json"),
      `${JSON.stringify(audit, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    exactInventory(stage, OUTPUT_FILES, "staging");
    options.beforeActivate?.();
    if (pathExistsNoFollow(outputDir)) fail("outputDir", "was created during packaging");
    fs.renameSync(stage, outputDir);
    stage = null;
    return audit;
  } finally {
    if (stage) removeStage(stage, parent);
  }
}

export { PUBLIC_FIELDS, SOURCE_FILES, OUTPUT_FILES };
