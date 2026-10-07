/** Browser artifact contracts. Validate JSON before it enters ranking or rendering. */

export type NodeType = "user" | "anime";
export type EdgeType = "user-anime" | "anime-anime";

export interface GraphNode {
  id: string;
  label: string;
  nodeType: NodeType;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  edgeType: EdgeType;
  weight: number;
  /** Co-raters in the producer's processed rows; a v1 edge set may be selected/capped. */
  support?: number;
}

export interface GraphDataV1 {
  generatedAt: string;
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
  nodes: GraphNode[];
  /** User-anime and anime-anime edges may both use the graph builder's selected rating subset. */
  edges: GraphEdge[];
}

export interface GraphV2Metadata {
  role: "recommendation" | "visualization";
  graphId: string;
  sourceGraphId?: string;
  dataset: { sha256: string; scope: "anonymized-ratings-content-v1"; source: string };
  semantics: {
    pairWeight: "centered-pair-preference-mean-v1";
    support: "co-raters-after-selection-v1";
    recommendationUse: "positive-only-v1";
  };
  config: {
    ratingSelectionPolicy: "all-ratings" | "sha256-bottom-k-v1";
    seed: number;
    maxRatingsPerUser: number;
    maxAnimeAnimeEdges: number;
    maxPairVisits: number;
    maxPairCandidates: number;
    minPairSupport: number;
    maxNeighborsPerAnime: number;
  };
  truncation: {
    inputRatings: number;
    selectedRatings: number;
    ratingsSkipped: number;
    potentialPairVisits: number;
    pairVisits: number;
    pairVisitsSkipped: number;
    candidatePairs: number;
    eligiblePairs: number;
    selectedPairs: number;
    excludedBySupport: number;
    excludedByNeighborLimit: number;
    excludedByOutputLimit: number;
  };
  visualization?: {
    policy: "abs-weight-top-k-v1";
    maxUserAnimeEdges: number;
    maxAnimeAnimeEdges: number;
    excludedUserAnimeEdges: number;
    excludedAnimeAnimeEdges: number;
  };
}

export interface GraphDataV2 extends GraphDataV1, GraphV2Metadata {
  format: "graph-legacy-v2";
  role: "recommendation";
}

export type GraphData = GraphDataV1 | GraphDataV2;

export type CompactAnimeEntry = [animeId: number, title: string];
export type CompactUserAnimeEdge = [userIndex: number, animeIndex: number, weight: number];
export type CompactAnimeAnimeEdge = [
  leftAnimeIndex: number,
  rightAnimeIndex: number,
  weight: number,
  support?: number, // co-raters in processed rows, not proof that all pair keys were exported
];

export interface CompactGraphDataV1 {
  format: "graph-compact-v1";
  generatedAt: string;
  userIds: string[];
  anime: CompactAnimeEntry[];
  /** Can be a seeded per-user subset of the separate full ratings dataset. */
  ua: CompactUserAnimeEdge[];
  aa: CompactAnimeAnimeEdge[];
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
}

export interface CompactGraphDataV2 extends Omit<CompactGraphDataV1, "format" | "aa">, GraphV2Metadata {
  format: "graph-compact-v2";
  aa: [leftAnimeIndex: number, rightAnimeIndex: number, weight: number, support: number][];
}

export interface CompactGraphDataV3 extends Omit<CompactGraphDataV2, "format"> {
  format: "graph-compact-v3";
  projection: { policy: "omit-user-anime-v1" };
}

export type CompactGraphData = CompactGraphDataV1 | CompactGraphDataV2 | CompactGraphDataV3;

export type LoadedGraphData = GraphData | CompactGraphData;

export interface AnimeRelation {
  kind: "prequel" | "sequel" | "alternative-version" | "side-story" | "spin-off";
  animeId: number;
  title: string;
}

export interface AnimeMetadata {
  animeId: number;
  /** Only titles already supplied by the loaded catalog or metadata response. */
  aliases?: string[];
  mediaFormat?: string | null;
  year: number | null;
  score: number | null;
  genres: string[];
  studios: string[];
  synopsis: string;
  imageUrl: string;
  season: string | null;
  /** Missing or null means unchecked; even [] is not proof that prerequisites do not exist. */
  relations?: AnimeRelation[] | null;
  /** Optional fields from a verified bundled metadata snapshot. */
  episodeCount?: number | null;
  runtimeMinutes?: number | null;
  contentClassification?: { jurisdiction: string; system: string; value: string } | null;
}

export interface DemoCatalogItem extends AnimeMetadata {
  title: string;
}

export interface ModelRecommendationAnime {
  animeId: number;
  title: string;
  bias: number;
  embedding: number[];
}

export interface ModelRecommendationData {
  generatedAt: string;
  sourceModelSha256?: string;
  globalMean: number;
  factors: number;
  anime: ModelRecommendationAnime[];
}

export interface CompactModelRecommendationData {
  format: "model-mf-compact-v1";
  generatedAt: string;
  sourceModel?: string;
  sourceModelSha256?: string;
  /** Declared graph-compatible input identity; required only by release-manifest-v1. */
  datasetSha256?: string;
  globalMean: number;
  factors: number;
  animeIds: number[];
  titles: string[];
  biases: number[];
  embeddings: number[][];
  animeCount?: number;
}

export interface ReleaseIdentityCatalog {
  format: "anime-catalog-v1";
  datasetSha256: string;
  anime: CompactAnimeEntry[];
}

/** Source-neutral metadata candidate. It is not a release-manifest-v1 asset or a use approval. */
export interface CatalogMetadataItemV1 {
  animeId: number;
  sourceItemId: string;
  /** Source canonical label; browser projection retains it as a known alias of the stable graph ID. */
  title: string;
  /** Null means unavailable; an empty array means checked with no values. */
  aliases: string[] | null;
  genres: string[] | null;
  year: number | null;
  mediaFormat: "TV" | "Movie" | "OVA" | "ONA" | "Special" | null;
  episodeCount: number | null;
  runtimeMinutes: number | null;
  contentClassification: { jurisdiction: string; system: string; value: string } | null;
  communityScore: number | null;
  /** Even an empty array does not prove there is no viewing prerequisite. */
  relations: AnimeRelation[] | null;
}

export interface CatalogMetadataSnapshotV1 {
  format: "anime-metadata-catalog-v1";
  /** Declared provenance; digest binds exact approved retained input bytes, not inferred transport bytes or source rights. */
  source: { name: string; snapshotAt: string; snapshotSha256: string };
  anime: CatalogMetadataItemV1[];
}

export type CatalogCoverageField = "aliases" | "genres" | "year" | "mediaFormat" |
  "episodeCount" | "runtimeMinutes" | "contentClassification" | "communityScore" | "relations";

export interface CatalogMetadataCoverage {
  total: number;
  missingItems: number;
  /** Present, non-null values; empty arrays are known but not usable for matching. */
  known: Record<CatalogCoverageField, number>;
  /** Nonempty arrays or present scalar values. */
  usable: Record<CatalogCoverageField, number>;
  /** Directed evidence is counted, never treated as proof that other prerequisites are absent. */
  directedRelationItems: number;
  directedTargetsOutsideUniverse: number;
}

export interface ReleaseManifestAsset {
  path: string;
  format: string;
  sha256: string;
  bytes: number;
}

export interface ReleaseManifestV1 {
  format: "release-manifest-v1";
  tag: string;
  bundleId: string;
  dataset: GraphV2Metadata["dataset"];
  catalog: ReleaseManifestAsset & { animeCount: number; itemMapSha256: string };
  neighborhood: ReleaseManifestAsset & { graphId: string };
  explorer: ReleaseManifestAsset & { graphId: string; sourceGraphId: string };
  model: (ReleaseManifestAsset & {
    datasetSha256: string;
    itemMapSha256: string;
    coverage: { mappedAnimeCount: number; totalCatalogAnimeCount: number };
  }) | null;
  lastKnownGood: { tag: string; bundleId: string; manifestSha256: string } | null;
}

/** Browser candidate only. Existing release installers and publishers still accept v1. */
export interface ReleaseManifestV2 extends Omit<ReleaseManifestV1, "format"> {
  format: "release-manifest-v2";
  metadata: ReleaseManifestAsset & {
    animeCount: number;
    itemMapSha256: string;
    sourceSnapshotSha256: string;
  };
}

export type BrowserReleaseManifest = ReleaseManifestV1 | ReleaseManifestV2;

export interface ActiveReleaseBundleV1 {
  format: "active-release-bundle-v1";
  tag: string;
  bundleId: string;
  manifestSha256: string;
}

export const RELEASE_BUNDLE_LIMITS = {
  manifestBytes: 256 * 1024,
  compressedAssetBytes: 64 * 1024 * 1024,
  plainAssetBytes: 256 * 1024 * 1024,
  totalPlainBytes: 512 * 1024 * 1024,
  activePointerBytes: 16 * 1024,
} as const;

export class ArtifactValidationError extends Error {
  constructor(label: string, location: string, reason: string) {
    super(`${label}: ${location} ${reason}. Rebuild or replace this artifact.`);
    this.name = "ArtifactValidationError";
  }
}

function invalid(label: string, location: string, reason: string): never {
  throw new ArtifactValidationError(label, location, reason);
}

function record(value: unknown, label: string, location: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    invalid(label, location, "must be an object");
  }
  return value as Record<string, unknown>;
}

function list(value: unknown, label: string, location: string): unknown[] {
  if (!Array.isArray(value)) invalid(label, location, "must be an array");
  return value;
}

function nonemptyText(value: unknown, label: string, location: string): string {
  if (typeof value !== "string" || !value.trim()) {
    invalid(label, location, "must be a nonempty string");
  }
  return value;
}

function text(value: unknown, label: string, location: string): string {
  if (typeof value !== "string") invalid(label, location, "must be a string");
  return value;
}

function finite(value: unknown, label: string, location: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    invalid(label, location, "must be a finite number");
  }
  return value;
}

function safeInteger(value: unknown, label: string, location: string, minimum: number): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) {
    invalid(label, location, `must be a safe integer >= ${minimum}`);
  }
  return value;
}

function index(value: unknown, bound: number, label: string, location: string): number {
  const parsed = safeInteger(value, label, location, 0);
  if (parsed >= bound) invalid(label, location, `references an index outside 0..${bound - 1}`);
  return parsed;
}

function generatedAt(value: unknown, label: string): void {
  const stamp = nonemptyText(value, label, "generatedAt");
  if (!Number.isFinite(Date.parse(stamp))) {
    invalid(label, "generatedAt", "must be a valid date-time string");
  }
}

function expectFormat(value: Record<string, unknown>, expected: string, label: string): void {
  if (value.format !== expected) {
    invalid(label, "format", `is unsupported; expected ${expected}`);
  }
  if (Object.hasOwn(value, "version")) {
    invalid(label, "version", "is unsupported; use the declared format version");
  }
}

function expectLegacyFormat(value: Record<string, unknown>, label: string): void {
  if (Object.hasOwn(value, "format") || Object.hasOwn(value, "version")) {
    invalid(label, "format/version", "is unsupported for the unversioned legacy artifact");
  }
}

function rejectV2MetadataOnV1(value: Record<string, unknown>, label: string): void {
  for (const field of ["role", "graphId", "sourceGraphId", "dataset", "semantics", "config", "truncation", "visualization", "projection"]) {
    if (Object.hasOwn(value, field)) {
      invalid(label, field, "requires a v2 graph format");
    }
  }
}

function checkCounts(
  value: Record<string, unknown>, label: string, users: number, anime: number, edges: number,
): void {
  for (const [field, expected] of [
    ["userCount", users], ["animeCount", anime],
    ["nodeCount", users + anime], ["edgeCount", edges],
  ] as const) {
    const actual = safeInteger(value[field], label, field, 0);
    if (actual !== expected) invalid(label, field, `must equal ${expected}`);
  }
}

function checkOptionalCount(
  value: Record<string, unknown>, label: string, field: string, expected: number,
): void {
  if (!Object.hasOwn(value, field)) return;
  const actual = safeInteger(value[field], label, field, 0);
  if (actual !== expected) invalid(label, field, `must equal ${expected}`);
}

function uniqueKey(key: string | number, seen: Set<string | number>, label: string, location: string): void {
  if (seen.has(key)) invalid(label, location, "duplicates an earlier ID or relationship");
  seen.add(key);
}

function relationshipKey(left: number, right: number, width: number, label: string, location: string): number {
  const key = left * width + right;
  if (!Number.isSafeInteger(key)) invalid(label, location, "has too many indexes for safe relationship IDs");
  return key;
}

function sha256(value: unknown, label: string, location: string): void {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {
    invalid(label, location, "must be a lowercase SHA-256 digest");
  }
}

function exactFields(value: Record<string, unknown>, fields: string[], label: string, location: string): void {
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) invalid(label, `${location}.${field}`, "is required");
  }
  for (const field of Object.keys(value)) {
    if (!fields.includes(field)) invalid(label, `${location}.${field}`, "is unsupported");
  }
}

function versionTag(value: unknown, label: string, location: string): void {
  if (typeof value !== "string" || !/^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)) {
    invalid(label, location, "must be a versioned data-v tag");
  }
}

function validateV2Metadata(
  graph: Record<string, unknown>, label: string,
  expectedRole?: "recommendation" | "visualization",
): "recommendation" | "visualization" {
  const role = graph.role;
  if (role !== "recommendation" && role !== "visualization") {
    invalid(label, "role", "must be recommendation or visualization");
  }
  if (expectedRole && role !== expectedRole) {
    invalid(label, "role", `must be ${expectedRole} for this artifact`);
  }
  sha256(graph.graphId, label, "graphId");
  const dataset = record(graph.dataset, label, "dataset");
  sha256(dataset.sha256, label, "dataset.sha256");
  if (dataset.scope !== "anonymized-ratings-content-v1") {
    invalid(label, "dataset.scope", "is unsupported");
  }
  nonemptyText(dataset.source, label, "dataset.source");
  const semantics = record(graph.semantics, label, "semantics");
  for (const [field, expected] of [
    ["pairWeight", "centered-pair-preference-mean-v1"],
    ["support", "co-raters-after-selection-v1"],
    ["recommendationUse", "positive-only-v1"],
  ] as const) {
    if (semantics[field] !== expected) invalid(label, `semantics.${field}`, "is unsupported");
  }
  const config = record(graph.config, label, "config");
  const maxRatings = safeInteger(config.maxRatingsPerUser, label, "config.maxRatingsPerUser", 0);
  const policy = maxRatings > 0 ? "sha256-bottom-k-v1" : "all-ratings";
  if (config.ratingSelectionPolicy !== policy) {
    invalid(label, "config.ratingSelectionPolicy", `must be ${policy}`);
  }
  const seed = safeInteger(config.seed, label, "config.seed", 0);
  if (seed > 0xffffffff) invalid(label, "config.seed", "must be an unsigned 32-bit integer");
  for (const [field, minimum] of [
    ["maxAnimeAnimeEdges", 0], ["maxPairVisits", 1], ["maxPairCandidates", 1],
    ["minPairSupport", 1], ["maxNeighborsPerAnime", 0],
  ] as const) safeInteger(config[field], label, `config.${field}`, minimum);
  const truncation = record(graph.truncation, label, "truncation");
  const counts = new Map<string, number>();
  for (const field of [
    "inputRatings", "selectedRatings", "ratingsSkipped", "potentialPairVisits",
    "pairVisits", "pairVisitsSkipped", "candidatePairs", "eligiblePairs",
    "selectedPairs", "excludedBySupport", "excludedByNeighborLimit", "excludedByOutputLimit",
  ]) {
    counts.set(field, safeInteger(truncation[field], label, `truncation.${field}`, 0));
  }
  const count = (field: string): number => counts.get(field)!;
  if (count("selectedRatings") + count("ratingsSkipped") !== count("inputRatings")) {
    invalid(label, "truncation.ratingsSkipped", "does not reconcile with input ratings");
  }
  if (count("pairVisits") + count("pairVisitsSkipped") !== count("potentialPairVisits")) {
    invalid(label, "truncation.pairVisitsSkipped", "does not reconcile with potential visits");
  }
  if (count("candidatePairs") - count("excludedBySupport") !== count("eligiblePairs") ||
      count("eligiblePairs") - count("excludedByNeighborLimit") - count("excludedByOutputLimit") !== count("selectedPairs")) {
    invalid(label, "truncation.selectedPairs", "does not reconcile with candidate exclusions");
  }
  if (role === "recommendation") {
    if (Object.hasOwn(graph, "sourceGraphId") || Object.hasOwn(graph, "visualization")) {
      invalid(label, "role", "recommendation graphs cannot carry visualization provenance");
    }
  } else {
    sha256(graph.sourceGraphId, label, "sourceGraphId");
    const visualization = record(graph.visualization, label, "visualization");
    if (visualization.policy !== "abs-weight-top-k-v1") {
      invalid(label, "visualization.policy", "is unsupported");
    }
    for (const field of [
      "maxUserAnimeEdges", "maxAnimeAnimeEdges", "excludedUserAnimeEdges", "excludedAnimeAnimeEdges",
    ]) safeInteger(visualization[field], label, `visualization.${field}`, 0);
  }
  return role;
}

function validateV2EdgeCounts(
  graph: Record<string, unknown>, label: string, role: "recommendation" | "visualization",
  userAnimeCount: number, animeAnimeCount: number, aggregateOnly = false,
): void {
  const truncation = graph.truncation as Record<string, number>;
  if (role === "recommendation") {
    if ((!aggregateOnly && truncation.selectedRatings !== userAnimeCount) ||
        truncation.selectedPairs !== animeAnimeCount) {
      invalid(label, "truncation", "must match recommendation edge counts");
    }
  } else {
    const visualization = graph.visualization as Record<string, number>;
    if (userAnimeCount > visualization.maxUserAnimeEdges || animeAnimeCount > visualization.maxAnimeAnimeEdges ||
        (aggregateOnly ? visualization.excludedUserAnimeEdges !== 0
          : truncation.selectedRatings - userAnimeCount !== visualization.excludedUserAnimeEdges) ||
        truncation.selectedPairs - animeAnimeCount !== visualization.excludedAnimeAnimeEdges) {
      invalid(label, "visualization", "must match its sampled edge counts");
    }
  }
}

function validateAggregateOnlyFields(graph: Record<string, unknown>, label: string): void {
  const fields = ["format", "role", "graphId", "dataset", "semantics", "config", "truncation",
    "projection", "generatedAt", "userIds", "anime", "ua", "aa", "userCount", "animeCount",
    "nodeCount", "edgeCount"];
  if (graph.role === "visualization") fields.push("sourceGraphId", "visualization");
  exactFields(graph, fields, label, "root");
  const projection = record(graph.projection, label, "projection");
  exactFields(projection, ["policy"], label, "projection");
  if (projection.policy !== "omit-user-anime-v1") {
    invalid(label, "projection.policy", "must be omit-user-anime-v1");
  }
  exactFields(record(graph.dataset, label, "dataset"), ["sha256", "scope", "source"], label, "dataset");
  exactFields(record(graph.semantics, label, "semantics"),
    ["pairWeight", "support", "recommendationUse"], label, "semantics");
  exactFields(record(graph.config, label, "config"), ["ratingSelectionPolicy", "seed",
    "maxRatingsPerUser", "maxAnimeAnimeEdges", "maxPairVisits", "maxPairCandidates",
    "minPairSupport", "maxNeighborsPerAnime"], label, "config");
  exactFields(record(graph.truncation, label, "truncation"), ["inputRatings", "selectedRatings",
    "ratingsSkipped", "potentialPairVisits", "pairVisits", "pairVisitsSkipped", "candidatePairs",
    "eligiblePairs", "selectedPairs", "excludedBySupport", "excludedByNeighborLimit",
    "excludedByOutputLimit"], label, "truncation");
  if (graph.role === "visualization") {
    exactFields(record(graph.visualization, label, "visualization"), ["policy",
      "maxUserAnimeEdges", "maxAnimeAnimeEdges", "excludedUserAnimeEdges",
      "excludedAnimeAnimeEdges"], label, "visualization");
  }
}

export function parseCompactGraph(
  value: unknown, label: string, expectedRole?: "recommendation" | "visualization",
): CompactGraphData {
  const graph = record(value, label, "root");
  if (graph.format !== "graph-compact-v1" && graph.format !== "graph-compact-v2" &&
      graph.format !== "graph-compact-v3") {
    invalid(label, "format", "is unsupported; expected graph-compact-v1, v2, or v3");
  }
  if (Object.hasOwn(graph, "version")) {
    invalid(label, "version", "is unsupported; use the declared format version");
  }
  if (graph.format === "graph-compact-v1") rejectV2MetadataOnV1(graph, label);
  const role = graph.format !== "graph-compact-v1"
    ? validateV2Metadata(graph, label, expectedRole) : null;
  if (graph.format === "graph-compact-v3") validateAggregateOnlyFields(graph, label);
  generatedAt(graph.generatedAt, label);
  const userIds = list(graph.userIds, label, "userIds");
  const anime = list(graph.anime, label, "anime");
  const ua = list(graph.ua, label, "ua");
  const aa = list(graph.aa, label, "aa");
  if (graph.format === "graph-compact-v3" && (userIds.length !== 0 || ua.length !== 0)) {
    invalid(label, "userIds/ua", "must be empty for aggregate-only v3");
  }
  const userSet = new Set<string | number>();
  const animeSet = new Set<string | number>();
  userIds.forEach((value, i) => {
    const id = nonemptyText(value, label, `userIds[${i}]`);
    uniqueKey(id, userSet, label, `userIds[${i}]`);
  });
  anime.forEach((value, i) => {
    const entry = list(value, label, `anime[${i}]`);
    if (entry.length !== 2) invalid(label, `anime[${i}]`, "must have exactly 2 values");
    const id = safeInteger(entry[0], label, `anime[${i}][0]`, 1);
    nonemptyText(entry[1], label, `anime[${i}][1]`);
    uniqueKey(id, animeSet, label, `anime[${i}][0]`);
  });
  const uaSet = new Set<string | number>();
  ua.forEach((value, i) => {
    const entry = list(value, label, `ua[${i}]`);
    if (entry.length !== 3) invalid(label, `ua[${i}]`, "must have exactly 3 values");
    const user = index(entry[0], userIds.length, label, `ua[${i}][0]`);
    const item = index(entry[1], anime.length, label, `ua[${i}][1]`);
    finite(entry[2], label, `ua[${i}][2]`);
    uniqueKey(relationshipKey(user, item, anime.length, label, `ua[${i}]`), uaSet, label, `ua[${i}]`);
  });
  const aaSet = new Set<string | number>();
  aa.forEach((value, i) => {
    const entry = list(value, label, `aa[${i}]`);
    if (graph.format !== "graph-compact-v1" ? entry.length !== 4 : entry.length !== 3 && entry.length !== 4) {
      invalid(label, `aa[${i}]`, graph.format !== "graph-compact-v1"
        ? "must have exactly 4 values with support" : "must have 3 values or 4 with support");
    }
    const left = index(entry[0], anime.length, label, `aa[${i}][0]`);
    const right = index(entry[1], anime.length, label, `aa[${i}][1]`);
    if (left === right) invalid(label, `aa[${i}]`, "must connect distinct anime");
    finite(entry[2], label, `aa[${i}][2]`);
    if (entry.length === 4) safeInteger(entry[3], label, `aa[${i}][3]`, 1);
    uniqueKey(
      relationshipKey(Math.min(left, right), Math.max(left, right), anime.length, label, `aa[${i}]`),
      aaSet, label, `aa[${i}]`,
    );
  });
  checkCounts(graph, label, userIds.length, anime.length, ua.length + aa.length);
  if (role) validateV2EdgeCounts(graph, label, role, ua.length, aa.length,
    graph.format === "graph-compact-v3");
  return value as CompactGraphData;
}

/** Demo assets must use the same aggregate-only graph contract as new public bundles. */
export function parseAggregateDemoGraph(
  value: unknown, label: string, role: "recommendation" | "visualization",
): CompactGraphDataV3 {
  const graph = parseCompactGraph(value, label, role);
  if (graph.format !== "graph-compact-v3") {
    invalid(label, "format", "must be graph-compact-v3 for the synthetic demo");
  }
  return graph as CompactGraphDataV3;
}

export function parseLegacyGraph(value: unknown, label: string): GraphData {
  const graph = record(value, label, "root");
  const v2 = graph.format === "graph-legacy-v2";
  if (v2) {
    if (Object.hasOwn(graph, "version")) {
      invalid(label, "version", "is unsupported; use the declared format version");
    }
    validateV2Metadata(graph, label, "recommendation");
  } else {
    expectLegacyFormat(graph, label);
    rejectV2MetadataOnV1(graph, label);
  }
  generatedAt(graph.generatedAt, label);
  const nodes = list(graph.nodes, label, "nodes");
  const edges = list(graph.edges, label, "edges");
  const nodeTypes = new Map<string, NodeType>();
  let users = 0;
  let anime = 0;
  nodes.forEach((value, i) => {
    const node = record(value, label, `nodes[${i}]`);
    const id = nonemptyText(node.id, label, `nodes[${i}].id`);
    nonemptyText(node.label, label, `nodes[${i}].label`);
    if (node.nodeType !== "user" && node.nodeType !== "anime") {
      invalid(label, `nodes[${i}].nodeType`, "must be user or anime");
    }
    if (node.nodeType === "user") {
      if (!id.startsWith("user:") || id.length <= 5) invalid(label, `nodes[${i}].id`, "must identify a user");
      users += 1;
    } else {
      const suffix = id.startsWith("anime:") ? id.slice(6) : "";
      const animeId = Number(suffix);
      if (!Number.isSafeInteger(animeId) || animeId <= 0 || String(animeId) !== suffix) {
        invalid(label, `nodes[${i}].id`, "must identify a positive anime ID");
      }
      anime += 1;
    }
    if (nodeTypes.has(id)) invalid(label, `nodes[${i}].id`, "duplicates an earlier node ID");
    nodeTypes.set(id, node.nodeType);
  });
  const edgeIds = new Set<string | number>();
  edges.forEach((value, i) => {
    const edge = record(value, label, `edges[${i}]`);
    const id = nonemptyText(edge.id, label, `edges[${i}].id`);
    uniqueKey(id, edgeIds, label, `edges[${i}].id`);
    const source = nonemptyText(edge.source, label, `edges[${i}].source`);
    const target = nonemptyText(edge.target, label, `edges[${i}].target`);
    const sourceType = nodeTypes.get(source);
    const targetType = nodeTypes.get(target);
    if (!sourceType || !targetType) invalid(label, `edges[${i}]`, "references a missing node");
    if (edge.edgeType === "user-anime") {
      if (sourceType === targetType) invalid(label, `edges[${i}]`, "must connect one user and one anime");
    } else if (edge.edgeType === "anime-anime") {
      if (sourceType !== "anime" || targetType !== "anime" || source === target) {
        invalid(label, `edges[${i}]`, "must connect two distinct anime");
      }
    } else {
      invalid(label, `edges[${i}].edgeType`, "is unsupported");
    }
    finite(edge.weight, label, `edges[${i}].weight`);
    if (edge.edgeType === "anime-anime" && v2 && !Object.hasOwn(edge, "support")) {
      invalid(label, `edges[${i}].support`, "is required for v2 pair edges");
    }
    if (Object.hasOwn(edge, "support")) safeInteger(edge.support, label, `edges[${i}].support`, 1);
  });
  checkCounts(graph, label, users, anime, edges.length);
  if (v2) validateV2EdgeCounts(graph, label, "recommendation",
    edges.filter((edge) => (edge as Record<string, unknown>).edgeType === "user-anime").length,
    edges.filter((edge) => (edge as Record<string, unknown>).edgeType === "anime-anime").length,
  );
  return value as GraphData;
}

export function isCompactGraphData(value: LoadedGraphData): value is CompactGraphData {
  return "format" in value && (value.format === "graph-compact-v1" ||
    value.format === "graph-compact-v2" || value.format === "graph-compact-v3");
}

export function parseDemoCatalog(value: unknown, label: string): DemoCatalogItem[] {
  const catalog = record(value, label, "root");
  expectFormat(catalog, "demo-catalog-v1", label);
  generatedAt(catalog.generatedAt, label);
  const anime = list(catalog.anime, label, "anime");
  const ids = new Set<string | number>();
  anime.forEach((value, i) => {
    const item = record(value, label, `anime[${i}]`);
    const id = safeInteger(item.animeId, label, `anime[${i}].animeId`, 1);
    uniqueKey(id, ids, label, `anime[${i}].animeId`);
    nonemptyText(item.title, label, `anime[${i}].title`);
    if (item.aliases !== undefined) {
      const aliases = list(item.aliases, label, `anime[${i}].aliases`);
      if (aliases.length > 20) invalid(label, `anime[${i}].aliases`, "must contain at most 20 titles");
      aliases.forEach((alias, j) => {
        nonemptyText(alias, label, `anime[${i}].aliases[${j}]`);
        if ((alias as string).length > 200) invalid(label, `anime[${i}].aliases[${j}]`, "is too long");
      });
    }
    if (item.mediaFormat !== undefined && item.mediaFormat !== null) {
      nonemptyText(item.mediaFormat, label, `anime[${i}].mediaFormat`);
      if ((item.mediaFormat as string).length > 80) invalid(label, `anime[${i}].mediaFormat`, "is too long");
    }
    if (item.year !== null) safeInteger(item.year, label, `anime[${i}].year`, 1);
    if (item.score !== null) finite(item.score, label, `anime[${i}].score`);
    for (const field of ["genres", "studios"] as const) {
      list(item[field], label, `anime[${i}].${field}`).forEach((entry, j) => {
        nonemptyText(entry, label, `anime[${i}].${field}[${j}]`);
      });
    }
    text(item.synopsis, label, `anime[${i}].synopsis`);
    text(item.imageUrl, label, `anime[${i}].imageUrl`);
    if (item.season !== null) nonemptyText(item.season, label, `anime[${i}].season`);
    if (item.relations !== undefined && item.relations !== null) {
      list(item.relations, label, `anime[${i}].relations`).forEach((entry, j) => {
        const location = `anime[${i}].relations[${j}]`;
        const relation = record(entry, label, location);
        if (!["prequel", "sequel", "alternative-version", "side-story", "spin-off"].includes(
          relation.kind as string)) invalid(label, `${location}.kind`, "must be a supported anime relationship");
        const relatedId = safeInteger(relation.animeId, label, `${location}.animeId`, 1);
        if (relatedId === id) invalid(label, `${location}.animeId`, "cannot refer to itself");
        nonemptyText(relation.title, label, `${location}.title`);
      });
    }
  });
  return anime as DemoCatalogItem[];
}

export function parseReleaseIdentityCatalog(value: unknown, label: string): ReleaseIdentityCatalog {
  const catalog = record(value, label, "root");
  exactFields(catalog, ["format", "datasetSha256", "anime"], label, "root");
  expectFormat(catalog, "anime-catalog-v1", label);
  sha256(catalog.datasetSha256, label, "datasetSha256");
  const anime = list(catalog.anime, label, "anime");
  if (anime.length === 0) invalid(label, "anime", "must contain at least one anime");
  let previousId = 0;
  anime.forEach((value, i) => {
    const entry = list(value, label, `anime[${i}]`);
    if (entry.length !== 2) invalid(label, `anime[${i}]`, "must have exactly 2 values");
    const id = safeInteger(entry[0], label, `anime[${i}][0]`, 1);
    if (id <= previousId) invalid(label, `anime[${i}][0]`, "must be sorted by unique ascending ID");
    nonemptyText(entry[1], label, `anime[${i}][1]`);
    previousId = id;
  });
  return value as ReleaseIdentityCatalog;
}

export function parseCatalogMetadataSnapshot(value: unknown, label: string): CatalogMetadataSnapshotV1 {
  const root = record(value, label, "root");
  exactFields(root, ["format", "source", "anime"], label, "root");
  expectFormat(root, "anime-metadata-catalog-v1", label);
  const source = record(root.source, label, "source");
  exactFields(source, ["name", "snapshotAt", "snapshotSha256"], label, "source");
  if (nonemptyText(source.name, label, "source.name").length > 120) {
    invalid(label, "source.name", "is too long");
  }
  const stamp = nonemptyText(source.snapshotAt, label, "source.snapshotAt");
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(stamp) ||
      !Number.isFinite(Date.parse(stamp)) || new Date(stamp).toISOString() !== stamp) {
    invalid(label, "source.snapshotAt", "must be a canonical UTC date-time");
  }
  sha256(source.snapshotSha256, label, "source.snapshotSha256");
  const anime = list(root.anime, label, "anime");
  if (anime.length === 0 || anime.length > 100_000) {
    invalid(label, "anime", "must contain 1 to 100000 items");
  }
  const sourceIds = new Set<string | number>();
  let previousId = 0;
  const bounded = (entry: unknown, location: string, maximum: number) => {
    const parsed = nonemptyText(entry, label, location);
    if (parsed.length > maximum) invalid(label, location, `must be at most ${maximum} characters`);
    return parsed;
  };
  anime.forEach((value, i) => {
    const location = `anime[${i}]`;
    const item = record(value, label, location);
    exactFields(item, ["animeId", "sourceItemId", "title", "aliases", "genres", "year",
      "mediaFormat", "episodeCount", "runtimeMinutes", "contentClassification",
      "communityScore", "relations"], label, location);
    const id = safeInteger(item.animeId, label, `${location}.animeId`, 1);
    if (id <= previousId) invalid(label, `${location}.animeId`, "must be sorted by unique ascending ID");
    previousId = id;
    uniqueKey(bounded(item.sourceItemId, `${location}.sourceItemId`, 80), sourceIds,
      label, `${location}.sourceItemId`);
    bounded(item.title, `${location}.title`, 200);
    for (const [field, maximumCount, maximumLength] of [
      ["aliases", 20, 200], ["genres", 30, 80],
    ] as const) {
      if (item[field] === null) continue;
      const values = list(item[field], label, `${location}.${field}`);
      if (values.length > maximumCount) invalid(label, `${location}.${field}`, "has too many values");
      const seen = new Set<string | number>();
      values.forEach((entry, j) => {
        const name = bounded(entry, `${location}.${field}[${j}]`, maximumLength).trim().toLowerCase();
        uniqueKey(name, seen, label, `${location}.${field}[${j}]`);
      });
    }
    if (item.year !== null) {
      const year = safeInteger(item.year, label, `${location}.year`, 1800);
      if (year > 3000) invalid(label, `${location}.year`, "must be at most 3000");
    }
    if (item.mediaFormat !== null &&
        !["TV", "Movie", "OVA", "ONA", "Special"].includes(item.mediaFormat as string)) {
      invalid(label, `${location}.mediaFormat`, "must be a supported format or null");
    }
    if (item.episodeCount !== null) {
      const count = safeInteger(item.episodeCount, label, `${location}.episodeCount`, 1);
      if (count > 100_000) invalid(label, `${location}.episodeCount`, "must be at most 100000");
    }
    if (item.runtimeMinutes !== null) {
      const runtime = finite(item.runtimeMinutes, label, `${location}.runtimeMinutes`);
      if (runtime <= 0 || runtime > 10_000) {
        invalid(label, `${location}.runtimeMinutes`, "must be greater than 0 and at most 10000");
      }
    }
    if (item.contentClassification !== null) {
      const classification = record(item.contentClassification, label, `${location}.contentClassification`);
      exactFields(classification, ["jurisdiction", "system", "value"], label,
        `${location}.contentClassification`);
      for (const field of ["jurisdiction", "system", "value"] as const) {
        bounded(classification[field], `${location}.contentClassification.${field}`, 80);
      }
    }
    if (item.communityScore !== null) {
      const score = finite(item.communityScore, label, `${location}.communityScore`);
      if (score < 0 || score > 10) invalid(label, `${location}.communityScore`, "must be within 0..10");
    }
    if (item.relations !== null) {
      const relations = list(item.relations, label, `${location}.relations`);
      if (relations.length > 50) invalid(label, `${location}.relations`, "has too many values");
      const seen = new Set<string | number>();
      relations.forEach((value, j) => {
        const where = `${location}.relations[${j}]`;
        const relation = record(value, label, where);
        exactFields(relation, ["kind", "animeId", "title"], label, where);
        if (!["prequel", "sequel", "alternative-version", "side-story", "spin-off"].includes(
          relation.kind as string)) invalid(label, `${where}.kind`, "must be a supported relationship");
        const relatedId = safeInteger(relation.animeId, label, `${where}.animeId`, 1);
        if (relatedId === id) invalid(label, `${where}.animeId`, "cannot refer to itself");
        bounded(relation.title, `${where}.title`, 200);
        uniqueKey(`${relation.kind}:${relatedId}`, seen, label, where);
      });
    }
  });
  return value as CatalogMetadataSnapshotV1;
}

export function catalogMetadataCoverage(snapshot: CatalogMetadataSnapshotV1,
  universe: readonly number[]): CatalogMetadataCoverage {
  const fields: CatalogCoverageField[] = ["aliases", "genres", "year", "mediaFormat",
    "episodeCount", "runtimeMinutes", "contentClassification", "communityScore", "relations"];
  const counts = () => Object.fromEntries(fields.map((field) => [field, 0])) as
    Record<CatalogCoverageField, number>;
  const known = counts();
  const usable = counts();
  const byId = new Map(snapshot.anime.map((item) => [item.animeId, item]));
  const universeIds = new Set<number>();
  for (const id of universe) {
    if (!Number.isSafeInteger(id) || id < 1 || universeIds.has(id)) {
      throw new Error("Catalog coverage universe must contain unique positive anime IDs.");
    }
    universeIds.add(id);
  }
  let missingItems = 0;
  let directedRelationItems = 0;
  let directedTargetsOutsideUniverse = 0;
  for (const id of universe) {
    const item = byId.get(id);
    if (!item) { missingItems += 1; continue; }
    for (const field of fields) {
      const value = item[field];
      if (value === null) continue;
      known[field] += 1;
      if (!Array.isArray(value) || value.length > 0) usable[field] += 1;
    }
    const directed = (item.relations ?? []).filter((relation) =>
      relation.kind === "prequel" || relation.kind === "sequel");
    if (directed.length > 0) directedRelationItems += 1;
    directedTargetsOutsideUniverse += directed.filter((relation) =>
      !universeIds.has(relation.animeId)).length;
  }
  return { total: universe.length, missingItems, known, usable,
    directedRelationItems, directedTargetsOutsideUniverse };
}

export function parseReleaseManifest(value: unknown, label: string): ReleaseManifestV1 {
  const manifest = record(value, label, "root");
  exactFields(manifest, ["format", "tag", "bundleId", "dataset", "catalog", "neighborhood",
    "explorer", "model", "lastKnownGood"], label, "root");
  expectFormat(manifest, "release-manifest-v1", label);
  versionTag(manifest.tag, label, "tag");
  sha256(manifest.bundleId, label, "bundleId");
  const dataset = record(manifest.dataset, label, "dataset");
  exactFields(dataset, ["sha256", "scope", "source"], label, "dataset");
  sha256(dataset.sha256, label, "dataset.sha256");
  if (dataset.scope !== "anonymized-ratings-content-v1") invalid(label, "dataset.scope", "is unsupported");
  nonemptyText(dataset.source, label, "dataset.source");

  const asset = (field: string, expectedPath: string, expectedFormat: string, extra: string[]) => {
    const entry = record(manifest[field], label, field);
    exactFields(entry, ["path", "format", "sha256", "bytes", ...extra], label, field);
    if (entry.path !== expectedPath) invalid(label, `${field}.path`, `must be ${expectedPath}`);
    if (entry.format !== expectedFormat) invalid(label, `${field}.format`, `must be ${expectedFormat}`);
    sha256(entry.sha256, label, `${field}.sha256`);
    safeInteger(entry.bytes, label, `${field}.bytes`, 1);
    return entry;
  };
  const catalog = asset("catalog", "catalog.identity.json", "anime-catalog-v1",
    ["animeCount", "itemMapSha256"]);
  safeInteger(catalog.animeCount, label, "catalog.animeCount", 1);
  sha256(catalog.itemMapSha256, label, "catalog.itemMapSha256");
  const declaredGraphFormat = record(manifest.neighborhood, label, "neighborhood").format;
  if (declaredGraphFormat !== "graph-compact-v2" && declaredGraphFormat !== "graph-compact-v3") {
    invalid(label, "neighborhood.format", "must be graph-compact-v2 or graph-compact-v3");
  }
  const neighborhood = asset("neighborhood", "graph.compact.json", declaredGraphFormat, ["graphId"]);
  sha256(neighborhood.graphId, label, "neighborhood.graphId");
  const explorer = asset("explorer", "graph-explorer.compact.json", declaredGraphFormat,
    ["graphId", "sourceGraphId"]);
  sha256(explorer.graphId, label, "explorer.graphId");
  sha256(explorer.sourceGraphId, label, "explorer.sourceGraphId");
  if (explorer.sourceGraphId !== neighborhood.graphId) {
    invalid(label, "explorer.sourceGraphId", "must match neighborhood.graphId");
  }
  if (manifest.model !== null) {
    const model = asset("model", "model-mf-web.compact.json", "model-mf-compact-v1",
      ["datasetSha256", "itemMapSha256", "coverage"]);
    sha256(model.datasetSha256, label, "model.datasetSha256");
    if (model.datasetSha256 !== dataset.sha256) {
      invalid(label, "model.datasetSha256", "must match dataset.sha256");
    }
    sha256(model.itemMapSha256, label, "model.itemMapSha256");
    const coverage = record(model.coverage, label, "model.coverage");
    exactFields(coverage, ["mappedAnimeCount", "totalCatalogAnimeCount"], label, "model.coverage");
    const mapped = safeInteger(coverage.mappedAnimeCount, label, "model.coverage.mappedAnimeCount", 1);
    const total = safeInteger(coverage.totalCatalogAnimeCount, label,
      "model.coverage.totalCatalogAnimeCount", 1);
    if (total !== catalog.animeCount || mapped > total) {
      invalid(label, "model.coverage", "must fit the catalog anime count");
    }
  }
  if (manifest.lastKnownGood !== null) {
    const previous = record(manifest.lastKnownGood, label, "lastKnownGood");
    exactFields(previous, ["tag", "bundleId", "manifestSha256"], label, "lastKnownGood");
    versionTag(previous.tag, label, "lastKnownGood.tag");
    if (previous.tag === manifest.tag) invalid(label, "lastKnownGood.tag", "must differ from current tag");
    sha256(previous.bundleId, label, "lastKnownGood.bundleId");
    sha256(previous.manifestSha256, label, "lastKnownGood.manifestSha256");
  }
  return value as ReleaseManifestV1;
}

/** Accepts the existing release contract plus a metadata-bearing browser candidate. */
export function parseBrowserReleaseManifest(value: unknown, label: string): BrowserReleaseManifest {
  const root = record(value, label, "root");
  if (root.format === "release-manifest-v1") return parseReleaseManifest(value, label);
  if (root.format !== "release-manifest-v2") {
    invalid(label, "format", "is unsupported");
  }
  exactFields(root, ["format", "tag", "bundleId", "dataset", "catalog", "neighborhood",
    "explorer", "model", "lastKnownGood", "metadata"], label, "root");
  const { metadata, ...v1Fields } = root;
  const base = parseReleaseManifest({ ...v1Fields, format: "release-manifest-v1" }, label);
  const entry = record(metadata, label, "metadata");
  exactFields(entry, ["path", "format", "sha256", "bytes", "animeCount",
    "itemMapSha256", "sourceSnapshotSha256"], label, "metadata");
  if (entry.path !== "catalog.metadata.json") {
    invalid(label, "metadata.path", "must be catalog.metadata.json");
  }
  if (entry.format !== "anime-metadata-catalog-v1") {
    invalid(label, "metadata.format", "must be anime-metadata-catalog-v1");
  }
  sha256(entry.sha256, label, "metadata.sha256");
  safeInteger(entry.bytes, label, "metadata.bytes", 1);
  const count = safeInteger(entry.animeCount, label, "metadata.animeCount", 1);
  if (count > base.catalog.animeCount) {
    invalid(label, "metadata.animeCount", "cannot exceed catalog.animeCount");
  }
  sha256(entry.itemMapSha256, label, "metadata.itemMapSha256");
  if (entry.itemMapSha256 !== base.catalog.itemMapSha256) {
    invalid(label, "metadata.itemMapSha256", "must match catalog.itemMapSha256");
  }
  sha256(entry.sourceSnapshotSha256, label, "metadata.sourceSnapshotSha256");
  return value as ReleaseManifestV2;
}

export function parseActiveReleaseBundle(value: unknown, label: string): ActiveReleaseBundleV1 {
  const active = record(value, label, "root");
  exactFields(active, ["format", "tag", "bundleId", "manifestSha256"], label, "root");
  expectFormat(active, "active-release-bundle-v1", label);
  versionTag(active.tag, label, "tag");
  sha256(active.bundleId, label, "bundleId");
  sha256(active.manifestSha256, label, "manifestSha256");
  return value as ActiveReleaseBundleV1;
}

function checkModelBase(model: Record<string, unknown>, label: string): number {
  generatedAt(model.generatedAt, label);
  finite(model.globalMean, label, "globalMean");
  const factors = safeInteger(model.factors, label, "factors", 1);
  if (Object.hasOwn(model, "sourceModel")) text(model.sourceModel, label, "sourceModel");
  if (Object.hasOwn(model, "sourceModelSha256") &&
      (typeof model.sourceModelSha256 !== "string" ||
       !/^[a-f0-9]{64}$/.test(model.sourceModelSha256))) {
    invalid(label, "sourceModelSha256", "must be a lowercase SHA-256 digest");
  }
  if (Object.hasOwn(model, "datasetSha256")) sha256(model.datasetSha256, label, "datasetSha256");
  return factors;
}

function checkEmbedding(value: unknown, factors: number, label: string, location: string): void {
  const embedding = list(value, label, location);
  if (embedding.length !== factors) invalid(label, location, `dimension must equal factors (${factors})`);
  embedding.forEach((weight, i) => finite(weight, label, `${location}[${i}]`));
}

export function parseCompactModel(value: unknown, label: string): CompactModelRecommendationData {
  const model = record(value, label, "root");
  const optional = ["sourceModel", "sourceModelSha256", "datasetSha256", "animeCount"]
    .filter((field) => Object.hasOwn(model, field));
  exactFields(model, ["format", "generatedAt", "globalMean", "factors", "animeIds",
    "titles", "biases", "embeddings", ...optional], label, "root");
  expectFormat(model, "model-mf-compact-v1", label);
  const factors = checkModelBase(model, label);
  const animeIds = list(model.animeIds, label, "animeIds");
  const titles = list(model.titles, label, "titles");
  const biases = list(model.biases, label, "biases");
  const embeddings = list(model.embeddings, label, "embeddings");
  const count = animeIds.length;
  if (count === 0) invalid(label, "animeIds", "must contain at least one anime");
  for (const [field, actual] of [
    ["titles", titles.length], ["biases", biases.length], ["embeddings", embeddings.length],
  ] as const) {
    if (actual !== count) invalid(label, field, `length must equal animeIds length (${count})`);
  }
  checkOptionalCount(model, label, "animeCount", count);
  const ids = new Set<string | number>();
  animeIds.forEach((value, i) => {
    const id = safeInteger(value, label, `animeIds[${i}]`, 1);
    uniqueKey(id, ids, label, `animeIds[${i}]`);
    nonemptyText(titles[i], label, `titles[${i}]`);
    finite(biases[i], label, `biases[${i}]`);
    checkEmbedding(embeddings[i], factors, label, `embeddings[${i}]`);
  });
  return value as CompactModelRecommendationData;
}

export function parseLegacyModel(value: unknown, label: string): ModelRecommendationData {
  const model = record(value, label, "root");
  expectLegacyFormat(model, label);
  const factors = checkModelBase(model, label);
  const anime = list(model.anime, label, "anime");
  if (anime.length === 0) invalid(label, "anime", "must contain at least one anime");
  checkOptionalCount(model, label, "animeCount", anime.length);
  const ids = new Set<string | number>();
  anime.forEach((value, i) => {
    const item = record(value, label, `anime[${i}]`);
    const id = safeInteger(item.animeId, label, `anime[${i}].animeId`, 1);
    uniqueKey(id, ids, label, `anime[${i}].animeId`);
    nonemptyText(item.title, label, `anime[${i}].title`);
    finite(item.bias, label, `anime[${i}].bias`);
    checkEmbedding(item.embedding, factors, label, `anime[${i}].embedding`);
  });
  return value as ModelRecommendationData;
}
