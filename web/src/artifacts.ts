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

export interface GraphData {
  generatedAt: string;
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export type CompactAnimeEntry = [animeId: number, title: string];
export type CompactUserAnimeEdge = [userIndex: number, animeIndex: number, weight: number];
export type CompactAnimeAnimeEdge = [
  leftAnimeIndex: number,
  rightAnimeIndex: number,
  weight: number,
  support?: number, // co-raters in processed rows, not proof that all pair keys were exported
];

export interface CompactGraphData {
  format: "graph-compact-v1";
  generatedAt: string;
  userIds: string[];
  anime: CompactAnimeEntry[];
  ua: CompactUserAnimeEdge[];
  aa: CompactAnimeAnimeEdge[];
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
}

export type LoadedGraphData = GraphData | CompactGraphData;

export interface AnimeMetadata {
  animeId: number;
  year: number | null;
  score: number | null;
  genres: string[];
  studios: string[];
  synopsis: string;
  imageUrl: string;
  season: string | null;
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
  globalMean: number;
  factors: number;
  anime: ModelRecommendationAnime[];
}

export interface CompactModelRecommendationData {
  format: "model-mf-compact-v1";
  generatedAt: string;
  globalMean: number;
  factors: number;
  animeIds: number[];
  titles: string[];
  biases: number[];
  embeddings: number[][];
}

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

export function parseCompactGraph(value: unknown, label: string): CompactGraphData {
  const graph = record(value, label, "root");
  expectFormat(graph, "graph-compact-v1", label);
  generatedAt(graph.generatedAt, label);
  const userIds = list(graph.userIds, label, "userIds");
  const anime = list(graph.anime, label, "anime");
  const ua = list(graph.ua, label, "ua");
  const aa = list(graph.aa, label, "aa");
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
    if (entry.length !== 3 && entry.length !== 4) {
      invalid(label, `aa[${i}]`, "must have 3 values or 4 with support");
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
  return value as CompactGraphData;
}

export function parseLegacyGraph(value: unknown, label: string): GraphData {
  const graph = record(value, label, "root");
  expectLegacyFormat(graph, label);
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
    if (Object.hasOwn(edge, "support")) safeInteger(edge.support, label, `edges[${i}].support`, 1);
  });
  checkCounts(graph, label, users, anime, edges.length);
  return value as GraphData;
}

export function isCompactGraphData(value: LoadedGraphData): value is CompactGraphData {
  return "format" in value;
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
  });
  return anime as DemoCatalogItem[];
}

function checkModelBase(model: Record<string, unknown>, label: string): number {
  generatedAt(model.generatedAt, label);
  finite(model.globalMean, label, "globalMean");
  const factors = safeInteger(model.factors, label, "factors", 1);
  if (Object.hasOwn(model, "sourceModel")) text(model.sourceModel, label, "sourceModel");
  return factors;
}

function checkEmbedding(value: unknown, factors: number, label: string, location: string): void {
  const embedding = list(value, label, location);
  if (embedding.length !== factors) invalid(label, location, `dimension must equal factors (${factors})`);
  embedding.forEach((weight, i) => finite(weight, label, `${location}[${i}]`));
}

export function parseCompactModel(value: unknown, label: string): CompactModelRecommendationData {
  const model = record(value, label, "root");
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
