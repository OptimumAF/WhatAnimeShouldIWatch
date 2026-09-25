import { visualizationGraphId } from "./graph-contract.js";
import type { CompactGraphData, CompactGraphDataV1, CompactGraphDataV2, GraphData } from "../types.js";

export const EXPLORER_AA_LIMIT = 8000;
export const EXPLORER_UA_LIMIT = 2500;

/** Convert a v2 legacy recommendation export so the explorer can be sampled independently. */
export function compactFromLegacyGraph(graph: GraphData): CompactGraphDataV2 {
  if (graph.role !== "recommendation") {
    throw new Error("A v2 legacy explorer source must be a recommendation graph.");
  }
  const userIds: string[] = [];
  const anime: [number, string][] = [];
  const userIndex = new Map<string, number>();
  const animeIndex = new Map<string, number>();
  for (const node of graph.nodes) {
    if (node.nodeType === "user") {
      if (!node.id.startsWith("user:") || node.id.length === 5 || userIndex.has(node.id)) {
        throw new Error(`Invalid v2 legacy user node: ${node.id}`);
      }
      userIndex.set(node.id, userIds.length);
      userIds.push(node.id.slice(5));
    } else if (node.nodeType === "anime") {
      const match = /^anime:([1-9]\d*)$/.exec(node.id);
      const id = match ? Number(match[1]) : NaN;
      if (!Number.isSafeInteger(id) || animeIndex.has(node.id)) {
        throw new Error(`Invalid v2 legacy anime node: ${node.id}`);
      }
      animeIndex.set(node.id, anime.length);
      anime.push([id, node.label]);
    } else {
      throw new Error(`Invalid v2 legacy node type: ${node.id}`);
    }
  }
  const ua: CompactGraphDataV2["ua"] = [];
  const aa: CompactGraphDataV2["aa"] = [];
  for (const edge of graph.edges) {
    if (edge.edgeType === "user-anime") {
      const user = userIndex.get(edge.source) ?? userIndex.get(edge.target);
      const item = animeIndex.get(edge.target) ?? animeIndex.get(edge.source);
      if (user === undefined || item === undefined) {
        throw new Error(`Invalid v2 legacy user-anime reference: ${edge.id}`);
      }
      ua.push([user, item, edge.weight]);
    } else if (edge.edgeType === "anime-anime") {
      const left = animeIndex.get(edge.source);
      const right = animeIndex.get(edge.target);
      if (left === undefined || right === undefined || edge.support === undefined) {
        throw new Error(`Invalid v2 legacy pair reference or support: ${edge.id}`);
      }
      aa.push([left, right, edge.weight, edge.support]);
    } else {
      throw new Error(`Invalid v2 legacy edge type: ${edge.id}`);
    }
  }
  if (userIds.length !== graph.userCount || anime.length !== graph.animeCount ||
      userIds.length + anime.length !== graph.nodeCount || ua.length + aa.length !== graph.edgeCount) {
    throw new Error("V2 legacy graph counts do not match its nodes and edges.");
  }
  const { nodes: _nodes, edges: _edges, ...metadata } = graph;
  return { ...metadata, format: "graph-compact-v2", userIds, anime, ua, aa };
}

/** Derive a separately identified, bounded visualization sample from a recommendation graph. */
export function buildExplorerGraph(
  graph: CompactGraphData,
  maxAnimeAnimeEdges = EXPLORER_AA_LIMIT,
  maxUserAnimeEdges = EXPLORER_UA_LIMIT,
): CompactGraphData {
  if (!Number.isSafeInteger(maxAnimeAnimeEdges) || maxAnimeAnimeEdges < 0 ||
      !Number.isSafeInteger(maxUserAnimeEdges) || maxUserAnimeEdges < 0) {
    throw new Error("Explorer edge limits must be nonnegative safe integers.");
  }
  if (graph.format === "graph-compact-v2" && graph.role !== "recommendation") {
    throw new Error("An explorer sample requires a v2 recommendation graph.");
  }
  const selectedUa = selectTopEdges(graph.ua, maxUserAnimeEdges);
  const selectedAa = selectTopEdges(graph.aa, maxAnimeAnimeEdges);
  const animeIndexMap = new Map<number, number>();
  const userIndexMap = new Map<number, number>();
  const anime: [number, string][] = [];
  const userIds: string[] = [];

  const remapAnime = (sourceIndex: number): number => {
    const existing = animeIndexMap.get(sourceIndex);
    if (existing !== undefined) return existing;
    const entry = graph.anime[sourceIndex];
    if (!entry) throw new Error("Explorer pair references a missing anime.");
    const nextIndex = anime.length;
    animeIndexMap.set(sourceIndex, nextIndex);
    anime.push(entry);
    return nextIndex;
  };
  const remapUser = (sourceIndex: number): number => {
    const existing = userIndexMap.get(sourceIndex);
    if (existing !== undefined) return existing;
    const userId = graph.userIds[sourceIndex];
    if (!userId) throw new Error("Explorer edge references a missing user.");
    const nextIndex = userIds.length;
    userIndexMap.set(sourceIndex, nextIndex);
    userIds.push(userId);
    return nextIndex;
  };

  const ua: [number, number, number][] = selectedUa.map(([user, item, weight]) =>
    [remapUser(user), remapAnime(item), weight],
  );
  const aa: CompactGraphDataV1["aa"] = selectedAa.map(([left, right, weight, support]) =>
    support === undefined
      ? [remapAnime(left), remapAnime(right), weight]
      : [remapAnime(left), remapAnime(right), weight, support],
  );
  const core: CompactGraphDataV1 = {
    format: "graph-compact-v1",
    generatedAt: graph.generatedAt,
    userIds,
    anime,
    ua,
    aa,
    userCount: userIds.length,
    animeCount: anime.length,
    nodeCount: userIds.length + anime.length,
    edgeCount: ua.length + aa.length,
  };
  if (graph.format === "graph-compact-v1") return core;

  const visualization = {
    policy: "abs-weight-top-k-v1" as const,
    maxUserAnimeEdges,
    maxAnimeAnimeEdges,
    excludedUserAnimeEdges: graph.ua.length - ua.length,
    excludedAnimeAnimeEdges: graph.aa.length - aa.length,
  };
  const { graphId: sourceGraphId, ...sourceMetadata } = graph;
  const withoutId: Omit<CompactGraphDataV2, "graphId"> = {
    ...sourceMetadata,
    ...core,
    format: "graph-compact-v2",
    role: "visualization",
    sourceGraphId,
    visualization,
    aa: aa.map(([left, right, weight, support]) => {
      if (support === undefined) throw new Error("V2 explorer pair support is missing.");
      return [left, right, weight, support];
    }),
  };
  return { ...withoutId, graphId: visualizationGraphId(withoutId) };
}

function selectTopEdges<T extends [number, number, number, number?]>(edges: T[], limit: number): T[] {
  if (edges.length <= limit) return [...edges];
  return [...edges].sort((a, b) =>
    Math.abs(b[2]) - Math.abs(a[2]) || a[0] - b[0] || a[1] - b[1],
  ).slice(0, limit);
}
