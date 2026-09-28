import { parseCompactGraph } from "../../../web/src/artifacts.js";
import type { CompactGraphDataV2, CompactGraphDataV3 } from "../types.js";
import { aggregateRecommendationGraphId, recommendationGraphId } from "./graph-contract.js";

/** Retain pair evidence and truthful source counts while removing every user row. */
export function projectAggregateGraph(source: CompactGraphDataV2): CompactGraphDataV3 {
  parseCompactGraph(source, "aggregate projection source", "recommendation");
  if (source.format !== "graph-compact-v2" || source.role !== "recommendation") {
    throw new Error("Aggregate projection requires a v2 recommendation graph.");
  }
  const { graphId: _sourceId, ...sourceWithoutId } = source;
  if (recommendationGraphId(sourceWithoutId) !== source.graphId) {
    throw new Error("Aggregate projection source graphId does not match its content.");
  }
  const withoutId: Omit<CompactGraphDataV3, "graphId"> = {
    format: "graph-compact-v3",
    role: "recommendation",
    dataset: { ...source.dataset },
    semantics: { ...source.semantics },
    config: { ...source.config },
    truncation: { ...source.truncation },
    projection: { policy: "omit-user-anime-v1" },
    generatedAt: source.generatedAt,
    userIds: [],
    anime: source.anime.map(([id, title]) => [id, title]),
    ua: [],
    aa: source.aa.map(([left, right, weight, support]) => [left, right, weight, support]),
    userCount: 0,
    animeCount: source.anime.length,
    nodeCount: source.anime.length,
    edgeCount: source.aa.length,
  };
  const result = { ...withoutId, graphId: aggregateRecommendationGraphId(withoutId) };
  parseCompactGraph(result, "aggregate projection", "recommendation");
  return result;
}
