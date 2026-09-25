import { createHash } from "node:crypto";
import { PAIR_CAP_POLICY, type PairSelectionStats } from "./pair-aggregation.js";
import type { AnonymizedDataset, CompactGraphDataV2, GraphV2Metadata } from "../types.js";

export const GRAPH_SEMANTICS: GraphV2Metadata["semantics"] = {
  pairWeight: "centered-pair-preference-mean-v1",
  support: "co-raters-after-selection-v1",
  recommendationUse: "positive-only-v1",
};

export function datasetIdentity(dataset: AnonymizedDataset): GraphV2Metadata["dataset"] {
  const hash = createHash("sha256");
  hash.update(`${JSON.stringify(["source", dataset.source])}\n`);
  for (const user of [...dataset.users].sort((a, b) =>
    a.userId < b.userId ? -1 : a.userId > b.userId ? 1 : 0,
  )) {
    hash.update(`${JSON.stringify(["user", user.userId])}\n`);
    for (const rating of [...user.ratings].sort((a, b) => a.animeId - b.animeId)) {
      hash.update(`${JSON.stringify(["rating", rating.animeId, rating.title, rating.rawScore, rating.normalizedScore])}\n`);
    }
  }
  return { sha256: hash.digest("hex"), scope: "anonymized-ratings-content-v1", source: dataset.source };
}

export function recommendationMetadata(
  dataset: GraphV2Metadata["dataset"],
  config: Omit<GraphV2Metadata["config"], "ratingSelectionPolicy">,
  stats: PairSelectionStats,
): Omit<GraphV2Metadata, "role" | "graphId" | "sourceGraphId" | "visualization"> {
  return {
    dataset,
    semantics: GRAPH_SEMANTICS,
    config: {
      ...config,
      ratingSelectionPolicy: config.maxRatingsPerUser > 0 ? PAIR_CAP_POLICY : "all-ratings",
    },
    truncation: {
      inputRatings: stats.inputRatings,
      selectedRatings: stats.selectedRatings,
      ratingsSkipped: stats.ratingsSkippedByUserCap,
      potentialPairVisits: stats.potentialPairVisits,
      pairVisits: stats.pairVisits,
      pairVisitsSkipped: stats.pairVisitsSkippedByUserCap,
      candidatePairs: stats.candidatePairs,
      eligiblePairs: stats.eligiblePairs,
      selectedPairs: stats.selectedPairs,
      excludedBySupport: stats.excludedBySupport,
      excludedByNeighborLimit: stats.excludedByNeighborLimit,
      excludedByOutputLimit: stats.excludedByOutputLimit,
    },
  };
}

export function recommendationGraphId(
  graph: Omit<CompactGraphDataV2, "graphId">,
): string {
  const { dataset, semantics, config, truncation, userIds, anime, ua, aa } = graph;
  return createHash("sha256")
    .update(JSON.stringify({ dataset, semantics, config, truncation, userIds, anime, ua, aa }))
    .digest("hex");
}

export function visualizationGraphId(
  graph: Omit<CompactGraphDataV2, "graphId">,
): string {
  const { sourceGraphId, visualization, userIds, anime, ua, aa } = graph;
  return createHash("sha256")
    .update(JSON.stringify({ sourceGraphId, visualization, userIds, anime, ua, aa }))
    .digest("hex");
}
