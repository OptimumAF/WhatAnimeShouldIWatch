import type { RecommendationResult } from "../src/domain.ts";
import { clampModelBlendWeight } from "../src/recommendations.ts";

const RRF_K = 60;
const RANK_POINT_SCALE = 1_000;

function ranked(items: readonly RecommendationResult[]): Map<number, { item: RecommendationResult; rank: number }> {
  const ordered = items.filter((item) => Number.isFinite(item.score))
    .sort((left, right) => right.score - left.score || left.anime.animeId - right.anime.animeId);
  const ranks = new Map<number, { item: RecommendationResult; rank: number }>();
  let rank = 0;
  let lastScore: number | undefined;
  for (let i = 0; i < ordered.length; i += 1) {
    const item = ordered[i];
    if (lastScore === undefined || item.score !== lastScore) rank = i + 1;
    lastScore = item.score;
    if (!ranks.has(item.anime.animeId)) ranks.set(item.anime.animeId, { item, rank });
  }
  return ranks;
}

/** Evaluation-only reference; no code from this module enters the browser bundle. */
export function fuseRankedRecommendations(
  graph: readonly RecommendationResult[], model: readonly RecommendationResult[], modelWeight: number,
): RecommendationResult[] {
  const graphRanks = ranked(graph);
  const modelRanks = ranked(model);
  if (graphRanks.size === 0 && modelRanks.size === 0) return [];
  const weight = clampModelBlendWeight(modelWeight);
  const graphWeight = modelRanks.size === 0 ? 1 : graphRanks.size === 0 ? 0 : 1 - weight;
  const effectiveModelWeight = graphRanks.size === 0 ? 1 : modelRanks.size === 0 ? 0 : weight;
  const activeIds = new Set<number>();
  if (graphWeight > 0) for (const id of graphRanks.keys()) activeIds.add(id);
  if (effectiveModelWeight > 0) for (const id of modelRanks.keys()) activeIds.add(id);
  return [...activeIds].map((id) => {
    const graphHit = graphRanks.get(id);
    const modelHit = modelRanks.get(id);
    const score = RANK_POINT_SCALE * (
      (graphHit && graphWeight > 0 ? graphWeight / (RRF_K + graphHit.rank) : 0) +
      (modelHit && effectiveModelWeight > 0 ? effectiveModelWeight / (RRF_K + modelHit.rank) : 0)
    );
    const source = graphHit?.item ?? modelHit!.item;
    return { ...source, score, contributions: [] };
  }).sort((left, right) => right.score - left.score || left.anime.animeId - right.anime.animeId);
}
