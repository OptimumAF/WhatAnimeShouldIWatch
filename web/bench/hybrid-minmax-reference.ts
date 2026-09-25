import type { RecommendationContribution, RecommendationResult } from "../src/domain.ts";
import { clampModelBlendWeight } from "../src/recommendations.ts";

/** Original browser min–max blend retained for protocol-locked comparison only. */
export function fuseMinMaxRecommendations(
  graphRecommendations: readonly RecommendationResult[],
  modelRecommendations: readonly RecommendationResult[],
  modelWeight: number,
): RecommendationResult[] {
  const clampedModelWeight = clampModelBlendWeight(modelWeight);
  const graphWeight = 1 - clampedModelWeight;
  const graphScale = createScoreScale(graphRecommendations);
  const modelScale = createScoreScale(modelRecommendations);
  const byAnime = new Map<number, {
    anime: RecommendationResult["anime"];
    score: number;
    strongest: number;
    supportCount: number;
    contributions: RecommendationContribution[];
  }>();
  for (const item of graphRecommendations) {
    const weighted = graphScale(item.score) * graphWeight;
    if (weighted <= 0) continue;
    byAnime.set(item.anime.animeId, {
      anime: item.anime, score: weighted, strongest: item.strongest * graphWeight,
      supportCount: item.supportCount,
      contributions: scaleContributions(item.contributions, graphWeight),
    });
  }
  for (const item of modelRecommendations) {
    const weighted = modelScale(item.score) * clampedModelWeight;
    if (weighted <= 0) continue;
    const current = byAnime.get(item.anime.animeId);
    if (!current) {
      byAnime.set(item.anime.animeId, {
        anime: item.anime, score: weighted,
        strongest: item.strongest * clampedModelWeight,
        supportCount: item.supportCount,
        contributions: scaleContributions(item.contributions, clampedModelWeight),
      });
      continue;
    }
    current.score += weighted;
    current.strongest = Math.max(current.strongest, item.strongest * clampedModelWeight);
    current.supportCount += item.supportCount;
    current.contributions = [...current.contributions]
      .concat(scaleContributions(item.contributions, clampedModelWeight))
      .sort((left, right) => right.weightedScore - left.weightedScore)
      .slice(0, 10);
  }
  return [...byAnime.values()].sort((left, right) =>
    right.score - left.score || right.supportCount - left.supportCount || right.strongest - left.strongest);
}

function createScoreScale(recommendations: readonly RecommendationResult[]): (score: number) => number {
  if (recommendations.length === 0) return () => 0;
  let min = recommendations[0].score;
  let max = recommendations[0].score;
  for (const item of recommendations) {
    if (item.score < min) min = item.score;
    if (item.score > max) max = item.score;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return (score: number) => Number.isFinite(score) ? 1 : 0;
  }
  const denominator = max - min;
  return (score: number) => !Number.isFinite(score) ? 0
    : Math.min(Math.max((score - min) / denominator, 0), 1);
}

function scaleContributions(contributions: readonly RecommendationContribution[],
  factor: number): RecommendationContribution[] {
  return contributions.map((item) => ({ ...item, weightedScore: item.weightedScore * factor }));
}
