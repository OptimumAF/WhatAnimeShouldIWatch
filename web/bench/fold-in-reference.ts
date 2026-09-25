/** Evaluation-only ridge fold-in candidate. The deployed browser scorer remains the reference average. */
import type { ModelRecommendationIndex, RecommendationIndex, RecommendationResult } from "../src/domain";
import type { AnimePreference } from "../src/preferences";

export interface FoldInObservation {
  embedding: readonly number[];
  itemBias: number;
  target: 1 | -1;
  weight: number;
}

/** Weighted ridge solution in observation space: Qᵀ(QQᵀ + λI)⁻¹y. */
export function fitFoldInVector(
  observations: readonly FoldInObservation[], globalMean: number, regularization = 1,
): number[] {
  if (observations.length === 0) return [];
  if (!Number.isFinite(regularization) || regularization <= 0) {
    throw new Error("Fold-in regularization must be positive.");
  }
  const factors = observations[0].embedding.length;
  const scales = observations.map((item) => {
    if (item.embedding.length !== factors || !Number.isFinite(item.weight) || item.weight <= 0) {
      throw new Error("Fold-in observations must have matching dimensions and positive weights.");
    }
    return Math.sqrt(item.weight);
  });
  const n = observations.length;
  const lower = Array.from({ length: n }, () => new Float64Array(n));
  for (let row = 0; row < n; row += 1) {
    for (let col = 0; col <= row; col += 1) {
      let value = row === col ? regularization : 0;
      for (let factor = 0; factor < factors; factor += 1) {
        value += scales[row] * scales[col] *
          observations[row].embedding[factor] * observations[col].embedding[factor];
      }
      for (let k = 0; k < col; k += 1) value -= lower[row][k] * lower[col][k];
      lower[row][col] = row === col ? Math.sqrt(value) : value / lower[col][col];
    }
  }
  const forward = new Float64Array(n);
  for (let row = 0; row < n; row += 1) {
    let value = scales[row] * (observations[row].target - globalMean - observations[row].itemBias);
    for (let col = 0; col < row; col += 1) value -= lower[row][col] * forward[col];
    forward[row] = value / lower[row][row];
  }
  const dual = new Float64Array(n);
  for (let row = n - 1; row >= 0; row -= 1) {
    let value = forward[row];
    for (let col = row + 1; col < n; col += 1) value -= lower[col][row] * dual[col];
    dual[row] = value / lower[row][row];
  }
  const vector = new Array<number>(factors).fill(0);
  for (let row = 0; row < n; row += 1) {
    for (let factor = 0; factor < factors; factor += 1) {
      vector[factor] += scales[row] * dual[row] * observations[row].embedding[factor];
    }
  }
  return vector;
}

export function buildFoldInRecommendations(
  preferences: readonly AnimePreference[], index: RecommendationIndex,
  modelIndex: ModelRecommendationIndex,
): RecommendationResult[] {
  const observed = preferences.flatMap((preference) => {
    if (preference.sentiment === "seen") return [];
    const anime = index.animeByNodeId.get(preference.nodeId);
    const modelAnime = anime && modelIndex.animeByAnimeId.get(anime.animeId);
    if (!anime || !modelAnime) return [];
    const weight = preference.importance * preference.confidence;
    return [{ anime, modelAnime, weight,
      signedWeight: preference.sentiment === "disliked" ? -weight : weight,
      target: preference.sentiment === "disliked" ? -1 as const : 1 as const }];
  });
  if (observed.length === 0) return [];
  const vector = fitFoldInVector(observed.map((item) => ({
    embedding: item.modelAnime.embedding, itemBias: item.modelAnime.bias,
    target: item.target, weight: item.weight,
  })), modelIndex.globalMean);
  const excluded = new Set(preferences.map((item) => item.nodeId));
  const results: RecommendationResult[] = [];
  for (const modelAnime of modelIndex.animeByAnimeId.values()) {
    const anime = index.animeByAnimeId.get(modelAnime.animeId);
    if (!anime || excluded.has(anime.nodeId)) continue;
    let score = modelIndex.globalMean + modelAnime.bias;
    for (let factor = 0; factor < vector.length; factor += 1) {
      score += vector[factor] * modelAnime.embedding[factor];
    }
    const contributions = observed.map((item) => {
      let similarity = 0;
      for (let factor = 0; factor < vector.length; factor += 1) {
        similarity += item.modelAnime.embedding[factor] * modelAnime.embedding[factor];
      }
      return { watched: item.anime, edgeWeight: similarity, weightFactor: item.signedWeight,
        weightedScore: similarity * item.signedWeight };
    }).sort((left, right) => right.weightedScore - left.weightedScore);
    results.push({ anime, score, strongest: contributions[0]?.weightedScore ?? 0,
      supportCount: contributions.filter((item) => item.weightedScore > 0).length,
      contributions });
  }
  return results.sort((left, right) => right.score - left.score ||
    right.supportCount - left.supportCount || right.strongest - left.strongest ||
    left.anime.animeId - right.anime.animeId);
}
