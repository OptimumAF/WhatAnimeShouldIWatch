/** Pure recommendation indexes, ranking, eligibility, and explanations. */
import type { AnimeMetadata, CompactGraphData, GraphData, ModelRecommendationAnime } from "./artifacts";
import type {
  AnimeInfo,
  ModelRecommendationIndex,
  RecommendationContribution,
  RecommendationIndex,
  RecommendationResult,
} from "./domain";

export interface RecommendationFilters {
  genre: string;
  minYear: number | null;
  maxYear: number | null;
  minScore: number | null;
}

export const MIN_WATCH_WEIGHT = 0.2;
export const MAX_WATCH_WEIGHT = 3;
export const MIN_MODEL_BLEND_WEIGHT = 0;
export const MAX_MODEL_BLEND_WEIGHT = 1;

export function buildRecommendationIndex(graphDataValue: GraphData): RecommendationIndex {
  const animeList: AnimeInfo[] = [];
  const animeByNodeId = new Map<string, AnimeInfo>();
  const animeByAnimeId = new Map<number, AnimeInfo>();
  const titleLookup = new Map<string, AnimeInfo[]>();
  const adjacency = new Map<string, { otherNodeId: string; weight: number }[]>();

  for (const node of graphDataValue.nodes) {
    if (node.nodeType !== "anime") {
      continue;
    }

    const animeId = parseAnimeId(node.id);
    const anime: AnimeInfo = {
      nodeId: node.id,
      animeId,
      label: node.label,
    };

    animeList.push(anime);
    animeByNodeId.set(node.id, anime);
    animeByAnimeId.set(animeId, anime);

    const normalizedTitle = normalizeTitle(node.label);
    const existing = titleLookup.get(normalizedTitle);
    if (!existing) {
      titleLookup.set(normalizedTitle, [anime]);
    } else {
      existing.push(anime);
    }
  }

  for (const edge of graphDataValue.edges) {
    if (edge.edgeType !== "anime-anime") {
      continue;
    }
    if (!animeByNodeId.has(edge.source) || !animeByNodeId.has(edge.target)) {
      continue;
    }

    pushAdjacency(adjacency, edge.source, edge.target, edge.weight);
    pushAdjacency(adjacency, edge.target, edge.source, edge.weight);
  }

  return {
    animeList,
    animeByNodeId,
    animeByAnimeId,
    titleLookup,
    adjacency,
  };
}

export function buildRecommendationIndexFromCompact(
  graphDataValue: CompactGraphData,
): RecommendationIndex {
  const animeList: AnimeInfo[] = [];
  const animeByNodeId = new Map<string, AnimeInfo>();
  const animeByAnimeId = new Map<number, AnimeInfo>();
  const titleLookup = new Map<string, AnimeInfo[]>();
  const adjacency = new Map<string, { otherNodeId: string; weight: number }[]>();

  for (const animeEntry of graphDataValue.anime) {
    const animeId = animeEntry[0];
    const label = String(animeEntry[1]);
    const nodeId = `anime:${animeId}`;
    const anime: AnimeInfo = {
      nodeId,
      animeId,
      label,
    };

    animeList.push(anime);
    animeByNodeId.set(nodeId, anime);
    animeByAnimeId.set(animeId, anime);

    const normalizedTitle = normalizeTitle(label);
    const existing = titleLookup.get(normalizedTitle);
    if (!existing) {
      titleLookup.set(normalizedTitle, [anime]);
    } else {
      existing.push(anime);
    }
  }

  for (const [leftAnimeIndex, rightAnimeIndex, weight] of graphDataValue.aa) {
    const leftAnime = graphDataValue.anime[leftAnimeIndex];
    const rightAnime = graphDataValue.anime[rightAnimeIndex];
    if (!leftAnime || !rightAnime || !Number.isFinite(weight)) {
      continue;
    }

    const leftNodeId = `anime:${leftAnime[0]}`;
    const rightNodeId = `anime:${rightAnime[0]}`;
    pushAdjacency(adjacency, leftNodeId, rightNodeId, weight);
    pushAdjacency(adjacency, rightNodeId, leftNodeId, weight);
  }

  return {
    animeList,
    animeByNodeId,
    animeByAnimeId,
    titleLookup,
    adjacency,
  };
}

function pushAdjacency(
  adjacency: Map<string, { otherNodeId: string; weight: number }[]>,
  source: string,
  target: string,
  weight: number,
): void {
  const list = adjacency.get(source);
  if (!list) {
    adjacency.set(source, [{ otherNodeId: target, weight }]);
    return;
  }
  list.push({ otherNodeId: target, weight });
}

function parseAnimeId(nodeId: string): number {
  const value = nodeId.startsWith("anime:") ? nodeId.slice("anime:".length) : nodeId;
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? -1 : parsed;
}

export function normalizeTitle(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

export function buildGraphRecommendations(
  selectedNodeIds: string[],
  selectedWeights: Map<string, number>,
  index: RecommendationIndex,
): RecommendationResult[] {
  const selected = new Set(selectedNodeIds);
  const scored = new Map<
    string,
    {
      score: number;
      strongest: number;
      supportCount: number;
      sourceMap: Map<
        string,
        { edgeWeight: number; weightFactor: number; weightedScore: number }
      >;
    }
  >();

  for (const selectedNodeId of selectedNodeIds) {
    const weightFactor = clampWatchWeight(selectedWeights.get(selectedNodeId) ?? 1);
    const neighbors = index.adjacency.get(selectedNodeId) ?? [];
    for (const neighbor of neighbors) {
      if (selected.has(neighbor.otherNodeId)) {
        continue;
      }
      const weightedScore = neighbor.weight * weightFactor;

      const current = scored.get(neighbor.otherNodeId);
      if (neighbor.weight <= 0) {
        if (current) {
          current.sourceMap.set(selectedNodeId, {
            edgeWeight: neighbor.weight,
            weightFactor,
            weightedScore,
          });
        }
        continue;
      }

      if (!current) {
        scored.set(neighbor.otherNodeId, {
          score: weightedScore,
          strongest: weightedScore,
          supportCount: 1,
          sourceMap: new Map([
            [
              selectedNodeId,
              {
                edgeWeight: neighbor.weight,
                weightFactor,
                weightedScore,
              },
            ],
          ]),
        });
        continue;
      }

      current.score += weightedScore;
      current.strongest = Math.max(current.strongest, weightedScore);
      current.supportCount += 1;
      current.sourceMap.set(selectedNodeId, {
        edgeWeight: neighbor.weight,
        weightFactor,
        weightedScore,
      });
    }
  }

  return [...scored.entries()]
    .map(([nodeId, aggregate]) => {
      const anime = index.animeByNodeId.get(nodeId);
      if (!anime) {
        return null;
      }
      const contributions = [...aggregate.sourceMap.entries()]
        .map(([watchedNodeId, source]) => {
          const watched = index.animeByNodeId.get(watchedNodeId);
          if (!watched) {
            return null;
          }
          return {
            watched,
            edgeWeight: source.edgeWeight,
            weightFactor: source.weightFactor,
            weightedScore: source.weightedScore,
          } satisfies RecommendationContribution;
        })
        .filter((value): value is RecommendationContribution => value !== null)
        .sort((left, right) => right.weightedScore - left.weightedScore);

      return {
        anime,
        score: aggregate.score,
        strongest: aggregate.strongest,
        supportCount: aggregate.supportCount,
        contributions,
      } satisfies RecommendationResult;
    })
    .filter((value): value is RecommendationResult => value !== null)
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      if (right.supportCount !== left.supportCount) {
        return right.supportCount - left.supportCount;
      }
      return right.strongest - left.strongest;
    });
}

export function buildModelRecommendations(
  selectedNodeIds: string[],
  selectedWeights: Map<string, number>,
  index: RecommendationIndex,
  modelIndex: ModelRecommendationIndex,
): RecommendationResult[] {
  const watchedEntries = selectedNodeIds
    .map((nodeId) => {
      const anime = index.animeByNodeId.get(nodeId);
      if (!anime) {
        return null;
      }
      const modelAnime = modelIndex.animeByAnimeId.get(anime.animeId);
      if (!modelAnime) {
        return null;
      }
      return {
        anime,
        modelAnime,
        weight: clampWatchWeight(selectedWeights.get(nodeId) ?? 1),
      };
    })
    .filter(
      (
        value,
      ): value is {
        anime: AnimeInfo;
        modelAnime: ModelRecommendationAnime;
        weight: number;
      } => value !== null,
    );

  if (watchedEntries.length === 0) {
    return [];
  }

  const factors = watchedEntries[0].modelAnime.embedding.length;
  if (factors === 0) {
    return [];
  }

  const watchedAnimeIds = new Set(watchedEntries.map((entry) => entry.anime.animeId));
  const userVector = new Float32Array(factors);
  let denominator = 0;

  for (const watched of watchedEntries) {
    denominator += Math.abs(watched.weight);
    for (let i = 0; i < factors; i += 1) {
      userVector[i] += watched.modelAnime.embedding[i] * watched.weight;
    }
  }

  if (denominator <= 0) {
    denominator = watchedEntries.length;
  }
  for (let i = 0; i < factors; i += 1) {
    userVector[i] /= denominator;
  }

  const scored: RecommendationResult[] = [];
  for (const [animeId, modelAnime] of modelIndex.animeByAnimeId.entries()) {
    if (watchedAnimeIds.has(animeId)) {
      continue;
    }
    if (modelAnime.embedding.length !== factors) {
      continue;
    }

    let score = modelAnime.bias + modelIndex.globalMean;
    for (let i = 0; i < factors; i += 1) {
      score += userVector[i] * modelAnime.embedding[i];
    }

    const contributions = watchedEntries
      .map((watched) => {
        let similarity = 0;
        for (let i = 0; i < factors; i += 1) {
          similarity += watched.modelAnime.embedding[i] * modelAnime.embedding[i];
        }
        const weightedScore = similarity * watched.weight;
        return {
          watched: watched.anime,
          edgeWeight: similarity,
          weightFactor: watched.weight,
          weightedScore,
        } satisfies RecommendationContribution;
      })
      .sort((left, right) => right.weightedScore - left.weightedScore);

    const anime =
      index.animeByAnimeId.get(animeId) ??
      ({
        nodeId: `anime:${animeId}`,
        animeId,
        label: modelAnime.title,
      } satisfies AnimeInfo);

    const strongest = contributions.length > 0 ? contributions[0].weightedScore : 0;
    const supportCount = contributions.filter((item) => item.weightedScore > 0).length;

    scored.push({
      anime,
      score,
      strongest,
      supportCount,
      contributions,
    });
  }

  return scored.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    if (right.supportCount !== left.supportCount) {
      return right.supportCount - left.supportCount;
    }
    return right.strongest - left.strongest;
  });
}

export function combineHybridRecommendations(
  graphRecommendations: RecommendationResult[],
  modelRecommendations: RecommendationResult[],
  modelWeight: number,
): RecommendationResult[] {
  const clampedModelWeight = clampModelBlendWeight(modelWeight);
  const graphWeight = 1 - clampedModelWeight;

  const graphScale = createScoreScale(graphRecommendations);
  const modelScale = createScoreScale(modelRecommendations);

  const byAnime = new Map<
    number,
    {
      anime: AnimeInfo;
      score: number;
      strongest: number;
      supportCount: number;
      contributions: RecommendationContribution[];
    }
  >();

  for (const item of graphRecommendations) {
    const normalized = graphScale(item.score);
    const weighted = normalized * graphWeight;
    if (weighted <= 0) {
      continue;
    }
    byAnime.set(item.anime.animeId, {
      anime: item.anime,
      score: weighted,
      strongest: item.strongest * graphWeight,
      supportCount: item.supportCount,
      contributions: scaleContributions(item.contributions, graphWeight),
    });
  }

  for (const item of modelRecommendations) {
    const normalized = modelScale(item.score);
    const weighted = normalized * clampedModelWeight;
    if (weighted <= 0) {
      continue;
    }
    const current = byAnime.get(item.anime.animeId);
    if (!current) {
      byAnime.set(item.anime.animeId, {
        anime: item.anime,
        score: weighted,
        strongest: item.strongest * clampedModelWeight,
        supportCount: item.supportCount,
        contributions: scaleContributions(item.contributions, clampedModelWeight),
      });
      continue;
    }

    current.score += weighted;
    current.strongest = Math.max(
      current.strongest,
      item.strongest * clampedModelWeight,
    );
    current.supportCount += item.supportCount;
    current.contributions = [...current.contributions]
      .concat(scaleContributions(item.contributions, clampedModelWeight))
      .sort((left, right) => right.weightedScore - left.weightedScore)
      .slice(0, 10);
  }

  return [...byAnime.values()]
    .map((value) => ({
      anime: value.anime,
      score: value.score,
      strongest: value.strongest,
      supportCount: value.supportCount,
      contributions: value.contributions,
    }))
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      if (right.supportCount !== left.supportCount) {
        return right.supportCount - left.supportCount;
      }
      return right.strongest - left.strongest;
    });
}

function createScoreScale(
  recommendations: RecommendationResult[],
): (score: number) => number {
  if (recommendations.length === 0) {
    return () => 0;
  }
  let min = recommendations[0].score;
  let max = recommendations[0].score;
  for (const item of recommendations) {
    if (item.score < min) {
      min = item.score;
    }
    if (item.score > max) {
      max = item.score;
    }
  }

  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return (score: number) => (Number.isFinite(score) ? 1 : 0);
  }

  const denominator = max - min;
  return (score: number) => {
    if (!Number.isFinite(score)) {
      return 0;
    }
    const normalized = (score - min) / denominator;
    return Math.min(Math.max(normalized, 0), 1);
  };
}

function scaleContributions(
  contributions: RecommendationContribution[],
  factor: number,
): RecommendationContribution[] {
  return contributions.map((item) => ({
    watched: item.watched,
    edgeWeight: item.edgeWeight,
    weightFactor: item.weightFactor,
    weightedScore: item.weightedScore * factor,
  }));
}

export function formatWeight(value: number): string {
  const normalized = Math.abs(value) < 0.0005 ? 0 : value;
  const rounded = normalized.toFixed(3);
  return normalized > 0 ? `+${rounded}` : rounded;
}

export function clampWatchWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(Math.max(value, MIN_WATCH_WEIGHT), MAX_WATCH_WEIGHT);
}

export function normalizeImportedScoreToWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }

  if (value <= MAX_WATCH_WEIGHT) {
    return clampWatchWeight(value);
  }

  const mapped = 1 + (value - 5) / 5;
  return clampWatchWeight(mapped);
}

export function clampModelBlendWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 0.5;
  }
  return Math.min(Math.max(value, MIN_MODEL_BLEND_WEIGHT), MAX_MODEL_BLEND_WEIGHT);
}

export function filterCandidateEligibility(
  recommendations: RecommendationResult[],
  includeCandidateNodeIds: readonly string[],
  excludeCandidateNodeIds: readonly string[],
): RecommendationResult[] {
  const includeSet = new Set(includeCandidateNodeIds);
  const excludeSet = new Set(excludeCandidateNodeIds);
  return recommendations.filter((item) => {
    if (excludeSet.has(item.anime.nodeId)) {
      return false;
    }
    if (includeSet.size > 0 && !includeSet.has(item.anime.nodeId)) {
      return false;
    }
    return true;
  });
}

export function hasActiveRecommendationFilters(recommendationFilters: RecommendationFilters): boolean {
  return (
    recommendationFilters.genre.length > 0 ||
    recommendationFilters.minYear !== null ||
    recommendationFilters.maxYear !== null ||
    (recommendationFilters.minScore ?? 0) > 0
  );
}

export function applyRecommendationFilters(
  recommendations: RecommendationResult[],
  recommendationFilters: RecommendationFilters,
  animeMetadataCache: ReadonlyMap<number, AnimeMetadata>,
): {
  recommendations: RecommendationResult[];
  missingMetadataCount: number;
} {
  if (!hasActiveRecommendationFilters(recommendationFilters)) {
    return {
      recommendations,
      missingMetadataCount: 0,
    };
  }

  const minYearRaw = recommendationFilters.minYear;
  const maxYearRaw = recommendationFilters.maxYear;
  const lowerYear =
    minYearRaw !== null && maxYearRaw !== null
      ? Math.min(minYearRaw, maxYearRaw)
      : minYearRaw;
  const upperYear =
    minYearRaw !== null && maxYearRaw !== null
      ? Math.max(minYearRaw, maxYearRaw)
      : maxYearRaw;
  const minScore = recommendationFilters.minScore ?? 0;
  const genreFilter = recommendationFilters.genre.trim().toLowerCase();

  let missingMetadataCount = 0;
  const filtered = recommendations.filter((item) => {
    const metadata = animeMetadataCache.get(item.anime.animeId);
    if (!metadata) {
      missingMetadataCount += 1;
      return false;
    }

    if (genreFilter) {
      const hasGenre = metadata.genres.some(
        (genre) => genre.trim().toLowerCase() === genreFilter,
      );
      if (!hasGenre) {
        return false;
      }
    }

    if (lowerYear !== null) {
      if (metadata.year === null || metadata.year < lowerYear) {
        return false;
      }
    }

    if (upperYear !== null) {
      if (metadata.year === null || metadata.year > upperYear) {
        return false;
      }
    }

    if (minScore > 0) {
      if (metadata.score === null || metadata.score < minScore) {
        return false;
      }
    }

    return true;
  });

  return {
    recommendations: filtered,
    missingMetadataCount,
  };
}

export type RecommendationExplanation =
  | { kind: "none" }
  | { kind: "contributors"; positiveLine: string; negativeLine: string };

export function explainRecommendation(result: RecommendationResult): RecommendationExplanation {
  if (result.contributions.length === 0) {
    return { kind: "none" };
  }

  const positives = result.contributions
    .filter((item) => item.weightedScore > 0)
    .sort((left, right) => right.weightedScore - left.weightedScore)
    .slice(0, 2);
  const negatives = result.contributions
    .filter((item) => item.weightedScore < 0)
    .sort((left, right) => left.weightedScore - right.weightedScore)
    .slice(0, 2);

  const positiveLine = positives.length > 0
    ? `Why+: ${positives.map((item) => `${item.watched.label} (${formatWeight(item.weightedScore)})`).join(" | ")}`
    : "Why+: no strong positive contributors.";
  const negativeLine = negatives.length > 0
    ? `Why-: ${negatives.map((item) => `${item.watched.label} (${formatWeight(item.weightedScore)})`).join(" | ")}`
    : "Why-: no notable negative contributors.";

  return { kind: "contributors", positiveLine, negativeLine };
}
