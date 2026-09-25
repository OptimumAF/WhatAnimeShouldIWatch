/** Pure recommendation indexes, ranking, eligibility, and explanations. */
import type { AnimeMetadata, CompactGraphData, GraphData, ModelRecommendationAnime } from "./artifacts";
import type {
  AnimeInfo,
  ModelRecommendationIndex,
  RecommendationContribution,
  RecommendationIndex,
  RecommendationResult,
} from "./domain";
import type { AnimePreference } from "./preferences";
import { seenHistoryNodeIds } from "./import-history";
import type { HistoryEntry } from "./import-history";
import { normalizeTitle } from "./title";
export { normalizeTitle } from "./title";

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
  const sampledRatingCountByNodeId = new Map<string, number>();

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
    sampledRatingCountByNodeId.set(node.id, 0);

    const normalizedTitle = normalizeTitle(node.label);
    const existing = titleLookup.get(normalizedTitle);
    if (!existing) {
      titleLookup.set(normalizedTitle, [anime]);
    } else {
      existing.push(anime);
    }
  }

  for (const edge of graphDataValue.edges) {
    if (edge.edgeType === "user-anime") {
      const animeNodeId = animeByNodeId.has(edge.source) ? edge.source : edge.target;
      if (animeByNodeId.has(animeNodeId)) {
        sampledRatingCountByNodeId.set(
          animeNodeId, (sampledRatingCountByNodeId.get(animeNodeId) ?? 0) + 1,
        );
      }
      continue;
    }
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
    sampledRatingCountByNodeId,
  };
}

export function buildRecommendationIndexFromCompact(
  graphDataValue: CompactGraphData,
): RecommendationIndex {
  if (graphDataValue.format === "graph-compact-v2" && graphDataValue.role !== "recommendation") {
    throw new Error("A visualization graph cannot be used as a recommendation neighborhood.");
  }
  const animeList: AnimeInfo[] = [];
  const animeByNodeId = new Map<string, AnimeInfo>();
  const animeByAnimeId = new Map<number, AnimeInfo>();
  const titleLookup = new Map<string, AnimeInfo[]>();
  const adjacency = new Map<string, { otherNodeId: string; weight: number }[]>();
  const sampledRatingCountByNodeId = new Map<string, number>();

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
    sampledRatingCountByNodeId.set(nodeId, 0);

    const normalizedTitle = normalizeTitle(label);
    const existing = titleLookup.get(normalizedTitle);
    if (!existing) {
      titleLookup.set(normalizedTitle, [anime]);
    } else {
      existing.push(anime);
    }
  }

  for (const [, animeIndex] of graphDataValue.ua) {
    const anime = graphDataValue.anime[animeIndex];
    if (!anime) continue;
    const nodeId = `anime:${anime[0]}`;
    sampledRatingCountByNodeId.set(nodeId, (sampledRatingCountByNodeId.get(nodeId) ?? 0) + 1);
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
    sampledRatingCountByNodeId,
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

export function buildGraphRecommendations(
  selectedNodeIds: string[],
  selectedWeights: Map<string, number>,
  index: RecommendationIndex,
): RecommendationResult[] {
  return scoreGraphRecommendations(selectedNodeIds.map((nodeId) => ({
    nodeId, weight: clampWatchWeight(selectedWeights.get(nodeId) ?? 1),
  })), selectedNodeIds, index);
}

export function buildGraphRecommendationsForPreferences(
  preferences: readonly AnimePreference[], index: RecommendationIndex,
): RecommendationResult[] {
  return scoreGraphRecommendations(preferences
    .filter((item) => item.sentiment === "liked")
    .map((item) => ({ nodeId: item.nodeId, weight: item.importance * item.confidence })),
  preferences.map((item) => item.nodeId), index);
}

/** A nonpersonalized fallback ordered by positive pair-neighborhood coverage. */
export function buildCatalogCoverageRecommendations(index: RecommendationIndex): RecommendationResult[] {
  return index.animeList.map((anime) => {
    let positiveConnections = 0;
    let strongest = 0;
    for (const neighbor of index.adjacency.get(anime.nodeId) ?? []) {
      if (Number.isFinite(neighbor.weight) && neighbor.weight > 0) {
        positiveConnections += 1;
        strongest = Math.max(strongest, neighbor.weight);
      }
    }
    return {
      anime,
      score: positiveConnections,
      strongest,
      supportCount: positiveConnections,
      contributions: [],
    } satisfies RecommendationResult;
  }).sort((left, right) => right.score - left.score || left.anime.animeId - right.anime.animeId);
}

/** Counts are a popularity proxy for the loaded graph sample, never global audience totals. */
export function buildSamplePopularityExploration(index: RecommendationIndex): RecommendationResult[] {
  return index.animeList.map((anime) => {
    const count = index.sampledRatingCountByNodeId.get(anime.nodeId) ?? 0;
    return { anime, score: count, strongest: 0, supportCount: count, contributions: [] } satisfies RecommendationResult;
  }).sort((left, right) => right.score - left.score || left.anime.animeId - right.anime.animeId);
}

/** Community score is compared only for catalog items whose metadata is already available. */
export function buildCommunityQualityExploration(
  index: RecommendationIndex, metadataByAnimeId: ReadonlyMap<number, AnimeMetadata>,
): RecommendationResult[] {
  return index.animeList.flatMap((anime) => {
    const score = metadataByAnimeId.get(anime.animeId)?.score;
    if (score === null || score === undefined || !Number.isFinite(score)) return [];
    return [{ anime, score, strongest: 0,
      supportCount: index.sampledRatingCountByNodeId.get(anime.nodeId) ?? 0,
      contributions: [] } satisfies RecommendationResult];
  }).sort((left, right) => right.score - left.score ||
    right.supportCount - left.supportCount || left.anime.animeId - right.anime.animeId);
}

export interface GenreOverlapRecommendation extends RecommendationResult {
  sharedGenres: string[];
  matchingLikedTitles: string[];
}

/** A content-only baseline: weighted exact genre overlap with explicit Liked titles. */
export function buildGenreOverlapExploration(
  preferences: readonly AnimePreference[], index: RecommendationIndex,
  metadataByAnimeId: ReadonlyMap<number, AnimeMetadata>,
): GenreOverlapRecommendation[] {
  const likedSources = preferences.flatMap((preference) => {
    if (preference.sentiment !== "liked") return [];
    const anime = index.animeByNodeId.get(preference.nodeId);
    const metadata = anime && metadataByAnimeId.get(anime.animeId);
    if (!anime || !metadata) return [];
    const genres = new Set(metadata.genres.map((genre) => normalizeTitle(genre)).filter(Boolean));
    if (genres.size === 0) return [];
    return [{ anime, genres, weight: preference.importance * preference.confidence }];
  });
  if (likedSources.length === 0) return [];

  return index.animeList.flatMap((anime) => {
    const metadata = metadataByAnimeId.get(anime.animeId);
    if (!metadata) return [];
    const genres = new Map(metadata.genres.map((genre) => [normalizeTitle(genre), genre.trim()]));
    let score = 0;
    let strongest = 0;
    const sharedGenres = new Map<string, string>();
    const matchingLikedTitles: string[] = [];
    for (const source of likedSources) {
      if (source.anime.nodeId === anime.nodeId) continue;
      const overlap = [...source.genres].filter((genre) => genres.has(genre));
      if (overlap.length === 0) continue;
      const weighted = overlap.length * source.weight;
      score += weighted;
      strongest = Math.max(strongest, weighted);
      matchingLikedTitles.push(source.anime.label);
      for (const genre of overlap) sharedGenres.set(genre, genres.get(genre)!);
    }
    if (score <= 0) return [];
    return [{ anime, score, strongest, supportCount: matchingLikedTitles.length,
      contributions: [], sharedGenres: [...sharedGenres.values()].sort((a, b) => a.localeCompare(b)),
      matchingLikedTitles } satisfies GenreOverlapRecommendation];
  }).sort((left, right) => right.score - left.score ||
    right.supportCount - left.supportCount ||
    (index.sampledRatingCountByNodeId.get(right.anime.nodeId) ?? 0) -
      (index.sampledRatingCountByNodeId.get(left.anime.nodeId) ?? 0) ||
    left.anime.animeId - right.anime.animeId);
}

function scoreGraphRecommendations(
  seeds: readonly { nodeId: string; weight: number }[],
  excludedNodeIds: readonly string[], index: RecommendationIndex,
): RecommendationResult[] {
  const selected = new Set(excludedNodeIds);
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

  for (const { nodeId: selectedNodeId, weight: weightFactor } of seeds) {
    const neighbors = index.adjacency.get(selectedNodeId) ?? [];
    for (const neighbor of neighbors) {
      if (selected.has(neighbor.otherNodeId)) {
        continue;
      }
      // V1 weights are pair preference, so a nonpositive mean does not prove
      // opposition to a selected anime. It cannot seed or penalize a pick.
      if (neighbor.weight <= 0) {
        continue;
      }
      const weightedScore = neighbor.weight * weightFactor;

      const current = scored.get(neighbor.otherNodeId);
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
  return scoreModelRecommendations(selectedNodeIds.map((nodeId) => ({
    nodeId, weight: clampWatchWeight(selectedWeights.get(nodeId) ?? 1),
  })), selectedNodeIds, index, modelIndex);
}

export function buildModelRecommendationsForPreferences(
  preferences: readonly AnimePreference[], index: RecommendationIndex,
  modelIndex: ModelRecommendationIndex,
): RecommendationResult[] {
  return scoreModelRecommendations(preferences
    .filter((item) => item.sentiment !== "seen")
    .map((item) => ({ nodeId: item.nodeId,
      weight: (item.sentiment === "disliked" ? -1 : 1) * item.importance * item.confidence })),
  preferences.map((item) => item.nodeId), index, modelIndex);
}

function scoreModelRecommendations(
  seeds: readonly { nodeId: string; weight: number }[],
  excludedNodeIds: readonly string[], index: RecommendationIndex,
  modelIndex: ModelRecommendationIndex,
): RecommendationResult[] {
  const watchedEntries = seeds
    .map(({ nodeId, weight }) => {
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
        weight,
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

  const watchedAnimeIds = new Set(excludedNodeIds.map((nodeId) => index.animeByNodeId.get(nodeId)?.animeId));
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
  graphRecommendations: readonly RecommendationResult[],
  modelRecommendations: readonly RecommendationResult[],
  modelWeight: number,
): RecommendationResult[] {
  const graphRanks = rankComponent(graphRecommendations);
  const modelRanks = rankComponent(modelRecommendations);
  if (graphRanks.size === 0 && modelRanks.size === 0) return [];
  const requestedModelWeight = clampModelBlendWeight(modelWeight);
  const graphWeight = modelRanks.size === 0 ? 1 : graphRanks.size === 0 ? 0 : 1 - requestedModelWeight;
  const effectiveModelWeight = graphRanks.size === 0 ? 1 : modelRanks.size === 0 ? 0 : requestedModelWeight;
  const candidateIds = new Set<number>();
  if (graphWeight > 0) for (const id of graphRanks.keys()) candidateIds.add(id);
  if (effectiveModelWeight > 0) for (const id of modelRanks.keys()) candidateIds.add(id);

  return [...candidateIds].map((animeId) => {
    const graph = graphRanks.get(animeId);
    const model = modelRanks.get(animeId);
    const activeGraphRank = graphWeight > 0 ? graph?.rank ?? null : null;
    const activeModelRank = effectiveModelWeight > 0 ? model?.rank ?? null : null;
    const score = 1_000 * (
      (activeGraphRank === null ? 0 : graphWeight / (60 + activeGraphRank)) +
      (activeModelRank === null ? 0 : effectiveModelWeight / (60 + activeModelRank))
    );
    return {
      anime: (graph?.item ?? model!.item).anime,
      score,
      strongest: 0,
      supportCount: 0,
      contributions: [],
      fusion: {
        graphRank: activeGraphRank,
        modelRank: activeModelRank,
        graphWeight,
        modelWeight: effectiveModelWeight,
        graphContributions: activeGraphRank === null ? [] : graph!.item.contributions,
        modelContributions: activeModelRank === null ? [] : model!.item.contributions,
      },
    } satisfies RecommendationResult;
  }).sort((left, right) => right.score - left.score || left.anime.animeId - right.anime.animeId);
}

function rankComponent(items: readonly RecommendationResult[]): Map<number, {
  item: RecommendationResult; rank: number;
}> {
  const ordered = items.filter((item) => Number.isFinite(item.score))
    .sort((left, right) => right.score - left.score || left.anime.animeId - right.anime.animeId);
  const ranks = new Map<number, { item: RecommendationResult; rank: number }>();
  let rank = 0;
  let previousScore: number | undefined;
  for (let index = 0; index < ordered.length; index += 1) {
    const item = ordered[index];
    if (previousScore === undefined || item.score !== previousScore) rank = index + 1;
    previousScore = item.score;
    if (!ranks.has(item.anime.animeId)) ranks.set(item.anime.animeId, { item, rank });
  }
  return ranks;
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

export function clampModelBlendWeight(value: number): number {
  if (!Number.isFinite(value)) {
    return 0.5;
  }
  return Math.min(Math.max(value, MIN_MODEL_BLEND_WEIGHT), MAX_MODEL_BLEND_WEIGHT);
}

export function hasActiveRecommendationFilters(recommendationFilters: RecommendationFilters): boolean {
  return (
    recommendationFilters.genre.trim().length > 0 ||
    recommendationFilters.minYear !== null ||
    recommendationFilters.maxYear !== null ||
    (recommendationFilters.minScore ?? 0) > 0
  );
}

export interface CandidateEligibilityOptions {
  /** The loaded recommendation graph defines the available anime catalog. */
  index: RecommendationIndex;
  preferences: readonly AnimePreference[];
  history: readonly HistoryEntry[];
  /** An allowlist over already scored candidates; it never creates a score. */
  includeOnlyNodeIds: readonly string[];
  excludeNodeIds: readonly string[];
  /** Genre is the current content preference; genre/year/score are required when set. */
  filters: RecommendationFilters;
}

export interface CandidateEligibilityResult {
  /** Candidates that can be checked without optional metadata. */
  structurallyEligible: RecommendationResult[];
  recommendations: RecommendationResult[];
  missingMetadataCount: number;
}

export interface CandidateEligibilityPolicy {
  evaluate(
    recommendations: readonly RecommendationResult[],
    metadataByAnimeId: ReadonlyMap<number, AnimeMetadata>,
  ): CandidateEligibilityResult;
}

/** One policy for every ranking source. Exclusion and watch status always beat inclusion. */
export function createCandidateEligibilityPolicy(options: CandidateEligibilityOptions): CandidateEligibilityPolicy {
  const { index, filters } = options;
  const includeOnly = new Set(options.includeOnlyNodeIds);
  const blocked = new Set([
    ...options.excludeNodeIds,
    ...options.preferences.map((item) => item.nodeId),
    ...seenHistoryNodeIds(options.history, index),
  ]);
  const requireMetadata = hasActiveRecommendationFilters(filters);
  const genreFilter = filters.genre.trim().toLowerCase();
  const minYearRaw = filters.minYear;
  const maxYearRaw = filters.maxYear;
  const lowerYear =
    minYearRaw !== null && maxYearRaw !== null
      ? Math.min(minYearRaw, maxYearRaw)
      : minYearRaw;
  const upperYear =
    minYearRaw !== null && maxYearRaw !== null
      ? Math.max(minYearRaw, maxYearRaw)
      : maxYearRaw;
  const minScore = filters.minScore ?? 0;

  function matchesRequiredMetadata(metadata: AnimeMetadata): boolean {
    if (genreFilter && !metadata.genres.some((genre) => genre.trim().toLowerCase() === genreFilter)) {
      return false;
    }
    if (lowerYear !== null && (metadata.year === null || metadata.year < lowerYear)) {
      return false;
    }
    if (upperYear !== null && (metadata.year === null || metadata.year > upperYear)) {
      return false;
    }
    return minScore <= 0 || metadata.score !== null && metadata.score >= minScore;
  }

  return {
    evaluate(recommendations, metadataByAnimeId) {
      const structurallyEligible: RecommendationResult[] = [];
      const eligible: RecommendationResult[] = [];
      let missingMetadataCount = 0;
      for (const item of recommendations) {
        const known = index.animeByNodeId.get(item.anime.nodeId);
        if (!known || known.animeId !== item.anime.animeId ||
            blocked.has(item.anime.nodeId) ||
            includeOnly.size > 0 && !includeOnly.has(item.anime.nodeId)) {
          continue;
        }
        structurallyEligible.push(item);
        if (!requireMetadata) {
          eligible.push(item);
          continue;
        }
        const metadata = metadataByAnimeId.get(item.anime.animeId);
        if (!metadata) {
          missingMetadataCount += 1;
        } else if (matchesRequiredMetadata(metadata)) {
          eligible.push(item);
        }
      }
      return { structurallyEligible, recommendations: eligible, missingMetadataCount };
    },
  };
}

export type EligibilityRankingMode = "graph" | "model" | "hybrid" | "fallback";

/** Filter components before assigning hybrid ranks, then guard the displayed list too. */
export function rankEligibleCandidates(
  mode: EligibilityRankingMode,
  sources: {
    graph?: readonly RecommendationResult[];
    model?: readonly RecommendationResult[];
    fallback?: readonly RecommendationResult[];
  },
  policy: CandidateEligibilityPolicy,
  metadataByAnimeId: ReadonlyMap<number, AnimeMetadata>,
  modelBlendWeight = 0.5,
): CandidateEligibilityResult {
  if (mode === "graph") return policy.evaluate(sources.graph ?? [], metadataByAnimeId);
  if (mode === "model") return policy.evaluate(sources.model ?? [], metadataByAnimeId);
  if (mode === "fallback") return policy.evaluate(sources.fallback ?? [], metadataByAnimeId);
  const graph = policy.evaluate(sources.graph ?? [], metadataByAnimeId);
  const model = policy.evaluate(sources.model ?? [], metadataByAnimeId);
  const structural = combineHybridRecommendations(
    graph.structurallyEligible, model.structurallyEligible, modelBlendWeight,
  );
  const scored = combineHybridRecommendations(graph.recommendations, model.recommendations, modelBlendWeight);
  const complete = policy.evaluate(scored, metadataByAnimeId);
  return {
    structurallyEligible: structural,
    recommendations: complete.recommendations,
    missingMetadataCount: policy.evaluate(structural, metadataByAnimeId).missingMetadataCount,
  };
}

export type RecommendationExplanation =
  | { kind: "none" }
  | { kind: "fusion"; line: string }
  | { kind: "contributors"; positiveLine: string; negativeLine: string };

export function explainRecommendation(result: RecommendationResult): RecommendationExplanation {
  if (result.fusion) {
    const fusion = result.fusion;
    const sourceLine = (source: "Graph" | "Model", rank: number | null,
      weight: number, contributions: readonly RecommendationContribution[]): string | null => {
      if (weight === 0) return null;
      if (rank === null) return `${source}: no candidate`;
      const positive = contributions.find((item) => item.weightedScore > 0)?.watched.label;
      const negative = contributions.find((item) => item.weightedScore < 0)?.watched.label;
      return `${source} rank #${rank}` +
        (positive ? `; positive evidence from ${positive}` : "") +
        (negative ? `; negative evidence from ${negative}` : "");
    };
    const parts = [
      sourceLine("Graph", fusion.graphRank, fusion.graphWeight, fusion.graphContributions),
      sourceLine("Model", fusion.modelRank, fusion.modelWeight, fusion.modelContributions),
    ].filter((part): part is string => part !== null);
    return { kind: "fusion",
      line: `Why: ${parts.join(" | ")}. Rank points combine relative positions; they are not a probability.` };
  }
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
