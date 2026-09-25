import type { EdgeType, ModelRecommendationAnime, NodeType } from "./artifacts";
import type { HistoryEntry } from "./import-history";
import type { AnimePreference } from "./preferences";

export interface ConnectedItem {
  nodeId: string;
  label: string;
  nodeType: NodeType;
  edgeType: EdgeType;
  weight: number;
}

export interface AnimeInfo {
  nodeId: string;
  animeId: number;
  label: string;
}

export interface RecommendationContribution {
  watched: AnimeInfo;
  edgeWeight: number;
  weightFactor: number;
  weightedScore: number;
}

export interface RecommendationResult {
  anime: AnimeInfo;
  score: number;
  strongest: number;
  supportCount: number;
  contributions: RecommendationContribution[];
}

export interface RecommendationIndex {
  animeList: AnimeInfo[];
  animeByNodeId: Map<string, AnimeInfo>;
  animeByAnimeId: Map<number, AnimeInfo>;
  titleLookup: Map<string, AnimeInfo[]>;
  adjacency: Map<string, { otherNodeId: string; weight: number }[]>;
  /** Counts only user-anime edges retained in the loaded recommendation graph. */
  sampledRatingCountByNodeId: Map<string, number>;
}

export interface ModelRecommendationIndex {
  generatedAt: string;
  factors: number;
  globalMean: number;
  animeByAnimeId: Map<number, ModelRecommendationAnime>;
}

export interface SeasonalAnimeItem {
  animeId: number;
  title: string;
  score: number | null;
  year: number | null;
  season: string | null;
  imageUrl: string;
}

export interface ImportedPreferenceEntry {
  anime: AnimeInfo;
  preference: AnimePreference;
}

export interface UsernameImportResult {
  entries: ImportedPreferenceEntry[];
  history: HistoryEntry[];
  duplicateCount: number;
  ratedCount: number;
  unmappedCount: number;
}
