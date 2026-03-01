export interface MalAnimeEntry {
  anime_id: number;
  anime_title: string;
  score: number;
}

export interface AnimeRating {
  animeId: number;
  title: string;
  rawScore: number;
  normalizedScore: number;
}

export interface AnonymizedUserRatings {
  userId: string;
  ratings: AnimeRating[];
}

export interface AnonymizedDataset {
  generatedAt: string;
  source: string;
  users: AnonymizedUserRatings[];
}

export type CompactAnimeEntry = [animeId: number, title: string];
export type CompactRatingEntry = [
  animeIndex: number,
  rawScore: number,
  normalizedScore: number,
];
export type CompactUserRatings = [userId: string, ratings: CompactRatingEntry[]];

export interface CompactAnonymizedDataset {
  format: "ratings-compact-v1";
  generatedAt: string;
  source: string;
  anime: CompactAnimeEntry[];
  users: CompactUserRatings[];
}

export interface GraphNode {
  id: string;
  label: string;
  nodeType: "user" | "anime";
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  edgeType: "user-anime" | "anime-anime";
  weight: number;
}

export interface GraphData {
  generatedAt: string;
  nodeCount: number;
  edgeCount: number;
  userCount: number;
  animeCount: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export type CompactUserAnimeEdge = [
  userIndex: number,
  animeIndex: number,
  weight: number,
];
export type CompactAnimeAnimeEdge = [
  leftAnimeIndex: number,
  rightAnimeIndex: number,
  weight: number,
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
