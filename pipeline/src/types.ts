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
