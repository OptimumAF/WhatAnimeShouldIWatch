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
  datasetSha256?: string;
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
  datasetSha256?: string;
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
  support?: number;
}

export interface GraphDataV1 {
  generatedAt: string;
  nodeCount: number;
  edgeCount: number;
  userCount: number;
  animeCount: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphV2Metadata {
  role: "recommendation" | "visualization";
  graphId: string;
  sourceGraphId?: string;
  dataset: {
    sha256: string;
    scope: "anonymized-ratings-content-v1";
    source: string;
  };
  semantics: {
    pairWeight: "centered-pair-preference-mean-v1";
    support: "co-raters-after-selection-v1";
    recommendationUse: "positive-only-v1";
  };
  config: {
    ratingSelectionPolicy: "all-ratings" | "sha256-bottom-k-v1";
    seed: number;
    maxRatingsPerUser: number;
    maxAnimeAnimeEdges: number;
    maxPairVisits: number;
    maxPairCandidates: number;
    minPairSupport: number;
    maxNeighborsPerAnime: number;
  };
  truncation: {
    inputRatings: number;
    selectedRatings: number;
    ratingsSkipped: number;
    potentialPairVisits: number;
    pairVisits: number;
    pairVisitsSkipped: number;
    candidatePairs: number;
    eligiblePairs: number;
    selectedPairs: number;
    excludedBySupport: number;
    excludedByNeighborLimit: number;
    excludedByOutputLimit: number;
  };
  visualization?: {
    policy: "abs-weight-top-k-v1";
    maxUserAnimeEdges: number;
    maxAnimeAnimeEdges: number;
    excludedUserAnimeEdges: number;
    excludedAnimeAnimeEdges: number;
  };
}

export interface GraphData extends GraphDataV1, GraphV2Metadata {
  format: "graph-legacy-v2";
  role: "recommendation";
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
  support?: number,
];

export interface CompactGraphDataV1 {
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

export interface CompactGraphDataV2 extends Omit<CompactGraphDataV1, "format" | "aa">, GraphV2Metadata {
  format: "graph-compact-v2";
  aa: [leftAnimeIndex: number, rightAnimeIndex: number, weight: number, support: number][];
}

export type CompactGraphData = CompactGraphDataV1 | CompactGraphDataV2;
