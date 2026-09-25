import type { AnimeMetadata } from "../src/artifacts.ts";
import type { RecommendationResult } from "../src/domain.ts";

const candidates: [number, string, number, number][] = [
  [201, "Orbit Knights Season 2", 10, 3],
  [202, "Orbit Knights Side Story", 9, 2],
  [203, "Cedar Tide", 8, 2],
  [204, "Glass Vale", 7, 1],
  [205, "Unknown Caravan", 6, 1],
  [206, "Orbit Knight Academy", 5, 1],
  [207, "Marble Passage", 4, 0],
  [208, "Cedar Tide Part 2", 3, 1],
];

export const franchiseResults: RecommendationResult[] = candidates.map(([animeId, label, score]) => ({
  anime: { animeId, nodeId: `anime:${animeId}`, label },
  score, strongest: 0, supportCount: 0, contributions: [],
}));

export const franchiseGains = new Map(candidates.map(([id, , , gain]) => [id, gain]));
export const franchiseTitles = new Map<number, string>([
  [200, "Orbit Knights"], ...candidates.map(([id, title]): [number, string] => [id, title]),
]);

function metadata(animeId: number, relations: AnimeMetadata["relations"]): AnimeMetadata {
  return { animeId, year: null, score: null, genres: [], studios: [], synopsis: "",
    imageUrl: "", season: null, relations };
}

export const franchiseMetadata = new Map<number, AnimeMetadata>([
  [201, metadata(201, [{ kind: "prequel", animeId: 200, title: "Orbit Knights" }])],
  [202, metadata(202, [{ kind: "side-story", animeId: 201, title: "Orbit Knights Season 2" }])],
  [203, metadata(203, [])],
  [204, metadata(204, [])],
  [206, metadata(206, [])],
  [207, metadata(207, [])],
]);

export const franchiseIdentity = new Map<number, string>([
  [201, "orbit"], [202, "orbit"], [203, "cedar"], [204, "glass"],
  [205, "unknown"], [206, "academy"], [207, "marble"], [208, "cedar"],
]);
