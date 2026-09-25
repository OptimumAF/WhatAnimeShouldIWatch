import fs from "node:fs";
import path from "node:path";
import { datasetIdentity, recommendationGraphId, recommendationMetadata } from "./core/graph-contract.js";
import { buildExplorerGraph } from "./core/explorer-graph.js";
import { aggregateAnimePairs } from "./core/pair-aggregation.js";
import { getRepoRoot } from "./paths.js";
import type { CompactGraphDataV2 } from "./types.js";

interface FixtureAnime {
  animeId: number;
  title: string;
  year: number;
  score: number;
  genres: string[];
  embedding: number[];
  bias: number;
  relations?: { kind: "prequel" | "sequel" | "alternative-version" | "side-story" | "spin-off";
    animeId: number; title: string }[];
}

interface FixtureInput {
  anime: FixtureAnime[];
  users: { userId: string; ratings: { animeId: number; score: number }[] }[];
}

const repoRoot = getRepoRoot(import.meta.url);
const sourcePath = path.join(repoRoot, "fixtures", "synthetic-input.json");
const outputDir = path.join(repoRoot, "web", "public", "demo-data");
const generatedAt = "2026-09-24T00:00:00.000Z";
const input = JSON.parse(fs.readFileSync(sourcePath, "utf8")) as FixtureInput;

const catalogById = new Map(input.anime.map((anime) => [anime.animeId, anime]));
if (catalogById.size !== input.anime.length) {
  throw new Error("Duplicate synthetic catalog anime ID.");
}
if (new Set(input.users.map((user) => user.userId)).size !== input.users.length) {
  throw new Error("Duplicate synthetic user ID.");
}
for (const anime of input.anime) {
  if (!Number.isSafeInteger(anime.animeId) || anime.animeId <= 0 ||
      anime.embedding.length !== 2 || !anime.embedding.every(Number.isFinite)) {
    throw new Error(`Invalid synthetic anime ${anime.animeId}.`);
  }
  for (const relation of anime.relations ?? []) {
    if (!catalogById.has(relation.animeId) || relation.animeId === anime.animeId) {
      throw new Error(`Invalid synthetic relationship from ${anime.animeId}.`);
    }
  }
}

let duplicateRatings = 0;
let unknownRatings = 0;
const users = input.users.map((user) => {
  const ratingsById = new Map<number, number>();
  for (const rating of user.ratings) {
    if (!Number.isFinite(rating.score) || rating.score < 0 || rating.score > 10) {
      throw new Error(`Invalid score in ${user.userId}.`);
    }
    if (!catalogById.has(rating.animeId)) {
      unknownRatings += 1;
      continue;
    }
    if (ratingsById.has(rating.animeId)) {
      duplicateRatings += 1;
    }
    ratingsById.set(rating.animeId, rating.score);
  }
  const mean = [...ratingsById.values()].reduce((sum, score) => sum + score, 0) /
    Math.max(ratingsById.size, 1);
  return {
    userId: user.userId,
    ratings: [...ratingsById].sort(([left], [right]) => left - right).map(([animeId, score]) => ({
      animeId,
      title: catalogById.get(animeId)!.title,
      rawScore: score,
      normalizedScore: score - mean,
    })),
  };
});

if (duplicateRatings === 0 || unknownRatings === 0 ||
    !users.some((user) => user.ratings.length === 0) ||
    !users.some((user) => user.ratings.length === 1) ||
    !input.anime.some((anime) => !users.some((user) => user.ratings.some((rating) => rating.animeId === anime.animeId))) ||
    !input.anime.some((anime) => /[^\u0000-\u007f]/.test(anime.title))) {
  throw new Error("The synthetic fixture is missing a required edge case.");
}

const anime = input.anime.map((item): [number, string] => [item.animeId, item.title]);
const animeIndex = new Map(anime.map(([animeId], index) => [animeId, index]));
const ua: CompactGraphDataV2["ua"] = [];
for (let userIndex = 0; userIndex < users.length; userIndex += 1) {
  for (const rating of users[userIndex].ratings) {
    const index = animeIndex.get(rating.animeId);
    if (index === undefined) throw new Error(`Missing anime ${rating.animeId}.`);
    ua.push([userIndex, index, roundWeight(rating.normalizedScore)]);
  }
}
const pairResult = aggregateAnimePairs(users, 0, 0);
const aa: CompactGraphDataV2["aa"] = [...pairResult.pairs].map(([key, pair]) => {
  const [low, high] = key.split(":").map(Number);
  const left = animeIndex.get(low);
  const right = animeIndex.get(high);
  if (left === undefined || right === undefined) throw new Error(`Missing pair ${key}.`);
  return [left, right, roundWeight(pair.weight), pair.support];
});

const dataset = { generatedAt, source: "synthetic-fixture", users };
const metadata = recommendationMetadata(datasetIdentity(dataset), {
  seed: 0,
  maxRatingsPerUser: 0,
  maxAnimeAnimeEdges: 0,
  maxPairVisits: 20_000_000,
  maxPairCandidates: 2_500_000,
  minPairSupport: 1,
  maxNeighborsPerAnime: 0,
}, pairResult.stats);
const graphWithoutId: Omit<CompactGraphDataV2, "graphId"> = {
  format: "graph-compact-v2",
  role: "recommendation",
  ...metadata,
  generatedAt,
  userIds: users.map((user) => user.userId),
  anime,
  ua,
  aa,
  userCount: users.length,
  animeCount: anime.length,
  nodeCount: users.length + anime.length,
  edgeCount: ua.length + aa.length,
};
const graph: CompactGraphDataV2 = {
  ...graphWithoutId,
  graphId: recommendationGraphId(graphWithoutId),
};
const explorerGraph = buildExplorerGraph(graph, 10, 10);
const catalog = {
  format: "demo-catalog-v1",
  generatedAt,
  anime: input.anime.map(({ animeId, title, year, score, genres, relations }) => ({
    animeId, title, year, score, genres,
    studios: [], synopsis: "Invented offline demo title.", imageUrl: "", season: null,
    ...(relations ? { relations } : {}),
  })),
};
const model = {
  format: "model-mf-compact-v1",
  generatedAt,
  globalMean: 0,
  factors: 2,
  animeIds: input.anime.map((item) => item.animeId),
  titles: input.anime.map((item) => item.title),
  biases: input.anime.map((item) => item.bias),
  embeddings: input.anime.map((item) => item.embedding),
};

const outputs = new Map<string, unknown>([
  ["graph.compact.json", graph],
  ["graph-explorer.compact.json", explorerGraph],
  ["catalog.json", catalog],
  ["model-mf-web.compact.json", model],
]);
const check = process.argv.includes("--check");
if (!check) fs.mkdirSync(outputDir, { recursive: true });
for (const [filename, value] of outputs) {
  const filenamePath = path.join(outputDir, filename);
  const expected = `${JSON.stringify(value, null, 2)}\n`;
  if (check) {
    if (!fs.existsSync(filenamePath) || fs.readFileSync(filenamePath, "utf8") !== expected) {
      throw new Error(`Synthetic artifact missing or stale: ${filenamePath}. Run npm run data:fixture.`);
    }
  } else {
    fs.writeFileSync(filenamePath, expected);
  }
}
process.stdout.write(`${check ? "Verified" : "Generated"} synthetic graph/catalog/model: ` +
  `${users.length} users, ${anime.length} anime, ${aa.length} pairs, ` +
  `${duplicateRatings} duplicate and ${unknownRatings} unknown input ratings handled.\n`);

function roundWeight(value: number): number {
  return Number(value.toFixed(4));
}
