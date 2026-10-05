/** Deterministic invented scale probe. Writes only to the ignored built demo directory. */
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { datasetIdentity, recommendationGraphId, recommendationMetadata } from
  "../../pipeline/src/core/graph-contract.js";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph.js";
import { projectAggregateGraph } from "../../pipeline/src/core/aggregate-projection.js";
import { aggregateAnimePairs } from "../../pipeline/src/core/pair-aggregation.js";
import { parseCompactGraph, parseCompactModel, parseDemoCatalog } from "../src/artifacts.js";
import type { AnonymizedDataset, CompactGraphDataV2 } from "../../pipeline/src/types.js";

const animeCount = 3_000;
const userCount = 3_000;
const ratingsPerUser = 6;
const modelFactors = 16;
const seed = 0x51a71c;
const generatedAt = "2026-10-05T00:00:00.000Z";
const outputDir = resolve(fileURLToPath(new URL("../dist/demo-data/", import.meta.url)));
const anime: [number, string][] = Array.from({ length: animeCount }, (_, index) =>
  [101 + index, index === 0 ? "Copper Comet" : `Invented Scale Title ${String(index + 1).padStart(4, "0")}`]);
const titleById = new Map(anime);
let randomState = seed;
function randomUint32(): number {
  randomState ^= randomState << 13;
  randomState ^= randomState >>> 17;
  randomState ^= randomState << 5;
  return randomState >>> 0;
}
const rounded = (value: number): number => Number(value.toFixed(4));

const users: AnonymizedDataset["users"] = [];
for (let userIndex = 0; userIndex < userCount; userIndex += 1) {
  const selected = new Set<number>([101]);
  while (selected.size < ratingsPerUser) selected.add(102 + randomUint32() % (animeCount - 1));
  const raw = [...selected].sort((left, right) => left - right)
    .map((animeId) => ({ animeId, score: animeId === 101 ? 9 : 1 + randomUint32() % 10 }));
  const mean = raw.reduce((sum, item) => sum + item.score, 0) / ratingsPerUser;
  users.push({
    userId: `invented-scale-${String(userIndex + 1).padStart(5, "0")}`,
    ratings: raw.map(({ animeId, score }) => ({
      animeId, title: titleById.get(animeId)!, rawScore: score,
      normalizedScore: rounded(score - mean),
    })),
  });
}
const dataset: AnonymizedDataset = { generatedAt, source: "invented-browser-scale-v1", users };
const pairResult = aggregateAnimePairs(users, 0, 0);
const animeIndex = new Map(anime.map(([animeId], index) => [animeId, index]));
const ua: CompactGraphDataV2["ua"] = users.flatMap((user, userIndex) =>
  user.ratings.map((rating) => [userIndex, animeIndex.get(rating.animeId)!,
    rating.normalizedScore] as [number, number, number]));
const aa: CompactGraphDataV2["aa"] = [...pairResult.pairs].map(([key, pair]) => {
  const [leftId, rightId] = key.split(":").map(Number);
  return [animeIndex.get(leftId)!, animeIndex.get(rightId)!, rounded(pair.weight), pair.support];
});
const metadata = recommendationMetadata(datasetIdentity(dataset), {
  seed: 0, maxRatingsPerUser: 0, maxAnimeAnimeEdges: 0,
  maxPairVisits: 20_000_000, maxPairCandidates: 2_500_000,
  minPairSupport: 1, maxNeighborsPerAnime: 0,
}, pairResult.stats);
const graphWithoutId: Omit<CompactGraphDataV2, "graphId"> = {
  format: "graph-compact-v2", role: "recommendation", ...metadata, generatedAt,
  userIds: users.map((user) => user.userId), anime, ua, aa,
  userCount, animeCount, nodeCount: userCount + animeCount, edgeCount: ua.length + aa.length,
};
const graph = projectAggregateGraph({ ...graphWithoutId,
  graphId: recommendationGraphId(graphWithoutId) });
const explorer = buildExplorerGraph(graph);
const catalog = {
  format: "demo-catalog-v1", generatedAt,
  anime: anime.map(([animeId, title], index) => ({
    animeId, title, year: 2000 + index % 26, score: 5 + (index % 45) / 10,
    genres: ["Adventure", index % 2 === 0 ? "Fantasy" : "Drama"],
    studios: [], synopsis: "Invented offline performance title.", imageUrl: "", season: null,
  })),
};
const model = {
  format: "model-mf-compact-v1", generatedAt, datasetSha256: graph.dataset.sha256,
  globalMean: 0, factors: modelFactors,
  animeIds: anime.map(([animeId]) => animeId), titles: anime.map(([, title]) => title),
  biases: anime.map((_, index) => rounded((index % 11 - 5) / 100)),
  embeddings: anime.map((_, index) => Array.from({ length: modelFactors }, (_, factor) =>
    rounded(Math.sin((index + 1) * (factor + 1)) / 4))),
};
parseCompactGraph(graph, "invented scale graph", "recommendation");
parseCompactGraph(explorer, "invented scale explorer", "visualization");
parseDemoCatalog(catalog, "invented scale catalog");
parseCompactModel(model, "invented scale model");

await mkdir(outputDir, { recursive: true });
const files = new Map<string, unknown>([
  ["graph.aggregate.compact.json", graph],
  ["graph-explorer.aggregate.compact.json", explorer],
  ["catalog.json", catalog],
  ["model-mf-web.compact.json", model],
]);
const assets = [];
for (const [name, value] of files) {
  const bytes = Buffer.from(`${JSON.stringify(value)}\n`);
  await writeFile(resolve(outputDir, name), bytes);
  assets.push({ name, bytes: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex") });
}
const scenario = {
  format: "invented-browser-scale-v1", seed, animeCount, userCount, ratingsPerUser,
  selectedRatings: pairResult.stats.selectedRatings,
  pairVisits: pairResult.stats.pairVisits, selectedPairs: pairResult.stats.selectedPairs,
  explorerPairs: explorer.aa.length, modelFactors, graphId: graph.graphId,
  datasetSha256: graph.dataset.sha256, assets,
};
await writeFile(resolve(outputDir, "performance-scenario.json"), `${JSON.stringify(scenario, null, 2)}\n`);
process.stdout.write(`${JSON.stringify(scenario, null, 2)}\n`);
