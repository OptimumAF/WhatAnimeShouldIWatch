import { readFileSync } from "node:fs";
import { arch, cpus, platform } from "node:os";
import { performance } from "node:perf_hooks";
import { parseCompactGraph, parseCompactModel } from "../src/artifacts.ts";
import type { GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex, RecommendationIndex, RecommendationResult } from "../src/domain.ts";
import type { HistoryEntry } from "../src/import-history.ts";
import type { AnimePreference } from "../src/preferences.ts";
import { preferenceFromHistory, manualPreference } from "../src/preferences.ts";
import {
  buildModelRecommendationsForPreferences, buildRecommendationIndex,
  buildRecommendationIndexFromCompact, createCandidateEligibilityPolicy, rankEligibleCandidates,
} from "../src/recommendations.ts";
import { buildFoldInRecommendations } from "./fold-in-reference.ts";

type RawRating = { animeId: number; score: number };
type RawUser = { userId: string; ratings: RawRating[] };
type EvalCase = { userId: string; heldOutId: number; observed: RawRating[]; preferences: AnimePreference[] };
type Method = "average" | "foldIn";
const emptyMetadata = new Map();
const emptyFilters = { genre: "", minYear: null, maxYear: null, minScore: null };

function readJson(url: URL): unknown {
  return JSON.parse(readFileSync(url, "utf8"));
}

function compactModelIndex(): ModelRecommendationIndex {
  const artifact = parseCompactModel(readJson(new URL("../public/demo-data/model-mf-web.compact.json", import.meta.url)),
    "synthetic model");
  return { generatedAt: artifact.generatedAt, factors: artifact.factors,
    globalMean: artifact.globalMean,
    animeByAnimeId: new Map(artifact.animeIds.map((animeId, i) => [animeId, {
      animeId, title: artifact.titles[i], bias: artifact.biases[i], embedding: artifact.embeddings[i],
    }])),
  };
}

function subsets<T>(items: readonly T[], size: number): T[][] {
  if (size === 0) return [[]];
  if (items.length < size) return [];
  return items.flatMap((item, i) => subsets(items.slice(i + 1), size - 1)
    .map((rest) => [item, ...rest]));
}

function observedPreference(rating: RawRating): AnimePreference {
  const entry: HistoryEntry = { provider: "local", sourceId: String(rating.animeId),
    title: `Invented ${rating.animeId}`, animeId: rating.animeId,
    status: "completed", sourceStatus: "completed", progressEpisodes: null,
    score: rating.score, scoreScale: "local-10" };
  return preferenceFromHistory(entry, `anime:${rating.animeId}`)!;
}

function enumerateCases(users: readonly RawUser[], index: RecommendationIndex, seedCount: number): EvalCase[] {
  const cases: EvalCase[] = [];
  for (const user of users) {
    const deduplicated = new Map<number, number>();
    for (const rating of user.ratings) {
      if (index.animeByAnimeId.has(rating.animeId)) deduplicated.set(rating.animeId, rating.score);
    }
    const ratings = [...deduplicated].map(([animeId, score]) => ({ animeId, score }))
      .sort((left, right) => left.animeId - right.animeId);
    for (const holdout of ratings.filter((item) => item.score >= 7)) {
      for (const observed of subsets(ratings.filter((item) => item.animeId !== holdout.animeId), seedCount)) {
        const preferences = observed.map(observedPreference);
        if (!preferences.some((item) => item.sentiment !== "seen")) continue;
        cases.push({ userId: user.userId, heldOutId: holdout.animeId, observed, preferences });
      }
    }
  }
  return cases;
}

function eligibleResults(
  method: Method, preferences: readonly AnimePreference[],
  index: RecommendationIndex, model: ModelRecommendationIndex,
): RecommendationResult[] {
  const policy = createCandidateEligibilityPolicy({ index, preferences, history: [],
    includeOnlyNodeIds: [], excludeNodeIds: [], filters: emptyFilters });
  const results = method === "average"
    ? buildModelRecommendationsForPreferences(preferences, index, model)
    : buildFoldInRecommendations(preferences, index, model);
  return rankEligibleCandidates("model", { model: results }, policy, emptyMetadata).recommendations;
}

function summarize(cases: readonly EvalCase[], method: Method,
  index: RecommendationIndex, model: ModelRecommendationIndex,
) {
  const ranks = cases.map((item) => {
    const results = eligibleResults(method, item.preferences, index, model);
    const position = results.findIndex((result) => result.anime.animeId === item.heldOutId);
    return { case: `${item.userId}:${item.heldOutId}:${item.observed.map((rating) => rating.animeId).join("+")}`,
      rank: position < 0 ? null : position + 1 };
  });
  return { count: cases.length,
    hitAt3: ranks.filter((item) => item.rank !== null && item.rank <= 3).length / cases.length,
    reciprocalRank: ranks.reduce((sum, item) => sum + (item.rank === null ? 0 : 1 / item.rank), 0) / cases.length,
    ranks };
}

function largeSyntheticWorkload(): { index: RecommendationIndex; model: ModelRecommendationIndex } {
  let state = 0x5eed2026;
  function random(): number {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  }
  const nodes = Array.from({ length: 1_000 }, (_, i) => ({
    id: `anime:${i + 1}`, label: `Invented ${i + 1}`, nodeType: "anime" as const,
  }));
  const graph: GraphData = { generatedAt: "invented", userCount: 0, animeCount: nodes.length,
    nodeCount: nodes.length, edgeCount: 0, nodes, edges: [] };
  const model: ModelRecommendationIndex = { generatedAt: "invented", factors: 96, globalMean: 0,
    animeByAnimeId: new Map(nodes.map((_, i) => [i + 1, { animeId: i + 1,
      title: `Invented ${i + 1}`, bias: (random() - 0.5) * 0.2,
      embedding: Array.from({ length: 96 }, () => (random() - 0.5) * 0.6),
    }])),
  };
  return { index: buildRecommendationIndex(graph), model };
}

function p95Latency(method: Method, seedCount: number,
  index: RecommendationIndex, model: ModelRecommendationIndex,
): number {
  const preferences = [manualPreference("anime:1", "liked"),
    manualPreference("anime:2", "disliked"), manualPreference("anime:3", "liked")]
    .slice(0, seedCount);
  for (let i = 0; i < 20; i += 1) eligibleResults(method, preferences, index, model);
  const times: number[] = [];
  for (let i = 0; i < 200; i += 1) {
    const start = performance.now();
    eligibleResults(method, preferences, index, model);
    times.push(performance.now() - start);
  }
  times.sort((left, right) => left - right);
  return Number(times[Math.ceil(times.length * 0.95) - 1].toFixed(3));
}

const input = readJson(new URL("../../fixtures/synthetic-input.json", import.meta.url)) as { users: RawUser[] };
const graph = parseCompactGraph(readJson(new URL("../public/demo-data/graph.compact.json", import.meta.url)),
  "synthetic graph");
const index = buildRecommendationIndexFromCompact(graph);
const model = compactModelIndex();
const workload = largeSyntheticWorkload();
const rows = [1, 3].map((seedCount) => {
  const cases = enumerateCases(input.users, index, seedCount);
  const average = summarize(cases, "average", index, model);
  const foldIn = summarize(cases, "foldIn", index, model);
  return { seedCount, average, foldIn,
    latencyP95Ms: { average: p95Latency("average", seedCount, workload.index, workload.model),
      foldIn: p95Latency("foldIn", seedCount, workload.index, workload.model) } };
});
const enoughCases = rows.every((row) => row.average.count >= 5);
const averagePass = enoughCases && rows.every((row) => row.average.hitAt3 >= 0.5 &&
  row.latencyP95Ms.average <= 50);
const foldInPass = enoughCases && rows.every((row) => row.foldIn.hitAt3 >= row.average.hitAt3 + 0.1 &&
  row.foldIn.reciprocalRank >= row.average.reciprocalRank && row.latencyP95Ms.foldIn <= 50);
const decision = foldInPass ? "promote-fold-in" : averagePass ? "retain-average" : "unresolved";
console.log(JSON.stringify({ protocol: "docs/decisions/0010-user-vector-evaluation.md",
  runtime: { node: process.version, platform: platform(), arch: arch(), cpu: cpus()[0]?.model },
  syntheticCatalogSize: index.animeList.length, latencyCatalogSize: workload.index.animeList.length,
  rows, enoughCases, averagePass, foldInPass, decision }, null, 2));
