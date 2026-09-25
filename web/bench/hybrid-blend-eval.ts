import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { arch, cpus, platform } from "node:os";
import { performance } from "node:perf_hooks";
import { aggregateAnimePairs } from "../../pipeline/src/core/pair-aggregation.ts";
import { parseCompactModel } from "../src/artifacts.ts";
import type { GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex, RecommendationIndex, RecommendationResult } from "../src/domain.ts";
import type { HistoryEntry } from "../src/import-history.ts";
import type { AnimePreference } from "../src/preferences.ts";
import { preferenceFromHistory } from "../src/preferences.ts";
import {
  buildGraphRecommendationsForPreferences, buildModelRecommendationsForPreferences,
  buildRecommendationIndex, combineHybridRecommendations, createCandidateEligibilityPolicy,
  rankEligibleCandidates,
} from "../src/recommendations.ts";
import { fuseRankedRecommendations } from "./hybrid-fusion-reference.ts";
import { fuseMinMaxRecommendations } from "./hybrid-minmax-reference.ts";

type RawRating = { animeId: number; score: number };
type RawUser = { userId: string; ratings: RawRating[] };
type Case = { userId: string; holdoutId: number; observed: RawRating[]; preferences: AnimePreference[] };
type Method = "minMax" | "rankFusion";
const noFilters = { genre: "", minYear: null, maxYear: null, minScore: null };
const noMetadata = new Map();

function readJson(url: URL): unknown {
  return JSON.parse(readFileSync(url, "utf8"));
}

function normalizedUsers(raw: readonly RawUser[], catalogIds: ReadonlySet<number>): RawUser[] {
  return raw.map((user) => {
    const byAnime = new Map<number, number>();
    for (const rating of user.ratings) {
      if (catalogIds.has(rating.animeId)) byAnime.set(rating.animeId, rating.score);
    }
    return { userId: user.userId,
      ratings: [...byAnime].map(([animeId, score]) => ({ animeId, score }))
        .sort((left, right) => left.animeId - right.animeId) };
  });
}

function graphWithoutUser(users: readonly RawUser[], excludedUserId: string,
  catalog: readonly { animeId: number; title: string }[]): RecommendationIndex {
  const train = users.filter((user) => user.userId !== excludedUserId).map((user) => {
    const mean = user.ratings.reduce((sum, rating) => sum + rating.score, 0) /
      Math.max(user.ratings.length, 1);
    return { userId: user.userId,
      ratings: user.ratings.map((rating) => ({ animeId: rating.animeId,
        normalizedScore: rating.score - mean })) };
  });
  const pairs = aggregateAnimePairs(train, 0, 0).pairs;
  const nodes = catalog.map((anime) => ({ id: `anime:${anime.animeId}`, label: anime.title,
    nodeType: "anime" as const }));
  const edges: GraphData["edges"] = [...pairs].map(([key, pair]) => {
    const [low, high] = key.split(":").map(Number);
    return { id: `aa:${key}`, source: `anime:${low}`, target: `anime:${high}`,
      edgeType: "anime-anime" as const, weight: pair.weight };
  });
  return buildRecommendationIndex({ generatedAt: "invented", userCount: train.length,
    animeCount: nodes.length, nodeCount: nodes.length, edgeCount: edges.length, nodes, edges });
}

function graphSnapshot(index: RecommendationIndex): [string, [string, number][]][] {
  return [...index.adjacency].map(([id, edges]) => [id,
    edges.map((edge) => [edge.otherNodeId, edge.weight] as [string, number])]);
}

function modelIndex(): ModelRecommendationIndex {
  const model = parseCompactModel(readJson(new URL("../public/demo-data/model-mf-web.compact.json", import.meta.url)),
    "synthetic model");
  return { generatedAt: model.generatedAt, factors: model.factors, globalMean: model.globalMean,
    animeByAnimeId: new Map(model.animeIds.map((animeId, i) => [animeId, { animeId,
      title: model.titles[i], bias: model.biases[i], embedding: model.embeddings[i] }])) };
}

function subsets<T>(items: readonly T[], size: number): T[][] {
  if (size === 0) return [[]];
  if (items.length < size) return [];
  return items.flatMap((item, i) => subsets(items.slice(i + 1), size - 1)
    .map((rest) => [item, ...rest]));
}

function preference(rating: RawRating): AnimePreference {
  const entry: HistoryEntry = { provider: "local", sourceId: String(rating.animeId),
    title: `Invented ${rating.animeId}`, animeId: rating.animeId,
    status: "completed", sourceStatus: "completed", progressEpisodes: null,
    score: rating.score, scoreScale: "local-10" };
  return preferenceFromHistory(entry, `anime:${rating.animeId}`)!;
}

function casesFor(user: RawUser, count: number): Case[] {
  return user.ratings.filter((item) => item.score >= 7).flatMap((holdout) =>
    subsets(user.ratings.filter((item) => item.animeId !== holdout.animeId), count)
      .map((observed) => ({ userId: user.userId, holdoutId: holdout.animeId,
        observed, preferences: observed.map(preference) }))
      .filter((item) => item.preferences.some((pref) => pref.sentiment !== "seen")));
}

function scoreCase(item: Case, index: RecommendationIndex, model: ModelRecommendationIndex,
  method: Method): number | null {
  const graph = buildGraphRecommendationsForPreferences(item.preferences, index);
  const modelResults = buildModelRecommendationsForPreferences(item.preferences, index, model);
  const policy = createCandidateEligibilityPolicy({ index, preferences: item.preferences, history: [],
    includeOnlyNodeIds: [], excludeNodeIds: [], filters: noFilters });
  const filteredGraph = policy.evaluate(graph, noMetadata).recommendations;
  const filteredModel = policy.evaluate(modelResults, noMetadata).recommendations;
  const referenceFusion = fuseRankedRecommendations(filteredGraph, filteredModel, 0.5);
  const deployedFusion = rankEligibleCandidates("hybrid", { graph, model: modelResults },
    policy, noMetadata, 0.5).recommendations;
  assert.deepEqual(deployedFusion.map((result) => [result.anime.animeId, result.score]),
    referenceFusion.map((result) => [result.anime.animeId, result.score]));
  const ranked = method === "minMax"
    ? fuseMinMaxRecommendations(filteredGraph, filteredModel, 0.5)
    : deployedFusion;
  const position = ranked.findIndex((result) => result.anime.animeId === item.holdoutId);
  return position < 0 ? null : position + 1;
}

function summarize(cases: readonly Case[], indexes: ReadonlyMap<string, RecommendationIndex>,
  model: ModelRecommendationIndex, method: Method) {
  const ranks = cases.map((item) => ({ case: `${item.userId}:${item.holdoutId}:` +
    item.observed.map((rating) => rating.animeId).join("+"),
    rank: scoreCase(item, indexes.get(item.userId)!, model, method) }));
  const hitCount = ranks.filter((item) => item.rank !== null && item.rank <= 3).length;
  return { count: cases.length, hitCount, hitAt3: hitCount / cases.length,
    reciprocalRank: ranks.reduce((sum, item) => sum + (item.rank === null ? 0 : 1 / item.rank), 0) /
      cases.length, ranks };
}

function timingLists(): { graph: RecommendationResult[]; model: RecommendationResult[] } {
  let state = 0x20260925;
  const random = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
  const items = Array.from({ length: 1_000 }, (_, i) => ({
    anime: { animeId: i + 1, nodeId: `anime:${i + 1}`, label: `Invented ${i + 1}` },
    score: random() * 4 - 2, strongest: 0, supportCount: 0, contributions: [],
  }));
  return { graph: items.slice(0, 800), model: items.slice(200).map((item) => ({
    ...item, score: random() * 4 - 2,
  })) };
}

function p95(method: Method | "deployedRankFusion", graph: RecommendationResult[],
  model: RecommendationResult[]): number {
  const call = method === "minMax"
    ? () => fuseMinMaxRecommendations(graph, model, 0.5)
    : method === "rankFusion" ? () => fuseRankedRecommendations(graph, model, 0.5)
      : () => combineHybridRecommendations(graph, model, 0.5);
  for (let i = 0; i < 20; i += 1) call();
  const times = [];
  for (let i = 0; i < 200; i += 1) {
    const start = performance.now();
    call();
    times.push(performance.now() - start);
  }
  times.sort((left, right) => left - right);
  return Number(times[Math.ceil(times.length * 0.95) - 1].toFixed(3));
}

const input = readJson(new URL("../../fixtures/synthetic-input.json", import.meta.url)) as {
  users: RawUser[]; anime: { animeId: number; title: string }[] };
const users = normalizedUsers(input.users, new Set(input.anime.map((anime) => anime.animeId)));
const indexes = new Map(users.map((user) => [user.userId,
  graphWithoutUser(users, user.userId, input.anime)]));
for (const user of users) {
  const changed = users.map((other) => other.userId === user.userId
    ? { ...other, ratings: other.ratings.map((rating) => ({ ...rating, score: 10 - rating.score })) }
    : other);
  assert.deepEqual(graphSnapshot(graphWithoutUser(changed, user.userId, input.anime)),
    graphSnapshot(indexes.get(user.userId)!));
}
const model = modelIndex();
const rows = [1, 3].map((count) => {
  const cases = users.flatMap((user) => casesFor(user, count));
  return { observedCount: count,
    minMax: summarize(cases, indexes, model, "minMax"),
    rankFusion: summarize(cases, indexes, model, "rankFusion") };
});
const timing = timingLists();
const latencyP95Ms = { minMax: p95("minMax", timing.graph, timing.model),
  rankFusionReference: p95("rankFusion", timing.graph, timing.model),
  deployedRankFusion: p95("deployedRankFusion", timing.graph, timing.model) };
const qualityGate = rows.every((row) => row.minMax.count >= 5 &&
  row.rankFusion.hitCount >= row.minMax.hitCount - 1 &&
  row.rankFusion.reciprocalRank >= row.minMax.reciprocalRank - 0.05);
const latencyGate = latencyP95Ms.deployedRankFusion <= 50;
console.log(JSON.stringify({ protocol: "docs/decisions/0011-hybrid-blend-validation.md",
  runtime: { node: process.version, platform: platform(), arch: arch(), cpu: cpus()[0]?.model },
  syntheticCatalogSize: input.anime.length, targetUserLeakageCheck: true,
  rows, latencyP95Ms, qualityGate, latencyGate,
  candidateDecision: qualityGate && latencyGate ? "promote-rank-fusion-if-behavior-tests-pass" : "retain-min-max-pending-review" },
null, 2));
if (!qualityGate || !latencyGate) process.exitCode = 1;
