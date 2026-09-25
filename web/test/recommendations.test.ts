import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseCompactGraph, parseCompactModel, parseDemoCatalog } from "../src/artifacts.ts";
import type { AnimeMetadata, GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex, RecommendationResult } from "../src/domain.ts";
import {
  applyRecommendationFilters,
  buildGraphRecommendations,
  buildModelRecommendations,
  buildRecommendationIndex,
  buildRecommendationIndexFromCompact,
  clampModelBlendWeight,
  clampWatchWeight,
  combineHybridRecommendations,
  explainRecommendation,
  filterCandidateEligibility,
  formatWeight,
  hasActiveRecommendationFilters,
} from "../src/recommendations.ts";
import type { RecommendationFilters } from "../src/recommendations.ts";

const fixtureRoot = new URL("../public/demo-data/", import.meta.url);
const fixture = (name: string): unknown => JSON.parse(readFileSync(new URL(name, fixtureRoot), "utf8"));
const graph = parseCompactGraph(fixture("graph.compact.json"), "synthetic graph");
const index = buildRecommendationIndexFromCompact(graph);
const compactModel = parseCompactModel(fixture("model-mf-web.compact.json"), "synthetic model");
const modelIndex: ModelRecommendationIndex = {
  generatedAt: compactModel.generatedAt,
  factors: compactModel.factors,
  globalMean: compactModel.globalMean,
  animeByAnimeId: new Map(compactModel.animeIds.map((animeId, i) => [animeId, {
    animeId,
    title: compactModel.titles[i],
    bias: compactModel.biases[i],
    embedding: compactModel.embeddings[i],
  }])),
};
const watched = ["anime:101"];
const weights = new Map([["anime:101", 1]]);

test("compact index and graph scoring keep the synthetic rank, weight, and watched exclusion", () => {
  const results = buildGraphRecommendations(watched, weights, index);
  assert.deepEqual(results.map((item) => [item.anime.label, formatWeight(item.score)]), [
    ["Moonlit Workshop", "+0.583"],
    ["星の航路", "+0.417"],
  ]);
  assert.equal(results.some((item) => item.anime.nodeId === "anime:101"), false);
  assert.equal(results[0].supportCount, 1);
  assert.equal(results[0].contributions[0].watched.nodeId, "anime:101");
  assert.equal(clampWatchWeight(Number.NaN), 1);
  assert.equal(clampWatchWeight(20), 3);
  assert.equal(clampWatchWeight(-2), 0.2);
  assert.deepEqual(
    buildGraphRecommendations(watched, new Map([["anime:101", 20]]), index)
      .map((item) => formatWeight(item.score)),
    ["+1.750", "+1.250"],
  );

  const legacy: GraphData = {
    generatedAt: graph.generatedAt,
    userCount: 0,
    animeCount: graph.anime.length,
    nodeCount: graph.anime.length,
    edgeCount: graph.aa.length,
    nodes: graph.anime.map(([animeId, label]) => ({ id: `anime:${animeId}`, label, nodeType: "anime" })),
    edges: graph.aa.map(([left, right, weight], i) => ({
      id: `aa:${i}`, source: `anime:${graph.anime[left][0]}`,
      target: `anime:${graph.anime[right][0]}`, edgeType: "anime-anime", weight,
    })),
  };
  const legacyResults = buildGraphRecommendations(watched, weights, buildRecommendationIndex(legacy));
  assert.deepEqual(legacyResults.map((item) => [item.anime.label, item.score]),
    results.map((item) => [item.anime.label, item.score]));
});

test("model and hybrid ranking retain their fixture scores and blend endpoints", () => {
  const graphResults = buildGraphRecommendations(watched, weights, index);
  const modelResults = buildModelRecommendations(watched, weights, index, modelIndex);
  assert.deepEqual(modelResults.map((item) => [item.anime.label, formatWeight(item.score)]), [
    ["星の航路", "+0.940"], ["Moonlit Workshop", "+0.760"],
    ["Quiet Satellite", "+0.520"], ["Café Nebula", "+0.490"],
    ["Glass Orchard", "+0.380"], ["Paper Current", "-0.090"],
    ["Ashen Harbor", "-0.400"],
  ]);
  assert.equal(modelResults.some((item) => item.anime.animeId === 101), false);
  assert.deepEqual(buildModelRecommendations([], weights, index, modelIndex), []);
  assert.equal(clampModelBlendWeight(Number.NaN), 0.5);
  assert.equal(clampModelBlendWeight(2), 1);

  const hybrid = combineHybridRecommendations(graphResults, modelResults, 0.5);
  assert.deepEqual(hybrid.map((item) => [item.anime.label, formatWeight(item.score)]), [
    ["Moonlit Workshop", "+0.933"], ["星の航路", "+0.500"],
    ["Quiet Satellite", "+0.343"], ["Café Nebula", "+0.332"],
    ["Glass Orchard", "+0.291"], ["Paper Current", "+0.116"],
  ]);
  // Existing min-max blending drops the lowest graph score, even at a zero model weight.
  assert.deepEqual(combineHybridRecommendations(graphResults, modelResults, 0)
    .map((item) => item.anime.label), ["Moonlit Workshop"]);
  assert.equal(combineHybridRecommendations(graphResults, modelResults, 1)[0].anime.label, "星の航路");
});

test("v1 negative pair preference never seeds or penalizes graph candidates", () => {
  const tinyGraph: GraphData = {
    generatedAt: "synthetic", userCount: 0, animeCount: 5, nodeCount: 5, edgeCount: 4,
    nodes: [
      { id: "anime:1", label: "Positive", nodeType: "anime" },
      { id: "anime:2", label: "Negative", nodeType: "anime" },
      { id: "anime:3", label: "Candidate", nodeType: "anime" },
      { id: "anime:4", label: "Neutral only", nodeType: "anime" },
      { id: "anime:5", label: "Negative only", nodeType: "anime" },
    ],
    edges: [
      { id: "aa:1:3", source: "anime:1", target: "anime:3", edgeType: "anime-anime", weight: 0.6 },
      { id: "aa:2:3", source: "anime:2", target: "anime:3", edgeType: "anime-anime", weight: -0.4 },
      { id: "aa:1:4", source: "anime:1", target: "anime:4", edgeType: "anime-anime", weight: 0 },
      { id: "aa:2:5", source: "anime:2", target: "anime:5", edgeType: "anime-anime", weight: -0.8 },
    ],
  };
  const tinyIndex = buildRecommendationIndex(tinyGraph);
  const positiveFirst = buildGraphRecommendations(["anime:1", "anime:2"], new Map(), tinyIndex);
  const negativeFirst = buildGraphRecommendations(["anime:2", "anime:1"], new Map(), tinyIndex);
  assert.deepEqual(buildGraphRecommendations(["anime:2"], new Map(), tinyIndex), []);
  assert.deepEqual(positiveFirst, negativeFirst);
  assert.deepEqual(positiveFirst.map((item) => item.anime.animeId), [3]);
  assert.equal(positiveFirst[0].score, 0.6);
  assert.deepEqual(positiveFirst[0].contributions.map((item) => formatWeight(item.weightedScore)),
    ["+0.600"]);
  const compact = parseCompactGraph({
    format: "graph-compact-v1", generatedAt: "2026-01-01T00:00:00.000Z", userIds: [], userCount: 0,
    animeCount: 5, nodeCount: 5, edgeCount: 4,
    anime: [[1, "Positive"], [2, "Negative"], [3, "Candidate"],
      [4, "Neutral only"], [5, "Negative only"]],
    ua: [], aa: [[0, 2, 0.6], [1, 2, -0.4], [0, 3, 0], [1, 4, -0.8]],
  }, "signed synthetic graph");
  assert.deepEqual(buildGraphRecommendations(
    ["anime:2", "anime:1"], new Map(), buildRecommendationIndexFromCompact(compact),
  ), positiveFirst);
});

test("candidate and metadata eligibility retain exclusion precedence and missing count", () => {
  const recommendations = buildModelRecommendations(watched, weights, index, modelIndex);
  const onlyMoonlit = filterCandidateEligibility(
    recommendations, ["anime:102", "anime:105"], ["anime:105"],
  );
  assert.deepEqual(onlyMoonlit.map((item) => item.anime.nodeId), ["anime:102"]);
  const noFilters: RecommendationFilters = { genre: "", minYear: null, maxYear: null, minScore: null };
  assert.equal(hasActiveRecommendationFilters(noFilters), false);
  assert.equal(applyRecommendationFilters(recommendations, noFilters, new Map()).recommendations,
    recommendations);

  const metadata = new Map<number, AnimeMetadata>(
    parseDemoCatalog(fixture("catalog.json"), "synthetic catalog")
      .map((item) => [item.animeId, item]),
  );
  const filters: RecommendationFilters = {
    genre: " slice of life ", minYear: 2022, maxYear: 2018, minScore: 7.7,
  };
  assert.equal(hasActiveRecommendationFilters(filters), true);
  const filtered = applyRecommendationFilters(recommendations, filters, metadata);
  assert.deepEqual(filtered.recommendations.map((item) => item.anime.label), ["Moonlit Workshop"]);
  assert.equal(filtered.missingMetadataCount, 0);
  metadata.delete(102);
  const missing = applyRecommendationFilters(recommendations, filters, metadata);
  assert.deepEqual(missing.recommendations, []);
  assert.equal(missing.missingMetadataCount, 1);
});

test("explanations select the two strongest contributors of each sign as plain text", () => {
  const item: RecommendationResult = {
    anime: { animeId: 9, nodeId: "anime:9", label: "Candidate" },
    score: 0.4, strongest: 0.8, supportCount: 3,
    contributions: [
      ["small", 0.1], ["<Large & watched>", 0.8], ["middle", 0.4],
      ["weak negative", -0.1], ["strong negative", -0.7], ["middle negative", -0.3],
      ["neutral", 0],
    ].map(([label, weightedScore]) => ({
      watched: { animeId: 1, nodeId: "anime:1", label: String(label) },
      edgeWeight: Number(weightedScore), weightFactor: 1, weightedScore: Number(weightedScore),
    })),
  };
  assert.deepEqual(explainRecommendation(item), {
    kind: "contributors",
    positiveLine: "Why+: <Large & watched> (+0.800) | middle (+0.400)",
    negativeLine: "Why-: strong negative (-0.700) | middle negative (-0.300)",
  });
  assert.deepEqual(explainRecommendation({ ...item, contributions: [] }), { kind: "none" });
});
