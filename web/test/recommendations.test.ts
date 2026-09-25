import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseCompactGraph, parseCompactModel, parseDemoCatalog } from "../src/artifacts.ts";
import type { AnimeMetadata, GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex, RecommendationResult } from "../src/domain.ts";
import { parseTextHistory } from "../src/import-history.ts";
import {
  buildGraphRecommendations,
  buildGraphRecommendationsForPreferences,
  buildCatalogCoverageRecommendations,
  buildCommunityQualityExploration,
  buildGenreOverlapExploration,
  buildModelRecommendations,
  buildModelRecommendationsForPreferences,
  buildRecommendationIndex,
  buildRecommendationIndexFromCompact,
  buildSamplePopularityExploration,
  clampModelBlendWeight,
  clampWatchWeight,
  combineHybridRecommendations,
  createCandidateEligibilityPolicy,
  explainRecommendation,
  formatWeight,
  hasActiveRecommendationFilters,
  rankEligibleCandidates,
} from "../src/recommendations.ts";
import type { RecommendationFilters } from "../src/recommendations.ts";
import { manualPreference } from "../src/preferences.ts";

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

test("model ranking retains fixture scores; rank fusion keeps full component endpoints", () => {
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
    ["Moonlit Workshop", "+16.261"], ["星の航路", "+16.261"],
    ["Quiet Satellite", "+7.937"], ["Café Nebula", "+7.813"],
    ["Glass Orchard", "+7.692"], ["Paper Current", "+7.576"],
    ["Ashen Harbor", "+7.463"],
  ]);
  assert.deepEqual(combineHybridRecommendations(graphResults, modelResults, 0)
    .map((item) => item.anime.label), graphResults.map((item) => item.anime.label));
  assert.deepEqual(combineHybridRecommendations(graphResults, modelResults, 1)
    .map((item) => item.anime.label), modelResults.map((item) => item.anime.label));
});

test("seen is exclusion only, likes seed graph, and dislikes provide signed model evidence", () => {
  const seen = manualPreference("anime:101");
  assert.deepEqual(buildGraphRecommendationsForPreferences([seen], index), []);
  assert.deepEqual(buildModelRecommendationsForPreferences([seen], index, modelIndex), []);

  const liked = { ...manualPreference("anime:101", "liked", 2), confidence: 0.25 };
  const graphResults = buildGraphRecommendationsForPreferences([liked], index);
  assert.equal(formatWeight(graphResults[0].score), "+0.292");
  assert.equal(graphResults[0].contributions[0].weightFactor, 0.5);
  const disliked = manualPreference("anime:102", "disliked", 1);
  const graphWithDislike = buildGraphRecommendationsForPreferences([liked, disliked], index);
  assert.equal(graphWithDislike.some((item) => item.anime.nodeId === "anime:102"), false);
  assert.deepEqual(buildGraphRecommendationsForPreferences([disliked], index), []);

  const modelResults = buildModelRecommendationsForPreferences([liked, disliked, seen], index, modelIndex);
  assert.equal(modelResults.some((item) => [101, 102].includes(item.anime.animeId)), false);
  assert.equal(modelResults.some((item) => item.contributions.some((part) =>
    part.watched.nodeId === "anime:102" && part.weightFactor < 0)), true);
  const likedOnly = buildModelRecommendationsForPreferences([liked], index, modelIndex);
  assert.notDeepEqual(modelResults.map((item) => [item.anime.animeId, item.score]),
    likedOnly.map((item) => [item.anime.animeId, item.score]));
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

test("one eligibility policy enforces catalog, watched, history, exclusion, and allowlist rules", () => {
  const recommendations = buildModelRecommendations(watched, weights, index, modelIndex);
  const noFilters: RecommendationFilters = { genre: "", minYear: null, maxYear: null, minScore: null };
  assert.equal(hasActiveRecommendationFilters(noFilters), false);
  assert.equal(hasActiveRecommendationFilters({ ...noFilters, genre: "  " }), false);
  const base = { index, preferences: [manualPreference("anime:101", "liked")],
    history: [], includeOnlyNodeIds: ["anime:102", "anime:105"],
    excludeNodeIds: ["anime:105"], filters: noFilters };
  const policy = createCandidateEligibilityPolicy(base);
  const unknown: RecommendationResult = { ...recommendations[0],
    anime: { nodeId: "anime:999", animeId: 999, label: "Model-only invention" } };
  const eligible = policy.evaluate([unknown, ...recommendations], new Map());
  assert.deepEqual(eligible.structurallyEligible.map((item) => item.anime.nodeId), ["anime:102"]);
  assert.deepEqual(eligible.recommendations.map((item) => item.anime.nodeId), ["anime:102"]);
  assert.equal(eligible.missingMetadataCount, 0);
  const history = parseTextHistory("102, 0, Watching, 3").entries;
  const withSeenHistory = createCandidateEligibilityPolicy({ ...base, history });
  assert.deepEqual(withSeenHistory.evaluate(recommendations, new Map()).recommendations, []);
  const withSeenPreference = createCandidateEligibilityPolicy({ ...base,
    preferences: [...base.preferences, manualPreference("anime:102")] });
  assert.deepEqual(withSeenPreference.evaluate(recommendations, new Map()).recommendations, []);
  const fallback = rankEligibleCandidates("fallback", { fallback: [unknown, ...recommendations] },
    policy, new Map());
  assert.deepEqual(fallback.recommendations.map((item) => item.anime.nodeId), ["anime:102"]);
});

test("required genre, year, and score filters reject missing metadata before hybrid rank assignment", () => {
  const recommendations = buildModelRecommendations(watched, weights, index, modelIndex);
  const metadata = new Map<number, AnimeMetadata>(
    parseDemoCatalog(fixture("catalog.json"), "synthetic catalog")
      .map((item) => [item.animeId, item]),
  );
  const filters: RecommendationFilters = {
    genre: " slice of life ", minYear: 2022, maxYear: 2018, minScore: 7.7,
  };
  assert.equal(hasActiveRecommendationFilters(filters), true);
  const policy = createCandidateEligibilityPolicy({ index,
    preferences: [manualPreference("anime:101", "liked")], history: [],
    includeOnlyNodeIds: [], excludeNodeIds: [], filters });
  const filtered = policy.evaluate(recommendations, metadata);
  assert.deepEqual(filtered.recommendations.map((item) => item.anime.label), ["Moonlit Workshop"]);
  assert.equal(filtered.missingMetadataCount, 0);
  const graphResults = buildGraphRecommendations(watched, weights, index);
  const hybrid = rankEligibleCandidates("hybrid", { graph: graphResults, model: recommendations },
    policy, metadata, 0.5);
  assert.deepEqual(hybrid.recommendations.map((item) => item.anime.nodeId), ["anime:102"]);
  assert.ok(Math.abs(hybrid.recommendations[0].score - 1_000 / 61) < 1e-12);
  assert.notEqual(combineHybridRecommendations(graphResults, recommendations, 0.5)
    .find((item) => item.anime.nodeId === "anime:102")?.score, hybrid.recommendations[0].score);
  metadata.delete(102);
  const missing = policy.evaluate(recommendations, metadata);
  assert.deepEqual(missing.recommendations, []);
  assert.equal(missing.missingMetadataCount, 1);
  assert.equal(rankEligibleCandidates("hybrid", { graph: graphResults, model: recommendations },
    policy, metadata).missingMetadataCount, 1);
});

test("catalog coverage fallback is deterministic and shares the eligibility policy", () => {
  const baseline = buildCatalogCoverageRecommendations(index);
  assert.equal(baseline.length, index.animeList.length);
  assert.ok(baseline.every((item, position) => position === 0 ||
    baseline[position - 1].score > item.score ||
    baseline[position - 1].score === item.score &&
      baseline[position - 1].anime.animeId < item.anime.animeId));
  assert.ok(baseline.every((item) => item.score === item.supportCount && item.contributions.length === 0));
  const policy = createCandidateEligibilityPolicy({ index,
    preferences: [manualPreference("anime:101", "disliked")], history: [],
    includeOnlyNodeIds: ["anime:101", "anime:102"], excludeNodeIds: ["anime:102"],
    filters: { genre: "", minYear: null, maxYear: null, minScore: null } });
  assert.deepEqual(rankEligibleCandidates("fallback", { fallback: baseline }, policy, new Map()).recommendations, []);
});

test("sample popularity counts retained user-anime edges in compact and legacy graphs", () => {
  const popularity = buildSamplePopularityExploration(index);
  assert.deepEqual(popularity.map((item) => [item.anime.animeId, item.score]), [
    [101, 4], [102, 4], [103, 3], [105, 3], [104, 2], [106, 1], [107, 1], [108, 0],
  ]);
  const legacy: GraphData = {
    generatedAt: graph.generatedAt, userCount: graph.userIds.length,
    animeCount: graph.anime.length, nodeCount: graph.userIds.length + graph.anime.length,
    edgeCount: graph.ua.length + graph.aa.length,
    nodes: [
      ...graph.userIds.map((id) => ({ id: `user:${id}`, label: id, nodeType: "user" as const })),
      ...graph.anime.map(([id, label]) => ({ id: `anime:${id}`, label, nodeType: "anime" as const })),
    ],
    edges: [
      ...graph.ua.map(([user, anime, weight], i) => ({ id: `ua:${i}`,
        source: `user:${graph.userIds[user]}`, target: `anime:${graph.anime[anime][0]}`,
        edgeType: "user-anime" as const, weight })),
      ...graph.aa.map(([left, right, weight], i) => ({ id: `aa:${i}`,
        source: `anime:${graph.anime[left][0]}`, target: `anime:${graph.anime[right][0]}`,
        edgeType: "anime-anime" as const, weight })),
    ],
  };
  assert.deepEqual(buildSamplePopularityExploration(buildRecommendationIndex(legacy)), popularity);
  assert.deepEqual(buildSamplePopularityExploration(buildRecommendationIndexFromCompact({ ...graph,
    ua: [...graph.ua].reverse(),
  })), popularity);
});

test("quality and content exploration use known metadata and only Liked genre evidence", () => {
  const metadata = new Map<number, AnimeMetadata>(
    parseDemoCatalog(fixture("catalog.json"), "synthetic catalog")
      .map((item) => [item.animeId, item]),
  );
  const quality = buildCommunityQualityExploration(index, metadata);
  assert.deepEqual(quality.slice(0, 3).map((item) => item.anime.animeId), [105, 101, 108]);
  metadata.set(105, { ...metadata.get(105)!, score: null });
  assert.equal(buildCommunityQualityExploration(index, metadata).some((item) => item.anime.animeId === 105), false);
  metadata.delete(105);
  assert.equal(buildCommunityQualityExploration(index, metadata).some((item) => item.anime.animeId === 105), false);
  metadata.set(105, parseDemoCatalog(fixture("catalog.json"), "synthetic catalog")
    .find((item) => item.animeId === 105)!);
  const related = buildGenreOverlapExploration([
    manualPreference("anime:101", "liked"), manualPreference("anime:102", "liked"),
    manualPreference("anime:103", "disliked"), manualPreference("anime:104", "seen"),
  ], index, metadata);
  assert.equal(related[0].anime.animeId, 105);
  assert.equal(related[0].score, 2);
  assert.deepEqual(related[0].sharedGenres, ["Adventure", "Fantasy"]);
  assert.deepEqual(related[0].matchingLikedTitles, ["Copper Comet", "Moonlit Workshop"]);
  const policy = createCandidateEligibilityPolicy({ index,
    preferences: [manualPreference("anime:101", "liked"), manualPreference("anime:102", "liked"),
      manualPreference("anime:103", "disliked"), manualPreference("anime:104", "seen")],
    history: [], includeOnlyNodeIds: [], excludeNodeIds: [],
    filters: { genre: "", minYear: null, maxYear: null, minScore: null } });
  assert.equal(rankEligibleCandidates("fallback", { fallback: related }, policy, metadata)
    .recommendations.some((item) => [101, 102, 103, 104].includes(item.anime.animeId)), false);
});

test("untyped legacy results label source evidence as qualitative", () => {
  const item: RecommendationResult = {
    anime: { animeId: 9, nodeId: "anime:9", label: "Candidate" },
    score: 0.4, strongest: 0.8, supportCount: 3,
    contributions: [
      ["small", 0.1], ["<Large & watched>", 0.8], ["middle", 0.4],
      ["weak negative", -0.1], ["strong negative", -0.7], ["middle negative", -0.3],
      ["neutral", 0],
    ].map(([label, weightedScore], index) => ({
      watched: { animeId: index + 1, nodeId: `anime:${index + 1}`, label: String(label) },
      edgeWeight: Number(weightedScore), weightFactor: 1, weightedScore: Number(weightedScore),
    })),
  };
  const explanation = explainRecommendation(item);
  assert.equal(explanation.kind, "qualitative");
  assert.ok(explanation.headline.includes("<Large & watched>"));
  assert.ok(explanation.headline.includes("No numeric attribution"));
  assert.deepEqual(explainRecommendation({ ...item, contributions: [] }),
    { kind: "none", headline: "Why: no direct contributing title is available." });
});
