import assert from "node:assert/strict";
import { test } from "node:test";
import type { GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex } from "../src/domain.ts";
import { buildRecommendationIndex } from "../src/recommendations.ts";
import { manualPreference } from "../src/preferences.ts";
import { buildFoldInRecommendations, fitFoldInVector } from "../bench/fold-in-reference.ts";

test("ridge fold-in subtracts global and item bias from observed targets", () => {
  const vector = fitFoldInVector([
    { embedding: [2], itemBias: 0.25, target: 1, weight: 1 },
  ], 0.1);
  assert.ok(Math.abs(vector[0] - 0.26) < 1e-12);
  const orthogonal = fitFoldInVector([
    { embedding: [1, 0], itemBias: 0, target: 1, weight: 1 },
    { embedding: [0, 1], itemBias: 0, target: -1, weight: 1 },
  ], 0);
  assert.ok(Math.abs(orthogonal[0] - 0.5) < 1e-12);
  assert.ok(Math.abs(orthogonal[1] + 0.5) < 1e-12);
  assert.throws(() => fitFoldInVector([
    { embedding: [1], itemBias: 0, target: 1, weight: 0 },
  ], 0), /positive weights/);
});

test("fold-in candidate scores only unwatched catalog titles using signed observations", () => {
  const graph: GraphData = {
    generatedAt: "invented", userCount: 0, animeCount: 4, nodeCount: 4, edgeCount: 0,
    nodes: [1, 2, 3, 4].map((id) => ({ id: `anime:${id}`, label: `Title ${id}`, nodeType: "anime" })),
    edges: [],
  };
  const index = buildRecommendationIndex(graph);
  const model: ModelRecommendationIndex = {
    generatedAt: "invented", factors: 2, globalMean: 0,
    animeByAnimeId: new Map([
      [1, { animeId: 1, title: "Title 1", bias: 0, embedding: [1, 0] }],
      [2, { animeId: 2, title: "Title 2", bias: 0, embedding: [0, 1] }],
      [3, { animeId: 3, title: "Title 3", bias: 0, embedding: [0, 0] }],
      [4, { animeId: 4, title: "Title 4", bias: 0.1, embedding: [1, -1] }],
    ]),
  };
  const preferences = [manualPreference("anime:1", "liked"),
    manualPreference("anime:2", "disliked"), manualPreference("anime:3", "seen")];
  const results = buildFoldInRecommendations(preferences, index, model);
  assert.deepEqual(results.map((item) => item.anime.animeId), [4]);
  assert.ok(Math.abs(results[0].score - 1.1) < 1e-12);
  assert.equal(results[0].supportCount, 2);
});
