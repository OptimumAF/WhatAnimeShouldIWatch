import assert from "node:assert/strict";
import { test } from "node:test";
import type { RecommendationResult } from "../src/domain.ts";
import { combineHybridRecommendations, explainRecommendation } from "../src/recommendations.ts";
import { fuseRankedRecommendations } from "../bench/hybrid-fusion-reference.ts";
import { fuseMinMaxRecommendations } from "../bench/hybrid-minmax-reference.ts";

function item(id: number, score: number): RecommendationResult {
  return { anime: { animeId: id, nodeId: `anime:${id}`, label: `Invented ${id}` },
    score, strongest: score, supportCount: 1, contributions: [] };
}

test("hybrid endpoints keep every candidate from the selected component", () => {
  const graph = [item(3, 1), item(1, -2)];
  const model = [item(2, 3), item(4, -4)];
  assert.deepEqual(combineHybridRecommendations(graph, model, 0).map((result) => result.anime.animeId), [3, 1]);
  assert.deepEqual(combineHybridRecommendations(graph, model, 1).map((result) => result.anime.animeId), [2, 4]);
  assert.equal(combineHybridRecommendations(graph, model, 0)[1].score, 1_000 / 62);
  assert.deepEqual(fuseMinMaxRecommendations(graph, model, 0)
    .map((result) => result.anime.animeId), [3]);
});

test("a missing component takes full effective weight, including at the opposite slider endpoint", () => {
  const graph = [item(1, -1), item(2, -2)];
  const model = [item(3, -4), item(4, -5)];
  for (const weight of [0, 0.5, 1]) {
    assert.deepEqual(combineHybridRecommendations(graph, [], weight)
      .map((result) => result.anime.animeId), [1, 2]);
    assert.deepEqual(combineHybridRecommendations([], model, weight)
      .map((result) => result.anime.animeId), [3, 4]);
  }
  assert.deepEqual(combineHybridRecommendations([], [], 0.5), []);
  assert.equal(combineHybridRecommendations([], model, 0)[0].fusion?.modelWeight, 1);
  assert.equal(combineHybridRecommendations(graph, [], 1)[0].fusion?.graphWeight, 1);
});

test("negative values retain rank, equal scores share rank, and anime ID breaks fusion ties", () => {
  const graph = [item(3, -2), item(2, -2), item(1, -1), item(4, Number.NaN)];
  const fused = combineHybridRecommendations(graph, [], 0.5);
  assert.deepEqual(fused.map((result) => [result.anime.animeId, result.fusion?.graphRank]),
    [[1, 1], [2, 2], [3, 2]]);
  assert.equal(fused[1].score, fused[2].score);
  assert.ok(fused.every((result) => result.score > 0));
  const differentSources = combineHybridRecommendations([item(9, -1)], [item(8, -10)], 0.5);
  assert.deepEqual(differentSources.map((result) => result.anime.animeId), [8, 9]);
  assert.equal(differentSources[0].fusion?.graphRank, null);
  assert.equal(differentSources[0].fusion?.modelRank, 1);
});

test("deployed fusion agrees with the separately written reference across weights and source shapes", () => {
  const inputs: [RecommendationResult[], RecommendationResult[]][] = [
    [[item(3, 2), item(1, 2), item(2, -1)], [item(1, -3), item(4, 0), item(2, -8)]],
    [[item(2, 1)], []], [[], [item(2, -1)]], [[], []],
  ];
  for (const [graph, model] of inputs) {
    for (const weight of [0, 0.2, 0.5, 0.8, 1]) {
      const actual = combineHybridRecommendations(graph, model, weight);
      const reference = fuseRankedRecommendations(graph, model, weight);
      assert.deepEqual(actual.map((result) => [result.anime.animeId, result.score]),
        reference.map((result) => [result.anime.animeId, result.score]));
    }
  }
});

test("fusion explanation names source ranks and evidence without equating raw evidence to rank points", () => {
  const graph = [{ ...item(2, 3), contributions: [{ watched: item(1, 0).anime,
    edgeWeight: 3, weightFactor: 1, weightedScore: 3 }] }];
  const model = [{ ...item(2, -1), contributions: [{ watched: item(4, 0).anime,
    edgeWeight: -2, weightFactor: 1, weightedScore: -2 }] }];
  const result = combineHybridRecommendations(graph, model, 0.5)[0];
  assert.deepEqual(explainRecommendation(result), { kind: "fusion",
    line: "Why: Graph rank #1; positive evidence from Invented 1 | " +
      "Model rank #1; negative evidence from Invented 4. " +
      "Rank points combine relative positions; they are not a probability." });
  assert.deepEqual(result.contributions, []);
});
