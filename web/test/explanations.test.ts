import assert from "node:assert/strict";
import { test } from "node:test";
import type { GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex } from "../src/domain.ts";
import { manualPreference } from "../src/preferences.ts";
import {
  buildGraphRecommendationsForPreferences, buildModelRecommendationsForPreferences,
  buildRecommendationIndex, combineHybridRecommendations, explainRecommendation,
  formatScoreEquation,
} from "../src/recommendations.ts";

const graph: GraphData = {
  generatedAt: "invented", userCount: 0, animeCount: 3, nodeCount: 3, edgeCount: 3,
  nodes: [
    { id: "anime:1", label: "Invented One", nodeType: "anime" },
    { id: "anime:2", label: "Invented Two", nodeType: "anime" },
    { id: "anime:3", label: "Invented Target", nodeType: "anime" },
  ],
  edges: [
    { id: "edge:a", source: "anime:1", target: "anime:3", edgeType: "anime-anime", weight: 0.5 },
    { id: "edge:b", source: "anime:1", target: "anime:3", edgeType: "anime-anime", weight: 0.25 },
    { id: "edge:c", source: "anime:2", target: "anime:3", edgeType: "anime-anime", weight: 0.4 },
  ],
};
const index = buildRecommendationIndex(graph);
const model: ModelRecommendationIndex = { generatedAt: "invented", factors: 2, globalMean: 0.2,
  animeByAnimeId: new Map([
    [1, { animeId: 1, title: "Invented One", bias: 0, embedding: [1, 0] }],
    [2, { animeId: 2, title: "Invented Two", bias: 0, embedding: [0, 1] }],
    [3, { animeId: 3, title: "Invented Target", bias: 0.1, embedding: [2, 3] }],
  ]) };

function scoredExplanation(result: ReturnType<typeof buildGraphRecommendationsForPreferences>[number]) {
  const explanation = explainRecommendation(result);
  assert.equal(explanation.kind, "score");
  if (explanation.kind !== "score") throw new Error("Expected a numeric score explanation");
  const shown = formatScoreEquation(explanation);
  assert.equal(shown.termUnits.reduce((sum, value) => sum + value, 0), shown.totalUnits);
  assert.ok(Math.abs(explanation.terms.reduce((sum, term) => sum + term.value, 0) -
    result.score) < 1e-8);
  return { explanation, shown };
}

test("graph explanation reconciles duplicate edges while counting distinct Liked source titles", () => {
  const result = buildGraphRecommendationsForPreferences([
    manualPreference("anime:1", "liked"), manualPreference("anime:2", "liked", 2),
  ], index)[0];
  assert.equal(result.anime.animeId, 3);
  assert.equal(result.score, 1.55);
  assert.equal(result.supportCount, 3);
  assert.equal(result.contributions.length, 3);
  const { explanation, shown } = scoredExplanation(result);
  assert.equal(explanation.engine, "graph");
  assert.equal(explanation.distinctSourceCount, 2);
  assert.deepEqual(explanation.terms.map((term) => [term.label, term.value]), [
    ["Invented Two", 0.8], ["Invented One", 0.75],
  ]);
  assert.equal(shown.line,
    "Graph score +1.550 = Invented Two (+0.800) + Invented One (+0.750).");
  assert.match(explanation.detailLines.join(" "), /3 contributing edges; 2 distinct source titles/);
  assert.match(explanation.uncertainty, /No calibrated confidence interval/);
});

test("model explanation includes global mean, item bias, normalized signed titles, and mapped counts", () => {
  const result = buildModelRecommendationsForPreferences([
    manualPreference("anime:1", "liked"), manualPreference("anime:2", "disliked"),
  ], index, model)[0];
  assert.equal(result.anime.animeId, 3);
  assert.ok(Math.abs(result.score + 0.2) < 1e-12);
  const { explanation, shown } = scoredExplanation(result);
  assert.equal(explanation.engine, "model");
  assert.equal(explanation.distinctSourceCount, 2);
  assert.deepEqual(explanation.terms.map((term) => [term.label, term.value]), [
    ["global mean", 0.2], ["item bias", 0.1],
    ["Invented Two", -1.5], ["Invented One", 1],
  ]);
  assert.equal(shown.line,
    "Model score -0.200 = global mean (+0.200) + item bias (+0.100) + " +
      "Invented Two (-1.500) + Invented One (+1.000).");
  assert.match(explanation.headline, /2\/2 supplied signals mapped/);
  assert.match(explanation.detailLines.join(" "), /÷ 2 \(/);
  assert.match(explanation.uncertainty, /not been calibrated for new users/);
});

test("fusion explanation reconciles displayed rank points and counts shared source titles once", () => {
  const graphResult = buildGraphRecommendationsForPreferences([
    manualPreference("anime:1", "liked"), manualPreference("anime:2", "liked", 2),
  ], index)[0];
  const modelResult = buildModelRecommendationsForPreferences([
    manualPreference("anime:1", "liked"), manualPreference("anime:2", "disliked"),
  ], index, model)[0];
  const result = combineHybridRecommendations([graphResult], [modelResult], 0.25)[0];
  const { explanation, shown } = scoredExplanation(result);
  assert.equal(explanation.engine, "fusion");
  assert.equal(explanation.distinctSourceCount, 2);
  assert.equal(shown.totalUnits, 1639);
  assert.match(shown.line, /^Rank points 16\.39 = graph rank #1 \(/);
  assert.match(explanation.headline, /Invented One, Invented Two/);
  assert.match(explanation.detailLines.join(" "), /Effective weights: graph 75%, model 25%/);
  assert.match(explanation.detailLines.join(" "), /Model input: Model score -0\.200/);
  assert.match(explanation.detailLines.join(" "), /global mean \(\+0\.200\).*item bias \(\+0\.100\)/);
  assert.match(explanation.uncertainty, /Ranks depend on the current eligible candidate pool/);
});

test("display rounding is an explicit term rather than changing a source contribution", () => {
  const shown = formatScoreEquation({ kind: "score", engine: "graph", headline: "",
    score: 0.0088, precision: 3, distinctSourceCount: 2,
    terms: [{ label: "Invented One", value: 0.0044 },
      { label: "Invented Two", value: 0.0044 }],
    detailLines: [], uncertainty: "" });
  assert.equal(shown.roundingAdjustmentUnits, 1);
  assert.deepEqual(shown.termUnits, [4, 4, 1]);
  assert.equal(shown.totalUnits, 9);
  assert.equal(shown.line, "Graph score +0.009 = Invented One (+0.004) + " +
    "Invented Two (+0.004) + display rounding adjustment (+0.001).");
});

test("model explanation reports preference signals absent from the loaded model", () => {
  const partialModel: ModelRecommendationIndex = { ...model,
    animeByAnimeId: new Map([[1, model.animeByAnimeId.get(1)!],
      [3, model.animeByAnimeId.get(3)!]]) };
  const result = buildModelRecommendationsForPreferences([
    manualPreference("anime:1", "liked"), manualPreference("anime:2", "disliked"),
  ], index, partialModel).find((item) => item.anime.animeId === 3);
  assert.ok(result);
  const { explanation } = scoredExplanation(result);
  assert.match(explanation.headline, /1\/2 supplied signals mapped/);
  assert.equal(explanation.distinctSourceCount, 1);
  assert.deepEqual(explanation.terms.slice(0, 3).map((term) => term.label),
    ["global mean", "item bias", "Invented One"]);
});
