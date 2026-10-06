import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { parseCompactGraph } from "../src/artifacts.js";
import type { GraphDataV1 } from "../src/artifacts.js";
import { selectAnimeNeighborhood } from "../src/network-neighborhood.js";

const invented: GraphDataV1 = {
  generatedAt: "2026-10-05T00:00:00.000Z", userCount: 1, animeCount: 7,
  nodeCount: 8, edgeCount: 8,
  nodes: [
    ...Array.from({ length: 7 }, (_, index) => ({ id: `anime:${index + 1}`,
      label: `Invented ${index + 1}`, nodeType: "anime" as const })),
    { id: "user:invented", label: "Invented user", nodeType: "user" },
  ],
  edges: [
    { id: "a2", source: "anime:1", target: "anime:2", edgeType: "anime-anime", weight: 0.8, support: 2 },
    { id: "a3", source: "anime:1", target: "anime:3", edgeType: "anime-anime", weight: -0.9, support: 4 },
    { id: "a4", source: "anime:1", target: "anime:4", edgeType: "anime-anime", weight: 0.9, support: 1 },
    { id: "a5", source: "anime:1", target: "anime:5", edgeType: "anime-anime", weight: -0.7, support: 5 },
    { id: "a6", source: "anime:1", target: "anime:6", edgeType: "anime-anime", weight: 0.7, support: 5 },
    { id: "a2-better", source: "anime:1", target: "anime:2", edgeType: "anime-anime", weight: 1.2, support: 1 },
    { id: "unrelated", source: "anime:5", target: "anime:6", edgeType: "anime-anime", weight: 4 },
    { id: "user-edge", source: "user:invented", target: "anime:1", edgeType: "user-anime", weight: 5 },
  ],
};

test("focus chooses strongest distinct signed pair evidence inside both budgets", () => {
  const focused = selectAnimeNeighborhood(invented, "anime:1", 4, 2);
  assert.ok(focused);
  assert.deepEqual(focused.edges.map((edge) => edge.id), ["a2-better", "a3"]);
  assert.deepEqual(focused.nodes.map((node) => node.id), ["anime:1", "anime:2", "anime:3"]);
  assert.equal(focused.eligiblePairEdges, 6);
  assert.equal(focused.omittedByBudget, 4);
  assert.equal(focused.edges[0].weight, 1.2);
  const byNodeBudget = selectAnimeNeighborhood(invented, "anime:1", 2, 5);
  assert.deepEqual(byNodeBudget?.edges.map((edge) => edge.id), ["a2-better"]);
});

test("equal absolute weights use support then numeric neighbor ID; threshold keeps signed values", () => {
  const focused = selectAnimeNeighborhood(invented, "anime:1", 6, 5, 0.8);
  assert.deepEqual(focused?.edges.map((edge) => edge.id),
    ["a2-better", "a3", "a4"]);
  assert.equal(focused?.edges[1].weight, -0.9);
  assert.deepEqual(selectAnimeNeighborhood(invented, "anime:1", 7, 6)?.edges.slice(-2)
    .map((edge) => edge.id), ["a5", "a6"]);
  const isolated = selectAnimeNeighborhood(invented, "anime:7", 6, 5);
  assert.equal(isolated?.center.label, "Invented 7");
  assert.deepEqual(isolated?.edges, []);
  assert.equal(selectAnimeNeighborhood(invented, "user:invented", 6, 5), null);
  assert.throws(() => selectAnimeNeighborhood(invented, "anime:1", 0, 5), RangeError);
});

test("strict aggregate graph offers only its retained one-hop pairs", () => {
  const source = JSON.parse(fs.readFileSync(new URL(
    "../public/demo-data/graph.aggregate.compact.json", import.meta.url), "utf8"));
  const graph = parseCompactGraph(source, "invented aggregate", "recommendation");
  const focused = selectAnimeNeighborhood(graph, "anime:101", 4, 3);
  assert.ok(focused);
  assert.ok(focused.edges.length <= 3);
  assert.ok(focused.edges.every((edge) => edge.edgeType === "anime-anime" &&
    (edge.source === "anime:101" || edge.target === "anime:101")));
  assert.equal(selectAnimeNeighborhood(graph, "anime:999", 4, 3), null);
});
