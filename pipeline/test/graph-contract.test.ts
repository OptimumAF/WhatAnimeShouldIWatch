import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { datasetIdentity, recommendationGraphId } from "../src/core/graph-contract.js";
import { buildExplorerGraph } from "../src/core/explorer-graph.js";
import type { AnonymizedDataset, CompactGraphDataV2 } from "../src/types.js";

const fixture = JSON.parse(fs.readFileSync(
  new URL("../../web/public/demo-data/graph.compact.json", import.meta.url), "utf8",
)) as CompactGraphDataV2;

test("dataset identity ignores row order and timestamps but detects content changes", () => {
  const dataset: AnonymizedDataset = {
    generatedAt: "2026-01-01T00:00:00.000Z",
    source: "invented-fixture",
    users: [
      { userId: "z", ratings: [
        { animeId: 8, title: "Invented Eight", rawScore: 8, normalizedScore: 1 },
        { animeId: 2, title: "Invented Two", rawScore: 6, normalizedScore: -1 },
      ] },
      { userId: "a", ratings: [{ animeId: 2, title: "Invented Two", rawScore: 7, normalizedScore: 0 }] },
    ],
  };
  const first = datasetIdentity(dataset);
  const shuffled = structuredClone(dataset);
  shuffled.generatedAt = "2026-09-25T00:00:00.000Z";
  shuffled.users.reverse();
  shuffled.users[1].ratings.reverse();
  assert.deepEqual(datasetIdentity(shuffled), first);
  shuffled.users[1].ratings[0].rawScore = 9;
  assert.notEqual(datasetIdentity(shuffled).sha256, first.sha256);
});

test("v2 explorer sample has a separate role, stable source identity, and bounded edges", () => {
  const sample = buildExplorerGraph(fixture, 2, 3) as CompactGraphDataV2;
  assert.equal(sample.format, "graph-compact-v2");
  assert.equal(sample.role, "visualization");
  assert.equal(sample.sourceGraphId, fixture.graphId);
  assert.equal(sample.dataset.sha256, fixture.dataset.sha256);
  assert.equal(sample.aa.length, 2);
  assert.equal(sample.ua.length, 3);
  assert.equal(sample.visualization?.excludedAnimeAnimeEdges, fixture.aa.length - 2);
  assert.equal(sample.visualization?.excludedUserAnimeEdges, fixture.ua.length - 3);
  assert.notEqual(sample.graphId, fixture.graphId);
  assert.deepEqual(buildExplorerGraph(fixture, 2, 3), sample);
  assert.throws(() => buildExplorerGraph(sample, 1, 1), /requires a v2 recommendation graph/);
  const { graphId, ...withoutId } = fixture;
  assert.equal(recommendationGraphId({ ...withoutId, generatedAt: "2030-01-01T00:00:00.000Z" }), graphId);
});
