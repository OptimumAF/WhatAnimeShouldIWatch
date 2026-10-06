import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { parseCompactGraph, parseLegacyGraph } from "../src/artifacts.js";
import { describeGraphScope } from "../src/graph-scope.js";

const fixture = (name: string) => JSON.parse(fs.readFileSync(
  new URL(`../public/demo-data/${name}`, import.meta.url), "utf8",
));

test("v3 scope separates source ratings, selected pairs, explorer omissions, and omitted user rows", () => {
  const recommendation = parseCompactGraph(fixture("graph.aggregate.compact.json"),
    "v3 recommendation", "recommendation");
  const explorer = parseCompactGraph(fixture("graph-explorer.aggregate.compact.json"),
    "v3 explorer", "visualization");
  const scope = describeGraphScope(recommendation, explorer, 12000, 4000);
  assert.match(scope.versions, /graph-compact-v3.*graph-compact-v3/);
  assert.match(scope.selection, /18\/18 source ratings selected/);
  assert.match(scope.selection, /11\/11 eligible pair edges retained/);
  assert.match(scope.explorer, /10\/11 retained pair edges \(1 omitted; sample cap 10\)/);
  assert.match(scope.explorer, /User rows are deliberately omitted from v3/);
  assert.match(scope.caveat, /does not prove no relationship/);
  assert.equal(scope.userRowsOmitted, true);
});

test("v2 scope names declared source selection and both explorer sample limits", () => {
  const source = fixture("graph.compact.json");
  source.truncation.inputRatings += 5;
  source.truncation.ratingsSkipped += 5;
  source.config.ratingSelectionPolicy = "sha256-bottom-k-v1";
  source.config.maxRatingsPerUser = 4;
  const recommendation = parseCompactGraph(source, "v2 recommendation", "recommendation");
  const explorer = parseCompactGraph(fixture("graph-explorer.compact.json"),
    "v2 explorer", "visualization");
  const scope = describeGraphScope(recommendation, explorer, 12000, 4000);
  assert.match(scope.selection, /18\/23 source ratings selected.*5 omitted before aggregation/);
  assert.match(scope.explorer, /10\/11 retained pair edges/);
  assert.match(scope.explorer, /10\/18 retained user-anime edges sampled/);
  assert.equal(scope.userRowsOmitted, false);
});

test("unversioned legacy scope refuses to infer source or explorer coverage", () => {
  const graph = parseLegacyGraph({
    generatedAt: "2026-09-24T00:00:00.000Z", userCount: 0, animeCount: 1,
    nodeCount: 1, edgeCount: 0,
    nodes: [{ id: "anime:101", label: "Invented title", nodeType: "anime" }],
    edges: [],
  }, "legacy graph");
  const scope = describeGraphScope(graph, graph, 12000, 4000);
  assert.match(scope.versions, /unversioned legacy graph.*same loaded graph/);
  assert.match(scope.selection, /does not declare.*coverage/);
  assert.match(scope.explorer, /cannot be verified/);
  assert.equal(scope.userRowsOmitted, false);
});
