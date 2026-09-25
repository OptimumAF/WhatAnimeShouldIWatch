import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  ArtifactValidationError,
  parseCompactGraph,
  parseCompactModel,
  parseDemoCatalog,
  parseLegacyGraph,
  parseLegacyModel,
} from "../src/artifacts.ts";

const fixtureRoot = new URL("../public/demo-data/", import.meta.url);
const fixture = (name: string): any => JSON.parse(readFileSync(new URL(name, fixtureRoot), "utf8"));
const copy = <T>(value: T): T => structuredClone(value);

function compactV1(): any {
  const { role, graphId, sourceGraphId, dataset, semantics, config, truncation, visualization, ...graph } =
    fixture("graph.compact.json");
  return { ...graph, format: "graph-compact-v1" };
}

function legacyGraph(): any {
  const compact = fixture("graph.compact.json");
  const nodes = [
    ...compact.userIds.map((id: string) => ({ id: `user:${id}`, label: `User ${id}`, nodeType: "user" })),
    ...compact.anime.map(([id, title]: [number, string]) => ({ id: `anime:${id}`, label: title, nodeType: "anime" })),
  ];
  const edges = [
    ...compact.ua.map(([user, item, weight]: [number, number, number]) => ({
      id: `ua:${user}:${item}`, source: `user:${compact.userIds[user]}`,
      target: `anime:${compact.anime[item][0]}`, edgeType: "user-anime", weight,
    })),
    ...compact.aa.map(([left, right, weight, support]: [number, number, number, number?]) => ({
      id: `aa:${left}:${right}`, source: `anime:${compact.anime[left][0]}`,
      target: `anime:${compact.anime[right][0]}`, edgeType: "anime-anime", weight, support,
    })),
  ];
  return { generatedAt: compact.generatedAt, userCount: compact.userCount,
    animeCount: compact.animeCount, nodeCount: nodes.length, edgeCount: edges.length, nodes, edges };
}

function legacyModel(): any {
  const compact = fixture("model-mf-web.compact.json");
  return { generatedAt: compact.generatedAt, globalMean: compact.globalMean,
    factors: compact.factors, animeCount: compact.animeIds.length,
    anime: compact.animeIds.map((animeId: number, i: number) => ({
      animeId, title: compact.titles[i], bias: compact.biases[i],
      embedding: compact.embeddings[i],
    })) };
}

test("accepts current synthetic compact and legacy contracts, including three-value pair tuples", () => {
  const graph = fixture("graph.compact.json");
  const explorer = fixture("graph-explorer.compact.json");
  const model = fixture("model-mf-web.compact.json");
  assert.equal(parseCompactGraph(graph, "graph fixture"), graph);
  assert.equal(parseCompactGraph(explorer, "explorer fixture", "visualization"), explorer);
  assert.equal(parseDemoCatalog(fixture("catalog.json"), "catalog fixture").length, 8);
  assert.equal(parseCompactModel(model, "model fixture"), model);
  assert.equal(parseLegacyGraph(legacyGraph(), "legacy graph").edgeCount, graph.edgeCount);
  const v2Legacy = { ...legacyGraph(), format: "graph-legacy-v2", role: "recommendation",
    graphId: graph.graphId, dataset: graph.dataset, semantics: graph.semantics,
    config: graph.config, truncation: graph.truncation };
  assert.equal(parseLegacyGraph(v2Legacy, "v2 legacy graph").edgeCount, graph.edgeCount);
  const missingSupport = copy(v2Legacy);
  delete missingSupport.edges.find((edge: { edgeType: string }) => edge.edgeType === "anime-anime").support;
  assert.throws(() => parseLegacyGraph(missingSupport, "v2 legacy graph"), /support/);
  assert.throws(() => parseLegacyGraph({ ...v2Legacy, version: 2 }, "v2 legacy graph"), /version.*unsupported/);
  const mislabeledCompact = { ...graph, format: "graph-compact-v1" };
  assert.throws(() => parseCompactGraph(mislabeledCompact, "mislabeled graph"), /role.*requires a v2/);
  assert.throws(() => parseLegacyGraph({ ...legacyGraph(), role: "recommendation" }, "mislabeled legacy"),
    /role.*requires a v2/);
  assert.equal(parseLegacyModel(legacyModel(), "legacy model").anime.length, model.animeIds.length);
  const oldPairs = compactV1();
  oldPairs.aa = oldPairs.aa.map(([left, right, weight]: [number, number, number]) => [left, right, weight]);
  assert.equal(parseCompactGraph(oldPairs, "three-value graph"), oldPairs);
});

test("v1 contracts accept selected user-rating and pair subsets with matching counts", () => {
  const compact = compactV1();
  compact.ua = compact.ua.slice(0, -1);
  compact.aa = [compact.aa[0]];
  compact.edgeCount = compact.ua.length + 1;
  assert.equal(parseCompactGraph(compact, "selected compact graph"), compact);
  assert.equal(compact.aa[0][3], 3);

  const legacy = legacyGraph();
  legacy.edges = legacy.edges.filter((edge: { edgeType: string }) => edge.edgeType === "user-anime")
    .slice(0, -1).concat([legacy.edges.find((edge: { edgeType: string }) => edge.edgeType === "anime-anime")]);
  legacy.edgeCount = legacy.edges.length;
  assert.equal(parseLegacyGraph(legacy, "selected legacy graph"), legacy);
});

test("rejects unsupported versions, malformed tuples, duplicate IDs, and broken graph references", () => {
  const original = fixture("graph.compact.json");
  const cases: [string, (value: any) => void, RegExp][] = [
    ["format", (v) => { v.format = "graph-compact-v3"; }, /format.*unsupported/],
    ["extra version", (v) => { v.version = 2; }, /version.*unsupported/],
    ["anime tuple", (v) => { v.anime[0].push("extra"); }, /anime\[0\].*exactly 2/],
    ["ua tuple", (v) => { v.ua[0].pop(); }, /ua\[0\].*exactly 3/],
    ["aa tuple", (v) => { v.aa[0].push(2); }, /aa\[0\].*exactly 4/],
    ["user ID", (v) => { v.userIds[1] = v.userIds[0]; }, /userIds\[1\].*duplicates/],
    ["anime ID", (v) => { v.anime[1][0] = v.anime[0][0]; }, /anime\[1\]\[0\].*duplicates/],
    ["user reference", (v) => { v.ua[0][0] = 999; }, /ua\[0\]\[0\].*outside/],
    ["anime reference", (v) => { v.aa[0][1] = 999; }, /aa\[0\]\[1\].*outside/],
    ["edge count", (v) => { v.edgeCount += 1; }, /edgeCount.*must equal/],
    ["duplicate pair", (v) => { v.aa.push([...v.aa[0]]); v.edgeCount += 1; }, /aa\[11\].*duplicates/],
  ];
  for (const [name, mutate, message] of cases) {
    const graph = copy(original);
    mutate(graph);
    assert.throws(() => parseCompactGraph(graph, "graph fixture"), message, name);
  }
});

test("v2 rejects changed semantics, missing support, mismatched counts, and wrong roles", () => {
  const original = fixture("graph.compact.json");
  const cases: [string, (value: any) => void, RegExp][] = [
    ["missing digest", (v) => { delete v.dataset.sha256; }, /dataset.sha256.*SHA-256/],
    ["semantic drift", (v) => { v.semantics.pairWeight = "item-cosine"; }, /semantics.pairWeight.*unsupported/],
    ["missing support", (v) => { v.aa[0].pop(); }, /aa\[0\].*exactly 4/],
    ["false truncation", (v) => { v.truncation.selectedPairs -= 1; }, /truncation.selectedPairs.*reconcile/],
    ["wrong role", (v) => { v.role = "visualization"; }, /role.*recommendation/],
  ];
  for (const [name, mutate, message] of cases) {
    const graph = copy(original);
    mutate(graph);
    assert.throws(() => parseCompactGraph(graph, "v2 recommendation", "recommendation"), message, name);
  }
  const explorer = fixture("graph-explorer.compact.json");
  assert.throws(() => parseCompactGraph(explorer, "v2 explorer", "recommendation"), /role.*recommendation/);
  assert.throws(() => parseCompactGraph(original, "v2 recommendation", "visualization"), /role.*visualization/);
});

test("rejects non-finite graph weights and duplicate or missing legacy graph identities", () => {
  for (const edge of ["ua", "aa"] as const) {
    const graph = fixture("graph.compact.json");
    graph[edge][0][2] = Number.NaN;
    assert.throws(() => parseCompactGraph(graph, "graph fixture"), /must be a finite number/);
  }
  const original = legacyGraph();
  const cases: [string, (value: any) => void, RegExp][] = [
    ["duplicate node", (v) => { v.nodes[1].id = v.nodes[0].id; }, /nodes\[1\].id.*duplicates/],
    ["duplicate edge", (v) => { v.edges[1].id = v.edges[0].id; }, /edges\[1\].id.*duplicates/],
    ["missing node", (v) => { v.edges[0].target = "anime:999999"; }, /edges\[0\].*missing node/],
    ["weight", (v) => { v.edges[0].weight = Number.POSITIVE_INFINITY; }, /weight.*finite/],
    ["version", (v) => { v.format = "graph-compact-v2"; }, /format.*unsupported/],
  ];
  for (const [name, mutate, message] of cases) {
    const graph = copy(original);
    mutate(graph);
    assert.throws(() => parseLegacyGraph(graph, "legacy fixture"), message, name);
  }
});

test("rejects invalid catalog IDs, metadata, and version", () => {
  const original = fixture("catalog.json");
  const cases: [string, (value: any) => void, RegExp][] = [
    ["version", (v) => { v.format = "demo-catalog-v2"; }, /format.*unsupported/],
    ["duplicate ID", (v) => { v.anime[1].animeId = v.anime[0].animeId; }, /anime\[1\].animeId.*duplicates/],
    ["score", (v) => { v.anime[0].score = Number.NaN; }, /anime\[0\].score.*finite/],
    ["genres", (v) => { v.anime[0].genres = "action"; }, /genres.*array/],
  ];
  for (const [name, mutate, message] of cases) {
    const catalog = copy(original);
    mutate(catalog);
    assert.throws(() => parseDemoCatalog(catalog, "catalog fixture"), message, name);
  }
});

test("rejects mismatched model arrays, factor dimensions, duplicate IDs, and non-finite values", () => {
  const original = fixture("model-mf-web.compact.json");
  const cases: [string, (value: any) => void, RegExp][] = [
    ["version", (v) => { v.format = "model-mf-compact-v2"; }, /format.*unsupported/],
    ["array length", (v) => { v.biases.pop(); }, /biases.*length must equal/],
    ["dimension", (v) => { v.embeddings[0].pop(); }, /embeddings\[0\].*dimension/],
    ["duplicate ID", (v) => { v.animeIds[1] = v.animeIds[0]; }, /animeIds\[1\].*duplicates/],
    ["bias", (v) => { v.biases[0] = Number.NaN; }, /biases\[0\].*finite/],
    ["embedding", (v) => { v.embeddings[0][0] = Number.POSITIVE_INFINITY; }, /embeddings\[0\]\[0\].*finite/],
    ["global mean", (v) => { v.globalMean = Number.NaN; }, /globalMean.*finite/],
  ];
  for (const [name, mutate, message] of cases) {
    const model = copy(original);
    mutate(model);
    assert.throws(() => parseCompactModel(model, "model fixture"), message, name);
  }
  const legacy = legacyModel();
  legacy.anime[0].embedding.pop();
  assert.throws(() => parseLegacyModel(legacy, "legacy model"), /embedding.*dimension/);
});

test("validation errors identify the artifact and field without echoing content", () => {
  const model = fixture("model-mf-web.compact.json");
  model.embeddings[0].pop();
  assert.throws(
    () => parseCompactModel(model, "model-mf-web.compact.json"),
    (error: unknown) => error instanceof ArtifactValidationError &&
      error.message.includes("model-mf-web.compact.json: embeddings[0]") &&
      error.message.includes("Rebuild or replace this artifact"),
  );
});
