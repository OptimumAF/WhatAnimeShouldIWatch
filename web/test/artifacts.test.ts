import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  ArtifactValidationError,
  parseActiveReleaseBundle,
  parseBrowserReleaseManifest,
  parseCompactGraph,
  parseCompactModel,
  catalogMetadataCoverage,
  parseCatalogMetadataSnapshot,
  parseDemoCatalog,
  parseLegacyGraph,
  parseLegacyModel,
  parseReleaseIdentityCatalog,
  parseReleaseManifest,
} from "../src/artifacts.ts";
import { projectCatalogMetadata } from "../src/catalog-metadata.ts";

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
  const catalog = parseDemoCatalog(fixture("catalog.json"), "catalog fixture");
  assert.equal(catalog.length, 8);
  assert.deepEqual(catalog.find((item) => item.animeId === 101)?.aliases, ["Galaxy Route"]);
  assert.equal(catalog.find((item) => item.animeId === 102)?.mediaFormat, "Movie");
  assert.deepEqual(catalog.find((item) => item.animeId === 105)?.relations,
    [{ kind: "side-story", animeId: 108, title: "Quiet Satellite" }]);
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
    ["format", (v) => { v.format = "graph-compact-v4"; }, /format.*unsupported/],
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

test("v3 retains pair statistics but rejects public user rows and hidden fields", () => {
  const graph = fixture("graph.compact.json");
  graph.format = "graph-compact-v3";
  graph.projection = { policy: "omit-user-anime-v1" };
  graph.userIds = [];
  graph.ua = [];
  graph.userCount = 0;
  graph.nodeCount = graph.anime.length;
  graph.edgeCount = graph.aa.length;
  assert.equal(parseCompactGraph(graph, "aggregate", "recommendation"), graph);
  assert.equal(graph.truncation.selectedRatings, 18);
  assert.throws(() => parseCompactGraph({ ...graph, userIds: ["invented"] }, "aggregate"),
    /userIds\/ua must be empty/);
  assert.throws(() => parseCompactGraph({ ...graph, hiddenHistory: [] }, "aggregate"),
    /root.hiddenHistory is unsupported/);
  const manifest = fixture("release-manifest.json");
  manifest.neighborhood.format = "graph-compact-v3";
  manifest.explorer.format = "graph-compact-v3";
  assert.equal(parseReleaseManifest(manifest, "aggregate manifest"), manifest);
  manifest.explorer.format = "graph-compact-v2";
  assert.throws(() => parseReleaseManifest(manifest, "mixed manifest"),
    /explorer.format must be graph-compact-v3/);
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
    ["aliases", (v) => { v.anime[0].aliases = "wrong"; }, /aliases.*array/],
    ["alias title", (v) => { v.anime[0].aliases = [""]; }, /aliases\[0\].*nonempty/],
    ["media format", (v) => { v.anime[0].mediaFormat = 5; }, /mediaFormat.*nonempty string/],
    ["relation kind", (v) => { v.anime[4].relations[0].kind = "unknown"; }, /relations\[0\].kind.*supported/],
    ["self relation", (v) => { v.anime[4].relations[0].animeId = 105; }, /relations\[0\].animeId.*itself/],
    ["relation title", (v) => { v.anime[4].relations[0].title = ""; }, /relations\[0\].title.*nonempty/],
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

test("optional numeric source digest is validated in compact and legacy models", () => {
  const compact = fixture("model-mf-web.compact.json");
  compact.sourceModelSha256 = "a".repeat(64);
  assert.equal(parseCompactModel(compact, "compact model").sourceModelSha256,
    "a".repeat(64));
  compact.sourceModelSha256 = "A".repeat(64);
  assert.throws(() => parseCompactModel(compact, "compact model"),
    /sourceModelSha256.*lowercase SHA-256/);
  const legacy = legacyModel();
  legacy.sourceModelSha256 = "invalid";
  assert.throws(() => parseLegacyModel(legacy, "legacy model"),
    /sourceModelSha256.*lowercase SHA-256/);
});

test("compact item model rejects undeclared user factors at the browser boundary", () => {
  const model = fixture("model-mf-web.compact.json");
  model.userFactors = [[1, 0]];
  assert.throws(() => parseCompactModel(model, "model-mf-web.compact.json"),
    /model-mf-web.compact.json: root.userFactors is unsupported/);
  delete model.userFactors;
  model.trainUserItems = { invented: [101] };
  assert.throws(() => parseCompactModel(model, "model-mf-web.compact.json"),
    /model-mf-web.compact.json: root.trainUserItems is unsupported/);
});

test("release identity and manifest contracts reject stale fields and mutable tags", () => {
  const catalog = fixture("catalog.identity.json");
  const manifest = fixture("release-manifest.json");
  assert.equal(parseReleaseIdentityCatalog(catalog, "catalog.identity.json"), catalog);
  assert.equal(parseReleaseManifest(manifest, "release-manifest.json"), manifest);
  const wrongOrder = copy(catalog);
  wrongOrder.anime.reverse();
  assert.throws(() => parseReleaseIdentityCatalog(wrongOrder, "catalog.identity.json"),
    /anime\[1\]\[0\].*sorted/);
  const wrongDigest = copy(catalog);
  wrongDigest.datasetSha256 = "invalid";
  assert.throws(() => parseReleaseIdentityCatalog(wrongDigest, "catalog.identity.json"),
    /datasetSha256.*SHA-256/);
  const mutable = copy(manifest);
  mutable.tag = "data-latest";
  assert.throws(() => parseReleaseManifest(mutable, "release-manifest.json"), /tag.*versioned/);
  const staleLink = copy(manifest);
  staleLink.explorer.sourceGraphId = "a".repeat(64);
  assert.throws(() => parseReleaseManifest(staleLink, "release-manifest.json"),
    /explorer.sourceGraphId.*neighborhood.graphId/);
  const staleModel = copy(manifest);
  staleModel.model.datasetSha256 = "a".repeat(64);
  assert.throws(() => parseReleaseManifest(staleModel, "release-manifest.json"),
    /model.datasetSha256.*dataset.sha256/);
  const unknown = copy(manifest);
  unknown.extra = true;
  assert.throws(() => parseReleaseManifest(unknown, "release-manifest.json"),
    /root.extra.*unsupported/);
  const malformedModel = fixture("model-mf-web.compact.json");
  malformedModel.datasetSha256 = "bad";
  assert.throws(() => parseCompactModel(malformedModel, "model-mf-web.compact.json"),
    /datasetSha256.*SHA-256/);
  delete malformedModel.datasetSha256;
  assert.equal(parseCompactModel(malformedModel, "older compact model"), malformedModel);
  const active = { format: "active-release-bundle-v1", tag: manifest.tag,
    bundleId: manifest.bundleId, manifestSha256: "b".repeat(64) };
  assert.equal(parseActiveReleaseBundle(active, "active.json"), active);
  assert.throws(() => parseActiveReleaseBundle({ ...active, tag: "data-latest" }, "active.json"),
    /tag.*versioned/);
  assert.throws(() => parseActiveReleaseBundle({ ...active, manifestSha256: "wrong" }, "active.json"),
    /manifestSha256.*SHA-256/);
  assert.throws(() => parseActiveReleaseBundle({ ...active, extra: true }, "active.json"),
    /root.extra.*unsupported/);
});

test("metadata candidate keeps unknown fields explicit and reports bounded coverage", () => {
  const snapshot: any = {
    format: "anime-metadata-catalog-v1",
    source: { name: "invented-fixture", snapshotAt: "2026-09-24T00:00:00.000Z",
      snapshotSha256: "a".repeat(64) },
    anime: [
      { animeId: 101, sourceItemId: "invented:101", title: "Copper Comet",
        aliases: ["Galaxy Route"], genres: ["Adventure"], year: 2021, mediaFormat: "TV",
        episodeCount: 12, runtimeMinutes: 24, contentClassification: null,
        communityScore: null, relations: [{ kind: "sequel", animeId: 102, title: "Moonlit Workshop" }] },
      { animeId: 102, sourceItemId: "invented:102", title: "Moonlit Workshop",
        aliases: [], genres: null, year: null, mediaFormat: "Movie",
        episodeCount: null, runtimeMinutes: 95,
        contentClassification: { jurisdiction: "Fixtureland", system: "Invented board", value: "All" },
        communityScore: null, relations: null },
    ],
  };
  assert.equal(parseCatalogMetadataSnapshot(snapshot, "metadata candidate"), snapshot);
  assert.deepEqual(projectCatalogMetadata(snapshot.anime[0]), {
    animeId: 101, aliases: ["Copper Comet", "Galaxy Route"], mediaFormat: "TV", year: 2021,
    score: null, genres: ["Adventure"], studios: [], synopsis: "", imageUrl: "",
    season: null, relations: [{ kind: "sequel", animeId: 102, title: "Moonlit Workshop" }],
    episodeCount: 12, runtimeMinutes: 24, contentClassification: null,
  });
  assert.deepEqual(projectCatalogMetadata(snapshot.anime[1]).genres, []);
  const coverage = catalogMetadataCoverage(snapshot, [101, 102, 103]);
  assert.equal(coverage.total, 3);
  assert.equal(coverage.missingItems, 1);
  assert.equal(coverage.known.aliases, 2);
  assert.equal(coverage.usable.aliases, 1);
  assert.equal(coverage.known.genres, 1);
  assert.equal(coverage.known.contentClassification, 1);
  assert.equal(coverage.known.communityScore, 0);
  assert.equal(coverage.usable.relations, 1);
  assert.equal(coverage.directedRelationItems, 1);
  assert.equal(coverage.directedTargetsOutsideUniverse, 0);
  assert.equal(catalogMetadataCoverage(snapshot, [101]).directedTargetsOutsideUniverse, 1);
  assert.throws(() => catalogMetadataCoverage(snapshot, [101, 101]), /unique positive anime IDs/);

  const cases: [string, (value: any) => void, RegExp][] = [
    ["hidden user rows", (v) => { v.userRows = []; }, /root.userRows.*unsupported/],
    ["image field", (v) => { v.anime[0].imageUrl = "https://example.test/cover"; },
      /anime\[0\].imageUrl.*unsupported/],
    ["missing unknown", (v) => { delete v.anime[1].genres; }, /anime\[1\].genres.*required/],
    ["duplicate source identity", (v) => { v.anime[1].sourceItemId = "invented:101"; },
      /anime\[1\].sourceItemId.*duplicates/],
    ["ambiguous aliases", (v) => { v.anime[0].aliases.push(" galaxy route "); },
      /anime\[0\].aliases\[1\].*duplicates/],
    ["unsupported format", (v) => { v.anime[0].mediaFormat = "Unknown"; },
      /anime\[0\].mediaFormat.*supported/],
    ["invalid runtime", (v) => { v.anime[0].runtimeMinutes = -1; },
      /anime\[0\].runtimeMinutes.*greater than 0/],
    ["classification structure", (v) => { v.anime[1].contentClassification.extra = "hidden"; },
      /contentClassification.extra.*unsupported/],
    ["relation reference", (v) => { v.anime[0].relations[0].animeId = 101; },
      /relations\[0\].animeId.*itself/],
    ["source digest", (v) => { v.source.snapshotSha256 = "bad"; },
      /source.snapshotSha256.*SHA-256/],
  ];
  for (const [name, mutate, pattern] of cases) {
    const changed = copy(snapshot);
    mutate(changed);
    assert.throws(() => parseCatalogMetadataSnapshot(changed, "metadata candidate"), pattern, name);
  }
});

test("browser-only metadata manifest binds a separate asset without weakening v1", () => {
  const base = fixture("release-manifest.json");
  const candidate: any = { ...base, format: "release-manifest-v2", metadata: {
    path: "catalog.metadata.json", format: "anime-metadata-catalog-v1",
    sha256: "a".repeat(64), bytes: 500, animeCount: 2,
    itemMapSha256: base.catalog.itemMapSha256, sourceSnapshotSha256: "b".repeat(64),
  } };
  assert.equal(parseBrowserReleaseManifest(base, "release-manifest.json"), base);
  assert.equal(parseBrowserReleaseManifest(candidate, "release-manifest.json"), candidate);
  assert.throws(() => parseReleaseManifest(candidate, "release-manifest.json"), /root.metadata.*unsupported/);
  const wrongMap = copy(candidate);
  wrongMap.metadata.itemMapSha256 = "c".repeat(64);
  assert.throws(() => parseBrowserReleaseManifest(wrongMap, "release-manifest.json"),
    /metadata.itemMapSha256.*catalog.itemMapSha256/);
  const extra = copy(candidate);
  extra.metadata.userCount = 1;
  assert.throws(() => parseBrowserReleaseManifest(extra, "release-manifest.json"),
    /metadata.userCount.*unsupported/);
  const tooMany = copy(candidate);
  tooMany.metadata.animeCount = base.catalog.animeCount + 1;
  assert.throws(() => parseBrowserReleaseManifest(tooMany, "release-manifest.json"),
    /metadata.animeCount.*catalog.animeCount/);
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
