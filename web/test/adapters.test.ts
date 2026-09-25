import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { createArtifactLoader } from "../src/artifact-loader.ts";
import { parseCompactGraph } from "../src/artifacts.ts";
import { createPersistenceAdapter } from "../src/persistence.ts";
import { createProviderAdapter } from "../src/providers.ts";
import { buildRecommendationIndexFromCompact } from "../src/recommendations.ts";
import { createSeededRandom } from "../src/runtime.ts";
import type { RuntimePorts, StoragePort } from "../src/runtime.ts";

const fixtureRoot = new URL("../public/demo-data/", import.meta.url);
const fixture = (name: string): unknown => JSON.parse(readFileSync(new URL(name, fixtureRoot), "utf8"));
const index = buildRecommendationIndexFromCompact(parseCompactGraph(fixture("graph.compact.json"), "synthetic graph"));
const jsonResponse = (value: unknown, status = 200): Response =>
  new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });

function fakeRuntime(
  responder: (url: string, init?: RequestInit) => Promise<Response> | Response,
  seed = 19,
) {
  const values = new Map<string, string>();
  const requests: Array<{ url: string; init?: RequestInit }> = [];
  const sleeps: number[] = [];
  const storage: StoragePort = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
    removeItem: (key) => { values.delete(key); },
  };
  const runtime: RuntimePorts = {
    fetch: async (input, init) => {
      const url = String(input);
      requests.push({ url, init });
      return await responder(url, init);
    },
    storage,
    now: () => new Date("2026-09-24T12:34:56.000Z"),
    monotonicNow: () => 1234,
    random: createSeededRandom(seed),
    sleep: async (ms) => { sleeps.push(ms); },
    schedule: (callback) => { callback(); },
    frame: (callback) => { callback(1234); },
  };
  return { runtime, values, requests, sleeps };
}

test("runtime randomness is repeatable from an injected synthetic seed", () => {
  const first = createSeededRandom(42);
  const second = createSeededRandom(42);
  const other = createSeededRandom(43);
  const sequence = [first(), first(), first()];
  assert.deepEqual(sequence, [second(), second(), second()]);
  assert.notEqual(sequence[0], other());
  assert.ok(sequence.every((value) => value >= 0 && value < 1));
});

test("persistence uses injected storage, prefix, and clock while preserving legacy keys", () => {
  const fake = fakeRuntime(() => { throw new Error("network must not be used"); });
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  const state = {
    version: 3,
    mode: "hybrid" as const,
    selected: [{ nodeId: "anime:101", weight: 1.7 }],
    modelBlendWeight: 0.35,
    includeCandidates: ["anime:102"],
    excludeCandidates: ["anime:105"],
  };
  persistence.persistRecommendationState(state);
  assert.equal(fake.values.has("wasiw.demo.recommendationState.v1"), true);
  assert.deepEqual(persistence.loadRecommendationState(index), {
    mode: "hybrid", selected: state.selected, modelBlendWeight: 0.35,
    includeCandidates: ["anime:102"], excludeCandidates: ["anime:105"],
  });

  fake.values.set("wasiw.demo.recommendationProfiles.v1", JSON.stringify([
    { name: " Fixture Profile ", state },
  ]));
  const profiles = persistence.loadRecommendationProfiles(index);
  assert.equal(profiles.get("Fixture Profile")?.updatedAt, "2026-09-24T12:34:56.000Z");
  persistence.persistRecommendationProfiles(profiles);
  assert.deepEqual(JSON.parse(fake.values.get("wasiw.demo.recommendationProfiles.v1") ?? "null")
    .map((item: { name: string }) => item.name), ["Fixture Profile"]);

  assert.equal(persistence.loadThemeModePreference(() => true), "light");
  persistence.persistThemeModePreference("dark");
  assert.equal(persistence.loadThemeModePreference(() => true), "dark");
  persistence.persistContrastModePreference("high");
  assert.equal(persistence.loadContrastModePreference(), "high");
  persistence.persistHelpTipsDismissed(true);
  assert.equal(persistence.loadHelpTipsDismissed(), true);
  persistence.persistCommandPinnedIds(["network"]);
  persistence.persistCommandHistoryIds(["recommendations"]);
  assert.deepEqual(persistence.loadCommandPinnedIds(), ["network"]);
  assert.deepEqual(persistence.loadCommandHistoryIds(), ["recommendations"]);
  assert.deepEqual(fake.requests, []);
});

test("denied browser storage has deterministic fallbacks without DOM setup", () => {
  const fake = fakeRuntime(() => { throw new Error("network must not be used"); });
  fake.runtime.storage = {
    getItem: () => { throw new Error("storage denied"); },
    setItem: () => { throw new Error("storage denied"); },
    removeItem: () => { throw new Error("storage denied"); },
  };
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  const oldWarn = console.warn;
  console.warn = () => {};
  try {
    assert.deepEqual(persistence.loadRecommendationState(index), {
      mode: "graph", selected: [], modelBlendWeight: 0.5,
      includeCandidates: [], excludeCandidates: [],
    });
    assert.equal(persistence.loadThemeModePreference(() => true), "light");
    assert.deepEqual(persistence.loadRecommendationProfiles(index), new Map());
    assert.doesNotThrow(() => persistence.persistRecommendationState({
      version: 3, mode: "graph", selected: [],
    }));
  } finally {
    console.warn = oldWarn;
  }
});

test("provider imports and metadata use only mocked direct transport and seeded retry timing", async () => {
  const counts = new Map<string, number>();
  const fake = fakeRuntime((url, init) => {
    const parsed = new URL(url);
    const key = `${parsed.hostname}${parsed.pathname}`;
    const attempt = (counts.get(key) ?? 0) + 1;
    counts.set(key, attempt);
    if (parsed.hostname === "myanimelist.net") {
      return attempt === 1 ? jsonResponse({}, 429)
        : jsonResponse([{ anime_id: 102, score: 8 }]);
    }
    if (parsed.hostname === "graphql.anilist.co") {
      assert.equal(init?.method, "POST");
      assert.equal(JSON.parse(String(init?.body)).variables.userName, "fixture-user");
      return jsonResponse({ data: { MediaListCollection: { lists: [
        { entries: [{ score: 9, media: { idMal: 101 } }] },
      ] } } });
    }
    if (parsed.pathname === "/v4/anime/102/full") {
      return attempt === 1 ? jsonResponse({}, 503) : jsonResponse({ data: {
        year: 2020, score: 7.8, genres: [{ name: "Fantasy" }], studios: [],
        synopsis: "Invented synopsis", images: { jpg: { image_url: "https://example.invalid/cover.png" } },
      } });
    }
    if (parsed.pathname === "/v4/anime/404/full") return jsonResponse({}, 404);
    if (parsed.pathname === "/v4/seasons/now") return jsonResponse({ data: [
      { mal_id: 102, title: "Moonlit Workshop", year: 2020, score: 7.8, season: "spring" },
    ] });
    throw new Error(`Unexpected mocked route: ${parsed.pathname}`);
  });
  const provider = createProviderAdapter(fake.runtime);
  const mal = await provider.fetchMalUsernameImport("fixture-user", index);
  assert.deepEqual(mal.entries.map((entry) => [entry.anime.animeId, entry.weight]), [[102, 1.6]]);
  const anilist = await provider.fetchAniListUsernameImport("fixture-user", index);
  assert.deepEqual(anilist.entries.map((entry) => [entry.anime.animeId, entry.weight]), [[101, 1.8]]);
  const metadata = await provider.fetchAnimeMetadataFromJikan(102);
  assert.deepEqual([metadata?.year, metadata?.genres, metadata?.score], [2020, ["Fantasy"], 7.8]);
  assert.equal(await provider.fetchAnimeMetadataFromJikan(404), null);
  const seasonal = await provider.fetchSeasonalAnime(12);
  assert.deepEqual(seasonal.map((item) => item.title), ["Moonlit Workshop"]);
  const random = createSeededRandom(19);
  assert.deepEqual(fake.sleeps, [1000 + Math.floor(random() * 200), 800 + Math.floor(random() * 220)]);
  assert.deepEqual(new Set(fake.requests.map((request) => new URL(request.url).hostname)),
    new Set(["myanimelist.net", "graphql.anilist.co", "api.jikan.moe"]));
  assert.equal(fake.requests.some((request) => request.url.includes("r.jina.ai")), false);
});

test("a provider rejection is testable without the page and never uses a proxy", async () => {
  const fake = fakeRuntime(() => jsonResponse({}, 403));
  const provider = createProviderAdapter(fake.runtime);
  await assert.rejects(provider.fetchMalUsernameImport("fixture-user", index),
    /Direct MAL import failed.*No proxy was contacted/);
  assert.equal(fake.requests.length, 1);
  assert.equal(new URL(fake.requests[0].url).hostname, "myanimelist.net");
  assert.deepEqual(fake.sleeps, []);
});

test("artifact loader validates synthetic files through injected local transport", async () => {
  const data = new Map<string, unknown>([
    ["./demo-data/graph.compact.json", fixture("graph.compact.json")],
    ["./demo-data/catalog.json", fixture("catalog.json")],
    ["./demo-data/model-mf-web.compact.json", fixture("model-mf-web.compact.json")],
  ]);
  const fake = fakeRuntime((url) => {
    if (!data.has(url)) throw new Error(`Unexpected local artifact: ${url}`);
    return jsonResponse(data.get(url));
  });
  const loader = createArtifactLoader(fake.runtime, true);
  const graph = await loader.fetchGraph();
  assert.equal(graph.animeCount, 8);
  assert.equal(await loader.fetchExplorerGraph(graph), graph);
  assert.equal((await loader.fetchModelRecommendationIndex())?.factors, 2);
  assert.equal((await loader.fetchDemoCatalog()).length, 8);
  assert.deepEqual(fake.requests.map((request) => request.url), [
    "./demo-data/graph.compact.json",
    "./demo-data/model-mf-web.compact.json",
    "./demo-data/catalog.json",
  ]);
});

test("artifact transport keeps the missing compact to legacy fallback deliberate", async () => {
  const legacy = {
    generatedAt: "2026-09-24T00:00:00.000Z", userCount: 0, animeCount: 1,
    nodeCount: 1, edgeCount: 0,
    nodes: [{ id: "anime:101", label: "Copper Comet", nodeType: "anime" }],
    edges: [],
  };
  const fake = fakeRuntime((url) => url === "./data/graph.json"
    ? jsonResponse(legacy) : new Response("", { status: 404 }));
  const graph = await createArtifactLoader(fake.runtime, false).fetchGraph();
  assert.equal(graph.animeCount, 1);
  assert.deepEqual(fake.requests.map((request) => request.url), [
    "./data/graph.compact.json.gz", "./data/graph.compact.json",
    "./data/graph.json.gz", "./data/graph.json",
  ]);
});
