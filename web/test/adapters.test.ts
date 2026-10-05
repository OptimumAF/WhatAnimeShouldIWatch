import assert from "node:assert/strict";
import crypto from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { gzipSync } from "node:zlib";
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
const bytesResponse = (bytes: Uint8Array, headers: Record<string, string> = {}): Response =>
  new Response(Uint8Array.from(bytes).buffer as ArrayBuffer, { headers });
const gzipResponse = (value: unknown): Response => bytesResponse(
  gzipSync(Buffer.from(JSON.stringify(value))), { "Content-Type": "application/gzip" });

function fakeRuntime(
  responder: (url: string, init?: RequestInit) => Promise<Response> | Response,
  seed = 19,
) {
  const values = new Map<string, string>();
  const requests: Array<{ url: string; init?: RequestInit; at: number }> = [];
  const sleeps: number[] = [];
  let elapsedMs = 0;
  const storage: StoragePort = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
    removeItem: (key) => { values.delete(key); },
  };
  const runtime: RuntimePorts = {
    fetch: async (input, init) => {
      const url = String(input);
      requests.push({ url, init, at: elapsedMs });
      return await responder(url, init);
    },
    storage,
    now: () => new Date(Date.parse("2026-09-24T12:34:56.000Z") + elapsedMs),
    monotonicNow: () => 1234 + elapsedMs,
    random: createSeededRandom(seed),
    sleep: async (ms) => { sleeps.push(ms); elapsedMs += ms; },
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

test("project Pages base pins asset URLs even after direct nested navigation", async () => {
  const prefix = "/WhatAnimeShouldIWatch/";
  const fake = fakeRuntime((url) => {
    if (url === `${prefix}data/active.json` ||
        url === `${prefix}data/graph.compact.json.gz`) return jsonResponse({}, 404);
    if (url === `${prefix}data/graph.compact.json`) {
      return jsonResponse(fixture("graph.compact.json"));
    }
    throw new Error(`Unexpected asset URL: ${url}`);
  });
  const graph = await createArtifactLoader(fake.runtime, false, prefix).fetchGraph();
  assert.equal(graph.format, "graph-compact-v2");
  assert.deepEqual(fake.requests.map((request) => request.url), [
    `${prefix}data/active.json`, `${prefix}data/graph.compact.json.gz`,
    `${prefix}data/graph.compact.json`,
  ]);
  assert.throws(() => createArtifactLoader(fake.runtime, false, "../"),
    /Artifact base path/);
});

test("persistence uses injected storage, prefix, and clock while migrating legacy keys", () => {
  const fake = fakeRuntime(() => { throw new Error("network must not be used"); });
  const persistence = createPersistenceAdapter(fake.runtime, "wasiw.demo");
  const state = {
    version: 5,
    mode: "hybrid" as const,
    preferences: [{ nodeId: "anime:101", sentiment: "liked" as const,
      importance: 1.7, confidence: 1, source: "manual" as const }],
    modelBlendWeight: 0.35,
    includeCandidates: ["anime:102"],
    excludeCandidates: ["anime:105"],
  };
  persistence.persistRecommendationState(state);
  assert.equal(fake.values.has("wasiw.demo.recommendationState.v5"), true);
  assert.deepEqual(persistence.loadRecommendationState(), {
    mode: "hybrid", preferences: state.preferences, modelBlendWeight: 0.35,
    allowRelatedTitles: false,
    includeCandidates: ["anime:102"], excludeCandidates: ["anime:105"], history: [], watchlist: [],
  });

  fake.values.set("wasiw.demo.recommendationProfiles.v1", JSON.stringify([
    { name: " Fixture Profile ", state: { version: 3, mode: "hybrid",
      selected: [{ nodeId: "anime:101", weight: 1.7 }] } },
  ]));
  const profiles = persistence.loadRecommendationProfiles();
  assert.equal(profiles.get("Fixture Profile")?.updatedAt, "2026-09-24T12:34:56.000Z");
  persistence.persistRecommendationProfiles(profiles);
  assert.deepEqual(JSON.parse(fake.values.get("wasiw.demo.recommendationProfiles.v5") ?? "null")
    .profiles.map((item: { name: string }) => item.name), ["Fixture Profile"]);

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
    assert.deepEqual(persistence.loadRecommendationState(), {
      mode: "graph", preferences: [], modelBlendWeight: 0.5,
      allowRelatedTitles: false,
      includeCandidates: [], excludeCandidates: [], history: [], watchlist: [],
    });
    assert.equal(persistence.loadThemeModePreference(() => true), "light");
    assert.deepEqual(persistence.loadRecommendationProfiles(), new Map());
    assert.doesNotThrow(() => persistence.persistRecommendationState({
      version: 5, mode: "graph", preferences: [],
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
      return jsonResponse({ data: { User: { mediaListOptions: { scoreFormat: "POINT_10_DECIMAL" } },
        MediaListCollection: { lists: [
        { entries: [{ score: 9, status: "COMPLETED", progress: 12,
          media: { id: 501, idMal: 101, title: { romaji: "Copper Comet" } } }] },
      ] } } });
    }
    if (parsed.pathname === "/v4/anime/102/full") {
      return attempt === 1 ? jsonResponse({}, 503) : jsonResponse({ data: {
        year: 2020, score: 7.8, genres: [{ name: "Fantasy" }], studios: [], type: "Movie",
        titles: [{ type: "Default", title: "Moonlit Workshop" }, { type: "English", title: "Moonlit Studio" }],
        title_synonyms: ["The Moon Workshop"],
        synopsis: "Invented synopsis", images: { jpg: { image_url: "https://example.invalid/cover.png" } },
        relations: [
          { relation: "Prequel", entry: [{ mal_id: 101, type: "anime", name: "Copper Comet" }] },
          { relation: "Adaptation", entry: [{ mal_id: 900, type: "manga", name: "Invented manga" }] },
        ],
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
  assert.deepEqual(mal.entries.map((entry) => [entry.anime.animeId, entry.preference.sentiment,
    entry.preference.confidence]), [[102, "liked", 0.5]]);
  assert.deepEqual([mal.history[0].provider, mal.history[0].sourceId, mal.history[0].scoreScale],
    ["mal", "102", "mal-10"]);
  const anilist = await provider.fetchAniListUsernameImport("fixture-user", index);
  assert.deepEqual(anilist.entries.map((entry) => [entry.anime.animeId, entry.preference.sentiment,
    entry.preference.confidence]), [[101, "liked", 0.75]]);
  assert.deepEqual([anilist.history[0].sourceId, anilist.history[0].status,
    anilist.history[0].progressEpisodes, anilist.history[0].scoreScale],
  ["501", "completed", 12, "POINT_10_DECIMAL"]);
  const metadata = await provider.fetchAnimeMetadataFromJikan(102);
  assert.equal(metadata.state, "ready");
  if (metadata.state === "ready") {
    assert.deepEqual([metadata.metadata.year, metadata.metadata.genres, metadata.metadata.score], [2020, ["Fantasy"], 7.8]);
    assert.deepEqual(metadata.metadata.relations,
      [{ kind: "prequel", animeId: 101, title: "Copper Comet" }]);
    assert.deepEqual(metadata.metadata.aliases,
      ["Moonlit Workshop", "Moonlit Studio", "The Moon Workshop"]);
    assert.equal(metadata.metadata.mediaFormat, "Movie");
  }
  assert.deepEqual(await provider.fetchAnimeMetadataFromJikan(404), { state: "unavailable" });
  const seasonal = await provider.fetchSeasonalAnime(12);
  assert.deepEqual(seasonal.map((item) => item.title), ["Moonlit Workshop"]);
  const random = createSeededRandom(19);
  assert.deepEqual(fake.sleeps, [1000 + Math.floor(random() * 350), 800 + Math.floor(random() * 250), 200]);
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

test("malformed optional Jikan relationships remain unknown", async () => {
  const fake = fakeRuntime(() => jsonResponse({ data: {
    year: 2020, relations: [{ relation: "Prequel", entry: "broken" }],
  } }));
  const result = await createProviderAdapter(fake.runtime).fetchAnimeMetadataFromJikan(102);
  assert.equal(result.state, "ready");
  if (result.state === "ready") assert.equal(result.metadata.relations, null);
  assert.equal(fake.requests.length, 1);
});

test("mocked username imports keep status, progress, unscored and unmapped provider identities", async () => {
  const fake = fakeRuntime((url) => new URL(url).hostname === "myanimelist.net"
    ? jsonResponse([
      { anime_id: 101, anime_title: "Copper Comet", score: 8, status: 2, num_watched_episodes: 12 },
      { anime_id: 102, anime_title: "Moonlit Workshop", score: 0, status: 1, num_watched_episodes: 3 },
      { anime_id: 99999, anime_title: "Unknown Fixture", score: 0, status: 6, num_watched_episodes: 0 },
      { anime_id: 101, anime_title: "Copper Comet", score: 9, status: 2, num_watched_episodes: 12 },
    ]) : jsonResponse({ data: { User: { mediaListOptions: { scoreFormat: "POINT_10_DECIMAL" } },
      MediaListCollection: { lists: [
      { entries: [
        { score: 0, status: "PLANNING", progress: 0,
          media: { id: 777, idMal: null, title: { romaji: "Invented Unmapped" } } },
        { score: 8.5, status: "CURRENT", progress: 4,
          media: { id: 778, idMal: 102, title: { romaji: "Moonlit Workshop" } } },
      ] },
    ] } } }));
  const adapter = createProviderAdapter(fake.runtime);
  const mal = await adapter.fetchMalUsernameImport("fixture-user", index);
  assert.equal(mal.duplicateCount, 1);
  assert.deepEqual(mal.history.map((entry) => [entry.sourceId, entry.status,
    entry.progressEpisodes, entry.score]), [
    ["101", "completed", 12, 9], ["102", "watching", 3, null],
    ["99999", "plan_to_watch", 0, null],
  ]);
  assert.equal(mal.ratedCount, 1);
  const anilist = await adapter.fetchAniListUsernameImport("fixture-user", index);
  assert.deepEqual(anilist.history.map((entry) => [entry.sourceId, entry.animeId,
    entry.status, entry.progressEpisodes, entry.score, entry.scoreScale]), [
    ["777", null, "plan_to_watch", 0, null, "POINT_10_DECIMAL"],
    ["778", 102, "watching", 4, 8.5, "POINT_10_DECIMAL"],
  ]);
  assert.deepEqual(fake.requests.map((request) => new URL(request.url).hostname),
    ["myanimelist.net", "graphql.anilist.co"]);
});

test("AniList import retains native score formats before preference conversion", async () => {
  const cases = [
    { scale: "POINT_100", score: 80, sentiment: "liked", confidence: 0.5 },
    { scale: "POINT_10_DECIMAL", score: 8.5, sentiment: "liked", confidence: 0.63 },
    { scale: "POINT_10", score: 8, sentiment: "liked", confidence: 0.5 },
    { scale: "POINT_5", score: 4, sentiment: "liked", confidence: 0.5 },
    { scale: "POINT_3", score: 3, sentiment: "liked", confidence: 1 },
    { scale: "POINT_3", score: 1, sentiment: "disliked", confidence: 0.75 },
  ];
  for (const { scale, score, sentiment, confidence } of cases) {
    const fake = fakeRuntime((_url, init) => {
      const query = JSON.parse(String(init?.body)).query as string;
      assert.match(query, /User\(name: \$userName\).*scoreFormat/s);
      assert.match(query, /\bscore\s*\n/);
      assert.doesNotMatch(query, /score\(format:/);
      return jsonResponse({ data: {
        User: { mediaListOptions: { scoreFormat: scale } },
        MediaListCollection: { lists: [{ entries: [
          { score, status: "COMPLETED", progress: 12,
            media: { id: 501, idMal: 101, title: { romaji: "Copper Comet" } } },
        ] }] },
      } });
    });
    const result = await createProviderAdapter(fake.runtime)
      .fetchAniListUsernameImport("fixture-user", index);
    assert.deepEqual([result.history[0].score, result.history[0].scoreScale], [score, scale]);
    assert.deepEqual(result.entries.map((entry) =>
      [entry.preference.sentiment, entry.preference.confidence]), [[sentiment, confidence]]);
  }
});

test("AniList import rejects missing or inconsistent score scale", async () => {
  for (const scale of [undefined, "POINT_5", "NOT_A_SCALE"]) {
    const fake = fakeRuntime(() => jsonResponse({ data: {
      User: { mediaListOptions: { scoreFormat: scale } },
      MediaListCollection: { lists: [{ entries: [
        { score: 8, status: "COMPLETED", progress: 1,
          media: { id: 501, idMal: 101, title: { romaji: "Copper Comet" } } },
      ] }] },
    } }));
    await assert.rejects(createProviderAdapter(fake.runtime)
      .fetchAniListUsernameImport("fixture-user", index), /score format|invalid list score/);
  }
});

test("metadata and seasonal reads share one Jikan quota in the browser adapter", async () => {
  const fake = fakeRuntime((url) => new URL(url).pathname === "/v4/seasons/now"
    ? jsonResponse({ data: [] }) : jsonResponse({ data: { year: 2026 } }));
  const provider = createProviderAdapter(fake.runtime);
  await Promise.all([
    provider.fetchAnimeMetadataFromJikan(101),
    provider.fetchAnimeMetadataFromJikan(102),
    provider.fetchAnimeMetadataFromJikan(103),
    provider.fetchAnimeMetadataFromJikan(104),
    provider.fetchSeasonalAnime(12),
  ]);
  const starts = fake.requests.map((request) => request.at);
  assert.equal(starts.length, 5);
  for (const at of starts) {
    assert.ok(starts.filter((other) => other >= at && other < at + 1_000).length <= 3);
  }
  assert.ok(starts[3] >= 1_000);
  assert.equal(fake.requests.some((request) => request.url.includes("r.jina.ai")), false);
});

test("aborted imports and metadata discard delayed responses and stop retrying", async () => {
  let releaseImport: (() => void) | undefined;
  let importSignal: AbortSignal | undefined;
  const delayed = fakeRuntime((_url, init) => {
    importSignal = init?.signal ?? undefined;
    return new Promise<Response>((resolve) => {
      releaseImport = () => resolve(jsonResponse([{ anime_id: 102, score: 8 }]));
    });
  });
  const importController = new AbortController();
  const importPromise = createProviderAdapter(delayed.runtime)
    .fetchMalUsernameImport("fixture-user", index, importController.signal);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.ok(importSignal);
  importController.abort();
  releaseImport?.();
  await assert.rejects(importPromise, { name: "AbortError" });
  assert.equal(importSignal.aborted, true);
  assert.equal(delayed.requests.length, 1);

  let releaseSleep: (() => void) | undefined;
  const retrying = fakeRuntime(() => jsonResponse({}, 429));
  retrying.runtime.sleep = async () => new Promise<void>((resolve) => { releaseSleep = resolve; });
  const retryController = new AbortController();
  const retryPromise = createProviderAdapter(retrying.runtime)
    .fetchAniListUsernameImport("fixture-user", index, retryController.signal);
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.ok(releaseSleep);
  retryController.abort();
  releaseSleep?.();
  await assert.rejects(retryPromise, { name: "AbortError" });
  assert.equal(retrying.requests.length, 1);

  let releaseMetadata: (() => void) | undefined;
  const metadata = fakeRuntime((_url, init) => {
    assert.ok(init?.signal);
    return new Promise<Response>((resolve) => {
      releaseMetadata = () => resolve(jsonResponse({ data: { year: 2026 } }));
    });
  });
  const metadataController = new AbortController();
  const metadataPromise = createProviderAdapter(metadata.runtime)
    .fetchAnimeMetadataFromJikan(102, metadataController.signal);
  await new Promise<void>((resolve) => setImmediate(resolve));
  metadataController.abort();
  releaseMetadata?.();
  await assert.rejects(metadataPromise, { name: "AbortError" });
});

test("AniList distinguishes a returned empty list from an unavailable collection", async () => {
  const empty = fakeRuntime(() => jsonResponse({ data: { MediaListCollection: { lists: [] } } }));
  const emptyResult = await createProviderAdapter(empty.runtime)
    .fetchAniListUsernameImport("fixture-user", index);
  assert.deepEqual(emptyResult, { entries: [], history: [], duplicateCount: 0,
    ratedCount: 0, unmappedCount: 0 });

  const unavailable = fakeRuntime(() => jsonResponse({ data: { MediaListCollection: null } }));
  await assert.rejects(
    createProviderAdapter(unavailable.runtime).fetchAniListUsernameImport("fixture-user", index),
    { name: "ProviderUnavailableError" },
  );
});

test("provider error text and username are not reflected in import errors", async () => {
  const privateText = "private-user raw-history 1,2,3";
  const failed = fakeRuntime(() => jsonResponse({ errors: [{ message: privateText }] }));
  await assert.rejects(
    createProviderAdapter(failed.runtime).fetchAniListUsernameImport(privateText, index),
    (error: unknown) => error instanceof Error &&
      error.message === "AniList API returned an error; import was not applied.",
  );
  const blocked = fakeRuntime(() => jsonResponse({}, 403));
  await assert.rejects(
    createProviderAdapter(blocked.runtime).fetchAniListUsernameImport(privateText, index),
    (error: unknown) => error instanceof Error &&
      error.message === "anilist request unavailable (403).",
  );
});

test("artifact loader validates synthetic files through injected local transport", async () => {
  const data = new Map<string, unknown>([
    ["./demo-data/graph.aggregate.compact.json", fixture("graph.aggregate.compact.json")],
    ["./demo-data/graph-explorer.aggregate.compact.json", fixture("graph-explorer.aggregate.compact.json")],
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
  assert.equal(graph.format, "graph-compact-v3");
  assert.deepEqual(graph.userIds, []);
  assert.deepEqual(graph.ua, []);
  const explorer = await loader.fetchExplorerGraph(graph);
  assert.notEqual(explorer, graph);
  assert.equal(explorer.format, "graph-compact-v3");
  assert.deepEqual(explorer.userIds, []);
  assert.deepEqual(explorer.ua, []);
  assert.equal(explorer.edgeCount < graph.edgeCount, true);
  assert.equal((await loader.fetchModelRecommendationIndex())?.factors, 2);
  assert.equal(loader.getLoadedModelFormat(), "model-mf-compact-v1");
  assert.equal(await loader.getActiveReleaseManifest(), null);
  assert.equal((await loader.fetchDemoCatalog()).length, 8);
  assert.deepEqual(fake.requests.map((request) => request.url), [
    "./demo-data/graph.aggregate.compact.json",
    "./demo-data/graph-explorer.aggregate.compact.json",
    "./demo-data/model-mf-web.compact.json",
    "./demo-data/catalog.json",
  ]);
});

test("v2 explorer loading rejects missing and mismatched provenance", async () => {
  const main = fixture("graph.compact.json");
  const explorer = fixture("graph-explorer.compact.json") as Record<string, unknown>;
  for (const [sample, expected] of [
    [null, /Unable to load graph-explorer.compact.json/],
    [{ ...explorer, sourceGraphId: "0".repeat(64) }, /sourceGraphId or graph provenance/],
    [{ ...explorer, dataset: { ...(explorer.dataset as object), sha256: "0".repeat(64) } }, /sourceGraphId or graph provenance/],
  ] as const) {
    const fake = fakeRuntime((url) => {
      if (url === "./data/graph.compact.json") return jsonResponse(main);
      if (url === "./data/graph-explorer.compact.json" && sample) return jsonResponse(sample);
      return new Response("", { status: 404 });
    });
    const loader = createArtifactLoader(fake.runtime, false);
    const graph = await loader.fetchGraph();
    await assert.rejects(loader.fetchExplorerGraph(graph), expected);
  }
  const reordered = { ...explorer,
    semantics: Object.fromEntries(Object.entries(explorer.semantics as object).reverse()),
    config: Object.fromEntries(Object.entries(explorer.config as object).reverse()) };
  const fake = fakeRuntime((url) => {
    if (url === "./data/graph.compact.json") return jsonResponse(main);
    if (url === "./data/graph-explorer.compact.json") return jsonResponse(reordered);
    return new Response("", { status: 404 });
  });
  const loader = createArtifactLoader(fake.runtime, false);
  const graph = await loader.fetchGraph();
  assert.equal((await loader.fetchExplorerGraph(graph)).edgeCount, explorer.edgeCount);
});

test("v1 recommendation loading rejects a mixed v2 explorer", async () => {
  const compact = fixture("graph.compact.json") as Record<string, unknown>;
  const { role, graphId, dataset, semantics, config, truncation, ...v1 } = compact;
  const fake = fakeRuntime((url) => {
    if (url === "./data/graph.compact.json") return jsonResponse({ ...v1, format: "graph-compact-v1" });
    if (url === "./data/graph-explorer.compact.json") return jsonResponse(fixture("graph-explorer.compact.json"));
    return new Response("", { status: 404 });
  });
  const loader = createArtifactLoader(fake.runtime, false);
  const graph = await loader.fetchGraph();
  await assert.rejects(loader.fetchExplorerGraph(graph), /v2 explorer requires a v2 recommendation graph/);
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
    "./data/active.json",
    "./data/graph.compact.json.gz", "./data/graph.compact.json",
    "./data/graph.json.gz", "./data/graph.json",
  ]);
  assert.equal(fake.requests[0].init?.cache, "no-store");
});

test("gzip-only legacy graph, explorer, and model load through bounded local decompression", async () => {
  const files = new Map<string, unknown>([
    ["graph.compact.json.gz", fixture("graph.compact.json")],
    ["graph-explorer.compact.json.gz", fixture("graph-explorer.compact.json")],
    ["model-mf-web.compact.json.gz", fixture("model-mf-web.compact.json")],
  ]);
  const fake = fakeRuntime((url) => {
    if (url === "./data/active.json") return jsonResponse({}, 404);
    const file = files.get(url.replace("./data/", ""));
    if (file !== undefined) return gzipResponse(file);
    throw new Error(`Unexpected plain request: ${url}`);
  });
  const loader = createArtifactLoader(fake.runtime, false);
  const graph = await loader.fetchGraph();
  assert.equal(graph.format, "graph-compact-v2");
  assert.equal((await loader.fetchExplorerGraph(graph)).role, "visualization");
  assert.equal((await loader.fetchModelRecommendationIndex())?.factors, 2);
  assert.equal(fake.requests.length, 4);
  assert.ok(fake.requests.slice(1).every((request) => request.init?.cache === "no-store"));
});

test("browser-decoded gzip JSON is parsed once and a missing gzip uses fresh plain JSON", async () => {
  const graph = fixture("graph.compact.json");
  const decoded = fakeRuntime((url) => url === "./data/active.json"
    ? jsonResponse({}, 404)
    : url === "./data/graph.compact.json.gz"
      ? new Response(JSON.stringify(graph), { headers: { "Content-Encoding": "gzip" } })
      : new Response("", { status: 404 }));
  assert.equal((await createArtifactLoader(decoded.runtime, false).fetchGraph()).format,
    "graph-compact-v2");
  assert.deepEqual(decoded.requests.map((request) => request.url), [
    "./data/active.json", "./data/graph.compact.json.gz",
  ]);
  const plain = fakeRuntime((url, init) => {
    if (url === "./data/active.json") return jsonResponse({}, 404);
    if (url === "./data/graph.compact.json.gz") return jsonResponse({}, 404);
    if (url === "./data/graph.compact.json") {
      assert.equal(init?.cache, "no-store");
      return jsonResponse(graph);
    }
    throw new Error(`Unexpected request: ${url}`);
  });
  assert.equal((await createArtifactLoader(plain.runtime, false).fetchGraph()).format,
    "graph-compact-v2");
});

test("unsupported gzip decompression uses a plain file or names the missing fallback", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "DecompressionStream");
  Object.defineProperty(globalThis, "DecompressionStream",
    { configurable: true, writable: true, value: undefined });
  try {
    const graph = fixture("graph.compact.json");
    for (const plainPresent of [true, false]) {
      const fake = fakeRuntime((url) => {
        if (url === "./data/active.json") return jsonResponse({}, 404);
        if (url === "./data/graph.compact.json.gz") return gzipResponse(graph);
        if (url === "./data/graph.compact.json") {
          return plainPresent ? jsonResponse(graph) : jsonResponse({}, 404);
        }
        throw new Error(`Unexpected request: ${url}`);
      });
      const loading = createArtifactLoader(fake.runtime, false).fetchGraph();
      if (plainPresent) assert.equal((await loading).format, "graph-compact-v2");
      else await assert.rejects(loading,
        /graph\.compact\.json\.gz: this browser cannot decompress gzip and graph\.compact\.json is missing/);
    }
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "DecompressionStream", descriptor);
    else Reflect.deleteProperty(globalThis, "DecompressionStream");
  }
});

test("present malformed gzip and HTTP errors fail without silently taking a plain graph", async () => {
  const graph = fixture("graph.compact.json");
  for (const [gzResponse, expected] of [
    [bytesResponse(new Uint8Array([0x1f, 0x8b, 0, 1, 2])),
      /graph\.compact\.json\.gz: invalid gzip stream/],
    [new Response("error", { status: 503 }),
      /graph\.compact\.json\.gz: unable to load \(503\)/],
  ] as const) {
    const fake = fakeRuntime((url) => {
      if (url === "./data/active.json") return jsonResponse({}, 404);
      if (url === "./data/graph.compact.json.gz") return gzResponse.clone();
      if (url === "./data/graph.compact.json") return jsonResponse(graph);
      throw new Error(`Unexpected request: ${url}`);
    });
    await assert.rejects(createArtifactLoader(fake.runtime, false).fetchGraph(), expected);
    assert.deepEqual(fake.requests.map((request) => request.url),
      ["./data/active.json", "./data/graph.compact.json.gz"]);
  }
});

test("a present gzip graph with an invalid field names its transport file", async () => {
  const graph = structuredClone(fixture("graph.compact.json")) as Record<string, any>;
  graph.aa[0][1] = graph.anime.length;
  const fake = fakeRuntime((url) => url === "./data/active.json" ? jsonResponse({}, 404)
    : url === "./data/graph.compact.json.gz" ? gzipResponse(graph)
      : jsonResponse({}, 404));
  await assert.rejects(createArtifactLoader(fake.runtime, false).fetchGraph(),
    /graph\.compact\.json\.gz: aa\[0\]\[1\] references an index outside/);
});

test("plain HTTP errors name the file while transport failures remain generic", async () => {
  for (const [responder, expected] of [
    [(url: string) => url.endsWith(".gz") ? jsonResponse({}, 404)
      : new Response("error", { status: 503 }), /Unable to load graph\.compact\.json \(503\)/],
    [(url: string) => { if (url.endsWith(".gz")) throw new Error("private transport text");
      return jsonResponse({}, 404); }, /Artifact transport failed/],
  ] as const) {
    const fake = fakeRuntime((url) => url === "./data/active.json"
      ? jsonResponse({}, 404) : responder(url));
    await assert.rejects(createArtifactLoader(fake.runtime, false).fetchGraph(), expected);
  }
});

test("legacy compressed, decoded, and plain size limits stop oversized payloads", async () => {
  const largeJson = { padding: "x".repeat(4096) };
  for (const [gz, plain, limits, expected] of [
    [gzipResponse(largeJson), jsonResponse({}, 404),
      { compressedBytes: 16, plainBytes: 512 }, /graph\.compact\.json\.gz: byte length exceeds transport limit/],
    [gzipResponse(largeJson), jsonResponse({}, 404),
      { compressedBytes: 256, plainBytes: 512 }, /graph\.compact\.json\.gz: byte length exceeds transport limit/],
    [jsonResponse({}, 404), jsonResponse(largeJson),
      { compressedBytes: 256, plainBytes: 512 }, /graph\.compact\.json: byte length exceeds transport limit/],
    [new Response(JSON.stringify(largeJson), { headers: { "Content-Encoding": "gzip" } }),
      jsonResponse({}, 404), { compressedBytes: 256, plainBytes: 512 },
      /graph\.compact\.json\.gz: byte length exceeds transport limit/],
    [new Response("{}", { headers: { "Content-Length": "999" } }),
      jsonResponse({}, 404), { compressedBytes: 256, plainBytes: 512 },
      /graph\.compact\.json\.gz: advertised byte length exceeds transport limit/],
  ] as const) {
    const fake = fakeRuntime((url) => url === "./data/active.json" ? jsonResponse({}, 404)
      : url === "./data/graph.compact.json.gz" ? gz.clone() : plain.clone());
    await assert.rejects(createArtifactLoader(fake.runtime, false, "./", limits).fetchGraph(), expected);
  }
  assert.throws(() => createArtifactLoader(fakeRuntime(() => jsonResponse({}, 404)).runtime,
    false, "./", { plainBytes: 257 * 1024 * 1024 }), /plainBytes: legacy transport limit/);
});

test("a stale versioned manifest and graph are retried without cache, then hash checked", async () => {
  const manifestBytes = readFileSync(new URL("release-manifest.json", fixtureRoot));
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  const graphBytes = readFileSync(new URL("graph.compact.json", fixtureRoot));
  const pointer = { format: "active-release-bundle-v1", tag: manifest.tag,
    bundleId: manifest.bundleId,
    manifestSha256: crypto.createHash("sha256").update(manifestBytes).digest("hex") };
  const prefix = `./data/bundles/${manifest.bundleId}/`;
  const fake = fakeRuntime((url, init) => {
    if (url === "./data/active.json") return jsonResponse(pointer);
    if (url === `${prefix}release-manifest.json`) {
      return bytesResponse(init?.cache === "no-store" ? manifestBytes : Buffer.from("{}"));
    }
    if (url === `${prefix}graph.compact.json`) {
      return bytesResponse(init?.cache === "no-store" ? graphBytes : Buffer.from("{}"));
    }
    throw new Error(`Unexpected legacy request: ${url}`);
  });
  assert.equal((await createArtifactLoader(fake.runtime, false).fetchGraph()).format,
    "graph-compact-v2");
  assert.deepEqual(fake.requests.map(({ url, init }) => [url, init?.cache ?? "default"]), [
    ["./data/active.json", "no-store"],
    [`${prefix}release-manifest.json`, "default"],
    [`${prefix}release-manifest.json`, "no-store"],
    [`${prefix}graph.compact.json`, "default"],
    [`${prefix}graph.compact.json`, "no-store"],
  ]);
});

test("a persistently stale manifest or asset fails after one no-store retry", async () => {
  const manifestBytes = readFileSync(new URL("release-manifest.json", fixtureRoot));
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  const pointer = { format: "active-release-bundle-v1", tag: manifest.tag,
    bundleId: manifest.bundleId,
    manifestSha256: crypto.createHash("sha256").update(manifestBytes).digest("hex") };
  const prefix = `./data/bundles/${manifest.bundleId}/`;
  for (const badManifest of [true, false]) {
    const fake = fakeRuntime((url) => {
      if (url === "./data/active.json") return jsonResponse(pointer);
      if (url === `${prefix}release-manifest.json`) {
        return bytesResponse(badManifest ? Buffer.from("{}") : manifestBytes);
      }
      if (url === `${prefix}graph.compact.json`) return bytesResponse(Buffer.from("{}"));
      throw new Error(`Unexpected legacy request: ${url}`);
    });
    await assert.rejects(createArtifactLoader(fake.runtime, false).fetchGraph(), badManifest
      ? /release-manifest\.json: SHA-256 differs from active\.json\.manifestSha256/
      : /graph\.compact\.json: byte length or SHA-256 differs from release-manifest\.json/);
    assert.equal(fake.requests.filter((request) => request.init?.cache === "no-store").length, 2);
    assert.ok(fake.requests.every((request) => !request.url.endsWith(".gz")));
  }
});
