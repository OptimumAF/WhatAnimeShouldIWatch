import assert from "node:assert/strict";
import { test } from "node:test";
import type { AnimeMetadata } from "../src/artifacts.ts";
import { selectFranchiseDiverseRecommendations } from "../src/franchise-diversity.ts";
import { parseTextHistory, seenHistoryAnimeIds } from "../src/import-history.ts";
import { buildRecommendationIndex } from "../src/recommendations.ts";
import type { GraphData } from "../src/artifacts.ts";
import { franchiseMetadata, franchiseResults, franchiseTitles } from
  "../bench/franchise-diversity-fixture.ts";

const ids = (items: readonly { anime: { animeId: number } }[]): number[] =>
  items.map((item) => item.anime.animeId);

test("known prequel and repeated franchise evidence narrow only the final eligible list", () => {
  const selected = selectFranchiseDiverseRecommendations(
    franchiseResults, franchiseMetadata, new Set([200]), false, franchiseTitles);
  assert.deepEqual(ids(selected.recommendations), [201, 203, 204, 205, 206, 207]);
  assert.deepEqual(selected.recommendations.map((item) => item.score), [10, 8, 7, 6, 5, 4]);
  assert.equal(selected.knownPrequelHidden, 0);
  assert.equal(selected.repeatedFranchiseHidden, 2);
  assert.equal(selected.relationshipPayloadCount, 6);
  assert.equal(selected.uncheckedCount, 2);
  assert.match(selected.notesByAnimeId.get(201) ?? "", /Orbit Knights \(in watched history\)/);
  assert.match(selected.notesByAnimeId.get(205) ?? "", /Prerequisites unverified/);
  assert.match(selected.notesByAnimeId.get(208) ?? "", /Possible same-series title match/);
});

test("a known unwatched prequel withholds its sequel; allow related restores exact rank and scores", () => {
  const preferred = selectFranchiseDiverseRecommendations(
    franchiseResults, franchiseMetadata, new Set(), false, franchiseTitles);
  assert.equal(preferred.knownPrequelHidden, 1);
  assert.ok(!ids(preferred.recommendations).includes(201));
  assert.match(preferred.notesByAnimeId.get(201) ?? "", /not in watched history/);
  const allowed = selectFranchiseDiverseRecommendations(
    franchiseResults, franchiseMetadata, new Set(), true, franchiseTitles);
  assert.deepEqual(ids(allowed.recommendations), ids(franchiseResults));
  assert.deepEqual(allowed.recommendations.map((item) => item.score),
    franchiseResults.map((item) => item.score));
  assert.equal(allowed.knownPrequelHidden, 0);
  assert.equal(allowed.repeatedFranchiseHidden, 0);
});

test("one-sided links and transitive groups do not turn missing predecessors into proof of safety", () => {
  const make = (id: number, relations: AnimeMetadata["relations"]): AnimeMetadata => ({
    animeId: id, year: null, score: null, genres: [], studios: [], synopsis: "", imageUrl: "",
    season: null, relations,
  });
  const metadata = new Map<number, AnimeMetadata>([
    [200, make(200, [{ kind: "sequel", animeId: 201, title: "Orbit Knights Season 2" }])],
    [201, make(201, [{ kind: "sequel", animeId: 202, title: "Orbit Knights Side Story" }])],
  ]);
  const preferred = selectFranchiseDiverseRecommendations(
    franchiseResults.slice(0, 4), metadata, new Set([200]), false, franchiseTitles);
  assert.deepEqual(ids(preferred.recommendations), [201, 203, 204]);
  assert.equal(preferred.knownPrequelHidden, 1);
  assert.match(preferred.notesByAnimeId.get(202) ?? "", /Orbit Knights Season 2 \(not in watched history\)/);
  assert.match(preferred.notesByAnimeId.get(203) ?? "", /Prerequisites unverified/);
});

test("serial title cues group likely duplicates but never assert a verified prequel", () => {
  const noMetadata = new Map<number, AnimeMetadata>();
  const candidates = [franchiseResults[0], franchiseResults[2], franchiseResults[7]];
  const selected = selectFranchiseDiverseRecommendations(
    candidates, noMetadata, new Set(), false, franchiseTitles);
  assert.deepEqual(ids(selected.recommendations), [201, 203]);
  assert.equal(selected.knownPrequelHidden, 0);
  assert.match(selected.notesByAnimeId.get(208) ?? "", /Possible same-series title match/);
  assert.match(selected.notesByAnimeId.get(208) ?? "", /Prerequisites unverified/);
});

test("missing and empty relationships stay uncertain, while repeated names group only for variety", () => {
  const repeated = { ...franchiseResults[4], anime: { ...franchiseResults[4].anime,
    label: "Cedar Tide" } };
  const metadata = new Map<number, AnimeMetadata>([
    [203, { animeId: 203, year: null, score: null, genres: [], studios: [],
      synopsis: "", imageUrl: "", season: null, relations: [] }],
  ]);
  const selected = selectFranchiseDiverseRecommendations(
    [franchiseResults[2], repeated, franchiseResults[5]], metadata,
    new Set(), false, franchiseTitles);
  assert.deepEqual(ids(selected.recommendations), [203, 206]);
  assert.equal(selected.relationshipPayloadCount, 1);
  assert.equal(selected.uncheckedCount, 2);
  assert.match(selected.notesByAnimeId.get(203) ?? "", /Prerequisites unverified/);
  assert.match(selected.notesByAnimeId.get(205) ?? "", /Possible same-series title match/);
});

test("an imported completed prequel outside the graph catalog still counts as watched", () => {
  const minimalGraph: GraphData = { generatedAt: "invented", userCount: 0, animeCount: 1,
    nodeCount: 1, edgeCount: 0,
    nodes: [{ id: "anime:201", label: "Orbit Knights Season 2", nodeType: "anime" }], edges: [] };
  const history = parseTextHistory("200, 9, Completed, 12\n209, 0, Plan to Watch, 0").entries;
  const seen = new Set(seenHistoryAnimeIds(history, buildRecommendationIndex(minimalGraph)));
  assert.deepEqual([...seen], [200]);
  const selected = selectFranchiseDiverseRecommendations(
    franchiseResults.slice(0, 1), franchiseMetadata, seen, false, franchiseTitles);
  assert.deepEqual(ids(selected.recommendations), [201]);
});
