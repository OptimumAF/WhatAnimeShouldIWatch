import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseCompactGraph } from "../src/artifacts.ts";
import { buildCatalogCoverageRecommendations, buildRecommendationIndexFromCompact,
  createCandidateEligibilityPolicy } from "../src/recommendations.ts";
import { manualPreference } from "../src/preferences.ts";
import { mergeWatchlistFeedback, validateWatchlist, watchedWatchlistAnimeIds } from "../src/watchlist.ts";
import type { WatchlistEntry } from "../src/watchlist.ts";

const graph = parseCompactGraph(JSON.parse(readFileSync(
  new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8")), "invented graph");
const index = buildRecommendationIndexFromCompact(graph);
const entry = (animeId: number, status: WatchlistEntry["status"], rating: number | null): WatchlistEntry =>
  ({ animeId, title: `<Invented ${animeId}>`, status, rating });

test("local watchlist validates all five statuses, integer ratings, unknown IDs, and strict fields", () => {
  const statuses: WatchlistEntry["status"][] = [
    "plan_to_watch", "watching", "completed", "on_hold", "dropped",
  ];
  const entries = statuses.map((status, index) => entry(900 + index, status, index ? index * 2 : null));
  assert.deepEqual(validateWatchlist(entries), entries);
  assert.deepEqual(watchedWatchlistAnimeIds(entries), [901, 902, 903, 904]);
  for (const invalid of [
    [...entries, entries[0]],
    [{ ...entries[0], rating: 0 }],
    [{ ...entries[0], rating: 10.5 }],
    [{ ...entries[0], status: "future" }],
    [{ ...entries[0], title: " " }],
    [{ ...entries[0], source: "provider" }],
  ]) assert.throws(() => validateWatchlist(invalid));
});

test("only explicit ratings create browser feedback; planned and manual preferences retain precedence", () => {
  const none = mergeWatchlistFeedback([], [entry(101, "watching", null)], index);
  assert.deepEqual(none, []);
  const liked = mergeWatchlistFeedback([], [entry(101, "completed", 9)], index);
  assert.deepEqual(liked, [{ nodeId: "anime:101", sentiment: "liked", importance: 1,
    confidence: 0.75, source: "manual" }]);
  assert.deepEqual(mergeWatchlistFeedback([], [entry(101, "plan_to_watch", 9)], index), []);
  assert.deepEqual(mergeWatchlistFeedback([], [entry(101, "dropped", 2)], index)[0], {
    nodeId: "anime:101", sentiment: "disliked", importance: 1, confidence: 0.75, source: "manual",
  });
  assert.equal(mergeWatchlistFeedback([], [entry(101, "on_hold", 5)], index)[0].sentiment, "seen");
  const imported = { ...manualPreference("anime:101", "liked"), source: "import" as const };
  assert.equal(mergeWatchlistFeedback([imported], [entry(101, "completed", 2)], index)[0].sentiment,
    "disliked");
  const manual = manualPreference("anime:101", "liked", 2.4);
  assert.deepEqual(mergeWatchlistFeedback([manual], [entry(101, "completed", 2)], index), [manual]);
  assert.deepEqual(mergeWatchlistFeedback([manual], [entry(101, "plan_to_watch", null)], index), []);
  assert.deepEqual(mergeWatchlistFeedback([], [entry(99999, "completed", 10)], index), []);
});

test("a planned shortlist title is excluded by central eligibility without becoming a preference", () => {
  const baseline = buildCatalogCoverageRecommendations(index);
  const policy = createCandidateEligibilityPolicy({ index, preferences: [], history: [],
    watchlist: [entry(102, "plan_to_watch", null)], includeOnlyNodeIds: ["anime:102"],
    excludeNodeIds: [], filters: { genre: "", minYear: null, maxYear: null, minScore: null } });
  assert.deepEqual(policy.evaluate(baseline, new Map()).recommendations, []);
  assert.equal(baseline.some((item) => item.anime.animeId === 102), true);
});
