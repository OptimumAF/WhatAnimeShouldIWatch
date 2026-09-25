import assert from "node:assert/strict";
import { test } from "node:test";
import { parseTextHistory } from "../src/import-history.ts";
import {
  manualPreference, migrateLegacyPreferences, preferenceFromHistory, validatePreferences,
} from "../src/preferences.ts";
import type { HistoryEntry } from "../src/import-history.ts";

function scoreHistory(score: number | null, scoreScale: HistoryEntry["scoreScale"] = "mal-10"): HistoryEntry {
  return {
    provider: scoreScale === "mal-10" ? "mal" : "anilist", sourceId: "101", title: "Invented A",
    animeId: 101, status: "completed", sourceStatus: "completed", progressEpisodes: 12,
    score, scoreScale,
  };
}

test("native provider scores map to seen, liked, or disliked without treating every rating as positive", () => {
  const cases: Array<[number | null, HistoryEntry["scoreScale"], string, number]> = [
    [null, "mal-10", "seen", 0], [1, "mal-10", "disliked", 1],
    [4, "mal-10", "disliked", 0.25], [5, "mal-10", "seen", 0],
    [6, "mal-10", "seen", 0], [7, "mal-10", "liked", 0.25],
    [10, "mal-10", "liked", 1], [20, "POINT_100", "disliked", 0.75],
    [45, "POINT_100", "seen", 0], [65, "POINT_100", "seen", 0],
    [3, "POINT_5", "seen", 0], [4, "POINT_5", "liked", 0.5],
    [1, "POINT_3", "disliked", 0.75], [2, "POINT_3", "seen", 0],
    [3, "POINT_3", "liked", 1],
  ];
  for (const [score, scale, sentiment, confidence] of cases) {
    assert.deepEqual(preferenceFromHistory(scoreHistory(score, scale), "anime:101"), {
      nodeId: "anime:101", sentiment, importance: 1, confidence, source: "import",
    }, `${scale}: ${score}`);
  }
  assert.equal(preferenceFromHistory({ ...scoreHistory(9), status: "plan_to_watch" }, "anime:101"), null);
  assert.deepEqual(manualPreference("anime:101"), {
    nodeId: "anime:101", sentiment: "seen", importance: 1, confidence: 0, source: "manual",
  });
});

test("legacy weights survive while ambiguous watches stay unrated and low scores become dislikes", () => {
  const history = parseTextHistory("101, 3, Completed, 12\n102, 9, Completed, 12\n103, 0, Watching, 2").entries;
  const migrated = migrateLegacyPreferences([
    { nodeId: "anime:101", weight: 0.6 }, { nodeId: "anime:102", weight: 1.8 },
    { nodeId: "anime:103", weight: 1 }, { nodeId: "anime:999", weight: 1 },
    { nodeId: "anime:998", weight: 2.4 },
  ], history);
  assert.deepEqual(migrated.map(({ nodeId, sentiment, importance, confidence }) =>
    [nodeId, sentiment, importance, confidence]), [
    ["anime:101", "disliked", 0.6, 0.5],
    ["anime:102", "liked", 1.8, 0.75],
    ["anime:103", "seen", 1, 0],
    ["anime:999", "seen", 1, 0],
    ["anime:998", "liked", 2.4, 0.5],
  ]);
  assert.deepEqual(validatePreferences(migrated), migrated);
  assert.throws(() => validatePreferences([...migrated, migrated[0]]), /Invalid saved preference/);
  assert.throws(() => validatePreferences([{ ...migrated[0], confidence: 0 }]), /Invalid saved preference/);
});

test("conflicting stored provider scores migrate to Seen regardless of history order", () => {
  const low = scoreHistory(2);
  const high = { ...scoreHistory(9), provider: "local" as const,
    scoreScale: "local-10" as const };
  const selected = [{ nodeId: "anime:101", weight: 2 }];
  for (const history of [[low, high], [high, low]]) {
    assert.deepEqual(migrateLegacyPreferences(selected, history), [{
      nodeId: "anime:101", sentiment: "seen", importance: 2, confidence: 0, source: "legacy",
    }]);
  }
});
