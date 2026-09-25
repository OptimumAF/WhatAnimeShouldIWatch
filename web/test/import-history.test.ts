import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseCompactGraph } from "../src/artifacts.ts";
import {
  mergeHistory, parseTextHistory, previewHistory, resolveHistoryAnime,
  seenHistoryNodeIds, validateHistoryEntries,
} from "../src/import-history.ts";
import { buildRecommendationIndexFromCompact } from "../src/recommendations.ts";

const graph = parseCompactGraph(JSON.parse(readFileSync(
  new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8",
)), "synthetic graph");
const index = buildRecommendationIndexFromCompact(graph);

test("text history retains scores, scale, statuses, progress, unscored and unmapped entries", () => {
  const parsed = parseTextHistory([
    "101, 9, Completed, 12",
    "102, 0, Watching, 3",
    "99999, 0, Plan to Watch, 0",
    "Unknown Fixture Title, 4, Dropped, 1",
  ].join("\n"));
  assert.equal(parsed.duplicates, 0);
  assert.deepEqual(parsed.entries.map((entry) => [
    entry.sourceId, entry.status, entry.progressEpisodes, entry.score, entry.scoreScale,
  ]), [
    ["anime:101", "completed", 12, 9, "local-10"],
    ["anime:102", "watching", 3, null, "local-10"],
    ["anime:99999", "plan_to_watch", 0, null, "local-10"],
    ["title:unknown fixture title", "dropped", 1, 4, "local-10"],
  ]);
  assert.equal(resolveHistoryAnime(parsed.entries[0], index)?.nodeId, "anime:101");
  assert.equal(resolveHistoryAnime(parsed.entries[2], index), null);
  assert.deepEqual(seenHistoryNodeIds(parsed.entries, index), ["anime:101", "anime:102"]);
  assert.deepEqual(previewHistory(parsed, [], index, "merge"), {
    total: 4, duplicates: 0, mapped: 2, unmapped: 2,
    unscored: 2, seen: 3, planned: 1, added: 4, updated: 0, unchanged: 0, removed: 0,
  });
});

test("identity deduplication is deterministic and repeated merge is idempotent", () => {
  const parsed = parseTextHistory("101, 8, Watching, 2\n101, 9, Completed, 12\n102, 0, Plan to Watch, 0");
  assert.equal(parsed.duplicates, 1);
  assert.deepEqual(parsed.entries.map((entry) => entry.score), [9, null]);
  const first = mergeHistory([], parsed.entries, "merge");
  assert.deepEqual(mergeHistory(first, parsed.entries, "merge"), first);
  assert.deepEqual(previewHistory(parsed, first, index, "merge"), {
    total: 2, duplicates: 1, mapped: 2, unmapped: 0,
    unscored: 1, seen: 1, planned: 1, added: 0, updated: 0, unchanged: 2, removed: 0,
  });
  const replacement = parseTextHistory("102, 6, Watching, 3");
  assert.equal(previewHistory(replacement, first, index, "replace").removed, 1);
  assert.deepEqual(mergeHistory(first, replacement.entries, "replace"), replacement.entries);
});

test("unscored local watched titles are seen without inventing a completed status", () => {
  const parsed = parseTextHistory("102");
  assert.equal(parsed.entries[0].status, "unknown");
  assert.equal(parsed.entries[0].score, null);
  assert.deepEqual(seenHistoryNodeIds(parsed.entries, index), ["anime:102"]);
});

test("text parser rejects oversized and invalid files instead of silently losing fields", () => {
  assert.throws(() => parseTextHistory("1".repeat(128 * 1024 + 1)), /128 KiB/);
  assert.throws(() => parseTextHistory("101, 8oops"), /Invalid score on line 1/);
  assert.throws(() => parseTextHistory("101, 11"), /Invalid score on line 1/);
  assert.throws(() => parseTextHistory("101, 8, completed, -1"), /Invalid progress on line 1/);
  assert.throws(() => parseTextHistory("101, 8, completed, 1, extra"), /Invalid text import line 1/);
  assert.throws(() => validateHistoryEntries([{ provider: "mal", sourceId: "101" }]),
    /Invalid saved history entry/);
});
