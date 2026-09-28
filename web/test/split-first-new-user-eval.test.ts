import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { preferenceFromHistory } from "../src/preferences.ts";
import type { HistoryEntry } from "../src/import-history.ts";
import {
  evaluateNewUsers, fitUserIds, loadEvalBundle, metricsForRanks,
  orderedObserved, parseEvalBundle, parseEvalFixture, summarizeNewUsers,
} from "../bench/split-first-new-user-eval.ts";

function fixture(name: string): unknown {
  return JSON.parse(readFileSync(new URL("../../fixtures/" + name, import.meta.url), "utf8"));
}
const bundle = loadEvalBundle();
const raw = fixture("synthetic-new-user-validation.json");
const fitUsers = fitUserIds(fixture("synthetic-new-user-fit.json"));
const cohort = parseEvalFixture(raw, bundle, fitUsers);

test("new-user fixture is disjoint, pinned, mapped, and checked against train-only catalog", () => {
  assert.equal(bundle.fitUserCount, 8);
  assert.equal(bundle.trainRowCount, 72);
  assert.equal(bundle.catalog.length, 24);
  assert.equal(cohort.users.length, 4);
  assert.ok(cohort.users.every((user) => !fitUsers.has(user.userId)));
  assert.ok(bundle.positivePairs.every((pair) => pair.weight > 0 && pair.support > 0));
  assert.equal(bundle.warmValidation.eligibleUsers, 6);
  const ranks = evaluateNewUsers(bundle, cohort);
  assert.equal(ranks.length, 16);
  assert.deepEqual([...new Set(ranks.map((row) => row.suppliedCount))], [1, 3, 5, 10]);
  assert.deepEqual(ranks.filter((row) => row.suppliedCount === 10)
    .map((row) => row.eligiblePositiveLabels), [2, 2, 2, 2]);
  const seenOnly = ranks.find((row) => row.userNumber === 2 && row.suppliedCount === 1)!;
  assert.equal(seenOnly.mappedSignalCount, 0);
  assert.equal(seenOnly.eligibleCandidateCount, 0);
  assert.equal(seenOnly.displayedEngine, "coverage");
  assert.ok(seenOnly.displayedCandidateCount > 0);
  assert.deepEqual(seenOnly.ranks, [null, null]);
  const report = summarizeNewUsers(bundle, ranks);
  assert.deepEqual(report.newUserValidation.map((row) => row.users), [4, 4, 4, 4]);
  assert.deepEqual(report.newUserValidation.map((row) => row.positiveLabels), [8, 8, 8, 8]);
  assert.deepEqual(report.newUserValidation.map((row) => row.suppliedRatings), [4, 12, 20, 40]);
  assert.deepEqual(report.newUserValidation.map((row) => row.vectorSignals), [3, 9, 16, 32]);
  assert.deepEqual(report.newUserValidation.map((row) => row.mappedSignals), [3, 9, 16, 32]);
  assert.ok(!JSON.stringify(report).includes("invented-new-"));
  assert.ok(!JSON.stringify(bundle).includes("invented-fit-"));
  assert.ok(!Object.hasOwn(bundle.model, "userFactors"));
});

test("native-scale mapping and score-independent nested supplied prefixes use browser functions", () => {
  const entry = (score: number): HistoryEntry => ({
    provider: "local", sourceId: "201", title: "Invented Amber", animeId: 201,
    status: "completed", sourceStatus: "completed", progressEpisodes: null,
    score, scoreScale: "local-10",
  });
  assert.deepEqual([4, 6, 7, 10].map((score) =>
    preferenceFromHistory(entry(score), "anime:201")?.sentiment),
  ["disliked", "seen", "liked", "liked"]);
  assert.deepEqual([4, 6, 7, 10].map((score) =>
    preferenceFromHistory(entry(score), "anime:201")?.confidence),
  [0.25, 0, 0.25, 1]);
  const user = cohort.users[0];
  const ordered = orderedObserved(user, cohort.seed);
  assert.equal(new Set(ordered.map((rating) => rating.animeId)).size, 10);
  const scoreEdited = { ...user, observed: user.observed.map((rating) =>
    ({ ...rating, rawScore: rating.rawScore === 9 ? 1 : 9 })) };
  assert.deepEqual(ordered.map((rating) => rating.animeId),
    orderedObserved(scoreEdited, cohort.seed).map((rating) => rating.animeId));
  for (const count of [1, 3, 5, 10]) {
    assert.deepEqual(ordered.slice(0, count).map((rating) => rating.animeId),
      ordered.slice(0, 10).map((rating) => rating.animeId).slice(0, count));
  }
});

test("editing a held-out label changes metrics but never model scores or candidate order", () => {
  const edited = structuredClone(raw) as Record<string, any>;
  edited.users[0].validation[0].rawScore = 3;
  const changed = parseEvalFixture(edited, bundle, fitUsers);
  const before = evaluateNewUsers(bundle, cohort);
  const after = evaluateNewUsers(bundle, changed);
  assert.deepEqual(before.map((row) => row.ranked), after.map((row) => row.ranked));
  assert.deepEqual(before.map((row) => row.displayedEngine),
    after.map((row) => row.displayedEngine));
  assert.deepEqual(before.map((row) => row.displayedCandidateCount),
    after.map((row) => row.displayedCandidateCount));
  assert.deepEqual(before.map((row) => row.mappedSignalCount),
    after.map((row) => row.mappedSignalCount));
  assert.notDeepEqual(before.map((row) => row.positiveLabels),
    after.map((row) => row.positiveLabels));
  assert.notDeepEqual(summarizeNewUsers(bundle, before).newUserValidation,
    summarizeNewUsers(bundle, after).newUserValidation);
});

test("browser candidate policy removes watched, history, exclusions, disallowed and filtered titles", () => {
  const rows = evaluateNewUsers(bundle, cohort);
  for (const row of rows) {
    const user = cohort.users[row.userNumber - 1];
    const supplied = new Set(orderedObserved(user, cohort.seed).slice(0, row.suppliedCount)
      .map((rating) => rating.animeId));
    assert.ok(row.ranked.every((item) => !supplied.has(item.animeId) &&
      !user.historySeen.includes(item.animeId) && !user.exclude.includes(item.animeId)));
    if (user.includeOnly.length) {
      assert.ok(row.ranked.every((item) => user.includeOnly.includes(item.animeId)));
    }
  }
  const user = structuredClone(cohort.users[0]);
  const target = user.validation[0].animeId;
  const matching = cohort.candidateMetadata.find((item) => item.animeId === target)!;
  user.includeOnly = [target];
  user.filters = { genre: matching.genres[1], minYear: matching.year,
    maxYear: matching.year, minScore: matching.score };
  const narrowed = evaluateNewUsers(bundle, { ...cohort, users: [user] })
    .find((row) => row.suppliedCount === 10)!;
  assert.deepEqual(narrowed.ranked.map((item) => item.animeId), [target]);
  assert.equal(narrowed.eligiblePositiveLabels, 1);
  assert.equal(narrowed.excludedPositiveLabels, 1);
  assert.equal(narrowed.excludedPositiveReasons.includeOnly, 1);
  user.filters = { ...user.filters, genre: "Absent Genre" };
  const filtered = evaluateNewUsers(bundle, { ...cohort, users: [user] })
    .find((row) => row.suppliedCount === 10)!;
  assert.equal(filtered.eligibleCandidateCount, 0);
  assert.equal(filtered.displayedCandidateCount, 0);
  assert.equal(filtered.eligiblePositiveLabels, 0);
  assert.equal(filtered.excludedPositiveLabels, 2);
  assert.deepEqual(filtered.excludedPositiveReasons,
    { watchedOrExcluded: 0, includeOnly: 1, metadataFilter: 1 });
  const blocked = { ...cohort.users[0],
    exclude: [...cohort.users[0].exclude, cohort.users[0].validation[0].animeId] };
  const blockedRow = evaluateNewUsers(bundle, { ...cohort, users: [blocked] })
    .find((row) => row.suppliedCount === 10)!;
  assert.equal(blockedRow.excludedPositiveReasons.watchedOrExcluded, 1);
});

test("malformed cohort, catalog, overlap, and policy inputs fail before scoring", () => {
  const changed = (change: (value: any) => void) => {
    const value = structuredClone(raw);
    change(value);
    assert.throws(() => parseEvalFixture(value, bundle, fitUsers), /Invalid new-user evaluation/);
  };
  changed((value) => { value.users[0].userId = "invented-fit-01"; });
  changed((value) => { value.users[1].userId = value.users[0].userId; });
  changed((value) => { value.users[0].validation[0].animeId = value.users[0].observed[0].animeId; });
  changed((value) => { value.users[0].observed[0].rawScore = Infinity; });
  changed((value) => { value.users[0].observed[0].animeId = 999; });
  changed((value) => { value.users[0].exclude = [999]; });
  changed((value) => { value.users[0].filters.minScore = -1; });
  changed((value) => { value.candidateMetadata.pop(); });
  const badBundle = structuredClone(bundle) as Record<string, any>;
  badBundle.catalog[0].animeId = 999;
  assert.throws(() => parseEvalBundle(badBundle), /Invalid new-user evaluation/);
});

test("hand-computed ranking metrics distinguish top-K quality from full reciprocal rank", () => {
  const metric = metricsForRanks([1, 3], 2, 3);
  assert.equal(metric.hitAtK, 1);
  assert.equal(metric.recallAtK, 1);
  assert.ok(Math.abs(metric.ndcgAtK - 1.5 / (1 + 1 / Math.log2(3))) < 1e-12);
  assert.equal(metric.reciprocalRank, 1);
  assert.deepEqual(metricsForRanks([4, null], 2, 3),
    { hitAtK: 0, recallAtK: 0, ndcgAtK: 0, reciprocalRank: 0.25 });
  assert.throws(() => metricsForRanks([0], 1, 3), /metric labels/);
});
