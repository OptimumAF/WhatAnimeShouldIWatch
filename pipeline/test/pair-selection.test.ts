import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { aggregateAnimePairs, type PairUser } from "../src/core/pair-aggregation.js";

const fixture = JSON.parse(fs.readFileSync(
  new URL("../../fixtures/synthetic-pair-selection.json", import.meta.url), "utf8",
)) as {
  users: PairUser[];
  expected: { pairVisits: number; candidatePairs: number; topPair: string; topSupport: number; topWeight: number };
};
const budgets = { maxPairVisits: 8, maxCandidatePairs: 6 };

test("support-ranked output cap selects the later strong pair, independent of input order", () => {
  for (const user of fixture.users) {
    assert.equal(user.ratings.reduce((sum, rating) => sum + rating.normalizedScore, 0), 0);
  }
  assert.equal(fixture.users[0].ratings.map((rating) => rating.animeId).join(":"), "1:2");

  const variants = [fixture.users, [...fixture.users].reverse(), fixture.users.map((user) => ({
    ...user, ratings: [...user.ratings].reverse(),
  }))];
  for (const users of variants) {
    const result = aggregateAnimePairs(users, 0, 1, budgets);
    assert.deepEqual([...result.pairs], [[fixture.expected.topPair, {
      support: fixture.expected.topSupport, weight: fixture.expected.topWeight,
    }]]);
    assert.equal(result.stats.pairVisits, fixture.expected.pairVisits);
    assert.equal(result.stats.candidatePairs, fixture.expected.candidatePairs);
    assert.equal(result.stats.excludedByOutputLimit, 5);
  }
});

test("minimum support and per-anime neighbor limits select from exact candidate statistics", () => {
  const full = aggregateAnimePairs(fixture.users, 0, 0, budgets);
  assert.deepEqual([...full.pairs.keys()], ["1:2", "3:4", "3:5", "3:6", "4:5", "4:6"]);
  assert.equal(full.stats.selectedPairs, 6);
  const supported = aggregateAnimePairs(fixture.users, 0, 0, { ...budgets, minSupport: 2 });
  assert.deepEqual([...supported.pairs.keys()], ["3:4"]);
  assert.equal(supported.stats.excludedBySupport, 5);

  const neighbors = aggregateAnimePairs(fixture.users, 0, 0, {
    ...budgets, maxNeighborsPerAnime: 1,
  });
  assert.deepEqual([...neighbors.pairs.keys()], ["1:2", "3:4"]);
  assert.equal(neighbors.stats.excludedByNeighborLimit, 4);
});

test("output cap is separate from fail-closed pair-visit and candidate-key budgets", () => {
  assert.throws(() => aggregateAnimePairs(fixture.users, 0, 1, {
    ...budgets, maxPairVisits: 7,
  }), /pair-visit budget.*8.*7/i);
  assert.throws(() => aggregateAnimePairs(fixture.users, 0, 1, {
    ...budgets, maxCandidatePairs: 5,
  }), /candidate-key budget.*5/i);
  assert.throws(() => aggregateAnimePairs(fixture.users, 0, 1, {
    ...budgets, maxPairVisits: 0,
  }), /maxPairVisits.*positive safe integer/i);
});

test("equal support and magnitude resolve by numeric pair IDs", () => {
  const users: PairUser[] = [
    { userId: "a", ratings: [{ animeId: 10, normalizedScore: 1 }, { animeId: 11, normalizedScore: -1 }] },
    { userId: "b", ratings: [{ animeId: 2, normalizedScore: 1 }, { animeId: 100, normalizedScore: -1 }] },
  ];
  const result = aggregateAnimePairs(users, 0, 1, { maxPairVisits: 2, maxCandidatePairs: 2 });
  assert.deepEqual([...result.pairs.keys()], ["2:100"]);
});

test("legacy per-user cap reports skipped ratings without hiding input-budget checks", () => {
  const result = aggregateAnimePairs(fixture.users, 2, 0, budgets);
  assert.equal(result.stats.ratingsSkippedByUserCap, 2);
  assert.equal(result.stats.pairVisits, 4);
  assert.deepEqual(result.pairs.get("3:4"), { support: 3, weight: 1 });
});

test("malformed centered inputs fail before exporting NaN weights or inflated support", () => {
  const valid = fixture.users[0];
  assert.throws(() => aggregateAnimePairs([{ ...valid, ratings: [
    { animeId: 1, normalizedScore: Number.NaN }, valid.ratings[1],
  ] }], 0, 0, budgets), /Invalid anime ID or centered score/);
  assert.throws(() => aggregateAnimePairs([{ ...valid, ratings: [
    valid.ratings[0], valid.ratings[0],
  ] }], 0, 0, budgets), /Duplicate anime ID/);
  assert.throws(() => aggregateAnimePairs([valid, valid], 0, 0, budgets), /Duplicate user ID/);
});
