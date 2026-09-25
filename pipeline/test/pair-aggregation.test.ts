import assert from "node:assert/strict";
import test from "node:test";
import { aggregateAnimePairs, type PairUser } from "../src/core/pair-aggregation.js";

const observations: PairUser[] = [
  { userId: "user-c", ratings: [{ animeId: 1, normalizedScore: 5 }, { animeId: 2, normalizedScore: 5 }] },
  { userId: "user-a", ratings: [{ animeId: 1, normalizedScore: 1 }, { animeId: 2, normalizedScore: 1 }] },
  { userId: "user-b", ratings: [{ animeId: 1, normalizedScore: 3 }, { animeId: 2, normalizedScore: 3 }] },
];

function legacyRecursiveAverage(scores: number[]): number {
  return scores.reduce((current, score) => current === null ? score : (current + score) / 2, null as number | null) ?? 0;
}

test("the previous recursive average depends on observation order", () => {
  assert.equal(legacyRecursiveAverage([1, 3, 5]), 3.5);
  assert.equal(legacyRecursiveAverage([5, 3, 1]), 2.5);
});

test("three pair observations use their arithmetic mean regardless of user and rating order", () => {
  const variants = [
    observations,
    [...observations].reverse(),
    [...observations].reverse().map((user) => ({
      ...user,
      ratings: [...user.ratings].reverse(),
    })),
  ];
  for (const users of variants) {
    const result = aggregateAnimePairs(users, 0, 0);
    assert.deepEqual(result.pairs.get("1:2"), { weight: 3, support: 3 });
    assert.equal(result.stats.pairVisits, 3);
    assert.equal(result.stats.candidatePairs, 1);
  }
});

test("fractional pair means are stable within a declared numeric tolerance", () => {
  const users: PairUser[] = [
    { userId: "z", ratings: [{ animeId: 8, normalizedScore: 0.1 }, { animeId: 9, normalizedScore: 0.2 }] },
    { userId: "a", ratings: [{ animeId: 8, normalizedScore: 0.3 }, { animeId: 9, normalizedScore: 0.4 }] },
    { userId: "m", ratings: [{ animeId: 8, normalizedScore: 0.5 }, { animeId: 9, normalizedScore: 0.6 }] },
  ];
  const first = aggregateAnimePairs(users, 0, 0).pairs.get("8:9");
  const reversed = aggregateAnimePairs(users.reverse().map((user) => ({
    ...user, ratings: [...user.ratings].reverse(),
  })), 0, 0).pairs.get("8:9");
  assert.ok(first && reversed);
  assert.equal(first.support, 3);
  assert.ok(Math.abs(first.weight - 0.35) <= 1e-12);
  assert.ok(Math.abs(first.weight - reversed.weight) <= 1e-12);
});
