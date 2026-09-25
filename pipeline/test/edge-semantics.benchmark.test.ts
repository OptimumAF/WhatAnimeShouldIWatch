import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { aggregateAnimePairs, type PairUser } from "../src/core/pair-aggregation.js";
import { scoreSupportShrunkAdjustedCosine, type CoRatedDeviation } from "../src/core/item-similarity.js";

const fixture = JSON.parse(fs.readFileSync(
  new URL("../../fixtures/synthetic-edge-semantics.json", import.meta.url), "utf8",
)) as {
  users: PairUser[];
  signedCases: { label: string; observations: CoRatedDeviation[]; cosine: number | null; shrunk: number | null }[];
  uniformRaters: { label: string; raw: number[]; centered: number[] }[];
};

function coRated(leftId: number, rightId: number, users: PairUser[] = fixture.users) {
  return users.flatMap((user) => {
    const left = user.ratings.find((rating) => rating.animeId === leftId);
    const right = user.ratings.find((rating) => rating.animeId === rightId);
    return left && right ? [{ left: left.normalizedScore, right: right.normalizedScore }] : [];
  });
}

function near(actual: number, expected: number): void {
  assert.ok(Math.abs(actual - expected) <= 1e-12, `${actual} differs from ${expected}`);
}

test("hand-computed fixture distinguishes pair preference from support-shrunk item similarity", () => {
  for (const user of fixture.users) {
    assert.equal(user.ratings.reduce((sum, rating) => sum + rating.normalizedScore, 0), 0);
  }
  const legacy = aggregateAnimePairs(fixture.users, 0, 0).pairs;
  const cases = [
    // label, pair, support, mean of pair deviations, adjusted cosine, shrunk with lambda=2
    ["aligned", 1, 3, 3, 2 / 3, 1, 3 / 5],
    ["co-disliked", 2, 4, 3, -1, 16 / Math.sqrt(288), (16 / Math.sqrt(288)) * (3 / 5)],
    ["opposed", 1, 2, 3, 0, -1, -3 / 5],
    ["one overlap", 1, 5, 1, 2, 1, 1 / 3],
  ] as const;
  for (const [label, leftId, rightId, support, mean, cosine, shrunk] of cases) {
    const pair = legacy.get(`${leftId}:${rightId}`);
    const observations = coRated(leftId, rightId);
    const similarity = scoreSupportShrunkAdjustedCosine(observations, 2);
    const reversed = scoreSupportShrunkAdjustedCosine(observations.reverse(), 2);
    assert.ok(pair && similarity && reversed, label);
    assert.equal(pair.support, support, label);
    assert.equal(similarity.support, support, label);
    near(pair.weight, mean);
    near(similarity.adjustedCosine, cosine);
    near(similarity.shrunkSimilarity, shrunk);
    near(reversed.shrunkSimilarity, similarity.shrunkSimilarity);
  }

  // The legacy statistic elevates a single shared rater above an aligned
  // three-rater pair and gives co-disliked items a negative edge.
  assert.ok(legacy.get("1:5")!.weight > legacy.get("1:3")!.weight);
  assert.ok(legacy.get("2:4")!.weight < 0);
  assert.ok(scoreSupportShrunkAdjustedCosine(coRated(1, 5), 2)!.shrunkSimilarity <
    scoreSupportShrunkAdjustedCosine(coRated(1, 3), 2)!.shrunkSimilarity);
  assert.ok(scoreSupportShrunkAdjustedCosine(coRated(2, 4), 2)!.shrunkSimilarity > 0);
});

test("zero magnitude and no overlap stay undefined without inventing an edge", () => {
  assert.equal(scoreSupportShrunkAdjustedCosine([], 2), null);
  assert.equal(scoreSupportShrunkAdjustedCosine([{ left: 0, right: 2 }], 2), null);
  assert.throws(() => scoreSupportShrunkAdjustedCosine([{ left: Number.NaN, right: 1 }], 2),
    /finite centered scores/);
  assert.throws(() => scoreSupportShrunkAdjustedCosine([{ left: 1, right: 1 }], -1),
    /finite and nonnegative/);
});

test("opposition, co-dislike, neutral, constant, flat, sparse, and missing pairs keep distinct signs", () => {
  for (const { label, observations, cosine, shrunk } of fixture.signedCases) {
    const result = scoreSupportShrunkAdjustedCosine(observations, 2);
    if (cosine === null || shrunk === null) {
      assert.equal(result, null, label);
      continue;
    }
    assert.ok(result, label);
    assert.equal(result.support, observations.length, label);
    near(result.adjustedCosine, cosine);
    near(result.shrunkSimilarity, shrunk);
  }
  const disliked = fixture.signedCases.find((entry) => entry.label === "universally disliked pair")!;
  assert.ok(disliked.observations.every(({ left, right }) => left < 0 && right < 0));
  assert.ok(scoreSupportShrunkAdjustedCosine(disliked.observations, 2)!.shrunkSimilarity > 0);
  const constant = fixture.signedCases.find((entry) => entry.label === "constant nonzero item vectors")!;
  assert.ok(constant.observations.every(({ left, right }) => left === 1 && right === 1));
  near(scoreSupportShrunkAdjustedCosine(constant.observations, 2)!.shrunkSimilarity, 1 / 2);
});

test("user centering removes high/low rater offsets and leaves flat raters undefined", () => {
  const centered = fixture.uniformRaters.map(({ label, raw, centered: expected }) => {
    const mean = raw.reduce((sum, score) => sum + score, 0) / raw.length;
    const deviations = raw.map((score) => score - mean);
    assert.deepEqual(deviations, expected, label);
    return deviations;
  });
  assert.deepEqual(centered[0], centered[1]);
  const aligned = scoreSupportShrunkAdjustedCosine(
    centered.slice(0, 2).map((row) => ({ left: row[0], right: row[1] })), 2,
  );
  const opposed = scoreSupportShrunkAdjustedCosine(
    centered.slice(0, 2).map((row) => ({ left: row[0], right: row[2] })), 2,
  );
  assert.ok(aligned && opposed);
  near(aligned.shrunkSimilarity, 1 / 2);
  near(opposed.shrunkSimilarity, -1 / 2);
  assert.equal(scoreSupportShrunkAdjustedCosine(
    centered.slice(2).map((row) => ({ left: row[0], right: row[1] })), 2,
  ), null);
});
