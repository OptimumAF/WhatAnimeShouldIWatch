import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  evaluateBaselineCases, loadBaselineBundle, parseBaselineSpec, summarizeBaselineCases,
} from "../bench/split-first-baseline-ablation.ts";
import {
  assertPinnedMetricSpec, bootstrapUserMean, catalogCoverage, genreDiversity,
  measureLatency, parseMetricSpec, percentile, popularityBias, positiveEdgeSupport,
  rankMetrics, summarizeMetricCases,
} from "../bench/split-first-metric-report.ts";
import { fitUserIds, parseEvalFixture } from "../bench/split-first-new-user-eval.ts";

function fixture(name: string): unknown {
  return JSON.parse(readFileSync(new URL("../../fixtures/" + name, import.meta.url), "utf8"));
}
const spec = parseMetricSpec(fixture("synthetic-metric-report-spec.json"));
const baselineSpec = parseBaselineSpec(fixture("synthetic-baseline-ablation-spec.json"));
const bundle = loadBaselineBundle(baselineSpec);
const cohort = parseEvalFixture(fixture("synthetic-new-user-validation.json"), bundle.base,
  fitUserIds(fixture("synthetic-new-user-fit.json")));
const cases = evaluateBaselineCases(bundle, cohort, baselineSpec);

test("hand-computed binary recall and NDCG obey both cutoffs and empty-label rule", () => {
  assert.deepEqual(rankMetrics([10, 11, 12], [10, 12], 1), { recall: 0.5, ndcg: 1 });
  const three = rankMetrics([10, 11, 12], [10, 12], 3)!;
  assert.equal(three.recall, 1);
  assert.ok(Math.abs(three.ndcg - 1.5 / (1 + 1 / Math.log2(3))) < 1e-12);
  const eleven = Array.from({ length: 11 }, (_, i) => i + 1);
  assert.deepEqual(rankMetrics(eleven, [11], 10), { recall: 0, ndcg: 0 });
  assert.ok(Math.abs(rankMetrics(eleven, [11], 20)!.ndcg - 1 / Math.log2(12)) < 1e-12);
  assert.equal(rankMetrics([1], [], 10), null);
  assert.throws(() => rankMetrics([1, 1], [1], 10), /displayed IDs/);
  assert.throws(() => rankMetrics([1, 2], [2, 2], 10), /positive IDs/);
});

test("hand-computed restricted coverage, genre spread and count bias have fixed denominators", () => {
  const rows = [
    { universeIds: [1, 2, 3], displayedIds: [1, 2] },
    { universeIds: [2, 3, 4], displayedIds: [4, 3] },
  ];
  assert.deepEqual(catalogCoverage(rows, 1),
    { distinctDisplayed: 2, eligibleUnion: 4, fraction: 0.5 });
  assert.deepEqual(catalogCoverage(rows, 2),
    { distinctDisplayed: 4, eligibleUnion: 4, fraction: 1 });
  assert.throws(() => catalogCoverage(
    [{ universeIds: [1], displayedIds: [2] }], 1), /coverage subset/);
  const genres = new Map<number, string[]>([
    [1, ["A"]], [2, ["A", "B"]], [3, ["C"]],
  ]);
  assert.ok(Math.abs(genreDiversity([1, 2, 3], genres)! - 5 / 6) < 1e-12);
  assert.equal(genreDiversity([1], genres), null);
  const counts = new Map([[1, 1], [2, 3], [3, 5]]);
  assert.equal(popularityBias([2, 3], [1, 2, 3], counts), 1);
  assert.equal(popularityBias([1], [1, 2, 3], counts), -2);
  assert.throws(() => popularityBias([4], [1, 2, 3], counts), /popularity subset/);
});

test("support slice treats absent positive evidence as zero and takes the strongest co-rater support", () => {
  const pairs = [
    { leftAnimeId: 1, rightAnimeId: 2, weight: 0.1, support: 1 },
    { leftAnimeId: 2, rightAnimeId: 3, weight: 0.2, support: 3 },
  ];
  assert.equal(positiveEdgeSupport(2, [1, 3], pairs), 3);
  assert.equal(positiveEdgeSupport(2, [1], pairs), 1);
  assert.equal(positiveEdgeSupport(2, [], pairs), 0);
  assert.equal(positiveEdgeSupport(4, [1, 3], pairs), 0);
});

test("user-cluster bootstrap, percentile interpolation and timing use declared mechanics", () => {
  assert.equal(percentile([0, 10, 20, 30], 0.25), 7.5);
  const rows = [
    { userNumber: 1, value: 0 }, { userNumber: 1, value: 0 },
    { userNumber: 2, value: 1 }, { userNumber: 2, value: 1 },
  ];
  const first = bootstrapUserMean(rows, 271828, 2048);
  assert.deepEqual(first, bootstrapUserMean(rows, 271828, 2048));
  assert.equal(first.users, 2);
  assert.equal(first.mean, 0.5);
  assert.equal(first.p025, 0);
  assert.equal(first.p975, 1);
  let clock = 0;
  const durations = [9, 1, 3, 2];
  assert.deepEqual(measureLatency(() => { clock += durations.shift()!; },
    () => clock, 1, 3), { warmups: 1, samples: 3, medianMs: 2, p95Ms: 3 });
  assert.throws(() => measureLatency(() => {}, () => NaN, 0, 1), /latency clock/);
});

test("pinned ten-method report retains M5.6 means and counts distinct users and support labels", () => {
  assert.doesNotThrow(() => assertPinnedMetricSpec(spec));
  const report = summarizeMetricCases(bundle, cohort, cases, spec);
  const baseline = summarizeBaselineCases(bundle, cases);
  assert.equal(report.validationUsers, 4);
  assert.equal(report.eligibleUsers, 4);
  assert.equal(report.measurableCases, 16);
  assert.equal(report.eligiblePositiveLabels, 32);
  assert.equal(report.methods.length, 10);
  assert.ok(report.lowSupportZeroEvidenceLabels >= 0);
  for (const method of report.methods) {
    const previous = baseline.methods.find((item) => item.method === method.method)!;
    assert.ok(Math.abs(method.full.ranking[0].meanNdcg! - previous.meanNdcgAt10) < 1e-12);
    assert.ok(Math.abs(method.full.ranking[0].meanRecall! - previous.meanRecallAt10) < 1e-12);
    assert.equal(method.perPrefix.length, 4);
    assert.equal(method.perPrefix.reduce((sum, row) => sum + row.eligiblePositiveLabels, 0), 32);
    assert.equal(method.byPositivePairSupport.lowOrMissing.eligiblePositiveLabels +
      method.byPositivePairSupport.atLeastTwo.eligiblePositiveLabels, 32);
    assert.equal(method.bootstrapNdcgAt10?.users, 4);
    assert.equal(method.full.ancillary[0].coverage.eligibleUnion, 24);
    assert.equal(method.full.ancillary[0].diversityCases, 16);
    assert.equal(method.full.ancillary[0].popularityCases, 16);
  }
});

test("zero-positive case is counted without a fabricated zero metric; invalid lists fail closed", () => {
  const zero = structuredClone(cases);
  zero[0].eligiblePositiveIds = [];
  const report = summarizeMetricCases(bundle, cohort, zero, spec);
  assert.equal(report.measurableCases, 15);
  assert.equal(report.casesWithoutEligiblePositive, 1);
  assert.equal(report.eligiblePositiveLabels, 30);
  const duplicate = structuredClone(cases);
  duplicate[0].methods[0].displayedIds[1] = duplicate[0].methods[0].displayedIds[0];
  assert.throws(() => summarizeMetricCases(bundle, cohort, duplicate, spec), /displayed IDs/);
  const impossible = structuredClone(cases);
  impossible[0].methods[0].displayedIds[0] = 999;
  assert.throws(() => summarizeMetricCases(bundle, cohort, impossible, spec),
    /method candidate\/selector contract/);
});

test("modified metric definitions or a stale baseline specification hash are rejected", () => {
  const altered = structuredClone(spec);
  altered.topKs = [10, 21];
  assert.throws(() => parseMetricSpec(altered), /specification protocol/);
  const stale = structuredClone(spec);
  stale.baselineSpecSha256 = "0".repeat(64);
  assert.throws(() => assertPinnedMetricSpec(stale), /baseline specification hash/);
});
