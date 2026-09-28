import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import {
  assertPinnedInputs, evaluateBaselineCases, loadBaselineBundle, METHODS,
  parseBaselineBundle, parseBaselineSpec, summarizeBaselineCases,
} from "../bench/split-first-baseline-ablation.ts";
import { fitUserIds, loadEvalBundle, parseEvalFixture } from "../bench/split-first-new-user-eval.ts";

function fixture(name: string): unknown {
  return JSON.parse(readFileSync(new URL("../../fixtures/" + name, import.meta.url), "utf8"));
}
const spec = parseBaselineSpec(fixture("synthetic-baseline-ablation-spec.json"));
const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const bundle = loadBaselineBundle(spec);
const raw = fixture("synthetic-new-user-validation.json");
const fitUsers = fitUserIds(fixture("synthetic-new-user-fit.json"));
const cohort = parseEvalFixture(raw, bundle.base, fitUsers);

test("pinned train-only bundle preserves the M5.5 graph model and declares all variants", () => {
  assert.doesNotThrow(() => assertPinnedInputs(spec));
  assert.equal(bundle.base.trainRowCount, 72);
  assert.equal(bundle.base.catalog.length, 24);
  assert.equal(bundle.base.modelSha256, loadEvalBundle().modelSha256);
  assert.equal([...bundle.trainCounts.values()].reduce((sum, count) => sum + count, 0), 72);
  assert.equal(new Set([bundle.base.modelSha256, ...Object.values(bundle.models)
    .map((variant) => variant.modelSha256)]).size, 4);
  assert.ok(bundle.audit.nonpositivePairEdgesExcluded > 0);
  assert.ok(bundle.similarityPairs.every((pair) => pair.support >= 2 && pair.weight > 0 &&
    Math.abs(pair.weight - pair.adjustedCosine * pair.support / (pair.support + 2)) < 1e-12));
});

test("every method ranks one centrally eligible catalog and the same labels per prefix", () => {
  const cases = evaluateBaselineCases(bundle, cohort, spec);
  assert.equal(cases.length, 16);
  assert.equal(cases.reduce((sum, row) => sum + row.eligiblePositiveIds.length, 0), 32);
  assert.deepEqual([...new Set(cases.map((row) => row.suppliedCount))], [1, 3, 5, 10]);
  for (const row of cases) {
    assert.deepEqual(row.methods.map((method) => method.method), METHODS);
    const user = cohort.users[row.userNumber - 1];
    const observed = new Set(user.observed.map((item) => item.animeId));
    for (const method of row.methods) {
      assert.deepEqual(method.eligibleCandidateIds, row.universeIds);
      assert.equal(method.ranked.length, row.universeIds.length);
      assert.equal(new Set(method.ranked.map((item) => item.animeId)).size, row.universeIds.length);
      assert.ok(method.ranked.every((item) => Number.isFinite(item.score)));
      assert.ok(method.ranked.every((item) => !user.historySeen.includes(item.animeId) &&
        !user.exclude.includes(item.animeId)));
      if (user.includeOnly.length) {
        assert.ok(method.ranked.every((item) => user.includeOnly.includes(item.animeId)));
      }
      assert.ok(method.displayedIds.every((id) => row.universeIds.includes(id)));
      assert.equal(method.selectorRemoved, 0, "unrelated invented titles should retain one common display set");
    }
    assert.ok(row.eligiblePositiveIds.every((id) => row.universeIds.includes(id) &&
      !observed.has(id)));
  }
  const report = summarizeBaselineCases(bundle, cases);
  assert.equal(report.methods.length, METHODS.length);
  assert.equal(report.measurableCases, 16);
  assert.equal(report.eligiblePositiveLabels, 32);
  assert.ok(report.methods.every((item) => item.perPrefix.every((slice) =>
    slice.measurableCases === 4)));
});

test("validation labels change metrics but cannot change any method score or rank", () => {
  const edited = structuredClone(raw) as Record<string, any>;
  edited.users[0].validation[0].rawScore = 3;
  const changed = parseEvalFixture(edited, bundle.base, fitUsers);
  const before = evaluateBaselineCases(bundle, cohort, spec);
  const after = evaluateBaselineCases(bundle, changed, spec);
  assert.deepEqual(before.map((row) => row.universeIds), after.map((row) => row.universeIds));
  assert.deepEqual(before.map((row) => row.methods.map((item) => item.ranked)),
    after.map((row) => row.methods.map((item) => item.ranked)));
  assert.deepEqual(before.map((row) => row.methods.map((item) => item.displayedIds)),
    after.map((row) => row.methods.map((item) => item.displayedIds)));
  assert.notDeepEqual(before.map((row) => row.eligiblePositiveIds),
    after.map((row) => row.eligiblePositiveIds));
  assert.notDeepEqual(summarizeBaselineCases(bundle, before).methods,
    summarizeBaselineCases(bundle, after).methods);
});

test("Seen-only prefix uses neutral zero instead of inserting an undeclared fallback", () => {
  const row = evaluateBaselineCases(bundle, cohort, spec).find((item) =>
    item.userNumber === 2 && item.suppliedCount === 1)!;
  for (const method of ["supported-adjusted-cosine", "genre-overlap",
    "v1-positive-pair-graph", "plain-mf", "positive-pair-mf",
    "unit-positive-pair-mf", "shrunk-positive-pair-mf",
    "hybrid-default-0.5"] as const) {
    const result = row.methods.find((item) => item.method === method)!;
    assert.equal(result.signalCandidates, 0, method);
    assert.deepEqual(result.ranked.map((item) => item.animeId), row.universeIds, method);
  }
  assert.ok(row.methods.find((item) => item.method === "train-count")!.signalCandidates > 0);
});

test("malformed method spec, missing counts, and unsupported similarity fail closed", () => {
  const changedSpec = structuredClone(spec);
  changedSpec.methods[0] = "test-informed-winner";
  assert.throws(() => parseBaselineSpec(changedSpec), /specification protocol/);
  const rawBundle = JSON.parse(execFileSync("python", ["ml/split_first_baseline_export.py"],
    { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 }));
  const short = structuredClone(rawBundle);
  short.trainCounts.pop();
  assert.throws(() => parseBaselineBundle(short, spec), /trainCounts coverage/);
  const negative = structuredClone(rawBundle);
  negative.similarityPairs[0].weight = -1;
  assert.throws(() => parseBaselineBundle(negative, spec), /similarityPairs/);
});
