import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { evaluateBaselineCases, loadBaselineBundle, parseBaselineSpec } from
  "../bench/split-first-baseline-ablation.ts";
import {
  assertExperimentInputs, evaluateExperimentCases, EXPERIMENT_METHODS,
  parseExperimentBundle, parseExperimentSpec, summarizeExperimentCases,
} from "../bench/split-first-experiment-eval.ts";
import { fitUserIds, parseEvalFixture } from "../bench/split-first-new-user-eval.ts";

function fixture(name: string): unknown {
  return JSON.parse(readFileSync(new URL("../../fixtures/" + name, import.meta.url), "utf8"));
}
const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const baselineSpec = parseBaselineSpec(fixture("synthetic-baseline-ablation-spec.json"));
const spec = parseExperimentSpec(fixture("synthetic-experiment-spec.json"));
const baseline = loadBaselineBundle(baselineSpec);
const raw = fixture("synthetic-new-user-validation.json");
const fitUsers = fitUserIds(fixture("synthetic-new-user-fit.json"));
const cohort = parseEvalFixture(raw, baseline.base, fitUsers);
const exported = JSON.parse(execFileSync("python", ["ml/split_first_experiment_export.py"],
  { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 }));
const bundle = parseExperimentBundle(exported, spec, baseline);

test("two candidates use the fixed training/catalog inputs and no provider features", () => {
  assert.doesNotThrow(() => assertExperimentInputs(spec, baselineSpec, cohort));
  assert.equal(bundle.trainRowCount, 72);
  assert.equal(bundle.positiveTrainInteractions, 35);
  assert.equal(bundle.positivePairEdges, 91);
  assert.deepEqual(bundle.contentVocabulary, ["genre:Adventure", "genre:Mystery",
    "studio:Studio A", "studio:Studio B", "studio:Studio C"]);
  assert.deepEqual(Object.keys(bundle.models), EXPERIMENT_METHODS);
  for (const method of EXPERIMENT_METHODS) {
    assert.deepEqual(bundle.models[method].model.animeIds,
      baseline.base.catalog.map((item) => item.animeId));
    assert.deepEqual(bundle.models[method].model.titles,
      baseline.base.catalog.map((item) => item.title));
  }
});

test("both candidates share every baseline eligible universe, label, and final selector", () => {
  const before = evaluateBaselineCases(baseline, cohort, baselineSpec);
  const cases = evaluateExperimentCases(baseline, cohort, baselineSpec, spec, bundle);
  assert.equal(cases.length, 16);
  assert.equal(cases.reduce((sum, row) => sum + row.eligiblePositiveIds.length, 0), 32);
  for (const [i, row] of cases.entries()) {
    assert.deepEqual(row.universeIds, before[i].universeIds);
    assert.deepEqual(row.eligiblePositiveIds, before[i].eligiblePositiveIds);
    assert.deepEqual(row.methods.slice(0, -2), before[i].methods);
    assert.deepEqual(row.methods.slice(-2).map((method) => method.method), EXPERIMENT_METHODS);
    const user = cohort.users[row.userNumber - 1];
    for (const method of row.methods.slice(-2)) {
      assert.deepEqual(method.eligibleCandidateIds, row.universeIds);
      assert.equal(method.ranked.length, row.universeIds.length);
      assert.ok(method.ranked.every((item) => Number.isFinite(item.score) &&
        !user.historySeen.includes(item.animeId) && !user.exclude.includes(item.animeId)));
      if (user.includeOnly.length) {
        assert.ok(method.ranked.every((item) => user.includeOnly.includes(item.animeId)));
      }
      assert.ok(method.displayedIds.every((id) => row.universeIds.includes(id)));
      assert.equal(method.selectorRemoved, 0);
    }
  }
  const seenOnly = cases.find((row) => row.userNumber === 2 && row.suppliedCount === 1)!;
  for (const method of seenOnly.methods.slice(-2)) {
    assert.equal(method.signalCandidates, 0);
    assert.deepEqual(method.ranked.map((item) => item.animeId), seenOnly.universeIds);
  }
  const report = summarizeExperimentCases(cases);
  assert.deepEqual(report.map((item) => item.method), EXPERIMENT_METHODS);
  assert.ok(report.every((item) => item.meanSignalCandidates > 0));
});

test("validation labels affect metrics but cannot alter candidate scores or displayed order", () => {
  const changed = structuredClone(raw) as Record<string, any>;
  changed.users[0].validation[0].rawScore = 3;
  const changedCohort = parseEvalFixture(changed, baseline.base, fitUsers);
  const before = evaluateExperimentCases(baseline, cohort, baselineSpec, spec, bundle);
  const after = evaluateExperimentCases(baseline, changedCohort, baselineSpec, spec, bundle);
  assert.deepEqual(before.map((row) => row.methods.slice(-2).map((item) => item.ranked)),
    after.map((row) => row.methods.slice(-2).map((item) => item.ranked)));
  assert.deepEqual(before.map((row) => row.methods.slice(-2).map((item) => item.displayedIds)),
    after.map((row) => row.methods.slice(-2).map((item) => item.displayedIds)));
  assert.notDeepEqual(summarizeExperimentCases(before), summarizeExperimentCases(after));
});

test("the content item model uses signed browser preferences", () => {
  const changed = structuredClone(raw) as Record<string, any>;
  for (const rating of changed.users[0].observed) {
    if (rating.rawScore <= 4) rating.rawScore = 5;
  }
  const changedCohort = parseEvalFixture(changed, baseline.base, fitUsers);
  const before = evaluateExperimentCases(baseline, cohort, baselineSpec, spec, bundle)
    .find((row) => row.userNumber === 1 && row.suppliedCount === 10)!;
  const after = evaluateExperimentCases(baseline, changedCohort, baselineSpec, spec, bundle)
    .find((row) => row.userNumber === 1 && row.suppliedCount === 10)!;
  assert.notDeepEqual(before.methods.at(-1)!.ranked, after.methods.at(-1)!.ranked);
});

test("stale specs, mismatched metadata, and tampered model bytes fail before scoring", () => {
  const changedSpec = structuredClone(spec);
  changedSpec.lightgcn.epochs = 3;
  assert.throws(() => parseExperimentSpec(changedSpec), /specification protocol/);
  const changedCohortRaw = structuredClone(raw) as Record<string, any>;
  changedCohortRaw.candidateMetadata[0].genres = ["Invented", "Comedy"];
  const changedCohort = parseEvalFixture(changedCohortRaw, baseline.base, fitUsers);
  assert.throws(() => assertExperimentInputs(spec, baselineSpec, changedCohort),
    /content metadata genre\/catalog mismatch/);
  assert.throws(() => evaluateExperimentCases(baseline, changedCohort,
    baselineSpec, spec, bundle), /content metadata genre\/catalog mismatch/);
  const changedBundle = structuredClone(exported);
  changedBundle.models["content-multihot"].model.embeddings[0][0] = 0.25;
  assert.throws(() => parseExperimentBundle(changedBundle, spec, baseline),
    /content-multihot provenance\/catalog/);
  const wrongTitle = structuredClone(exported);
  wrongTitle.models["lightgcn-bpr"].model.titles[0] = "Changed invented title";
  assert.throws(() => parseExperimentBundle(wrongTitle, spec, baseline),
    /lightgcn-bpr provenance\/catalog/);
});
