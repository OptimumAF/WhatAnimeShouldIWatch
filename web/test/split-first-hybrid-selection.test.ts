import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, sep } from "node:path";
import { test } from "node:test";
import type { TestContext } from "node:test";
import {
  evaluateNewUsers, fitUserIds, loadEvalBundle, parseEvalFixture,
} from "../bench/split-first-new-user-eval.ts";
import {
  parseHybridSpec, reportFrozenHybridTest, scoreHybridValidation, selectHybrid,
} from "../bench/split-first-hybrid-selection.ts";

const ROOT = resolve(import.meta.dirname, "../..");
const bundle = loadEvalBundle();
const validationPath = join(ROOT, "fixtures/synthetic-new-user-validation.json");
const reservedPath = join(ROOT, "fixtures/synthetic-new-user-final-test.json");
const candidatesPath = join(ROOT, "fixtures/synthetic-hybrid-candidates.json");
const rawValidation = JSON.parse(readFileSync(validationPath, "utf8"));
const fitUsers = fitUserIds(JSON.parse(readFileSync(join(ROOT,
  "fixtures/synthetic-new-user-fit.json"), "utf8")));
const validation = parseEvalFixture(rawValidation, bundle, fitUsers);
const spec = parseHybridSpec(JSON.parse(readFileSync(candidatesPath, "utf8")));

function tempDirectory(t: TestContext): string {
  const directory = mkdtempSync(join(tmpdir(), "wasiw-hybrid-"));
  assert.ok(directory.startsWith(resolve(tmpdir()) + sep));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  return directory;
}
function paths(directory: string, finalFixture = reservedPath) {
  const candidateFile = finalFixture === reservedPath ? candidatesPath
    : join(directory, "candidates.json");
  if (finalFixture !== reservedPath) {
    mkdirSync(directory, { recursive: true });
    writeFileSync(candidateFile, JSON.stringify({
      ...spec, testFixtureSha256: fixtureHash(finalFixture),
    }), "utf8");
  }
  return { validationFixture: validationPath, testFixture: finalFixture,
    candidates: candidateFile, selection: join(directory, "selection.json"),
    testReport: join(directory, "final-report.json") };
}
function fixtureHash(path: string): string {
  const bytes = readFileSync(path);
  const normalized = Buffer.from(bytes.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return createHash("sha256").update(normalized).digest("hex");
}
function temporaryFinal(directory: string, filename = "temp-final.json"): string {
  const value = structuredClone(rawValidation);
  value.format = "new-user-final-test-fixture-v1";
  for (const [index, user] of value.users.entries()) {
    user.userId = `invented-temp-final-${index + 1}`;
    user.test = user.validation;
    delete user.validation;
  }
  const path = join(directory, filename);
  writeFileSync(path, JSON.stringify(value), "utf8");
  return path;
}

test("declared hybrid weights use the browser path and keep endpoints, fallback, and eligibility", () => {
  assert.deepEqual(spec.modelWeights, [0, 0.25, 0.5, 0.75, 1]);
  const graphEnd = evaluateNewUsers(bundle, validation, "hybrid", 0);
  const modelEnd = evaluateNewUsers(bundle, validation, "hybrid", 1);
  const pureGraph = evaluateNewUsers(bundle, validation, "graph");
  const pureModel = evaluateNewUsers(bundle, validation, "model");
  assert.equal(graphEnd.length, 16);
  assert.equal(modelEnd.length, 16);
  for (const [index, row] of graphEnd.entries()) {
    assert.deepEqual(row.ranked.map((item) => item.animeId),
      pureGraph[index].ranked.map((item) => item.animeId));
    if (pureModel[index].eligibleCandidateCount > 0) {
      assert.deepEqual(modelEnd[index].ranked.map((item) => item.animeId),
        pureModel[index].ranked.map((item) => item.animeId));
    }
  }
  assert.ok(graphEnd.some((row, index) =>
    row.ranked.map((item) => item.animeId).join(",") !==
    modelEnd[index].ranked.map((item) => item.animeId).join(",")));
  const seenOnly = graphEnd.find((row) => row.userNumber === 2 && row.suppliedCount === 1)!;
  assert.equal(seenOnly.displayedEngine, "coverage");
  assert.ok(seenOnly.displayedCandidateCount > 0);
  for (const row of [...graphEnd, ...modelEnd]) {
    const user = validation.users[row.userNumber - 1];
    assert.ok(row.ranked.every((item) =>
      !user.historySeen.includes(item.animeId) && !user.exclude.includes(item.animeId)));
    if (user.includeOnly.length) {
      assert.ok(row.ranked.every((item) => user.includeOnly.includes(item.animeId)));
    }
  }
  const metric = scoreHybridValidation(bundle, validation, 0.5);
  assert.equal(metric.cases, 16);
  assert.equal(metric.measurableCases, 16);
  assert.equal(metric.perPrefix.length, 4);
  assert.equal(Object.values(metric.displayedEngines).reduce((sum, n) => sum + n, 0), 16);
  assert.throws(() => scoreHybridValidation(bundle, validation, 0.1), /undeclared model weight/);
  const noGraph = { ...bundle, positivePairs: [] };
  const soleModel = evaluateNewUsers(noGraph, validation, "hybrid", 0.5);
  const soleModelEndpoint = evaluateNewUsers(noGraph, validation, "hybrid", 1);
  assert.deepEqual(soleModel.map((row) => row.ranked),
    soleModelEndpoint.map((row) => row.ranked));
  const noModel = { ...bundle, model: { ...bundle.model,
    animeIds: [], titles: [], biases: [], embeddings: [] } };
  const soleGraph = evaluateNewUsers(noModel, validation, "hybrid", 0.5);
  assert.deepEqual(soleGraph.map((row) => row.ranked),
    pureGraph.map((row) => row.ranked));
});

test("validation labels change metrics, never candidate scores or displayed engines", () => {
  const changedRaw = structuredClone(rawValidation);
  changedRaw.users[0].validation[0].rawScore = 3;
  const changed = parseEvalFixture(changedRaw, bundle, fitUsers);
  const before = evaluateNewUsers(bundle, validation, "hybrid", 0.5);
  const after = evaluateNewUsers(bundle, changed, "hybrid", 0.5);
  assert.deepEqual(before.map((row) => row.ranked), after.map((row) => row.ranked));
  assert.deepEqual(before.map((row) => row.displayedEngine),
    after.map((row) => row.displayedEngine));
  assert.notDeepEqual(scoreHybridValidation(bundle, validation, 0.5),
    scoreHybridValidation(bundle, changed, 0.5));
});

test("hybrid validation scores the final franchise-selected order", () => {
  const baseline = evaluateNewUsers(bundle, validation, "hybrid", 0.5);
  const target = baseline.find((row) => row.displayedEngine === "hybrid" &&
    row.ranked.length > 2 && row.displayedCandidateCount === row.ranked.length)!;
  assert.ok(target);
  const linkedIds = target.ranked.slice(0, 2).map((item) => item.animeId);
  const related = structuredClone(bundle);
  for (const [index, animeId] of linkedIds.entries()) {
    related.catalog.find((item) => item.animeId === animeId)!.title =
      `Invented Shared Season ${index + 1}`;
  }
  const changed = evaluateNewUsers(related, validation, "hybrid", 0.5)
    .find((row) => row.userNumber === target.userNumber &&
      row.suppliedCount === target.suppliedCount)!;
  assert.deepEqual(changed.ranked.map((item) => item.animeId),
    target.ranked.map((item) => item.animeId));
  assert.equal(changed.displayedCandidateCount, target.displayedCandidateCount - 1);
});

test("reserved final scores cannot choose a weight or alter validation trials", (t) => {
  const directory = tempDirectory(t);
  const baseline = selectHybrid(bundle, paths(join(directory, "baseline")));
  const changed = JSON.parse(readFileSync(reservedPath, "utf8"));
  changed.users[0].test[0].rawScore = 1;
  const changedPath = join(directory, "changed-final.json");
  writeFileSync(changedPath, JSON.stringify(changed), "utf8");
  const variant = selectHybrid(bundle, paths(join(directory, "variant"), changedPath));
  assert.equal(variant.selectedModelWeight, baseline.selectedModelWeight);
  assert.deepEqual(variant.trials, baseline.trials);
  assert.equal(variant.validationFixtureSha256, baseline.validationFixtureSha256);
  assert.notEqual(variant.testFixtureSha256, baseline.testFixtureSha256);
  assert.equal(baseline.testFixtureSha256,
    fixtureHash(reservedPath));
  assert.ok(!JSON.stringify(baseline).includes("invented-new-"));
  assert.ok(!JSON.stringify(baseline).includes("invented-final-"));
  assert.ok(!existsSync(paths(join(directory, "baseline")).testReport));
  assert.ok(!existsSync(baseline.selectionPath + ".test-used"));
});

test("frozen temporary final report is one-use and rejects drift before or after marker", (t) => {
  const directory = tempDirectory(t);
  const finalPath = temporaryFinal(directory);
  const successPaths = paths(join(directory, "success"), finalPath);
  const selection = selectHybrid(bundle, successPaths);
  const copyPath = join(directory, "copied.json");
  writeFileSync(copyPath, readFileSync(selection.selectionPath));
  assert.throws(() => reportFrozenHybridTest(bundle, copyPath), /original path/);
  assert.ok(!existsSync(selection.selectionPath + ".test-used"));
  const changedBundle = { ...bundle, modelSha256: "0".repeat(64) };
  assert.throws(() => reportFrozenHybridTest(changedBundle, selection.selectionPath),
    /frozen inputs/);
  assert.ok(!existsSync(selection.selectionPath + ".test-used"));
  const report = reportFrozenHybridTest(bundle, selection.selectionPath) as Record<string, any>;
  assert.equal(report.format, "split-first-hybrid-final-test-v1");
  assert.equal(report.selectedModelWeight, selection.selectedModelWeight);
  assert.equal(report.test.cases, 16);
  assert.ok(existsSync(selection.selectionPath + ".test-used"));
  assert.throws(() => reportFrozenHybridTest(bundle, selection.selectionPath),
    /one-use marker already exists/);

  const badPath = join(directory, "malformed-final.json");
  writeFileSync(badPath, JSON.stringify({ format: "new-user-final-test-fixture-v1" }), "utf8");
  const badSelection = selectHybrid(bundle, paths(join(directory, "malformed"), badPath));
  assert.throws(() => reportFrozenHybridTest(bundle, badSelection.selectionPath),
    /Invalid new-user evaluation/);
  assert.ok(existsSync(badSelection.selectionPath + ".test-used"));
  assert.ok(!existsSync(badSelection.testReportPath));
  assert.throws(() => reportFrozenHybridTest(bundle, badSelection.selectionPath),
    /one-use marker already exists/);

  const changedPath = temporaryFinal(directory, "changed-final.json");
  const changedSelection = selectHybrid(bundle, paths(join(directory, "changed-choice"), changedPath));
  const changed = JSON.parse(readFileSync(changedPath, "utf8"));
  changed.users[0].test[0].rawScore = 1;
  writeFileSync(changedPath, JSON.stringify(changed), "utf8");
  assert.throws(() => reportFrozenHybridTest(bundle, changedSelection.selectionPath),
    /frozen final fixture hash/);
  assert.ok(existsSync(changedSelection.selectionPath + ".test-used"));
  assert.ok(!existsSync(changedSelection.testReportPath));
});

test("malformed specifications and reused or public paths fail before selection", (t) => {
  const mutated = structuredClone(spec) as Record<string, any>;
  mutated.modelWeights = [0, 0.5, 0.5, 1];
  assert.throws(() => parseHybridSpec(mutated), /candidate specification protocol/);
  mutated.modelWeights = [0, 0.25, 0.5, 0.75, 1];
  mutated.topK = 3;
  assert.throws(() => parseHybridSpec(mutated), /candidate specification protocol/);
  const directory = tempDirectory(t);
  const options = paths(directory);
  selectHybrid(bundle, options);
  assert.throws(() => selectHybrid(bundle, options), /unused output paths/);
  assert.throws(() => selectHybrid(bundle, {
    ...paths(join(directory, "public")),
    selection: join(ROOT, "web/public/data/invented-hybrid-selection.json"),
  }), /public or release output path/);
  assert.throws(() => selectHybrid(bundle, {
    ...paths(join(directory, "same")), testFixture: validationPath,
  }), /separate validation and test fixtures/);
  const changedFinal = join(directory, "changed-test.json");
  writeFileSync(changedFinal, readFileSync(reservedPath, "utf8") + "\n", "utf8");
  assert.throws(() => selectHybrid(bundle, {
    ...paths(join(directory, "stale"), changedFinal), candidates: candidatesPath,
  }), /predeclared fixture hashes/);
});

test("final cohort overlap is rejected after the marker; validation drift before it", (t) => {
  const directory = tempDirectory(t);
  const overlapping = structuredClone(rawValidation);
  overlapping.format = "new-user-final-test-fixture-v1";
  for (const user of overlapping.users) {
    user.test = user.validation;
    delete user.validation;
  }
  const overlapPath = join(directory, "overlap.json");
  writeFileSync(overlapPath, JSON.stringify(overlapping), "utf8");
  const overlapRecord = selectHybrid(bundle, paths(join(directory, "overlap-choice"), overlapPath));
  assert.throws(() => reportFrozenHybridTest(bundle, overlapRecord.selectionPath),
    /final cohort overlap/);
  assert.ok(existsSync(overlapRecord.selectionPath + ".test-used"));
  assert.ok(!existsSync(overlapRecord.testReportPath));

  const validationCopy = join(directory, "validation-copy.json");
  writeFileSync(validationCopy, readFileSync(validationPath));
  const stableFinal = temporaryFinal(directory);
  const driftRecord = selectHybrid(bundle, {
    ...paths(join(directory, "drift-choice"), stableFinal),
    validationFixture: validationCopy,
  });
  const changedValidation = structuredClone(rawValidation);
  changedValidation.users[0].validation[0].rawScore = 1;
  writeFileSync(validationCopy, JSON.stringify(changedValidation), "utf8");
  assert.throws(() => reportFrozenHybridTest(bundle, driftRecord.selectionPath),
    /frozen inputs/);
  assert.ok(!existsSync(driftRecord.selectionPath + ".test-used"));
});
