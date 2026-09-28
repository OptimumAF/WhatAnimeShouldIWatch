import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { freezeServingEvidence, markServingFinalUsed, servingSha256,
  verifyServingFreeze } from "../src/core/model-serving-freeze.js";

const H = (letter: string) => letter.repeat(64);

function setup(t: TestContext) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-serving-freeze-"));
  t.after(() => {
    const resolved = fs.realpathSync(root);
    assert.ok(resolved.startsWith(path.resolve(os.tmpdir()) + path.sep));
    fs.rmSync(root, { recursive: true, force: true });
  });
  const evidence = path.join(root, "private");
  const candidate = path.join(root, "candidate");
  fs.mkdirSync(evidence);
  fs.mkdirSync(candidate);
  const finalBytes = Buffer.from('{"format":"model-promotion-final-v1","users":[]}\n');
  fs.writeFileSync(path.join(evidence, "serving-final.json"), finalBytes);
  const finalSha256 = servingSha256(finalBytes);
  const cohort = { format: "model-promotion-cohort-v2", sourceName: "invented-source",
    datasetSha256: H("b"), seed: 42, finalSha256 };
  const cohortBytes = Buffer.from(JSON.stringify(cohort) + "\n");
  fs.writeFileSync(path.join(evidence, "serving-cohort.json"), cohortBytes);
  const plan = { format: "model-serving-quality-plan-v1", sourceName: "invented-source",
    rawContentSha256: H("a"), graphDatasetSha256: H("b"),
    cohortSha256: servingSha256(cohortBytes), finalSha256,
    baselineCandidates: ["graph", "genre", "coverage"],
    decisionRef: "docs/decisions/0035-serving-evaluation-freeze.md",
    seed: 42, suppliedCount: 1, positiveRawScoreMin: 7, topK: 5,
    minimumEligibleUsers: 4, minimumPositiveLabels: 4,
    minimumServingCoverage: 0.5, minimumNdcgLift: 0.05,
    maximumP95LatencyMs: 1000, latencyWarmups: 1, latencySamples: 7 };
  const expected = { sourceName: plan.sourceName, rawContentSha256: plan.rawContentSha256,
    graphDatasetSha256: plan.graphDatasetSha256,
    cohortSha256: plan.cohortSha256, finalSha256 };
  return { evidence, candidate, plan, expected };
}

test("prefit freeze binds exact private bytes and final scoring is one-use", (t) => {
  const fixture = setup(t);
  const result = freezeServingEvidence(fixture.evidence, fixture.candidate, fixture.plan);
  const expected = { ...fixture.expected, freezeSha256: result.freezeSha256 };
  assert.deepEqual(verifyServingFreeze(fixture.evidence, expected, fixture.plan, false), fixture.plan);
  assert.throws(() => verifyServingFreeze(fixture.evidence, expected, fixture.plan),
    /serving-final.json.test-used/);
  markServingFinalUsed(fixture.evidence, result.freezeSha256, fixture.expected.finalSha256);
  verifyServingFreeze(fixture.evidence, expected, fixture.plan);
  assert.throws(() => markServingFinalUsed(fixture.evidence,
    result.freezeSha256, fixture.expected.finalSha256), /EEXIST/);
  fs.appendFileSync(path.join(fixture.evidence, "serving-final.json"), " ");
  assert.throws(() => verifyServingFreeze(fixture.evidence, expected, fixture.plan),
    /cohort or reserved final bytes changed/);
});

test("selection or candidate model present before a freeze refuses the ceremony", (t) => {
  const selected = setup(t);
  fs.writeFileSync(path.join(selected.evidence, "selection.json"), "{}");
  assert.throws(() => freezeServingEvidence(selected.evidence, selected.candidate, selected.plan),
    /selection.json: already exists/);
  const fitted = setup(t);
  fs.writeFileSync(path.join(fitted.candidate, "model-mf-web.compact.json"), "{}");
  assert.throws(() => freezeServingEvidence(fitted.evidence, fitted.candidate, fitted.plan),
    /candidate.model: already exists/);
});

test("an invalid predeclared quality floor cannot be frozen", (t) => {
  const fixture = setup(t);
  const invalid = { ...fixture.plan, minimumNdcgLift: 0 };
  assert.throws(() => freezeServingEvidence(fixture.evidence, fixture.candidate, invalid),
    /quality-plan.json.minimumNdcgLift: must be in \(0, 1\]/);
  assert.equal(fs.existsSync(path.join(fixture.evidence, "quality-plan.json")), false);
});
