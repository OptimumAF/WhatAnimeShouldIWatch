import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import {
  compareRows, evaluateParity, parseParitySpec, runParity,
} from "../bench/model-parity.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
function fixture(name: string): any {
  return JSON.parse(readFileSync(new URL("../../fixtures/" + name, import.meta.url), "utf8"));
}
const spec = parseParitySpec(fixture("synthetic-model-parity-spec.json"));
const input = fixture("synthetic-model-parity-input.json");
const reference = JSON.parse(execFileSync("python", ["ml/model_parity_reference.py"],
  { cwd: ROOT, encoding: "utf8" }));

test("same exported numeric model agrees on all raw/eligible IDs, scores, exclusions, and top K", () => {
  const report = runParity();
  assert.equal(report.maximumScoreDelta, 0);
  assert.equal(report.scoreAbsoluteTolerance, 0.00001);
  assert.deepEqual(report.cases.map((item) => item.topKIds),
    [[302, 307], [307, 305], []]);
  assert.deepEqual(report.cases[0].sourceExcludedIds, [301, 303, 304]);
  assert.deepEqual(report.cases[0].policyExcludedIds, [305, 306]);
  assert.deepEqual(report.cases[1].eligibleCandidateIds, [307, 305, 303]);
  assert.equal(report.cases[2].rawCandidateIds.length, 0);
  assert.equal(report.archiveSha256, reference.model.sourceModelSha256);
});

test("score drift, candidate drift, exclusion drift, and malformed model fail closed", () => {
  const score = structuredClone(reference);
  score.cases[0].eligible[0].score += 0.01;
  assert.throws(() => evaluateParity(spec, input, score), /eligible score\[0\]/);
  const candidate = structuredClone(reference);
  candidate.cases[0].raw[0].animeId = 999;
  assert.throws(() => evaluateParity(spec, input, candidate), /raw candidate IDs\/order/);
  const excluded = structuredClone(reference);
  excluded.cases[0].policyExcludedIds = [];
  assert.throws(() => evaluateParity(spec, input, excluded), /exclusions\/top K/);
  const shape = structuredClone(reference);
  shape.model.embeddings[0].pop();
  assert.throws(() => evaluateParity(spec, input, shape),
    /synthetic parity model: embeddings\[0\].*dimension/);
  const digest = structuredClone(reference);
  digest.model.sourceModelSha256 = "0".repeat(64);
  assert.throws(() => evaluateParity(spec, input, digest), /source archive hash/);
});

test("declared score tolerance and input hash are enforced", () => {
  assert.ok(Math.abs(compareRows([{ animeId: 1, score: 1.000005 }],
    [{ animeId: 1, score: 1 }], 0.00001, "hand case") - 0.000005) < 1e-12);
  assert.throws(() => compareRows([{ animeId: 1, score: 1.01 }],
    [{ animeId: 1, score: 1 }], 0.00001, "hand case"), /hand case score\[0\]/);
  const changedSpec = structuredClone(spec);
  changedSpec.inputSha256 = "0".repeat(64);
  assert.throws(() => parseParitySpec(changedSpec), /specification protocol\/hash/);
});
