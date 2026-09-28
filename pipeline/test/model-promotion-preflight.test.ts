import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { GRAPH_SEMANTICS, recommendationGraphId } from "../src/core/graph-contract.js";
import { verifyModelPromotion } from "../src/model-promotion-preflight.js";
import type { CompactGraphDataV2 } from "../src/types.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const H = (digit: string) => digit.repeat(64);
const sha = (bytes: Buffer | string) => crypto.createHash("sha256").update(bytes).digest("hex");
const write = (directory: string, name: string, value: unknown) => {
  const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
  fs.writeFileSync(path.join(directory, name), bytes);
  return sha(bytes);
};

function inventedGraph(): CompactGraphDataV2 {
  const graphWithoutId = {
    format: "graph-compact-v2" as const,
    role: "recommendation" as const,
    dataset: { sha256: H("a"), scope: "anonymized-ratings-content-v1" as const,
      source: "invented-fixture" },
    semantics: GRAPH_SEMANTICS,
    config: { ratingSelectionPolicy: "all-ratings" as const, seed: 0,
      maxRatingsPerUser: 0, maxAnimeAnimeEdges: 0, maxPairVisits: 100,
      maxPairCandidates: 100, minPairSupport: 1, maxNeighborsPerAnime: 0 },
    truncation: { inputRatings: 0, selectedRatings: 0, ratingsSkipped: 0,
      potentialPairVisits: 0, pairVisits: 0, pairVisitsSkipped: 0,
      candidatePairs: 0, eligiblePairs: 0, selectedPairs: 0,
      excludedBySupport: 0, excludedByNeighborLimit: 0, excludedByOutputLimit: 0 },
    generatedAt: "2026-09-28T00:00:00.000Z",
    userIds: [], anime: [[101, "Invented Alpha"]] as [number, string][],
    ua: [], aa: [], userCount: 0, animeCount: 1, nodeCount: 1, edgeCount: 0,
  };
  return { ...graphWithoutId, graphId: recommendationGraphId(graphWithoutId) };
}

function setup() {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "invented-promotion-"));
  const candidateDir = path.join(temporary, "candidate");
  const rollbackDir = path.join(temporary, "rollback");
  const approvalsPath = path.join(temporary, "approvals.json");
  fs.mkdirSync(candidateDir);
  fs.mkdirSync(rollbackDir);
  const graph = inventedGraph();
  const catalog = { format: "demo-catalog-v1", generatedAt: graph.generatedAt,
    anime: [{ animeId: 101, title: "Invented Alpha", year: 2026, score: 7,
      genres: ["Adventure"], studios: [], synopsis: "Invented.", imageUrl: "",
      season: null }] };
  const model = { format: "model-mf-compact-v1", generatedAt: graph.generatedAt,
    sourceModelSha256: H("b"), globalMean: 0, factors: 2,
    animeIds: [101], titles: ["Invented Alpha"], biases: [0], embeddings: [[1, 0]] };
  const modelSha256 = write(candidateDir, "model-mf-web.compact.json", model);
  const graphSha256 = write(candidateDir, "graph.compact.json", graph);
  const catalogSha256 = write(candidateDir, "catalog.json", catalog);
  const previousFiles = {
    "graph.compact.json": write(rollbackDir, "graph.compact.json", graph),
    "catalog.json": write(rollbackDir, "catalog.json", catalog),
    "model-mf-web.compact.json": write(rollbackDir, "model-mf-web.compact.json",
      { ...model, biases: [0.1] }),
  };
  const previousManifestSha256 = write(rollbackDir, "rollback-manifest.json", {
    format: "model-rollback-bundle-v1", tag: "data-v-invented-previous",
    files: previousFiles,
  });
  const refit = { format: "split-first-final-refit-v1", fitMembership: "train-plus-validation",
    releaseStatus: "unapproved experiment artifact", trainRows: 7, validationRows: 3,
    testRowsExcluded: 3, refitRows: 10, webModelSha256: modelSha256,
    numericArchiveSha256: H("b"), rawContentSha256: H("c"),
    finalReportSha256: H("d"), selectedCandidateId: "invented-mf" };
  const refitRecordSha256 = write(candidateDir, "refit-record.json", refit);
  const ownerApprovalRef = "https://example.test/invented-owner-approval";
  const publicationApprovalRef = "https://example.test/invented-publication-approval";
  const deploymentApprovalRef = "https://example.test/invented-deployment-approval";
  const record = { format: "model-promotion-v1", promotionId: "invented-1",
    targetTag: "data-v-invented-next", previousTag: "data-v-invented-previous",
    modelSha256, graphSha256, catalogSha256, graphId: graph.graphId,
    graphDatasetSha256: graph.dataset.sha256, rawTrainingSnapshotSha256: H("c"),
    refitRecordSha256, finalReportSha256: H("d"), selectedCandidateId: "invented-mf",
    previousManifestSha256,
    qualityDecisionRef: "docs/decisions/0024-experiment-refit-promotion-boundary.md",
    ownerApprovalRef, publicationApprovalRef, deploymentApprovalRef,
    minimumModelCoverage: 1 };
  const recordSha256 = write(candidateDir, "model-promotion.json", record);
  const approval = { format: "model-promotion-approvals-v1", approvals: [{
    promotionId: record.promotionId, targetTag: record.targetTag, recordSha256,
    approved: true, owner: "Invented Reviewer", approvedAt: "2026-09-28",
    qualityGatePassed: true, ownerApprovalRef, publicationApprovalRef,
    deploymentApprovalRef,
  }] };
  write(temporary, "approvals.json", approval);
  return { temporary, candidateDir, rollbackDir, approvalsPath, repoRoot,
    record, approval };
}

test("invented approved compatible candidate has an intact rollback bundle", () => {
  const fixture = setup();
  try {
    const result = verifyModelPromotion(fixture);
    assert.equal(result.promotionId, "invented-1");
    assert.equal(result.modelCoverage, 1);
    assert.equal(result.previousTag, "data-v-invented-previous");
  } finally { fs.rmSync(fixture.temporary, { recursive: true, force: true }); }
});

test("unrecorded and stale approval refuse a candidate", () => {
  const fixture = setup();
  try {
    write(fixture.temporary, "approvals.json", {
      format: "model-promotion-approvals-v1", approvals: [],
    });
    assert.throws(() => verifyModelPromotion(fixture), /exactly one committed owner review/);
    const stale = structuredClone(fixture.approval);
    stale.approvals[0].recordSha256 = H("e");
    write(fixture.temporary, "approvals.json", stale);
    assert.throws(() => verifyModelPromotion(fixture), /recordSha256/);
  } finally { fs.rmSync(fixture.temporary, { recursive: true, force: true }); }
});

test("the executable preflight uses the committed empty review manifest", () => {
  const fixture = setup();
  try {
    const result = spawnSync(process.execPath, [
      "--import", "tsx", path.join(repoRoot, "pipeline/src/check-model-promotion.ts"),
      "--candidate-dir", fixture.candidateDir,
      "--rollback-dir", fixture.rollbackDir,
    ], { cwd: repoRoot, encoding: "utf8" });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /exactly one committed owner review/);
  } finally { fs.rmSync(fixture.temporary, { recursive: true, force: true }); }
});

test("model or graph corruption and catalog title drift refuse promotion", () => {
  const fixture = setup();
  try {
    fs.appendFileSync(path.join(fixture.candidateDir, "model-mf-web.compact.json"), " ");
    assert.throws(() => verifyModelPromotion(fixture), /record.modelSha256/);
    const original = setup();
    try {
      const catalogPath = path.join(original.candidateDir, "catalog.json");
      const catalog = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
      catalog.anime[0].title = "Invented Other";
      write(original.candidateDir, "catalog.json", catalog);
      assert.throws(() => verifyModelPromotion(original), /catalog.anime\[101\]/);
    } finally { fs.rmSync(original.temporary, { recursive: true, force: true }); }
    const graphCase = setup();
    try {
      const graphPath = path.join(graphCase.candidateDir, "graph.compact.json");
      const graph = JSON.parse(fs.readFileSync(graphPath, "utf8"));
      graph.graphId = H("f");
      write(graphCase.candidateDir, "graph.compact.json", graph);
      assert.throws(() => verifyModelPromotion(graphCase), /candidate.graph.graphId/);
    } finally { fs.rmSync(graphCase.temporary, { recursive: true, force: true }); }
  } finally { fs.rmSync(fixture.temporary, { recursive: true, force: true }); }
});

test("missing or changed rollback bytes and mutable tags refuse promotion", () => {
  const fixture = setup();
  try {
    fs.appendFileSync(path.join(fixture.rollbackDir, "model-mf-web.compact.json"), " ");
    assert.throws(() => verifyModelPromotion(fixture), /rollback-manifest.files.model-mf-web/);
    const tagCase = setup();
    try {
      const record = { ...tagCase.record, previousTag: "data-latest" };
      write(tagCase.candidateDir, "model-promotion.json", record);
      assert.throws(() => verifyModelPromotion(tagCase), /immutable data-v tags/);
    } finally { fs.rmSync(tagCase.temporary, { recursive: true, force: true }); }
  } finally { fs.rmSync(fixture.temporary, { recursive: true, force: true }); }
});

test("refit provenance or quality approval failure refuses promotion", () => {
  const fixture = setup();
  try {
    const refitPath = path.join(fixture.candidateDir, "refit-record.json");
    const refit = JSON.parse(fs.readFileSync(refitPath, "utf8"));
    refit.testRowsExcluded = 0;
    write(fixture.candidateDir, "refit-record.json", refit);
    assert.throws(() => verifyModelPromotion(fixture), /refitRecordSha256/);
    const approvalCase = setup();
    try {
      const denied = structuredClone(approvalCase.approval);
      denied.approvals[0].qualityGatePassed = false;
      write(approvalCase.temporary, "approvals.json", denied);
      assert.throws(() => verifyModelPromotion(approvalCase), /qualityGatePassed/);
    } finally { fs.rmSync(approvalCase.temporary, { recursive: true, force: true }); }
  } finally { fs.rmSync(fixture.temporary, { recursive: true, force: true }); }
});
