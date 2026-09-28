import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import { projectAggregateGraph } from "../src/core/aggregate-projection.js";
import { buildExplorerGraph } from "../src/core/explorer-graph.js";
import { aggregateRecommendationGraphId } from "../src/core/graph-contract.js";
import { MODEL_OUTPUT_FILES, packageModelRelease } from "../src/package-model-release.js";
import { OUTPUT_FILES, packageDataRelease, PUBLIC_FIELDS,
  type PublicationReviewV1 } from "../src/package-data-release.js";
import { RELEASE_FILES, releaseSha256, writeReleaseManifest } from "../src/release-manifest.js";
import { verifyModelReleasePackage } from "../src/verify-model-release-package.js";
import type { CompactGraphDataV2 } from "../src/types.js";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const demo = path.join(repoRoot, "web/public/demo-data");
const encode = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;
const H = (character: string) => character.repeat(64);
const decision = "docs/decisions/0029-aggregate-only-graph.md";
const promotionDecision = "docs/decisions/0033-public-model-promotion-contract.md";
const refs = {
  publication: "https://example.test/invented-publication-approval",
  training: "https://example.test/invented-training-approval",
  deployment: "https://example.test/invented-deployment-approval",
  owner: "https://example.test/invented-model-approval",
  bootstrap: "https://example.test/invented-bootstrap-approval",
  bridge: "https://example.test/invented-bridge-review",
};

function tempRoot(t: TestContext): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-model-package-"));
  t.after(() => {
    const real = fs.realpathSync(root);
    if (!real.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(root, { recursive: true, force: true });
  });
  return root;
}

function write(directory: string, name: string, value: unknown) {
  const bytes = Buffer.from(encode(value));
  fs.writeFileSync(path.join(directory, name), bytes);
  return releaseSha256(bytes);
}

function setup(t: TestContext, synthetic = true) {
  const root = tempRoot(t);
  const baseDir = path.join(root, "base");
  const candidateDir = path.join(root, "candidate");
  const evidenceDir = path.join(root, "evidence");
  const outputDir = path.join(root, "package");
  for (const directory of [baseDir, candidateDir, evidenceDir]) fs.mkdirSync(directory);
  const graph = projectAggregateGraph(JSON.parse(fs.readFileSync(path.join(demo,
    RELEASE_FILES.neighborhood), "utf8")) as CompactGraphDataV2);
  const sourceName = synthetic ? "synthetic-fixture" : "invented-reviewed-source";
  if (!synthetic) {
    graph.dataset.source = sourceName;
    const { graphId: _graphId, ...withoutId } = graph;
    graph.graphId = aggregateRecommendationGraphId(withoutId);
  }
  write(baseDir, RELEASE_FILES.neighborhood, graph);
  write(baseDir, RELEASE_FILES.explorer, buildExplorerGraph(graph, 5, 0));
  fs.copyFileSync(path.join(demo, RELEASE_FILES.catalog), path.join(baseDir, RELEASE_FILES.catalog));
  const baseManifest = writeReleaseManifest(baseDir, "data-vinvented-base", undefined,
    synthetic, !synthetic);
  for (const name of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer, RELEASE_FILES.catalog]) {
    fs.copyFileSync(path.join(baseDir, name), path.join(candidateDir, name));
  }
  const archive = Buffer.from("invented numeric archive; no user data");
  fs.writeFileSync(path.join(evidenceDir, "model.npz"), archive);
  const numericArchiveSha256 = releaseSha256(archive);
  const model = JSON.parse(fs.readFileSync(path.join(demo, RELEASE_FILES.model), "utf8"));
  model.sourceModelSha256 = numericArchiveSha256;
  const modelSha256 = write(candidateDir, RELEASE_FILES.model, model);
  const manifest = writeReleaseManifest(candidateDir, "data-vinvented-model", baseDir);
  const rawContentSha256 = H("a");
  const selectionSha256 = H("b");
  const selectionFileSha256 = write(evidenceDir, "selection.json", {
    format: "split-first-selection-v1", rawContentSha256, selectionSha256,
    selectedCandidate: { id: "invented-mf" },
  });
  const finalReportSha256 = write(evidenceDir, "final-report.json", {
    format: "split-first-final-test-v1", selectionSha256,
    selectedCandidateId: "invented-mf", status: synthetic
      ? "single synthetic warm-user report; no release claim"
      : "reviewed permitted serving-path report",
    test: { ndcgAtK: 0.4 },
  });
  write(evidenceDir, "final-report.sha256.json", {
    format: "split-first-final-test-digest-v1", selectionSha256,
    reportSha256: finalReportSha256,
  });
  write(evidenceDir, "selection.test-used.json", {
    format: "split-first-test-used-v1", selectionSha256,
  });
  const refitRecordSha256 = write(evidenceDir, "refit-record.json", {
    format: "split-first-final-refit-v1", releaseStatus: "unapproved experiment artifact",
    fitMembership: "train-plus-validation", selectionSha256, finalReportSha256,
    selectedCandidateId: "invented-mf", rawContentSha256,
    candidateSpecSha256: H("c"), splitIdentitySha256: H("d"), metadataSha256: H("e"),
    originalTrainSha256: H("f"), refitTrainSha256: H("1"),
    refitFitSha256: H("2"), refitModelSha256: H("3"),
    numericArchiveSha256, webModelSha256: modelSha256,
    trainRows: 7, validationRows: 3, testRowsExcluded: 3, refitRows: 10,
    evaluation: "none; this record contains no held-out quality metric",
  });
  const datasetBridgeSha256 = write(evidenceDir, "dataset-bridge.json", {
    format: "model-dataset-bridge-v1", sourceName, rawContentSha256,
    graphDatasetSha256: graph.dataset.sha256, decisionRef: promotionDecision,
    reviewRef: refs.bridge,
  });
  const qualityPolicySha256 = write(evidenceDir, "quality-policy.json", {
    format: "model-promotion-quality-policy-v1", decisionRef: promotionDecision,
    baselineName: "invented genre baseline", topK: 10, minimumEligibleUsers: 4,
    minimumPositiveLabels: 8, minimumModelCoverage: 0.9,
    minimumNdcgLift: 0.05, maximumP95LatencyMs: 100,
  });
  const servingReportSha256 = write(evidenceDir, "serving-report.json", {
    format: "model-serving-evaluation-v1", policySha256: qualityPolicySha256,
    tag: manifest.tag, bundleId: manifest.bundleId,
    rawContentSha256, graphDatasetSha256: graph.dataset.sha256,
    selectionSha256, finalReportSha256, topK: 10, eligibleUsers: 4, positiveLabels: 8,
    baselineName: "invented genre baseline",
    baseline: { ndcgAtK: 0.25, coverage: 0.8, p95LatencyMs: 40 },
    model: { ndcgAtK: 0.4, coverage: 1, p95LatencyMs: 80 },
  });
  const review = {
    format: "model-promotion-review-v2", promotionId: "invented-promotion-1",
    tag: manifest.tag, bundleId: manifest.bundleId,
    manifestSha256: releaseSha256(fs.readFileSync(path.join(candidateDir, RELEASE_FILES.manifest))),
    baseTag: baseManifest.tag, baseBundleId: baseManifest.bundleId,
    baseManifestSha256: releaseSha256(fs.readFileSync(path.join(baseDir, RELEASE_FILES.manifest))),
    sourceName, sourceDecisionRef: decision, datasetBridgeDecisionRef: promotionDecision,
    graphDatasetSha256: graph.dataset.sha256, rawContentSha256, modelSha256,
    numericArchiveSha256, selectionFileSha256, selectionSha256, finalReportSha256,
    refitRecordSha256, datasetBridgeSha256, qualityPolicySha256, servingReportSha256,
    minimumModelCoverage: 0.9,
    owner: "Invented Reviewer", ownerApprovalRef: refs.owner,
    trainingApprovalRef: refs.training, publicationApprovalRef: refs.publication,
    deploymentApprovalRef: refs.deployment,
  };
  write(evidenceDir, "model-promotion-review.json", review);
  return { root, baseDir, candidateDir, evidenceDir, outputDir,
    graph, baseManifest, manifest, review, sourceName };
}

function updateReview(fixture: ReturnType<typeof setup>, update: Record<string, unknown>): void {
  write(fixture.evidenceDir, "model-promotion-review.json", { ...fixture.review, ...update });
}

test("invented v3 model package copies only item assets and records private hashes", (t) => {
  const fixture = setup(t);
  const audit = packageModelRelease({ ...fixture, syntheticFixture: true });
  assert.equal(audit.status, "synthetic-only");
  assert.deepEqual(fs.readdirSync(fixture.outputDir).sort(), [...MODEL_OUTPUT_FILES].sort());
  assert.equal(audit.modelCoverage, 1);
  assert.equal(audit.base.bundleId, fixture.baseManifest.bundleId);
  assert.equal(audit.evidence.finalReportSha256, fixture.review.finalReportSha256);
  for (const name of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer, RELEASE_FILES.catalog,
    RELEASE_FILES.model, RELEASE_FILES.manifest]) {
    assert.deepEqual(fs.readFileSync(path.join(fixture.candidateDir, name)),
      fs.readFileSync(path.join(fixture.outputDir, name)));
  }
  for (const privateName of ["model.npz", "selection.json", "final-report.json",
    "refit-record.json", "dataset-bridge.json", "serving-report.json"]) {
    assert.equal(fs.existsSync(path.join(fixture.outputDir, privateName)), false);
  }
  assert.equal(fs.readdirSync(fixture.root).some((name) => name.startsWith(".model-package-")), false);
  assert.throws(() => packageModelRelease({ ...fixture, syntheticFixture: true }),
    /outputDir.*unused path/);
});

test("extra public/private files, changed data bytes, and user factors refuse packaging", (t) => {
  const extra = setup(t);
  fs.writeFileSync(path.join(extra.candidateDir, "ratings.sqlite"), "invented");
  assert.throws(() => packageModelRelease({ ...extra, syntheticFixture: true }),
    /candidate.*exactly/);
  const privateExtra = setup(t);
  fs.writeFileSync(path.join(privateExtra.evidenceDir, "usernames.txt"), "invented");
  assert.throws(() => packageModelRelease({ ...privateExtra, syntheticFixture: true }),
    /evidence.*exactly/);
  const changed = setup(t);
  fs.appendFileSync(path.join(changed.candidateDir, RELEASE_FILES.neighborhood), " ");
  assert.throws(() => packageModelRelease({ ...changed, syntheticFixture: true }),
    /graph.compact.json|release-manifest.json/);
  const factors = setup(t);
  const modelPath = path.join(factors.candidateDir, RELEASE_FILES.model);
  const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));
  model.userFactors = [[1, 0]];
  write(factors.candidateDir, RELEASE_FILES.model, model);
  assert.throws(() => packageModelRelease({ ...factors, syntheticFixture: true }),
    /model-mf-web.compact.json: root.userFactors is unsupported/);
  const wrongSource = setup(t, false);
  const bridge = JSON.parse(fs.readFileSync(path.join(wrongSource.evidenceDir,
    "dataset-bridge.json"), "utf8"));
  bridge.sourceName = "invented-other-source";
  updateReview(wrongSource, { sourceName: bridge.sourceName,
    datasetBridgeSha256: write(wrongSource.evidenceDir, "dataset-bridge.json", bridge) });
  assert.throws(() => packageModelRelease(wrongSource), /review.sourceName/);
});

test("private refit, final report, archive, and quality drift refuse packaging", (t) => {
  const report = setup(t);
  fs.appendFileSync(path.join(report.evidenceDir, "final-report.json"), " ");
  assert.throws(() => packageModelRelease({ ...report, syntheticFixture: true }),
    /review.finalReportSha256/);
  const archive = setup(t);
  fs.appendFileSync(path.join(archive.evidenceDir, "model.npz"), " ");
  assert.throws(() => packageModelRelease({ ...archive, syntheticFixture: true }), /model.npz/);
  const lowQuality = setup(t);
  const reportPath = path.join(lowQuality.evidenceDir, "serving-report.json");
  const serving = JSON.parse(fs.readFileSync(reportPath, "utf8"));
  serving.model.ndcgAtK = 0.24;
  const servingReportSha256 = write(lowQuality.evidenceDir, "serving-report.json", serving);
  updateReview(lowQuality, { servingReportSha256 });
  assert.throws(() => packageModelRelease({ ...lowQuality, syntheticFixture: true }),
    /serving-report.*floors/);
  const badBridge = setup(t);
  const bridgePath = path.join(badBridge.evidenceDir, "dataset-bridge.json");
  const bridge = JSON.parse(fs.readFileSync(bridgePath, "utf8"));
  bridge.graphDatasetSha256 = H("e");
  updateReview(badBridge, { datasetBridgeSha256: write(badBridge.evidenceDir,
    "dataset-bridge.json", bridge) });
  assert.throws(() => packageModelRelease({ ...badBridge, syntheticFixture: true }),
    /dataset-bridge.graphDatasetSha256/);
  const changedPolicy = setup(t);
  const policyPath = path.join(changedPolicy.evidenceDir, "quality-policy.json");
  const policy = JSON.parse(fs.readFileSync(policyPath, "utf8"));
  policy.baselineName = "invented weaker baseline";
  updateReview(changedPolicy, { qualityPolicySha256: write(changedPolicy.evidenceDir,
    "quality-policy.json", policy) });
  assert.throws(() => packageModelRelease({ ...changedPolicy, syntheticFixture: true }),
    /serving-report.policySha256/);
});

test("failed staging leaves no output and the empty committed registry rejects synthetic packages", (t) => {
  const fixture = setup(t);
  assert.throws(() => packageModelRelease({ ...fixture, syntheticFixture: true,
    beforeActivate: () => { throw new Error("invented interruption"); } }), /invented interruption/);
  assert.equal(fs.existsSync(fixture.outputDir), false);
  assert.equal(fs.readdirSync(fixture.root).some((name) => name.startsWith(".model-package-")), false);
  packageModelRelease({ ...fixture, syntheticFixture: true });
  const result = spawnSync(process.execPath, ["--import", "tsx",
    path.join(repoRoot, "pipeline/src/verify-model-release-package.ts"),
    "--package", fixture.outputDir, "--base", fixture.baseDir,
    "--evidence", fixture.evidenceDir, "--tag", fixture.manifest.tag,
    "--base-tag", fixture.baseManifest.tag, "--run-id", "123",
    "--artifact-name", "invented-model"], { cwd: repoRoot, encoding: "utf8" });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /syntheticFixture|synthetic|unapproved/);
});

test("an invented reviewed package requires exact model, provider, and data-base approvals", (t) => {
  const fixture = setup(t, false);
  const baseReview: PublicationReviewV1 = {
    format: "publication-review-v1", tag: fixture.baseManifest.tag,
    bundleId: fixture.baseManifest.bundleId,
    manifestSha256: fixture.review.baseManifestSha256,
    source: { name: fixture.sourceName, datasetSha256: fixture.graph.dataset.sha256,
      derivation: "Invented aggregate projection", decisionRef: decision },
    redistribution: { status: "reviewed-allowed", basis: "Invented test review",
      approvalRef: refs.publication, owner: "Invented Reviewer",
      allowedFields: [...PUBLIC_FIELDS], attribution: "Invented",
      deletionCorrection: "Replace invented test bundle" },
    changes: { previousTag: null, summary: "Invented reviewed base" },
    quality: { checks: [
      { id: "schema", evidenceRef: "invented schema test" },
      { id: "privacy", evidenceRef: "invented privacy test" },
      { id: "coverage", evidenceRef: "invented coverage test" },
    ] },
  };
  const basePackage = path.join(fixture.root, "base-package");
  packageDataRelease({ candidateDir: fixture.baseDir, outputDir: basePackage,
    review: baseReview,
    approval: { scope: "publication", approved: true, approvalRef: refs.publication,
      decisionRef: decision, owner: "Invented Reviewer", sources: [fixture.sourceName] },
    bootstrapApproval: { scope: "first-real-bundle", approved: true,
      tag: fixture.baseManifest.tag, bundleId: fixture.baseManifest.bundleId,
      manifestSha256: fixture.review.baseManifestSha256,
      decisionRef: "docs/decisions/0032-reviewed-first-release-bootstrap.md",
      approvalRef: refs.bootstrap, owner: "Invented Reviewer" } });
  assert.deepEqual(fs.readdirSync(basePackage).sort(), [...OUTPUT_FILES].sort());
  const audit = packageModelRelease({ ...fixture, baseDir: basePackage });
  assert.equal(audit.status, "pending-approval");
  const providerApprovals = { schemaVersion: 1, approvals: Object.fromEntries(
    (["training", "publication", "deployment"] as const).map((use) => [use, {
      approved: true, sources: [fixture.sourceName], sourceBasis: "Invented source",
      use: "Invented test permission", owner: "Invented Reviewer", approvedAt: "2026-09-28",
      decisionRef: decision, approvalRef: refs[use],
    }])) };
  const publicationApprovals = { schemaVersion: 1, packages: [{
    tag: fixture.baseManifest.tag, bundleId: fixture.baseManifest.bundleId,
    manifestSha256: fixture.review.baseManifestSha256,
    auditSha256: releaseSha256(fs.readFileSync(path.join(basePackage, "publication-audit.json"))),
    sourceRunId: 122, artifactName: "invented-data", previousTag: null,
    decisionRef: decision, approvalRef: refs.publication, owner: "Invented Reviewer",
    bootstrap: { decisionRef: "docs/decisions/0032-reviewed-first-release-bootstrap.md",
      approvalRef: refs.bootstrap, owner: "Invented Reviewer" },
  }] };
  const modelApprovals = { schemaVersion: 1, promotions: [{
    promotionId: audit.promotionId, tag: audit.tag, bundleId: audit.bundleId,
    manifestSha256: audit.manifestSha256,
    auditSha256: releaseSha256(fs.readFileSync(path.join(fixture.outputDir,
      "model-promotion-audit.json"))),
    baseTag: audit.base.tag, baseBundleId: audit.base.bundleId,
    baseManifestSha256: audit.base.manifestSha256,
    sourceRunId: 123, artifactName: "invented-model", decisionRef: promotionDecision,
    datasetBridgeSha256: audit.evidence.datasetBridgeSha256,
    datasetBridgeReviewRef: refs.bridge,
    qualityPolicySha256: audit.evidence.qualityPolicySha256,
    servingReportSha256: audit.evidence.servingReportSha256,
    owner: audit.approvals.owner, ownerApprovalRef: refs.owner,
    trainingApprovalRef: refs.training, publicationApprovalRef: refs.publication,
    deploymentApprovalRef: refs.deployment,
  }] };
  const options = { packageDir: fixture.outputDir, baseDir: basePackage,
    evidenceDir: fixture.evidenceDir,
    dispatch: { tag: audit.tag, baseTag: audit.base.tag, sourceRunId: 123,
      artifactName: "invented-model" },
    providerApprovals, publicationApprovals, modelApprovals };
  assert.deepEqual(verifyModelReleasePackage(options), audit);
  assert.throws(() => verifyModelReleasePackage({ ...options,
    modelApprovals: { schemaVersion: 1, promotions: [] } }), /exact committed owner review/);
  assert.throws(() => verifyModelReleasePackage({ ...options,
    providerApprovals: { schemaVersion: 1, approvals: {} } }), /providerApprovals.*training/);
  const stale = structuredClone(modelApprovals);
  stale.promotions[0].auditSha256 = H("f");
  assert.throws(() => verifyModelReleasePackage({ ...options, modelApprovals: stale }),
    /no exact independently reviewed model package/);
  fs.appendFileSync(path.join(fixture.outputDir, "model-promotion-audit.json"), " ");
  assert.throws(() => verifyModelReleasePackage(options),
    /model-promotion-audit.json.*differs/);
});
