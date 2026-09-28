import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import { projectAggregateGraph } from "../src/core/aggregate-projection.js";
import { buildExplorerGraph } from "../src/core/explorer-graph.js";
import { graphFromPrepared, prepareGraphBridge, verifyGraphDatasetBridge } from
  "../src/core/split-graph-bridge.js";
import { freezeServingEvidence } from "../src/core/model-serving-freeze.js";
import { MODEL_OUTPUT_FILES, packageModelRelease } from "../src/package-model-release.js";
import { OUTPUT_FILES, packageDataRelease, PUBLIC_FIELDS,
  type PublicationReviewV1 } from "../src/package-data-release.js";
import { RELEASE_FILES, releaseSha256, writeReleaseManifest } from "../src/release-manifest.js";
import { verifyModelReleasePackage } from "../src/verify-model-release-package.js";
import { verifyModelPublicRelease } from "../src/verify-model-release-public.js";
import { generateServingReport, verifyServingReport } from
  "../../web/bench/model-promotion-serving-evaluation.ts";
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
  freeze: "https://example.test/invented-prefit-freeze-approval",
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

function setup(t: TestContext, synthetic = true, precommitPlan = false) {
  const root = tempRoot(t);
  const baseDir = path.join(root, "base");
  const candidateDir = path.join(root, "candidate");
  const evidenceDir = path.join(root, "evidence");
  const sourceDir = path.join(root, "source");
  const outputDir = path.join(root, "package");
  for (const directory of [baseDir, candidateDir, evidenceDir]) fs.mkdirSync(directory);
  const demoGraph = JSON.parse(fs.readFileSync(path.join(demo,
    RELEASE_FILES.neighborhood), "utf8")) as CompactGraphDataV2;
  const sourceName = synthetic ? "synthetic-fixture" : "invented-reviewed-source";
  if (!synthetic) {
    fs.mkdirSync(sourceDir);
    for (const [source, target] of [
      ["fixtures/synthetic-split-input.json", "raw-ratings.json"],
      ["fixtures/synthetic-anime-metadata.json", "anime-metadata.json"],
    ]) fs.copyFileSync(path.join(repoRoot, source), path.join(sourceDir, target));
    const rawPath = path.join(sourceDir, "raw-ratings.json");
    const raw = JSON.parse(fs.readFileSync(rawPath, "utf8"));
    raw.interactions.push({ userId: "invented-a", animeId: 106, rawScore: 7 });
    fs.writeFileSync(rawPath, encode(raw));
    const split = spawnSync("python", [path.join(repoRoot, "ml/raw_interaction_split.py"),
      "--input", path.join(sourceDir, "raw-ratings.json"),
      "--out", path.join(sourceDir, "split-manifest.json"),
      "--policy", "seeded", "--seed", "8"], { cwd: repoRoot, encoding: "utf8" });
    assert.equal(split.status, 0, split.stderr);
  }
  const prepared = synthetic ? null : prepareGraphBridge(path.join(sourceDir, "raw-ratings.json"),
    path.join(sourceDir, "split-manifest.json"), path.join(sourceDir, "anime-metadata.json"));
  const graph = synthetic ? projectAggregateGraph(demoGraph)
    : graphFromPrepared(prepared!, sourceName, demoGraph.generatedAt, demoGraph.config);
  write(baseDir, RELEASE_FILES.neighborhood, graph);
  write(baseDir, RELEASE_FILES.explorer, buildExplorerGraph(graph, 5, 0));
  if (synthetic) {
    fs.copyFileSync(path.join(demo, RELEASE_FILES.catalog), path.join(baseDir, RELEASE_FILES.catalog));
  } else {
    write(baseDir, RELEASE_FILES.catalog, { format: "anime-catalog-v1",
      datasetSha256: graph.dataset.sha256,
      anime: graph.anime });
  }
  const baseManifest = writeReleaseManifest(baseDir, "data-vinvented-base", undefined,
    synthetic, !synthetic);
  for (const name of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer, RELEASE_FILES.catalog]) {
    fs.copyFileSync(path.join(baseDir, name), path.join(candidateDir, name));
  }
  const rawContentSha256 = prepared?.rawContentSha256 ?? H("a");
  const caseFor = (userId: string, labelId: number) => ({
    userId, observed: [{ animeId: 101, rawScore: 10 }],
    labels: [{ animeId: labelId, rawScore: 9 },
      ...(labelId === 103 ? [] : [{ animeId: 103, rawScore: 3 }])],
    historySeen: [], exclude: [], includeOnly: [],
    filters: { genre: "", minYear: null, maxYear: null, minScore: null },
  });
  const cohortMetadata = JSON.parse(fs.readFileSync(path.join(demo, "catalog.json"), "utf8"));
  if (!synthetic) {
    const titles = new Map((JSON.parse(fs.readFileSync(path.join(sourceDir,
      "anime-metadata.json"), "utf8"))).anime.map(
        ({ animeId, title }: { animeId: number; title: string }) => [animeId, title]));
    for (const item of cohortMetadata.anime) item.title = titles.get(item.animeId) ?? item.title;
    cohortMetadata.anime.push({ animeId: 109, title: titles.get(109), year: 2022,
      score: 7.1, genres: ["Fantasy"], studios: [], synopsis: "Invented title.",
      imageUrl: "", season: null });
  }
  const finalPayload = { format: "model-promotion-final-v1", users:
    Array.from({ length: 4 }, (_, index) =>
      caseFor(`invented-promo-final-${index + 1}`, synthetic ? 106 : 105)) };
  const servingFinalSha256 = write(evidenceDir, "serving-final.json", finalPayload);
  const trainingUsers = synthetic
    ? ["fixture-overlap-a", "fixture-overlap-b", "fixture-opposite", "fixture-equal",
      "fixture-sparse", "fixture-empty", "fixture-duplicate-unknown"]
    : [...new Set(prepared!.rows.map((row) => row.userId))].sort();
  const cohort = { format: "model-promotion-cohort-v2", seed: 42, sourceName,
    datasetSha256: graph.dataset.sha256, trainingUsers, metadata: cohortMetadata,
    baselineValidation: Array.from({ length: 4 }, (_, index) =>
      caseFor(`invented-promo-validation-${index + 1}`, 102)),
    finalSha256: servingFinalSha256 };
  const servingCohortSha256 = write(evidenceDir, "serving-cohort.json", cohort);
  const plan = { format: "model-serving-quality-plan-v1", sourceName,
    rawContentSha256, graphDatasetSha256: graph.dataset.sha256,
    cohortSha256: servingCohortSha256, finalSha256: servingFinalSha256,
    baselineCandidates: ["graph", "genre", "coverage"],
    decisionRef: promotionDecision, seed: 42, suppliedCount: 1,
    positiveRawScoreMin: 7, topK: 5, minimumEligibleUsers: 4,
    minimumPositiveLabels: 4, minimumServingCoverage: 0.5,
    minimumNdcgLift: 0.05, maximumP95LatencyMs: 1000,
    latencyWarmups: 1, latencySamples: 7 };
  const { planSha256: qualityPlanSha256, freezeSha256: servingFreezeSha256 } =
    freezeServingEvidence(evidenceDir, candidateDir, plan);
  const planApproval = precommitPlan ? (() => {
    const planApprovals = { schemaVersion: 1, freezes: [{
      sourceName, graphDatasetSha256: graph.dataset.sha256,
      qualityPlanSha256, servingCohortSha256, servingFinalSha256,
      servingFreezeSha256, owner: "Invented Reviewer", approvalRef: refs.freeze,
      decisionRef: "docs/decisions/0035-serving-evaluation-freeze.md",
    }] };
    const approvalRepoDir = path.join(root, "approval-history");
    const approvalFiles = path.join(approvalRepoDir, "docs", "approvals");
    fs.mkdirSync(approvalFiles, { recursive: true });
    const git = (...args: string[]) => {
      const result = spawnSync("git", args, { cwd: approvalRepoDir, encoding: "utf8" });
      assert.equal(result.status, 0, result.stderr);
      return result.stdout.trim();
    };
    git("init", "-q");
    write(approvalFiles, "model-evaluation-plans.json", planApprovals);
    write(approvalFiles, "model-release-bundles.json", { schemaVersion: 1, promotions: [] });
    git("add", ".");
    git("-c", "user.name=Invented Reviewer", "-c", "user.email=invented@example.test",
      "commit", "-qm", "precommit invented plan before selecting an MF candidate");
    return { planApprovals, approvalRepoDir, approvalFiles, freezeRevision: git("rev-parse", "HEAD") };
  })() : undefined;
  const generated = spawnSync("python", [path.join(repoRoot, synthetic
    ? "ml/synthetic_promotion_archive.py" : "ml/synthetic_promotion_refit.py"),
    "--candidate-dir", candidateDir, "--evidence-dir", evidenceDir,
    ...(!synthetic ? ["--source-dir", sourceDir,
      "--dataset-sha256", graph.dataset.sha256,
      "--serving-freeze", path.join(evidenceDir, "serving-freeze.json")] : [])],
  { cwd: repoRoot, encoding: "utf8" });
  assert.equal(generated.status, 0, generated.stderr);
  const refitFromFit = synthetic ? null : JSON.parse(fs.readFileSync(path.join(evidenceDir,
    "refit-record.json"), "utf8"));
  const archiveEvidence = JSON.parse(generated.stdout);
  const { numericArchiveSha256, numericMetadataSha256, refitModelSha256 } =
    refitFromFit ?? archiveEvidence;
  const model = JSON.parse(fs.readFileSync(path.join(candidateDir, RELEASE_FILES.model), "utf8"));
  if (synthetic) fs.copyFileSync(path.join(candidateDir, RELEASE_FILES.model),
    path.join(evidenceDir, RELEASE_FILES.model));
  const modelSha256 = releaseSha256(fs.readFileSync(path.join(candidateDir, RELEASE_FILES.model)));
  const manifest = writeReleaseManifest(candidateDir, "data-vinvented-model", baseDir);
  const selectionSha256 = refitFromFit
    ? JSON.parse(fs.readFileSync(path.join(evidenceDir, "selection.json"), "utf8")).selectionSha256
    : H("b");
  const selectionFileSha256 = refitFromFit
    ? releaseSha256(fs.readFileSync(path.join(evidenceDir, "selection.json")))
    : write(evidenceDir, "selection.json", {
    format: "split-first-selection-v1", rawContentSha256, selectionSha256,
    servingFreezeSha256,
    selectedCandidate: { id: "invented-mf" },
  });
  const finalReportSha256 = refitFromFit
    ? releaseSha256(fs.readFileSync(path.join(evidenceDir, "final-report.json")))
    : write(evidenceDir, "final-report.json", {
    format: "split-first-final-test-v1", selectionSha256,
    selectedCandidateId: "invented-mf", status: "single synthetic warm-user report; no release claim",
    test: { ndcgAtK: 0.4 },
  });
  if (synthetic) {
    write(evidenceDir, "final-report.json.sha256.json", {
      format: "split-first-final-test-digest-v1", selectionSha256,
      reportSha256: finalReportSha256,
    });
    write(evidenceDir, "selection.json.test-used", {
      format: "split-first-test-used-v1", selectionSha256,
    });
  }
  const refitRecordSha256 = refitFromFit
    ? releaseSha256(fs.readFileSync(path.join(evidenceDir, "refit-record.json")))
    : write(evidenceDir, "refit-record.json", {
    format: "split-first-final-refit-v1", releaseStatus: "unapproved experiment artifact",
    fitMembership: "train-plus-validation", selectionSha256, finalReportSha256,
    selectedCandidateId: "invented-mf", rawContentSha256,
    candidateSpecSha256: H("c"), splitIdentitySha256: H("d"),
    metadataSha256: H("e"), originalTrainSha256: H("f"),
    refitTrainSha256: H("1"), refitFitSha256: H("2"), refitModelSha256,
    numericArchiveSha256, numericMetadataSha256, webModelSha256: modelSha256,
    trainRows: 7, validationRows: 3,
    testRowsExcluded: 3, refitRows: 10,
    evaluation: "none; this record contains no held-out quality metric",
  });
  const bridgeRecord = {
    format: synthetic ? "model-dataset-bridge-v1" : "model-dataset-bridge-v2",
    sourceName, rawContentSha256,
    graphDatasetSha256: graph.dataset.sha256, decisionRef: promotionDecision,
    reviewRef: refs.bridge,
    ...(!synthetic ? { verification: verifyGraphDatasetBridge(prepared!, graph, sourceName,
      JSON.parse(fs.readFileSync(path.join(evidenceDir, "refit-record.json"), "utf8"))) } : {}),
  };
  const datasetBridgeSha256 = write(evidenceDir, "dataset-bridge.json", bridgeRecord);
  const policy = {
    format: "model-promotion-quality-policy-v1", decisionRef: promotionDecision,
    candidateBundleId: manifest.bundleId, cohortSha256: servingCohortSha256,
    baselineName: synthetic ? "graph" : "coverage", seed: 42, suppliedCount: 1,
    positiveRawScoreMin: 7, topK: 5, minimumEligibleUsers: 4,
    minimumPositiveLabels: 4, minimumServingCoverage: 0.5,
    minimumNdcgLift: 0.05, maximumP95LatencyMs: 1000,
    latencyWarmups: 1, latencySamples: 7,
  };
  const qualityPolicySha256 = write(evidenceDir, "quality-policy.json", policy);
  const generatedServing = spawnSync(process.execPath, ["--import", "tsx",
    path.join(repoRoot, "web/bench/model-promotion-serving-evaluation.ts"),
    "--write-frozen", candidateDir, evidenceDir], { cwd: repoRoot, encoding: "utf8" });
  assert.equal(generatedServing.status, 0, generatedServing.stderr);
  const serving = JSON.parse(fs.readFileSync(path.join(evidenceDir, "serving-report.json"), "utf8"));
  const servingReportSha256 = releaseSha256(fs.readFileSync(path.join(evidenceDir,
    "serving-report.json")));
  const review = {
    format: "model-promotion-review-v3", promotionId: "invented-promotion-1",
    tag: manifest.tag, bundleId: manifest.bundleId,
    manifestSha256: releaseSha256(fs.readFileSync(path.join(candidateDir, RELEASE_FILES.manifest))),
    baseTag: baseManifest.tag, baseBundleId: baseManifest.bundleId,
    baseManifestSha256: releaseSha256(fs.readFileSync(path.join(baseDir, RELEASE_FILES.manifest))),
    sourceName, sourceDecisionRef: decision, datasetBridgeDecisionRef: promotionDecision,
    graphDatasetSha256: graph.dataset.sha256, rawContentSha256, modelSha256,
    numericArchiveSha256, numericMetadataSha256,
    selectionFileSha256, selectionSha256, finalReportSha256,
    refitRecordSha256, datasetBridgeSha256, qualityPlanSha256, qualityPolicySha256,
    servingCohortSha256, servingFinalSha256, servingFreezeSha256, servingReportSha256,
    minimumModelCoverage: 0.9,
    owner: "Invented Reviewer", ownerApprovalRef: refs.owner,
    freezeApprovalRef: refs.freeze,
    trainingApprovalRef: refs.training, publicationApprovalRef: refs.publication,
    deploymentApprovalRef: refs.deployment,
  };
  write(evidenceDir, "model-promotion-review.json", review);
  return { root, baseDir, candidateDir, evidenceDir,
    sourceDir: synthetic ? undefined : sourceDir, outputDir,
    graph, baseManifest, manifest, review, sourceName, serving, planApproval };
}

function updateReview(fixture: ReturnType<typeof setup>, update: Record<string, unknown>): void {
  write(fixture.evidenceDir, "model-promotion-review.json", { ...fixture.review, ...update });
}

function servingInputs(fixture: ReturnType<typeof setup>) {
  const read = (directory: string, name: string) => JSON.parse(fs.readFileSync(
    path.join(directory, name), "utf8"));
  return { graph: read(fixture.candidateDir, RELEASE_FILES.neighborhood),
    model: read(fixture.candidateDir, RELEASE_FILES.model),
    catalog: read(fixture.candidateDir, RELEASE_FILES.catalog),
    cohort: read(fixture.evidenceDir, "serving-cohort.json"),
    finalBytes: fs.readFileSync(path.join(fixture.evidenceDir, "serving-final.json")),
    policy: read(fixture.evidenceDir, "quality-policy.json"),
    tag: fixture.manifest.tag, bundleId: fixture.manifest.bundleId,
    rawContentSha256: fixture.review.rawContentSha256,
    selectionSha256: fixture.review.selectionSha256,
    finalReportSha256: fixture.review.finalReportSha256,
    graphSha256: releaseSha256(fs.readFileSync(path.join(fixture.candidateDir,
      RELEASE_FILES.neighborhood))),
    modelSha256: fixture.review.modelSha256,
    cohortSha256: fixture.review.servingCohortSha256,
    freezeSha256: fixture.review.servingFreezeSha256,
    policySha256: fixture.review.qualityPolicySha256,
    now: (() => { let tick = 0; return () => ++tick; })(),
  };
}

function changedFinal<T extends ReturnType<typeof servingInputs>>(inputs: T,
  change: (users: any[]) => void): T {
  const final = JSON.parse(inputs.finalBytes.toString("utf8"));
  change(final.users);
  const finalBytes = Buffer.from(encode(final));
  return { ...inputs, finalBytes, cohort: { ...inputs.cohort,
    finalSha256: releaseSha256(finalBytes) } };
}

test("invented v3 model package copies only item assets and records private hashes", (t) => {
  const fixture = setup(t);
  const audit = packageModelRelease({ ...fixture, syntheticFixture: true });
  assert.equal(audit.status, "synthetic-only");
  assert.deepEqual(fs.readdirSync(fixture.outputDir).sort(), [...MODEL_OUTPUT_FILES].sort());
  assert.equal(audit.modelCoverage, 1);
  assert.equal(audit.base.bundleId, fixture.baseManifest.bundleId);
  assert.equal(audit.evidence.finalReportSha256, fixture.review.finalReportSha256);
  assert.equal(audit.evidence.servingCohortSha256, fixture.review.servingCohortSha256);
  for (const name of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer, RELEASE_FILES.catalog,
    RELEASE_FILES.model, RELEASE_FILES.manifest]) {
    assert.deepEqual(fs.readFileSync(path.join(fixture.candidateDir, name)),
      fs.readFileSync(path.join(fixture.outputDir, name)));
  }
  for (const privateName of ["model.npz", "model.metadata.json", "selection.json", "final-report.json",
    "refit-record.json", "dataset-bridge.json", "serving-cohort.json",
    "serving-report.json"]) {
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
    /serving-report\.json: browser scorer recomputation or frozen quality gate failed/);
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

test("a rehashed, passing-looking authored quality claim fails browser recomputation", (t) => {
  const fixture = setup(t);
  const report = { ...fixture.serving,
    model: { ...fixture.serving.model, ndcgAtK: fixture.serving.model.ndcgAtK + 0.01 } };
  assert.ok(report.model.ndcgAtK >= report.baseline.ndcgAtK + 0.05);
  const servingReportSha256 = write(fixture.evidenceDir, "serving-report.json", report);
  updateReview(fixture, { servingReportSha256 });
  assert.throws(() => packageModelRelease({ ...fixture, syntheticFixture: true }),
    /serving-report\.json: browser scorer recomputation or frozen quality gate failed/);
});

test("a rehashed private sidecar cannot change the archive fit-user set", (t) => {
  const fixture = setup(t);
  const sidecar = JSON.parse(fs.readFileSync(path.join(fixture.evidenceDir,
    "model.metadata.json"), "utf8"));
  sidecar.userIds[0] = "invented-other-fit-user";
  const numericMetadataSha256 = write(fixture.evidenceDir, "model.metadata.json", sidecar);
  const refit = JSON.parse(fs.readFileSync(path.join(fixture.evidenceDir,
    "refit-record.json"), "utf8"));
  refit.numericMetadataSha256 = numericMetadataSha256;
  const refitRecordSha256 = write(fixture.evidenceDir, "refit-record.json", refit);
  updateReview(fixture, { numericMetadataSha256, refitRecordSha256 });
  assert.throws(() => packageModelRelease({ ...fixture, syntheticFixture: true }),
    /model.npz: safe archive or frozen split-first refit reproduction failed/);
});

test("nonfixture packaging requires the exact private split and recomputed bridge", (t) => {
  const missing = setup(t, false);
  assert.throws(() => packageModelRelease({ ...missing, sourceDir: undefined }),
    /sourceDir: nonfixture packaging requires private raw ratings/);

  const extra = setup(t, false);
  fs.writeFileSync(path.join(extra.sourceDir!, "invented-extra.json"), "{}");
  assert.throws(() => packageModelRelease(extra), /source: must contain exactly/);

  const forged = setup(t, false);
  const bridge = JSON.parse(fs.readFileSync(path.join(forged.evidenceDir,
    "dataset-bridge.json"), "utf8"));
  bridge.verification.selectedPairs += 1;
  updateReview(forged, { datasetBridgeSha256: write(forged.evidenceDir,
    "dataset-bridge.json", bridge) });
  assert.throws(() => packageModelRelease(forged),
    /dataset-bridge.verification.selectedPairs/);

  const stale = setup(t, false);
  const rawPath = path.join(stale.sourceDir!, "raw-ratings.json");
  const raw = JSON.parse(fs.readFileSync(rawPath, "utf8"));
  raw.interactions[0].rawScore -= 1;
  fs.writeFileSync(rawPath, encode(raw));
  assert.throws(() => packageModelRelease(stale),
    /Graph dataset bridge private inputs: raw snapshot, split manifest, or metadata failed validation/);
});

test("nonfixture package rejects an altered selection after its review hash is refreshed", (t) => {
  const fixture = setup(t, false);
  const selection = JSON.parse(fs.readFileSync(path.join(fixture.evidenceDir,
    "selection.json"), "utf8"));
  selection.selectionSpec.modelSeed += 1;
  updateReview(fixture, { selectionFileSha256: write(fixture.evidenceDir,
    "selection.json", selection) });
  assert.throws(() => packageModelRelease(fixture),
    /model.npz: safe archive or frozen split-first refit reproduction failed/);
});

test("frozen serving marker precedes final parsing and a report cannot be replayed", (t) => {
  const fixture = setup(t, false);
  const originalReport = fs.readFileSync(path.join(fixture.evidenceDir, "serving-report.json"));
  const replay = spawnSync(process.execPath, ["--import", "tsx",
    path.join(repoRoot, "web/bench/model-promotion-serving-evaluation.ts"),
    "--write-frozen", fixture.candidateDir, fixture.evidenceDir],
  { cwd: repoRoot, encoding: "utf8" });
  assert.notEqual(replay.status, 0);
  assert.match(replay.stderr, /EEXIST/);
  assert.deepEqual(fs.readFileSync(path.join(fixture.evidenceDir, "serving-report.json")),
    originalReport);

  const inputs = servingInputs(fixture);
  const malformed = Buffer.from("{invalid reserved labels");
  let markerWritten = false;
  assert.throws(() => generateServingReport({ ...inputs, finalBytes: malformed,
    cohort: { ...inputs.cohort, finalSha256: releaseSha256(malformed) },
    beforeFinal: () => { markerWritten = true; } }), /serving-final.json: must be valid JSON/);
  assert.equal(markerWritten, true);
});

test("a rehashed policy or reserved final file cannot escape the prefit plan", (t) => {
  const policyFixture = setup(t, false);
  const policy = JSON.parse(fs.readFileSync(path.join(policyFixture.evidenceDir,
    "quality-policy.json"), "utf8"));
  policy.minimumNdcgLift = 0.01;
  const qualityPolicySha256 = write(policyFixture.evidenceDir, "quality-policy.json", policy);
  const report = JSON.parse(fs.readFileSync(path.join(policyFixture.evidenceDir,
    "serving-report.json"), "utf8"));
  report.policySha256 = qualityPolicySha256;
  const servingReportSha256 = write(policyFixture.evidenceDir, "serving-report.json", report);
  updateReview(policyFixture, { qualityPolicySha256, servingReportSha256 });
  assert.throws(() => packageModelRelease(policyFixture),
    /quality-policy.json.minimumNdcgLift: differs from the prefit plan/);

  const finalFixture = setup(t, false);
  fs.appendFileSync(path.join(finalFixture.evidenceDir, "serving-final.json"), " ");
  updateReview(finalFixture, { servingFinalSha256: releaseSha256(fs.readFileSync(
    path.join(finalFixture.evidenceDir, "serving-final.json"))) });
  assert.throws(() => packageModelRelease(finalFixture),
    /quality-plan.json.finalSha256: differs from the reviewed input/);
});

test("a generated report must meet the frozen coverage and fresh latency gates", (t) => {
  const fixture = setup(t);
  const inputs = servingInputs(fixture);
  const policy = { ...inputs.policy, minimumServingCoverage: 0.95 };
  const qualityPolicySha256 = write(fixture.evidenceDir, "quality-policy.json", policy);
  const serving = generateServingReport({ ...inputs, policy,
    policySha256: qualityPolicySha256 });
  assert.ok(serving.model.coverage < policy.minimumServingCoverage);
  const servingReportSha256 = write(fixture.evidenceDir, "serving-report.json", serving);
  updateReview(fixture, { qualityPolicySha256, servingReportSha256 });
  assert.throws(() => packageModelRelease({ ...fixture, syntheticFixture: true }),
    /serving-report: does not meet the reviewed quality, coverage, and latency floors/);
  const realPolicyInputs = servingInputs(fixture);
  let tick = 0;
  assert.throws(() => verifyServingReport({ ...realPolicyInputs,
    now: () => (tick += policy.maximumP95LatencyMs + 1) }, serving),
  /latency: fresh serving-path measurement exceeds the policy ceiling/);
});

test("serving evaluation isolates fit, baseline-selection, and final-test users", (t) => {
  const fixture = setup(t);
  const inputs = servingInputs(fixture);
  const overlap = changedFinal(inputs, (users) => {
    users[0].userId = inputs.cohort.baselineValidation[0].userId;
  });
  assert.throws(() => generateServingReport(overlap),
    /cohort\.finalTest\[0\]\.userId: overlaps fit or another evaluation group/);
  const fitOverlap = changedFinal(inputs, (users) => {
    users[0].userId = inputs.cohort.trainingUsers[0];
  });
  assert.throws(() => generateServingReport(fitOverlap),
    /cohort\.finalTest\[0\]\.userId: overlaps fit or another evaluation group/);
  const noValidationPositives = structuredClone(inputs.cohort);
  for (const user of noValidationPositives.baselineValidation) user.labels[0].rawScore = 3;
  assert.throws(() => generateServingReport({ ...inputs, cohort: noValidationPositives }),
    /baselineValidation: has too few eligible users or positive labels/);
});

test("serving report depends on held-out labels but not their input row order", (t) => {
  const fixture = setup(t);
  const inputs = servingInputs(fixture);
  const base = generateServingReport(inputs);
  const reordered = changedFinal(inputs, (users) => {
    for (const user of users) user.labels.reverse();
  });
  const reorderedReport = generateServingReport(reordered);
  assert.deepEqual(reorderedReport.model, base.model);
  assert.deepEqual(reorderedReport.baseline, base.baseline);
  const changed = changedFinal(inputs, (users) => {
    for (const user of users) user.labels[0].animeId = 102;
  });
  const changedReport = generateServingReport(changed);
  assert.notEqual(changedReport.baseline.ndcgAtK, base.baseline.ndcgAtK);
  const observedOrder = changedFinal(inputs, (users) => {
    for (const user of users) user.observed.push({ animeId: 104, rawScore: 8 });
  });
  const prefixA = generateServingReport(observedOrder);
  const reversedOrder = changedFinal(observedOrder, (users) => {
    for (const user of users) user.observed.reverse();
  });
  const prefixB = generateServingReport(reversedOrder);
  assert.deepEqual(prefixA.baseline, prefixB.baseline);
  assert.deepEqual(prefixA.model, prefixB.model);
  assert.equal(releaseSha256(fs.readFileSync(path.join(fixture.candidateDir,
    RELEASE_FILES.model))), fixture.review.modelSha256);
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
  const fixture = setup(t, false, true);
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
  assert.deepEqual(fs.readdirSync(fixture.outputDir).sort(), [...MODEL_OUTPUT_FILES].sort());
  for (const name of ["raw-ratings.json", "split-manifest.json", "anime-metadata.json",
    "dataset-bridge.json", "model.metadata.json", "model.npz"]) {
    assert.equal(fs.existsSync(path.join(fixture.outputDir, name)), false);
  }
  assert.equal(fs.readFileSync(path.join(fixture.outputDir,
    "model-promotion-audit.json"), "utf8").includes("invented-a"), false);
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
  assert.ok(fixture.planApproval);
  const { planApprovals, approvalRepoDir, approvalFiles, freezeRevision } = fixture.planApproval;
  assert.equal(planApprovals.freezes[0].qualityPlanSha256,
    audit.evidence.qualityPlanSha256);
  const git = (...args: string[]) => {
    const result = spawnSync("git", args, { cwd: approvalRepoDir, encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
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
    qualityPlanSha256: audit.evidence.qualityPlanSha256,
    qualityPolicySha256: audit.evidence.qualityPolicySha256,
    servingFinalSha256: audit.evidence.servingFinalSha256,
    servingFreezeSha256: audit.evidence.servingFreezeSha256,
    freezeRevision, freezeApprovalRef: refs.freeze,
    numericMetadataSha256: audit.numericMetadataSha256,
    servingCohortSha256: audit.evidence.servingCohortSha256,
    servingReportSha256: audit.evidence.servingReportSha256,
    owner: audit.approvals.owner, ownerApprovalRef: refs.owner,
    trainingApprovalRef: refs.training, publicationApprovalRef: refs.publication,
    deploymentApprovalRef: refs.deployment,
  }] };
  write(approvalFiles, "model-release-bundles.json", modelApprovals);
  git("add", ".");
  git("-c", "user.name=Invented Reviewer", "-c", "user.email=invented@example.test",
    "commit", "-qm", "approve invented model after serving freeze");
  const options = { packageDir: fixture.outputDir, baseDir: basePackage,
    evidenceDir: fixture.evidenceDir, sourceDir: fixture.sourceDir!,
    approvalRepoDir, planApprovals,
    dispatch: { tag: audit.tag, baseTag: audit.base.tag, sourceRunId: 123,
      artifactName: "invented-model" },
    providerApprovals, publicationApprovals, modelApprovals };
  assert.deepEqual(verifyModelReleasePackage(options), audit);
  const publicOptions = { packageDir: fixture.outputDir, baseDir: basePackage,
    approvalRepoDir, planApprovals, providerApprovals,
    publicationApprovals, modelApprovals,
    dispatch: { ...options.dispatch, ownerApprovalRef: refs.owner } };
  assert.deepEqual(verifyModelPublicRelease(publicOptions), audit);
  const wrongPublicApproval = structuredClone(modelApprovals);
  wrongPublicApproval.promotions[0].auditSha256 = H("f");
  assert.throws(() => verifyModelPublicRelease({ ...publicOptions,
    modelApprovals: wrongPublicApproval }),
  /modelApprovals.promotion.auditSha256: differs from exact package/);
  assert.throws(() => verifyModelPublicRelease({ ...publicOptions,
    dispatch: { ...publicOptions.dispatch, ownerApprovalRef: refs.freeze } }),
  /dispatch.ownerApprovalRef: differs from exact owner approval/);
  const originalPublicModel = fs.readFileSync(path.join(fixture.outputDir, RELEASE_FILES.model));
  fs.appendFileSync(path.join(fixture.outputDir, RELEASE_FILES.model), " ");
  assert.throws(() => verifyModelPublicRelease(publicOptions),
    /Release bundle .*model-mf-web.compact.json|Release bundle release-manifest.json/);
  fs.writeFileSync(path.join(fixture.outputDir, RELEASE_FILES.model), originalPublicModel);
  fs.writeFileSync(path.join(fixture.outputDir, "model.npz"), "invented private archive");
  assert.throws(() => verifyModelPublicRelease(publicOptions),
    /packageDir: has an unsupported file inventory/);
  fs.unlinkSync(path.join(fixture.outputDir, "model.npz"));
  const originalBaseAudit = fs.readFileSync(path.join(basePackage, "publication-audit.json"));
  fs.appendFileSync(path.join(basePackage, "publication-audit.json"), " ");
  assert.throws(() => verifyModelPublicRelease(publicOptions),
    /publicationApprovals.base: differs from reviewed data base bytes or owner/);
  fs.writeFileSync(path.join(basePackage, "publication-audit.json"), originalBaseAudit);
  const publicCli = spawnSync(process.execPath, ["--import", "tsx",
    path.join(repoRoot, "pipeline/src/verify-model-release-public.ts"),
    "--package", fixture.outputDir, "--base", basePackage,
    "--tag", audit.tag, "--base-tag", audit.base.tag,
    "--run-id", "123", "--artifact-name", "invented-model",
    "--owner-approval-ref", refs.owner], { cwd: repoRoot, encoding: "utf8" });
  assert.notEqual(publicCli.status, 0);
  assert.match(publicCli.stderr, /providerApprovals\.approvals\.training|modelApprovals/);
  const lateFreeze = structuredClone(modelApprovals);
  lateFreeze.promotions[0].freezeRevision = git("rev-parse", "HEAD");
  assert.throws(() => verifyModelReleasePackage({ ...options,
    sourceDir: path.join(fixture.root, "missing-source"), modelApprovals: lateFreeze }),
  /freezeRevision: must identify a prior commit/);
  assert.throws(() => verifyModelReleasePackage({ ...options,
    sourceDir: path.join(fixture.root, "missing-source"),
    planApprovals: { schemaVersion: 1, freezes: [] } }),
  /planApprovals: must be the committed current plan registry/);
  assert.throws(() => verifyModelReleasePackage({ ...options,
    modelApprovals: { schemaVersion: 1, promotions: [] } }), /exact committed owner record/);
  assert.throws(() => verifyModelReleasePackage({ ...options,
    sourceDir: path.join(fixture.root, "missing-source"),
    providerApprovals: { schemaVersion: 1, approvals: {} } }), /providerApprovals.*training/);
  const stale = structuredClone(modelApprovals);
  stale.promotions[0].auditSha256 = H("f");
  assert.throws(() => verifyModelReleasePackage({ ...options, modelApprovals: stale }),
    /exact committed owner record/);
  fs.appendFileSync(path.join(fixture.outputDir, "model-promotion-audit.json"), " ");
  assert.throws(() => verifyModelReleasePackage(options),
    /exact committed owner record/);
});
