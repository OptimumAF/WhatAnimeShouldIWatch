import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import { RELEASE_BUNDLE_LIMITS } from "../../web/src/artifacts.js";
import { projectAggregateGraph } from "../src/core/aggregate-projection.js";
import { buildExplorerGraph } from "../src/core/explorer-graph.js";
import { aggregateRecommendationGraphId } from "../src/core/graph-contract.js";
import {
  OUTPUT_FILES, packageDataRelease, PUBLIC_FIELDS, type PublicationReviewV1,
  type ValidatedPublicationApproval,
} from "../src/package-data-release.js";
import { RELEASE_FILES, releaseSha256, writeReleaseManifest } from "../src/release-manifest.js";
import type { CompactGraphDataV2 } from "../src/types.js";

const fixtureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../web/public/demo-data");
const encode = (value: unknown): string => `${JSON.stringify(value, null, 2)}\n`;

function tempRoot(t: TestContext): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-publication-package-"));
  t.after(() => {
    const resolved = fs.realpathSync(root);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  });
  return root;
}

function makeBundle(root: string, name: string, previous?: string,
  sourceName = "synthetic-fixture") {
  const candidateDir = path.join(root, name);
  fs.mkdirSync(candidateDir);
  const source = JSON.parse(fs.readFileSync(path.join(fixtureDir,
    RELEASE_FILES.neighborhood), "utf8")) as CompactGraphDataV2;
  const graph = projectAggregateGraph(source);
  if (sourceName !== "synthetic-fixture") {
    graph.dataset.source = sourceName;
    const { graphId: _previousGraphId, ...withoutId } = graph;
    graph.graphId = aggregateRecommendationGraphId(withoutId);
  }
  fs.writeFileSync(path.join(candidateDir, RELEASE_FILES.neighborhood), encode(graph));
  fs.writeFileSync(path.join(candidateDir, RELEASE_FILES.explorer),
    encode(buildExplorerGraph(graph, 5, 0)));
  fs.copyFileSync(path.join(fixtureDir, RELEASE_FILES.catalog),
    path.join(candidateDir, RELEASE_FILES.catalog));
  const manifest = writeReleaseManifest(candidateDir, `data-v${name}`, previous, !previous);
  return { candidateDir, graph, manifest };
}

function reviewFor(bundle: ReturnType<typeof makeBundle>, previousTag: string | null = null):
  PublicationReviewV1 {
  return {
    format: "publication-review-v1", tag: bundle.manifest.tag,
    bundleId: bundle.manifest.bundleId,
    manifestSha256: releaseSha256(fs.readFileSync(path.join(bundle.candidateDir,
      RELEASE_FILES.manifest))),
    source: { name: bundle.manifest.dataset.source,
      datasetSha256: bundle.manifest.dataset.sha256,
      derivation: "Invented ratings projected to aggregate pairs",
      decisionRef: "docs/decisions/0029-aggregate-only-graph.md" },
    redistribution: { status: "synthetic-only", basis: "Invented fixture only",
      approvalRef: null, owner: "Test fixture owner", allowedFields: [...PUBLIC_FIELDS],
      attribution: "Invented fixture", deletionCorrection: "Replace test fixture if corrected" },
    changes: { previousTag, summary: "Invented bundle package check" },
    quality: { checks: [
      { id: "schema", evidenceRef: "pipeline/test/aggregate-projection.test.ts" },
      { id: "privacy", evidenceRef: "pipeline/test/package-data-release.test.ts" },
      { id: "coverage", evidenceRef: "Synthetic pair counts in publication audit" },
    ] },
  };
}

function stageNames(root: string): string[] {
  return fs.readdirSync(root).filter((name) => name.startsWith(".publication-"));
}

test("invented v3 package has only four verified assets and an unpublishable computed audit", (t) => {
  const root = tempRoot(t);
  const bundle = makeBundle(root, "fixture");
  const outputDir = path.join(root, "package");
  const audit = packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
    review: reviewFor(bundle), fixtureGenesis: true });
  assert.equal(audit.publishable, false);
  assert.deepEqual(fs.readdirSync(outputDir).sort(), [...OUTPUT_FILES].sort());
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(outputDir, "publication-audit.json"), "utf8")),
    audit);
  assert.equal(audit.quality.computed.animeCount, bundle.graph.anime.length);
  assert.equal(audit.quality.computed.pairCount, bundle.graph.aa.length);
  assert.equal(audit.quality.computed.positivePairCount,
    bundle.graph.aa.filter((edge) => edge[2] > 0).length);
  assert.equal(audit.quality.computed.minimumPairSupport,
    Math.min(...bundle.graph.aa.map((edge) => edge[3])));
  assert.equal(audit.compatibility.neighborhoodGraphId, bundle.graph.graphId);
  for (const asset of audit.assets) {
    const candidate = fs.readFileSync(path.join(bundle.candidateDir, asset.path));
    const packaged = fs.readFileSync(path.join(outputDir, asset.path));
    assert.deepEqual(packaged, candidate);
    assert.equal(asset.sha256, releaseSha256(candidate));
    assert.equal(asset.bytes, candidate.length);
  }
  assert.equal(stageNames(root).length, 0);
  assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
    review: reviewFor(bundle), fixtureGenesis: true }), /outputDir.*unused path/);
  assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir,
    outputDir: path.join(bundle.candidateDir, "nested-package"),
    review: reviewFor(bundle), fixtureGenesis: true }), /outputDir.*outside the candidate/);
  assert.equal(fs.existsSync(path.join(bundle.candidateDir, "nested-package")), false);
});

test("invented package refuses extra sensitive files, directories, and source links", (t) => {
  const root = tempRoot(t);
  const bundle = makeBundle(root, "fixture");
  const outputDir = path.join(root, "package");
  for (const name of ["anonymized-ratings.compact.json.gz", "ratings.sqlite", "user-salt.txt",
    "training-user-factors.json", RELEASE_FILES.model, ".hidden"]) {
    const extra = path.join(bundle.candidateDir, name);
    fs.writeFileSync(extra, "invented only");
    assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
      review: reviewFor(bundle), fixtureGenesis: true }), /candidate.*inventory/);
    fs.rmSync(extra);
  }
  const nested = path.join(bundle.candidateDir, "nested");
  fs.mkdirSync(nested);
  assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
    review: reviewFor(bundle), fixtureGenesis: true }), /candidate.*inventory/);
  fs.rmdirSync(nested);
  const graphFile = path.join(bundle.candidateDir, RELEASE_FILES.neighborhood);
  const original = fs.readFileSync(graphFile);
  fs.rmSync(graphFile);
  try {
    fs.symlinkSync(path.join(fixtureDir, RELEASE_FILES.neighborhood), graphFile, "file");
    assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
      review: reviewFor(bundle), fixtureGenesis: true }), /candidate.*regular file/);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EPERM") throw error;
    t.diagnostic("Windows symlink privilege unavailable; CI exercises this case");
  } finally {
    fs.rmSync(graphFile, { force: true });
    fs.writeFileSync(graphFile, original);
  }
  assert.equal(fs.existsSync(outputDir), false);
});

test("candidate byte limits, stale bytes, and malformed graph fail before output", (t) => {
  const root = tempRoot(t);
  const bundle = makeBundle(root, "fixture");
  const outputDir = path.join(root, "package");
  const review = reviewFor(bundle);
  const graphFile = path.join(bundle.candidateDir, RELEASE_FILES.neighborhood);
  const original = fs.readFileSync(graphFile);
  fs.appendFileSync(graphFile, " ");
  assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
    review, fixtureGenesis: true }), /release-manifest.json.*hashes/);
  fs.writeFileSync(graphFile, original);
  const graph = JSON.parse(original.toString("utf8"));
  graph.ua = [[0, 0, 1]];
  fs.writeFileSync(graphFile, encode(graph));
  assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
    review, fixtureGenesis: true }), /graph.compact.json.*userIds\/ua/);
  fs.writeFileSync(graphFile, original);
  fs.truncateSync(graphFile, RELEASE_BUNDLE_LIMITS.plainAssetBytes + 1);
  assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir, outputDir,
    review, fixtureGenesis: true }), /graph.compact.json.*byte limit/);
  fs.writeFileSync(graphFile, original);
  assert.equal(fs.existsSync(outputDir), false);
});

test("review fields, quality evidence, rights, and manifest bindings fail closed", (t) => {
  const root = tempRoot(t);
  const bundle = makeBundle(root, "fixture");
  const outputDir = path.join(root, "package");
  const run = (review: unknown, approval?: ValidatedPublicationApproval) =>
    packageDataRelease({ candidateDir: bundle.candidateDir, outputDir, review,
      fixtureGenesis: true, approval });
  assert.throws(() => run({ ...reviewFor(bundle), rawUsers: ["invented"] }),
    /review.rawUsers.*unsupported/);
  assert.throws(() => run({ ...reviewFor(bundle), manifestSha256: "a".repeat(64) }),
    /review.*manifest-byte hash/);
  assert.throws(() => run({ ...reviewFor(bundle), source: {
    ...reviewFor(bundle).source, datasetSha256: "b".repeat(64) } }),
  /review.source.*manifest dataset provenance/);
  assert.throws(() => run({ ...reviewFor(bundle), quality: { checks: [] } }),
    /review.quality.checks.*schema, privacy, and coverage/);
  assert.throws(() => run({ ...reviewFor(bundle), redistribution: {
    ...reviewFor(bundle).redistribution, allowedFields: ["user.id"] } }),
  /review.redistribution.allowedFields/);
  assert.throws(() => run({ ...reviewFor(bundle), redistribution: {
    ...reviewFor(bundle).redistribution, status: "reviewed-allowed",
    approvalRef: "https://example.test/review" } }),
  /fixtureGenesis.*publishable genesis/);
  assert.equal(fs.existsSync(outputDir), false);
});

test("a mocked real review binds the named prior and separately validated approval", (t) => {
  const root = tempRoot(t);
  const prior = makeBundle(root, "prior", undefined, "invented-reviewed-source");
  const bundle = makeBundle(root, "next", prior.candidateDir, "invented-reviewed-source");
  const outputDir = path.join(root, "package");
  const review = reviewFor(bundle, prior.manifest.tag);
  review.redistribution = { ...review.redistribution, status: "reviewed-allowed",
    approvalRef: "https://example.test/invented-approval" };
  const approval: ValidatedPublicationApproval = {
    scope: "publication", approved: true, approvalRef: review.redistribution.approvalRef!,
    decisionRef: review.source.decisionRef, owner: review.redistribution.owner,
    sources: [review.source.name],
  };
  const run = (changes = review.changes, useApproval: ValidatedPublicationApproval | null = approval) =>
    packageDataRelease({ candidateDir: bundle.candidateDir, previousDir: prior.candidateDir,
      outputDir, review: { ...review, changes }, approval: useApproval ?? undefined });
  assert.throws(() => run(review.changes, null), /approval.*separately validated/);
  assert.throws(() => run({ ...review.changes, previousTag: "data-vwrong" }),
    /review.changes.previousTag.*verified prior/);
  assert.throws(() => run(review.changes, { ...approval, owner: "Wrong owner" }),
    /approval.*separately validated/);
  const priorManifest = path.join(prior.candidateDir, RELEASE_FILES.manifest);
  const originalPrior = fs.readFileSync(priorManifest);
  fs.appendFileSync(priorManifest, " ");
  assert.throws(() => run(), /lastKnownGood.*manifest-byte hash/);
  fs.writeFileSync(priorManifest, originalPrior);
  const syntheticPrior = makeBundle(root, "synthetic-prior");
  const syntheticNext = makeBundle(root, "synthetic-next", syntheticPrior.candidateDir,
    "invented-reviewed-source");
  assert.throws(() => packageDataRelease({ candidateDir: syntheticNext.candidateDir,
    previousDir: syntheticPrior.candidateDir,
    outputDir: path.join(root, "synthetic-prior-package"),
    review: { ...reviewFor(syntheticNext, syntheticPrior.manifest.tag),
      redistribution: review.redistribution }, approval }),
  /previousDir.*reviewed data-only v3 predecessor/);
  const audit = run();
  assert.equal(audit.publishable, true);
  assert.equal(audit.changes.previousTag, prior.manifest.tag);
  assert.equal(fs.existsSync(path.join(outputDir, "publication-audit.json")), true);
});

test("failed activation cleans staging and permits an exact retry", (t) => {
  const root = tempRoot(t);
  const bundle = makeBundle(root, "fixture");
  const outputDir = path.join(root, "package");
  const options = { candidateDir: bundle.candidateDir, outputDir,
    review: reviewFor(bundle), fixtureGenesis: true };
  assert.throws(() => packageDataRelease({ ...options,
    beforeActivate: () => { throw new Error("invented staging failure"); } }),
  /invented staging failure/);
  assert.equal(fs.existsSync(outputDir), false);
  assert.deepEqual(stageNames(root), []);
  const audit = packageDataRelease(options);
  assert.equal(audit.publishable, false);
  assert.deepEqual(stageNames(root), []);
});

test("a destination created during staging is preserved, not overwritten", (t) => {
  const root = tempRoot(t);
  const bundle = makeBundle(root, "fixture");
  const outputDir = path.join(root, "package");
  assert.throws(() => packageDataRelease({ candidateDir: bundle.candidateDir,
    outputDir, review: reviewFor(bundle), fixtureGenesis: true,
    beforeActivate: () => {
      fs.mkdirSync(outputDir);
      fs.writeFileSync(path.join(outputDir, "marker.txt"), "invented existing content");
    } }), /outputDir.*created during packaging/);
  assert.equal(fs.readFileSync(path.join(outputDir, "marker.txt"), "utf8"),
    "invented existing content");
  assert.deepEqual(stageNames(root), []);
});
