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
import { packageDataRelease, PUBLIC_FIELDS, type PublicationReviewV1,
  type ValidatedFirstBundleApproval, type ValidatedPublicationApproval } from
  "../src/package-data-release.js";
import { RELEASE_FILES, releaseSha256, writeReleaseManifest } from "../src/release-manifest.js";
import { verifyPublicationPackage, type PublicationDispatch } from
  "../src/verify-publication-package.js";
import type { CompactGraphDataV2 } from "../src/types.js";

const fixtureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../web/public/demo-data");
const encoded = (value: unknown): string => `${JSON.stringify(value, null, 2)}\n`;

function root(t: TestContext): string {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "invented-package-approval-"));
  t.after(() => {
    const resolved = fs.realpathSync(directory);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  });
  return directory;
}

function bundle(directory: string, name: string, previous?: string, reviewed = false,
  reviewedGenesis = false) {
  const out = path.join(directory, name);
  fs.mkdirSync(out);
  const source = JSON.parse(fs.readFileSync(path.join(fixtureDir,
    RELEASE_FILES.neighborhood), "utf8")) as CompactGraphDataV2;
  const graph = projectAggregateGraph(source);
  if (reviewed) {
    graph.dataset.source = "invented-reviewed-source";
    const { graphId: _old, ...withoutId } = graph;
    graph.graphId = aggregateRecommendationGraphId(withoutId);
  }
  fs.writeFileSync(path.join(out, RELEASE_FILES.neighborhood), encoded(graph));
  fs.writeFileSync(path.join(out, RELEASE_FILES.explorer),
    encoded(buildExplorerGraph(graph, 5, 0)));
  fs.copyFileSync(path.join(fixtureDir, RELEASE_FILES.catalog),
    path.join(out, RELEASE_FILES.catalog));
  const manifest = writeReleaseManifest(out, `data-v${name}`, previous,
    !previous && !reviewedGenesis, reviewedGenesis);
  return { out, manifest };
}

function setup(t: TestContext) {
  const directory = root(t);
  const prior = bundle(directory, "prior", undefined, true);
  const current = bundle(directory, "current", prior.out, true);
  const approvalRef = "https://example.test/invented-owner-approval";
  const decisionRef = "docs/decisions/0029-aggregate-only-graph.md";
  const owner = "Invented owner";
  const review: PublicationReviewV1 = {
    format: "publication-review-v1", tag: current.manifest.tag,
    bundleId: current.manifest.bundleId,
    manifestSha256: releaseSha256(fs.readFileSync(path.join(current.out,
      RELEASE_FILES.manifest))),
    source: { name: current.manifest.dataset.source,
      datasetSha256: current.manifest.dataset.sha256,
      derivation: "Invented projected pair statistics", decisionRef },
    redistribution: { status: "reviewed-allowed", basis: "Mocked license basis",
      approvalRef, owner, allowedFields: [...PUBLIC_FIELDS],
      attribution: "Invented", deletionCorrection: "Rebuild invented snapshot" },
    changes: { previousTag: prior.manifest.tag, summary: "Invented candidate" },
    quality: { checks: [
      { id: "schema", evidenceRef: "Invented parser test" },
      { id: "privacy", evidenceRef: "Invented no-user-row test" },
      { id: "coverage", evidenceRef: "Invented pair-count test" },
    ] },
  };
  const approval: ValidatedPublicationApproval = { scope: "publication", approved: true,
    approvalRef, decisionRef, owner, sources: [review.source.name] };
  const packageDir = path.join(directory, "package");
  const audit = packageDataRelease({ candidateDir: current.out, previousDir: prior.out,
    outputDir: packageDir, review, approval });
  const dispatch: PublicationDispatch = { tag: audit.tag, sourceRunId: 12345,
    artifactName: "invented-reviewed-package", previousTag: prior.manifest.tag, approvalRef };
  const providerApprovals = { schemaVersion: 1, approvals: { publication: {
    approved: true, sources: [review.source.name], sourceBasis: "Invented approval evidence",
    use: "Public aggregate graph", owner, approvedAt: "2026-09-28", decisionRef,
    approvalRef,
  } } };
  const approvedPackage = { tag: audit.tag, bundleId: audit.bundleId,
    manifestSha256: audit.manifestSha256,
    auditSha256: releaseSha256(fs.readFileSync(path.join(packageDir, "publication-audit.json"))),
    sourceRunId: dispatch.sourceRunId, artifactName: dispatch.artifactName,
    previousTag: prior.manifest.tag, decisionRef, approvalRef, owner, bootstrap: null };
  const priorApprovedPackage = { tag: prior.manifest.tag,
    bundleId: prior.manifest.bundleId,
    manifestSha256: releaseSha256(fs.readFileSync(path.join(prior.out, RELEASE_FILES.manifest))),
    auditSha256: "a".repeat(64), sourceRunId: 12344,
    artifactName: "invented-prior-approved-package", previousTag: null,
    decisionRef, approvalRef, owner,
    bootstrap: { decisionRef: "docs/decisions/0032-reviewed-first-release-bootstrap.md",
      approvalRef: "https://example.test/invented-prior-bootstrap-review", owner } };
  const packageApprovals = { schemaVersion: 1, packages: [priorApprovedPackage, approvedPackage] };
  return { directory, prior, current, packageDir, audit, dispatch, providerApprovals,
    packageApprovals, approvedPackage, priorApprovedPackage, review, approval };
}

function setupFirst(t: TestContext) {
  const fixture = setup(t);
  const first = bundle(fixture.directory, "first", undefined, true, true);
  const review: PublicationReviewV1 = {
    ...fixture.review, tag: first.manifest.tag, bundleId: first.manifest.bundleId,
    manifestSha256: releaseSha256(fs.readFileSync(path.join(first.out, RELEASE_FILES.manifest))),
    changes: { previousTag: null, summary: "Invented first candidate" },
  };
  const bootstrapApproval: ValidatedFirstBundleApproval = {
    scope: "first-real-bundle", approved: true, tag: review.tag,
    bundleId: review.bundleId, manifestSha256: review.manifestSha256,
    decisionRef: "docs/decisions/0032-reviewed-first-release-bootstrap.md",
    approvalRef: "https://example.test/invented-separate-bootstrap-review",
    owner: fixture.approval.owner,
  };
  const packageDir = path.join(fixture.directory, "first-package");
  const audit = packageDataRelease({ candidateDir: first.out, outputDir: packageDir,
    review, approval: fixture.approval, bootstrapApproval });
  const dispatch: PublicationDispatch = { ...fixture.dispatch, tag: review.tag,
    previousTag: null };
  const approvedPackage = { ...fixture.approvedPackage,
    tag: review.tag, bundleId: review.bundleId,
    manifestSha256: review.manifestSha256,
    auditSha256: releaseSha256(fs.readFileSync(path.join(packageDir, "publication-audit.json"))),
    previousTag: null,
    bootstrap: { decisionRef: bootstrapApproval.decisionRef,
      approvalRef: bootstrapApproval.approvalRef, owner: bootstrapApproval.owner } };
  return { ...fixture, first, review, bootstrapApproval, packageDir, audit, dispatch,
    approvedPackage, packageApprovals: { schemaVersion: 1, packages: [approvedPackage] } };
}

test("a first publishable bundle needs separate exact bootstrap approval and no invented prior", (t) => {
  const fixture = setupFirst(t);
  assert.equal(fixture.first.manifest.lastKnownGood, null);
  assert.equal(verifyPublicationPackage({ packageDir: fixture.packageDir,
    dispatch: fixture.dispatch, providerApprovals: fixture.providerApprovals,
    packageApprovals: fixture.packageApprovals }).publishable, true);
  assert.throws(() => packageDataRelease({ candidateDir: fixture.first.out,
    outputDir: path.join(fixture.directory, "without-bootstrap"), review: fixture.review,
    approval: fixture.approval }), /bootstrapApproval.*separate exact/);
  assert.throws(() => verifyPublicationPackage({ packageDir: fixture.packageDir,
    dispatch: fixture.dispatch, providerApprovals: fixture.providerApprovals,
    packageApprovals: { schemaVersion: 1, packages: [{ ...fixture.approvedPackage,
      bootstrap: null }] } }), /bootstrap.*object/);
  assert.throws(() => verifyPublicationPackage({ packageDir: fixture.packageDir,
    dispatch: fixture.dispatch, providerApprovals: fixture.providerApprovals,
    packageApprovals: { schemaVersion: 1, packages: [{ ...fixture.approvedPackage,
      bootstrap: { ...fixture.approvedPackage.bootstrap,
        approvalRef: fixture.approval.approvalRef } }] } }), /bootstrap.*separate approval/);
  assert.throws(() => verifyPublicationPackage({ packageDir: fixture.packageDir,
    previousDir: fixture.prior.out, dispatch: fixture.dispatch,
    providerApprovals: fixture.providerApprovals,
    packageApprovals: fixture.packageApprovals }), /previousDir.*absent/);
  assert.throws(() => verifyPublicationPackage({ packageDir: fixture.packageDir,
    dispatch: { ...fixture.dispatch, previousTag: fixture.prior.manifest.tag },
    providerApprovals: fixture.providerApprovals,
    packageApprovals: fixture.packageApprovals }), /previousDir.*absent/);
});

test("exact five-file package and two independent approvals recompute byte-identically", (t) => {
  const fixture = setup(t);
  const verified = verifyPublicationPackage({ packageDir: fixture.packageDir,
    previousDir: fixture.prior.out, dispatch: fixture.dispatch,
    providerApprovals: fixture.providerApprovals,
    packageApprovals: fixture.packageApprovals });
  assert.deepEqual(verified, fixture.audit);
  assert.equal(verified.publishable, true);
  assert.throws(() => verifyPublicationPackage({ packageDir: fixture.packageDir,
    previousDir: fixture.prior.out, dispatch: fixture.dispatch,
    providerApprovals: fixture.providerApprovals,
    packageApprovals: { schemaVersion: 1, packages: [fixture.approvedPackage] } }),
  /packageApprovals.*predecessor needs an exact approved/);
});

test("empty, stale, duplicate, or mismatched independent approvals fail closed", (t) => {
  const fixture = setup(t);
  const run = (providerApprovals: unknown, packageApprovals: unknown,
    dispatch = fixture.dispatch) => verifyPublicationPackage({ packageDir: fixture.packageDir,
    previousDir: fixture.prior.out, dispatch, providerApprovals, packageApprovals });
  assert.throws(() => run({ schemaVersion: 1, approvals: {} }, fixture.packageApprovals),
    /providerApprovals.approvals.publication.*object/);
  assert.throws(() => run(fixture.providerApprovals, { schemaVersion: 1, packages: [] }),
    /packageApprovals.*no exact/);
  assert.throws(() => run(fixture.providerApprovals, { schemaVersion: 1,
    packages: [{ ...fixture.approvedPackage, auditSha256: "a".repeat(64) }] }),
  /packageApprovals.*no exact/);
  assert.throws(() => run(fixture.providerApprovals, fixture.packageApprovals,
    { ...fixture.dispatch, sourceRunId: 12346 }), /packageApprovals.*no exact/);
  assert.throws(() => run(fixture.providerApprovals, { schemaVersion: 1,
    packages: [fixture.approvedPackage, fixture.approvedPackage] }),
  /packageApprovals.packages.*duplicate tags/);
  assert.throws(() => run({ ...fixture.providerApprovals, approvals: { publication: {
    ...fixture.providerApprovals.approvals.publication, sources: ["other-source"] } } },
  fixture.packageApprovals), /providerApprovals.approvals.publication.*source\/use/);
  assert.throws(() => run({ ...fixture.providerApprovals, approvals: { publication: {
    ...fixture.providerApprovals.approvals.publication, approvedAt: "2026-02-30" } } },
  fixture.packageApprovals), /providerApprovals.approvals.publication.approvedAt.*real ISO date/);
});

test("extra private-looking assets and changed package bytes never pass", (t) => {
  const fixture = setup(t);
  const run = (packageApprovals = fixture.packageApprovals) =>
    verifyPublicationPackage({ packageDir: fixture.packageDir, previousDir: fixture.prior.out,
      dispatch: fixture.dispatch, providerApprovals: fixture.providerApprovals, packageApprovals });
  const extra = path.join(fixture.packageDir, "ratings.sqlite");
  fs.writeFileSync(extra, "invented only");
  assert.throws(() => run(), /packageDir.*exactly the five/);
  fs.rmSync(extra);
  const graphFile = path.join(fixture.packageDir, RELEASE_FILES.neighborhood);
  const originalGraph = fs.readFileSync(graphFile);
  fs.appendFileSync(graphFile, " ");
  assert.throws(() => run(), /release-manifest.json.*hashes/);
  fs.writeFileSync(graphFile, originalGraph);
  const auditFile = path.join(fixture.packageDir, "publication-audit.json");
  const originalAudit = fs.readFileSync(auditFile);
  const changed = JSON.parse(originalAudit.toString("utf8"));
  changed.quality.computed.pairCount++;
  fs.writeFileSync(auditFile, encoded(changed));
  assert.throws(() => run(), /packageApprovals.*no exact/);
  const approvedChangedAudit = { schemaVersion: 1, packages: [fixture.priorApprovedPackage,
    { ...fixture.approvedPackage,
      auditSha256: releaseSha256(fs.readFileSync(auditFile)) }] };
  assert.throws(() => run(approvedChangedAudit),
    /publication-audit.json.*independently recomputed package/);
  fs.writeFileSync(auditFile, originalAudit);
});

test("prior drift, unpublishable audit, and invalid dispatch fail before release work", (t) => {
  const fixture = setup(t);
  const run = (dispatch = fixture.dispatch) => verifyPublicationPackage({
    packageDir: fixture.packageDir, previousDir: fixture.prior.out, dispatch,
    providerApprovals: fixture.providerApprovals, packageApprovals: fixture.packageApprovals });
  assert.throws(() => run({ ...fixture.dispatch, tag: "data-latest" }),
    /dispatch.tag.*invalid/);
  assert.throws(() => run({ ...fixture.dispatch, previousTag: fixture.dispatch.tag }),
    /dispatch.previousTag.*differ/);
  const priorManifest = path.join(fixture.prior.out, RELEASE_FILES.manifest);
  const originalPrior = fs.readFileSync(priorManifest);
  fs.appendFileSync(priorManifest, " ");
  assert.throws(() => run(), /lastKnownGood.*manifest-byte hash/);
  fs.writeFileSync(priorManifest, originalPrior);
  const auditFile = path.join(fixture.packageDir, "publication-audit.json");
  const originalAudit = fs.readFileSync(auditFile);
  const changed = JSON.parse(originalAudit.toString("utf8"));
  changed.publishable = false;
  fs.writeFileSync(auditFile, encoded(changed));
  assert.throws(() => run(), /publication-audit.json.*publishable/);
  fs.writeFileSync(auditFile, originalAudit);
});

test("the executable verifier refuses the committed empty approval records", (t) => {
  const fixture = setup(t);
  const entry = path.resolve(path.dirname(fileURLToPath(import.meta.url)),
    "../src/verify-publication-package.ts");
  const result = spawnSync(process.execPath, ["--import", "tsx", entry,
    "--package", fixture.packageDir, "--previous", fixture.prior.out,
    "--tag", fixture.dispatch.tag, "--run-id", String(fixture.dispatch.sourceRunId),
    "--artifact-name", fixture.dispatch.artifactName,
    "--prior-tag", fixture.dispatch.previousTag,
    "--approval-ref", fixture.dispatch.approvalRef],
  { cwd: path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../.."),
    encoding: "utf8" });
  assert.equal(result.status, 1, result.stderr);
  assert.match(result.stderr, /providerApprovals.approvals.publication.*must be an object/);
});
