import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  parseCompactGraph, parseCompactModel, parseDemoCatalog,
} from "../../web/src/artifacts.js";
import { recommendationGraphId } from "./core/graph-contract.js";
import type { CompactGraphDataV2 } from "./types.js";

const DIGEST = /^[a-f0-9]{64}$/;
const VERSION_TAG = /^data-v[a-zA-Z0-9._-]+$/;
const FILES = {
  graph: "graph.compact.json",
  catalog: "catalog.json",
  model: "model-mf-web.compact.json",
} as const;

export interface PromotionPaths {
  candidateDir: string;
  rollbackDir: string;
  approvalsPath: string;
  repoRoot: string;
}

export interface PromotionCheck {
  promotionId: string;
  targetTag: string;
  previousTag: string;
  modelSha256: string;
  graphId: string;
  graphDatasetSha256: string;
  modelCoverage: number;
}

function fail(field: string, reason: string): never {
  throw new Error(`Model promotion ${field}: ${reason}`);
}

function object(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    fail(field, "must be an object");
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, field: string, keys: string[]): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    fail(field, `expected fields ${expected.join(", ")}`);
  }
}

function word(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim() || value !== value.trim()) {
    fail(field, "must be nonempty trimmed text");
  }
  return value as string;
}

function digest(value: unknown, field: string): string {
  if (typeof value !== "string" || !DIGEST.test(value)) {
    fail(field, "must be a lowercase SHA-256 digest");
  }
  return value as string;
}

function https(value: unknown, field: string): string {
  const text = word(value, field);
  try {
    const url = new URL(text);
    if (url.protocol !== "https:" || !url.hostname) fail(field, "must be an HTTPS URL");
  } catch {
    fail(field, "must be an HTTPS URL");
  }
  return text;
}

function sha(bytes: Buffer): string {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function read(directory: string, filename: string): Buffer {
  return fs.readFileSync(path.join(directory, filename));
}

function json(bytes: Buffer, field: string): Record<string, unknown> {
  try {
    return object(JSON.parse(bytes.toString("utf8")), field);
  } catch (error) {
    if (error instanceof SyntaxError) fail(field, "contains invalid JSON");
    throw error;
  }
}

function same(actual: unknown, expected: unknown, field: string): void {
  if (actual !== expected) fail(field, "does not match the bound artifact or approval");
}

function mapTitles(graph: CompactGraphDataV2, catalogValue: unknown, modelValue: unknown,
                   label: string): number {
  const catalog = parseDemoCatalog(catalogValue, `${label}/catalog.json`);
  const model = parseCompactModel(modelValue, `${label}/model-mf-web.compact.json`);
  const catalogTitles = new Map(catalog.map((item) => [item.animeId, item.title]));
  const graphTitles = new Map(graph.anime.map(([id, title]) => [id, title]));
  if (graphTitles.size === 0) fail(`${label}.graph.anime`, "must contain candidates");
  for (const [id, title] of graphTitles) {
    same(catalogTitles.get(id), title, `${label}.catalog.anime[${id}]`);
  }
  model.animeIds.forEach((id, index) => {
    same(graphTitles.get(id), model.titles[index], `${label}.model.animeIds[${index}]`);
  });
  return model.animeIds.length / graphTitles.size;
}

function bundle(directory: string, label: string) {
  const graphBytes = read(directory, FILES.graph);
  const catalogBytes = read(directory, FILES.catalog);
  const modelBytes = read(directory, FILES.model);
  const graph = parseCompactGraph(json(graphBytes, `${label}.graph`),
    `${label}/graph.compact.json`, "recommendation");
  if (graph.format !== "graph-compact-v2") {
    fail(`${label}.graph.format`, "model promotion requires graph-compact-v2");
  }
  same(graph.graphId, recommendationGraphId(graph), `${label}.graph.graphId`);
  const coverage = mapTitles(graph as CompactGraphDataV2,
    json(catalogBytes, `${label}.catalog`), json(modelBytes, `${label}.model`), label);
  return { graph: graph as CompactGraphDataV2, coverage,
    hashes: { graph: sha(graphBytes), catalog: sha(catalogBytes), model: sha(modelBytes) },
    model: json(modelBytes, `${label}.model`) };
}

export function verifyModelPromotion(paths: PromotionPaths): PromotionCheck {
  const recordBytes = read(paths.candidateDir, "model-promotion.json");
  const record = json(recordBytes, "record");
  const fields = ["format", "promotionId", "targetTag", "previousTag", "modelSha256",
    "graphSha256", "catalogSha256", "graphId", "graphDatasetSha256",
    "rawTrainingSnapshotSha256", "refitRecordSha256", "finalReportSha256",
    "selectedCandidateId", "previousManifestSha256", "qualityDecisionRef",
    "ownerApprovalRef", "publicationApprovalRef", "deploymentApprovalRef",
    "minimumModelCoverage"];
  exactKeys(record, "record", fields);
  same(record.format, "model-promotion-v1", "record.format");
  const promotionId = word(record.promotionId, "record.promotionId");
  const targetTag = word(record.targetTag, "record.targetTag");
  const previousTag = word(record.previousTag, "record.previousTag");
  if (!VERSION_TAG.test(targetTag) || !VERSION_TAG.test(previousTag) ||
      targetTag === previousTag) {
    fail("record.targetTag/previousTag", "must be distinct immutable data-v tags");
  }
  for (const field of ["modelSha256", "graphSha256", "catalogSha256", "graphId",
    "graphDatasetSha256", "rawTrainingSnapshotSha256", "refitRecordSha256",
    "finalReportSha256", "previousManifestSha256"] as const) {
    digest(record[field], `record.${field}`);
  }
  word(record.selectedCandidateId, "record.selectedCandidateId");
  for (const field of ["ownerApprovalRef", "publicationApprovalRef",
    "deploymentApprovalRef"] as const) {
    https(record[field], `record.${field}`);
  }
  if (typeof record.minimumModelCoverage !== "number" ||
      !Number.isFinite(record.minimumModelCoverage) ||
      record.minimumModelCoverage <= 0 || record.minimumModelCoverage > 1) {
    fail("record.minimumModelCoverage", "must be in (0, 1]");
  }
  const decisionRef = word(record.qualityDecisionRef, "record.qualityDecisionRef");
  const decisionsDir = fs.realpathSync(path.resolve(paths.repoRoot, "docs/decisions"));
  const unresolvedDecisionPath = path.resolve(paths.repoRoot, decisionRef);
  if (!fs.existsSync(unresolvedDecisionPath)) {
    fail("record.qualityDecisionRef", "must name an existing decision file");
  }
  const decisionPath = fs.realpathSync(unresolvedDecisionPath);
  if (!decisionPath.startsWith(`${decisionsDir}${path.sep}`) ||
      !decisionPath.endsWith(".md")) {
    fail("record.qualityDecisionRef", "must name an existing decision file");
  }

  const candidate = bundle(paths.candidateDir, "candidate");
  same(candidate.hashes.model, record.modelSha256, "record.modelSha256");
  same(candidate.hashes.graph, record.graphSha256, "record.graphSha256");
  same(candidate.hashes.catalog, record.catalogSha256, "record.catalogSha256");
  same(candidate.graph.graphId, record.graphId, "record.graphId");
  same(candidate.graph.dataset.sha256, record.graphDatasetSha256,
    "record.graphDatasetSha256");
  if (candidate.coverage < (record.minimumModelCoverage as number)) {
    fail("record.minimumModelCoverage", "candidate model covers too few graph titles");
  }

  const refitBytes = read(paths.candidateDir, "refit-record.json");
  const refit = json(refitBytes, "refit-record");
  same(sha(refitBytes), record.refitRecordSha256, "record.refitRecordSha256");
  same(refit.format, "split-first-final-refit-v1", "refit-record.format");
  same(refit.fitMembership, "train-plus-validation", "refit-record.fitMembership");
  same(refit.releaseStatus, "unapproved experiment artifact", "refit-record.releaseStatus");
  same(refit.webModelSha256, record.modelSha256, "refit-record.webModelSha256");
  same(refit.numericArchiveSha256, candidate.model.sourceModelSha256,
    "refit-record.numericArchiveSha256");
  same(refit.rawContentSha256, record.rawTrainingSnapshotSha256,
    "record.rawTrainingSnapshotSha256");
  same(refit.finalReportSha256, record.finalReportSha256,
    "record.finalReportSha256");
  same(refit.selectedCandidateId, record.selectedCandidateId,
    "record.selectedCandidateId");
  if (!Number.isSafeInteger(refit.trainRows) || !Number.isSafeInteger(refit.validationRows) ||
      !Number.isSafeInteger(refit.refitRows) || !Number.isSafeInteger(refit.testRowsExcluded) ||
      (refit.trainRows as number) < 1 || (refit.validationRows as number) < 1 ||
      (refit.testRowsExcluded as number) < 1 ||
      refit.refitRows !== (refit.trainRows as number) + (refit.validationRows as number)) {
    fail("refit-record", "membership counts are inconsistent");
  }
  if ("test" in refit || "ndcgAtK" in refit || "recallAtK" in refit) {
    fail("refit-record", "must not contain held-out test metrics");
  }

  const manifestBytes = read(paths.rollbackDir, "rollback-manifest.json");
  const manifest = json(manifestBytes, "rollback-manifest");
  exactKeys(manifest, "rollback-manifest", ["format", "tag", "files"]);
  same(manifest.format, "model-rollback-bundle-v1", "rollback-manifest.format");
  same(manifest.tag, previousTag, "rollback-manifest.tag");
  same(sha(manifestBytes), record.previousManifestSha256,
    "record.previousManifestSha256");
  const previous = bundle(paths.rollbackDir, "rollback");
  if (previous.hashes.model === candidate.hashes.model) {
    fail("rollback.model", "must preserve a distinct previous model");
  }
  const previousFiles = object(manifest.files, "rollback-manifest.files");
  exactKeys(previousFiles, "rollback-manifest.files", [FILES.graph, FILES.catalog, FILES.model]);
  for (const [kind, filename] of Object.entries(FILES) as [keyof typeof FILES, string][]) {
    same(previousFiles[filename], previous.hashes[kind],
      `rollback-manifest.files.${filename}`);
  }

  const approvals = json(fs.readFileSync(paths.approvalsPath), "approvals");
  exactKeys(approvals, "approvals", ["format", "approvals"]);
  same(approvals.format, "model-promotion-approvals-v1", "approvals.format");
  if (!Array.isArray(approvals.approvals)) fail("approvals.approvals", "must be a list");
  const matches = approvals.approvals.filter((item) =>
    object(item, "approvals.entry").promotionId === promotionId);
  if (matches.length !== 1) fail("approvals", "requires exactly one committed owner review");
  const approved = object(matches[0], "approvals.entry");
  exactKeys(approved, "approvals.entry", ["promotionId", "targetTag", "recordSha256",
    "approved", "owner", "approvedAt", "qualityGatePassed", "ownerApprovalRef",
    "publicationApprovalRef", "deploymentApprovalRef"]);
  same(approved.approved, true, "approvals.entry.approved");
  same(approved.qualityGatePassed, true, "approvals.entry.qualityGatePassed");
  word(approved.owner, "approvals.entry.owner");
  const approvedAt = word(approved.approvedAt, "approvals.entry.approvedAt");
  const approvedDate = new Date(`${approvedAt}T00:00:00Z`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(approvedAt) ||
      Number.isNaN(approvedDate.getTime()) ||
      approvedDate.toISOString().slice(0, 10) !== approvedAt) {
    fail("approvals.entry.approvedAt", "must be an ISO date");
  }
  same(approved.targetTag, targetTag, "approvals.entry.targetTag");
  same(approved.recordSha256, sha(recordBytes), "approvals.entry.recordSha256");
  for (const field of ["ownerApprovalRef", "publicationApprovalRef",
    "deploymentApprovalRef"] as const) {
    same(approved[field], record[field], `approvals.entry.${field}`);
  }
  return { promotionId, targetTag, previousTag,
    modelSha256: candidate.hashes.model, graphId: candidate.graph.graphId,
    graphDatasetSha256: candidate.graph.dataset.sha256,
    modelCoverage: candidate.coverage };
}
