/** Offline M5.8 parity against an independent Python reference on invented data. */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseCompactModel } from "../src/artifacts.ts";
import type { AnimeMetadata, GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex } from "../src/domain.ts";
import { selectFranchiseDiverseRecommendations } from "../src/franchise-diversity.ts";
import type { HistoryEntry } from "../src/import-history.ts";
import type { AnimePreference } from "../src/preferences.ts";
import {
  buildModelRecommendationsForPreferences, buildRecommendationIndex,
  createCandidateEligibilityPolicy, rankEligibleCandidates,
} from "../src/recommendations.ts";
import type { RecommendationFilters } from "../src/recommendations.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const INPUT = "fixtures/synthetic-model-parity-input.json";
const SPEC = "fixtures/synthetic-model-parity-spec.json";
const CASES = ["signed-and-excluded", "metadata-and-allowlist", "seen-only"];

type ParitySpec = {
  format: "synthetic-model-parity-spec-v1"; inputSha256: string;
  numericArchiveFormat: "model-numeric-npz-v1";
  metadataFormat: "model-numeric-sidecar-v1"; webFormat: "model-mf-compact-v1";
  exportRoundDigits: 8; scoreAbsoluteTolerance: number;
  tieBreak: "score-support-strongest-source-order"; cases: string[];
};
type ParityPreference = {
  animeId: number; sentiment: "liked" | "disliked" | "seen";
  importance: number; confidence: number;
};
type ParityCase = {
  id: string; preferences: ParityPreference[]; historySeen: number[];
  exclude: number[]; includeOnly: number[]; filters: RecommendationFilters; topK: number;
};
type ParityInput = { format: string; cases: ParityCase[] };
type ParityRow = { animeId: number; score: number };
type ReferenceCase = {
  id: string; raw: ParityRow[]; eligible: ParityRow[]; topKIds: number[];
  sourceExcludedIds: number[]; policyExcludedIds: number[];
};
type Reference = {
  format: string; inputSha256: string; archiveSha256: string;
  model: unknown; metadata: { animeId: number; year: number; score: number;
    genres: string[] }[]; cases: ReferenceCase[];
};

function fail(field: string): never { throw new Error("Invalid model parity " + field + "."); }
function readJson(relative: string): unknown {
  return JSON.parse(readFileSync(resolve(ROOT, relative), "utf8"));
}
function inputSha(): string {
  const bytes = readFileSync(resolve(ROOT, INPUT));
  return createHash("sha256").update(Buffer.from(
    bytes.toString("latin1").replace(/\r\n/g, "\n"), "latin1")).digest("hex");
}
function ids(rows: readonly ParityRow[]): number[] { return rows.map((row) => row.animeId); }
function same(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((item, i) => item === right[i]);
}
function localHistory(animeId: number, title: string): HistoryEntry {
  return { provider: "local", sourceId: String(animeId), animeId, title,
    status: "completed", sourceStatus: "completed", progressEpisodes: null,
    score: null, scoreScale: "local-10" };
}

export function parseParitySpec(value: unknown): ParitySpec {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("specification");
  const item = value as Record<string, unknown>;
  const keys = ["format", "inputSha256", "numericArchiveFormat", "metadataFormat",
    "webFormat", "exportRoundDigits", "scoreAbsoluteTolerance", "tieBreak", "cases"];
  if (Object.keys(item).sort().join("|") !== keys.sort().join("|") ||
      item.format !== "synthetic-model-parity-spec-v1" ||
      item.inputSha256 !== inputSha() ||
      item.numericArchiveFormat !== "model-numeric-npz-v1" ||
      item.metadataFormat !== "model-numeric-sidecar-v1" ||
      item.webFormat !== "model-mf-compact-v1" ||
      item.exportRoundDigits !== 8 || item.scoreAbsoluteTolerance !== 0.00001 ||
      item.tieBreak !== "score-support-strongest-source-order" ||
      JSON.stringify(item.cases) !== JSON.stringify(CASES)) fail("specification protocol/hash");
  return item as ParitySpec;
}

export function compareRows(
  actual: readonly ParityRow[], expected: readonly ParityRow[],
  tolerance: number, field: string,
): number {
  if (!Number.isFinite(tolerance) || tolerance < 0 || !same(ids(actual), ids(expected))) {
    fail(field + " candidate IDs/order");
  }
  let maximum = 0;
  for (let i = 0; i < actual.length; i += 1) {
    const delta = Math.abs(actual[i].score - expected[i].score);
    if (!Number.isFinite(delta) || delta > tolerance) fail(field + " score[" + i + "]");
    maximum = Math.max(maximum, delta);
  }
  return maximum;
}

export function evaluateParity(spec: ParitySpec, input: ParityInput,
                               reference: Reference) {
  if (input.format !== "synthetic-model-parity-input-v1" ||
      reference.format !== "synthetic-model-parity-reference-v1" ||
      reference.inputSha256 !== spec.inputSha256 ||
      JSON.stringify(input.cases.map((item) => item.id)) !== JSON.stringify(spec.cases) ||
      JSON.stringify(reference.cases.map((item) => item.id)) !== JSON.stringify(spec.cases)) {
    fail("input/reference protocol");
  }
  const model = parseCompactModel(reference.model, "synthetic parity model");
  if (model.sourceModelSha256 !== reference.archiveSha256 ||
      !/^[a-f0-9]{64}$/.test(reference.archiveSha256)) fail("source archive hash");
  const graphNodes: GraphData["nodes"] = model.animeIds.map((animeId, i) =>
    ({ id: "anime:" + animeId, label: model.titles[i], nodeType: "anime" }));
  const index = buildRecommendationIndex({
    generatedAt: model.generatedAt, userCount: 0, animeCount: graphNodes.length,
    nodeCount: graphNodes.length, edgeCount: 0, nodes: graphNodes, edges: [],
  });
  const modelIndex: ModelRecommendationIndex = {
    generatedAt: model.generatedAt, factors: model.factors, globalMean: model.globalMean,
    animeByAnimeId: new Map(model.animeIds.map((animeId, i) =>
      [animeId, { animeId, title: model.titles[i], bias: model.biases[i],
        embedding: model.embeddings[i] }])),
  };
  const metadata = new Map<number, AnimeMetadata>(reference.metadata.map((item) =>
    [item.animeId, { ...item, studios: [], synopsis: "", imageUrl: "", season: null }]));
  const titleById = new Map(model.animeIds.map((animeId, i) => [animeId, model.titles[i]]));
  let maximumScoreDelta = 0;
  const cases = input.cases.map((item, caseIndex) => {
    const expected = reference.cases[caseIndex];
    const preferences: AnimePreference[] = item.preferences.map((entry) => ({
      nodeId: "anime:" + entry.animeId, sentiment: entry.sentiment,
      importance: entry.importance, confidence: entry.confidence, source: "manual",
    }));
    const history = item.historySeen.map((animeId) =>
      localHistory(animeId, titleById.get(animeId) ?? "Invented unknown"));
    const policy = createCandidateEligibilityPolicy({
      index, preferences, history,
      includeOnlyNodeIds: item.includeOnly.map((id) => "anime:" + id),
      excludeNodeIds: item.exclude.map((id) => "anime:" + id),
      filters: item.filters,
    });
    const raw = buildModelRecommendationsForPreferences(preferences, index, modelIndex);
    const eligible = rankEligibleCandidates("model", { model: raw },
      policy, metadata).recommendations;
    const watched = new Set([...item.preferences.map((entry) => entry.animeId),
      ...item.historySeen]);
    const displayed = selectFranchiseDiverseRecommendations(eligible, metadata,
      watched, false, titleById).recommendations;
    const rawRows = raw.map((result) =>
      ({ animeId: result.anime.animeId, score: result.score }));
    const eligibleRows = eligible.map((result) =>
      ({ animeId: result.anime.animeId, score: result.score }));
    maximumScoreDelta = Math.max(maximumScoreDelta,
      compareRows(rawRows, expected.raw, spec.scoreAbsoluteTolerance, item.id + " raw"),
      compareRows(eligibleRows, expected.eligible, spec.scoreAbsoluteTolerance,
        item.id + " eligible"));
    const sourceExcluded = model.animeIds.filter((animeId) =>
      item.preferences.some((entry) => entry.animeId === animeId));
    const eligibleSet = new Set(eligible.map((result) => result.anime.animeId));
    const policyExcluded = raw.filter((result) => !eligibleSet.has(result.anime.animeId))
      .map((result) => result.anime.animeId);
    const topKIds = displayed.slice(0, item.topK).map((result) => result.anime.animeId);
    if (!same(sourceExcluded, expected.sourceExcludedIds) ||
        !same(policyExcluded, expected.policyExcludedIds) ||
        !same(topKIds, expected.topKIds) ||
        displayed.length !== eligible.length) fail(item.id + " exclusions/top K/selector");
    return { id: item.id, rawCandidateIds: ids(rawRows),
      eligibleCandidateIds: ids(eligibleRows), sourceExcludedIds: sourceExcluded,
      policyExcludedIds: policyExcluded, topKIds };
  });
  return { format: "synthetic-model-parity-v1", inputSha256: spec.inputSha256,
    archiveSha256: reference.archiveSha256,
    scoreAbsoluteTolerance: spec.scoreAbsoluteTolerance, maximumScoreDelta, cases };
}

export function runParity() {
  const spec = parseParitySpec(readJson(SPEC));
  const input = readJson(INPUT) as ParityInput;
  const output = execFileSync("python", ["ml/model_parity_reference.py"],
    { cwd: ROOT, encoding: "utf8", maxBuffer: 4 * 1024 * 1024 });
  return evaluateParity(spec, input, JSON.parse(output) as Reference);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  process.stdout.write(JSON.stringify(runParity(), null, 2) + "\n");
}
