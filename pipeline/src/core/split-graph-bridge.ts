/** Private raw-split to v3 graph check. No private row is returned in its report. */
import { isDeepStrictEqual } from "node:util";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseCompactGraph } from "../../../web/src/artifacts.js";
import type { AnonymizedDataset, CompactGraphDataV2, CompactGraphDataV3 } from "../types.js";
import { aggregateAnimePairs } from "./pair-aggregation.js";
import { datasetIdentity, recommendationGraphId, recommendationMetadata } from "./graph-contract.js";
import { projectAggregateGraph } from "./aggregate-projection.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const sha = /^[a-f0-9]{64}$/;

interface PrivateRow {
  userId: string;
  animeId: number;
  title: string;
  rawScore: number;
  normalizedScore: number;
}

export interface PreparedGraphBridge {
  format: "private-graph-bridge-rows-v1";
  rawContentSha256: string;
  splitIdentitySha256: string;
  metadataSha256: string;
  trainRows: number;
  validationRows: number;
  testRowsExcluded: number;
  refitTrainSha256: string;
  refitFitSha256: string;
  rows: PrivateRow[];
}

export interface GraphBridgeReport {
  format: "model-dataset-bridge-verification-v1";
  sourceName: string;
  fitMembership: "train-plus-validation";
  rawContentSha256: string;
  splitIdentitySha256: string;
  metadataSha256: string;
  trainRows: number;
  validationRows: number;
  testRowsExcluded: number;
  refitTrainSha256: string;
  refitFitSha256: string;
  graphDatasetSha256: string;
  graphId: string;
  selectedRatings: number;
  selectedPairs: number;
}

function fail(field: string, message: string): never {
  throw new Error(`Graph dataset bridge ${field}: ${message}`);
}

function digest(value: unknown, field: string): void {
  if (typeof value !== "string" || !sha.test(value)) fail(field, "invalid SHA-256 digest");
}

function count(value: unknown, field: string, minimum = 0): void {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) fail(field, "invalid count");
}

function validatePrepared(value: unknown): PreparedGraphBridge {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("prepared", "invalid object");
  const input = value as Record<string, unknown>;
  const keys = ["format", "rawContentSha256", "splitIdentitySha256", "metadataSha256",
    "trainRows", "validationRows", "testRowsExcluded", "refitTrainSha256", "refitFitSha256", "rows"];
  if (Object.keys(input).sort().join() !== keys.sort().join() ||
      input.format !== "private-graph-bridge-rows-v1") fail("prepared", "unsupported fields or format");
  for (const key of ["rawContentSha256", "splitIdentitySha256", "metadataSha256",
    "refitTrainSha256", "refitFitSha256"]) digest(input[key], `prepared.${key}`);
  for (const key of ["trainRows", "validationRows", "testRowsExcluded"]) {
    count(input[key], `prepared.${key}`, 1);
  }
  if (!Array.isArray(input.rows) || input.rows.length !==
      (input.trainRows as number) + (input.validationRows as number)) {
    fail("prepared.rows", "must contain exactly the fit rows");
  }
  for (const [index, raw] of input.rows.entries()) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) fail(`prepared.rows[${index}]`, "invalid row");
    const row = raw as Record<string, unknown>;
    if (Object.keys(row).sort().join() !==
        ["userId", "animeId", "title", "rawScore", "normalizedScore"].sort().join() ||
        typeof row.userId !== "string" || !row.userId ||
        typeof row.title !== "string" || !row.title ||
        !Number.isSafeInteger(row.animeId) || (row.animeId as number) < 1 ||
        typeof row.rawScore !== "number" || !Number.isFinite(row.rawScore) ||
        typeof row.normalizedScore !== "number" || !Number.isFinite(row.normalizedScore)) {
      fail(`prepared.rows[${index}]`, "invalid fields");
    }
  }
  return value as PreparedGraphBridge;
}

/** Calls the existing Python split/fit path; stdout is private and never logged. */
export function prepareGraphBridge(rawRatings: string, splitManifest: string,
  metadata: string): PreparedGraphBridge {
  const result = spawnSync("python", [path.join(root, "ml/prepare_graph_dataset_bridge.py"),
    "--raw-ratings", rawRatings, "--split-manifest", splitManifest,
    "--metadata", metadata], { cwd: root, encoding: "utf8", maxBuffer: 128 * 1024 * 1024,
    env: { ...process.env, WASIW_PRIVATE_BRIDGE_PIPE: "1" } });
  if (result.error || result.status !== 0) {
    fail("private inputs", "raw snapshot, split manifest, or metadata failed validation");
  }
  try { return validatePrepared(JSON.parse(result.stdout)); }
  catch (error) {
    if (error instanceof SyntaxError) fail("prepared", "invalid Python output");
    throw error;
  }
}

function firstDifference(actual: unknown, expected: unknown, field: string): string | null {
  if (isDeepStrictEqual(actual, expected)) return null;
  if (actual && expected && typeof actual === "object" && typeof expected === "object") {
    const left = actual as Record<string, unknown>;
    const right = expected as Record<string, unknown>;
    for (const key of new Set([...Object.keys(left), ...Object.keys(right)])) {
      const difference = firstDifference(left[key], right[key], `${field}.${key}`);
      if (difference) return difference;
    }
  }
  return field;
}

/** Derive the same pair selection, v2 provenance, and v3 projection from fit rows. */
export function graphFromPrepared(prepared: PreparedGraphBridge, sourceName: string,
  generatedAt: string, config: CompactGraphDataV3["config"]): CompactGraphDataV3 {
  validatePrepared(prepared);
  if (!sourceName || sourceName.trim() !== sourceName) fail("sourceName", "invalid source label");
  const byUser = new Map<string, AnonymizedDataset["users"][number]>();
  for (const row of prepared.rows) {
    let user = byUser.get(row.userId);
    if (!user) {
      user = { userId: row.userId, ratings: [] };
      byUser.set(row.userId, user);
    }
    user.ratings.push({ animeId: row.animeId, title: row.title,
      rawScore: row.rawScore, normalizedScore: row.normalizedScore });
  }
  const dataset: AnonymizedDataset = { generatedAt, source: sourceName,
    users: [...byUser.values()].sort((a, b) =>
      a.userId < b.userId ? -1 : a.userId > b.userId ? 1 : 0) };
  for (const user of dataset.users) user.ratings.sort((a, b) => a.animeId - b.animeId);
  const identity = datasetIdentity(dataset);
  const pairResult = aggregateAnimePairs(dataset.users, config.maxRatingsPerUser,
    config.maxAnimeAnimeEdges, { maxPairVisits: config.maxPairVisits,
      maxCandidatePairs: config.maxPairCandidates, minSupport: config.minPairSupport,
      maxNeighborsPerAnime: config.maxNeighborsPerAnime, selectionSeed: config.seed });
  const expectedPolicy = config.maxRatingsPerUser > 0 ? "sha256-bottom-k-v1" : "all-ratings";
  if (config.ratingSelectionPolicy !== expectedPolicy) fail("graph.config.ratingSelectionPolicy", "inconsistent selection");
  const anime: CompactGraphDataV2["anime"] = [];
  const animeIndex = new Map<number, number>();
  const ua: CompactGraphDataV2["ua"] = [];
  const userIds: string[] = [];
  for (const user of pairResult.selectedUsers) {
    const userIndex = userIds.length;
    userIds.push(user.userId);
    for (const rating of [...user.ratings].sort((a, b) => a.animeId - b.animeId)) {
      let index = animeIndex.get(rating.animeId);
      if (index === undefined) {
        index = anime.length;
        animeIndex.set(rating.animeId, index);
        anime.push([rating.animeId, rating.title]);
      }
      ua.push([userIndex, index, Number(rating.normalizedScore.toFixed(4))]);
    }
  }
  const aa: CompactGraphDataV2["aa"] = [...pairResult.pairs].map(([key, pair]) => {
    const [left, right] = key.split(":").map(Number);
    return [animeIndex.get(left)!, animeIndex.get(right)!, Number(pair.weight.toFixed(4)),
      pair.support];
  });
  const metadata = recommendationMetadata(identity, {
    seed: config.seed, maxRatingsPerUser: config.maxRatingsPerUser,
    maxAnimeAnimeEdges: config.maxAnimeAnimeEdges, maxPairVisits: config.maxPairVisits,
    maxPairCandidates: config.maxPairCandidates, minPairSupport: config.minPairSupport,
    maxNeighborsPerAnime: config.maxNeighborsPerAnime,
  }, pairResult.stats);
  const withoutId: Omit<CompactGraphDataV2, "graphId"> = {
    format: "graph-compact-v2", role: "recommendation", ...metadata, generatedAt,
    userIds, anime, ua, aa, userCount: userIds.length, animeCount: anime.length,
    nodeCount: userIds.length + anime.length, edgeCount: ua.length + aa.length,
  };
  return projectAggregateGraph({ ...withoutId, graphId: recommendationGraphId(withoutId) });
}

export function verifyGraphDatasetBridge(prepared: PreparedGraphBridge,
  graph: CompactGraphDataV3, sourceName: string,
  refit: Record<string, unknown>): GraphBridgeReport {
  parseCompactGraph(graph, "graph.compact.json", "recommendation");
  if (graph.format !== "graph-compact-v3") fail("graph.format", "requires aggregate v3");
  if (graph.dataset.source !== sourceName) fail("graph.dataset.source", "source label mismatch");
  const expected = graphFromPrepared(prepared, sourceName, graph.generatedAt, graph.config);
  const difference = firstDifference(graph, expected, "graph.compact.json");
  if (difference) fail(difference, "does not match exact split fit producer output");
  const fields: (keyof PreparedGraphBridge)[] = ["rawContentSha256", "splitIdentitySha256",
    "metadataSha256", "trainRows", "validationRows", "testRowsExcluded",
    "refitTrainSha256", "refitFitSha256"];
  for (const key of fields) {
    if (refit[key] !== prepared[key]) fail(`refit-record.${key}`, "does not match validated split fit");
  }
  if (refit.format !== "split-first-final-refit-v1" ||
      refit.fitMembership !== "train-plus-validation" ||
      refit.refitRows !== prepared.rows.length) fail("refit-record.fitMembership", "invalid fit membership/count");
  return { format: "model-dataset-bridge-verification-v1", sourceName,
    fitMembership: "train-plus-validation", rawContentSha256: prepared.rawContentSha256,
    splitIdentitySha256: prepared.splitIdentitySha256,
    metadataSha256: prepared.metadataSha256, trainRows: prepared.trainRows,
    validationRows: prepared.validationRows, testRowsExcluded: prepared.testRowsExcluded,
    refitTrainSha256: prepared.refitTrainSha256, refitFitSha256: prepared.refitFitSha256,
    graphDatasetSha256: graph.dataset.sha256, graphId: graph.graphId,
    selectedRatings: graph.truncation.selectedRatings,
    selectedPairs: graph.truncation.selectedPairs };
}

export function verifyGraphDatasetBridgeFiles(options: { rawRatings: string;
  splitManifest: string; metadata: string; graph: string; refitRecord: string;
  sourceName: string }): GraphBridgeReport {
  const prepared = prepareGraphBridge(options.rawRatings, options.splitManifest, options.metadata);
  const readJson = (file: string, field: string, limit: number) => {
    try {
      const stat = fs.lstatSync(file);
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > limit) {
        fail(field, "must be a bounded regular file");
      }
      return JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
    } catch (error) {
      if (error instanceof SyntaxError) fail(field, "invalid JSON");
      throw error;
    }
  };
  const graph = readJson(options.graph, "graph.compact.json", 64 * 1024 * 1024) as CompactGraphDataV3;
  const refit = readJson(options.refitRecord, "refit-record.json", 1024 * 1024) as Record<string, unknown>;
  return verifyGraphDatasetBridge(prepared, graph, options.sourceName, refit);
}
