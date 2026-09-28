import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import { aggregateRecommendationGraphId } from "../src/core/graph-contract.js";
import { projectAggregateGraph } from "../src/core/aggregate-projection.js";
import { graphFromPrepared, prepareGraphBridge, verifyGraphDatasetBridge,
  verifyGraphDatasetBridgeFiles } from "../src/core/split-graph-bridge.js";
import { openDatabase, upsertAnime, upsertRating, upsertUser } from "../src/db.js";
import type { CompactGraphDataV2 } from "../src/types.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixtureRaw = path.join(root, "fixtures/synthetic-split-input.json");
const fixtureSplit = path.join(root, "fixtures/synthetic-split-manifest.json");
const fixtureMetadata = path.join(root, "fixtures/synthetic-anime-metadata.json");
const sourceName = "invented-restricted-fit";
const generatedAt = "2026-01-01T00:00:00.000Z";
const config = {
  seed: 0, maxRatingsPerUser: 0, maxAnimeAnimeEdges: 0,
  maxPairVisits: 20_000_000, maxPairCandidates: 2_500_000,
  minPairSupport: 1, maxNeighborsPerAnime: 0,
  ratingSelectionPolicy: "all-ratings" as const,
};

function setup(t: TestContext) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "invented-graph-bridge-"));
  t.after(() => {
    const resolved = fs.realpathSync(directory);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) throw new Error("Unsafe temp cleanup");
    fs.rmSync(directory, { recursive: true, force: true });
  });
  const raw = path.join(directory, "raw-ratings.json");
  const split = path.join(directory, "split-manifest.json");
  const metadata = path.join(directory, "anime-metadata.json");
  fs.copyFileSync(fixtureRaw, raw);
  fs.copyFileSync(fixtureSplit, split);
  fs.copyFileSync(fixtureMetadata, metadata);
  return { directory, raw, split, metadata };
}

function refit(prepared: ReturnType<typeof prepareGraphBridge>): Record<string, unknown> {
  return { format: "split-first-final-refit-v1", fitMembership: "train-plus-validation",
    rawContentSha256: prepared.rawContentSha256,
    splitIdentitySha256: prepared.splitIdentitySha256,
    metadataSha256: prepared.metadataSha256,
    trainRows: prepared.trainRows, validationRows: prepared.validationRows,
    testRowsExcluded: prepared.testRowsExcluded, refitRows: prepared.rows.length,
    refitTrainSha256: prepared.refitTrainSha256,
    refitFitSha256: prepared.refitFitSha256,
    originalTrainSha256: prepared.originalTrainSha256 };
}

function interactionId(userId: string, animeId: number): string {
  return createHash("sha256").update("wasiw-interaction-v1\n")
    .update(JSON.stringify([userId, animeId])).digest("hex");
}

function changePartitionScore(paths: ReturnType<typeof setup>, partition: "trainIds" | "testIds") {
  const split = JSON.parse(fs.readFileSync(paths.split, "utf8"));
  const raw = JSON.parse(fs.readFileSync(paths.raw, "utf8"));
  const target = raw.interactions.find((row: { userId: string; animeId: number }) =>
    split[partition].includes(interactionId(row.userId, row.animeId)) &&
    raw.interactions.filter((other: { userId: string; animeId: number }) =>
      other.userId === row.userId && other.animeId === row.animeId).length === 1);
  assert.ok(target);
  target.rawScore += 1;
  fs.writeFileSync(paths.raw, JSON.stringify(raw));
  const rebuilt = spawnSync("python", [path.join(root, "ml/raw_interaction_split.py"),
    "--input", paths.raw, "--out", path.join(paths.directory, "fresh-manifest.json"),
    "--policy", "seeded"], { cwd: root, encoding: "utf8" });
  assert.equal(rebuilt.status, 0, rebuilt.stderr);
  return path.join(paths.directory, "fresh-manifest.json");
}

function sqliteProjection(paths: ReturnType<typeof setup>,
  prepared: ReturnType<typeof prepareGraphBridge>) {
  const db = openDatabase(path.join(paths.directory, "fit-only.sqlite"));
  try {
    for (const row of prepared.rows) {
      upsertUser(db, row.userId);
      upsertAnime(db, row.animeId, row.title);
      upsertRating(db, row.userId, row.animeId, row.rawScore);
    }
  } finally { db.close(); }
  const compact = path.join(paths.directory, "graph.compact.json");
  const built = spawnSync(process.execPath, ["--import", "tsx",
    path.join(root, "pipeline/src/build-graph.ts"),
    "--db", path.join(paths.directory, "fit-only.sqlite"),
    "--out-dataset-compact", path.join(paths.directory, "ratings.compact.json"),
    "--out-graph-compact", compact,
    "--out-report", path.join(paths.directory, "graph.report.json"),
    "--max-anime-anime-edges", "0", "--compact-only"],
  { cwd: root, encoding: "utf8" });
  assert.equal(built.status, 0, built.stderr);
  const v2 = JSON.parse(fs.readFileSync(compact, "utf8")) as CompactGraphDataV2;
  return { v2, projected: projectAggregateGraph(v2) };
}

test("invented raw split produces a private fit-only v3 graph and sanitized bridge report", (t) => {
  const paths = setup(t);
  const direct = spawnSync("python", [path.join(root, "ml/prepare_graph_dataset_bridge.py"),
    "--raw-ratings", paths.raw, "--split-manifest", paths.split,
    "--metadata", paths.metadata], { cwd: root, encoding: "utf8",
    env: { ...process.env, WASIW_PRIVATE_BRIDGE_PIPE: "" } });
  assert.notEqual(direct.status, 0);
  assert.equal(direct.stdout, "");
  const prepared = prepareGraphBridge(paths.raw, paths.split, paths.metadata);
  const graph = graphFromPrepared(prepared, sourceName, generatedAt, config);
  assert.equal(prepared.trainRows, 7);
  assert.equal(prepared.validationRows, 3);
  assert.equal(prepared.testRowsExcluded, 3);
  assert.equal(graph.format, "graph-compact-v3");
  assert.deepEqual(graph.userIds, []);
  assert.deepEqual(graph.ua, []);
  assert.equal(graph.truncation.inputRatings, 10);
  assert.equal(graph.truncation.selectedRatings, 10);
  assert.deepEqual(graph.aa.map(([left, right, weight, support]) =>
    [graph.anime[left][0], graph.anime[right][0], weight, support]), [
    [101, 102, 0.75, 2], [101, 103, -1.5, 1], [101, 105, 2, 1],
    [101, 107, -1.5, 1], [102, 103, -2, 1], [102, 105, 1.5, 1],
    [102, 107, 1, 1], [103, 105, -1, 1], [105, 108, 0, 1],
  ]);
  const graphFile = path.join(paths.directory, "graph.compact.json");
  const refitFile = path.join(paths.directory, "refit-record.json");
  fs.writeFileSync(graphFile, JSON.stringify(graph));
  fs.writeFileSync(refitFile, JSON.stringify(refit(prepared)));
  const report = verifyGraphDatasetBridgeFiles({ rawRatings: paths.raw,
    splitManifest: paths.split, metadata: paths.metadata, graph: graphFile,
    refitRecord: refitFile, sourceName });
  assert.equal(report.graphId, graph.graphId);
  assert.equal(report.refitFitSha256, prepared.refitFitSha256);
  assert.equal(report.fitMembership, "train-plus-validation");
  assert.equal(JSON.stringify(report).includes("invented-a"), false);
  assert.equal(JSON.stringify(report).includes("rawScore"), false);
  const cli = spawnSync(process.execPath, ["--import", "tsx",
    path.join(root, "pipeline/src/verify-graph-dataset-bridge.ts"),
    "--raw-ratings", fixtureRaw, "--split-manifest", fixtureSplit,
    "--metadata", fixtureMetadata, "--graph", graphFile,
    "--refit-record", refitFile, "--source-name", sourceName],
  { cwd: root, encoding: "utf8" });
  assert.equal(cli.status, 0, cli.stderr);
  assert.deepEqual(JSON.parse(cli.stdout), report);
  assert.equal(cli.stdout.includes("invented-a"), false);
  const unapproved = spawnSync(process.execPath, ["--import", "tsx",
    path.join(root, "pipeline/src/verify-graph-dataset-bridge.ts"),
    "--raw-ratings", paths.raw, "--split-manifest", paths.split,
    "--metadata", paths.metadata, "--graph", graphFile,
    "--refit-record", refitFile, "--source-name", sourceName],
  { cwd: root, encoding: "utf8" });
  assert.notEqual(unapproved.status, 0);
  assert.match(unapproved.stderr, /recorded training approval/);
  assert.equal(unapproved.stdout, "");
});

test("row order and identical duplicate do not alter the split-derived graph", (t) => {
  const paths = setup(t);
  const first = prepareGraphBridge(paths.raw, paths.split, paths.metadata);
  const raw = JSON.parse(fs.readFileSync(paths.raw, "utf8"));
  raw.interactions.reverse();
  fs.writeFileSync(paths.raw, JSON.stringify(raw));
  const reordered = prepareGraphBridge(paths.raw, paths.split, paths.metadata);
  assert.deepEqual(reordered, first);
  assert.deepEqual(graphFromPrepared(reordered, sourceName, generatedAt, config),
    graphFromPrepared(first, sourceName, generatedAt, config));
});

test("the existing SQLite graph producer agrees on exact invented fit-only v3 output", (t) => {
  const paths = setup(t);
  const prepared = prepareGraphBridge(paths.raw, paths.split, paths.metadata);
  const { v2, projected } = sqliteProjection(paths, prepared);
  const fromRaw = graphFromPrepared(prepared, "mixed-or-unverified", v2.generatedAt, config);
  assert.deepEqual(fromRaw, projected);
});

test("decimal scores retain exact graph-producer identity", (t) => {
  const paths = setup(t);
  const raw = JSON.parse(fs.readFileSync(paths.raw, "utf8"));
  for (const row of raw.interactions) row.rawScore /= 10;
  fs.writeFileSync(paths.raw, JSON.stringify(raw));
  const rebuilt = spawnSync("python", [path.join(root, "ml/raw_interaction_split.py"),
    "--input", paths.raw, "--out", path.join(paths.directory, "decimal-split.json"),
    "--policy", "seeded"], { cwd: root, encoding: "utf8" });
  assert.equal(rebuilt.status, 0, rebuilt.stderr);
  const prepared = prepareGraphBridge(paths.raw,
    path.join(paths.directory, "decimal-split.json"), paths.metadata);
  const { v2, projected } = sqliteProjection(paths, prepared);
  assert.deepEqual(graphFromPrepared(prepared, "mixed-or-unverified", v2.generatedAt, config),
    projected);
});

test("held-out score changes require a refreshed split but cannot alter fit graph", (t) => {
  const paths = setup(t);
  const before = prepareGraphBridge(paths.raw, paths.split, paths.metadata);
  const graph = graphFromPrepared(before, sourceName, generatedAt, config);
  const fresh = changePartitionScore(paths, "testIds");
  assert.throws(() => prepareGraphBridge(paths.raw, paths.split, paths.metadata),
    /raw snapshot, split manifest, or metadata failed validation/);
  const after = prepareGraphBridge(paths.raw, fresh, paths.metadata);
  assert.notEqual(after.rawContentSha256, before.rawContentSha256);
  assert.equal(after.splitIdentitySha256, before.splitIdentitySha256);
  assert.deepEqual(after.rows, before.rows);
  assert.deepEqual(graphFromPrepared(after, sourceName, generatedAt, config), graph);
  assert.equal(verifyGraphDatasetBridge(after, graph, sourceName, refit(after)).graphId, graph.graphId);
  assert.throws(() => verifyGraphDatasetBridge(after, graph, sourceName, refit(before)),
    /refit-record.rawContentSha256/);
});

test("fit score, fixed titles, signed pairs, config, and refit drift are refused", (t) => {
  const paths = setup(t);
  const before = prepareGraphBridge(paths.raw, paths.split, paths.metadata);
  const graph = graphFromPrepared(before, sourceName, generatedAt, config);
  const fresh = changePartitionScore(paths, "trainIds");
  const changed = prepareGraphBridge(paths.raw, fresh, paths.metadata);
  assert.notEqual(changed.refitTrainSha256, before.refitTrainSha256);
  assert.throws(() => verifyGraphDatasetBridge(changed, graph, sourceName, refit(changed)),
    /graph.compact.json.dataset.sha256/);

  const titlePaths = setup(t);
  const alteredMetadata = JSON.parse(fs.readFileSync(titlePaths.metadata, "utf8"));
  alteredMetadata.anime[0].title = "Invented Different Orbit";
  fs.writeFileSync(titlePaths.metadata, JSON.stringify(alteredMetadata));
  const renamed = prepareGraphBridge(titlePaths.raw, titlePaths.split, titlePaths.metadata);
  assert.throws(() => verifyGraphDatasetBridge(renamed, graph, sourceName, refit(renamed)),
    /graph.compact.json.dataset.sha256/);

  const signed = structuredClone(graph);
  signed.aa[0][2] *= -1;
  const { graphId: _id, ...withoutId } = signed;
  signed.graphId = aggregateRecommendationGraphId(withoutId);
  assert.throws(() => verifyGraphDatasetBridge(before, signed, sourceName, refit(before)),
    /graph.compact.json.graphId|graph.compact.json.aa/);
  const wrongConfig = structuredClone(graph);
  wrongConfig.config.maxAnimeAnimeEdges = 1;
  assert.throws(() => verifyGraphDatasetBridge(before, wrongConfig, sourceName, refit(before)),
    /graph.compact.json/);
  assert.throws(() => verifyGraphDatasetBridge(before, graph, sourceName,
    { ...refit(before), refitFitSha256: "0".repeat(64) }), /refit-record.refitFitSha256/);
});
