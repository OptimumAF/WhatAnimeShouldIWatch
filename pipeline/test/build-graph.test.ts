import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { openDatabase, upsertAnime, upsertRating, upsertUser } from "../src/db.js";
import type { CompactGraphData, GraphData } from "../src/types.js";

test("the graph CLI exports the three-user mean and true pair support in both formats", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anime-graph-test-"));
  try {
    const dbPath = path.join(dir, "fixture.sqlite");
    const db = openDatabase(dbPath);
    try {
      for (const [animeId, title] of [[1, "A"], [2, "B"], [3, "C"]] as const) {
        upsertAnime(db, animeId, title);
      }
      for (const [userId, pairScore, thirdScore] of [
        ["u-c", 6, 3], ["u-a", 10, 1], ["u-b", 8, 2],
      ] as const) {
        upsertUser(db, userId);
        upsertRating(db, userId, 1, pairScore);
        upsertRating(db, userId, 2, pairScore);
        upsertRating(db, userId, 3, thirdScore);
      }
    } finally {
      db.close();
    }

    const graphPath = path.join(dir, "graph.json");
    const compactPath = path.join(dir, "graph.compact.json");
    const command = spawnSync(process.execPath, [
      "--import", "tsx", "src/build-graph.ts",
      "--db", dbPath,
      "--out-dataset", path.join(dir, "ratings.json"),
      "--out-graph", graphPath,
      "--out-dataset-compact", path.join(dir, "ratings.compact.json"),
      "--out-graph-compact", compactPath,
      "--max-anime-anime-edges", "0",
    ], { cwd: path.resolve("."), encoding: "utf8" });
    assert.equal(command.status, 0, `${command.stdout}\n${command.stderr}`);

    const graph = JSON.parse(fs.readFileSync(graphPath, "utf8")) as GraphData;
    assert.deepEqual(
      graph.edges.find((edge) => edge.id === "aa:1:2"),
      { id: "aa:1:2", source: "anime:1", target: "anime:2", edgeType: "anime-anime", weight: 2, support: 3 },
    );
    const compact = JSON.parse(fs.readFileSync(compactPath, "utf8")) as CompactGraphData;
    const left = compact.anime.findIndex(([id]) => id === 1);
    const right = compact.anime.findIndex(([id]) => id === 2);
    assert.deepEqual(compact.aa.find(([a, b]) => a === left && b === right), [left, right, 2, 3]);
    const report = JSON.parse(fs.readFileSync(`${graphPath}.report.json`, "utf8"));
    assert.equal(report.selection.policy, "all-ratings");
    assert.equal(report.selection.seed, 0);
    assert.equal(report.coverage.approximationLevel, "exact-input");
    assert.equal(report.coverage.pairVisitsSkipped, 0);
    assert.equal(report.coverage.pairKeyRecall, 1);
    assert.equal(report.coverage.outputTruncated, false);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("the graph CLI selects the later supported pair and fails before output on input-budget exhaustion", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anime-graph-bounded-test-"));
  try {
    const dbPath = path.join(dir, "fixture.sqlite");
    const db = openDatabase(dbPath);
    try {
      for (const animeId of [1, 2, 3, 4, 5, 6]) {
        upsertAnime(db, animeId, `Invented ${animeId}`);
      }
      const rows = [
        ["u-a", [[1, 6], [2, 4]]],
        ["u-b", [[3, 7], [4, 3]]],
        ["u-c", [[3, 7], [4, 7], [5, 1]]],
        ["u-d", [[3, 6], [4, 6], [6, 3]]],
      ] as const;
      for (const [userId, ratings] of rows) {
        upsertUser(db, userId);
        for (const [animeId, rawScore] of ratings) {
          upsertRating(db, userId, animeId, rawScore);
        }
      }
    } finally {
      db.close();
    }

    function runGraph(name: string, budgetFlag: string, budgetValue: string) {
      const graphPath = path.join(dir, `${name}.json`);
      const compactPath = path.join(dir, `${name}.compact.json`);
      const command = spawnSync(process.execPath, [
        "--import", "tsx", "src/build-graph.ts",
        "--db", dbPath,
        "--out-dataset", path.join(dir, `${name}.ratings.json`),
        "--out-graph", graphPath,
        "--out-dataset-compact", path.join(dir, `${name}.ratings.compact.json`),
        "--out-graph-compact", compactPath,
        "--max-anime-anime-edges", "1",
        "--max-pair-visits", "8",
        "--max-pair-candidates", "6",
        budgetFlag, budgetValue,
      ], { cwd: path.resolve("."), encoding: "utf8" });
      return { command, graphPath, compactPath };
    }

    const selected = runGraph("selected", "--min-pair-support", "1");
    assert.equal(selected.command.status, 0, `${selected.command.stdout}\n${selected.command.stderr}`);
    assert.match(selected.command.stdout, /8 visits, 6 candidate keys/);
    assert.match(selected.command.stdout, /5 output-limited/);
    const graph = JSON.parse(fs.readFileSync(selected.graphPath, "utf8")) as GraphData;
    assert.deepEqual(graph.edges.filter((edge) => edge.edgeType === "anime-anime"), [
      { id: "aa:3:4", source: "anime:3", target: "anime:4", edgeType: "anime-anime", weight: 1, support: 3 },
    ]);
    const compact = JSON.parse(fs.readFileSync(selected.compactPath, "utf8")) as CompactGraphData;
    assert.deepEqual(compact.aa, [[2, 3, 1, 3]]);
    const selectedReport = JSON.parse(fs.readFileSync(`${selected.graphPath}.report.json`, "utf8"));
    assert.equal(selectedReport.coverage.outputTruncated, true);
    assert.equal(selectedReport.coverage.pairKeyRecall, 1);

    const visits = runGraph("visits-blocked", "--max-pair-visits", "7");
    assert.notEqual(visits.command.status, 0);
    assert.match(visits.command.stderr, /Pair-visit budget exceeded/);
    assert.equal(fs.existsSync(visits.graphPath), false);
    assert.equal(fs.existsSync(visits.compactPath), false);
    assert.equal(fs.existsSync(`${visits.graphPath}.report.json`), false);
    const candidates = runGraph("candidates-blocked", "--max-pair-candidates", "5");
    assert.notEqual(candidates.command.status, 0);
    assert.match(candidates.command.stderr, /Candidate-key budget exceeded/);
    assert.equal(fs.existsSync(candidates.graphPath), false);
    assert.equal(fs.existsSync(candidates.compactPath), false);
    const malformed = runGraph("malformed-budget", "--max-pair-visits", "8suffix");
    assert.notEqual(malformed.command.status, 0);
    assert.match(malformed.command.stderr, /Invalid max pair visits/);
    assert.equal(fs.existsSync(malformed.graphPath), false);
    const invalidSeed = runGraph("invalid-seed", "--pair-selection-seed", "4294967296");
    assert.notEqual(invalidSeed.command.status, 0);
    assert.match(invalidSeed.command.stderr, /Invalid pair selection seed.*unsigned 32-bit/);
    assert.equal(fs.existsSync(invalidSeed.graphPath), false);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("a seeded cap uses one selected subset for both graph edge types and records coverage", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anime-graph-seeded-test-"));
  try {
    const dbPath = path.join(dir, "fixture.sqlite");
    const db = openDatabase(dbPath);
    try {
      upsertUser(db, "invented-user");
      for (let animeId = 1; animeId <= 8; animeId += 1) {
        upsertAnime(db, animeId, `Invented ${animeId}`);
        upsertRating(db, "invented-user", animeId, animeId);
      }
    } finally {
      db.close();
    }
    const graphPath = path.join(dir, "graph.json");
    const reportPath = path.join(dir, "report.json");
    const command = spawnSync(process.execPath, [
      "--import", "tsx", "src/build-graph.ts",
      "--db", dbPath,
      "--out-dataset", path.join(dir, "ratings.json"),
      "--out-graph", graphPath,
      "--out-dataset-compact", path.join(dir, "ratings.compact.json"),
      "--out-graph-compact", path.join(dir, "graph.compact.json"),
      "--out-report", reportPath,
      "--max-ratings-per-user", "3",
      "--pair-selection-seed", "17",
      "--max-pair-visits", "3",
      "--max-pair-candidates", "3",
    ], { cwd: path.resolve("."), encoding: "utf8" });
    assert.equal(command.status, 0, `${command.stdout}\n${command.stderr}`);
    const graph = JSON.parse(fs.readFileSync(graphPath, "utf8")) as GraphData;
    const selectedIds = graph.edges.filter((edge) => edge.edgeType === "user-anime")
      .map((edge) => Number(edge.target.slice("anime:".length))).sort((a, b) => a - b);
    assert.deepEqual(selectedIds, [1, 3, 5]);
    assert.deepEqual(graph.edges.filter((edge) => edge.edgeType === "anime-anime").map((edge) => edge.id), [
      `aa:${selectedIds[0]}:${selectedIds[1]}`,
      `aa:${selectedIds[0]}:${selectedIds[2]}`,
      `aa:${selectedIds[1]}:${selectedIds[2]}`,
    ]);
    const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    assert.equal(report.format, "graph-build-report-v1");
    assert.equal(report.selection.policy, "sha256-bottom-k-v1");
    assert.equal(report.selection.seed, 17);
    assert.equal(report.selection.maxRatingsPerUser, 3);
    assert.equal(report.coverage.approximationLevel, "seeded-per-user-subset");
    assert.equal(report.coverage.inputRatings, 8);
    assert.equal(report.coverage.selectedRatings, 3);
    assert.equal(report.coverage.ratingsSkipped, 5);
    assert.equal(report.coverage.potentialPairVisits, 28);
    assert.equal(report.coverage.pairVisits, 3);
    assert.equal(report.coverage.pairVisitsSkipped, 25);
    assert.equal(report.coverage.inputAnimeCount, 8);
    assert.equal(report.coverage.selectedAnimeCount, 3);
    assert.equal(report.coverage.pairKeyRecall, null);
    assert.ok(report.measurement.elapsedMs > 0);
    assert.ok(report.measurement.peakRssBytes > 0);
    assert.doesNotMatch(JSON.stringify(report), /invented-user/);
    assert.match(command.stdout, /sha256-bottom-k-v1.*seed=17/);

    const compactOnlyPath = path.join(dir, "only.compact.json");
    const compactOnly = spawnSync(process.execPath, [
      "--import", "tsx", "src/build-graph.ts",
      "--db", dbPath,
      "--out-dataset", path.join(dir, "only.ratings.json"),
      "--out-graph", path.join(dir, "only.graph.json"),
      "--out-dataset-compact", path.join(dir, "only.ratings.compact.json"),
      "--out-graph-compact", compactOnlyPath,
      "--compact-only",
      "--max-ratings-per-user", "3",
      "--pair-selection-seed", "17",
    ], { cwd: path.resolve("."), encoding: "utf8" });
    assert.equal(compactOnly.status, 0, `${compactOnly.stdout}\n${compactOnly.stderr}`);
    assert.equal(fs.existsSync(path.join(dir, "only.graph.json")), false);
    assert.equal(fs.existsSync(compactOnlyPath), true);
    assert.equal(JSON.parse(fs.readFileSync(`${compactOnlyPath}.report.json`, "utf8")).selection.seed, 17);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
