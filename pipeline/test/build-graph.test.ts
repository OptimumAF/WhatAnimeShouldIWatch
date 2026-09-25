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
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
