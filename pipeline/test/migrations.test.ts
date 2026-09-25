import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import Database from "better-sqlite3";
import { collectUserSnapshot } from "../src/collection.js";
import { loadDatasetFromDb, openDatabase, recomputeNormalizedScores } from "../src/db.js";
import { DATABASE_SCHEMA_VERSION, migrateDatabase, NORMALIZED_FIELD_VERSION } from "../src/migrations.js";

const userId = "bbbbbbbbbbbbbbbbbbbbbbbb";

function tempDatabase(): { dbPath: string; cleanup(): void } {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "wasiw-migration-"));
  return { dbPath: path.join(directory, "fixture.sqlite"),
    cleanup: () => fs.rmSync(directory, {
      recursive: true, force: true, maxRetries: 10, retryDelay: 100,
    }) };
}

function createLegacyBase(db: Database.Database): void {
  db.exec(`
    CREATE TABLE users (id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE anime (id INTEGER PRIMARY KEY, title TEXT NOT NULL);
    CREATE TABLE ratings (
      user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      anime_id INTEGER NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
      raw_score REAL NOT NULL, normalized_score REAL NOT NULL DEFAULT 0,
      PRIMARY KEY (user_id, anime_id)
    );
  `);
}

test("fresh schema is versioned and reopening does not change it", () => {
  const db = openDatabase(":memory:");
  try {
    assert.equal(db.pragma("user_version", { simple: true }), DATABASE_SCHEMA_VERSION);
    migrateDatabase(db);
    assert.equal(db.pragma("user_version", { simple: true }), DATABASE_SCHEMA_VERSION);
    assert.ok(db.prepare("SELECT name FROM sqlite_master WHERE name = 'import_runs'").get());
    assert.ok(db.prepare("SELECT name FROM sqlite_master WHERE name = 'provider_anime_ids'").get());
  } finally {
    db.close();
  }
});

test("an unversioned legacy database keeps ratings and leaves unknown provenance null", () => {
  const fixture = tempDatabase();
  try {
    const old = new Database(fixture.dbPath);
    createLegacyBase(old);
    old.prepare("INSERT INTO users (id) VALUES (?)").run(userId);
    old.prepare("INSERT INTO anime (id, title) VALUES (42, 'Invented Star')").run();
    old.prepare(`INSERT INTO ratings (user_id, anime_id, raw_score, normalized_score)
      VALUES (?, 42, 8, 1.25)`).run(userId);
    old.close();

    const upgraded = openDatabase(fixture.dbPath);
    try {
      assert.equal(upgraded.pragma("user_version", { simple: true }), DATABASE_SCHEMA_VERSION);
      const row = upgraded.prepare(`SELECT raw_score AS rawScore,
        normalized_score AS normalizedScore, source_provider AS provider,
        source_anime_id AS sourceId, fetched_at AS fetchedAt,
        provider_updated_at AS providerUpdatedAt,
        normalized_field_version AS normalizedVersion FROM ratings`).get() as Record<string, unknown>;
      assert.deepEqual(row, {
        rawScore: 8, normalizedScore: 1.25, provider: null, sourceId: null,
        fetchedAt: null, providerUpdatedAt: null, normalizedVersion: 0,
      });
      assert.equal(loadDatasetFromDb(upgraded).source, "mixed-or-unverified");
      recomputeNormalizedScores(upgraded);
      assert.equal(upgraded.prepare(`SELECT normalized_field_version AS version FROM ratings`).get().version,
        NORMALIZED_FIELD_VERSION);
    } finally {
      upgraded.close();
    }
  } finally {
    fixture.cleanup();
  }
});

test("a pre-migration collection checkpoint survives and resumes without invented fetch dates", async () => {
  const fixture = tempDatabase();
  try {
    const old = new Database(fixture.dbPath);
    createLegacyBase(old);
    old.exec(`
      CREATE TABLE collection_status (
        user_id TEXT PRIMARY KEY, state TEXT NOT NULL, next_offset INTEGER NOT NULL,
        pages_fetched INTEGER NOT NULL, staged_entries INTEGER NOT NULL,
        committed_scored_count INTEGER, last_completed_at TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      CREATE TABLE collection_pages (
        user_id TEXT NOT NULL REFERENCES collection_status(user_id),
        offset INTEGER NOT NULL, entry_count INTEGER NOT NULL, scored_count INTEGER NOT NULL,
        PRIMARY KEY (user_id, offset)
      );
      CREATE TABLE collection_staged_entries (
        user_id TEXT NOT NULL REFERENCES collection_status(user_id),
        anime_id INTEGER NOT NULL, page_offset INTEGER NOT NULL,
        anime_title TEXT NOT NULL, score INTEGER NOT NULL,
        PRIMARY KEY (user_id, anime_id),
        FOREIGN KEY (user_id, page_offset) REFERENCES collection_pages(user_id, offset)
      );
    `);
    old.prepare(`INSERT INTO collection_status
      (user_id, state, next_offset, pages_fetched, staged_entries)
      VALUES (?, 'paused', 300, 1, 300)`).run(userId);
    old.prepare(`INSERT INTO collection_pages
      (user_id, offset, entry_count, scored_count) VALUES (?, 0, 300, 300)`).run(userId);
    const insert = old.prepare(`INSERT INTO collection_staged_entries
      (user_id, anime_id, page_offset, anime_title, score) VALUES (?, ?, 0, ?, 8)`);
    old.transaction(() => {
      for (let index = 0; index < 300; index += 1) {
        insert.run(userId, 1_000 + index, `Invented ${index}`);
      }
    })();
    old.close();

    const upgraded = openDatabase(fixture.dbPath);
    try {
      assert.equal(upgraded.prepare(`SELECT COUNT(*) AS count FROM collection_staged_entries`).get().count, 300);
      const result = await collectUserSnapshot(upgraded, userId, async (offset) => {
        assert.equal(offset, 300);
        return [{ anime_id: 2_000, anime_title: "Invented Finale", score: 10 }];
      }, { now: () => "2026-09-24T12:00:00.000Z" });
      assert.equal(result.outcome, "complete");
      const oldRow = upgraded.prepare(`SELECT source_provider AS provider,
        fetched_at AS fetchedAt, normalized_field_version AS normalizedVersion
        FROM ratings WHERE user_id = ? AND anime_id = 1000`).get(userId) as Record<string, unknown>;
      assert.deepEqual(oldRow, { provider: null, fetchedAt: null,
        normalizedVersion: NORMALIZED_FIELD_VERSION });
      const newRow = upgraded.prepare(`SELECT source_provider AS provider,
        fetched_at AS fetchedAt FROM ratings WHERE user_id = ? AND anime_id = 2000`
      ).get(userId) as Record<string, unknown>;
      assert.deepEqual(newRow, { provider: "mal", fetchedAt: "2026-09-24T12:00:00.000Z" });
    } finally {
      upgraded.close();
    }
  } finally {
    fixture.cleanup();
  }
});

test("a newer unknown version is rejected without changing its tables", () => {
  const fixture = tempDatabase();
  try {
    const future = new Database(fixture.dbPath);
    future.exec("CREATE TABLE future_fixture (value TEXT); PRAGMA user_version = 99;");
    future.close();
    assert.throws(() => openDatabase(fixture.dbPath), /Unsupported SQLite schema version 99/);
    const unchanged = new Database(fixture.dbPath);
    try {
      assert.equal(unchanged.pragma("user_version", { simple: true }), 99);
      assert.ok(unchanged.prepare("SELECT name FROM sqlite_master WHERE name = 'future_fixture'").get());
      assert.equal(unchanged.prepare("SELECT name FROM sqlite_master WHERE name = 'ratings'").get(), undefined);
    } finally {
      unchanged.close();
    }
  } finally {
    fixture.cleanup();
  }
});

test("a malformed legacy schema rolls back the entire migration", () => {
  const fixture = tempDatabase();
  try {
    const malformed = new Database(fixture.dbPath);
    malformed.exec("CREATE TABLE ratings (user_id TEXT); PRAGMA user_version = 0;");
    malformed.close();
    assert.throws(() => openDatabase(fixture.dbPath), /ratings is missing anime_id/);
    const unchanged = new Database(fixture.dbPath);
    try {
      assert.equal(unchanged.pragma("user_version", { simple: true }), 0);
      assert.equal(unchanged.prepare("SELECT name FROM sqlite_master WHERE name = 'users'").get(), undefined);
      assert.equal(unchanged.prepare("SELECT name FROM sqlite_master WHERE name = 'anime'").get(), undefined);
    } finally {
      unchanged.close();
    }
  } finally {
    fixture.cleanup();
  }
});

test("an orphaned legacy rating is rejected without deleting it", () => {
  const old = new Database(":memory:");
  try {
    old.pragma("foreign_keys = OFF");
    createLegacyBase(old);
    old.prepare(`INSERT INTO ratings (user_id, anime_id, raw_score)
      VALUES (?, 999, 7)`).run(userId);
    old.pragma("foreign_keys = ON");
    assert.throws(() => migrateDatabase(old), /foreign-key check failed/);
    assert.equal(old.pragma("user_version", { simple: true }), 0);
    assert.equal(old.prepare("SELECT COUNT(*) AS count FROM ratings").get().count, 1);
  } finally {
    old.close();
  }
});
