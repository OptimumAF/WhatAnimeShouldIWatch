import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import type Database from "better-sqlite3";
import { ProviderScheduler } from "../../shared/provider-scheduler.js";
import {
  collectUserSnapshot, getCollectionStatus, MAL_PAGE_SIZE,
  MAL_SOURCE_PROVIDER, MAL_SOURCE_ROUTE,
} from "../src/collection.js";
import { loadDatasetFromDb, openDatabase, upsertAnime, upsertRating, upsertUser } from "../src/db.js";
import { fetchMalPage, MalPageHttpError } from "../src/mal.js";
import { NORMALIZED_FIELD_VERSION } from "../src/migrations.js";

const userId = "aaaaaaaaaaaaaaaaaaaaaaaa";
type Entry = { anime_id: number; anime_title: string; score: number };
const entry = (animeId: number, score: number): Entry => ({
  anime_id: animeId, anime_title: `Invented ${animeId}`, score,
});
const fullPage = (start = 1_000): Entry[] => Array.from({ length: MAL_PAGE_SIZE },
  (_unused, index) => entry(start + index, 8));

function rows(db: Database.Database): Array<{ animeId: number; rawScore: number; normalizedScore: number }> {
  return db.prepare(`SELECT anime_id AS animeId, raw_score AS rawScore,
    normalized_score AS normalizedScore FROM ratings WHERE user_id = ? ORDER BY anime_id`
  ).all(userId) as Array<{ animeId: number; rawScore: number; normalizedScore: number }>;
}

function seed(db: Database.Database): void {
  upsertUser(db, userId);
  for (const item of [entry(11, 6), entry(22, 8)]) {
    upsertAnime(db, item.anime_id, item.anime_title);
    upsertRating(db, userId, item.anime_id, item.score);
  }
}

test("complete re-import is idempotent and atomically changes scores and removes entries", async () => {
  const db = openDatabase(":memory:");
  try {
    seed(db);
    const next = [entry(11, 9), entry(33, 7)];
    for (let index = 0; index < 2; index += 1) {
      const result = await collectUserSnapshot(db, userId, async (offset) => {
        assert.equal(offset, 0);
        return next;
      });
      assert.deepEqual(result, {
        outcome: "complete", nextOffset: 0, pagesFetched: 1, scoredCount: 2,
      });
      assert.deepEqual(rows(db), [
        { animeId: 11, rawScore: 9, normalizedScore: 1 },
        { animeId: 33, rawScore: 7, normalizedScore: -1 },
      ]);
      assert.equal(getCollectionStatus(db, userId)?.committedScoredCount, 2);
      assert.equal(db.prepare("SELECT COUNT(*) AS count FROM collection_pages").get().count, 0);
    }
  } finally {
    db.close();
  }
});

test("a capped full page checkpoints, survives database reopen, and resumes without refetching", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "wasiw-collection-"));
  const dbPath = path.join(directory, "fixture.sqlite");
  let db = openDatabase(dbPath);
  try {
    seed(db);
    const result = await collectUserSnapshot(db, userId, async (offset) => {
      assert.equal(offset, 0);
      return fullPage();
    }, { maxPages: 1 });
    assert.equal(result.outcome, "paused");
    assert.equal(result.nextOffset, 300);
    assert.deepEqual(rows(db).map((row) => row.animeId), [11, 22]);
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM collection_pages").get().count, 1);
    db.close();

    db = openDatabase(dbPath);
    const resumed = await collectUserSnapshot(db, userId, async (offset) => {
      assert.equal(offset, 300);
      return [entry(2_000, 10)];
    });
    assert.equal(resumed.outcome, "complete");
    assert.equal(resumed.pagesFetched, 2);
    assert.equal(resumed.scoredCount, 301);
    assert.equal(rows(db).length, 301);
    assert.equal(rows(db).some((row) => row.animeId === 11), false);
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM collection_staged_entries").get().count, 0);
  } finally {
    if (db.open) db.close();
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("a failed or canceled later page preserves committed ratings and its resumable checkpoint", async () => {
  const db = openDatabase(":memory:");
  try {
    seed(db);
    const offsets: number[] = [];
    const failed = await collectUserSnapshot(db, userId, async (offset) => {
      offsets.push(offset);
      if (offset === 0) return fullPage();
      throw new Error("synthetic timeout");
    });
    assert.equal(failed.outcome, "failed");
    assert.deepEqual(offsets, [0, 300]);
    assert.equal(getCollectionStatus(db, userId)?.nextOffset, 300);
    assert.deepEqual(rows(db).map((row) => row.animeId), [11, 22]);

    const controller = new AbortController();
    const canceled = await collectUserSnapshot(db, userId, async (offset) => {
      assert.equal(offset, 300);
      controller.abort();
      throw Object.assign(new Error("synthetic cancellation"), { name: "AbortError" });
    }, { signal: controller.signal });
    assert.equal(canceled.outcome, "canceled");
    assert.equal(canceled.nextOffset, 300);
    assert.deepEqual(rows(db).map((row) => row.animeId), [11, 22]);

    const complete = await collectUserSnapshot(db, userId, async (offset) => {
      assert.equal(offset, 300);
      return [entry(2_000, 9)];
    });
    assert.equal(complete.outcome, "complete");
    assert.equal(rows(db).length, 301);
    assert.deepEqual(db.prepare("SELECT state FROM import_runs ORDER BY rowid").all(), [
      { state: "failed" }, { state: "canceled" }, { state: "complete" },
    ]);
  } finally {
    db.close();
  }
});

test("a later failed fetch records its outcome while retaining the last completion", async () => {
  const db = openDatabase(":memory:");
  try {
    await collectUserSnapshot(db, userId, async () => [entry(77, 9)]);
    const completedAt = getCollectionStatus(db, userId)?.lastCompletedAt;
    assert.ok(completedAt);
    const failed = await collectUserSnapshot(db, userId, async () => {
      throw new Error("synthetic timeout");
    });
    assert.equal(failed.outcome, "failed");
    assert.deepEqual(rows(db).map((row) => row.animeId), [77]);
    assert.deepEqual(getCollectionStatus(db, userId), {
      userId, state: "failed", nextOffset: 0, pagesFetched: 0,
      stagedEntries: 0, committedScoredCount: 1, lastCompletedAt: completedAt,
    });
  } finally {
    db.close();
  }
});

test("empty, private, and malformed pages cannot erase the last complete ratings", async () => {
  const db = openDatabase(":memory:");
  try {
    seed(db);
    const before = rows(db);
    for (const fetcher of [
      async () => [],
      async () => { throw new MalPageHttpError(403); },
      async () => ({ private: true }),
      async () => [entry(44, 11)],
    ]) {
      const result = await collectUserSnapshot(db, userId, fetcher);
      assert.ok(["unavailable", "invalid"].includes(result.outcome));
      assert.deepEqual(rows(db), before);
      assert.equal(getCollectionStatus(db, userId)?.nextOffset, 0);
      assert.deepEqual(db.prepare("SELECT state FROM import_runs ORDER BY rowid DESC LIMIT 1").get(),
        { state: result.outcome });
    }

    const exactMultiple = await collectUserSnapshot(db, userId,
      async (offset) => offset === 0 ? fullPage() : []);
    assert.equal(exactMultiple.outcome, "unavailable");
    assert.equal(exactMultiple.pagesFetched, 1);
    assert.deepEqual(rows(db), before);
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM collection_pages").get().count, 0);
  } finally {
    db.close();
  }
});

test("a damaged checkpoint refuses to resume without changing committed ratings", async () => {
  const db = openDatabase(":memory:");
  try {
    seed(db);
    await collectUserSnapshot(db, userId, async () => fullPage(), { maxPages: 1 });
    db.prepare(`DELETE FROM collection_staged_entries
      WHERE user_id = ? AND anime_id = ?`).run(userId, 1_000);
    await assert.rejects(collectUserSnapshot(db, userId, async () => {
      throw new Error("must not fetch after corrupt checkpoint");
    }), /checkpoint is inconsistent/);
    assert.deepEqual(rows(db).map((row) => row.animeId), [11, 22]);
    assert.equal(getCollectionStatus(db, userId)?.state, "paused");
  } finally {
    db.close();
  }
});

test("overlapping pages are rejected and a later fresh run starts at zero", async () => {
  const db = openDatabase(":memory:");
  try {
    seed(db);
    const invalid = await collectUserSnapshot(db, userId,
      async (offset) => offset === 0 ? fullPage() : [entry(1_000, 9)]);
    assert.equal(invalid.outcome, "invalid");
    assert.deepEqual(rows(db).map((row) => row.animeId), [11, 22]);
    const fresh = await collectUserSnapshot(db, userId, async (offset) => {
      assert.equal(offset, 0);
      return [entry(55, 9)];
    });
    assert.equal(fresh.outcome, "complete");
    assert.deepEqual(rows(db).map((row) => row.animeId), [55]);
  } finally {
    db.close();
  }
});

test("a complete but below-threshold list is recorded without adding a user", async () => {
  const db = openDatabase(":memory:");
  try {
    const result = await collectUserSnapshot(db, userId,
      async () => [entry(77, 9)], { minScoredEntries: 2 });
    assert.equal(result.outcome, "below_threshold");
    assert.equal(result.scoredCount, 1);
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM users").get().count, 0);
    assert.equal(getCollectionStatus(db, userId)?.state, "below_threshold");
  } finally {
    db.close();
  }
});

test("a nonempty all-unscored list can remove prior scored entries", async () => {
  const db = openDatabase(":memory:");
  try {
    seed(db);
    const result = await collectUserSnapshot(db, userId, async () => [entry(11, 0), entry(22, 0)]);
    assert.equal(result.outcome, "complete");
    assert.equal(result.scoredCount, 0);
    assert.deepEqual(rows(db), []);
  } finally {
    db.close();
  }
});

test("a database error during replacement rolls back deletions and the new page", async () => {
  const db = openDatabase(":memory:");
  try {
    seed(db);
    const before = rows(db);
    db.exec(`CREATE TRIGGER reject_fixture_rating BEFORE INSERT ON ratings
      WHEN NEW.anime_id = 999 BEGIN SELECT RAISE(ABORT, 'synthetic insert failure'); END;`);
    await assert.rejects(collectUserSnapshot(db, userId, async () => [entry(999, 9)]),
      /synthetic insert failure/);
    assert.deepEqual(rows(db), before);
    assert.equal(getCollectionStatus(db, userId)?.state, "in_progress");
    assert.equal(db.prepare("SELECT COUNT(*) AS count FROM collection_pages").get().count, 0);
    db.exec("DROP TRIGGER reject_fixture_rating");
    const retry = await collectUserSnapshot(db, userId, async () => [entry(999, 9)]);
    assert.equal(retry.outcome, "complete");
    assert.deepEqual(rows(db).map((row) => row.animeId), [999]);
    assert.deepEqual(db.prepare("SELECT state FROM import_runs ORDER BY rowid").all(), [
      { state: "interrupted" }, { state: "complete" },
    ]);
  } finally {
    db.close();
  }
});

test("the collector and MAL adapter share a mocked scheduled page request", async () => {
  const db = openDatabase(":memory:");
  try {
    let elapsed = 0;
    const requested: string[] = [];
    const scheduler = new ProviderScheduler({
      fetch: async (url) => {
        requested.push(url);
        return new Response(JSON.stringify([entry(123, 9)]));
      },
      monotonicNow: () => elapsed,
      wallNow: () => 0,
      sleep: async (ms) => { elapsed += ms; },
      random: () => 0.5,
    });
    const result = await collectUserSnapshot(db, userId,
      (offset, signal) => fetchMalPage("invented-fixture", offset, 0, signal, scheduler));
    assert.equal(result.outcome, "complete");
    assert.equal(requested.length, 1);
    assert.equal(new URL(requested[0]).searchParams.get("offset"), "0");
    assert.deepEqual(rows(db).map((row) => row.animeId), [123]);
  } finally {
    db.close();
  }
});

test("new ratings retain provider IDs, fetch time, normalization version, and a completed import run", async () => {
  const db = openDatabase(":memory:");
  const fetchedAt = "2026-09-24T12:00:00.000Z";
  try {
    const result = await collectUserSnapshot(db, userId,
      async () => [{ ...entry(123, 9), updated_at: "2025-01-01T01:00:00Z" }],
      { now: () => fetchedAt });
    assert.equal(result.outcome, "complete");
    const rating = db.prepare(`SELECT source_provider AS provider,
      source_route AS route, source_anime_id AS sourceId,
      fetched_at AS fetchedAt, provider_updated_at AS providerUpdatedAt,
      normalized_field_version AS normalizedVersion, fetch_run_id AS fetchRunId
      FROM ratings WHERE user_id = ? AND anime_id = 123`).get(userId) as Record<string, unknown>;
    assert.equal(rating.provider, MAL_SOURCE_PROVIDER);
    assert.equal(rating.route, MAL_SOURCE_ROUTE);
    assert.equal(rating.sourceId, "123");
    assert.equal(rating.fetchedAt, fetchedAt);
    assert.equal(rating.providerUpdatedAt, null);
    assert.equal(rating.normalizedVersion, NORMALIZED_FIELD_VERSION);
    assert.equal(typeof rating.fetchRunId, "string");

    const run = db.prepare(`SELECT id, source_provider AS provider,
      source_route AS route, started_at AS startedAt, finished_at AS finishedAt,
      state, start_offset AS startOffset, end_offset AS endOffset,
      pages_fetched AS pagesFetched, entries_fetched AS entriesFetched
      FROM import_runs`).get() as Record<string, unknown>;
    assert.deepEqual(run, {
      id: rating.fetchRunId, provider: MAL_SOURCE_PROVIDER, route: MAL_SOURCE_ROUTE,
      startedAt: fetchedAt, finishedAt: fetchedAt, state: "complete",
      startOffset: 0, endOffset: 1, pagesFetched: 1, entriesFetched: 1,
    });
    assert.deepEqual(db.prepare(`SELECT anime_id AS animeId,
      source_provider AS provider, source_id AS sourceId,
      first_fetched_at AS firstFetchedAt, last_fetched_at AS lastFetchedAt
      FROM provider_anime_ids`).get(), {
      animeId: 123, provider: MAL_SOURCE_PROVIDER, sourceId: "123",
      firstFetchedAt: fetchedAt, lastFetchedAt: fetchedAt,
    });
    assert.equal(loadDatasetFromDb(db).source, MAL_SOURCE_ROUTE);
    await collectUserSnapshot(db, userId, async () => [entry(123, 8)], {
      now: () => "2026-09-25T12:00:00.000Z",
    });
    assert.deepEqual(db.prepare(`SELECT first_fetched_at AS firstFetchedAt,
      last_fetched_at AS lastFetchedAt FROM provider_anime_ids
      WHERE source_provider = 'mal' AND source_id = '123'`).get(), {
      firstFetchedAt: fetchedAt, lastFetchedAt: "2026-09-25T12:00:00.000Z",
    });
    assert.deepEqual(db.prepare(`SELECT fetched_at AS fetchedAt,
      normalized_field_version AS version FROM ratings WHERE user_id = ?`
    ).get(userId), {
      fetchedAt: "2026-09-25T12:00:00.000Z", version: NORMALIZED_FIELD_VERSION,
    });
    await collectUserSnapshot(db, userId, async () => [entry(123, 8)], {
      now: () => "2026-09-23T12:00:00.000Z",
    });
    assert.deepEqual(db.prepare(`SELECT first_fetched_at AS firstFetchedAt,
      last_fetched_at AS lastFetchedAt FROM provider_anime_ids
      WHERE source_provider = 'mal' AND source_id = '123'`).get(), {
      firstFetchedAt: "2026-09-23T12:00:00.000Z",
      lastFetchedAt: "2026-09-25T12:00:00.000Z",
    });
    upsertRating(db, userId, 123, 7);
    assert.deepEqual(db.prepare(`SELECT raw_score AS rawScore,
      normalized_score AS normalizedScore, normalized_field_version AS version,
      source_provider AS provider, fetched_at AS fetchedAt,
      provider_updated_at AS providerUpdatedAt, fetch_run_id AS fetchRunId
      FROM ratings WHERE user_id = ? AND anime_id = 123`).get(userId), {
      rawScore: 7, normalizedScore: 0, version: 0, provider: null,
      fetchedAt: null, providerUpdatedAt: null, fetchRunId: null,
    });
    assert.equal(loadDatasetFromDb(db).source, "mixed-or-unverified");
  } finally {
    db.close();
  }
});

test("provider update time is stored only from an explicit verified map", async () => {
  const db = openDatabase(":memory:");
  try {
    const result = await collectUserSnapshot(db, userId, async () => ({
      entries: [entry(124, 8)],
      providerUpdatedAtByAnimeId: new Map([[124, "2026-09-23T11:00:00+01:00"]]),
    }), { now: () => "2026-09-24T12:00:00.000Z" });
    assert.equal(result.outcome, "complete");
    const row = db.prepare(`SELECT fetched_at AS fetchedAt,
      provider_updated_at AS providerUpdatedAt FROM ratings WHERE user_id = ?`
    ).get(userId) as Record<string, unknown>;
    assert.deepEqual(row, {
      fetchedAt: "2026-09-24T12:00:00.000Z",
      providerUpdatedAt: "2026-09-23T10:00:00.000Z",
    });

    const invalid = await collectUserSnapshot(db, userId, async () => ({
      entries: [entry(125, 7)],
      providerUpdatedAtByAnimeId: new Map([[999, "2026-09-23T00:00:00Z"]]),
    }));
    assert.equal(invalid.outcome, "invalid");
    const dateOnly = await collectUserSnapshot(db, userId, async () => ({
      entries: [entry(125, 7)],
      providerUpdatedAtByAnimeId: new Map([[125, "2026-09-23"]]),
    }));
    assert.equal(dateOnly.outcome, "invalid");
    assert.deepEqual(rows(db).map((item) => item.animeId), [124]);
  } finally {
    db.close();
  }
});

test("a resumed snapshot retains the fetch run and fetch time for each page", async () => {
  const db = openDatabase(":memory:");
  try {
    const first = await collectUserSnapshot(db, userId, async () => fullPage(), {
      maxPages: 1, now: () => "2026-09-24T12:00:00.000Z",
    });
    assert.equal(first.outcome, "paused");
    const second = await collectUserSnapshot(db, userId,
      async (offset) => {
        assert.equal(offset, 300);
        return [entry(2_000, 9)];
      }, { now: () => "2026-09-24T13:00:00.000Z" });
    assert.equal(second.outcome, "complete");
    const runs = db.prepare(`SELECT id, state, start_offset AS startOffset,
      end_offset AS endOffset, pages_fetched AS pagesFetched
      FROM import_runs ORDER BY started_at`).all() as Array<Record<string, unknown>>;
    assert.equal(runs.length, 2);
    assert.deepEqual(runs.map(({ state, startOffset, endOffset, pagesFetched }) =>
      ({ state, startOffset, endOffset, pagesFetched })), [
      { state: "paused", startOffset: 0, endOffset: 300, pagesFetched: 1 },
      { state: "complete", startOffset: 300, endOffset: 301, pagesFetched: 1 },
    ]);
    const ratings = db.prepare(`SELECT anime_id AS animeId, fetched_at AS fetchedAt,
      fetch_run_id AS fetchRunId FROM ratings WHERE user_id = ?
      AND anime_id IN (1000, 2000) ORDER BY anime_id`).all(userId) as Array<Record<string, unknown>>;
    assert.deepEqual(ratings, [
      { animeId: 1_000, fetchedAt: "2026-09-24T12:00:00.000Z", fetchRunId: runs[0].id },
      { animeId: 2_000, fetchedAt: "2026-09-24T13:00:00.000Z", fetchRunId: runs[1].id },
    ]);
  } finally {
    db.close();
  }
});

test("a superseded concurrent attempt cannot replace a newer completed snapshot", async () => {
  const db = openDatabase(":memory:");
  try {
    let releaseFirst: ((value: unknown) => void) | undefined;
    const first = collectUserSnapshot(db, userId,
      () => new Promise<unknown>((resolve) => { releaseFirst = resolve; }),
      { now: () => "2026-09-24T12:00:00.000Z" });
    assert.ok(releaseFirst);
    const second = await collectUserSnapshot(db, userId,
      async () => [entry(777, 9)],
      { now: () => "2026-09-24T13:00:00.000Z" });
    assert.equal(second.outcome, "complete");
    releaseFirst([entry(888, 8)]);
    await assert.rejects(first, /superseded/);
    assert.deepEqual(rows(db).map((item) => item.animeId), [777]);
    assert.deepEqual(db.prepare("SELECT state FROM import_runs ORDER BY started_at").all(), [
      { state: "interrupted" }, { state: "complete" },
    ]);
  } finally {
    db.close();
  }
});
