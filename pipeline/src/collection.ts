import { randomUUID } from "node:crypto";
import type Database from "better-sqlite3";
import { upsertAnime, upsertUser } from "./db.js";
import { MalPageHttpError } from "./mal.js";
import { NORMALIZED_FIELD_VERSION } from "./migrations.js";

export const MAL_PAGE_SIZE = 300;
export const MAL_SOURCE_PROVIDER = "mal";
export const MAL_SOURCE_ROUTE = "myanimelist.net/animelist/{username}/load.json";

export type CollectionOutcome =
  | "in_progress" | "paused" | "complete" | "below_threshold"
  | "failed" | "unavailable" | "invalid" | "canceled";

export interface CollectionStatus {
  userId: string;
  state: CollectionOutcome;
  nextOffset: number;
  pagesFetched: number;
  stagedEntries: number;
  committedScoredCount: number | null;
  lastCompletedAt: string | null;
}

export interface CollectionResult {
  outcome: Exclude<CollectionOutcome, "in_progress">;
  nextOffset: number;
  pagesFetched: number;
  scoredCount: number;
}

export interface CollectionOptions {
  /** Page cap for this invocation. A full capped page is a checkpoint, not a snapshot. */
  maxPages?: number;
  /** Expansion can reject small complete lists without adding a user. */
  minScoredEntries?: number;
  signal?: AbortSignal;
  /** Local fetch/attempt time; never a provider viewing or rating timestamp. */
  now?: () => string;
}

export type MalPageFetcher = (offset: number, signal?: AbortSignal) => Promise<unknown>;

/** A source adapter may supply only provider-documented update times here. */
export interface VerifiedProviderPage {
  entries: unknown;
  providerUpdatedAtByAnimeId: ReadonlyMap<number, string>;
}

interface MalListEntry {
  animeId: number;
  title: string;
  score: number;
  providerUpdatedAt: string | null;
}

interface StagedRating extends MalListEntry {
  sourceProvider: string | null;
  sourceRoute: string | null;
  sourceAnimeId: string | null;
  fetchedAt: string | null;
  fetchRunId: string | null;
}

class InvalidSnapshotError extends Error {}
const DATE_TIME_WITH_ZONE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/;

export function getCollectionStatus(db: Database.Database, userId: string): CollectionStatus | undefined {
  return db.prepare(`
    SELECT user_id AS userId, state, next_offset AS nextOffset,
      pages_fetched AS pagesFetched, staged_entries AS stagedEntries,
      committed_scored_count AS committedScoredCount,
      last_completed_at AS lastCompletedAt
    FROM collection_status WHERE user_id = ?
  `).get(userId) as CollectionStatus | undefined;
}

/** Keep incomplete pages private to this database until a nonempty terminal page is validated. */
export async function collectUserSnapshot(
  db: Database.Database,
  userId: string,
  fetchPage: MalPageFetcher,
  options: CollectionOptions = {},
): Promise<CollectionResult> {
  if (!/^[a-f0-9]{24}$/.test(userId)) {
    throw new Error("Collection requires a 24-character anonymized user ID.");
  }
  const maxPages = options.maxPages ?? 0;
  const minScoredEntries = options.minScoredEntries ?? 0;
  if (!Number.isSafeInteger(maxPages) || maxPages < 0 ||
      !Number.isSafeInteger(minScoredEntries) || minScoredEntries < 0) {
    throw new Error("Collection limits must be nonnegative integers.");
  }

  const now = options.now ?? (() => new Date().toISOString());
  const { checkpoint: initialCheckpoint, runId } = startOrResume(db, userId, timestamp(now));
  let checkpoint = initialCheckpoint;
  let pagesThisRun = 0;
  while (true) {
    if (options.signal?.aborted) {
      return recordOutcome(db, userId, runId, "canceled", false, timestamp(now));
    }

    let rawPage: unknown;
    try {
      rawPage = await fetchPage(checkpoint.nextOffset, options.signal);
    } catch (error) {
      if (options.signal?.aborted || (error instanceof Error && error.name === "AbortError")) {
        return recordOutcome(db, userId, runId, "canceled", false, timestamp(now));
      }
      if (error instanceof MalPageHttpError && [401, 403, 404].includes(error.status)) {
        return recordOutcome(db, userId, runId, "unavailable", true, timestamp(now));
      }
      return recordOutcome(db, userId, runId, "failed", false, timestamp(now));
    }

    const fetchedAt = timestamp(now);
    let page: MalListEntry[];
    try {
      page = validatePage(rawPage);
    } catch (error) {
      if (error instanceof InvalidSnapshotError) {
        return recordOutcome(db, userId, runId, "invalid", true, timestamp(now));
      }
      throw error;
    }
    // An empty load.json response cannot distinguish a genuinely empty list from a
    // private/unavailable one. An exact multiple of 300 therefore stays incomplete.
    if (page.length === 0) {
      return recordOutcome(db, userId, runId, "unavailable", true, timestamp(now));
    }
    if (options.signal?.aborted) {
      return recordOutcome(db, userId, runId, "canceled", false, timestamp(now));
    }

    try {
      const result = stagePage(db, userId, checkpoint, page, minScoredEntries,
        runId, fetchedAt, timestamp(now));
      if (result) return result;
    } catch (error) {
      if (error instanceof InvalidSnapshotError) {
        return recordOutcome(db, userId, runId, "invalid", true, timestamp(now));
      }
      throw error;
    }

    pagesThisRun += 1;
    checkpoint = getCollectionStatus(db, userId)!;
    if (maxPages > 0 && pagesThisRun >= maxPages) {
      return recordOutcome(db, userId, runId, "paused", false, timestamp(now));
    }
  }
}

function startOrResume(
  db: Database.Database,
  userId: string,
  startedAt: string,
): { checkpoint: CollectionStatus; runId: string } {
  const runId = randomUUID();
  const checkpoint = db.transaction(() => {
    const previous = getCollectionStatus(db, userId);
    if (previous?.nextOffset &&
        ["in_progress", "paused", "failed", "canceled"].includes(previous.state)) {
      verifyCheckpoint(db, previous);
      db.prepare(`UPDATE collection_status SET state = 'in_progress',
        updated_at = ? WHERE user_id = ?`).run(startedAt, userId);
    } else {
      clearStage(db, userId);
      db.prepare(`
        INSERT INTO collection_status (user_id, state, updated_at) VALUES (?, 'in_progress', ?)
        ON CONFLICT(user_id) DO UPDATE SET
          state = 'in_progress', next_offset = 0, pages_fetched = 0,
          staged_entries = 0, updated_at = excluded.updated_at
      `).run(userId, startedAt);
    }
    const current = getCollectionStatus(db, userId)!;
    // A process killed between pages leaves a running row. The next attempt
    // records that interruption without pretending its pages were committed.
    db.prepare(`UPDATE import_runs SET state = 'interrupted', finished_at = ?
      WHERE user_id = ? AND state = 'running'`).run(startedAt, userId);
    db.prepare(`INSERT INTO import_runs (
      id, user_id, source_provider, source_route, started_at,
      state, start_offset, end_offset
    ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?)`
    ).run(runId, userId, MAL_SOURCE_PROVIDER, MAL_SOURCE_ROUTE,
      startedAt, current.nextOffset, current.nextOffset);
    return current;
  })();
  return { checkpoint, runId };
}

function verifyCheckpoint(db: Database.Database, status: CollectionStatus): void {
  const pageState = db.prepare(`SELECT COUNT(*) AS count,
    COALESCE(SUM(entry_count), 0) AS entries,
    COALESCE(SUM(CASE WHEN entry_count != ? OR offset % ? != 0 THEN 1 ELSE 0 END), 0) AS invalid,
    MIN(offset) AS firstOffset, MAX(offset) AS lastOffset
    FROM collection_pages WHERE user_id = ?`
  ).get(MAL_PAGE_SIZE, MAL_PAGE_SIZE, status.userId) as {
    count: number; entries: number; invalid: number;
    firstOffset: number | null; lastOffset: number | null;
  };
  const stageState = db.prepare(`SELECT COUNT(*) AS count FROM collection_staged_entries WHERE user_id = ?`
  ).get(status.userId) as { count: number };
  if (status.nextOffset !== status.pagesFetched * MAL_PAGE_SIZE ||
      pageState.count !== status.pagesFetched ||
      pageState.entries !== status.stagedEntries ||
      pageState.invalid !== 0 || pageState.firstOffset !== 0 ||
      pageState.lastOffset !== status.nextOffset - MAL_PAGE_SIZE ||
      stageState.count !== status.stagedEntries) {
    throw new Error("Collection checkpoint is inconsistent; committed ratings were not changed.");
  }
}

function validatePage(raw: unknown): MalListEntry[] {
  let updateTimes: ReadonlyMap<number, string> | undefined;
  if (raw && typeof raw === "object" && !Array.isArray(raw) &&
      "providerUpdatedAtByAnimeId" in raw &&
      raw.providerUpdatedAtByAnimeId instanceof Map && "entries" in raw) {
    const page = raw as VerifiedProviderPage;
    updateTimes = page.providerUpdatedAtByAnimeId;
    raw = page.entries;
  }
  if (!Array.isArray(raw) || raw.length > MAL_PAGE_SIZE) {
    throw new InvalidSnapshotError("MAL page must be an array of at most 300 entries.");
  }
  const ids = new Set<number>();
  const entries = raw.map((value): MalListEntry => {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new InvalidSnapshotError("MAL page contains a malformed entry.");
    }
    const entry = value as Record<string, unknown>;
    const animeId = entry.anime_id;
    const title = entry.anime_title;
    const score = entry.score;
    if (typeof animeId !== "number" || !Number.isSafeInteger(animeId) || animeId <= 0 ||
        typeof title !== "string" || !title.trim() ||
        typeof score !== "number" || !Number.isInteger(score) || score < 0 || score > 10 ||
        ids.has(animeId)) {
      throw new InvalidSnapshotError("MAL page contains invalid or duplicate anime data.");
    }
    ids.add(animeId);
    const providerUpdatedAt = updateTimes?.get(animeId);
    const parsedUpdate = providerUpdatedAt === undefined
      ? null : canonicalInstant(providerUpdatedAt);
    if (providerUpdatedAt !== undefined && parsedUpdate === null) {
      throw new InvalidSnapshotError("Verified provider update time must be a date-time.");
    }
    return { animeId, title, score, providerUpdatedAt: parsedUpdate };
  });
  if (updateTimes && [...updateTimes.keys()].some((animeId) => !ids.has(animeId))) {
    throw new InvalidSnapshotError("Provider update time references an absent anime ID.");
  }
  return entries;
}

function timestamp(now: () => string): string {
  const value = canonicalInstant(now());
  if (value === null) {
    throw new Error("Collection clock must return a valid date-time.");
  }
  return value;
}

function canonicalInstant(value: unknown): string | null {
  return typeof value === "string" && DATE_TIME_WITH_ZONE.test(value) &&
      Number.isFinite(Date.parse(value))
    ? new Date(value).toISOString() : null;
}

function stagePage(
  db: Database.Database,
  userId: string,
  checkpoint: CollectionStatus,
  page: MalListEntry[],
  minScoredEntries: number,
  runId: string,
  fetchedAt: string,
  completedAt: string,
): CollectionResult | undefined {
  return db.transaction(() => {
    assertRunning(db, runId);
    const offset = checkpoint.nextOffset;
    const scoredOnPage = page.filter((entry) => entry.score > 0).length;
    db.prepare(`INSERT INTO collection_pages (
      user_id, offset, entry_count, scored_count, fetched_at, fetch_run_id
    ) VALUES (?, ?, ?, ?, ?, ?)`
    ).run(userId, offset, page.length, scoredOnPage, fetchedAt, runId);
    const insert = db.prepare(`INSERT OR IGNORE INTO collection_staged_entries
      (user_id, anime_id, page_offset, anime_title, score, source_provider,
       source_route, source_anime_id, fetched_at, provider_updated_at, fetch_run_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
    for (const entry of page) {
      if (insert.run(userId, entry.animeId, offset, entry.title, entry.score,
        MAL_SOURCE_PROVIDER, MAL_SOURCE_ROUTE, String(entry.animeId), fetchedAt,
        entry.providerUpdatedAt, runId).changes !== 1) {
        throw new InvalidSnapshotError("MAL pages overlap; restart from the first page.");
      }
    }

    const pagesFetched = checkpoint.pagesFetched + 1;
    const stagedEntries = checkpoint.stagedEntries + page.length;
    db.prepare(`UPDATE import_runs SET pages_fetched = pages_fetched + 1,
      entries_fetched = entries_fetched + ?, end_offset = ? WHERE id = ?`
    ).run(page.length, offset + page.length, runId);
    if (page.length === MAL_PAGE_SIZE) {
      db.prepare(`UPDATE collection_status SET state = 'in_progress',
        next_offset = ?, pages_fetched = ?, staged_entries = ?,
        updated_at = ? WHERE user_id = ?`
      ).run(offset + page.length, pagesFetched, stagedEntries, completedAt, userId);
      return undefined;
    }

    const scored = db.prepare(`SELECT anime_id AS animeId, anime_title AS title, score,
      source_provider AS sourceProvider, source_route AS sourceRoute,
      source_anime_id AS sourceAnimeId, fetched_at AS fetchedAt,
      provider_updated_at AS providerUpdatedAt, fetch_run_id AS fetchRunId
      FROM collection_staged_entries WHERE user_id = ? AND score > 0 ORDER BY anime_id`
    ).all(userId) as StagedRating[];
    if (scored.length < minScoredEntries) {
      clearStage(db, userId);
      db.prepare(`UPDATE collection_status SET state = 'below_threshold',
        next_offset = 0, pages_fetched = ?, staged_entries = 0,
        updated_at = ? WHERE user_id = ?`).run(pagesFetched, completedAt, userId);
      finishRun(db, runId, "below_threshold", completedAt);
      return { outcome: "below_threshold" as const, nextOffset: 0, pagesFetched, scoredCount: scored.length };
    }

    upsertUser(db, userId);
    const findMapping = db.prepare(`SELECT anime_id AS animeId FROM provider_anime_ids
      WHERE source_provider = ? AND source_id = ?`);
    const upsertMapping = db.prepare(`INSERT INTO provider_anime_ids (
      anime_id, source_provider, source_id, first_fetched_at, last_fetched_at
    ) VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(source_provider, source_id) DO UPDATE SET
      first_fetched_at = MIN(provider_anime_ids.first_fetched_at, excluded.first_fetched_at),
      last_fetched_at = MAX(provider_anime_ids.last_fetched_at, excluded.last_fetched_at)`);
    for (const entry of scored) {
      upsertAnime(db, entry.animeId, entry.title);
      if (entry.sourceProvider && entry.sourceAnimeId && entry.fetchedAt) {
        const existing = findMapping.get(entry.sourceProvider, entry.sourceAnimeId) as
          { animeId: number } | undefined;
        if (existing && existing.animeId !== entry.animeId) {
          throw new Error("Provider anime ID maps to a different internal anime ID.");
        }
        upsertMapping.run(entry.animeId, entry.sourceProvider,
          entry.sourceAnimeId, entry.fetchedAt, entry.fetchedAt);
      }
    }
    const mean = scored.reduce((sum, entry) => sum + entry.score, 0) / Math.max(scored.length, 1);
    db.prepare("DELETE FROM ratings WHERE user_id = ?").run(userId);
    const insertRating = db.prepare(`INSERT INTO ratings
      (user_id, anime_id, raw_score, normalized_score, source_provider,
       source_route, source_anime_id, fetched_at, provider_updated_at,
       normalized_field_version, fetch_run_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
    for (const entry of scored) {
      insertRating.run(userId, entry.animeId, entry.score, entry.score - mean,
        entry.sourceProvider, entry.sourceRoute, entry.sourceAnimeId,
        entry.fetchedAt, entry.providerUpdatedAt, NORMALIZED_FIELD_VERSION,
        entry.fetchRunId);
    }
    clearStage(db, userId);
    db.prepare(`UPDATE collection_status SET state = 'complete',
      next_offset = 0, pages_fetched = ?, staged_entries = 0,
      committed_scored_count = ?, last_completed_at = ?,
      updated_at = ? WHERE user_id = ?`
    ).run(pagesFetched, scored.length, completedAt, completedAt, userId);
    finishRun(db, runId, "complete", completedAt);
    return { outcome: "complete" as const, nextOffset: 0, pagesFetched, scoredCount: scored.length };
  })();
}

function recordOutcome(
  db: Database.Database,
  userId: string,
  runId: string,
  outcome: Exclude<CollectionOutcome, "in_progress" | "complete" | "below_threshold">,
  reset: boolean,
  finishedAt: string,
): CollectionResult {
  db.transaction(() => {
    assertRunning(db, runId);
    if (reset) clearStage(db, userId);
    db.prepare(`UPDATE collection_status SET state = ?,
      next_offset = CASE WHEN ? THEN 0 ELSE next_offset END,
      staged_entries = CASE WHEN ? THEN 0 ELSE staged_entries END,
      updated_at = ? WHERE user_id = ?`
    ).run(outcome, Number(reset), Number(reset), finishedAt, userId);
    finishRun(db, runId, outcome, finishedAt);
  })();
  const status = getCollectionStatus(db, userId)!;
  const scoredCount = db.prepare(`SELECT COUNT(*) AS count FROM collection_staged_entries
    WHERE user_id = ? AND score > 0`).get(userId) as { count: number };
  return { outcome, nextOffset: status.nextOffset,
    pagesFetched: status.pagesFetched, scoredCount: scoredCount.count };
}

function assertRunning(db: Database.Database, runId: string): void {
  const row = db.prepare("SELECT state FROM import_runs WHERE id = ?").get(runId) as
    { state: string } | undefined;
  if (row?.state !== "running") {
    throw new Error("Collection attempt was superseded; committed ratings were not changed.");
  }
}

function finishRun(db: Database.Database, runId: string, state: string, finishedAt: string): void {
  db.prepare(`UPDATE import_runs SET state = ?, finished_at = ? WHERE id = ?`
  ).run(state, finishedAt, runId);
}

function clearStage(db: Database.Database, userId: string): void {
  db.prepare("DELETE FROM collection_staged_entries WHERE user_id = ?").run(userId);
  db.prepare("DELETE FROM collection_pages WHERE user_id = ?").run(userId);
}
