import type Database from "better-sqlite3";
import { upsertAnime, upsertUser } from "./db.js";
import { MalPageHttpError } from "./mal.js";

export const MAL_PAGE_SIZE = 300;

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
}

export type MalPageFetcher = (offset: number, signal?: AbortSignal) => Promise<unknown>;

interface MalListEntry {
  animeId: number;
  title: string;
  score: number;
}

class InvalidSnapshotError extends Error {}

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

  let checkpoint = startOrResume(db, userId);
  let pagesThisRun = 0;
  while (true) {
    if (options.signal?.aborted) return recordOutcome(db, userId, "canceled", false);

    let rawPage: unknown;
    try {
      rawPage = await fetchPage(checkpoint.nextOffset, options.signal);
    } catch (error) {
      if (options.signal?.aborted || (error instanceof Error && error.name === "AbortError")) {
        return recordOutcome(db, userId, "canceled", false);
      }
      if (error instanceof MalPageHttpError && [401, 403, 404].includes(error.status)) {
        return recordOutcome(db, userId, "unavailable", true);
      }
      return recordOutcome(db, userId, "failed", false);
    }

    let page: MalListEntry[];
    try {
      page = validatePage(rawPage);
    } catch (error) {
      if (error instanceof InvalidSnapshotError) {
        return recordOutcome(db, userId, "invalid", true);
      }
      throw error;
    }
    // An empty load.json response cannot distinguish a genuinely empty list from a
    // private/unavailable one. An exact multiple of 300 therefore stays incomplete.
    if (page.length === 0) return recordOutcome(db, userId, "unavailable", true);
    if (options.signal?.aborted) return recordOutcome(db, userId, "canceled", false);

    try {
      const result = stagePage(db, userId, checkpoint, page, minScoredEntries);
      if (result) return result;
    } catch (error) {
      if (error instanceof InvalidSnapshotError) {
        return recordOutcome(db, userId, "invalid", true);
      }
      throw error;
    }

    pagesThisRun += 1;
    checkpoint = getCollectionStatus(db, userId)!;
    if (maxPages > 0 && pagesThisRun >= maxPages) {
      return recordOutcome(db, userId, "paused", false);
    }
  }
}

function startOrResume(db: Database.Database, userId: string): CollectionStatus {
  const previous = getCollectionStatus(db, userId);
  if (previous?.nextOffset &&
      ["in_progress", "paused", "failed", "canceled"].includes(previous.state)) {
    verifyCheckpoint(db, previous);
    db.prepare(`UPDATE collection_status SET state = 'in_progress',
      updated_at = CURRENT_TIMESTAMP WHERE user_id = ?`).run(userId);
    return { ...previous, state: "in_progress" };
  }

  db.transaction(() => {
    clearStage(db, userId);
    db.prepare(`
      INSERT INTO collection_status (user_id, state) VALUES (?, 'in_progress')
      ON CONFLICT(user_id) DO UPDATE SET
        state = 'in_progress', next_offset = 0, pages_fetched = 0,
        staged_entries = 0, updated_at = CURRENT_TIMESTAMP
    `).run(userId);
  })();
  return getCollectionStatus(db, userId)!;
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
  if (!Array.isArray(raw) || raw.length > MAL_PAGE_SIZE) {
    throw new InvalidSnapshotError("MAL page must be an array of at most 300 entries.");
  }
  const ids = new Set<number>();
  return raw.map((value) => {
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
    return { animeId, title, score };
  });
}

function stagePage(
  db: Database.Database,
  userId: string,
  checkpoint: CollectionStatus,
  page: MalListEntry[],
  minScoredEntries: number,
): CollectionResult | undefined {
  return db.transaction(() => {
    const offset = checkpoint.nextOffset;
    const scoredOnPage = page.filter((entry) => entry.score > 0).length;
    db.prepare(`INSERT INTO collection_pages (user_id, offset, entry_count, scored_count)
      VALUES (?, ?, ?, ?)`).run(userId, offset, page.length, scoredOnPage);
    const insert = db.prepare(`INSERT OR IGNORE INTO collection_staged_entries
      (user_id, anime_id, page_offset, anime_title, score) VALUES (?, ?, ?, ?, ?)`);
    for (const entry of page) {
      if (insert.run(userId, entry.animeId, offset, entry.title, entry.score).changes !== 1) {
        throw new InvalidSnapshotError("MAL pages overlap; restart from the first page.");
      }
    }

    const pagesFetched = checkpoint.pagesFetched + 1;
    const stagedEntries = checkpoint.stagedEntries + page.length;
    if (page.length === MAL_PAGE_SIZE) {
      db.prepare(`UPDATE collection_status SET state = 'in_progress',
        next_offset = ?, pages_fetched = ?, staged_entries = ?,
        updated_at = CURRENT_TIMESTAMP WHERE user_id = ?`
      ).run(offset + page.length, pagesFetched, stagedEntries, userId);
      return undefined;
    }

    const scored = db.prepare(`SELECT anime_id AS animeId, anime_title AS title, score
      FROM collection_staged_entries WHERE user_id = ? AND score > 0 ORDER BY anime_id`
    ).all(userId) as MalListEntry[];
    if (scored.length < minScoredEntries) {
      clearStage(db, userId);
      db.prepare(`UPDATE collection_status SET state = 'below_threshold',
        next_offset = 0, pages_fetched = ?, staged_entries = 0,
        updated_at = CURRENT_TIMESTAMP WHERE user_id = ?`).run(pagesFetched, userId);
      return { outcome: "below_threshold" as const, nextOffset: 0, pagesFetched, scoredCount: scored.length };
    }

    upsertUser(db, userId);
    for (const entry of scored) upsertAnime(db, entry.animeId, entry.title);
    const mean = scored.reduce((sum, entry) => sum + entry.score, 0) / Math.max(scored.length, 1);
    db.prepare("DELETE FROM ratings WHERE user_id = ?").run(userId);
    const insertRating = db.prepare(`INSERT INTO ratings
      (user_id, anime_id, raw_score, normalized_score) VALUES (?, ?, ?, ?)`);
    for (const entry of scored) {
      insertRating.run(userId, entry.animeId, entry.score, entry.score - mean);
    }
    clearStage(db, userId);
    db.prepare(`UPDATE collection_status SET state = 'complete',
      next_offset = 0, pages_fetched = ?, staged_entries = 0,
      committed_scored_count = ?, last_completed_at = CURRENT_TIMESTAMP,
      updated_at = CURRENT_TIMESTAMP WHERE user_id = ?`
    ).run(pagesFetched, scored.length, userId);
    return { outcome: "complete" as const, nextOffset: 0, pagesFetched, scoredCount: scored.length };
  })();
}

function recordOutcome(
  db: Database.Database,
  userId: string,
  outcome: Exclude<CollectionOutcome, "in_progress" | "complete" | "below_threshold">,
  reset: boolean,
): CollectionResult {
  db.transaction(() => {
    if (reset) clearStage(db, userId);
    db.prepare(`UPDATE collection_status SET state = ?,
      next_offset = CASE WHEN ? THEN 0 ELSE next_offset END,
      staged_entries = CASE WHEN ? THEN 0 ELSE staged_entries END,
      updated_at = CURRENT_TIMESTAMP WHERE user_id = ?`
    ).run(outcome, Number(reset), Number(reset), userId);
  })();
  const status = getCollectionStatus(db, userId)!;
  const scoredCount = db.prepare(`SELECT COUNT(*) AS count FROM collection_staged_entries
    WHERE user_id = ? AND score > 0`).get(userId) as { count: number };
  return { outcome, nextOffset: status.nextOffset,
    pagesFetched: status.pagesFetched, scoredCount: scoredCount.count };
}

function clearStage(db: Database.Database, userId: string): void {
  db.prepare("DELETE FROM collection_staged_entries WHERE user_id = ?").run(userId);
  db.prepare("DELETE FROM collection_pages WHERE user_id = ?").run(userId);
}
