import fs from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import { migrateDatabase, NORMALIZED_FIELD_VERSION } from "./migrations.js";
import type { AnonymizedDataset, AnonymizedUserRatings } from "./types.js";

export interface AnimeRatingRow {
  animeId: number;
  title: string;
  rawScore: number;
  normalizedScore: number;
}

export function openDatabase(dbPath: string): Database.Database {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);
  try {
    db.pragma("journal_mode = WAL");
    db.pragma("foreign_keys = ON");
    migrateDatabase(db);
    return db;
  } catch (error) {
    db.close();
    throw error;
  }
}

export function upsertUser(db: Database.Database, userId: string): void {
  db.prepare(
    `
      INSERT INTO users (id)
      VALUES (?)
      ON CONFLICT(id) DO NOTHING;
    `,
  ).run(userId);
}

export function upsertAnime(
  db: Database.Database,
  animeId: number,
  title: string,
): void {
  db.prepare(
    `
      INSERT INTO anime (id, title)
      VALUES (?, ?)
      ON CONFLICT(id) DO UPDATE SET title = excluded.title;
    `,
  ).run(animeId, title);
}

export function upsertRating(
  db: Database.Database,
  userId: string,
  animeId: number,
  rawScore: number,
): void {
  db.prepare(
    `
      INSERT INTO ratings (user_id, anime_id, raw_score)
      VALUES (?, ?, ?)
      ON CONFLICT(user_id, anime_id) DO UPDATE SET
        raw_score = excluded.raw_score,
        normalized_score = 0,
        normalized_field_version = 0,
        source_provider = NULL,
        source_route = NULL,
        source_anime_id = NULL,
        fetched_at = NULL,
        provider_updated_at = NULL,
        fetch_run_id = NULL;
    `,
  ).run(userId, animeId, rawScore);
}

export function recomputeNormalizedScores(db: Database.Database): void {
  const users = db.prepare("SELECT id FROM users").all() as { id: string }[];
  const avgScoreStmt = db.prepare(
    `
      SELECT AVG(raw_score) AS avg_score
      FROM ratings
      WHERE user_id = ?;
    `,
  );
  const updateStmt = db.prepare(
    `
      UPDATE ratings
      SET normalized_score = raw_score - ?, normalized_field_version = ?
      WHERE user_id = ?;
    `,
  );

  const tx = db.transaction(() => {
    for (const user of users) {
      const row = avgScoreStmt.get(user.id) as { avg_score: number | null };
      const avg = row.avg_score ?? 0;
      updateStmt.run(avg, NORMALIZED_FIELD_VERSION, user.id);
    }
  });

  tx();
}

export function loadDatasetFromDb(db: Database.Database): AnonymizedDataset {
  const sources = db.prepare(`SELECT DISTINCT source_provider AS provider,
    source_route AS route FROM ratings`).all() as
    { provider: string | null; route: string | null }[];
  const rows = db.prepare(
    `
      SELECT
        r.user_id AS userId,
        r.anime_id AS animeId,
        a.title AS title,
        r.raw_score AS rawScore,
        r.normalized_score AS normalizedScore
      FROM ratings r
      JOIN anime a ON a.id = r.anime_id
      ORDER BY r.user_id, r.anime_id;
    `,
  ).all() as AnimeRatingRowWithUser[];

  const byUser = new Map<string, AnonymizedUserRatings>();
  for (const row of rows) {
    let user = byUser.get(row.userId);
    if (!user) {
      user = { userId: row.userId, ratings: [] };
      byUser.set(row.userId, user);
    }
    user.ratings.push({
      animeId: row.animeId,
      title: row.title,
      rawScore: row.rawScore,
      normalizedScore: row.normalizedScore,
    });
  }

  return {
    generatedAt: new Date().toISOString(),
    source: sources.length === 1 && sources[0].provider && sources[0].route
      ? sources[0].route : "mixed-or-unverified",
    users: [...byUser.values()],
  };
}

interface AnimeRatingRowWithUser extends AnimeRatingRow {
  userId: string;
}
