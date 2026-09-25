import type Database from "better-sqlite3";

export const DATABASE_SCHEMA_VERSION = 3;
export const NORMALIZED_FIELD_VERSION = 1;

/** Upgrade an unversioned legacy or M2.5 database without inventing old provenance. */
export function migrateDatabase(db: Database.Database): void {
  const version = db.pragma("user_version", { simple: true }) as number;
  if (!Number.isSafeInteger(version) || version < 0 || version > DATABASE_SCHEMA_VERSION) {
    throw new Error(`Unsupported SQLite schema version ${version}; expected 0–${DATABASE_SCHEMA_VERSION}.`);
  }

  db.transaction(() => {
    if (version < 1) {
      createBaseTables(db);
      db.pragma("user_version = 1");
    }
    verifyColumns(db, "users", ["id", "created_at"]);
    verifyColumns(db, "anime", ["id", "title"]);
    verifyColumns(db, "ratings", ["user_id", "anime_id", "raw_score", "normalized_score"]);

    if (version < 2) {
      createCollectionTables(db);
      db.pragma("user_version = 2");
    }
    verifyColumns(db, "collection_status", [
      "user_id", "state", "next_offset", "pages_fetched", "staged_entries",
      "committed_scored_count", "last_completed_at", "updated_at",
    ]);
    verifyColumns(db, "collection_pages", ["user_id", "offset", "entry_count", "scored_count"]);
    verifyColumns(db, "collection_staged_entries", [
      "user_id", "anime_id", "page_offset", "anime_title", "score",
    ]);

    if (version < 3) {
      addProvenance(db);
      db.pragma("user_version = 3");
    }
    verifyColumns(db, "ratings", [
      "source_provider", "source_route", "source_anime_id", "fetched_at",
      "provider_updated_at", "normalized_field_version", "fetch_run_id",
    ]);
    verifyColumns(db, "collection_pages", ["fetched_at", "fetch_run_id"]);
    verifyColumns(db, "collection_staged_entries", [
      "source_provider", "source_route", "source_anime_id", "fetched_at",
      "provider_updated_at", "fetch_run_id",
    ]);
    verifyColumns(db, "import_runs", [
      "id", "user_id", "source_provider", "source_route", "started_at",
      "finished_at", "state", "start_offset", "end_offset",
      "pages_fetched", "entries_fetched",
    ]);
    verifyColumns(db, "provider_anime_ids", [
      "anime_id", "source_provider", "source_id", "first_fetched_at", "last_fetched_at",
    ]);
    const broken = db.pragma("foreign_key_check") as unknown[];
    if (broken.length) throw new Error("SQLite foreign-key check failed; migration was rolled back.");
  })();
}

function createBaseTables(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS anime (
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS ratings (
      user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      anime_id INTEGER NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
      raw_score REAL NOT NULL,
      normalized_score REAL NOT NULL DEFAULT 0,
      PRIMARY KEY (user_id, anime_id)
    );
  `);
}

function createCollectionTables(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS collection_status (
      user_id TEXT PRIMARY KEY,
      state TEXT NOT NULL CHECK (state IN (
        'in_progress', 'paused', 'complete', 'below_threshold',
        'failed', 'unavailable', 'invalid', 'canceled'
      )),
      next_offset INTEGER NOT NULL DEFAULT 0 CHECK (next_offset >= 0),
      pages_fetched INTEGER NOT NULL DEFAULT 0 CHECK (pages_fetched >= 0),
      staged_entries INTEGER NOT NULL DEFAULT 0 CHECK (staged_entries >= 0),
      committed_scored_count INTEGER,
      last_completed_at TEXT,
      updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS collection_pages (
      user_id TEXT NOT NULL REFERENCES collection_status(user_id) ON DELETE CASCADE,
      offset INTEGER NOT NULL CHECK (offset >= 0),
      entry_count INTEGER NOT NULL CHECK (entry_count > 0),
      scored_count INTEGER NOT NULL CHECK (scored_count >= 0),
      PRIMARY KEY (user_id, offset)
    );
    CREATE TABLE IF NOT EXISTS collection_staged_entries (
      user_id TEXT NOT NULL REFERENCES collection_status(user_id) ON DELETE CASCADE,
      anime_id INTEGER NOT NULL CHECK (anime_id > 0),
      page_offset INTEGER NOT NULL,
      anime_title TEXT NOT NULL,
      score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 10),
      PRIMARY KEY (user_id, anime_id),
      FOREIGN KEY (user_id, page_offset) REFERENCES collection_pages(user_id, offset)
    );
  `);
}

function addProvenance(db: Database.Database): void {
  addColumn(db, "ratings", "source_provider TEXT");
  addColumn(db, "ratings", "source_route TEXT");
  addColumn(db, "ratings", "source_anime_id TEXT");
  addColumn(db, "ratings", "fetched_at TEXT");
  addColumn(db, "ratings", "provider_updated_at TEXT");
  addColumn(db, "ratings", "normalized_field_version INTEGER NOT NULL DEFAULT 0");
  addColumn(db, "ratings", "fetch_run_id TEXT");
  addColumn(db, "collection_pages", "fetched_at TEXT");
  addColumn(db, "collection_pages", "fetch_run_id TEXT");
  addColumn(db, "collection_staged_entries", "source_provider TEXT");
  addColumn(db, "collection_staged_entries", "source_route TEXT");
  addColumn(db, "collection_staged_entries", "source_anime_id TEXT");
  addColumn(db, "collection_staged_entries", "fetched_at TEXT");
  addColumn(db, "collection_staged_entries", "provider_updated_at TEXT");
  addColumn(db, "collection_staged_entries", "fetch_run_id TEXT");

  db.exec(`
    CREATE TABLE IF NOT EXISTS import_runs (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      source_provider TEXT NOT NULL,
      source_route TEXT NOT NULL,
      started_at TEXT NOT NULL,
      finished_at TEXT,
      state TEXT NOT NULL CHECK (state IN (
        'running', 'paused', 'complete', 'below_threshold', 'failed',
        'unavailable', 'invalid', 'canceled', 'interrupted'
      )),
      start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
      end_offset INTEGER NOT NULL CHECK (end_offset >= 0),
      pages_fetched INTEGER NOT NULL DEFAULT 0 CHECK (pages_fetched >= 0),
      entries_fetched INTEGER NOT NULL DEFAULT 0 CHECK (entries_fetched >= 0)
    );
    CREATE INDEX IF NOT EXISTS import_runs_user_started
      ON import_runs(user_id, started_at);
    CREATE TABLE IF NOT EXISTS provider_anime_ids (
      anime_id INTEGER NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
      source_provider TEXT NOT NULL,
      source_id TEXT NOT NULL,
      first_fetched_at TEXT NOT NULL,
      last_fetched_at TEXT NOT NULL,
      PRIMARY KEY (source_provider, source_id),
      UNIQUE (anime_id, source_provider)
    );
  `);
}

function addColumn(db: Database.Database, table: string, definition: string): void {
  const column = definition.split(" ", 1)[0];
  const existing = db.pragma(`table_info(${table})`) as Array<{ name: string }>;
  if (!existing.some((item) => item.name === column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${definition}`);
  }
}

function verifyColumns(db: Database.Database, table: string, columns: string[]): void {
  const actual = new Set((db.pragma(`table_info(${table})`) as Array<{ name: string }>).map((item) => item.name));
  for (const column of columns) {
    if (!actual.has(column)) throw new Error(`SQLite table ${table} is missing ${column}; migration was rolled back.`);
  }
}
