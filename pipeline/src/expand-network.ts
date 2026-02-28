import path from "node:path";
import { Command } from "commander";
import { anonymizeUsername } from "./anonymize.js";
import {
  openDatabase,
  recomputeNormalizedScores,
  upsertAnime,
  upsertRating,
  upsertUser,
} from "./db.js";
import { fetchJikanAnimeUserUpdates, fetchJikanUsersPage } from "./jikan.js";
import { fetchMalRatings } from "./mal.js";
import { getRepoRoot } from "./paths.js";

interface ExpandNetworkOptions {
  seeds?: string;
  salt?: string;
  targetTotalUsers?: string;
  db?: string;
  malDelayMs?: string;
  jikanDelayMs?: string;
  discoveryAnimePerUser?: string;
  updatesPagesPerAnime?: string;
  fallbackUsersPages?: string;
  minScoredAnime?: string;
}

interface DiscoveryRating {
  animeId: number;
  rawScore: number;
}

const repoRoot = getRepoRoot(import.meta.url);
const program = new Command()
  .argument("[seeds]", "Seed MAL usernames, comma-separated")
  .argument("[salt]", "Anonymization salt")
  .argument("[targetTotalUsers]", "Target total anonymized users in DB")
  .argument("[db]", "SQLite path")
  .option("--seeds <value>", "Seed MAL usernames, comma-separated")
  .option("--salt <value>", "Anonymization salt")
  .option("--target-total-users <count>", "Target total anonymized users in DB")
  .option("--db <path>", "SQLite path")
  .option("--mal-delay-ms <milliseconds>", "Delay between MAL requests")
  .option("--jikan-delay-ms <milliseconds>", "Delay between Jikan requests")
  .option(
    "--discovery-anime-per-user <count>",
    "Number of top-rated anime to use for discovering new users",
  )
  .option(
    "--updates-pages-per-anime <count>",
    "How many Jikan userupdates pages to scan per anime",
  )
  .option(
    "--fallback-users-pages <count>",
    "Number of Jikan /users pages to enqueue before crawl loop",
  )
  .option(
    "--min-scored-anime <count>",
    "Minimum rated anime required to keep a user",
  );

program.parse(process.argv);
const options = program.opts<ExpandNetworkOptions>();
const [argSeeds, argSalt, argTargetTotalUsers, argDb] = program.args as string[];

const dbPath = path.resolve(
  repoRoot,
  options.db ?? process.env.GRAPH_DB ?? argDb ?? "data/anime.sqlite",
);
const seedCsv = options.seeds ?? process.env.MAL_SEEDS ?? argSeeds ?? "Gigguk";
const salt =
  options.salt ?? process.env.ANON_SALT ?? argSalt ?? "change-me";
const malDelayMs = parseNonNegativeInt(
  options.malDelayMs ?? process.env.MAL_DELAY_MS ?? "800",
  "mal-delay-ms",
);
const jikanDelayMs = parseNonNegativeInt(
  options.jikanDelayMs ?? process.env.JIKAN_DELAY_MS ?? "450",
  "jikan-delay-ms",
);
const discoveryAnimePerUser = parseNonNegativeInt(
  options.discoveryAnimePerUser ??
    process.env.DISCOVERY_ANIME_PER_USER ??
    "8",
  "discovery-anime-per-user",
);
const updatesPagesPerAnime = parseNonNegativeInt(
  options.updatesPagesPerAnime ??
    process.env.UPDATES_PAGES_PER_ANIME ??
    "1",
  "updates-pages-per-anime",
);
const fallbackUsersPages = parseNonNegativeInt(
  options.fallbackUsersPages ??
    process.env.FALLBACK_USERS_PAGES ??
    "2",
  "fallback-users-pages",
);
const minScoredAnime = parseNonNegativeInt(
  options.minScoredAnime ?? process.env.MIN_SCORED_ANIME ?? "30",
  "min-scored-anime",
);

const db = openDatabase(dbPath);
const countUsersStmt = db.prepare(
  "SELECT COUNT(*) AS count FROM users",
) as unknown as {
  get(): { count: number };
};
const userExistsStmt = db.prepare(
  "SELECT 1 AS ok FROM users WHERE id = ?",
) as unknown as {
  get(userId: string): { ok: number } | undefined;
};
const existingRatingsStmt = db.prepare(
  `
    SELECT
      anime_id AS animeId,
      raw_score AS rawScore
    FROM ratings
    WHERE user_id = ?
    ORDER BY raw_score DESC, anime_id ASC
    LIMIT ?;
  `,
) as unknown as {
  all(userId: string, limit: number): DiscoveryRating[];
};

const insertTx = db.transaction(
  (
    userId: string,
    ratings: { anime_id: number; anime_title: string; score: number }[],
  ) => {
    upsertUser(db, userId);
    for (const rating of ratings) {
      upsertAnime(db, rating.anime_id, rating.anime_title);
      upsertRating(db, userId, rating.anime_id, rating.score);
    }
  },
);

try {
  const currentUsers = countUsersStmt.get().count;
  const targetTotalUsers = parseNonNegativeInt(
    options.targetTotalUsers ??
      process.env.TARGET_TOTAL_USERS ??
      argTargetTotalUsers ??
      String(currentUsers + 30),
    "target-total-users",
  );

  const queue: string[] = [];
  const queued = new Set<string>();
  const processed = new Set<string>();

  for (const seed of splitCsv(seedCsv)) {
    enqueue(seed, queue, queued, processed);
  }

  process.stdout.write(
    `Network expansion starting at ${currentUsers} users; target ${targetTotalUsers}\n`,
  );

  if (fallbackUsersPages > 0) {
    process.stdout.write(
      `Priming queue using Jikan /users pages: ${fallbackUsersPages}\n`,
    );
    for (let page = 1; page <= fallbackUsersPages; page += 1) {
      try {
        const users = await fetchJikanUsersPage(page);
        for (const username of users) {
          enqueue(username, queue, queued, processed);
        }
      } catch (error) {
        process.stderr.write(
          `Jikan /users page ${page} failed: ${(error as Error).message}\n`,
        );
      }
      if (jikanDelayMs > 0) {
        await sleep(jikanDelayMs);
      }
    }
  }

  let insertedUsers = 0;
  let skippedUsers = 0;
  let failedUsers = 0;
  let discoveredUsers = 0;

  while (queue.length > 0) {
    const totalUsers = countUsersStmt.get().count;
    if (totalUsers >= targetTotalUsers) {
      break;
    }

    const username = queue.shift();
    if (!username) {
      continue;
    }

    const key = normalizeUsername(username);
    queued.delete(key);
    if (processed.has(key)) {
      continue;
    }
    processed.add(key);

    const userId = anonymizeUsername(username, salt);
    let discoveryRatings: DiscoveryRating[] = [];

    if (userExistsStmt.get(userId)) {
      skippedUsers += 1;
      discoveryRatings = existingRatingsStmt.all(userId, discoveryAnimePerUser);
      process.stdout.write(
        `[skip] ${username} already exists; queue=${queue.length}\n`,
      );
    } else {
      try {
        const ratings = await fetchMalRatings(username, malDelayMs);
        if (ratings.length < minScoredAnime) {
          skippedUsers += 1;
          process.stdout.write(
            `[skip] ${username} has ${ratings.length} rated anime (< ${minScoredAnime})\n`,
          );
          continue;
        }

        insertTx(userId, ratings);
        insertedUsers += 1;
        discoveryRatings = ratings
          .map((rating) => ({ animeId: rating.anime_id, rawScore: rating.score }))
          .sort((left, right) => right.rawScore - left.rawScore)
          .slice(0, discoveryAnimePerUser);

        process.stdout.write(
          `[add] ${username} -> ${ratings.length} rated anime; total users ${countUsersStmt.get().count}\n`,
        );
      } catch (error) {
        failedUsers += 1;
        process.stderr.write(
          `[fail] ${username}: ${(error as Error).message}\n`,
        );
        continue;
      }
    }

    if (updatesPagesPerAnime === 0 || discoveryRatings.length === 0) {
      continue;
    }

    for (const rating of discoveryRatings.slice(0, discoveryAnimePerUser)) {
      for (let page = 1; page <= updatesPagesPerAnime; page += 1) {
        let usernames: string[] = [];
        try {
          usernames = await fetchJikanAnimeUserUpdates(rating.animeId, page);
        } catch (error) {
          process.stderr.write(
            `Jikan updates failed anime=${rating.animeId}, page=${page}: ${
              (error as Error).message
            }\n`,
          );
          break;
        }

        if (usernames.length === 0) {
          break;
        }

        for (const discovered of usernames) {
          const preLen = queue.length;
          enqueue(discovered, queue, queued, processed);
          if (queue.length > preLen) {
            discoveredUsers += 1;
          }
        }

        if (jikanDelayMs > 0) {
          await sleep(jikanDelayMs);
        }
      }
    }
  }

  process.stdout.write("Recomputing normalized scores... ");
  recomputeNormalizedScores(db);
  process.stdout.write("done\n");

  const finalUsers = countUsersStmt.get().count;
  process.stdout.write(`DB: ${dbPath}\n`);
  process.stdout.write(
    `Summary: +${insertedUsers} inserted, ${skippedUsers} skipped, ${failedUsers} failed, ${discoveredUsers} discovered, total users=${finalUsers}\n`,
  );
} finally {
  db.close();
}

function splitCsv(value: string): string[] {
  return value
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function normalizeUsername(username: string): string {
  return username.trim().toLowerCase();
}

function enqueue(
  username: string,
  queue: string[],
  queued: Set<string>,
  processed: Set<string>,
): void {
  const normalized = normalizeUsername(username);
  if (!normalized || queued.has(normalized) || processed.has(normalized)) {
    return;
  }
  queue.push(username.trim());
  queued.add(normalized);
}

function parseNonNegativeInt(raw: string, field: string): number {
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < 0) {
    throw new Error(`Invalid ${field}: ${raw}`);
  }
  return parsed;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
