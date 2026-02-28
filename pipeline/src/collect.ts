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
import { fetchMalRatings } from "./mal.js";
import { getRepoRoot } from "./paths.js";

interface CollectOptions {
  users?: string;
  db?: string;
  salt?: string;
  delayMs?: string;
}

const repoRoot = getRepoRoot(import.meta.url);

const program = new Command()
  .argument("[users]", "Comma-separated MAL usernames (positional fallback)")
  .argument("[salt]", "Anonymization salt (positional fallback)")
  .argument("[delayMs]", "Delay in ms (positional fallback)")
  .argument("[db]", "SQLite path (positional fallback)")
  .option("--users <usernames>", "Comma-separated MAL usernames")
  .option("--db <path>", "Path to SQLite database")
  .option("--salt <value>", "Salt used for username anonymization")
  .option("--delay-ms <milliseconds>", "Delay between MAL page requests per user");

program.parse(process.argv);
const options = program.opts<CollectOptions>();
const [argUsers, argSalt, argDelayMs, argDb] = program.args as string[];

const usersRaw = options.users ?? process.env.MAL_USERS ?? argUsers;
if (!usersRaw) {
  throw new Error(
    'No usernames provided. Use "--users user1,user2" or set MAL_USERS.',
  );
}

const userNames = usersRaw
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);

if (userNames.length === 0) {
  throw new Error("No usernames provided.");
}

const dbPath = path.resolve(repoRoot, options.db ?? argDb ?? "data/anime.sqlite");
const salt =
  options.salt ?? process.env.ANON_SALT ?? argSalt ?? "change-me";
const delayMs = Number.parseInt(options.delayMs ?? argDelayMs ?? "800", 10);
if (Number.isNaN(delayMs) || delayMs < 0) {
  throw new Error(`Invalid delay value: ${options.delayMs ?? argDelayMs}`);
}
const db = openDatabase(dbPath);

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
  for (const username of userNames) {
    const anonymizedId = anonymizeUsername(username, salt);
    process.stdout.write(`Fetching MAL list for "${username}"... `);
    const ratings = await fetchMalRatings(username, delayMs);
    process.stdout.write(`done (${ratings.length} scored anime)\n`);

    insertTx(anonymizedId, ratings);
  }

  process.stdout.write("Recomputing normalized scores... ");
  recomputeNormalizedScores(db);
  process.stdout.write("done\n");
  process.stdout.write(`Database written: ${dbPath}\n`);
} finally {
  db.close();
}
