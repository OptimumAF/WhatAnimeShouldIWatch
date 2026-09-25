import path from "node:path";
import { Command } from "commander";
import { anonymizeUsername } from "./anonymize.js";
import { collectUserSnapshot } from "./collection.js";
import { openDatabase } from "./db.js";
import { fetchMalPage } from "./mal.js";
import { getRepoRoot } from "./paths.js";

interface CollectOptions {
  users?: string;
  db?: string;
  salt?: string;
  delayMs?: string;
  maxPagesPerUser?: string;
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
  .option("--delay-ms <milliseconds>", "Delay between MAL page requests per user")
  .option(
    "--max-pages-per-user <count>",
    "Maximum MAL pages per user in this run (0 = unlimited; full pages checkpoint)",
  );

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
const maxPagesPerUser = Number.parseInt(
  options.maxPagesPerUser ?? process.env.MAX_MAL_PAGES_PER_USER ?? "0",
  10,
);
if (Number.isNaN(maxPagesPerUser) || maxPagesPerUser < 0) {
  throw new Error(
    `Invalid max-pages-per-user value: ${options.maxPagesPerUser}`,
  );
}
const db = openDatabase(dbPath);
const controller = new AbortController();
process.once("SIGINT", () => controller.abort());
process.once("SIGTERM", () => controller.abort());

try {
  let incomplete = 0;
  for (const [index, username] of userNames.entries()) {
    if (controller.signal.aborted) {
      process.exitCode = 130;
      break;
    }
    const anonymizedId = anonymizeUsername(username, salt);
    const result = await collectUserSnapshot(
      db,
      anonymizedId,
      (offset, signal) => fetchMalPage(username, offset, delayMs, signal),
      { maxPages: maxPagesPerUser, signal: controller.signal },
    );
    process.stdout.write(
      `User ${index + 1}: ${result.outcome}; pages=${result.pagesFetched}, ` +
      `next offset=${result.nextOffset}, ` +
      `${result.outcome === "complete" ? "committed" : "staged"} scored=${result.scoredCount}\n`,
    );
    if (result.outcome !== "complete") incomplete += 1;
    if (result.outcome === "canceled") {
      process.exitCode = 130;
      break;
    }
  }
  if (incomplete > 0 && !process.exitCode) process.exitCode = 1;
  process.stdout.write(`Incomplete user fetches: ${incomplete}\n`);
  process.stdout.write(`Database written: ${dbPath}\n`);
} finally {
  db.close();
}
