import path from "node:path";
import { spawn } from "node:child_process";
import { Command } from "commander";
import { openDatabase } from "./db.js";
import { getRepoRoot } from "./paths.js";

interface Expand100xOptions {
  db?: string;
  seeds?: string;
  salt?: string;
  factor?: string;
  targetTotalUsers?: string;
  minTargetUsers?: string;
  malDelayMs?: string;
  jikanDelayMs?: string;
  discoveryAnimePerUser?: string;
  updatesPagesPerAnime?: string;
  fallbackUsersPages?: string;
  minScoredAnime?: string;
  maxMalPagesPerUser?: string;
}

const repoRoot = getRepoRoot(import.meta.url);
const pipelineRoot = path.resolve(repoRoot, "pipeline");

const program = new Command()
  .option("--db <path>", "SQLite path", "data/anime.sqlite")
  .option("--seeds <value>", "Seed MAL usernames, comma-separated", "Gigguk,TheAnimeMan")
  .option("--salt <value>", "Anonymization salt", process.env.ANON_SALT ?? "change-me")
  .option("--factor <count>", "Target user multiplier", "100")
  .option("--target-total-users <count>", "Explicit target total users (overrides factor)")
  .option("--min-target-users <count>", "Minimum absolute target users", "200")
  .option("--mal-delay-ms <milliseconds>", "Delay between MAL requests", "350")
  .option("--jikan-delay-ms <milliseconds>", "Delay between Jikan requests", "250")
  .option(
    "--discovery-anime-per-user <count>",
    "Top rated anime per user used for discovery",
    "10",
  )
  .option(
    "--updates-pages-per-anime <count>",
    "Jikan updates pages to scan per anime",
    "1",
  )
  .option(
    "--fallback-users-pages <count>",
    "Jikan /users pages to enqueue before crawl",
    "40",
  )
  .option("--min-scored-anime <count>", "Minimum ratings required per user", "15")
  .option(
    "--max-mal-pages-per-user <count>",
    "Maximum MAL pages fetched per user (0 = unlimited)",
    "2",
  );

program.parse(process.argv);
const options = program.opts<Expand100xOptions>();

const dbPath = path.resolve(repoRoot, options.db ?? "data/anime.sqlite");
const currentUsers = getUserCount(dbPath);
const factor = parseNonNegativeInt(options.factor ?? "100", "factor");
const minTargetUsers = parseNonNegativeInt(
  options.minTargetUsers ?? "200",
  "min-target-users",
);
const explicitTargetRaw = options.targetTotalUsers;
const targetUsers = explicitTargetRaw
  ? parseNonNegativeInt(explicitTargetRaw, "target-total-users")
  : Math.max(minTargetUsers, Math.ceil(currentUsers * factor));

process.stdout.write(
  `Scale-up run: users=${currentUsers}, factor=${factor}, target=${targetUsers}, db=${dbPath}\n`,
);

if (targetUsers <= currentUsers) {
  process.stdout.write("No-op: target already satisfied.\n");
  process.exit(0);
}

const exitCode = await runExpandNetworkCli([
  "--seeds",
  options.seeds ?? "Gigguk,TheAnimeMan",
  "--salt",
  options.salt ?? "change-me",
  "--target-total-users",
  String(targetUsers),
  "--db",
  path.relative(repoRoot, dbPath),
  "--mal-delay-ms",
  options.malDelayMs ?? "350",
  "--jikan-delay-ms",
  options.jikanDelayMs ?? "250",
  "--discovery-anime-per-user",
  options.discoveryAnimePerUser ?? "10",
  "--updates-pages-per-anime",
  options.updatesPagesPerAnime ?? "1",
  "--fallback-users-pages",
  options.fallbackUsersPages ?? "40",
  "--min-scored-anime",
  options.minScoredAnime ?? "15",
  "--max-mal-pages-per-user",
  options.maxMalPagesPerUser ?? "2",
]);

if (exitCode !== 0) {
  process.exit(exitCode);
}

function getUserCount(dbPathValue: string): number {
  const db = openDatabase(dbPathValue);
  try {
    const stmt = db.prepare(
      "SELECT COUNT(*) AS count FROM users",
    ) as unknown as { get(): { count: number } };
    return stmt.get().count;
  } finally {
    db.close();
  }
}

function parseNonNegativeInt(raw: string, field: string): number {
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < 0) {
    throw new Error(`Invalid ${field}: ${raw}`);
  }
  return parsed;
}

function runExpandNetworkCli(args: string[]): Promise<number> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      ["--import", "tsx", "src/expand-network.ts", ...args],
      {
        cwd: pipelineRoot,
        stdio: "inherit",
      },
    );

    child.on("error", reject);
    child.on("close", (code) => resolve(code ?? 1));
  });
}
