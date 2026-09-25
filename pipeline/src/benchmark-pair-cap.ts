import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { aggregateAnimePairs, PAIR_CAP_POLICY, type PairUser } from "./core/pair-aggregation.js";

// Invented, deterministic medium fixture. No SQLite, provider, or user history is read.
const recipe = {
  users: 250,
  anime: 180,
  ratingsPerUser: 48,
  cap: 24,
  selectionSeed: 17,
  generator: "modular-ids-centered-scores-v1",
};

function makeUsers(): PairUser[] {
  return Array.from({ length: recipe.users }, (_, userIndex) => {
    const rawScores = Array.from({ length: recipe.ratingsPerUser }, (_, ratingIndex) =>
      ((userIndex * 7 + ratingIndex * 13) % 21) - 10,
    );
    const mean = rawScores.reduce((sum, score) => sum + score, 0) / rawScores.length;
    return {
      userId: `invented-${userIndex}`,
      ratings: rawScores.map((score, ratingIndex) => ({
        animeId: 1 + ((userIndex * 37 + ratingIndex * 53) % recipe.anime),
        normalizedScore: score - mean,
      })),
    };
  });
}

interface ChildResult {
  mode: "exact" | "capped";
  stats: ReturnType<typeof aggregateAnimePairs>["stats"];
  keys: string[];
  elapsedMs: number;
  peakRssBytes: number;
}

function childRun(mode: ChildResult["mode"]): void {
  const users = makeUsers();
  const startedAt = performance.now();
  const result = aggregateAnimePairs(users, mode === "exact" ? 0 : recipe.cap, 0, {
    selectionSeed: recipe.selectionSeed,
    maxPairVisits: 300_000,
    maxCandidatePairs: 20_000,
  });
  const elapsedMs = Number((performance.now() - startedAt).toFixed(3));
  const maxRssKiB = process.resourceUsage().maxRSS;
  if (maxRssKiB <= 0) {
    throw new Error("This benchmark requires process.resourceUsage().maxRSS for peak RSS.");
  }
  const childResult: ChildResult = {
    mode,
    stats: result.stats,
    keys: [...result.pairs.keys()],
    elapsedMs,
    peakRssBytes: maxRssKiB * 1024,
  };
  process.stdout.write(JSON.stringify(childResult));
}

function runChild(mode: ChildResult["mode"]): ChildResult {
  const result = spawnSync(process.execPath, [
    "--import", "tsx", fileURLToPath(import.meta.url), "--child", mode,
  ], { encoding: "utf8", maxBuffer: 10 * 1024 * 1024 });
  if (result.status !== 0) {
    throw new Error(`Synthetic ${mode} benchmark failed: ${result.stderr}`);
  }
  return JSON.parse(result.stdout) as ChildResult;
}

if (process.argv[2] === "--child") {
  if (process.argv[3] !== "exact" && process.argv[3] !== "capped") {
    throw new Error("Invalid synthetic benchmark mode.");
  }
  childRun(process.argv[3]);
} else if (process.argv.length === 2) {
  const exact = runChild("exact");
  const capped = runChild("capped");
  const exactKeys = new Set(exact.keys);
  const sharedKeys = capped.keys.filter((key) => exactKeys.has(key)).length;
  const report = {
    format: "synthetic-pair-cap-benchmark-v1",
    recipe,
    policy: PAIR_CAP_POLICY,
    exact: { stats: exact.stats, elapsedMs: exact.elapsedMs, peakRssBytes: exact.peakRssBytes },
    capped: { stats: capped.stats, elapsedMs: capped.elapsedMs, peakRssBytes: capped.peakRssBytes },
    coverage: {
      ratingsRetainedFraction: capped.stats.selectedRatings / exact.stats.inputRatings,
      pairVisitsRetainedFraction: capped.stats.pairVisits / exact.stats.pairVisits,
      animeRetainedFraction: capped.stats.selectedAnimeCount / exact.stats.inputAnimeCount,
      pairKeyRecallAgainstExact: sharedKeys / exactKeys.size,
    },
    note: "Peak RSS is each child process high-water mark; elapsed time covers aggregation only. Pair-key recall is synthetic coverage, not recommendation quality.",
  };
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} else {
  throw new Error("Unexpected synthetic benchmark arguments.");
}
