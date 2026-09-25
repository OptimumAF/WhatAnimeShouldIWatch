import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { Command } from "commander";
import { loadDatasetFromDb, openDatabase } from "./db.js";
import { getRepoRoot } from "./paths.js";
import {
  aggregateAnimePairs,
  DEFAULT_MAX_CANDIDATE_PAIRS,
  DEFAULT_MAX_PAIR_VISITS,
  DEFAULT_PAIR_SELECTION_SEED,
  PAIR_CAP_POLICY,
  type PairSelectionOptions,
  type PairSelectionStats,
} from "./core/pair-aggregation.js";
import type {
  CompactAnonymizedDataset,
  CompactGraphData,
  GraphData,
  GraphEdge,
  GraphNode,
} from "./types.js";

interface BuildGraphOptions {
  db?: string;
  outDataset?: string;
  outGraph?: string;
  outDatasetCompact?: string;
  outGraphCompact?: string;
  outReport?: string;
  maxRatingsPerUser?: string;
  pairSelectionSeed?: string;
  maxAnimeAnimeEdges?: string;
  maxPairVisits?: string;
  maxPairCandidates?: string;
  minPairSupport?: string;
  maxNeighborsPerAnime?: string;
  prettyJson?: boolean;
  compact?: boolean;
  compactOnly?: boolean;
}

const repoRoot = getRepoRoot(import.meta.url);
const program = new Command()
  .argument("[db]", "SQLite path (positional fallback)")
  .argument("[outDataset]", "Dataset JSON path (positional fallback)")
  .argument("[outGraph]", "Graph JSON path (positional fallback)")
  .argument(
    "[maxRatingsPerUser]",
    "Per-user ratings cap for graph generation (positional fallback)",
  )
  .option("--db <path>", "Path to SQLite database")
  .option(
    "--out-dataset <path>",
    "Output path for anonymized user ratings JSON",
  )
  .option("--out-graph <path>", "Output path for generated graph JSON")
  .option(
    "--out-dataset-compact <path>",
    "Output path for compact anonymized user ratings JSON",
  )
  .option(
    "--out-graph-compact <path>",
    "Output path for compact graph JSON",
  )
  .option("--out-report <path>", "Output path for the graph build report JSON")
  .option(
    "--max-ratings-per-user <count>",
    "Optional seeded per-user rating cap for graph generation only (0 = unlimited)",
  )
  .option("--pair-selection-seed <uint32>", "Stable seed for per-user rating selection")
  .option(
    "--max-anime-anime-edges <count>",
    "Maximum selected anime-anime edges after exact pair aggregation (0 = unlimited)",
  )
  .option(
    "--max-pair-visits <count>",
    "Hard budget for pair observations before enumeration",
  )
  .option(
    "--max-pair-candidates <count>",
    "Hard budget for distinct candidate pair keys in memory",
  )
  .option(
    "--min-pair-support <count>",
    "Minimum co-rater count before a pair can be selected",
  )
  .option(
    "--max-neighbors-per-anime <count>",
    "Maximum selected pair degree per anime (0 = unlimited)",
  )
  .option(
    "--pretty-json",
    "Write pretty-printed JSON instead of minified output",
  )
  .option(
    "--compact-only",
    "Skip legacy JSON outputs and write compact outputs only",
  )
  .option(
    "--no-compact",
    "Disable compact output generation",
  );

program.parse(process.argv);
const options = program.opts<BuildGraphOptions>();
const [argDb, argOutDataset, argOutGraph, argMaxRatingsPerUser] =
  program.args as string[];

const dbPath = path.resolve(
  repoRoot,
  options.db ?? process.env.GRAPH_DB ?? argDb ?? "data/anime.sqlite",
);
const outDatasetPath = path.resolve(
  repoRoot,
  options.outDataset ??
    process.env.GRAPH_DATASET_OUT ??
    argOutDataset ??
    "data/anonymized-ratings.json",
);
const outGraphPath = path.resolve(
  repoRoot,
  options.outGraph ??
    process.env.GRAPH_OUT ??
    argOutGraph ??
    "data/graph.json",
);
const outDatasetCompactPath = path.resolve(
  repoRoot,
  options.outDatasetCompact ??
    process.env.GRAPH_DATASET_COMPACT_OUT ??
    "data/anonymized-ratings.compact.json",
);
const outGraphCompactPath = path.resolve(
  repoRoot,
  options.outGraphCompact ??
    process.env.GRAPH_COMPACT_OUT ??
    "data/graph.compact.json",
);
const maxRatingsPerUser = parseCount(
  options.maxRatingsPerUser ??
    process.env.GRAPH_MAX_RATINGS_PER_USER ??
    argMaxRatingsPerUser ??
    "0",
  "max ratings per user",
);
const pairSelectionSeed = parseCount(
  options.pairSelectionSeed ?? process.env.GRAPH_PAIR_SELECTION_SEED ?? String(DEFAULT_PAIR_SELECTION_SEED),
  "pair selection seed", 0, 0xffffffff,
);
const maxAnimeAnimeEdges = parseCount(
  options.maxAnimeAnimeEdges ??
    process.env.GRAPH_MAX_ANIME_ANIME_EDGES ??
    "2000000",
  "max anime-anime edges",
);
const maxPairVisits = parseCount(
  options.maxPairVisits ?? process.env.GRAPH_MAX_PAIR_VISITS ?? String(DEFAULT_MAX_PAIR_VISITS),
  "max pair visits", 1,
);
const maxPairCandidates = parseCount(
  options.maxPairCandidates ?? process.env.GRAPH_MAX_PAIR_CANDIDATES ?? String(DEFAULT_MAX_CANDIDATE_PAIRS),
  "max pair candidates", 1,
);
const minPairSupport = parseCount(
  options.minPairSupport ?? process.env.GRAPH_MIN_PAIR_SUPPORT ?? "1",
  "min pair support", 1,
);
const maxNeighborsPerAnime = parseCount(
  options.maxNeighborsPerAnime ?? process.env.GRAPH_MAX_NEIGHBORS_PER_ANIME ?? "0",
  "max neighbors per anime",
);
const prettyJson =
  options.prettyJson === true || isTruthy(process.env.GRAPH_PRETTY_JSON);
const writeCompact =
  options.compact !== false && !isTruthy(process.env.GRAPH_DISABLE_COMPACT);
const writeLegacy =
  options.compactOnly !== true && !isTruthy(process.env.GRAPH_COMPACT_ONLY);
if (!writeLegacy && !writeCompact) {
  throw new Error("At least one graph output format must be enabled.");
}
const outReportPath = path.resolve(
  repoRoot,
  options.outReport ?? process.env.GRAPH_REPORT_OUT ??
    `${writeLegacy ? outGraphPath : outGraphCompactPath}.report.json`,
);
if ([dbPath, outDatasetPath, outGraphPath, outDatasetCompactPath, outGraphCompactPath]
    .some((filePath) => filePath.toLowerCase() === outReportPath.toLowerCase())) {
  throw new Error("The graph build report path must be distinct from the database and graph artifacts.");
}

const startedAt = performance.now();
const db = openDatabase(dbPath);
try {
  const dataset = loadDatasetFromDb(db);

  for (const user of dataset.users) {
    const avg =
      user.ratings.reduce((sum, rating) => sum + rating.rawScore, 0) /
      Math.max(user.ratings.length, 1);
    for (const rating of user.ratings) {
      rating.normalizedScore = rating.rawScore - avg;
    }
  }

  const graphResult = createGraph(
    dataset,
    maxRatingsPerUser,
    maxAnimeAnimeEdges,
    {
      maxPairVisits,
      maxCandidatePairs: maxPairCandidates,
      minSupport: minPairSupport,
      maxNeighborsPerAnime,
      selectionSeed: pairSelectionSeed,
    },
  );
  const graph = graphResult.graph;
  if (writeLegacy) {
    fs.mkdirSync(path.dirname(outDatasetPath), { recursive: true });
    fs.writeFileSync(
      outDatasetPath,
      JSON.stringify(dataset, null, prettyJson ? 2 : 0),
    );
    fs.mkdirSync(path.dirname(outGraphPath), { recursive: true });
    fs.writeFileSync(outGraphPath, JSON.stringify(graph, null, prettyJson ? 2 : 0));
    process.stdout.write(`Dataset written: ${outDatasetPath}\n`);
    process.stdout.write(`Graph written: ${outGraphPath}\n`);
  }

  if (writeCompact) {
    const compactDataset = createCompactDataset(dataset);
    const compactGraph = createCompactGraph(graph);

    fs.mkdirSync(path.dirname(outDatasetCompactPath), { recursive: true });
    fs.writeFileSync(
      outDatasetCompactPath,
      JSON.stringify(compactDataset, null, prettyJson ? 2 : 0),
    );
    fs.mkdirSync(path.dirname(outGraphCompactPath), { recursive: true });
    fs.writeFileSync(
      outGraphCompactPath,
      JSON.stringify(compactGraph, null, prettyJson ? 2 : 0),
    );
    process.stdout.write(`Compact dataset written: ${outDatasetCompactPath}\n`);
    process.stdout.write(`Compact graph written: ${outGraphCompactPath}\n`);
  }

  process.stdout.write(
    `Graph stats: ${graph.userCount} users, ${graph.animeCount} anime, ${graph.edgeCount} edges\n`,
  );
  const pairStats = graphResult.pairStats;
  const maxRssKiB = process.resourceUsage().maxRSS;
  const peakRssBytes = maxRssKiB > 0 ? maxRssKiB * 1024 : process.memoryUsage().rss;
  const report = {
    format: "graph-build-report-v1",
    selection: {
      policy: maxRatingsPerUser > 0 ? PAIR_CAP_POLICY : "all-ratings",
      seed: pairSelectionSeed,
      maxRatingsPerUser,
      maxAnimeAnimeEdges,
      maxPairVisits,
      maxPairCandidates,
      minPairSupport,
      maxNeighborsPerAnime,
    },
    coverage: {
      approximationLevel: pairStats.ratingsSkippedByUserCap > 0 ? "seeded-per-user-subset" : "exact-input",
      inputUsers: pairStats.inputUsers,
      usersCapped: pairStats.usersCapped,
      inputRatings: pairStats.inputRatings,
      selectedRatings: pairStats.selectedRatings,
      ratingsSkipped: pairStats.ratingsSkippedByUserCap,
      ratingsRetainedFraction: fraction(pairStats.selectedRatings, pairStats.inputRatings),
      potentialPairVisits: pairStats.potentialPairVisits,
      pairVisits: pairStats.pairVisits,
      pairVisitsSkipped: pairStats.pairVisitsSkippedByUserCap,
      pairVisitsRetainedFraction: fraction(pairStats.pairVisits, pairStats.potentialPairVisits),
      inputAnimeCount: pairStats.inputAnimeCount,
      selectedAnimeCount: pairStats.selectedAnimeCount,
      animeRetainedFraction: fraction(pairStats.selectedAnimeCount, pairStats.inputAnimeCount),
      candidatePairs: pairStats.candidatePairs,
      eligiblePairs: pairStats.eligiblePairs,
      selectedPairs: pairStats.selectedPairs,
      excludedBySupport: pairStats.excludedBySupport,
      excludedByNeighborLimit: pairStats.excludedByNeighborLimit,
      excludedByOutputLimit: pairStats.excludedByOutputLimit,
      outputTruncated: pairStats.selectedPairs !== pairStats.candidatePairs,
      pairKeyRecall: pairStats.ratingsSkippedByUserCap > 0 ? null : 1,
    },
    measurement: {
      elapsedMs: Number((performance.now() - startedAt).toFixed(3)),
      peakRssBytes,
      peakRssSource: maxRssKiB > 0 ? "process.resourceUsage.maxRSS" : "process.memoryUsage.rss-snapshot",
    },
  };
  fs.mkdirSync(path.dirname(outReportPath), { recursive: true });
  fs.writeFileSync(outReportPath, JSON.stringify(report, null, prettyJson ? 2 : 0));
  process.stdout.write(`Graph build report written: ${outReportPath}\n`);
  process.stdout.write(
    `Pair selection: ${pairStats.pairVisits} visits, ${pairStats.candidatePairs} candidate keys, ` +
    `${pairStats.eligiblePairs} with support >= ${minPairSupport}, ${pairStats.selectedPairs} retained; ` +
    `${pairStats.excludedBySupport} below support, ${pairStats.excludedByNeighborLimit} neighbor-limited, ` +
    `${pairStats.excludedByOutputLimit} output-limited.\n`,
  );
  process.stdout.write(
    `Per-user selection: ${report.selection.policy}, seed=${pairSelectionSeed}, cap=${maxRatingsPerUser || "unlimited"}; ` +
    `${pairStats.ratingsSkippedByUserCap}/${pairStats.inputRatings} ratings and ` +
    `${pairStats.pairVisitsSkippedByUserCap}/${pairStats.potentialPairVisits} pair visits skipped; ` +
    `${pairStats.selectedAnimeCount}/${pairStats.inputAnimeCount} anime covered.\n`,
  );
  process.stdout.write(`Build measurement: ${report.measurement.elapsedMs} ms, peak RSS ${peakRssBytes} bytes (${report.measurement.peakRssSource}).\n`);
  process.stdout.write(
    `Pair budgets: visits=${maxPairVisits}, candidate keys=${maxPairCandidates}, ` +
    `output edges=${maxAnimeAnimeEdges || "unlimited"}, neighbors/anime=${maxNeighborsPerAnime || "unlimited"}.\n`,
  );
} finally {
  db.close();
}

function createGraph(
  dataset: {
    users: {
      userId: string;
      ratings: {
        animeId: number;
        title: string;
        normalizedScore: number;
      }[];
    }[];
  },
  maxRatingsPerUser: number,
  maxAnimeAnimeEdges: number,
  pairOptions: PairSelectionOptions,
): {
  graph: GraphData;
  pairStats: PairSelectionStats;
} {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];

  const pairResult = aggregateAnimePairs(dataset.users, maxRatingsPerUser, maxAnimeAnimeEdges, pairOptions);
  for (const user of pairResult.selectedUsers) {
    const userNodeId = `user:${user.userId}`;
    nodes.set(userNodeId, {
      id: userNodeId,
      label: `User ${user.userId.slice(0, 8)}`,
      nodeType: "user",
    });

    for (const rating of [...user.ratings].sort((a, b) => a.animeId - b.animeId)) {
      const animeNodeId = `anime:${rating.animeId}`;
      if (!nodes.has(animeNodeId)) {
        nodes.set(animeNodeId, {
          id: animeNodeId,
          label: rating.title,
          nodeType: "anime",
        });
      }

      edges.push({
        id: `ua:${user.userId}:${rating.animeId}`,
        source: userNodeId,
        target: animeNodeId,
        edgeType: "user-anime",
        weight: roundWeight(rating.normalizedScore),
      });
    }
  }

  for (const [pair, aggregate] of pairResult.pairs.entries()) {
    const [low, high] = pair.split(":");
    edges.push({
      id: `aa:${pair}`,
      source: `anime:${low}`,
      target: `anime:${high}`,
      edgeType: "anime-anime",
      weight: roundWeight(aggregate.weight),
      support: aggregate.support,
    });
  }

  const nodeList = [...nodes.values()];
  const userCount = nodeList.filter((node) => node.nodeType === "user").length;
  const animeCount = nodeList.length - userCount;

  return {
    graph: {
      generatedAt: new Date().toISOString(),
      nodeCount: nodeList.length,
      edgeCount: edges.length,
      userCount,
      animeCount,
      nodes: nodeList,
      edges,
    },
    pairStats: pairResult.stats,
  };
}

function parseCount(raw: string, label: string, minimum = 0, maximum = Number.MAX_SAFE_INTEGER): number {
  const trimmed = raw.trim();
  const requirement = maximum === 0xffffffff ? "an unsigned 32-bit integer" :
    `a ${minimum === 0 ? "nonnegative" : "positive"} safe integer`;
  if (!/^(0|[1-9]\d*)$/.test(trimmed)) {
    throw new Error(`Invalid ${label}: expected ${requirement}.`);
  }
  const value = Number(trimmed);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(`Invalid ${label}: expected ${requirement}.`);
  }
  return value;
}

function fraction(numerator: number, denominator: number): number {
  return denominator === 0 ? 1 : numerator / denominator;
}

function roundWeight(value: number): number {
  return Number(value.toFixed(4));
}

function isTruthy(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  return /^(1|true|yes|on)$/i.test(value.trim());
}

function createCompactDataset(dataset: {
  generatedAt: string;
  source: string;
  users: {
    userId: string;
    ratings: {
      animeId: number;
      title: string;
      rawScore: number;
      normalizedScore: number;
    }[];
  }[];
}): CompactAnonymizedDataset {
  const anime: [number, string][] = [];
  const animeIdToIndex = new Map<number, number>();
  const users: [string, [number, number, number][]][] = [];

  for (const user of dataset.users) {
    const compactRatings: [number, number, number][] = [];
    for (const rating of user.ratings) {
      let animeIndex = animeIdToIndex.get(rating.animeId);
      if (animeIndex === undefined) {
        animeIndex = anime.length;
        animeIdToIndex.set(rating.animeId, animeIndex);
        anime.push([rating.animeId, rating.title]);
      }
      compactRatings.push([
        animeIndex,
        rating.rawScore,
        roundWeight(rating.normalizedScore),
      ]);
    }
    users.push([user.userId, compactRatings]);
  }

  return {
    format: "ratings-compact-v1",
    generatedAt: dataset.generatedAt,
    source: dataset.source,
    anime,
    users,
  };
}

function createCompactGraph(graph: GraphData): CompactGraphData {
  const userIds: string[] = [];
  const anime: [number, string][] = [];
  const userNodeIdToIndex = new Map<string, number>();
  const animeNodeIdToIndex = new Map<string, number>();

  for (const node of graph.nodes) {
    if (node.nodeType === "user") {
      const userId = node.id.startsWith("user:")
        ? node.id.slice("user:".length)
        : node.id;
      const userIndex = userIds.length;
      userIds.push(userId);
      userNodeIdToIndex.set(node.id, userIndex);
      continue;
    }

    const animeIdValue = node.id.startsWith("anime:")
      ? node.id.slice("anime:".length)
      : node.id;
    const animeId = Number.parseInt(animeIdValue, 10);
    if (!Number.isFinite(animeId)) {
      continue;
    }
    const animeIndex = anime.length;
    anime.push([animeId, node.label]);
    animeNodeIdToIndex.set(node.id, animeIndex);
  }

  const ua: [number, number, number][] = [];
  const aa: CompactGraphData["aa"] = [];

  for (const edge of graph.edges) {
    if (edge.edgeType === "user-anime") {
      const sourceUser = userNodeIdToIndex.get(edge.source);
      const targetAnime = animeNodeIdToIndex.get(edge.target);
      const sourceAnime = animeNodeIdToIndex.get(edge.source);
      const targetUser = userNodeIdToIndex.get(edge.target);

      if (sourceUser !== undefined && targetAnime !== undefined) {
        ua.push([sourceUser, targetAnime, edge.weight]);
      } else if (targetUser !== undefined && sourceAnime !== undefined) {
        ua.push([targetUser, sourceAnime, edge.weight]);
      }
      continue;
    }

    const left = animeNodeIdToIndex.get(edge.source);
    const right = animeNodeIdToIndex.get(edge.target);
    if (left === undefined || right === undefined) {
      continue;
    }
    aa.push(edge.support === undefined
      ? [left, right, edge.weight]
      : [left, right, edge.weight, edge.support]);
  }

  return {
    format: "graph-compact-v1",
    generatedAt: graph.generatedAt,
    userIds,
    anime,
    ua,
    aa,
    userCount: userIds.length,
    animeCount: anime.length,
    nodeCount: userIds.length + anime.length,
    edgeCount: ua.length + aa.length,
  };
}
