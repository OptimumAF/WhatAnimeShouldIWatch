import fs from "node:fs";
import path from "node:path";
import { Command } from "commander";
import { loadDatasetFromDb, openDatabase } from "./db.js";
import { getRepoRoot } from "./paths.js";
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
  maxRatingsPerUser?: string;
  maxAnimeAnimeEdges?: string;
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
  .option(
    "--max-ratings-per-user <count>",
    "Optional per-user cap for graph generation only (0 = unlimited)",
  )
  .option(
    "--max-anime-anime-edges <count>",
    "Maximum number of unique anime-anime edges to keep (0 = unlimited)",
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
const maxRatingsPerUser = Number.parseInt(
  options.maxRatingsPerUser ??
    process.env.GRAPH_MAX_RATINGS_PER_USER ??
    argMaxRatingsPerUser ??
    "0",
  10,
);
if (Number.isNaN(maxRatingsPerUser) || maxRatingsPerUser < 0) {
  throw new Error(
    `Invalid max ratings value: ${
      options.maxRatingsPerUser ?? argMaxRatingsPerUser
    }`,
  );
}
const maxAnimeAnimeEdges = Number.parseInt(
  options.maxAnimeAnimeEdges ??
    process.env.GRAPH_MAX_ANIME_ANIME_EDGES ??
    "2000000",
  10,
);
if (Number.isNaN(maxAnimeAnimeEdges) || maxAnimeAnimeEdges < 0) {
  throw new Error(
    `Invalid max anime-anime edge value: ${options.maxAnimeAnimeEdges}`,
  );
}
const prettyJson =
  options.prettyJson === true || isTruthy(process.env.GRAPH_PRETTY_JSON);
const writeCompact =
  options.compact !== false && !isTruthy(process.env.GRAPH_DISABLE_COMPACT);
const writeLegacy =
  options.compactOnly !== true && !isTruthy(process.env.GRAPH_COMPACT_ONLY);

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
  if (graphResult.skippedNewAnimeAnimePairs > 0) {
    process.stdout.write(
      `Anime-anime edge guard hit: skipped ${graphResult.skippedNewAnimeAnimePairs} new pair keys after reaching cap=${graphResult.maxAnimeAnimeEdges}\n`,
    );
  }
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
): {
  graph: GraphData;
  skippedNewAnimeAnimePairs: number;
  maxAnimeAnimeEdges: number;
} {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const animePairEdgeWeights = new Map<string, number>();
  let skippedNewAnimeAnimePairs = 0;

  for (const user of dataset.users) {
    const userNodeId = `user:${user.userId}`;
    nodes.set(userNodeId, {
      id: userNodeId,
      label: `User ${user.userId.slice(0, 8)}`,
      nodeType: "user",
    });

    const userRatings =
      maxRatingsPerUser > 0
        ? user.ratings.slice(0, maxRatingsPerUser)
        : user.ratings;

    for (const rating of userRatings) {
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

    for (let i = 0; i < userRatings.length; i += 1) {
      const left = userRatings[i];
      for (let j = i + 1; j < userRatings.length; j += 1) {
        const right = userRatings[j];
        const low = Math.min(left.animeId, right.animeId);
        const high = Math.max(left.animeId, right.animeId);
        const key = `${low}:${high}`;
        const userPairScore = (left.normalizedScore + right.normalizedScore) / 2;
        const current = animePairEdgeWeights.get(key);
        if (
          current === undefined &&
          maxAnimeAnimeEdges > 0 &&
          animePairEdgeWeights.size >= maxAnimeAnimeEdges
        ) {
          skippedNewAnimeAnimePairs += 1;
          continue;
        }
        const next =
          current === undefined ? userPairScore : (current + userPairScore) / 2;
        animePairEdgeWeights.set(key, next);
      }
    }
  }

  for (const [pair, weight] of animePairEdgeWeights.entries()) {
    const [low, high] = pair.split(":");
    edges.push({
      id: `aa:${pair}`,
      source: `anime:${low}`,
      target: `anime:${high}`,
      edgeType: "anime-anime",
      weight: roundWeight(weight),
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
    skippedNewAnimeAnimePairs,
    maxAnimeAnimeEdges,
  };
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
  const aa: [number, number, number][] = [];

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
    aa.push([left, right, edge.weight]);
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
