import fs from "node:fs";
import path from "node:path";
import { Command } from "commander";
import { loadDatasetFromDb, openDatabase } from "./db.js";
import { getRepoRoot } from "./paths.js";
import type { GraphData, GraphEdge, GraphNode } from "./types.js";

interface BuildGraphOptions {
  db?: string;
  outDataset?: string;
  outGraph?: string;
  maxRatingsPerUser?: string;
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
    "--max-ratings-per-user <count>",
    "Optional per-user cap for graph generation only (0 = unlimited)",
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

  fs.mkdirSync(path.dirname(outDatasetPath), { recursive: true });
  fs.writeFileSync(outDatasetPath, JSON.stringify(dataset, null, 2));

  const graph = createGraph(dataset, maxRatingsPerUser);
  fs.mkdirSync(path.dirname(outGraphPath), { recursive: true });
  fs.writeFileSync(outGraphPath, JSON.stringify(graph, null, 2));

  process.stdout.write(`Dataset written: ${outDatasetPath}\n`);
  process.stdout.write(`Graph written: ${outGraphPath}\n`);
  process.stdout.write(
    `Graph stats: ${graph.userCount} users, ${graph.animeCount} anime, ${graph.edgeCount} edges\n`,
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
): GraphData {
  const nodes = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const animePairEdgeWeights = new Map<string, number>();

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
    generatedAt: new Date().toISOString(),
    nodeCount: nodeList.length,
    edgeCount: edges.length,
    userCount,
    animeCount,
    nodes: nodeList,
    edges,
  };
}

function roundWeight(value: number): number {
  return Number(value.toFixed(4));
}
