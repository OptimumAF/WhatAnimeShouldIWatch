import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { getRepoRoot } from "./paths.js";

interface CompactGraphData {
  format: "graph-compact-v1";
  generatedAt: string;
  userIds: string[];
  anime: [number, string][];
  ua: [number, number, number][];
  aa: [number, number, number][];
  userCount: number;
  animeCount: number;
  nodeCount: number;
  edgeCount: number;
}

const repoRoot = getRepoRoot(import.meta.url);
const sourceDir = path.resolve(repoRoot, "data");
const targetDir = path.resolve(repoRoot, "web", "public", "data");
const EXPLORER_AA_LIMIT = 8000;
const EXPLORER_UA_LIMIT = 2500;

const includeDataset = isTruthy(process.env.SYNC_WEB_INCLUDE_DATASET);
const keepPlainJson = isTruthy(process.env.SYNC_WEB_KEEP_JSON);
const gzipLevel = parsePositiveInt(process.env.SYNC_WEB_GZIP_LEVEL, 9);

fs.mkdirSync(targetDir, { recursive: true });

syncRequiredWithFallback(["graph.compact.json", "graph.json"]);

if (includeDataset) {
  syncRequiredWithFallback([
    "anonymized-ratings.compact.json",
    "anonymized-ratings.json",
  ]);
} else {
  removeSyncedOutputs("anonymized-ratings.compact.json");
  removeSyncedOutputs("anonymized-ratings.json");
  process.stdout.write(
    "Skipped anonymized ratings dataset (set SYNC_WEB_INCLUDE_DATASET=1 to include it)\n",
  );
}

syncOptionalWithFallback(["model-mf-web.compact.json", "model-mf-web.json"]);

function syncRequiredWithFallback(candidates: string[]): void {
  const selected = candidates.find((filename) =>
    fs.existsSync(path.join(sourceDir, filename)),
  );
  if (!selected) {
    throw new Error(
      `Missing source file. Checked: ${candidates
        .map((filename) => path.join(sourceDir, filename))
        .join(", ")}. Run "npm run build:graph --workspace pipeline --" first.`,
    );
  }
  for (const filename of candidates) {
    if (filename !== selected) {
      removeSyncedOutputs(filename);
    }
  }
  syncFileOutputs(selected, path.join(sourceDir, selected));
}

function syncOptionalWithFallback(candidates: string[]): void {
  const selected = candidates.find((filename) =>
    fs.existsSync(path.join(sourceDir, filename)),
  );
  if (!selected) {
    for (const filename of candidates) {
      removeSyncedOutputs(filename);
    }
    process.stdout.write(
      `Skipped optional data; none found (${candidates.join(", ")})\n`,
    );
    return;
  }
  for (const filename of candidates) {
    if (filename !== selected) {
      removeSyncedOutputs(filename);
    }
  }
  syncFileOutputs(selected, path.join(sourceDir, selected));
}

function syncFileOutputs(filename: string, sourcePath: string): void {
  const targetPath = path.join(targetDir, filename);
  const sourceBuffer = fs.readFileSync(sourcePath);
  const gzipBuffer = zlib.gzipSync(sourceBuffer, { level: gzipLevel });

  fs.writeFileSync(`${targetPath}.gz`, gzipBuffer);
  process.stdout.write(
    `Synced ${filename}.gz -> ${targetPath}.gz (${formatBytes(sourceBuffer.length)} -> ${formatBytes(gzipBuffer.length)})\n`,
  );

  if (keepPlainJson) {
    fs.copyFileSync(sourcePath, targetPath);
    process.stdout.write(`Synced ${filename} -> ${targetPath}\n`);
  } else {
    removeIfExists(targetPath);
  }

  if (filename === "graph.compact.json") {
    syncExplorerGraphOutputs(sourceBuffer);
  } else if (filename === "graph.json") {
    removeSyncedOutputs("graph-explorer.compact.json");
  }
}

function removeSyncedOutputs(filename: string): void {
  const targetPath = path.join(targetDir, filename);
  removeIfExists(targetPath);
  removeIfExists(`${targetPath}.gz`);
}

function removeIfExists(filepath: string): void {
  if (fs.existsSync(filepath)) {
    fs.unlinkSync(filepath);
  }
}

function syncExplorerGraphOutputs(sourceBuffer: Buffer): void {
  const compactGraph = JSON.parse(sourceBuffer.toString("utf8")) as CompactGraphData;
  if (compactGraph.format !== "graph-compact-v1") {
    removeSyncedOutputs("graph-explorer.compact.json");
    return;
  }

  const explorerGraph = buildExplorerGraph(compactGraph);
  const explorerJson = Buffer.from(JSON.stringify(explorerGraph), "utf8");
  const targetPath = path.join(targetDir, "graph-explorer.compact.json");
  const gzipBuffer = zlib.gzipSync(explorerJson, { level: gzipLevel });

  fs.writeFileSync(`${targetPath}.gz`, gzipBuffer);
  process.stdout.write(
    `Synced graph-explorer.compact.json.gz -> ${targetPath}.gz (${formatBytes(explorerJson.length)} -> ${formatBytes(gzipBuffer.length)})\n`,
  );

  if (keepPlainJson) {
    fs.writeFileSync(targetPath, explorerJson);
    process.stdout.write(`Synced graph-explorer.compact.json -> ${targetPath}\n`);
  } else {
    removeIfExists(targetPath);
  }
}

function buildExplorerGraph(graph: CompactGraphData): CompactGraphData {
  const selectedUa = selectTopEdges(graph.ua, EXPLORER_UA_LIMIT);
  const selectedAa = selectTopEdges(graph.aa, EXPLORER_AA_LIMIT);
  const animeIndexMap = new Map<number, number>();
  const userIndexMap = new Map<number, number>();
  const anime: [number, string][] = [];
  const userIds: string[] = [];

  const remapAnime = (sourceIndex: number): number => {
    const existing = animeIndexMap.get(sourceIndex);
    if (existing !== undefined) {
      return existing;
    }
    const entry = graph.anime[sourceIndex];
    const nextIndex = anime.length;
    animeIndexMap.set(sourceIndex, nextIndex);
    anime.push(entry);
    return nextIndex;
  };

  const remapUser = (sourceIndex: number): number => {
    const existing = userIndexMap.get(sourceIndex);
    if (existing !== undefined) {
      return existing;
    }
    const userId = graph.userIds[sourceIndex];
    const nextIndex = userIds.length;
    userIndexMap.set(sourceIndex, nextIndex);
    userIds.push(userId);
    return nextIndex;
  };

  const ua: [number, number, number][] = [];
  const aa: [number, number, number][] = [];

  for (const [userIndex, animeIndex, weight] of selectedUa) {
    if (!graph.userIds[userIndex] || !graph.anime[animeIndex]) {
      continue;
    }
    ua.push([remapUser(userIndex), remapAnime(animeIndex), weight]);
  }

  for (const [leftAnimeIndex, rightAnimeIndex, weight] of selectedAa) {
    if (!graph.anime[leftAnimeIndex] || !graph.anime[rightAnimeIndex]) {
      continue;
    }
    aa.push([remapAnime(leftAnimeIndex), remapAnime(rightAnimeIndex), weight]);
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

function selectTopEdges(
  edges: [number, number, number][],
  limit: number,
): [number, number, number][] {
  if (edges.length <= limit) {
    return edges;
  }

  const top = [...edges]
    .sort((left, right) => Math.abs(right[2]) - Math.abs(left[2]))
    .slice(0, limit);

  return top;
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  if (!value) {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 1 || parsed > 9) {
    return fallback;
  }
  return parsed;
}

function isTruthy(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  return /^(1|true|yes|on)$/i.test(value.trim());
}

function formatBytes(value: number): string {
  const megabytes = value / (1024 * 1024);
  return `${megabytes.toFixed(2)} MB`;
}
