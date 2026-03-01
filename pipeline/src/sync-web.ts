import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { getRepoRoot } from "./paths.js";

const repoRoot = getRepoRoot(import.meta.url);
const sourceDir = path.resolve(repoRoot, "data");
const targetDir = path.resolve(repoRoot, "web", "public", "data");

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
