import fs from "node:fs";
import path from "node:path";
import { getRepoRoot } from "./paths.js";

const repoRoot = getRepoRoot(import.meta.url);
const sourceDir = path.resolve(repoRoot, "data");
const targetDir = path.resolve(repoRoot, "web", "public", "data");

fs.mkdirSync(targetDir, { recursive: true });

for (const filename of ["anonymized-ratings.json", "graph.json"]) {
  const sourcePath = path.join(sourceDir, filename);
  const targetPath = path.join(targetDir, filename);
  if (!fs.existsSync(sourcePath)) {
    throw new Error(
      `Missing source file: ${sourcePath}. Run "npm run build:graph --workspace pipeline --" first.`,
    );
  }
  fs.copyFileSync(sourcePath, targetPath);
  process.stdout.write(`Synced ${filename} -> ${targetPath}\n`);
}

for (const optionalFilename of ["model-mf-web.json"]) {
  const sourcePath = path.join(sourceDir, optionalFilename);
  const targetPath = path.join(targetDir, optionalFilename);
  if (!fs.existsSync(sourcePath)) {
    process.stdout.write(
      `Skipped optional ${optionalFilename}; file not found at ${sourcePath}\n`,
    );
    continue;
  }
  fs.copyFileSync(sourcePath, targetPath);
  process.stdout.write(`Synced ${optionalFilename} -> ${targetPath}\n`);
}
