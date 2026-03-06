import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import zlib from "node:zlib";
import crypto from "node:crypto";
import { Command } from "commander";
import { getRepoRoot } from "./paths.js";

interface PublishOptions {
  owner: string;
  repo: string;
  tag: string;
  releaseName: string;
  sourceDir: string;
  workDir: string;
}

const repoRoot = getRepoRoot(import.meta.url);

const program = new Command()
  .name("publish-release-data")
  .description("Package local compact data files and upload them to a GitHub release.")
  .option("--owner <value>", "GitHub org/user", process.env.GH_OWNER ?? "OptimumAF")
  .option(
    "--repo <value>",
    "GitHub repo",
    process.env.GH_REPO ?? "WhatAnimeShouldIWatch",
  )
  .option("--tag <value>", "Release tag", process.env.DATA_RELEASE_TAG ?? "data-latest")
  .option(
    "--release-name <value>",
    "Release title",
    process.env.DATA_RELEASE_NAME ?? "Data Assets (latest)",
  )
  .option(
    "--source-dir <path>",
    "Directory containing local compact JSON data",
    path.resolve(repoRoot, "data"),
  )
  .option(
    "--work-dir <path>",
    "Temporary packaging directory",
    path.resolve(repoRoot, "release-data"),
  )
  .parse(process.argv);

const options = program.opts<PublishOptions>();

const requiredInputFiles = [
  "graph.compact.json",
  "anonymized-ratings.compact.json",
];
const optionalInputFiles = ["model-mf-web.compact.json"];

main();

function main(): void {
  for (const filename of requiredInputFiles) {
    const filepath = path.join(options.sourceDir, filename);
    if (!fs.existsSync(filepath)) {
      throw new Error(`Missing required local data file: ${filepath}`);
    }
  }

  fs.mkdirSync(options.workDir, { recursive: true });
  cleanDirectory(options.workDir);

  const packagedFiles = [
    ...requiredInputFiles,
    ...optionalInputFiles.filter((filename) =>
      fs.existsSync(path.join(options.sourceDir, filename)),
    ),
  ].map((filename) => packageGzipAsset(filename));

  writeChecksums(packagedFiles);
  writeManifest(packagedFiles);
  ensureReleaseExists();
  uploadReleaseAssets(packagedFiles);

  process.stdout.write(
    `Published ${packagedFiles.length} data assets to ${options.owner}/${options.repo}@${options.tag}\n`,
  );
}

function cleanDirectory(dir: string): void {
  for (const entry of fs.readdirSync(dir)) {
    fs.rmSync(path.join(dir, entry), { recursive: true, force: true });
  }
}

function packageGzipAsset(filename: string): string {
  const inputPath = path.join(options.sourceDir, filename);
  const outputName = `${filename}.gz`;
  const outputPath = path.join(options.workDir, outputName);
  const sourceBuffer = fs.readFileSync(inputPath);
  const gzipBuffer = zlib.gzipSync(sourceBuffer, { level: 9 });
  fs.writeFileSync(outputPath, gzipBuffer);
  process.stdout.write(
    `Packaged ${filename} -> ${outputName} (${formatMegabytes(sourceBuffer.length)} -> ${formatMegabytes(gzipBuffer.length)})\n`,
  );
  return outputPath;
}

function writeChecksums(assetPaths: string[]): void {
  const lines = assetPaths
    .map((assetPath) => {
      const name = path.basename(assetPath);
      const hash = crypto
        .createHash("sha256")
        .update(fs.readFileSync(assetPath))
        .digest("hex");
      return `${hash} *${name}`;
    })
    .sort();
  fs.writeFileSync(path.join(options.workDir, "SHA256SUMS.txt"), `${lines.join("\n")}\n`);
}

function writeManifest(assetPaths: string[]): void {
  const files = assetPaths
    .map((assetPath) => {
      const stat = fs.statSync(assetPath);
      return {
        name: path.basename(assetPath),
        bytes: stat.size,
      };
    })
    .sort((left, right) => left.name.localeCompare(right.name));
  fs.writeFileSync(
    path.join(options.workDir, "data-manifest.json"),
    `${JSON.stringify({ generatedAt: new Date().toISOString(), files }, null, 2)}\n`,
    "utf8",
  );
}

function ensureReleaseExists(): void {
  try {
    runGh(["release", "view", options.tag, "--repo", `${options.owner}/${options.repo}`]);
  } catch {
    runGh([
      "release",
      "create",
      options.tag,
      "--repo",
      `${options.owner}/${options.repo}`,
      "--title",
      options.releaseName,
      "--notes",
      "",
    ]);
  }
}

function uploadReleaseAssets(assetPaths: string[]): void {
  runGh([
    "release",
    "upload",
    options.tag,
    ...assetPaths,
    path.join(options.workDir, "data-manifest.json"),
    path.join(options.workDir, "SHA256SUMS.txt"),
    "--clobber",
    "--repo",
    `${options.owner}/${options.repo}`,
  ]);
}

function runGh(args: string[]): void {
  execFileSync("gh", args, {
    cwd: repoRoot,
    stdio: "inherit",
  });
}

function formatMegabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
