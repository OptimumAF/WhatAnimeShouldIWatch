import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { Command } from "commander";
import { getRepoRoot } from "./paths.js";

interface GitHubAsset {
  name: string;
  browser_download_url: string;
}

interface GitHubRelease {
  tag_name: string;
  draft: boolean;
  prerelease: boolean;
  assets: GitHubAsset[];
}

interface FetchReleaseOptions {
  owner: string;
  repo: string;
  tag: string;
  outDir: string;
  keepGzip: boolean;
  allowLocalFallback: boolean;
  retries: number;
  retryDelayMs: number;
}

const repoRoot = getRepoRoot(import.meta.url);

const program = new Command()
  .name("fetch-release-data")
  .description(
    "Download large data artifacts from GitHub Releases into local data/.",
  )
  .option("--owner <value>", "GitHub org/user", process.env.GH_OWNER ?? "OptimumAF")
  .option(
    "--repo <value>",
    "GitHub repo",
    process.env.GH_REPO ?? "WhatAnimeShouldIWatch",
  )
  .option(
    "--tag <value>",
    'Release tag to fetch. Use "latest" or explicit tag. Default: newest tag prefixed with "data-".',
    process.env.DATA_RELEASE_TAG ?? "",
  )
  .option(
    "--out-dir <path>",
    "Output directory for downloaded JSON files",
    path.resolve(repoRoot, "data"),
  )
  .option(
    "--keep-gzip",
    "Keep downloaded .gz files in output directory",
    false,
  )
  .option(
    "--allow-local-fallback",
    "If release fetch fails, continue when required local files already exist",
    isTruthy(process.env.DATA_FETCH_ALLOW_LOCAL_FALLBACK),
  )
  .option(
    "--retries <count>",
    "Retry attempts for transient release-asset mismatches",
    process.env.DATA_FETCH_RETRIES ?? "6",
  )
  .option(
    "--retry-delay-ms <milliseconds>",
    "Base delay between retry attempts",
    process.env.DATA_FETCH_RETRY_DELAY_MS ?? "3000",
  )
  .parse(process.argv);

const options = program.opts<FetchReleaseOptions>();

const requiredAssets = [
  "graph.compact.json.gz",
  "anonymized-ratings.compact.json.gz",
];
const optionalAssets = ["model-mf-web.compact.json.gz"];

const token = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN ?? "";

async function main(): Promise<void> {
  fs.mkdirSync(options.outDir, { recursive: true });
  try {
    await fetchReleaseAssetsWithRetry(options, token);
  } catch (error) {
    if (!options.allowLocalFallback) {
      throw error;
    }
    const hasAllLocalRequired = requiredAssets.every((assetName) => {
      const jsonName = assetName.endsWith(".gz")
        ? assetName.slice(0, -".gz".length)
        : assetName;
      return fs.existsSync(path.join(options.outDir, jsonName));
    });
    if (!hasAllLocalRequired) {
      throw error;
    }
    process.stdout.write(
      `Release fetch failed but local required data exists in ${options.outDir}; continuing.\n`,
    );
    process.stdout.write(`Reason: ${(error as Error).message}\n`);
  }
}

async function fetchReleaseAssetsWithRetry(
  config: FetchReleaseOptions,
  authToken: string,
): Promise<void> {
  const maxAttempts = Math.max(1, parseNonNegativeInt(config.retries, "retries") + 1);
  const retryDelayMs = parseNonNegativeInt(
    config.retryDelayMs,
    "retry-delay-ms",
  );
  let lastError: Error | undefined;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const release = await resolveRelease(
        config.owner,
        config.repo,
        config.tag.trim(),
        authToken,
      );
      process.stdout.write(
        `Using release tag ${release.tag_name} from ${config.owner}/${config.repo}\n`,
      );

      const byName = new Map(release.assets.map((asset) => [asset.name, asset]));
      const missingRequired = requiredAssets.filter((required) => !byName.has(required));
      if (missingRequired.length > 0) {
        throw new Error(
          `Missing required assets in release ${release.tag_name}: ${missingRequired.join(", ")}`,
        );
      }

      for (const required of requiredAssets) {
        await downloadAndExtractGzip(
          byName.get(required)!,
          config.outDir,
          config.keepGzip,
          authToken,
        );
      }

      for (const optional of optionalAssets) {
        const asset = byName.get(optional);
        if (!asset) {
          process.stdout.write(`Optional asset missing: ${optional}\n`);
          continue;
        }
        await downloadAndExtractGzip(asset, config.outDir, config.keepGzip, authToken);
      }
      return;
    } catch (error) {
      lastError = error as Error;
      if (attempt >= maxAttempts || !isTransientReleaseError(lastError)) {
        break;
      }
      const delayMs = retryDelayMs * attempt;
      process.stdout.write(
        `Release asset fetch retry ${attempt}/${maxAttempts - 1} after transient failure: ${lastError.message}\n`,
      );
      await sleep(delayMs);
    }
  }

  throw lastError ?? new Error("Release asset fetch failed for unknown reason");
}

async function resolveRelease(
  owner: string,
  repo: string,
  tag: string,
  authToken: string,
): Promise<GitHubRelease> {
  if (tag && tag !== "latest") {
    return fetchJson<GitHubRelease>(
      `https://api.github.com/repos/${owner}/${repo}/releases/tags/${encodeURIComponent(tag)}`,
      authToken,
    );
  }

  if (tag === "latest") {
    return fetchJson<GitHubRelease>(
      `https://api.github.com/repos/${owner}/${repo}/releases/latest`,
      authToken,
    );
  }

  const releases = await fetchJson<GitHubRelease[]>(
    `https://api.github.com/repos/${owner}/${repo}/releases?per_page=100`,
    authToken,
  );
  const dataRelease = releases.find(
    (release) =>
      !release.draft &&
      !release.prerelease &&
      typeof release.tag_name === "string" &&
      release.tag_name.startsWith("data-"),
  );
  if (dataRelease) {
    return dataRelease;
  }

  return fetchJson<GitHubRelease>(
    `https://api.github.com/repos/${owner}/${repo}/releases/latest`,
    authToken,
  );
}

async function fetchJson<T>(url: string, authToken: string): Promise<T> {
  const response = await fetch(url, {
    headers: buildApiHeaders(authToken),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`GitHub API request failed (${response.status}): ${message}`);
  }
  return (await response.json()) as T;
}

function buildApiHeaders(authToken: string): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "User-Agent": "WhatAnimeShouldIWatch-fetch-release-data",
  };
  if (authToken.trim()) {
    headers.Authorization = `Bearer ${authToken.trim()}`;
  }
  return headers;
}

async function downloadAndExtractGzip(
  asset: GitHubAsset,
  outDir: string,
  keepGzip: boolean,
  authToken: string,
): Promise<void> {
  const response = await fetch(asset.browser_download_url, {
    headers: buildApiHeaders(authToken),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(
      `Asset download failed for ${asset.name} (${response.status}): ${message}`,
    );
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  const gzPath = path.join(outDir, asset.name);
  if (keepGzip) {
    fs.writeFileSync(gzPath, buffer);
  } else if (fs.existsSync(gzPath)) {
    fs.unlinkSync(gzPath);
  }

  const jsonName = asset.name.endsWith(".gz")
    ? asset.name.slice(0, -".gz".length)
    : asset.name;
  const jsonPath = path.join(outDir, jsonName);
  const extracted = asset.name.endsWith(".gz") ? zlib.gunzipSync(buffer) : buffer;
  fs.writeFileSync(jsonPath, extracted);

  process.stdout.write(
    `Downloaded ${asset.name} -> ${jsonName} (${formatBytes(buffer.length)} -> ${formatBytes(extracted.length)})\n`,
  );
}

function isTransientReleaseError(error: Error): boolean {
  const message = error.message.toLowerCase();
  return (
    message.includes("missing required asset") ||
    message.includes("missing required assets") ||
    message.includes("asset download failed") ||
    message.includes("502") ||
    message.includes("503") ||
    message.includes("504") ||
    message.includes("secondary rate limit")
  );
}

function parseNonNegativeInt(value: string | number, field: string): number {
  const parsed =
    typeof value === "number" ? value : Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`Invalid ${field}: ${value}`);
  }
  return parsed;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatBytes(value: number): string {
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function isTruthy(value: string | undefined): boolean {
  if (!value) {
    return false;
  }
  return /^(1|true|yes|on)$/i.test(value.trim());
}

await main();
