import { readReleaseResponse, type ReleaseAssetTransport } from "./install-release-bundle.js";
import { RELEASE_FILES } from "./release-manifest.js";

interface GitHubRelease {
  tag_name: string;
  draft: boolean;
  prerelease: boolean;
  assets: { name: string }[];
}

export interface GitHubTransportOptions {
  owner: string;
  repo: string;
  tag: string;
  token?: string;
  fetcher?: typeof fetch;
}

function fail(field: string, reason: string): never {
  throw new Error(`GitHub release transport ${field}: ${reason}`);
}

/** Resolve an explicit versioned GitHub release; the caller decides whether source use is approved. */
export async function createGitHubReleaseTransport(options: GitHubTransportOptions):
  Promise<ReleaseAssetTransport> {
  for (const [field, value] of [["owner", options.owner], ["repo", options.repo]] as const) {
    if (!/^[A-Za-z0-9_.-]+$/.test(value) || value === "." || value === "..") {
      fail(field, "must be a GitHub owner or repository name");
    }
  }
  if (!/^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(options.tag)) {
    fail("tag", "must be an explicit versioned data-v tag");
  }
  const fetcher = options.fetcher ?? globalThis.fetch;
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "User-Agent": "WhatAnimeShouldIWatch-verified-release-installer",
  };
  if (options.token) headers.Authorization = `Bearer ${options.token}`;
  const apiUrl = `https://api.github.com/repos/${options.owner}/${options.repo}/releases/tags/${encodeURIComponent(options.tag)}`;
  const response = await fetcher(apiUrl, { headers });
  const bytes = await readReleaseResponse(response, 1024 * 1024, "GitHub release metadata");
  let release: unknown;
  try {
    release = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    fail("metadata", "invalid JSON or UTF-8");
  }
  if (!release || typeof release !== "object" || Array.isArray(release)) {
    fail("metadata", "must be an object");
  }
  const candidate = release as Partial<GitHubRelease>;
  if (candidate.tag_name !== options.tag || candidate.draft !== false ||
      candidate.prerelease !== false || !Array.isArray(candidate.assets)) {
    fail("metadata", "tag, publication state, or asset list is invalid");
  }
  const assets = candidate.assets.map((asset, index) => {
    if (!asset || typeof asset.name !== "string" || !asset.name || asset.name.includes("/") ||
        asset.name.includes("\\")) {
      fail(`assets[${index}].name`, "must be a simple asset filename");
    }
    return asset.name;
  });
  const allowed = new Set(assets);
  const bundleNames = new Set<string>([RELEASE_FILES.manifest]);
  for (const file of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    RELEASE_FILES.catalog, RELEASE_FILES.model]) {
    bundleNames.add(file);
    bundleNames.add(`${file}.gz`);
  }
  return {
    tag: options.tag,
    assets,
    fetchAsset(name: string) {
      if (!allowed.has(name) || !bundleNames.has(name)) {
        fail("asset", `unlisted or unsupported filename ${name}`);
      }
      const url = `https://github.com/${options.owner}/${options.repo}/releases/download/${encodeURIComponent(options.tag)}/${encodeURIComponent(name)}`;
      return fetcher(url, { headers: {
        Accept: "application/octet-stream", "User-Agent": headers["User-Agent"],
      } });
    },
  };
}
