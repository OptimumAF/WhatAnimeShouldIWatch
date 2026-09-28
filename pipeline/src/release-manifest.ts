import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { isDeepStrictEqual } from "node:util";
import {
  parseCompactGraph, parseCompactModel, parseReleaseIdentityCatalog, parseReleaseManifest,
  type CompactAnimeEntry, type CompactGraphDataV2, type ReleaseManifestV1,
} from "../../web/src/artifacts.js";
import { recommendationGraphId, visualizationGraphId } from "./core/graph-contract.js";
import { buildExplorerGraph } from "./core/explorer-graph.js";

export const RELEASE_FILES = {
  neighborhood: "graph.compact.json",
  explorer: "graph-explorer.compact.json",
  catalog: "catalog.identity.json",
  model: "model-mf-web.compact.json",
  manifest: "release-manifest.json",
} as const;

export interface ReleaseFileBytes {
  neighborhood: Buffer;
  explorer: Buffer;
  catalog: Buffer;
  model?: Buffer;
}

type Previous = NonNullable<ReleaseManifestV1["lastKnownGood"]>;

export interface ReleaseBuildOptions {
  tag: string;
  lastKnownGood?: Previous;
  fixtureGenesis?: boolean;
}

function fail(field: string, reason: string): never {
  throw new Error(`Release bundle ${field}: ${reason}`);
}

export function releaseSha256(bytes: Buffer | string): string {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

/** The mapping digest is SHA-256 of compact JSON over numeric-ID-sorted [ID, title] pairs. */
export function releaseItemMapSha256(anime: CompactAnimeEntry[]): string {
  return releaseSha256(JSON.stringify([...anime].sort(([left], [right]) => left - right)));
}

function parseJson(bytes: Buffer, filename: string): unknown {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    fail(filename, "invalid JSON or UTF-8");
  }
}

function graphWithoutId(graph: CompactGraphDataV2): Omit<CompactGraphDataV2, "graphId"> {
  const { graphId: _graphId, ...withoutId } = graph;
  return withoutId;
}

function asset(bytes: Buffer, file: string, format: string) {
  return { path: file, format, sha256: releaseSha256(bytes), bytes: bytes.length };
}

export function buildReleaseManifest(files: ReleaseFileBytes, options: ReleaseBuildOptions): ReleaseManifestV1 {
  if (!/^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(options.tag)) {
    fail("tag", "must be a versioned data-v tag");
  }
  if (!options.lastKnownGood && !options.fixtureGenesis) {
    fail("lastKnownGood", "a prior bundle is required unless fixtureGenesis is explicit");
  }
  if (options.lastKnownGood && options.fixtureGenesis) {
    fail("fixtureGenesis", "cannot be combined with a previous bundle");
  }
  if (options.lastKnownGood?.tag === options.tag) fail("lastKnownGood.tag", "must differ from current tag");

  const neighborhood = parseCompactGraph(parseJson(files.neighborhood, RELEASE_FILES.neighborhood),
    RELEASE_FILES.neighborhood, "recommendation");
  const explorer = parseCompactGraph(parseJson(files.explorer, RELEASE_FILES.explorer),
    RELEASE_FILES.explorer, "visualization");
  if (neighborhood.format !== "graph-compact-v2") fail(RELEASE_FILES.neighborhood, "requires graph-compact-v2");
  if (explorer.format !== "graph-compact-v2") fail(RELEASE_FILES.explorer, "requires graph-compact-v2");
  if (neighborhood.graphId !== recommendationGraphId(graphWithoutId(neighborhood))) {
    fail(`${RELEASE_FILES.neighborhood}.graphId`, "does not match graph content");
  }
  if (explorer.graphId !== visualizationGraphId(graphWithoutId(explorer))) {
    fail(`${RELEASE_FILES.explorer}.graphId`, "does not match graph content");
  }
  if (explorer.sourceGraphId !== neighborhood.graphId) {
    fail(`${RELEASE_FILES.explorer}.sourceGraphId`, "does not match recommendation graphId");
  }
  for (const field of ["dataset", "semantics", "config", "truncation"] as const) {
    if (!isDeepStrictEqual(explorer[field], neighborhood[field])) {
      fail(`${RELEASE_FILES.explorer}.${field}`, "does not match recommendation graph");
    }
  }
  const visualization = explorer.visualization;
  if (!visualization) fail(`${RELEASE_FILES.explorer}.visualization`, "is required");
  const expectedExplorer = buildExplorerGraph(neighborhood,
    visualization.maxAnimeAnimeEdges, visualization.maxUserAnimeEdges);
  if (!isDeepStrictEqual(explorer, expectedExplorer)) {
    fail(RELEASE_FILES.explorer, "is not the declared selection from the recommendation graph");
  }

  const catalog = parseReleaseIdentityCatalog(parseJson(files.catalog, RELEASE_FILES.catalog), RELEASE_FILES.catalog);
  if (catalog.datasetSha256 !== neighborhood.dataset.sha256) {
    fail(`${RELEASE_FILES.catalog}.datasetSha256`, "does not match recommendation dataset");
  }
  const sortedGraphAnime = [...neighborhood.anime].sort(([left], [right]) => left - right);
  if (!isDeepStrictEqual(catalog.anime, sortedGraphAnime)) {
    fail(`${RELEASE_FILES.catalog}.anime`, "must exactly match the recommendation ID/title map");
  }
  const itemMapSha256 = releaseItemMapSha256(catalog.anime);
  let modelEntry: ReleaseManifestV1["model"] = null;
  if (files.model) {
    const model = parseCompactModel(parseJson(files.model, RELEASE_FILES.model), RELEASE_FILES.model);
    if (model.datasetSha256 !== neighborhood.dataset.sha256) {
      fail(`${RELEASE_FILES.model}.datasetSha256`, "does not match recommendation dataset");
    }
    const catalogTitles = new Map(catalog.anime);
    const modelAnime: CompactAnimeEntry[] = model.animeIds.map((id, i) => [id, model.titles[i]]);
    modelAnime.forEach(([id, title], i) => {
      if (catalogTitles.get(id) !== title) {
        fail(`${RELEASE_FILES.model}.titles[${i}]`, `anime ${id} is absent or differs from the catalog`);
      }
    });
    modelEntry = {
      ...asset(files.model, RELEASE_FILES.model, "model-mf-compact-v1"),
      datasetSha256: model.datasetSha256,
      itemMapSha256: releaseItemMapSha256(modelAnime),
      coverage: { mappedAnimeCount: modelAnime.length, totalCatalogAnimeCount: catalog.anime.length },
    };
  }

  const payload: Omit<ReleaseManifestV1, "bundleId"> = {
    format: "release-manifest-v1", tag: options.tag, dataset: neighborhood.dataset,
    catalog: { ...asset(files.catalog, RELEASE_FILES.catalog, "anime-catalog-v1"),
      animeCount: catalog.anime.length, itemMapSha256 },
    neighborhood: { ...asset(files.neighborhood, RELEASE_FILES.neighborhood, "graph-compact-v2"),
      graphId: neighborhood.graphId },
    explorer: { ...asset(files.explorer, RELEASE_FILES.explorer, "graph-compact-v2"),
      graphId: explorer.graphId, sourceGraphId: explorer.sourceGraphId },
    model: modelEntry,
    lastKnownGood: options.lastKnownGood ?? null,
  };
  const manifest: ReleaseManifestV1 = {
    ...payload,
    bundleId: releaseSha256(JSON.stringify(payload)),
  };
  return parseReleaseManifest(manifest, RELEASE_FILES.manifest);
}

function readRequired(directory: string, filename: string): Buffer {
  try {
    return fs.readFileSync(path.join(directory, filename));
  } catch {
    fail(filename, `missing or unreadable in ${directory}`);
  }
}

function readFiles(directory: string): ReleaseFileBytes {
  const modelPath = path.join(directory, RELEASE_FILES.model);
  return {
    neighborhood: readRequired(directory, RELEASE_FILES.neighborhood),
    explorer: readRequired(directory, RELEASE_FILES.explorer),
    catalog: readRequired(directory, RELEASE_FILES.catalog),
    ...(fs.existsSync(modelPath) ? { model: readRequired(directory, RELEASE_FILES.model) } : {}),
  };
}

function verifyOne(directory: string): { manifest: ReleaseManifestV1; manifestBytes: Buffer } {
  const manifestBytes = readRequired(directory, RELEASE_FILES.manifest);
  const manifest = parseReleaseManifest(parseJson(manifestBytes, RELEASE_FILES.manifest), RELEASE_FILES.manifest);
  const files = readFiles(directory);
  if ((files.model !== undefined) !== (manifest.model !== null)) {
    fail(RELEASE_FILES.model, "presence does not match manifest.model");
  }
  const expected = buildReleaseManifest(files, {
    tag: manifest.tag, lastKnownGood: manifest.lastKnownGood ?? undefined,
    fixtureGenesis: manifest.lastKnownGood === null,
  });
  if (!isDeepStrictEqual(manifest, expected)) {
    fail(RELEASE_FILES.manifest, "fields, hashes, or bundleId differ from verified artifact bytes");
  }
  return { manifest, manifestBytes };
}

/** Verify all current bytes and the complete named prior bundle, without following the prior bundle's own chain. */
export function verifyReleaseBundle(directory: string, previousDirectory?: string,
  fixtureGenesis = false): ReleaseManifestV1 {
  const current = verifyOne(directory).manifest;
  if (current.lastKnownGood === null) {
    if (!fixtureGenesis || previousDirectory) {
      fail("lastKnownGood", "genesis requires explicit fixtureGenesis and no previous directory");
    }
    return current;
  }
  if (fixtureGenesis || !previousDirectory) {
    fail("lastKnownGood", "a separate previous bundle directory is required");
  }
  if (fs.realpathSync(directory) === fs.realpathSync(previousDirectory)) {
    fail("lastKnownGood", "previous bundle resolves to the current directory");
  }
  const previous = verifyOne(previousDirectory);
  const pointer = current.lastKnownGood;
  if (pointer.tag !== previous.manifest.tag || pointer.bundleId !== previous.manifest.bundleId ||
      pointer.manifestSha256 !== releaseSha256(previous.manifestBytes)) {
    fail("lastKnownGood", "tag, bundleId, or manifest-byte hash differs from named previous bundle");
  }
  return current;
}

/** Write once; a prior bundle must already pass the same byte and compatibility checks. */
export function writeReleaseManifest(directory: string, tag: string, previousDirectory?: string,
  fixtureGenesis = false): ReleaseManifestV1 {
  if (fs.existsSync(path.join(directory, RELEASE_FILES.manifest))) {
    fail(RELEASE_FILES.manifest, "already exists and will not be overwritten");
  }
  let lastKnownGood: Previous | undefined;
  if (previousDirectory) {
    if (fs.realpathSync(directory) === fs.realpathSync(previousDirectory)) {
      fail("lastKnownGood", "previous bundle resolves to the current directory");
    }
    const previous = verifyOne(previousDirectory);
    lastKnownGood = { tag: previous.manifest.tag, bundleId: previous.manifest.bundleId,
      manifestSha256: releaseSha256(previous.manifestBytes) };
  }
  const manifest = buildReleaseManifest(readFiles(directory), { tag, lastKnownGood, fixtureGenesis });
  fs.writeFileSync(path.join(directory, RELEASE_FILES.manifest), `${JSON.stringify(manifest, null, 2)}\n`,
    { encoding: "utf8", flag: "wx" });
  verifyReleaseBundle(directory, previousDirectory, fixtureGenesis);
  return manifest;
}
