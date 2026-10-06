/** Local v2 metadata-bundle construction. Publication workflows remain v1-only. */
import fs from "node:fs";
import path from "node:path";
import { isDeepStrictEqual } from "node:util";
import {
  parseBrowserReleaseManifest, parseCatalogMetadataSnapshot, parseReleaseIdentityCatalog,
  RELEASE_BUNDLE_LIMITS,
  type BrowserReleaseManifest, type ReleaseManifestV2,
} from "../../web/src/artifacts.js";
import {
  RELEASE_FILES, buildReleaseManifest, releaseSha256,
  type ReleaseBuildOptions, type ReleaseFileBytes,
} from "./release-manifest.js";

export const METADATA_RELEASE_FILE = "catalog.metadata.json";

export type MetadataReleaseFiles = Pick<ReleaseFileBytes, "neighborhood" | "explorer" | "catalog"> & {
  metadata: Buffer;
};

export type MetadataBuildOptions = Pick<ReleaseBuildOptions,
  "tag" | "lastKnownGood" | "fixtureGenesis">;

function fail(field: string, reason: string): never {
  throw new Error(`Metadata bundle ${field}: ${reason}`);
}

function json(bytes: Buffer, label: string): unknown {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    fail(label, "invalid JSON or UTF-8");
  }
}

function bounded(bytes: Buffer, label: string, maximum = RELEASE_BUNDLE_LIMITS.plainAssetBytes): void {
  if (bytes.length < 1 || bytes.length > maximum) fail(label, "is empty or exceeds the byte limit");
}

/** Rebuild the public manifest from public bytes; sourceDigest is only a declaration here. */
function assemble(files: MetadataReleaseFiles, options: MetadataBuildOptions,
  sourceDigest: string): ReleaseManifestV2 {
  if (options.fixtureGenesis && !options.tag.startsWith("data-vsynthetic-")) {
    fail("tag", "fixture genesis requires a data-vsynthetic- tag");
  }
  if ("model" in files) fail(RELEASE_FILES.model, "is not allowed in the data-only metadata candidate");
  for (const [name, bytes] of [
    [RELEASE_FILES.neighborhood, files.neighborhood],
    [RELEASE_FILES.explorer, files.explorer],
    [RELEASE_FILES.catalog, files.catalog],
    [METADATA_RELEASE_FILE, files.metadata],
  ] as const) bounded(bytes, name);
  const total = files.neighborhood.length + files.explorer.length +
    files.catalog.length + files.metadata.length;
  if (total > RELEASE_BUNDLE_LIMITS.totalPlainBytes) fail("assets", "total bytes exceed limit");

  const base = buildReleaseManifest({ neighborhood: files.neighborhood,
    explorer: files.explorer, catalog: files.catalog }, options);
  if (base.neighborhood.format !== "graph-compact-v3" || base.model !== null) {
    fail(RELEASE_FILES.neighborhood, "requires a data-only aggregate graph-compact-v3 bundle");
  }
  const snapshot = parseCatalogMetadataSnapshot(json(files.metadata, METADATA_RELEASE_FILE),
    METADATA_RELEASE_FILE);
  if (snapshot.source.snapshotSha256 !== sourceDigest) {
    fail(`${METADATA_RELEASE_FILE}.source.snapshotSha256`, "does not match the supplied source bytes");
  }
  const catalog = parseReleaseIdentityCatalog(json(files.catalog, RELEASE_FILES.catalog),
    RELEASE_FILES.catalog);
  const catalogIds = new Set(catalog.anime.map(([id]) => id));
  snapshot.anime.forEach((item, index) => {
    if (!catalogIds.has(item.animeId)) {
      fail(`${METADATA_RELEASE_FILE}.anime[${index}].animeId`,
        "is outside the identity catalog");
    }
  });
  const { bundleId: _baseId, ...basePayload } = base;
  const payload: Omit<ReleaseManifestV2, "bundleId"> = {
    ...basePayload,
    format: "release-manifest-v2",
    metadata: {
      path: METADATA_RELEASE_FILE,
      format: "anime-metadata-catalog-v1",
      sha256: releaseSha256(files.metadata),
      bytes: files.metadata.length,
      animeCount: snapshot.anime.length,
      itemMapSha256: base.catalog.itemMapSha256,
      sourceSnapshotSha256: sourceDigest,
    },
  };
  const manifest = { ...payload, bundleId: releaseSha256(JSON.stringify(payload)) };
  const parsed = parseBrowserReleaseManifest(manifest, RELEASE_FILES.manifest);
  if (parsed.format !== "release-manifest-v2") fail(RELEASE_FILES.manifest, "must be v2");
  return parsed;
}

/** The source bytes stay private and are never written into the release bundle. */
export function buildMetadataReleaseManifest(files: MetadataReleaseFiles,
  options: MetadataBuildOptions, sourceBytes: Buffer): ReleaseManifestV2 {
  bounded(sourceBytes, "sourceBytes");
  return assemble(files, options, releaseSha256(sourceBytes));
}

function readRequired(directory: string, filename: string, maximum: number): Buffer {
  const filepath = path.join(directory, filename);
  let stat: fs.Stats;
  try {
    stat = fs.lstatSync(filepath);
  } catch {
    fail(filename, "missing or unreadable");
  }
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > maximum) {
    fail(filename, "missing, linked, empty, or oversized");
  }
  try {
    return fs.readFileSync(filepath);
  } catch {
    fail(filename, "missing or unreadable");
  }
}

function assertRealDirectory(directory: string): void {
  try {
    const stat = fs.lstatSync(directory);
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      fail("directory", "must be a real directory");
    }
  } catch {
    fail("directory", "missing or unreadable real directory");
  }
}

function assertExactInventory(directory: string, expected: readonly string[]): void {
  assertRealDirectory(directory);
  let names: string[];
  try {
    names = fs.readdirSync(directory);
  } catch {
    fail("directory", "missing or unreadable real directory");
  }
  if (!isDeepStrictEqual(names.sort(), [...expected].sort())) {
    fail("directory", "contains missing or undeclared files");
  }
}

function sameDirectory(left: string, right: string): boolean {
  try {
    return fs.realpathSync(left) === fs.realpathSync(right);
  } catch {
    fail("lastKnownGood", "previous bundle directory is missing or unreadable");
  }
}

function assertTotal(files: ReleaseFileBytes): void {
  const total = files.neighborhood.length + files.explorer.length + files.catalog.length +
    (files.model?.length ?? 0);
  if (total > RELEASE_BUNDLE_LIMITS.totalPlainBytes) fail("assets", "total bytes exceed limit");
}

/** Verify one bundle's public bytes; prior linkage is checked by the caller. */
function verifyOne(directory: string): { manifest: BrowserReleaseManifest; manifestBytes: Buffer } {
  assertRealDirectory(directory);
  const manifestBytes = readRequired(directory, RELEASE_FILES.manifest,
    RELEASE_BUNDLE_LIMITS.manifestBytes);
  const manifest = parseBrowserReleaseManifest(json(manifestBytes, RELEASE_FILES.manifest),
    RELEASE_FILES.manifest);
  if (manifest.format === "release-manifest-v1") {
    assertExactInventory(directory, [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood,
      RELEASE_FILES.explorer, RELEASE_FILES.catalog,
      ...(manifest.model ? [RELEASE_FILES.model] : [])]);
    const files: ReleaseFileBytes = {
      neighborhood: readRequired(directory, RELEASE_FILES.neighborhood,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes),
      explorer: readRequired(directory, RELEASE_FILES.explorer,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes),
      catalog: readRequired(directory, RELEASE_FILES.catalog,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes),
      ...(manifest.model ? { model: readRequired(directory, RELEASE_FILES.model,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes) } : {}),
    };
    assertTotal(files);
    const expected = buildReleaseManifest(files, { tag: manifest.tag,
      lastKnownGood: manifest.lastKnownGood ?? undefined,
      fixtureGenesis: manifest.lastKnownGood === null });
    if (!isDeepStrictEqual(manifest, expected)) {
      fail(RELEASE_FILES.manifest, "fields, hashes, or bundleId differ from public bytes");
    }
  } else {
    assertExactInventory(directory, [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood,
      RELEASE_FILES.explorer, RELEASE_FILES.catalog, METADATA_RELEASE_FILE]);
    const files: MetadataReleaseFiles = {
      neighborhood: readRequired(directory, RELEASE_FILES.neighborhood,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes),
      explorer: readRequired(directory, RELEASE_FILES.explorer,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes),
      catalog: readRequired(directory, RELEASE_FILES.catalog,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes),
      metadata: readRequired(directory, METADATA_RELEASE_FILE,
        RELEASE_BUNDLE_LIMITS.plainAssetBytes),
    };
    const expected = assemble(files, { tag: manifest.tag,
      lastKnownGood: manifest.lastKnownGood ?? undefined,
      fixtureGenesis: manifest.lastKnownGood === null },
    manifest.metadata.sourceSnapshotSha256);
    if (!isDeepStrictEqual(manifest, expected)) {
      fail(RELEASE_FILES.manifest, "fields, hashes, or bundleId differ from public bytes");
    }
  }
  return { manifest, manifestBytes };
}

/** Public-byte check only; the source digest still needs private-source and rights review. */
export function verifyMetadataReleaseBundle(directory: string, previousDirectory?: string,
  fixtureGenesis = false): ReleaseManifestV2 {
  const current = verifyOne(directory);
  if (current.manifest.format !== "release-manifest-v2") {
    fail(RELEASE_FILES.manifest, "must be release-manifest-v2");
  }
  const priorLink = current.manifest.lastKnownGood;
  if (priorLink === null) {
    if (!fixtureGenesis || previousDirectory) {
      fail("lastKnownGood", "first metadata bundle is fixture-only without a reviewed prior");
    }
    return current.manifest;
  }
  if (fixtureGenesis || !previousDirectory) {
    fail("lastKnownGood", "requires a separate previous bundle directory");
  }
  if (sameDirectory(directory, previousDirectory)) {
    fail("lastKnownGood", "previous bundle resolves to current directory");
  }
  const prior = verifyOne(previousDirectory);
  if (prior.manifest.neighborhood.format !== "graph-compact-v3" || prior.manifest.model !== null) {
    fail("lastKnownGood", "requires a data-only aggregate predecessor");
  }
  if (priorLink.tag !== prior.manifest.tag || priorLink.bundleId !== prior.manifest.bundleId ||
      priorLink.manifestSha256 !== releaseSha256(prior.manifestBytes)) {
    fail("lastKnownGood", "tag, bundleId, or manifest-byte hash differs from predecessor");
  }
  return current.manifest;
}

/** Write once from a separate private source byte stream; no installer or publisher is changed. */
export function writeMetadataReleaseManifest(directory: string, tag: string, sourceBytes: Buffer,
  previousDirectory?: string, fixtureGenesis = false): ReleaseManifestV2 {
  if (fs.existsSync(path.join(directory, RELEASE_FILES.manifest))) {
    fail(RELEASE_FILES.manifest, "already exists and will not be overwritten");
  }
  let lastKnownGood: NonNullable<ReleaseManifestV2["lastKnownGood"]> | undefined;
  if (previousDirectory) {
    if (sameDirectory(directory, previousDirectory)) {
      fail("lastKnownGood", "previous bundle resolves to current directory");
    }
    const prior = verifyOne(previousDirectory);
    if (prior.manifest.neighborhood.format !== "graph-compact-v3" || prior.manifest.model !== null) {
      fail("lastKnownGood", "requires a data-only aggregate predecessor");
    }
    lastKnownGood = { tag: prior.manifest.tag, bundleId: prior.manifest.bundleId,
      manifestSha256: releaseSha256(prior.manifestBytes) };
  }
  assertExactInventory(directory, [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    RELEASE_FILES.catalog, METADATA_RELEASE_FILE]);
  const files: MetadataReleaseFiles = {
    neighborhood: readRequired(directory, RELEASE_FILES.neighborhood,
      RELEASE_BUNDLE_LIMITS.plainAssetBytes),
    explorer: readRequired(directory, RELEASE_FILES.explorer,
      RELEASE_BUNDLE_LIMITS.plainAssetBytes),
    catalog: readRequired(directory, RELEASE_FILES.catalog,
      RELEASE_BUNDLE_LIMITS.plainAssetBytes),
    metadata: readRequired(directory, METADATA_RELEASE_FILE,
      RELEASE_BUNDLE_LIMITS.plainAssetBytes),
  };
  const manifest = buildMetadataReleaseManifest(files,
    { tag, lastKnownGood, fixtureGenesis }, sourceBytes);
  fs.writeFileSync(path.join(directory, RELEASE_FILES.manifest),
    `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  verifyMetadataReleaseBundle(directory, previousDirectory, fixtureGenesis);
  return manifest;
}
