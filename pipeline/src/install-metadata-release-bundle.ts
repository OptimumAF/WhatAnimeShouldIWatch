/** Local synthetic v2 installation. No GitHub transport or publication entry point uses this. */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import {
  parseActiveReleaseBundle, parseBrowserReleaseManifest, RELEASE_BUNDLE_LIMITS,
  type ActiveReleaseBundleV1, type BrowserReleaseManifest, type ReleaseManifestAsset,
  type ReleaseManifestV2,
} from "../../web/src/artifacts.js";
import { readReleaseResponse, type ReleaseAssetTransport } from "./install-release-bundle.js";
import {
  METADATA_RELEASE_FILE, verifyLocalDataOnlyBundle, verifyMetadataReleaseBundle,
} from "./metadata-release-bundle.js";
import { RELEASE_FILES, releaseSha256 } from "./release-manifest.js";

export interface InstallMetadataReleaseOptions {
  storeDir: string;
  tag: string;
  transport: ReleaseAssetTransport;
  /** This candidate route accepts invented bundles only. */
  fixtureOnly: true;
  /** Test seam after staging a complete versioned directory and before pointer activation. */
  beforeActivate?: () => void;
}

export interface InstallMetadataReleaseResult {
  manifest: ReleaseManifestV2;
  bundleDir: string;
  changed: boolean;
}

function fail(field: string, reason: string): never {
  throw new Error(`Metadata install ${field}: ${reason}`);
}

function parseJson(bytes: Buffer, label: string): unknown {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    fail(label, "invalid JSON or UTF-8");
  }
}

function readSmall(filepath: string, maximum: number, label: string): Buffer {
  let stat: fs.Stats;
  try {
    stat = fs.lstatSync(filepath);
  } catch {
    fail(label, "missing or unreadable");
  }
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > maximum) {
    fail(label, "missing, linked, empty, or oversized");
  }
  try {
    return fs.readFileSync(filepath);
  } catch {
    fail(label, "missing or unreadable");
  }
}

function assertRealDirectory(directory: string, label: string): void {
  let stat: fs.Stats;
  try {
    stat = fs.lstatSync(directory);
  } catch {
    fail(label, "missing or unreadable real directory");
  }
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    fail(label, "must be a real directory");
  }
}

function pathPresent(filepath: string, label: string): boolean {
  try {
    fs.lstatSync(filepath);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    fail(label, "cannot inspect path");
  }
}

function bundlePath(storeDir: string, bundleId: string): string {
  if (!/^[a-f0-9]{64}$/.test(bundleId)) fail("bundleId", "must be a lowercase SHA-256 digest");
  return path.join(storeDir, "bundles", bundleId);
}

function assertFixtureGenesisStore(storeDir: string, bundleId: string): void {
  const names = fs.readdirSync(storeDir);
  if (names.some((name) => name !== ".install.lock" && name !== "bundles")) {
    fail("storeDir", "fixture genesis requires an empty store");
  }
  const bundlesDir = path.join(storeDir, "bundles");
  if (pathPresent(bundlesDir, "bundles")) {
    assertRealDirectory(bundlesDir, "bundles");
    if (fs.readdirSync(bundlesDir).some((name) => name !== bundleId)) {
      fail("bundles", "fixture genesis permits only its unactivated candidate");
    }
  }
}

function parseManifest(bytes: Buffer): BrowserReleaseManifest {
  return parseBrowserReleaseManifest(parseJson(bytes, RELEASE_FILES.manifest), RELEASE_FILES.manifest);
}

function readActive(storeDir: string):
  { pointer: ActiveReleaseBundleV1; manifest: BrowserReleaseManifest; directory: string } | null {
  const pointerPath = path.join(storeDir, "active.json");
  if (!pathPresent(pointerPath, "active.json")) return null;
  const pointer = parseActiveReleaseBundle(parseJson(readSmall(pointerPath,
    RELEASE_BUNDLE_LIMITS.activePointerBytes, "active.json"), "active.json"), "active.json");
  const directory = bundlePath(storeDir, pointer.bundleId);
  assertRealDirectory(directory, "active bundle");
  const manifestBytes = readSmall(path.join(directory, RELEASE_FILES.manifest),
    RELEASE_BUNDLE_LIMITS.manifestBytes, RELEASE_FILES.manifest);
  if (releaseSha256(manifestBytes) !== pointer.manifestSha256) {
    fail("active.json.manifestSha256", "differs from the active manifest bytes");
  }
  const draft = parseManifest(manifestBytes);
  if (!draft.tag.startsWith("data-vsynthetic-")) {
    fail("active.json.tag", "fixture installer requires a synthetic active bundle");
  }
  const priorDirectory = draft.lastKnownGood
    ? bundlePath(storeDir, draft.lastKnownGood.bundleId) : undefined;
  const manifest = verifyLocalDataOnlyBundle(directory, priorDirectory, !priorDirectory);
  if (pointer.tag !== manifest.tag || pointer.bundleId !== manifest.bundleId) {
    fail("active.json", "tag or bundleId differs from its verified manifest");
  }
  return { pointer, manifest, directory };
}

function entries(manifest: ReleaseManifestV2): ReleaseManifestAsset[] {
  return [manifest.catalog, manifest.neighborhood, manifest.explorer, manifest.metadata];
}

function declaredSizes(manifest: ReleaseManifestV2): void {
  let total = 0;
  for (const entry of entries(manifest)) {
    if (entry.bytes > RELEASE_BUNDLE_LIMITS.plainAssetBytes) {
      fail(entry.path, "declared plain-byte size exceeds limit");
    }
    total += entry.bytes;
  }
  if (total > RELEASE_BUNDLE_LIMITS.totalPlainBytes) {
    fail("assets", "declared total plain bytes exceed limit");
  }
}

function selectInventory(transport: ReleaseAssetTransport, manifest: ReleaseManifestV2):
  Map<string, { name: string; compressed: boolean }> {
  const seen = new Set<string>();
  for (const name of transport.assets) {
    if (seen.has(name)) fail("assets", `duplicate asset ${name}`);
    seen.add(name);
  }
  if (!seen.has(RELEASE_FILES.manifest)) fail(RELEASE_FILES.manifest, "missing release asset");
  const selected = new Map<string, { name: string; compressed: boolean }>();
  const expected = new Set<string>([RELEASE_FILES.manifest]);
  for (const entry of entries(manifest)) {
    const plain = seen.has(entry.path);
    const compressed = seen.has(`${entry.path}.gz`);
    if (plain === compressed) {
      fail(entry.path, plain ? "ambiguous plain and gzip assets" : "missing release asset");
    }
    const name = compressed ? `${entry.path}.gz` : entry.path;
    selected.set(entry.path, { name, compressed });
    expected.add(name);
  }
  for (const name of seen) {
    if (!expected.has(name)) fail("assets", `undeclared asset ${name}`);
  }
  return selected;
}

async function downloadAsset(transport: ReleaseAssetTransport, entry: ReleaseManifestAsset,
  selected: { name: string; compressed: boolean }): Promise<Buffer> {
  const maximum = selected.compressed ? RELEASE_BUNDLE_LIMITS.compressedAssetBytes : entry.bytes;
  const received = await readReleaseResponse(await transport.fetchAsset(selected.name),
    maximum, selected.name);
  let plain: Buffer;
  if (selected.compressed) {
    try {
      plain = zlib.gunzipSync(received,
        { maxOutputLength: Math.min(entry.bytes, RELEASE_BUNDLE_LIMITS.plainAssetBytes) + 1 });
    } catch {
      fail(selected.name, "invalid gzip or decompressed size exceeds limit");
    }
  } else {
    plain = received;
  }
  if (plain.length !== entry.bytes || releaseSha256(plain) !== entry.sha256) {
    fail(entry.path, "plain-byte length or SHA-256 differs from manifest");
  }
  return plain;
}

function removeStaging(stage: string, storeDir: string): void {
  if (!fs.existsSync(stage)) return;
  const resolvedStore = fs.realpathSync(storeDir);
  const resolvedStage = fs.realpathSync(stage);
  if (path.dirname(resolvedStage) !== resolvedStore ||
      !path.basename(resolvedStage).startsWith(".incoming-metadata-") ||
      fs.lstatSync(stage).isSymbolicLink()) {
    fail("staging", "refusing cleanup outside the installer store");
  }
  fs.rmSync(stage, { recursive: true, force: true });
}

function activate(storeDir: string, pointer: ActiveReleaseBundleV1): void {
  const temporary = path.join(storeDir, `.active-metadata-${crypto.randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(pointer, null, 2)}\n`,
      { encoding: "utf8", flag: "wx" });
    fs.renameSync(temporary, path.join(storeDir, "active.json"));
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

/** Install a fully verified invented v2 bundle before changing the sole active pointer. */
export async function installMetadataReleaseBundle(options: InstallMetadataReleaseOptions):
  Promise<InstallMetadataReleaseResult> {
  if (options.fixtureOnly !== true || !/^data-vsynthetic-[A-Za-z0-9._-]+$/.test(options.tag) ||
      options.transport.tag !== options.tag) {
    fail("tag", "requires a matching explicit data-vsynthetic- tag and fixture mode");
  }
  const storeDir = path.resolve(options.storeDir);
  fs.mkdirSync(storeDir, { recursive: true });
  assertRealDirectory(storeDir, "storeDir");
  const bundlesDir = path.join(storeDir, "bundles");
  if (pathPresent(bundlesDir, "bundles")) assertRealDirectory(bundlesDir, "bundles");
  const lockPath = path.join(storeDir, ".install.lock");
  let lock: number;
  try {
    lock = fs.openSync(lockPath, "wx");
  } catch {
    fail("lock", "another installation or stale lock is present");
  }
  let stage: string | null = null;
  try {
    const current = readActive(storeDir);
    const activePath = path.join(storeDir, "active.json");
    const originalPointerBytes = current ? readSmall(activePath,
      RELEASE_BUNDLE_LIMITS.activePointerBytes, "active.json") : null;
    const manifestBytes = await readReleaseResponse(
      await options.transport.fetchAsset(RELEASE_FILES.manifest),
      RELEASE_BUNDLE_LIMITS.manifestBytes, RELEASE_FILES.manifest);
    const parsed = parseManifest(manifestBytes);
    if (parsed.format !== "release-manifest-v2") {
      fail(RELEASE_FILES.manifest, "must be release-manifest-v2");
    }
    const manifest = parsed;
    if (manifest.tag !== options.tag) fail("release-manifest.json.tag", "differs from requested tag");
    if (manifest.model !== null || manifest.neighborhood.format !== "graph-compact-v3") {
      fail(RELEASE_FILES.neighborhood, "requires a data-only aggregate graph-compact-v3 bundle");
    }
    declaredSizes(manifest);
    const inventory = selectInventory(options.transport, manifest);
    const manifestSha256 = releaseSha256(manifestBytes);
    if (current?.pointer.bundleId === manifest.bundleId) {
      if (current.manifest.format !== "release-manifest-v2" ||
          current.pointer.manifestSha256 !== manifestSha256) {
        fail(RELEASE_FILES.manifest, "same bundleId has different manifest bytes");
      }
      return { manifest: current.manifest,
        bundleDir: current.directory, changed: false };
    }
    if (current) {
      const prior = manifest.lastKnownGood;
      if (!prior || prior.tag !== current.pointer.tag ||
          prior.bundleId !== current.pointer.bundleId ||
          prior.manifestSha256 !== current.pointer.manifestSha256) {
        fail("lastKnownGood", "must exactly identify the currently active bundle");
      }
    } else if (manifest.lastKnownGood !== null) {
      fail("lastKnownGood", "an empty fixture store requires a genesis bundle");
    }
    if (!current) assertFixtureGenesisStore(storeDir, manifest.bundleId);

    const destination = bundlePath(storeDir, manifest.bundleId);
    if (pathPresent(destination, "bundleId")) {
      const existing = verifyMetadataReleaseBundle(destination, current?.directory, !current);
      const existingBytes = readSmall(path.join(destination, RELEASE_FILES.manifest),
        RELEASE_BUNDLE_LIMITS.manifestBytes, RELEASE_FILES.manifest);
      if (existing.bundleId !== manifest.bundleId || releaseSha256(existingBytes) !== manifestSha256) {
        fail("bundleId", "pre-existing versioned directory differs from downloaded manifest");
      }
    } else {
      fs.mkdirSync(bundlesDir, { recursive: true });
      assertRealDirectory(bundlesDir, "bundles");
      stage = fs.mkdtempSync(path.join(storeDir, ".incoming-metadata-"));
      fs.writeFileSync(path.join(stage, RELEASE_FILES.manifest), manifestBytes, { flag: "wx" });
      for (const entry of entries(manifest)) {
        const selected = inventory.get(entry.path)!;
        fs.writeFileSync(path.join(stage, entry.path),
          await downloadAsset(options.transport, entry, selected), { flag: "wx" });
      }
      verifyMetadataReleaseBundle(stage, current?.directory, !current);
      fs.renameSync(stage, destination);
      stage = null;
    }
    options.beforeActivate?.();
    if (originalPointerBytes) {
      if (!readSmall(activePath, RELEASE_BUNDLE_LIMITS.activePointerBytes,
        "active.json").equals(originalPointerBytes)) {
        fail("active.json", "changed during installation verification");
      }
    } else if (pathPresent(activePath, "active.json")) {
      fail("active.json", "appeared during installation verification");
    }
    verifyMetadataReleaseBundle(destination, current?.directory, !current);
    activate(storeDir, { format: "active-release-bundle-v1", tag: manifest.tag,
      bundleId: manifest.bundleId, manifestSha256 });
    return { manifest, bundleDir: destination, changed: true };
  } finally {
    try {
      if (stage) removeStaging(stage, storeDir);
    } finally {
      fs.closeSync(lock);
      fs.unlinkSync(lockPath);
    }
  }
}
