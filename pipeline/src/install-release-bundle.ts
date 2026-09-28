import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import {
  parseActiveReleaseBundle, parseReleaseManifest, RELEASE_BUNDLE_LIMITS,
  type ActiveReleaseBundleV1, type ReleaseManifestAsset, type ReleaseManifestV1,
} from "../../web/src/artifacts.js";
import { RELEASE_FILES, releaseSha256, verifyReleaseBundle } from "./release-manifest.js";

export const RELEASE_DOWNLOAD_LIMITS = RELEASE_BUNDLE_LIMITS;

/** A release inventory and fetch implementation can be injected without contacting GitHub. */
export interface ReleaseAssetTransport {
  tag: string;
  assets: readonly string[];
  fetchAsset(name: string): Promise<Response>;
}

export interface InstallReleaseOptions {
  storeDir: string;
  tag: string;
  transport: ReleaseAssetTransport;
  /** Only invented fixture bootstrap may install a genesis bundle. */
  fixtureBootstrap?: boolean;
  /** Supplied only after a separate exact published-release and owner-approval check. */
  reviewedGenesis?: { tag: string; bundleId: string; manifestSha256: string };
  /** An independently verified immutable predecessor for an empty deployment store. */
  verifiedPriorDir?: string;
  /** Test seam for a failure after complete directory staging but before pointer activation. */
  beforeActivate?: () => void;
}

export interface InstallReleaseResult {
  manifest: ReleaseManifestV1;
  bundleDir: string;
  changed: boolean;
}

function fail(field: string, reason: string): never {
  throw new Error(`Release install ${field}: ${reason}`);
}

function json(bytes: Buffer, label: string): unknown {
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
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size > maximum || stat.size < 1) {
    fail(label, "missing, linked, empty, or oversized");
  }
  return fs.readFileSync(filepath);
}

function assertStoredLimits(directory: string): void {
  readSmall(path.join(directory, RELEASE_FILES.manifest), RELEASE_DOWNLOAD_LIMITS.manifestBytes,
    RELEASE_FILES.manifest);
  let total = 0;
  for (const filename of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    RELEASE_FILES.catalog, RELEASE_FILES.model]) {
    const filepath = path.join(directory, filename);
    if (!fs.existsSync(filepath)) continue;
    const stat = fs.lstatSync(filepath);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 ||
        stat.size > RELEASE_DOWNLOAD_LIMITS.plainAssetBytes) {
      fail(filename, "stored file is empty or oversized");
    }
    total += stat.size;
  }
  if (total > RELEASE_DOWNLOAD_LIMITS.totalPlainBytes) fail("bundle", "stored plain bytes exceed limit");
}

function bundlePath(storeDir: string, bundleId: string): string {
  if (!/^[a-f0-9]{64}$/.test(bundleId)) fail("bundleId", "must be a lowercase SHA-256 digest");
  return path.join(storeDir, "bundles", bundleId);
}

function assertDirectoryNotLink(directory: string, label: string): void {
  const stat = fs.lstatSync(directory);
  if (!stat.isDirectory() || stat.isSymbolicLink()) fail(label, "must be a real directory");
}

function verifyStored(directory: string, priorDirectory: string | undefined,
  fixtureBootstrap: boolean): ReleaseManifestV1 {
  assertDirectoryNotLink(directory, directory);
  assertStoredLimits(directory);
  if (priorDirectory) {
    assertDirectoryNotLink(priorDirectory, priorDirectory);
    assertStoredLimits(priorDirectory);
  }
  return verifyReleaseBundle(directory, priorDirectory, fixtureBootstrap);
}

function readActive(storeDir: string, fixtureBootstrap: boolean):
  { pointer: ActiveReleaseBundleV1; manifest: ReleaseManifestV1; directory: string } | null {
  const activePath = path.join(storeDir, "active.json");
  if (!fs.existsSync(activePath)) return null;
  const pointer = parseActiveReleaseBundle(json(readSmall(activePath,
    RELEASE_DOWNLOAD_LIMITS.activePointerBytes, "active.json"), "active.json"), "active.json");
  const directory = bundlePath(storeDir, pointer.bundleId);
  assertDirectoryNotLink(directory, directory);
  const manifestPath = path.join(directory, RELEASE_FILES.manifest);
  const manifestBytes = readSmall(manifestPath, RELEASE_DOWNLOAD_LIMITS.manifestBytes,
    RELEASE_FILES.manifest);
  if (releaseSha256(manifestBytes) !== pointer.manifestSha256) {
    fail("active.json.manifestSha256", "does not match the active manifest bytes");
  }
  const draft = parseReleaseManifest(json(manifestBytes, RELEASE_FILES.manifest), RELEASE_FILES.manifest);
  const priorDirectory = draft.lastKnownGood
    ? bundlePath(storeDir, draft.lastKnownGood.bundleId) : undefined;
  if (!priorDirectory && !fixtureBootstrap) fail("active.json", "genesis is fixture-only");
  const manifest = verifyStored(directory, priorDirectory, !priorDirectory);
  if (pointer.tag !== manifest.tag || pointer.bundleId !== manifest.bundleId) {
    fail("active.json", "tag or bundleId differs from its verified manifest");
  }
  return { pointer, manifest, directory };
}

export async function readReleaseResponse(response: Response, maximum: number, label: string): Promise<Buffer> {
  if (!response.ok) fail(label, `download failed with status ${response.status}`);
  const advertised = response.headers.get("content-length");
  if (advertised !== null && /^\d+$/.test(advertised) && Number(advertised) > maximum) {
    fail(label, "advertised size exceeds limit");
  }
  if (!response.body) fail(label, "missing response body");
  const reader = response.body.getReader();
  const chunks: Buffer[] = [];
  let length = 0;
  let completed = false;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) { completed = true; break; }
      length += next.value.byteLength;
      if (length > maximum) fail(label, "download exceeds size limit");
      chunks.push(Buffer.from(next.value));
    }
  } finally {
    if (!completed) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
  if (length === 0) fail(label, "download is empty");
  return Buffer.concat(chunks, length);
}

function validateInventory(transport: ReleaseAssetTransport, modelPresent: boolean): void {
  const allowed = new Set<string>([RELEASE_FILES.manifest]);
  for (const filename of [RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    RELEASE_FILES.catalog, ...(modelPresent ? [RELEASE_FILES.model] : [])]) {
    allowed.add(filename);
    allowed.add(`${filename}.gz`);
  }
  const seen = new Set<string>();
  for (const name of transport.assets) {
    if (seen.has(name)) fail("assets", `duplicate asset ${name}`);
    seen.add(name);
    if (!allowed.has(name)) fail("assets", `undeclared asset ${name}`);
  }
  if (!seen.has(RELEASE_FILES.manifest)) fail(RELEASE_FILES.manifest, "missing release asset");
}

function selectAsset(names: ReadonlySet<string>, entry: ReleaseManifestAsset):
  { name: string; compressed: boolean } {
  const plain = names.has(entry.path);
  const compressed = names.has(`${entry.path}.gz`);
  if (plain === compressed) fail(entry.path, plain ? "ambiguous plain and gzip assets" : "missing release asset");
  return compressed ? { name: `${entry.path}.gz`, compressed: true }
    : { name: entry.path, compressed: false };
}

async function downloadAsset(transport: ReleaseAssetTransport, names: ReadonlySet<string>,
  entry: ReleaseManifestAsset): Promise<Buffer> {
  const selected = selectAsset(names, entry);
  const downloadLimit = selected.compressed
    ? RELEASE_DOWNLOAD_LIMITS.compressedAssetBytes : Math.min(entry.bytes,
      RELEASE_DOWNLOAD_LIMITS.plainAssetBytes);
  const received = await readReleaseResponse(await transport.fetchAsset(selected.name), downloadLimit,
    selected.name);
  let plain: Buffer;
  if (selected.compressed) {
    try {
      plain = zlib.gunzipSync(received,
        { maxOutputLength: Math.min(entry.bytes, RELEASE_DOWNLOAD_LIMITS.plainAssetBytes) + 1 });
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

function assertDeclaredSizes(manifest: ReleaseManifestV1): void {
  const entries = [manifest.catalog, manifest.neighborhood, manifest.explorer,
    ...(manifest.model ? [manifest.model] : [])];
  let total = 0;
  for (const entry of entries) {
    if (entry.bytes > RELEASE_DOWNLOAD_LIMITS.plainAssetBytes) {
      fail(entry.path, "declared plain-byte size exceeds limit");
    }
    total += entry.bytes;
  }
  if (total > RELEASE_DOWNLOAD_LIMITS.totalPlainBytes) {
    fail("bundle", "declared total plain-byte size exceeds limit");
  }
}

function removeStaging(stage: string, storeDir: string): void {
  if (!fs.existsSync(stage)) return;
  const resolvedStore = fs.realpathSync(storeDir);
  const resolvedStage = fs.realpathSync(stage);
  if (path.dirname(resolvedStage) !== resolvedStore ||
      !path.basename(resolvedStage).startsWith(".incoming-") ||
      fs.lstatSync(stage).isSymbolicLink()) {
    fail("staging", "refusing cleanup outside the installer store");
  }
  fs.rmSync(stage, { recursive: true, force: true });
}

function activate(storeDir: string, pointer: ActiveReleaseBundleV1): void {
  const temporary = path.join(storeDir, `.active-${crypto.randomUUID()}.tmp`);
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(pointer, null, 2)}\n`,
      { encoding: "utf8", flag: "wx" });
    fs.renameSync(temporary, path.join(storeDir, "active.json"));
  } finally {
    if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
  }
}

/** Keep the exact verified predecessor beside a fresh Pages candidate for recovery. */
function storeVerifiedPrior(storeDir: string, external: string, candidateDirectory: string): void {
  verifyStored(candidateDirectory, external, false);
  const priorBytes = readSmall(path.join(external, RELEASE_FILES.manifest),
    RELEASE_DOWNLOAD_LIMITS.manifestBytes, "verifiedPriorDir.release-manifest.json");
  const prior = parseReleaseManifest(json(priorBytes, "verifiedPriorDir.release-manifest.json"),
    "verifiedPriorDir.release-manifest.json");
  const stored = bundlePath(storeDir, prior.bundleId);
  if (fs.existsSync(stored)) {
    verifyStored(candidateDirectory, stored, false);
    return;
  }
  const stage = fs.mkdtempSync(path.join(storeDir, ".incoming-prior-"));
  try {
    for (const name of [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood,
      RELEASE_FILES.explorer, RELEASE_FILES.catalog,
      ...(prior.model ? [RELEASE_FILES.model] : [])]) {
      const maximum = name === RELEASE_FILES.manifest
        ? RELEASE_DOWNLOAD_LIMITS.manifestBytes : RELEASE_DOWNLOAD_LIMITS.plainAssetBytes;
      fs.writeFileSync(path.join(stage, name), readSmall(path.join(external, name),
        maximum, `verifiedPriorDir.${name}`), { flag: "wx" });
    }
    verifyStored(candidateDirectory, stage, false);
    fs.renameSync(stage, stored);
  } finally {
    if (fs.existsSync(stage)) removeStaging(stage, storeDir);
  }
}

/** Install a fully verified local bundle before changing the sole active pointer. */
export async function installReleaseBundle(options: InstallReleaseOptions): Promise<InstallReleaseResult> {
  if (!/^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(options.tag) ||
      options.transport.tag !== options.tag) {
    fail("tag", "requires a matching explicit versioned release tag");
  }
  if ((options.fixtureBootstrap && options.reviewedGenesis) ||
      (options.verifiedPriorDir && (options.fixtureBootstrap || options.reviewedGenesis))) {
    fail("bootstrap", "reviewed genesis, fixture genesis, and a named prior are exclusive");
  }
  const storeDir = path.resolve(options.storeDir);
  fs.mkdirSync(storeDir, { recursive: true });
  assertDirectoryNotLink(storeDir, "storeDir");
  const lockPath = path.join(storeDir, ".install.lock");
  let lock: number;
  try {
    lock = fs.openSync(lockPath, "wx");
  } catch {
    fail("lock", "another installation or stale lock is present");
  }
  let stage: string | null = null;
  try {
    const current = readActive(storeDir,
      options.fixtureBootstrap === true || options.reviewedGenesis !== undefined);
    if (current && options.verifiedPriorDir) fail("verifiedPriorDir", "requires an empty store");
    if (!current && !options.fixtureBootstrap && !options.reviewedGenesis &&
        !options.verifiedPriorDir) fail("active.json", "a verified prior bundle is required");
    const inventory = new Set(options.transport.assets);
    if (!inventory.has(RELEASE_FILES.manifest)) fail(RELEASE_FILES.manifest, "missing release asset");
    const manifestBytes = await readReleaseResponse(await options.transport.fetchAsset(RELEASE_FILES.manifest),
      RELEASE_DOWNLOAD_LIMITS.manifestBytes, RELEASE_FILES.manifest);
    const manifest = parseReleaseManifest(json(manifestBytes, RELEASE_FILES.manifest), RELEASE_FILES.manifest);
    if (manifest.tag !== options.tag) fail("release-manifest.json.tag", "differs from requested tag");
    assertDeclaredSizes(manifest);
    validateInventory(options.transport, manifest.model !== null);
    const manifestSha256 = releaseSha256(manifestBytes);
    if (options.reviewedGenesis && (manifest.lastKnownGood !== null ||
        options.reviewedGenesis.tag !== manifest.tag ||
        options.reviewedGenesis.bundleId !== manifest.bundleId ||
        options.reviewedGenesis.manifestSha256 !== manifestSha256)) {
      fail("reviewedGenesis", "does not match the exact first published bundle");
    }
    let priorDirectory = current?.directory;
    if (options.verifiedPriorDir) {
      const external = path.resolve(options.verifiedPriorDir);
      assertDirectoryNotLink(external, "verifiedPriorDir");
      const priorBytes = readSmall(path.join(external, RELEASE_FILES.manifest),
        RELEASE_DOWNLOAD_LIMITS.manifestBytes, "verifiedPriorDir.release-manifest.json");
      const prior = parseReleaseManifest(json(priorBytes, "verifiedPriorDir.release-manifest.json"),
        "verifiedPriorDir.release-manifest.json");
      if (!manifest.lastKnownGood || manifest.lastKnownGood.tag !== prior.tag ||
          manifest.lastKnownGood.bundleId !== prior.bundleId ||
          manifest.lastKnownGood.manifestSha256 !== releaseSha256(priorBytes)) {
        fail("verifiedPriorDir", "does not match the candidate's exact named predecessor");
      }
      priorDirectory = external;
    }
    if (current?.pointer.bundleId === manifest.bundleId) {
      if (current.pointer.manifestSha256 !== manifestSha256) {
        fail("release-manifest.json", "same bundleId has different manifest bytes");
      }
      return { manifest: current.manifest, bundleDir: current.directory, changed: false };
    }
    if (current) {
      const prior = manifest.lastKnownGood;
      if (!prior || prior.tag !== current.pointer.tag ||
          prior.bundleId !== current.pointer.bundleId ||
          prior.manifestSha256 !== current.pointer.manifestSha256) {
        fail("lastKnownGood", "must exactly identify the currently active bundle");
      }
    } else if (!options.verifiedPriorDir && manifest.lastKnownGood !== null) {
      fail("lastKnownGood", "genesis must not name a prior bundle");
    }

    const destination = bundlePath(storeDir, manifest.bundleId);
    if (fs.existsSync(destination)) {
      const existing = verifyStored(destination, priorDirectory, !priorDirectory);
      const existingManifestSha256 = releaseSha256(fs.readFileSync(path.join(destination,
        RELEASE_FILES.manifest)));
      if (existing.bundleId !== manifest.bundleId || existingManifestSha256 !== manifestSha256) {
        fail("bundleId", "pre-existing versioned directory differs from downloaded manifest");
      }
      if (options.verifiedPriorDir && priorDirectory) {
        storeVerifiedPrior(storeDir, priorDirectory, destination);
      }
    } else {
      const bundlesDir = path.join(storeDir, "bundles");
      fs.mkdirSync(bundlesDir, { recursive: true });
      assertDirectoryNotLink(bundlesDir, "bundles");
      stage = fs.mkdtempSync(path.join(storeDir, ".incoming-"));
      fs.writeFileSync(path.join(stage, RELEASE_FILES.manifest), manifestBytes, { flag: "wx" });
      const entries = [manifest.catalog, manifest.neighborhood, manifest.explorer,
        ...(manifest.model ? [manifest.model] : [])];
      for (const entry of entries) {
        const plain = await downloadAsset(options.transport, inventory, entry);
        fs.writeFileSync(path.join(stage, entry.path), plain, { flag: "wx" });
      }
      verifyStored(stage, priorDirectory, !priorDirectory);
      if (options.verifiedPriorDir && priorDirectory) {
        storeVerifiedPrior(storeDir, priorDirectory, stage);
      }
      fs.renameSync(stage, destination);
      stage = null;
    }
    options.beforeActivate?.();
    const pointer: ActiveReleaseBundleV1 = {
      format: "active-release-bundle-v1", tag: manifest.tag,
      bundleId: manifest.bundleId, manifestSha256,
    };
    activate(storeDir, pointer);
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
