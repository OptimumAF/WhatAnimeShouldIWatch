import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";
import { installReleaseBundle, type ReleaseAssetTransport } from
  "../src/install-release-bundle.js";
import { installMetadataReleaseBundle } from "../src/install-metadata-release-bundle.js";
import { buildMetadataReleaseManifest, METADATA_RELEASE_FILE,
  type MetadataReleaseFiles } from "../src/metadata-release-bundle.js";
import { buildReleaseManifest, RELEASE_FILES, releaseSha256 } from "../src/release-manifest.js";

const fixtureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)),
  "../../web/public/demo-data");
const fixture = (name: string): Buffer => fs.readFileSync(path.join(fixtureDir, name));
const encode = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const sourceBytes = Buffer.from("invented-local-metadata-source");

function files(): MetadataReleaseFiles {
  return {
    neighborhood: fixture("graph.aggregate.compact.json"),
    explorer: fixture("graph-explorer.aggregate.compact.json"),
    catalog: fixture(RELEASE_FILES.catalog),
    metadata: encode({ format: "anime-metadata-catalog-v1",
      source: { name: "invented-fixture", snapshotAt: "2026-09-24T00:00:00.000Z",
        snapshotSha256: releaseSha256(sourceBytes) },
      anime: [
        { animeId: 101, sourceItemId: "invented:101", title: "Copper Comet",
          aliases: ["Copper Voyage"], genres: ["Adventure"], year: 2021,
          mediaFormat: "TV", episodeCount: 12, runtimeMinutes: 24,
          contentClassification: null, communityScore: null, relations: null },
      ] }),
  };
}

function directory(t: TestContext): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-metadata-install-"));
  t.after(() => {
    const resolved = fs.realpathSync(root);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  });
  return root;
}

function transport(tag: string, manifestBytes: Buffer,
  source: MetadataReleaseFiles, compressed = true): ReleaseAssetTransport & {
    bytes: Map<string, Buffer>; requested: string[];
  } {
  const bytes = new Map<string, Buffer>([[RELEASE_FILES.manifest, manifestBytes]]);
  for (const [name, value] of [
    [RELEASE_FILES.neighborhood, source.neighborhood],
    [RELEASE_FILES.explorer, source.explorer],
    [RELEASE_FILES.catalog, source.catalog],
    [METADATA_RELEASE_FILE, source.metadata],
  ] as [string, Buffer][]) {
    bytes.set(compressed ? `${name}.gz` : name,
      compressed ? zlib.gzipSync(value) : value);
  }
  const requested: string[] = [];
  return { tag, bytes, requested, get assets() { return [...bytes.keys()]; },
    async fetchAsset(name) {
      requested.push(name);
      const value = bytes.get(name);
      return value ? new Response(new Uint8Array(value), { status: 200 })
        : new Response("missing", { status: 404 });
    } };
}

function candidate(suffix: string, previous?: { tag: string; manifest: { bundleId: string };
  manifestBytes: Buffer }, compressed = true) {
  const source = files();
  const tag = `data-vsynthetic-${suffix}`;
  const manifest = buildMetadataReleaseManifest(source, { tag,
    ...(previous ? { lastKnownGood: { tag: previous.tag, bundleId: previous.manifest.bundleId,
      manifestSha256: releaseSha256(previous.manifestBytes) } } : { fixtureGenesis: true }) },
  sourceBytes);
  const manifestBytes = encode(manifest);
  return { tag, manifest, manifestBytes, transport: transport(tag, manifestBytes, source, compressed) };
}

function activeBytes(store: string): Buffer {
  return fs.readFileSync(path.join(store, "active.json"));
}

function installedDir(store: string, bundleId: string): string {
  return path.join(store, "bundles", bundleId);
}

test("invented v2 genesis installs exact plain bytes and pins an atomic pointer", async (t) => {
  const storeDir = path.join(directory(t), "store");
  const first = candidate("genesis");
  const installed = await installMetadataReleaseBundle({ storeDir, tag: first.tag,
    transport: first.transport, fixtureOnly: true });
  assert.equal(installed.changed, true);
  assert.equal(installed.bundleDir, installedDir(storeDir, first.manifest.bundleId));
  assert.deepEqual(JSON.parse(activeBytes(storeDir).toString("utf8")), {
    format: "active-release-bundle-v1", tag: first.tag,
    bundleId: first.manifest.bundleId,
    manifestSha256: releaseSha256(first.manifestBytes),
  });
  assert.deepEqual(fs.readdirSync(installed.bundleDir).sort(), [
    RELEASE_FILES.manifest, RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    RELEASE_FILES.catalog, METADATA_RELEASE_FILE,
  ].sort());
  assert.deepEqual(first.transport.requested, [RELEASE_FILES.manifest,
    RELEASE_FILES.catalog, RELEASE_FILES.neighborhood, RELEASE_FILES.explorer,
    METADATA_RELEASE_FILE].map((name, index) => index === 0 ? name : `${name}.gz`));
  assert.equal(fs.readFileSync(path.join(installed.bundleDir, METADATA_RELEASE_FILE)).length,
    files().metadata.length);
  assert.equal((await installMetadataReleaseBundle({ storeDir, tag: first.tag,
    transport: first.transport, fixtureOnly: true })).changed, false);
  assert.equal(fs.readdirSync(storeDir).some((name) => name.startsWith(".incoming-")), false);
});

test("a v2 successor accepts an exact installed v1 data-only predecessor", async (t) => {
  const storeDir = path.join(directory(t), "store");
  const source = files();
  const firstTag = "data-vsynthetic-v1-prior";
  const firstManifest = buildReleaseManifest({ neighborhood: source.neighborhood,
    explorer: source.explorer, catalog: source.catalog },
  { tag: firstTag, fixtureGenesis: true });
  const firstManifestBytes = encode(firstManifest);
  const firstTransport = transport(firstTag, firstManifestBytes, source);
  firstTransport.bytes.delete(`${METADATA_RELEASE_FILE}.gz`);
  await installReleaseBundle({ storeDir, tag: firstTag, transport: firstTransport,
    fixtureBootstrap: true });
  const firstPointer = activeBytes(storeDir);
  const second = candidate("v2-after-v1", { tag: firstTag, manifest: firstManifest,
    manifestBytes: firstManifestBytes }, false);
  const installed = await installMetadataReleaseBundle({ storeDir, tag: second.tag,
    transport: second.transport, fixtureOnly: true });
  assert.equal(installed.changed, true);
  assert.equal(fs.existsSync(installedDir(storeDir, firstManifest.bundleId)), true);
  assert.notDeepEqual(activeBytes(storeDir), firstPointer);
  assert.equal(JSON.parse(activeBytes(storeDir).toString("utf8")).bundleId,
    second.manifest.bundleId);
  const third = candidate("v2-after-v2", second);
  assert.equal((await installMetadataReleaseBundle({ storeDir, tag: third.tag,
    transport: third.transport, fixtureOnly: true })).manifest.bundleId,
  third.manifest.bundleId);
});

test("bad inventory and bytes fail before activation without reading undeclared assets", async (t) => {
  const storeDir = path.join(directory(t), "store");
  const first = candidate("inventory-base");
  await installMetadataReleaseBundle({ storeDir, tag: first.tag,
    transport: first.transport, fixtureOnly: true });
  const before = activeBytes(storeDir);
  const second = candidate("inventory-next", first);
  second.transport.bytes.set("private-user-ratings.sqlite", Buffer.from("invented"));
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: second.tag,
    transport: second.transport, fixtureOnly: true }), /assets.*undeclared asset/);
  assert.deepEqual(second.transport.requested, [RELEASE_FILES.manifest]);
  second.transport.bytes.delete("private-user-ratings.sqlite");
  second.transport.bytes.set(METADATA_RELEASE_FILE, files().metadata);
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: second.tag,
    transport: second.transport, fixtureOnly: true }), /catalog.metadata.json.*ambiguous/);
  second.transport.bytes.delete(METADATA_RELEASE_FILE);
  second.transport.bytes.set(`${METADATA_RELEASE_FILE}.gz`, zlib.gzipSync(Buffer.from("{}")));
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: second.tag,
    transport: second.transport, fixtureOnly: true }), /catalog.metadata.json.*plain-byte length/);
  assert.deepEqual(activeBytes(storeDir), before);
  assert.equal(fs.existsSync(installedDir(storeDir, second.manifest.bundleId)), false);
  assert.equal(fs.readdirSync(storeDir).some((name) => name.startsWith(".incoming-")), false);
});

test("stale predecessor, bad active bytes, lock, and interrupted activation preserve the pointer", async (t) => {
  const storeDir = path.join(directory(t), "store");
  const first = candidate("atomic-base");
  await installMetadataReleaseBundle({ storeDir, tag: first.tag,
    transport: first.transport, fixtureOnly: true });
  const before = activeBytes(storeDir);
  const stale = candidate("stale", { tag: first.tag,
    manifest: { bundleId: "a".repeat(64) }, manifestBytes: first.manifestBytes });
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: stale.tag,
    transport: stale.transport, fixtureOnly: true }), /lastKnownGood.*currently active/);
  const next = candidate("atomic-next", first);
  const activePath = path.join(storeDir, "active.json");
  fs.writeFileSync(activePath, "{}\n");
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: next.tag,
    transport: next.transport, fixtureOnly: true }), /active.json.*format/);
  fs.writeFileSync(activePath, before);
  const lock = path.join(storeDir, ".install.lock");
  fs.writeFileSync(lock, "invented");
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: next.tag,
    transport: next.transport, fixtureOnly: true }), /lock/);
  fs.unlinkSync(lock);
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: next.tag,
    transport: next.transport, fixtureOnly: true,
    beforeActivate: () => { throw new Error("invented interrupted activation"); } }),
  /invented interrupted activation/);
  assert.deepEqual(activeBytes(storeDir), before);
  assert.equal(fs.existsSync(installedDir(storeDir, next.manifest.bundleId)), true);
  const stagedMetadataPath = path.join(installedDir(storeDir, next.manifest.bundleId),
    METADATA_RELEASE_FILE);
  const stagedMetadata = fs.readFileSync(stagedMetadataPath);
  fs.writeFileSync(stagedMetadataPath, "{}\n");
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: next.tag,
    transport: next.transport, fixtureOnly: true }), /catalog.metadata.json: root.format/);
  assert.deepEqual(activeBytes(storeDir), before);
  fs.writeFileSync(stagedMetadataPath, stagedMetadata);
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: next.tag,
    transport: next.transport, fixtureOnly: true,
    beforeActivate: () => fs.writeFileSync(stagedMetadataPath, "{}\n") }),
  /catalog.metadata.json: root.format/);
  assert.deepEqual(activeBytes(storeDir), before);
  fs.writeFileSync(stagedMetadataPath, stagedMetadata);
  assert.equal((await installMetadataReleaseBundle({ storeDir, tag: next.tag,
    transport: next.transport, fixtureOnly: true })).changed, true);
  const currentBytes = activeBytes(storeDir);
  const currentManifestPath = path.join(installedDir(storeDir, next.manifest.bundleId),
    RELEASE_FILES.manifest);
  fs.appendFileSync(currentManifestPath, " ");
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: next.tag,
    transport: next.transport, fixtureOnly: true }), /active.json.manifestSha256/);
  assert.deepEqual(activeBytes(storeDir), currentBytes);
});

test("v1 manifest and real-looking tags cannot enter the synthetic v2 route", async (t) => {
  const storeDir = path.join(directory(t), "store");
  const source = files();
  const v1 = buildReleaseManifest({ neighborhood: source.neighborhood,
    explorer: source.explorer, catalog: source.catalog },
  { tag: "data-vsynthetic-v1-candidate", fixtureGenesis: true });
  const v1Transport = transport(v1.tag, encode(v1), source);
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: v1.tag,
    transport: v1Transport, fixtureOnly: true }), /release-manifest.json.*must be release-manifest-v2/);
  const v2 = candidate("tag-check");
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir,
    tag: "data-vreal-looking", transport: v2.transport, fixtureOnly: true }),
  /tag.*data-vsynthetic-/);
  const extraPath = path.join(storeDir, "private-ratings.json");
  fs.writeFileSync(extraPath, "[]\n");
  await assert.rejects(() => installMetadataReleaseBundle({ storeDir, tag: v2.tag,
    transport: v2.transport, fixtureOnly: true }), /storeDir.*empty store/);
  fs.unlinkSync(extraPath);
  assert.equal(fs.existsSync(path.join(storeDir, "active.json")), false);
});
