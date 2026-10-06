import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import { parseReleaseManifest } from "../../web/src/artifacts.js";
import {
  METADATA_RELEASE_FILE, buildMetadataReleaseManifest, verifyMetadataReleaseBundle,
  writeMetadataReleaseManifest, type MetadataReleaseFiles,
} from "../src/metadata-release-bundle.js";
import { RELEASE_FILES, releaseSha256, writeReleaseManifest } from "../src/release-manifest.js";

const fixtureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)),
  "../../web/public/demo-data");
const fixture = (name: string): Buffer => fs.readFileSync(path.join(fixtureDir, name));
const encode = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const sourceBytes = Buffer.from("invented-structured-source-only");

function files(): MetadataReleaseFiles {
  const metadata = {
    format: "anime-metadata-catalog-v1",
    source: { name: "invented-fixture", snapshotAt: "2026-09-24T00:00:00.000Z",
      snapshotSha256: releaseSha256(sourceBytes) },
    anime: [
      { animeId: 101, sourceItemId: "invented:101", title: "Copper Comet",
        aliases: ["Copper Voyage"], genres: ["Adventure"], year: 2021,
        mediaFormat: "TV", episodeCount: 12, runtimeMinutes: 24,
        contentClassification: null, communityScore: null, relations: null },
      { animeId: 102, sourceItemId: "invented:102", title: "Moonlit Workshop",
        aliases: [], genres: null, year: null, mediaFormat: null,
        episodeCount: null, runtimeMinutes: null, contentClassification: null,
        communityScore: null, relations: null },
    ],
  };
  return {
    neighborhood: fixture("graph.aggregate.compact.json"),
    explorer: fixture("graph-explorer.aggregate.compact.json"),
    catalog: fixture(RELEASE_FILES.catalog),
    metadata: encode(metadata),
  };
}

function directory(t: TestContext): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-metadata-bundle-"));
  t.after(() => {
    const resolved = fs.realpathSync(root);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  });
  return root;
}

function writeFiles(directory: string, source: MetadataReleaseFiles): void {
  fs.mkdirSync(directory);
  fs.writeFileSync(path.join(directory, RELEASE_FILES.neighborhood), source.neighborhood);
  fs.writeFileSync(path.join(directory, RELEASE_FILES.explorer), source.explorer);
  fs.writeFileSync(path.join(directory, RELEASE_FILES.catalog), source.catalog);
  fs.writeFileSync(path.join(directory, METADATA_RELEASE_FILE), source.metadata);
}

test("invented v2 metadata producer binds public bytes and private source digest", (t) => {
  const source = files();
  const root = directory(t);
  const candidate = path.join(root, "candidate");
  writeFiles(candidate, source);
  const manifest = writeMetadataReleaseManifest(candidate, "data-vsynthetic-metadata",
    sourceBytes, undefined, true);
  assert.equal(manifest.format, "release-manifest-v2");
  assert.equal(manifest.neighborhood.format, "graph-compact-v3");
  assert.equal(manifest.model, null);
  assert.equal(manifest.metadata.sha256, releaseSha256(source.metadata));
  assert.equal(manifest.metadata.sourceSnapshotSha256, releaseSha256(sourceBytes));
  assert.equal(manifest.metadata.itemMapSha256, manifest.catalog.itemMapSha256);
  assert.equal(verifyMetadataReleaseBundle(candidate, undefined, true).bundleId, manifest.bundleId);
  assert.throws(() => verifyMetadataReleaseBundle(candidate), /lastKnownGood.*fixture-only/);
  assert.throws(() => parseReleaseManifest(manifest, RELEASE_FILES.manifest),
    /root.metadata.*unsupported/);
  assert.equal(fs.readdirSync(candidate).length, 5);
  assert.equal(fs.readdirSync(candidate).includes("sourceBytes"), false);
  assert.throws(() => writeMetadataReleaseManifest(candidate, "data-vsecond",
    sourceBytes, undefined, true), /release-manifest.json.*already exists/);
});

test("source mismatch, hidden fields, unknown IDs, and user-row graph refuse v2 creation", () => {
  const source = files();
  assert.throws(() => buildMetadataReleaseManifest(source,
    { tag: "data-vreal-looking", fixtureGenesis: true }, sourceBytes),
  /tag.*data-vsynthetic-/);
  assert.throws(() => buildMetadataReleaseManifest(source,
    { tag: "data-vsynthetic-invented", fixtureGenesis: true }, Buffer.from("different invented source")),
  /source.snapshotSha256.*source bytes/);
  const hidden = files();
  const hiddenValue = JSON.parse(hidden.metadata.toString("utf8"));
  hiddenValue.anime[0].userHistory = [];
  hidden.metadata = encode(hiddenValue);
  assert.throws(() => buildMetadataReleaseManifest(hidden,
    { tag: "data-vsynthetic-invented", fixtureGenesis: true }, sourceBytes),
  /catalog.metadata.json: anime\[0\].userHistory.*unsupported/);
  const extraId = files();
  const extraValue = JSON.parse(extraId.metadata.toString("utf8"));
  extraValue.anime[1].animeId = 999;
  extraId.metadata = encode(extraValue);
  assert.throws(() => buildMetadataReleaseManifest(extraId,
    { tag: "data-vsynthetic-invented", fixtureGenesis: true }, sourceBytes),
  /catalog.metadata.json.anime\[1\].animeId.*identity catalog/);
  const userGraph = files();
  userGraph.neighborhood = fixture(RELEASE_FILES.neighborhood);
  userGraph.explorer = fixture(RELEASE_FILES.explorer);
  assert.throws(() => buildMetadataReleaseManifest(userGraph,
    { tag: "data-vsynthetic-invented", fixtureGenesis: true }, sourceBytes),
  /graph.compact.json.*aggregate graph-compact-v3/);
});

test("v2 public-byte verifier rejects tampering and undeclared files", (t) => {
  const root = directory(t);
  const candidate = path.join(root, "candidate");
  writeFiles(candidate, files());
  writeMetadataReleaseManifest(candidate, "data-vsynthetic-invented", sourceBytes, undefined, true);
  const metadataPath = path.join(candidate, METADATA_RELEASE_FILE);
  const original = fs.readFileSync(metadataPath);
  fs.writeFileSync(metadataPath, Buffer.concat([original, Buffer.from(" ")]));
  assert.throws(() => verifyMetadataReleaseBundle(candidate, undefined, true),
    /release-manifest.json.*public bytes/);
  fs.writeFileSync(metadataPath, original);
  fs.writeFileSync(path.join(candidate, "user-ratings.json"), "[]");
  assert.throws(() => verifyMetadataReleaseBundle(candidate, undefined, true),
    /directory.*undeclared files/);
});

test("v2 binds an exact verified v1 data-only predecessor", (t) => {
  const root = directory(t);
  const prior = path.join(root, "prior");
  const candidate = path.join(root, "candidate");
  const source = files();
  fs.mkdirSync(prior);
  fs.writeFileSync(path.join(prior, RELEASE_FILES.neighborhood), source.neighborhood);
  fs.writeFileSync(path.join(prior, RELEASE_FILES.explorer), source.explorer);
  fs.writeFileSync(path.join(prior, RELEASE_FILES.catalog), source.catalog);
  const priorManifest = writeReleaseManifest(prior, "data-vsynthetic-prior", undefined, true);
  writeFiles(candidate, source);
  const manifest = writeMetadataReleaseManifest(candidate, "data-vinvented-next",
    sourceBytes, prior);
  assert.equal(manifest.lastKnownGood?.bundleId, priorManifest.bundleId);
  assert.equal(verifyMetadataReleaseBundle(candidate, prior).bundleId, manifest.bundleId);
  assert.throws(() => verifyMetadataReleaseBundle(candidate), /lastKnownGood.*separate previous/);
  const priorPath = path.join(prior, RELEASE_FILES.manifest);
  fs.appendFileSync(priorPath, " ");
  assert.throws(() => verifyMetadataReleaseBundle(candidate, prior),
    /lastKnownGood.*manifest-byte hash/);
});

test("a later invented v2 bundle can name the exact previous v2 bytes", (t) => {
  const root = directory(t);
  const prior = path.join(root, "prior");
  const candidate = path.join(root, "candidate");
  writeFiles(prior, files());
  const priorManifest = writeMetadataReleaseManifest(prior, "data-vsynthetic-first",
    sourceBytes, undefined, true);
  writeFiles(candidate, files());
  const next = writeMetadataReleaseManifest(candidate, "data-vinvented-second",
    sourceBytes, prior);
  assert.equal(next.lastKnownGood?.bundleId, priorManifest.bundleId);
  assert.equal(verifyMetadataReleaseBundle(candidate, prior).bundleId, next.bundleId);
});
