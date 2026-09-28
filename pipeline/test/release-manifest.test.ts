import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import { recommendationGraphId, visualizationGraphId } from "../src/core/graph-contract.js";
import {
  buildReleaseManifest, releaseItemMapSha256, releaseSha256, verifyReleaseBundle,
  writeReleaseManifest, RELEASE_FILES, type ReleaseFileBytes,
} from "../src/release-manifest.js";

const fixtureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../web/public/demo-data");
const encoded = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const fixture = (name: string): Buffer => fs.readFileSync(path.join(fixtureDir, name));
const read = (bytes: Buffer): any => JSON.parse(bytes.toString("utf8"));

function files(): ReleaseFileBytes {
  return {
    neighborhood: fixture(RELEASE_FILES.neighborhood),
    explorer: fixture(RELEASE_FILES.explorer),
    catalog: fixture(RELEASE_FILES.catalog),
    model: fixture(RELEASE_FILES.model),
  };
}

function mutate(files: ReleaseFileBytes, key: keyof ReleaseFileBytes, change: (value: any) => void): void {
  const value = read(files[key]!);
  change(value);
  files[key] = encoded(value);
}

function directory(t: TestContext): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-release-bundle-"));
  t.after(() => {
    const resolved = fs.realpathSync(root);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  });
  return root;
}

function writeFiles(dir: string, source: ReleaseFileBytes): void {
  fs.mkdirSync(dir);
  fs.writeFileSync(path.join(dir, RELEASE_FILES.neighborhood), source.neighborhood);
  fs.writeFileSync(path.join(dir, RELEASE_FILES.explorer), source.explorer);
  fs.writeFileSync(path.join(dir, RELEASE_FILES.catalog), source.catalog);
  if (source.model) fs.writeFileSync(path.join(dir, RELEASE_FILES.model), source.model);
}

test("synthetic manifest pins exact bytes, graph IDs, item maps, and explicit genesis", (t) => {
  const source = files();
  const fixtureTag = read(fixture(RELEASE_FILES.manifest)).tag;
  const manifest = buildReleaseManifest(source,
    { tag: fixtureTag, fixtureGenesis: true });
  const fixtureManifest = read(fixture(RELEASE_FILES.manifest));
  assert.deepEqual(manifest, fixtureManifest);
  assert.equal(manifest.neighborhood.sha256, releaseSha256(source.neighborhood));
  assert.equal(manifest.neighborhood.bytes, source.neighborhood.length);
  assert.equal(manifest.catalog.itemMapSha256, releaseItemMapSha256(read(source.catalog).anime));
  assert.deepEqual(manifest.model?.coverage, { mappedAnimeCount: 8, totalCatalogAnimeCount: 8 });
  assert.equal(manifest.explorer.sourceGraphId, manifest.neighborhood.graphId);
  assert.throws(() => buildReleaseManifest(source, { tag: "data-latest" }), /tag.*versioned/);
  assert.throws(() => buildReleaseManifest(source, { tag: "data-vnext" }), /lastKnownGood.*required/);

  const root = directory(t);
  const genesis = path.join(root, "genesis");
  writeFiles(genesis, source);
  writeReleaseManifest(genesis, fixtureTag, undefined, true);
  assert.equal(verifyReleaseBundle(genesis, undefined, true).bundleId, manifest.bundleId);
  assert.throws(() => verifyReleaseBundle(genesis), /lastKnownGood.*genesis/);
  assert.throws(() => writeReleaseManifest(genesis, fixtureTag, undefined, true),
    /release-manifest.json.*already exists/);
});

test("a distinct intact previous bundle is required and pinned by exact manifest bytes", (t) => {
  const root = directory(t);
  const previous = path.join(root, "previous");
  const current = path.join(root, "current");
  writeFiles(previous, files());
  const previousManifest = writeReleaseManifest(previous, "data-vprevious", undefined, true);
  const next = files();
  mutate(next, "model", (model) => { model.biases[0] += 0.1; });
  writeFiles(current, next);
  const currentManifest = writeReleaseManifest(current, "data-vcurrent", previous);
  assert.equal(currentManifest.lastKnownGood?.tag, previousManifest.tag);
  assert.equal(currentManifest.lastKnownGood?.bundleId, previousManifest.bundleId);
  assert.equal(currentManifest.lastKnownGood?.manifestSha256,
    releaseSha256(fs.readFileSync(path.join(previous, RELEASE_FILES.manifest))));
  assert.equal(verifyReleaseBundle(current, previous).bundleId, currentManifest.bundleId);
  assert.throws(() => verifyReleaseBundle(current), /lastKnownGood.*required/);
  assert.throws(() => verifyReleaseBundle(current, current), /lastKnownGood.*current directory/);
  assert.throws(() => verifyReleaseBundle(current, path.join(root, "missing")), /ENOENT|missing/);
  assert.throws(() => writeReleaseManifest(current, "data-vother", previous), /already exists/);

  const previousManifestPath = path.join(previous, RELEASE_FILES.manifest);
  const originalPreviousBytes = fs.readFileSync(previousManifestPath);
  fs.writeFileSync(previousManifestPath, Buffer.concat([originalPreviousBytes, Buffer.from(" ")]));
  assert.throws(() => verifyReleaseBundle(current, previous), /lastKnownGood.*manifest-byte hash/);
  fs.writeFileSync(previousManifestPath, originalPreviousBytes);
  fs.writeFileSync(path.join(previous, RELEASE_FILES.catalog), Buffer.from("{}"));
  assert.throws(() => verifyReleaseBundle(current, previous), /catalog.identity.json/);
});

test("manifest rejects changed bytes, a missing required model, and an undeclared model", (t) => {
  const root = directory(t);
  const current = path.join(root, "current");
  writeFiles(current, files());
  writeReleaseManifest(current, "data-vcurrent", undefined, true);
  const graphPath = path.join(current, RELEASE_FILES.neighborhood);
  const graphBytes = fs.readFileSync(graphPath);
  fs.writeFileSync(graphPath, Buffer.concat([graphBytes, Buffer.from(" ")]));
  assert.throws(() => verifyReleaseBundle(current, undefined, true), /release-manifest.json.*hashes/);
  fs.writeFileSync(graphPath, graphBytes);
  const modelPath = path.join(current, RELEASE_FILES.model);
  const modelBytes = fs.readFileSync(modelPath);
  fs.rmSync(modelPath);
  assert.throws(() => verifyReleaseBundle(current, undefined, true), /model-mf-web.compact.json.*presence/);
  const dataOnly = path.join(root, "data-only");
  const noModel = files();
  delete noModel.model;
  writeFiles(dataOnly, noModel);
  writeReleaseManifest(dataOnly, "data-vdata-only", undefined, true);
  assert.equal(verifyReleaseBundle(dataOnly, undefined, true).model, null);
  fs.writeFileSync(path.join(dataOnly, RELEASE_FILES.model), modelBytes);
  assert.throws(() => verifyReleaseBundle(dataOnly, undefined, true), /model-mf-web.compact.json.*presence/);
  const invalidUtf8 = files();
  invalidUtf8.catalog = Buffer.from([0x7b, 0x22, 0x78, 0x22, 0x3a, 0x22, 0xff, 0x22, 0x7d]);
  assert.throws(() => buildReleaseManifest(invalidUtf8,
    { tag: "data-vinvalid", fixtureGenesis: true }), /catalog.identity.json.*invalid JSON/);
});

test("cross-version, dataset, catalog, model, and explorer drift all fail before manifest creation", () => {
  const cases: [string, keyof ReleaseFileBytes, (value: any) => void, RegExp][] = [
    ["old graph", "neighborhood", (v) => { v.format = "graph-compact-v1"; }, /graph.compact.json.*requires graph-compact-v2|graph.compact.json.*role/],
    ["stale graph ID", "neighborhood", (v) => { v.anime[0][1] = "Wrong title"; }, /graph.compact.json.graphId.*content/],
    ["dataset", "catalog", (v) => { v.datasetSha256 = "a".repeat(64); }, /catalog.identity.json.datasetSha256.*dataset/],
    ["catalog title", "catalog", (v) => { v.anime[0][1] = "Wrong title"; }, /catalog.identity.json.anime.*match/],
    ["model dataset", "model", (v) => { v.datasetSha256 = "a".repeat(64); }, /model-mf-web.compact.json.datasetSha256.*dataset/],
    ["model title", "model", (v) => { v.titles[0] = "Wrong title"; }, /model-mf-web.compact.json.titles\[0\].*catalog/],
    ["explorer link", "explorer", (v) => { v.sourceGraphId = "b".repeat(64); }, /graph-explorer.compact.json.graphId.*content|graph-explorer.compact.json.sourceGraphId.*recommendation/],
    ["explorer semantics", "explorer", (v) => { v.semantics.recommendationUse = "other"; }, /semantics.recommendationUse.*unsupported/],
  ];
  for (const [name, key, change, message] of cases) {
    const source = files();
    mutate(source, key, change);
    assert.throws(() => buildReleaseManifest(source,
      { tag: "data-vcandidate", fixtureGenesis: true }), message, name);
  }
  const subset = files();
  mutate(subset, "model", (model) => {
    for (const key of ["animeIds", "titles", "biases", "embeddings"]) model[key] = model[key].slice(0, 2);
  });
  assert.deepEqual(buildReleaseManifest(subset,
    { tag: "data-vpartial", fixtureGenesis: true }).model?.coverage,
  { mappedAnimeCount: 2, totalCatalogAnimeCount: 8 });
  const missingDataset = files();
  mutate(missingDataset, "model", (model) => { delete model.datasetSha256; });
  assert.throws(() => buildReleaseManifest(missingDataset,
    { tag: "data-vcandidate", fixtureGenesis: true }), /model-mf-web.compact.json.datasetSha256/);

  const newGraphWithOldExplorer = files();
  mutate(newGraphWithOldExplorer, "neighborhood", (graph) => {
    graph.dataset.sha256 = "c".repeat(64);
    const { graphId: _old, ...content } = graph;
    graph.graphId = recommendationGraphId(content);
  });
  assert.throws(() => buildReleaseManifest(newGraphWithOldExplorer,
    { tag: "data-vcandidate", fixtureGenesis: true }), /graph-explorer.compact.json.sourceGraphId/);

  const forgedExplorer = files();
  mutate(forgedExplorer, "explorer", (graph) => {
    graph.sourceGraphId = "b".repeat(64);
    const { graphId: _old, ...content } = graph;
    graph.graphId = visualizationGraphId(content);
  });
  assert.throws(() => buildReleaseManifest(forgedExplorer,
    { tag: "data-vcandidate", fixtureGenesis: true }), /graph-explorer.compact.json.sourceGraphId/);
});
