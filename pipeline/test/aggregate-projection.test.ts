import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { parseCompactGraph } from "../../web/src/artifacts.js";
import type { CompactGraphDataV2 } from "../src/types.js";
import { projectAggregateGraph } from "../src/core/aggregate-projection.js";
import { buildExplorerGraph } from "../src/core/explorer-graph.js";
import { installReleaseBundle } from "../src/install-release-bundle.js";
import { buildReleaseManifest, RELEASE_FILES, releaseSha256, verifyReleaseBundle,
  writeReleaseManifest } from "../src/release-manifest.js";

const fixtureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../web/public/demo-data");
const source = (): CompactGraphDataV2 => JSON.parse(fs.readFileSync(
  path.join(fixtureDir, RELEASE_FILES.neighborhood), "utf8")) as CompactGraphDataV2;
const encoded = (value: unknown): string => `${JSON.stringify(value, null, 2)}\n`;

test("aggregate projection preserves pair evidence and source counts without user rows", () => {
  const privateGraph = source();
  const projected = projectAggregateGraph(privateGraph);
  assert.equal(projected.format, "graph-compact-v3");
  assert.notEqual(projected.graphId, privateGraph.graphId);
  assert.equal(projected.graphId, projectAggregateGraph(privateGraph).graphId);
  assert.deepEqual(projected.aa, privateGraph.aa);
  assert.deepEqual(projected.truncation, privateGraph.truncation);
  assert.deepEqual(projected.dataset, privateGraph.dataset);
  assert.deepEqual(projected.userIds, []);
  assert.deepEqual(projected.ua, []);
  assert.equal(projected.userCount, 0);
  assert.equal(projected.edgeCount, privateGraph.aa.length);
  assert.equal(projected.truncation.selectedRatings, privateGraph.ua.length);
  assert.equal(encoded(projected).includes(privateGraph.userIds[0]), false);
  const explorer = buildExplorerGraph(projected, 5, 0);
  assert.equal(explorer.format, "graph-compact-v3");
  if (explorer.format !== "graph-compact-v3") throw new Error("Expected v3 explorer");
  assert.equal(explorer.role, "visualization");
  assert.equal(explorer.sourceGraphId, projected.graphId);
  assert.deepEqual(explorer.userIds, []);
  assert.deepEqual(explorer.ua, []);
  assert.equal(explorer.aa.length, 5);
  assert.equal(explorer.visualization?.excludedUserAnimeEdges, 0);
  assert.equal(explorer.visualization?.excludedAnimeAnimeEdges, privateGraph.aa.length - 5);
  parseCompactGraph(projected, "aggregate recommendation", "recommendation");
  parseCompactGraph(explorer, "aggregate explorer", "visualization");
});

test("v3 strict parser rejects user data, hidden fields, and false counts", () => {
  const graph = projectAggregateGraph(source());
  const withField = (field: string, value: unknown) => ({ ...graph, [field]: value });
  assert.throws(() => parseCompactGraph(withField("rawRatings", [{ userId: "invented" }]), "v3"),
    /v3: root.rawRatings is unsupported/);
  assert.throws(() => parseCompactGraph({ ...graph, dataset: {
    ...graph.dataset, userId: "invented" } }, "v3"), /v3: dataset.userId is unsupported/);
  assert.throws(() => parseCompactGraph({ ...graph, userIds: ["invented"] }, "v3"),
    /v3: userIds\/ua must be empty/);
  assert.throws(() => parseCompactGraph({ ...graph, ua: [[0, 0, 1]] }, "v3"),
    /v3: userIds\/ua must be empty/);
  assert.throws(() => parseCompactGraph({ ...graph, truncation: {
    ...graph.truncation, selectedPairs: graph.truncation.selectedPairs - 1 } }, "v3"),
    /v3: truncation.selectedPairs does not reconcile/);
  assert.throws(() => parseCompactGraph({ ...graph, projection: { policy: "none" } }, "v3"),
    /v3: projection.policy must be omit-user-anime-v1/);
});

test("a data-only v3 bundle verifies and installs exact manifest, catalog, and explorer bytes", async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "invented-aggregate-bundle-"));
  try {
    const graph = projectAggregateGraph(source());
    const explorer = buildExplorerGraph(graph, 5, 0);
    fs.writeFileSync(path.join(directory, RELEASE_FILES.neighborhood), encoded(graph));
    fs.writeFileSync(path.join(directory, RELEASE_FILES.explorer), encoded(explorer));
    fs.copyFileSync(path.join(fixtureDir, RELEASE_FILES.catalog),
      path.join(directory, RELEASE_FILES.catalog));
    const manifest = writeReleaseManifest(directory, "data-vinvented-aggregate", undefined, true);
    assert.equal(manifest.neighborhood.format, "graph-compact-v3");
    assert.equal(manifest.explorer.format, "graph-compact-v3");
    assert.equal(manifest.model, null);
    assert.deepEqual(verifyReleaseBundle(directory, undefined, true), manifest);
    const assetNames = [RELEASE_FILES.manifest, RELEASE_FILES.neighborhood,
      RELEASE_FILES.explorer, RELEASE_FILES.catalog];
    const installed = await installReleaseBundle({ storeDir: path.join(directory, "store"),
      tag: manifest.tag, fixtureBootstrap: true,
      transport: { tag: manifest.tag, assets: assetNames,
        async fetchAsset(name) {
          return new Response(new Uint8Array(fs.readFileSync(path.join(directory, name))));
        } },
    });
    assert.equal(installed.manifest.bundleId, manifest.bundleId);
    const mixed = buildExplorerGraph(source(), 5, 0);
    fs.writeFileSync(path.join(directory, RELEASE_FILES.explorer), encoded(mixed));
    assert.throws(() => verifyReleaseBundle(directory, undefined, true),
      /graph-explorer.compact.json.*same compact graph format/);
  } finally {
    const resolved = fs.realpathSync(directory);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside the temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  }
});

test("a model-bearing v3 bundle preserves the exact aggregate data base and item map", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-aggregate-model-"));
  try {
    const base = path.join(root, "base");
    const promoted = path.join(root, "promoted");
    fs.mkdirSync(base);
    fs.mkdirSync(promoted);
    const graph = projectAggregateGraph(source());
    const assets = new Map([
      [RELEASE_FILES.neighborhood, encoded(graph)],
      [RELEASE_FILES.explorer, encoded(buildExplorerGraph(graph, 5, 0))],
      [RELEASE_FILES.catalog, fs.readFileSync(path.join(fixtureDir,
        RELEASE_FILES.catalog))],
    ]);
    for (const [name, bytes] of assets) {
      fs.writeFileSync(path.join(base, name), bytes);
      fs.writeFileSync(path.join(promoted, name), bytes);
    }
    const baseManifest = writeReleaseManifest(base, "data-vinvented-model-base", undefined, true);
    fs.copyFileSync(path.join(fixtureDir, RELEASE_FILES.model),
      path.join(promoted, RELEASE_FILES.model));
    const modelManifest = writeReleaseManifest(promoted, "data-vinvented-model-next", base);
    assert.equal(modelManifest.neighborhood.format, "graph-compact-v3");
    assert.equal(modelManifest.model?.coverage.mappedAnimeCount, graph.anime.length);
    assert.equal(modelManifest.model?.itemMapSha256, baseManifest.catalog.itemMapSha256);
    assert.equal(modelManifest.neighborhood.sha256, baseManifest.neighborhood.sha256);
    assert.equal(modelManifest.explorer.sha256, baseManifest.explorer.sha256);
    assert.equal(modelManifest.catalog.sha256, baseManifest.catalog.sha256);
    assert.equal(modelManifest.lastKnownGood?.bundleId, baseManifest.bundleId);
    assert.deepEqual(verifyReleaseBundle(promoted, base), modelManifest);
    const badModel = JSON.parse(fs.readFileSync(path.join(promoted, RELEASE_FILES.model), "utf8"));
    badModel.userFactors = [[1, 0]];
    assert.throws(() => buildReleaseManifest({
      neighborhood: fs.readFileSync(path.join(promoted, RELEASE_FILES.neighborhood)),
      explorer: fs.readFileSync(path.join(promoted, RELEASE_FILES.explorer)),
      catalog: fs.readFileSync(path.join(promoted, RELEASE_FILES.catalog)),
      model: Buffer.from(encoded(badModel)),
    }, { tag: "data-vinvented-hidden-factor",
      lastKnownGood: { tag: baseManifest.tag, bundleId: baseManifest.bundleId,
        manifestSha256: releaseSha256(fs.readFileSync(path.join(base, RELEASE_FILES.manifest))) } }),
    /model-mf-web.compact.json: root.userFactors is unsupported/);
  } finally {
    const resolved = fs.realpathSync(root);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside the temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  }
});
