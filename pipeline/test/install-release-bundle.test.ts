import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";
import { createGitHubReleaseTransport } from "../src/github-release-transport.js";
import { installReleaseBundle, RELEASE_DOWNLOAD_LIMITS,
  readReleaseResponse,
  type ReleaseAssetTransport } from "../src/install-release-bundle.js";
import { buildReleaseManifest, RELEASE_FILES, releaseSha256,
  type ReleaseFileBytes } from "../src/release-manifest.js";
import type { ReleaseManifestV1 } from "../../web/src/artifacts.js";

const fixtureDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../web/public/demo-data");
const encoded = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const fixture = (name: string): Buffer => fs.readFileSync(path.join(fixtureDir, name));
const sourceFiles = (): ReleaseFileBytes => ({
  neighborhood: fixture(RELEASE_FILES.neighborhood),
  explorer: fixture(RELEASE_FILES.explorer),
  catalog: fixture(RELEASE_FILES.catalog),
  model: fixture(RELEASE_FILES.model),
});

function directory(t: TestContext): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-bundle-install-"));
  t.after(() => {
    const resolved = fs.realpathSync(root);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  });
  return root;
}

function transport(tag: string, manifestBytes: Buffer, files: ReleaseFileBytes,
  compressed = true): ReleaseAssetTransport & { bytes: Map<string, Buffer> } {
  const bytes = new Map<string, Buffer>([[RELEASE_FILES.manifest, manifestBytes]]);
  for (const [name, value] of [
    [RELEASE_FILES.neighborhood, files.neighborhood],
    [RELEASE_FILES.explorer, files.explorer],
    [RELEASE_FILES.catalog, files.catalog],
    ...(files.model ? [[RELEASE_FILES.model, files.model]] : []),
  ] as [string, Buffer][]) {
    bytes.set(compressed ? `${name}.gz` : name, compressed ? zlib.gzipSync(value) : value);
  }
  return {
    tag, bytes, get assets() { return [...bytes.keys()]; },
    async fetchAsset(name) {
      const value = bytes.get(name);
      return value ? new Response(new Uint8Array(value), { status: 200 })
        : new Response("missing", { status: 404 });
    },
  };
}

function genesis() {
  const files = sourceFiles();
  const tag = "data-vinvented-genesis";
  const manifest = buildReleaseManifest(files, { tag, fixtureGenesis: true });
  const manifestBytes = encoded(manifest);
  return { files, tag, manifest, manifestBytes, transport: transport(tag, manifestBytes, files) };
}

function successor(previous: { tag: string; manifest: ReleaseManifestV1; manifestBytes: Buffer },
  suffix: string, changeBias = 0.1) {
  const files = sourceFiles();
  const model = JSON.parse(files.model!.toString("utf8"));
  model.biases[0] += changeBias;
  files.model = encoded(model);
  const tag = `data-vinvented-${suffix}`;
  const manifest = buildReleaseManifest(files, { tag, lastKnownGood: {
    tag: previous.tag, bundleId: previous.manifest.bundleId,
    manifestSha256: releaseSha256(previous.manifestBytes),
  } });
  const manifestBytes = encoded(manifest);
  return { files, tag, manifest, manifestBytes, transport: transport(tag, manifestBytes, files) };
}

async function install(storeDir: string, bundle: ReturnType<typeof genesis>,
  beforeActivate?: () => void) {
  return installReleaseBundle({ storeDir, tag: bundle.tag, transport: bundle.transport,
    fixtureBootstrap: true, beforeActivate });
}

test("complete gzip or plain bundles install through one pointer and preserve the prior", async (t) => {
  const storeDir = directory(t);
  const first = genesis();
  const initial = await install(storeDir, first);
  assert.equal(initial.changed, true);
  assert.equal(initial.manifest.bundleId, first.manifest.bundleId);
  const firstDir = initial.bundleDir;
  const pointerPath = path.join(storeDir, "active.json");
  assert.equal(JSON.parse(fs.readFileSync(pointerPath, "utf8")).bundleId, first.manifest.bundleId);
  assert.equal((await install(storeDir, first)).changed, false);

  const second = successor(first, "second");
  second.transport = transport(second.tag, second.manifestBytes, second.files, false);
  const installed = await install(storeDir, second);
  assert.equal(installed.changed, true);
  assert.equal(JSON.parse(fs.readFileSync(pointerPath, "utf8")).bundleId, second.manifest.bundleId);
  assert.equal(fs.existsSync(path.join(firstDir, RELEASE_FILES.manifest)), true);
  assert.equal(fs.existsSync(path.join(installed.bundleDir, RELEASE_FILES.model)), true);
  assert.deepEqual(fs.readdirSync(storeDir).filter((name) => name.startsWith(".incoming-")), []);
});

test("failed complete activation leaves the old pointer and can reuse the verified directory", async (t) => {
  const storeDir = directory(t);
  const first = genesis();
  await install(storeDir, first);
  const second = successor(first, "second");
  const before = fs.readFileSync(path.join(storeDir, "active.json"));
  await assert.rejects(() => install(storeDir, second, () => { throw new Error("invented crash"); }),
    /invented crash/);
  assert.deepEqual(fs.readFileSync(path.join(storeDir, "active.json")), before);
  assert.equal(fs.existsSync(path.join(storeDir, "bundles", second.manifest.bundleId)), true);
  assert.equal((await install(storeDir, second)).changed, true);
  assert.equal(JSON.parse(fs.readFileSync(path.join(storeDir, "active.json"), "utf8")).bundleId,
    second.manifest.bundleId);
});

test("partial, corrupt, ambiguous, stale, and oversized downloads never activate", async (t) => {
  const storeDir = directory(t);
  const first = genesis();
  await install(storeDir, first);
  const pointerPath = path.join(storeDir, "active.json");
  const before = fs.readFileSync(pointerPath);
  const cases: [string, (candidate: ReturnType<typeof successor>) => void, RegExp][] = [
    ["missing graph", (v) => { v.transport.bytes.delete(`${RELEASE_FILES.neighborhood}.gz`); },
      /graph.compact.json.*missing release asset/],
    ["corrupt graph", (v) => { v.transport.bytes.set(`${RELEASE_FILES.neighborhood}.gz`,
      zlib.gzipSync(Buffer.from("{}"))); }, /graph.compact.json.*length or SHA-256/],
    ["partial gzip", (v) => { const key = `${RELEASE_FILES.catalog}.gz`;
      v.transport.bytes.set(key, v.transport.bytes.get(key)!.subarray(0, 6)); },
      /catalog.identity.json.gz.*invalid gzip/],
    ["ambiguous", (v) => { v.transport.bytes.set(RELEASE_FILES.catalog, v.files.catalog); },
      /catalog.identity.json.*ambiguous/],
    ["undeclared", (v) => { v.transport.bytes.set("anonymized-ratings.compact.json.gz", Buffer.from("x")); },
      /assets.*undeclared asset/],
    ["gzip bomb", (v) => { v.transport.bytes.set(`${RELEASE_FILES.catalog}.gz`,
      zlib.gzipSync(Buffer.alloc(16_384, 65))); },
      /catalog.identity.json.gz.*decompressed size/],
    ["wrong tag", (v) => { v.transport.tag = "data-vdifferent"; }, /tag.*matching/],
  ];
  for (const [name, mutate, message] of cases) {
    const candidate = successor(first, name.replace(/\s/g, "-"), 0.2);
    mutate(candidate);
    await assert.rejects(() => install(storeDir, candidate), message, name);
    assert.deepEqual(fs.readFileSync(pointerPath), before, name);
    assert.deepEqual(fs.readdirSync(storeDir).filter((entry) => entry.startsWith(".incoming-")), [], name);
  }
  const oversized = successor(first, "oversized");
  oversized.transport.bytes.set(RELEASE_FILES.manifest,
    Buffer.alloc(RELEASE_DOWNLOAD_LIMITS.manifestBytes + 1, 32));
  await assert.rejects(() => install(storeDir, oversized), /release-manifest.json.*size limit/);
  const declaredHuge = successor(first, "declared-huge");
  const hugeManifest = { ...declaredHuge.manifest,
    catalog: { ...declaredHuge.manifest.catalog,
      bytes: RELEASE_DOWNLOAD_LIMITS.plainAssetBytes + 1 } };
  declaredHuge.transport.bytes.set(RELEASE_FILES.manifest, encoded(hugeManifest));
  await assert.rejects(() => install(storeDir, declaredHuge), /catalog.identity.json.*declared plain-byte size/);
  const compressedHuge = successor(first, "compressed-huge");
  const originalFetch = compressedHuge.transport.fetchAsset.bind(compressedHuge.transport);
  compressedHuge.transport.fetchAsset = async (name) => name === `${RELEASE_FILES.catalog}.gz`
    ? new Response("small", { status: 200, headers: {
      "Content-Length": String(RELEASE_DOWNLOAD_LIMITS.compressedAssetBytes + 1),
    } }) : originalFetch(name);
  await assert.rejects(() => install(storeDir, compressedHuge),
    /catalog.identity.json.gz.*advertised size exceeds limit/);
  assert.deepEqual(fs.readFileSync(pointerPath), before);

  const retried = successor(first, "retried");
  const key = `${RELEASE_FILES.neighborhood}.gz`;
  const correct = retried.transport.bytes.get(key)!;
  retried.transport.bytes.set(key, correct.subarray(0, 4));
  await assert.rejects(() => install(storeDir, retried), /graph.compact.json.gz.*invalid gzip/);
  assert.deepEqual(fs.readFileSync(pointerPath), before);
  retried.transport.bytes.set(key, correct);
  assert.equal((await install(storeDir, retried)).changed, true);
});

test("prior linkage, corrupted active state, and an installation lock fail closed", async (t) => {
  const storeDir = directory(t);
  const first = genesis();
  await install(storeDir, first);
  const pointerPath = path.join(storeDir, "active.json");
  const before = fs.readFileSync(pointerPath);
  const stale = successor(first, "stale");
  const staleManifest = { ...stale.manifest,
    lastKnownGood: { ...stale.manifest.lastKnownGood!, bundleId: "a".repeat(64) } };
  stale.transport.bytes.set(RELEASE_FILES.manifest, encoded(staleManifest));
  await assert.rejects(() => install(storeDir, stale), /lastKnownGood.*currently active/);
  assert.deepEqual(fs.readFileSync(pointerPath), before);
  const lockPath = path.join(storeDir, ".install.lock");
  fs.writeFileSync(lockPath, "invented lock");
  await assert.rejects(() => install(storeDir, successor(first, "locked")), /lock.*stale lock/);
  fs.unlinkSync(lockPath);
  const pointer = JSON.parse(before.toString("utf8"));
  pointer.manifestSha256 = "b".repeat(64);
  fs.writeFileSync(pointerPath, encoded(pointer));
  await assert.rejects(() => install(storeDir, successor(first, "bad-active")),
    /active.json.manifestSha256/);
  assert.equal(JSON.parse(fs.readFileSync(pointerPath, "utf8")).manifestSha256,
    pointer.manifestSha256);
});

test("only explicit fixture bootstrap can create the first active bundle", async (t) => {
  const storeDir = directory(t);
  const first = genesis();
  await assert.rejects(() => installReleaseBundle({ storeDir, tag: first.tag,
    transport: first.transport }), /active.json.*verified prior bundle/);
  assert.equal(fs.existsSync(path.join(storeDir, "active.json")), false);
  const dataOnlyFiles = sourceFiles();
  delete dataOnlyFiles.model;
  const manifest = buildReleaseManifest(dataOnlyFiles,
    { tag: "data-vinvented-data-only", fixtureGenesis: true });
  const unexpectedModel = transport(manifest.tag, encoded(manifest), dataOnlyFiles);
  unexpectedModel.bytes.set(`${RELEASE_FILES.model}.gz`, zlib.gzipSync(first.files.model!));
  await assert.rejects(() => installReleaseBundle({ storeDir, tag: manifest.tag,
    transport: unexpectedModel, fixtureBootstrap: true }), /assets.*undeclared asset/);
  assert.equal(fs.existsSync(path.join(storeDir, "active.json")), false);
});

test("an explicit GitHub release transport works with mocked HTTPS responses only", async (t) => {
  const storeDir = directory(t);
  const first = genesis();
  const requested: string[] = [];
  const assetAuth: Array<string | undefined> = [];
  const mockedFetch: typeof fetch = async (input, init) => {
    const url = String(input);
    requested.push(url);
    if (url.startsWith("https://api.github.com/")) {
      return new Response(JSON.stringify({ tag_name: first.tag, draft: false,
        prerelease: false, assets: [...first.transport.bytes.keys()].map((name) => ({ name })) }),
      { status: 200 });
    }
    assetAuth.push((init?.headers as Record<string, string> | undefined)?.Authorization);
    const name = decodeURIComponent(new URL(url).pathname.split("/").at(-1)!);
    const bytes = first.transport.bytes.get(name);
    return bytes ? new Response(new Uint8Array(bytes), { status: 200 })
      : new Response("missing", { status: 404 });
  };
  const transport = await createGitHubReleaseTransport({ owner: "InventedOwner",
    repo: "InventedRepo", tag: first.tag, token: "invented-token", fetcher: mockedFetch });
  const installed = await installReleaseBundle({ storeDir, tag: first.tag, transport,
    fixtureBootstrap: true });
  assert.equal(installed.manifest.bundleId, first.manifest.bundleId);
  assert.equal(requested[0],
    `https://api.github.com/repos/InventedOwner/InventedRepo/releases/tags/${first.tag}`);
  assert.ok(requested.slice(1).every((url) =>
    url.startsWith(`https://github.com/InventedOwner/InventedRepo/releases/download/${first.tag}/`)));
  assert.equal(requested.some((url) => url.includes("anonymized-ratings")), false);
  assert.ok(assetAuth.every((value) => value === undefined));
  assert.throws(() => transport.fetchAsset("unlisted.sqlite"), /unlisted or unsupported filename/);
  await assert.rejects(() => createGitHubReleaseTransport({ owner: "InventedOwner",
    repo: "InventedRepo", tag: "data-latest", fetcher: mockedFetch }), /tag.*versioned/);
  assert.equal(requested.filter((url) => url.startsWith("https://api.github.com/")).length, 1);
});

test("streamed download counts actual bytes even without a useful length header", async () => {
  const response = new Response(Buffer.from("five!"), { status: 200 });
  await assert.rejects(() => readReleaseResponse(response, 4, "invented asset"),
    /invented asset.*download exceeds size limit/);
});
