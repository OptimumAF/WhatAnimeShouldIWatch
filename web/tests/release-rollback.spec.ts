import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import { projectAggregateGraph } from "../../pipeline/src/core/aggregate-projection";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph";
import { installReleaseBundle, restorePreviousRelease } from "../../pipeline/src/install-release-bundle";
import { buildReleaseManifest, RELEASE_FILES, releaseSha256 } from "../../pipeline/src/release-manifest";

const normalAppUrl = "http://127.0.0.1:5174/";
const fixture = (name: string): Buffer =>
  readFileSync(new URL(`../public/demo-data/${name}`, import.meta.url));
const encoded = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);

function stagingStore() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "invented-browser-rollback-"));
  const store = path.join(root, "store");
  const source = JSON.parse(fixture(RELEASE_FILES.neighborhood).toString("utf8"));
  const graph = projectAggregateGraph(source);
  const files = {
    neighborhood: encoded(graph), explorer: encoded(buildExplorerGraph(graph, 5, 0)),
    catalog: fixture(RELEASE_FILES.catalog), model: fixture(RELEASE_FILES.model),
  };
  const firstFiles = { neighborhood: files.neighborhood, explorer: files.explorer,
    catalog: files.catalog };
  const first = buildReleaseManifest(firstFiles,
    { tag: "data-vinvented-browser-prior", fixtureGenesis: true });
  const firstBytes = encoded(first);
  const second = buildReleaseManifest(files, { tag: "data-vinvented-browser-candidate",
    lastKnownGood: { tag: first.tag, bundleId: first.bundleId,
      manifestSha256: releaseSha256(firstBytes) } });
  const secondBytes = encoded(second);
  const firstAssets = new Map<string, Buffer>([
    [RELEASE_FILES.manifest, firstBytes],
    [RELEASE_FILES.neighborhood, files.neighborhood],
    [RELEASE_FILES.explorer, files.explorer],
    [RELEASE_FILES.catalog, files.catalog],
  ]);
  const secondAssets = new Map(firstAssets);
  secondAssets.set(RELEASE_FILES.manifest, secondBytes);
  secondAssets.set(RELEASE_FILES.model, files.model);
  const install = (tag: string, assets: Map<string, Buffer>) =>
    installReleaseBundle({ storeDir: store, tag, fixtureBootstrap: true,
      transport: { tag, assets: [...assets.keys()], async fetchAsset(name) {
        const bytes = assets.get(name);
        return bytes ? new Response(new Uint8Array(bytes), { status: 200 })
          : new Response("missing", { status: 404 });
      } } });
  const restore = () => restorePreviousRelease({ storeDir: store,
    expectedCurrent: { tag: second.tag, bundleId: second.bundleId,
      manifestSha256: releaseSha256(secondBytes) },
    expectedPrevious: { tag: first.tag, bundleId: first.bundleId,
      manifestSha256: releaseSha256(firstBytes) } });
  const cleanup = () => {
    const resolved = fs.realpathSync(root);
    if (!resolved.startsWith(path.resolve(os.tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    fs.rmSync(resolved, { recursive: true, force: true });
  };
  return { store, first, second, installFirst: () => install(first.tag, firstAssets),
    installSecond: () => install(second.tag, secondAssets), restore, cleanup };
}

async function routeStagingStore(page: Page, store: string): Promise<string[]> {
  const blockedExternal: string[] = [];
  const allowed = new Set(["active.json", RELEASE_FILES.manifest,
    RELEASE_FILES.neighborhood, RELEASE_FILES.explorer, RELEASE_FILES.catalog,
    RELEASE_FILES.model]);
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/data/")) {
      const relative = url.pathname.slice("/data/".length);
      const parts = relative.split("/");
      const valid = (parts.length === 1 && allowed.has(parts[0])) ||
        (parts.length === 3 && parts[0] === "bundles" && /^[a-f0-9]{64}$/.test(parts[1]) &&
          allowed.has(parts[2]));
      if (!valid) return route.fulfill({ status: 404, body: "" });
      const filepath = path.join(store, ...parts);
      return fs.existsSync(filepath)
        ? route.fulfill({ contentType: "application/json", body: fs.readFileSync(filepath),
          headers: { "Cache-Control": "no-store" } })
        : route.fulfill({ status: 404, body: "" });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" }, body: '{"data":[]}' });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
      blockedExternal.push(url.href);
      return route.abort();
    }
    return route.continue();
  });
  return blockedExternal;
}

for (const legacyVersion of [1, 4] as const) {
  test(`staged candidate and rollback preserve invented v${legacyVersion} profile migration`, async ({ page }) => {
    const staging = stagingStore();
    try {
      await staging.installFirst();
      const blockedExternal = await routeStagingStore(page, staging.store);
      await page.goto(normalAppUrl);
      await expect(page.locator("#diagnostic-data")).toContainText(staging.first.tag);
      const profileState = legacyVersion === 1 ? {
        version: 3, mode: "hybrid", selected: [
          { nodeId: "anime:101", weight: 1.7 }, { nodeId: "anime:999", weight: 2.4 }],
        includeCandidates: ["anime:998"], excludeCandidates: ["anime:997"],
      } : {
        version: 4, mode: "hybrid", selected: [
          { nodeId: "anime:101", weight: 1.7 }, { nodeId: "anime:999", weight: 1 }],
        history: [{ provider: "local", sourceId: "anime:101", title: "Copper Comet",
          animeId: 101, status: "completed", sourceStatus: "Completed", progressEpisodes: 12,
          score: 9, scoreScale: "local-10" }],
        includeCandidates: ["anime:998"], excludeCandidates: ["anime:997"],
      };
      const profileName = `Invented v${legacyVersion} profile`;
      const raw = legacyVersion === 1
        ? JSON.stringify([{ name: profileName, updatedAt: "2026-01-01T00:00:00Z",
          state: profileState }])
        : JSON.stringify({ version: 4, profiles: [{ name: profileName,
          updatedAt: "2026-01-01T00:00:00Z", state: profileState }] });
      await page.evaluate(({ version, raw }) => {
        localStorage.setItem(`wasiw.recommendationProfiles.v${version}`, raw);
      }, { version: legacyVersion, raw });
      await staging.installSecond();
      await page.reload();
      await expect(page.locator("#diagnostic-data")).toContainText(staging.second.tag);
      await expect(page.locator("#diagnostic-model")).toContainText("sha256");
      await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
      await expect(page.locator("#profile-select")).toContainText(profileName);
      await page.locator("#profile-select").selectOption(profileName);
      await page.locator("#profile-load-btn").click();
      await expect(page.locator("#selected-anime")).toContainText("anime:999");
      const migrated = await page.evaluate((version) => ({
        old: localStorage.getItem(`wasiw.recommendationProfiles.v${version}`),
        backup: localStorage.getItem(`wasiw.recommendationProfiles.v${version}.backup`),
        current: localStorage.getItem("wasiw.recommendationProfiles.v5"),
      }), legacyVersion);
      expect(migrated.old).toBe(raw);
      expect(migrated.backup).toBe(raw);
      const migratedState = JSON.parse(migrated.current ?? "null").profiles[0].state;
      expect(migratedState).toMatchObject({ version: 5, mode: "hybrid",
        includeCandidates: ["anime:998"], excludeCandidates: ["anime:997"] });
      expect(migratedState.preferences).toEqual([
        expect.objectContaining({ nodeId: "anime:101", importance: 1.7, sentiment: "liked" }),
        expect.objectContaining({ nodeId: "anime:999", importance: legacyVersion === 1 ? 2.4 : 1,
          sentiment: legacyVersion === 1 ? "liked" : "seen" }),
      ]);
      expect(migratedState.history).toHaveLength(legacyVersion === 1 ? 0 : 1);
      staging.restore();
      await page.reload();
      await expect(page.locator("#diagnostic-data")).toContainText(staging.first.tag);
      await expect(page.locator("#diagnostic-model")).toHaveText("Not included in this data release");
      await expect(page.locator("#rec-engine-status")).toContainText("Using catalog coverage baseline");
      await expect(page.locator("#selected-anime")).toContainText("anime:999");
      await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
      await page.locator("#profile-select").selectOption(profileName);
      await page.locator("#profile-load-btn").click();
      await expect(page.locator("#selected-anime")).toContainText("anime:999");
      const after = await page.evaluate((version) => ({
        old: localStorage.getItem(`wasiw.recommendationProfiles.v${version}`),
        backup: localStorage.getItem(`wasiw.recommendationProfiles.v${version}.backup`),
        current: localStorage.getItem("wasiw.recommendationProfiles.v5"),
      }), legacyVersion);
      expect(after).toEqual(migrated);
      expect(blockedExternal.every((url) => new URL(url).hostname === "fonts.googleapis.com"))
        .toBe(true);
    } finally {
      staging.cleanup();
    }
  });
}
