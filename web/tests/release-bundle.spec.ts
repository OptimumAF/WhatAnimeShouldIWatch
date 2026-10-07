import crypto from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { projectAggregateGraph } from "../../pipeline/src/core/aggregate-projection";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph";
import { installMetadataReleaseBundle } from "../../pipeline/src/install-metadata-release-bundle";
import { buildMetadataReleaseManifest } from "../../pipeline/src/metadata-release-bundle";
import { buildReleaseManifest, releaseSha256 } from "../../pipeline/src/release-manifest";
import { mapWikibaseMetadata, type WikibaseMappingPolicy } from "../../pipeline/src/wikibase-metadata";

const normalAppUrl = "http://127.0.0.1:5174/";
const appVersion = JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8")).version as string;
const sourceRevision = /^[a-f0-9]{40}$/.test(process.env.GITHUB_SHA ?? "")
  ? process.env.GITHUB_SHA!.slice(0, 12) : "local build";
const fixture = (name: string): Buffer =>
  readFileSync(new URL(`../public/demo-data/${name}`, import.meta.url));
const manifestBytes = fixture("release-manifest.json");
const manifest = JSON.parse(manifestBytes.toString("utf8"));
const pointer = {
  format: "active-release-bundle-v1", tag: manifest.tag, bundleId: manifest.bundleId,
  manifestSha256: crypto.createHash("sha256").update(manifestBytes).digest("hex"),
};

async function routeBundle(page: Page, options: { corruptGraph?: boolean; corruptPointer?: boolean;
  corruptExplorer?: boolean; corruptManifest?: boolean; withoutModel?: boolean } = {}) {
  const { bundleId: _originalId, ...originalPayload } = manifest;
  const dataOnlyPayload = { ...originalPayload, model: null };
  const dataOnlyManifest = { ...dataOnlyPayload,
    bundleId: crypto.createHash("sha256").update(JSON.stringify(dataOnlyPayload)).digest("hex") };
  const deliveredManifest = options.withoutModel ? dataOnlyManifest : manifest;
  const deliveredManifestBytes = options.withoutModel
    ? Buffer.from(`${JSON.stringify(dataOnlyManifest, null, 2)}\n`) : manifestBytes;
  const deliveredPointer = options.withoutModel ? {
    format: "active-release-bundle-v1", tag: deliveredManifest.tag,
    bundleId: deliveredManifest.bundleId,
    manifestSha256: crypto.createHash("sha256").update(deliveredManifestBytes).digest("hex"),
  } : pointer;
  const requests: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/data/")) {
      requests.push(url.pathname);
      if (url.pathname === "/data/active.json") {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify(options.corruptPointer
          ? { ...deliveredPointer, manifestSha256: "invalid" } : deliveredPointer) });
      }
      const prefix = `/data/bundles/${deliveredManifest.bundleId}/`;
      if (url.pathname.startsWith(prefix)) {
        const name = url.pathname.slice(prefix.length);
        if (name === "release-manifest.json") {
          return route.fulfill({ contentType: "application/json",
            body: options.corruptManifest ? Buffer.from("{}") : deliveredManifestBytes });
        }
        if (["graph.compact.json", "graph-explorer.compact.json", "catalog.identity.json",
          "model-mf-web.compact.json"].includes(name)) {
          if (options.withoutModel && name === "model-mf-web.compact.json") {
            return route.fulfill({ status: 404, body: "" });
          }
          const body = options.corruptGraph && name === "graph.compact.json"
            ? Buffer.from("{}")
            : options.corruptExplorer && name === "graph-explorer.compact.json"
              ? Buffer.from("{}") : fixture(name);
          return route.fulfill({ contentType: "application/json", body });
        }
      }
      if (options.withoutModel && url.pathname === "/data/model-mf-web.compact.json.gz") {
        return route.fulfill({ contentType: "application/json", body: fixture("model-mf-web.compact.json") });
      }
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" }, body: '{"data":[]}' });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  return requests;
}

async function routeAggregateBundle(page: Page, options: { withModel?: boolean } = {}) {
  const source = JSON.parse(fixture("graph.compact.json").toString("utf8"));
  const graph = projectAggregateGraph(source);
  const explorer = buildExplorerGraph(graph, 5, 0);
  const graphBytes = Buffer.from(`${JSON.stringify(graph, null, 2)}\n`);
  const explorerBytes = Buffer.from(`${JSON.stringify(explorer, null, 2)}\n`);
  const catalogBytes = fixture("catalog.identity.json");
  const base = buildReleaseManifest({ neighborhood: graphBytes, explorer: explorerBytes,
    catalog: catalogBytes }, { tag: "data-vinvented-aggregate-browser", fixtureGenesis: true });
  const baseBytes = Buffer.from(`${JSON.stringify(base, null, 2)}\n`);
  const modelBytes = fixture("model-mf-web.compact.json");
  const release = options.withModel
    ? buildReleaseManifest({ neighborhood: graphBytes, explorer: explorerBytes,
      catalog: catalogBytes, model: modelBytes }, {
      tag: "data-vinvented-aggregate-model-browser",
      lastKnownGood: { tag: base.tag, bundleId: base.bundleId,
        manifestSha256: crypto.createHash("sha256").update(baseBytes).digest("hex") },
    }) : base;
  const releaseBytes = Buffer.from(`${JSON.stringify(release, null, 2)}\n`);
  const active = { format: "active-release-bundle-v1", tag: release.tag,
    bundleId: release.bundleId,
    manifestSha256: crypto.createHash("sha256").update(releaseBytes).digest("hex") };
  const files = new Map<string, Buffer>([
    ["release-manifest.json", releaseBytes],
    ["graph.compact.json", graphBytes],
    ["graph-explorer.compact.json", explorerBytes],
    ["catalog.identity.json", catalogBytes],
    ...(options.withModel ? [["model-mf-web.compact.json", modelBytes] as [string, Buffer]] : []),
  ]);
  const requests: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/data/")) {
      requests.push(url.pathname);
      if (url.pathname === "/data/active.json") {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify(active) });
      }
      const prefix = `/data/bundles/${release.bundleId}/`;
      const bytes = url.pathname.startsWith(prefix) ? files.get(url.pathname.slice(prefix.length)) : null;
      return bytes ? route.fulfill({ contentType: "application/json", body: bytes })
        : route.fulfill({ status: 404, body: "" });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" }, body: '{"data":[]}' });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  return { requests, graphBytes, release };
}

async function routeMetadataBundle(page: Page, options: {
  corruptBytes?: boolean; malformedYear?: boolean; unknownId?: boolean;
  classificationMarkup?: boolean; catalogGraphMismatch?: boolean; mappedSource?: boolean;
  missingYear?: boolean; missingGenre?: boolean;
} = {}) {
  const authoredMetadata = {
    format: "anime-metadata-catalog-v1",
    source: { name: "invented-fixture", snapshotAt: "2026-09-24T00:00:00.000Z",
      snapshotSha256: "b".repeat(64) },
    anime: [
      { animeId: 101, sourceItemId: "invented:101", title: "Copper Comet",
        aliases: ["Copper Voyage"], genres: ["Adventure"], year: 2021,
        mediaFormat: "TV", episodeCount: 12, runtimeMinutes: 24,
        contentClassification: null, communityScore: null, relations: null },
      { animeId: options.unknownId ? 999 : 102, sourceItemId: "invented:102",
        title: "Moonlit Workshop", aliases: [], genres: options.missingGenre ? null : ["Adventure"],
        year: options.malformedYear ? "unknown" : options.missingYear ? null : 2022, mediaFormat: "Movie",
        episodeCount: 1, runtimeMinutes: 95,
        contentClassification: { jurisdiction: "Fixtureland", system: "Invented board",
          value: options.classificationMarkup ? "<img src=x onerror=alert(1)>" : "All" },
        communityScore: 8.1, relations: null },
    ],
  };
  if (options.missingGenre) authoredMetadata.anime.push({ animeId: 105, sourceItemId: "invented:105",
    title: "星の航路", aliases: [], genres: ["Adventure"], year: 2022,
    mediaFormat: "TV", episodeCount: 12, runtimeMinutes: 24,
    contentClassification: null, communityScore: null, relations: null });
  const metadata = options.mappedSource ? (() => {
    const raw = JSON.parse(readFileSync(new URL("../../fixtures/synthetic-wikibase-entities.json", import.meta.url), "utf8"));
    raw.entities.Q910000101.labels.en.value = "Invented Copper Sky";
    return mapWikibaseMetadata(Buffer.from(JSON.stringify(raw)), [101, 102, 103, 104],
      JSON.parse(readFileSync(new URL("../../fixtures/synthetic-wikibase-policy.json", import.meta.url), "utf8")) as WikibaseMappingPolicy,
      { name: "invented-wikibase-fixture", snapshotAt: "2026-10-06T00:00:00.000Z" }).snapshot!;
  })() : authoredMetadata;
  const metadataBytes = Buffer.from(`${JSON.stringify(metadata, null, 2)}\n`);
  const identity = JSON.parse(fixture("catalog.identity.json").toString("utf8"));
  if (options.catalogGraphMismatch) identity.anime[1][1] = "Invented title mismatch";
  const identityBytes = Buffer.from(`${JSON.stringify(identity, null, 2)}\n`);
  const itemMapSha256 = crypto.createHash("sha256")
    .update(JSON.stringify(identity.anime)).digest("hex");
  const payload = { ...manifest, format: "release-manifest-v2", model: null,
    catalog: { ...manifest.catalog,
      sha256: crypto.createHash("sha256").update(identityBytes).digest("hex"),
      bytes: identityBytes.length, itemMapSha256 },
    metadata: { path: "catalog.metadata.json", format: "anime-metadata-catalog-v1",
      sha256: crypto.createHash("sha256").update(metadataBytes).digest("hex"),
      bytes: metadataBytes.length, animeCount: metadata.anime.length,
      itemMapSha256,
      sourceSnapshotSha256: metadata.source.snapshotSha256 } };
  const { bundleId: _oldId, ...withoutId } = payload;
  const release = { ...withoutId,
    bundleId: crypto.createHash("sha256").update(JSON.stringify(withoutId)).digest("hex") };
  const releaseBytes = Buffer.from(`${JSON.stringify(release, null, 2)}\n`);
  const active = { format: "active-release-bundle-v1", tag: release.tag,
    bundleId: release.bundleId,
    manifestSha256: crypto.createHash("sha256").update(releaseBytes).digest("hex") };
  const files = new Map<string, Buffer>([
    ["release-manifest.json", releaseBytes],
    ["graph.compact.json", fixture("graph.compact.json")],
    ["catalog.identity.json", identityBytes],
    ["catalog.metadata.json", options.corruptBytes ? Buffer.from("{}") : metadataBytes],
  ]);
  const requests: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/data/")) {
      requests.push(url.pathname);
      if (url.pathname === "/data/active.json") {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify(active) });
      }
      const prefix = `/data/bundles/${release.bundleId}/`;
      const bytes = url.pathname.startsWith(prefix) ? files.get(url.pathname.slice(prefix.length)) : null;
      return bytes ? route.fulfill({ contentType: "application/json", body: bytes })
        : route.fulfill({ status: 404, body: "" });
    }
    if (url.hostname === "api.jikan.moe") {
      requests.push(`jikan:${url.pathname}`);
      return route.abort();
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  return { requests, release };
}

test("normal mode pins one verified bundle for graph, model, and explorer", async ({ page }) => {
  const requests = await routeBundle(page);
  await page.goto(normalAppUrl);
  await page.locator("#advanced-recommendation-settings summary").click();
  await expect(page.locator("#diagnostic-app")).toHaveText(`${appVersion} · source ${sourceRevision}`);
  await expect(page.locator("#diagnostic-data")).toContainText(manifest.tag);
  await expect(page.locator("#diagnostic-data")).toContainText(manifest.bundleId.slice(0, 12));
  await expect(page.locator("#diagnostic-model")).toContainText("declared; not loaded");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations (2 factors)");
  await expect(page.locator("#diagnostic-model")).toContainText("loaded");
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-render-status")).toContainText("nodes");
  await expect(page.locator("#network-versions")).toContainText("graph-compact-v2");
  await expect(page.locator("#network-versions")).toContainText("model-mf-compact-v1");
  await expect(page.locator("#network-selection")).toContainText("18/18 source ratings selected");
  await expect(page.locator("#network-explorer-sample")).toContainText("10/11 retained pair edges");
  await expect(page.locator("#network-explorer-sample")).toContainText("10/18 retained user-anime edges sampled");
  await expect(page.locator("#toggle-users")).toBeEnabled();
  await expect(page.locator("#toggle-users")).not.toBeChecked();
  await page.locator("#network-search-input").fill("anime:101");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-mode-status")).toContainText("recommendation graph");
  await expect(page.locator("#toggle-users")).toBeDisabled();
  await page.locator("#reset-neighborhood").click();
  await expect(page.locator("#network-mode-status")).toContainText("Explorer sample overview");
  await expect(page.locator("#toggle-users")).toBeEnabled();
  expect(requests.filter((name) => name === "/data/active.json")).toHaveLength(1);
  expect(requests).toContain(`/data/bundles/${manifest.bundleId}/graph.compact.json`);
  expect(requests).toContain(`/data/bundles/${manifest.bundleId}/model-mf-web.compact.json`);
  expect(requests).toContain(`/data/bundles/${manifest.bundleId}/graph-explorer.compact.json`);
  expect(requests.some((name) => name === "/data/graph.json" ||
    name === "/data/graph.compact.json.gz")).toBe(false);
});

test("browser candidate reads hash-bound invented metadata without provider enrichment", async ({ page }) => {
  const { requests, release } = await routeMetadataBundle(page);
  await page.goto(normalAppUrl);
  await expect(page.locator("#diagnostic-data")).toContainText(release.bundleId.slice(0, 12));
  expect(requests).toContain(`/data/bundles/${release.bundleId}/catalog.identity.json`);
  expect(requests).toContain(`/data/bundles/${release.bundleId}/catalog.metadata.json`);
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  const card = page.locator(".rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(card).toContainText("Invented board All (Fixtureland)");
  await expect(card).toContainText("95 min");
  await expect(card).toContainText("Catalog community score: 8.10/10");
  await page.locator("#filter-year-min").fill("2022");
  await page.locator("#filter-year-min").press("Tab");
  await expect(card).toBeVisible();
  expect(requests.some((name) => name.startsWith("jikan:/v4/anime/"))).toBe(false);
});

test("mapped invented Wikibase statements reach browser filters with unknown community scores", async ({ page }) => {
  const { requests } = await routeMetadataBundle(page, { mappedSource: true });
  await page.goto(normalAppUrl);
  await page.locator("#anime-input").fill("Invented Copper Sky");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#selected-anime .chip-title")).toHaveText("Copper Comet");
  const card = page.locator(".rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(card).toContainText("Invented board All (Fixtureland)");
  await expect(card).toContainText("95 min");
  await expect(card.locator(".rec-community-score")).toHaveCount(0);
  await page.locator("#filter-genre").selectOption("adventure");
  await page.locator("#filter-year-min").fill("2022");
  await page.locator("#filter-year-min").press("Tab");
  await expect(card).toBeVisible();
  await page.locator("#filter-min-score").evaluate((input: HTMLInputElement) => {
    input.value = "1";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#rec-results .rec-item")).toHaveCount(0);
  expect(requests.some((name) => name.startsWith("jikan:"))).toBe(false);
});

for (const missing of ["year", "genre"] as const) {
  test(`combined filters exclude a bundled title with unknown ${missing} and recover when cleared`, async ({ page }) => {
    const { requests } = await routeMetadataBundle(page, { missingYear: missing === "year", missingGenre: missing === "genre" });
    await page.goto(normalAppUrl);
    await page.locator("#anime-input").fill("Copper Comet");
    await page.locator("#add-preference").selectOption("liked");
    await page.locator("#add-anime-form button").click();
    const card = page.locator(".rec-item").filter({ hasText: "Moonlit Workshop" });
    await expect(card).toBeVisible();
    if (missing === "year") await page.locator("#filter-genre").selectOption("adventure");
    else {
      await page.locator("#filter-year-min").fill("2022");
      await page.locator("#filter-year-min").press("Tab");
    }
    await expect(card).toBeVisible();
    if (missing === "year") {
      await page.locator("#filter-year-min").fill("2022");
      await page.locator("#filter-year-min").press("Tab");
    } else await page.locator("#filter-genre").selectOption("adventure");
    await expect(card).toHaveCount(0);
    if (missing === "year") {
      await page.locator("#filter-year-min").fill("");
      await page.locator("#filter-year-min").press("Tab");
    } else await page.locator("#filter-genre").selectOption("");
    await expect(card).toBeVisible();
    expect(requests.some((name) => name.startsWith("jikan:"))).toBe(false);
  });
}

test("browser reads the exact synthetic v2 directory activated by the local installer", async ({ page }) => {
  const root = mkdtempSync(path.join(tmpdir(), "invented-browser-v2-install-"));
  try {
    const storeDir = path.join(root, "data");
    const sourceBytes = Buffer.from("invented-browser-source-bytes");
    const files = {
      neighborhood: fixture("graph.aggregate.compact.json"),
      explorer: fixture("graph-explorer.aggregate.compact.json"),
      catalog: fixture("catalog.identity.json"),
      metadata: Buffer.from(`${JSON.stringify({ format: "anime-metadata-catalog-v1",
        source: { name: "invented-fixture", snapshotAt: "2026-09-24T00:00:00.000Z",
          snapshotSha256: releaseSha256(sourceBytes) },
        anime: [{ animeId: 102, sourceItemId: "invented:102", title: "Moonlit Workshop",
          aliases: [], genres: ["Adventure"], year: 2022, mediaFormat: "Movie",
          episodeCount: 1, runtimeMinutes: 95, contentClassification: null,
          communityScore: 8.1, relations: null }] }, null, 2)}\n`),
    };
    const tag = "data-vsynthetic-browser-install";
    const release = buildMetadataReleaseManifest(files, { tag, fixtureGenesis: true }, sourceBytes);
    const manifestBytes = Buffer.from(`${JSON.stringify(release, null, 2)}\n`);
    const assets = new Map<string, Buffer>([
      ["release-manifest.json", manifestBytes], ["graph.compact.json", files.neighborhood],
      ["graph-explorer.compact.json", files.explorer],
      ["catalog.identity.json", files.catalog], ["catalog.metadata.json", files.metadata],
    ]);
    const installed = await installMetadataReleaseBundle({ storeDir, tag, fixtureOnly: true,
      transport: { tag, assets: [...assets.keys()],
        async fetchAsset(name) {
          const bytes = assets.get(name);
          return bytes ? new Response(new Uint8Array(bytes), { status: 200 })
            : new Response("missing", { status: 404 });
        } },
    });
    const requests: string[] = [];
    const externalRequests: string[] = [];
    await page.route("**/*", (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.startsWith("/data/")) {
        requests.push(url.pathname);
        const name = url.pathname.slice(`/data/bundles/${release.bundleId}/`.length);
        const file = url.pathname === "/data/active.json"
          ? path.join(storeDir, "active.json")
          : url.pathname.startsWith(`/data/bundles/${release.bundleId}/`) && assets.has(name)
            ? path.join(installed.bundleDir, name) : null;
        return file ? route.fulfill({ contentType: "application/json", body: readFileSync(file) })
          : route.fulfill({ status: 404, body: "" });
      }
      if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
        externalRequests.push(`${url.hostname}${url.pathname}`);
        return route.abort();
      }
      return route.continue();
    });
    await page.goto(normalAppUrl);
    await expect(page.locator("#diagnostic-data")).toContainText(tag);
    await page.locator("#anime-input").fill("Copper Comet");
    await page.locator("#add-preference").selectOption("liked");
    await page.locator("#add-anime-form button").click();
    await expect(page.locator(".rec-item").filter({ hasText: "Moonlit Workshop" }))
      .toContainText("Catalog community score: 8.10/10");
    expect(requests).toContain(`/data/bundles/${release.bundleId}/catalog.metadata.json`);
    expect(requests.some((name) => name.startsWith("/data/graph.compact.json"))).toBe(false);
    expect(externalRequests.filter((name) => name.startsWith("api.jikan.moe/v4/anime/")))
      .toEqual([]);
  } finally {
    const resolved = path.resolve(root);
    if (!resolved.startsWith(path.resolve(tmpdir()) + path.sep)) {
      throw new Error("Refusing cleanup outside temporary directory");
    }
    rmSync(resolved, { recursive: true, force: true });
  }
});

test("bundled classification is text, never executable card markup", async ({ page }) => {
  await routeMetadataBundle(page, { classificationMarkup: true });
  await page.goto(normalAppUrl);
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  const card = page.locator(".rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(card).toContainText("<img src=x onerror=alert(1)>");
  await expect(card.locator("img")).toHaveCount(0);
});

test("present bad metadata bytes fail closed at their named asset", async ({ page }) => {
  const { requests } = await routeMetadataBundle(page, { corruptBytes: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "catalog.metadata.json: byte length or SHA-256 differs from release-manifest.json");
  await expect(page.locator("#diagnostic-code")).toContainText("DATA-002");
  expect(requests.some((name) => name.startsWith("jikan:/v4/anime/"))).toBe(false);
});

test("present invalid metadata names its field without provider fallback", async ({ page }) => {
  const { requests } = await routeMetadataBundle(page, { malformedYear: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText("catalog.metadata.json: anime[1].year");
  expect(requests.some((name) => name.startsWith("jikan:/v4/anime/"))).toBe(false);
});

test("metadata mapped outside identity catalog fails before use", async ({ page }) => {
  const { requests } = await routeMetadataBundle(page, { unknownId: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "catalog.metadata.json: anime[1].animeId is outside catalog.identity.json");
  expect(requests.some((name) => name.startsWith("jikan:/v4/anime/"))).toBe(false);
});

test("a declared identity map that differs from the graph fails before metadata use", async ({ page }) => {
  await routeMetadataBundle(page, { catalogGraphMismatch: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "catalog.identity.json: anime IDs or titles differ from graph.compact.json");
});

test("a present corrupt bundle graph fails without reading a legacy graph", async ({ page }) => {
  const requests = await routeBundle(page, { corruptGraph: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "graph.compact.json: byte length or SHA-256 differs from release-manifest.json",
  );
  await expect(page.locator("#diagnostic-code")).toContainText("DATA-002");
  await expect(page.locator("#diagnostic-action")).toContainText("last verified data release");
  expect(requests.some((name) => name === "/data/graph.json" ||
    name === "/data/graph.compact.json.gz")).toBe(false);
});

test("a transport exception cannot echo invented private text from a required data read", async ({ page }) => {
  const privateText = "invented-private-user raw-history-1-2-3";
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.addInitScript((message) => {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) => String(input).endsWith("/data/active.json")
      ? Promise.reject(new Error(message)) : nativeFetch(input, init);
  }, privateText);
  await page.goto(normalAppUrl);
  await expect(page.locator("#diagnostic-code")).toContainText("DATA-001");
  await expect(page.locator("#rec-message")).toContainText("unable to load or verify the required asset");
  const visibleFailure = await page.locator("#rec-message, #local-diagnostics").allTextContents();
  expect([...visibleFailure, ...pageErrors].join(" ")).not.toContain(privateText);
});

test("a present malformed active pointer fails closed", async ({ page }) => {
  const requests = await routeBundle(page, { corruptPointer: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "active.json: manifestSha256 must be a lowercase SHA-256 digest",
  );
  expect(requests.some((name) => name === "/data/graph.json" ||
    name === "/data/graph.compact.json.gz")).toBe(false);
});

test("a changed manifest cannot select a different asset set", async ({ page }) => {
  const requests = await routeBundle(page, { corruptManifest: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "release-manifest.json: SHA-256 differs from active.json.manifestSha256",
  );
  expect(requests.some((name) => name === "/data/graph.json" ||
    name === "/data/graph.compact.json.gz")).toBe(false);
});

test("a stale explorer cache is rejected inside the pinned bundle", async ({ page }) => {
  await routeBundle(page, { corruptExplorer: true });
  await page.goto(normalAppUrl);
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-render-status")).toContainText(
    "graph-explorer.compact.json: byte length or SHA-256 differs from release-manifest.json",
  );
  await expect(page.locator("#diagnostic-code")).toContainText("EXPLORER-001");
});

test("an explorer transport exception stays out of local status and diagnostics", async ({ page }) => {
  const privateText = "invented-private-user raw-history-4-5-6";
  await page.addInitScript((message) => {
    const nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) => String(input).includes("/graph-explorer.compact.json")
      ? Promise.reject(new Error(message)) : nativeFetch(input, init);
  }, privateText);
  await routeBundle(page);
  await page.goto(normalAppUrl);
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#diagnostic-code")).toContainText("EXPLORER-001");
  await expect(page.locator("#network-render-status")).toContainText("unable to load or verify the optional asset");
  const visibleFailure = await page.locator("#network-render-status, #local-diagnostics").allTextContents();
  expect(visibleFailure.join(" ")).not.toContain(privateText);
});

test("a data-only bundle does not borrow a legacy model", async ({ page }) => {
  const requests = await routeBundle(page, { withoutModel: true });
  await page.goto(normalAppUrl);
  await page.locator("#advanced-recommendation-settings summary").click();
  await expect(page.locator("#diagnostic-model")).toHaveText("Not included in this data release");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
  expect(requests.some((name) => name.includes("model-mf-web"))).toBe(false);
});

test("aggregate-only bundle ranks pairs and labels sampled popularity unavailable", async ({ page }) => {
  const { requests, graphBytes, release } = await routeAggregateBundle(page);
  expect(graphBytes.toString("utf8")).not.toContain("fixture-overlap-a");
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-engine-status")).toContainText("community-score exploration");
  expect(requests).toEqual([
    "/data/active.json",
    `/data/bundles/${release.bundleId}/release-manifest.json`,
    `/data/bundles/${release.bundleId}/graph.compact.json`,
  ]);
  const initialGraph = JSON.parse(graphBytes.toString("utf8"));
  expect(initialGraph).toMatchObject({ format: "graph-compact-v3", role: "recommendation",
    userIds: [], ua: [], userCount: 0 });
  await page.locator("#discovery-view").selectOption("popularity");
  await expect(page.locator("#rec-engine-status")).toContainText(
    "Popularity proxy unavailable in this aggregate-only graph");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#discovery-view").selectOption("auto");
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph recommendations");
  await expect(page.locator("#rec-results")).toContainText("Moonlit Workshop");
  expect(requests).toContain(`/data/bundles/${release.bundleId}/graph.compact.json`);
  expect(requests.some((name) => name.includes("anonymized-ratings"))).toBe(false);
});

test("aggregate-only graph can serve a pinned item-only model without user history", async ({ page }) => {
  const { requests, release } = await routeAggregateBundle(page, { withModel: true });
  expect(release.model?.coverage.mappedAnimeCount).toBe(8);
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-engine-status")).toBeVisible();
  expect(requests.some((name) => name.includes("model-mf-web") ||
    name.includes("graph-explorer"))).toBe(false);
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations (2 factors)");
  expect(requests).toContain(`/data/bundles/${release.bundleId}/model-mf-web.compact.json`);
  expect(requests.some((name) => name.includes("graph-explorer"))).toBe(false);
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-render-status")).toContainText("nodes");
  expect(requests).toContain(`/data/bundles/${release.bundleId}/graph-explorer.compact.json`);
  expect(requests.some((name) => name.includes("anonymized-ratings"))).toBe(false);
});

test("synthetic demo requests aggregate neighborhoods first and defers optional assets", async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (request) => {
    const pathname = new URL(request.url()).pathname;
    if (pathname.startsWith("/demo-data/")) requests.push(pathname);
  });
  await page.goto("/");
  await expect(page.locator("#rec-engine-status")).toContainText("community-score exploration");
  expect(requests).toEqual([
    "/demo-data/catalog.json", "/demo-data/graph.aggregate.compact.json",
  ]);
  const graph = JSON.parse(fixture("graph.aggregate.compact.json").toString("utf8"));
  expect(graph).toMatchObject({ format: "graph-compact-v3", role: "recommendation",
    userIds: [], ua: [], userCount: 0 });
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations");
  expect(requests).toContain("/demo-data/model-mf-web.compact.json");
  expect(requests).not.toContain("/demo-data/graph-explorer.aggregate.compact.json");
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-render-status")).toContainText("nodes");
  expect(requests).toContain("/demo-data/graph-explorer.aggregate.compact.json");
});
