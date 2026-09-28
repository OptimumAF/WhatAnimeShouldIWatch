import crypto from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import { projectAggregateGraph } from "../../pipeline/src/core/aggregate-projection";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph";
import { buildReleaseManifest } from "../../pipeline/src/release-manifest";

const normalAppUrl = "http://127.0.0.1:5174/";
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

test("normal mode pins one verified bundle for graph, model, and explorer", async ({ page }) => {
  const requests = await routeBundle(page);
  await page.goto(normalAppUrl);
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations (2 factors)");
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-render-status")).toContainText("nodes");
  expect(requests.filter((name) => name === "/data/active.json")).toHaveLength(1);
  expect(requests).toContain(`/data/bundles/${manifest.bundleId}/graph.compact.json`);
  expect(requests).toContain(`/data/bundles/${manifest.bundleId}/model-mf-web.compact.json`);
  expect(requests).toContain(`/data/bundles/${manifest.bundleId}/graph-explorer.compact.json`);
  expect(requests.some((name) => name === "/data/graph.json" ||
    name === "/data/graph.compact.json.gz")).toBe(false);
});

test("a present corrupt bundle graph fails without reading a legacy graph", async ({ page }) => {
  const requests = await routeBundle(page, { corruptGraph: true });
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "graph.compact.json: byte length or SHA-256 differs from release-manifest.json",
  );
  expect(requests.some((name) => name === "/data/graph.json" ||
    name === "/data/graph.compact.json.gz")).toBe(false);
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
});

test("a data-only bundle does not borrow a legacy model", async ({ page }) => {
  const requests = await routeBundle(page, { withoutModel: true });
  await page.goto(normalAppUrl);
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
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations (2 factors)");
  expect(requests).toContain(`/data/bundles/${release.bundleId}/model-mf-web.compact.json`);
  expect(requests.some((name) => name.includes("anonymized-ratings"))).toBe(false);
});
