import crypto from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { projectAggregateGraph } from "../../pipeline/src/core/aggregate-projection";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph";
import { buildReleaseManifest } from "../../pipeline/src/release-manifest";

const base = "/WhatAnimeShouldIWatch/";
const appVersion = JSON.parse(readFileSync(new URL("../../package.json", import.meta.url), "utf8")).version as string;
const sourceRevision = /^[a-f0-9]{40}$/.test(process.env.GITHUB_SHA ?? "")
  ? process.env.GITHUB_SHA!.slice(0, 12) : "local build";
const fixture = (name: string): Buffer =>
  readFileSync(new URL(`../public/demo-data/${name}`, import.meta.url));
const encoded = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);

test("direct project-path navigation loads a pinned data-only release and graph fallback", async ({ page }) => {
  const graph = projectAggregateGraph(JSON.parse(fixture("graph.compact.json").toString("utf8")));
  const files = {
    neighborhood: encoded(graph), explorer: encoded(buildExplorerGraph(graph, 5, 0)),
    catalog: fixture("catalog.identity.json"),
  };
  const manifest = buildReleaseManifest(files,
    { tag: "data-vinvented-pages-path", fixtureGenesis: true });
  const manifestBytes = encoded(manifest);
  const pointer = { format: "active-release-bundle-v1", tag: manifest.tag,
    bundleId: manifest.bundleId,
    manifestSha256: crypto.createHash("sha256").update(manifestBytes).digest("hex") };
  const assets = new Map<string, Buffer>([
    ["release-manifest.json", manifestBytes], ["graph.compact.json", files.neighborhood],
    ["graph-explorer.compact.json", files.explorer],
    ["catalog.identity.json", files.catalog],
  ]);
  const requested: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" }, body: '{"data":[]}' });
    }
    if (url.pathname.startsWith(`${base}data/`)) {
      requested.push(url.pathname);
      if (url.pathname === `${base}data/active.json`) {
        return route.fulfill({ contentType: "application/json", body: JSON.stringify(pointer) });
      }
      const prefix = `${base}data/bundles/${manifest.bundleId}/`;
      const bytes = url.pathname.startsWith(prefix) ? assets.get(url.pathname.slice(prefix.length)) : null;
      return bytes ? route.fulfill({ contentType: "application/json", body: bytes })
        : route.fulfill({ status: 404, body: "" });
    }
    if (url.hostname !== "127.0.0.1") return route.abort();
    return route.continue();
  });
  await page.goto("http://127.0.0.1:5175/WhatAnimeShouldIWatch/deep/link/");
  await expect(page.locator("#diagnostic-app")).toHaveText(`${appVersion} · source ${sourceRevision}`);
  await expect(page.locator("#diagnostic-data")).toContainText(manifest.tag);
  await expect(page.locator("#diagnostic-model")).toHaveText("Not included in this data release");
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute("href", `${base}favicon.svg`);
  await expect(page.locator('link[rel="manifest"]')).toHaveAttribute("href",
    `${base}manifest.webmanifest`);
  await expect(page.locator('link[rel="apple-touch-icon"]')).toHaveAttribute("href",
    `${base}icons/apple-touch-icon.png`);
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
  expect(requested).toContain(`${base}data/active.json`);
  expect(requested).toContain(`${base}data/bundles/${manifest.bundleId}/graph.compact.json`);
  expect(requested.some((name) => name.includes("/deep/link/data/") ||
    name.includes("model-mf-web"))).toBe(false);
});
