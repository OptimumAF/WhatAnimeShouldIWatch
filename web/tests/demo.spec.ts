import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph";
import { aggregateRecommendationGraphId } from "../../pipeline/src/core/graph-contract";
import type { CompactGraphDataV3 } from "../../pipeline/src/types";

test("offline demo loads synthetic graph, catalog, and model without provider requests", async ({ page }) => {
  const unexpectedRequests: string[] = [];
  const pageErrors: string[] = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith("/data/") ||
        /api\.jikan\.moe|graphql\.anilist\.co|myanimelist\.net|r\.jina\.ai/.test(url.hostname)) {
      unexpectedRequests.push(request.url());
    }
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
      return route.abort();
    }
    return route.continue();
  });

  await page.goto("/");
  await page.locator("#advanced-recommendation-settings summary").click();
  await expect(page.getByText("SYNTHETIC DEMO DATA")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Demo Suggestions" })).toBeVisible();
  await expect(page.locator("#username-import-submit")).toBeDisabled();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "demo");
  await expect(page.locator("#seasonal-status")).toHaveAttribute("data-state", "demo");

  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results")).toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Copper Comet" })).toHaveCount(0);
  await expect(page.locator("#metadata-status")).toContainText("2/2");
  await expect(page.locator("#metadata-status")).toHaveAttribute("data-state", "demo");
  expect(await page.evaluate(() => Object.keys(window.localStorage)))
    .toContain("wasiw.demo.recommendationState.v5");

  await page.locator("#filter-genre").selectOption("slice of life");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);
  await page.locator("#clear-rec-filters").click();

  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("ML model recommendations");
  await expect(page.locator("#rec-results li").first()).toBeVisible();
  await page.locator("#rec-method").selectOption("hybrid");
  await expect(page.locator("#rec-engine-status")).toContainText("hybrid recommendations");
  await expect(page.locator("#rec-results li").first()).toBeVisible();

  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#graph svg")).toBeVisible();
  await expect(page.locator("#network-render-status")).toContainText("nodes");
  await expect(page.locator("#network-versions")).toContainText("graph-compact-v3");
  await expect(page.locator("#network-versions")).toContainText("model-mf-compact-v1");

  expect(unexpectedRequests).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test("aggregate explorer labels selected evidence and omissions without implying no users", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-render-status")).toContainText("nodes");
  await expect(page.locator("#network-versions")).toContainText("Recommendation graph: graph-compact-v3");
  await expect(page.locator("#network-versions")).toContainText("expected format model-mf-compact-v1 when requested");
  await expect(page.locator("#network-selection")).toContainText("18/18 source ratings selected");
  await expect(page.locator("#network-selection")).toContainText("11/11 eligible pair edges retained");
  await expect(page.locator("#network-explorer-sample")).toContainText("10/11 retained pair edges (1 omitted; sample cap 10)");
  await expect(page.locator("#network-explorer-sample")).toContainText("User rows are deliberately omitted from v3");
  await expect(page.locator("#network-drawing-limits")).toContainText("12000 pair edges and 4000 user-anime edges");
  await expect(page.locator("#network-scope-caveat")).toContainText("does not prove no relationship");
  await expect(page.locator("#toggle-users")).toBeDisabled();
  await expect(page.locator("#toggle-users-label")).toContainText("User rows omitted");
  await expect(page.locator("#network-search-input")).toHaveAttribute("aria-label", "Search loaded anime nodes");
  await expect(page.locator("#stats")).toContainText("User rows");
  await expect(page.locator("#stats")).toContainText("omitted from v3");
  await expect(page.locator("#network-scope-caveat")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("network explorer distinguishes signed v1 pair preference without calling it similarity", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await page.locator("#min-weight").evaluate((input: HTMLInputElement) => {
    input.value = "0";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const positive = page.locator("#graph .graph-edge-layer line[data-edge-sign='positive']");
  const negative = page.locator("#graph .graph-edge-layer line[data-edge-sign='negative']");
  const neutral = page.locator("#graph .graph-edge-layer line[data-edge-sign='neutral']");
  await expect.poll(() => positive.count()).toBeGreaterThan(0);
  expect(await negative.count()).toBeGreaterThan(0);
  expect(await neutral.count()).toBeGreaterThan(0);
  expect(await positive.first().getAttribute("stroke"))
    .not.toBe(await negative.first().getAttribute("stroke"));
  expect(await neutral.first().getAttribute("stroke-dasharray")).toBe("3 3");
  await expect(page.locator("#network-edge-legend")).toContainText("pair preference");
  await expect(page.locator("#network-edge-legend")).not.toContainText("similarity");
});

test("a valid selected-pair aggregate graph still loads recommendations and the network", async ({ page }) => {
  const original = JSON.parse(readFileSync(
    new URL("../public/demo-data/graph.aggregate.compact.json", import.meta.url), "utf8",
  )) as CompactGraphDataV3;
  const selectedWithoutId = { ...original,
    config: { ...original.config, maxAnimeAnimeEdges: 1 },
    truncation: { ...original.truncation, selectedPairs: 1,
      excludedByOutputLimit: original.truncation.excludedByOutputLimit + original.aa.length - 1 },
    aa: [original.aa[0]], edgeCount: 1 };
  const { graphId: _oldGraphId, ...core } = selectedWithoutId;
  const graph = { ...core, graphId: aggregateRecommendationGraphId(core) };
  const explorer = buildExplorerGraph(graph, 1, 0);
  await page.route("**/demo-data/graph.aggregate.compact.json", (route) => route.fulfill({ json: graph }));
  await page.route("**/demo-data/graph-explorer.aggregate.compact.json", (route) =>
    route.fulfill({ json: explorer }));
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#graph svg")).toBeVisible();
  await expect(page.locator("#network-render-status")).toContainText("edges");
  await page.locator("#network-search-input").fill("anime:105");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-search-message")).toContainText("Focused neighborhood");
  await expect(page.locator("#graph circle[data-node-id='anime:105']")).toBeVisible();
  await expect(page.locator("#network-mode-status")).toContainText("No retained pair edge meets this filter");
  await page.locator("#network-search-input").fill("anime:999");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-search-message")).toContainText("does not prove no relationship");
});
