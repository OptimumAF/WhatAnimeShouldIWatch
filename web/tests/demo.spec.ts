import { expect, test } from "@playwright/test";

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
  await expect(page.getByText("SYNTHETIC DEMO DATA")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Demo Suggestions" })).toBeVisible();
  await expect(page.locator("#username-import-submit")).toBeDisabled();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "demo");
  await expect(page.locator("#seasonal-status")).toHaveAttribute("data-state", "demo");

  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results")).toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Copper Comet" })).toHaveCount(0);
  await expect(page.locator("#metadata-status")).toContainText("2/2");
  await expect(page.locator("#metadata-status")).toHaveAttribute("data-state", "demo");
  expect(await page.evaluate(() => Object.keys(window.localStorage)))
    .toContain("wasiw.demo.recommendationState.v4");

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

  expect(unexpectedRequests).toEqual([]);
  expect(pageErrors).toEqual([]);
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

test("a valid selected-rating and selected-pair graph still loads recommendations and the network", async ({ page }) => {
  let selectedLegacyCompact: Record<string, unknown> | null = null;
  await page.route("**/demo-data/graph.compact.json", async (route) => {
    const response = await route.fetch();
    const graph = await response.json();
    for (const field of ["role", "graphId", "dataset", "semantics", "config", "truncation"]) delete graph[field];
    graph.format = "graph-compact-v1";
    graph.ua = graph.ua.slice(0, -1);
    graph.aa = [graph.aa[0]];
    graph.edgeCount = graph.ua.length + graph.aa.length;
    selectedLegacyCompact = graph;
    await route.fulfill({ response, json: graph });
  });
  await page.route("**/demo-data/graph-explorer.compact.json", async (route) => {
    if (!selectedLegacyCompact) throw new Error("Selected graph was not loaded before the explorer.");
    await route.fulfill({ json: selectedLegacyCompact });
  });
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#graph svg")).toBeVisible();
  await expect(page.locator("#network-render-status")).toContainText("edges");
});
