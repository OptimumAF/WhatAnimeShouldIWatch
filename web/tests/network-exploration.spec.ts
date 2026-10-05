import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { buildExplorerGraph } from "../../pipeline/src/core/explorer-graph";
import { aggregateRecommendationGraphId } from "../../pipeline/src/core/graph-contract";
import type { CompactGraphDataV3 } from "../../pipeline/src/types";
import { selectAnimeNeighborhood } from "../src/network-neighborhood";

test("bounded local neighborhood keeps sampled overview, keyboard list, viewport, and reset usable", async ({ page }) => {
  const original = JSON.parse(readFileSync(new URL(
    "../public/demo-data/graph.aggregate.compact.json", import.meta.url), "utf8",
  )) as CompactGraphDataV3;
  const anime = Array.from({ length: 41 }, (_, index) =>
    [101 + index, `Invented Node ${String(index + 1).padStart(2, "0")}`] as [number, string]);
  const pairs: [number, number, number, number][] = anime.slice(1).map((_, index) =>
    [0, index + 1, (index % 5 === 0 ? -1 : 1) * (80 - index) / 40, 1 + index % 3]);
  const withoutId = { ...original, anime, aa: pairs, animeCount: anime.length,
    nodeCount: anime.length, edgeCount: pairs.length,
    truncation: { ...original.truncation, inputRatings: 41, selectedRatings: 41,
      potentialPairVisits: 40, pairVisits: 40, candidatePairs: 40,
      eligiblePairs: 40, selectedPairs: 40 } };
  const { graphId: _oldGraphId, ...core } = withoutId;
  const graph = { ...core, graphId: aggregateRecommendationGraphId(core) };
  expect(selectAnimeNeighborhood(graph, "anime:141", 25, 24, 1)?.edges).toHaveLength(1);
  const explorer = buildExplorerGraph(graph, 8, 0);
  const providerRequests: string[] = [];
  page.on("request", (request) => {
    if (/api\.jikan\.moe|graphql\.anilist\.co|myanimelist\.net|r\.jina\.ai/.test(new URL(request.url()).hostname)) {
      providerRequests.push(request.url());
    }
  });
  await page.route("**/*", (route) => {
    const host = new URL(route.request().url()).hostname;
    return host === "127.0.0.1" || host === "localhost" ? route.continue() : route.abort();
  });
  await page.route("**/demo-data/graph.aggregate.compact.json", (route) => route.fulfill({ json: graph }));
  await page.route("**/demo-data/graph-explorer.aggregate.compact.json", (route) =>
    route.fulfill({ json: explorer }));

  await page.goto("/");
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-mode-status")).toContainText("Explorer sample overview");
  await expect(page.locator("#graph .graph-edge-layer line")).toHaveCount(8);
  await expect(page.locator("#graph circle")).toHaveCount(9);
  await expect(page.locator("#graph circle[data-node-id='anime:141']")).toHaveCount(0);
  await page.locator("#min-weight").evaluate((input: HTMLInputElement) => {
    input.value = "0";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });

  await page.locator("#network-search-input").fill("anime:141");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-mode-status")).toContainText("recommendation graph");
  await expect(page.locator("#graph circle[data-node-id='anime:141']")).toBeVisible();
  await expect(page.locator("#graph circle")).toHaveCount(2);

  await page.locator("#network-search-input").fill("anime:101");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-mode-status")).toContainText("24/40 retained signed pair edges");
  await expect(page.locator("#network-mode-status")).toContainText("16 matching pair edges omitted");
  await expect(page.locator("#graph circle")).toHaveCount(25);
  await expect(page.locator("#graph .graph-edge-layer line")).toHaveCount(24);
  await expect(page.locator("#stats")).toContainText("25 / 41");
  await expect(page.locator("#graph .graph-edge-layer line[data-edge-sign='negative']"))
    .not.toHaveCount(0);
  await expect(page.locator("#toggle-users")).toBeDisabled();
  await expect(page.locator("#graph svg circle[tabindex]")).toHaveCount(0);
  await expect(page.locator("#inspect-count")).toContainText("absolute signed weight");
  await expect(page.locator("#inspect-list .inspect-item-label").first()).toHaveText("Invented Node 02");

  await page.locator("#network-node-list summary").click();
  await expect(page.locator("#network-node-list-results button")).toHaveCount(15);
  await expect(page.locator("#network-node-list-results button").first()).toHaveText("Invented Node 01");
  await expect(page.locator("#network-node-list-results button").nth(1))
    .toContainText("Invented Node 02 · -2.000 pair preference · support 1");
  await page.locator("#network-node-filter").fill("Invented Node 10");
  await expect(page.locator("#network-node-list-results button")).toHaveCount(1);
  await expect(page.locator("#network-node-list-results button")).toContainText("support 3");
  await page.locator("#network-node-list-results button").press("Enter");
  await expect(page.locator("#network-node-list-results button")).toBeFocused();
  await expect(page.locator("#network-node-list-results button")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#focus-neighborhood").click();
  await expect(page.locator("#focus-neighborhood")).toBeFocused();
  await expect(page.locator("#focus-neighborhood")).toBeInViewport();
  await expect(page.locator("#network-mode-status")).toContainText("Invented Node 10");
  await expect(page.locator("#graph circle")).toHaveCount(2);

  await expect(page.locator("#graph svg")).toHaveAttribute("viewBox", "0 0 1200 900");
  await page.locator("#graph-zoom-in").click();
  await expect(page.locator("#graph svg")).toHaveAttribute("viewBox", "120 90 960 720");
  const plot = await page.locator("#graph svg").boundingBox();
  expect(plot).not.toBeNull();
  await page.mouse.move(plot!.x + 35, plot!.y + 80);
  await page.mouse.down();
  await page.mouse.move(plot!.x + 95, plot!.y + 80, { steps: 4 });
  await page.mouse.up();
  const draggedX = Number((await page.locator("#graph svg").getAttribute("viewBox"))?.split(" ")[0]);
  expect(draggedX).toBeLessThan(120);
  await page.locator("#graph-shell").focus();
  await page.keyboard.press("ArrowRight");
  await expect.poll(async () => Number((await page.locator("#graph svg").getAttribute("viewBox"))?.split(" ")[0]))
    .toBeGreaterThan(draggedX);
  await page.keyboard.press("0");
  await expect(page.locator("#graph svg")).toHaveAttribute("viewBox", "0 0 1200 900");

  await page.locator("#reset-neighborhood").click();
  await expect(page.locator("#reset-neighborhood")).toBeFocused();
  await expect(page.locator("#network-mode-status")).toContainText("Explorer sample overview");
  await expect(page.locator("#graph .graph-edge-layer line")).toHaveCount(8);
  await expect(page.locator("#graph circle")).toHaveCount(9);
  await page.setViewportSize({ width: 320, height: 720 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  const button = await page.locator("#graph-zoom-in").boundingBox();
  expect(button?.height).toBeGreaterThanOrEqual(44);
  expect(providerRequests).toEqual([]);
});
