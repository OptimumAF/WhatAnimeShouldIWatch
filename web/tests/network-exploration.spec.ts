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
  const widths = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth,
    viewport: window.innerWidth,
    extending: [...document.querySelectorAll("#view-network *")]
      .filter((element) => element.getBoundingClientRect().right > window.innerWidth + 1)
      .slice(0, 8).map((element) => `${element.tagName.toLowerCase()}#${element.id || "-"}`),
  }));
  expect(widths.document, JSON.stringify(widths)).toBeLessThanOrEqual(widths.viewport);
  const button = await page.locator("#graph-zoom-in").boundingBox();
  expect(button?.height).toBeGreaterThanOrEqual(44);
  expect(providerRequests).toEqual([]);
});

test("large invented overview cancels stale builds, draws every signed pair, and keeps focus reversible", async ({ page }) => {
  const original = JSON.parse(readFileSync(new URL(
    "../public/demo-data/graph.aggregate.compact.json", import.meta.url), "utf8",
  )) as CompactGraphDataV3;
  const anime = Array.from({ length: 80 }, (_, index) =>
    [101 + index, `Invented Batch Title ${index + 1}`] as [number, string]);
  const pairs: [number, number, number, number][] = [];
  for (let left = 0; left < anime.length && pairs.length < 1200; left += 1) {
    for (let right = left + 1; right < anime.length && pairs.length < 1200; right += 1) {
      const weight = pairs.length % 3 === 0 ? 1 : pairs.length % 3 === 1 ? -1 : 0;
      pairs.push([left, right, weight, 1]);
    }
  }
  const withoutId = { ...original, anime, aa: pairs, animeCount: anime.length,
    nodeCount: anime.length, edgeCount: pairs.length,
    truncation: { ...original.truncation, inputRatings: 1200, selectedRatings: 1200,
      potentialPairVisits: 1200, pairVisits: 1200, candidatePairs: 1200,
      eligiblePairs: 1200, selectedPairs: 1200 } };
  const { graphId: _oldGraphId, ...core } = withoutId;
  const graph = { ...core, graphId: aggregateRecommendationGraphId(core) };
  const explorer = buildExplorerGraph(graph, 1200, 0);
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
  await page.evaluate(() => {
    const browser = window as Window & {
      graphYieldHeld?: boolean;
      releaseHeldGraphYield?: () => void;
    };
    const NativeChannel = window.MessageChannel;
    let holdFirstYield = true;
    window.MessageChannel = new Proxy(NativeChannel, {
      construct(target, args) {
        const channel = Reflect.construct(target, args) as MessageChannel;
        const post = channel.port2.postMessage.bind(channel.port2);
        channel.port2.postMessage = ((message: unknown) => {
          if (holdFirstYield) {
            holdFirstYield = false;
            browser.graphYieldHeld = true;
            browser.releaseHeldGraphYield = () => post(message);
          } else {
            post(message);
          }
        }) as MessagePort["postMessage"];
        return channel;
      },
    });
  });
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await page.locator("#min-weight").evaluate((input: HTMLInputElement) => {
    input.value = "0";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.waitForFunction(() => (window as Window & { graphYieldHeld?: boolean }).graphYieldHeld === true);
  await page.locator("#toggle-anime-edges").uncheck();
  await expect(page.locator("#network-mode-status")).toContainText("0 edges visible");
  await page.evaluate(() => {
    (window as Window & { releaseHeldGraphYield?: () => void }).releaseHeldGraphYield?.();
  });
  await expect(page.locator("#graph .graph-edge-layer path")).toHaveCount(0);
  await page.locator("#toggle-anime-edges").check();
  await expect(page.locator("#network-mode-status")).toContainText("1200 edges visible");
  const edges = page.locator("#graph .graph-edge-layer");
  await expect(edges).toHaveAttribute("data-render-mode", "batched-paths");
  await expect(edges.locator("line")).toHaveCount(0);
  expect(await edges.locator("path").count()).toBeLessThanOrEqual(3);
  for (const sign of ["positive", "negative", "neutral"]) {
    await expect(edges.locator(`path[data-edge-sign='${sign}']`)).toHaveCount(1);
  }
  const drawnPairs = await edges.locator("path").evaluateAll((paths) =>
    paths.reduce((sum, path) => sum + Number(path.getAttribute("data-edge-count")), 0));
  expect(drawnPairs).toBe(1200);
  const pathGeometry = await edges.locator("path").evaluateAll((paths) => paths.map((path) => ({
    declared: Number(path.getAttribute("data-edge-count")),
    segments: (path.getAttribute("d")?.match(/M/g) ?? []).length,
    hasInvalidNumber: /NaN|Infinity/.test(path.getAttribute("d") ?? ""),
    pointerEvents: getComputedStyle(path).pointerEvents,
  })));
  expect(pathGeometry.every((item) => item.declared === item.segments &&
    !item.hasInvalidNumber && item.pointerEvents === "none")).toBe(true);
  await expect(edges.locator("path[data-edge-sign='neutral']"))
    .toHaveAttribute("stroke-dasharray", "3 3");
  await expect(page.locator("#graph circle")).toHaveCount(80);
  const firstOverview = await page.locator("#graph svg.graph-svg").elementHandle();
  await page.locator("#toggle-anime-edges").uncheck();
  await expect(page.locator("#network-mode-status")).toContainText("0 edges visible");
  await page.locator("#toggle-anime-edges").check();
  await expect(page.locator("#network-mode-status")).toContainText("1200 edges visible");
  expect(await firstOverview!.evaluate((element) =>
    element === document.querySelector("#graph svg.graph-svg"))).toBe(true);
  await page.locator("#graph circle[data-node-id='anime:101']").click();
  await expect(edges.locator("path[stroke='var(--graph-dim-edge)']")).not.toHaveCount(0);

  await page.locator("#network-search-input").fill("anime:101");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-mode-status")).toContainText("Focused on");
  await expect(edges.locator("line")).toHaveCount(24);
  await expect(edges.locator("path")).toHaveCount(0);
  await page.locator("#reset-neighborhood").click();
  await expect(edges).toHaveAttribute("data-render-mode", "batched-paths");
  expect(await edges.locator("path").count()).toBeLessThanOrEqual(3);
  expect(providerRequests).toEqual([]);
});

test("large invented node overview keeps every list entry and pointer selection with grouped SVG paths", async ({ page }) => {
  const original = JSON.parse(readFileSync(new URL(
    "../public/demo-data/graph.aggregate.compact.json", import.meta.url), "utf8",
  )) as CompactGraphDataV3;
  const anime = Array.from({ length: 600 }, (_, index) =>
    [101 + index, `Invented Batch Node ${String(index + 1).padStart(3, "0")}`] as [number, string]);
  const pairs: [number, number, number, number][] = anime.map((_, index) => {
    const other = (index + 1) % anime.length;
    return [Math.min(index, other), Math.max(index, other), index % 3 === 0 ? -1 : 1, 1];
  });
  const withoutId = { ...original, anime, aa: pairs, animeCount: anime.length,
    nodeCount: anime.length, edgeCount: pairs.length,
    truncation: { ...original.truncation, inputRatings: 600, selectedRatings: 600,
      potentialPairVisits: 600, pairVisits: 600, candidatePairs: 600,
      eligiblePairs: 600, selectedPairs: 600 } };
  const { graphId: _oldGraphId, ...core } = withoutId;
  const graph = { ...core, graphId: aggregateRecommendationGraphId(core) };
  const explorer = buildExplorerGraph(graph, 600, 0);
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

  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await page.locator("#min-weight").evaluate((input: HTMLInputElement) => {
    input.value = "0";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#network-mode-status")).toContainText("600 nodes and 600 edges visible");
  const nodeLayer = page.locator("#graph .graph-node-layer");
  await expect(nodeLayer).toHaveAttribute("data-render-mode", "batched-paths");
  await expect(nodeLayer.locator("circle")).toHaveCount(0);
  expect(await nodeLayer.locator("path").evaluateAll((paths) =>
    paths.reduce((sum, path) => sum + Number(path.getAttribute("data-node-count")), 0))).toBe(600);
  await expect(page.locator("#network-node-list-status")).toContainText("600 visible nodes");
  const firstOverview = await page.locator("#graph svg.graph-svg").elementHandle();
  await page.locator("#toggle-anime-edges").uncheck();
  await expect(page.locator("#network-mode-status")).toContainText("0 edges visible");
  await page.locator("#toggle-anime-edges").check();
  await expect(page.locator("#network-mode-status")).toContainText("600 nodes and 600 edges visible");
  expect(await firstOverview!.evaluate((element) =>
    element === document.querySelector("#graph svg.graph-svg"))).toBe(true);

  await page.locator("#network-node-list summary").click();
  await page.locator("#network-node-filter").fill("Invented Batch Node 001");
  const choice = page.locator("#network-node-list-results button[data-node-id='anime:101']");
  await expect(choice).toHaveCount(1);
  await choice.press("Enter");
  await expect(choice).toBeFocused();
  await expect(choice).toHaveAttribute("aria-pressed", "true");
  const selectedLabel = page.locator("#graph .graph-label-layer text")
    .filter({ hasText: "Invented Batch Node 001" });
  await expect(selectedLabel).toHaveCount(1);
  const targetX = Number(await selectedLabel.getAttribute("x")) - 8;
  const targetY = Number(await selectedLabel.getAttribute("y")) + 8;
  await page.locator("#clear-selection").click();
  const graphPoint = (svg: SVGSVGElement, target: { x: number; y: number }) => {
    const point = svg.createSVGPoint();
    point.x = target.x;
    point.y = target.y;
    const screen = point.matrixTransform(svg.getScreenCTM()!);
    return { x: screen.x, y: screen.y };
  };
  const offscreenPoint = await page.locator("#graph svg.graph-svg")
    .evaluate(graphPoint, { x: targetX, y: targetY });
  await page.evaluate((targetY) => window.scrollBy(0, targetY - innerHeight / 2), offscreenPoint.y);
  const clickPoint = await page.locator("#graph svg.graph-svg")
    .evaluate(graphPoint, { x: targetX, y: targetY });
  await page.mouse.click(clickPoint.x, clickPoint.y);
  await expect(page.locator("#network-search-message")).toContainText("Invented Batch Node 001 (anime:101)");
  await expect(choice).toHaveAttribute("aria-pressed", "true");
  expect(await nodeLayer.locator("path").evaluateAll((paths) =>
    paths.reduce((sum, path) => sum + Number(path.getAttribute("data-node-count")), 0))).toBe(600);
  await page.setViewportSize({ width: 320, height: 720 });
  const pageWidth = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth,
    viewport: window.innerWidth,
  }));
  expect(pageWidth.document).toBeLessThanOrEqual(pageWidth.viewport);
  await expect(choice).toHaveAttribute("aria-pressed", "true");
  expect(providerRequests).toEqual([]);
});
