import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

async function mockNormalModeCatalog(page: Page): Promise<() => number> {
  type SyntheticMetadata = { animeId: number; year: number | null; score: number | null; genres: string[] };
  const graph = JSON.parse(readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8"));
  const catalog: { anime: SyntheticMetadata[] } = JSON.parse(
    readFileSync(new URL("../public/demo-data/catalog.json", import.meta.url), "utf8"),
  );
  const byId = new Map<number, SyntheticMetadata>(catalog.anime.map((item) => [item.animeId, item]));
  let metadataRequests = 0;
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/data/graph.compact.json.gz") return route.fulfill({ status: 404, body: "" });
    if (url.pathname === "/data/graph.compact.json") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(graph) });
    }
    if (url.pathname.startsWith("/data/")) return route.fulfill({ status: 404, body: "" });
    const metadataMatch = /^\/v4\/anime\/(\d+)\/full$/.exec(url.pathname);
    if (url.hostname === "api.jikan.moe" && metadataMatch) {
      metadataRequests += 1;
      const item = byId.get(Number(metadataMatch[1]));
      return route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        body: JSON.stringify({ data: item ? { year: item.year, score: item.score,
          genres: item.genres.map((name: string) => ({ name })), synopsis: "Synthetic metadata" } : null }) });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" }, body: '{"data":[]}' });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  return () => metadataRequests;
}

test("no-preference demo offers sampled popularity, community quality, and genre exploration", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#watched-count")).toHaveText("0");
  await expect(page.locator("#rec-engine-status")).toContainText("catalog popularity proxy");
  await expect(page.locator("#rec-results .rec-title").first()).toHaveText("Copper Comet");
  await expect(page.locator("#rec-results .rec-score").first()).toHaveText("4 sampled ratings");
  await expect(page.locator("#rec-summary")).toContainText("loaded recommendation graph");
  const storageBefore = await page.evaluate(() => localStorage.getItem("wasiw.demo.recommendationState.v5"));

  await page.locator("#discovery-view").selectOption("quality");
  await expect(page.locator("#rec-engine-status")).toContainText("community-score exploration");
  await expect(page.locator("#rec-results .rec-title").first()).toHaveText("星の航路");
  await expect(page.locator("#rec-results .rec-score").first()).toHaveText("8.30 / 10");
  await page.locator("#filter-genre").selectOption("fantasy");
  await expect(page.locator("#rec-results .rec-title")).toHaveText([
    "星の航路", "Moonlit Workshop", "Glass Orchard",
  ]);
  await expect(page.locator("#watched-count")).toHaveText("0");
  expect(await page.evaluate(() => localStorage.getItem("wasiw.demo.recommendationState.v5"))).toBe(storageBefore);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await expect(page.locator("#rec-results .rec-item").first()).toBeVisible();
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await page.locator("#exclude-input").fill("Glass Orchard");
  await page.locator("#add-exclude-form button").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["星の航路", "Moonlit Workshop"]);
});

test("sparse Liked history can browse exact genre overlap without treating Seen as a seed", async ({ page }) => {
  await page.goto("/");
  for (const title of ["Copper Comet", "Moonlit Workshop"] as const) {
    await page.locator("#anime-input").fill(title);
    await page.locator("#add-preference").selectOption("liked");
    await page.locator("#add-anime-form button").click();
  }
  await page.locator("#anime-input").fill("Glass Orchard");
  await page.locator("#add-preference").selectOption("seen");
  await page.locator("#add-anime-form button").click();
  await page.locator("#discovery-view").selectOption("related");
  await expect(page.locator("#rec-engine-status")).toContainText("shared-genre content baseline");
  await expect(page.locator("#rec-results .rec-title").first()).toHaveText("星の航路");
  await expect(page.locator("#rec-results .rec-why").first())
    .toContainText("Adventure, Fantasy with Liked title(s) Copper Comet, Moonlit Workshop");
  for (const title of ["Copper Comet", "Moonlit Workshop", "Glass Orchard"]) {
    await expect(page.locator("#rec-results .rec-title").filter({ hasText: title })).toHaveCount(0);
  }
  await page.locator("#discovery-view").selectOption("auto");
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph recommendations");
});

test("a sparse Liked title without graph neighbors falls back to supported genre overlap", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Quiet Satellite");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-engine-status")).toContainText("shared-genre content baseline");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Copper Comet", "Paper Current"]);
  await expect(page.locator("#rec-results .rec-why").first()).toContainText("Sci-Fi with Liked title(s) Quiet Satellite");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("seen");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Paper Current"]);
});

test("three sparse Liked titles still receive a content fallback", async ({ page }) => {
  await page.goto("/");
  for (const title of ["Café Nebula", "Paper Current", "Quiet Satellite"]) {
    await page.locator("#anime-input").fill(title);
    await page.locator("#add-preference").selectOption("liked");
    await page.locator("#add-anime-form button").click();
  }
  await expect(page.locator("#rec-engine-status")).toContainText("shared-genre content baseline");
  await expect(page.locator("#rec-results .rec-title")).toHaveText([
    "Copper Comet", "Moonlit Workshop", "Ashen Harbor",
  ]);
});

test("normal-mode catalog metadata is fetched only after a deliberate exploration action", async ({ page }) => {
  const metadataRequests = await mockNormalModeCatalog(page);

  await page.goto("http://127.0.0.1:5174/");
  await expect(page.locator("#rec-engine-status")).toContainText("catalog popularity proxy");
  expect(metadataRequests()).toBe(0);
  await page.locator("#discovery-view").selectOption("related");
  await expect(page.locator("#rec-summary")).toContainText("Mark a title Liked");
  expect(metadataRequests()).toBe(0);
  await page.locator("#discovery-view").selectOption("popularity");
  await page.locator("#discovery-load-metadata").click();
  await expect.poll(metadataRequests).toBe(8);
  await expect(page.locator("#filter-genre option[value='fantasy']")).toHaveCount(1);
  await page.locator("#discovery-view").selectOption("quality");
  await expect(page.locator("#rec-results .rec-title").first()).toHaveText("星の航路");
  expect(metadataRequests()).toBe(8);
});

test("normal-mode sparse Liked title checks a bounded sample for a content fallback", async ({ page }) => {
  const metadataRequests = await mockNormalModeCatalog(page);
  await page.goto("http://127.0.0.1:5174/");
  await expect(page.locator("#rec-engine-status")).toContainText("catalog popularity proxy");
  expect(metadataRequests()).toBe(0);
  await page.locator("#anime-input").fill("Quiet Satellite");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-engine-status")).toContainText("shared-genre content baseline");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Copper Comet", "Paper Current"]);
  expect(metadataRequests()).toBe(8);
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("seen");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Paper Current"]);
  expect(metadataRequests()).toBe(8);
});
