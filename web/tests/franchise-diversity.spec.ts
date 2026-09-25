import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

async function likeCopper(page: import("@playwright/test").Page): Promise<void> {
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
}

test("demo prefers known franchise variety and saves the allow-related option with profiles", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#allow-related-titles")).not.toBeChecked();
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Quiet Satellite" })).toHaveCount(0);
  await page.locator("#discovery-view").selectOption("quality");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Quiet Satellite" })).toHaveCount(0);
  await page.locator("#allow-related-titles").check();
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Quiet Satellite" })).toHaveCount(1);
  await page.locator("#discovery-view").selectOption("auto");
  await page.locator("#allow-related-titles").uncheck();
  await likeCopper(page);
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-results .rec-title")).toHaveCount(6);
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Quiet Satellite" })).toHaveCount(0);
  await expect(page.locator("#rec-summary")).toContainText("1 repeated known/title-suggested series entry");
  await expect(page.locator("#rec-results .rec-relationship").first()).toContainText(
    "Prerequisites unverified");

  await page.locator("#allow-related-titles").check();
  await expect(page.locator("#rec-results .rec-title")).toHaveText([
    "星の航路", "Moonlit Workshop", "Quiet Satellite", "Café Nebula",
    "Glass Orchard", "Paper Current", "Ashen Harbor",
  ]);
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Allow related fixture");
  await page.locator("#profile-save-submit").click();
  await page.locator("#allow-related-titles").uncheck();
  await expect(page.locator("#rec-results .rec-title")).toHaveCount(6);
  await page.locator("#profile-select").selectOption("Allow related fixture");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#allow-related-titles")).toBeChecked();
  await expect(page.locator("#rec-results .rec-title")).toHaveCount(7);
  await page.reload();
  await expect(page.locator("#allow-related-titles")).toBeChecked();
  await expect(page.locator("#rec-results .rec-title")).toHaveCount(7);
});

test("mocked normal metadata withholds a known unwatched sequel without extra provider reads", async ({ page }) => {
  const graph = JSON.parse(readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8"));
  const unsafeTitle = '<img src=x onerror="window.__relationUnsafe=1">';
  let metadataRequests = 0;
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/data/graph.compact.json.gz") return route.fulfill({ status: 404, body: "" });
    if (url.pathname === "/data/graph.compact.json") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(graph) });
    }
    if (url.pathname.startsWith("/data/")) return route.fulfill({ status: 404, body: "" });
    const match = /^\/v4\/anime\/(\d+)\/full$/.exec(url.pathname);
    if (url.hostname === "api.jikan.moe" && match) {
      metadataRequests += 1;
      const animeId = Number(match[1]);
      const relations = animeId === 105
        ? [{ relation: "Prequel", entry: [{ mal_id: 102, type: "anime", name: unsafeTitle }] }]
        : [];
      return route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        body: JSON.stringify({ data: { year: 2023, score: 8, genres: [], relations } }) });
    }
    if (url.hostname === "api.jikan.moe") return route.abort();
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  await page.goto("http://127.0.0.1:5174/");
  await likeCopper(page);
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);
  await expect(page.locator("#rec-summary")).toContainText("1 known-unwatched sequel");
  expect(metadataRequests).toBe(2);
  await page.locator("#allow-related-titles").check();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop", "星の航路"]);
  await expect(page.locator("#rec-results .rec-relationship").nth(1)).toContainText(unsafeTitle);
  await expect(page.locator("#rec-results .rec-relationship img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __relationUnsafe?: number }).__relationUnsafe))
    .toBeUndefined();
  expect(metadataRequests).toBe(2);
});
