import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

const unsafeMarkup = '<img src=x onerror="window.__unsafe=1">';
const unsafeImportLine = '<svg onload="window.__unsafe=3">';

test("catalog metadata stays text and unsafe recommendation images are omitted", async ({ page }) => {
  await page.route("https://images.example.test/**", (route) => route.abort());
  await page.route("**/demo-data/catalog.json", async (route) => {
    const response = await route.fetch();
    const catalog = await response.json();
    const recommended = catalog.anime.find((item: { animeId: number }) => item.animeId === 102);
    recommended.synopsis = unsafeMarkup;
    recommended.genres = [unsafeMarkup];
    recommended.studios = [unsafeMarkup];
    recommended.imageUrl = "javascript:window.__unsafe=2";
    catalog.anime.find((item: { animeId: number }) => item.animeId === 105).imageUrl =
      "https://images.example.test/cover.png?x=1&y=2";
    await route.fulfill({ response, json: catalog });
  });

  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  const card = page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(card.locator(".rec-synopsis")).toContainText(unsafeMarkup);
  await expect(card.locator(".rec-meta-details")).toContainText(unsafeMarkup);
  await expect(card.locator(".rec-cover-placeholder")).toHaveCount(1);
  await expect(card.locator("img")).toHaveCount(0);
  await expect(page.locator("#rec-results .rec-item").filter({ hasText: "星の航路" }).locator("img"))
    .toHaveAttribute("src", "https://images.example.test/cover.png?x=1&y=2");
  await expect(page.locator("#rec-results .rec-item").filter({ hasText: "星の航路" }).locator("img"))
    .toHaveAttribute("referrerpolicy", "no-referrer");
  expect(await page.evaluate(() => (window as unknown as { __unsafe?: number }).__unsafe)).toBeUndefined();
});

test("mocked seasonal provider text is literal and unsafe images are omitted", async ({ page }) => {
  const graphJson = readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8");
  const unsafeUrls = ["data:image/svg+xml,<svg onload=alert(1)>", "javascript:alert(1)"];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.hostname === "api.jikan.moe") {
      if (url.pathname === "/v4/seasons/now") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          headers: { "Access-Control-Allow-Origin": "*" },
          json: { data: unsafeUrls.map((imageUrl, index) => ({
            mal_id: 101 + index,
            title: `${unsafeMarkup} ${index}`,
            season: unsafeMarkup,
            images: { jpg: { image_url: imageUrl } },
          })) },
        });
      }
      return route.fulfill({ status: 404, headers: { "Access-Control-Allow-Origin": "*" }, body: "{}" });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    if (url.pathname === "/data/graph.compact.json.gz") return route.fulfill({ status: 404, body: "" });
    if (url.pathname === "/data/graph.compact.json") {
      return route.fulfill({ status: 200, contentType: "application/json", body: graphJson });
    }
    return route.continue();
  });

  await page.goto("http://127.0.0.1:5174/");
  await expect(page.locator("#seasonal-list .seasonal-item")).toHaveCount(2);
  await expect(page.locator("#seasonal-list .seasonal-title").first()).toContainText(unsafeMarkup);
  await expect(page.locator("#seasonal-list .seasonal-meta").first()).toContainText(unsafeMarkup);
  await expect(page.locator("#seasonal-list .seasonal-cover-placeholder")).toHaveCount(2);
  await expect(page.locator("#seasonal-list img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __unsafe?: number }).__unsafe)).toBeUndefined();
});

test("graph labels, imported lines, and profile names remain text", async ({ page }) => {
  await page.route("**/demo-data/graph.compact.json", async (route) => {
    const response = await route.fetch();
    const graph = await response.json();
    graph.anime[0][1] = unsafeMarkup;
    await route.fulfill({ response, json: graph });
  });
  await page.goto("/");
  await page.locator("#anime-input").fill("101");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#selected-anime .chip-title")).toContainText(unsafeMarkup);

  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#bulk-import-input").fill(unsafeImportLine);
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-unmapped")).toContainText(unsafeImportLine);
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#history-list")).toContainText(unsafeImportLine);
  await page.locator("#profile-name-input").fill(unsafeMarkup);
  await page.locator("#profile-save-submit").click();
  await expect(page.locator("#profile-select option")).toHaveText(unsafeMarkup);

  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#graph svg text").filter({ hasText: unsafeMarkup })).toHaveCount(1);
  await page.locator("#graph svg [data-node-id='anime:101']").click();
  await expect(page.locator("#inspect-meta .inspect-title")).toContainText(unsafeMarkup);
  await expect(page.locator("#app img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __unsafe?: number }).__unsafe)).toBeUndefined();
});
