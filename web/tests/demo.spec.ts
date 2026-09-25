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

  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results")).toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Copper Comet" })).toHaveCount(0);
  await expect(page.locator("#metadata-status")).toContainText("2/2");
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
