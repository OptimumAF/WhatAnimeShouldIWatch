import { expect, test } from "@playwright/test";

test("seen is exclusion only; explicit likes and dislikes change the active engines", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']"))
    .toHaveValue("seen");
  await expect(page.locator("#rec-results .rec-item")).toHaveCount(0);
  await expect(page.locator("#rec-engine-status")).toContainText("waiting for a Liked title");
  await page.locator("#selected-anime select[data-preference-node-id='anime:101']")
    .selectOption("liked");
  await expect(page.locator("#rec-results .rec-title").first()).toContainText("Moonlit Workshop");
  await expect(page.locator("#selected-anime .chip-confidence")).toContainText("100% confidence");
  await page.locator("#selected-anime select[data-preference-node-id='anime:101']")
    .selectOption("disliked");
  await expect(page.locator("#rec-results .rec-item")).toHaveCount(0);
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-results .rec-item").first()).toBeVisible();
  expect(await page.locator("#rec-results .rec-title").allTextContents()).not.toContain("Copper Comet");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null")
    ?.preferences[0])).toEqual({ nodeId: "anime:101", sentiment: "disliked", importance: 1,
    confidence: 1, source: "manual" });
});

test("local import preserves low, unscored, and high preferences and honors manual overrides", async ({ page }) => {
  await page.goto("/");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#bulk-import-input").fill("101, 2, Completed, 12\n102, 0, Watching, 3\n103, 9, Completed, 12");
  await page.locator("#bulk-import-form button").click();
  await page.locator("#history-import-apply").click();
  const preference = (id: number) => page.locator(`#selected-anime select[data-preference-node-id='anime:${id}']`);
  await expect(preference(101)).toHaveValue("disliked");
  await expect(preference(102)).toHaveValue("seen");
  await expect(preference(103)).toHaveValue("liked");
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-results .rec-item").first()).toBeVisible();
  const titles = await page.locator("#rec-results .rec-title").allTextContents();
  expect(titles).not.toContain("Copper Comet");
  expect(titles).not.toContain("Moonlit Workshop");
  expect(titles).not.toContain("Ashen Harbor");

  await preference(101).selectOption("liked");
  await page.locator("#bulk-import-input").fill("101, 1, Completed, 12");
  await page.locator("#bulk-import-form button").click();
  await page.locator("#history-import-apply").click();
  await expect(preference(101)).toHaveValue("liked");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null")
    ?.preferences.find((item: { nodeId: string }) => item.nodeId === "anime:101"))).toMatchObject({ source: "manual" });
});

test("planned status clears an imported preference, and browsing seasonal ideas does not add watches", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#seasonal-status")).toHaveAttribute("data-state", "demo");
  await page.locator("#quickstart-seasonal").click();
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#bulk-import-input").fill("101, 8, Completed, 12");
  await page.locator("#bulk-import-form button").click();
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#watched-count")).toHaveText("1");
  await page.locator("#bulk-import-input").fill("101, 0, Plan to Watch, 0");
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-summary")).toContainText("cleared by planned status: 1");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#watched-count")).toHaveText("0");
});
