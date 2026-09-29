import { expect, test } from "@playwright/test";

test("first-use favorites path selects an explicit like and keeps manual Seen as the default", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "How would you like to find your next anime?" })).toBeVisible();
  await expect(page.locator("#add-preference")).toHaveValue("seen");
  await expect(page.locator("#advanced-recommendation-settings")).not.toHaveAttribute("open", "");

  await page.getByRole("button", { name: /Add favorites/ }).click();
  await expect(page.locator("#anime-input")).toBeFocused();
  await expect(page.locator("#add-preference")).toHaveValue("liked");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']"))
    .toHaveValue("liked");
  await expect(page.locator("#rec-results .rec-title").first()).toHaveText("Moonlit Workshop");
  await expect(page.locator("#add-preference")).toHaveValue("liked");
  await page.locator("#anime-input").fill("Glass Orchard");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:104']"))
    .toHaveValue("liked");
  await page.getByRole("button", { name: /Browse without a list/ }).click();
  await expect(page.locator("#add-preference")).toHaveValue("seen");
});

test("first-use import path opens a local preview without applying until requested", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Import a list/ }).click();
  await expect(page.locator("#import-profiles")).toHaveAttribute("open", "");
  await expect(page.locator("#bulk-import-input")).toBeFocused();
  await expect(page.locator("#bulk-import-file")).toBeVisible();
  await page.locator("#bulk-import-input").fill("101, 9, Completed, 12");
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-preview")).toBeVisible();
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']"))
    .toHaveValue("liked");
});

test("first-use browsing does not add history and advanced ranking controls remain reachable", async ({ page }) => {
  await page.goto("/");
  const stateBefore = await page.evaluate(() => localStorage.getItem("wasiw.demo.recommendationState.v5"));
  await page.getByRole("button", { name: /Browse without a list/ }).click();
  await expect(page.locator("#discovery-view")).toHaveValue("popularity");
  await expect(page.locator("#rec-engine-status")).toContainText("catalog popularity proxy");
  await expect(page.locator("#rec-results .rec-item").first()).toBeVisible();
  await expect(page.locator("#watched-count")).toHaveText("0");
  expect(await page.evaluate(() => localStorage.getItem("wasiw.demo.recommendationState.v5")))
    .toBe(stateBefore);

  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#rec-method").selectOption("hybrid");
  await expect(page.locator("#rec-blend-control")).toBeVisible();
  await page.locator("#allow-related-titles").check();
  await expect(page.locator("#allow-related-titles")).toBeChecked();
});

test("first-use choices remain usable at a narrow viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator("#quickstart-favorites")).toBeVisible();
  await expect(page.locator("#quickstart-import")).toBeVisible();
  await expect(page.locator("#quickstart-browse")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("optional help tips stay available without covering the first-use choices", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#tips-recommendations")).toBeHidden();
  await page.locator("#tips-toggle").click();
  await expect(page.locator("#tips-recommendations")).toBeVisible();
  await page.locator("#tips-dismiss-recommendations").click();
  await expect(page.locator("#tips-recommendations")).toBeHidden();
  await page.reload();
  await expect(page.locator("#tips-recommendations")).toBeHidden();
});
