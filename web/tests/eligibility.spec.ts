import { expect, test } from "@playwright/test";

test("Include Only intersects scored candidates and exclusions win in every engine", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await expect(page.getByText("Include Only narrows the current ranking to listed titles.")).toBeVisible();

  // Ashen Harbor has no positive graph pair from Copper Comet, so allowlisting it creates no score.
  await page.locator("#include-input").fill("Ashen Harbor");
  await page.locator("#add-include-form button").click();
  await expect(page.locator("#rec-results .rec-item")).toHaveCount(0);
  await expect(page.locator("#rec-summary")).toContainText("No scored candidates match Include Only");

  await page.locator("#include-input").fill("Moonlit Workshop");
  await page.locator("#add-include-form button").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);

  await page.locator("#exclude-input").fill("Moonlit Workshop");
  await page.locator("#add-exclude-form button").click();
  await expect(page.locator("#include-anime")).toContainText("Moonlit Workshop");
  await expect(page.locator("#exclude-anime")).toContainText("Moonlit Workshop");
  for (const mode of ["graph", "model", "hybrid"] as const) {
    await page.locator("#rec-method").selectOption(mode);
    await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);
  }

  await page.reload();
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await expect(page.locator("#include-anime")).toContainText("Moonlit Workshop");
  await expect(page.locator("#exclude-anime")).toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);

  await page.locator("#exclude-anime button[data-exclude-node-id='anime:102']").click();
  await page.locator("#anime-input").fill("Moonlit Workshop");
  await page.locator("#add-preference").selectOption("seen");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);
});

test("required content filters are applied to graph, model, and hybrid results", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#filter-genre").selectOption("slice of life");
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-max").fill("2020");
  await page.locator("#filter-min-score").evaluate((input) => {
    (input as HTMLInputElement).value = "7.7";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  for (const mode of ["graph", "model", "hybrid"] as const) {
    await page.locator("#rec-method").selectOption(mode);
    await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);
  }
  await page.locator("#filter-min-score").evaluate((input) => {
    (input as HTMLInputElement).value = "7.9";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#rec-results .rec-item")).toHaveCount(0);
});
