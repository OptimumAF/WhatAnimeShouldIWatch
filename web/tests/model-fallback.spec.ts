import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

function modelFixture(): any {
  return JSON.parse(readFileSync(new URL("../public/demo-data/model-mf-web.compact.json", import.meta.url), "utf8"));
}

function removeModelAnime(model: any, animeId: number): void {
  const index = model.animeIds.indexOf(animeId);
  for (const field of ["animeIds", "titles", "biases", "embeddings"] as const) {
    model[field].splice(index, 1);
  }
  model.animeCount = model.animeIds.length;
}

async function addPreference(page: import("@playwright/test").Page, title: string, sentiment: string): Promise<void> {
  await page.locator("#anime-input").fill(title);
  await page.locator("#add-preference").selectOption(sentiment);
  await page.locator("#add-anime-form button").click();
}

for (const variant of ["absent", "corrupt", "incompatible", "unmapped seed"] as const) {
  test(`${variant} model falls back to eligible graph results without changing the requested mode`, async ({ page }) => {
    const model = modelFixture();
    if (variant === "incompatible") model.format = "model-mf-compact-v9";
    if (variant === "unmapped seed") removeModelAnime(model, 101);
    await page.route("**/demo-data/model-mf-web.compact.json*", (route) => {
      if (route.request().url().endsWith(".gz") || variant === "absent") {
        return route.fulfill({ status: 404, body: "" });
      }
      return route.fulfill({ contentType: "application/json",
        body: variant === "corrupt" ? "{broken" : JSON.stringify(model) });
    });
    await page.goto("/");
    await addPreference(page, "Copper Comet", "liked");
    await page.locator("#rec-method").selectOption("hybrid");
    await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
    await expect(page.locator("#rec-method")).toHaveValue("hybrid");
    await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop", "星の航路"]);
    const expectedDetail = {
      absent: "ML model data not found",
      corrupt: "synthetic demo model: invalid JSON",
      incompatible: "model-mf-compact-v1",
      "unmapped seed": "ML model maps no selected preference signals",
    }[variant];
    await expect(page.locator("#rec-engine-status")).toContainText(expectedDetail);
  });
}

test("a valid but partial model names its mapped preference coverage", async ({ page }) => {
  const model = modelFixture();
  removeModelAnime(model, 103);
  await page.route("**/demo-data/model-mf-web.compact.json*", (route) =>
    route.request().url().endsWith(".gz")
      ? route.fulfill({ status: 404, body: "" })
      : route.fulfill({ contentType: "application/json", body: JSON.stringify(model) }));
  await page.goto("/");
  await addPreference(page, "Copper Comet", "liked");
  await addPreference(page, "Ashen Harbor", "disliked");
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations");
  await expect(page.locator("#rec-engine-status")).toContainText("model mapped 1/2 preference signals");
  await expect(page.locator("#rec-results .rec-item").first()).toBeVisible();
});

test("without graph candidates, missing model uses an eligible catalog coverage baseline", async ({ page }) => {
  await page.route("**/demo-data/model-mf-web.compact.json*", (route) =>
    route.fulfill({ status: 404, body: "" }));
  await page.goto("/");
  await addPreference(page, "Copper Comet", "disliked");
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using catalog coverage baseline");
  await expect(page.locator("#rec-summary")).toContainText("catalog coverage baseline");
  await expect(page.locator("#rec-results .rec-item").first()).toBeVisible();
  await expect(page.locator("#rec-results .rec-why").first()).toContainText("Catalog coverage:");
  await expect(page.locator("#rec-results .rec-score").first()).toContainText("connections");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Copper Comet" })).toHaveCount(0);
  await expect(page.locator("#rec-method")).toHaveValue("model");

  await page.locator("#filter-genre").selectOption("slice of life");
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await page.locator("#include-input").fill("Moonlit Workshop");
  await page.locator("#add-include-form button").click();
  await page.locator("#exclude-input").fill("Moonlit Workshop");
  await page.locator("#add-exclude-form button").click();
  await expect(page.locator("#rec-results .rec-item")).toHaveCount(0);
});
