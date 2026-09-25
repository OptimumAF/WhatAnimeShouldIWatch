import { expect, test } from "@playwright/test";

test("synthetic selection, overrides, mode, and named profile survive reload", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await page.locator("#selected-anime input[data-weight-node-id='anime:101']").evaluate((input) => {
    (input as HTMLInputElement).value = "1.7";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });

  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await page.locator("#include-input").fill("Moonlit Workshop");
  await page.locator("#add-include-form button").click();
  await page.locator("#exclude-input").fill("星の航路");
  await page.locator("#add-exclude-form button").click();
  await page.locator("#rec-method").selectOption("hybrid");
  await page.locator("#rec-blend").evaluate((input) => {
    (input as HTMLInputElement).value = "0.35";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });

  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Fixture Profile");
  await page.locator("#profile-save-submit").click();
  const before = await page.evaluate(() => ({
    state: JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v1") ?? "null"),
    profiles: JSON.parse(localStorage.getItem("wasiw.demo.recommendationProfiles.v1") ?? "null"),
  }));
  expect(before.state).toMatchObject({
    version: 3, mode: "hybrid", modelBlendWeight: 0.35,
    selected: [{ nodeId: "anime:101", weight: 1.7 }],
    includeCandidates: ["anime:102"], excludeCandidates: ["anime:105"],
  });
  expect(before.profiles).toHaveLength(1);
  expect(before.profiles[0].name).toBe("Fixture Profile");
  expect(before.profiles[0].state).toEqual(before.state);

  await page.reload();
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#rec-method")).toHaveValue("hybrid");
  await expect(page.locator("#rec-blend")).toHaveValue("0.35");
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await expect(page.locator("#include-anime")).toContainText("Moonlit Workshop");
  await expect(page.locator("#exclude-anime")).toContainText("星の航路");

  await page.locator("#clear-watched").click();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-select").selectOption("Fixture Profile");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#selected-anime .chip-weight-value")).toHaveText("1.7x");
});
