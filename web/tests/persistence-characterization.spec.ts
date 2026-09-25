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
    state: JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v4") ?? "null"),
    profiles: JSON.parse(localStorage.getItem("wasiw.demo.recommendationProfiles.v4") ?? "null"),
  }));
  expect(before.state).toMatchObject({
    version: 4, mode: "hybrid", modelBlendWeight: 0.35,
    selected: [{ nodeId: "anime:101", weight: 1.7 }],
    includeCandidates: ["anime:102"], excludeCandidates: ["anime:105"],
  });
  expect(before.profiles.version).toBe(4);
  expect(before.profiles.profiles).toHaveLength(1);
  expect(before.profiles.profiles[0].name).toBe("Fixture Profile");
  expect(before.profiles.profiles[0].state).toEqual(before.state);

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

test("legacy profiles keep catalog-missing selections and overrides through load and save", async ({ page }) => {
  await page.goto("/");
  const legacy = {
    version: 3, mode: "hybrid", modelBlendWeight: 0.35,
    selected: [{ nodeId: "anime:101", weight: 1.7 }, { nodeId: "anime:999", weight: 2.4 }],
    includeCandidates: ["anime:998"], excludeCandidates: ["anime:997"],
  };
  await page.evaluate((state) => {
    localStorage.setItem("wasiw.demo.recommendationState.v1", JSON.stringify(state));
    localStorage.setItem("wasiw.demo.recommendationProfiles.v1", JSON.stringify([
      { name: "Fixture Profile", updatedAt: "2026-01-01T00:00:00Z", state },
    ]));
  }, legacy);
  await page.reload();

  await expect(page.locator("#watched-count")).toHaveText("2");
  await expect(page.locator("#selected-anime")).toContainText("anime:999");
  await expect(page.locator("#selected-anime")).toContainText("Unavailable in this catalog");
  await expect(page.locator("#selected-anime input[data-weight-node-id='anime:999']")).toHaveValue("2.4");
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await expect(page.locator("#include-anime")).toContainText("anime:998");
  await expect(page.locator("#exclude-anime")).toContainText("anime:997");
  const migrated = await page.evaluate(() => ({
    state: JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v4") ?? "null"),
    stateBackup: localStorage.getItem("wasiw.demo.recommendationState.v1.backup"),
    profileBackup: localStorage.getItem("wasiw.demo.recommendationProfiles.v1.backup"),
  }));
  expect(migrated.state).toEqual({ ...legacy, version: 4 });
  expect(migrated.stateBackup).toBe(JSON.stringify(legacy));
  expect(migrated.profileBackup).toContain("Fixture Profile");

  await page.locator("#clear-watched").click();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-select").selectOption("Fixture Profile");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#watched-count")).toHaveText("2");
  await page.locator("#profile-name-input").fill("Copied Profile");
  await page.locator("#profile-save-submit").click();
  const copied = await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationProfiles.v4") ?? "null"));
  expect(copied.profiles.find((profile: { name: string }) => profile.name === "Copied Profile").state)
    .toMatchObject({ ...legacy, version: 4 });
});

test("a rejected profile write reports failure instead of claiming it was saved", async ({ page }) => {
  await page.addInitScript(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key.endsWith(".recommendationProfiles.v4")) {
        throw new DOMException("Synthetic quota exceeded", "QuotaExceededError");
      }
      return original.call(this, key, value);
    };
  });
  await page.goto("/");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Fixture Profile");
  await page.locator("#profile-save-submit").click();
  await expect(page.locator("#rec-message")).toContainText("Could not save profile");
  await expect(page.locator("#storage-status")).toContainText("storage rejected changes");
  await expect(page.locator("#profile-select")).toBeDisabled();
});

test("a corrupt current copy restores its backup and shows a durable warning", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => {
    localStorage.setItem("wasiw.demo.recommendationState.v4", "{broken");
    localStorage.setItem("wasiw.demo.recommendationState.v4.backup", JSON.stringify({
      version: 4, mode: "graph", selected: [{ nodeId: "anime:999", weight: 2.4 }],
      modelBlendWeight: 0.5, includeCandidates: [], excludeCandidates: [],
    }));
  });
  await page.reload();
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#selected-anime")).toContainText("anime:999");
  await expect(page.locator("#storage-status")).toContainText("Recovered recommendation state from a backup");
  expect(await page.evaluate(() => localStorage.getItem("wasiw.demo.recommendationState.v4"))).toBe("{broken");
});
