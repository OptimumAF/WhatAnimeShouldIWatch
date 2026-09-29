import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";

test("save a recommendation, change every watch status, rate it, and restore a named profile", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  const result = page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(result).toBeVisible();
  await result.locator("button[data-watchlist-save-id]").click();
  await expect(page.locator("#watchlist-count")).toHaveText("1");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);
  const status = page.locator("#watchlist-list select[data-watchlist-status-id='102']");
  const rating = page.locator("#watchlist-list select[data-watchlist-rating-id='102']");
  for (const value of ["watching", "completed", "on_hold", "dropped", "plan_to_watch"]) {
    await status.selectOption(value);
    await expect(status).toHaveValue(value);
  }
  await status.selectOption("completed");
  await rating.selectOption("9");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Invented shortlist");
  await page.locator("#profile-save-submit").click();
  await status.selectOption("dropped");
  await rating.selectOption("2");
  await page.locator("#profile-select").selectOption("Invented shortlist");
  await page.locator("#profile-load-btn").click();
  await expect(status).toHaveValue("completed");
  await expect(rating).toHaveValue("9");
  await page.reload();
  await expect(status).toHaveValue("completed");
  await expect(rating).toHaveValue("9");
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null"));
  expect(stored.watchlist).toEqual([{ animeId: 102, title: "Moonlit Workshop", status: "completed", rating: 9 }]);
});

test("status alone supplies no preference; explicit rating affects only local ranking", async ({ page }) => {
  await page.goto("/");
  await page.locator("#watchlist-input").fill("Copper Comet");
  await page.locator("#watchlist-form button").click();
  await expect(page.locator("#watchlist-count")).toHaveText("1");
  const status = page.locator("select[data-watchlist-status-id='101']");
  const rating = page.locator("select[data-watchlist-rating-id='101']");
  await rating.selectOption("9");
  await expect(page.locator("#rec-engine-status")).not.toContainText("Using graph recommendations");
  await status.selectOption("watching");
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph recommendations");
  await expect(page.locator("#rec-results .rec-title").first()).toContainText("Moonlit Workshop");
  await rating.selectOption("");
  await expect(page.locator("#rec-engine-status")).not.toContainText("Using graph recommendations");
  await rating.selectOption("2");
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations");
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null"));
  expect(stored.preferences).toEqual([]);
  expect(stored.watchlist[0]).toMatchObject({ status: "watching", rating: 2 });
});

test("watchlist input refuses partial names; unknown backup titles render as text", async ({ page }) => {
  await page.goto("/");
  await page.locator("#watchlist-input").fill("Moonlit");
  await page.locator("#watchlist-form button").click();
  await expect(page.locator("#watchlist-status")).toContainText("exact catalog title");
  await expect(page.locator("#watchlist-count")).toHaveText("0");
  await page.evaluate(() => {
    const state = { version: 5, mode: "graph", preferences: [], watchlist: [
      { animeId: 99999, title: "<img src=x onerror=alert(1)>", status: "on_hold", rating: 4 },
    ] };
    localStorage.setItem("wasiw.demo.recommendationState.v5", JSON.stringify(state));
  });
  await page.reload();
  await expect(page.locator("#watchlist-list")).toContainText("<img src=x onerror=alert(1)>");
  await expect(page.locator("#watchlist-list img")).toHaveCount(0);
  await expect(page.locator("#watchlist-list")).toContainText("Unavailable in this catalog");
});

test("rejected browser storage rolls back a saved recommendation and reports failure", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" })).toBeVisible();
  await page.evaluate(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === "wasiw.demo.recommendationState.v5") {
        throw new DOMException("Invented quota rejection", "QuotaExceededError");
      }
      return original.call(this, key, value);
    };
  });
  await page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" })
    .locator("button[data-watchlist-save-id]").click();
  await expect(page.locator("#watchlist-count")).toHaveText("0");
  await expect(page.locator("#watchlist-status")).toContainText("storage rejected");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toBeVisible();
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null").watchlist ?? []))
    .toEqual([]);
});

test("v2 backup retains unknown rated watchlist entries through reset and replace", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => {
    localStorage.setItem("wasiw.demo.recommendationState.v5", JSON.stringify({
      version: 5, mode: "graph", preferences: [], watchlist: [
        { animeId: 99999, title: "Invented Missing", status: "dropped", rating: 3 },
      ],
    }));
  });
  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Missing title");
  await page.locator("#profile-save-submit").click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#profile-export-btn").click();
  const raw = await readFile(await (await downloadPromise).path(), "utf8");
  const backup = JSON.parse(raw);
  expect(backup.version).toBe(2);
  expect(backup.state.watchlist[0]).toMatchObject({ animeId: 99999, status: "dropped", rating: 3 });
  expect(backup.profiles[0].state.watchlist[0].animeId).toBe(99999);
  await page.locator("#profile-reset-preview-btn").click();
  await expect(page.locator("#profile-backup-summary")).toContainText("1 watchlist titles");
  await page.locator("#profile-backup-apply").click();
  await expect(page.locator("#watchlist-count")).toHaveText("0");
  await page.locator("#profile-backup-file").setInputFiles({
    name: "invented-rated-backup.json", mimeType: "application/json", buffer: Buffer.from(raw),
  });
  await expect(page.locator("#profile-backup-summary")).toContainText("1 watchlist titles");
  await page.locator("#profile-backup-mode").selectOption("replace");
  await page.locator("#profile-backup-apply").click();
  await page.reload();
  await expect(page.locator("#watchlist-list")).toContainText("Invented Missing");
  await expect(page.locator("select[data-watchlist-status-id='99999']")).toHaveValue("dropped");
  await expect(page.locator("select[data-watchlist-rating-id='99999']")).toHaveValue("3");
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null").watchlist))
    .toEqual(backup.state.watchlist);
});
