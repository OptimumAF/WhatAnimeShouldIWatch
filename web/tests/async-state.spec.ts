import { readFileSync } from "node:fs";
import { expect, test, type Page, type Route } from "@playwright/test";

const appUrl = "http://127.0.0.1:5174/";
const graphJson = readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8");

async function openMockedApp(page: Page, external: (route: Route, url: URL) => Promise<void> | void) {
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.hostname === "myanimelist.net" || url.hostname === "graphql.anilist.co" || url.hostname === "api.jikan.moe") {
      await external(route, url);
      return;
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
      await route.abort();
      return;
    }
    if (url.pathname === "/data/graph.compact.json.gz") {
      await route.fulfill({ status: 404, body: "" });
      return;
    }
    if (url.pathname === "/data/graph.compact.json") {
      await route.fulfill({ status: 200, contentType: "application/json", body: graphJson });
      return;
    }
    await route.continue();
  });
  await page.goto(appUrl);
  await expect(page.getByRole("heading", { name: "What Anime Should I Watch" })).toBeVisible();
}

function jsonRoute(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status, contentType: "application/json", headers: { "Access-Control-Allow-Origin": "*" },
    body: JSON.stringify(body),
  });
}

test("a delayed username import cannot overwrite a newer loaded profile", async ({ page }) => {
  let releaseImport: (() => void) | undefined;
  await openMockedApp(page, async (route, url) => {
    if (url.hostname === "myanimelist.net") {
      await new Promise<void>((resolve) => { releaseImport = resolve; });
      await jsonRoute(route, [{ anime_id: 102, score: 8 }]).catch(() => {});
      return;
    }
    await jsonRoute(route, { data: [] });
  });
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Copper Profile");
  await page.locator("#profile-save-submit").click();
  await page.locator("#clear-watched").click();

  await page.locator("#username-import-provider").selectOption("mal");
  await page.locator("#username-import-input").fill("fixture-user");
  await page.locator("#username-import-submit").click();
  await expect(page.locator("#username-import-submit")).toBeDisabled();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "loading");
  await page.locator("#profile-select").selectOption("Copper Profile");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "stale");
  releaseImport?.();
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#selected-anime")).toContainText("Copper Comet");
  await expect(page.locator("#selected-anime")).not.toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-message")).toContainText("Loaded profile");
  const preferences = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("wasiw.recommendationState.v5") ?? "null")?.preferences);
  expect(preferences).toEqual([{ nodeId: "anime:101", sentiment: "seen",
    importance: 1, confidence: 0, source: "manual" }]);
});

test("canceling during provider backoff prevents a later retry", async ({ page }) => {
  let malRequests = 0;
  await openMockedApp(page, async (route, url) => {
    if (url.hostname === "myanimelist.net") {
      malRequests += 1;
      await jsonRoute(route, {}, 429);
      return;
    }
    await jsonRoute(route, { data: [] });
  });
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Empty Profile");
  await page.locator("#profile-save-submit").click();
  await page.locator("#username-import-provider").selectOption("mal");
  await page.locator("#username-import-input").fill("fixture-user");
  await page.locator("#username-import-submit").click();
  await expect.poll(() => malRequests).toBe(1);
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "stale");
  await page.waitForTimeout(1400);
  expect(malRequests).toBe(1);
});

test("superseded metadata cannot enter the cache or replace current status", async ({ page }) => {
  let releaseFirst: (() => void) | undefined;
  let metadataRequests = 0;
  await openMockedApp(page, async (route, url) => {
    if (url.pathname === "/v4/anime/102/full") {
      metadataRequests += 1;
      if (metadataRequests === 1) {
        await new Promise<void>((resolve) => { releaseFirst = resolve; });
      }
      await jsonRoute(route, { data: { year: 2026, score: 8, synopsis: "Synthetic metadata", genres: [] } }).catch(() => {});
      return;
    }
    await jsonRoute(route, { data: [] });
  });
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect.poll(() => metadataRequests).toBeGreaterThan(0);
  await expect(page.locator("#metadata-status")).toHaveAttribute("data-state", "loading");
  await page.locator("#clear-watched").click();
  await expect(page.locator("#metadata-status")).toHaveAttribute("data-state", "empty");
  releaseFirst?.();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect.poll(() => metadataRequests).toBeGreaterThan(1);
  await expect(page.locator("#metadata-status")).toHaveAttribute("data-state", "ready");
});

for (const [responseStatus, expectedState] of [[404, "unavailable"], [400, "failed"]] as const) {
  test(`metadata response ${responseStatus} is ${expectedState}`, async ({ page }) => {
    await openMockedApp(page, async (route, url) => {
      await jsonRoute(route, url.pathname.startsWith("/v4/anime/") ? {} : { data: [] },
        url.pathname.startsWith("/v4/anime/") ? responseStatus : 200);
    });
    await page.locator("#anime-input").fill("Copper Comet");
    await page.locator("#add-preference").selectOption("liked");
    await page.locator("#add-anime-form button").click();
    await expect(page.locator("#metadata-status")).toHaveAttribute("data-state", expectedState);
  });
}

test("seasonal status separates empty, unavailable, failed, and ready responses", async ({ page }) => {
  let seasonStatus = 200;
  let seasonItems: unknown[] = [];
  await openMockedApp(page, async (route, url) => {
    if (url.pathname === "/v4/seasons/now") {
      await jsonRoute(route, { data: seasonItems }, seasonStatus);
    } else {
      await jsonRoute(route, {}, 404);
    }
  });
  await expect(page.locator("#seasonal-status")).toHaveAttribute("data-state", "empty");
  seasonStatus = 404;
  await page.locator("#refresh-seasonal").click();
  await expect(page.locator("#seasonal-status")).toHaveAttribute("data-state", "unavailable");
  seasonStatus = 400;
  await page.locator("#refresh-seasonal").click();
  await expect(page.locator("#seasonal-status")).toHaveAttribute("data-state", "failed");
  seasonStatus = 200;
  seasonItems = [{ mal_id: 102, title: "Moonlit Workshop", year: 2026, score: 8 }];
  await page.locator("#refresh-seasonal").click();
  await expect(page.locator("#seasonal-status")).toHaveAttribute("data-state", "ready");
  await expect(page.locator("#seasonal-list")).toContainText("Moonlit Workshop");
});

test("a delayed local file read is superseded by profile load and a newer file", async ({ page }) => {
  await page.addInitScript(() => {
    const pending = new Map<string, (content: string) => void>();
    (window as unknown as { releaseSyntheticFile: (name: string, content: string) => void }).releaseSyntheticFile =
      (name, content) => { pending.get(name)?.(content); pending.delete(name); };
    File.prototype.text = function () {
      return new Promise<string>((resolve) => { pending.set(this.name, resolve); });
    };
  });
  await page.goto("/");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Fixture Profile");
  await page.locator("#profile-save-submit").click();
  await page.locator("#bulk-import-file").setInputFiles({
    name: "old.txt", mimeType: "text/plain", buffer: Buffer.from("101, 9"),
  });
  await expect(page.locator("#bulk-import-status")).toHaveAttribute("data-state", "loading");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#bulk-import-status")).toHaveAttribute("data-state", "stale");
  await page.evaluate(() => (window as unknown as { releaseSyntheticFile: (name: string, content: string) => void })
    .releaseSyntheticFile("old.txt", "101, 9"));
  await expect(page.locator("#bulk-import-input")).toHaveValue("");

  await page.locator("#bulk-import-file").setInputFiles({
    name: "older.txt", mimeType: "text/plain", buffer: Buffer.from("101, 9"),
  });
  await page.locator("#bulk-import-file").setInputFiles({
    name: "newer.txt", mimeType: "text/plain", buffer: Buffer.from("102, 8"),
  });
  await page.evaluate(() => (window as unknown as { releaseSyntheticFile: (name: string, content: string) => void })
    .releaseSyntheticFile("newer.txt", "102, 8"));
  await expect(page.locator("#bulk-import-input")).toHaveValue("102, 8");
  await page.evaluate(() => (window as unknown as { releaseSyntheticFile: (name: string, content: string) => void })
    .releaseSyntheticFile("older.txt", "101, 9"));
  await expect(page.locator("#bulk-import-input")).toHaveValue("102, 8");
});
