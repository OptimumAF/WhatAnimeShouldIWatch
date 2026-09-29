import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

test("ambiguous alias presents both invented titles with year and format before a preference is saved", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Galaxy Route");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  const dialog = page.locator("#title-search-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".title-search-choice")).toHaveCount(2);
  await expect(dialog.locator(".title-search-choice").first()).toContainText("Copper Comet");
  await expect(dialog.locator(".title-search-choice").first()).toContainText("2021 · TV");
  await expect(dialog.locator(".title-search-choice").nth(1)).toContainText("Glass Orchard");
  await expect(dialog.locator(".title-search-choice").nth(1)).toContainText("2022 · OVA");
  await expect(page.locator("#watched-count")).toHaveText("0");
  await dialog.locator(".title-search-choice").nth(1).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#selected-anime")).toContainText("Glass Orchard");
  await expect(page.locator("#selected-anime")).not.toContainText("Copper Comet");
});

test("partial title requires confirmation; cancel leaves state unchanged", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper");
  await page.locator("#add-anime-form button").click();
  const dialog = page.locator("#title-search-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".title-search-choice")).toHaveCount(1);
  await dialog.locator("#title-search-close").click();
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#add-anime-form button").click();
  await dialog.locator(".title-search-choice").click();
  await expect(page.locator("#selected-anime")).toContainText("Copper Comet");
});

test("watchlist and candidate overrides use the same explicit alias and partial choices", async ({ page }) => {
  await page.goto("/");
  await page.locator("#watchlist-input").fill("Moonlit Studio");
  await page.locator("#watchlist-form button").click();
  await expect(page.locator("#watchlist-list")).toContainText("Moonlit Workshop");
  await page.locator("#watchlist-input").fill("Glass");
  await page.locator("#watchlist-form button").click();
  await expect(page.locator("#title-search-dialog")).toBeVisible();
  await expect(page.locator("#watchlist-count")).toHaveText("1");
  await page.locator("#title-search-dialog .title-search-choice").click();
  await expect(page.locator("#watchlist-count")).toHaveText("2");

  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await page.locator("#include-input").fill("Galaxy Route");
  await page.locator("#add-include-form button").click();
  await page.locator("#title-search-dialog .title-search-choice").first().click();
  await expect(page.locator("#include-anime")).toContainText("Copper Comet");
  await page.locator("#exclude-input").fill("Paper");
  await page.locator("#add-exclude-form button").click();
  await expect(page.locator("#title-search-dialog")).toBeVisible();
  await page.locator("#title-search-dialog .title-search-choice").click();
  await expect(page.locator("#exclude-anime")).toContainText("Paper Current");
});

test("disambiguation treats alias text as text and permits only safe already-known covers", async ({ page }) => {
  const unsafe = '<img src=x onerror="window.__titleUnsafe=1">';
  await page.route("https://covers.example.invalid/**", (route) => route.abort());
  await page.route("**/demo-data/catalog.json", async (route) => {
    const response = await route.fetch();
    const catalog = await response.json();
    const copper = catalog.anime.find((item: { animeId: number }) => item.animeId === 101);
    const glass = catalog.anime.find((item: { animeId: number }) => item.animeId === 104);
    copper.aliases = [`Galaxy ${unsafe}`];
    glass.aliases = [`Galaxy ${unsafe}`];
    copper.imageUrl = "javascript:window.__titleUnsafe=2";
    glass.imageUrl = "https://covers.example.invalid/glass.png";
    await route.fulfill({ response, json: catalog });
  });
  await page.goto("/");
  await page.locator("#anime-input").fill("Galaxy");
  await page.locator("#add-anime-form button").click();
  const dialog = page.locator("#title-search-dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".title-search-cover-placeholder")).toHaveCount(1);
  await expect(dialog.locator("img")).toHaveAttribute("src", "https://covers.example.invalid/glass.png");
  await expect(dialog).toContainText(unsafe);
  await expect(dialog.locator(".title-search-choice img")).toHaveCount(1);
  expect(await page.evaluate(() => (window as unknown as { __titleUnsafe?: number }).__titleUnsafe)).toBeUndefined();
});

test("unknown imported history remains unmapped after a separate title choice", async ({ page }) => {
  await page.goto("/");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#bulk-import-input").fill("999, 8, Completed, 12");
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-summary")).toContainText("unmapped: 1");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#history-list")).toContainText("unmapped, kept");
  await page.locator("#anime-input").fill("Copper");
  await page.locator("#add-anime-form button").click();
  await page.locator("#title-search-dialog .title-search-choice").click();
  await page.reload();
  await expect(page.locator("#history-list")).toContainText("unmapped, kept");
});

test("normal mode uses aliases only after mocked metadata is already loaded", async ({ page }) => {
  const graph = JSON.parse(readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8"));
  let metadataRequests = 0;
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/data/graph.compact.json.gz") return route.fulfill({ status: 404, body: "" });
    if (url.pathname === "/data/graph.compact.json") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(graph) });
    }
    if (url.pathname.startsWith("/data/")) return route.fulfill({ status: 404, body: "" });
    const match = /^\/v4\/anime\/(\d+)\/full$/.exec(url.pathname);
    if (url.hostname === "api.jikan.moe" && match) {
      metadataRequests += 1;
      const id = Number(match[1]);
      return route.fulfill({ contentType: "application/json", headers: { "Access-Control-Allow-Origin": "*" },
        json: { data: { year: 2020, score: 7.8, type: "Movie", genres: [],
          titles: [{ type: "English", title: id === 102 ? "Moonlit Studio" : `Other ${id}` }] } } });
    }
    if (url.hostname === "api.jikan.moe") return route.fulfill({ status: 404, body: "{}" });
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  await page.goto("http://127.0.0.1:5174/");
  await page.locator("#anime-input").fill("Moonlit Studio");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-message")).toContainText("No catalog title or known alias");
  expect(metadataRequests).toBe(0);
  await page.locator("#discovery-load-metadata").click();
  await expect.poll(() => metadataRequests).toBe(8);
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#selected-anime")).toContainText("Moonlit Workshop");
  expect(metadataRequests).toBe(8);
});

test("graph search refuses an ambiguous partial node label", async ({ page }) => {
  await page.goto("/");
  await page.locator("#nav-network").click();
  await page.locator("#network-search-input").fill("o");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-search-message")).toContainText("Several graph nodes match");
  await page.locator("#network-search-input").fill("anime:101");
  await page.locator("#network-search-form button").click();
  await expect(page.locator("#network-search-message")).toContainText("Copper Comet");
});

test("title choices remain inside a narrow viewport and Escape leaves input intact", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.locator("#anime-input").fill("Galaxy Route");
  await page.locator("#add-anime-form button").click();
  const dialog = page.locator("#title-search-dialog");
  await expect(dialog).toBeVisible();
  const bounds = await dialog.boundingBox();
  expect(bounds).not.toBeNull();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390);
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(page.locator("#anime-input")).toHaveValue("Galaxy Route");
  await expect(page.locator("#watched-count")).toHaveText("0");
});
