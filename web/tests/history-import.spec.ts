import { expect, test, type Page } from "@playwright/test";

const malXml = `<?xml version="1.0" encoding="UTF-8"?>
<myanimelist>
  <myinfo><user_export_type>1</user_export_type></myinfo>
  <anime><series_animedb_id>101</series_animedb_id><series_title>Copper Comet</series_title><my_watched_episodes>12</my_watched_episodes><my_score>9</my_score><my_status>Completed</my_status></anime>
  <anime><series_animedb_id>102</series_animedb_id><series_title>Moonlit Workshop</series_title><my_watched_episodes>3</my_watched_episodes><my_score>0</my_score><my_status>Watching</my_status></anime>
  <anime><series_animedb_id>99999</series_animedb_id><series_title><![CDATA[Unmapped <svg onload="window.__imported=1">]]></series_title><my_watched_episodes>0</my_watched_episodes><my_score>0</my_score><my_status>Plan to Watch</my_status></anime>
</myanimelist>`;

async function openDemo(page: Page): Promise<string[]> {
  const external: string[] = [];
  page.on("request", (request) => {
    const hostname = new URL(request.url()).hostname;
    if (["myanimelist.net", "graphql.anilist.co", "api.jikan.moe", "r.jina.ai"].includes(hostname)) {
      external.push(request.url());
    }
  });
  await page.route("**/*", (route) => {
    const hostname = new URL(route.request().url()).hostname;
    return hostname === "127.0.0.1" || hostname === "localhost"
      ? route.continue() : route.abort();
  });
  await page.goto("/");
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  return external;
}

async function setXmlFile(page: Page, content: string, name = "invented-mal.xml"): Promise<void> {
  await page.locator("#bulk-import-file").setInputFiles({
    name, mimeType: "application/xml", buffer: Buffer.from(content),
  });
}

async function applyText(page: Page, content: string): Promise<void> {
  await page.locator("#bulk-import-input").fill(content);
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-preview")).toBeVisible();
  await page.locator("#history-import-apply").click();
}

test("local MAL XML previews and retains unscored, status, progress, scale, unmapped, and reload state", async ({ page }) => {
  const external = await openDemo(page);
  await setXmlFile(page, malXml);
  await expect(page.locator("#history-import-preview")).toBeVisible();
  await expect(page.locator("#history-import-summary")).toContainText("3 entries");
  await expect(page.locator("#history-import-summary")).toContainText("unmapped: 1");
  await expect(page.locator("#history-import-summary")).toContainText("unscored: 2");
  await expect(page.locator("#history-import-summary")).toContainText("seen: 2");
  await expect(page.locator("#history-import-unmapped")).toContainText('Unmapped <svg onload="window.__imported=1">');
  await expect(page.locator("#watched-count")).toHaveText("0");
  await expect(page.locator("#history-count")).toHaveText("0");

  await page.locator("#history-import-apply").click();
  await expect(page.locator("#watched-count")).toHaveText("2");
  await expect(page.locator("#history-count")).toHaveText("3");
  await expect(page.locator("#history-list")).toContainText("Moonlit Workshop — Watching; episodes: 3; score: unscored (mal-10)");
  await expect(page.locator("#history-list")).toContainText("unmapped, kept");
  await expect(page.locator("#rec-results")).not.toContainText("Moonlit Workshop");
  expect(await page.evaluate(() => (window as unknown as { __imported?: number }).__imported)).toBeUndefined();

  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await expect(page.locator("#history-count")).toHaveText("3");
  await expect(page.locator("#history-list")).toContainText("unmapped, kept");
  await expect(page.locator("#rec-results")).not.toContainText("Moonlit Workshop");
  await setXmlFile(page, malXml);
  await expect(page.locator("#history-import-summary")).toContainText("unchanged: 3");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#history-count")).toHaveText("3");
  expect(external).toEqual([]);
});

test("merge and replace preview their effect and named profiles restore imported history", async ({ page }) => {
  await openDemo(page);
  await applyText(page, "101, 9, Completed, 12");
  await page.locator("#profile-name-input").fill("Before replace");
  await page.locator("#profile-save-submit").click();
  await page.locator("#anime-input").fill("星の航路");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#watched-count")).toHaveText("2");

  await page.locator("#bulk-import-input").fill("102, 8, Completed, 12");
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-summary")).toContainText("removed by replace: 0");
  await page.locator("#history-import-mode").selectOption("replace");
  await expect(page.locator("#history-import-summary")).toContainText("removed by replace: 1");
  await expect(page.locator("#history-import-summary")).toContainText("Preferences removed by replace: 2");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#history-count")).toHaveText("1");
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#selected-anime")).toContainText("Moonlit Workshop");
  await expect(page.locator("#selected-anime")).not.toContainText("星の航路");

  await page.locator("#profile-select").selectOption("Before replace");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#history-list")).toContainText("Copper Comet — Completed; episodes: 12");
  await expect(page.locator("#selected-anime")).toContainText("Copper Comet");
  await expect(page.locator("#history-list")).not.toContainText("Moonlit Workshop");
});

test("malformed XML, DTDs, oversized files, and storage failure preserve prior history", async ({ page }) => {
  await openDemo(page);
  await applyText(page, "101, 9, Completed, 12");
  const key = "wasiw.demo.recommendationState.v5";
  const previous = await page.evaluate((storageKey) => localStorage.getItem(storageKey), key);

  await setXmlFile(page, '<!DOCTYPE myanimelist [<!ENTITY x "bad">]><myanimelist><anime>&x;</anime></myanimelist>');
  await expect(page.locator("#bulk-import-status")).toHaveAttribute("data-state", "failed");
  await expect(page.locator("#rec-message")).toContainText("DTDs or entities are not supported");
  await expect(page.locator("#history-count")).toHaveText("1");
  await setXmlFile(page, malXml.replace("<my_score>9</my_score>", "<my_score>11</my_score>"));
  await expect(page.locator("#rec-message")).toContainText("Invalid MAL XML score");
  await page.locator("#bulk-import-file").setInputFiles({
    name: "oversized.xml", mimeType: "application/xml", buffer: Buffer.alloc(2 * 1024 * 1024 + 1, "x"),
  });
  await expect(page.locator("#bulk-import-status")).toHaveAttribute("data-state", "failed");
  await expect(page.locator("#history-count")).toHaveText("1");

  await page.evaluate(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (storageKey, value) {
      if (storageKey === "wasiw.demo.recommendationState.v5") throw new Error("synthetic quota");
      return original.call(this, storageKey, value);
    };
  });
  await page.locator("#bulk-import-input").fill("102, 8, Completed, 12");
  await page.locator("#bulk-import-form button").click();
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#bulk-import-status")).toHaveAttribute("data-state", "failed");
  await expect(page.locator("#history-count")).toHaveText("1");
  expect(await page.evaluate((storageKey) => localStorage.getItem(storageKey), key)).toBe(previous);
});
