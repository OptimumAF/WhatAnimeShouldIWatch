import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

const normalAppUrl = "http://127.0.0.1:5174/";
const syntheticUsername = "fixture-user";

async function openNormalAppWithMockedProviders(
  page: Page,
  malStatus: number,
  malBody: string,
  anilistBody?: string,
): Promise<string[]> {
  const graphJson = readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8");
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.hostname === "myanimelist.net") {
      if (malStatus === 0) {
        return route.abort();
      }
      return route.fulfill({
        status: malStatus,
        contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        body: malBody,
      });
    }
    if (url.hostname === "graphql.anilist.co" && anilistBody !== undefined) {
      return route.fulfill({
        status: 200, contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" }, body: anilistBody,
      });
    }
    if (url.hostname === "r.jina.ai") {
      // A legacy fallback would succeed here, making accidental proxy use visible.
      return route.fulfill({ status: 200, body: "Markdown Content: []" });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        body: '{"data":[]}',
      });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
      return route.abort();
    }
    if (url.pathname === "/data/graph.compact.json.gz") {
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.pathname === "/data/graph.compact.json") {
      return route.fulfill({ status: 200, contentType: "application/json", body: graphJson });
    }
    return route.continue();
  });

  await page.goto(normalAppUrl);
  await expect(page.getByRole("heading", { name: "What Anime Should I Watch" })).toBeVisible();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#username-import-provider").selectOption("mal");
  await page.locator("#username-import-input").fill(syntheticUsername);
  return requests;
}

test("mocked AniList import stores a native five-star score and maps a clear like", async ({ page }) => {
  const requests = await openNormalAppWithMockedProviders(page, 200, "[]", JSON.stringify({ data: {
    User: { mediaListOptions: { scoreFormat: "POINT_5" } },
    MediaListCollection: { lists: [{ entries: [{
      score: 4, status: "COMPLETED", progress: 12,
      media: { id: 501, idMal: 101, title: { romaji: "Copper Comet" } },
    }] }] },
  } }));
  await page.locator("#username-import-provider").selectOption("anilist");
  await page.locator("#username-import-submit").click();
  await expect(page.locator("#history-import-preview")).toBeVisible();
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#history-list")).toContainText("score: 4 (POINT_5)");
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']")).toHaveValue("liked");
  await expect(page.locator("#selected-anime .chip-confidence")).toContainText("50% confidence");
  expect(requests.filter((url) => new URL(url).hostname === "graphql.anilist.co")).toHaveLength(1);
});

test("MAL failure keeps the existing watched list and never routes through a proxy", async ({ page }) => {
  const requests = await openNormalAppWithMockedProviders(page, 403, "{}");
  await expect(page.locator(".username-import .muted").first()).toContainText("username is sent directly");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#watched-count")).toHaveText("1");

  await page.locator("#username-import-submit").click();
  await expect(page.locator("#rec-message")).not.toContainText("Importing rated anime");
  expect(requests.filter((url) => new URL(url).hostname === "r.jina.ai")).toEqual([]);
  await expect(page.locator("#rec-message")).toContainText("Direct MAL import failed");
  await expect(page.locator("#rec-message")).toContainText("No proxy was contacted");
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "unavailable");
  await expect(page.locator("#watched-count")).toHaveText("1");
  expect(requests.filter((url) => new URL(url).hostname === "myanimelist.net")).toHaveLength(1);
});

test("a direct MAL response still imports a mapped synthetic rating", async ({ page }) => {
  const requests = await openNormalAppWithMockedProviders(
    page, 200, '[{"anime_id":102,"score":8}]',
  );
  await page.locator("#username-import-submit").click();
  await expect(page.locator("#history-import-preview")).toBeVisible();
  await expect(page.locator("#history-import-summary")).toContainText("1 entries");
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#rec-message")).toContainText("Import applied");
  await expect(page.locator("#selected-anime")).toContainText("Moonlit Workshop");
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "ready");
  expect(requests.filter((url) => new URL(url).hostname === "r.jina.ai")).toEqual([]);
});

test("mocked MAL username history keeps unscored seen and unmapped entries through preview and reload", async ({ page }) => {
  const requests = await openNormalAppWithMockedProviders(page, 200, JSON.stringify([
    { anime_id: 101, anime_title: "Copper Comet", score: 9, status: 2, num_watched_episodes: 12 },
    { anime_id: 102, anime_title: "Moonlit Workshop", score: 0, status: 1, num_watched_episodes: 3 },
    { anime_id: 99999, anime_title: "Unknown Fixture", score: 0, status: 6, num_watched_episodes: 0 },
  ]));
  await page.locator("#username-import-submit").click();
  await expect(page.locator("#history-import-summary")).toContainText("3 entries");
  await expect(page.locator("#history-import-summary")).toContainText("unmapped: 1");
  await expect(page.locator("#history-import-summary")).toContainText("unscored: 2");
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#watched-count")).toHaveText("2");
  await expect(page.locator("#history-count")).toHaveText("3");
  await expect(page.locator("#history-list")).toContainText("Unknown Fixture — 6; episodes: 0; score: unscored (mal-10); unmapped, kept");
  await expect(page.locator("#rec-results")).not.toContainText("Moonlit Workshop");
  await page.reload();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await expect(page.locator("#history-count")).toHaveText("3");
  expect(requests.filter((url) => new URL(url).hostname === "r.jina.ai")).toEqual([]);
});

test("an empty rated-list response leaves the watched list intact", async ({ page }) => {
  await openNormalAppWithMockedProviders(page, 200, "[]");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-anime-form button").click();
  await page.locator("#username-import-submit").click();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "empty");
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#selected-anime")).toContainText("Copper Comet");
});

test("a browser transport failure gives an actionable error without proxying", async ({ page }) => {
  const requests = await openNormalAppWithMockedProviders(page, 0, "");
  await page.locator("#username-import-submit").click();
  await expect(page.locator("#rec-message")).toContainText("Direct MAL import failed", { timeout: 15_000 });
  await expect(page.locator("#rec-message")).toContainText("local file or text import");
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "failed");
  expect(requests.filter((url) => new URL(url).hostname === "r.jina.ai")).toEqual([]);
  expect(requests.filter((url) => new URL(url).hostname === "myanimelist.net")).toHaveLength(4);
});

test("local file import waits for review and never sends the entries to a provider", async ({ page }) => {
  const requests = await openNormalAppWithMockedProviders(page, 403, "{}");
  await expect(page.getByRole("heading", { name: "Import From Local File or Text (Recommended)" })).toBeVisible();
  await page.locator("#bulk-import-file").setInputFiles({
    name: "too-large.txt",
    mimeType: "text/plain",
    buffer: Buffer.alloc(128 * 1024 + 1, "1"),
  });
  await expect(page.locator("#rec-message")).toContainText("up to 128 KiB");
  await expect(page.locator("#watched-count")).toHaveText("0");

  await page.locator("#bulk-import-file").setInputFiles({
    name: "synthetic.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("101, 9\n102, 8\n"),
  });
  await expect(page.locator("#rec-message")).toContainText("Preview Text Import");
  await expect(page.locator("#bulk-import-input")).toHaveValue("101, 9\n102, 8\n");
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-preview")).toBeVisible();
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#watched-count")).toHaveText("2");
  expect(requests.filter((url) => ["r.jina.ai", "myanimelist.net", "graphql.anilist.co"].includes(new URL(url).hostname))).toEqual([]);
});
