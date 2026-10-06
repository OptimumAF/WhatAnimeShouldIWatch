import { readFileSync } from "node:fs";
import { expect, test, type Page, type Route } from "@playwright/test";

const graphJson = readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url), "utf8");
const localHosts = new Set(["127.0.0.1", "localhost"]);

async function openDemo(page: Page): Promise<string[]> {
  const externalRequests: string[] = [];
  page.on("request", (request) => {
    if (!localHosts.has(new URL(request.url()).hostname)) externalRequests.push(request.url());
  });
  await page.route("**/*", (route) =>
    localHosts.has(new URL(route.request().url()).hostname) ? route.continue() : route.abort());
  await page.goto("/");
  await expect(page.getByText("SYNTHETIC DEMO DATA")).toBeVisible({ timeout: 15_000 });
  return externalRequests;
}

async function openNormalWithMocks(
  page: Page,
  provider: (route: Route, url: URL) => Promise<void> | void,
): Promise<string[]> {
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (["myanimelist.net", "graphql.anilist.co", "api.jikan.moe"].includes(url.hostname)) {
      await provider(route, url);
    } else if (!localHosts.has(url.hostname)) {
      await route.abort();
    } else if (url.pathname === "/data/graph.compact.json.gz") {
      await route.fulfill({ status: 404, body: "" });
    } else if (url.pathname === "/data/graph.compact.json") {
      await route.fulfill({ contentType: "application/json", body: graphJson });
    } else if (url.pathname.includes("model-mf-web")) {
      await route.fulfill({ status: 404, body: "" });
    } else {
      await route.continue();
    }
  });
  await page.goto("http://127.0.0.1:5174/");
  await expect(page.getByRole("heading", { name: "What Anime Should I Watch" })).toBeVisible();
  return requests;
}

async function addLikedCopper(page: Page): Promise<void> {
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']"))
    .toHaveValue("liked");
}

function expectNoProviderOrProxyRequests(externalRequests: string[]): void {
  expect(externalRequests.filter((request) =>
    ["myanimelist.net", "graphql.anilist.co", "api.jikan.moe", "r.jina.ai"]
      .includes(new URL(request).hostname))).toEqual([]);
}

test("manual first use reaches a saved shortlist and restores it after reload on a phone", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const external = await openDemo(page);
  await page.getByRole("button", { name: /Add favorites/ }).click();
  await expect(page.locator("#anime-input")).toBeFocused();
  await addLikedCopper(page);
  const result = page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(result).toBeVisible();
  await result.locator("button[data-watchlist-save-id='102']").click();
  await expect(page.locator("#rec-action-status")).toContainText("Moonlit Workshop");
  await expect(page.locator("#watchlist-count")).toHaveText("1");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);

  await page.reload();
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']"))
    .toHaveValue("liked");
  await expect(page.locator("#watchlist-list")).toContainText("Moonlit Workshop");
  await expect(page.locator("#watchlist-list select[data-watchlist-status-id='102']"))
    .toHaveValue("plan_to_watch");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);
  expectNoProviderOrProxyRequests(external);
});

test("reviewed local history flows into eligibility and content filters after reload", async ({ page }) => {
  const external = await openDemo(page);
  await page.getByRole("button", { name: /Import a list/ }).click();
  await page.locator("#bulk-import-input").fill("101, 9, Completed, 12\n103, 0, Watching, 2");
  await page.locator("#bulk-import-form button").click();
  await expect(page.locator("#history-import-summary")).toContainText("2 entries");
  await expect(page.locator("#watched-count")).toHaveText("0");
  await page.locator("#history-import-apply").click();
  await expect(page.locator("#history-count")).toHaveText("2");
  await expect(page.locator("#watched-count")).toHaveText("2");
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']"))
    .toHaveValue("liked");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Copper Comet" })).toHaveCount(0);
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Ashen Harbor" })).toHaveCount(0);

  await page.locator("#filter-genre").selectOption("slice of life");
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-max").fill("2020");
  await page.locator("#filter-year-max").press("Tab");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);

  await page.reload();
  await expect(page.locator("#history-count")).toHaveText("2");
  await expect(page.locator("#watched-count")).toHaveText("2");
  await page.locator("#filter-genre").selectOption("slice of life");
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-max").fill("2020");
  await page.locator("#filter-year-max").press("Tab");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop"]);
  expectNoProviderOrProxyRequests(external);
});

test("switching profiles cancels a delayed mocked username import without changing saved picks", async ({ page }) => {
  let releaseImport: (() => void) | undefined;
  let importSettled = false;
  const requests = await openNormalWithMocks(page, async (route, url) => {
    if (url.hostname === "myanimelist.net") {
      await new Promise<void>((resolve) => { releaseImport = resolve; });
      await route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        body: JSON.stringify([{ anime_id: 103, score: 9 }]) }).catch(() => {});
      importSettled = true;
    } else {
      await route.fulfill({ status: 404, body: "" });
    }
  });
  await addLikedCopper(page);
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Invented favorite profile");
  await page.locator("#profile-save-submit").click();
  await page.locator("#clear-watched").click();
  await expect(page.locator("#watched-count")).toHaveText("0");

  await page.locator("#username-import-provider").selectOption("mal");
  await page.locator("#username-import-input").fill("fixture-user");
  await page.locator("#username-import-submit").click();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "loading");
  await expect.poll(() => typeof releaseImport).toBe("function");
  await page.locator("#profile-select").selectOption("Invented favorite profile");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#username-import-status")).toHaveAttribute("data-state", "stale");
  releaseImport?.();
  await expect.poll(() => importSettled).toBe(true);
  await expect(page.locator("#selected-anime")).toContainText("Copper Comet");
  await expect(page.locator("#selected-anime")).not.toContainText("Ashen Harbor");
  await expect(page.locator("#history-count")).toHaveText("0");
  await page.reload();
  await expect(page.locator("#selected-anime select[data-preference-node-id='anime:101']"))
    .toHaveValue("liked");
  await expect(page.locator("#history-count")).toHaveText("0");
  expect(requests.filter((request) => new URL(request).hostname === "myanimelist.net")).toHaveLength(1);
  expect(requests.some((request) => new URL(request).hostname === "r.jina.ai")).toBe(false);
});

test("profile changes discard delayed metadata and refetch for the restored preference", async ({ page }) => {
  let releaseFirst: (() => void) | undefined;
  let firstSettled = false;
  let detailRequests = 0;
  await openNormalWithMocks(page, async (route, url) => {
    if (url.pathname === "/v4/anime/102/full") {
      detailRequests += 1;
      if (detailRequests === 1) {
        await new Promise<void>((resolve) => { releaseFirst = resolve; });
      }
      await route.fulfill({ contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        body: JSON.stringify({ data: { year: 2020, score: 7.8, genres: [{ name: "Fantasy" }] } }),
      }).catch(() => {});
      if (detailRequests === 1) firstSettled = true;
    } else {
      await route.fulfill({ status: 404, body: "" });
    }
  });
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Invented empty profile");
  await page.locator("#profile-save-submit").click();
  await addLikedCopper(page);
  await expect.poll(() => detailRequests).toBeGreaterThan(0);
  await expect.poll(() => typeof releaseFirst).toBe("function");
  await page.locator("#profile-select").selectOption("Invented empty profile");
  await page.locator("#profile-load-btn").click();
  await expect(page.locator("#watched-count")).toHaveText("0");
  await expect(page.locator("#rec-engine-status")).toContainText("catalog popularity proxy");
  releaseFirst?.();
  await expect.poll(() => firstSettled).toBe(true);
  await expect(page.locator("#rec-engine-status")).toContainText("catalog popularity proxy");
  await addLikedCopper(page);
  await expect.poll(() => detailRequests).toBeGreaterThan(1);
  await expect(page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" }))
    .toHaveAttribute("data-metadata-state", "ready");
});

test("missing model and unavailable provider details still allow a saved graph result", async ({ page }) => {
  const requests = await openNormalWithMocks(page, (route) =>
    route.fulfill({ status: 404, body: "" }));
  await page.locator("#advanced-recommendation-settings summary").click();
  await addLikedCopper(page);
  await page.locator("#rec-method").selectOption("hybrid");
  await expect(page.locator("#rec-method")).toHaveValue("hybrid");
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
  const result = page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(result).toBeVisible();
  await expect(result).toHaveAttribute("data-metadata-state", "unavailable");
  await result.locator("button[data-watchlist-save-id='102']").click();
  await expect(page.locator("#watchlist-list")).toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);
  expect(requests.some((request) => new URL(request).pathname.includes("model-mf-web"))).toBe(true);
  expect(requests.some((request) => {
    const url = new URL(request);
    return url.hostname === "api.jikan.moe" && url.pathname === "/v4/anime/102/full";
  })).toBe(true);
  expect(requests.some((request) => new URL(request).hostname === "r.jina.ai")).toBe(false);
});

test("corrupt current state recovers a real saved shortlist and preserves raw evidence", async ({ page }) => {
  const external = await openDemo(page);
  await addLikedCopper(page);
  await page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" })
    .locator("button[data-watchlist-save-id='102']").click();
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await page.locator("#profile-name-input").fill("Invented recovery profile");
  await page.locator("#profile-save-submit").click();
  await page.evaluate(() => {
    const stateKey = "wasiw.demo.recommendationState.v5";
    const profilesKey = "wasiw.demo.recommendationProfiles.v5";
    localStorage.setItem(`${stateKey}.backup`, localStorage.getItem(stateKey) ?? "");
    localStorage.setItem(`${profilesKey}.backup`, localStorage.getItem(profilesKey) ?? "");
    localStorage.setItem(stateKey, "{invented corrupt state");
    localStorage.setItem(profilesKey, "{invented corrupt profiles");
  });
  await page.reload();
  await expect(page.locator("#storage-status")).toContainText("Recovered recommendation state from a backup");
  await expect(page.locator("#watchlist-list")).toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);
  await page.locator("summary").filter({ hasText: "Import & Profiles" }).click();
  await expect(page.locator("#profile-select")).toContainText("Invented recovery profile");
  await page.locator("#profile-repair-btn").click();
  await expect(page.locator("#profile-backup-status")).toContainText("restored to current browser storage");
  await page.reload();
  await expect(page.locator("#watchlist-list")).toContainText("Moonlit Workshop");
  const saved = await page.evaluate(() => ({
    state: JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null"),
    oldState: localStorage.getItem("wasiw.demo.recommendationState.v5.corrupt"),
    oldProfiles: localStorage.getItem("wasiw.demo.recommendationProfiles.v5.corrupt"),
  }));
  expect(saved.state.watchlist).toEqual([
    { animeId: 102, title: "Moonlit Workshop", status: "plan_to_watch", rating: null },
  ]);
  expect(saved.oldState).toBe("{invented corrupt state");
  expect(saved.oldProfiles).toBe("{invented corrupt profiles");
  expectNoProviderOrProxyRequests(external);
});
