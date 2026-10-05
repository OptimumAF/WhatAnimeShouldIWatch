import { expect, test } from "@playwright/test";

test("result cards separate ranking evidence from invented community scores", async ({ page }) => {
  await page.goto("/");
  const first = page.locator("#rec-results .rec-item").first();
  await expect(first.locator(".rec-score-label")).toHaveText("Community score");
  await expect(first.locator(".rec-why")).toContainText("not a personal prediction");
  await expect(first.locator(".rec-meta-details")).toContainText("2023");
  await expect(first.locator(".rec-community-score")).toHaveCount(0);
  await expect(first.locator(".rec-actions button")).toHaveText(["Plan to Watch", "Seen", "Not interested"]);

  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  const ranked = page.locator("#rec-results .rec-item").first();
  await expect(ranked.locator(".rec-title")).toHaveText("Moonlit Workshop");
  await expect(ranked.locator(".rec-score-label")).toHaveText("Graph ranking");
  await expect(ranked.locator(".rec-score")).toHaveText("+0.583");
  await expect(ranked.locator(".rec-community-score")).toHaveText("Demo community score: 7.80/10");
  await expect(ranked.locator(".rec-why-line").nth(1)).toContainText("No calibrated confidence interval");

  await page.locator("#discovery-view").selectOption("quality");
  await expect(page.locator("#rec-results .rec-score-label").first()).toHaveText("Community score");
  await expect(page.locator("#rec-results .rec-community-score")).toHaveCount(0);
  await expect(page.locator("#rec-results .rec-why").first()).toContainText("not a personal prediction");
});

test("Plan to Watch, Seen, and Not interested save three distinct local states", async ({ page }) => {
  await page.goto("/");
  await page.locator("#rec-results .rec-item").filter({ hasText: "Copper Comet" })
    .locator("button[data-card-action='save']").click();
  await expect(page.locator("#rec-action-status")).toContainText("Plan to Watch");
  await expect(page.locator("#watchlist-count")).toHaveText("1");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Copper Comet" })).toHaveCount(0);

  await page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" })
    .locator("button[data-card-action='seen']").click();
  await expect(page.locator("#rec-action-status")).toContainText("without treating it as Liked");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Moonlit Workshop" })).toHaveCount(0);
  await expect(page.locator("#rec-engine-status")).toContainText("community-score exploration");

  await page.locator("#rec-results .rec-item").filter({ hasText: "Ashen Harbor" })
    .locator("button[data-card-action='hide']").click();
  await expect(page.locator("#rec-action-status")).toContainText("does not create a Disliked model signal");
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Ashen Harbor" })).toHaveCount(0);
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem("wasiw.demo.recommendationState.v5") ?? "null"));
  expect(saved.watchlist).toEqual([{ animeId: 101, title: "Copper Comet", status: "plan_to_watch", rating: null }]);
  expect(saved.preferences).toEqual([{ nodeId: "anime:102", sentiment: "seen", importance: 1,
    confidence: 0, source: "manual" }]);
  expect(saved.excludeCandidates).toEqual(["anime:103"]);
  await page.reload();
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Ashen Harbor" })).toHaveCount(0);
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await page.locator("#exclude-anime button[data-exclude-node-id='anime:103']").click();
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Ashen Harbor" })).toHaveCount(1);
});

test("card feedback rolls back visibly when browser storage rejects it", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === "wasiw.demo.recommendationState.v5") {
        throw new DOMException("Invented quota rejection", "QuotaExceededError");
      }
      return original.call(this, key, value);
    };
  });
  const copper = page.locator("#rec-results .rec-item").filter({ hasText: "Copper Comet" });
  await copper.locator("button[data-card-action='seen']").click();
  await expect(page.locator("#rec-action-status")).toContainText("no preference was saved");
  await expect(page.locator("#watched-count")).toHaveText("0");
  await expect(copper).toBeVisible();
  await copper.locator("button[data-card-action='hide']").click();
  await expect(page.locator("#rec-action-status")).toContainText("title was not hidden");
  await expect(copper).toBeVisible();
});

test("unsafe covers are omitted and failed image loads show an explicit placeholder", async ({ page }) => {
  await page.route("https://covers.example.invalid/**", (route) => route.abort());
  await page.route("**/demo-data/catalog.json", async (route) => {
    const response = await route.fetch();
    const catalog = await response.json();
    catalog.anime.find((item: { animeId: number }) => item.animeId === 101).imageUrl = "javascript:window.__coverUnsafe=1";
    catalog.anime.find((item: { animeId: number }) => item.animeId === 102).imageUrl =
      "https://covers.example.invalid/moonlit.png";
    await route.fulfill({ response, json: catalog });
  });
  await page.goto("/");
  const copper = page.locator("#rec-results .rec-item").filter({ hasText: "Copper Comet" });
  const moonlit = page.locator("#rec-results .rec-item").filter({ hasText: "Moonlit Workshop" });
  await expect(copper.locator(".rec-cover-placeholder")).toContainText("Cover blocked");
  await expect(copper.locator("img")).toHaveCount(0);
  await expect(moonlit.locator(".rec-cover-error")).toContainText("Cover could not load");
  expect(await page.evaluate(() => (window as unknown as { __coverUnsafe?: number }).__coverUnsafe))
    .toBeUndefined();
});

test("result actions stay inside a narrow viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const first = page.locator("#rec-results .rec-item").first();
  const viewportWidth = page.viewportSize()!.width;
  for (const action of await first.locator(".rec-actions button").all()) {
    const bounds = await action.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(viewportWidth);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(viewportWidth);
});

test("untrusted result titles remain text in action labels and status", async ({ page }) => {
  const title = 'Moonlit "<img src=x onerror="window.__cardUnsafe=1">';
  for (const path of ["graph.aggregate.compact.json", "graph-explorer.aggregate.compact.json"]) {
    await page.route(`**/demo-data/${path}`, async (route) => {
      const response = await route.fetch();
      const graph = await response.json();
      graph.anime.find((item: [number, string]) => item[0] === 102)[1] = title;
      await route.fulfill({ response, json: graph });
    });
  }
  await page.route("**/demo-data/catalog.json", async (route) => {
    const response = await route.fetch();
    const catalog = await response.json();
    catalog.anime.find((item: { animeId: number }) => item.animeId === 102).title = title;
    await route.fulfill({ response, json: catalog });
  });
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  const card = page.locator("#rec-results .rec-item").filter({ has: page.locator(".rec-title", { hasText: title }) });
  await expect(card.locator(".rec-actions")).toHaveAttribute("aria-label", `Actions for ${title}`);
  await expect(card.locator("button[data-card-action='seen']"))
    .toHaveAttribute("aria-label", `Mark ${title} Seen`);
  await card.locator("button[data-card-action='hide']").click();
  await expect(page.locator("#rec-action-status")).toContainText(`Hidden ${title}.`);
  await expect(page.locator("#rec-action-status img, #rec-results img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __cardUnsafe?: number }).__cardUnsafe))
    .toBeUndefined();
});
