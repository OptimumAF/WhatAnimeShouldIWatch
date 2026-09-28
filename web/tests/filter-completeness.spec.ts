import { expect, test, type Page } from "@playwright/test";

interface MockMetadata {
  year: number;
  score: number;
  genres: string[];
}

async function openRankedFixture(
  page: Page,
  candidateCount: number,
  respond: (animeId: number) => { status: number; metadata?: MockMetadata },
  withModel = false,
  withGraphEdges = true,
): Promise<() => number[]> {
  const seedId = 101;
  const anime = [[seedId, "Fixture Seed"], ...Array.from({ length: candidateCount }, (_, index) =>
    [seedId + index + 1, index === candidateCount - 1 ? "Late Match" : `Fixture Candidate ${index + 1}`])];
  const graph = {
    format: "graph-compact-v1", generatedAt: "2026-09-28T00:00:00.000Z",
    userIds: [], anime, ua: [],
    aa: withGraphEdges
      ? Array.from({ length: candidateCount }, (_, index) => [0, index + 1, 1 - index * 0.01, 1]) : [],
    userCount: 0, animeCount: anime.length, nodeCount: anime.length,
    edgeCount: withGraphEdges ? candidateCount : 0,
  };
  const model = {
    format: "model-mf-compact-v1", generatedAt: graph.generatedAt,
    globalMean: 5, factors: 1, animeCount: anime.length,
    animeIds: anime.map(([id]) => id), titles: anime.map(([, title]) => title),
    biases: anime.map((_, index) => -index * 0.01),
    embeddings: anime.map(() => [1]),
  };
  const requests: number[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/data/graph.compact.json.gz") return route.fulfill({ status: 404, body: "" });
    if (url.pathname === "/data/graph.compact.json") return route.fulfill({ json: graph });
    if (withModel && url.pathname === "/data/model-mf-web.compact.json") return route.fulfill({ json: model });
    if (url.pathname.startsWith("/data/")) return route.fulfill({ status: 404, body: "" });
    const match = /^\/v4\/anime\/(\d+)\/full$/.exec(url.pathname);
    if (url.hostname === "api.jikan.moe" && match) {
      const animeId = Number(match[1]);
      requests.push(animeId);
      const result = respond(animeId);
      return route.fulfill({ status: result.status, contentType: "application/json",
        headers: { "Access-Control-Allow-Origin": "*" },
        json: result.metadata ? { data: {
          year: result.metadata.year, score: result.metadata.score,
          genres: result.metadata.genres.map((name) => ({ name })),
        } } : {} });
    }
    if (url.hostname === "api.jikan.moe") return route.fulfill({ status: 404,
      headers: { "Access-Control-Allow-Origin": "*" }, body: "{}" });
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  await page.goto("http://127.0.0.1:5174/");
  await page.locator("#anime-input").fill("Fixture Seed");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-engine-status")).toContainText(withGraphEdges
    ? "Using graph recommendations" : "shared-genre content baseline");
  await expect(page.locator("#watched-count")).toHaveText("1");
  await expect(page.locator("#rec-summary")).toContainText(withGraphEdges
    ? "graph edge ranking" : "shared-genre content baseline");
  return () => [...requests];
}

test("a filtered match after the first thirty ranked candidates is reachable", async ({ page }) => {
  test.setTimeout(90_000);
  const candidateCount = 32;
  const lateId = 101 + candidateCount;
  const requests = await openRankedFixture(page, candidateCount, (animeId) => ({ status: 200,
    metadata: { year: animeId === lateId ? 2025 : 2010, score: 8,
      genres: [animeId === lateId ? "Late Genre" : "Fantasy"] },
  }));
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary")).toContainText("No confirmed matches yet", { timeout: 30_000 });
  await expect(page.locator("#filter-metadata-note")).toContainText("2 unchecked");
  await expect(page.locator("#filter-metadata-more")).toBeVisible();
  await expect(page.locator("#rec-results .rec-item")).toHaveCount(0);
  await page.locator("#filter-metadata-more").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Late Match"]);
  await expect(page.locator("#filter-metadata-note")).toContainText("32/32 resolved");
  await expect(page.locator("#filter-metadata-more")).toBeHidden();
  await expect(page.locator("#filter-genre option[value='late genre']")).toHaveCount(1);
  expect(requests()).toContain(lateId);
  await page.locator("#filter-year-min").fill("2030");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary")).toContainText("No candidates meet the required filters");
  await expect(page.locator("#filter-metadata-note")).toContainText("32/32 resolved");
});

test("a failed required-metadata check stays partial until an explicit retry succeeds", async ({ page }) => {
  let ready = false;
  const requests = await openRankedFixture(page, 1, () => ready
    ? { status: 200, metadata: { year: 2025, score: 8, genres: ["Fantasy"] } }
    : { status: 400 });
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary")).toContainText("No confirmed matches yet");
  await expect(page.locator("#filter-metadata-note")).toContainText("1 failed");
  await expect(page.locator("#filter-metadata-more")).toContainText("Retry");
  ready = true;
  await page.locator("#filter-metadata-more").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Late Match"]);
  await expect(page.locator("#filter-metadata-note")).toContainText("1/1 resolved");
  expect(requests()).toEqual([102, 102]);
});

test("a model with unchecked candidates does not fall back before a late filtered match is checked", async ({ page }) => {
  test.setTimeout(90_000);
  const lateId = 133;
  await openRankedFixture(page, 32, (animeId) => ({ status: 200,
    metadata: { year: animeId === lateId ? 2025 : 2010, score: 8, genres: ["Fantasy"] },
  }), true);
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations");
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary"))
    .toContainText("No confirmed matches yet", { timeout: 30_000 });
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations");
  await page.locator("#filter-metadata-more").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Late Match"]);
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations");
});

test("community-score exploration checks beyond its first bounded batch", async ({ page }) => {
  test.setTimeout(90_000);
  const lateId = 133;
  const requests = await openRankedFixture(page, 32, (animeId) => ({ status: 200,
    metadata: { year: animeId === lateId ? 2025 : 2010, score: 8, genres: ["Fantasy"] },
  }));
  await page.locator("#discovery-view").selectOption("quality");
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary"))
    .toContainText("No confirmed matches yet", { timeout: 30_000 });
  await expect(page.locator("#discovery-metadata-note")).toContainText("full 32-title eligible catalog");
  const unchecked = Number(/(\d+) unchecked/.exec(
    await page.locator("#discovery-metadata-note").textContent() ?? "")?.[1]);
  expect(unchecked).toBeGreaterThan(0);
  for (let batch = 0; batch < Math.ceil(unchecked / 12); batch += 1) {
    await page.locator("#discovery-load-metadata").click();
    await expect(page.locator("#discovery-metadata-note"))
      .toContainText(`${Math.max(0, unchecked - (batch + 1) * 12)} unchecked`, { timeout: 30_000 });
  }
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Late Match"]);
  await expect(page.locator("#discovery-metadata-note")).toContainText("32/32 resolved");
  expect(requests()).toContain(lateId);
  await page.locator("#filter-year-min").fill("2030");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary")).toContainText("Metadata scan complete");
  await expect(page.locator("#rec-summary")).not.toContainText("Check more metadata");
  await expect(page.locator("#discovery-load-metadata")).toBeDisabled();
});

test("a sparse shared-genre fallback keeps filtered unseen catalog titles reachable", async ({ page }) => {
  test.setTimeout(90_000);
  await openRankedFixture(page, 32, (animeId) => ({ status: 200,
    metadata: { year: animeId === 133 ? 2025 : 2010, score: 8, genres: ["Fantasy"] },
  }), false, false);
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary"))
    .toContainText("No confirmed matches yet", { timeout: 30_000 });
  await expect(page.locator("#filter-metadata-note")).toContainText("32");
  await page.locator("#filter-metadata-more").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Late Match"]);
  await expect(page.locator("#rec-engine-status")).toContainText("shared-genre content baseline");
  await expect(page.locator("#filter-metadata-note")).toContainText("32/32 resolved");
});

test("unavailable metadata cannot pass a required filter and is explained after complete checks", async ({ page }) => {
  await openRankedFixture(page, 1, () => ({ status: 404 }));
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#rec-summary")).toContainText("No candidates meet the required filters");
  await expect(page.locator("#filter-metadata-note")).toContainText("1 unavailable");
  await expect(page.locator("#filter-metadata-more")).toBeHidden();
  await expect(page.locator("#metadata-status")).toHaveAttribute("data-state", "unavailable");
});

test("a superseded follow-up batch cannot add stale filter metadata", async ({ page }) => {
  test.setTimeout(90_000);
  await openRankedFixture(page, 32, (animeId) => ({ status: 200,
    metadata: { year: animeId === 133 ? 2025 : 2010, score: 8, genres: ["Fantasy"] },
  }));
  await page.locator("#filter-year-min").fill("2020");
  await page.locator("#filter-year-min").press("Tab");
  await expect(page.locator("#filter-metadata-note"))
    .toContainText("2 unchecked", { timeout: 30_000 });
  let releaseLate: (() => void) | undefined;
  await page.route("https://api.jikan.moe/v4/anime/133/full", async (route) => {
    await new Promise<void>((resolve) => { releaseLate = resolve; });
    await route.fulfill({ contentType: "application/json",
      headers: { "Access-Control-Allow-Origin": "*" },
      json: { data: { year: 2025, score: 8, genres: [{ name: "Fantasy" }] } },
    }).catch(() => {});
  });
  const requestedLate = page.waitForRequest((request) => request.url().endsWith("/v4/anime/133/full"));
  await page.locator("#filter-metadata-more").click();
  await requestedLate;
  await page.locator("#discovery-view").selectOption("popularity");
  await expect(page.locator("#filter-metadata-controls")).toBeHidden();
  await expect(page.locator("#rec-engine-status")).toContainText("catalog popularity proxy");
  releaseLate?.();
  await page.locator("#discovery-view").selectOption("auto");
  // The other ID in the canceled batch may have completed before the view switch.
  await expect(page.locator("#filter-metadata-note")).toContainText(/[12] unchecked/);
  await expect(page.locator("#rec-results .rec-title").filter({ hasText: "Late Match" })).toHaveCount(0);
});
