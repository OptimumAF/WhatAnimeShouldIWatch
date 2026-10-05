import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

function demoArtifact(name: string): any {
  return JSON.parse(readFileSync(new URL(`../public/demo-data/${name}`, import.meta.url), "utf8"));
}

const normalAppUrl = "http://127.0.0.1:5174/";

function legacyGraph(): any {
  const compact = demoArtifact("graph.compact.json");
  const nodes = [
    ...compact.userIds.map((id: string) => ({ id: `user:${id}`, label: `User ${id}`, nodeType: "user" })),
    ...compact.anime.map(([id, title]: [number, string]) => ({ id: `anime:${id}`, label: title, nodeType: "anime" })),
  ];
  const edges = [
    ...compact.ua.map(([user, item, weight]: [number, number, number]) => ({
      id: `ua:${user}:${item}`, source: `user:${compact.userIds[user]}`,
      target: `anime:${compact.anime[item][0]}`, edgeType: "user-anime", weight,
    })),
    ...compact.aa.map(([left, right, weight, support]: [number, number, number, number?]) => ({
      id: `aa:${left}:${right}`, source: `anime:${compact.anime[left][0]}`,
      target: `anime:${compact.anime[right][0]}`, edgeType: "anime-anime", weight, support,
    })),
  ];
  return { generatedAt: compact.generatedAt, userCount: compact.userCount,
    animeCount: compact.animeCount, nodeCount: nodes.length, edgeCount: edges.length, nodes, edges };
}

function legacyModel(): any {
  const compact = demoArtifact("model-mf-web.compact.json");
  return { generatedAt: compact.generatedAt, globalMean: compact.globalMean,
    factors: compact.factors, animeCount: compact.animeIds.length,
    anime: compact.animeIds.map((animeId: number, i: number) => ({
      animeId, title: compact.titles[i], bias: compact.biases[i],
      embedding: compact.embeddings[i],
    })) };
}

test("a broken graph reference is named before recommendations start", async ({ page }) => {
  const graph = demoArtifact("graph.aggregate.compact.json");
  graph.aa[0][1] = graph.anime.length;
  await page.route("**/demo-data/graph.aggregate.compact.json", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(graph),
  }));

  await page.goto("/");
  await expect(page.locator("#rec-message")).toContainText(
    "synthetic demo graph: aa[0][1] references an index outside",
  );
  await expect(page.locator("#rec-engine-status")).toContainText("artifact is repaired");
});

test("demo refuses a v2 recommendation asset that would reintroduce user rows", async ({ page }) => {
  const graph = demoArtifact("graph.compact.json");
  await page.route("**/demo-data/graph.aggregate.compact.json", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(graph),
  }));
  await page.goto("/");
  await expect(page.locator("#rec-message")).toContainText(
    "synthetic demo graph: format must be graph-compact-v3 for the synthetic demo",
  );
});

test("a duplicate catalog ID reports the affected artifact", async ({ page }) => {
  const catalog = demoArtifact("catalog.json");
  catalog.anime[1].animeId = catalog.anime[0].animeId;
  await page.route("**/demo-data/catalog.json", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(catalog),
  }));

  await page.goto("/");
  await expect(page.locator("#rec-message")).toContainText(
    "synthetic demo catalog: anime[1].animeId duplicates",
  );
});

test("a model with the wrong embedding width is named while eligible graph results remain", async ({ page }) => {
  const model = demoArtifact("model-mf-web.compact.json");
  model.embeddings[0].pop();
  await page.route("**/demo-data/model-mf-web.compact.json", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(model),
  }));

  await page.goto("/");
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText(
    "synthetic demo model: embeddings[0] dimension must equal factors (2)",
  );
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
  await expect(page.locator("#diagnostic-code")).toContainText("MODEL-001");
  await expect(page.locator("#diagnostic-model")).toContainText("graph fallback active");
  await expect(page.locator("#rec-method")).toHaveValue("model");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop", "星の航路"]);
});

test("a present malformed source model digest is named and graph fallback remains eligible", async ({ page }) => {
  const model = demoArtifact("model-mf-web.compact.json");
  model.sourceModelSha256 = "not-a-sha";
  await page.route("**/demo-data/model-mf-web.compact.json", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(model),
  }));
  await page.goto("/");
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText(
    "synthetic demo model: sourceModelSha256 must be a lowercase SHA-256 digest",
  );
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
});

test("a compact model with undeclared user factors fails visibly and keeps graph fallback", async ({ page }) => {
  const model = demoArtifact("model-mf-web.compact.json");
  model.userFactors = [[1, 0]];
  await page.route("**/demo-data/model-mf-web.compact.json", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(model),
  }));
  await page.goto("/");
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText(
    "synthetic demo model: root.userFactors is unsupported",
  );
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
});

test("a present malformed model dataset digest names the field and keeps graph fallback", async ({ page }) => {
  const model = demoArtifact("model-mf-web.compact.json");
  model.datasetSha256 = "not-a-sha";
  await page.route("**/demo-data/model-mf-web.compact.json", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(model),
  }));
  await page.goto("/");
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText(
    "synthetic demo model: datasetSha256 must be a lowercase SHA-256 digest",
  );
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback");
});

test("normal mode still loads valid unversioned legacy graph and model artifacts", async ({ page }) => {
  const graph = legacyGraph();
  const model = legacyModel();
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/data/") && url.pathname.endsWith(".gz")) {
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.pathname === "/data/graph.compact.json" ||
        url.pathname === "/data/model-mf-web.compact.json") {
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.pathname === "/data/graph.json") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(graph) });
    }
    if (url.pathname === "/data/model-mf-web.json") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(model) });
    }
    if (url.pathname.startsWith("/data/")) {
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json", headers: { "Access-Control-Allow-Origin": "*" }, body: '{"data":[]}' });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
      return route.abort();
    }
    return route.continue();
  });

  await page.goto(normalAppUrl);
  await page.locator("#advanced-recommendation-settings summary").click();
  await expect(page.locator("#diagnostic-data")).toContainText("Legacy unversioned data");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations (2 factors)");
  await expect(page.locator("#diagnostic-model")).toContainText("legacy unpinned asset");
  await expect(page.locator("#rec-results li").first()).toBeVisible();
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-render-status")).toContainText("nodes");
  await expect(page.locator("#network-versions")).toContainText("unversioned legacy graph");
  await expect(page.locator("#network-selection")).toContainText("does not declare source rating or pair-selection coverage");
  await expect(page.locator("#network-explorer-sample")).toContainText("cannot be verified");
  await expect(page.locator("#network-scope-caveat")).toContainText("does not prove no relationship");
  expect(pageErrors).toEqual([]);
});

test("normal mode keeps graph suggestions when the optional model files are absent", async ({ page }) => {
  const graph = legacyGraph();
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/data/") && url.pathname !== "/data/graph.json") {
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.pathname === "/data/graph.json") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(graph) });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json", headers: { "Access-Control-Allow-Origin": "*" }, body: '{"data":[]}' });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });

  await page.goto(normalAppUrl);
  await page.locator("#advanced-recommendation-settings summary").click();
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using graph fallback. ML model data not found");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop", "星の航路"]);
  await expect(page.locator("#rec-method")).toHaveValue("model");
});

test("an unsupported compact graph version is rejected instead of using the legacy fallback", async ({ page }) => {
  const unsupported = demoArtifact("graph.compact.json");
  unsupported.format = "graph-compact-v4";
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/data/graph.compact.json.gz") {
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.pathname === "/data/graph.compact.json") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(unsupported) });
    }
    if (url.pathname === "/data/graph.json" || url.pathname === "/data/graph.json.gz") {
      return route.fulfill({ contentType: "application/json", body: JSON.stringify(legacyGraph()) });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") {
      return route.abort();
    }
    return route.continue();
  });

  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText(
    "graph.compact.json: format is unsupported; expected graph-compact-v1, v2, or v3",
  );
  await expect(page.locator("#rec-engine-status")).toContainText("artifact is repaired");
});
