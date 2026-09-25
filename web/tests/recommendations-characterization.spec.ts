import { expect, test } from "@playwright/test";

test("synthetic recommendation modes retain their visible ranking and explanations", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();

  const expected = {
    graph: [
      ["Moonlit Workshop", "+0.583"], ["星の航路", "+0.417"],
    ],
    model: [
      ["星の航路", "+0.940"], ["Moonlit Workshop", "+0.760"],
      ["Quiet Satellite", "+0.520"], ["Café Nebula", "+0.490"],
      ["Glass Orchard", "+0.380"], ["Paper Current", "-0.090"],
      ["Ashen Harbor", "-0.400"],
    ],
    hybrid: [
      ["Moonlit Workshop", "16.26 rank points"], ["星の航路", "16.26 rank points"],
      ["Quiet Satellite", "7.94 rank points"], ["Café Nebula", "7.81 rank points"],
      ["Glass Orchard", "7.69 rank points"], ["Paper Current", "7.58 rank points"],
      ["Ashen Harbor", "7.46 rank points"],
    ],
  };

  for (const mode of ["graph", "model", "hybrid"] as const) {
    await page.locator("#rec-method").selectOption(mode);
    await expect(page.locator("#rec-engine-status")).toContainText({
      graph: "Using graph recommendations",
      model: "Using ML model recommendations",
      hybrid: "Using hybrid recommendations",
    }[mode]);
    await expect(page.locator("#rec-results .rec-item")).toHaveCount(expected[mode].length);
    const cards = await page.locator("#rec-results .rec-item").evaluateAll((items) =>
      items.map((item) => [
        item.querySelector(".rec-title")?.textContent,
        item.querySelector(".rec-score")?.textContent,
      ]),
    );
    expect(cards).toEqual(expected[mode]);
    const first = page.locator("#rec-results .rec-item").first();
    await expect(first.locator(".rec-why-line").first()).toContainText("Copper Comet");
    await expect(first.locator(".rec-why-line").nth(1)).toContainText("No calibrated confidence interval");
    await first.locator(".rec-score-breakdown summary").click();
    const equation = first.locator(".rec-score-equation");
    if (mode === "graph") {
      await expect(equation).toHaveText("Graph score +0.583 = Copper Comet (+0.583).");
      await expect(first.locator(".rec-score-detail").first()).toContainText("1 distinct source title");
    } else if (mode === "model") {
      await expect(equation).toHaveText(
        "Model score +0.940 = global mean (0.000) + item bias (+0.120) + Copper Comet (+0.820).");
      await expect(first.locator(".rec-score-detail").first()).toContainText("sum of absolute mapped weights");
    } else {
      await expect(equation).toContainText("Rank points 16.26 = graph rank #1");
      await expect(first.locator(".rec-score-detail")).toContainText([
        "Eligible raw graph/model scores set source ranks",
        "Effective weights: graph 50%, model 50%",
        "Graph input: Graph score +0.583",
        "Each title term sums",
        "There is no global mean",
        "Model input: Model score +0.760",
        "Each title term is its item-vector dot product",
        "The global mean and candidate item bias",
      ]);
    }
  }
});

test("browser model ranks from three explicit preference signals", async ({ page }) => {
  await page.goto("/");
  for (const [title, sentiment] of [
    ["Copper Comet", "liked"], ["Moonlit Workshop", "liked"], ["Ashen Harbor", "disliked"],
  ] as const) {
    await page.locator("#anime-input").fill(title);
    await page.locator("#add-preference").selectOption(sentiment);
    await page.locator("#add-anime-form button").click();
  }
  await page.locator("#rec-method").selectOption("model");
  await expect(page.locator("#rec-engine-status")).toContainText("Using ML model recommendations");
  await expect(page.locator("#rec-results .rec-title").first()).toHaveText("星の航路");
  await expect(page.locator("#rec-results .rec-title").nth(1)).toHaveText("Quiet Satellite");
  await expect(page.locator("#rec-results .rec-title").nth(2)).toHaveText("Café Nebula");
  for (const title of ["Copper Comet", "Moonlit Workshop", "Ashen Harbor"]) {
    await expect(page.locator("#rec-results .rec-title").filter({ hasText: title })).toHaveCount(0);
  }
  const first = page.locator("#rec-results .rec-item").first();
  await expect(first.locator(".rec-meta").first()).toContainText("3 distinct mapped titles | 3/3 signals mapped");
  await expect(first.locator(".rec-why-line").first()).toContainText(
    "Copper Comet, Moonlit Workshop, Ashen Harbor");
  await first.locator(".rec-score-breakdown summary").click();
  const equation = first.locator(".rec-score-equation");
  await expect(equation).toContainText("global mean (");
  await expect(equation).toContainText("item bias (");
  for (const title of ["Copper Comet", "Moonlit Workshop", "Ashen Harbor"]) {
    await expect(equation).toContainText(title);
  }
  await expect(first.locator(".rec-score-detail").first()).toContainText(
    "signed importance × confidence ÷");
});

test("hybrid endpoints keep the lowest source item and a missing graph gives the model full weight", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await page.locator("#rec-method").selectOption("hybrid");
  const weight = page.locator("#rec-blend");
  const setWeight = (value: string) => weight.evaluate((element, next) => {
    (element as HTMLInputElement).value = next;
    element.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);
  await setWeight("0");
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Moonlit Workshop", "星の航路"]);
  await expect(page.locator("#rec-results .rec-score")).toContainText(["rank points", "rank points"]);
  await setWeight("1");
  await expect(page.locator("#rec-results .rec-title")).toHaveCount(7);
  await expect(page.locator("#rec-results .rec-title").last()).toHaveText("Ashen Harbor");
  await setWeight("0");
  await page.locator("summary").filter({ hasText: "Candidate Overrides" }).click();
  await page.locator("#include-input").fill("Ashen Harbor");
  await page.locator("#add-include-form button").click();
  await expect(page.locator("#rec-results .rec-title")).toHaveText(["Ashen Harbor"]);
  await expect(page.locator("#rec-engine-status")).toContainText("model-only rank fusion");
  await expect(page.locator("#rec-results .rec-score")).toHaveText("16.39 rank points");
  await expect(page.locator("#rec-results .rec-why")).toContainText("not a probability");
});

test("recommendation explanation renders an invented unsafe label as text", async ({ page }) => {
  const unsafeLabel = '<img src=x onerror="window.__unsafe=1">';
  await page.route("**/demo-data/graph.compact.json", async (route) => {
    const response = await route.fetch();
    const graph = await response.json();
    graph.anime[0][1] = unsafeLabel;
    await route.fulfill({ response, json: graph });
  });
  await page.goto("/");
  await page.locator("#anime-input").fill("101");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-why").first()).toContainText(unsafeLabel);
  await page.locator("#rec-results .rec-score-breakdown summary").first().click();
  await expect(page.locator("#rec-results .rec-score-equation").first()).toContainText(unsafeLabel);
  await expect(page.locator("#rec-results .rec-why img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __unsafe?: number }).__unsafe)).toBeUndefined();
});
