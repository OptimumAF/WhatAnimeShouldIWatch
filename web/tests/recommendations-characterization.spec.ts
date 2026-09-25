import { expect, test } from "@playwright/test";

test("synthetic recommendation modes retain their visible ranking and explanations", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();

  const expected = {
    graph: [
      ["Moonlit Workshop", "+0.583", "Why+: Copper Comet (+0.583)Why-: no notable negative contributors."],
      ["星の航路", "+0.417", "Why+: Copper Comet (+0.417)Why-: no notable negative contributors."],
    ],
    model: [
      ["星の航路", "+0.940", "Why+: Copper Comet (+0.820)Why-: no notable negative contributors."],
      ["Moonlit Workshop", "+0.760", "Why+: Copper Comet (+0.710)Why-: no notable negative contributors."],
      ["Quiet Satellite", "+0.520", "Why+: Copper Comet (+0.480)Why-: no notable negative contributors."],
      ["Café Nebula", "+0.490", "Why+: Copper Comet (+0.470)Why-: no notable negative contributors."],
      ["Glass Orchard", "+0.380", "Why+: Copper Comet (+0.430)Why-: no notable negative contributors."],
      ["Paper Current", "-0.090", "Why+: no strong positive contributors.Why-: Copper Comet (-0.060)"],
      ["Ashen Harbor", "-0.400", "Why+: no strong positive contributors.Why-: Copper Comet (-0.400)"],
    ],
    hybrid: [
      ["Moonlit Workshop", "16.26 rank points", "Why: Graph rank #1; positive evidence from Copper Comet | Model rank #2; positive evidence from Copper Comet. Rank points combine relative positions; they are not a probability."],
      ["星の航路", "16.26 rank points", "Why: Graph rank #2; positive evidence from Copper Comet | Model rank #1; positive evidence from Copper Comet. Rank points combine relative positions; they are not a probability."],
      ["Quiet Satellite", "7.94 rank points", "Why: Graph: no candidate | Model rank #3; positive evidence from Copper Comet. Rank points combine relative positions; they are not a probability."],
      ["Café Nebula", "7.81 rank points", "Why: Graph: no candidate | Model rank #4; positive evidence from Copper Comet. Rank points combine relative positions; they are not a probability."],
      ["Glass Orchard", "7.69 rank points", "Why: Graph: no candidate | Model rank #5; positive evidence from Copper Comet. Rank points combine relative positions; they are not a probability."],
      ["Paper Current", "7.58 rank points", "Why: Graph: no candidate | Model rank #6; negative evidence from Copper Comet. Rank points combine relative positions; they are not a probability."],
      ["Ashen Harbor", "7.46 rank points", "Why: Graph: no candidate | Model rank #7; negative evidence from Copper Comet. Rank points combine relative positions; they are not a probability."],
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
        item.querySelector(".rec-why")?.textContent,
      ]),
    );
    expect(cards).toEqual(expected[mode]);
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
  await expect(page.locator("#rec-results .rec-why img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __unsafe?: number }).__unsafe)).toBeUndefined();
});
