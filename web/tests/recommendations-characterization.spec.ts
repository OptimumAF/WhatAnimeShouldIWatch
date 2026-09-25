import { expect, test } from "@playwright/test";

test("synthetic recommendation modes retain their visible ranking and explanations", async ({ page }) => {
  await page.goto("/");
  await page.locator("#anime-input").fill("Copper Comet");
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
      ["Moonlit Workshop", "+0.933", "Why+: Copper Comet (+0.355) | Copper Comet (+0.292)Why-: no notable negative contributors."],
      ["星の航路", "+0.500", "Why+: Copper Comet (+0.410)Why-: no notable negative contributors."],
      ["Quiet Satellite", "+0.343", "Why+: Copper Comet (+0.240)Why-: no notable negative contributors."],
      ["Café Nebula", "+0.332", "Why+: Copper Comet (+0.235)Why-: no notable negative contributors."],
      ["Glass Orchard", "+0.291", "Why+: Copper Comet (+0.215)Why-: no notable negative contributors."],
      ["Paper Current", "+0.116", "Why+: no strong positive contributors.Why-: Copper Comet (-0.030)"],
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
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-why").first()).toContainText(unsafeLabel);
  await expect(page.locator("#rec-results .rec-why img")).toHaveCount(0);
  expect(await page.evaluate(() => (window as unknown as { __unsafe?: number }).__unsafe)).toBeUndefined();
});
