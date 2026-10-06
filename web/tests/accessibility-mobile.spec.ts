import { expect, test, type Page } from "@playwright/test";

async function blockExternalRequests(page: Page): Promise<void> {
  await page.route("**/*", (route) => {
    const host = new URL(route.request().url()).hostname;
    return host === "localhost" || host === "127.0.0.1"
      ? route.continue() : route.abort();
  });
}

async function openDemo(page: Page): Promise<void> {
  await blockExternalRequests(page);
  await page.goto("/");
  await expect(page.getByText("SYNTHETIC DEMO DATA")).toBeVisible({ timeout: 15_000 });
}

async function horizontalOverflow(page: Page): Promise<{ width: number; viewport: number; offenders: string[] }> {
  return page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    viewport: innerWidth,
    offenders: [...document.querySelectorAll<HTMLElement>("body *")]
      .filter((element) => element.getClientRects().length > 0 &&
        element.getBoundingClientRect().right > innerWidth + 1)
      .slice(0, 10).map((element) => `${element.tagName.toLowerCase()}#${element.id}.${element.className}`),
  }));
}

async function tabTo(page: Page, selector: string, limit = 90): Promise<void> {
  for (let index = 0; index < limit; index += 1) {
    await page.keyboard.press("Tab");
    if (await page.evaluate((target) => document.activeElement?.matches(target) ?? false, selector)) return;
  }
  throw new Error(`Keyboard focus did not reach ${selector} after ${limit} tabs.`);
}

test("keyboard-only favorite and shortlist flow announces the saved result", async ({ page }) => {
  await openDemo(page);
  await page.keyboard.press("Alt+/");
  await expect(page.locator("#anime-input")).toBeFocused();
  await page.keyboard.type("Copper Comet");
  await page.keyboard.press("Tab");
  await expect(page.locator("#add-preference")).toBeFocused();
  await page.keyboard.press("l");
  await expect(page.locator("#add-preference")).toHaveValue("liked");
  await tabTo(page, "#add-anime-form button");
  await page.keyboard.press("Enter");
  await expect(page.locator("#rec-results .rec-title")).toContainText(["Moonlit Workshop"]);
  await tabTo(page, "#rec-results [data-card-action='save']");
  await page.keyboard.press("Enter");
  await expect(page.locator("#rec-action-status")).toHaveAttribute("role", "status");
  await expect(page.locator("#rec-action-status")).toContainText("Moonlit Workshop");
  await expect(page.locator("#rec-action-status")).toBeFocused();
  await expect(page.locator("#watchlist-list")).toContainText("Moonlit Workshop");
});

test("the visible graph has a bounded keyboard list instead of dozens of SVG tab stops", async ({ page }) => {
  const anime = Array.from({ length: 55 }, (_, index) => [index + 101, `Invented Node ${String(index + 1).padStart(2, "0")}`]);
  await blockExternalRequests(page);
  await page.route("**/demo-data/graph-explorer.aggregate.compact.json", async (route) => {
    const response = await route.fetch();
    const graph = await response.json();
    graph.anime = anime;
    graph.aa = anime.slice(1).map((_, index) => [0, index + 1, 0.5, 1]);
    graph.animeCount = anime.length;
    graph.nodeCount = graph.userCount + anime.length;
    graph.edgeCount = graph.ua.length + graph.aa.length;
    graph.truncation.candidatePairs = graph.aa.length;
    graph.truncation.eligiblePairs = graph.aa.length;
    graph.truncation.selectedPairs = graph.aa.length;
    graph.truncation.excludedBySupport = 0;
    graph.truncation.excludedByNeighborLimit = 0;
    graph.truncation.excludedByOutputLimit = 0;
    graph.visualization.maxAnimeAnimeEdges = graph.aa.length;
    graph.visualization.excludedAnimeAnimeEdges = 0;
    await route.fulfill({ response, json: graph });
  });
  await page.goto("/");
  await expect(page.getByText("SYNTHETIC DEMO DATA")).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await page.locator("#min-weight").evaluate((input: HTMLInputElement) => {
    input.value = "0";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.locator("#graph svg circle")).toHaveCount(55);
  await expect(page.locator("#graph svg circle[tabindex='0']")).toHaveCount(0);
  await page.locator("#network-node-list summary").click();
  await expect(page.locator("#network-node-list-results button")).toHaveCount(15);
  await expect(page.locator("#network-node-list-status")).toContainText("55 visible nodes");
  await page.locator("#network-node-filter").fill("Invented Node 55");
  await expect(page.locator("#network-node-list-results button")).toHaveText(["Invented Node 55"]);
  await page.locator("#network-node-list-results button").press("Enter");
  await expect(page.locator("#inspect-meta")).toContainText("Invented Node 55");
  await expect(page.locator("#network-search-message")).toContainText("Invented Node 55");
  await expect(page.locator("#network-node-list-results button")).toBeFocused();
  await expect(page.locator("#network-node-list-results button")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#network-node-filter").fill("");
  await page.locator("#network-node-next").click();
  await expect(page.locator("#network-node-list-status")).toContainText("16–30 of 55");
  await expect(page.locator("#network-node-list-results button")).toHaveCount(15);
  await page.locator("#network-node-next").click();
  await page.locator("#network-node-next").click();
  await expect(page.locator("#network-node-list-status")).toContainText("46–55 of 55");
  await expect(page.locator("#network-node-list-results button")).toHaveCount(10);
  await expect(page.locator("#network-node-next")).toBeDisabled();
  await page.locator("#network-node-filter").fill("no such invented node");
  await expect(page.locator("#network-node-list-status")).toContainText("No nodes match");
  await expect(page.locator("#network-node-list-results button")).toHaveCount(0);
});

test("a keyboard-opened command palette returns focus to its opener", async ({ page }) => {
  await openDemo(page);
  await page.locator("#anime-input").focus();
  await page.keyboard.press("Control+k");
  await expect(page.locator("#command-input")).toBeFocused();
  await expect(page.locator("#command-palette")).toHaveAttribute("aria-hidden", "false");
  await expect(page.locator(".topbar")).toHaveAttribute("inert", "");
  await page.keyboard.press("Shift+Tab");
  await expect(page.locator("#command-close")).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(page.locator("#command-palette")).toContainText("Quick Actions");
  expect(await page.evaluate(() => document.activeElement?.closest("#command-palette") !== null)).toBe(true);
  await page.keyboard.press("Escape");
  await expect(page.locator("#command-palette")).toHaveAttribute("aria-hidden", "true");
  await expect(page.locator("#anime-input")).toBeFocused();
  await expect(page.locator(".topbar")).not.toHaveAttribute("inert", "");
});

test("palette navigation moves focus out of the view it hides", async ({ page }) => {
  await openDemo(page);
  await page.locator("#anime-input").focus();
  await page.keyboard.press("Control+k");
  await page.locator("#command-input").fill("Open Network Explorer");
  await page.keyboard.press("Enter");
  await expect(page.locator("#view-network")).toBeVisible();
  await expect(page.locator("#nav-network")).toBeFocused();

  await page.locator("#network-search-input").focus();
  await page.keyboard.press("Control+k");
  await page.locator("#command-input").fill("Open Recommendations");
  await page.keyboard.press("Enter");
  await expect(page.locator("#view-recommendations")).toBeVisible();
  await expect(page.locator("#nav-recommendations")).toBeFocused();
});

test("ambiguous keyboard title choice restores the input after Escape", async ({ page }) => {
  await openDemo(page);
  await page.keyboard.press("Alt+/");
  await page.keyboard.type("Copper");
  await page.keyboard.press("Enter");
  await expect(page.locator("#title-search-dialog")).toBeVisible();
  await expect(page.locator("#title-search-dialog .title-search-choice").first()).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.locator("#title-search-dialog")).toBeHidden();
  await expect(page.locator("#anime-input")).toBeFocused();
  await expect(page.locator("#watched-count")).toHaveText("0");
});

test("mobile controls remain usable at narrow width, zoom, high contrast, and reduced motion", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 720 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openDemo(page);
  await expect(page.locator(".topbar")).toHaveCSS("position", "relative");
  await page.getByRole("button", { name: "Switch to high contrast mode" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-contrast", "high");
  const initialTheme = await page.locator("html").getAttribute("data-theme");
  await page.locator("#theme-toggle").click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", initialTheme === "light" ? "dark" : "light");
  const initialOverflow = await horizontalOverflow(page);
  expect(initialOverflow.width, initialOverflow.offenders.join(", "))
    .toBeLessThanOrEqual(initialOverflow.viewport);
  const reducedTransition = await page.locator("#theme-toggle").evaluate((element) =>
    getComputedStyle(element).transitionDuration);
  expect(reducedTransition.split(",").every((part) => Number.parseFloat(part) < 0.01)).toBe(true);
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#nav-network")).toHaveAttribute("aria-current", "page");
  await expect(page.locator("#network-mobile-toggle")).toBeVisible();
  await page.locator("#network-mobile-toggle").click();
  await expect(page.locator("#network-mobile-toggle")).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator("#network-search-input")).toBeVisible();
  const networkOverflow = await horizontalOverflow(page);
  expect(networkOverflow.width, networkOverflow.offenders.join(", "))
    .toBeLessThanOrEqual(networkOverflow.viewport);
  const targetSize = await page.locator("#network-mobile-toggle").boundingBox();
  expect(targetSize?.height).toBeGreaterThanOrEqual(44);
});

test("compact controls hand focus to the reveal button when a resize hides them", async ({ page }) => {
  await page.setViewportSize({ width: 1100, height: 800 });
  await openDemo(page);
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await expect(page.locator("#network-search-input")).toBeVisible();
  await page.locator("#network-search-input").focus();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("#network-mobile-toggle")).toBeFocused();
  await expect(page.locator("#network-mobile-toggle")).toHaveAttribute("aria-expanded", "false");
  await page.keyboard.press("Enter");
  await expect(page.locator("#network-search-input")).toBeVisible();
  await expect(page.locator("#network-mobile-toggle")).toHaveAttribute("aria-expanded", "true");
});

test("normal and high-contrast controls remain readable in light and dark themes", async ({ page }) => {
  await openDemo(page);
  for (const theme of ["dark", "light"] as const) {
    if (await page.locator("html").getAttribute("data-theme") !== theme) {
      await page.locator("#theme-toggle").click();
    }
    for (const contrast of ["normal", "high"] as const) {
      if (await page.locator("html").getAttribute("data-contrast") !== contrast) {
        await page.locator("#contrast-toggle").click();
      }
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
      await expect(page.locator("html")).toHaveAttribute("data-contrast", contrast);
      const readContrast = () => page.locator("#theme-toggle").evaluate((element) => {
        const style = getComputedStyle(element);
        const channels = (value: string) => (value.match(/[\d.]+/g) ?? []).slice(0, 3)
          .map((part) => Number(part) / (value.startsWith("color(srgb") ? 1 : 255))
          .map((channel) => channel <= 0.04045
            ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
        const luminance = (value: string) => {
          const [red, green, blue] = channels(value);
          return red * 0.2126 + green * 0.7152 + blue * 0.0722;
        };
        const foreground = luminance(style.color);
        const background = luminance(style.backgroundColor);
        return { foreground: style.color, background: style.backgroundColor,
          ratio: (Math.max(foreground, background) + 0.05) /
            (Math.min(foreground, background) + 0.05) };
      });
      await expect.poll(async () => (await readContrast()).ratio,
        { message: `${theme}/${contrast} toolbar contrast after the theme transition` })
        .toBeGreaterThanOrEqual(4.5);
    }
  }
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.locator("html")).toHaveAttribute("data-contrast", "high");
});

test("primary touch controls have 44-pixel targets on a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openDemo(page);
  await page.locator("#anime-input").fill("Copper Comet");
  await page.locator("#add-preference").selectOption("liked");
  await page.locator("#add-anime-form button").click();
  await expect(page.locator("#rec-results .rec-actions").first()).toBeVisible();
  const controls = ["#anime-input", "#add-preference", "#add-anime-form button",
    "#quickstart-favorites", "#rec-results .rec-actions button:first-child",
    "#rec-results .rec-actions button:nth-child(2)"];
  for (const selector of controls) {
    const bounds = await page.locator(selector).first().boundingBox();
    expect(bounds?.height, selector).toBeGreaterThanOrEqual(44);
    expect(bounds?.width, selector).toBeGreaterThanOrEqual(44);
  }
  await page.getByRole("button", { name: "Open network explorer page" }).click();
  await page.locator("#network-mobile-toggle").click();
  await page.locator("#network-node-list summary").click();
  for (const selector of ["#network-mobile-toggle", "#network-search-input", "#min-weight",
    "#network-search-form button", "#network-node-list summary",
    "#network-node-filter", "#network-node-list-results button:first-child"] as const) {
    const bounds = await page.locator(selector).first().boundingBox();
    expect(bounds?.height, selector).toBeGreaterThanOrEqual(44);
    expect(bounds?.width, selector).toBeGreaterThanOrEqual(44);
  }
});
