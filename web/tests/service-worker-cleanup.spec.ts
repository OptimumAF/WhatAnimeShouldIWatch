import { expect, test } from "@playwright/test";

test("loading this app leaves another same-origin app's service worker registered", async ({ page }) => {
  await page.addInitScript(() => {
    const state = window as typeof window & { foreignUnregisterCalls: number };
    state.foreignUnregisterCalls = 0;
    Object.defineProperty(navigator, "serviceWorker", {
      configurable: true,
      value: {
        getRegistrations: async () => [{
          scope: `${window.location.origin}/other-app/`,
          unregister: async () => {
            state.foreignUnregisterCalls += 1;
            return true;
          },
        }],
      },
    });
  });

  await page.goto("/");
  await expect(page.getByText("SYNTHETIC DEMO DATA")).toBeVisible();
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 0)));
  const calls = await page.evaluate(
    () => (window as typeof window & { foreignUnregisterCalls: number }).foreignUnregisterCalls,
  );
  expect(calls).toBe(0);
});
