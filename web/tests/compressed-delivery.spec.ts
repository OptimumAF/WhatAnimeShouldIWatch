import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { expect, test, type Page } from "@playwright/test";

const normalAppUrl = "http://127.0.0.1:5174/";
const graphBytes = readFileSync(new URL("../public/demo-data/graph.compact.json", import.meta.url));
const compressedGraph = gzipSync(graphBytes);

async function routeLegacyGraph(page: Page, gzipBody: Buffer, withPlain: boolean,
  browserDecoded = false) {
  const requested: string[] = [];
  await page.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/data/")) {
      requested.push(url.pathname);
      if (url.pathname === "/data/graph.compact.json.gz") {
        return route.fulfill({ body: gzipBody, headers: browserDecoded
          ? { "Content-Type": "application/json", "Content-Encoding": "gzip" }
          : { "Content-Type": "application/gzip" } });
      }
      if (url.pathname === "/data/graph.compact.json" && withPlain) {
        return route.fulfill({ body: graphBytes, contentType: "application/json" });
      }
      return route.fulfill({ status: 404, body: "" });
    }
    if (url.hostname === "api.jikan.moe") {
      return route.fulfill({ contentType: "application/json", body: '{"data":[]}' });
    }
    if (url.hostname !== "127.0.0.1" && url.hostname !== "localhost") return route.abort();
    return route.continue();
  });
  return requested;
}

test("gzip-only normal-mode graph loads and does not request a plain copy", async ({ page }) => {
  const requested = await routeLegacyGraph(page, compressedGraph, false);
  await page.goto(normalAppUrl);
  await expect(page.locator("#diagnostic-data")).toContainText("graph-compact-v2");
  expect(requested).toEqual(["/data/active.json", "/data/graph.compact.json.gz"]);
});

test("a mocked browser-decoded gzip body reaches the loader as JSON", async ({ page }) => {
  // Route fulfillment supplies the browser-exposed body after HTTP decoding.
  const requested = await routeLegacyGraph(page, graphBytes, false, true);
  await page.goto(normalAppUrl);
  await expect(page.locator("#diagnostic-data")).toContainText("graph-compact-v2");
  const prefix = await page.evaluate(async () => {
    const bytes = new Uint8Array(await (await fetch("/data/graph.compact.json.gz")).arrayBuffer());
    return [...bytes.slice(0, 2)];
  });
  expect(prefix).toEqual([...graphBytes.subarray(0, 2)]);
  expect(requested).not.toContain("/data/graph.compact.json");
});

test("unsupported decompression falls back to plain JSON without a proxy", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "DecompressionStream",
      { configurable: true, value: undefined });
  });
  const requested = await routeLegacyGraph(page, compressedGraph, true);
  await page.goto(normalAppUrl);
  await expect(page.locator("#diagnostic-data")).toContainText("graph-compact-v2");
  expect(requested).toEqual(["/data/active.json", "/data/graph.compact.json.gz",
    "/data/graph.compact.json"]);
});

test("malformed present gzip fails visibly even when a plain copy exists", async ({ page }) => {
  const requested = await routeLegacyGraph(page, Buffer.from([0x1f, 0x8b, 0, 1, 2]), true);
  await page.goto(normalAppUrl);
  await expect(page.locator("#rec-message")).toContainText("graph.compact.json.gz: invalid gzip stream");
  expect(requested).toEqual(["/data/active.json", "/data/graph.compact.json.gz"]);
});
