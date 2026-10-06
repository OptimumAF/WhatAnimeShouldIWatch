/** Synthetic built-demo browser benchmark. No external request is allowed through. */
import { chromium } from "@playwright/test";
import { preview } from "vite";
import { gzipSync } from "node:zlib";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { resolve, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { cpus, platform, release, totalmem } from "node:os";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { createHash } from "node:crypto";

const webRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const repoRoot = resolve(webRoot, "..");
const distRoot = join(webRoot, "dist");
const scenarioFlag = process.argv.find((item) => item.startsWith("--scenario="));
const scenarioName = scenarioFlag ? scenarioFlag.slice("--scenario=".length) : "fixture";
if (scenarioName !== "fixture" && scenarioName !== "scale") {
  throw new Error("--scenario must be fixture or scale.");
}
const outputPath = join(webRoot, "test-results",
  scenarioName === "scale" ? "performance-scale.json" : "performance-baseline.json");
const runs = readPositiveInteger("--runs", 3);
const updates = readPositiveInteger("--updates", 8);
const graphRenders = readPositiveInteger("--graph-renders", 4);
const profiles = [
  { name: "desktop", viewport: { width: 1365, height: 768 }, deviceScaleFactor: 1,
    cpuRate: 1, latencyMs: 0, downBytesPerSecond: -1, upBytesPerSecond: -1 },
  { name: "mobile", viewport: { width: 390, height: 844 }, deviceScaleFactor: 3,
    cpuRate: 4, latencyMs: 150, downBytesPerSecond: 1_600_000 / 8,
    upBytesPerSecond: 750_000 / 8 },
];

function readPositiveInteger(flag, fallback) {
  const argument = process.argv.find((item) => item.startsWith(`${flag}=`));
  if (!argument) return fallback;
  const value = Number(argument.slice(flag.length + 1));
  if (!Number.isSafeInteger(value) || value < 1 || value > 30) {
    throw new Error(`${flag} must be an integer from 1 to 30.`);
  }
  return value;
}

function metricSummary(values) {
  if (!values.length) return null;
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error("A required performance measurement is missing or nonfinite.");
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return {
    count: sorted.length,
    median: round(sorted.length % 2 === 0
      ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]),
    p95: round(sorted[Math.ceil(sorted.length * 0.95) - 1]),
    max: round(sorted.at(-1)),
  };
}

function round(value) { return Math.round(value * 100) / 100; }

function measureEntries(measures, name) {
  return measures.filter((item) => item.name === name).map((item) => item.duration);
}

function phaseSummaries(samples, names, measuresFor, longTasksFor) {
  return Object.fromEntries(names.map((name) => {
    const measured = samples.flatMap((sample) =>
      measuresFor(sample).filter((item) => item.name === name));
    const overlaps = samples.reduce((sum, sample) => {
      const phaseMeasures = measuresFor(sample).filter((item) => item.name === name);
      return sum + longTasksFor(sample).filter((task) => phaseMeasures.some((item) =>
        task.startTime < item.startTime + item.duration &&
        task.startTime + task.duration > item.startTime)).length;
    }, 0);
    return [name.slice("wasiw:".length), {
      durationMs: metricSummary(measured.map((item) => item.duration)),
      longTaskOverlaps: overlaps,
    }];
  }));
}

async function pageMetrics(page, session) {
  const browserMetrics = await page.evaluate(() => ({
    measures: performance.getEntriesByType("measure")
      .filter((item) => item.name.startsWith("wasiw:"))
      .map((item) => ({ name: item.name, startTime: item.startTime, duration: item.duration })),
    resources: [...performance.getEntriesByType("navigation"),
      ...performance.getEntriesByType("resource")].map((item) => ({
      path: new URL(item.name).pathname,
      transferBytes: item.transferSize,
      encodedBodyBytes: item.encodedBodySize,
      decodedBodyBytes: item.decodedBodySize,
    })),
    longTasks: window.__wasiwLongTasks ?? [],
    longTaskSupported: PerformanceObserver.supportedEntryTypes.includes("longtask"),
  }));
  const cdp = await session.send("Performance.getMetrics");
  const metric = new Map(cdp.metrics.map((item) => [item.name, item.value]));
  return {
    ...browserMetrics,
    heapUsedBytes: metric.get("JSHeapUsedSize") ?? null,
    heapTotalBytes: metric.get("JSHeapTotalSize") ?? null,
    domNodes: metric.get("Nodes") ?? null,
  };
}

async function clearMeasures(page) {
  await page.evaluate(() => {
    performance.clearMeasures();
    window.__wasiwLongTasks = [];
  });
}

async function waitForMeasure(page, name, previousCount) {
  await page.waitForFunction(([metricName, count]) =>
    performance.getEntriesByName(metricName, "measure").length > count,
  [name, previousCount], { timeout: 15_000 });
  return page.evaluate((metricName) => {
    const entries = performance.getEntriesByName(metricName, "measure");
    return entries.at(-1).duration;
  }, name);
}

async function measureCount(page, name) {
  return page.evaluate((metricName) =>
    performance.getEntriesByName(metricName, "measure").length, name);
}

function attachNetwork(session, baseOrigin) {
  const responses = new Map();
  const servedFromCache = new Set();
  const phases = { cold: [], warm: [], interaction: [] };
  let phase = "cold";
  session.on("Network.requestServedFromCache", ({ requestId }) => {
    servedFromCache.add(requestId);
  });
  session.on("Network.responseReceived", ({ requestId, response }) => {
    if (new URL(response.url).origin !== baseOrigin) return;
    const headers = Object.fromEntries(Object.entries(response.headers)
      .map(([key, value]) => [key.toLowerCase(), value]));
    responses.set(requestId, {
      path: new URL(response.url).pathname,
      status: response.status,
      fromDiskCache: response.fromDiskCache,
      contentEncoding: headers["content-encoding"] ?? "identity",
    });
  });
  session.on("Network.loadingFinished", ({ requestId, encodedDataLength }) => {
    const response = responses.get(requestId);
    if (response) phases[phase].push({ ...response, wireBytes: encodedDataLength,
      servedFromCache: servedFromCache.has(requestId) });
    responses.delete(requestId);
    servedFromCache.delete(requestId);
  });
  return { phases, setPhase: (value) => { phase = value; } };
}

async function localAssetSizes(paths) {
  const seen = new Set(paths);
  const rows = [];
  for (const pathname of seen) {
    const relativePath = pathname === "/" ? "index.html" : pathname.replace(/^\//, "");
    const fullPath = resolve(distRoot, relativePath);
    if (!fullPath.startsWith(`${distRoot}${sep}`)) continue;
    try {
      const bytes = await readFile(fullPath);
      rows.push({ path: pathname, plainBytes: bytes.byteLength,
        gzipBytes: gzipSync(bytes).byteLength });
    } catch {
      // Runtime-only or unsuccessful paths are absent from the built file inventory.
    }
  }
  return rows.sort((left, right) => left.path.localeCompare(right.path));
}

async function oneRun(browser, baseUrl, profile, proxy) {
  const context = await browser.newContext({
    viewport: profile.viewport,
    deviceScaleFactor: profile.deviceScaleFactor,
    serviceWorkers: "block",
  });
  const blockedBefore = proxy.blockedRequests;
  const page = await context.newPage();
  await page.addInitScript(() => {
    window.__wasiwLongTasks = [];
    if (PerformanceObserver.supportedEntryTypes.includes("longtask")) {
      new PerformanceObserver((list) => {
        for (const task of list.getEntries()) {
          window.__wasiwLongTasks.push({ startTime: task.startTime, duration: task.duration });
        }
      }).observe({ type: "longtask", buffered: true });
    }
  });
  const session = await context.newCDPSession(page);
  await session.send("Network.enable");
  await session.send("Network.setCacheDisabled", { cacheDisabled: true });
  await session.send("Network.emulateNetworkConditions", {
    offline: false, latency: profile.latencyMs,
    downloadThroughput: profile.downBytesPerSecond,
    uploadThroughput: profile.upBytesPerSecond,
  });
  await session.send("Emulation.setCPUThrottlingRate", { rate: profile.cpuRate });
  await session.send("Performance.enable");
  const network = attachNetwork(session, new URL(baseUrl).origin);
  try {
    await page.goto(`${baseUrl}?perf=1`, { waitUntil: "load", timeout: 60_000 });
    await page.locator(".demo-banner").waitFor({ timeout: 15_000 }).catch(() => {
      throw new Error("Built app is not the synthetic demo; run npm run bench:browser:fixture.");
    });
    await page.locator("#rec-summary").getByText(/Showing top/).waitFor({ timeout: 60_000 });
    await page.waitForLoadState("networkidle");
    const cold = await pageMetrics(page, session);
    if (!cold.longTaskSupported) throw new Error("Long Tasks API is unavailable in this browser.");
    const firstRecommendation = cold.measures.find((item) => item.name === "wasiw:recommendation:update");
    if (!firstRecommendation) throw new Error("First-view recommendation timing is missing.");
    const coldFirstViewMs = firstRecommendation.startTime + firstRecommendation.duration;

    network.setPhase("warm");
    await session.send("Network.setCacheDisabled", { cacheDisabled: false });
    await page.reload({ waitUntil: "load", timeout: 60_000 });
    await page.locator("#rec-results .rec-item").first().waitFor({ timeout: 60_000 });
    await page.waitForLoadState("networkidle");
    const warm = await pageMetrics(page, session);
    const warmFirstRecommendation = warm.measures.find((item) =>
      item.name === "wasiw:recommendation:update");
    if (!warmFirstRecommendation) throw new Error("Warm-view recommendation timing is missing.");
    const warmFirstViewMs = warmFirstRecommendation.startTime + warmFirstRecommendation.duration;

    network.setPhase("interaction");
    await clearMeasures(page);
    await page.locator("#quickstart-favorites").click();
    await page.locator("#anime-input").fill("Copper Comet");
    const beforeFavorite = await measureCount(page, "wasiw:recommendation:update");
    await page.locator("#add-anime-form button").click();
    await waitForMeasure(page, "wasiw:recommendation:update", beforeFavorite);
    await page.locator("#rec-results .rec-item").first().waitFor();
    await clearMeasures(page);
    await page.locator("#advanced-recommendation-settings summary").click();
    const rankUpdates = { graph: [], model: [], hybrid: [] };
    for (const mode of ["graph", "model", "hybrid"]) {
      if (mode !== "graph") {
        const count = await measureCount(page, "wasiw:recommendation:update");
        await page.locator("#rec-method").selectOption(mode);
        await waitForMeasure(page, "wasiw:recommendation:update", count);
        await page.locator("#rec-engine-status").getByText(
          mode === "model" ? /Using ML model recommendations/ : /Using hybrid recommendations/,
        ).waitFor({ timeout: 15_000 });
      }
      for (let index = 0; index < updates; index += 1) {
        const count = await measureCount(page, "wasiw:recommendation:update");
        await page.locator("#allow-related-titles").setChecked(index % 2 === 0);
        rankUpdates[mode].push(await waitForMeasure(page, "wasiw:recommendation:update", count));
      }
    }
    const ranking = await pageMetrics(page, session);
    await clearMeasures(page);
    const graphTimes = [];
    const navStart = await page.evaluate(() => performance.now());
    await page.locator("#nav-network").click();
    graphTimes.push(await waitForMeasure(page, "wasiw:network:render", 0));
    const graphFirst = await pageMetrics(page, session);
    const firstGraphEntry = graphFirst.measures.find((item) => item.name === "wasiw:network:render");
    const networkReadyMs = firstGraphEntry.startTime + firstGraphEntry.duration - navStart;
    if (await page.locator("#network-mobile-toggle").isVisible()) {
      await page.locator("#network-mobile-toggle").click();
    }
    const userEdgesAvailable = await page.locator("#toggle-users").isEnabled();
    const renderToggle = page.locator(userEdgesAvailable ? "#toggle-users" : "#toggle-anime-edges");
    for (let index = 1; index < graphRenders; index += 1) {
      const count = await measureCount(page, "wasiw:network:render");
      await renderToggle.setChecked(userEdgesAvailable ? index % 2 === 1 : index % 2 === 0);
      graphTimes.push(await waitForMeasure(page, "wasiw:network:render", count));
    }
    const graph = await pageMetrics(page, session);
    const resourcePaths = [...cold.resources, ...warm.resources,
      ...graph.resources].map((item) => item.path);
    return {
      cold: { firstViewMs: coldFirstViewMs, ...cold,
        network: network.phases.cold },
      warm: { firstViewMs: warmFirstViewMs, ...warm,
        network: network.phases.warm },
      rankUpdates,
      rankingMeasures: ranking.measures,
      rankingLongTasks: ranking.longTasks,
      graph: { firstNavigationMs: networkReadyMs, renderMs: graphTimes,
        rerenderControl: userEdgesAvailable ? "sampled user edges" : "aggregate pair edges",
        heapUsedBytes: graph.heapUsedBytes, domNodes: graph.domNodes,
        longTasks: graph.longTasks, measures: graph.measures,
        network: network.phases.interaction },
      assetSizes: await localAssetSizes(resourcePaths),
      blockedExternalRequests: proxy.blockedRequests - blockedBefore,
    };
  } finally {
    await context.close();
  }
}

function summarizeProfile(samples) {
  const values = (selector) => samples.map(selector);
  const flatten = (selector) => samples.flatMap(selector);
  const wire = (phase) => values((sample) =>
    sample[phase].network.reduce((sum, item) => sum + item.wireBytes, 0));
  const body = (phase, field) => values((sample) =>
    sample[phase].resources.reduce((sum, item) => sum + item[field], 0));
  const stalls = (phase) => flatten((sample) => sample[phase].longTasks.map((task) => task.duration));
  return {
    coldFirstViewMs: metricSummary(values((sample) => sample.cold.firstViewMs)),
    warmFirstViewMs: metricSummary(values((sample) => sample.warm.firstViewMs)),
    coldWireBytes: metricSummary(wire("cold")),
    warmWireBytes: metricSummary(wire("warm")),
    coldEncodedBodyBytes: metricSummary(body("cold", "encodedBodyBytes")),
    coldDecodedBodyBytes: metricSummary(body("cold", "decodedBodyBytes")),
    coldHeapUsedBytes: metricSummary(values((sample) => sample.cold.heapUsedBytes)),
    warmHeapUsedBytes: metricSummary(values((sample) => sample.warm.heapUsedBytes)),
    graphHeapUsedBytes: metricSummary(values((sample) => sample.graph.heapUsedBytes)),
    coldLongTasks: { count: flatten((sample) => sample.cold.longTasks).length,
      durationMs: metricSummary(stalls("cold")) },
    warmLongTasks: { count: flatten((sample) => sample.warm.longTasks).length,
      durationMs: metricSummary(stalls("warm")) },
    rankingLongTasks: { count: flatten((sample) => sample.rankingLongTasks).length,
      durationMs: metricSummary(flatten((sample) => sample.rankingLongTasks.map((task) => task.duration))) },
    graphLongTasks: { count: flatten((sample) => sample.graph.longTasks).length,
      durationMs: metricSummary(stalls("graph")) },
    jsonCatalogMs: metricSummary(values((sample) => measureEntries(sample.cold.measures,
      "wasiw:json:catalog")[0])),
    jsonGraphMs: metricSummary(values((sample) => measureEntries(sample.cold.measures,
      "wasiw:json:graph")[0])),
    schemaCatalogMs: metricSummary(values((sample) => measureEntries(sample.cold.measures,
      "wasiw:schema:catalog")[0])),
    schemaGraphMs: metricSummary(values((sample) => measureEntries(sample.cold.measures,
      "wasiw:schema:graph")[0])),
    graphIndexMs: metricSummary(values((sample) => measureEntries(sample.cold.measures,
      "wasiw:index:graph")[0])),
    jsonModelMs: metricSummary(values((sample) => measureEntries(sample.rankingMeasures,
      "wasiw:json:model")[0])),
    schemaModelMs: metricSummary(values((sample) => measureEntries(sample.rankingMeasures,
      "wasiw:schema:model")[0])),
    modelIndexMs: metricSummary(values((sample) => measureEntries(sample.rankingMeasures,
      "wasiw:index:model")[0])),
    jsonExplorerMs: metricSummary(values((sample) => measureEntries(sample.graph.measures,
      "wasiw:json:explorer")[0])),
    schemaExplorerMs: metricSummary(values((sample) => measureEntries(sample.graph.measures,
      "wasiw:schema:explorer")[0])),
    rankUpdateMs: Object.fromEntries(["graph", "model", "hybrid"].map((mode) =>
      [mode, metricSummary(flatten((sample) => sample.rankUpdates[mode]))])),
    recommendationPhases: phaseSummaries(samples, [
      "wasiw:recommendation:graph-score", "wasiw:recommendation:model-score",
      "wasiw:recommendation:eligibility", "wasiw:recommendation:filter-ui",
      "wasiw:recommendation:franchise", "wasiw:recommendation:cards",
      "wasiw:recommendation:dom",
    ], (sample) => sample.rankingMeasures, (sample) => sample.rankingLongTasks),
    graphNavigationMs: metricSummary(values((sample) => sample.graph.firstNavigationMs)),
    graphRenderMs: metricSummary(flatten((sample) => sample.graph.renderMs)),
    graphPhases: phaseSummaries(samples, [
      "wasiw:network:select", "wasiw:network:construct", "wasiw:network:layout",
      "wasiw:network:construct-node-batch", "wasiw:network:construct-edge-batch",
      "wasiw:network:scope", "wasiw:network:svg", "wasiw:network:svg-coordinates",
      "wasiw:network:svg-edge-batch", "wasiw:network:svg-paths",
      "wasiw:network:svg-node-batch", "wasiw:network:svg-node-paths",
      "wasiw:network:svg-dom-commit",
    ], (sample) => sample.graph.measures, (sample) => sample.graph.longTasks),
    blockedExternalRequests: samples.reduce((sum, sample) =>
      sum + sample.blockedExternalRequests, 0),
  };
}

async function rejectingProxy() {
  let blockedRequests = 0;
  const server = createServer((_request, response) => {
    blockedRequests += 1;
    response.writeHead(502);
    response.end();
  });
  server.on("connect", (_request, socket) => {
    blockedRequests += 1;
    socket.write("HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n");
    socket.destroy();
  });
  await new Promise((done) => server.listen(0, "127.0.0.1", done));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Rejecting proxy did not bind.");
  return {
    url: `http://127.0.0.1:${address.port}`,
    get blockedRequests() { return blockedRequests; },
    close: () => new Promise((done, reject) =>
      server.close((error) => error ? reject(error) : done())),
  };
}

await readFile(join(distRoot, "index.html"));
const scenario = scenarioName === "scale"
  ? JSON.parse(await readFile(join(distRoot, "demo-data", "performance-scenario.json"), "utf8"))
  : null;
if (scenarioName === "scale" && scenario.format !== "invented-browser-scale-v1") {
  throw new Error("The built scale scenario is missing or unsupported.");
}
if (scenarioName === "scale") {
  for (const asset of scenario.assets) {
    const bytes = await readFile(join(distRoot, "demo-data", asset.name));
    if (bytes.length !== asset.bytes ||
        createHash("sha256").update(bytes).digest("hex") !== asset.sha256) {
      throw new Error(`The built invented scale asset ${asset.name} changed after generation.`);
    }
  }
}
const server = await preview({
  root: webRoot,
  configFile: join(webRoot, "vite.config.ts"),
  preview: { host: "127.0.0.1", port: 4173, strictPort: true },
});
const address = server.httpServer.address();
if (!address || typeof address === "string") throw new Error("Preview did not bind a TCP port.");
const baseUrl = `http://127.0.0.1:${address.port}/`;
const proxy = await rejectingProxy();
let browser;
try {
  browser = await chromium.launch({ headless: true, args: ["--enable-precise-memory-info"],
    proxy: { server: proxy.url, bypass: "127.0.0.1,localhost" } });
  const samples = {};
  for (const profile of profiles) {
    samples[profile.name] = [];
    for (let run = 0; run < runs; run += 1) {
      samples[profile.name].push(await oneRun(browser, baseUrl, profile, proxy));
    }
  }
  const assetSizes = samples.desktop[0].assetSizes;
  const initialAssetSizes = await localAssetSizes(samples.desktop[0].cold.resources
    .map((item) => item.path));
  const summary = Object.fromEntries(profiles.map((profile) =>
    [profile.name, summarizeProfile(samples[profile.name])]));
  const report = {
    format: "synthetic-browser-performance-v1",
    scenarioName, scenario,
    sourceRevision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot,
      encoding: "utf8" }).trim(),
    workingTreeDirty: execFileSync("git", ["status", "--porcelain"], { cwd: repoRoot,
      encoding: "utf8" }).trim().length > 0,
    generatedAt: new Date().toISOString(),
    build: scenarioName === "scale"
      ? "vite build --mode demo; ignored invented scale assets replace built demo assets; vite preview"
      : "vite build --mode demo; vite preview; invented demo assets only",
    browser: await browser.version(),
    host: { platform: platform(), release: release(), cpu: cpus()[0]?.model ?? "unknown",
      logicalCpus: cpus().length, ramBytes: totalmem() },
    profiles, runs, updatesPerModePerRun: updates,
    graphRendersPerRun: graphRenders,
    graphRerenderControl: samples.desktop[0].graph.rerenderControl,
    initialAssetSizes,
    initialAssetTotals: {
      plainBytes: initialAssetSizes.reduce((sum, item) => sum + item.plainBytes, 0),
      gzipBytes: initialAssetSizes.reduce((sum, item) => sum + item.gzipBytes, 0),
    },
    assetSizes,
    assetTotals: {
      plainBytes: assetSizes.reduce((sum, item) => sum + item.plainBytes, 0),
      gzipBytes: assetSizes.reduce((sum, item) => sum + item.gzipBytes, 0),
    },
    summary, samples,
  };
  await mkdir(resolve(outputPath, ".."), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    format: report.format, sourceRevision: report.sourceRevision,
    scenarioName: report.scenarioName, scenario: report.scenario,
    workingTreeDirty: report.workingTreeDirty,
    browser: report.browser, host: report.host, profiles: report.profiles,
    runs: report.runs, initialAssetTotals: report.initialAssetTotals,
    graphRerenderControl: report.graphRerenderControl,
    assetTotals: report.assetTotals, summary: report.summary,
    output: relative(repoRoot, outputPath),
  }, null, 2)}\n`);
} finally {
  await browser?.close();
  await proxy.close();
  await new Promise((done, reject) => server.httpServer.close((error) => error ? reject(error) : done()));
}
