/** Local synthetic budget gate; real-data/device acceptance remains a separate review. */
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("../test-results/", import.meta.url)));
const reports = await Promise.all(["performance-baseline.json", "performance-scale.json"]
  .map(async (name) => {
    try {
      return JSON.parse(await readFile(resolve(root, name), "utf8"));
    } catch (error) {
      if (error?.code === "ENOENT") {
        process.stderr.write(`Missing ${name}; run both browser benchmark commands first.\n`);
        process.exit(1);
      }
      throw error;
    }
  }));
const [fixture, scale] = reports;
const failures = [];
function requireCondition(condition, message) {
  if (!condition) failures.push(message);
}
function mobileSummary(report, name) {
  requireCondition(report.format === "synthetic-browser-performance-v1", `${name}: format`);
  requireCondition(report.workingTreeDirty === false,
    `${name}: benchmark source checkout was dirty`);
  requireCondition(report.runs >= 3, `${name}: fewer than 3 cold/warm runs`);
  const profile = report.profiles?.find((item) => item.name === "mobile");
  requireCondition(profile?.cpuRate === 4 && profile?.latencyMs === 150 &&
    profile?.downBytesPerSecond === 200_000 &&
    profile?.viewport?.width === 390, `${name}: mobile profile changed`);
  requireCondition(report.summary?.mobile?.coldFirstViewMs?.count === report.runs,
    `${name}: first-view sample count`);
  return report.summary?.mobile;
}
requireCondition(fixture.scenarioName === "fixture" && fixture.scenario === null,
  "fixture: wrong scenario");
requireCondition(scale.scenarioName === "scale" &&
  scale.scenario?.format === "invented-browser-scale-v1" &&
  scale.scenario?.animeCount === 3_000 && scale.scenario?.userCount === 3_000 &&
  scale.scenario?.ratingsPerUser === 6 && scale.scenario?.selectedPairs >= 30_000 &&
  scale.scenario?.explorerPairs === 8_000 && scale.scenario?.modelFactors === 16,
"scale: declared invented workload changed");
requireCondition(fixture.sourceRevision === scale.sourceRevision &&
  fixture.browser === scale.browser, "reports: different source revisions or browsers");
const fixtureMobile = mobileSummary(fixture, "fixture");
const scaleMobile = mobileSummary(scale, "scale");
const firstViewBudgetMs = 3_000;
const warmUpdateP95BudgetMs = 200;
for (const [name, summary] of [["fixture", fixtureMobile], ["scale", scaleMobile]]) {
  const measured = summary?.coldFirstViewMs?.p95;
  requireCondition(Number.isFinite(measured) && measured <= firstViewBudgetMs,
    `${name}: mobile cold first-view p95 ${measured ?? "missing"} ms exceeds ${firstViewBudgetMs} ms`);
}
for (const mode of ["graph", "model", "hybrid"]) {
  const metric = scaleMobile?.rankUpdateMs?.[mode];
  requireCondition(metric?.count >= scale.runs * 8,
    `scale: insufficient ${mode} warm-update samples`);
  requireCondition(Number.isFinite(metric?.p95) && metric.p95 <= warmUpdateP95BudgetMs,
    `scale: ${mode} warm-update p95 ${metric?.p95 ?? "missing"} ms exceeds ${warmUpdateP95BudgetMs} ms`);
}
const result = {
  fixtureMobileColdFirstViewP95Ms: fixtureMobile?.coldFirstViewMs?.p95 ?? null,
  scaleMobileColdFirstViewP95Ms: scaleMobile?.coldFirstViewMs?.p95 ?? null,
  scaleMobileWarmUpdateP95Ms: Object.fromEntries(["graph", "model", "hybrid"]
    .map((mode) => [mode, scaleMobile?.rankUpdateMs?.[mode]?.p95 ?? null])),
  firstViewBudgetMs, warmUpdateP95BudgetMs,
  status: failures.length ? "failed" : "passed", failures,
};
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
if (failures.length) process.exitCode = 1;
