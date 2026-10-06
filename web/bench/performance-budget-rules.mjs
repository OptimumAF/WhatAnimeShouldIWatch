/** Pure validation for the local invented browser budget reports. */
export function checkPerformanceBudgets(fixture, scale, {
  currentRevision, currentWorkingTreeDirty,
}) {
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
  requireCondition(currentWorkingTreeDirty === false, "reports: current checkout is dirty");
  requireCondition(fixture.sourceRevision === currentRevision &&
    scale.sourceRevision === currentRevision, "reports: stale source revision");
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
  for (const [name, report] of [["fixture", fixture], ["scale", scale]]) {
    for (const profile of ["desktop", "mobile"]) {
      const samples = report.samples?.[profile];
      requireCondition(report.summary?.[profile]?.blockedExternalRequests === 0 &&
        Array.isArray(samples) && samples.length === report.runs &&
        samples.every((sample) => sample.blockedExternalRequests === 0),
      `${name}: ${profile} external requests or incomplete samples`);
    }
  }
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
  return {
    fixtureMobileColdFirstViewP95Ms: fixtureMobile?.coldFirstViewMs?.p95 ?? null,
    scaleMobileColdFirstViewP95Ms: scaleMobile?.coldFirstViewMs?.p95 ?? null,
    scaleMobileWarmUpdateP95Ms: Object.fromEntries(["graph", "model", "hybrid"]
      .map((mode) => [mode, scaleMobile?.rankUpdateMs?.[mode]?.p95 ?? null])),
    firstViewBudgetMs, warmUpdateP95BudgetMs,
    status: failures.length ? "failed" : "passed", failures,
  };
}
