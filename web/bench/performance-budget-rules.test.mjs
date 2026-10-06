import assert from "node:assert/strict";
import test from "node:test";
import { checkPerformanceBudgets } from "./performance-budget-rules.mjs";

const revision = "invented-clean-revision";
function report(scenarioName) {
  const scale = scenarioName === "scale";
  return {
    format: "synthetic-browser-performance-v1",
    scenarioName,
    scenario: scale ? {
      format: "invented-browser-scale-v1", animeCount: 3_000, userCount: 3_000,
      ratingsPerUser: 6, selectedPairs: 32_886, explorerPairs: 8_000, modelFactors: 16,
    } : null,
    sourceRevision: revision,
    workingTreeDirty: false,
    browser: "invented-browser",
    runs: 3,
    profiles: [{ name: "mobile", cpuRate: 4, latencyMs: 150,
      downBytesPerSecond: 200_000, viewport: { width: 390 } }],
    samples: {
      desktop: Array.from({ length: 3 }, () => ({ blockedExternalRequests: 0 })),
      mobile: Array.from({ length: 3 }, () => ({ blockedExternalRequests: 0 })),
    },
    summary: {
      desktop: { blockedExternalRequests: 0 },
      mobile: {
        blockedExternalRequests: 0,
        coldFirstViewMs: { count: 3, p95: scale ? 2_700 : 1_300 },
        rankUpdateMs: Object.fromEntries(["graph", "model", "hybrid"]
          .map((mode) => [mode, { count: 24, p95: 100 }])),
      },
    },
  };
}
function check(fixture = report("fixture"), scale = report("scale"),
  currentWorkingTreeDirty = false) {
  return checkPerformanceBudgets(fixture, scale, {
    currentRevision: revision, currentWorkingTreeDirty,
  });
}

test("clean invented reports for the current revision pass the local budget gate", () => {
  assert.deepEqual(check().failures, []);
});

test("a stale report or dirty current checkout cannot pass the local gate", () => {
  const staleFixture = report("fixture");
  const staleScale = report("scale");
  staleFixture.sourceRevision = staleScale.sourceRevision = "older-invented-revision";
  assert.ok(check(staleFixture, staleScale).failures.includes("reports: stale source revision"));
  assert.ok(check(report("fixture"), report("scale"), true).failures
    .includes("reports: current checkout is dirty"));
});

test("an attempted external request or missing sample cannot pass the local gate", () => {
  const fixture = report("fixture");
  fixture.samples.desktop[1].blockedExternalRequests = 1;
  assert.ok(check(fixture).failures
    .includes("fixture: desktop external requests or incomplete samples"));
  const scale = report("scale");
  scale.summary.mobile.blockedExternalRequests = 1;
  assert.ok(check(report("fixture"), scale).failures
    .includes("scale: mobile external requests or incomplete samples"));
  scale.summary.mobile.blockedExternalRequests = 0;
  scale.samples.mobile.pop();
  assert.ok(check(report("fixture"), scale).failures
    .includes("scale: mobile external requests or incomplete samples"));
});

test("the existing first-view and warm-update limits still fail above budget", () => {
  const fixture = report("fixture");
  const scale = report("scale");
  fixture.summary.mobile.coldFirstViewMs.p95 = 3_001;
  scale.summary.mobile.rankUpdateMs.hybrid.p95 = 201;
  const failures = check(fixture, scale).failures;
  assert.ok(failures.some((message) => message.includes("fixture: mobile cold first-view p95")));
  assert.ok(failures.some((message) => message.includes("scale: hybrid warm-update p95")));
});
