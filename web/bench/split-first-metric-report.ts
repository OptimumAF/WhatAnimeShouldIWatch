/** M5.7 invented validation metrics for the fixed M5.6 common-candidate comparison. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { cpus } from "node:os";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import {
  evaluateBaselineCases, loadBaselineBundle, METHODS, parseBaselineSpec,
} from "./split-first-baseline-ablation.ts";
import type { BaselineBundle, BaselineCase, Method, MethodCase } from "./split-first-baseline-ablation.ts";
import { fitUserIds, parseEvalFixture } from "./split-first-new-user-eval.ts";
import type { EvalFixture } from "./split-first-new-user-eval.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const TOP_KS = [10, 20] as const;
const COUNTS = [1, 3, 5, 10] as const;
export type MetricSpec = {
  format: "split-first-metric-report-spec-v1"; baselineSpecSha256: string;
  topKs: number[]; suppliedCounts: number[]; positiveRawScoreMin: 7;
  lowSupportMax: 1; bootstrapSeed: 271828; bootstrapReplicates: 2048;
  latencyWarmups: 5; latencySamples: 31;
};
export type RankMetrics = { recall: number; ndcg: number };
type CaseLabels = { row: BaselineCase; labels: number[] };
type PositivePair = BaselineBundle["base"]["positivePairs"][number];

function fail(field: string): never { throw new Error(`Invalid metric report ${field}.`); }
function mean(values: readonly number[]): number | null {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}
function readJson(relative: string): unknown {
  return JSON.parse(readFileSync(resolve(ROOT, relative), "utf8"));
}
function normalizedFileSha(relative: string): string {
  const bytes = readFileSync(resolve(ROOT, relative));
  return createHash("sha256").update(Buffer.from(
    bytes.toString("latin1").replace(/\r\n/g, "\n"), "latin1")).digest("hex");
}
function sameIds(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, i) => value === right[i]);
}
function uniquePositiveIds(ids: readonly number[], field: string): void {
  if (new Set(ids).size !== ids.length ||
      ids.some((id) => !Number.isSafeInteger(id) || id < 1)) fail(field);
}

export function parseMetricSpec(value: unknown): MetricSpec {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("specification");
  const item = value as Record<string, unknown>;
  const keys = ["format", "baselineSpecSha256", "topKs", "suppliedCounts",
    "positiveRawScoreMin", "lowSupportMax", "bootstrapSeed", "bootstrapReplicates",
    "latencyWarmups", "latencySamples"];
  if (Object.keys(item).sort().join("|") !== keys.sort().join("|") ||
      item.format !== "split-first-metric-report-spec-v1" ||
      typeof item.baselineSpecSha256 !== "string" ||
      !/^[a-f0-9]{64}$/.test(item.baselineSpecSha256) ||
      JSON.stringify(item.topKs) !== JSON.stringify(TOP_KS) ||
      JSON.stringify(item.suppliedCounts) !== JSON.stringify(COUNTS) ||
      item.positiveRawScoreMin !== 7 || item.lowSupportMax !== 1 ||
      item.bootstrapSeed !== 271828 || item.bootstrapReplicates !== 2048 ||
      item.latencyWarmups !== 5 || item.latencySamples !== 31) fail("specification protocol");
  return item as MetricSpec;
}

export function assertPinnedMetricSpec(spec: MetricSpec): void {
  if (normalizedFileSha("fixtures/synthetic-baseline-ablation-spec.json") !==
      spec.baselineSpecSha256) fail("baseline specification hash");
}

/** Binary ranking metrics; null means that no eligible positive label exists. */
export function rankMetrics(displayedIds: readonly number[], positiveIds: readonly number[],
                            topK: number): RankMetrics | null {
  uniquePositiveIds(displayedIds, "displayed IDs");
  uniquePositiveIds(positiveIds, "positive IDs");
  if (!Number.isSafeInteger(topK) || topK < 1) fail("top K");
  if (!positiveIds.length) return null;
  const positive = new Set(positiveIds);
  const hits = displayedIds.slice(0, topK).flatMap((id, i) => positive.has(id) ? [i + 1] : []);
  const dcg = hits.reduce((sum, rank) => sum + 1 / Math.log2(rank + 1), 0);
  const ideal = Array.from({ length: Math.min(positive.size, topK) }, (_, i) =>
    1 / Math.log2(i + 2)).reduce((sum, gain) => sum + gain, 0);
  return { recall: hits.length / positive.size, ndcg: dcg / ideal };
}

/** Mean pairwise Jaccard distance in the fixed authored genre vocabulary. */
export function genreDiversity(ids: readonly number[],
                               genres: ReadonlyMap<number, readonly string[]>): number | null {
  uniquePositiveIds(ids, "diversity IDs");
  if (ids.length < 2) return null;
  let distance = 0;
  let pairs = 0;
  for (let i = 0; i < ids.length; i += 1) {
    const left = genres.get(ids[i]);
    if (!left?.length) fail(`genres ${ids[i]}`);
    for (let j = i + 1; j < ids.length; j += 1) {
      const right = genres.get(ids[j]);
      if (!right?.length) fail(`genres ${ids[j]}`);
      const a = new Set(left);
      const b = new Set(right);
      const intersection = [...a].filter((genre) => b.has(genre)).length;
      distance += 1 - intersection / new Set([...a, ...b]).size;
      pairs += 1;
    }
  }
  return distance / pairs;
}

/** Difference from the same case's eligible-universe mean train-only count. */
export function popularityBias(displayedIds: readonly number[], universeIds: readonly number[],
                               counts: ReadonlyMap<number, number>): number | null {
  uniquePositiveIds(displayedIds, "popularity displayed IDs");
  uniquePositiveIds(universeIds, "popularity universe IDs");
  const eligible = new Set(universeIds);
  if (displayedIds.some((id) => !eligible.has(id))) fail("popularity subset");
  if (!displayedIds.length || !universeIds.length) return null;
  const average = (ids: readonly number[]) => {
    const values = ids.map((id) => counts.get(id));
    if (values.some((value) => value === undefined ||
        !Number.isSafeInteger(value) || value < 0)) fail("train count");
    return values.reduce<number>((sum, value) => sum + value!, 0) / values.length;
  };
  return average(displayedIds) - average(universeIds);
}

export function catalogCoverage(rows: readonly { universeIds: readonly number[];
                                                  displayedIds: readonly number[] }[],
                                topK: number) {
  if (!Number.isSafeInteger(topK) || topK < 1) fail("coverage top K");
  const eligible = new Set<number>();
  const shown = new Set<number>();
  for (const row of rows) {
    uniquePositiveIds(row.universeIds, "coverage universe IDs");
    uniquePositiveIds(row.displayedIds, "coverage displayed IDs");
    const own = new Set(row.universeIds);
    if (row.displayedIds.some((id) => !own.has(id))) fail("coverage subset");
    row.universeIds.forEach((id) => eligible.add(id));
    row.displayedIds.slice(0, topK).forEach((id) => shown.add(id));
  }
  return { distinctDisplayed: shown.size, eligibleUnion: eligible.size,
    fraction: eligible.size ? shown.size / eligible.size : null };
}

/** Linear interpolation between sorted observations, including endpoints. */
export function percentile(values: readonly number[], probability: number): number {
  if (!values.length || values.some((value) => !Number.isFinite(value)) ||
      !Number.isFinite(probability) || probability < 0 || probability > 1) fail("percentile");
  const sorted = [...values].sort((a, b) => a - b);
  const location = (sorted.length - 1) * probability;
  const lower = Math.floor(location);
  const fraction = location - lower;
  return sorted[lower] + (sorted[Math.ceil(location)] - sorted[lower]) * fraction;
}

function random32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Resamples complete user clusters, retaining every measurable prefix together. */
export function bootstrapUserMean(rows: readonly { userNumber: number; value: number }[],
                                  seed: number, replicates: number) {
  if (!rows.length || !Number.isSafeInteger(seed) || !Number.isSafeInteger(replicates) ||
      replicates < 1 || rows.some((row) => !Number.isSafeInteger(row.userNumber) ||
        row.userNumber < 1 || !Number.isFinite(row.value))) fail("bootstrap input");
  const clusters = new Map<number, number[]>();
  for (const row of rows) clusters.set(row.userNumber,
    [...(clusters.get(row.userNumber) ?? []), row.value]);
  const groups = [...clusters.values()];
  const random = random32(seed);
  const samples: number[] = [];
  for (let repeat = 0; repeat < replicates; repeat += 1) {
    let total = 0;
    let count = 0;
    for (let draw = 0; draw < groups.length; draw += 1) {
      const group = groups[Math.floor(random() * groups.length)];
      total += group.reduce((sum, value) => sum + value, 0);
      count += group.length;
    }
    samples.push(total / count);
  }
  return { users: groups.length, replicates, seed, mean: mean(rows.map((row) => row.value))!,
    p025: percentile(samples, 0.025), p975: percentile(samples, 0.975) };
}

/** Measures only the supplied callback; caller excludes fit, I/O and serialization. */
export function measureLatency(run: () => void, now: () => number,
                               warmups: number, samples: number) {
  if (!Number.isSafeInteger(warmups) || warmups < 0 ||
      !Number.isSafeInteger(samples) || samples < 1) fail("latency protocol");
  for (let i = 0; i < warmups; i += 1) run();
  const durations: number[] = [];
  for (let i = 0; i < samples; i += 1) {
    const start = now();
    run();
    const duration = now() - start;
    if (!Number.isFinite(duration) || duration < 0) fail("latency clock");
    durations.push(duration);
  }
  const sorted = [...durations].sort((a, b) => a - b);
  return { warmups, samples, medianMs: percentile(sorted, 0.5),
    p95Ms: sorted[Math.ceil(0.95 * samples) - 1] };
}

export function positiveEdgeSupport(candidateId: number, likedSourceIds: readonly number[],
                                    pairs: readonly PositivePair[]): number {
  uniquePositiveIds(likedSourceIds, "liked source IDs");
  const liked = new Set(likedSourceIds);
  return pairs.reduce((highest, pair) => {
    if (pair.leftAnimeId === candidateId && liked.has(pair.rightAnimeId) ||
        pair.rightAnimeId === candidateId && liked.has(pair.leftAnimeId)) {
      return Math.max(highest, pair.support);
    }
    return highest;
  }, 0);
}

function methodRow(row: BaselineCase, method: Method): MethodCase {
  const found = row.methods.find((item) => item.method === method);
  if (!found) fail(`missing method ${method}`);
  return found;
}
function validateCases(cases: readonly BaselineCase[], fixture: EvalFixture, spec: MetricSpec): void {
  if (fixture.positiveRawScoreMin !== spec.positiveRawScoreMin ||
      !sameIds(spec.suppliedCounts, COUNTS) ||
      cases.length !== fixture.users.length * COUNTS.length) fail("case protocol");
  const seen = new Set<string>();
  for (const row of cases) {
    const key = `${row.userNumber}:${row.suppliedCount}`;
    if (!Number.isSafeInteger(row.userNumber) || row.userNumber < 1 ||
        row.userNumber > fixture.users.length || !COUNTS.includes(row.suppliedCount as 1) ||
        seen.has(key)) fail("case identity");
    seen.add(key);
    uniquePositiveIds(row.universeIds, "universe IDs");
    uniquePositiveIds(row.eligiblePositiveIds, "eligible positive IDs");
    uniquePositiveIds(row.likedSourceIds, "liked source IDs");
    const universe = new Set(row.universeIds);
    if (row.eligiblePositiveIds.some((id) => !universe.has(id)) ||
        !sameIds(row.universeIds, [...row.universeIds].sort((a, b) => a - b)) ||
        row.methods.length !== METHODS.length ||
        row.methods.some((item, i) => item.method !== METHODS[i])) fail("case universe/methods");
    for (const item of row.methods) {
      uniquePositiveIds(item.displayedIds, "displayed IDs");
      const rankedIds = item.ranked.map((entry) => entry.animeId);
      uniquePositiveIds(rankedIds, "ranked IDs");
      if (!sameIds(item.eligibleCandidateIds, row.universeIds) ||
          !sameIds([...rankedIds].sort((a, b) => a - b), row.universeIds) ||
          item.displayedIds.some((id) => !universe.has(id)) ||
          item.selectorRemoved !== rankedIds.length - item.displayedIds.length) {
        fail("method candidate/selector contract");
      }
      let previous = -1;
      for (const id of item.displayedIds) {
        const at = rankedIds.indexOf(id);
        if (at <= previous) fail("displayed order");
        previous = at;
      }
      const top10 = rankMetrics(item.displayedIds, row.eligiblePositiveIds, 10);
      if (top10 && (Math.abs(top10.recall - item.recallAt10) > 1e-12 ||
                    Math.abs(top10.ndcg - item.ndcgAt10) > 1e-12)) fail("M5.6 metric parity");
    }
  }
}

function rankingSummary(rows: readonly CaseLabels[], method: Method) {
  const measurable = rows.filter((item) => item.labels.length > 0);
  const values = TOP_KS.map((topK) => {
    const metrics = measurable.map((item) =>
      rankMetrics(methodRow(item.row, method).displayedIds, item.labels, topK)!);
    return { topK, meanRecall: mean(metrics.map((item) => item.recall)),
      meanNdcg: mean(metrics.map((item) => item.ndcg)) };
  });
  return { eligibleUsers: new Set(measurable.map((item) => item.row.userNumber)).size,
    measurableCases: measurable.length, casesWithoutEligiblePositive: rows.length - measurable.length,
    eligiblePositiveLabels: measurable.reduce((sum, item) => sum + item.labels.length, 0),
    ranking: values };
}

function groupSummary(rows: readonly CaseLabels[], method: Method,
                      bundle: BaselineBundle, genres: ReadonlyMap<number, readonly string[]>) {
  const ranked = rankingSummary(rows, method);
  const measurable = rows.filter((item) => item.labels.length > 0);
  const ancillary = TOP_KS.map((topK) => {
    const displayed = measurable.map((item) => ({
      universeIds: item.row.universeIds,
      displayedIds: methodRow(item.row, method).displayedIds.slice(0, topK),
    }));
    const diversity = displayed.map((item) => genreDiversity(item.displayedIds, genres))
      .filter((value): value is number => value !== null);
    const bias = displayed.map((item) =>
      popularityBias(item.displayedIds, item.universeIds, bundle.trainCounts))
      .filter((value): value is number => value !== null);
    return { topK, coverage: catalogCoverage(displayed, topK),
      meanGenreJaccardDistance: mean(diversity), diversityCases: diversity.length,
      meanTrainCountBias: mean(bias), popularityCases: bias.length };
  });
  const counts = measurable.map((item) => item.row.universeIds.length);
  return { ...ranked, eligibleCandidateCountRange: counts.length
    ? [Math.min(...counts), Math.max(...counts)] : null, ancillary };
}

export function summarizeMetricCases(bundle: BaselineBundle, fixture: EvalFixture,
                                     cases: readonly BaselineCase[], spec: MetricSpec) {
  validateCases(cases, fixture, spec);
  const genres = new Map(fixture.candidateMetadata.map((item) =>
    [item.animeId, item.genres] as const));
  const all: CaseLabels[] = cases.map((row) => ({ row, labels: row.eligiblePositiveIds }));
  const supportRows = (low: boolean): CaseLabels[] => cases.map((row) => ({
    row, labels: row.eligiblePositiveIds.filter((id) =>
      (positiveEdgeSupport(id, row.likedSourceIds, bundle.base.positivePairs) <=
        spec.lowSupportMax) === low),
  }));
  const low = supportRows(true);
  const higher = supportRows(false);
  const zeroSupportLabels = cases.reduce((sum, row) => sum + row.eligiblePositiveIds.filter((id) =>
    positiveEdgeSupport(id, row.likedSourceIds, bundle.base.positivePairs) === 0).length, 0);
  const methods = METHODS.map((method) => {
    const full = groupSummary(all, method, bundle, genres);
    const perPrefix = COUNTS.map((suppliedCount) =>
      ({ suppliedCount, ...groupSummary(all.filter((item) =>
        item.row.suppliedCount === suppliedCount), method, bundle, genres) }));
    const lowSupport = rankingSummary(low, method);
    const higherSupport = rankingSummary(higher, method);
    const bootstrapRows = all.filter((item) => item.labels.length > 0)
      .map((item) => ({ userNumber: item.row.userNumber,
        value: rankMetrics(methodRow(item.row, method).displayedIds, item.labels, 10)!.ndcg }));
    const bootstrap = bootstrapRows.length
      ? bootstrapUserMean(bootstrapRows, spec.bootstrapSeed, spec.bootstrapReplicates) : null;
    return { method, full, perPrefix,
      byPositivePairSupport: { lowOrMissing: lowSupport, atLeastTwo: higherSupport },
      bootstrapNdcgAt10: bootstrap };
  });
  return {
    format: "split-first-metric-validation-v1",
    baselineSpecSha256: spec.baselineSpecSha256,
    trainSha256: bundle.base.trainSha256, fitSha256: bundle.base.fitSha256,
    validationUsers: fixture.users.length, cases: cases.length,
    eligibleUsers: new Set(all.filter((item) => item.labels.length)
      .map((item) => item.row.userNumber)).size,
    measurableCases: all.filter((item) => item.labels.length).length,
    casesWithoutEligiblePositive: all.filter((item) => !item.labels.length).length,
    eligiblePositiveLabels: all.reduce((sum, item) => sum + item.labels.length, 0),
    lowSupportZeroEvidenceLabels: zeroSupportLabels,
    methods,
  };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const spec = parseMetricSpec(readJson("fixtures/synthetic-metric-report-spec.json"));
  assertPinnedMetricSpec(spec);
  const baselineSpec = parseBaselineSpec(readJson("fixtures/synthetic-baseline-ablation-spec.json"));
  const bundle = loadBaselineBundle(baselineSpec);
  const fixture = parseEvalFixture(readJson("fixtures/synthetic-new-user-validation.json"),
    bundle.base, fitUserIds(readJson("fixtures/synthetic-new-user-fit.json")));
  const cases = evaluateBaselineCases(bundle, fixture, baselineSpec);
  const report = summarizeMetricCases(bundle, fixture, cases, spec);
  const latency = measureLatency(() => {
    evaluateBaselineCases(bundle, fixture, baselineSpec);
  }, () => performance.now(), spec.latencyWarmups, spec.latencySamples);
  process.stdout.write(JSON.stringify({ ...report,
    metricSpecSha256: normalizedFileSha("fixtures/synthetic-metric-report-spec.json"),
    latency: {
    ...latency, operation: "full-16-case-ten-method-evaluateBaselineCases",
    excludes: ["Python fit/export", "fixture/JSON I/O", "report serialization"],
    runtime: { node: process.version, platform: process.platform, arch: process.arch,
      cpu: cpus()[0]?.model ?? "unknown" },
  } }, null, 2) + "\n");
}
