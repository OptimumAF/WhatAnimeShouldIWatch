/** Private M8.4 serving-path comparison. No user rows or labels leave the evidence directory. */
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { markServingFinalUsed, servingSha256, verifyServingFreeze } from
  "../../pipeline/src/core/model-serving-freeze.ts";
import {
  parseCompactGraph, parseCompactModel, parseDemoCatalog, parseReleaseIdentityCatalog,
  parseReleaseManifest,
  type AnimeMetadata, type CompactGraphDataV3,
} from "../src/artifacts.ts";
import type { ModelRecommendationIndex, RecommendationResult } from "../src/domain.ts";
import { selectFranchiseDiverseRecommendations } from "../src/franchise-diversity.ts";
import type { HistoryEntry } from "../src/import-history.ts";
import { preferenceFromHistory } from "../src/preferences.ts";
import {
  buildCatalogCoverageRecommendations, buildGenreOverlapExploration,
  buildGraphRecommendationsForPreferences, buildModelRecommendationsForPreferences,
  buildRecommendationIndexFromCompact, createCandidateEligibilityPolicy, rankEligibleCandidates,
  type RecommendationFilters,
} from "../src/recommendations.ts";

const DIGEST = /^[a-f0-9]{64}$/;
const BASELINES = ["graph", "genre", "coverage"] as const;
type Baseline = typeof BASELINES[number];
type Rating = { animeId: number; rawScore: number };
type UserCase = { userId: string; observed: Rating[]; labels: Rating[]; historySeen: number[];
  exclude: number[]; includeOnly: number[]; filters: RecommendationFilters };
type Policy = { format: "model-promotion-quality-policy-v1"; decisionRef: string;
  candidateBundleId: string; cohortSha256: string; baselineName: Baseline;
  seed: number; suppliedCount: number; positiveRawScoreMin: number; topK: number;
  minimumEligibleUsers: number; minimumPositiveLabels: number;
  minimumServingCoverage: number; minimumNdcgLift: number; maximumP95LatencyMs: number;
  latencyWarmups: number; latencySamples: number };

export interface ServingReportInputs {
  graph: unknown;
  model: unknown;
  catalog: unknown;
  cohort: unknown;
  /** Reserved bytes are hashed before, and parsed only after, baseline selection. */
  finalBytes: Buffer;
  policy: unknown;
  tag: string;
  bundleId: string;
  rawContentSha256: string;
  selectionSha256: string;
  finalReportSha256: string;
  graphSha256: string;
  modelSha256: string;
  cohortSha256: string;
  freezeSha256: string;
  policySha256: string;
  now?: () => number;
  beforeFinal?: () => void;
}

export interface ServingReportV1 {
  format: "model-serving-evaluation-v1";
  evaluator: "browser-preference-eligibility-selector-v1";
  policySha256: string;
  cohortSha256: string;
  freezeSha256: string;
  graphSha256: string;
  modelSha256: string;
  tag: string;
  bundleId: string;
  rawContentSha256: string;
  graphDatasetSha256: string;
  selectionSha256: string;
  finalReportSha256: string;
  topK: number;
  suppliedCount: number;
  baselineValidation: { graph: number; genre: number; coverage: number };
  validationUsers: number;
  finalUsers: number;
  eligibleUsers: number;
  positiveLabels: number;
  baselineName: Baseline;
  baseline: { ndcgAtK: number; coverage: number; p95LatencyMs: number };
  model: { ndcgAtK: number; coverage: number; p95LatencyMs: number };
}

function fail(field: string, reason: string): never {
  throw new Error(`Model serving evaluation ${field}: ${reason}`);
}

function record(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as Record<string, unknown>;
}

function exact(value: unknown, keys: readonly string[], field: string): Record<string, unknown> {
  const item = record(value, field);
  for (const key of keys) if (!Object.hasOwn(item, key)) fail(`${field}.${key}`, "is required");
  for (const key of Object.keys(item)) if (!keys.includes(key)) fail(`${field}.${key}`, "is unsupported");
  return item;
}

function integer(value: unknown, field: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    fail(field, `must be a safe integer at least ${minimum}`);
  }
  return value as number;
}

function number(value: unknown, field: string, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    fail(field, `must be finite in [${minimum}, ${maximum}]`);
  }
  return value as number;
}

function word(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim() || value !== value.trim() || value.length > 200) {
    fail(field, "must be bounded nonempty trimmed text");
  }
  return value as string;
}

function digest(value: unknown, field: string): string {
  if (typeof value !== "string" || !DIGEST.test(value)) fail(field, "must be lowercase SHA-256");
  return value as string;
}

function list(value: unknown, field: string, maximum: number): unknown[] {
  if (!Array.isArray(value) || value.length > maximum) fail(field, "must be a bounded list");
  return value;
}

function ids(value: unknown, field: string, catalog: ReadonlySet<number>): number[] {
  const result = list(value, field, catalog.size).map((item, index) =>
    integer(item, `${field}[${index}]`, 1));
  if (new Set(result).size !== result.length || result.some((id) => !catalog.has(id))) {
    fail(field, "must contain unique catalog IDs");
  }
  return result;
}

function ratings(value: unknown, field: string, catalog: ReadonlySet<number>): Rating[] {
  const result = list(value, field, catalog.size).map((item, index) => {
    const row = exact(item, ["animeId", "rawScore"], `${field}[${index}]`);
    const animeId = integer(row.animeId, `${field}[${index}].animeId`, 1);
    if (!catalog.has(animeId)) fail(`${field}[${index}].animeId`, "is absent from the catalog");
    return { animeId, rawScore: number(row.rawScore, `${field}[${index}].rawScore`, 1, 10) };
  });
  if (new Set(result.map((row) => row.animeId)).size !== result.length) {
    fail(field, "contains duplicate anime IDs");
  }
  return result;
}

function users(value: unknown, field: string, catalog: ReadonlySet<number>,
  used: Set<string>, suppliedCount: number): UserCase[] {
  const result = list(value, field, 1_000).map((item, index) => {
    const label = `${field}[${index}]`;
    const row = exact(item, ["userId", "observed", "labels", "historySeen", "exclude",
      "includeOnly", "filters"], label);
    const userId = word(row.userId, `${label}.userId`);
    if (used.has(userId)) fail(`${label}.userId`, "overlaps fit or another evaluation group");
    used.add(userId);
    const observed = ratings(row.observed, `${label}.observed`, catalog);
    const labels = ratings(row.labels, `${label}.labels`, catalog);
    if (observed.length < suppliedCount || !labels.length ||
        observed.some((rating) => labels.some((held) => rating.animeId === held.animeId))) {
      fail(label, "needs a supplied prefix and disjoint held-out labels");
    }
    const filters = exact(row.filters, ["genre", "minYear", "maxYear", "minScore"],
      `${label}.filters`);
    if (typeof filters.genre !== "string" || filters.genre.length > 100) {
      fail(`${label}.filters.genre`, "must be bounded text");
    }
    const year = (key: "minYear" | "maxYear") => filters[key] === null ? null
      : integer(filters[key], `${label}.filters.${key}`, 1);
    const minScore = filters.minScore === null ? null
      : number(filters.minScore, `${label}.filters.minScore`, 0, 10);
    return { userId, observed, labels,
      historySeen: ids(row.historySeen, `${label}.historySeen`, catalog),
      exclude: ids(row.exclude, `${label}.exclude`, catalog),
      includeOnly: ids(row.includeOnly, `${label}.includeOnly`, catalog),
      filters: { genre: filters.genre, minYear: year("minYear"),
        maxYear: year("maxYear"), minScore } };
  });
  if (!result.length) fail(field, "must contain users");
  return result;
}

function parsePolicy(value: unknown, bundleId: string, cohortSha256: string): Policy {
  const policy = exact(value, ["format", "decisionRef", "candidateBundleId", "cohortSha256",
    "baselineName", "seed", "suppliedCount", "positiveRawScoreMin", "topK",
    "minimumEligibleUsers", "minimumPositiveLabels", "minimumServingCoverage",
    "minimumNdcgLift", "maximumP95LatencyMs", "latencyWarmups", "latencySamples"], "policy");
  if (policy.format !== "model-promotion-quality-policy-v1" ||
      policy.candidateBundleId !== bundleId || policy.cohortSha256 !== cohortSha256 ||
      !BASELINES.includes(policy.baselineName as Baseline)) {
    fail("policy", "format, candidate, cohort, or baseline is unsupported");
  }
  word(policy.decisionRef, "policy.decisionRef");
  digest(policy.cohortSha256, "policy.cohortSha256");
  integer(policy.seed, "policy.seed");
  integer(policy.suppliedCount, "policy.suppliedCount", 1);
  integer(policy.positiveRawScoreMin, "policy.positiveRawScoreMin", 1);
  integer(policy.topK, "policy.topK", 1);
  integer(policy.minimumEligibleUsers, "policy.minimumEligibleUsers", 1);
  integer(policy.minimumPositiveLabels, "policy.minimumPositiveLabels", 1);
  number(policy.minimumServingCoverage, "policy.minimumServingCoverage", Number.MIN_VALUE, 1);
  number(policy.minimumNdcgLift, "policy.minimumNdcgLift", Number.MIN_VALUE, 1);
  number(policy.maximumP95LatencyMs, "policy.maximumP95LatencyMs", Number.MIN_VALUE, 60_000);
  integer(policy.latencyWarmups, "policy.latencyWarmups");
  integer(policy.latencySamples, "policy.latencySamples", 1);
  if ((policy.latencySamples as number) > 1_000 || (policy.latencyWarmups as number) > 1_000 ||
      (policy.topK as number) > 100 || (policy.suppliedCount as number) > 100) {
    fail("policy", "measurement or rank budget is too large");
  }
  return policy as unknown as Policy;
}

function localHistory(rating: Rating, title: string): HistoryEntry {
  return { provider: "local", sourceId: String(rating.animeId), title,
    animeId: rating.animeId, status: "completed", sourceStatus: "completed",
    progressEpisodes: null, score: rating.rawScore, scoreScale: "local-10" };
}

function orderedObserved(user: UserCase, seed: number): Rating[] {
  const key = (rating: Rating) => createHash("sha256").update("wasiw-promotion-observed-v1\n" +
    JSON.stringify([seed, user.userId, rating.animeId])).digest("hex");
  return [...user.observed].sort((a, b) => key(a).localeCompare(key(b)) || a.animeId - b.animeId);
}

function ndcg(ids: readonly number[], positives: readonly number[], topK: number): number {
  if (!positives.length) fail("ndcg", "needs eligible positive labels");
  const expected = new Set(positives);
  const actual = ids.slice(0, topK).reduce((sum, id, index) =>
    sum + (expected.has(id) ? 1 / Math.log2(index + 2) : 0), 0);
  const ideal = Array.from({ length: Math.min(topK, expected.size) }, (_, index) =>
    1 / Math.log2(index + 2)).reduce((sum, gain) => sum + gain, 0);
  return actual / ideal;
}

function percentile95(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(0.95 * sorted.length) - 1];
}

function measure(run: () => void, now: () => number, warmups: number, samples: number): number {
  for (let i = 0; i < warmups; i += 1) run();
  const durations: number[] = [];
  for (let i = 0; i < samples; i += 1) {
    const start = now();
    run();
    const duration = now() - start;
    if (!Number.isFinite(duration) || duration < 0) fail("latency", "clock returned invalid duration");
    durations.push(duration);
  }
  return percentile95(durations);
}

export function generateServingReport(inputs: ServingReportInputs): ServingReportV1 {
  const graph = parseCompactGraph(inputs.graph, "promotion graph", "recommendation");
  if (graph.format !== "graph-compact-v3") fail("graph.format", "requires aggregate v3");
  const model = parseCompactModel(inputs.model, "promotion model");
  const catalog = parseReleaseIdentityCatalog(inputs.catalog, "promotion catalog");
  if (catalog.datasetSha256 !== graph.dataset.sha256 ||
      JSON.stringify(catalog.anime) !== JSON.stringify(graph.anime) ||
      model.datasetSha256 !== graph.dataset.sha256) {
    fail("dataset", "graph, catalog, and model disagree");
  }
  const titleById = new Map(catalog.anime);
  model.animeIds.forEach((id, index) => {
    if (titleById.get(id) !== model.titles[index]) fail(`model.titles[${index}]`, "differs from catalog");
  });
  for (const key of ["bundleId", "rawContentSha256", "selectionSha256",
    "finalReportSha256", "graphSha256", "modelSha256", "cohortSha256", "freezeSha256",
    "policySha256"] as const) digest(inputs[key], key);
  const policy = parsePolicy(inputs.policy, inputs.bundleId, inputs.cohortSha256);
  const cohortValue = exact(inputs.cohort, ["format", "seed", "sourceName",
    "datasetSha256", "trainingUsers", "metadata", "baselineValidation", "finalSha256"], "cohort");
  if (cohortValue.format !== "model-promotion-cohort-v2" ||
      cohortValue.datasetSha256 !== graph.dataset.sha256 ||
      cohortValue.sourceName !== graph.dataset.source || cohortValue.seed !== policy.seed ||
      cohortValue.finalSha256 !== servingSha256(inputs.finalBytes)) {
    fail("cohort", "format, source, dataset, or seed differs from pinned artifacts");
  }
  const catalogIds = new Set(catalog.anime.map(([id]) => id));
  const metadataRows = parseDemoCatalog(cohortValue.metadata, "promotion metadata");
  if (metadataRows.length !== catalogIds.size ||
      metadataRows.some((row) => titleById.get(row.animeId) !== row.title)) {
    fail("cohort.metadata", "must cover the exact catalog ID/title map");
  }
  const metadata = new Map<number, AnimeMetadata>(metadataRows.map((row) => [row.animeId, row]));
  const used = new Set<string>();
  const trainingUsers = list(cohortValue.trainingUsers, "cohort.trainingUsers", 100_000)
    .map((value, index) => word(value, `cohort.trainingUsers[${index}]`));
  if (!trainingUsers.length || new Set(trainingUsers).size !== trainingUsers.length) {
    fail("cohort.trainingUsers", "must contain unique fit IDs");
  }
  trainingUsers.forEach((id) => used.add(id));
  const validation = users(cohortValue.baselineValidation, "cohort.baselineValidation",
    catalogIds, used, policy.suppliedCount);
  const index = buildRecommendationIndexFromCompact(graph as CompactGraphDataV3);
  const modelIndex: ModelRecommendationIndex = { generatedAt: model.generatedAt,
    factors: model.factors, globalMean: model.globalMean,
    animeByAnimeId: new Map(model.animeIds.map((animeId, index) =>
      [animeId, { animeId, title: model.titles[index], bias: model.biases[index],
        embedding: model.embeddings[index] }])) };
  const catalogRanking = buildCatalogCoverageRecommendations(index);
  const titleMap = new Map(catalog.anime);

  function scoreUser(user: UserCase, method: Baseline | "model") {
    const preferences = orderedObserved(user, policy.seed).slice(0, policy.suppliedCount)
      .map((rating) => preferenceFromHistory(localHistory(rating, titleById.get(rating.animeId)!),
        `anime:${rating.animeId}`)!);
    const history = user.historySeen.map((animeId) => ({
      ...localHistory({ animeId, rawScore: 5 }, titleById.get(animeId)!), score: null,
    }));
    const eligibility = createCandidateEligibilityPolicy({ index, preferences, history,
      includeOnlyNodeIds: user.includeOnly.map((id) => `anime:${id}`),
      excludeNodeIds: user.exclude.map((id) => `anime:${id}`), filters: user.filters });
    const eligibleCatalog = rankEligibleCandidates("fallback", { fallback: catalogRanking },
      eligibility, metadata).recommendations.map((item) => item.anime.animeId);
    const positive = user.labels.filter((item) => item.rawScore >= policy.positiveRawScoreMin &&
      eligibleCatalog.includes(item.animeId)).map((item) => item.animeId);
    let ranked: RecommendationResult[];
    if (method === "model") {
      ranked = rankEligibleCandidates("model", { model: buildModelRecommendationsForPreferences(
        preferences, index, modelIndex) }, eligibility, metadata).recommendations;
    } else if (method === "graph") {
      ranked = rankEligibleCandidates("graph", { graph: buildGraphRecommendationsForPreferences(
        preferences, index) }, eligibility, metadata).recommendations;
    } else {
      const baseline = method === "genre"
        ? buildGenreOverlapExploration(preferences, index, metadata) : catalogRanking;
      ranked = rankEligibleCandidates("fallback", { fallback: baseline },
        eligibility, metadata).recommendations;
    }
    const watched = new Set([...preferences.map((item) => index.animeByNodeId.get(item.nodeId)!.animeId),
      ...user.historySeen]);
    const displayed = selectFranchiseDiverseRecommendations(ranked, metadata,
      watched, false, titleMap).recommendations.map((item) => item.anime.animeId);
    return { positive, eligibleCatalog, displayed };
  }

  function summary(group: readonly UserCase[], method: Baseline | "model") {
    const rows = group.map((user) => scoreUser(user, method))
      .filter((row) => row.positive.length > 0);
    if (!rows.length) return { eligibleUsers: 0, positiveLabels: 0, ndcgAtK: 0, coverage: 0 };
    const shown = new Set(rows.flatMap((row) => row.displayed.slice(0, policy.topK)));
    const available = new Set(rows.flatMap((row) => row.eligibleCatalog));
    return { eligibleUsers: rows.length,
      positiveLabels: rows.reduce((sum, row) => sum + row.positive.length, 0),
      ndcgAtK: rows.reduce((sum, row) => sum + ndcg(row.displayed, row.positive, policy.topK), 0) /
        rows.length,
      coverage: available.size ? shown.size / available.size : 0 };
  }

  const validationSummaries = Object.fromEntries(BASELINES.map((baseline) =>
    [baseline, summary(validation, baseline)])) as Record<Baseline, ReturnType<typeof summary>>;
  for (const baseline of BASELINES) {
    if (validationSummaries[baseline].eligibleUsers < policy.minimumEligibleUsers ||
        validationSummaries[baseline].positiveLabels < policy.minimumPositiveLabels) {
      fail("baselineValidation", "has too few eligible users or positive labels");
    }
  }
  const validationScores = Object.fromEntries(BASELINES.map((baseline) =>
    [baseline, validationSummaries[baseline].ndcgAtK])) as Record<Baseline, number>;
  const best = [...BASELINES].sort((a, b) => validationScores[b] - validationScores[a] ||
    BASELINES.indexOf(a) - BASELINES.indexOf(b))[0];
  if (policy.baselineName !== best) {
    fail("policy.baselineName", `must be the validation-selected simple baseline (${best})`);
  }
  inputs.beforeFinal?.();
  let finalPayload: unknown;
  try { finalPayload = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(inputs.finalBytes)); }
  catch { fail("serving-final.json", "must be valid JSON and UTF-8"); }
  const reserved = exact(finalPayload, ["format", "users"], "serving-final.json");
  if (reserved.format !== "model-promotion-final-v1") fail("serving-final.json.format", "is unsupported");
  const final = users(reserved.users, "cohort.finalTest", catalogIds, used, policy.suppliedCount);
  const baseline = summary(final, best);
  const measured = summary(final, "model");
  if (baseline.eligibleUsers !== measured.eligibleUsers ||
      baseline.positiveLabels !== measured.positiveLabels) {
    fail("finalTest", "baseline and model must use the same eligible positive labels");
  }
  const now = inputs.now ?? (() => performance.now());
  const baselineP95 = measure(() => { for (const user of final) scoreUser(user, best); },
    now, policy.latencyWarmups, policy.latencySamples);
  const modelP95 = measure(() => { for (const user of final) scoreUser(user, "model"); },
    now, policy.latencyWarmups, policy.latencySamples);
  return { format: "model-serving-evaluation-v1",
    evaluator: "browser-preference-eligibility-selector-v1",
    policySha256: inputs.policySha256, cohortSha256: inputs.cohortSha256,
    freezeSha256: inputs.freezeSha256,
    graphSha256: inputs.graphSha256, modelSha256: inputs.modelSha256,
    tag: inputs.tag, bundleId: inputs.bundleId,
    rawContentSha256: inputs.rawContentSha256, graphDatasetSha256: graph.dataset.sha256,
    selectionSha256: inputs.selectionSha256, finalReportSha256: inputs.finalReportSha256,
    topK: policy.topK, suppliedCount: policy.suppliedCount,
    baselineValidation: validationScores, validationUsers: validation.length,
    finalUsers: final.length, eligibleUsers: measured.eligibleUsers,
    positiveLabels: measured.positiveLabels, baselineName: best,
    baseline: { ndcgAtK: baseline.ndcgAtK, coverage: baseline.coverage,
      p95LatencyMs: baselineP95 },
    model: { ndcgAtK: measured.ndcgAtK, coverage: measured.coverage,
      p95LatencyMs: modelP95 } };
}

/** Recompute ranks and coverage; latency is measured again against the same declared ceiling. */
export function verifyServingReport(inputs: ServingReportInputs, value: unknown): ServingReportV1 {
  const report = exact(value, ["format", "evaluator", "policySha256", "cohortSha256",
    "freezeSha256",
    "graphSha256", "modelSha256", "tag", "bundleId", "rawContentSha256",
    "graphDatasetSha256", "selectionSha256", "finalReportSha256", "topK",
    "suppliedCount", "baselineValidation", "validationUsers", "finalUsers",
    "eligibleUsers", "positiveLabels", "baselineName", "baseline", "model"], "report");
  const expected = generateServingReport(inputs);
  const reportedBaseline = exact(report.baseline, ["ndcgAtK", "coverage", "p95LatencyMs"],
    "report.baseline");
  const reportedModel = exact(report.model, ["ndcgAtK", "coverage", "p95LatencyMs"],
    "report.model");
  const policy = parsePolicy(inputs.policy, inputs.bundleId, inputs.cohortSha256);
  for (const [name, row] of [["baseline", reportedBaseline], ["model", reportedModel]] as const) {
    number(row.p95LatencyMs, `report.${name}.p95LatencyMs`, Number.MIN_VALUE,
      policy.maximumP95LatencyMs);
  }
  if (expected.baseline.p95LatencyMs > policy.maximumP95LatencyMs ||
      expected.model.p95LatencyMs > policy.maximumP95LatencyMs) {
    fail("latency", "fresh serving-path measurement exceeds the policy ceiling");
  }
  const removeTiming = (item: Record<string, unknown>) => ({ ...item,
    baseline: { ...record(item.baseline, "baseline"), p95LatencyMs: 0 },
    model: { ...record(item.model, "model"), p95LatencyMs: 0 } });
  if (!isDeepStrictEqual(removeTiming(report), removeTiming(expected as unknown as
      Record<string, unknown>))) {
    fail("report", "ranking, coverage, provenance, or baseline differs from browser recomputation");
  }
  if (expected.eligibleUsers < policy.minimumEligibleUsers ||
      expected.positiveLabels < policy.minimumPositiveLabels ||
      expected.model.coverage < policy.minimumServingCoverage ||
      expected.model.ndcgAtK < expected.baseline.ndcgAtK + policy.minimumNdcgLift) {
    fail("report", "does not meet frozen quality or coverage floors");
  }
  return value as ServingReportV1;
}

function fileSha(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

/** Write once for an initial final report, then verify read-only during packaging. */
function main(): void {
  const writing = process.argv[2] === "--write-frozen";
  if (process.argv.length !== (writing ? 5 : 4)) {
    fail("arguments", "expected candidate and private evidence directories");
  }
  const candidate = resolve(process.argv[writing ? 3 : 2]);
  const evidence = resolve(process.argv[writing ? 4 : 3]);
  const read = (directory: string, name: string) => JSON.parse(readFileSync(join(directory, name), "utf8"));
  const manifest = parseReleaseManifest(read(candidate, "release-manifest.json"),
    "release-manifest.json");
  const review = writing ? null : record(read(evidence, "model-promotion-review.json"), "review");
  const selection = writing ? record(read(evidence, "selection.json"), "selection") : null;
  const graphPath = join(candidate, "graph.compact.json");
  const modelPath = join(candidate, "model-mf-web.compact.json");
  const cohortPath = join(evidence, "serving-cohort.json");
  const finalPath = join(evidence, "serving-final.json");
  const freezePath = join(evidence, "serving-freeze.json");
  const policyPath = join(evidence, "quality-policy.json");
  const cohortSha256 = fileSha(cohortPath);
  const freezeSha256 = fileSha(freezePath);
  const finalBytes = readFileSync(finalPath);
  const qualityPlan = record(read(evidence, "quality-plan.json"), "quality-plan.json");
  const policy = read(evidence, "quality-policy.json");
  const rawContentSha256 = (review?.rawContentSha256 ?? selection?.rawContentSha256) as string;
  verifyServingFreeze(evidence, { sourceName: manifest.dataset.source,
    rawContentSha256, graphDatasetSha256: manifest.dataset.sha256,
    cohortSha256, finalSha256: qualityPlan.finalSha256 as string,
    freezeSha256 }, policy, !writing);
  const inputs: ServingReportInputs = { graph: read(candidate, "graph.compact.json"),
    model: read(candidate, "model-mf-web.compact.json"),
    catalog: read(candidate, "catalog.identity.json"),
    cohort: read(evidence, "serving-cohort.json"), finalBytes, policy,
    tag: manifest.tag, bundleId: manifest.bundleId,
    rawContentSha256,
    selectionSha256: (review?.selectionSha256 ?? selection?.selectionSha256) as string,
    finalReportSha256: (review?.finalReportSha256 ?? fileSha(join(evidence,
      "final-report.json"))) as string,
    graphSha256: fileSha(graphPath), modelSha256: fileSha(modelPath),
    cohortSha256, freezeSha256, policySha256: fileSha(policyPath) };
  if (writing) {
    const report = generateServingReport({ ...inputs,
      beforeFinal: () => markServingFinalUsed(evidence, freezeSha256,
        qualityPlan.finalSha256 as string) });
    writeFileSync(join(evidence, "serving-report.json"), JSON.stringify(report) + "\n",
      { flag: "wx" });
    process.stdout.write(`Wrote one frozen serving report for ${manifest.tag}.\n`);
  } else {
    verifyServingReport(inputs, read(evidence, "serving-report.json"));
    process.stdout.write(`Verified generated serving report for ${manifest.tag}.\n`);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  try { main(); } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
