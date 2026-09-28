/** Invented M5.6 validation comparison over one eligible catalog per user/prefix case. */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseCompactModel } from "../src/artifacts.ts";
import type { AnimeMetadata, GraphData } from "../src/artifacts.ts";
import type { ModelRecommendationIndex, RecommendationResult } from "../src/domain.ts";
import { selectFranchiseDiverseRecommendations } from "../src/franchise-diversity.ts";
import type { HistoryEntry } from "../src/import-history.ts";
import { preferenceFromHistory } from "../src/preferences.ts";
import type { AnimePreference } from "../src/preferences.ts";
import {
  buildGenreOverlapExploration, buildGraphRecommendationsForPreferences,
  buildModelRecommendationsForPreferences, buildRecommendationIndex,
  createCandidateEligibilityPolicy, rankEligibleCandidates,
} from "../src/recommendations.ts";
import { fitUserIds, metricsForRanks, orderedObserved, parseEvalBundle,
  parseEvalFixture } from "./split-first-new-user-eval.ts";
import type { EvalFixture } from "./split-first-new-user-eval.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const COUNTS = [1, 3, 5, 10] as const;
export const METHODS = [
  "train-count", "metadata-score", "supported-adjusted-cosine", "genre-overlap",
  "v1-positive-pair-graph", "plain-mf", "positive-pair-mf", "unit-positive-pair-mf",
  "shrunk-positive-pair-mf", "hybrid-default-0.5",
] as const;
export type Method = typeof METHODS[number];
type BaseBundle = ReturnType<typeof parseEvalBundle>;
type CompactModel = ReturnType<typeof parseCompactModel>;
type Variant = { modelSha256: string; model: CompactModel };
export type BaselineBundle = {
  specSha256: string; base: BaseBundle;
  models: { plain: Variant; unitPositive: Variant; shrunkPositive: Variant };
  trainCounts: ReadonlyMap<number, number>;
  similarityPairs: readonly { leftAnimeId: number; rightAnimeId: number;
    support: number; adjustedCosine: number; weight: number }[];
  audit: { positivePairEdges: number; nonpositivePairEdgesExcluded: number;
    similarity: { observedPairs: number; lowSupportPairs: number;
      definedSupportedPairs: number; nonpositiveSupportedPairs: number;
      positiveSupportedPairs: number } };
};
export type BaselineSpec = {
  format: "split-first-baseline-ablation-spec-v1";
  fitSnapshotSha256: string; fitManifestSha256: string; metadataSha256: string;
  validationSha256: string; mfCandidatesSha256: string;
  modelCandidateId: "graph-two-epochs"; suppliedCounts: number[];
  positiveRawScoreMin: 7; topK: 10; similarityMinSupport: 2;
  similarityShrinkage: 2; graphShrinkage: 2; hybridModelWeight: 0.5;
  missingSignalScore: 0; tieBreak: "anime-id-ascending";
  objective: "mean-displayed-ndcg-at-10"; methods: string[];
};
export type MethodCase = {
  method: Method; eligibleCandidateIds: number[]; ranked: { animeId: number; score: number }[];
  displayedIds: number[]; signalCandidates: number; selectorRemoved: number;
  hitAt10: number; recallAt10: number; ndcgAt10: number;
};
export type BaselineCase = {
  userNumber: number; suppliedCount: number; universeIds: number[];
  universeSha256: string; eligiblePositiveIds: number[]; likedSourceIds: number[];
  methods: MethodCase[];
};

function fail(field: string): never { throw new Error(`Invalid baseline ablation ${field}.`); }
function record(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field);
  return value as Record<string, unknown>;
}
function fields(value: Record<string, unknown>, expected: readonly string[], field: string): void {
  if (Object.keys(value).sort().join("|") !== [...expected].sort().join("|")) fail(`${field} fields`);
}
function array(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) fail(field);
  return value;
}
function integer(value: unknown, field: string, min = 0): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min) fail(field);
  return value;
}
function finite(value: unknown, field: string, min = -Infinity, max = Infinity): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) fail(field);
  return value;
}
function digest(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const item = value as Record<string, unknown>;
    return `{${Object.keys(item).sort()
      .map((key) => `${JSON.stringify(key)}:${canonical(item[key])}`).join(",")}}`;
  }
  const encoded = JSON.stringify(value);
  if (encoded === undefined) fail("canonical value");
  return encoded;
}
function fileSha(path: string): string {
  const bytes = readFileSync(resolve(ROOT, path));
  return digest(Buffer.from(bytes.toString("latin1").replace(/\r\n/g, "\n"), "latin1"));
}
function readJson(path: string): unknown {
  return JSON.parse(readFileSync(resolve(ROOT, path), "utf8"));
}

export function parseBaselineSpec(value: unknown): BaselineSpec {
  const item = record(value, "specification");
  fields(item, ["format", "fitSnapshotSha256", "fitManifestSha256", "metadataSha256",
    "validationSha256", "mfCandidatesSha256", "modelCandidateId", "suppliedCounts",
    "positiveRawScoreMin", "topK", "similarityMinSupport", "similarityShrinkage",
    "graphShrinkage", "hybridModelWeight", "missingSignalScore", "tieBreak",
    "objective", "methods"], "specification");
  for (const key of ["fitSnapshotSha256", "fitManifestSha256", "metadataSha256",
    "validationSha256", "mfCandidatesSha256"]) {
    if (typeof item[key] !== "string" || !/^[a-f0-9]{64}$/.test(item[key])) fail(`specification.${key}`);
  }
  if (item.format !== "split-first-baseline-ablation-spec-v1" ||
      item.modelCandidateId !== "graph-two-epochs" ||
      JSON.stringify(item.suppliedCounts) !== JSON.stringify(COUNTS) ||
      item.positiveRawScoreMin !== 7 || item.topK !== 10 ||
      item.similarityMinSupport !== 2 || item.similarityShrinkage !== 2 ||
      item.graphShrinkage !== 2 || item.hybridModelWeight !== 0.5 ||
      item.missingSignalScore !== 0 || item.tieBreak !== "anime-id-ascending" ||
      item.objective !== "mean-displayed-ndcg-at-10" ||
      JSON.stringify(item.methods) !== JSON.stringify(METHODS)) fail("specification protocol");
  return item as BaselineSpec;
}

export function assertPinnedInputs(spec: BaselineSpec): void {
  const inputs: [keyof BaselineSpec, string][] = [
    ["fitSnapshotSha256", "fixtures/synthetic-new-user-fit.json"],
    ["fitManifestSha256", "fixtures/synthetic-new-user-fit-manifest.json"],
    ["metadataSha256", "fixtures/synthetic-new-user-anime-metadata.json"],
    ["validationSha256", "fixtures/synthetic-new-user-validation.json"],
    ["mfCandidatesSha256", "fixtures/synthetic-mf-candidates.json"],
  ];
  for (const [key, path] of inputs) {
    if (fileSha(path) !== spec[key]) fail(`${key} stale input`);
  }
}

export function parseBaselineBundle(value: unknown, spec: BaselineSpec): BaselineBundle {
  const root = record(value, "bundle");
  fields(root, ["format", "specSha256", "baseBundle", "models", "trainCounts",
    "similarityPairs", "audit"], "bundle");
  if (root.format !== "split-first-baseline-bundle-v1" ||
      root.specSha256 !== digest(canonical(spec))) fail("bundle format/spec hash");
  const base = parseEvalBundle(root.baseBundle);
  const catalogIds = new Set(base.catalog.map((item) => item.animeId));
  const models = record(root.models, "models");
  fields(models, ["plain", "unitPositive", "shrunkPositive"], "models");
  const parseVariant = (name: "plain" | "unitPositive" | "shrunkPositive"): Variant => {
    const item = record(models[name], `models.${name}`);
    fields(item, ["modelSha256", "model"], `models.${name}`);
    if (typeof item.modelSha256 !== "string" ||
        !/^[a-f0-9]{64}$/.test(item.modelSha256)) fail(`models.${name}.modelSha256`);
    const model = parseCompactModel(item.model, `baseline ${name} model`);
    if (model.animeIds.length !== catalogIds.size ||
        model.animeIds.some((id) => !catalogIds.has(id))) fail(`models.${name}.catalog`);
    return { modelSha256: item.modelSha256, model };
  };
  const trainCounts = new Map<number, number>();
  for (const [i, raw] of array(root.trainCounts, "trainCounts").entries()) {
    const item = record(raw, `trainCounts[${i}]`);
    fields(item, ["animeId", "count"], `trainCounts[${i}]`);
    const id = integer(item.animeId, `trainCounts[${i}].animeId`, 1);
    if (!catalogIds.has(id) || trainCounts.has(id)) fail(`trainCounts[${i}] catalog/duplicate`);
    trainCounts.set(id, integer(item.count, `trainCounts[${i}].count`, 1));
  }
  if (trainCounts.size !== catalogIds.size) fail("trainCounts coverage");
  const seenPairs = new Set<string>();
  const similarityPairs = array(root.similarityPairs, "similarityPairs").map((raw, i) => {
    const item = record(raw, `similarityPairs[${i}]`);
    fields(item, ["leftAnimeId", "rightAnimeId", "support", "adjustedCosine", "weight"],
      `similarityPairs[${i}]`);
    const leftAnimeId = integer(item.leftAnimeId, `similarityPairs[${i}].leftAnimeId`, 1);
    const rightAnimeId = integer(item.rightAnimeId, `similarityPairs[${i}].rightAnimeId`, 1);
    const support = integer(item.support, `similarityPairs[${i}].support`, spec.similarityMinSupport);
    const adjustedCosine = finite(item.adjustedCosine,
      `similarityPairs[${i}].adjustedCosine`, Number.MIN_VALUE, 1);
    const weight = finite(item.weight, `similarityPairs[${i}].weight`, Number.MIN_VALUE, 1);
    const key = `${leftAnimeId}:${rightAnimeId}`;
    if (!catalogIds.has(leftAnimeId) || !catalogIds.has(rightAnimeId) ||
        leftAnimeId >= rightAnimeId || seenPairs.has(key) ||
        Math.abs(weight - adjustedCosine * support / (support + spec.similarityShrinkage)) > 1e-12) {
      fail(`similarityPairs[${i}] semantics/duplicate`);
    }
    seenPairs.add(key);
    return { leftAnimeId, rightAnimeId, support, adjustedCosine, weight };
  });
  const audit = record(root.audit, "audit");
  fields(audit, ["positivePairEdges", "nonpositivePairEdgesExcluded", "similarity"], "audit");
  const similarity = record(audit.similarity, "audit.similarity");
  fields(similarity, ["observedPairs", "lowSupportPairs", "definedSupportedPairs",
    "nonpositiveSupportedPairs", "positiveSupportedPairs"], "audit.similarity");
  const checkedSimilarity = {
    observedPairs: integer(similarity.observedPairs, "audit.similarity.observedPairs"),
    lowSupportPairs: integer(similarity.lowSupportPairs, "audit.similarity.lowSupportPairs"),
    definedSupportedPairs: integer(similarity.definedSupportedPairs,
      "audit.similarity.definedSupportedPairs"),
    nonpositiveSupportedPairs: integer(similarity.nonpositiveSupportedPairs,
      "audit.similarity.nonpositiveSupportedPairs"),
    positiveSupportedPairs: integer(similarity.positiveSupportedPairs,
      "audit.similarity.positiveSupportedPairs"),
  };
  if (checkedSimilarity.positiveSupportedPairs !== similarityPairs.length ||
      integer(audit.positivePairEdges, "audit.positivePairEdges") !== base.positivePairs.length) {
    fail("audit counts");
  }
  return {
    specSha256: root.specSha256, base,
    models: { plain: parseVariant("plain"), unitPositive: parseVariant("unitPositive"),
      shrunkPositive: parseVariant("shrunkPositive") },
    trainCounts, similarityPairs,
    audit: { positivePairEdges: base.positivePairs.length,
      nonpositivePairEdgesExcluded: integer(audit.nonpositivePairEdgesExcluded,
        "audit.nonpositivePairEdgesExcluded"), similarity: checkedSimilarity },
  };
}

export function loadBaselineBundle(spec: BaselineSpec): BaselineBundle {
  assertPinnedInputs(spec);
  const output = execFileSync("python", ["ml/split_first_baseline_export.py"],
    { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  return parseBaselineBundle(JSON.parse(output), spec);
}

function modelIndex(model: CompactModel): ModelRecommendationIndex {
  return { generatedAt: model.generatedAt, factors: model.factors,
    globalMean: model.globalMean,
    animeByAnimeId: new Map(model.animeIds.map((animeId, i) =>
      [animeId, { animeId, title: model.titles[i], bias: model.biases[i],
        embedding: model.embeddings[i] }])) };
}
function localHistory(animeId: number, rawScore: number | null, title: string): HistoryEntry {
  return { provider: "local", sourceId: String(animeId), title, animeId,
    status: "completed", sourceStatus: "completed", progressEpisodes: null,
    score: rawScore, scoreScale: "local-10" };
}
function asResults(ids: readonly number[], byId: ReadonlyMap<number, RecommendationResult>,
                   index: ReturnType<typeof buildRecommendationIndex>): RecommendationResult[] {
  return ids.map((id) => byId.get(id) ?? {
    anime: index.animeByAnimeId.get(id)!, score: 0, strongest: 0,
    supportCount: 0, contributions: [],
  }).sort((a, b) => b.score - a.score || a.anime.animeId - b.anime.animeId);
}
function fromScores(ids: readonly number[], score: (id: number) => number,
                    index: ReturnType<typeof buildRecommendationIndex>): RecommendationResult[] {
  return ids.map((id) => {
    const value = score(id);
    if (!Number.isFinite(value)) fail(`score ${id}`);
    return { anime: index.animeByAnimeId.get(id)!, score: value,
      strongest: 0, supportCount: 0, contributions: [] };
  }).sort((a, b) => b.score - a.score || a.anime.animeId - b.anime.animeId);
}
function sourceMap(items: readonly RecommendationResult[]): Map<number, RecommendationResult> {
  return new Map(items.map((item) => [item.anime.animeId, item]));
}

export function evaluateBaselineCases(bundle: BaselineBundle, fixture: EvalFixture,
                                      spec: BaselineSpec): BaselineCase[] {
  if (fixture.topK !== spec.topK || fixture.positiveRawScoreMin !== spec.positiveRawScoreMin) {
    fail("cohort protocol");
  }
  const base = bundle.base;
  const nodes: GraphData["nodes"] = base.catalog.map((item) =>
    ({ id: `anime:${item.animeId}`, label: item.title, nodeType: "anime" }));
  const edges: GraphData["edges"] = base.positivePairs.map((item) => ({
    id: `aa:${item.leftAnimeId}:${item.rightAnimeId}`,
    source: `anime:${item.leftAnimeId}`, target: `anime:${item.rightAnimeId}`,
    edgeType: "anime-anime", weight: item.weight, support: item.support,
  }));
  const index = buildRecommendationIndex({ generatedAt: base.model.generatedAt,
    userCount: base.fitUserCount, animeCount: nodes.length, nodeCount: nodes.length,
    edgeCount: edges.length, nodes, edges });
  const metadata = new Map<number, AnimeMetadata>(fixture.candidateMetadata.map((item) =>
    [item.animeId, { ...item, studios: [], synopsis: "", imageUrl: "", season: null }]));
  const titleById = new Map(base.catalog.map((item) => [item.animeId, item.title]));
  const model = {
    plain: modelIndex(bundle.models.plain.model),
    weighted: modelIndex(base.model),
    unit: modelIndex(bundle.models.unitPositive.model),
    shrunk: modelIndex(bundle.models.shrunkPositive.model),
  };
  const sim = new Map<string, number>();
  for (const pair of bundle.similarityPairs) {
    sim.set(`${pair.leftAnimeId}:${pair.rightAnimeId}`, pair.weight);
  }
  const cases: BaselineCase[] = [];
  for (const [userNumber, user] of fixture.users.entries()) {
    const ordered = orderedObserved(user, fixture.seed);
    const history = user.historySeen.map((id) => localHistory(id, null, titleById.get(id)!));
    for (const suppliedCount of COUNTS) {
      const preferences: AnimePreference[] = ordered.slice(0, suppliedCount).map((rating) =>
        preferenceFromHistory(localHistory(rating.animeId, rating.rawScore,
          titleById.get(rating.animeId)!), `anime:${rating.animeId}`)!);
      const policy = createCandidateEligibilityPolicy({
        index, preferences, history,
        includeOnlyNodeIds: user.includeOnly.map((id) => `anime:${id}`),
        excludeNodeIds: user.exclude.map((id) => `anime:${id}`), filters: user.filters,
      });
      const probes = index.animeList.map((anime) =>
        ({ anime, score: 0, strongest: 0, supportCount: 0, contributions: [] }));
      const universeIds = policy.evaluate(probes, metadata).recommendations
        .map((item) => item.anime.animeId).sort((a, b) => a - b);
      const eligible = new Set(universeIds);
      const eligiblePositiveIds = user.validation.filter((rating) =>
        rating.rawScore >= spec.positiveRawScoreMin && eligible.has(rating.animeId))
        .map((rating) => rating.animeId);
      const graph = buildGraphRecommendationsForPreferences(preferences, index);
      const content = buildGenreOverlapExploration(preferences, index, metadata);
      const modelScores = {
        plain: buildModelRecommendationsForPreferences(preferences, index, model.plain),
        weighted: buildModelRecommendationsForPreferences(preferences, index, model.weighted),
        unit: buildModelRecommendationsForPreferences(preferences, index, model.unit),
        shrunk: buildModelRecommendationsForPreferences(preferences, index, model.shrunk),
      };
      const liked = preferences.flatMap((item) => {
        if (item.sentiment !== "liked") return [];
        const id = index.animeByNodeId.get(item.nodeId)?.animeId;
        return id === undefined ? [] : [{ id, weight: item.importance * item.confidence }];
      });
      const similarityScore = (candidateId: number) => liked.reduce((total, source) => {
        const key = `${Math.min(source.id, candidateId)}:${Math.max(source.id, candidateId)}`;
        return total + (sim.get(key) ?? 0) * source.weight;
      }, 0);
      const rawSources: Record<Exclude<Method, "hybrid-default-0.5">, RecommendationResult[]> = {
        "train-count": fromScores(universeIds, (id) => bundle.trainCounts.get(id)!, index),
        "metadata-score": fromScores(universeIds, (id) => metadata.get(id)!.score!, index),
        "supported-adjusted-cosine": fromScores(universeIds, similarityScore, index),
        "genre-overlap": content,
        "v1-positive-pair-graph": graph,
        "plain-mf": modelScores.plain,
        "positive-pair-mf": modelScores.weighted,
        "unit-positive-pair-mf": modelScores.unit,
        "shrunk-positive-pair-mf": modelScores.shrunk,
      };
      const graphComplete = asResults(universeIds, sourceMap(graph), index);
      const weightedComplete = asResults(universeIds, sourceMap(modelScores.weighted), index);
      const watchedIds = new Set([...ordered.slice(0, suppliedCount).map((item) => item.animeId),
        ...user.historySeen]);
      const methods: MethodCase[] = METHODS.map((method) => {
        const raw = method === "hybrid-default-0.5" ? [...graph, ...modelScores.weighted]
          : rawSources[method];
        const signalCandidates = new Set(raw.filter((item) =>
          eligible.has(item.anime.animeId) && item.score !== 0).map((item) => item.anime.animeId)).size;
        let ranked: RecommendationResult[];
        if (method === "hybrid-default-0.5") {
          ranked = rankEligibleCandidates("hybrid", { graph: graphComplete,
            model: weightedComplete }, policy, metadata, spec.hybridModelWeight).recommendations;
        } else {
          const complete = asResults(universeIds, sourceMap(raw), index);
          const mode = method === "v1-positive-pair-graph" ? "graph"
            : method.endsWith("-mf") ? "model" : "fallback";
          ranked = rankEligibleCandidates(mode, { [mode]: complete },
            policy, metadata).recommendations;
        }
        const eligibleCandidateIds = ranked.map((item) => item.anime.animeId).sort((a, b) => a - b);
        if (JSON.stringify(eligibleCandidateIds) !== JSON.stringify(universeIds)) {
          fail(`${method} candidate universe drift`);
        }
        const displayed = selectFranchiseDiverseRecommendations(ranked, metadata, watchedIds,
          false, titleById).recommendations;
        const ranks = eligiblePositiveIds.map((id) => {
          const position = displayed.findIndex((item) => item.anime.animeId === id);
          return position < 0 ? null : position + 1;
        });
        const metrics = eligiblePositiveIds.length
          ? metricsForRanks(ranks, eligiblePositiveIds.length, spec.topK)
          : { hitAtK: 0, recallAtK: 0, ndcgAtK: 0 };
        return { method, eligibleCandidateIds,
          ranked: ranked.map((item) => ({ animeId: item.anime.animeId, score: item.score })),
          displayedIds: displayed.map((item) => item.anime.animeId),
          signalCandidates, selectorRemoved: ranked.length - displayed.length,
          hitAt10: metrics.hitAtK, recallAt10: metrics.recallAtK, ndcgAt10: metrics.ndcgAtK };
      });
      cases.push({ userNumber: userNumber + 1, suppliedCount, universeIds,
        universeSha256: digest(JSON.stringify(universeIds)), eligiblePositiveIds,
        likedSourceIds: liked.map((item) => item.id), methods });
    }
  }
  return cases;
}

export function summarizeBaselineCases(bundle: BaselineBundle, cases: readonly BaselineCase[]) {
  const measurable = cases.filter((item) => item.eligiblePositiveIds.length > 0);
  if (cases.length !== 16 || measurable.length === 0) fail("case coverage");
  const mean = (rows: readonly MethodCase[], field: "ndcgAt10" | "hitAt10" | "recallAt10") =>
    rows.reduce((sum, item) => sum + item[field], 0) / rows.length;
  return {
    format: "split-first-baseline-validation-v1",
    specSha256: bundle.specSha256, trainSha256: bundle.base.trainSha256,
    fitSha256: bundle.base.fitSha256, graphModelSha256: bundle.base.modelSha256,
    modelSha256: { plain: bundle.models.plain.modelSha256,
      unitPositive: bundle.models.unitPositive.modelSha256,
      shrunkPositive: bundle.models.shrunkPositive.modelSha256 },
    fitTrainRows: bundle.base.trainRowCount, catalogCount: bundle.base.catalog.length,
    cases: cases.length, measurableCases: measurable.length,
    eligiblePositiveLabels: measurable.reduce((sum, item) =>
      sum + item.eligiblePositiveIds.length, 0),
    candidateUniverses: cases.map((item) => ({ userNumber: item.userNumber,
      suppliedCount: item.suppliedCount, count: item.universeIds.length,
      sha256: item.universeSha256 })),
    graphAudit: bundle.audit,
    methods: METHODS.map((method) => {
      const rows = measurable.map((item) => item.methods.find((row) => row.method === method)!);
      return { method, meanNdcgAt10: mean(rows, "ndcgAt10"),
        meanHitAt10: mean(rows, "hitAt10"), meanRecallAt10: mean(rows, "recallAt10"),
        meanSignalCandidates: rows.reduce((sum, item) => sum + item.signalCandidates, 0) / rows.length,
        selectorRemoved: rows.reduce((sum, item) => sum + item.selectorRemoved, 0),
        perPrefix: COUNTS.map((suppliedCount) => {
          const matching = measurable.filter((item) => item.suppliedCount === suppliedCount)
            .map((item) => item.methods.find((row) => row.method === method)!);
          return { suppliedCount, measurableCases: matching.length,
            meanNdcgAt10: matching.length ? mean(matching, "ndcgAt10") : 0 };
        }) };
    }),
  };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const spec = parseBaselineSpec(readJson("fixtures/synthetic-baseline-ablation-spec.json"));
  const bundle = loadBaselineBundle(spec);
  const fixture = parseEvalFixture(readJson("fixtures/synthetic-new-user-validation.json"),
    bundle.base, fitUserIds(readJson("fixtures/synthetic-new-user-fit.json")));
  process.stdout.write(JSON.stringify(summarizeBaselineCases(bundle,
    evaluateBaselineCases(bundle, fixture, spec)), null, 2) + "\n");
}
