/** Invented M5.9 candidates on the exact M5.6 new-user eligibility and ranking cases. */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseCompactModel } from "../src/artifacts.ts";
import {
  assertPinnedInputs, evaluateBaselineCases, loadBaselineBundle,
  parseBaselineSpec, summarizeBaselineCases,
} from "./split-first-baseline-ablation.ts";
import type { BaselineBundle, BaselineCase, BaselineSpec, ExperimentalMethod,
  MethodCase } from "./split-first-baseline-ablation.ts";
import { fitUserIds, parseEvalFixture } from "./split-first-new-user-eval.ts";
import type { EvalFixture } from "./split-first-new-user-eval.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
export const EXPERIMENT_METHODS = ["lightgcn-bpr", "content-multihot"] as const;
type CompactModel = ReturnType<typeof parseCompactModel>;
type ExperimentSpec = {
  format: string; baselineSpecSha256: string; contentMetadataSha256: string;
  methods: ExperimentalMethod[]; suppliedCounts: number[]; topK: number;
  positiveRawScoreMin: number; missingSignalScore: number; tieBreak: string;
  objective: string; lightgcn: Record<string, unknown>; content: Record<string, unknown>;
};
export type ExperimentBundle = {
  specSha256: string; trainSha256: string; fitSha256: string;
  trainRowCount: number; positiveTrainInteractions: number; positivePairEdges: number;
  contentVocabulary: string[];
  models: Record<ExperimentalMethod, { modelSha256: string; model: CompactModel }>;
};

function fail(field: string): never { throw new Error(`Invalid split-first experiment ${field}.`); }
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
function sha(value: unknown, field: string): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) fail(field);
  return value;
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const item = value as Record<string, unknown>;
    return `{${Object.keys(item).sort().map((key) =>
      `${JSON.stringify(key)}:${canonical(item[key])}`).join(",")}}`;
  }
  const encoded = JSON.stringify(value);
  if (encoded === undefined) fail("canonical value");
  return encoded;
}
function digest(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}
function fileSha(name: string): string {
  const bytes = readFileSync(resolve(ROOT, "fixtures", name));
  return digest(Buffer.from(bytes.toString("latin1").replace(/\r\n/g, "\n"), "latin1"));
}
function fixture(name: string): unknown {
  return JSON.parse(readFileSync(resolve(ROOT, "fixtures", name), "utf8"));
}

export function parseExperimentSpec(value: unknown): ExperimentSpec {
  const spec = record(value, "specification");
  fields(spec, ["format", "baselineSpecSha256", "contentMetadataSha256", "methods",
    "suppliedCounts", "topK", "positiveRawScoreMin", "missingSignalScore",
    "tieBreak", "objective", "lightgcn", "content"], "specification");
  sha(spec.baselineSpecSha256, "specification.baselineSpecSha256");
  sha(spec.contentMetadataSha256, "specification.contentMetadataSha256");
  const gcn = record(spec.lightgcn, "specification.lightgcn");
  const content = record(spec.content, "specification.content");
  const expectedGcn = { factors: 8, layers: 1, epochs: 2, batchSize: 128,
    lr: 0.01, reg: 0.0001, seed: 42, centeredPositiveThreshold: 0,
    pairEdges: "train-positive-only" };
  const expectedContent = { featureFields: ["genres", "studios"],
    dropUniversalTokens: true, normalize: "item-l2",
    scorer: "browser-signed-item-vector" };
  if (spec.format !== "split-first-experiment-spec-v1" ||
      canonical(spec.methods) !== canonical(EXPERIMENT_METHODS) ||
      canonical(spec.suppliedCounts) !== canonical([1, 3, 5, 10]) ||
      spec.topK !== 10 || spec.positiveRawScoreMin !== 7 ||
      spec.missingSignalScore !== 0 || spec.tieBreak !== "anime-id-ascending" ||
      spec.objective !== "mean-displayed-ndcg-at-10" ||
      canonical(gcn) !== canonical(expectedGcn) ||
      canonical(content) !== canonical(expectedContent)) fail("specification protocol");
  return spec as ExperimentSpec;
}

export function assertExperimentInputs(spec: ExperimentSpec, baseline: BaselineSpec,
                                       fixtureData: EvalFixture): void {
  assertPinnedInputs(baseline);
  if (fileSha("synthetic-baseline-ablation-spec.json") !== spec.baselineSpecSha256 ||
      fileSha("synthetic-experiment-content-metadata.json") !== spec.contentMetadataSha256 ||
      fixtureData.topK !== spec.topK ||
      fixtureData.positiveRawScoreMin !== spec.positiveRawScoreMin) fail("stale input/protocol");
  const content = record(fixture("synthetic-experiment-content-metadata.json"), "content metadata");
  fields(content, ["format", "source", "anime"], "content metadata");
  if (content.format !== "experiment-content-metadata-v1" ||
      content.source !== "invented-fixture") fail("content metadata format/source");
  const genres = new Map<number, unknown>();
  for (const [i, raw] of array(content.anime, "content metadata.anime").entries()) {
    const item = record(raw, `content metadata.anime[${i}]`);
    fields(item, ["animeId", "genres", "studios"], `content metadata.anime[${i}]`);
    const id = integer(item.animeId, `content metadata.anime[${i}].animeId`, 1);
    if (genres.has(id) || !Array.isArray(item.genres) || !Array.isArray(item.studios)) {
      fail(`content metadata.anime[${i}]`);
    }
    genres.set(id, item.genres);
  }
  if (genres.size !== fixtureData.candidateMetadata.length ||
      fixtureData.candidateMetadata.some((item) =>
        canonical(genres.get(item.animeId)) !== canonical(item.genres))) {
    fail("content metadata genre/catalog mismatch");
  }
}

export function parseExperimentBundle(value: unknown, spec: ExperimentSpec,
                                      baseline: BaselineBundle): ExperimentBundle {
  const root = record(value, "bundle");
  fields(root, ["format", "specSha256", "baselineSpecSha256", "trainSha256",
    "fitSha256", "metadataSha256", "contentMetadataSha256", "trainRowCount",
    "positiveTrainInteractions", "positivePairEdges", "contentVocabulary", "models"],
  "bundle");
  if (root.format !== "split-first-experiment-bundle-v1" ||
      root.specSha256 !== digest(canonical(spec)) ||
      root.baselineSpecSha256 !== spec.baselineSpecSha256 ||
      root.contentMetadataSha256 !== spec.contentMetadataSha256 ||
      root.trainSha256 !== baseline.base.trainSha256 ||
      root.fitSha256 !== baseline.base.fitSha256 ||
      root.metadataSha256 !== baseline.base.metadataSha256 ||
      root.trainRowCount !== baseline.base.trainRowCount ||
      root.positivePairEdges !== baseline.audit.positivePairEdges) fail("bundle provenance");
  const positiveTrainInteractions = integer(root.positiveTrainInteractions,
    "bundle.positiveTrainInteractions", 1);
  if (positiveTrainInteractions > baseline.base.trainRowCount) fail("positive train count");
  const vocabulary = array(root.contentVocabulary, "bundle.contentVocabulary");
  if (!vocabulary.length || vocabulary.some((token) => typeof token !== "string" ||
      !/^(genre|studio):\S/.test(token)) ||
      canonical(vocabulary) !== canonical([...vocabulary].sort()) ||
      new Set(vocabulary).size !== vocabulary.length) fail("content vocabulary");
  const variants = record(root.models, "bundle.models");
  fields(variants, EXPERIMENT_METHODS, "bundle.models");
  const catalog = baseline.base.catalog;
  const models = {} as ExperimentBundle["models"];
  for (const method of EXPERIMENT_METHODS) {
    const item = record(variants[method], `bundle.models.${method}`);
    fields(item, ["modelSha256", "model"], `bundle.models.${method}`);
    const model = parseCompactModel(item.model, `experiment ${method} model`);
    if (sha(item.modelSha256, `bundle.models.${method}.modelSha256`) !==
        digest(canonical(item.model)) || model.animeIds.length !== catalog.length ||
        model.animeIds.some((id, i) => id !== catalog[i].animeId ||
          model.titles[i] !== catalog[i].title || model.biases[i] !== 0) ||
        model.globalMean !== 0 ||
        (method === "lightgcn-bpr" ? model.factors !== 8 :
          model.factors !== vocabulary.length)) fail(`bundle.models.${method} provenance/catalog`);
    models[method] = { modelSha256: item.modelSha256 as string, model };
  }
  return { specSha256: root.specSha256 as string,
    trainSha256: root.trainSha256 as string, fitSha256: root.fitSha256 as string,
    trainRowCount: root.trainRowCount as number, positiveTrainInteractions,
    positivePairEdges: root.positivePairEdges as number,
    contentVocabulary: vocabulary as string[], models };
}

export function loadExperimentBundle(spec: ExperimentSpec,
                                     baseline: BaselineBundle): ExperimentBundle {
  const output = execFileSync("python", ["ml/split_first_experiment_export.py"],
    { cwd: ROOT, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  return parseExperimentBundle(JSON.parse(output), spec, baseline);
}

export function evaluateExperimentCases(baseline: BaselineBundle, fixtureData: EvalFixture,
                                        baselineSpec: BaselineSpec, spec: ExperimentSpec,
                                        experiment: ExperimentBundle): BaselineCase[] {
  assertExperimentInputs(spec, baselineSpec, fixtureData);
  if (spec.baselineSpecSha256 !== fileSha("synthetic-baseline-ablation-spec.json") ||
      experiment.specSha256 !== digest(canonical(spec))) fail("evaluation specification");
  return evaluateBaselineCases(baseline, fixtureData, baselineSpec,
    EXPERIMENT_METHODS.map((method) => ({ method,
      model: experiment.models[method].model })));
}

export function summarizeExperimentCases(cases: readonly BaselineCase[]) {
  const measurable = cases.filter((item) => item.eligiblePositiveIds.length > 0);
  if (cases.length !== 16 || measurable.length !== 16) fail("case coverage");
  return EXPERIMENT_METHODS.map((method) => {
    const rows = measurable.map((item) => item.methods.find((row) => row.method === method));
    if (rows.some((row) => !row)) fail(`${method} missing case`);
    const results = rows as MethodCase[];
    const mean = (field: "hitAt10" | "recallAt10" | "ndcgAt10") =>
      results.reduce((sum, row) => sum + row[field], 0) / results.length;
    return { method, meanHitAt10: mean("hitAt10"), meanRecallAt10: mean("recallAt10"),
      meanNdcgAt10: mean("ndcgAt10"),
      meanSignalCandidates: results.reduce((sum, row) => sum + row.signalCandidates, 0) /
        results.length,
      perPrefix: [1, 3, 5, 10].map((suppliedCount) => {
        const selected = measurable.filter((row) => row.suppliedCount === suppliedCount)
          .map((row) => row.methods.find((item) => item.method === method)!);
        return { suppliedCount, meanNdcgAt10: selected.reduce((sum, row) =>
          sum + row.ndcgAt10, 0) / selected.length };
      }) };
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  const baselineSpec = parseBaselineSpec(fixture("synthetic-baseline-ablation-spec.json"));
  const spec = parseExperimentSpec(fixture("synthetic-experiment-spec.json"));
  const baseline = loadBaselineBundle(baselineSpec);
  const cohort = parseEvalFixture(fixture("synthetic-new-user-validation.json"), baseline.base,
    fitUserIds(fixture("synthetic-new-user-fit.json")));
  assertExperimentInputs(spec, baselineSpec, cohort);
  const experiment = loadExperimentBundle(spec, baseline);
  const cases = evaluateExperimentCases(baseline, cohort, baselineSpec, spec, experiment);
  process.stdout.write(JSON.stringify({
    format: "split-first-experiment-validation-v1",
    status: "invented-engineering-check-only", specSha256: experiment.specSha256,
    trainSha256: experiment.trainSha256, fitSha256: experiment.fitSha256,
    positiveTrainInteractions: experiment.positiveTrainInteractions,
    positivePairEdges: experiment.positivePairEdges,
    contentVocabulary: experiment.contentVocabulary,
    models: Object.fromEntries(EXPERIMENT_METHODS.map((method) =>
      [method, experiment.models[method].modelSha256])),
    baseline: summarizeBaselineCases(baseline, cases),
    experiments: summarizeExperimentCases(cases),
  }, null, 2) + "\n");
}
