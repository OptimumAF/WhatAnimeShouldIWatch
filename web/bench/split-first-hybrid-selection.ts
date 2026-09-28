/** Frozen validation-only browser hybrid selection and one-use final reporting. */
import { createHash } from "node:crypto";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import {
  evaluateNewUsers, fitUserIds, loadEvalBundle, metricsForRanks, parseEvalFixture,
} from "./split-first-new-user-eval.ts";
import type { EvalFixture } from "./split-first-new-user-eval.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const COUNTS = [1, 3, 5, 10] as const;
const WEIGHTS = [0, 0.25, 0.5, 0.75, 1] as const;
type Bundle = ReturnType<typeof loadEvalBundle>;
export type HybridSpec = {
  format: "split-first-hybrid-candidates-v1";
  objective: "mean-displayed-ndcg-at-10";
  modelCandidateId: "graph-two-epochs";
  suppliedCounts: number[];
  positiveRawScoreMin: 7;
  topK: 10;
  tieBreak: "smaller-model-weight";
  validationFixtureSha256: string;
  testFixtureSha256: string;
  modelWeights: number[];
};
export type HybridTrial = {
  modelWeight: number; cases: number; measurableCases: number;
  eligiblePositiveLabels: number; meanNdcgAt10: number;
  meanHitAt10: number; meanRecallAt10: number;
  perPrefix: { suppliedCount: number; measurableCases: number; meanNdcgAt10: number }[];
  displayedEngines: { hybrid: number; graph: number; coverage: number };
};
type Selection = {
  format: "split-first-hybrid-selection-v1";
  selectionPath: string; testReportPath: string;
  validationFixturePath: string; testFixturePath: string;
  validationFixtureSha256: string; testFixtureSha256: string;
  trainSha256: string; fitSha256: string; modelSha256: string; metadataSha256: string;
  candidateSpec: HybridSpec; candidateSpecSha256: string;
  trials: HybridTrial[]; selectedModelWeight: number;
  selectionSha256: string;
};
type Paths = {
  validationFixture: string; testFixture: string; candidates: string;
  selection: string; testReport: string;
};

function fail(field: string): never { throw new Error(`Invalid hybrid selection ${field}.`); }
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function keys(value: Record<string, unknown>, expected: readonly string[], field: string): void {
  if (Object.keys(value).sort().join("|") !== [...expected].sort().join("|")) fail(`${field} fields`);
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (isRecord(value)) return `{${Object.keys(value).sort()
    .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(",")}}`;
  const encoded = JSON.stringify(value);
  if (encoded === undefined) fail("canonical value");
  return encoded;
}
function sha(value: unknown): string {
  return createHash("sha256").update(canonical(value)).digest("hex");
}
function fileSha(path: string): string {
  const bytes = readFileSync(path);
  const normalized = Buffer.from(bytes.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  return createHash("sha256").update(normalized).digest("hex");
}
function readJson(path: string): unknown {
  return JSON.parse(readFileSync(path, "utf8"));
}
function privateOutput(path: string): string {
  const result = resolve(path);
  for (const directory of [resolve(ROOT, "web/public"), resolve(ROOT, "release-data")]) {
    if (result === directory || result.startsWith(directory + sep)) {
      fail("public or release output path");
    }
  }
  return result;
}
function writeNew(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  const handle = openSync(path, "wx");
  try {
    writeFileSync(handle, JSON.stringify(value, null, 2) + "\n", "utf8");
  } finally {
    closeSync(handle);
  }
}
function fitIds(): ReadonlySet<string> {
  return fitUserIds(readJson(resolve(ROOT, "fixtures/synthetic-new-user-fit.json")));
}
function bundleHashes(bundle: Bundle) {
  return { trainSha256: bundle.trainSha256, fitSha256: bundle.fitSha256,
    modelSha256: bundle.modelSha256, metadataSha256: bundle.metadataSha256 };
}

export function parseHybridSpec(value: unknown): HybridSpec {
  if (!isRecord(value)) fail("candidate specification");
  keys(value, ["format", "objective", "modelCandidateId", "suppliedCounts",
    "positiveRawScoreMin", "topK", "tieBreak", "validationFixtureSha256",
    "testFixtureSha256", "modelWeights"], "candidate specification");
  if (value.format !== "split-first-hybrid-candidates-v1" ||
      value.objective !== "mean-displayed-ndcg-at-10" ||
      value.modelCandidateId !== "graph-two-epochs" ||
      value.positiveRawScoreMin !== 7 || value.topK !== 10 ||
      value.tieBreak !== "smaller-model-weight" ||
      !["validationFixtureSha256", "testFixtureSha256"].every((key) =>
        typeof value[key] === "string" && /^[a-f0-9]{64}$/.test(value[key])) ||
      JSON.stringify(value.suppliedCounts) !== JSON.stringify(COUNTS) ||
      JSON.stringify(value.modelWeights) !== JSON.stringify(WEIGHTS)) {
    fail("candidate specification protocol");
  }
  return value as HybridSpec;
}

export function scoreHybridValidation(bundle: Bundle, fixture: EvalFixture,
                                      modelWeight: number): HybridTrial {
  if (!WEIGHTS.includes(modelWeight as typeof WEIGHTS[number])) fail("undeclared model weight");
  const rows = evaluateNewUsers(bundle, fixture, "hybrid", modelWeight);
  if (rows.length !== fixture.users.length * COUNTS.length) fail("case coverage");
  const measurable = rows.filter((row) => row.eligiblePositiveLabels > 0);
  if (!measurable.length) fail("no measurable validation cases");
  const scored = measurable.map((row) => ({
    row, metric: metricsForRanks(row.displayedRanks, row.eligiblePositiveLabels, fixture.topK),
  }));
  const average = (field: "ndcgAtK" | "hitAtK" | "recallAtK",
                   items = scored) => items.length
    ? items.reduce((sum, item) => sum + item.metric[field], 0) / items.length : 0;
  return {
    modelWeight, cases: rows.length, measurableCases: measurable.length,
    eligiblePositiveLabels: measurable.reduce((sum, row) => sum + row.eligiblePositiveLabels, 0),
    meanNdcgAt10: average("ndcgAtK"), meanHitAt10: average("hitAtK"),
    meanRecallAt10: average("recallAtK"),
    perPrefix: COUNTS.map((suppliedCount) => {
      const matching = scored.filter((item) => item.row.suppliedCount === suppliedCount);
      return { suppliedCount, measurableCases: matching.length,
        meanNdcgAt10: average("ndcgAtK", matching) };
    }),
    displayedEngines: {
      hybrid: rows.filter((row) => row.displayedEngine === "hybrid").length,
      graph: rows.filter((row) => row.displayedEngine === "graph").length,
      coverage: rows.filter((row) => row.displayedEngine === "coverage").length,
    },
  };
}

export function selectHybrid(bundle: Bundle, paths: Paths): Selection {
  const selectionPath = privateOutput(paths.selection);
  const testReportPath = privateOutput(paths.testReport);
  const markerPath = privateOutput(selectionPath + ".test-used");
  if (selectionPath === testReportPath || existsSync(selectionPath) ||
      existsSync(testReportPath) || existsSync(markerPath)) fail("distinct unused output paths");
  const spec = parseHybridSpec(readJson(paths.candidates));
  if (bundle.candidateId !== spec.modelCandidateId) fail("model candidate");
  const validationFixturePath = resolve(paths.validationFixture);
  const testFixturePath = resolve(paths.testFixture);
  if (validationFixturePath === testFixturePath) fail("separate validation and test fixtures");
  const validationFixtureSha256 = fileSha(validationFixturePath);
  // Only bind the reserved test bytes. Do not parse scores or identities here.
  const testFixtureSha256 = fileSha(testFixturePath);
  if (validationFixtureSha256 !== spec.validationFixtureSha256 ||
      testFixtureSha256 !== spec.testFixtureSha256) fail("predeclared fixture hashes");
  const validation = parseEvalFixture(readJson(validationFixturePath), bundle, fitIds());
  if (validation.topK !== spec.topK ||
      validation.positiveRawScoreMin !== spec.positiveRawScoreMin) fail("validation protocol");
  const trials = spec.modelWeights.map((weight) => scoreHybridValidation(bundle, validation, weight));
  const selectedModelWeight = [...trials].sort((left, right) =>
    right.meanNdcgAt10 - left.meanNdcgAt10 || left.modelWeight - right.modelWeight)[0].modelWeight;
  const payload = {
    format: "split-first-hybrid-selection-v1" as const,
    selectionPath, testReportPath, validationFixturePath, testFixturePath,
    validationFixtureSha256, testFixtureSha256,
    ...bundleHashes(bundle), candidateSpec: spec, candidateSpecSha256: sha(spec),
    trials, selectedModelWeight,
  };
  const result = { ...payload, selectionSha256: sha(payload) };
  writeNew(selectionPath, result);
  return result;
}

function checkedSelection(value: unknown, selectionPath: string, bundle: Bundle): Selection {
  if (!isRecord(value)) fail("frozen record");
  keys(value, ["format", "selectionPath", "testReportPath", "validationFixturePath",
    "testFixturePath", "validationFixtureSha256", "testFixtureSha256", "trainSha256",
    "fitSha256", "modelSha256", "metadataSha256", "candidateSpec", "candidateSpecSha256",
    "trials", "selectedModelWeight", "selectionSha256"], "frozen record");
  if (value.format !== "split-first-hybrid-selection-v1" ||
      value.selectionPath !== selectionPath ||
      value.selectionSha256 !== sha(Object.fromEntries(
        Object.entries(value).filter(([key]) => key !== "selectionSha256")))) {
    fail("frozen integrity or original path");
  }
  const record = value as Selection;
  const spec = parseHybridSpec(record.candidateSpec);
  if (record.candidateSpecSha256 !== sha(spec) ||
      spec.modelCandidateId !== bundle.candidateId ||
      Object.entries(bundleHashes(bundle)).some(([key, hash]) =>
        record[key as keyof Selection] !== hash) ||
      record.validationFixtureSha256 !== fileSha(record.validationFixturePath) ||
      record.testReportPath !== privateOutput(record.testReportPath) ||
      record.testFixturePath === record.validationFixturePath) {
    fail("frozen inputs or output path");
  }
  if (!Array.isArray(record.trials) ||
      record.trials.length !== spec.modelWeights.length ||
      record.trials.some((trial, index) =>
        trial.modelWeight !== spec.modelWeights[index] ||
        typeof trial.meanNdcgAt10 !== "number" ||
        !Number.isFinite(trial.meanNdcgAt10))) {
    fail("frozen validation trials");
  }
  const best = [...record.trials].sort((left, right) =>
    right.meanNdcgAt10 - left.meanNdcgAt10 || left.modelWeight - right.modelWeight)[0];
  if (record.selectedModelWeight !== best.modelWeight) fail("frozen chosen weight");
  return record;
}

export function reportFrozenHybridTest(bundle: Bundle, selectionFile: string): object {
  const selectionPath = privateOutput(selectionFile);
  const record = checkedSelection(readJson(selectionPath), selectionPath, bundle);
  const reportPath = privateOutput(record.testReportPath);
  const markerPath = privateOutput(selectionPath + ".test-used");
  if (existsSync(reportPath) || existsSync(markerPath)) fail("test report or one-use marker already exists");
  const fitUsers = fitIds();
  const validation = parseEvalFixture(readJson(record.validationFixturePath), bundle, fitUsers);
  // This exclusive marker is written before reading or parsing the final cohort.
  writeNew(markerPath, { format: "split-first-hybrid-test-used-v1",
    selectionSha256: record.selectionSha256 });
  if (fileSha(record.testFixturePath) !== record.testFixtureSha256) fail("frozen final fixture hash");
  const final = parseEvalFixture(readJson(record.testFixturePath), bundle, fitUsers, "test");
  const validationIds = new Set(validation.users.map((user) => user.userId));
  if (final.users.some((user) => validationIds.has(user.userId)) ||
      canonical(final.candidateMetadata) !== canonical(validation.candidateMetadata)) {
    fail("final cohort overlap or metadata drift");
  }
  const metric = scoreHybridValidation(bundle, final, record.selectedModelWeight);
  const report = {
    format: "split-first-hybrid-final-test-v1",
    selectionSha256: record.selectionSha256, selectedModelWeight: record.selectedModelWeight,
    modelSha256: bundle.modelSha256, test: metric,
    status: "single invented new-user report; no release claim",
  };
  writeNew(reportPath, report);
  return report;
}

function argumentsFor(command: string, args: string[]): Record<string, string> {
  const allowed = command === "select"
    ? ["validation-fixture", "test-fixture", "candidates", "out-selection", "out-test-report"]
    : command === "report-test" ? ["selection"] : [];
  if (!allowed.length || args.length % 2 !== 0) fail("command arguments");
  const values: Record<string, string> = {};
  for (let index = 0; index < args.length; index += 2) {
    const name = args[index];
    if (!name.startsWith("--") || !allowed.includes(name.slice(2)) ||
        !args[index + 1] || Object.hasOwn(values, name.slice(2))) fail("command arguments");
    values[name.slice(2)] = args[index + 1];
  }
  return values;
}

function main(): void {
  const [command, ...args] = process.argv.slice(2);
  const values = argumentsFor(command, args);
  const bundle = loadEvalBundle();
  if (command === "select") {
    const result = selectHybrid(bundle, {
      validationFixture: values["validation-fixture"] ??
        resolve(ROOT, "fixtures/synthetic-new-user-validation.json"),
      testFixture: values["test-fixture"] ??
        resolve(ROOT, "fixtures/synthetic-new-user-final-test.json"),
      candidates: values.candidates ?? resolve(ROOT, "fixtures/synthetic-hybrid-candidates.json"),
      selection: values["out-selection"] ?? resolve(ROOT, "data/synthetic-hybrid-selection-local.json"),
      testReport: values["out-test-report"] ?? resolve(ROOT, "data/synthetic-hybrid-final-test-local.json"),
    });
    process.stdout.write(`Froze validation-selected model weight ${result.selectedModelWeight} at ` +
      `${result.selectionPath}; reserved final cohort was not scored.\n`);
  } else {
    if (!values.selection) fail("selection path");
    reportFrozenHybridTest(bundle, values.selection);
    process.stdout.write("Wrote one frozen invented new-user final report.\n");
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`Hybrid selection failed: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
