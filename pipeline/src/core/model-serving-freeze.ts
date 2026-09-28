/** Private prefit commitment for M8.4 serving evaluation. Never exports labels. */
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const DIGEST = /^[a-f0-9]{64}$/;
// Match the package's private JSON limit so a valid freeze can reach packaging.
const MAX_JSON = 1024 * 1024;
export const STATIC_POLICY_FIELDS = ["decisionRef", "seed", "suppliedCount",
  "positiveRawScoreMin", "topK", "minimumEligibleUsers", "minimumPositiveLabels",
  "minimumServingCoverage", "minimumNdcgLift", "maximumP95LatencyMs",
  "latencyWarmups", "latencySamples"] as const;
export const SERVING_FREEZE_FILES = ["quality-plan.json", "serving-freeze.json",
  "serving-final.json", "serving-final.json.test-used"] as const;

type ObjectValue = Record<string, unknown>;

function fail(field: string, reason: string): never {
  throw new Error(`Model serving freeze ${field}: ${reason}`);
}

function exact(value: unknown, keys: readonly string[], field: string): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  const item = value as ObjectValue;
  for (const key of keys) if (!Object.hasOwn(item, key)) fail(`${field}.${key}`, "is required");
  for (const key of Object.keys(item)) if (!keys.includes(key)) fail(`${field}.${key}`, "is unsupported");
  return item;
}

function digest(value: unknown, field: string): string {
  if (typeof value !== "string" || !DIGEST.test(value)) fail(field, "must be lowercase SHA-256");
  return value as string;
}

function boundedFile(filepath: string, field: string): Buffer {
  const stat = fs.lstatSync(filepath);
  if (!stat.isFile() || stat.isSymbolicLink() || stat.size < 1 || stat.size > MAX_JSON) {
    fail(field, "must be a bounded regular file");
  }
  return fs.readFileSync(filepath);
}

function json(bytes: Buffer, field: string): ObjectValue {
  let value: unknown;
  try { value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
  catch { fail(field, "must be valid JSON and UTF-8"); }
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as ObjectValue;
}

export function servingSha256(bytes: Buffer | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function planFields(): string[] {
  return ["format", "sourceName", "rawContentSha256", "graphDatasetSha256",
    "cohortSha256", "finalSha256", "baselineCandidates", ...STATIC_POLICY_FIELDS];
}

function checkedPlan(value: unknown): ObjectValue {
  const plan = exact(value, planFields(), "quality-plan.json");
  if (plan.format !== "model-serving-quality-plan-v1") fail("quality-plan.json.format", "is unsupported");
  for (const key of ["rawContentSha256", "graphDatasetSha256", "cohortSha256",
    "finalSha256"] as const) digest(plan[key], `quality-plan.json.${key}`);
  if (typeof plan.sourceName !== "string" || !plan.sourceName.trim() ||
      plan.sourceName.length > 200) fail("quality-plan.json.sourceName", "is invalid");
  if (JSON.stringify(plan.baselineCandidates) !== JSON.stringify(["graph", "genre", "coverage"])) {
    fail("quality-plan.json.baselineCandidates", "must predeclare all three simple baselines in order");
  }
  for (const key of STATIC_POLICY_FIELDS) {
    if (key === "decisionRef") {
      if (typeof plan[key] !== "string" || !plan[key]) fail(`quality-plan.json.${key}`, "is invalid");
    } else if (typeof plan[key] !== "number" || !Number.isFinite(plan[key])) {
      fail(`quality-plan.json.${key}`, "must be finite");
    }
  }
  for (const key of ["seed", "suppliedCount", "positiveRawScoreMin", "topK",
    "minimumEligibleUsers", "minimumPositiveLabels", "latencyWarmups",
    "latencySamples"] as const) {
    if (!Number.isSafeInteger(plan[key])) fail(`quality-plan.json.${key}`, "must be a safe integer");
  }
  for (const key of ["suppliedCount", "positiveRawScoreMin", "topK",
    "minimumEligibleUsers", "minimumPositiveLabels", "latencySamples"] as const) {
    if ((plan[key] as number) < 1) fail(`quality-plan.json.${key}`, "must be positive");
  }
  if ((plan.latencyWarmups as number) < 0 || (plan.latencyWarmups as number) > 1_000 ||
      (plan.latencySamples as number) > 1_000 || (plan.topK as number) > 100 ||
      (plan.suppliedCount as number) > 100) {
    fail("quality-plan.json", "measurement or rank budget is too large");
  }
  for (const key of ["minimumServingCoverage", "minimumNdcgLift"] as const) {
    if ((plan[key] as number) <= 0 || (plan[key] as number) > 1) {
      fail(`quality-plan.json.${key}`, "must be in (0, 1]");
    }
  }
  if ((plan.maximumP95LatencyMs as number) <= 0 ||
      (plan.maximumP95LatencyMs as number) > 60_000) {
    fail("quality-plan.json.maximumP95LatencyMs", "must be in (0, 60000]");
  }
  return plan;
}

function privateDirectory(directory: string): string {
  const resolved = path.resolve(directory);
  const stat = fs.lstatSync(resolved);
  if (!stat.isDirectory() || stat.isSymbolicLink()) fail("evidenceDir", "must be a real directory");
  const repository = path.resolve(import.meta.dirname, "../../..");
  for (const forbidden of [path.join(repository, "web", "public"),
    path.join(repository, "release-data")]) {
    const relative = path.relative(forbidden, resolved);
    if (relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." &&
        !path.isAbsolute(relative))) fail("evidenceDir", "cannot be public or release assets");
  }
  return resolved;
}

/** Create the commitment before a selected model or serving report exists. */
export function freezeServingEvidence(evidenceDir: string, candidateDir: string,
  declaredPlan: unknown): { freezeSha256: string; planSha256: string } {
  const directory = privateDirectory(evidenceDir);
  for (const filename of ["selection.json", "model.npz", "serving-report.json"]) {
    if (fs.existsSync(path.join(directory, filename))) fail(filename, "already exists before the freeze");
  }
  if (fs.existsSync(path.join(candidateDir, "model-mf-web.compact.json"))) {
    fail("candidate.model", "already exists before the freeze");
  }
  const plan = checkedPlan(declaredPlan);
  const cohort = boundedFile(path.join(directory, "serving-cohort.json"), "serving-cohort.json");
  const final = boundedFile(path.join(directory, "serving-final.json"), "serving-final.json");
  if (plan.cohortSha256 !== servingSha256(cohort) ||
      plan.finalSha256 !== servingSha256(final)) {
    fail("quality-plan.json", "cohort or reserved final bytes differ from the declaration");
  }
  const cohortRecord = json(cohort, "serving-cohort.json");
  if (cohortRecord.format !== "model-promotion-cohort-v2" ||
      cohortRecord.finalSha256 !== plan.finalSha256 ||
      cohortRecord.datasetSha256 !== plan.graphDatasetSha256 ||
      cohortRecord.sourceName !== plan.sourceName || cohortRecord.seed !== plan.seed) {
    fail("serving-cohort.json", "source, dataset, seed, or reserved final digest differs");
  }
  const planBytes = Buffer.from(JSON.stringify(plan) + "\n");
  const planSha256 = servingSha256(planBytes);
  const freeze = { format: "model-serving-freeze-v1", planSha256,
    cohortSha256: plan.cohortSha256, finalSha256: plan.finalSha256 };
  const freezeBytes = Buffer.from(JSON.stringify(freeze) + "\n");
  fs.writeFileSync(path.join(directory, "quality-plan.json"), planBytes, { flag: "wx" });
  fs.writeFileSync(path.join(directory, "serving-freeze.json"), freezeBytes, { flag: "wx" });
  return { planSha256, freezeSha256: servingSha256(freezeBytes) };
}

/** Verify immutable byte links; optional marker check is used after final scoring. */
export function verifyServingFreeze(evidenceDir: string, expected: {
  sourceName: string; rawContentSha256: string; graphDatasetSha256: string;
  cohortSha256: string; finalSha256: string; freezeSha256: string;
}, policy: unknown, requireUsed = true): ObjectValue {
  const directory = privateDirectory(evidenceDir);
  const planBytes = boundedFile(path.join(directory, "quality-plan.json"), "quality-plan.json");
  const plan = checkedPlan(json(planBytes, "quality-plan.json"));
  const freezeBytes = boundedFile(path.join(directory, "serving-freeze.json"), "serving-freeze.json");
  const freeze = exact(json(freezeBytes, "serving-freeze.json"),
    ["format", "planSha256", "cohortSha256", "finalSha256"], "serving-freeze.json");
  if (freeze.format !== "model-serving-freeze-v1" ||
      freeze.planSha256 !== servingSha256(planBytes) ||
      freeze.cohortSha256 !== plan.cohortSha256 ||
      freeze.finalSha256 !== plan.finalSha256 ||
      expected.freezeSha256 !== servingSha256(freezeBytes)) {
    fail("serving-freeze.json", "does not match the original plan or reviewed digest");
  }
  for (const key of ["sourceName", "rawContentSha256", "graphDatasetSha256",
    "cohortSha256", "finalSha256"] as const) {
    if (plan[key] !== expected[key]) fail(`quality-plan.json.${key}`, "differs from the reviewed input");
  }
  const cohortBytes = boundedFile(path.join(directory, "serving-cohort.json"), "serving-cohort.json");
  const finalBytes = boundedFile(path.join(directory, "serving-final.json"), "serving-final.json");
  if (servingSha256(cohortBytes) !== plan.cohortSha256 ||
      servingSha256(finalBytes) !== plan.finalSha256) {
    fail("quality-plan.json", "cohort or reserved final bytes changed after the freeze");
  }
  const cohort = json(cohortBytes, "serving-cohort.json");
  if (cohort.format !== "model-promotion-cohort-v2" ||
      cohort.finalSha256 !== plan.finalSha256 ||
      cohort.datasetSha256 !== plan.graphDatasetSha256 ||
      cohort.sourceName !== plan.sourceName || cohort.seed !== plan.seed) {
    fail("serving-cohort.json", "differs from the frozen source, dataset, seed, or final digest");
  }
  const finalPolicy = policy as ObjectValue;
  for (const key of STATIC_POLICY_FIELDS) {
    if (finalPolicy?.[key] !== plan[key]) fail(`quality-policy.json.${key}`, "differs from the prefit plan");
  }
  if (requireUsed) {
    const marker = exact(json(boundedFile(path.join(directory,
      "serving-final.json.test-used"), "serving-final.json.test-used"),
    "serving-final.json.test-used"), ["format", "freezeSha256", "finalSha256"],
    "serving-final.json.test-used");
    if (marker.format !== "model-serving-final-used-v1" ||
        marker.freezeSha256 !== expected.freezeSha256 ||
        marker.finalSha256 !== expected.finalSha256) {
      fail("serving-final.json.test-used", "does not match the frozen final cohort");
    }
  }
  return plan;
}

/** Consume the final-label attempt before any caller parses reserved JSON. */
export function markServingFinalUsed(evidenceDir: string, freezeSha256: string,
  finalSha256: string): void {
  const directory = privateDirectory(evidenceDir);
  const marker = { format: "model-serving-final-used-v1", freezeSha256, finalSha256 };
  fs.writeFileSync(path.join(directory, "serving-final.json.test-used"),
    JSON.stringify(marker) + "\n", { flag: "wx" });
}
