/** Read a prior committed plan approval before any private serving labels are read. */
import { execFileSync } from "node:child_process";
import path from "node:path";
import { isDeepStrictEqual } from "node:util";

const REVISION = /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/;
const REGISTRY_PATH = "docs/approvals/model-evaluation-plans.json";
const MODEL_PATH = "docs/approvals/model-release-bundles.json";

type ObjectValue = Record<string, unknown>;

function fail(field: string, reason: string): never {
  throw new Error(`Model evaluation approval ${field}: ${reason}`);
}

function object(value: unknown, field: string): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as ObjectValue;
}

function exact(value: unknown, keys: readonly string[], field: string): ObjectValue {
  const item = object(value, field);
  for (const key of keys) if (!Object.hasOwn(item, key)) fail(`${field}.${key}`, "is required");
  for (const key of Object.keys(item)) if (!keys.includes(key)) fail(`${field}.${key}`, "is unsupported");
  return item;
}

function git(repository: string, args: readonly string[]): string {
  try {
    return execFileSync("git", ["-C", repository, ...args], { encoding: "utf8",
      timeout: 10_000, maxBuffer: 1024 * 1024, stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch { fail("revision", "cannot read a verified prior repository commit"); }
}

function registryAt(repository: string, revision: string, filename: string): ObjectValue {
  let value: unknown;
  try { value = JSON.parse(git(repository, ["show", `${revision}:${filename}`])); }
  catch { fail("revision", `cannot parse ${filename} at the claimed commit`); }
  return object(value, `${filename}@${revision}`);
}

export interface PlanApprovalIdentity {
  sourceName: string;
  graphDatasetSha256: string;
  qualityPlanSha256: string;
  servingCohortSha256: string;
  servingFinalSha256: string;
  servingFreezeSha256: string;
  owner: string;
  approvalRef: string;
  decisionRef: string;
}

/** The plan entry must exist at a strict ancestor where this model tag is absent. */
export function verifyPriorPlanApproval(repository: string, freezeRevision: unknown,
  currentApprovals: unknown, currentModels: unknown, modelTag: string,
  expected: PlanApprovalIdentity): void {
  if (typeof freezeRevision !== "string" || !REVISION.test(freezeRevision)) {
    fail("freezeRevision", "must be a full Git commit ID");
  }
  const root = path.resolve(repository);
  const head = git(root, ["rev-parse", "HEAD"]);
  if (head === freezeRevision || git(root, ["cat-file", "-t", freezeRevision]) !== "commit") {
    fail("freezeRevision", "must identify a prior commit");
  }
  git(root, ["merge-base", "--is-ancestor", freezeRevision, head]);
  const current = registryAt(root, "HEAD", REGISTRY_PATH);
  if (!isDeepStrictEqual(current, currentApprovals)) {
    fail("planApprovals", "must be the committed current plan registry");
  }
  if (!isDeepStrictEqual(registryAt(root, "HEAD", MODEL_PATH), currentModels)) {
    fail("modelApprovals", "must be the committed current model registry");
  }
  const present = exact(current, ["schemaVersion", "freezes"], "planApprovals");
  if (present.schemaVersion !== 1 || !Array.isArray(present.freezes) ||
      present.freezes.filter((value) =>
        object(value, "planApprovals.freeze").servingFreezeSha256 === expected.servingFreezeSha256
      ).length !== 1) {
    fail("planApprovals", "must retain one exact approved freeze");
  }
  const prior = exact(registryAt(root, freezeRevision, REGISTRY_PATH),
    ["schemaVersion", "freezes"], "priorPlanApprovals");
  if (prior.schemaVersion !== 1 || !Array.isArray(prior.freezes)) {
    fail("priorPlanApprovals", "is unsupported");
  }
  const matches = prior.freezes.filter((value) =>
    object(value, "priorPlanApprovals.freezes").servingFreezeSha256 === expected.servingFreezeSha256);
  if (matches.length !== 1 || !isDeepStrictEqual(exact(matches[0],
    ["sourceName", "graphDatasetSha256", "qualityPlanSha256",
      "servingCohortSha256", "servingFinalSha256", "servingFreezeSha256",
      "owner", "approvalRef", "decisionRef"], "priorPlanApprovals.freeze"), expected)) {
    fail("priorPlanApprovals.freeze", "does not contain one exact prior owner precommitment");
  }
  if (!isDeepStrictEqual(present.freezes.find((value) =>
    object(value, "planApprovals.freeze").servingFreezeSha256 === expected.servingFreezeSha256),
  expected)) {
    fail("planApprovals.freeze", "changed after the prior commitment");
  }
  const priorModels = exact(registryAt(root, freezeRevision, MODEL_PATH),
    ["schemaVersion", "promotions"], "priorModelApprovals");
  if (priorModels.schemaVersion !== 1 || !Array.isArray(priorModels.promotions) ||
      priorModels.promotions.some((value) => object(value, "priorModelApprovals.promotion").tag === modelTag)) {
    fail("freezeRevision", "model promotion already existed at the claimed freeze commit");
  }
}
