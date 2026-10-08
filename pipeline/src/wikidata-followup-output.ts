/** Private output candidate only. No network, default writer, CLI or approval registry. */
import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import { parseWikibaseJson } from "./wikibase-metadata.js";
import { collectFollowupDefinitions, verifyFollowupApproval, FOLLOWUP_LIMITS as limits,
  FOLLOWUP_SCOPE_SHA256, FOLLOWUP_STUDY_ID, FOLLOWUP_IDS, type FollowupReservation } from "./wikidata-followup-gates.js";

type Value = Record<string, any>;
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const encode = (value: unknown) => Buffer.from(`${JSON.stringify(value)}\n`);
function fail(): never { throw new Error("Wikidata follow-up output: invalid, changed, expired or mismatched private evidence."); }
const equal = (a: unknown, b: unknown) => { if (!isDeepStrictEqual(a, b)) fail(); };
function object(value: unknown, keys?: string[]): Value {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail();
  if (keys) equal(Object.keys(value).sort(), [...keys].sort());
  return value as Value;
}
function timestamp(value: unknown): number {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) fail();
  const parsed = Date.parse(value); if (!Number.isFinite(parsed) || new Date(parsed).toISOString() !== value) fail(); return parsed;
}
export interface FollowupOutputPayload {
  format: "private-followup-output-payload-v1";
  reservation: FollowupReservation;
  files: { name: string; bytes: number; sha256: string; base64: string }[];
}
/** Recompute private bytes and cross-bindings before an OS writer receives them. Declared receipts are not source authentication. */
export function prepareFollowupOutput(approval: unknown, value: unknown, now: number): FollowupOutputPayload {
  try {
    const result = object(value, ["sourceBytes", "definitionBytes", "inventory", "reservation", "receipt"]);
    const record = object(result.reservation, ["format", "studyId", "scopeSha256", "approvalSha256", "state", "startedAt", "expiresAt", "publicArtifacts"]);
    const started = timestamp(record.startedAt), window = verifyFollowupApproval(approval, now, record.approvalSha256);
    verifyFollowupApproval(approval, started, record.approvalSha256);
    equal(record, { format: "wikidata-followup-reservation-v1", studyId: FOLLOWUP_STUDY_ID, scopeSha256: FOLLOWUP_SCOPE_SHA256,
      approvalSha256: window.approvalSha256, state: "started", startedAt: record.startedAt,
      expiresAt: new Date(Math.min(started + 7 * 86400000, window.expiresAt)).toISOString(), publicArtifacts: false });
    if (started > now || now >= timestamp(record.expiresAt)) fail();
    for (const key of ["sourceBytes", "definitionBytes"]) if (!(result[key] instanceof Uint8Array) || result[key].length > limits.bodyBytes) fail();
    const sourceBytes = Buffer.from(result.sourceBytes), definitionBytes = Buffer.from(result.definitionBytes);
    const inventory = collectFollowupDefinitions(sourceBytes); equal(result.inventory, inventory);
    const source = object(parseWikibaseJson(sourceBytes), ["entities"]).entities;
    for (const item of Object.values(source) as Value[]) {
      if (!(item.claims.P4086 ?? []).some((statement: Value) => statement.rank !== "deprecated" && statement.mainsnak.snaktype === "value" &&
        statement.mainsnak.datatype === "external-id" && statement.mainsnak.datavalue?.type === "string" &&
        FOLLOWUP_IDS.some((id) => statement.mainsnak.datavalue.value === String(id)))) fail();
    }
    const definitions = object(object(parseWikibaseJson(definitionBytes), ["entities"]).entities);
    equal(Object.keys(definitions).sort(), [...inventory.requiredIds].sort());
    let missing = 0;
    for (const id of inventory.requiredIds) {
      const item = object(definitions[id], ["id", "labels"]); if (item.id !== id) fail();
      const labels = object(item.labels); if (Object.keys(labels).some((key) => key !== "en")) fail();
      if (Object.hasOwn(labels, "en")) {
        const term = object(labels.en, ["language", "value"]);
        if (term.language !== "en" || typeof term.value !== "string" || !term.value.length || term.value.length > 4000 ||
          term.value.trim() !== term.value || /[\u0000-\u001f\u007f]/.test(term.value)) fail();
      } else missing++;
      if (inventory.reusedIds.includes(id)) equal(labels, Object.hasOwn(source[id].labels, "en") ? { en: source[id].labels.en } : {});
    }
    const receipt = object(result.receipt, ["format", "studyId", "scopeSha256", "approvalSha256", "startedAt", "expiresAt", "snapshotAt",
      "projectionSha256", "definitionsSha256", "attempts", "totalBytes", "lookupRows", "animeEntities", "requiredDefinitionEntities",
      "reusedDefinitionEntities", "fetchedDefinitionEntities", "missingDefinitionLabels", "mappingReviewed", "unreviewedDefinitionEntities",
      "definitionOmissions", "transportBodies", "revisions", "atomicAcrossRequests", "publicArtifacts"]);
    const fixed = { format: "private-followup-transport-receipt-v1", studyId: FOLLOWUP_STUDY_ID, scopeSha256: FOLLOWUP_SCOPE_SHA256,
      approvalSha256: record.approvalSha256, startedAt: record.startedAt, expiresAt: record.expiresAt,
      projectionSha256: hash(sourceBytes), definitionsSha256: hash(definitionBytes), animeEntities: Object.keys(source).length,
      requiredDefinitionEntities: inventory.requiredIds.length, reusedDefinitionEntities: inventory.reusedIds.length,
      fetchedDefinitionEntities: inventory.fetchIds.length, missingDefinitionLabels: missing, mappingReviewed: false,
      unreviewedDefinitionEntities: inventory.requiredIds.length, definitionOmissions: 0, atomicAcrossRequests: false, publicArtifacts: false };
    for (const [key, expected] of Object.entries(fixed)) equal(receipt[key], expected);
    const snapshot = timestamp(receipt.snapshotAt);
    if (snapshot < started || snapshot > now || snapshot >= timestamp(record.expiresAt) || !Number.isSafeInteger(receipt.lookupRows) ||
      receipt.lookupRows < Object.keys(source).length || receipt.lookupRows > 2000) fail();
    const bodies = receipt.transportBodies;
    if (!Array.isArray(bodies) || !bodies.length || bodies.length > limits.attempts) fail();
    const logical = [...Array(10).fill("lookup"), ...Array(Math.ceil(Object.keys(source).length / 20)).fill("entities"),
      ...Array(inventory.fetchBatches.length).fill("definitions")];
    let index = 0, retried = false, total = 0;
    for (const bodyValue of bodies) {
      const body = object(bodyValue, ["kind", "status", "bytes", "outcome", "sha256"]);
      if (body.kind !== logical[index] || !Number.isSafeInteger(body.bytes) || body.bytes < 0 || body.bytes > limits.bodyBytes ||
        !Number.isSafeInteger(body.status)) fail();
      total += body.bytes;
      if (body.outcome === "success") {
        if (body.status < 200 || body.status >= 300 || typeof body.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(body.sha256)) fail();
        index++; retried = false;
      } else {
        if (retried || body.sha256 !== null || !(body.outcome === "http-retry" && [429, 503].includes(body.status) ||
          body.outcome === "maxlag-retry" && body.kind !== "lookup" && body.status >= 200 && body.status < 300)) fail();
        retried = true;
      }
    }
    if (index !== logical.length || retried || total > limits.totalBytes) fail();
    equal(receipt.attempts, bodies.length); equal(receipt.totalBytes, total);
    const revisions = object(receipt.revisions);
    equal(Object.keys(revisions).sort(), [...new Set([...Object.keys(source), ...inventory.fetchIds])].sort());
    if (Object.values(revisions).some((revision) => revision !== null && (!Number.isSafeInteger(revision) || revision <= 0))) fail();
    const files = [{ name: "source-projection.json", data: sourceBytes }, { name: "definition-labels.json", data: definitionBytes },
      { name: "inventory.json", data: encode(inventory) }, { name: "receipt.json", data: encode(receipt) }];
    if (files.slice(2).some((file) => file.data.length > 256 * 1024)) fail();
    const completion = { format: "private-followup-completion-v1", studyId: FOLLOWUP_STUDY_ID, scopeSha256: FOLLOWUP_SCOPE_SHA256,
      approvalSha256: record.approvalSha256, startedAt: record.startedAt, expiresAt: record.expiresAt,
      completedAt: new Date(now).toISOString(), files: files.map((file) => ({ name: file.name, bytes: file.data.length, sha256: hash(file.data) })),
      state: "completed", mappingReviewed: false, publicArtifacts: false };
    files.push({ name: "completed.json", data: encode(completion) });
    return { format: "private-followup-output-payload-v1", reservation: structuredClone(record) as FollowupReservation,
      files: files.map((file) => ({ name: file.name, bytes: file.data.length, sha256: hash(file.data), base64: file.data.toString("base64") })) };
  } catch { fail(); }
}
export interface FollowupOutputPort { save(payload: FollowupOutputPayload): Promise<unknown> }
/** Explicit writer only; a failed or partial write stays consumed, with no cleanup or retry. */
export async function saveFollowupOutput(approval: unknown, result: unknown, now: () => number, port: FollowupOutputPort) {
  let payload: FollowupOutputPayload;
  try { payload = prepareFollowupOutput(approval, result, now()); } catch { fail(); }
  let saved: unknown;
  try { saved = await port.save(payload); } catch { fail(); }
  equal(saved, { saved: true });
  try {
    verifyFollowupApproval(approval, now(), payload.reservation.approvalSha256);
    if (now() >= timestamp(payload.reservation.expiresAt)) fail();
  } catch { fail(); }
  return { saved: true as const, publicArtifacts: false as const };
}
