/** Offline follow-up design gates. No filesystem implementation, transport, CLI or approval registry. */
import { createHash } from "node:crypto";
import path from "node:path";
import { parseWikibaseJson } from "./wikibase-metadata.js";

type RecordValue = Record<string, unknown>;
export const FOLLOWUP_STUDY_ID = "wikidata-followup-101-200-v1";
export const FOLLOWUP_IDS: readonly number[] = Object.freeze(Array.from({ length: 100 }, (_, index) => index + 101));
export const FOLLOWUP_PROPERTIES = Object.freeze(["P4086", "P31", "P136", "P577", "P580", "P1113", "P2047", "P155", "P156", "P2756"]);
export const FOLLOWUP_LIMITS = Object.freeze({ animeEntities: 100, definitions: 100, attempts: 40,
  bodyBytes: 4 * 1024 * 1024, totalBytes: 16 * 1024 * 1024, lookupBatch: 10, lookupRows: 200,
  entityBatch: 20, definitionBatch: 20, minSpacingMs: 2000, timeoutMs: 30000, maxRetryMs: 60000,
  retries: 1, maxlag: 5, retentionDays: 7 });
const properties = FOLLOWUP_PROPERTIES;
const roles = ["mainFormat", "mainGenre", "mainClassification", "mainUnit", "mainCalendar",
  "qualifierItem", "qualifierUnit", "qualifierCalendar"] as const;
type Role = typeof roles[number];
const hash = (value: Uint8Array | string) => createHash("sha256").update(value).digest("hex");
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as RecordValue;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonical(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
// Binding covers the reviewed scope, not source identity, actual permissions or a future adapter implementation.
export const FOLLOWUP_SCOPE_SHA256 = hash(canonical({ studyId: FOLLOWUP_STUDY_ID,
  ids: FOLLOWUP_IDS, properties, roles,
  endpoints: ["https://query.wikidata.org/sparql", "https://www.wikidata.org/w/api.php"],
  ...FOLLOWUP_LIMITS,
  animeLanguages: ["en", "ja"], definitionLanguages: ["en"], relationExpansion: false,
  publicArtifacts: false, training: false, deployment: false, productCache: false }));
function fail(field: string, reason = "unsupported, missing or mismatched evidence"): never {
  throw new Error(`Wikidata follow-up ${field}: ${reason}.`);
}
function object(value: unknown, field: string, required: string[], optional: string[] = []): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field);
  const result = value as RecordValue;
  if (required.some((key) => !Object.hasOwn(result, key)) || Object.keys(result).some((key) => ![...required, ...optional].includes(key))) fail(field);
  return result;
}
function boundedObject(value: unknown, field: string, maximum: number): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).length > maximum) fail(field);
  return value as RecordValue;
}
function text(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum && value.trim() === value && !/[\u0000-\u001f\u007f]/.test(value);
}
function approvedWindow(value: unknown, now: number): { expiresAt: number } {
  const approval = object(value, "approval", ["format", "studyId", "state", "approved", "owner", "authority",
    "decisionRef", "scopeSha256", "approvedAt", "expiresAt", "use", "publicArtifacts", "training", "deployment", "productCache"]);
  const exact = { format: "wikidata-followup-approval-v1", studyId: FOLLOWUP_STUDY_ID, state: "approved", approved: true,
    owner: "Avery", decisionRef: "docs/decisions/0046-bounded-followup-study-gates.md", scopeSha256: FOLLOWUP_SCOPE_SHA256,
    use: "one-local-feasibility-study-only", publicArtifacts: false, training: false, deployment: false, productCache: false };
  if (Object.entries(exact).some(([key, expected]) => approval[key] !== expected) || !text(approval.authority, 200)) fail("approval");
  const timestamp = (value: unknown): number => {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) fail("approval");
    const result = Date.parse(value);
    if (!Number.isFinite(result) || new Date(result).toISOString() !== value) fail("approval");
    return result;
  };
  const approvedAt = timestamp(approval.approvedAt), expiresAt = timestamp(approval.expiresAt);
  if (!Number.isFinite(now) || now < approvedAt || now >= expiresAt || expiresAt <= approvedAt || expiresAt - approvedAt > 7 * 86400000) fail("approval", "outside the exact approval window");
  return { expiresAt };
}
export interface FollowupReservation {
  format: "wikidata-followup-reservation-v1"; studyId: string; scopeSha256: string; approvalSha256: string;
  state: "started"; startedAt: string; expiresAt: string; publicArtifacts: false;
}
/** Pure recheck for the separate transport; a declared record/hash is not proof of owner permission. */
export function verifyFollowupApproval(value: unknown, now: number, expectedSha256?: string) {
  const window = approvedWindow(value, now), approvalSha256 = hash(canonical(value));
  if (expectedSha256 !== undefined && approvalSha256 !== expectedSha256) fail("approval", "changed during transport");
  return { ...window, approvalSha256 };
}
/** Facts must come from an independent OS probe. Mocked facts cannot verify actual ACLs or paths. */
export interface FollowupGatePorts {
  now(): number;
  inspect(): Promise<unknown>;
  /** Must exclusively create a durable record outside deletable output; false means already reserved. */
  reserveAtomic(record: FollowupReservation): Promise<boolean>;
}
function preflight(value: unknown): void {
  const facts = object(value, "preflight", ["repoRoot", "privateParent", "resolvedPrivateParent", "targetPath",
    "ancestorsHaveReparsePoints", "priorPilotExists", "targetExists", "reservationExists", "access"]);
  const location = (value: unknown): string => {
    if (typeof value !== "string" || !/^[A-Za-z]:\\/.test(value) || path.win32.normalize(value) !== value) fail("preflight");
    return value.toLowerCase();
  };
  const root = location(facts.repoRoot), parent = location(facts.privateParent), resolved = location(facts.resolvedPrivateParent);
  const target = location(facts.targetPath);
  if (parent !== path.win32.join(path.win32.dirname(root), "anime-private") || resolved !== parent ||
      target !== path.win32.join(parent, FOLLOWUP_STUDY_ID) ||
      facts.ancestorsHaveReparsePoints !== false || facts.priorPilotExists !== false || facts.targetExists !== false ||
      facts.reservationExists !== false || facts.access !== "verified-owner-only") fail("preflight");
}
export async function reserveFollowupStudy(approval: unknown, ports: FollowupGatePorts, signal?: AbortSignal): Promise<FollowupReservation> {
  let approvalSha256: string | undefined;
  const check = () => {
    if (signal?.aborted) fail("preflight", "cancelled");
    const now = ports.now(), window = approvedWindow(approval, now), currentHash = hash(canonical(approval));
    if (approvalSha256 !== undefined && currentHash !== approvalSha256) fail("approval", "changed during preflight");
    approvalSha256 = currentHash; return { ...window, now };
  };
  check();
  let facts: unknown;
  try { facts = await ports.inspect(); } catch { fail("preflight", "inspection failed"); }
  preflight(facts); const window = check(), startedAt = window.now;
  const record: FollowupReservation = { format: "wikidata-followup-reservation-v1", studyId: FOLLOWUP_STUDY_ID,
    scopeSha256: FOLLOWUP_SCOPE_SHA256, approvalSha256: approvalSha256!, state: "started", startedAt: new Date(startedAt).toISOString(),
    expiresAt: new Date(Math.min(window.expiresAt, startedAt + 7 * 86400000)).toISOString(), publicArtifacts: false };
  let reserved: boolean;
  try { reserved = await ports.reserveAtomic(record); } catch { fail("preflight", "reservation failed"); }
  if (reserved !== true) fail("preflight", "study already reserved");
  check(); // Expiration here still leaves the reservation consumed.
  return record;
}

/** Complete required definition inventory from local selected bytes; no definition fetch or mapper. */
export function collectFollowupDefinitions(sourceBytes: Uint8Array) {
  let parsed: unknown;
  try { parsed = parseWikibaseJson(sourceBytes); } catch { fail("inventory", "invalid bounded JSON"); }
  const root = object(parsed, "inventory", ["entities"]), entities = boundedObject(root.entities, "inventory", 100);
  const byRole = Object.fromEntries(roles.map((role) => [role, new Set<string>()])) as Record<Role, Set<string>>;
  const required = new Set<string>(), relations = new Set<string>();
  const itemId = (value: unknown): string => {
    if (typeof value !== "string" || !/^Q[1-9]\d{0,38}$/.test(value)) fail("inventory", "invalid entity reference");
    return value;
  };
  const uriId = (value: unknown, dimensionless = false): string | null => {
    if (dimensionless && value === "1") return null;
    if (typeof value !== "string" || !/^https?:\/\/www\.wikidata\.org\/entity\/Q[1-9]\d{0,38}$/.test(value)) fail("inventory", "invalid entity reference");
    return itemId(value.slice(value.lastIndexOf("/") + 1));
  };
  const add = (role: Role | null, id: string | null) => {
    if (!role || !id) return;
    byRole[role].add(id); required.add(id);
    if (required.size > 100) fail("inventory", "definition budget exceeded");
  };
  const snak = (value: unknown, property: string, role: Role | null, unitRole: Role | null, calendarRole: Role | null, relation: boolean) => {
    const raw = object(value, "inventory", ["snaktype", "property"], ["datatype", "datavalue", "hash"]);
    if (raw.property !== property || !["value", "somevalue", "novalue"].includes(raw.snaktype as string) ||
        (Object.hasOwn(raw, "datatype") && !text(raw.datatype, 80)) ||
        (Object.hasOwn(raw, "hash") && (typeof raw.hash !== "string" || !/^[a-f0-9]{40}$/.test(raw.hash)))) fail("inventory");
    if (raw.snaktype !== "value") { if (Object.hasOwn(raw, "datavalue")) fail("inventory"); return; }
    const data = object(raw.datavalue, "inventory", ["type", "value"]);
    if (data.type === "wikibase-entityid") {
      const value = object(data.value, "inventory", ["entity-type", "id"], ["numeric-id"]), id = itemId(value.id);
      if (raw.datatype !== "wikibase-item" || value["entity-type"] !== "item" ||
          (Object.hasOwn(value, "numeric-id") && (!Number.isSafeInteger(value["numeric-id"]) || String(value["numeric-id"]) !== id.slice(1)))) fail("inventory");
      add(role, id); if (relation) relations.add(id);
    } else if (data.type === "quantity") {
      const value = object(data.value, "inventory", ["amount", "unit"], ["lowerBound", "upperBound"]);
      if (raw.datatype !== "quantity" || !text(value.amount, 80) || ["lowerBound", "upperBound"].some((key) => Object.hasOwn(value, key) && value[key] !== null && !text(value[key], 80))) fail("inventory");
      add(unitRole, uriId(value.unit, true));
    } else if (data.type === "time") {
      const value = object(data.value, "inventory", ["time", "timezone", "before", "after", "precision", "calendarmodel"]);
      if (raw.datatype !== "time" || !text(value.time, 80) || ["timezone", "before", "after", "precision"].some((key) => !Number.isFinite(value[key]))) fail("inventory");
      add(calendarRole, uriId(value.calendarmodel));
    } else if (data.type !== "string" || !text(data.value, 4000) || !text(raw.datatype, 80) || ["wikibase-item", "time", "quantity"].includes(raw.datatype)) {
      fail("inventory", "unsupported structured snak");
    }
  };
  for (const [key, value] of Object.entries(entities)) {
    const entity = object(value, "inventory", ["id", "type", "labels", "aliases", "claims"]);
    if (itemId(key) !== entity.id || entity.type !== "item") fail("inventory");
    for (const component of ["labels", "aliases"]) {
      const terms = boundedObject(entity[component], "inventory", 2);
      for (const [language, values] of Object.entries(terms)) {
        if (!["en", "ja"].includes(language)) fail("inventory");
        const list = component === "aliases" ? values : [values];
        if (!Array.isArray(list) || list.length > 100) fail("inventory");
        for (const value of list) {
          const term = object(value, "inventory", ["language", "value"]);
          if (term.language !== language || !text(term.value, 4000)) fail("inventory");
        }
      }
    }
    const claims = boundedObject(entity.claims, "inventory", properties.length);
    for (const [property, statements] of Object.entries(claims)) {
      if (!properties.includes(property) || !Array.isArray(statements) || statements.length > 100) fail("inventory");
      for (const value of statements) {
        const statement = object(value, "inventory", ["type", "rank", "mainsnak"], ["qualifiers"]);
        if (statement.type !== "statement" || !["normal", "preferred", "deprecated"].includes(statement.rank as string)) fail("inventory");
        const live = statement.rank !== "deprecated";
        const mainRole: Role | null = live ? ({ P31: "mainFormat", P136: "mainGenre", P2756: "mainClassification" } as Record<string, Role>)[property] ?? null : null;
        snak(statement.mainsnak, property, mainRole, live && property === "P2047" ? "mainUnit" : null,
          live && ["P577", "P580"].includes(property) ? "mainCalendar" : null, live && ["P155", "P156"].includes(property));
        const qualifiers = boundedObject(statement.qualifiers ?? {}, "inventory", 100);
        for (const [qualifier, snaks] of Object.entries(qualifiers)) {
          if (!/^P[1-9]\d{0,38}$/.test(qualifier) || !Array.isArray(snaks) || snaks.length > 100) fail("inventory");
          for (const value of snaks) snak(value, qualifier, live ? "qualifierItem" : null, live ? "qualifierUnit" : null, live ? "qualifierCalendar" : null, false);
        }
      }
    }
  }
  const sort = (values: Iterable<string>) => [...values].sort((a, b) => {
    const left = BigInt(a.slice(1)), right = BigInt(b.slice(1)); return left < right ? -1 : left > right ? 1 : 0;
  });
  const requiredIds = sort(required), reusedIds = requiredIds.filter((id) => Object.hasOwn(entities, id));
  const fetchIds = requiredIds.filter((id) => !Object.hasOwn(entities, id));
  const inventory = { requiredIds, roleIds: Object.fromEntries(roles.map((role) => [role, sort(byRole[role])])) };
  return { format: "private-followup-definition-inventory-v1" as const, sourceSha256: hash(sourceBytes),
    inventorySha256: hash(canonical(inventory)), requiredIds, reusedIds, fetchIds,
    fetchBatches: Array.from({ length: Math.ceil(fetchIds.length / 20) }, (_, index) => fetchIds.slice(index * 20, index * 20 + 20)),
    byRole: Object.fromEntries(roles.map((role) => [role, byRole[role].size])) as Record<Role, number>,
    reusedMissingEnglishLabels: reusedIds.filter((id) => !Object.hasOwn((entities[id] as RecordValue).labels as RecordValue, "en")).length,
    outsideAcquiredRelationTargets: [...relations].filter((id) => !Object.hasOwn(entities, id)).length,
    publicationAuthorized: false as const };
}
