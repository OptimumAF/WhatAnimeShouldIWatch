/** Private offline design audit. No catalog mutation, transport, retention, or publication entry point. */
import { createHash } from "node:crypto";
import { mapWikibaseMetadata, parseWikibaseJson, type WikibaseMappingPolicy } from "./wikibase-metadata.js";

type ObjectValue = Record<string, unknown>;
type Kind = "series" | "season";
type DateIssue = "missing" | "qualified" | "unknown" | "invalid" | "conflict";
type DateResult = { year: number; precision: number; issue?: undefined } | { year: null; precision: null; issue: DateIssue };
interface Statement { mainsnak: ObjectValue; rank: "normal" | "preferred" | "deprecated"; qualified: boolean; raw: ObjectValue }
const reasons = ["missingIdentity", "undeclaredScope", "scopeMismatch", "primaryDeprecatedOnly",
  "primaryQualified", "primaryUnknown", "primaryInvalid", "primaryConflict", "fallbackMissing",
  "fallbackDeprecatedOnly", "fallbackQualified", "fallbackUnknown", "fallbackInvalid", "fallbackConflict",
  "crossPropertyConflict"] as const;
type Reason = typeof reasons[number];
const hash = (value: Uint8Array | string) => createHash("sha256").update(value).digest("hex");
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as ObjectValue;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonical(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
function fail(field: string): never { throw new Error(`TV date candidate ${field}: unsupported, missing or mismatched evidence.`); }
function object(value: unknown, field: string, keys?: string[]): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field);
  const result = value as ObjectValue;
  if (keys && (Object.keys(result).length !== keys.length || keys.some((key) => !Object.hasOwn(result, key)))) fail(field);
  return result;
}
function json(bytes: Uint8Array, field: string): unknown {
  try { return parseWikibaseJson(bytes); } catch { fail(field); }
}
function statements(entity: ObjectValue, property: string): Statement[] {
  const claims = object(entity.claims, "source.claims");
  if (!Object.hasOwn(claims, property)) return [];
  const raw = claims[property];
  if (!Array.isArray(raw) || raw.length > 100) fail(`source.claims.${property}`);
  return raw.map((value) => {
    const statement = object(value, `source.claims.${property}.statement`);
    const mainsnak = object(statement.mainsnak, `source.claims.${property}.mainsnak`);
    if (statement.type !== "statement" || !["normal", "preferred", "deprecated"].includes(statement.rank as string) ||
        mainsnak.property !== property || !["value", "somevalue", "novalue"].includes(mainsnak.snaktype as string)) {
      fail(`source.claims.${property}.statement`);
    }
    const qualifiers = statement.qualifiers === undefined ? {} : object(statement.qualifiers, `source.claims.${property}.qualifiers`);
    return { mainsnak, rank: statement.rank as Statement["rank"], qualified: Object.keys(qualifiers).length > 0, raw: statement };
  });
}
function time(snak: ObjectValue): DateResult {
  const invalid: DateResult = { year: null, precision: null, issue: "invalid" };
  if (snak.datatype !== "time" || !snak.datavalue || typeof snak.datavalue !== "object" || Array.isArray(snak.datavalue)) return invalid;
  const data = snak.datavalue as ObjectValue;
  if (data.type !== "time" || Object.keys(data).length !== 2 || !data.value || typeof data.value !== "object" || Array.isArray(data.value)) return invalid;
  const raw = data.value as ObjectValue;
  const match = typeof raw.time === "string" && /^\+(\d{4})-(\d{2})-(\d{2})T00:00:00Z$/.exec(raw.time);
  if (Object.keys(raw).length !== 6 || !match || raw.calendarmodel !== "http://www.wikidata.org/entity/Q1985727" ||
      raw.timezone !== 0 || raw.before !== 0 || raw.after !== 0 || ![9, 10, 11].includes(raw.precision as number)) return invalid;
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  if (year < 1800 || year > 3000 ||
      (raw.precision === 9 && (month !== 0 || day !== 0)) ||
      (raw.precision === 10 && (month < 1 || month > 12 || day !== 0)) ||
      (raw.precision === 11 && (month < 1 || month > 12 || day < 1 || day > 31 ||
        new Date(Date.UTC(year, month - 1, day)).getUTCDate() !== day))) return invalid;
  return { year, precision: raw.precision as number };
}
function resolve(live: Statement[]): DateResult {
  if (!live.length) return { year: null, precision: null, issue: "missing" };
  if (live.some((entry) => entry.qualified)) return { year: null, precision: null, issue: "qualified" };
  if (live.some((entry) => entry.mainsnak.snaktype !== "value")) return { year: null, precision: null, issue: "unknown" };
  const decoded = live.map((entry) => time(entry.mainsnak));
  if (decoded.some((entry) => entry.issue)) return { year: null, precision: null, issue: "invalid" };
  const years = new Set(decoded.map((entry) => entry.year));
  return years.size === 1 ? { year: decoded[0].year!, precision: Math.min(...decoded.map((entry) => entry.precision!)) }
    : { year: null, precision: null, issue: "conflict" };
}

/** Source bytes and independent scope/policy hashes are checked, but declarations do not prove meaning or rights. */
export function auditTvDateCandidate(input: {
  sourceBytes: Uint8Array; universe: readonly number[]; mappingPolicy: WikibaseMappingPolicy;
  datePolicyBytes: Uint8Array; scopeBytes: Uint8Array; source: { name: string; snapshotAt: string };
}) {
  const mapped = mapWikibaseMetadata(input.sourceBytes, input.universe, input.mappingPolicy, input.source);
  const policy = object(json(input.datePolicyBytes, "policy"), "policy",
    ["format", "primaryProperty", "fallbackProperty", "dateScope", "typeScopes"]);
  if (policy.format !== "tv-first-airing-policy-v1" || policy.primaryProperty !== "P577" || policy.fallbackProperty !== "P580" ||
      policy.dateScope !== "whole-work-first-airing") fail("policy");
  const types = object(policy.typeScopes, "policy.typeScopes");
  if (!Object.keys(types).length || Object.keys(types).length > 100 || Object.entries(types).some(([id, kind]) =>
    !/^Q[1-9]\d{0,38}$/.test(id) || !["series", "season"].includes(kind as string) ||
    !Object.hasOwn(input.mappingPolicy.mediaFormats, id) || input.mappingPolicy.mediaFormats[id] !== "TV")) fail("policy.typeScopes");
  const scope = object(json(input.scopeBytes, "scope"), "scope", ["format", "sourceSnapshotSha256",
    "mappingPolicySha256", "datePolicySha256", "universeSha256", "items"]);
  if (scope.format !== "declared-tv-first-airing-scope-v1" || scope.sourceSnapshotSha256 !== hash(input.sourceBytes) ||
      scope.mappingPolicySha256 !== mapped.report.policySha256 || scope.datePolicySha256 !== hash(input.datePolicyBytes) ||
      scope.universeSha256 !== mapped.report.universeSha256) fail("scope.bindings");
  if (!Array.isArray(scope.items) || scope.items.length > 100) fail("scope.items");
  const declared = new Map<number, { sourceItemId: string; kind: Kind }>();
  const sourceItems = new Set<string>();
  let previousId = 0;
  for (const [index, value] of scope.items.entries()) {
    const entry = object(value, `scope.items[${index}]`, ["animeId", "sourceItemId", "kind", "dateScope"]);
    if (!Number.isSafeInteger(entry.animeId) || !input.universe.includes(entry.animeId as number) || Number(entry.animeId) <= previousId ||
        typeof entry.sourceItemId !== "string" || !/^Q[1-9]\d{0,38}$/.test(entry.sourceItemId) || sourceItems.has(entry.sourceItemId) ||
        !["series", "season"].includes(entry.kind as string) || entry.dateScope !== "whole-work-first-airing") fail(`scope.items[${index}]`);
    previousId = Number(entry.animeId); sourceItems.add(entry.sourceItemId);
    declared.set(previousId, { sourceItemId: entry.sourceItemId, kind: entry.kind as Kind });
  }
  const entities = object(object(json(input.sourceBytes, "source"), "source").entities, "source.entities");
  const byId = new Map((mapped.snapshot?.anime ?? []).map((item) => [item.animeId, item]));
  const refusals = Object.fromEntries(reasons.map((reason) => [reason, 0])) as Record<Reason, number>;
  const rows = [...input.universe].sort((a, b) => a - b).map((animeId) => {
    const item = byId.get(animeId), declaration = declared.get(animeId);
    const base = { animeId, sourceItemId: item?.sourceItemId ?? null, kind: declaration?.kind ?? null };
    const reject = (reason: Reason) => {
      refusals[reason] += 1;
      return { ...base, year: null, selectedProperty: null, minimumPrecision: null, selectedStatementsSha256: null, reason };
    };
    if (!item) return reject("missingIdentity");
    if (!declaration) return reject("undeclaredScope");
    if (item.sourceItemId !== declaration.sourceItemId) fail(`scope.items.animeId-${animeId}.sourceItemId`);
    const entity = object(entities[item.sourceItemId], "source.entity");
    const liveTypes = statements(entity, "P31").filter((entry) => entry.rank !== "deprecated");
    if (item.mediaFormat !== "TV" || !liveTypes.length || liveTypes.some((entry) => {
      const snak = entry.mainsnak, data = snak.datavalue as ObjectValue | undefined, value = data?.value as ObjectValue | undefined;
      return entry.qualified || snak.snaktype !== "value" || snak.datatype !== "wikibase-item" || data?.type !== "wikibase-entityid" ||
        value?.["entity-type"] !== "item" || typeof value.id !== "string" ||
        (value["numeric-id"] !== undefined && String(value["numeric-id"]) !== value.id.slice(1)) || types[value.id] !== declaration.kind;
    })) return reject("scopeMismatch");
    const primaryAll = statements(entity, "P577"), fallbackAll = statements(entity, "P580");
    const primary = primaryAll.filter((entry) => entry.rank !== "deprecated"), fallback = fallbackAll.filter((entry) => entry.rank !== "deprecated");
    if (primaryAll.length && !primary.length) return reject("primaryDeprecatedOnly");
    const p = resolve(primary), f = resolve(fallback);
    if (primaryAll.length && p.issue) return reject(`primary${p.issue[0].toUpperCase()}${p.issue.slice(1)}` as Reason);
    if (fallbackAll.length && !fallback.length) return reject("fallbackDeprecatedOnly");
    if (fallbackAll.length && f.issue) return reject(`fallback${f.issue[0].toUpperCase()}${f.issue.slice(1)}` as Reason);
    if (!primaryAll.length && f.issue) return reject("fallbackMissing");
    if (!p.issue && !f.issue && p.year !== f.year) return reject("crossPropertyConflict");
    const selectedProperty = primaryAll.length ? "P577" as const : "P580" as const;
    const result = primaryAll.length ? p : f;
    const selected = primaryAll.length ? primary : fallback;
    return { ...base, year: result.year, selectedProperty, minimumPrecision: result.precision,
      selectedStatementsSha256: hash(`[${selected.map((entry) => canonical(entry.raw)).sort().join(",")}]`), reason: null };
  });
  return { format: "private-tv-first-airing-audit-v1" as const, sourceSnapshotSha256: hash(input.sourceBytes),
    mappingPolicySha256: mapped.report.policySha256, datePolicySha256: hash(input.datePolicyBytes),
    scopeSha256: hash(input.scopeBytes), universeSha256: mapped.report.universeSha256,
    universeItems: rows.length, usableItems: rows.filter((row) => row.year !== null).length,
    primaryItems: rows.filter((row) => row.selectedProperty === "P577").length,
    fallbackItems: rows.filter((row) => row.selectedProperty === "P580").length, refusals, rows,
    publicationAuthorized: false as const };
}
