/** Exact, one-study acquisition adapter; no filesystem, provider proxy, or release entry point. */
import { createHash } from "node:crypto";
import { mapWikibaseMetadata, parseWikibaseJson, type WikibaseMappingPolicy } from "./wikibase-metadata.js";

const queryEndpoint = "https://query.wikidata.org/sparql";
const apiEndpoint = "https://www.wikidata.org/w/api.php";
export const PILOT_USER_AGENT = "WhatAnimeShouldIWatch-Feasibility/0.1 (https://github.com/OptimumAF/WhatAnimeShouldIWatch)";
export const PILOT_IDS: readonly number[] = Object.freeze(Array.from({ length: 100 }, (_, index) => index + 1));
export const PILOT_PROPERTIES = Object.freeze(["P4086", "P31", "P136", "P577", "P1113", "P2047", "P155", "P156", "P2756"]);
export const PILOT_LIMITS = Object.freeze({ attempts: 40, entities: 100, definitions: 20,
  bodyBytes: 4 * 1024 * 1024, totalBytes: 16 * 1024 * 1024,
  minSpacingMs: 2000, timeoutMs: 30000, maxRetryMs: 60000 } as const);
type RecordValue = Record<string, any>;
export interface PilotPorts {
  fetch: typeof fetch; now: () => number; sleep: (milliseconds: number) => Promise<void>;
}
const hash = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");
const encode = (value: unknown): Buffer => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
function fail(field: string, reason: string): never { throw new Error(`Wikidata pilot ${field}: ${reason}.`); }
function record(value: unknown, field: string): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "requires an object");
  return value as RecordValue;
}

export function verifyPilotApproval(value: unknown, now: number): void {
  const entry = record(value, "approval");
  if (entry.state !== "approved") fail("approval", "study is completed or not active");
  const expected = { schemaVersion: 1, studyId: "wikidata-pilot-2026-10-07", approved: true,
    state: "approved",
    owner: "Avery", authority: "Owner delegated approval/revision in this task on 2026-10-07",
    decisionRef: "docs/decisions/0043-catalog-source-candidate.md", approvedAt: "2026-10-07",
    expiresAt: "2026-10-14T23:59:59Z", selection: "public-numeric-anime-ids-1-through-100",
    use: "one-local-feasibility-study-only", maxAnimeEntities: 100, maxDefinitionEntities: 20,
    maxAttempts: 40, maxRetentionDays: 7, publicArtifacts: false, training: false, deployment: false };
  if (Object.keys(entry).length !== Object.keys(expected).length || Object.entries(expected).some(([key, expectedValue]) => entry[key] !== expectedValue)) {
    fail("approval", "does not match the exact delegated local study");
  }
  if (!Number.isFinite(now) || now < Date.parse("2026-10-07T00:00:00Z") || now >= Date.parse(expected.expiresAt)) {
    fail("approval", "is outside the approved time window");
  }
}

function selectedSnak(value: unknown): RecordValue {
  const snak = record(value, "entity.snak");
  const result: RecordValue = { snaktype: snak.snaktype, property: snak.property, datatype: snak.datatype };
  if (!snak.datavalue) return result;
  const raw = record(snak.datavalue, "entity.datavalue");
  let selected = raw.value;
  if (selected && typeof selected === "object" && !Array.isArray(selected)) {
    const keys = raw.type === "wikibase-entityid" ? ["entity-type", "numeric-id", "id"]
      : raw.type === "quantity" ? ["amount", "unit", "lowerBound", "upperBound"]
      : raw.type === "time" ? ["time", "timezone", "before", "after", "precision", "calendarmodel"] : [];
    selected = Object.fromEntries(keys.filter((key) => Object.hasOwn(raw.value, key)).map((key) => [key, raw.value[key]]));
  }
  result.datavalue = { type: raw.type, value: selected };
  return result;
}
function selectedTerms(value: unknown, aliases: boolean): RecordValue {
  const terms = record(value ?? {}, "entity.terms"), result: RecordValue = {};
  for (const language of ["en", "ja"]) {
    if (!Object.hasOwn(terms, language)) continue;
    const list = aliases ? terms[language] : [terms[language]];
    if (!Array.isArray(list) || list.length > 100) fail("entity.terms", "term bound exceeded");
    const selected = list.map((term: unknown) => { const raw = record(term, "entity.term"); return { language: raw.language, value: raw.value }; });
    result[language] = aliases ? selected : selected[0];
  }
  return result;
}

export function pilotLookupQuery(ids: readonly number[]): string {
  if (!ids.length || ids.length > 10 || new Set(ids).size !== ids.length || ids.some((id) => !PILOT_IDS.includes(id))) {
    fail("query", "requires up to ten unique IDs in the approved range");
  }
  return `SELECT DISTINCT ?animeId ?entity WHERE { VALUES ?animeId { ${ids.map((id) => `"${id}"`).join(" ")} }
    ?entity p:P4086 ?statement. ?statement ps:P4086 ?animeId; wikibase:rank ?rank.
    FILTER(?rank != wikibase:DeprecatedRank) } LIMIT 201`;
}

export async function acquireWikidataPilot(approval: unknown, ports: PilotPorts, signal?: AbortSignal) {
  verifyPilotApproval(approval, ports.now());
  let attempts = 0, totalBytes = 0, lastStarted = -Infinity;
  const receipts: { kind: "lookup" | "entities" | "definitions"; bytes: number; sha256: string }[] = [];
  const request = async (url: URL, kind: typeof receipts[number]["kind"]): Promise<unknown> => {
    for (let retry = 0; retry < 2; retry += 1) {
      verifyPilotApproval(approval, ports.now());
      signal?.throwIfAborted();
      if (attempts >= PILOT_LIMITS.attempts) fail("transport", "attempt budget exhausted");
      await ports.sleep(Math.max(0, lastStarted + PILOT_LIMITS.minSpacingMs - ports.now()));
      verifyPilotApproval(approval, ports.now());
      signal?.throwIfAborted();
      lastStarted = ports.now(); attempts += 1;
      let response: Response;
      try {
        response = await ports.fetch(url, { headers: { "User-Agent": PILOT_USER_AGENT,
          Accept: kind === "lookup" ? "application/sparql-results+json" : "application/json" },
          credentials: "omit", redirect: "error", cache: "no-store",
          signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(PILOT_LIMITS.timeoutMs)]) : AbortSignal.timeout(PILOT_LIMITS.timeoutMs) });
      } catch { signal?.throwIfAborted(); fail("transport", "request failed or timed out"); }
      if (response.status === 429 || response.status === 503) {
        await response.body?.cancel();
        const retryAfter = response.headers.get("retry-after");
        const seconds = retryAfter && /^\d+$/.test(retryAfter) ? Number(retryAfter) : NaN;
        const date = retryAfter ? Date.parse(retryAfter) : NaN;
        const delay = Number.isFinite(seconds) ? seconds * 1000 : Number.isFinite(date) ? Math.max(0, date - ports.now()) : 5000;
        if (retry === 1 || delay > PILOT_LIMITS.maxRetryMs) fail("transport", "retry budget or Retry-After bound exceeded");
        await ports.sleep(Math.max(PILOT_LIMITS.minSpacingMs, delay));
        continue;
      }
      if (!response.ok) { await response.body?.cancel(); fail("transport", `HTTP ${response.status}`); }
      if (!response.body) fail("transport", "has no response body");
      const reader = response.body.getReader(), chunks: Uint8Array[] = [];
      let size = 0;
      try {
        while (true) {
          signal?.throwIfAborted();
          const { value, done } = await reader.read();
          if (done) break;
          size += value.byteLength; totalBytes += value.byteLength;
          if (size > PILOT_LIMITS.bodyBytes || totalBytes > PILOT_LIMITS.totalBytes) fail("transport", "byte budget exceeded");
          chunks.push(value);
        }
      } finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
      const bytes = Buffer.concat(chunks);
      const data = parseWikibaseJson(bytes);
      receipts.push({ kind, bytes: size, sha256: hash(bytes) });
      return data;
    }
    fail("transport", "retry budget exhausted");
  };
  const entityIds = new Set<string>();
  let lookupRows = 0;
  for (let offset = 0; offset < PILOT_IDS.length; offset += 10) {
    const ids = PILOT_IDS.slice(offset, offset + 10);
    const url = new URL(queryEndpoint);
    url.searchParams.set("query", pilotLookupQuery(ids)); url.searchParams.set("format", "json");
    const root = record(await request(url, "lookup"), "lookup");
    const rows = record(root.results, "lookup.results").bindings;
    if (!Array.isArray(rows) || rows.length > 200) fail("lookup", "row bound exceeded or malformed result");
    lookupRows += rows.length;
    for (const raw of rows) {
      const row = record(raw, "lookup.row"), id = record(row.animeId, "lookup.animeId"), entity = record(row.entity, "lookup.entity");
      if (id.type !== "literal" || !ids.includes(Number(id.value)) || String(Number(id.value)) !== id.value || entity.type !== "uri" ||
          typeof entity.value !== "string" || !/^http:\/\/www\.wikidata\.org\/entity\/Q[1-9]\d{0,38}$/.test(entity.value)) {
        fail("lookup.row", "contains an unrequested ID or noncanonical entity");
      }
      entityIds.add(entity.value.split("/").at(-1)!);
      if (entityIds.size > PILOT_LIMITS.entities) fail("lookup", "entity bound exceeded");
    }
  }
  const entityRead = async (ids: string[], definitions: boolean): Promise<RecordValue> => {
    const url = new URL(apiEndpoint);
    url.searchParams.set("action", "wbgetentities"); url.searchParams.set("ids", ids.join("|"));
    url.searchParams.set("props", definitions ? "info|labels" : "info|labels|aliases|claims");
    url.searchParams.set("languages", definitions ? "en" : "en|ja");
    url.searchParams.set("format", "json");
    const root = record(await request(url, definitions ? "definitions" : "entities"), "entity response");
    if (root.error || !root.entities) fail("entity response", "API error or missing entities");
    const result = record(root.entities, "entity response.entities");
    if (Object.keys(result).length !== ids.length || Object.keys(result).some((id) => !ids.includes(id))) fail("entity response", "does not match the requested inventory");
    for (const id of ids) {
      const entity = record(result[id], "entity response.item");
      if (entity.id !== id || entity.type !== "item" || Object.hasOwn(entity, "missing")) fail("entity response.item", "is missing or mismatched");
    }
    return result;
  };
  const entities: RecordValue = {};
  const revisions: Record<string, number | null> = {};
  const orderedIds = [...entityIds].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  for (let offset = 0; offset < orderedIds.length; offset += 20) {
    const raw = await entityRead(orderedIds.slice(offset, offset + 20), false);
    for (const [id, value] of Object.entries(raw)) {
      const claims = record(value.claims, "entity.claims"), selected: RecordValue = {};
      for (const property of PILOT_PROPERTIES) {
        if (!Object.hasOwn(claims, property)) continue;
        if (!Array.isArray(claims[property]) || claims[property].length > 100) fail("entity.claims", "statement bound exceeded");
        selected[property] = claims[property].map((statement: unknown) => {
          const s = record(statement, "entity.statement");
          const qualifiers: RecordValue = {};
          if (s.qualifiers) {
            const raw = record(s.qualifiers, "entity.qualifiers");
            if (Object.keys(raw).length > 100) fail("entity.qualifiers", "property bound exceeded");
            for (const [key, snaks] of Object.entries(raw)) {
              if (!/^P[1-9]\d{0,38}$/.test(key) || !Array.isArray(snaks) || snaks.length > 100) fail("entity.qualifiers", "invalid or over-budget property");
              qualifiers[key] = snaks.map(selectedSnak);
            }
          }
          return { type: s.type, rank: s.rank, mainsnak: selectedSnak(s.mainsnak), ...(s.qualifiers ? { qualifiers } : {}) };
        });
      }
      entities[id] = { id, type: "item", labels: selectedTerms(value.labels, false), aliases: selectedTerms(value.aliases, true), claims: selected };
      revisions[id] = Number.isSafeInteger(value.lastrevid) && value.lastrevid > 0 ? value.lastrevid : null;
    }
  }
  const sourceBytes = encode({ entities });
  // Validate the retained projection before any file can be written or definition fetched.
  const emptyPolicy: WikibaseMappingPolicy = { format: "wikibase-metadata-policy-v1", languageOrder: ["en", "ja"],
    genreLabels: {}, mediaFormats: {}, durationUnits: {}, classification: null };
  const source = { name: "Wikidata selected-statement local feasibility projection", snapshotAt: new Date(ports.now()).toISOString() };
  mapWikibaseMetadata(sourceBytes, PILOT_IDS, emptyPolicy, source);
  const definitionIds = new Set<string>();
  const referencedByProperty: Record<string, number> = {};
  // Predeclared priority: units, formats, genres, classification; numeric IDs break ties.
  for (const property of ["P2047", "P31", "P136", "P2756"]) {
    const referenced = new Set<string>();
    for (const entity of Object.values(entities)) for (const statement of entity.claims[property] ?? []) {
      if (statement.rank === "deprecated") continue;
      const value = statement.mainsnak?.datavalue?.value;
      const id = property === "P2047" && typeof value?.unit === "string" ? value.unit.split("/").at(-1) : value?.id;
      if (typeof id === "string" && /^Q[1-9]\d{0,38}$/.test(id)) referenced.add(id);
    }
    referencedByProperty[property] = referenced.size;
    for (const id of [...referenced].sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)))) definitionIds.add(id);
  }
  const selectedDefinitions = [...definitionIds].slice(0, PILOT_LIMITS.definitions);
  const definitions: RecordValue = {};
  if (selectedDefinitions.length) {
    const raw = await entityRead(selectedDefinitions, true);
    for (const [id, value] of Object.entries(raw)) {
      const labels = selectedTerms(value.labels, false); delete labels.ja;
      definitions[id] = { id, labels };
      revisions[id] = Number.isSafeInteger(value.lastrevid) && value.lastrevid > 0 ? value.lastrevid : null;
    }
  }
  const statementPresence = Object.fromEntries(PILOT_PROPERTIES.map((property) => [property,
    Object.values(entities).filter((entity) => (entity.claims[property] ?? []).some((s: RecordValue) => s.rank !== "deprecated")).length]));
  return { sourceBytes, definitionBytes: encode({ entities: definitions }), source,
    receipt: { format: "wikidata-local-feasibility-receipt-v1", studyId: "wikidata-pilot-2026-10-07",
      snapshotAt: source.snapshotAt, projectionSha256: hash(sourceBytes), attempts, totalBytes,
      lookupRows, animeEntities: orderedIds.length, definitionEntities: selectedDefinitions.length,
      omittedDefinitionEntities: definitionIds.size - selectedDefinitions.length, referencedByProperty,
      statementPresence, transportBodies: receipts, revisions, atomicAcrossRequests: false, publicArtifacts: false } };
}
