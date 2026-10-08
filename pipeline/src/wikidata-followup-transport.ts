/** Separate injected transport candidate. No default fetch, filesystem writer, CLI or approval registry. */
import { createHash } from "node:crypto";
import { parseWikibaseJson } from "./wikibase-metadata.js";
import { FOLLOWUP_IDS, FOLLOWUP_PROPERTIES, FOLLOWUP_LIMITS as limits, FOLLOWUP_SCOPE_SHA256,
  reserveFollowupStudy, verifyFollowupApproval, collectFollowupDefinitions, type FollowupGatePorts } from "./wikidata-followup-gates.js";

type RecordValue = Record<string, any>;
type Kind = "lookup" | "entities" | "definitions";
export interface FollowupTransportPorts extends FollowupGatePorts {
  fetch: typeof fetch;
  sleep(milliseconds: number, signal: AbortSignal): Promise<void>;
  /** Must schedule the callback at the whole-operation bound; return a cancellation function. */
  deadline(milliseconds: number, expire: () => void): () => void;
}
export const FOLLOWUP_USER_AGENT = "WhatAnimeShouldIWatch-Followup/0.1 (https://github.com/OptimumAF/WhatAnimeShouldIWatch)";
class TransportFailure extends Error {}
function fail(field: string, reason: string): never { throw new TransportFailure(`Wikidata follow-up ${field}: ${reason}.`); }
const encode = (value: unknown) => Buffer.from(`${JSON.stringify(value)}\n`);
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
function object(value: unknown, field: string): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "requires a bounded object");
  return value as RecordValue;
}
const ordered = (ids: Iterable<string>) => [...ids].sort((a, b) => BigInt(a.slice(1)) < BigInt(b.slice(1)) ? -1 : BigInt(a.slice(1)) > BigInt(b.slice(1)) ? 1 : 0);
export function followupLookupQuery(ids: readonly number[]): string {
  if (!ids.length || ids.length > limits.lookupBatch || new Set(ids).size !== ids.length || ids.some((id) => !FOLLOWUP_IDS.includes(id))) fail("query", "requires unique IDs in the exact follow-up range");
  return `SELECT DISTINCT ?animeId ?entity WHERE { VALUES ?animeId { ${ids.map((id) => `"${id}"`).join(" ")} }
?entity p:P4086 ?statement. ?statement ps:P4086 ?animeId; wikibase:rank ?rank.
FILTER(?rank != wikibase:DeprecatedRank) } LIMIT 201`;
}
function terms(value: unknown, aliases: boolean, languages: string[]): RecordValue {
  const raw = object(value ?? {}, "projection.terms"), result: RecordValue = {};
  for (const language of languages) {
    if (!Object.hasOwn(raw, language)) continue;
    const list = aliases ? raw[language] : [raw[language]];
    if (!Array.isArray(list) || list.length > 100) fail("projection.terms", "term bound exceeded");
    const selected = list.map((value: unknown) => {
      const term = object(value, "projection.term");
      if (term.language !== language || typeof term.value !== "string" || !term.value.length || term.value.length > 4000 || term.value.trim() !== term.value || /[\u0000-\u001f\u007f]/.test(term.value)) fail("projection.term", "malformed selected term");
      return { language, value: term.value };
    });
    result[language] = aliases ? selected : selected[0];
  }
  return result;
}
function snak(value: unknown): RecordValue {
  const raw = object(value, "projection.snak"), result: RecordValue = { snaktype: raw.snaktype, property: raw.property };
  if (Object.hasOwn(raw, "datatype")) result.datatype = raw.datatype;
  if (Object.hasOwn(raw, "datavalue")) {
    const data = object(raw.datavalue, "projection.datavalue"); let selected = data.value;
    if (selected && typeof selected === "object" && !Array.isArray(selected)) {
      const keys = data.type === "wikibase-entityid" ? ["entity-type", "numeric-id", "id"]
        : data.type === "quantity" ? ["amount", "unit", "lowerBound", "upperBound"]
        : data.type === "time" ? ["time", "timezone", "before", "after", "precision", "calendarmodel"] : [];
      selected = Object.fromEntries(keys.filter((key) => Object.hasOwn(data.value, key)).map((key) => [key, data.value[key]]));
    }
    result.datavalue = { type: data.type, value: selected };
  }
  return result;
}
function project(value: RecordValue, id: string) {
  const claims = object(value.claims, "projection.claims"), selected: RecordValue = {};
  for (const property of FOLLOWUP_PROPERTIES) {
    if (!Object.hasOwn(claims, property)) continue;
    if (!Array.isArray(claims[property]) || claims[property].length > 100) fail("projection.claims", "statement bound exceeded");
    selected[property] = claims[property].map((value: unknown) => {
      const statement = object(value, "projection.statement"), qualifiers: RecordValue = {};
      if (Object.hasOwn(statement, "qualifiers")) {
        const raw = object(statement.qualifiers, "projection.qualifiers");
        if (Object.keys(raw).length > 100) fail("projection.qualifiers", "property bound exceeded");
        for (const [key, values] of Object.entries(raw)) {
          if (!/^P[1-9]\d{0,38}$/.test(key) || !Array.isArray(values) || values.length > 100) fail("projection.qualifiers", "invalid or over-budget property");
          qualifiers[key] = values.map(snak);
        }
      }
      return { type: statement.type, rank: statement.rank, mainsnak: snak(statement.mainsnak),
        ...(Object.hasOwn(statement, "qualifiers") ? { qualifiers } : {}) };
    });
  }
  return { id, type: "item", labels: terms(value.labels, false, ["en", "ja"]), aliases: terms(value.aliases, true, ["en", "ja"]), claims: selected };
}

export async function runFollowupTransport(approval: unknown, ports: FollowupTransportPorts, signal?: AbortSignal) {
  if ([ports.fetch, ports.sleep, ports.deadline].some((port) => typeof port !== "function")) fail("preflight", "requires explicit transport and deadline ports");
  const reservation = await reserveFollowupStudy(approval, ports, signal);
  const check = () => {
    if (signal?.aborted) fail("transport", "cancelled");
    try { verifyFollowupApproval(approval, ports.now(), reservation.approvalSha256); }
    catch { fail("approval", "invalid, expired or changed approval"); }
    if (ports.now() >= Date.parse(reservation.expiresAt)) fail("transport", "retention window expired");
  };
  let attempts = 0, totalBytes = 0, lastStarted = -Infinity;
  const bodies: { kind: Kind; status: number; bytes: number; outcome: "success" | "http-retry" | "maxlag-retry"; sha256: string | null }[] = [];
  const wait = async (milliseconds: number) => {
    check(); const controller = new AbortController(); let closed = false, rejectCutoff!: (reason: Error) => void, cancelDeadline: (() => void) | undefined;
    const cutoff = new Promise<never>((_, reject) => { rejectCutoff = reject; });
    void cutoff.catch(() => undefined); // A faulty port may expire and throw before the race is installed.
    const abort = () => { if (!closed) { controller.abort(); rejectCutoff(new TransportFailure("Wikidata follow-up transport: cancelled.")); } };
    signal?.addEventListener("abort", abort, { once: true });
    try {
      cancelDeadline = ports.deadline(Date.parse(reservation.expiresAt) - ports.now(), () => {
        if (!closed) { controller.abort(); rejectCutoff(new TransportFailure("Wikidata follow-up approval: expired while waiting.")); }
      });
      if (typeof cancelDeadline !== "function") fail("transport", "invalid deadline port");
      const sleeping = (async () => { try { await ports.sleep(milliseconds, controller.signal); } catch { fail("transport", "wait failed"); } })();
      await Promise.race([sleeping, cutoff]); check();
    } catch (error) { if (error instanceof TransportFailure) throw error; fail("transport", "wait failed"); }
    finally { closed = true; controller.abort(); signal?.removeEventListener("abort", abort); try { cancelDeadline?.(); } catch { /* Fixed redacted boundary. */ } }
  };
  const attempt = async (url: URL, kind: Kind) => {
    const controller = new AbortController(); let reader: ReadableStreamDefaultReader<Uint8Array> | undefined,
      body: ReadableStream<Uint8Array> | null = null, closed = false;
    let rejectCutoff!: (reason: Error) => void;
    const cutoff = new Promise<never>((_, reject) => { rejectCutoff = reject; });
    void cutoff.catch(() => undefined);
    const cancelBody = () => { if (reader) { void reader.cancel().catch(() => undefined); try { reader.releaseLock(); } catch { /* Pending read remains handled by the operation. */ } }
      else if (body) { void body.cancel().catch(() => undefined); } };
    const stop = (reason: string) => { if (!closed) { controller.abort(); cancelBody(); rejectCutoff(new TransportFailure(`Wikidata follow-up transport: ${reason}.`)); } };
    const aborted = () => stop("cancelled"); signal?.addEventListener("abort", aborted, { once: true });
    let cancelDeadline: (() => void) | undefined;
    const started = ports.now();
    const live = () => {
      if (closed || controller.signal.aborted) fail("transport", "request cancelled or timed out");
      check(); if (ports.now() - started >= limits.timeoutMs) fail("transport", "whole-operation timeout");
    };
    try {
      const remaining = Date.parse(reservation.expiresAt) - started;
      cancelDeadline = ports.deadline(Math.min(limits.timeoutMs, remaining), () => stop(remaining <= limits.timeoutMs ? "approval expired during operation" : "whole-operation timeout"));
      if (typeof cancelDeadline !== "function") fail("transport", "invalid deadline port");
      const operation = (async () => {
        live(); let response: Response;
        try { response = await ports.fetch(url, { headers: { "User-Agent": FOLLOWUP_USER_AGENT, "Accept-Encoding": "gzip, deflate",
          Accept: kind === "lookup" ? "application/sparql-results+json" : "application/json" },
          credentials: "omit", redirect: "error", cache: "no-store", signal: controller.signal }); }
        catch { fail("transport", "request failed"); }
        body = response.body;
        if (closed || controller.signal.aborted) { void response.body?.cancel().catch(() => undefined); live(); }
        live();
        if (response.redirected || (response.url && response.url !== String(url))) { void response.body?.cancel().catch(() => undefined); fail("transport", "unexpected response location"); }
        if (!response.body) fail("transport", "missing response body");
        reader = response.body.getReader(); let size = 0; const chunks: Uint8Array[] = [];
        while (true) {
          live(); const next = await reader.read(); live(); if (next.done) break;
          size += next.value.byteLength; totalBytes += next.value.byteLength;
          if (size > limits.bodyBytes || totalBytes > limits.totalBytes) fail("transport", "byte budget exceeded");
          if (response.ok) chunks.push(next.value);
        }
        if (response.status === 429 || response.status === 503) {
          if (response.status === 503 && response.headers.has("x-squid-error") && !response.headers.has("retry-after")) fail("transport", "cache timeout is not retryable");
          bodies.push({ kind, status: response.status, bytes: size, outcome: "http-retry", sha256: null });
          return { retry: true, lag: false, retryAfter: response.headers.get("retry-after"), data: undefined };
        }
        if (!response.ok) fail("transport", "HTTP failure");
        const bytes = Buffer.concat(chunks); let data: unknown;
        try { data = parseWikibaseJson(bytes); } catch { fail("transport", "invalid bounded JSON"); }
        live(); const root = object(data, "response");
        if (Object.hasOwn(root, "error")) {
          if (kind !== "lookup" && object(root.error, "response.error").code === "maxlag") {
            bodies.push({ kind, status: response.status, bytes: size, outcome: "maxlag-retry", sha256: null });
            return { retry: true, lag: true, retryAfter: response.headers.get("retry-after"), data: undefined };
          }
          fail("transport", "API error");
        }
        bodies.push({ kind, status: response.status, bytes: size, outcome: "success", sha256: hash(bytes) });
        return { retry: false, lag: false, retryAfter: null, data };
      })();
      return await Promise.race([operation, cutoff]);
    } catch (error) { if (error instanceof TransportFailure) throw error; fail("transport", "operation failed"); }
    finally { closed = true; controller.abort(); signal?.removeEventListener("abort", aborted); try { cancelDeadline?.(); } catch { /* Do not reveal port exceptions. */ } cancelBody(); }
  };
  const request = async (url: URL, kind: Kind): Promise<unknown> => {
    for (let retry = 0; retry <= limits.retries; retry++) {
      check(); if (attempts >= limits.attempts) fail("transport", "attempt budget exhausted");
      await wait(Math.max(0, lastStarted + limits.minSpacingMs - ports.now()));
      lastStarted = ports.now(); attempts++;
      const result = await attempt(url, kind); check(); if (!result.retry) return result.data;
      if (retry === limits.retries) fail("transport", "retry budget exhausted");
      const header = result.retryAfter;
      if (header && header.length > 128) fail("transport", "Retry-After bound exceeded");
      const delay = header && /^\d+$/.test(header) ? Number(header) * 1000
        : header && Number.isFinite(Date.parse(header)) ? Math.max(0, Date.parse(header) - ports.now()) : 5000;
      if (!Number.isFinite(delay) || delay > limits.maxRetryMs) fail("transport", "Retry-After bound exceeded");
      // Action API maxlag asks noninteractive clients to pause at least five seconds.
      await wait(Math.max(result.lag ? 5000 : limits.minSpacingMs, delay));
    }
    fail("transport", "retry budget exhausted");
  };
  const lookup = new Map<string, Set<string>>(); let lookupRows = 0;
  for (let offset = 0; offset < FOLLOWUP_IDS.length; offset += limits.lookupBatch) {
    const ids = FOLLOWUP_IDS.slice(offset, offset + limits.lookupBatch), url = new URL("https://query.wikidata.org/sparql");
    url.searchParams.set("query", followupLookupQuery(ids)); url.searchParams.set("format", "json");
    const root = object(await request(url, "lookup"), "lookup"), rows = object(root.results, "lookup.results").bindings;
    if (!Array.isArray(rows) || rows.length > limits.lookupRows) fail("lookup", "row bound exceeded or malformed result");
    lookupRows += rows.length;
    for (const value of rows) {
      const row = object(value, "lookup.row"), external = object(row.animeId, "lookup.id"), entity = object(row.entity, "lookup.entity");
      if (external.type !== "literal" || !ids.includes(Number(external.value)) || String(Number(external.value)) !== external.value || entity.type !== "uri" ||
          typeof entity.value !== "string" || !/^http:\/\/www\.wikidata\.org\/entity\/Q[1-9]\d{0,38}$/.test(entity.value)) fail("lookup", "unrequested or noncanonical identity");
      const id = entity.value.slice(entity.value.lastIndexOf("/") + 1);
      if (!lookup.has(id)) lookup.set(id, new Set()); lookup.get(id)!.add(external.value);
      if (lookup.size > limits.animeEntities) fail("lookup", "anime entity budget exceeded");
    }
  }
  const readEntities = async (ids: string[], definitions: boolean) => {
    const url = new URL("https://www.wikidata.org/w/api.php");
    for (const [key, value] of Object.entries({ action: "wbgetentities", ids: ids.join("|"), props: definitions ? "info|labels" : "info|labels|aliases|claims",
      languages: definitions ? "en" : "en|ja", format: "json", maxlag: String(limits.maxlag) })) url.searchParams.set(key, value);
    const root = object(await request(url, definitions ? "definitions" : "entities"), "entities"), entities = object(root.entities, "entities.inventory");
    if (Object.keys(entities).length !== ids.length || Object.keys(entities).some((id) => !ids.includes(id))) fail("entities", "requested inventory mismatch");
    for (const id of ids) { const entity = object(entities[id], "entities.item");
      if (entity.id !== id || entity.type !== "item" || Object.hasOwn(entity, "missing")) fail("entities", "missing or mismatched item"); }
    return entities;
  };
  const entities: RecordValue = {}, revisions: Record<string, number | null> = {};
  const revision = (value: RecordValue) => Number.isSafeInteger(value.lastrevid) && value.lastrevid > 0 ? value.lastrevid : null;
  const animeIds = ordered(lookup.keys());
  for (let offset = 0; offset < animeIds.length; offset += limits.entityBatch) {
    const raw = await readEntities(animeIds.slice(offset, offset + limits.entityBatch), false);
    for (const [id, value] of Object.entries(raw)) { entities[id] = project(value, id); revisions[id] = revision(value); }
  }
  const sourceBytes = encode({ entities }), inventory = collectFollowupDefinitions(sourceBytes);
  for (const [id, expected] of lookup) {
    const live = new Set((entities[id].claims.P4086 ?? []).filter((s: RecordValue) => s.rank !== "deprecated" && s.mainsnak.snaktype === "value" && s.mainsnak.datatype === "external-id" && s.mainsnak.datavalue?.type === "string")
      .map((s: RecordValue) => s.mainsnak.datavalue.value));
    if ([...expected].some((value) => !live.has(value))) fail("identity", "lookup and retained live claims disagree");
  }
  const definitions: RecordValue = {};
  for (const id of inventory.reusedIds) definitions[id] = { id, labels: terms(entities[id].labels, false, ["en"]) };
  for (const batch of inventory.fetchBatches) {
    const raw = await readEntities(batch, true);
    for (const [id, value] of Object.entries(raw)) { definitions[id] = { id, labels: terms(value.labels, false, ["en"]) }; revisions[id] = revision(value); }
  }
  check(); const definitionBytes = encode({ entities: definitions });
  if (definitionBytes.length > limits.bodyBytes) fail("projection", "definition projection bound exceeded");
  return { sourceBytes, definitionBytes, inventory, reservation,
    receipt: { format: "private-followup-transport-receipt-v1", studyId: reservation.studyId, scopeSha256: FOLLOWUP_SCOPE_SHA256,
      approvalSha256: reservation.approvalSha256, startedAt: reservation.startedAt, expiresAt: reservation.expiresAt, snapshotAt: new Date(ports.now()).toISOString(),
      projectionSha256: hash(sourceBytes), definitionsSha256: hash(definitionBytes), attempts, totalBytes, lookupRows,
      animeEntities: animeIds.length, requiredDefinitionEntities: inventory.requiredIds.length, reusedDefinitionEntities: inventory.reusedIds.length,
      fetchedDefinitionEntities: inventory.fetchIds.length, missingDefinitionLabels: Object.values(definitions).filter((value) => !Object.hasOwn(value.labels, "en")).length,
      mappingReviewed: false, unreviewedDefinitionEntities: inventory.requiredIds.length,
      definitionOmissions: 0, transportBodies: bodies, revisions, atomicAcrossRequests: false, publicArtifacts: false } };
}
