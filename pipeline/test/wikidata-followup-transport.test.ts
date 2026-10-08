import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runFollowupTransport, followupLookupQuery, FOLLOWUP_USER_AGENT } from "../src/wikidata-followup-transport.js";
import { FOLLOWUP_IDS, FOLLOWUP_SCOPE_SHA256, FOLLOWUP_STUDY_ID, FOLLOWUP_LIMITS } from "../src/wikidata-followup-gates.js";

const start = Date.parse("2026-10-08T00:00:00.000Z");
const approval = () => ({ format: "wikidata-followup-approval-v1", studyId: FOLLOWUP_STUDY_ID, state: "approved", approved: true,
  owner: "Avery", authority: "Invented mocked transport approval only", decisionRef: "docs/decisions/0046-bounded-followup-study-gates.md",
  scopeSha256: FOLLOWUP_SCOPE_SHA256, approvedAt: new Date(start).toISOString(), expiresAt: new Date(start + 3600000).toISOString(),
  use: "one-local-feasibility-study-only", publicArtifacts: false, training: false, deployment: false, productCache: false });
const response = (value: unknown, init?: ResponseInit) => new Response(JSON.stringify(value), init);
const statement = (property: string, value: string, item = false) => ({ type: "statement", rank: "normal",
  mainsnak: { snaktype: "value", property, datatype: item ? "wikibase-item" : "external-id",
    datavalue: { type: item ? "wikibase-entityid" : "string", value: item ? { "entity-type": "item", id: value } : value } } });
function fixture() {
  const source = JSON.parse(readFileSync(new URL("../../fixtures/synthetic-wikibase-entities.json", import.meta.url), "utf8"));
  Object.values(source.entities).forEach((entity: any, index) => { entity.claims.P4086[0].mainsnak.datavalue.value = String(index + 101); });
  return source.entities;
}
function minimal(definitions = 0, count = 1) {
  return Object.fromEntries(Array.from({ length: count }, (_, index) => {
    const id = `Q${910000101 + index}`;
    return [id, { id, type: "item", labels: { en: { language: "en", value: "Invented anime" } }, aliases: {}, lastrevid: index + 1,
      claims: { P4086: [statement("P4086", String(index + 101))], P136: Array.from({ length: Math.min(definitions, 100) }, (_, i) => statement("P136", `Q${920000001 + i}`, true)),
        ...(definitions > 100 ? { P31: [statement("P31", "Q920000101", true)] } : {}) } }];
  }));
}
function mock(source = fixture(), override?: (url: URL, count: number) => Response | Promise<Response> | undefined) {
  let clock = start, reserved = false;
  const calls: { url: URL; at: number; init?: RequestInit }[] = [], sleeps: number[] = [], writes: unknown[] = [];
  const timers: { milliseconds: number; active: boolean; expire: () => void }[] = [];
  const evidence: any = { repoRoot: "C:\\invented\\Anime", privateParent: "C:\\invented\\Anime-private", resolvedPrivateParent: "C:\\invented\\Anime-private",
    targetPath: `C:\\invented\\Anime-private\\${FOLLOWUP_STUDY_ID}`, ancestorsHaveReparsePoints: false,
    priorPilotExists: false, targetExists: false, reservationExists: false, access: "verified-owner-only" };
  let inspections = 0;
  const ports = { now: () => clock, inspect: async () => { inspections++; return { ...evidence, reservationExists: reserved }; },
    reserveAtomic: async (value: unknown) => { if (reserved) return false; reserved = true; writes.push(value); return true; },
    sleep: async (ms: number) => { sleeps.push(ms); clock += ms; },
    deadline: (milliseconds: number, expire: () => void) => { const timer = { milliseconds, active: true, expire }; timers.push(timer); return () => { timer.active = false; }; },
    fetch: (async (input: string | URL | Request, init?: RequestInit) => {
      const url = new URL(String(input)); calls.push({ url, at: clock, init }); const special = override?.(url, calls.length); if (special) return special;
      if (url.hostname === "query.wikidata.org") {
        const ids = [...url.searchParams.get("query")!.matchAll(/"(\d+)"/g)].map((match) => match[1]);
        return response({ results: { bindings: Object.entries(source).flatMap(([id, entity]: [string, any]) =>
          entity.claims.P4086.filter((s: any) => ids.includes(s.mainsnak.datavalue.value)).map((s: any) => ({
            animeId: { type: "literal", value: s.mainsnak.datavalue.value }, entity: { type: "uri", value: `http://www.wikidata.org/entity/${id}` } }))) } });
      }
      return response({ entities: Object.fromEntries(url.searchParams.get("ids")!.split("|").map((id) => [id, source[id] ?? {
        id, type: "item", lastrevid: 1, labels: { en: { language: "en", value: "Invented definition" } } }])) });
    }) as typeof fetch };
  return { ports, calls, sleeps, writes, evidence, timers, inspections: () => inspections,
    setClock: (value: number) => { clock = value; }, expire: () => { clock += FOLLOWUP_LIMITS.timeoutMs; const timer = timers.find((value) => value.active && value.milliseconds === FOLLOWUP_LIMITS.timeoutMs); assert.ok(timer); timer.expire(); } };
}

test("closed/wrong approvals and unresolved cleanup make zero transport calls; failed starts remain consumed", async () => {
  const old = JSON.parse(readFileSync(new URL("../../docs/approvals/wikidata-feasibility.json", import.meta.url), "utf8"));
  for (const changed of [old, null, { ...approval(), state: "completed" }, { ...approval(), scopeSha256: "0".repeat(64) }, { ...approval(), publicArtifacts: true }]) {
    const context = mock(); await assert.rejects(runFollowupTransport(changed, context.ports), /approval/);
    assert.equal(context.calls.length, 0); assert.equal(context.writes.length, 0); assert.equal(context.inspections(), 0);
  }
  const held = mock(); held.evidence.priorPilotExists = true;
  await assert.rejects(runFollowupTransport(approval(), held.ports), /preflight/); assert.equal(held.calls.length, 0); assert.equal(held.writes.length, 0);
  const failed = mock(minimal(), () => { throw new Error("Invented provider secret"); });
  await assert.rejects(runFollowupTransport(approval(), failed.ports), (error: any) => /request failed/.test(error.message) && !error.message.includes("secret"));
  await assert.rejects(runFollowupTransport(approval(), failed.ports), /preflight/); assert.equal(failed.calls.length, 1); assert.equal(failed.writes.length, 1);
  assert.throws(() => followupLookupQuery([100]), /range/); assert.throws(() => followupLookupQuery([101, 101]), /unique/);
  assert.throws(() => (FOLLOWUP_IDS as number[]).push(1));
});

test("invented exact projection uses identified serialized requests, complete roles and private byte bindings", async () => {
  const source = fixture(); source.Q910000101.claims.P580 = [{ type: "statement", rank: "normal", mainsnak: { snaktype: "value", property: "P580", datatype: "time", datavalue: { type: "time", value: {
    time: "+2020-00-00T00:00:00Z", timezone: 0, before: 0, after: 0, precision: 9, calendarmodel: "http://www.wikidata.org/entity/Q930000001", ignored: "not retained" } } } }];
  source.Q910000101.references = "not retained"; source.Q910000101.labels.fr = { language: "fr", value: "not retained" };
  const context = mock(source), result = await runFollowupTransport(approval(), context.ports);
  assert.equal(context.calls.filter((call) => call.url.hostname === "query.wikidata.org").length, 10);
  assert.equal(result.receipt.animeEntities, 3); assert.equal(result.receipt.definitionOmissions, 0);
  assert.equal(result.receipt.requiredDefinitionEntities, result.receipt.fetchedDefinitionEntities + result.receipt.reusedDefinitionEntities);
  assert.equal(result.receipt.atomicAcrossRequests, false); assert.equal(result.receipt.publicArtifacts, false); assert.equal(result.receipt.mappingReviewed, false);
  assert.equal(result.receipt.unreviewedDefinitionEntities, result.inventory.requiredIds.length);
  assert.equal(result.receipt.projectionSha256, createHash("sha256").update(result.sourceBytes).digest("hex"));
  assert.equal(result.inventory.sourceSha256, result.receipt.projectionSha256);
  assert.equal(result.receipt.definitionsSha256, createHash("sha256").update(result.definitionBytes).digest("hex"));
  assert.equal(result.receipt.totalBytes, result.receipt.transportBodies.reduce((sum, body) => sum + body.bytes, 0));
  const selected = JSON.parse(result.sourceBytes.toString());
  assert.equal(selected.entities.Q910000101.references, undefined); assert.equal(selected.entities.Q910000101.labels.fr, undefined);
  assert.equal(selected.entities.Q910000101.claims.P18, undefined); assert.equal(selected.entities.Q910000101.claims.P580[0].mainsnak.datavalue.value.ignored, undefined);
  assert.ok(result.inventory.byRole.mainCalendar > 0);
  for (let i = 1; i < context.calls.length; i++) assert.ok(context.calls[i].at - context.calls[i - 1].at >= 2000);
  for (const call of context.calls) {
    assert.equal(call.init?.credentials, "omit"); assert.equal(call.init?.redirect, "error"); assert.equal(call.init?.cache, "no-store");
    assert.equal((call.init?.headers as any)["User-Agent"], FOLLOWUP_USER_AGENT); assert.equal((call.init?.headers as any)["Accept-Encoding"], "gzip, deflate");
    if (call.url.hostname === "www.wikidata.org") assert.equal(call.url.searchParams.get("maxlag"), "5");
  }
  assert.ok(context.timers.every((timer) => !timer.active));
});

test("100 definitions are read completely in exact twenty-item batches; 101 fails before any definition read", async () => {
  const context = mock(minimal(100)), result = await runFollowupTransport(approval(), context.ports);
  assert.equal(result.receipt.requiredDefinitionEntities, 100); assert.equal(result.receipt.fetchedDefinitionEntities, 100);
  const calls = context.calls.filter((call) => call.url.searchParams.get("props") === "info|labels");
  assert.equal(calls.length, 5); assert.ok(calls.every((call) => call.url.searchParams.get("ids")!.split("|").length === 20));
  const exhausted = mock(minimal(101)); await assert.rejects(runFollowupTransport(approval(), exhausted.ports), /definition budget/);
  assert.equal(exhausted.calls.filter((call) => call.url.searchParams.get("props") === "info|labels").length, 0);
});

test("reused anime definitions are not fetched twice; missing labels remain unresolved without retry", async () => {
  const source = minimal(); source.Q910000101.claims.P31 = [statement("P31", "Q910000101", true)]; source.Q910000101.labels = {};
  const context = mock(source), result = await runFollowupTransport(approval(), context.ports);
  assert.equal(result.receipt.requiredDefinitionEntities, 1); assert.equal(result.receipt.reusedDefinitionEntities, 1);
  assert.equal(result.receipt.fetchedDefinitionEntities, 0); assert.equal(result.receipt.missingDefinitionLabels, 1);
});

test("429/503 and HTTP-200 maxlag consume bodies/attempts and honor bounded one-retry waits", async () => {
  for (const status of [429, 503]) {
    const context = mock(minimal(), (_url, count) => count === 1 ? new Response("invented error payload", { status, headers: { "retry-after": "3" } }) : undefined);
    const result = await runFollowupTransport(approval(), context.ports);
    assert.equal(result.receipt.transportBodies[0].bytes, Buffer.byteLength("invented error payload"));
    assert.equal(result.receipt.transportBodies[0].sha256, null); assert.equal(result.receipt.attempts, 12);
    assert.equal(context.calls[1].at - context.calls[0].at, 3000); assert.ok(!JSON.stringify(result.receipt).includes("payload"));
  }
  let lag = false;
  const context = mock(minimal(), (url) => { if (!lag && url.hostname === "www.wikidata.org") { lag = true; return response({ error: { code: "maxlag", info: "invented private error" } }, { headers: { "retry-after": "1" } }); } });
  const result = await runFollowupTransport(approval(), context.ports), index = result.receipt.transportBodies.findIndex((body) => body.outcome === "maxlag-retry");
  assert.ok(index >= 0); assert.equal(context.calls[index + 1].at - context.calls[index].at, 5000); assert.equal(result.receipt.transportBodies[index].sha256, null);
  for (const special of [() => new Response("invented", { status: 429 }), () => new Response("invented", { status: 503, headers: { "retry-after": "61" } })]) {
    const denied = mock(minimal(), special); await assert.rejects(runFollowupTransport(approval(), denied.ports), /retry budget|Retry-After/);
    assert.ok(denied.calls.length <= 2);
  }
});

test("error body and aggregate byte limits stop safely before another retry", async () => {
  const oversized = mock(minimal(), () => new Response(new Uint8Array(FOLLOWUP_LIMITS.bodyBytes + 1), { status: 503 }));
  await assert.rejects(runFollowupTransport(approval(), oversized.ports), /byte budget/); assert.equal(oversized.calls.length, 1);
  const aggregate = mock(minimal(), (_url, count) => count % 2 ? new Response(new Uint8Array(FOLLOWUP_LIMITS.bodyBytes), { status: 503 }) : undefined);
  await assert.rejects(runFollowupTransport(approval(), aggregate.ports), /byte budget/); assert.equal(aggregate.calls.length, 7);
});

test("whole-operation deadline ends an uncooperative fetch and cancels its late body", async () => {
  let resolve!: (response: Response) => void, cancelled = false;
  const context = mock(minimal(), () => new Promise<Response>((done) => { resolve = done; }));
  const pending = runFollowupTransport(approval(), context.ports); const rejection = assert.rejects(pending, /whole-operation timeout/);
  await new Promise(setImmediate); context.expire(); await rejection;
  resolve(new Response(new ReadableStream({ cancel() { cancelled = true; } }))); await new Promise(setImmediate);
  assert.equal(cancelled, true); assert.equal(context.calls.length, 1); assert.equal(context.calls[0].init?.signal?.aborted, true);
});

test("whole-body deadline and external abort do not wait for a stuck cancellation promise", async () => {
  for (const external of [false, true]) {
    let cancelled = false; const controller = new AbortController();
    const context = mock(minimal(), () => new Response(new ReadableStream({ start(stream) { stream.enqueue(new Uint8Array([123])); }, cancel() { cancelled = true; return new Promise(() => {}); } })));
    const pending = runFollowupTransport(approval(), context.ports, controller.signal), rejection = assert.rejects(pending, /timeout|cancelled/);
    await new Promise(setImmediate); if (external) controller.abort(); else context.expire(); await rejection;
    assert.equal(cancelled, true); assert.equal(context.calls.length, 1);
  }
});

test("expiry and approval mutation after waits or reads prevent later requests", async () => {
  const context = mock(minimal()); const sleep = context.ports.sleep;
  context.ports.sleep = async (ms) => { await sleep(ms); context.setClock(start + 3600000); };
  await assert.rejects(runFollowupTransport(approval(), context.ports), /approval/); assert.equal(context.calls.length, 0); assert.equal(context.writes.length, 1);
  const changed = approval(), mutation = mock(minimal(), () => { changed.authority = "changed invented authority"; return undefined; });
  await assert.rejects(runFollowupTransport(changed, mutation.ports), /operation failed|approval/); assert.equal(mutation.calls.length, 1);
});

test("unexpected identities/inventories, stale live claims and malformed JSON fail before definitions", async () => {
  const contexts = [
    mock(minimal(), (_url, count) => count === 1 ? response({ results: { bindings: Array.from({ length: 201 }, () => ({})) } }) : undefined),
    mock(minimal(), (_url, count) => count === 1 ? response({ results: { bindings: [{ animeId: { type: "literal", value: "1" }, entity: { type: "uri", value: "http://www.wikidata.org/entity/Q910000101" } }] } }) : undefined),
    mock(minimal(), (url) => url.hostname === "www.wikidata.org" ? response({ entities: {} }) : undefined),
    mock(minimal(), (url) => url.hostname === "www.wikidata.org" ? response({ entities: minimal(0, 2) }) : undefined),
    mock(minimal(), (_url, count) => count === 1 ? new Response('{"results":{},"results":{}}') : undefined),
    mock(minimal(), (url) => { if (url.hostname !== "www.wikidata.org") return; const source = minimal(); source.Q910000101.claims.P4086[0].mainsnak.datavalue.value = "102"; return response({ entities: source }); }),
  ];
  for (const context of contexts) {
    await assert.rejects(runFollowupTransport(approval(), context.ports), /Wikidata follow-up/);
    assert.equal(context.calls.filter((call) => call.url.searchParams.get("props") === "info|labels").length, 0);
  }
});

test("the full 100-anime/100-definition route allows exactly forty bounded attempts", async () => {
  const source = minimal(100, 100), context = mock(source, (_url, count) => count % 2 ? new Response("invented retry", { status: 503 }) : undefined);
  const result = await runFollowupTransport(approval(), context.ports);
  assert.equal(result.receipt.attempts, 40); assert.equal(result.receipt.animeEntities, 100); assert.equal(result.receipt.fetchedDefinitionEntities, 100);
  assert.equal(result.receipt.transportBodies.filter((body) => body.sha256 === null).length, 20);
});

test("abort and expiry settle stalled spacing/retry waits without a later request", async () => {
  for (const external of [false, true]) {
    const controller = new AbortController(), context = mock(minimal());
    context.ports.sleep = async () => new Promise(() => {});
    const pending = runFollowupTransport(approval(), context.ports, controller.signal), rejection = assert.rejects(pending, /cancelled|expired/);
    await new Promise(setImmediate);
    if (external) controller.abort(); else { const timer = context.timers.find((value) => value.active); assert.ok(timer); timer.expire(); }
    await rejection; assert.equal(context.calls.length, 0); assert.equal(context.writes.length, 1); assert.ok(context.timers.every((timer) => !timer.active));
  }
});

test("cache timeout, missing body, malformed UTF-8 and bad ports fail redacted and never retry", async () => {
  const candidates = [
    () => new Response("invented cache error", { status: 503, headers: { "x-squid-error": "invented timeout" } }),
    () => new Response(null), () => new Response(new Uint8Array([0xff])),
    () => new Response("invented server error", { status: 500 }),
  ];
  for (const special of candidates) {
    const context = mock(minimal(), special);
    await assert.rejects(runFollowupTransport(approval(), context.ports), (error: any) => /Wikidata follow-up/.test(error.message) && !error.message.includes("invented"));
    assert.equal(context.calls.length, 1);
  }
  const faulty = mock(minimal()); faulty.ports.deadline = (_ms, expire) => { expire(); throw new Error("invented private port error"); };
  await assert.rejects(runFollowupTransport(approval(), faulty.ports), (error: any) => /wait failed/.test(error.message) && !error.message.includes("private"));
  assert.equal(faulty.calls.length, 0);
});

test("HTTP dates/default waits and exact definition response inventory remain bounded", async () => {
  const context = mock(minimal(), (_url, count) => count === 1 ? new Response("invented", { status: 429,
    headers: { "retry-after": new Date(start + 10000).toUTCString() } }) : undefined);
  const result = await runFollowupTransport(approval(), context.ports); assert.equal(context.calls[1].at - context.calls[0].at, 10000);
  assert.equal(result.receipt.attempts, 12);
  const defaults = mock(minimal(), (_url, count) => count === 1 ? new Response("invented", { status: 503 }) : undefined);
  await runFollowupTransport(approval(), defaults.ports); assert.equal(defaults.calls[1].at - defaults.calls[0].at, 5000);
  const badDefinition = mock(minimal(1), (url) => url.searchParams.get("props") === "info|labels" ? response({ entities: {} }) : undefined);
  await assert.rejects(runFollowupTransport(approval(), badDefinition.ports), /requested inventory/);
  assert.equal(badDefinition.calls.filter((call) => call.url.searchParams.get("props") === "info|labels").length, 1);
});

test("approval expiry shorter than the request deadline ends a stalled body", async () => {
  const value = { ...approval(), expiresAt: new Date(start + 1000).toISOString() };
  let cancelled = false;
  const context = mock(minimal(), () => new Response(new ReadableStream({ cancel() { cancelled = true; } })));
  const pending = runFollowupTransport(value, context.ports), rejection = assert.rejects(pending, /approval expired/);
  await new Promise(setImmediate); const timer = context.timers.find((entry) => entry.active && entry.milliseconds === 1000);
  assert.ok(timer); context.setClock(start + 1000); timer.expire(); await rejection;
  assert.equal(cancelled, true); assert.equal(context.calls.length, 1); assert.equal(context.writes.length, 1);
});
