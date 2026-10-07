import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { acquireWikidataPilot, PILOT_IDS, PILOT_LIMITS, PILOT_USER_AGENT, pilotLookupQuery, verifyPilotApproval } from "../src/wikidata-feasibility.js";

const recorded = JSON.parse(readFileSync(new URL("../../docs/approvals/wikidata-feasibility.json", import.meta.url), "utf8"));
// Open only a mocked copy; the real record is consumed and cannot authorize another run.
const approval = { ...recorded, state: "approved" };
const response = (value: unknown) => new Response(JSON.stringify(value), { headers: { "content-type": "application/json" } });
const binding = (animeId: number, entityId: string) => ({ animeId: { type: "literal", value: String(animeId) },
  entity: { type: "uri", value: `http://www.wikidata.org/entity/${entityId}` } });
function fixture() {
  const source = JSON.parse(readFileSync(new URL("../../fixtures/synthetic-wikibase-entities.json", import.meta.url), "utf8"));
  Object.values(source.entities).forEach((entity: any, index) => { entity.claims.P4086[0].mainsnak.datavalue.value = String(index + 1); });
  return source;
}
function mock(override?: (url: URL, count: number) => Response | undefined) {
  let clock = Date.parse("2026-10-08T00:00:00Z");
  const calls: { url: URL; at: number; init?: RequestInit }[] = [], sleeps: number[] = [];
  const source = fixture();
  const ports = { now: () => clock, sleep: async (ms: number) => { sleeps.push(ms); clock += ms; },
    fetch: (async (input: string | URL | Request, init?: RequestInit) => {
      const url = new URL(String(input)); calls.push({ url, at: clock, init });
      const special = override?.(url, calls.length); if (special) return special;
      if (url.hostname === "query.wikidata.org") return response({ results: { bindings: url.searchParams.get("query")!.includes('"1"')
        ? Object.keys(source.entities).map((id, index) => binding(index + 1, id)) : [] } });
      const ids = url.searchParams.get("ids")!.split("|");
      return response({ entities: Object.fromEntries(ids.map((id) => [id, source.entities[id] ?? { id, type: "item", labels: { en: { language: "en", value: "Invented definition" } } }])) });
    }) as typeof fetch };
  return { ports, calls, sleeps, setClock: (value: number) => { clock = value; } };
}

test("exact approval is required before requests; expired and broadened uses remain held", async () => {
  const context = mock();
  for (const changed of [{ ...approval, approved: false }, { ...approval, publicArtifacts: true }, { ...approval, selection: "user-history" }, { ...approval, extra: true }]) {
    await assert.rejects(acquireWikidataPilot(changed, context.ports), /approval/);
  }
  verifyPilotApproval(approval, context.ports.now());
  assert.throws(() => verifyPilotApproval({ ...approval, state: "completed" }, context.ports.now()), /approval/);
  assert.throws(() => verifyPilotApproval(approval, Date.parse("2026-10-15T00:00:00Z")), /time window/);
  assert.throws(() => verifyPilotApproval(approval, Date.parse("2026-10-06T00:00:00Z")), /time window/);
  assert.equal(context.calls.length, 0);
  assert.throws(() => pilotLookupQuery([101]), /approved range/);
  assert.throws(() => pilotLookupQuery([1, 1]), /unique IDs/);
  assert.throws(() => (PILOT_IDS as number[]).push(101), /extensible|read only/);
});

test("invented acquisition is exact, sequential, byte-bound, projected, and has separate transport provenance", async () => {
  const context = mock(), result = await acquireWikidataPilot(approval, context.ports);
  assert.equal(result.receipt.animeEntities, 3);
  assert.equal(context.calls.filter((call) => call.url.hostname === "query.wikidata.org").length, 10);
  assert.ok(result.receipt.definitionEntities > 0 && result.receipt.definitionEntities <= 20);
  for (let index = 1; index < context.calls.length; index += 1) assert.ok(context.calls[index].at - context.calls[index - 1].at >= 2000);
  for (const call of context.calls) {
    assert.ok(["query.wikidata.org", "www.wikidata.org"].includes(call.url.hostname));
    assert.equal(call.init?.credentials, "omit"); assert.equal(call.init?.redirect, "error");
    assert.equal(call.init?.cache, "no-store"); assert.equal((call.init?.headers as any)["User-Agent"], PILOT_USER_AGENT);
    assert.ok(call.init?.signal);
  }
  const selected = JSON.parse(result.sourceBytes.toString());
  assert.equal(selected.entities.Q910000101.claims.P18, undefined);
  assert.equal(selected.entities.Q910000103.claims.P179, undefined);
  assert.equal(result.receipt.statementPresence.P4086, 3);
  assert.equal(result.receipt.atomicAcrossRequests, false);
  assert.equal(result.receipt.publicArtifacts, false);
  assert.ok(result.receipt.transportBodies.every((body) => body.sha256 !== result.receipt.projectionSha256));
  assert.ok(!JSON.stringify(result.receipt).includes("Copper"));
});

test("Retry-After is honored once with a total attempt count; excessive or repeated waits fail closed", async () => {
  const context = mock((_url, count) => count === 1 ? new Response(null, { status: 429, headers: { "retry-after": "10" } }) : undefined);
  const result = await acquireWikidataPilot(approval, context.ports);
  assert.ok(context.sleeps.includes(10000));
  assert.equal(context.calls[1].at - context.calls[0].at, 10000);
  assert.equal(result.receipt.attempts, context.calls.length);
  const repeated = mock(() => new Response(null, { status: 503 }));
  await assert.rejects(acquireWikidataPilot(approval, repeated.ports), /retry budget/);
  assert.equal(repeated.calls.length, 2);
  const excessive = mock(() => new Response(null, { status: 429, headers: { "retry-after": "120" } }));
  await assert.rejects(acquireWikidataPilot(approval, excessive.ports), /Retry-After bound/);
  assert.equal(excessive.calls.length, 1);
});

test("unrequested IDs, noncanonical entities, excessive rows/entities, and malformed API inventory stop acquisition", async () => {
  for (const rows of [[binding(101, "Q910000101")], [binding(1, "not-an-item")], Array.from({ length: 201 }, () => binding(1, "Q910000101"))]) {
    const context = mock(() => response({ results: { bindings: rows } }));
    await assert.rejects(acquireWikidataPilot(approval, context.ports), /lookup/);
    assert.equal(context.calls.length, 1);
  }
  const tooMany = mock((_url, count) => response({ results: { bindings: count === 1
    ? Array.from({ length: 100 }, (_, index) => binding(1, `Q${910000001 + index}`)) : [binding(11, "Q910000999")] } }));
  await assert.rejects(acquireWikidataPilot(approval, tooMany.ports), /entity bound/);
  for (const raw of [{ error: { info: "invented-secret-token" } }, { entities: {} }, { entities: { Q910000101: { id: "Q910000101", type: "item", missing: "" } } }]) {
    const context = mock((url) => url.hostname === "www.wikidata.org" ? response(raw) : undefined);
    await assert.rejects(acquireWikidataPilot(approval, context.ports), (error: any) => /entity response/.test(error.message) && !error.message.includes("secret-token"));
  }
});

test("body budgets, malformed JSON and duplicate keys are refused before retention", async () => {
  const cases = [new Response("x".repeat(PILOT_LIMITS.bodyBytes + 1)), new Response("not-json"), new Response('{"results":{},"results":{}}')];
  for (const value of cases) {
    const context = mock(() => value);
    await assert.rejects(acquireWikidataPilot(approval, context.ports), /byte budget|UTF-8 JSON|duplicate JSON/);
  }
  const context = mock(() => response({ results: { bindings: [] }, padding: "x".repeat(3500000) }));
  await assert.rejects(acquireWikidataPilot(approval, context.ports), /byte budget/);
  assert.equal(context.calls.length, 5);
});

test("cancellation and expiration after a delay prevent the next request; transport exceptions are redacted", async () => {
  const context = mock(), controller = new AbortController(); controller.abort();
  await assert.rejects(acquireWikidataPilot(approval, context.ports, controller.signal), /abort/i);
  assert.equal(context.calls.length, 0);
  context.setClock(Date.parse(approval.expiresAt) - 1000);
  await assert.rejects(acquireWikidataPilot(approval, context.ports), /time window/);
  assert.equal(context.calls.length, 1);
  const broken = mock(); broken.ports.fetch = async () => { throw new Error("invented-secret-url"); };
  await assert.rejects(acquireWikidataPilot(approval, broken.ports), (error: any) => /request failed/.test(error.message) && !error.message.includes("secret-url"));
});

test("retention projection strips unexpected nested fields and languages while preserving qualifier evidence", async () => {
  const context = mock((url) => {
    if (url.hostname !== "www.wikidata.org" || url.searchParams.get("props") === "info|labels") return undefined;
    const source = fixture();
    source.entities.Q910000101.labels.en.unexpected = "invented-secret";
    source.entities.Q910000101.labels.fr = { language: "fr", value: "invented-secret" };
    const statement = source.entities.Q910000101.claims.P31[0];
    statement.mainsnak.unexpected = "invented-secret";
    statement.mainsnak.datavalue.value.unexpected = "invented-secret";
    statement.references = ["invented-secret"];
    statement.qualifiers = { P518: [{ snaktype: "somevalue", property: "P518", unexpected: "invented-secret" }] };
    return response(source);
  });
  const result = await acquireWikidataPilot(approval, context.ports);
  assert.equal(result.sourceBytes.toString().includes("invented-secret"), false);
  assert.deepEqual(JSON.parse(result.sourceBytes.toString()).entities.Q910000101.claims.P31[0].qualifiers,
    { P518: [{ snaktype: "somevalue", property: "P518" }] });
});
