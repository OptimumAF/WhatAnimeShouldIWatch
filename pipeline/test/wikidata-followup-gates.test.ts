import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  FOLLOWUP_SCOPE_SHA256, FOLLOWUP_STUDY_ID, collectFollowupDefinitions, reserveFollowupStudy,
  type FollowupGatePorts,
} from "../src/wikidata-followup-gates.js";

const start = Date.parse("2026-10-08T00:00:00.000Z");
const approval = () => ({ format: "wikidata-followup-approval-v1", studyId: FOLLOWUP_STUDY_ID,
  state: "approved", approved: true, owner: "Avery", authority: "Invented owner approval for mocked test only",
  decisionRef: "docs/decisions/0046-bounded-followup-study-gates.md", scopeSha256: FOLLOWUP_SCOPE_SHA256,
  approvedAt: new Date(start).toISOString(), expiresAt: new Date(start + 7 * 86400000).toISOString(),
  use: "one-local-feasibility-study-only", publicArtifacts: false, training: false, deployment: false, productCache: false });
function mock() {
  let clock = start;
  const root = "C:\\invented\\Anime", parent = "C:\\invented\\Anime-private";
  const evidence = { repoRoot: root, privateParent: parent, resolvedPrivateParent: parent,
    targetPath: `${parent}\\${FOLLOWUP_STUDY_ID}`, ancestorsHaveReparsePoints: false,
    priorPilotExists: false, targetExists: false, reservationExists: false, access: "verified-owner-only" };
  const inspections: number[] = [], writes: unknown[] = [];
  const ports: FollowupGatePorts = { now: () => clock, inspect: async () => { inspections.push(clock); return { ...evidence }; },
    reserveAtomic: async (record) => { if (evidence.reservationExists) return false;
      evidence.reservationExists = true; writes.push(record); return true; } };
  return { ports, evidence, inspections, writes, setClock: (value: number) => { clock = value; } };
}
const bytes = (value: unknown) => Buffer.from(JSON.stringify(value));
const item = (property: string, id: string) => ({ snaktype: "value", property, datatype: "wikibase-item",
  datavalue: { type: "wikibase-entityid", value: { "entity-type": "item", id } } });
const statement = (snak: any, rank = "normal", qualifiers?: any) => ({ type: "statement", rank, mainsnak: snak,
  ...(qualifiers === undefined ? {} : { qualifiers }) });
const entity = (claims: any, id = "Q910000101") => ({ id, type: "item", labels: {}, aliases: {}, claims });
const projection = (claims: any) => ({ entities: { Q910000101: entity(claims) } });
const quantity = (property: string, unit: string) => ({ snaktype: "value", property, datatype: "quantity",
  datavalue: { type: "quantity", value: { amount: "+24", unit, lowerBound: null, upperBound: null } } });
const time = (property: string, calendar: string) => ({ snaktype: "value", property, datatype: "time",
  datavalue: { type: "time", value: { time: "+2024-00-00T00:00:00Z", timezone: 0, before: 0, after: 0,
    precision: 9, calendarmodel: calendar } } });

test("absent, consumed, wrong-scope or broadened approvals fail before inspection and reservation", async () => {
  const context = mock();
  const old = JSON.parse(readFileSync(new URL("../../docs/approvals/wikidata-feasibility.json", import.meta.url), "utf8"));
  for (const value of [null, old, { ...approval(), approved: false }, { ...approval(), state: "completed" },
    { ...approval(), state: "started" }, { ...approval(), state: "failed" }, { ...approval(), state: "interrupted" },
    { ...approval(), scopeSha256: "0".repeat(64) }, { ...approval(), studyId: "other" },
    ...["publicArtifacts", "training", "deployment", "productCache"].map((key) => ({ ...approval(), [key]: true })),
    { ...approval(), unexpected: true }, { ...approval(), authority: "" }]) {
    await assert.rejects(reserveFollowupStudy(value, context.ports), /approval/);
  }
  assert.equal(context.inspections.length, 0); assert.equal(context.writes.length, 0);
});

test("approval windows and cleanup/path/access holds refuse reservation", async () => {
  for (const now of [start - 1, start + 7 * 86400000, NaN]) {
    const context = mock(); context.setClock(now);
    await assert.rejects(reserveFollowupStudy(approval(), context.ports), /approval/);
    assert.equal(context.writes.length, 0);
  }
  for (const changes of [{ priorPilotExists: true }, { targetExists: true }, { reservationExists: true },
    { ancestorsHaveReparsePoints: true }, { access: "unknown" }, { access: "posix-mode-only" },
    { privateParent: "C:\\invented\\Anime\\private", resolvedPrivateParent: "C:\\invented\\Anime\\private" },
    { resolvedPrivateParent: "C:\\other" }, { targetPath: "C:\\invented\\Anime-private\\other" },
    { priorPilotExists: undefined }]) {
    const context = mock(); Object.assign(context.evidence, changes);
    await assert.rejects(reserveFollowupStudy(approval(), context.ports), /preflight/);
    assert.equal(context.writes.length, 0);
  }
  for (const changes of [{ approvedAt: "bad" }, { approvedAt: "2026-10-08" },
    { expiresAt: new Date(start + 8 * 86400000).toISOString() }, { expiresAt: new Date(start).toISOString() }]) {
    await assert.rejects(reserveFollowupStudy({ ...approval(), ...changes }, mock().ports), /approval/);
  }
});

test("one-use reservation is durable across failed/interrupted work and concurrent starts", async () => {
  const context = mock();
  const results = await Promise.allSettled([reserveFollowupStudy(approval(), context.ports), reserveFollowupStudy(approval(), context.ports)]);
  assert.equal(results.filter((result) => result.status === "fulfilled").length, 1);
  assert.equal(context.writes.length, 1);
  const record: any = context.writes[0];
  assert.equal(record.state, "started"); assert.equal(record.scopeSha256, FOLLOWUP_SCOPE_SHA256);
  assert.match(record.approvalSha256, /^[a-f0-9]{64}$/);
  assert.equal(record.expiresAt, approval().expiresAt); assert.equal(record.publicArtifacts, false);
  context.evidence.targetExists = false; // Removing an output cannot remove the independent reservation.
  await assert.rejects(reserveFollowupStudy(approval(), context.ports), /preflight/);
  assert.equal(context.writes.length, 1);
});

test("expiry and cancellation are rechecked after inspection/reservation; port errors stay redacted", async () => {
  const context = mock(); context.ports.inspect = async () => { context.setClock(start + 7 * 86400000); return context.evidence; };
  await assert.rejects(reserveFollowupStudy(approval(), context.ports), /approval/); assert.equal(context.writes.length, 0);
  const cancelled = mock(), controller = new AbortController(); controller.abort("invented-secret");
  await assert.rejects(reserveFollowupStudy(approval(), cancelled.ports, controller.signal), /cancelled/);
  assert.equal(cancelled.inspections.length, 0); assert.equal(cancelled.writes.length, 0);
  for (const stage of ["inspect", "reserve"] as const) {
    const context = mock(), during = new AbortController();
    if (stage === "inspect") context.ports.inspect = async () => { during.abort(); return context.evidence; };
    else {
      const reserve = context.ports.reserveAtomic;
      context.ports.reserveAtomic = async (record) => { const result = await reserve(record); during.abort(); return result; };
    }
    await assert.rejects(reserveFollowupStudy(approval(), context.ports, during.signal), /cancelled/);
    assert.equal(context.writes.length, stage === "inspect" ? 0 : 1);
  }
  const expired = mock(), reserve = expired.ports.reserveAtomic;
  expired.ports.reserveAtomic = async (record) => { const result = await reserve(record); expired.setClock(start + 7 * 86400000); return result; };
  await assert.rejects(reserveFollowupStudy(approval(), expired.ports), /approval/); assert.equal(expired.writes.length, 1);
  for (const method of ["inspect", "reserveAtomic"] as const) {
    const broken = mock(); broken.ports[method] = async () => { throw new Error("invented-secret-path"); };
    await assert.rejects(reserveFollowupStudy(approval(), broken.ports), (error: any) => !error.message.includes("secret-path"));
  }
});

test("an approval cannot change across awaited inspection or reservation", async () => {
  for (const method of ["inspect", "reserveAtomic"] as const) {
    const context = mock(), record = approval();
    if (method === "inspect") context.ports.inspect = async () => { record.authority = "Another invented authority"; return context.evidence; };
    else {
      const reserve = context.ports.reserveAtomic;
      context.ports.reserveAtomic = async (value) => { const result = await reserve(value); record.authority = "Another invented authority"; return result; };
    }
    await assert.rejects(reserveFollowupStudy(record, context.ports), /changed during preflight/);
    assert.equal(context.writes.length, method === "inspect" ? 0 : 1);
  }
});

test("all live main and qualifier item/unit/calendar roles are complete and deduplicated", () => {
  const raw = projection({ P31: [statement(item("P31", "Q910000101"), "preferred"), statement(item("P31", "Q920000002"))],
    P136: [statement(item("P136", "Q920000002")), statement(item("P136", "Q920000099"), "deprecated")],
    P2756: [statement(item("P2756", "Q920000003"))],
    P2047: [statement(quantity("P2047", "http://www.wikidata.org/entity/Q920000004"))],
    P577: [statement(time("P577", "http://www.wikidata.org/entity/Q920000005"))],
    P580: [statement(time("P580", "http://www.wikidata.org/entity/Q920000005"), "normal", {
      P518: [item("P518", "Q920000002")], P1114: [quantity("P1114", "http://www.wikidata.org/entity/Q920000006")],
      P585: [time("P585", "http://www.wikidata.org/entity/Q920000007")],
      P2676: [{ snaktype: "value", property: "P2676", datatype: "string", datavalue: { type: "string", value: "invented certificate" } }],
    })], P155: [statement(item("P155", "Q930000001"))], P156: [statement(item("P156", "Q910000101"))] });
  const result = collectFollowupDefinitions(bytes(raw));
  assert.equal(result.requiredIds.length, 7); assert.deepEqual(result.reusedIds, ["Q910000101"]);
  assert.equal(result.reusedMissingEnglishLabels, 1);
  assert.equal(result.fetchIds.length, 6); assert.equal(result.outsideAcquiredRelationTargets, 1);
  assert.equal(result.byRole.mainFormat, 2); assert.equal(result.byRole.mainGenre, 1);
  assert.equal(result.byRole.qualifierItem, 1); assert.equal(result.byRole.qualifierUnit, 1); assert.equal(result.byRole.qualifierCalendar, 1);
  assert.equal(result.publicationAuthorized, false); assert.equal(JSON.stringify(result).includes("invented certificate"), false);
  const reversed = structuredClone(raw); reversed.entities.Q910000101.claims.P31.reverse();
  assert.equal(collectFollowupDefinitions(bytes(reversed)).inventorySha256, result.inventorySha256);
});

test("100 definitions yields all bounded batches; 101 fails without a partial inventory", () => {
  const raw = projection({ P136: Array.from({ length: 100 }, (_, index) => statement(item("P136", `Q${920000001 + index}`))) });
  const result = collectFollowupDefinitions(bytes(raw));
  assert.equal(result.requiredIds.length, 100); assert.equal(result.fetchBatches.length, 5);
  assert.deepEqual(result.fetchBatches.flat(), result.requiredIds);
  raw.entities.Q910000101.claims.P31 = [statement(item("P31", "Q940000001"))];
  assert.throws(() => collectFollowupDefinitions(bytes(raw)), /definition budget/);
});

test("canonical large IDs sort exactly; dimensionless units and unknown snaks add no definitions", () => {
  const ids = ["Q999999999999999999999999999999999999999", "Q999999999999999999999999999999999999998", "Q920000001"];
  const raw = projection({ P136: ids.map((id) => statement(item("P136", id))),
    P2047: [statement(quantity("P2047", "1"))], P31: [statement({ snaktype: "somevalue", property: "P31", datatype: "wikibase-item" })] });
  assert.deepEqual(collectFollowupDefinitions(bytes(raw)).requiredIds, [ids[2], ids[1], ids[0]]);
});

test("malformed references, hidden fields and structural bounds refuse complete accounting", () => {
  for (const snak of [item("P136", "Q01"), item("P136", "P31"),
    { ...item("P136", "Q920000001"), property: "P31" },
    quantity("P136", "https://unapproved.example/Q1"), time("P136", "not-a-calendar"),
    { snaktype: "value", property: "P136", datatype: "string", datavalue: { type: "mystery", value: { id: "Q920000001" } } }]) {
    assert.throws(() => collectFollowupDefinitions(bytes(projection({ P136: [statement(snak)] }))), /inventory/);
  }
  const bad = projection({ P136: [statement(item("P136", "Q920000001"), "normal", { P518: Array(101).fill(item("P518", "Q920000002")) })] });
  assert.throws(() => collectFollowupDefinitions(bytes(bad)), /inventory/);
  assert.throws(() => collectFollowupDefinitions(bytes(projection({ P136: Array(101).fill(statement(item("P136", "Q920000001"))) }))), /inventory/);
  assert.throws(() => collectFollowupDefinitions(bytes(projection({ P18: [] }))), /inventory/);
  assert.throws(() => collectFollowupDefinitions(bytes({ ...projection({}), hidden: true })), /inventory/);
  assert.throws(() => collectFollowupDefinitions(Buffer.from('{"entities":{},"entities":{}}')), /inventory/);
  assert.throws(() => collectFollowupDefinitions(Buffer.alloc(4 * 1024 * 1024 + 1)), /inventory/);
});

test("qualifier count and anime inventory bounds remain fail closed; deprecated contexts do not consume definitions", () => {
  const raw = projection({ P136: [statement(item("P136", "Q920000001"), "deprecated", { P518: [item("P518", "Q920000002")] })] });
  assert.equal(collectFollowupDefinitions(bytes(raw)).requiredIds.length, 0);
  raw.entities.Q910000101.claims.P136[0].rank = "normal";
  raw.entities.Q910000101.claims.P136[0].qualifiers = Object.fromEntries(Array.from({ length: 101 }, (_, index) => [`P${index + 1}`, []]));
  assert.throws(() => collectFollowupDefinitions(bytes(raw)), /inventory/);
  const tooMany = { entities: Object.fromEntries(Array.from({ length: 101 }, (_, index) => {
    const id = `Q${910000101 + index}`; return [id, entity({}, id)];
  })) };
  assert.throws(() => collectFollowupDefinitions(bytes(tooMany)), /inventory/);
  const badNumber = item("P136", "Q920000001"); (badNumber.datavalue.value as any)["numeric-id"] = 920000002;
  assert.throws(() => collectFollowupDefinitions(bytes(projection({ P136: [statement(badNumber)] }))), /inventory/);
  const badLabel = projection({}); (badLabel.entities.Q910000101.labels as any).en = { language: "ja", value: "Invented title" };
  assert.throws(() => collectFollowupDefinitions(bytes(badLabel)), /inventory/);
});
