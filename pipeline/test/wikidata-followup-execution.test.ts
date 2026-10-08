import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { executeFollowupStudy, createFollowupTimingPorts, type FollowupExecutionPorts } from "../src/wikidata-followup-execution.js";
import { inventedFollowupPorts } from "./invented-followup-result.js";
import { FOLLOWUP_SCOPE_SHA256 } from "../src/wikidata-followup-gates.js";

function context() {
  const fixture = inventedFollowupPorts(), events: string[] = [];
  const inspect = fixture.ports.inspect, reserve = fixture.ports.reserveAtomic, fetch = fixture.ports.fetch;
  fixture.ports.inspect = async () => { events.push("inspect"); return inspect(); };
  fixture.ports.reserveAtomic = async (record) => { events.push("reserve"); return reserve(record); };
  fixture.ports.fetch = async (input, init) => { events.push("fetch"); return fetch(input, init); };
  const ports: FollowupExecutionPorts = { transport: fixture.ports,
    verifyOwnerApproval: async (binding) => { events.push("owner"); assert.equal(binding.scopeSha256, FOLLOWUP_SCOPE_SHA256); assert.match(binding.approvalSha256, /^[a-f0-9]{64}$/); return true; },
    output: { save: async (payload) => { events.push("save"); assert.equal(payload.files.length, 5); return { saved: true }; } } };
  return { ...fixture, ports, events };
}
test("closed approvals and absent/false/declared owner checks refuse before OS, reservation, transport or output", async () => {
  for (const missing of ["output", "fetch", "sleep", "inspect"] as const) {
    const fixture = context();
    if (missing === "output") fixture.ports.output.save = undefined as any;
    else fixture.ports.transport[missing] = undefined as any;
    const result = await executeFollowupStudy(fixture.approval, fixture.ports);
    assert.equal(result.state, "refused"); assert.equal(result.phase, "configuration"); assert.deepEqual(fixture.events, []);
  }
  const old = JSON.parse(readFileSync(new URL("../../docs/approvals/wikidata-feasibility.json", import.meta.url), "utf8"));
  for (const approval of [old, null]) {
    const fixture = context(), result = await executeFollowupStudy(approval, fixture.ports);
    assert.equal(result.state, "refused"); assert.equal(result.phase, "approval"); assert.deepEqual(fixture.events, []);
  }
  for (const verifier of [undefined, async () => false, async () => ({ approved: true }), async () => { throw new Error("invented private authority"); }]) {
    const fixture = context(); fixture.ports.verifyOwnerApproval = verifier as any;
    const result = await executeFollowupStudy(fixture.approval, fixture.ports);
    assert.equal(result.state, "refused"); assert.equal(result.phase, "owner-review"); assert.deepEqual(fixture.events, []);
    assert.ok(!JSON.stringify(result).includes("authority"));
  }
});
test("owner-verification timeout, abort, late success and approval mutation cannot start a study", async () => {
  for (const external of [false, true]) {
    const fixture = context(), controller = new AbortController(); let expire!: () => void, resolve!: (value: boolean) => void;
    let ownerSignal!: AbortSignal, cancelled = false;
    fixture.ports.transport.deadline = (_ms, callback) => { expire = callback; return () => { cancelled = true; }; };
    fixture.ports.verifyOwnerApproval = async (_binding, signal) => { ownerSignal = signal!; return new Promise((done) => { resolve = done; }); };
    const pending = executeFollowupStudy(fixture.approval, fixture.ports, controller.signal);
    await new Promise(setImmediate); if (external) controller.abort(); else expire();
    const result = await pending; assert.equal(result.state, "refused"); assert.equal(ownerSignal.aborted, true); assert.equal(cancelled, true);
    resolve(true); await new Promise(setImmediate); assert.deepEqual(fixture.events, []);
  }
  const fixture = context(); fixture.ports.verifyOwnerApproval = async () => { fixture.approval.authority = "changed"; return true; };
  assert.equal((await executeFollowupStudy(fixture.approval, fixture.ports)).state, "refused"); assert.deepEqual(fixture.events, []);
});
test("invented complete execution checks the owner verifier first, consumes once and uses the private output boundary", async () => {
  const fixture = context(), result = await executeFollowupStudy(fixture.approval, fixture.ports);
  assert.equal(result.state, "completed"); assert.equal(result.consumption, "consumed"); assert.equal(result.outputStatus, "verified");
  assert.deepEqual(fixture.events.slice(0, 4), ["owner", "inspect", "reserve", "fetch"]); assert.equal(fixture.events.at(-1), "save");
  assert.ok(!/Q910|Invented output anime|authority|Anime-private/.test(JSON.stringify(result))); assert.equal(result.publicArtifacts, false);
  const fetches = fixture.events.filter((event) => event === "fetch").length;
  const second = await executeFollowupStudy(fixture.approval, fixture.ports);
  assert.equal(second.state, "refused"); assert.equal(second.consumption, "not-attempted");
  assert.equal(fixture.events.filter((event) => event === "fetch").length, fetches); assert.equal(fixture.events.filter((event) => event === "save").length, 1);
});
test("transport/output failures stay consumed; uncertain reservation never fabricates a verified retention deadline", async () => {
  for (const phase of ["transport", "output"] as const) {
    const fixture = context();
    if (phase === "transport") fixture.ports.transport.fetch = async () => { throw new Error("invented provider/path secret"); };
    else fixture.ports.output.save = async () => { throw new Error("invented output/path secret"); };
    const result = await executeFollowupStudy(fixture.approval, fixture.ports);
    assert.equal(result.state, "failed"); assert.equal(result.phase, phase); assert.equal(result.consumption, "consumed");
    assert.equal(result.outputStatus, phase === "transport" ? "not-attempted" : "attempted"); assert.ok(result.expiresAt);
    assert.ok(!JSON.stringify(result).includes("secret")); assert.equal((await executeFollowupStudy(fixture.approval, fixture.ports)).state, "refused");
  }
  const uncertain = context(), reserve = uncertain.ports.transport.reserveAtomic;
  uncertain.ports.transport.reserveAtomic = async (record) => { await reserve(record); throw new Error("invented after-create failure"); };
  const result = await executeFollowupStudy(uncertain.approval, uncertain.ports);
  assert.equal(result.consumption, "uncertain"); assert.equal(result.state, "failed"); assert.equal(result.expiresAt, null); assert.equal(result.startedAt, null);
  assert.ok(!uncertain.events.includes("fetch")); assert.equal((await executeFollowupStudy(uncertain.approval, uncertain.ports)).state, "refused");
  const stalled = context(); let expire!: () => void, outputSignal!: AbortSignal;
  stalled.ports.transport.deadline = (_ms, callback) => { expire = callback; return () => {}; };
  stalled.ports.output.save = async (_payload, signal) => { outputSignal = signal!; return new Promise(() => {}); };
  const pending = executeFollowupStudy(stalled.approval, stalled.ports); await new Promise(setImmediate);
  assert.ok(outputSignal); expire(); const failed = await pending;
  assert.equal(failed.state, "failed"); assert.equal(failed.outputStatus, "attempted"); assert.equal(outputSignal.aborted, true);
});
test("actual Node timing ports bound waits, cancel deadlines and settle abortable sleep", async () => {
  const timing = createFollowupTimingPorts(); const before = Date.now(); let fired = false;
  assert.ok(timing.now() >= before);
  const cancel = timing.deadline(10, () => { fired = true; }); cancel(); await timing.sleep(20, new AbortController().signal); assert.equal(fired, false);
  await new Promise<void>((resolve) => { timing.deadline(5, resolve); });
  const controller = new AbortController(), waiting = timing.sleep(60000, controller.signal); controller.abort(); await assert.rejects(waiting, /abort/i);
  for (const value of [-1, 0.5, NaN, 7 * 86400000 + 1]) assert.throws(() => timing.deadline(value, () => {}), /execution/);
  await assert.rejects(timing.sleep(60001, new AbortController().signal), /execution/);
});
test("real Node deadline ends a mocked stalled body at approval expiry without saving source bytes", { timeout: 10000 }, async () => {
  const fixture = context(), timing = createFollowupTimingPorts(); Object.assign(fixture.ports.transport, timing);
  fixture.approval.approvedAt = new Date(Date.now() - 60000).toISOString(); fixture.approval.expiresAt = new Date(Date.now() + 1500).toISOString();
  let cancelled = false, fetched = 0;
  fixture.ports.transport.fetch = async () => { fetched++; return new Response(new ReadableStream({ cancel() { cancelled = true; } })); };
  const result = await executeFollowupStudy(fixture.approval, fixture.ports);
  assert.equal(result.state, "failed"); assert.equal(result.consumption, "consumed"); assert.equal(fetched, 1); assert.equal(cancelled, true);
  assert.equal(result.outputStatus, "not-attempted"); assert.ok(!fixture.events.includes("save"));
});
