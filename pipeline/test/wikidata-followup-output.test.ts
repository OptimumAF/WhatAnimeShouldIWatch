import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import { inventedFollowupResult } from "./invented-followup-result.js";
import { prepareFollowupOutput, saveFollowupOutput } from "../src/wikidata-followup-output.js";
import { collectFollowupDefinitions } from "../src/wikidata-followup-gates.js";

test("private output recomputes exact five-file bytes and binds completion to private inventory", async () => {
  const context = await inventedFollowupResult(), payload = prepareFollowupOutput(context.approval, context.result, context.now());
  assert.deepEqual(payload.files.map((file) => file.name), ["source-projection.json", "definition-labels.json", "inventory.json", "receipt.json", "completed.json"]);
  for (const file of payload.files) {
    const bytes = Buffer.from(file.base64, "base64"); assert.equal(file.bytes, bytes.length);
    assert.equal(file.sha256, createHash("sha256").update(bytes).digest("hex"));
  }
  const completion = JSON.parse(Buffer.from(payload.files[4].base64, "base64").toString());
  assert.deepEqual(completion.files, payload.files.slice(0, 4).map(({ name, bytes, sha256 }) => ({ name, bytes, sha256 })));
  assert.equal(completion.mappingReviewed, false); assert.equal(completion.publicArtifacts, false);
  let calls = 0;
  assert.deepEqual(await saveFollowupOutput(context.approval, context.result, context.now, { save: async () => { calls++; return { saved: true }; } }), { saved: true, publicArtifacts: false });
  assert.equal(calls, 1);
});
test("changed bytes/inventory/receipt, hidden fields and wrong reservation fail before any writer", async () => {
  const context = await inventedFollowupResult();
  const mutations = [
    (value: any) => { value.sourceBytes = Buffer.from('{"entities":{}}'); },
    (value: any) => { value.inventory.requiredIds = []; },
    (value: any) => { value.reservation.approvalSha256 = "0".repeat(64); },
    (value: any) => { value.receipt.projectionSha256 = "0".repeat(64); },
    (value: any) => { value.receipt.publicArtifacts = true; },
    (value: any) => { value.receipt.mappingReviewed = true; },
    (value: any) => { value.receipt.privateExtra = "invented secret"; },
    (value: any) => { value.receipt.totalBytes++; },
    (value: any) => { value.receipt.transportBodies[0].outcome = "http-retry"; },
    (value: any) => { value.receipt.transportBodies[0].kind = "definitions"; },
    (value: any) => { value.receipt.revisions.Q920000001 = -1; },
    (value: any) => { value.definitionBytes = Buffer.from('{"entities":{}}'); },
    (value: any) => { value.receipt.snapshotAt = new Date(context.now() + 1000).toISOString(); },
    (value: any) => {
      const source = JSON.parse(Buffer.from(value.sourceBytes).toString());
      source.entities.Q910000101.claims.P4086[0].mainsnak.datavalue.value = "1";
      value.sourceBytes = Buffer.from(JSON.stringify(source)); value.inventory = collectFollowupDefinitions(value.sourceBytes);
      value.receipt.projectionSha256 = createHash("sha256").update(value.sourceBytes).digest("hex");
    },
  ];
  let writes = 0;
  for (const mutate of mutations) {
    const result = structuredClone(context.result); mutate(result);
    await assert.rejects(saveFollowupOutput(context.approval, result, context.now, { save: async () => { writes++; return { saved: true }; } }),
      (error: any) => /Wikidata follow-up output/.test(error.message) && !error.message.includes("secret"));
  }
  assert.equal(writes, 0);
});
test("closed/expired approval and input bounds refuse writes; failed ports and changed approval stay failed", async () => {
  const context = await inventedFollowupResult(); let writes = 0;
  const port = { save: async () => { writes++; return { saved: true }; } };
  for (const approval of [null, { ...context.approval, state: "completed" }, { ...context.approval, expiresAt: new Date(context.now()).toISOString() }])
    await assert.rejects(saveFollowupOutput(approval, context.result, context.now, port), /output/);
  const oversized = { ...context.result, definitionBytes: new Uint8Array(4 * 1024 * 1024 + 1) };
  await assert.rejects(saveFollowupOutput(context.approval, oversized, context.now, port), /output/); assert.equal(writes, 0);
  await assert.rejects(saveFollowupOutput(context.approval, context.result, context.now, { save: async () => { throw new Error("invented private path"); } }),
    (error: any) => /output/.test(error.message) && !error.message.includes("path"));
  await assert.rejects(saveFollowupOutput(context.approval, context.result, context.now, { save: async () => ({ saved: true, hidden: true }) }), /output/);
  await assert.rejects(saveFollowupOutput(context.approval, context.result, () => { throw new Error("invented private clock error"); }, port),
    (error: any) => /output/.test(error.message) && !error.message.includes("clock"));
  const changed = { ...context.approval };
  await assert.rejects(saveFollowupOutput(changed, context.result, context.now, { save: async () => { changed.authority = "changed"; return { saved: true }; } }), /output/);
});
