import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { auditTvDateCandidate } from "../src/tv-date-candidate.js";
import { mapWikibaseMetadata, type WikibaseMappingPolicy } from "../src/wikibase-metadata.js";

const fixture = (name: string) => JSON.parse(readFileSync(new URL(`../../fixtures/${name}`, import.meta.url), "utf8"));
const bytes = (value: unknown) => Buffer.from(JSON.stringify(value));
const hash = (value: Uint8Array) => createHash("sha256").update(value).digest("hex");
const raw = () => fixture("synthetic-wikibase-entities.json");
const entity = (value: any) => value.entities.Q910000101;
const statement = (value: any) => entity(value).claims.P577[0];
const source = { name: "invented-date-scope-fixture", snapshotAt: "2026-10-07T00:00:00.000Z" };
function setup(value = raw()) {
  const sourceBytes = bytes(value), universe = [101, 102, 103, 104];
  const mappingPolicy = fixture("synthetic-wikibase-policy.json") as WikibaseMappingPolicy;
  mappingPolicy.mediaFormats.Q910001003 = "TV";
  const policy = fixture("synthetic-tv-date-policy.json"), datePolicyBytes = bytes(policy);
  const baseline = mapWikibaseMetadata(sourceBytes, universe, mappingPolicy, source);
  const scope = { format: "declared-tv-first-airing-scope-v1", sourceSnapshotSha256: hash(sourceBytes),
    mappingPolicySha256: baseline.report.policySha256, datePolicySha256: hash(datePolicyBytes), universeSha256: baseline.report.universeSha256,
    items: [{ animeId: 101, sourceItemId: "Q910000101", kind: "series", dateScope: "whole-work-first-airing" }] };
  return { input: { sourceBytes, universe, mappingPolicy, datePolicyBytes, scopeBytes: bytes(scope), source }, baseline, policy, scope };
}
const audit = (value = raw()) => auditTvDateCandidate(setup(value).input);
const first = (value: ReturnType<typeof audit>) => value.rows.find((row) => row.animeId === 101)!;
function fallback(value: any, year = 2021) {
  const date = structuredClone(statement(value)); date.mainsnak.property = "P580";
  date.mainsnak.datavalue.value.time = `+${year}-00-00T00:00:00Z`;
  entity(value).claims.P580 = [date];
}

test("byte-bound whole-series primary year has private provenance and fixed missing-ID denominators", () => {
  const result = audit(), row = first(result);
  assert.equal(row.year, 2021); assert.equal(row.selectedProperty, "P577"); assert.equal(row.minimumPrecision, 9);
  assert.match(row.selectedStatementsSha256!, /^[a-f0-9]{64}$/);
  assert.equal(result.universeItems, 4); assert.equal(result.usableItems, 1); assert.equal(result.primaryItems, 1); assert.equal(result.fallbackItems, 0);
  assert.equal(result.refusals.undeclaredScope, 2); assert.equal(result.refusals.missingIdentity, 1);
  assert.equal(result.publicationAuthorized, false);
  assert.equal(JSON.stringify(result).includes("+2021-00-00"), false);
  assert.equal(Object.values(result.refusals).reduce((n, count) => n + count, 0) + result.usableItems, result.universeItems);
});

test("fallback requires zero primary statements, never unusable or deprecated-only primary evidence", () => {
  const missing = raw(); fallback(missing, 2020); delete entity(missing).claims.P577;
  const result = audit(missing);
  assert.equal(first(result).year, 2020); assert.equal(first(result).selectedProperty, "P580"); assert.equal(result.fallbackItems, 1);
  assert.equal(setup(missing).baseline.snapshot!.anime[0].year, null); // Public v1 mapper is untouched.
  entity(missing).claims.P577 = []; assert.equal(first(audit(missing)).year, 2020);
  for (const [mutate, reason] of [
    [(s: any) => { s.qualifiers = { P518: [] }; }, "primaryQualified"],
    [(s: any) => { s.mainsnak.snaktype = "somevalue"; }, "primaryUnknown"],
    [(s: any) => { s.mainsnak.datavalue.value.precision = 8; }, "primaryInvalid"],
    [(s: any) => { s.rank = "deprecated"; }, "primaryDeprecatedOnly"],
  ] as const) {
    const value = raw(); fallback(value, 2020); mutate(statement(value));
    assert.equal(first(audit(value)).reason, reason);
  }
});

test("both properties must agree at year precision; unusable fallback cannot be hidden by a valid primary", () => {
  const agree = raw(); fallback(agree);
  assert.equal(first(audit(agree)).selectedProperty, "P577");
  for (const [mutate, reason] of [
    [(s: any) => { s.mainsnak.datavalue.value.time = "+2020-00-00T00:00:00Z"; }, "crossPropertyConflict"],
    [(s: any) => { s.qualifiers = { P580: [] }; }, "fallbackQualified"],
    [(s: any) => { s.mainsnak.snaktype = "novalue"; }, "fallbackUnknown"],
    [(s: any) => { s.mainsnak.datavalue.value.calendarmodel = "invented-calendar"; }, "fallbackInvalid"],
    [(s: any) => { s.rank = "deprecated"; }, "fallbackDeprecatedOnly"],
  ] as const) {
    const value = raw(); fallback(value); mutate(entity(value).claims.P580[0]);
    assert.equal(first(audit(value)).reason, reason);
  }
  const absent = raw(); delete entity(absent).claims.P577;
  assert.equal(first(audit(absent)).reason, "fallbackMissing");
});

test("all live date ranks must agree and statement ordering never chooses a year or provenance", () => {
  const value = raw(), other = structuredClone(statement(value)); statement(value).rank = "preferred";
  other.mainsnak.datavalue.value.time = "+2022-00-00T00:00:00Z";
  entity(value).claims.P577.push(other);
  assert.equal(first(audit(value)).reason, "primaryConflict");
  assert.equal(setup(value).baseline.snapshot!.anime[0].year, 2021); // Legacy best rank remains separate.
  other.mainsnak.datavalue.value.time = "+2021-00-00T00:00:00Z";
  const before = first(audit(value)); entity(value).claims.P577.reverse();
  const after = first(audit(value));
  assert.deepEqual(after, before);
  const reordered = structuredClone(value);
  const original = statement(reordered).mainsnak.datavalue.value;
  statement(reordered).mainsnak.datavalue.value = Object.fromEntries(Object.entries(original).reverse());
  assert.deepEqual(first(audit(reordered)), after);
  other.rank = "deprecated"; other.mainsnak.datavalue.value.time = "+2022-00-00T00:00:00Z";
  assert.equal(first(audit(value)).year, 2021);
  const fallbackConflict = raw(); fallback(fallbackConflict); delete entity(fallbackConflict).claims.P577;
  const conflicting = structuredClone(entity(fallbackConflict).claims.P580[0]); conflicting.mainsnak.datavalue.value.time = "+2022-00-00T00:00:00Z";
  entity(fallbackConflict).claims.P580.push(conflicting);
  assert.equal(first(audit(fallbackConflict)).reason, "fallbackConflict");
});

test("date precision, uncertainty and calendar rules reject noncanonical or impossible evidence", () => {
  for (const change of [
    { precision: 8 }, { before: 1 }, { after: 1 }, { timezone: 60 }, { calendarmodel: "invented-calendar" },
    { time: "+2021-01-01T00:00:00Z", precision: 9 }, { time: "+2021-01-01T00:00:00Z", precision: 10 },
    { time: "+2021-02-29T00:00:00Z", precision: 11 }, { time: "+2020-13-01T00:00:00Z", precision: 11 },
    { time: "+1799-00-00T00:00:00Z" }, { time: "+3001-00-00T00:00:00Z" }, { extra: "invented-hidden" },
  ]) {
    const value = raw(); Object.assign(statement(value).mainsnak.datavalue.value, change);
    assert.equal(first(audit(value)).reason, "primaryInvalid");
  }
  for (const date of [{ time: "+2021-06-00T00:00:00Z", precision: 10 }, { time: "+2020-02-29T00:00:00Z", precision: 11 }]) {
    const value = raw(); Object.assign(statement(value).mainsnak.datavalue.value, date);
    const result = first(audit(value)); assert.equal(result.year, Number(date.time.slice(1, 5))); assert.equal(result.minimumPrecision, date.precision);
  }
});

test("TV format alone cannot establish series/season scope or exempt qualified type/date evidence", () => {
  for (const mutate of [
    (value: any) => { entity(value).claims.P31[0].mainsnak.datavalue.value.id = "Q910001002"; },
    (value: any) => { entity(value).claims.P31[0].mainsnak.datavalue.value.id = "Q910001003"; },
    (value: any) => { entity(value).claims.P31[0].qualifiers = { P518: [] }; },
    (value: any) => { delete entity(value).claims.P31; },
    (value: any) => { const other = structuredClone(entity(value).claims.P31[0]); other.mainsnak.datavalue.value.id = "Q910001003"; entity(value).claims.P31.push(other); },
  ]) {
    const value = raw(); mutate(value); assert.equal(first(audit(value)).reason, "scopeMismatch");
  }
  const season = raw(); entity(season).claims.P31[0].mainsnak.datavalue.value.id = "Q910001003";
  const data = setup(season); data.scope.items[0].kind = "season"; data.input.scopeBytes = bytes(data.scope);
  assert.equal(first(auditTvDateCandidate(data.input)).year, 2021);
  const unreviewed = setup(); unreviewed.scope.items = []; unreviewed.input.scopeBytes = bytes(unreviewed.scope);
  assert.equal(first(auditTvDateCandidate(unreviewed.input)).reason, "undeclaredScope");
});

test("source, mapping, date policy and universe bindings cannot be replaced by an authored digest", () => {
  for (const field of ["sourceSnapshotSha256", "mappingPolicySha256", "datePolicySha256", "universeSha256"] as const) {
    const value = setup(); value.scope[field] = "a".repeat(64); value.input.scopeBytes = bytes(value.scope);
    assert.throws(() => auditTvDateCandidate(value.input), /scope.bindings/);
  }
  const changed = setup(); changed.input.sourceBytes = bytes({ ...raw(), extra: "invented-byte-change" });
  assert.throws(() => auditTvDateCandidate(changed.input), /scope.bindings/);
  const wrongItem = setup(); wrongItem.scope.items[0].sourceItemId = "Q910000102"; wrongItem.input.scopeBytes = bytes(wrongItem.scope);
  assert.throws(() => auditTvDateCandidate(wrongItem.input), /scope.items.animeId-101.sourceItemId/);
  const differentClock = setup(); differentClock.input.source.snapshotAt = "2025-01-01T00:00:00.000Z";
  assert.equal(first(auditTvDateCandidate(differentClock.input)).year, 2021);
});

test("malformed policies/scopes and over-budget source dates fail without echoing private payloads", () => {
  for (const change of [
    (v: any) => { v.primaryProperty = "P580"; }, (v: any) => { v.dateScope = "episode-airing"; },
    (v: any) => { v.typeScopes.Q910001001 = "franchise"; }, (v: any) => { v.typeScopes.Q910001002 = "series"; },
    (v: any) => { v.ignoreQualifier = "invented-private"; },
  ]) {
    const value = setup(); change(value.policy); value.input.datePolicyBytes = bytes(value.policy);
    assert.throws(() => auditTvDateCandidate(value.input), (error: any) => /policy/.test(error.message) && !error.message.includes("invented-private"));
  }
  for (const change of [
    (v: any) => { v.items[0].dateScope = "part-airing"; }, (v: any) => { v.items[0].kind = "franchise"; },
    (v: any) => { v.items.push(v.items[0]); }, (v: any) => { v.items[0].animeId = 999; },
    (v: any) => { v.items[0].extra = "invented-private"; },
  ]) {
    const value = setup(); change(value.scope); value.input.scopeBytes = bytes(value.scope);
    assert.throws(() => auditTvDateCandidate(value.input), (error: any) => /scope/.test(error.message) && !error.message.includes("invented-private"));
  }
  const many = raw(); fallback(many); entity(many).claims.P580 = Array.from({ length: 101 }, () => entity(many).claims.P580[0]);
  assert.throws(() => audit(many), /source.claims.P580/);
  const malformed = raw(); fallback(malformed); entity(malformed).claims.P580[0].mainsnak.property = "P9999";
  assert.throws(() => audit(malformed), /source.claims.P580.statement/);
  const duplicateJson = setup(); duplicateJson.input.scopeBytes = Buffer.from('{"format":"invented-private","format":"duplicate"}');
  assert.throws(() => auditTvDateCandidate(duplicateJson.input), (error: any) => /candidate scope/.test(error.message) && !error.message.includes("invented-private"));
  const oversized = setup(); oversized.input.datePolicyBytes = Buffer.alloc(4 * 1024 * 1024 + 1, 32);
  assert.throws(() => auditTvDateCandidate(oversized.input), /candidate policy/);
});
