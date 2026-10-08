import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import { parseCatalogMetadataSnapshot } from "../../web/src/artifacts.js";
import { mapSyntheticScopedMetadata } from "../src/scoped-metadata-candidate.js";
import { mapWikibaseMetadata } from "../src/wikibase-metadata.js";
import { scopedMetadataFixture, syntheticWikibaseSource } from "./synthetic-scoped-metadata.js";

const raw = syntheticWikibaseSource;
const entity = (value: any) => value.entities.Q910000101;
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
function start(value: any, year = 2024) {
  const statement = structuredClone(entity(value).claims.P577[0]); statement.mainsnak.property = "P580";
  statement.mainsnak.datavalue.value.time = `+${year}-00-00T00:00:00Z`; entity(value).claims.P580 = [statement];
}
const mapped = (value = raw(), ids = [101]) => mapSyntheticScopedMetadata(scopedMetadataFixture(value, ids));
const year = (result: ReturnType<typeof mapped>) => result.snapshot!.anime.find((item) => item.animeId === 101)!.year;

test("explicit synthetic mapper uses scoped fallback and binds exact catalog bytes without private fields", () => {
  const value = raw(); start(value); delete entity(value).claims.P577;
  const input = scopedMetadataFixture(value), before = mapWikibaseMetadata(input.sourceBytes, input.universe, input.mappingPolicy, input.source);
  const result = mapSyntheticScopedMetadata(input);
  assert.equal(before.snapshot!.anime[0].year, null); assert.equal(year(result), 2024);
  assert.equal(result.privateAudit.dateAudit.rows[0].selectedProperty, "P580");
  assert.equal(result.privateAudit.metadataSha256, hash(result.metadataBytes!));
  const decoded = JSON.parse(Buffer.from(result.metadataBytes!).toString());
  assert.deepEqual(decoded, result.snapshot); assert.deepEqual(parseCatalogMetadataSnapshot(decoded), result.snapshot);
  for (const text of ["selectedProperty", "minimumPrecision", "selectedStatementsSha256", "privateAudit", "scopeSha256", "P580"]) {
    assert.equal(Buffer.from(result.metadataBytes!).toString().includes(text), false, text);
  }
  assert.equal(result.privateAudit.publicationAuthorized, false);
  assert.equal(result.privateAudit.coverage.known.year, 2); // Scoped TV plus unchanged movie primary.
  assert.equal(before.report.coverage.known.year, 1);
  assert.equal(result.privateAudit.baseMappingAudit.fieldIssues.year.missing, 2); // Deliberately labeled baseline only.
  assert.equal(result.privateAudit.yearBasis.scopedTv, 1);
});

test("conflict, qualified primary, undeclared TV scope and unknown format clear a previously known candidate year", () => {
  const conflict = raw(); start(conflict);
  assert.equal(year(mapped(conflict)), null);
  assert.equal(mapped(conflict).privateAudit.dateAudit.rows[0].reason, "crossPropertyConflict");
  const qualified = raw(); start(qualified, 2021); entity(qualified).claims.P577[0].qualifiers = { P518: [] };
  assert.equal(year(mapped(qualified)), null);
  assert.equal(year(mapped(raw(), [])), null);
  const unknown = raw(); delete entity(unknown).claims.P31;
  assert.equal(year(mapped(unknown)), null);
  const input = scopedMetadataFixture(conflict);
  assert.equal(mapWikibaseMetadata(input.sourceBytes, input.universe, input.mappingPolicy, input.source).snapshot!.anime[0].year, 2021);
});

test("non-TV publication years and every unrelated field stay at their base mapper values", () => {
  const value = raw(); start(value, 2021);
  const input = scopedMetadataFixture(value), baseline = mapWikibaseMetadata(input.sourceBytes, input.universe, input.mappingPolicy, input.source);
  const result = mapSyntheticScopedMetadata(input);
  assert.equal(result.snapshot!.anime.find((item) => item.animeId === 102)!.year, 2022);
  assert.equal(result.privateAudit.yearBasis.legacyPublication, 1);
  for (const entry of result.snapshot!.anime) {
    const { year: _year, ...other } = entry;
    const { year: _baselineYear, ...base } = baseline.snapshot!.anime.find((item) => item.animeId === entry.animeId)!;
    assert.deepEqual(other, base);
  }
  assert.deepEqual(baseline.snapshot!.source, result.snapshot!.source);
});

test("v2 certificate candidate composes without leaking its certificate or changing v1/v2 outputs", () => {
  const value = raw(); start(value); delete entity(value).claims.P577;
  value.entities.Q910000102.claims.P2756[0].qualifiers = { P2676: [{ snaktype: "value", property: "P2676", datatype: "string",
    datavalue: { type: "string", value: "invented-private-certificate" } }] };
  const input = scopedMetadataFixture(value, [101], "v2"), result = mapSyntheticScopedMetadata(input);
  assert.equal(year(result), 2024);
  assert.equal(result.snapshot!.anime.find((item) => item.animeId === 102)!.contentClassification!.value, "All");
  assert.equal(Buffer.from(result.metadataBytes!).toString().includes("invented-private-certificate"), false);
  assert.equal(JSON.stringify(result.privateAudit).includes("invented-private-certificate"), false);
  assert.equal(mapWikibaseMetadata(input.sourceBytes, input.universe, input.mappingPolicy, input.source).snapshot!.anime[0].year, null);
});

test("fixture purpose is explicit and stale or malformed scope fails before candidate output", () => {
  for (const change of [
    (input: any) => { input.purpose = "production"; }, (input: any) => { input.format = "wikibase-metadata-policy-v3"; },
    (input: any) => { input.source.name = "actual-source"; }, (input: any) => { input.extra = "invented-private"; },
  ]) {
    const input = scopedMetadataFixture(); change(input);
    assert.throws(() => mapSyntheticScopedMetadata(input), (error: any) => /Synthetic scoped metadata/.test(error.message) && !error.message.includes("invented-private"));
  }
  const stale = scopedMetadataFixture(); const value = raw(); start(value);
  stale.sourceBytes = Buffer.from(JSON.stringify(value));
  assert.throws(() => mapSyntheticScopedMetadata(stale), /scope.bindings/);
  const malformed = scopedMetadataFixture(); malformed.scopeBytes = Buffer.from("{}");
  assert.throws(() => mapSyntheticScopedMetadata(malformed), /candidate scope/);
});

test("ambiguous identities cannot acquire a fallback and an empty identity result is not a fabricated catalog", () => {
  const value = raw(); start(value); delete entity(value).claims.P577;
  const duplicate = structuredClone(entity(value)); duplicate.id = "Q910000999"; value.entities.Q910000999 = duplicate;
  const result = mapped(value);
  assert.equal(result.snapshot!.anime.some((item) => item.animeId === 101), false);
  assert.equal(result.privateAudit.dateAudit.refusals.missingIdentity, 2); // Quarantined 101 plus absent 104.
  assert.equal(result.privateAudit.coverage.missingItems, 2);
  const empty = raw(); for (const item of Object.values(empty.entities) as any[]) delete item.claims.P4086;
  const none = mapped(empty);
  assert.equal(none.snapshot, null); assert.equal(none.metadataBytes, null); assert.equal(none.privateAudit.metadataSha256, null);
  assert.equal(none.privateAudit.dateAudit.usableItems, 0); assert.equal(none.privateAudit.coverage.missingItems, 4);
});
