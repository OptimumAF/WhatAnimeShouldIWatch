import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseCatalogMetadataSnapshot } from "../../web/src/artifacts.js";
import { projectCatalogMetadata } from "../../web/src/catalog-metadata.js";
import { mapWikibaseMetadata, WIKIBASE_MAPPING_LIMITS, type WikibaseMappingPolicy } from "../src/wikibase-metadata.js";

const fixtureBytes = readFileSync(new URL("../../fixtures/synthetic-wikibase-entities.json", import.meta.url));
const policy = JSON.parse(readFileSync(new URL("../../fixtures/synthetic-wikibase-policy.json", import.meta.url), "utf8")) as WikibaseMappingPolicy;
const source = { name: "invented-wikibase-fixture", snapshotAt: "2026-10-06T00:00:00.000Z" };
const universe = [101, 102, 103, 104];
const raw = (): any => JSON.parse(fixtureBytes.toString("utf8"));
const encode = (value: unknown) => Buffer.from(JSON.stringify(value));
const map = (value = raw(), mapping = policy) => mapWikibaseMetadata(encode(value), universe, mapping, source);
const first = (value: any, property: string) => value.entities.Q910000101.claims[property][0];
const item = (property: string, id: string, rank = "normal") => ({ type: "statement", rank,
  mainsnak: { snaktype: "value", property, datatype: "wikibase-item",
    datavalue: { type: "wikibase-entityid", value: { "entity-type": "item", id } } } });

test("invented canonical statements produce strict metadata, actual source hash, and hand-counted coverage", () => {
  const result = mapWikibaseMetadata(fixtureBytes, universe, policy, source);
  assert.ok(result.snapshot);
  assert.equal(parseCatalogMetadataSnapshot(result.snapshot, "mapped fixture"), result.snapshot);
  assert.equal(result.snapshot.source.snapshotSha256, createHash("sha256").update(fixtureBytes).digest("hex"));
  assert.deepEqual(result.snapshot.anime[0], { animeId: 101, sourceItemId: "Q910000101", title: "Copper Comet",
    aliases: ["Copper Voyage", "銅の彗星"], genres: ["Adventure"], year: 2021, mediaFormat: "TV",
    episodeCount: 12, runtimeMinutes: 24, contentClassification: null, communityScore: null,
    relations: [{ kind: "sequel", animeId: 102, title: "Moonlit Workshop" }] });
  const film = result.snapshot.anime[1];
  assert.equal(film.runtimeMinutes, 95); // 5,700 invented seconds with the declared unit map.
  assert.deepEqual(film.contentClassification, { jurisdiction: "Fixtureland", system: "Invented board", value: "All" });
  assert.deepEqual(projectCatalogMetadata(film).relations, [{ kind: "prequel", animeId: 101, title: "Copper Comet" }]);
  assert.equal(result.snapshot.anime[2].genres, null);
  assert.equal(result.snapshot.anime[2].relations, null); // P179 is no prerequisite.
  assert.deepEqual(result.report.coverage, { total: 4, missingItems: 1,
    known: { aliases: 3, genres: 2, year: 2, mediaFormat: 3, episodeCount: 2, runtimeMinutes: 2,
      contentClassification: 1, communityScore: 0, relations: 2 },
    usable: { aliases: 2, genres: 2, year: 2, mediaFormat: 3, episodeCount: 2, runtimeMinutes: 2,
      contentClassification: 1, communityScore: 0, relations: 2 },
    directedRelationItems: 2, directedTargetsOutsideUniverse: 0 });
  assert.equal(result.report.ambiguousTitleKeys, 1);
  assert.equal(result.report.oneSidedDirectedEdges, 0);
  assert.equal(result.report.fieldIssues.genres.unmapped, 1);
  assert.equal(JSON.stringify(result.report).includes("Copper"), false);
  assert.equal(JSON.stringify(result.report).includes("Q910"), false);
  assert.equal(JSON.stringify(result.snapshot).includes("Invented cover"), false);
  assert.throws(() => parseCatalogMetadataSnapshot({ ...result.snapshot, images: [] }, "mapped fixture"), /unsupported/);
});

test("entity, statement, and policy key order never select a different value; byte provenance changes independently", () => {
  const before = map();
  const changed = raw();
  changed.entities = Object.fromEntries(Object.entries(changed.entities).reverse());
  for (const entity of Object.values(changed.entities) as any[]) {
    entity.claims = Object.fromEntries(Object.entries(entity.claims).reverse());
    for (const statements of Object.values(entity.claims) as any[][]) statements.reverse();
  }
  const changedPolicy = Object.fromEntries(Object.entries(policy).reverse()) as unknown as WikibaseMappingPolicy;
  const after = map(changed, changedPolicy);
  assert.deepEqual(after.snapshot?.anime, before.snapshot?.anime);
  assert.equal(after.report.policySha256, before.report.policySha256);
  assert.equal(after.report.universeSha256, before.report.universeSha256);
  assert.notEqual(after.report.sourceSnapshotSha256, before.report.sourceSnapshotSha256);
});

test("duplicate entities, qualified identity, and multiple live external IDs remain unresolved", () => {
  const duplicate = raw();
  duplicate.entities.Q910000104 = { ...structuredClone(duplicate.entities.Q910000101), id: "Q910000104" };
  let result = map(duplicate);
  assert.equal(result.report.identity.duplicateMappings, 1);
  assert.deepEqual(result.snapshot?.anime.map((entry) => entry.animeId), [102, 103]);
  assert.equal(result.snapshot?.anime[0].relations, null);
  const ambiguous = raw();
  const second = structuredClone(first(ambiguous, "P4086"));
  second.rank = "preferred";
  second.mainsnak.datavalue.value = "104";
  ambiguous.entities.Q910000101.claims.P4086.push(second);
  result = map(ambiguous);
  assert.equal(result.report.identity.unresolved, 1); // Preferred does not guess external identity.
  assert.deepEqual(result.snapshot?.anime.map((entry) => entry.animeId), [102, 103]);
  ambiguous.entities.Q910000104 = { ...structuredClone(raw().entities.Q910000101), id: "Q910000104" };
  result = map(ambiguous);
  assert.equal(result.report.identity.duplicateMappings, 1);
  assert.deepEqual(result.snapshot?.anime.map((entry) => entry.animeId), [102, 103]);
  const qualified = raw();
  first(qualified, "P4086").qualifiers = { P1545: [] };
  assert.equal(map(qualified).report.identity.unresolved, 1);
  const empty = map({ entities: {} });
  assert.equal(empty.snapshot, null);
  assert.equal(empty.report.coverage.missingItems, universe.length);
});

test("best-rank fields preserve conflicts, qualifiers, special snaks, and unmapped genres as explicit unknowns", () => {
  const ranked = raw();
  ranked.entities.Q910000101.claims.P31.push(item("P31", "Q910001002", "preferred"));
  first(ranked, "P31").rank = "deprecated";
  assert.equal(map(ranked).snapshot?.anime[0].mediaFormat, "Movie");
  const conflict = raw();
  conflict.entities.Q910000101.claims.P31.push(item("P31", "Q910001002"));
  conflict.entities.Q910000101.claims.P136.push(item("P136", "Q910002999"));
  first(conflict, "P1113").qualifiers = { P585: [] };
  first(conflict, "P2047").mainsnak = { snaktype: "somevalue", property: "P2047" };
  const result = map(conflict);
  assert.equal(result.snapshot?.anime[0].mediaFormat, null);
  assert.equal(result.snapshot?.anime[0].genres, null);
  assert.equal(result.snapshot?.anime[0].episodeCount, null);
  assert.equal(result.snapshot?.anime[0].runtimeMinutes, null);
  assert.equal(result.report.fieldIssues.mediaFormat.conflict, 1);
  assert.equal(result.report.fieldIssues.genres.unmapped, 2);
  assert.equal(result.report.fieldIssues.episodeCount.qualified, 1);
  assert.equal(result.report.fieldIssues.runtimeMinutes.unknown, 1);
  const absent = raw();
  delete absent.entities.Q910000101.aliases;
  assert.equal(map(absent).snapshot?.anime[0].aliases, null);
  const mixed = raw();
  const invalid = item("P136", "Q910002001");
  invalid.mainsnak.datavalue.type = "invalid";
  mixed.entities.Q910000101.claims.P136 = [item("P136", "Q910002999"), invalid];
  const before = map(mixed).report.fieldIssues;
  mixed.entities.Q910000101.claims.P136.reverse();
  assert.deepEqual(map(mixed).report.fieldIssues, before);
  assert.equal(before.genres.invalid, 1);
});

test("release year, duration units, and quantity scope never use fetch time, rounding, or an implicit unit", () => {
  const mutations = [
    ["year", "P577", (s: any) => { s.mainsnak.datavalue.value.precision = 8; }],
    ["year", "P577", (s: any) => { s.mainsnak.datavalue.value.calendarmodel = "invented-unknown-calendar"; }],
    ["year", "P577", (s: any) => { Object.assign(s.mainsnak.datavalue.value, { time: "+2021-02-30T00:00:00Z", precision: 11 }); }],
    ["episodeCount", "P1113", (s: any) => { s.mainsnak.datavalue.value.amount = "+12.5"; }],
    ["episodeCount", "P1113", (s: any) => { s.mainsnak.datavalue.value.unit = "http://www.wikidata.org/entity/Q910003001"; }],
    ["runtimeMinutes", "P2047", (s: any) => { s.mainsnak.datavalue.value.unit = "1"; }],
    ["runtimeMinutes", "P2047", (s: any) => { s.mainsnak.datavalue.value.lowerBound = "+23"; s.mainsnak.datavalue.value.upperBound = "+25"; }],
  ] as const;
  for (const [field, property, mutate] of mutations) {
    const value = raw(); mutate(first(value, property));
    assert.equal(map(value).snapshot?.anime[0][field], null, `${field}/${property}`);
  }
  const conflict = raw();
  const second = structuredClone(first(conflict, "P577"));
  second.mainsnak.datavalue.value.time = "+2022-00-00T00:00:00Z";
  conflict.entities.Q910000101.claims.P577.push(second);
  assert.equal(map(conflict).report.fieldIssues.year.conflict, 1);
  second.mainsnak.datavalue.value.time = "+2021-00-00T00:00:00Z";
  assert.equal(map(conflict).snapshot?.anime[0].year, 2021);
});

test("relations are directed and bounded to unambiguous mapped items; classification needs the exact reviewed table", () => {
  const oneSided = raw();
  delete oneSided.entities.Q910000102.claims.P155;
  assert.deepEqual(map(oneSided).snapshot?.anime[0].relations, [{ kind: "sequel", animeId: 102, title: "Moonlit Workshop" }]);
  assert.equal(map(oneSided).snapshot?.anime[1].relations, null);
  assert.equal(map(oneSided).report.oneSidedDirectedEdges, 1);
  oneSided.entities.Q910000101.claims.P156.push(item("P156", "Q910000103"));
  assert.equal(map(oneSided).snapshot?.anime[0].relations, null);
  assert.equal(map(oneSided).report.fieldIssues.relations.conflict, 1);
  const mapping = structuredClone(policy);
  mapping.classification!.values = {};
  assert.equal(map(raw(), mapping).snapshot?.anime[1].contentClassification, null);
  assert.equal(map(raw(), mapping).report.fieldIssues.contentClassification.unmapped, 1);
  const unsafe = raw();
  unsafe.entities.Q910000102.labels.en.value = "<img src=x onerror=alert(1)>";
  const result = map(unsafe);
  assert.equal(result.snapshot?.anime[0].relations?.[0].title, "<img src=x onerror=alert(1)>");
  assert.equal(JSON.stringify(result.report).includes("onerror"), false);
});

test("malformed or over-budget source/policy fails at a field without echoing source values", () => {
  const malformed = raw();
  first(malformed, "P4086").rank = "invented-private-token";
  assert.throws(() => map(malformed), (error: any) => /claims.P4086\[0\]/.test(error.message) && !error.message.includes("private-token"));
  assert.throws(() => mapWikibaseMetadata(Buffer.from("invented-private-json"), universe, policy, source), /source: must be UTF-8 JSON/);
  assert.throws(() => mapWikibaseMetadata(Buffer.from('{"entities":{},"entities":{}}'), universe, policy, source), /duplicate JSON object key/);
  assert.throws(() => mapWikibaseMetadata(Buffer.from('{"entities":{},"entit\\u0069es":{}}'), universe, policy, source), /duplicate JSON object key/);
  assert.throws(() => mapWikibaseMetadata(Buffer.alloc(WIKIBASE_MAPPING_LIMITS.sourceBytes + 1), universe, policy, source), /byte limit/);
  assert.throws(() => mapWikibaseMetadata(fixtureBytes, [101, 101], policy, source), /universe/);
  assert.throws(() => mapWikibaseMetadata(fixtureBytes, Array.from({ length: 101 }, (_, i) => i + 1), policy, source), /universe/);
  const wrong = structuredClone(policy) as any;
  wrong.images = [];
  assert.throws(() => map(raw(), wrong), /policy.*unsupported/);
  const longKey = structuredClone(policy);
  longKey.genreLabels[`Q${"1".repeat(100)}`] = "Invented genre";
  assert.throws(() => map(raw(), longKey), /policy.genreLabels.*key/);
  assert.throws(() => mapWikibaseMetadata(fixtureBytes, universe, policy, { ...source, snapshotAt: "2026-02-30T00:00:00.000Z" }), /source.snapshotAt/);
  const bounded = raw();
  bounded.entities.Q910000101.claims.P31 = Array.from({ length: 101 }, () => item("P31", "Q910001001"));
  assert.throws(() => map(bounded), /claims.P31.*bounded array/);
});
