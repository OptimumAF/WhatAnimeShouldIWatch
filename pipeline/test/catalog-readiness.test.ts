import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { auditCatalogReadiness } from "../src/catalog-readiness.js";
import { mapWikibaseMetadata, type WikibaseMappingPolicy } from "../src/wikibase-metadata.js";

const fixture = (name: string) => readFileSync(new URL(`../../fixtures/${name}`, import.meta.url));
const planBytes = fixture("synthetic-catalog-readiness-plan.json");
const plan = () => JSON.parse(planBytes.toString());
const policy = JSON.parse(fixture("synthetic-wikibase-policy.json").toString()) as WikibaseMappingPolicy;
const source = { name: "invented-readiness-fixture", snapshotAt: "2026-10-07T00:00:00.000Z" };
const mapped = () => mapWikibaseMetadata(fixture("synthetic-wikibase-entities.json"), [101, 102, 103, 104], policy, source).snapshot!;
const bytes = (value: unknown) => Buffer.from(JSON.stringify(value));
const audit = (snapshot = complete(), rules = plan()) => auditCatalogReadiness(bytes(snapshot), bytes(rules));
function complete() {
  const snapshot = structuredClone(mapped());
  const third = snapshot.anime[2];
  Object.assign(third, { genres: ["Adventure"], year: 2023, episodeCount: 13, runtimeMinutes: 25 });
  snapshot.anime.push({ ...structuredClone(snapshot.anime[0]), animeId: 104, sourceItemId: "Q910000104",
    title: "Invented fourth work", aliases: [], relations: null });
  return snapshot;
}

test("the existing invented mapping fails the declared denominators and a complete invented candidate meets them", () => {
  const partial = audit(mapped());
  assert.equal(partial.identityComplete, false);
  assert.equal(partial.declaredTargetsMet, false);
  assert.deepEqual(partial.checks.map((check) => [check.key, check.total, check.usableTogether, check.requiredUsable]),
    [["core", 4, 2, 4], ["tv-details", 3, 1, 3], ["movie-classification", 1, 1, 1]]);
  assert.equal(partial.checks[0].missingItems, 1);
  const full = audit();
  assert.equal(full.identityComplete, true);
  assert.equal(full.declaredTargetsMet, true);
  assert.equal(full.publicationAuthorized, false);
  assert.equal(JSON.stringify(full).includes("Copper"), false);
  assert.equal(JSON.stringify(full).includes("Q910"), false);
  assert.equal(JSON.stringify(full).includes("Fixtureland"), false);
});

test("95 percent marginal field coverage cannot conceal 90 percent joint coverage", () => {
  const snapshot = complete(), template = structuredClone(snapshot.anime[0]);
  snapshot.anime = Array.from({ length: 100 }, (_, index) => ({ ...structuredClone(template),
    animeId: index + 1, sourceItemId: `Q${910020000 + index}`, title: `Invented ${index + 1}`, aliases: [], relations: null,
    genres: index < 5 ? null : ["Adventure"], year: index >= 5 && index < 10 ? null : 2021 }));
  const rules = plan(); rules.universeIds = snapshot.anime.map((item) => item.animeId);
  rules.profiles = [{ ...rules.profiles[0], animeIds: rules.universeIds }];
  let result = audit(snapshot, rules);
  assert.equal(result.coverage.usable.genres, 95);
  assert.equal(result.coverage.usable.year, 95);
  assert.equal(result.checks[0].usableTogether, 90);
  assert.equal(result.checks[0].requiredUsable, 95);
  assert.equal(result.declaredTargetsMet, false);
  snapshot.anime.slice(5, 10).forEach((item) => { item.year = 2021; });
  result = audit(snapshot, rules);
  assert.equal(result.checks[0].usableTogether, 95);
  assert.equal(result.declaredTargetsMet, true); // Exact threshold, integer arithmetic.
  snapshot.anime[5].genres = [];
  result = audit(snapshot, rules);
  assert.equal(result.checks[0].knownTogether, 95);
  assert.equal(result.checks[0].usableTogether, 94);
  assert.equal(result.declaredTargetsMet, false);
});

test("wrong format or classification scope fails even when all requested fields are present", () => {
  const snapshot = complete(); snapshot.anime[2].mediaFormat = "Movie";
  let result = audit(snapshot);
  assert.equal(result.checks[1].usableTogether, 3);
  assert.equal(result.checks[1].formatMismatchItems, 1);
  assert.equal(result.declaredTargetsMet, false);
  snapshot.anime[2].mediaFormat = "TV";
  snapshot.anime[1].contentClassification!.jurisdiction = "Other invented territory";
  result = audit(snapshot);
  assert.equal(result.checks[2].classificationMismatchItems, 1);
  assert.equal(result.declaredTargetsMet, false);
});

test("core membership and floors cannot be narrowed, and invalid plan fields fail without payload values", () => {
  const mutations: [(value: any) => void, RegExp][] = [
    [(v) => { v.profiles[0].animeIds.pop(); }, /profiles.core/],
    [(v) => { v.profiles[0].minimumUsableBasisPoints = 9499; }, /profiles.core/],
    [(v) => { v.profiles[1].minimumUsableBasisPoints = 8999; }, /9000 basis points/],
    [(v) => { v.profiles[2].minimumUsableBasisPoints = 8999; }, /9000 basis points/],
    [(v) => { v.universeIds.push(104); }, /universeIds/],
    [(v) => { v.profiles[0].animeIds.push(999); }, /animeIds/],
    [(v) => { v.profiles[0].fields.push("invented-private-field"); }, /fields/],
    [(v) => { v.profiles[1].fields.push("year", "year"); }, /fields/],
    [(v) => { v.profiles[2].classificationScope = null; }, /classificationScope/],
    [(v) => { v.profiles[1].fields = ["runtimeMinutes"]; }, /expectedMediaFormat/],
    [(v) => { v.profiles[0].credential = "invented-private-value"; }, /unsupported/],
  ];
  for (const [mutate, pattern] of mutations) {
    const rules = plan(); mutate(rules);
    assert.throws(() => audit(complete(), rules), (error: any) => pattern.test(error.message) && !error.message.includes("invented-private"));
  }
  assert.throws(() => auditCatalogReadiness(bytes(complete()), Buffer.from('{"format":1,"format":2}')), /plan.*duplicate keys/);
  assert.throws(() => auditCatalogReadiness(bytes(complete()), Buffer.alloc(4 * 1024 * 1024 + 1)), /bounded/);
});

test("actual bytes and declared source identity stay distinct; hidden or outside-universe metadata is rejected", () => {
  const snapshot = complete(), metadataBytes = bytes(snapshot);
  const before = auditCatalogReadiness(metadataBytes, planBytes);
  assert.equal(before.metadataSha256, createHash("sha256").update(metadataBytes).digest("hex"));
  assert.equal(before.planSha256, createHash("sha256").update(planBytes).digest("hex"));
  assert.equal(before.declaredSourceSnapshotSha256, snapshot.source.snapshotSha256);
  const reformatted = auditCatalogReadiness(Buffer.from(JSON.stringify(snapshot, null, 2)), planBytes);
  assert.notEqual(reformatted.metadataSha256, before.metadataSha256);
  assert.deepEqual(reformatted.checks, before.checks);
  assert.equal(reformatted.publicationAuthorized, false);
  snapshot.anime[3].animeId = 999;
  assert.throws(() => audit(snapshot), /catalog.metadata.json.anime.*outside/);
  const hidden: any = complete(); hidden.anime[0].userRows = ["invented-private-value"];
  assert.throws(() => audit(hidden), (error: any) => /anime\[0\].userRows/.test(error.message) && !error.message.includes("invented-private-value"));
});

test("current mapper keeps proposed start-time and certificate-qualifier rules inactive", () => {
  const raw = JSON.parse(fixture("synthetic-wikibase-entities.json").toString());
  const claims = raw.entities.Q910000101.claims;
  claims.P580 = structuredClone(claims.P577); claims.P580[0].mainsnak.property = "P580";
  delete claims.P577;
  raw.entities.Q910000102.claims.P2756[0].qualifiers = { P2676: [{ snaktype: "value", property: "P2676",
    datatype: "string", datavalue: { type: "string", value: "invented-certificate" } }] };
  const result = mapWikibaseMetadata(bytes(raw), [101, 102, 103, 104], policy, source);
  assert.equal(result.snapshot!.anime[0].year, null);
  assert.equal(result.snapshot!.anime[1].contentClassification, null);
  assert.equal(result.report.fieldIssues.contentClassification.qualified, 1);
});
