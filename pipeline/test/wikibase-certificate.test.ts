import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { parseCatalogMetadataSnapshot } from "../../web/src/artifacts.js";
import { mapWikibaseMetadata, type WikibaseMappingPolicy } from "../src/wikibase-metadata.js";

const fixture = (name: string) => readFileSync(new URL(`../../fixtures/${name}`, import.meta.url));
const legacyPolicy = JSON.parse(fixture("synthetic-wikibase-policy.json").toString()) as WikibaseMappingPolicy;
const policy = () => JSON.parse(fixture("synthetic-wikibase-certificate-policy.json").toString()) as WikibaseMappingPolicy;
const certificateValue = "invented-certificate-<not-public>";
const reference = (value = certificateValue): any => ({ snaktype: "value", property: "P2676", datatype: "string",
  datavalue: { type: "string", value } });
const source = { name: "invented-certificate-fixture", snapshotAt: "2026-10-07T00:00:00.000Z" };
const raw = (): any => {
  const value = JSON.parse(fixture("synthetic-wikibase-entities.json").toString());
  value.entities.Q910000102.claims.P2756[0].qualifiers = { P2676: [reference()] };
  return value;
};
const film = (value: any) => value.entities.Q910000102;
const statement = (value: any) => film(value).claims.P2756[0];
const map = (value = raw(), mapping = policy()) => mapWikibaseMetadata(Buffer.from(JSON.stringify(value)), [101, 102, 103, 104], mapping, source);
const classification = (value: ReturnType<typeof map>) => value.snapshot!.anime.find((item) => item.animeId === 102)!.contentClassification;

test("v2 accepts the exact invented film certificate while v1 keeps qualified classification unknown", () => {
  const before = map(raw(), legacyPolicy), after = map();
  assert.equal(classification(before), null);
  assert.deepEqual(classification(after), { jurisdiction: "Fixtureland", system: "Invented board", value: "All" });
  assert.equal(before.report.format, "wikibase-metadata-audit-v1");
  assert.equal(after.report.format, "wikibase-metadata-audit-v2");
  assert.equal(after.report.classificationCertificate!.acceptedItems, 1);
  assert.equal(after.report.classificationCertificate!.evaluatedItems, 1);
  assert.equal(after.report.sourceSnapshotSha256, before.report.sourceSnapshotSha256);
  assert.notEqual(after.report.policySha256, before.report.policySha256);
  assert.equal(JSON.stringify(after.snapshot).includes(certificateValue), false);
  assert.equal(JSON.stringify(after.report).includes(certificateValue), false);
  assert.equal(parseCatalogMetadataSnapshot(after.snapshot, "invented certificate catalog"), after.snapshot);
  const baseline = JSON.parse(fixture("synthetic-wikibase-entities.json").toString());
  assert.deepEqual(classification(map(baseline, legacyPolicy)), { jurisdiction: "Fixtureland", system: "Invented board", value: "All" });
  assert.equal(classification(map(baseline)), null); // v2 explicitly requires the reference.
});

test("missing, unknown, malformed, extra, and multiple certificate qualifiers remain unknown", () => {
  const mutations: [(s: any) => void, string][] = [
    [(s) => { delete s.qualifiers; }, "missingCertificate"],
    [(s) => { s.qualifiers.P2676 = []; }, "missingCertificate"],
    [(s) => { s.qualifiers.P2676.push(reference()); }, "multipleCertificates"],
    [(s) => { s.qualifiers.P518 = []; }, "unsupportedQualifier"],
    [(s) => { s.qualifiers.P2676[0].snaktype = "somevalue"; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].snaktype = "novalue"; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].datatype = "external-id"; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].property = "P518"; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].datavalue.type = "quantity"; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].datavalue.value = 12; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].datavalue.value = " "; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].datavalue.value = "x".repeat(81); }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].datavalue.value = "invented\nvalue"; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].private = "invented-hidden"; }, "invalidCertificate"],
    [(s) => { s.qualifiers.P2676[0].datavalue.extra = true; }, "invalidCertificate"],
  ];
  for (const [mutate, reason] of mutations) {
    const value = raw(); mutate(statement(value));
    const result = map(value);
    assert.equal(classification(result), null, reason);
    assert.equal(result.report.classificationCertificate!.rejections[reason], 1);
    assert.equal(JSON.stringify(result.report).includes("invented-hidden"), false);
  }
});

test("non-film, unknown, conflicting or qualified formats cannot borrow film classification", () => {
  const mutations = [
    (v: any) => { film(v).claims.P31[0].mainsnak.datavalue.value.id = "Q910001001"; },
    (v: any) => { delete film(v).claims.P31; },
    (v: any) => { film(v).claims.P31[0].qualifiers = { P518: [] }; },
    (v: any) => { const extra = structuredClone(film(v).claims.P31[0]); extra.mainsnak.datavalue.value.id = "Q910001001"; film(v).claims.P31.push(extra); },
    (v: any) => { film(v).claims.P31[0].mainsnak.datavalue.value.id = "Q910009999"; },
  ];
  for (const mutate of mutations) {
    const value = raw(); mutate(value);
    const result = map(value);
    assert.equal(classification(result), null);
    assert.equal(result.report.classificationCertificate!.rejections.nonFilmOrUnknownFormat, 1);
  }
});

test("rating conflicts, unmapped values, and different certificates cannot be merged by order", () => {
  for (const change of ["rating", "certificate", "unmapped"] as const) {
    const value = raw(), second = structuredClone(statement(value));
    if (change === "certificate") second.qualifiers.P2676[0].datavalue.value = "other-invented-certificate";
    else second.mainsnak.datavalue.value.id = change === "rating" ? "Q910004002" : "Q910009999";
    film(value).claims.P2756.push(second);
    const before = map(value);
    assert.equal(classification(before), null);
    film(value).claims.P2756.reverse();
    const after = map(value);
    assert.deepEqual(after.report.classificationCertificate, before.report.classificationCertificate);
    assert.deepEqual(after.report.fieldIssues, before.report.fieldIssues);
  }
  const duplicate = raw(); film(duplicate).claims.P2756.push(structuredClone(statement(duplicate)));
  assert.ok(classification(map(duplicate))); // Same evidence duplicated, no distinct certificate.
  const competing = raw(); const extra = structuredClone(statement(competing));
  extra.mainsnak.datavalue.value.id = "Q910004002"; extra.rank = "deprecated"; film(competing).claims.P2756.push(extra);
  assert.equal(classification(map(competing))!.value, "All");
});

test("a valid reference does not exempt qualifiers on genres, identity, dates or relationships", () => {
  for (const property of ["P136", "P577", "P155", "P4086"]) {
    const value = raw(); film(value).claims[property][0].qualifiers = { P2676: [reference()] };
    const result = map(value);
    if (property === "P4086") assert.equal(result.snapshot!.anime.some((item) => item.animeId === 102), false);
    else {
      const entry = result.snapshot!.anime.find((item) => item.animeId === 102)!;
      assert.equal(entry[property === "P136" ? "genres" : property === "P577" ? "year" : "relations"], null);
    }
  }
});

test("v1 rejects v2 rule fields; v2 rejects broad or malformed rule policies without echoing values", () => {
  const changes = [
    (p: any) => { p.format = "wikibase-metadata-policy-v1"; },
    (p: any) => { p.classification.certificateReference.property = "P518"; },
    (p: any) => { p.classification.certificateReference.datatype = "external-id"; },
    (p: any) => { p.classification.certificateReference.maximumLength = 1000; },
    (p: any) => { p.classification.allowedMediaFormats = ["Movie", "TV"]; },
    (p: any) => { p.classification.property = "P9999"; },
    (p: any) => { p.classification.datatype = "string"; },
    (p: any) => { p.classification.certificateReference.ignore = "invented-private"; },
  ];
  for (const change of changes) {
    const mapping = policy(); change(mapping);
    assert.throws(() => map(raw(), mapping), (error: any) => /policy.classification/.test(error.message) && !error.message.includes("invented-private"));
  }
});

test("only selected best-rank classification statements count and all preferred conflicts remain unknown", () => {
  const value = raw(), competing = structuredClone(statement(value));
  statement(value).rank = "preferred";
  competing.mainsnak.datavalue.value.id = "Q910004002";
  competing.qualifiers.P518 = []; // Normal-rank evidence is not selected.
  film(value).claims.P2756.push(competing);
  assert.equal(classification(map(value))!.value, "All");
  competing.rank = "preferred";
  assert.equal(classification(map(value)), null);
  delete competing.qualifiers.P518;
  const result = map(value);
  assert.equal(classification(result), null);
  assert.equal(result.report.fieldIssues.contentClassification.conflict, 1);
  assert.equal(result.report.classificationCertificate!.rejections.ratingRejected, 1);
});

test("certificate snak hashes, whitespace, and shape are validated without treating them as a rating", () => {
  const valid = raw(); statement(valid).qualifiers.P2676[0].hash = "a".repeat(40);
  assert.equal(classification(map(valid))!.value, "All");
  for (const mutate of [
    (s: any) => { s.qualifiers.P2676 = {}; },
    (s: any) => { s.qualifiers.P2676 = [null]; },
    (s: any) => { s.qualifiers.P2676[0].hash = "invalid"; },
    (s: any) => { s.qualifiers.P2676[0].datavalue = []; },
    (s: any) => { s.qualifiers.P2676[0].datavalue.value = " leading"; },
    (s: any) => { s.qualifiers.P2676[0].datavalue.value = "trailing "; },
    (s: any) => { s.qualifiers.P2676[0].datavalue.value = "invented\u007fvalue"; },
  ]) {
    const value = raw(); mutate(statement(value));
    const result = map(value);
    assert.equal(classification(result), null);
    assert.equal(result.report.classificationCertificate!.rejections.invalidCertificate, 1);
  }
  for (const type of ["somevalue", "novalue"]) {
    const value = raw(); statement(value).mainsnak.snaktype = type;
    const result = map(value);
    assert.equal(classification(result), null);
    assert.equal(result.report.fieldIssues.contentClassification.unknown, 1);
    assert.equal(result.report.classificationCertificate!.rejections.ratingRejected, 1);
  }
});

test("mixed certificate refusals use stable precedence and disabled classification emits no audit activity", () => {
  const value = raw(), missing = structuredClone(statement(value)), invalid = structuredClone(statement(value));
  delete missing.qualifiers;
  invalid.qualifiers.P2676[0].snaktype = "novalue";
  statement(value).qualifiers.P518 = [];
  film(value).claims.P2756.push(missing, invalid);
  const before = map(value);
  film(value).claims.P2756.reverse();
  const after = map(value);
  assert.deepEqual(after.report.classificationCertificate, before.report.classificationCertificate);
  assert.deepEqual(after.report.fieldIssues, before.report.fieldIssues);
  assert.equal(after.report.classificationCertificate!.rejections.unsupportedQualifier, 1);
  assert.equal(after.report.classificationCertificate!.rejections.missingCertificate, 0);
  const mapping = policy(); mapping.classification = null;
  const disabled = map(value, mapping);
  assert.equal(classification(disabled), null);
  assert.equal(disabled.report.classificationCertificate!.evaluatedItems, 0);
  assert.equal(disabled.report.classificationCertificate!.acceptedItems, 0);
  assert.equal(Object.values(disabled.report.classificationCertificate!.rejections).reduce((sum, n) => sum + n), 0);
  assert.equal(Object.hasOwn(map(raw(), legacyPolicy).report, "classificationCertificate"), false);
});
