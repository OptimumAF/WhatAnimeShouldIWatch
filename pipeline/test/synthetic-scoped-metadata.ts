/** Invented test inputs only; never used by acquisition or browser runtime. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { mapWikibaseMetadata, type WikibaseMappingPolicy } from "../src/wikibase-metadata.js";

const fixture = (name: string) => JSON.parse(readFileSync(new URL(`../../fixtures/${name}`, import.meta.url), "utf8"));
export const syntheticWikibaseSource = () => fixture("synthetic-wikibase-entities.json");
export function scopedMetadataFixture(raw = syntheticWikibaseSource(), animeIds = [101], version: "v1" | "v2" = "v1") {
  const sourceBytes = Buffer.from(JSON.stringify(raw)), universe = [101, 102, 103, 104];
  const mappingPolicy = fixture(version === "v1" ? "synthetic-wikibase-policy.json" : "synthetic-wikibase-certificate-policy.json") as WikibaseMappingPolicy;
  mappingPolicy.mediaFormats.Q910001003 = "TV";
  const source = { name: "invented-scoped-fixture", snapshotAt: "2026-10-07T00:00:00.000Z" };
  const datePolicyBytes = Buffer.from(JSON.stringify(fixture("synthetic-tv-date-policy.json")));
  const mapped = mapWikibaseMetadata(sourceBytes, universe, mappingPolicy, source);
  const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
  const scopeBytes = Buffer.from(JSON.stringify({ format: "declared-tv-first-airing-scope-v1",
    sourceSnapshotSha256: hash(sourceBytes), mappingPolicySha256: mapped.report.policySha256,
    datePolicySha256: hash(datePolicyBytes), universeSha256: mapped.report.universeSha256,
    items: animeIds.map((animeId) => ({ animeId, sourceItemId: `Q910000${animeId}`,
      kind: "series", dateScope: "whole-work-first-airing" })) }));
  return { format: "synthetic-scoped-metadata-candidate-v1" as const, purpose: "fixture-only" as const,
    sourceBytes, universe, mappingPolicy, datePolicyBytes, scopeBytes, source };
}
