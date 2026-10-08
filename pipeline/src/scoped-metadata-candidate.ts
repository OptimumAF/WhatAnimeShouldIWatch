/** Explicit invented integration only. No transport, writes, installer or publication entry point. */
import { createHash } from "node:crypto";
import { catalogMetadataCoverage, parseCatalogMetadataSnapshot } from "../../web/src/artifacts.js";
import { auditTvDateCandidate } from "./tv-date-candidate.js";
import { mapWikibaseMetadata, type WikibaseMappingPolicy } from "./wikibase-metadata.js";

export interface SyntheticScopedMetadataInput {
  format: "synthetic-scoped-metadata-candidate-v1";
  purpose: "fixture-only";
  sourceBytes: Uint8Array; universe: readonly number[]; mappingPolicy: WikibaseMappingPolicy;
  datePolicyBytes: Uint8Array; scopeBytes: Uint8Array; source: { name: string; snapshotAt: string };
}
/** Fixture markers prevent accidental default use; they do not prove invention, semantic meaning or rights. */
export function mapSyntheticScopedMetadata(input: SyntheticScopedMetadataInput) {
  const keys = ["format", "purpose", "sourceBytes", "universe", "mappingPolicy", "datePolicyBytes", "scopeBytes", "source"];
  if (!input || typeof input !== "object" || Array.isArray(input) || Object.keys(input).length !== keys.length ||
      keys.some((key) => !Object.hasOwn(input, key)) || input.format !== "synthetic-scoped-metadata-candidate-v1" || input.purpose !== "fixture-only") {
    throw new Error("Synthetic scoped metadata input: requires the exact fixture-only contract.");
  }
  if (!input.source || typeof input.source.name !== "string" || !/^(invented|synthetic)-/.test(input.source.name)) {
    throw new Error("Synthetic scoped metadata source.name: requires an invented fixture identity.");
  }
  const base = mapWikibaseMetadata(input.sourceBytes, input.universe, input.mappingPolicy, input.source);
  const dateAudit = auditTvDateCandidate(input);
  const dates = new Map(dateAudit.rows.map((row) => [row.animeId, row]));
  const yearBasis = { scopedTv: 0, legacyPublication: 0, unknown: 0 };
  const candidate = base.snapshot && { ...base.snapshot, anime: base.snapshot.anime.map((item) => {
    const year = item.mediaFormat === "TV" ? dates.get(item.animeId)!.year : item.mediaFormat === null ? null : item.year;
    yearBasis[year === null ? "unknown" : item.mediaFormat === "TV" ? "scopedTv" : "legacyPublication"] += 1;
    return { ...item, year };
  }) };
  const snapshot = candidate ? parseCatalogMetadataSnapshot(candidate, "synthetic scoped catalog.metadata.json") : null;
  const metadataBytes = snapshot ? Buffer.from(`${JSON.stringify(snapshot, null, 2)}\n`) : null;
  // An empty source result still uses the fixed universe for coverage; it is not a catalog export.
  const coverage = catalogMetadataCoverage(snapshot ?? { format: "anime-metadata-catalog-v1",
    source: { ...input.source, snapshotSha256: base.report.sourceSnapshotSha256 }, anime: [] }, input.universe);
  const privateAudit = { format: "private-scoped-metadata-audit-v1" as const, purpose: "fixture-only" as const,
    metadataSha256: metadataBytes ? createHash("sha256").update(metadataBytes).digest("hex") : null,
    baseMappingAudit: base.report, dateAudit, yearBasis, coverage, publicationAuthorized: false as const };
  return { snapshot, metadataBytes, privateAudit };
}
