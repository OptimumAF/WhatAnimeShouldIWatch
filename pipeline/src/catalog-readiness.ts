/** Local, bounded source-design audit. No transport, file writes, release gate, or approval. */
import { createHash } from "node:crypto";
import { catalogMetadataCoverage, catalogJointMetadataCoverage, parseCatalogMetadataSnapshot,
  type CatalogCoverageField, type CatalogMetadataItemV1 } from "../../web/src/artifacts.js";
import { parseWikibaseJson } from "./wikibase-metadata.js";

type MediaFormat = NonNullable<CatalogMetadataItemV1["mediaFormat"]>;
type RecordValue = Record<string, unknown>;
interface Profile {
  key: string; animeIds: number[]; fields: CatalogCoverageField[];
  minimumUsableBasisPoints: number; expectedMediaFormat: MediaFormat | null;
  classificationScope: { jurisdiction: string; system: string } | null;
}
const supportedFields: CatalogCoverageField[] = ["aliases", "genres", "year", "mediaFormat",
  "episodeCount", "runtimeMinutes", "contentClassification", "communityScore", "relations"];
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
function fail(field: string, reason: string): never { throw new Error(`Catalog readiness ${field}: ${reason}.`); }
function record(value: unknown, field: string, keys: string[]): RecordValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "requires an object");
  const result = value as RecordValue;
  if (Object.keys(result).length !== keys.length || keys.some((key) => !Object.hasOwn(result, key))) {
    fail(field, "has unsupported or missing fields");
  }
  return result;
}
function ids(value: unknown, field: string): number[] {
  if (!Array.isArray(value) || !value.length || value.length > 100 || value.some((id, index) =>
    !Number.isSafeInteger(id) || id < 1 || index > 0 && id <= value[index - 1])) {
    fail(field, "requires one to 100 sorted unique positive IDs");
  }
  return value as number[];
}
function text(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim() || value.length > 80) fail(field, "requires bounded text");
  return value;
}
function json(bytes: Uint8Array, label: string): unknown {
  try { return parseWikibaseJson(bytes); }
  catch { fail(label, "requires bounded valid UTF-8 JSON without duplicate keys"); }
}
function plan(bytes: Uint8Array) {
  const root = record(json(bytes, "plan"), "plan", ["format", "purpose", "universeIds", "profiles"]);
  if (root.format !== "catalog-readiness-plan-v1" || root.purpose !== "local-source-feasibility-only") {
    fail("plan", "requires the local feasibility contract");
  }
  const universeIds = ids(root.universeIds, "plan.universeIds");
  if (!Array.isArray(root.profiles) || !root.profiles.length || root.profiles.length > 20) fail("plan.profiles", "requires one to 20 profiles");
  const keys = new Set<string>();
  const profiles: Profile[] = root.profiles.map((value, index) => {
    const field = `plan.profiles[${index}]`;
    const raw = record(value, field, ["key", "animeIds", "fields", "minimumUsableBasisPoints", "expectedMediaFormat", "classificationScope"]);
    const key = text(raw.key, `${field}.key`);
    if (!/^[a-z][a-z0-9-]{0,39}$/.test(key) || keys.has(key)) fail(`${field}.key`, "requires a unique fixed code");
    keys.add(key);
    const animeIds = ids(raw.animeIds, `${field}.animeIds`);
    if (animeIds.some((id) => !universeIds.includes(id))) fail(`${field}.animeIds`, "contains an ID outside the declared universe");
    if (!Array.isArray(raw.fields) || !raw.fields.length || new Set(raw.fields).size !== raw.fields.length ||
        raw.fields.some((entry) => !supportedFields.includes(entry as CatalogCoverageField))) fail(`${field}.fields`, "requires unique supported fields");
    const fields = raw.fields as CatalogCoverageField[];
    const minimum = raw.minimumUsableBasisPoints;
    if (!Number.isSafeInteger(minimum) || Number(minimum) < 1 || Number(minimum) > 10000) fail(`${field}.minimumUsableBasisPoints`, "requires an integer from one through 10000");
    if ((fields.includes("contentClassification") || fields.includes("episodeCount") && fields.includes("runtimeMinutes")) && Number(minimum) < 9000) {
      fail(`${field}.minimumUsableBasisPoints`, "detail and classification checks require at least 9000 basis points");
    }
    const format = raw.expectedMediaFormat;
    if (format !== null && (!["TV", "Movie", "OVA", "ONA", "Special"].includes(format as string) || !fields.includes("mediaFormat"))) {
      fail(`${field}.expectedMediaFormat`, "requires a supported format and its field check");
    }
    let classificationScope: Profile["classificationScope"] = null;
    if (raw.classificationScope !== null) {
      const scope = record(raw.classificationScope, `${field}.classificationScope`, ["jurisdiction", "system"]);
      if (!fields.includes("contentClassification")) fail(`${field}.classificationScope`, "requires the classification field check");
      classificationScope = { jurisdiction: text(scope.jurisdiction, `${field}.classificationScope.jurisdiction`),
        system: text(scope.system, `${field}.classificationScope.system`) };
    } else if (fields.includes("contentClassification")) fail(`${field}.classificationScope`, "requires an explicit jurisdiction and system");
    return { key, animeIds, fields, minimumUsableBasisPoints: Number(minimum), expectedMediaFormat: format as MediaFormat | null, classificationScope };
  });
  const core = profiles.find((profile) => profile.key === "core");
  if (!core || JSON.stringify(core.animeIds) !== JSON.stringify(universeIds) ||
      core.fields.length !== 3 || ["genres", "year", "mediaFormat"].some((field) => !core.fields.includes(field as CatalogCoverageField)) ||
      core.minimumUsableBasisPoints < 9500 || core.expectedMediaFormat !== null || core.classificationScope !== null) {
    fail("plan.profiles.core", "requires the full-universe genre year format check at at least 9500 basis points");
  }
  return { universeIds, profiles };
}

/** Hashes actual metadata/plan bytes; the declared source digest does not prove source identity. */
export function auditCatalogReadiness(metadataBytes: Uint8Array, planBytes: Uint8Array) {
  const { universeIds, profiles } = plan(planBytes);
  const snapshot = parseCatalogMetadataSnapshot(json(metadataBytes, "catalog.metadata.json"), "catalog.metadata.json");
  if (snapshot.anime.some((item) => !universeIds.includes(item.animeId))) fail("catalog.metadata.json.anime", "contains an ID outside the declared universe");
  const coverage = catalogMetadataCoverage(snapshot, universeIds);
  const byId = new Map(snapshot.anime.map((item) => [item.animeId, item]));
  const checks = profiles.map((profile) => {
    const together = catalogJointMetadataCoverage(snapshot, profile.animeIds, profile.fields);
    let formatMismatchItems = 0, classificationMismatchItems = 0;
    for (const id of profile.animeIds) {
      const item = byId.get(id);
      if (profile.expectedMediaFormat && item?.mediaFormat && item.mediaFormat !== profile.expectedMediaFormat) formatMismatchItems += 1;
      const rating = item?.contentClassification, scope = profile.classificationScope;
      if (scope && rating && (scope.jurisdiction !== rating.jurisdiction || scope.system !== rating.system)) classificationMismatchItems += 1;
    }
    const requiredUsable = Math.ceil(together.total * profile.minimumUsableBasisPoints / 10000);
    return { key: profile.key, fields: profile.fields, ...together, requiredUsable,
      minimumUsableBasisPoints: profile.minimumUsableBasisPoints, formatMismatchItems, classificationMismatchItems,
      targetMet: together.usableTogether >= requiredUsable && formatMismatchItems === 0 && classificationMismatchItems === 0 };
  });
  return { format: "catalog-readiness-audit-v1" as const, metadataSha256: hash(metadataBytes), planSha256: hash(planBytes),
    declaredSourceSnapshotSha256: snapshot.source.snapshotSha256,
    identityComplete: coverage.missingItems === 0, coverage, checks,
    declaredTargetsMet: coverage.missingItems === 0 && checks.every((check) => check.targetMet),
    publicationAuthorized: false as const };
}
