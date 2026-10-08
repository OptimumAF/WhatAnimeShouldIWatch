/** Offline, bounded statement mapping. No transport, cache, or publication entry point. */
import { createHash } from "node:crypto";
import {
  catalogMetadataCoverage, parseCatalogMetadataSnapshot,
  type CatalogCoverageField, type CatalogMetadataItemV1, type CatalogMetadataSnapshotV1,
} from "../../web/src/artifacts.js";

type MediaFormat = NonNullable<CatalogMetadataItemV1["mediaFormat"]>;
interface WikibaseMappingBase {
  languageOrder: string[];
  genreLabels: Record<string, string>;
  mediaFormats: Record<string, MediaFormat>;
  /** Exact unit URI to minutes multiplier; no implicit duration unit. */
  durationUnits: Record<string, number>;
}
interface ClassificationMapping {
  property: string; datatype: "wikibase-item" | "string";
  jurisdiction: string; system: string; values: Record<string, string>;
}
export interface WikibaseMappingPolicyV1 extends WikibaseMappingBase {
  format: "wikibase-metadata-policy-v1";
  classification: ClassificationMapping | null;
}
/** Synthetic candidate rule; source-derived use still requires separate scope/mapping review. */
export interface WikibaseMappingPolicyV2 extends WikibaseMappingBase {
  format: "wikibase-metadata-policy-v2";
  classification: (ClassificationMapping & {
    certificateReference: { property: "P2676"; datatype: "string"; maximumLength: 80 };
    allowedMediaFormats: ["Movie"];
  }) | null;
}
export type WikibaseMappingPolicy = WikibaseMappingPolicyV1 | WikibaseMappingPolicyV2;
type ObjectValue = Record<string, unknown>;
type Issue = "missing" | "qualified" | "unknown" | "unmapped" | "invalid" | "conflict";
type Resolution<T> = { value: T; issue?: undefined } | { value: null; issue: Issue };
const fields: CatalogCoverageField[] = ["aliases", "genres", "year", "mediaFormat",
  "episodeCount", "runtimeMinutes", "contentClassification", "communityScore", "relations"];
const issues: Issue[] = ["missing", "qualified", "unknown", "unmapped", "invalid", "conflict"];
const gregorian = "http://www.wikidata.org/entity/Q1985727";
const itemPattern = /^Q[1-9]\d*$/;
const propertyPattern = /^P[1-9]\d*$/;
export const WIKIBASE_MAPPING_LIMITS = { sourceBytes: 4 * 1024 * 1024, universe: 100,
  entities: 100, statementsPerProperty: 100, languages: 100, aliasesPerLanguage: 100 } as const;

function fail(field: string, reason: string): never {
  throw new Error(`Wikibase metadata ${field}: ${reason}.`);
}
function object(value: unknown, field: string): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(field, "must be an object");
  return value as ObjectValue;
}
function array(value: unknown, field: string, maximum: number): unknown[] {
  if (!Array.isArray(value) || value.length > maximum) fail(field, "must be a bounded array");
  return value;
}
function text(value: unknown, field: string, maximum: number): string {
  if (typeof value !== "string" || !value.trim() || value.length > maximum) {
    fail(field, "must be bounded nonempty text");
  }
  return value;
}
function exact(value: ObjectValue, keys: string[], field: string): void {
  if (Object.keys(value).some((key) => !keys.includes(key)) ||
      keys.some((key) => !Object.hasOwn(value, key))) fail(field, "has unsupported or missing fields");
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as ObjectValue;
    return `{${Object.keys(record).sort().map((key) =>
      `${JSON.stringify(key)}:${canonical(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
const digest = (value: Uint8Array | string): string => createHash("sha256").update(value).digest("hex");
function stringMap(value: unknown, field: string, keyPattern: RegExp, maximum: number): ObjectValue {
  const result = object(value, field);
  if (Object.keys(result).length > 1000) fail(field, "has too many entries");
  Object.entries(result).forEach(([key, entry], index) => {
    if (key.length > 80 || !keyPattern.test(key)) fail(`${field}[${index}].key`, "is unsupported");
    text(entry, `${field}[${index}].value`, maximum);
  });
  return result;
}
function validatePolicy(policy: WikibaseMappingPolicy): void {
  const root = object(policy, "policy");
  exact(root, ["format", "languageOrder", "genreLabels", "mediaFormats", "durationUnits", "classification"], "policy");
  if (!["wikibase-metadata-policy-v1", "wikibase-metadata-policy-v2"].includes(root.format as string)) {
    fail("policy.format", "is unsupported");
  }
  const languages = array(root.languageOrder, "policy.languageOrder", 4);
  if (!languages.length || new Set(languages).size !== languages.length || languages.some((entry) =>
    typeof entry !== "string" || entry.length > 40 || !/^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$/.test(entry))) {
    fail("policy.languageOrder", "must contain one to four unique language codes");
  }
  stringMap(root.genreLabels, "policy.genreLabels", itemPattern, 80);
  const formats = stringMap(root.mediaFormats, "policy.mediaFormats", itemPattern, 80);
  if (Object.values(formats).some((value) => !["TV", "Movie", "OVA", "ONA", "Special"].includes(value as string))) {
    fail("policy.mediaFormats", "contains an unsupported format");
  }
  const units = object(root.durationUnits, "policy.durationUnits");
  if (Object.keys(units).length > 20) fail("policy.durationUnits", "has too many entries");
  Object.entries(units).forEach(([key, value], index) => {
    if (key.length > 120 || !/^https?:\/\/www\.wikidata\.org\/entity\/Q[1-9]\d*$/.test(key) ||
        typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 10000) {
      fail(`policy.durationUnits[${index}]`, "requires an exact unit URI and positive minutes multiplier");
    }
  });
  if (root.classification !== null) {
    const rating = object(root.classification, "policy.classification");
    exact(rating, ["property", "datatype", "jurisdiction", "system", "values",
      ...(root.format === "wikibase-metadata-policy-v2" ? ["certificateReference", "allowedMediaFormats"] : [])], "policy.classification");
    if (typeof rating.property !== "string" || rating.property.length > 40 || !propertyPattern.test(rating.property) ||
        ["P4086", "P136", "P31", "P577", "P1113", "P2047", "P155", "P156"].includes(rating.property)) {
      fail("policy.classification.property", "must be a separate property");
    }
    if (!["wikibase-item", "string"].includes(rating.datatype as string)) fail("policy.classification.datatype", "is unsupported");
    text(rating.jurisdiction, "policy.classification.jurisdiction", 80);
    text(rating.system, "policy.classification.system", 80);
    stringMap(rating.values, "policy.classification.values",
      rating.datatype === "wikibase-item" ? itemPattern : /^.{1,80}$/, 80);
    if (root.format === "wikibase-metadata-policy-v2") {
      const reference = object(rating.certificateReference, "policy.classification.certificateReference");
      exact(reference, ["property", "datatype", "maximumLength"], "policy.classification.certificateReference");
      if (rating.property !== "P2756" || rating.datatype !== "wikibase-item" ||
          reference.property !== "P2676" || reference.datatype !== "string" || reference.maximumLength !== 80 ||
          !Array.isArray(rating.allowedMediaFormats) || rating.allowedMediaFormats.length !== 1 ||
          rating.allowedMediaFormats[0] !== "Movie") {
        fail("policy.classification", "requires the exact film certificate candidate rule");
      }
    }
  }
}

interface Statement { mainsnak: ObjectValue; rank: "normal" | "preferred" | "deprecated";
  qualified: boolean; qualifiers: ObjectValue }
interface Entity { id: string; labels: ObjectValue; aliases: ObjectValue | null;
  claims: Map<string, Statement[]> }
/** JSON.parse alone loses duplicate object keys before identity validation can see them. */
function checkJsonKeys(json: string): void {
  const tokens = json.match(/"(?:\\.|[^"\\])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|[{}\[\]:,]/g) ?? [];
  if (tokens.length > 500000) fail("source", "has too many JSON tokens");
  let index = 0;
  const visit = (depth: number): void => {
    if (depth > 64) fail("source", "exceeds the JSON nesting limit");
    const token = tokens[index++];
    if (token === "{") {
      const seen = new Set<string>();
      while (tokens[index] !== "}") {
        const key = JSON.parse(tokens[index++]) as string;
        if (seen.has(key)) fail("source", "contains a duplicate JSON object key");
        seen.add(key);
        index += 1; // colon; JSON.parse has already validated syntax.
        visit(depth + 1);
        if (tokens[index] === ",") index += 1;
      }
      index += 1;
    } else if (token === "[") {
      while (tokens[index] !== "]") {
        visit(depth + 1);
        if (tokens[index] === ",") index += 1;
      }
      index += 1;
    }
  };
  visit(0);
}
/** Shared bounded JSON decoding preserves duplicate-key refusals before projection. */
export function parseWikibaseJson(source: Uint8Array): unknown {
  if (source.byteLength === 0 || source.byteLength > WIKIBASE_MAPPING_LIMITS.sourceBytes) {
    fail("source", "exceeds the bounded byte limit or is empty");
  }
  let parsed: unknown;
  let json: string;
  try { json = new TextDecoder("utf-8", { fatal: true }).decode(source); parsed = JSON.parse(json); }
  catch { fail("source", "must be UTF-8 JSON"); }
  checkJsonKeys(json);
  return parsed;
}
function readEntities(source: Uint8Array, policy: WikibaseMappingPolicy): Entity[] {
  const parsed = parseWikibaseJson(source);
  const root = object(parsed, "source");
  const entities = object(root.entities, "source.entities");
  const entries = Object.entries(entities);
  if (entries.length > WIKIBASE_MAPPING_LIMITS.entities) fail("source.entities", "has too many entities");
  const properties = ["P4086", "P136", "P31", "P577", "P1113", "P2047", "P155", "P156",
    ...(policy.classification ? [policy.classification.property] : [])];
  return entries.map(([key, value], entityIndex) => {
    const where = `source.entities[${entityIndex}]`;
    const entity = object(value, where);
    if (!itemPattern.test(key) || key.length > 40 || entity.id !== key || entity.type !== "item") {
      fail(where, "requires a matching canonical item ID and type");
    }
    const labels = object(entity.labels, `${where}.labels`);
    const aliases = entity.aliases === undefined ? null : object(entity.aliases, `${where}.aliases`);
    for (const [component, terms] of [["labels", labels], ["aliases", aliases]] as const) {
      if (!terms) continue;
      if (Object.keys(terms).length > WIKIBASE_MAPPING_LIMITS.languages) fail(`${where}.${component}`, "has too many languages");
      Object.entries(terms).forEach(([language, raw], index) => {
        const values = component === "aliases"
          ? array(raw, `${where}.${component}[${index}]`, WIKIBASE_MAPPING_LIMITS.aliasesPerLanguage) : [raw];
        values.forEach((term) => {
          const entry = object(term, `${where}.${component}[${index}]`);
          if (entry.language !== language) fail(`${where}.${component}[${index}].language`, "does not match its key");
          text(entry.value, `${where}.${component}[${index}].value`, 4000);
        });
      });
    }
    const rawClaims = object(entity.claims, `${where}.claims`);
    const claims = new Map<string, Statement[]>();
    for (const property of properties) {
      if (!Object.hasOwn(rawClaims, property)) continue;
      const statements = array(rawClaims[property], `${where}.claims.${property}`, WIKIBASE_MAPPING_LIMITS.statementsPerProperty);
      claims.set(property, statements.map((value, index) => {
        const field = `${where}.claims.${property}[${index}]`;
        const statement = object(value, field);
        const snak = object(statement.mainsnak, `${field}.mainsnak`);
        if (statement.type !== "statement" || !["normal", "preferred", "deprecated"].includes(statement.rank as string) ||
            snak.property !== property || !["value", "somevalue", "novalue"].includes(snak.snaktype as string)) {
          fail(field, "has an invalid statement, rank, or main snak");
        }
        const qualifiers = statement.qualifiers === undefined ? {} : object(statement.qualifiers, `${field}.qualifiers`);
        return { mainsnak: snak, rank: statement.rank as Statement["rank"], qualifiers,
          qualified: Object.keys(qualifiers).length > 0 };
      }));
    }
    return { id: key, labels, aliases, claims };
  });
}
function bestStatements(entity: Entity, property: string): Statement[] {
  const live = (entity.claims.get(property) ?? []).filter((entry) => entry.rank !== "deprecated");
  return live.some((entry) => entry.rank === "preferred")
    ? live.filter((entry) => entry.rank === "preferred") : live;
}
function values<T>(statements: Statement[], decode: (snak: ObjectValue) => Resolution<T>): Resolution<T[]> {
  if (!statements.length) return { value: null, issue: "missing" };
  // Qualifier meaning needs a reviewed rule, even when an unqualified value also exists.
  if (statements.some((entry) => entry.qualified)) return { value: null, issue: "qualified" };
  if (statements.some((entry) => entry.mainsnak.snaktype !== "value")) return { value: null, issue: "unknown" };
  const decoded = statements.map((entry) => decode(entry.mainsnak));
  const rejected = (["invalid", "unmapped"] as const).find((issue) => decoded.some((entry) => entry.issue === issue));
  if (rejected) return { value: null, issue: rejected };
  return { value: decoded.map((entry) => entry.value as T) };
}
function scalar<T>(resolution: Resolution<T[]>): Resolution<T> {
  if (resolution.issue) return resolution;
  const unique = new Map(resolution.value.map((entry) => [canonical(entry), entry]));
  return unique.size === 1 ? { value: [...unique.values()][0] } : { value: null, issue: "conflict" };
}
function dataValue(snak: ObjectValue, datatype: string, type: string): Resolution<unknown> {
  if (snak.datatype !== datatype || !snak.datavalue || typeof snak.datavalue !== "object" || Array.isArray(snak.datavalue)) {
    return { value: null, issue: "invalid" };
  }
  const raw = snak.datavalue as ObjectValue;
  return raw.type === type && Object.hasOwn(raw, "value")
    ? { value: raw.value } : { value: null, issue: "invalid" };
}
function itemValue(snak: ObjectValue): Resolution<string> {
  const data = dataValue(snak, "wikibase-item", "wikibase-entityid");
  if (data.issue) return data;
  const raw = data.value;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { value: null, issue: "invalid" };
  const item = raw as ObjectValue;
  if (item["entity-type"] !== "item" || typeof item.id !== "string" || !itemPattern.test(item.id) || item.id.length > 40 ||
      (item["numeric-id"] !== undefined && String(item["numeric-id"]) !== item.id.slice(1))) {
    return { value: null, issue: "invalid" };
  }
  return { value: item.id };
}
function mappedItem<T>(snak: ObjectValue, mapping: Record<string, T>): Resolution<T> {
  const item = itemValue(snak);
  if (item.issue) return item;
  return Object.hasOwn(mapping, item.value) ? { value: mapping[item.value] } : { value: null, issue: "unmapped" };
}
function classificationValue(snak: ObjectValue, mapping: ClassificationMapping): Resolution<string> {
  const key = mapping.datatype === "wikibase-item" ? itemValue(snak) : dataValue(snak, "string", "string");
  if (key.issue) return key;
  return typeof key.value === "string" && Object.hasOwn(mapping.values, key.value)
    ? { value: mapping.values[key.value] } : { value: null, issue: "unmapped" };
}
const certificateReasons = ["nonFilmOrUnknownFormat", "unsupportedQualifier", "multipleCertificates",
  "missingCertificate", "invalidCertificate", "ratingRejected"] as const;
type CertificateReason = typeof certificateReasons[number];
interface CertificateAudit {
  evaluatedItems: number; acceptedItems: number; rejections: Record<CertificateReason, number>;
}
function certificateReference(statement: Statement, maximum: number):
  { value: string; reason?: undefined } | { value: null; reason: CertificateReason } {
  const qualifiers = statement.qualifiers;
  if (Object.keys(qualifiers).some((property) => property !== "P2676")) {
    return { value: null, reason: "unsupportedQualifier" };
  }
  const references = qualifiers.P2676;
  if (references === undefined || (Array.isArray(references) && !references.length)) {
    return { value: null, reason: "missingCertificate" };
  }
  if (!Array.isArray(references)) return { value: null, reason: "invalidCertificate" };
  if (references.length !== 1) return { value: null, reason: "multipleCertificates" };
  const raw = references[0];
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { value: null, reason: "invalidCertificate" };
  const snak = raw as ObjectValue;
  if (Object.keys(snak).some((key) => !["snaktype", "property", "datatype", "datavalue", "hash"].includes(key)) ||
      snak.snaktype !== "value" || snak.property !== "P2676" || snak.datatype !== "string" ||
      (snak.hash !== undefined && (typeof snak.hash !== "string" || !/^[a-f0-9]{40}$/.test(snak.hash)))) {
    return { value: null, reason: "invalidCertificate" };
  }
  const data = snak.datavalue;
  if (!data || typeof data !== "object" || Array.isArray(data)) return { value: null, reason: "invalidCertificate" };
  const { type, value } = data as ObjectValue;
  if (Object.keys(data).length !== 2 || type !== "string" || typeof value !== "string" ||
      !value.length || value.length > maximum || value !== value.trim() || /[\u0000-\u001f\u007f]/.test(value)) {
    return { value: null, reason: "invalidCertificate" };
  }
  return { value };
}
function certificateClassification(statements: Statement[], mapping: NonNullable<WikibaseMappingPolicyV2["classification"]>,
  mediaFormat: Resolution<MediaFormat>, audit: CertificateAudit): Resolution<string> {
  if (!statements.length) return { value: null, issue: "missing" };
  audit.evaluatedItems += 1;
  const reject = (reason: CertificateReason, issue: Issue): Resolution<string> => {
    audit.rejections[reason] += 1;
    return { value: null, issue };
  };
  if (mediaFormat.issue || mediaFormat.value !== "Movie") return reject("nonFilmOrUnknownFormat", "unmapped");
  const references = statements.map((entry) => certificateReference(entry, mapping.certificateReference.maximumLength));
  // Fixed precedence makes mixed refusals independent of statement order.
  const reason = certificateReasons.find((code) => references.some((entry) => entry.reason === code));
  if (reason) return reject(reason, reason === "missingCertificate" ? "missing" :
    reason === "unsupportedQualifier" ? "qualified" : reason === "multipleCertificates" ? "conflict" : "invalid");
  if (new Set(references.map((entry) => entry.value)).size !== 1) return reject("multipleCertificates", "conflict");
  // Only this property, after exact certificate and film checks, is exempt from v1's qualifier refusal.
  const rating = scalar(values(statements.map((entry) => ({ ...entry, qualified: false })),
    (snak) => classificationValue(snak, mapping)));
  if (rating.issue) return reject("ratingRejected", rating.issue);
  audit.acceptedItems += 1;
  return rating;
}
function decimal(value: unknown): number | null {
  if (typeof value !== "string" || value.length > 40 || !/^[+-]\d+(?:\.\d+)?$/.test(value)) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}
function quantity(snak: ObjectValue, units: Record<string, number>, integer: boolean, maximum: number): Resolution<number> {
  const data = dataValue(snak, "quantity", "quantity");
  if (data.issue) return data;
  if (!data.value || typeof data.value !== "object" || Array.isArray(data.value)) return { value: null, issue: "invalid" };
  const raw = data.value as ObjectValue;
  const amount = decimal(raw.amount);
  if (amount === null || typeof raw.unit !== "string") return { value: null, issue: "invalid" };
  if (!Object.hasOwn(units, raw.unit)) return { value: null, issue: "unmapped" };
  const lower = raw.lowerBound ?? null, upper = raw.upperBound ?? null;
  if ((lower === null) !== (upper === null) ||
      (lower !== null && (decimal(lower) !== amount || decimal(upper) !== amount))) {
    return { value: null, issue: "invalid" };
  }
  const converted = amount * units[raw.unit];
  if (converted <= 0 || converted > maximum || (integer && !Number.isSafeInteger(converted))) {
    return { value: null, issue: "invalid" };
  }
  return { value: converted };
}
function releaseYear(snak: ObjectValue): Resolution<number> {
  const data = dataValue(snak, "time", "time");
  if (data.issue) return data;
  if (!data.value || typeof data.value !== "object" || Array.isArray(data.value)) return { value: null, issue: "invalid" };
  const raw = data.value as ObjectValue;
  const match = typeof raw.time === "string" && /^\+(\d{4,16})-(\d{2})-(\d{2})T00:00:00Z$/.exec(raw.time);
  if (!match || raw.calendarmodel !== gregorian || raw.before !== 0 || raw.after !== 0 || raw.timezone !== 0 ||
      ![9, 10, 11].includes(raw.precision as number)) return { value: null, issue: "invalid" };
  const year = Number(match[1]), month = Number(match[2]), day = Number(match[3]);
  if (year < 1800 || year > 3000 || month > 12 || day > 31 ||
      (Number(raw.precision) >= 10 && month < 1) ||
      (raw.precision === 11 && (day < 1 || new Date(Date.UTC(year, month - 1, day)).getUTCDate() !== day))) {
    return { value: null, issue: "invalid" };
  }
  return { value: year };
}
function titleOf(entity: Entity, policy: WikibaseMappingPolicy): string | null {
  for (const language of policy.languageOrder) {
    const entry = entity.labels[language] as ObjectValue | undefined;
    if (entry) return typeof entry.value === "string" && entry.value.trim() && entry.value.length <= 200 ? entry.value : null;
  }
  return null;
}
function aliasesOf(entity: Entity, policy: WikibaseMappingPolicy, title: string): Resolution<string[]> {
  if (!entity.aliases) return { value: null, issue: "missing" };
  const names = policy.languageOrder.flatMap((language) =>
    (entity.aliases![language] as ObjectValue[] | undefined ?? []).map((entry) => entry.value as string));
  if (names.some((name) => name.length > 200)) return { value: null, issue: "invalid" };
  const unique = new Map<string, string>();
  for (const name of names.sort()) {
    const key = name.trim().toLowerCase();
    if (key !== title.trim().toLowerCase() && !unique.has(key)) unique.set(key, name);
  }
  return unique.size > 20 ? { value: null, issue: "invalid" } : { value: [...unique.values()] };
}
function animeIdentifier(snak: ObjectValue): Resolution<number> {
  const data = dataValue(snak, "external-id", "string");
  if (data.issue) return data;
  const raw = data.value;
  return typeof raw === "string" && /^[1-9]\d*$/.test(raw) && Number.isSafeInteger(Number(raw))
    ? { value: Number(raw) } : { value: null, issue: "invalid" };
}

/** Actual bytes bind provenance; the policy and universe are independently hashed in the audit. */
export function mapWikibaseMetadata(sourceBytes: Uint8Array, universe: readonly number[],
  policy: WikibaseMappingPolicy, source: { name: string; snapshotAt: string }) {
  validatePolicy(policy);
  if (!universe.length || universe.length > WIKIBASE_MAPPING_LIMITS.universe || new Set(universe).size !== universe.length ||
      universe.some((id) => !Number.isSafeInteger(id) || id < 1)) fail("universe", "requires one to 100 unique positive anime IDs");
  exact(object(source, "source identity"), ["name", "snapshotAt"], "source identity");
  text(source.name, "source.name", 120);
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(source.snapshotAt) ||
      !Number.isFinite(Date.parse(source.snapshotAt)) || new Date(source.snapshotAt).toISOString() !== source.snapshotAt) {
    fail("source.snapshotAt", "requires a canonical UTC date-time");
  }
  const entities = readEntities(sourceBytes, policy);
  const ids = new Set(universe);
  const byAnime = new Map<number, Entity[]>();
  const possibleByAnime = new Map<number, Set<string>>();
  const identity = { missing: 0, unresolved: 0, outsideUniverse: 0, duplicateMappings: 0, missingTitle: 0 };
  for (const entity of entities) {
    // Identity conflicts at any nondeprecated rank require review, not preferred-rank guessing.
    const statements = (entity.claims.get("P4086") ?? []).filter((entry) => entry.rank !== "deprecated");
    for (const statement of statements) {
      if (statement.mainsnak.snaktype !== "value") continue;
      const possible = animeIdentifier(statement.mainsnak);
      if (!possible.issue && ids.has(possible.value)) {
        possibleByAnime.set(possible.value, new Set([...(possibleByAnime.get(possible.value) ?? []), entity.id]));
      }
    }
    const result = scalar(values(statements, animeIdentifier));
    if (result.issue) { identity[result.issue === "missing" ? "missing" : "unresolved"] += 1; continue; }
    if (!ids.has(result.value)) { identity.outsideUniverse += 1; continue; }
    byAnime.set(result.value, [...(byAnime.get(result.value) ?? []), entity]);
  }
  identity.duplicateMappings = [...possibleByAnime.values()].filter((matches) => matches.size > 1).length;
  const accepted = new Map<string, { animeId: number; entity: Entity; title: string }>();
  for (const [animeId, matches] of byAnime) {
    if (matches.length !== 1 || possibleByAnime.get(animeId)!.size !== 1) continue;
    const title = titleOf(matches[0], policy);
    if (!title) { identity.missingTitle += 1; continue; }
    accepted.set(matches[0].id, { animeId, entity: matches[0], title });
  }
  const fieldIssues = Object.fromEntries(fields.map((field) => [field,
    Object.fromEntries(issues.map((issue) => [issue, 0]))])) as Record<CatalogCoverageField, Record<Issue, number>>;
  const classificationCertificate: CertificateAudit = { evaluatedItems: 0, acceptedItems: 0,
    rejections: Object.fromEntries(certificateReasons.map((reason) => [reason, 0])) as Record<CertificateReason, number> };
  const record = <T>(field: CatalogCoverageField, resolution: Resolution<T>): T | null => {
    if (resolution.issue) fieldIssues[field][resolution.issue] += 1;
    return resolution.value;
  };
  const anime: CatalogMetadataItemV1[] = [...accepted.values()].sort((left, right) => left.animeId - right.animeId).map(({ animeId, entity, title }) => {
    const read = <T>(property: string, decode: (snak: ObjectValue) => Resolution<T>) => values(bestStatements(entity, property), decode);
    const genres = read("P136", (snak) => mappedItem(snak, policy.genreLabels));
    const mediaFormat = scalar(read("P31", (snak) => mappedItem(snak, policy.mediaFormats)));
    const classification = policy.classification;
    const rating = policy.format === "wikibase-metadata-policy-v2" && policy.classification
      ? certificateClassification(bestStatements(entity, policy.classification.property), policy.classification,
        mediaFormat, classificationCertificate)
      : classification ? scalar(read(classification.property, (snak) => classificationValue(snak, classification)))
        : { value: null, issue: "missing" } as Resolution<string>;
    const relations: NonNullable<CatalogMetadataItemV1["relations"]> = [];
    const relationIssues: Issue[] = [];
    for (const [property, kind] of [["P155", "prequel"], ["P156", "sequel"]] as const) {
      const target = scalar(read(property, itemValue));
      if (target.issue) { relationIssues.push(target.issue); continue; }
      const mapped = accepted.get(target.value);
      if (!mapped || mapped.animeId === animeId) { relationIssues.push("unmapped"); continue; }
      relations.push({ kind, animeId: mapped.animeId, title: mapped.title });
    }
    // Report each unresolved direction; never infer the inverse edge or use shared P179 membership.
    for (const issue of relationIssues) fieldIssues.relations[issue] += 1;
    return { animeId, sourceItemId: entity.id, title,
      aliases: record("aliases", aliasesOf(entity, policy, title)),
      genres: record("genres", genres.issue ? genres : { value: [...new Set(genres.value)].sort() }),
      year: record("year", scalar(read("P577", releaseYear))),
      mediaFormat: record("mediaFormat", mediaFormat),
      episodeCount: record("episodeCount", scalar(read("P1113", (snak) => quantity(snak, { "1": 1 }, true, 100000)))),
      runtimeMinutes: record("runtimeMinutes", scalar(read("P2047", (snak) => quantity(snak, policy.durationUnits, false, 10000)))),
      contentClassification: record("contentClassification", rating.issue ? rating : { value: {
        jurisdiction: classification!.jurisdiction, system: classification!.system, value: rating.value } }),
      communityScore: record("communityScore", { value: null, issue: "missing" }),
      relations: relations.length ? relations : null };
  });
  const candidate: CatalogMetadataSnapshotV1 = { format: "anime-metadata-catalog-v1",
    source: { ...source, snapshotSha256: digest(sourceBytes) }, anime };
  // An empty feasibility result is a report with no valid export, not a fabricated catalog item.
  const snapshot = anime.length ? parseCatalogMetadataSnapshot(candidate, "mapped metadata candidate") : null;
  const names = new Map<string, Set<number>>();
  for (const item of anime) {
    for (const name of [item.title, ...(item.aliases ?? [])]) {
      const key = name.trim().toLowerCase();
      names.set(key, new Set([...(names.get(key) ?? []), item.animeId]));
    }
  }
  const byId = new Map(anime.map((item) => [item.animeId, item]));
  const oneSidedDirectedEdges = anime.reduce((count, item) => count + (item.relations ?? []).filter((relation) =>
    !(byId.get(relation.animeId)?.relations ?? []).some((inverse) => inverse.animeId === item.animeId &&
      inverse.kind === (relation.kind === "prequel" ? "sequel" : "prequel"))).length, 0);
  return { snapshot, report: { format: policy.format === "wikibase-metadata-policy-v1"
    ? "wikibase-metadata-audit-v1" as const : "wikibase-metadata-audit-v2" as const,
    ...(policy.format === "wikibase-metadata-policy-v2" ? { classificationCertificate } : {}),
    sourceSnapshotSha256: candidate.source.snapshotSha256, policySha256: digest(canonical(policy)),
    universeSha256: digest(canonical([...universe].sort((a, b) => a - b))),
    sourceEntities: entities.length, emittedItems: anime.length, identity, fieldIssues,
    ambiguousTitleKeys: [...names.values()].filter((matches) => matches.size > 1).length,
    oneSidedDirectedEdges,
    coverage: catalogMetadataCoverage(candidate, universe) } };
}
