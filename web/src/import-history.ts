import type { AnimeInfo, RecommendationIndex } from "./domain";
import { normalizeTitle } from "./recommendations";

export const MAX_TEXT_IMPORT_BYTES = 128 * 1024;
export const MAX_MAL_XML_IMPORT_BYTES = 2 * 1024 * 1024;
export const MAX_HISTORY_ENTRIES = 10_000;

export type HistoryProvider = "mal" | "anilist" | "local";
export type HistoryStatus = "watching" | "completed" | "on_hold" | "dropped" | "plan_to_watch" | "unknown";
export type HistoryScoreScale =
  | "mal-10" | "local-10"
  | "POINT_100" | "POINT_10_DECIMAL" | "POINT_10" | "POINT_5" | "POINT_3";
export type ImportMode = "merge" | "replace";

export interface HistoryEntry {
  provider: HistoryProvider;
  sourceId: string;
  title: string;
  animeId: number | null;
  status: HistoryStatus;
  sourceStatus: string;
  progressEpisodes: number | null;
  score: number | null;
  scoreScale: HistoryScoreScale;
}

export interface ParsedHistory {
  entries: HistoryEntry[];
  duplicates: number;
}

export interface ImportPreview {
  total: number;
  duplicates: number;
  mapped: number;
  unmapped: number;
  unscored: number;
  seen: number;
  planned: number;
  added: number;
  updated: number;
  unchanged: number;
  removed: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function byteLength(text: string): number {
  return new TextEncoder().encode(text).byteLength;
}

function integerToken(raw: string, maximum: number): number | null {
  if (!/^(?:0|[1-9]\d*)$/.test(raw)) return null;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) && parsed <= maximum ? parsed : null;
}

function scoreToken(raw: string, integerOnly: boolean): number | null {
  if (!/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(raw)) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed <= 10 && (!integerOnly || Number.isInteger(parsed))
    ? parsed : null;
}

export function normalizeHistoryStatus(raw: string): HistoryStatus {
  const value = raw.trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (value === "watching" || value === "current" || value === "repeating" || value === "1") return "watching";
  if (value === "completed" || value === "2") return "completed";
  if (value === "on_hold" || value === "paused" || value === "3") return "on_hold";
  if (value === "dropped" || value === "4") return "dropped";
  if (value === "plan_to_watch" || value === "planning" || value === "6") return "plan_to_watch";
  return "unknown";
}

export function historyIdentity(entry: Pick<HistoryEntry, "provider" | "sourceId">): string {
  return `${entry.provider}:${entry.sourceId}`;
}

export function historyScoreToTen(entry: HistoryEntry): number | null {
  if (entry.score === null) return null;
  switch (entry.scoreScale) {
    case "POINT_100": return entry.score / 10;
    case "POINT_5": return entry.score * 2;
    // AniList smileys are categorical. Keep sad negative and neutral unrated.
    case "POINT_3": return entry.score === 1 ? 2 : entry.score === 2 ? 5 : 10;
    default: return entry.score;
  }
}

function maximumScore(scale: HistoryScoreScale): number {
  if (scale === "POINT_100") return 100;
  if (scale === "POINT_5") return 5;
  if (scale === "POINT_3") return 3;
  return 10;
}

function integerScore(scale: HistoryScoreScale): boolean {
  return scale === "mal-10" || scale === "POINT_100" || scale === "POINT_10" ||
    scale === "POINT_5" || scale === "POINT_3";
}

export function validateHistoryEntries(value: unknown): HistoryEntry[] {
  if (!Array.isArray(value) || value.length > MAX_HISTORY_ENTRIES) {
    throw new Error("Invalid saved history size.");
  }
  const seen = new Set<string>();
  const statuses = new Set<HistoryStatus>(["watching", "completed", "on_hold", "dropped", "plan_to_watch", "unknown"]);
  const scales = new Set<HistoryScoreScale>([
    "mal-10", "local-10", "POINT_100", "POINT_10_DECIMAL", "POINT_10", "POINT_5", "POINT_3",
  ]);
  const providers = new Set<HistoryProvider>(["mal", "anilist", "local"]);
  return value.map((entry: unknown) => {
    if (!isRecord(entry) || !providers.has(entry.provider as HistoryProvider) ||
      typeof entry.sourceId !== "string" || !entry.sourceId || entry.sourceId.length > 512 ||
      typeof entry.title !== "string" || !entry.title.trim() || entry.title.length > 500 ||
      !statuses.has(entry.status as HistoryStatus) ||
      typeof entry.sourceStatus !== "string" || entry.sourceStatus.length > 80 ||
      !(entry.animeId === null || (Number.isSafeInteger(entry.animeId) && (entry.animeId as number) > 0)) ||
      !(entry.progressEpisodes === null ||
        (Number.isSafeInteger(entry.progressEpisodes) && (entry.progressEpisodes as number) >= 0 &&
          (entry.progressEpisodes as number) <= 1_000_000)) ||
      !scales.has(entry.scoreScale as HistoryScoreScale) ||
      (entry.provider === "mal" && entry.scoreScale !== "mal-10") ||
      (entry.provider === "local" && entry.scoreScale !== "local-10") ||
      (entry.provider === "anilist" && !String(entry.scoreScale).startsWith("POINT_")) ||
      !(entry.score === null || (typeof entry.score === "number" && Number.isFinite(entry.score) &&
        entry.score > 0 && entry.score <= maximumScore(entry.scoreScale as HistoryScoreScale) &&
        (!integerScore(entry.scoreScale as HistoryScoreScale) || Number.isInteger(entry.score)) &&
        (entry.scoreScale !== "POINT_10_DECIMAL" || Number.isInteger(entry.score * 10))))) {
      throw new Error("Invalid saved history entry.");
    }
    const checked = entry as unknown as HistoryEntry;
    const key = historyIdentity(checked);
    if (seen.has(key)) throw new Error("Duplicate saved history identity.");
    seen.add(key);
    return {
      provider: checked.provider, sourceId: checked.sourceId, title: checked.title,
      animeId: checked.animeId, status: checked.status, sourceStatus: checked.sourceStatus,
      progressEpisodes: checked.progressEpisodes, score: checked.score, scoreScale: checked.scoreScale,
    };
  });
}

export function deduplicateHistory(entries: HistoryEntry[]): ParsedHistory {
  const byIdentity = new Map<string, HistoryEntry>();
  for (const entry of entries) byIdentity.set(historyIdentity(entry), entry);
  return { entries: validateHistoryEntries([...byIdentity.values()]), duplicates: entries.length - byIdentity.size };
}

export function parseTextHistory(text: string): ParsedHistory {
  if (byteLength(text) > MAX_TEXT_IMPORT_BYTES) throw new Error("Text import exceeds 128 KiB.");
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length > MAX_HISTORY_ENTRIES) throw new Error("Text import has too many entries.");
  const entries: HistoryEntry[] = lines.map((line, index) => {
    const parts = line.split(/[,\t|]/).map((part) => part.trim());
    if (parts.length > 4 || !parts[0]) throw new Error(`Invalid text import line ${index + 1}.`);
    const token = parts[0];
    const numeric = /^(?:anime:)?(\d+)$/.exec(token);
    const animeId = numeric ? integerToken(numeric[1], Number.MAX_SAFE_INTEGER) : null;
    if (numeric && (!animeId || animeId <= 0)) throw new Error(`Invalid anime ID on line ${index + 1}.`);
    if (!numeric && token.length > 500) throw new Error(`Title on line ${index + 1} is too long.`);
    const scoreValue = parts[1] ? scoreToken(parts[1], false) : 0;
    if (scoreValue === null) throw new Error(`Invalid score on line ${index + 1}.`);
    const sourceStatus = parts[2] ?? "";
    if (sourceStatus.length > 80) throw new Error(`Status on line ${index + 1} is too long.`);
    const progress = parts[3] ? integerToken(parts[3], 1_000_000) : null;
    if (parts[3] && progress === null) throw new Error(`Invalid progress on line ${index + 1}.`);
    return {
      provider: "local", sourceId: animeId ? `anime:${animeId}` : `title:${normalizeTitle(token)}`,
      title: token, animeId, status: normalizeHistoryStatus(sourceStatus), sourceStatus,
      progressEpisodes: progress, score: scoreValue === 0 ? null : scoreValue,
      scoreScale: "local-10",
    };
  });
  return deduplicateHistory(entries);
}

/** A deliberately limited MAL-style XML subset. DTDs/entities are rejected before DOM parsing. */
export function parseMalXmlHistory(text: string): ParsedHistory {
  if (byteLength(text) > MAX_MAL_XML_IMPORT_BYTES) throw new Error("MAL XML import exceeds 2 MiB.");
  if (/<!\s*(?:DOCTYPE|ENTITY)/i.test(text)) throw new Error("XML declarations with DTDs or entities are not supported.");
  const document = new DOMParser().parseFromString(text, "application/xml");
  if (document.querySelector("parsererror") || document.documentElement.tagName !== "myanimelist") {
    throw new Error("Invalid MAL XML document.");
  }
  const animeNodes = [...document.documentElement.children].filter((child) => child.tagName === "anime");
  if (animeNodes.length > MAX_HISTORY_ENTRIES) throw new Error("MAL XML import has too many entries.");
  const directText = (node: Element, tag: string): string => {
    const children = [...node.children].filter((child) => child.tagName === tag);
    if (children.length > 1 || children[0]?.children.length) throw new Error(`Invalid MAL XML ${tag} field.`);
    return children[0]?.textContent?.trim() ?? "";
  };
  const entries: HistoryEntry[] = animeNodes.map((node, index) => {
    const id = integerToken(directText(node, "series_animedb_id"), Number.MAX_SAFE_INTEGER);
    const title = directText(node, "series_title");
    if (!id || !title || title.length > 500) throw new Error(`Invalid MAL XML anime at entry ${index + 1}.`);
    const rawScore = directText(node, "my_score");
    const scoreValue = rawScore ? scoreToken(rawScore, true) : 0;
    if (scoreValue === null) throw new Error(`Invalid MAL XML score at entry ${index + 1}.`);
    const rawProgress = directText(node, "my_watched_episodes");
    const progress = rawProgress ? integerToken(rawProgress, 1_000_000) : null;
    if (rawProgress && progress === null) throw new Error(`Invalid MAL XML progress at entry ${index + 1}.`);
    const sourceStatus = directText(node, "my_status");
    if (sourceStatus.length > 80) throw new Error(`Invalid MAL XML status at entry ${index + 1}.`);
    return {
      provider: "mal", sourceId: String(id), title, animeId: id,
      status: normalizeHistoryStatus(sourceStatus), sourceStatus,
      progressEpisodes: progress, score: scoreValue === 0 ? null : scoreValue,
      scoreScale: "mal-10",
    };
  });
  return deduplicateHistory(entries);
}

export function resolveHistoryAnime(entry: HistoryEntry, index: RecommendationIndex): AnimeInfo | null {
  if (entry.animeId !== null) return index.animeByAnimeId.get(entry.animeId) ?? null;
  const matches = index.titleLookup.get(normalizeTitle(entry.title)) ?? [];
  return matches.length === 1 ? matches[0] : null;
}

export function mergeHistory(existing: HistoryEntry[], incoming: HistoryEntry[], mode: ImportMode): HistoryEntry[] {
  if (mode === "replace") return deduplicateHistory(incoming).entries;
  return deduplicateHistory([...existing, ...incoming]).entries;
}

function isSeenHistoryEntry(entry: HistoryEntry): boolean {
  return ["watching", "completed", "on_hold", "dropped"].includes(entry.status) ||
    (entry.provider === "local" && entry.status === "unknown");
}

export function previewHistory(
  incoming: ParsedHistory,
  existing: HistoryEntry[],
  index: RecommendationIndex,
  mode: ImportMode,
): ImportPreview {
  const prior = new Map(existing.map((entry) => [historyIdentity(entry), entry]));
  const next = new Set(incoming.entries.map(historyIdentity));
  let mapped = 0;
  let unscored = 0;
  let seen = 0;
  let planned = 0;
  let added = 0;
  let updated = 0;
  let unchanged = 0;
  for (const entry of incoming.entries) {
    if (resolveHistoryAnime(entry, index)) mapped += 1;
    if (entry.score === null) unscored += 1;
    if (isSeenHistoryEntry(entry)) seen += 1;
    if (entry.status === "plan_to_watch") planned += 1;
    const old = prior.get(historyIdentity(entry));
    if (!old) added += 1;
    else if (JSON.stringify(old) === JSON.stringify(entry)) unchanged += 1;
    else updated += 1;
  }
  return {
    total: incoming.entries.length, duplicates: incoming.duplicates,
    mapped, unmapped: incoming.entries.length - mapped, unscored, seen, planned,
    added, updated, unchanged,
    removed: mode === "replace" ? existing.filter((entry) => !next.has(historyIdentity(entry))).length : 0,
  };
}

export function seenHistoryNodeIds(entries: HistoryEntry[], index: RecommendationIndex): string[] {
  return entries
    .filter(isSeenHistoryEntry)
    .map((entry) => resolveHistoryAnime(entry, index)?.nodeId)
    .filter((nodeId): nodeId is string => nodeId !== undefined);
}
