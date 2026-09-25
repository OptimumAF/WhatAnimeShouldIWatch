/** Local intent, separate from observed watch history and graph edge weights. */
import { historyScoreToTen } from "./import-history";
import type { HistoryEntry } from "./import-history";

export type PreferenceSentiment = "seen" | "liked" | "disliked";
export type PreferenceSource = "manual" | "import" | "legacy";

export interface AnimePreference {
  nodeId: string;
  sentiment: PreferenceSentiment;
  /** User-controlled emphasis, independent of certainty in the sentiment. */
  importance: number;
  /** 0 for seen/unrated; 0..1 for rated or explicitly stated preference. */
  confidence: number;
  source: PreferenceSource;
}

export const MIN_IMPORTANCE = 0.2;
export const MAX_IMPORTANCE = 3;

export function clampImportance(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.min(Math.max(value, MIN_IMPORTANCE), MAX_IMPORTANCE);
}

export function manualPreference(nodeId: string, sentiment: PreferenceSentiment = "seen", importance = 1): AnimePreference {
  return { nodeId, sentiment, importance: clampImportance(importance),
    confidence: sentiment === "seen" ? 0 : 1, source: "manual" };
}

/** Thresholds are deliberately conservative: 1–4 disliked, 5–6 unrated, 7–10 liked. */
export function preferenceFromHistory(entry: HistoryEntry, nodeId: string): AnimePreference | null {
  if (entry.status === "plan_to_watch") return null;
  const scoreTen = historyScoreToTen(entry);
  if (scoreTen === null || scoreTen > 4 && scoreTen < 7) {
    return { nodeId, sentiment: "seen", importance: 1, confidence: 0, source: "import" };
  }
  if (scoreTen >= 7) {
    return { nodeId, sentiment: "liked", importance: 1,
      confidence: roundConfidence((scoreTen - 6) / 4), source: "import" };
  }
  return { nodeId, sentiment: "disliked", importance: 1,
    confidence: roundConfidence((5 - scoreTen) / 4), source: "import" };
}

/** Old weights expressed emphasis, not a signed preference. Keep them without inventing a strong like. */
export function migrateLegacyPreferences(
  selected: readonly { nodeId: string; weight: number }[], history: readonly HistoryEntry[],
): AnimePreference[] {
  const historyByNodeId = new Map<string, HistoryEntry[]>();
  for (const entry of history) {
    if (entry.animeId !== null) {
      const nodeId = `anime:${entry.animeId}`;
      historyByNodeId.set(nodeId, [...(historyByNodeId.get(nodeId) ?? []), entry]);
    }
  }
  const byNodeId = new Map<string, AnimePreference>();
  for (const entry of selected) {
    const known = historyByNodeId.get(entry.nodeId);
    const signals = known?.map((item) => preferenceFromHistory(item, entry.nodeId));
    const agreement = signals?.length && signals.every((item) => item?.sentiment === signals[0]?.sentiment)
      ? signals[0] : null;
    const sentiment = agreement?.sentiment ?? (known ? "seen" : entry.weight > 1 ? "liked" : "seen");
    const confidence = sentiment === "seen" ? 0 : agreement
      ? Math.min(...signals!.map((item) => item!.confidence)) : 0.5;
    byNodeId.set(entry.nodeId, {
      nodeId: entry.nodeId, sentiment, importance: entry.weight,
      confidence,
      source: "legacy",
    });
  }
  return [...byNodeId.values()];
}

export function validatePreferences(value: unknown): AnimePreference[] {
  if (!Array.isArray(value) || value.length > 10_000) throw new Error("Invalid saved preference list.");
  const seen = new Set<string>();
  return value.map((entry: unknown) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error("Invalid saved preference entry.");
    }
    const item = entry as Record<string, unknown>;
    if (typeof item.nodeId !== "string" || !item.nodeId ||
        !["seen", "liked", "disliked"].includes(String(item.sentiment)) ||
        typeof item.importance !== "number" || !Number.isFinite(item.importance) ||
        item.importance < MIN_IMPORTANCE || item.importance > MAX_IMPORTANCE ||
        typeof item.confidence !== "number" || !Number.isFinite(item.confidence) ||
        (item.sentiment === "seen" ? item.confidence !== 0 : item.confidence <= 0 || item.confidence > 1) ||
        !["manual", "import", "legacy"].includes(String(item.source)) || seen.has(item.nodeId)) {
      throw new Error("Invalid saved preference entry.");
    }
    seen.add(item.nodeId);
    return {
      nodeId: item.nodeId, sentiment: item.sentiment as PreferenceSentiment,
      importance: item.importance, confidence: item.confidence,
      source: item.source as PreferenceSource,
    };
  });
}

function roundConfidence(value: number): number {
  return Math.round(Math.min(Math.max(value, 0.01), 1) * 100) / 100;
}
