/** Local watch plans and explicit ratings, separate from imported provider history. */
import type { RecommendationIndex } from "./domain";
import type { AnimePreference } from "./preferences";

export type WatchlistStatus = "plan_to_watch" | "watching" | "completed" | "on_hold" | "dropped";

export interface WatchlistEntry {
  animeId: number;
  /** Kept when a later catalog no longer contains this anime ID. */
  title: string;
  status: WatchlistStatus;
  /** Explicit local score on a 1–10 integer scale, or null for unrated. */
  rating: number | null;
}

const statuses = new Set<WatchlistStatus>([
  "plan_to_watch", "watching", "completed", "on_hold", "dropped",
]);

export function validateWatchlist(value: unknown): WatchlistEntry[] {
  if (!Array.isArray(value) || value.length > 10_000) throw new Error("Invalid local watchlist size.");
  const seen = new Set<number>();
  return value.map((entry: unknown) => {
    if (entry === null || typeof entry !== "object" || Array.isArray(entry)) {
      throw new Error("Invalid local watchlist entry.");
    }
    const item = entry as Record<string, unknown>;
    if (Object.keys(item).some((key) => !["animeId", "title", "status", "rating"].includes(key)) ||
        !Number.isSafeInteger(item.animeId) || (item.animeId as number) <= 0 ||
        typeof item.title !== "string" || !item.title.trim() || item.title.length > 500 ||
        !statuses.has(item.status as WatchlistStatus) ||
        !(item.rating === null || Number.isInteger(item.rating) &&
          (item.rating as number) >= 1 && (item.rating as number) <= 10) ||
        seen.has(item.animeId as number)) {
      throw new Error("Invalid local watchlist entry.");
    }
    seen.add(item.animeId as number);
    return { animeId: item.animeId as number, title: item.title,
      status: item.status as WatchlistStatus, rating: item.rating as number | null };
  });
}

export function watchedWatchlistAnimeIds(entries: readonly WatchlistEntry[]): number[] {
  return entries.filter((entry) => entry.status !== "plan_to_watch").map((entry) => entry.animeId);
}

/** A local rating is feedback for this browser's scorer. Status alone never creates a signal. */
export function mergeWatchlistFeedback(
  preferences: readonly AnimePreference[], entries: readonly WatchlistEntry[], index: RecommendationIndex,
): AnimePreference[] {
  const result = new Map(preferences.map((item) => [item.nodeId, item]));
  for (const entry of entries) {
    const anime = index.animeByAnimeId.get(entry.animeId);
    if (!anime) continue;
    if (entry.status === "plan_to_watch") {
      result.delete(anime.nodeId);
      continue;
    }
    if (entry.rating === null || result.get(anime.nodeId)?.source === "manual") continue;
    const sentiment = entry.rating >= 7 ? "liked" : entry.rating <= 4 ? "disliked" : "seen";
    const confidence = sentiment === "seen" ? 0 : sentiment === "liked"
      ? (entry.rating - 6) / 4 : (5 - entry.rating) / 4;
    result.set(anime.nodeId, { nodeId: anime.nodeId, sentiment,
      importance: 1, confidence, source: "manual" });
  }
  return [...result.values()];
}
