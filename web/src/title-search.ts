/** Local title lookup over catalog identities and metadata already loaded in this browser. */
import type { AnimeMetadata } from "./artifacts";
import type { AnimeInfo, RecommendationIndex } from "./domain";
import { normalizeTitle } from "./title";

export interface TitleMatch {
  anime: AnimeInfo;
  matchedAlias: string | null;
}

export interface TitleSearchResult {
  matches: TitleMatch[];
  total: number;
  /** Only an unambiguous exact ID, canonical title, or known alias is automatic. */
  automatic: AnimeInfo | null;
}

interface RankedMatch extends TitleMatch {
  priority: number;
}

export function searchAnimeTitles(
  raw: string,
  index: RecommendationIndex,
  knownMetadata: ReadonlyMap<number, AnimeMetadata>,
  limit = 8,
): TitleSearchResult {
  const query = normalizeTitle(raw);
  if (!query || raw.length > 500) return { matches: [], total: 0, automatic: null };

  const numeric = /^(?:anime:)?([1-9]\d*)$/i.exec(raw.trim());
  if (numeric) {
    const anime = index.animeByAnimeId.get(Number(numeric[1]));
    return anime
      ? { matches: [{ anime, matchedAlias: null }], total: 1, automatic: anime }
      : { matches: [], total: 0, automatic: null };
  }

  const ranked: RankedMatch[] = [];
  for (const anime of index.animeList) {
    const canonical = normalizeTitle(anime.label);
    const aliases = knownMetadata.get(anime.animeId)?.aliases ?? [];
    let best: RankedMatch | null = canonical === query
      ? { anime, matchedAlias: null, priority: 0 }
      : canonical.startsWith(query)
        ? { anime, matchedAlias: null, priority: 2 }
        : canonical.includes(query)
          ? { anime, matchedAlias: null, priority: 4 }
          : null;
    for (const alias of aliases) {
      const normalized = normalizeTitle(alias);
      const priority = normalized === query ? 1
        : normalized.startsWith(query) ? 3
          : normalized.includes(query) ? 5 : null;
      if (priority !== null && (!best || priority < best.priority)) {
        best = { anime, matchedAlias: alias, priority };
      }
    }
    if (best) ranked.push(best);
  }

  const hasExact = ranked.some((item) => item.priority <= 1);
  const eligible = hasExact ? ranked.filter((item) => item.priority <= 1) : ranked;
  eligible.sort((left, right) => left.priority - right.priority ||
    left.anime.label.localeCompare(right.anime.label) || left.anime.animeId - right.anime.animeId);
  const bounded = eligible.slice(0, Math.max(1, Math.min(12, Math.trunc(limit))))
    .map(({ anime, matchedAlias }) => ({ anime, matchedAlias }));
  return {
    matches: bounded,
    total: eligible.length,
    automatic: hasExact && eligible.length === 1 ? eligible[0].anime : null,
  };
}
