/** Existing optional public-provider reads behind injectable transport and timing. */
import type { AnimeMetadata, AnimeRelation } from "./artifacts";
import type { ImportedPreferenceEntry, RecommendationIndex, SeasonalAnimeItem, UsernameImportResult } from "./domain";
import { deduplicateHistory, normalizeHistoryStatus } from "./import-history";
import type { HistoryEntry, HistoryScoreScale } from "./import-history";
import { preferenceFromHistory } from "./preferences";
import { throwIfAborted } from "./runtime";
import type { RuntimePorts } from "./runtime";
import { createProviderScheduler, type Provider, type ProviderScheduler } from "../../shared/provider-scheduler";

const USERNAME_IMPORT_PAGE_SIZE = 300;

function providerScore(value: unknown, scale: HistoryScoreScale): number | null {
  if (value === null || value === undefined || value === 0) return null;
  const maximum = scale === "POINT_100" ? 100 : scale === "POINT_5" ? 5 : scale === "POINT_3" ? 3 : 10;
  const integerOnly = scale !== "POINT_10_DECIMAL" && scale !== "local-10";
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > maximum ||
    (integerOnly && !Number.isInteger(value)) ||
    (scale === "POINT_10_DECIMAL" && !Number.isInteger(value * 10))) {
    throw new Error("Provider returned an invalid list score; import was not applied.");
  }
  return value;
}

function providerProgress(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0 || value > 1_000_000) {
    throw new Error("Provider returned invalid episode progress; import was not applied.");
  }
  return value;
}

function mappedPreferenceEntries(history: HistoryEntry[], index: RecommendationIndex): {
  entries: ImportedPreferenceEntry[]; ratedCount: number; unmappedCount: number;
} {
  const byNodeId = new Map<string, ImportedPreferenceEntry>();
  let ratedCount = 0;
  let unmappedCount = 0;
  for (const entry of history) {
    if (entry.score !== null) ratedCount += 1;
    if (entry.status === "plan_to_watch") continue;
    const anime = entry.animeId === null ? undefined : index.animeByAnimeId.get(entry.animeId);
    if (!anime) {
      unmappedCount += 1;
      continue;
    }
    const preference = preferenceFromHistory(entry, anime.nodeId);
    if (preference) byNodeId.set(anime.nodeId, { anime, preference });
  }
  return { entries: [...byNodeId.values()], ratedCount, unmappedCount };
}

export type MetadataReadResult =
  | { state: "ready"; metadata: AnimeMetadata }
  | { state: "unavailable" }
  | { state: "failed" };

export class ProviderUnavailableError extends Error {
  readonly name = "ProviderUnavailableError";
}

export function createProviderAdapter(
  runtime: RuntimePorts,
  scheduler: ProviderScheduler = createProviderScheduler({
    fetch: (url, init) => runtime.fetch(url, init),
    monotonicNow: () => runtime.monotonicNow(),
    wallNow: () => runtime.now().getTime(),
    sleep: (ms, signal) => runtime.sleep(ms, signal),
    random: () => runtime.random(),
  }),
) {
  async function fetchAniListUsernameImport(
    username: string,
    index: RecommendationIndex,
    signal?: AbortSignal,
  ): Promise<UsernameImportResult> {
    const query = `
      query ($userName: String) {
        User(name: $userName) { mediaListOptions { scoreFormat } }
        MediaListCollection(userName: $userName, type: ANIME) {
          lists {
            entries {
              score
              status
              progress
              media {
                id
                idMal
                title { romaji }
              }
            }
          }
        }
      }
    `;

    const payload = await fetchJsonWithRetries<{
      data?: {
        User?: { mediaListOptions?: { scoreFormat?: string | null } | null } | null;
        MediaListCollection?: {
          lists?: Array<{
            entries?: Array<{
              score?: number;
              status?: string;
              progress?: number;
              media?: {
                id?: number;
                idMal?: number | null;
                title?: { romaji?: string | null } | null;
              } | null;
            }>;
          }>;
        } | null;
      };
      errors?: Array<{ message?: string }>;
    }>("anilist", "https://graphql.anilist.co", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        query,
        variables: { userName: username },
      }),
    }, signal);

    if (payload.errors && payload.errors.length > 0) {
      const message = payload.errors[0]?.message ?? "AniList API returned an error.";
      throw new Error(message);
    }

    const lists = payload.data?.MediaListCollection?.lists;
    if (!lists) throw new ProviderUnavailableError("No AniList anime list was available for that user.");

    const rawScoreScale = payload.data?.User?.mediaListOptions?.scoreFormat;
    const scoreScales = new Set<HistoryScoreScale>(
      ["POINT_100", "POINT_10_DECIMAL", "POINT_10", "POINT_5", "POINT_3"],
    );
    if (lists.some((list) => (list.entries?.length ?? 0) > 0) &&
      !scoreScales.has(rawScoreScale as HistoryScoreScale)) {
      throw new Error("AniList did not return a recognized user score format; import was not applied.");
    }
    const scoreScale = rawScoreScale as HistoryScoreScale;

    const history: HistoryEntry[] = [];
    for (const list of lists) {
      const entries = list.entries ?? [];
      for (const entry of entries) {
        const sourceId = entry.media?.id;
        if (!Number.isSafeInteger(sourceId) || (sourceId ?? 0) <= 0) {
          throw new Error("AniList returned an entry without a media ID; import was not applied.");
        }
        const malId = entry.media?.idMal;
        const animeId = Number.isSafeInteger(malId) && (malId ?? 0) > 0 ? malId! : null;
        const sourceStatus = entry.status ?? "";
        if (typeof sourceStatus !== "string") throw new Error("Invalid AniList status.");
        history.push({
          provider: "anilist", sourceId: String(sourceId),
          title: entry.media?.title?.romaji?.trim() ||
            (animeId === null ? `AniList anime ${sourceId}` : index.animeByAnimeId.get(animeId)?.label || `AniList anime ${sourceId}`),
          animeId, status: normalizeHistoryStatus(sourceStatus), sourceStatus,
          progressEpisodes: providerProgress(entry.progress),
          score: providerScore(entry.score, scoreScale), scoreScale,
        });
      }
    }
    const parsed = deduplicateHistory(history);
    return { ...mappedPreferenceEntries(parsed.entries, index), history: parsed.entries,
      duplicateCount: parsed.duplicates };
  }

  async function fetchMalUsernameImport(
    username: string,
    index: RecommendationIndex,
    signal?: AbortSignal,
  ): Promise<UsernameImportResult> {
    const allEntries: Array<{
      anime_id?: number; anime_title?: string; score?: number;
      status?: string | number; my_status?: string | number;
      num_watched_episodes?: number; my_watched_episodes?: number;
    }> = [];
    let offset = 0;

    while (true) {
      throwIfAborted(signal);
      const page = await fetchMalUsernamePage(username, offset, signal);
      if (!Array.isArray(page) || page.length === 0) {
        break;
      }

      allEntries.push(...page);
      if (page.length < USERNAME_IMPORT_PAGE_SIZE) {
        break;
      }

      offset += page.length;
    }
    throwIfAborted(signal);

    const history: HistoryEntry[] = [];
    for (const entry of allEntries) {
      const animeId = entry.anime_id;
      if (!Number.isSafeInteger(animeId) || (animeId ?? 0) <= 0) {
        throw new Error("MAL returned an entry without a valid anime ID; import was not applied.");
      }
      const rawStatus = entry.status ?? entry.my_status;
      const sourceStatus = rawStatus === undefined ? "" : String(rawStatus);
      history.push({
        provider: "mal", sourceId: String(animeId),
        title: entry.anime_title?.trim() || index.animeByAnimeId.get(animeId!)?.label || `MAL anime ${animeId}`,
        animeId: animeId!, status: normalizeHistoryStatus(sourceStatus), sourceStatus,
        progressEpisodes: providerProgress(entry.num_watched_episodes ?? entry.my_watched_episodes),
        score: providerScore(entry.score, "mal-10"), scoreScale: "mal-10",
      });
    }
    const parsed = deduplicateHistory(history);
    return { ...mappedPreferenceEntries(parsed.entries, index), history: parsed.entries,
      duplicateCount: parsed.duplicates };
  }

  async function fetchMalUsernamePage(
    username: string,
    offset: number,
    signal?: AbortSignal,
  ): Promise<Array<{
    anime_id?: number; anime_title?: string; score?: number;
    status?: string | number; my_status?: string | number;
    num_watched_episodes?: number; my_watched_episodes?: number;
  }>> {
    const malUrl = new URL(
      `https://myanimelist.net/animelist/${encodeURIComponent(username)}/load.json`,
    );
    malUrl.searchParams.set("status", "7");
    malUrl.searchParams.set("offset", String(offset));

    try {
      const page = await fetchJsonWithRetries<unknown>(
        "mal",
        malUrl.toString(),
        {
          headers: {
            Accept: "application/json",
          },
        },
        signal,
      );
      if (!Array.isArray(page)) {
        throw new Error("Unexpected MAL response shape.");
      }
      return page as Array<{
        anime_id?: number; anime_title?: string; score?: number;
        status?: string | number; my_status?: string | number;
        num_watched_episodes?: number; my_watched_episodes?: number;
      }>;
    } catch (error) {
      throwIfAborted(signal);
      const message = "Direct MAL import failed. Browser access may be blocked, the profile may be private, or MAL may be rate limiting. No proxy was contacted. Try the local file or text import above, or AniList.";
      if (error instanceof ProviderUnavailableError) {
        throw new ProviderUnavailableError(message);
      }
      throw new Error(message);
    }
  }

  async function fetchJsonWithRetries<T>(
    provider: Provider,
    url: string,
    init?: RequestInit,
    signal?: AbortSignal,
  ): Promise<T> {
    const response = await scheduler.request(provider, url, init, { signal });
    throwIfAborted(signal);
    if (response.ok) {
      const payload = (await response.json()) as T;
      throwIfAborted(signal);
      return payload;
    }
    if (response.status === 403 || response.status === 404) {
      throw new ProviderUnavailableError(`Request unavailable (${response.status}) for ${url}`);
    }
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }

  async function fetchAnimeMetadataFromJikan(
    animeId: number,
    signal?: AbortSignal,
  ): Promise<MetadataReadResult> {
    const url = `https://api.jikan.moe/v4/anime/${animeId}/full`;

    try {
      const response = await scheduler.request("jikan", url, {
        headers: { Accept: "application/json" },
      }, { signal });
      throwIfAborted(signal);
      if (response.ok) {
        const payload = (await response.json()) as { data?: Record<string, unknown> };
        throwIfAborted(signal);
        const metadata = parseAnimeMetadataPayload(animeId, payload.data);
        return metadata ? { state: "ready", metadata } : { state: "failed" };
      }
      return response.status === 403 || response.status === 404
        ? { state: "unavailable" } : { state: "failed" };
    } catch {
      throwIfAborted(signal);
      return { state: "failed" };
    }
  }

  function parseAnimeMetadataPayload(
    animeId: number,
    raw: Record<string, unknown> | undefined,
  ): AnimeMetadata | null {
    if (!raw || typeof raw !== "object") {
      return null;
    }

    const year =
      typeof raw.year === "number" && Number.isFinite(raw.year)
        ? Math.trunc(raw.year)
        : null;
    const score =
      typeof raw.score === "number" && Number.isFinite(raw.score)
        ? Number(raw.score)
        : null;
    const synopsis = typeof raw.synopsis === "string" ? raw.synopsis.trim() : "";
    const season = typeof raw.season === "string" ? raw.season : null;
    const genres = parseNameList(raw.genres);
    const studios = parseNameList(raw.studios);
    const relations = parseAnimeRelations(raw.relations, animeId);

    const images = raw.images as Record<string, unknown> | undefined;
    const webp = images?.webp as Record<string, unknown> | undefined;
    const jpg = images?.jpg as Record<string, unknown> | undefined;
    const imageUrl = [
      webp?.large_image_url,
      webp?.image_url,
      jpg?.large_image_url,
      jpg?.image_url,
    ].find((value): value is string => typeof value === "string" && value.length > 0);

    return {
      animeId,
      year,
      score,
      genres,
      studios,
      synopsis,
      imageUrl: imageUrl ?? "",
      season,
      relations,
    };
  }

  function parseAnimeRelations(raw: unknown, animeId: number): AnimeRelation[] | null {
    if (!Array.isArray(raw)) return null;
    const kinds = new Map<string, AnimeRelation["kind"]>([
      ["prequel", "prequel"], ["sequel", "sequel"],
      ["alternative version", "alternative-version"], ["side story", "side-story"],
      ["spin-off", "spin-off"], ["spin off", "spin-off"],
    ]);
    const relations = new Map<string, AnimeRelation>();
    for (const group of raw) {
      if (!group || typeof group !== "object" || Array.isArray(group)) return null;
      const relation = group as Record<string, unknown>;
      if (typeof relation.relation !== "string" || !Array.isArray(relation.entry)) return null;
      const kind = kinds.get(relation.relation.trim().toLowerCase());
      if (!kind) continue;
      for (const entry of relation.entry) {
        if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
        const related = entry as Record<string, unknown>;
        if (typeof related.type !== "string" || related.type.toLowerCase() !== "anime") continue;
        const id = related.mal_id;
        const title = related.name;
        if (!Number.isSafeInteger(id) || (id as number) <= 0 || id === animeId ||
            typeof title !== "string" || !title.trim()) return null;
        relations.set(`${kind}:${id}`, { kind, animeId: id as number, title: title.trim() });
      }
    }
    return [...relations.values()];
  }

  function parseNameList(raw: unknown): string[] {
    if (!Array.isArray(raw)) {
      return [];
    }
    const values: string[] = [];
    for (const entry of raw) {
      if (!entry || typeof entry !== "object") {
        continue;
      }
      const name = (entry as Record<string, unknown>).name;
      if (typeof name !== "string") {
        continue;
      }
      const trimmed = name.trim();
      if (!trimmed || values.includes(trimmed)) {
        continue;
      }
      values.push(trimmed);
    }
    return values;
  }

  async function fetchSeasonalAnime(limit: number, signal?: AbortSignal): Promise<SeasonalAnimeItem[]> {
    const payload = await fetchJsonWithRetries<{
      data?: Array<Record<string, unknown>>;
    }>(
      "jikan",
      `https://api.jikan.moe/v4/seasons/now?limit=${limit}`,
      { headers: { Accept: "application/json" } },
      signal,
    );
    const incoming = Array.isArray(payload.data) ? payload.data : [];
    return incoming
      .map((entry) => {
        const animeId = Number(entry.mal_id ?? 0);
        const title = String(entry.title ?? "").trim();
        if (!Number.isFinite(animeId) || animeId <= 0 || !title) {
          return null;
        }
        const scoreRaw = entry.score;
        const score = typeof scoreRaw === "number" && Number.isFinite(scoreRaw) ? scoreRaw : null;
        const yearRaw = entry.year;
        const year = typeof yearRaw === "number" && Number.isFinite(yearRaw)
          ? Math.trunc(yearRaw) : null;
        const season = typeof entry.season === "string" ? entry.season : null;
        const imageUrl = parseAnimeMetadataPayload(animeId, entry)?.imageUrl ?? "";
        return { animeId, title, score, year, season, imageUrl } satisfies SeasonalAnimeItem;
      })
      .filter((entry): entry is SeasonalAnimeItem => entry !== null);
  }

  return {
    fetchAniListUsernameImport,
    fetchMalUsernameImport,
    fetchAnimeMetadataFromJikan,
    fetchSeasonalAnime,
  };
}
