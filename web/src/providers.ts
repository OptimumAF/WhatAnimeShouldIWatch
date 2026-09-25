/** Existing optional public-provider reads behind injectable transport and timing. */
import type { AnimeMetadata } from "./artifacts";
import type { ImportedWatchedEntry, RecommendationIndex, SeasonalAnimeItem, UsernameImportResult } from "./domain";
import { normalizeImportedScoreToWeight } from "./recommendations";
import type { RuntimePorts } from "./runtime";

const USERNAME_IMPORT_MAX_RETRIES = 3;
const USERNAME_IMPORT_PAGE_SIZE = 300;
const USERNAME_IMPORT_PAGE_DELAY_MS = 350;
const METADATA_MAX_RETRIES = 2;

export function createProviderAdapter(runtime: RuntimePorts) {
  async function fetchAniListUsernameImport(
    username: string,
    index: RecommendationIndex,
  ): Promise<UsernameImportResult> {
    const query = `
      query ($userName: String) {
        MediaListCollection(userName: $userName, type: ANIME) {
          lists {
            entries {
              score(format: POINT_10_DECIMAL)
              media {
                idMal
              }
            }
          }
        }
      }
    `;

    const payload = await fetchJsonWithRetries<{
      data?: {
        MediaListCollection?: {
          lists?: Array<{
            entries?: Array<{
              score?: number;
              media?: {
                idMal?: number | null;
              } | null;
            }>;
          }>;
        } | null;
      };
      errors?: Array<{ message?: string }>;
    }>("https://graphql.anilist.co", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        query,
        variables: { userName: username },
      }),
    });

    if (payload.errors && payload.errors.length > 0) {
      const message = payload.errors[0]?.message ?? "AniList API returned an error.";
      throw new Error(message);
    }

    const lists = payload.data?.MediaListCollection?.lists ?? [];
    if (lists.length === 0) {
      throw new Error(
        "No AniList anime list found for that user (private, empty, or not found).",
      );
    }

    let ratedCount = 0;
    let unmappedCount = 0;
    const byNodeId = new Map<string, ImportedWatchedEntry>();

    for (const list of lists) {
      const entries = list.entries ?? [];
      for (const entry of entries) {
        const score = Number(entry.score ?? 0);
        const malId = Number(entry.media?.idMal ?? 0);
        if (!Number.isFinite(score) || score <= 0 || !Number.isFinite(malId) || malId <= 0) {
          continue;
        }

        ratedCount += 1;
        const anime = index.animeByAnimeId.get(Math.trunc(malId));
        if (!anime) {
          unmappedCount += 1;
          continue;
        }

        const weight = normalizeImportedScoreToWeight(score);
        byNodeId.set(anime.nodeId, { anime, weight });
      }
    }

    return {
      entries: [...byNodeId.values()],
      ratedCount,
      unmappedCount,
    };
  }

  async function fetchMalUsernameImport(
    username: string,
    index: RecommendationIndex,
  ): Promise<UsernameImportResult> {
    const allEntries: Array<{ anime_id?: number; score?: number }> = [];
    let offset = 0;

    while (true) {
      const page = await fetchMalUsernamePage(username, offset);
      if (!Array.isArray(page) || page.length === 0) {
        break;
      }

      allEntries.push(...page);
      if (page.length < USERNAME_IMPORT_PAGE_SIZE) {
        break;
      }

      offset += page.length;
      await runtime.sleep(USERNAME_IMPORT_PAGE_DELAY_MS);
    }

    if (allEntries.length === 0) {
      throw new Error("No rated MAL entries found for that user (private, empty, or not found).");
    }

    let ratedCount = 0;
    let unmappedCount = 0;
    const byNodeId = new Map<string, ImportedWatchedEntry>();

    for (const entry of allEntries) {
      const score = Number(entry.score ?? 0);
      const animeId = Number(entry.anime_id ?? 0);
      if (!Number.isFinite(score) || score <= 0 || !Number.isFinite(animeId) || animeId <= 0) {
        continue;
      }

      ratedCount += 1;
      const anime = index.animeByAnimeId.get(Math.trunc(animeId));
      if (!anime) {
        unmappedCount += 1;
        continue;
      }

      const weight = normalizeImportedScoreToWeight(score);
      byNodeId.set(anime.nodeId, { anime, weight });
    }

    return {
      entries: [...byNodeId.values()],
      ratedCount,
      unmappedCount,
    };
  }

  async function fetchMalUsernamePage(
    username: string,
    offset: number,
  ): Promise<Array<{ anime_id?: number; score?: number }>> {
    const malUrl = new URL(
      `https://myanimelist.net/animelist/${encodeURIComponent(username)}/load.json`,
    );
    malUrl.searchParams.set("status", "7");
    malUrl.searchParams.set("offset", String(offset));

    try {
      const page = await fetchJsonWithRetries<unknown>(
        malUrl.toString(),
        {
          headers: {
            Accept: "application/json",
          },
        },
        USERNAME_IMPORT_MAX_RETRIES,
      );
      if (!Array.isArray(page)) {
        throw new Error("Unexpected MAL response shape.");
      }
      return page as Array<{ anime_id?: number; score?: number }>;
    } catch {
      throw new Error(
        "Direct MAL import failed. Browser access may be blocked, the profile may be private, or MAL may be rate limiting. No proxy was contacted. Try the local file or text import above, or AniList.",
      );
    }
  }

  async function fetchJsonWithRetries<T>(
    url: string,
    init?: RequestInit,
    maxRetries = USERNAME_IMPORT_MAX_RETRIES,
  ): Promise<T> {
    for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
      const response = await runtime.fetch(url, init);
      if (response.ok) {
        return (await response.json()) as T;
      }

      if (!isRetryableStatus(response.status) || attempt >= maxRetries) {
        throw new Error(`Request failed (${response.status}) for ${url}`);
      }

      await runtime.sleep(Math.min(1000 * 2 ** attempt, 5000) + Math.floor(runtime.random() * 200));
    }

    throw new Error("Request retries exhausted.");
  }

  function isRetryableStatus(status: number): boolean {
    return status === 408 || status === 425 || status === 429 || status >= 500;
  }

  async function fetchAnimeMetadataFromJikan(
    animeId: number,
  ): Promise<AnimeMetadata | null> {
    const url = `https://api.jikan.moe/v4/anime/${animeId}/full`;

    for (let attempt = 0; attempt <= METADATA_MAX_RETRIES; attempt += 1) {
      try {
        const response = await runtime.fetch(url, {
          headers: {
            Accept: "application/json",
          },
        });
        if (response.ok) {
          const payload = (await response.json()) as {
            data?: Record<string, unknown>;
          };
          return parseAnimeMetadataPayload(animeId, payload.data);
        }
        if (response.status === 404) {
          return null;
        }
        if (!isRetryableStatus(response.status) || attempt >= METADATA_MAX_RETRIES) {
          return null;
        }
      } catch {
        if (attempt >= METADATA_MAX_RETRIES) {
          return null;
        }
      }

      await runtime.sleep(Math.min(800 * 2 ** attempt, 5000) + Math.floor(runtime.random() * 220));
    }

    return null;
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
    };
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

  async function fetchSeasonalAnime(limit: number): Promise<SeasonalAnimeItem[]> {
    const payload = await fetchJsonWithRetries<{
      data?: Array<Record<string, unknown>>;
    }>(
      `https://api.jikan.moe/v4/seasons/now?limit=${limit}`,
      { headers: { Accept: "application/json" } },
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
