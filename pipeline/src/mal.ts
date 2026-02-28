import type { MalAnimeEntry } from "./types.js";

const MAL_PAGE_SIZE = 300;
const MAL_MAX_RETRIES = 5;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function fetchMalRatings(
  username: string,
  delayMs: number,
): Promise<MalAnimeEntry[]> {
  const ratings: MalAnimeEntry[] = [];
  let offset = 0;

  while (true) {
    const url = new URL(
      `https://myanimelist.net/animelist/${encodeURIComponent(username)}/load.json`,
    );
    url.searchParams.set("status", "7");
    url.searchParams.set("offset", String(offset));

    const page = await fetchMalPage(username, offset, url);
    if (page.length === 0) {
      break;
    }

    for (const entry of page) {
      if (entry.score > 0) {
        ratings.push({
          anime_id: entry.anime_id,
          anime_title: entry.anime_title,
          score: entry.score,
        });
      }
    }

    if (page.length < MAL_PAGE_SIZE) {
      break;
    }

    offset += page.length;
    if (delayMs > 0) {
      await sleep(delayMs);
    }
  }

  return ratings;
}

async function fetchMalPage(
  username: string,
  offset: number,
  url: URL,
): Promise<MalAnimeEntry[]> {
  for (let attempt = 0; attempt <= MAL_MAX_RETRIES; attempt += 1) {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "WhatAnimeShouldIWatch/0.1",
      },
    });

    if (response.ok) {
      return (await response.json()) as MalAnimeEntry[];
    }

    const retryable = isRetryableStatus(response.status);
    if (retryable && attempt < MAL_MAX_RETRIES) {
      const retryAfterHeader = response.headers.get("retry-after");
      const retryAfterSeconds = retryAfterHeader
        ? Number.parseInt(retryAfterHeader, 10)
        : Number.NaN;

      const baseBackoffMs = Number.isNaN(retryAfterSeconds)
        ? Math.min(1000 * 2 ** attempt, 20_000)
        : Math.max(retryAfterSeconds, 1) * 1000;

      const jitterMs = Math.floor(Math.random() * 350);
      await sleep(baseBackoffMs + jitterMs);
      continue;
    }

    throw new Error(
      `MAL request failed for "${username}" at offset ${offset}: ${response.status} ${response.statusText}`,
    );
  }

  throw new Error(
    `MAL request failed for "${username}" at offset ${offset}: retries exhausted`,
  );
}

function isRetryableStatus(status: number): boolean {
  return status === 405 || status === 408 || status === 429 || status >= 500;
}
