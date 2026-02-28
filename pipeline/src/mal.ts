import type { MalAnimeEntry } from "./types.js";

const MAL_PAGE_SIZE = 300;

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

    const response = await fetch(url, {
      headers: {
        "User-Agent": "WhatAnimeShouldIWatch/0.1",
      },
    });

    if (!response.ok) {
      throw new Error(
        `MAL request failed for "${username}" at offset ${offset}: ${response.status} ${response.statusText}`,
      );
    }

    const page = (await response.json()) as MalAnimeEntry[];
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
