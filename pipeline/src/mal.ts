import type { MalAnimeEntry } from "./types.js";
import type { ProviderScheduler } from "../../shared/provider-scheduler.js";
import { crawlerProviderScheduler } from "./provider-client.js";

const MAL_PAGE_SIZE = 300;

export async function fetchMalRatings(
  username: string,
  delayMs: number,
  maxPages = 0,
  signal?: AbortSignal,
  scheduler: ProviderScheduler = crawlerProviderScheduler,
): Promise<MalAnimeEntry[]> {
  const ratings: MalAnimeEntry[] = [];
  let offset = 0;
  let pageCount = 0;

  while (true) {
    if (signal?.aborted) throw Object.assign(new Error("Collection canceled"), { name: "AbortError" });
    if (maxPages > 0 && pageCount >= maxPages) {
      break;
    }

    const url = new URL(
      `https://myanimelist.net/animelist/${encodeURIComponent(username)}/load.json`,
    );
    url.searchParams.set("status", "7");
    url.searchParams.set("offset", String(offset));

    const page = await fetchMalPage(username, offset, url, scheduler, delayMs, signal);
    if (page.length === 0) {
      break;
    }
    pageCount += 1;

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
  }

  return ratings;
}

async function fetchMalPage(
  username: string,
  offset: number,
  url: URL,
  scheduler: ProviderScheduler,
  delayMs: number,
  signal?: AbortSignal,
): Promise<MalAnimeEntry[]> {
  const response = await scheduler.request("mal", url.toString(), {
    headers: { "User-Agent": "WhatAnimeShouldIWatch/0.1" },
  }, { minIntervalMs: delayMs, maxAttempts: 6, retryBudgetMs: 120_000, signal });
  if (response.ok) {
    return (await response.json()) as MalAnimeEntry[];
  }
  throw new Error(
    `MAL request failed for "${username}" at offset ${offset}: ${response.status} ${response.statusText}`,
  );
}
