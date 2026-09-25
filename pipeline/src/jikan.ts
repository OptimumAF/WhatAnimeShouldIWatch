import type { ProviderScheduler } from "../../shared/provider-scheduler.js";
import { crawlerProviderScheduler } from "./provider-client.js";

interface JikanAnimeUserUpdatesResponse {
  data: {
    user: {
      username: string;
    };
  }[];
}

interface JikanUsersResponse {
  data: {
    username: string;
  }[];
}

export async function fetchJikanAnimeUserUpdates(
  animeId: number,
  page: number,
  minIntervalMs = 0,
  signal?: AbortSignal,
  scheduler: ProviderScheduler = crawlerProviderScheduler,
): Promise<string[]> {
  const url = new URL(`https://api.jikan.moe/v4/anime/${animeId}/userupdates`);
  url.searchParams.set("page", String(page));

  const json = await fetchJsonWithRetry<JikanAnimeUserUpdatesResponse>(
    url.toString(),
    `Jikan anime/${animeId}/userupdates page=${page}`,
    minIntervalMs,
    signal,
    scheduler,
  );

  return json.data
    .map((entry) => entry.user.username.trim())
    .filter((username) => username.length > 0);
}

export async function fetchJikanUsersPage(
  page: number,
  minIntervalMs = 0,
  signal?: AbortSignal,
  scheduler: ProviderScheduler = crawlerProviderScheduler,
): Promise<string[]> {
  const url = new URL("https://api.jikan.moe/v4/users");
  url.searchParams.set("page", String(page));

  const json = await fetchJsonWithRetry<JikanUsersResponse>(
    url.toString(),
    `Jikan users page=${page}`,
    minIntervalMs,
    signal,
    scheduler,
  );

  return json.data
    .map((entry) => entry.username.trim())
    .filter((username) => username.length > 0);
}

async function fetchJsonWithRetry<T>(
  url: string,
  label: string,
  minIntervalMs: number,
  signal: AbortSignal | undefined,
  scheduler: ProviderScheduler,
): Promise<T> {
  const response = await scheduler.request("jikan", url, {
    headers: { "User-Agent": "WhatAnimeShouldIWatch/0.1" },
  }, { minIntervalMs, maxAttempts: 5, retryBudgetMs: 120_000, signal });
  if (response.ok) {
    return (await response.json()) as T;
  }
  const body = await response.text();
  throw new Error(`${label} failed: ${response.status} ${response.statusText} ${body}`);
}
