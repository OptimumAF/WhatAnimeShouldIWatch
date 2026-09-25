import type { ProviderScheduler } from "../../shared/provider-scheduler.js";
import { crawlerProviderScheduler } from "./provider-client.js";

export class MalPageHttpError extends Error {
  constructor(readonly status: number) {
    super(`MAL page request returned HTTP ${status}`);
    this.name = "MalPageHttpError";
  }
}

export async function fetchMalPage(
  username: string,
  offset: number,
  delayMs: number,
  signal?: AbortSignal,
  scheduler: ProviderScheduler = crawlerProviderScheduler,
): Promise<unknown> {
  const url = new URL(
    `https://myanimelist.net/animelist/${encodeURIComponent(username)}/load.json`,
  );
  url.searchParams.set("status", "7");
  url.searchParams.set("offset", String(offset));

  const response = await scheduler.request("mal", url.toString(), {
    headers: { "User-Agent": "WhatAnimeShouldIWatch/0.1" },
  }, { minIntervalMs: delayMs, maxAttempts: 6, retryBudgetMs: 120_000, signal });
  if (!response.ok) throw new MalPageHttpError(response.status);
  return response.json() as Promise<unknown>;
}
