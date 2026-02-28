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
): Promise<string[]> {
  const url = new URL(`https://api.jikan.moe/v4/anime/${animeId}/userupdates`);
  url.searchParams.set("page", String(page));

  const json = await fetchJsonWithRetry<JikanAnimeUserUpdatesResponse>(
    url.toString(),
    `Jikan anime/${animeId}/userupdates page=${page}`,
  );

  return json.data
    .map((entry) => entry.user.username.trim())
    .filter((username) => username.length > 0);
}

export async function fetchJikanUsersPage(page: number): Promise<string[]> {
  const url = new URL("https://api.jikan.moe/v4/users");
  url.searchParams.set("page", String(page));

  const json = await fetchJsonWithRetry<JikanUsersResponse>(
    url.toString(),
    `Jikan users page=${page}`,
  );

  return json.data
    .map((entry) => entry.username.trim())
    .filter((username) => username.length > 0);
}

async function fetchJsonWithRetry<T>(
  url: string,
  label: string,
  maxRetries = 4,
): Promise<T> {
  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "WhatAnimeShouldIWatch/0.1",
      },
    });

    if (response.ok) {
      return (await response.json()) as T;
    }

    const retryable = response.status === 429 || response.status >= 500;
    if (retryable && attempt < maxRetries) {
      const retryAfterHeader = response.headers.get("retry-after");
      const retryAfterSeconds = retryAfterHeader
        ? Number.parseInt(retryAfterHeader, 10)
        : Number.NaN;

      const backoffMs = Number.isNaN(retryAfterSeconds)
        ? Math.min(1000 * 2 ** attempt, 10_000)
        : Math.max(retryAfterSeconds, 1) * 1000;

      await sleep(backoffMs);
      continue;
    }

    const body = await response.text();
    throw new Error(
      `${label} failed: ${response.status} ${response.statusText} ${body}`,
    );
  }

  throw new Error(`${label} exhausted retries`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
