import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ProviderScheduler,
  RequestBudgetExceededError,
  RequestTimeoutError,
  parseRetryAfter,
  type SchedulerPorts,
} from "../../shared/provider-scheduler.js";
import { fetchJikanAnimeUserUpdates, fetchJikanUsersPage } from "../src/jikan.js";
import { fetchMalPage } from "../src/mal.js";

const epoch = Date.parse("2026-09-24T12:00:00.000Z");
const response = (status = 200, headers?: HeadersInit) => new Response("{}", { status, headers });

function fakeScheduler(responder: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  let elapsed = 0;
  const starts: Array<{ url: string; at: number; signal?: AbortSignal }> = [];
  const sleeps: number[] = [];
  const ports: SchedulerPorts = {
    fetch: async (url, init) => {
      starts.push({ url, at: elapsed, signal: init?.signal ?? undefined });
      return responder(url, init);
    },
    monotonicNow: () => elapsed,
    wallNow: () => epoch + elapsed,
    sleep: async (ms, signal) => {
      if (signal?.aborted) throw Object.assign(new Error("canceled"), { name: "AbortError" });
      sleeps.push(ms);
      elapsed += ms;
      await Promise.resolve();
      if (signal?.aborted) throw Object.assign(new Error("canceled"), { name: "AbortError" });
    },
    random: () => 0.5,
    setTimer: () => () => {},
  };
  return { scheduler: new ProviderScheduler(ports), starts, sleeps, ports };
}

test("Retry-After accepts seconds and HTTP dates but rejects invalid values", () => {
  assert.equal(parseRetryAfter("2", epoch), 2_000);
  assert.equal(parseRetryAfter(new Date(epoch + 5_000).toUTCString(), epoch), 5_000);
  assert.equal(parseRetryAfter("bad", epoch), null);
  assert.equal(parseRetryAfter("-1", epoch), null);
});

test("metadata and seasonal requests share Jikan's second and minute windows", async () => {
  const fake = fakeScheduler(() => response());
  await Promise.all(Array.from({ length: 65 }, (_unused, index) => fake.scheduler.request(
    "jikan", index % 2 ? "https://api.jikan.moe/v4/seasons/now"
      : `https://api.jikan.moe/v4/anime/${index}/full`,
    {}, { retryBudgetMs: 120_000 },
  )));
  assert.equal(fake.starts.length, 65);
  const times = fake.starts.map((start) => start.at).sort((a, b) => a - b);
  for (const at of times) {
    assert.ok(times.filter((time) => time >= at && time < at + 1_000).length <= 3);
    assert.ok(times.filter((time) => time >= at && time < at + 60_000).length <= 60);
  }
  assert.ok(times[60] >= 60_000);
});

test("Retry-After and AniList reset headers delay later requests", async () => {
  let jikanAttempts = 0;
  const jikan = fakeScheduler(() => ++jikanAttempts === 1
    ? response(429, { "Retry-After": "2" }) : response());
  assert.equal((await jikan.scheduler.request("jikan", "https://api.jikan.moe/v4/anime/101/full")).status, 200);
  assert.deepEqual(jikan.starts.map((start) => start.at), [0, 2_000]);

  const anilist = fakeScheduler(() => response(200, {
    "X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": String((epoch + 5_000) / 1_000),
  }));
  await anilist.scheduler.request("anilist", "https://graphql.anilist.co");
  await anilist.scheduler.request("anilist", "https://graphql.anilist.co");
  assert.deepEqual(anilist.starts.map((start) => start.at), [0, 5_000]);
});

test("a Retry-After longer than the retry budget is not bypassed", async () => {
  const fake = fakeScheduler(() => response(429, { "Retry-After": "120" }));
  assert.equal((await fake.scheduler.request("jikan", "https://api.jikan.moe/v4/users")).status, 429);
  await assert.rejects(fake.scheduler.request("jikan", "https://api.jikan.moe/v4/users"),
    RequestBudgetExceededError);
  assert.equal(fake.starts.length, 1);
});

test("a Jikan cooldown does not delay a separate provider bucket", async () => {
  const fake = fakeScheduler((url) => url.includes("api.jikan.moe")
    ? response(429, { "Retry-After": "120" }) : response());
  await fake.scheduler.request("jikan", "https://api.jikan.moe/v4/users", {}, { maxAttempts: 1 });
  await fake.scheduler.request("mal", "https://myanimelist.net/fixture", {}, { maxAttempts: 1 });
  assert.deepEqual(fake.starts.map((start) => start.at), [0, 0]);
});

test("a rejected method is not retried", async () => {
  const fake = fakeScheduler(() => response(405));
  assert.equal((await fake.scheduler.request("mal", "https://myanimelist.net/fixture")).status, 405);
  assert.equal(fake.starts.length, 1);
  assert.deepEqual(fake.sleeps, []);
});

test("Jikan's in-flight cap queues a fourth call", async () => {
  const releases: Array<() => void> = [];
  const fake = fakeScheduler(() => new Promise<Response>((resolve) => {
    releases.push(() => resolve(response()));
  }));
  const pending = Array.from({ length: 4 }, (_unused, index) => fake.scheduler.request(
    "jikan", `https://api.jikan.moe/v4/anime/${index}/full`,
  ));
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fake.starts.length, 3);
  releases.shift()?.();
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(fake.starts.length, 4);
  for (const release of releases) release();
  await Promise.all(pending);
});

test("timeout aborts a stalled fetch even when transport ignores abort", async () => {
  let fireTimer: (() => void) | undefined;
  let canceled = false;
  const fake = fakeScheduler(() => new Promise<Response>(() => {}));
  fake.ports.setTimer = (callback) => { fireTimer = callback; return () => { canceled = true; }; };
  const pending = fake.scheduler.request("jikan", "https://api.jikan.moe/v4/users", {}, { maxAttempts: 1 });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.ok(fireTimer);
  fireTimer();
  await assert.rejects(pending, RequestTimeoutError);
  assert.equal(fake.starts[0].signal?.aborted, true);
  assert.equal(canceled, true);
});

test("a queued request can be canceled without entering transport", async () => {
  let releaseFirst: (() => void) | undefined;
  const fake = fakeScheduler(() => new Promise<Response>((resolve) => {
    releaseFirst = () => resolve(response());
  }));
  const first = fake.scheduler.request("anilist", "https://graphql.anilist.co/first");
  await new Promise<void>((resolve) => setImmediate(resolve));
  const controller = new AbortController();
  const second = fake.scheduler.request("anilist", "https://graphql.anilist.co/second", {}, {
    signal: controller.signal,
  });
  controller.abort();
  await assert.rejects(second, { name: "AbortError" });
  assert.equal(fake.starts.length, 1);
  releaseFirst?.();
  await first;
});

test("crawler MAL pages and Jikan discovery use the scheduler with synthetic responses", async () => {
  const malPage = Array.from({ length: 300 }, (_unused, index) => ({
    anime_id: 1_000 + index, anime_title: `Invented ${index}`, score: 8,
  }));
  let usersAttempts = 0;
  const fake = fakeScheduler((rawUrl) => {
    const url = new URL(rawUrl);
    if (url.hostname === "myanimelist.net") {
      return new Response(JSON.stringify(url.searchParams.get("offset") === "0"
        ? malPage : [{ anime_id: 101, anime_title: "Copper Comet", score: 9 }]));
    }
    if (url.pathname === "/v4/users") {
      return ++usersAttempts === 1
        ? response(429, { "Retry-After": "2" })
        : new Response(JSON.stringify({ data: [{ username: "fixture-user" }] }));
    }
    if (url.pathname === "/v4/anime/101/userupdates") {
      return new Response(JSON.stringify({ data: [{ user: { username: "fixture-neighbor" } }] }));
    }
    throw new Error(`Unexpected synthetic URL: ${url.pathname}`);
  });
  const firstPage = await fetchMalPage("fixture-user", 0, 1_200, undefined, fake.scheduler);
  const secondPage = await fetchMalPage("fixture-user", 300, 1_200, undefined, fake.scheduler);
  assert.ok(Array.isArray(firstPage));
  assert.equal(firstPage.length, 300);
  assert.deepEqual(secondPage, [{ anime_id: 101, anime_title: "Copper Comet", score: 9 }]);
  const malStarts = fake.starts.filter((start) => start.url.includes("myanimelist.net"));
  assert.equal(malStarts.length, 2);
  assert.ok(malStarts[1].at - malStarts[0].at >= 1_200);

  assert.deepEqual(await fetchJikanUsersPage(1, 450, undefined, fake.scheduler), ["fixture-user"]);
  assert.deepEqual(await fetchJikanAnimeUserUpdates(101, 1, 450, undefined, fake.scheduler), ["fixture-neighbor"]);
  const jikanStarts = fake.starts.filter((start) => start.url.includes("api.jikan.moe"));
  assert.equal(jikanStarts.length, 3);
  assert.ok(jikanStarts[1].at - jikanStarts[0].at >= 2_000);
  assert.ok(jikanStarts[2].at - jikanStarts[1].at >= 450);
});

test("crawler MAL and Jikan requests propagate cancellation into transport", async () => {
  for (const start of [
    (scheduler: ProviderScheduler, signal: AbortSignal) =>
      fetchMalPage("fixture-user", 0, 0, signal, scheduler),
    (scheduler: ProviderScheduler, signal: AbortSignal) =>
      fetchJikanUsersPage(1, 0, signal, scheduler),
  ]) {
    const fake = fakeScheduler(() => new Promise<Response>(() => {}));
    const controller = new AbortController();
    const pending = start(fake.scheduler, controller.signal);
    await new Promise<void>((resolve) => setImmediate(resolve));
    assert.equal(fake.starts.length, 1);
    controller.abort();
    await assert.rejects(pending, { name: "AbortError" });
    assert.equal(fake.starts[0].signal?.aborted, true);
  }
});
