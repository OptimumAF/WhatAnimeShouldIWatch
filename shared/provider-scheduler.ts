/** Per-runtime provider pacing. All retries reserve a fresh provider slot. */
export type Provider = "anilist" | "jikan" | "mal";

export interface RateWindow {
  limit: number;
  periodMs: number;
}

export interface ProviderPolicy {
  windows: RateWindow[];
  maxConcurrent: number;
  timeoutMs: number;
  maxAttempts: number;
  retryBudgetMs: number;
  backoffBaseMs: number;
  backoffMaxMs: number;
  jitterMs: number;
}

export interface SchedulerPorts {
  fetch(url: string, init?: RequestInit): Promise<Response>;
  monotonicNow(): number;
  wallNow(): number;
  sleep(ms: number, signal?: AbortSignal): Promise<void>;
  random(): number;
  setTimer?(callback: () => void, ms: number): () => void;
}

export interface RequestOptions {
  signal?: AbortSignal;
  minIntervalMs?: number;
  timeoutMs?: number;
  maxAttempts?: number;
  retryBudgetMs?: number;
}

const DEFAULT_POLICIES: Record<Provider, ProviderPolicy> = {
  // AniList documents a temporary 30/minute limit and an additional burst limiter.
  anilist: {
    windows: [{ limit: 1, periodMs: 2_000 }, { limit: 30, periodMs: 60_000 }],
    maxConcurrent: 1, timeoutMs: 10_000, maxAttempts: 3, retryBudgetMs: 30_000,
    backoffBaseMs: 1_000, backoffMaxMs: 10_000, jitterMs: 250,
  },
  // Jikan documents 3/second and 60/minute. Its upstream MAL can also rate-limit it.
  jikan: {
    windows: [{ limit: 3, periodMs: 1_000 }, { limit: 60, periodMs: 60_000 }],
    maxConcurrent: 3, timeoutMs: 10_000, maxAttempts: 3, retryBudgetMs: 30_000,
    backoffBaseMs: 800, backoffMaxMs: 10_000, jitterMs: 250,
  },
  // The current MAL site route has no documented application quota or permission.
  // This conservative local floor is not a claim that the route is authorized.
  mal: {
    windows: [{ limit: 1, periodMs: 1_000 }],
    maxConcurrent: 1, timeoutMs: 10_000, maxAttempts: 4, retryBudgetMs: 45_000,
    backoffBaseMs: 1_000, backoffMaxMs: 20_000, jitterMs: 350,
  },
};

interface Waiter {
  resolve(): void;
  reject(error: Error): void;
  signal?: AbortSignal;
  onAbort?: () => void;
}

interface ProviderState {
  active: number;
  waiters: Waiter[];
  starts: number[];
  cooldownUntil: number;
  minIntervalMs: number;
  advertisedMinuteLimit: number | null;
}

export class RequestTimeoutError extends Error {
  readonly name = "RequestTimeoutError";
}

export class RequestBudgetExceededError extends Error {
  readonly name = "RequestBudgetExceededError";
}

function abortError(): Error {
  const error = new Error("Request canceled");
  error.name = "AbortError";
  return error;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError();
}

export function parseRetryAfter(value: string | null, wallNowMs: number): number | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (/^\d+(?:\.\d+)?$/.test(trimmed)) {
    const ms = Number(trimmed) * 1_000;
    return Number.isFinite(ms) ? ms : null;
  }
  if (/^[+-]\d+(?:\.\d+)?$/.test(trimmed)) return null;
  const date = Date.parse(trimmed);
  return Number.isFinite(date) ? Math.max(0, date - wallNowMs) : null;
}

function retryable(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

export class ProviderScheduler {
  private readonly states: Record<Provider, ProviderState>;
  private readonly ports: SchedulerPorts;
  private readonly policies: Record<Provider, ProviderPolicy>;

  constructor(
    ports: SchedulerPorts,
    policies: Record<Provider, ProviderPolicy> = DEFAULT_POLICIES,
  ) {
    this.ports = ports;
    this.policies = policies;
    const createState = (): ProviderState => ({
      active: 0, waiters: [], starts: [], cooldownUntil: 0,
      minIntervalMs: 0, advertisedMinuteLimit: null,
    });
    this.states = { anilist: createState(), jikan: createState(), mal: createState() };
  }

  async request(
    provider: Provider,
    url: string,
    init: RequestInit = {},
    options: RequestOptions = {},
  ): Promise<Response> {
    const policy = this.policies[provider];
    const state = this.states[provider];
    const signal = options.signal ?? init.signal ?? undefined;
    const maxAttempts = options.maxAttempts ?? policy.maxAttempts;
    const timeoutMs = options.timeoutMs ?? policy.timeoutMs;
    const retryBudgetMs = options.retryBudgetMs ?? policy.retryBudgetMs;
    if (!Number.isInteger(maxAttempts) || maxAttempts < 1 ||
        !Number.isFinite(timeoutMs) || timeoutMs <= 0 ||
        !Number.isFinite(retryBudgetMs) || retryBudgetMs <= 0) {
      throw new Error("Invalid provider request limits");
    }
    if (options.minIntervalMs !== undefined) {
      if (!Number.isFinite(options.minIntervalMs) || options.minIntervalMs < 0) {
        throw new Error("Invalid provider interval");
      }
      state.minIntervalMs = Math.max(state.minIntervalMs, options.minIntervalMs);
    }
    const deadline = this.ports.monotonicNow() + retryBudgetMs;
    const release = await this.acquire(provider, signal);
    try {
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        await this.waitForRate(provider, deadline, signal);
        let response: Response;
        try {
          response = await this.fetchWithTimeout(url, init, timeoutMs, signal);
        } catch (error) {
          throwIfAborted(signal);
          if (attempt + 1 >= maxAttempts) throw error;
          const delay = this.backoff(policy, attempt);
          if (!this.fitsBudget(deadline, delay)) throw error;
          await this.ports.sleep(delay, signal);
          throwIfAborted(signal);
          continue;
        }

        const retryAfterMs = this.observe(provider, response);
        if (!retryable(response.status) || attempt + 1 >= maxAttempts) return response;
        const delay = Math.max(this.backoff(policy, attempt), retryAfterMs ?? 0);
        if (!this.fitsBudget(deadline, delay)) return response;
        void response.body?.cancel().catch(() => {});
        await this.ports.sleep(delay, signal);
        throwIfAborted(signal);
      }
      throw new Error("Provider retry budget exhausted");
    } finally {
      release();
    }
  }

  private backoff(policy: ProviderPolicy, attempt: number): number {
    const jitter = Math.floor(Math.min(1, Math.max(0, this.ports.random())) * policy.jitterMs);
    return Math.min(policy.backoffBaseMs * 2 ** attempt, policy.backoffMaxMs) + jitter;
  }

  private fitsBudget(deadline: number, delay: number): boolean {
    return this.ports.monotonicNow() + delay < deadline;
  }

  private async acquire(provider: Provider, signal?: AbortSignal): Promise<() => void> {
    throwIfAborted(signal);
    const state = this.states[provider];
    const policy = this.policies[provider];
    if (state.active < policy.maxConcurrent && state.waiters.length === 0) {
      state.active += 1;
    } else {
      await new Promise<void>((resolve, reject) => {
        const waiter: Waiter = { resolve, reject, signal };
        waiter.onAbort = () => {
          const index = state.waiters.indexOf(waiter);
          if (index >= 0) state.waiters.splice(index, 1);
          reject(abortError());
        };
        signal?.addEventListener("abort", waiter.onAbort, { once: true });
        state.waiters.push(waiter);
      });
    }
    return () => {
      state.active -= 1;
      while (state.waiters.length > 0) {
        const waiter = state.waiters.shift()!;
        waiter.signal?.removeEventListener("abort", waiter.onAbort!);
        if (waiter.signal?.aborted) {
          waiter.reject(abortError());
          continue;
        }
        state.active += 1;
        waiter.resolve();
        break;
      }
    };
  }

  private async waitForRate(provider: Provider, deadline: number, signal?: AbortSignal): Promise<void> {
    const state = this.states[provider];
    const policy = this.policies[provider];
    while (true) {
      throwIfAborted(signal);
      const now = this.ports.monotonicNow();
      if (now >= deadline) throw new RequestBudgetExceededError("Provider request waited past its budget");
      const windows = state.minIntervalMs > 0
        ? [...policy.windows, { limit: 1, periodMs: state.minIntervalMs }]
        : policy.windows;
      const longest = Math.max(...windows.map((window) => window.periodMs));
      state.starts = state.starts.filter((start) => start > now - longest);
      let next = Math.max(now, state.cooldownUntil);
      for (const window of windows) {
        const limit = provider === "anilist" && window.periodMs === 60_000 &&
          state.advertisedMinuteLimit !== null
          ? Math.min(window.limit, state.advertisedMinuteLimit) : window.limit;
        const recent = state.starts.filter((start) => start > now - window.periodMs);
        if (recent.length >= limit) {
          next = Math.max(next, recent[recent.length - limit] + window.periodMs);
        }
      }
      if (next <= now) {
        state.starts.push(now);
        return;
      }
      const delay = next - now;
      if (now + delay >= deadline) {
        throw new RequestBudgetExceededError("Provider rate wait exceeds its request budget");
      }
      await this.ports.sleep(delay, signal);
    }
  }

  private observe(provider: Provider, response: Response): number | null {
    const state = this.states[provider];
    const now = this.ports.monotonicNow();
    const wallNow = this.ports.wallNow();
    const retryAfter = response.status === 429 || response.status === 503
      ? parseRetryAfter(response.headers.get("retry-after"), wallNow) : null;
    if (retryAfter !== null) {
      state.cooldownUntil = Math.max(state.cooldownUntil, now + retryAfter);
    }
    if (provider === "anilist") {
      const advertised = Number(response.headers.get("x-ratelimit-limit"));
      if (Number.isInteger(advertised) && advertised > 0) {
        state.advertisedMinuteLimit = state.advertisedMinuteLimit === null
          ? advertised : Math.min(state.advertisedMinuteLimit, advertised);
      }
      if (response.headers.get("x-ratelimit-remaining") === "0") {
        const reset = Number(response.headers.get("x-ratelimit-reset"));
        if (Number.isFinite(reset) && reset > 0) {
          state.cooldownUntil = Math.max(state.cooldownUntil, now + Math.max(0, reset * 1_000 - wallNow));
        }
      }
    }
    return retryAfter;
  }

  private async fetchWithTimeout(
    url: string,
    init: RequestInit,
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<Response> {
    throwIfAborted(signal);
    const controller = new AbortController();
    let cancelTimer: (() => void) | undefined;
    let onAbort: (() => void) | undefined;
    const deadline = new Promise<never>((_resolve, reject) => {
      const setTimer = this.ports.setTimer ?? ((callback: () => void, ms: number) => {
        const timer = setTimeout(callback, ms);
        return () => clearTimeout(timer);
      });
      cancelTimer = setTimer(() => {
        controller.abort();
        reject(new RequestTimeoutError(`Provider request timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      onAbort = () => {
        controller.abort();
        reject(abortError());
      };
      signal?.addEventListener("abort", onAbort, { once: true });
    });
    try {
      return await Promise.race([
        this.ports.fetch(url, { ...init, signal: controller.signal }),
        deadline,
      ]);
    } finally {
      cancelTimer?.();
      if (onAbort) signal?.removeEventListener("abort", onAbort);
    }
  }
}

export function createProviderScheduler(ports: SchedulerPorts): ProviderScheduler {
  return new ProviderScheduler(ports);
}
