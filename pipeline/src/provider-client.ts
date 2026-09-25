import { createProviderScheduler } from "../../shared/provider-scheduler.js";

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(Object.assign(new Error("Request canceled"), { name: "AbortError" }));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(Object.assign(new Error("Request canceled"), { name: "AbortError" }));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** One coordinator for every MAL/Jikan call in this crawler process. */
export const crawlerProviderScheduler = createProviderScheduler({
  fetch: (url, init) => fetch(url, init),
  monotonicNow: () => performance.now(),
  wallNow: () => Date.now(),
  sleep,
  random: () => Math.random(),
});
