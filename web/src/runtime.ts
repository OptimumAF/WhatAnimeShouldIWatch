/** Side-effect ports used by data and persistence adapters. */
export interface StoragePort {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface RuntimePorts {
  fetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response>;
  storage: StoragePort;
  now(): Date;
  monotonicNow(): number;
  random(): number;
  sleep(ms: number): Promise<void>;
  schedule(callback: () => void, ms: number): void;
  frame(callback: FrameRequestCallback): void;
}

export function createBrowserRuntime(): RuntimePorts {
  return {
    fetch: (input, init) => globalThis.fetch(input, init),
    // Resolve localStorage inside each operation so denied access remains catchable by callers.
    storage: {
      getItem: (key) => window.localStorage.getItem(key),
      setItem: (key, value) => window.localStorage.setItem(key, value),
      removeItem: (key) => window.localStorage.removeItem(key),
    },
    now: () => new Date(),
    monotonicNow: () => performance.now(),
    random: () => Math.random(),
    sleep: (ms) => new Promise((resolve) => window.setTimeout(resolve, ms)),
    schedule: (callback, ms) => { window.setTimeout(callback, ms); },
    frame: (callback) => { window.requestAnimationFrame(callback); },
  };
}

/** Deterministic non-cryptographic source for synthetic tests. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}
