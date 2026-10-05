/** Fixed, local-only timing names. Enabled only with ?perf=1; never persisted or sent. */
export type LocalPerformanceMetric =
  | "wasiw:json:catalog"
  | "wasiw:json:graph"
  | "wasiw:json:explorer"
  | "wasiw:json:model"
  | "wasiw:schema:catalog"
  | "wasiw:schema:graph"
  | "wasiw:schema:explorer"
  | "wasiw:schema:model"
  | "wasiw:index:graph"
  | "wasiw:index:model"
  | "wasiw:recommendation:update"
  | "wasiw:recommendation:graph-score"
  | "wasiw:recommendation:model-score"
  | "wasiw:recommendation:eligibility"
  | "wasiw:recommendation:filter-ui"
  | "wasiw:recommendation:franchise"
  | "wasiw:recommendation:cards"
  | "wasiw:recommendation:dom"
  | "wasiw:network:select"
  | "wasiw:network:construct"
  | "wasiw:network:layout"
  | "wasiw:network:scope"
  | "wasiw:network:svg"
  | "wasiw:network:render";

const enabled = typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("perf") === "1";

function record(name: LocalPerformanceMetric, startedAt: number): void {
  performance.measure(name, { start: startedAt, end: performance.now() });
}

export function measureLocal<T>(name: LocalPerformanceMetric, work: () => T): T {
  if (!enabled) return work();
  const startedAt = performance.now();
  try {
    return work();
  } finally {
    record(name, startedAt);
  }
}

export function measureLocalAsync<T>(
  name: LocalPerformanceMetric,
  work: () => Promise<T>,
): Promise<T> {
  if (!enabled) return work();
  const startedAt = performance.now();
  try {
    return work().finally(() => record(name, startedAt));
  } catch (error) {
    record(name, startedAt);
    throw error;
  }
}
