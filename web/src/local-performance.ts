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
  | "wasiw:init:anime-options"
  | "wasiw:init:anime-option-batch"
  | "wasiw:init:network-options"
  | "wasiw:init:network-option-batch"
  | "wasiw:init:other-ui"
  | "wasiw:init:active-view"
  | "wasiw:init:yield"
  | "wasiw:recommendation:update"
  | "wasiw:recommendation:graph-score"
  | "wasiw:recommendation:model-score"
  | "wasiw:recommendation:eligibility"
  | "wasiw:recommendation:filter-ui"
  | "wasiw:recommendation:franchise"
  | "wasiw:recommendation:cards"
  | "wasiw:recommendation:dom"
  | "wasiw:recommendation:yield"
  | "wasiw:network:select"
  | "wasiw:network:construct"
  | "wasiw:network:construct-node-batch"
  | "wasiw:network:construct-edge-batch"
  | "wasiw:network:layout"
  | "wasiw:network:visible-node-map"
  | "wasiw:network:visible-node-sort"
  | "wasiw:network:visible-node-list"
  | "wasiw:network:control-update"
  | "wasiw:network:scope"
  | "wasiw:network:svg"
  | "wasiw:network:svg-coordinates"
  | "wasiw:network:svg-edge-batch"
  | "wasiw:network:svg-paths"
  | "wasiw:network:svg-node-batch"
  | "wasiw:network:svg-node-paths"
  | "wasiw:network:svg-dom-commit"
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

/** Record one synchronous batch; the caller excludes the time spent yielding. */
export function recordLocalDuration(name: LocalPerformanceMetric, startedAt: number): void {
  if (enabled) record(name, startedAt);
}
