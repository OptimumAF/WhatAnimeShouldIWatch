/** Shared title normalization without coupling history parsing to recommendation scoring. */
export function normalizeTitle(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}
