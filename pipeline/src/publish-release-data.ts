/**
 * The old data-latest publisher uploaded per-user rating rows and could update a
 * mutable release. Keep its CLI entry point as an explicit fail-closed guard
 * until M8.3 supplies a verified, audited, aggregate-only publication route.
 */
throw new Error(
  "Legacy data release publication is disabled by decision 0028; " +
  "a verified aggregate-only bundle and reviewed source/use approval are required.",
);
