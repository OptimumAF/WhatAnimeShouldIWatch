# 0008 — Seeded, reproducible per-user graph rating selection

**Status:** M3.6 synthetic implementation and measurement, 2026-09-25. This supersedes the first-N `--max-ratings-per-user` policy. It does not approve a production build, change the v1 pair-preference formula, or version the graph format.

## Selection policy

With cap `0` (the default), every valid rating is used. With a positive cap `k`, each user with more than `k` ratings keeps the `k` smallest SHA-256 ranks. The rank input is the UTF-8 JSON encoding of `["sha256-bottom-k-v1", seed, userId, animeId]`; equal ranks break by numeric anime ID. The unsigned 32-bit seed defaults to `0` and can be set with `--pair-selection-seed` or `GRAPH_PAIR_SELECTION_SEED`. Selected rows are then sorted by anime ID for stable pair accumulation. Scores are centered against **all** of that user's ratings before selection, as in the prior graph builder. The same selected rows supply user-anime and anime-anime graph edges; the separate ratings dataset export retains all rows. No raw ID or rating is written to the build report.

For unchanged user/anime IDs and scores, cap, and seed, rearranging users or ratings preserves selected memberships and pair statistics. A different seed deliberately changes memberships. A source that reassigns anonymized user IDs changes the sample even with the same seed. The hash is a deterministic selection device, not an anonymization or access-control mechanism. Equal inclusion opportunity across one user's rated items does not make a capped graph representative of all pair relationships or prove ranking quality.

This policy chooses a subset before pair enumeration: for a user with `n` ratings, the graph visits `min(n,k)(min(n,k)-1)/2` pairs instead of `n(n-1)/2`. Sorting hash ranks costs `O(n log n)` time and `O(n)` temporary memory for a capped user. The separate visit and candidate-key limits from [decision 0007](0007-bounded-pair-selection.md) still fail closed. No streaming key eviction or approximate support estimator is used; among the retained observations, pair sums and support are exact.

## Build report and measurement

Every successful `build:graph` writes `graph-build-report-v1` beside the selected graph output, or to `--out-report`. It records the policy, seed, caps, budgets, input/selected/skipped rating counts, full/selected/skipped pair observation counts, anime coverage, candidate and selected pair counts, output truncation, and exclusion counts. `approximationLevel` describes **input sampling**: `exact-input` if no rating was omitted, otherwise `seeded-per-user-subset`. Retained fractions use `1` for an empty denominator. Ordinary capped builds set candidate `pairKeyRecall` to `null` because the full pair-key set is not enumerated; with no omitted ratings, it is `1`. An output cap can still omit edges in either case. The CLI also prints the key counts and seed.

`measurement.elapsedMs` covers database loading, centering, graph construction, and graph/dataset writes, ending before the report write. `measurement.peakRssBytes` comes from the process high-water RSS (`process.resourceUsage().maxRSS`, converted from KiB); if that API reports zero, the report identifies a current RSS snapshot instead. Peak RSS includes the Node runtime and loaded dataset, so it is a process measure rather than the aggregator's isolated heap cost. Runtime and peak RSS vary with machine and cache state. The report is a sidecar, not a v1 graph field; M3.7 must bind configuration and dataset identity to a versioned graph artifact and keep visualization samples distinct from recommendation neighborhoods.

## Reproducible synthetic comparison

`npm run benchmark:pair-cap` generates 250 invented users, 180 invented anime, and 48 centered ratings per user from the documented modular recipe in `pipeline/src/benchmark-pair-cap.ts`. It runs exact and cap-24 builds in separate Node processes with seed 17. Its output reports runtime and peak RSS for each process and compares capped pair keys with the exact synthetic set. This is a coverage demonstration, not a recommendation-quality or production-capacity benchmark. The full dataset's pair relationships are available only because this small synthetic case fits within the exact input limits.

One local Windows / Node v23.10.0 run on 2026-09-25 measured:

| Variant | Ratings used | Pair visits | Distinct pair keys | Aggregation time | Process peak RSS |
|---|---:|---:|---:|---:|---:|
| Exact | 12,000 | 282,000 | 8,460 | 36.405 ms | 84,115,456 bytes |
| Seed 17, cap 24 | 6,000 | 69,000 | 8,105 | 43.724 ms | 84,660,224 bytes |

The cap retained 50% of ratings, 24.47% of pair observations, all 180 anime, and 95.80% of distinct synthetic pair keys. Hash ranking cost more than the saved enumeration in this small run; the measurements do not justify a speed or memory improvement claim. Repeated timing and larger synthetic shapes would be needed to tune a production cap, and source-use approval would still be required before a provider-derived build.

At the time of this decision, the graph format still exposed a single edge set without build configuration or dataset identity. [Decision 0009](0009-graph-v2-contract.md) subsequently bound configuration and dataset identity to v2 graph exports and separated the browser's visualization sample; the report remains a sidecar and is not loaded by the browser or trainer. M5 must evaluate any cap or alternative neighborhood policy on a split-first recommendation benchmark. Decision 0001's source-use and public-artifact holds remain unchanged.
