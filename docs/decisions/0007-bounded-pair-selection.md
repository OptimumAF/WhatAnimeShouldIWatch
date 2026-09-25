# 0007 — Exact pair selection inside separate input and output limits

**Status:** M3.5 design and synthetic verification, 2026-09-25. This changes which existing v1 pair-preference edges a capped build exports; it does not change their formula or format. No provider-derived graph was regenerated or published.

## Why the old guard failed

The previous `max-anime-anime-edges` guard kept the first distinct pair keys reached during user traversal. Later users could increase only already retained pairs, so even those weights and support counts were censored when a key was rejected earlier. The guard bounded map keys but still visited every selected user's `k(k-1)/2` pair observations. On the invented [selection fixture](../../fixtures/synthetic-pair-selection.json), output cap 1 kept the first `1:2` pair (support 1, weight 0) and missed `3:4` (support 3, weight 1).

## Strategies evaluated

| Strategy | Candidate work and memory | Selection consequence |
|---|---|---|
| First-encounter guard | Visits all pair observations; stores at most output cap keys | Order-biased keys and incomplete support/means |
| Aggregate every key, then top-K | Visits all observations; stores every distinct key | Correct selection but no input-work or memory bound |
| Streaming sample or heavy-hitter sketch | Can bound keys, but support and means are approximate after eviction/sampling | Needs a recorded seed, explicit error/coverage report, and validation; M3.6 owns this path |
| **Chosen exact bounded aggregation** | Preflight pair visits and stop above that budget; stop when distinct candidate keys exceed a separate budget; sort only the bounded exact candidates | No partial artifact is returned when an input budget fails; an output cap selects by evidence instead of arrival |

For processed ratings, let `P = sum_u k_u(k_u-1)/2` and `C` be distinct pair keys. Preflight costs `O(users)` before enumeration. Aggregation costs `O(P)` time and `O(C)` key storage. Sorting eligible candidates costs `O(C log C)` time and `O(C)` references. The default hard limits are `P <= 20,000,000` and `C <= 2,500,000`, separate from the existing output cap of 2,000,000 edges. These are guard values, not measured production capacity or tuned quality thresholds. `0` still removes only the **output** cap; input limits remain positive and mandatory. The CLI prints counts and limits, and an exceeded input limit fails before new graph/dataset files are written.

All eligible pairs are ranked by descending co-rater support, then descending absolute v1 pair mean, then numeric `(lowAnimeId, highAnimeId)`. `--min-pair-support` filters before ranking (default 1 to preserve current coverage). `--max-neighbors-per-anime` optionally applies a greedy maximum selected degree after ranking (default 0, no degree limit). The global output cap is then applied. The sorted selected pairs are exported in numeric ID order. A support-first rank avoids letting one extreme observation outrank a repeated pair, but neither this order nor the default support threshold is an empirical recommendation-quality result. Recommendation neighborhoods and network samples remain coupled in v1; M3.7 must separate them.

## Hand-computed fixture evidence

The fixture has four centered invented user rows: eight pair visits and six distinct keys. Both compact and legacy CLI outputs retain the same selected pair and true processed-row support.

| Setting | Retained pair keys | Observed effect |
|---|---|---|
| No output/support/degree filter | `1:2`, `3:4`, `3:5`, `3:6`, `4:5`, `4:6` | 8 visits, 6 candidates |
| Output cap 1 | `3:4` | Later support-3 pair wins; five output exclusions |
| Minimum support 2 | `3:4` | Five support-1 keys excluded |
| Per-anime degree 1 | `1:2`, `3:4` | Four incident keys excluded after stronger `3:4` fills both endpoints |
| Pair-visit budget 7 | No artifact | Preflight rejects eight required visits |
| Candidate-key budget 5 | No artifact | Rejects the sixth distinct key before export |

Reversing user order and each rating row leaves the uncapped-input selection identical in tests. The current `--max-ratings-per-user` first-N policy **still** depends on rating order and omits data; the CLI reports that count, but M3.6 must replace the selection policy and measure skipped data, runtime, peak memory, and coverage. A support value under that cap counts co-raters in the processed subset. The budgets count pair visits and keys, not exact heap bytes; M3.6 must measure actual peak memory before production-scale use. M3.7 must put selection configuration, dataset identity, and truncation into a versioned artifact. M5 must evaluate ranking quality after splitting data. Decision 0001 continues to hold provider-derived collection, training, and publication.

V1 compact and legacy contracts allow a selected pair subset without treating absent edges as malformed. A mocked browser load of a one-pair graph revealed that rounding the initial network threshold **up** to the slider step could hide its only edge. The default threshold now rounds down; the selected-pair browser test checks that recommendations and the network both load without manually changing the slider. This is a v1 viewing fix, not a substitute for M3.7 truncation metadata.
