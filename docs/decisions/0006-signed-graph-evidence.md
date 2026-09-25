# 0006 — Signed graph evidence by consumer

**Status:** M3.4 decision, verified with invented fixtures on 2026-09-25. Applies to the current `graph-compact-v1` pair-preference weight and specifies the later signed-neighborhood behavior; it does not change graph exports, published models, or provider-derived data. Read with [decision 0005](0005-graph-edge-semantics.md).

## What the sign establishes

V1 sign is the mean of two user-centered ratings, not opposition or item similarity. A negative mean may come from two below-baseline items. Its magnitude must not be turned into an attractive model constraint, and it must not be described to a user as evidence that a candidate opposes a selected item. A positive mean remains a provisional compatibility signal, not measured recommendation quality.

For the proposed support-shrunk adjusted cosine, a positive weight means co-rated deviations align, a negative weight means they oppose, and zero means the observed vectors are orthogonal. Co-disliked items can align positively. Zero is a defined result when both vector magnitudes exist; no overlap or a zero-magnitude side is undefined and must have no relationship. A single overlap has a defined cosine but weak support, so the benchmark shrinks it; M3.5 still must select a support gate and M5 must tune shrinkage on split-first validation data.

The [synthetic fixture](../../fixtures/synthetic-edge-semantics.json) and `node --import tsx --test pipeline/test/edge-semantics.benchmark.test.ts` check these hand calculations at `1e-12` with benchmark-only `lambda = 2`:

| Case | Centered pair observations | Cosine | Shrunk weight | Interpretation |
|---|---|---:|---:|---|
| Opposite tastes | `(2,-2), (-2,2)` | -1 | -0.5 | Opposing deviations |
| Universally co-disliked | `(-2,-1), (-4,-2)` | 1 | 0.5 | Aligned deviations despite dislike |
| Neutral | `(1,0), (0,1)` | 0 | 0 | Observed, orthogonal |
| Constant nonzero item vectors | `(1,1), (1,1)` | 1 | 0.5 | Adjusted cosine exists although item-centered Pearson has no variance |
| No within-user variation | `(0,0), (0,0)` | undefined | undefined | Zero magnitude |
| One opposite observation | `(2,-1)` | -1 | -1/3 | Defined but sparse |
| No overlap | none | undefined | undefined | No relationship |

The fixture also centers uniformly high `[9,9,7,7]` and low `[3,3,1,1]` raters to the same `[1,1,-1,-1]` vector. Flat high/low raters center to zero and provide no cosine magnitude. This checks invariance to a rater's score offset; it does not establish robustness to every scoring habit.

## Consumer policy

| Consumer | Current v1 handling | Required handling for a later signed-neighborhood format |
|---|---|---|
| Network visualization | Keep positive, negative, and zero anime-pair edges visible when the absolute-weight filter permits them. Use distinct colors and a dashed zero edge, label the quantity **pair preference**, and retain the signed value in inspection. Negative means below-baseline pair mean, not opposite tastes. User-anime edges retain their separate centered-rating meaning and also show sign. | Label the new semantic format as adjusted-cosine similarity, distinguish alignment/opposition/neutral visually, expose support and any truncation, and keep the visualization sample separate from ranking neighborhoods. Do not relabel v1 edges. |
| Graph candidate generation | Only positive v1 pair-preference edges seed and score candidates. Ignore nonpositive v1 edges entirely; a negative mean cannot be a candidate penalty. This removes the old selection-order-dependent negative explanation without changing positive-only ranking. | With an explicitly positive selected preference, positive similarity may seed and add evidence; negative similarity may subtract from a candidate seeded by positive evidence but must not seed a recommendation alone. Zero contributes nothing. A genuinely negative selected preference needs the M4.1 preference mapping before its interaction with signed similarity is implemented. No new ranking path is wired in M3.4. |
| Model regularization | The legacy attractive-distance term receives **positive v1 weights only**. Negative and zero v1 pair means are excluded; the old `abs(weight)` transformation was unjustified. The `--graph-min-abs-weight` CLI spelling stays for compatibility but now thresholds retained positive weights. Existing trained model artifacts are unchanged. | Positive similarity may be considered for attractive regularization after M3.7 format migration and M5 split-first evaluation. Negative similarity must not be made attractive through absolute value. A repulsive objective is a separate, stability-tested model decision; until then exclude negative and zero edges from regularization. |

The browser and both Python graph loaders have synthetic regression checks for the current policy; the browser demo verifies the visible sign key. The v1 trainer still consumes a provisional graph statistic and the graph may already include held-out information. No empirical ranking gain, safe support threshold, or production model improvement is claimed. M3.5–M3.7 and M5 remain required before a new similarity weight enters runtime consumers. Decision 0001 still holds provider-derived collection, training, and release work.
