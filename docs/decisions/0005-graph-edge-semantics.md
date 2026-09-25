# 0005 — Graph edge meanings and candidate item similarity

**Status:** M3.3 design decision, verified on invented fixtures on 2026-09-25. No production graph, browser ranking, trainer, or published artifact changes in this slice. Provider-derived graph generation remains held by [decision 0001](0001-provider-data-permissions.md).

## Terms and current contract

| Quantity | Definition | What its sign or size says |
|---|---|---|
| Pair preference (current `graph-compact-v1` anime edge) | For each co-rater, average the two scores after subtracting that user's overall scored-list mean; then average those pair values. `support` is the co-rater count. | Positive can mean both items are often above a user's baseline; negative can mean both are below it. Zero can mean opposition or cancellation. It is not a correlation or similarity. |
| Co-occurrence | Number of users who rated both items, regardless of score. | Counts overlap and popularity exposure; it has no taste direction. The current `support` field is this count. |
| Item correlation | Pearson correlation of two item rating vectors on their common users, each centered by that item's mean over the overlap. | Measures linear association; undefined if either overlap vector has zero variance. This is not the current pair mean. |
| Adjusted-cosine item similarity | Cosine of the two item vectors of deviations from each user's mean, restricted to common users. | Positive means deviations tend to align, including two below-baseline items; negative means they tend to oppose. It is not Pearson correlation because it does not subtract item means. |

The existing `aggregateAnimePairs` sum/count correction makes its pair-preference mean order independent under the tested tolerance, but does not change its meaning. The current browser uses positive edges to seed candidates and applies negative edges only after a candidate exists; `ml/train_graph_mf.py` takes the absolute edge weight for attractive regularization. Neither consumer is ready to reinterpret a signed similarity in place. The graph builder also computes user means from the full list before applying its first-N rating cap. [Sarwar et al. (2001)](https://www.ra.ethz.ch/CDstore/www10/papers/519/node14.html) describe adjusted cosine by subtracting user means; their [correlation definition](https://www.ra.ethz.ch/CDstore/www10/papers/519/node13.html) centers item vectors instead.

## Chosen future recommendation-edge meaning

For users `U_ij` who scored both items, let `x_ui = r_ui - mean_u` and `x_uj = r_uj - mean_u`, with the user mean fitted on the eligible training partition. The proposed signed neighborhood weight is:

```text
adjusted_cosine(i,j) = sum_U(x_ui * x_uj) /
  sqrt(sum_U(x_ui^2) * sum_U(x_uj^2))
weight(i,j) = adjusted_cosine(i,j) * n_ij / (n_ij + lambda)
support(i,j) = n_ij
```

Missing overlap or a zero-magnitude side has no defined weight; it must not become a zero-strength relationship or a NaN. `lambda = 2` is fixed **only for the hand-computed benchmark below**. It is a design parameter to select on validation data after split-first preprocessing and source permission review, not a production tuning result. Support shrinkage reduces confidence in one shared rating without changing the sign. Keep support as a separate field; do not infer it from the shrunken weight. This candidate is an item relationship derived from expressed rating deviations, not content similarity, viewing status, or a direct prediction that a particular user likes either item.

## Reproducible synthetic semantic benchmark

[`fixtures/synthetic-edge-semantics.json`](../../fixtures/synthetic-edge-semantics.json) contains three invented users with user-centered rows summing to zero. Item IDs 1–5 are invented. `node --import tsx --test pipeline/test/edge-semantics.benchmark.test.ts` checks the fixture and the following hand calculations against both implementations (tolerance `1e-12`):

| Pair | Constructed relationship | Support | Corrected v1 pair preference | Adjusted cosine | Shrunk weight, λ=2 |
|---|---|---:|---:|---:|---:|
| 1–3 | Aligned deviations | 3 | 0.666667 | 1 | 0.6 |
| 2–4 | Mostly co-disliked, deviations align | 3 | -1 | 0.942809 | 0.565685 |
| 1–2 | Opposite deviations | 3 | 0 | -1 | -0.6 |
| 1–5 | Aligned, one common user | 1 | 2 | 1 | 0.333333 |

The pair preference ranks the single-overlap pair 1–5 above three-rater aligned pair 1–3; shrinkage reverses that ranking. It calls co-disliked pair 2–4 negative, while their deviations align. It calls opposed pair 1–2 neutral. Co-occurrence alone is 3 for each of those three relationships and cannot distinguish them. These are deliberately constructed semantic counterexamples, **not** measured recommendation accuracy or evidence that `lambda = 2` is optimal. The prototype pair scorer in `pipeline/src/core/item-similarity.ts` is not wired into graph exports or runtime consumers.

## Migration and open gates

- Retain v1 pair-preference weights and support under their existing format; never relabel old edges as similarity. M3.7 must introduce a versioned semantic identifier, formula/configuration, support, and dataset/split identity with a deliberate old-format path. Recommendation neighborhoods and visualization samples may require distinct artifacts.
- M3.4 must define how negative, positive, and neutral evidence affects visualization, candidate generation, and model regularization. Do not take the absolute value of a negative similarity and then use it as an attractive model constraint. The benchmark's co-disliked pair also shows why a positive item relationship is not by itself positive evidence about a new user's preference.
- M3.5–M3.6 must bound input pair enumeration and memory and replace first-encounter output caps. A top-K output pass alone leaves `O(sum_u k_u^2)` pair visits and can retain too many candidate keys. The prototype scorer makes no scalability claim.
- M5.1–M5.6 must split interactions before fitting user means and graph statistics, then compare this candidate against simple baselines on permitted data and the browser's new-user path. Changing held-out scores must not change fitted graph edges. No provider-derived training, artifact regeneration, publication, or deployment is authorized by this design decision.
