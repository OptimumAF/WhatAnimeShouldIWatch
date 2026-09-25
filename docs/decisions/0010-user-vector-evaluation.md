# 0010 — Browser new-user vector comparison protocol

**Status:** Complete local M4.5 decision; protocol fixed before running the comparison on 2026-09-25. It makes no provider-data or production-model decision.

## Question and inputs

The browser currently averages signed item embeddings for an unseen user's explicit Liked and Disliked preferences. Training metrics for stored user factors do not evaluate that path. Compare the exact browser scorer with a regularized fold-in candidate using only `fixtures/synthetic-input.json` and its generated compact demo model. The eight item vectors and biases are authored fixture values, not factors learned from these users' ratings. This avoids a training holdout leak in this narrow check but does not make the invented tastes representative.

Deduplicate a fixture user's ratings by anime ID with the last score winning, omit unknown IDs, and enumerate every held-out rating of at least 7/10 for users with at least two mapped ratings. For each holdout, enumerate all one- and three-rating subsets of the other ratings. Convert each observed rating through the browser's native 10-point preference mapper. Skip a subset with no Liked or Disliked signal. Seen observations exclude their titles but do not fit the vector. The held-out title is not supplied as a preference, exclusion, or model-fitting input. Keep the current compact model and candidate catalog fixed; apply the same browser eligibility policy to both methods. Do not use graph edge weights or the retained rating counts to fit either vector.

## Methods and measurements

- **Current browser average:** call `buildModelRecommendationsForPreferences` and the normal model eligibility path. It averages item vectors using signed importance × confidence weights and scores candidates with global and item biases.
- **Fold-in candidate:** minimize `sum_j w_j (y_j - globalMean - itemBias_j - q_j · u)^2 + ||u||²`, where `y_j` is +1 for Liked and −1 for Disliked, `w_j` is importance × confidence, and Seen has no fit term. Solve the small dual ridge system with regularization fixed at 1.0. Keep global and item biases in candidate scores. A user bias is omitted because it shifts every candidate equally and cannot affect rank. Do not tune targets or regularization on the held-out cases.
- **Quality:** report case count, Hit@3, mean reciprocal rank, and the rank of every held-out positive separately for one and three observed ratings. Do not count unrated titles as negative labels. A comparison is inconclusive if either slice has fewer than five cases.
- **Latency:** warm both methods, then measure p95 of 200 inference calls on a deterministic 1,000-title, 96-factor invented catalog for one and three preference signals on this host. Report the host/runtime and measured times; timing is advisory outside this host.

The minimum local engineering gate for the current browser method is Hit@3 ≥ 0.50 in each slice and p95 ≤ 50 ms on the declared synthetic workload. Replace it with fold-in only if fold-in improves Hit@3 by at least 0.10 in **both** slices, does not lower reciprocal rank in either slice, and meets the same latency gate. Otherwise keep the simpler browser average if it meets its gate. If neither method meets the gate, leave M4.5 unchecked and identify the failure. A passing invented-data gate is a serving-path decision for this slice, not evidence of production ranking quality; M5's split-first, permitted-data evaluation remains required before promotion.

## Result and decision

The protocol above was committed as `abe3e7e` before the fixture comparison ran. `npm run eval:user-vector:fixture` verified the generated fixture, enumerated the cases, called the exact browser average scorer plus a separate ridge reference, applied the same model eligibility policy, and measured latency. The local host was Windows x64, Node 23.10.0, Intel i7-12700K. A hand-computed ridge test checks bias subtraction and a browser test exercises one and three explicit preference signals through the deployed model mode.

| Observed ratings | Positive holdout cases | Average Hit@3 / MRR | Fold-in Hit@3 / MRR | Average / fold-in p95, 1,000 items × 96 factors |
|---|---:|---:|---:|---:|
| 1 | 29 | 20/29 = 0.690 / 0.492 | 20/29 = 0.690 / 0.528 | 0.846 / 0.657 ms |
| 3 | 7 | 5/7 = 0.714 / 0.612 | 5/7 = 0.714 / 0.643 | 1.276 / 1.333 ms |

**Decision:** Retain the current browser item-embedding average. It passed the local Hit@3 and latency gates; fold-in did not improve Hit@3 in either slice, so it failed the predeclared promotion rule. The fold-in code remains an evaluation-only reference and is not shipped in the web bundle. The cases share a tiny, hand-authored eight-title fixture, overlap within users, and cannot establish a real-world quality difference or calibrate the model scores. M5.1–M5.5 must rebuild split-first, leakage-checked new-user evaluation on permitted data before any production method or quality claim is promoted.
