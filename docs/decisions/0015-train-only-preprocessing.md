# 0015 — Train-only preprocessing boundary

**Status:** Protocol for M5.2, 2026-09-25. Synthetic verification is recorded in the living plan and progress log.

## Inputs and isolation

Read a raw snapshot and validate its immutable split manifest with decision 0014 before fitting. The fit function receives only the resulting `train` tuple. Validation and test rows are never passed to centering, popularity, graph construction, or the model trainer. Rechecking a manifest against a modified snapshot is required; a held-out score edit needs a new content digest even when its membership IDs stay fixed.

The allowed metadata snapshot is a separate, fixed `anime-metadata-snapshot-v1` file. It contains only a source label, UTC snapshot time, and unique positive anime IDs with titles. It may define the item universe and display names, but cannot contain scores, counts, user IDs, ranks, or derived popularity. For this phase the checked-in snapshot is invented. Any later real metadata snapshot needs source/use review, a cutoff appropriate to the evaluation, and a separate permission decision. Metadata never decides train/validation/test membership.

## Fitted values

For each training user, compute the arithmetic mean of that user's **training raw scores**; center only training rows against that mean. Calculate item interaction counts and raw-score sums only from train. There are no learned content features in this path. Keep zero-count metadata items in the universe with zero popularity, rather than learning their presence from holdouts. Sort by stable user and anime ID before accumulation and hashing.

Build exact pair means and support from the centered training rows using the existing TypeScript `aggregateAnimePairs` implementation through a narrow stdin/stdout adapter. Pass only centered training rows across this local process boundary. Use all ratings, minimum support one, no output/degree cap, and the existing pair visit/key budgets; budget failure aborts without a partial fit. Retain signed pair weights and support in the fit; the current attractive MF regularizer may use only positive weights. No full-snapshot graph file can enter this path.

The fit fingerprint covers the fixed metadata digest, sorted training raw/centered rows, user baselines, popularity, graph pairs/support, and pair configuration. It excludes timestamps, validation/test labels, and report paths. A separate training-row digest is recorded. The new split-first runner fits an MF model from this fit and emits no evaluation metric in M5.2; M5.3 will test parameter invariance under held-out perturbations. The legacy MF/Optuna/LightGCN commands remain compatibility research paths and cannot support M5 claims until they use the same boundary.

## Synthetic acceptance

On the pinned 7/3/3 invented split, assert exact train-only means, centered rows, popularity, pair means/support, and deterministic fit hashes. Reject malformed or rating-derived metadata, missing training IDs, stale/tampered split manifests, and graph budget failures. Run the split-first MF route with fixed seed into an ignored local output directory; report fit and model hashes without reading holdout labels. Do not fetch provider data or publish model artifacts.
