# 0019 — Split-first Optuna selection

**Status:** Protocol fixed before implementation and Optuna trial results on 2026-09-27. M5.4 remains open until the browser hybrid blend and other candidate choices use validation-only selection.

## Scope and boundary

Replace the `ml:search` command's full-snapshot MF/graph experiment with a synthetic-safe Optuna route. It must load a raw-score snapshot, validate its immutable train/validation/test manifest, and fit each trial using only the M5.2 train-only preprocessing path. Optuna receives only the validation NDCG@K from decision 0017. It must not use the old precentered dataset, prebuilt graph, or trainer's train/test resplit as M5 evidence.

The checked-in invented three-candidate specification from decision 0017 is the fixed search space for the routine command. Use an in-memory, seeded Optuna grid sampler to visit each candidate exactly once with one worker. Factor count, epoch checkpoint, learning rate, bias/factor and graph regularization, positive graph threshold, graph sampling, and model-score floor are locked in each declared candidate. Raw relevance threshold, K, split, metadata, and model seed are fixed across trials. The selection objective is mean validation NDCG@K for warm users with positive labels; equal scores choose the lexicographically first candidate ID. A future broader search must predeclare its bounded space and cannot promote a validation result to a production quality claim.

Freeze the Optuna result in the existing `split-first-selection-v1` record: exact raw/metadata/candidate/training/fit/model fingerprints, ordered validation trials, one chosen candidate, original absolute output paths, and an integrity digest. The existing separate `report-test` command is the sole route to a synthetic final test report; it checks the frozen candidate and writes a one-use marker before scoring test. The search command emits no test metric and creates no resumable study database, so a stale study cannot mix snapshots or trials. Local selection/test outputs stay outside public or release assets.

## Checks

Run a real Optuna grid on the invented fixture and compare its ordered trial metrics and chosen candidate to the existing deterministic validation grid. Perturb only a test score under refreshed manifest membership: candidate choices, validation metrics, train/fit/model hashes must stay equal. A validation-label change must affect validation results. Refuse stale manifests, reused output paths, missing/duplicate trials, and public/release output paths before a final test read. The checked-in fixture and mocked-provider gates still apply. The old legacy MF training script remains a compatibility experiment and its metrics are invalid M5 evidence; `ml:search` no longer invokes the legacy search path.
