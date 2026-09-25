# 0016 — Fixed-split leakage regression

**Status:** M5.3 protocol, 2026-09-25. Results are recorded in the living plan and progress log.

## What must remain fixed

Use the checked-in invented 13-unique-interaction raw snapshot, its 7/3/3 split, and the same fixed metadata snapshot and MF seed/hyperparameters. Derive a fresh manifest for each perturbed full snapshot, then require that the **training interaction IDs and raw train rows are exactly unchanged** before comparing fits. A changed score on an existing validation/test interaction should preserve all partition IDs. Changing a hidden interaction's anime ID may change its own validation/test ID, but must keep every train ID and row fixed; the deterministic seeded policy must still place it in the same held-out partition. These assertions prevent a changed split from masquerading as leakage or invariance.

Predeclare three invented perturbations: change validation `invented-b`/101 and test `invented-c`/103 raw scores; replace validation `invented-b`/101 with `invented-b`/108; replace test `invented-c`/103 with `invented-c`/104. The replacement anime IDs already exist in the fixed metadata snapshot, and neither user rated that replacement in the baseline. Reject stale manifests for every change. An added or removed interaction that changes derived training membership cannot be compared as a fixed-split experiment; require the original manifest to fail closed and record the membership change instead of claiming invariance.

## Observation boundary and assertions

Exercise the same orchestration function used by the split-first MF CLI. It validates the manifest, passes only `.train` into the fitter, builds MF training arrays from that fit and fixed metadata, and trains with a fixed seed. Compare user baselines, centered rows, train-only popularity counts/sums, signed pair weights/support and pair-selection statistics, numeric training arrays, positive regularization edges, training-row digest, fit digest, every fitted MF array (`P`, `Q`, `bu`, `bi`, global mean), and model digest. Require exact equality on this invented integer-score fixture. Full-snapshot identity/content digests should change as appropriate; they are not training hashes. Also compare the CLI's reported hashes for one perturbation to protect the file-backed path. No test label, score, or item identity is given to the fitter or model trainer.

## Limits

This proves isolation for the split-first synthetic route and its fixed metadata/seed. It does not validate old MF/Optuna/LightGCN metrics, tuning/test discipline, a production snapshot, provider rights, temporal availability, or new-user ranking quality. Keep M5.4–M5.9 and the M5 exit gate open.
