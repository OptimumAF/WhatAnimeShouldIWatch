# 0017 — Validation selection and final-test boundary

**Status:** Initial split-first MF protocol implemented and synthetic checks passed on 2026-09-27. M5.4 remains unchecked until every M5 tuner that can affect the selected engine uses validation and the final-report boundary.

## Fixed synthetic experiment

Use the invented raw snapshot, M5.1 manifest, and fixed ID/title metadata snapshot. Keep the train/validation/test IDs pinned. Define relevance *before* selection as a raw score of at least 7 on the invented ten-point scale; do not tune the relevance label to improve a metric. Use mean NDCG@3 for warm users with an eligible positive holdout. Mask train items, rank the fixed metadata universe by model score then anime ID, and report the eligible user and positive-label counts. This is an engineering check of isolation, not a quality estimate.

Predeclare a small candidate JSON list. A candidate can vary MF factor count, epoch count (a validation-selected stopping checkpoint), learning rate, bias/factor regularization, graph regularization and its edge-weight threshold, graph sample rate, and a predicted-score floor. The positive-label threshold, top-K, split, metadata, objective, tie rule, and candidate list are fixed. Candidate IDs are unique; equal validation scores choose the lexicographically first ID. Candidate specifications are rejected if malformed or nonfinite. Do not use a test metric to add, remove, or reorder candidates. Old Optuna/LightGCN/full-graph scripts remain invalid M5 evidence; any future Optuna trial, blend weight, threshold, early-stop choice, or other candidate-selection parameter must use this validation-only scoring boundary before it can support M5.4.

## Read boundary

Selection validates the full split manifest but passes only `.train` to fitting and only `.validation` to scoring. It never reads `.test` labels, scores, or interaction IDs to decide a candidate. A synthetic test-score mutation with a refreshed manifest must leave trial scores and chosen candidate unchanged. The selection artifact binds the full raw-content digest only so a final report can verify the exact snapshot; this binding is not a selection input. It records training/metadata/candidate hashes, frozen configuration, validation-only trial metrics, and an integrity digest. Do not print a test metric during selection.

The final-test command requires that frozen artifact at its original resolved path, verifies its integrity and the exact raw/metadata snapshot, and creates a one-use marker exclusively *before scoring or inspecting `.test`*. Parsing and manifest validation necessarily load the full raw snapshot earlier; they do not choose a candidate or compute a test metric. The command refuses an existing marker or report. A failure after the marker is created needs explicit investigation; the command does not silently retry on test labels. It trains only the frozen candidate from train and writes one final test report to an ignored local path. Copying a selection file to a new path cannot bypass the original-path check. This is a workflow guard, not tamper-proof access control.

## Acceptance for the initial slice

Prove validation-only choice with at least two invented candidate configurations, a test-label perturbation, and a validation-label positive control; prove the frozen candidate cannot be replaced by edited selection JSON. Verify one final report, duplicate refusal, stale snapshot refusal, no public/release output, and no test read before the marker. Leave M5.4 unchecked while Optuna/hybrid blend and other M5 selection paths remain outside this boundary, and record that dependency in the plan.
