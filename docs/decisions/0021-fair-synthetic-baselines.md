# 0021 — Fair split-first synthetic baselines and graph ablations

**Status:** Protocol fixed on 2026-09-27 before running M5.6 baseline or ablation metrics. Five focused TypeScript and five Python tests, the full local synthetic/mock gate, and [fresh PR #35 CI](https://github.com/OptimumAF/WhatAnimeShouldIWatch/actions/runs/36390572870) passed on `c9ca431`. M5.6 is checked for this invented comparison; it is not a model-selection or release decision.

## Cohort and common candidate set

Use the validated 72/24/24 raw fit split, fixed ID/title metadata, and four disjoint invented new users from decision 0018. Only training rows may fit user means, item counts, pair statistics, item similarity, and model parameters. The existing new-user validation labels score all methods. The separate checked-in new-user final cohort remains unread by this comparison. `fixtures/synthetic-baseline-ablation-spec.json` pins newline-normalized input hashes, methods, and parameters before metrics.

For each user and nested 1/3/5/10 observed prefix, map native scores through `preferenceFromHistory`. Construct one candidate universe from the train-present 24-title catalog and `createCandidateEligibilityPolicy`, using the same watched/history, exclusions, Include Only, and genre/year/score metadata for every method. Each method must rank **every** eligible ID. Give missing graph, similarity, content, or model signal a neutral score of zero; report nonzero signal coverage separately. Sort equal scores by ascending anime ID. Apply the existing final franchise selector to every ordered list. The common universe is before that selector; any selector-caused retained-set difference must be disclosed. An eligible positive label has raw score at least 7 and belongs to this common universe.

**Interpretation clarification:** Completing sparse source lists with neutral zeros is an evaluation intervention for common-candidate fairness. The graph, content, and hybrid rows therefore do not reproduce the current browser's sparse candidate generation or fallback rates. Decisions 0018 and 0020 remain the serving-path checks. This clarification changes neither the declared methods nor their scores.

## Predeclared methods

| Method | Score source |
|---|---|
| `train-count` | Number of raw training ratings for the item, as a popularity proxy for this invented restricted snapshot. |
| `metadata-score` | The fixed fixture's already known community-score field; do not use rating-derived metadata or claim real quality. |
| `supported-adjusted-cosine` | Train-centered co-rater adjusted cosine, shrunk by `n/(n+2)`; require at least two co-raters. Only positive similarity from explicitly Liked supplied titles contributes, weighted by browser importance × confidence. |
| `genre-overlap` | Browser's exact-genre, Liked-only content scorer using the same fixed metadata. |
| `v1-positive-pair-graph` | Browser's current positive pair-preference graph scorer; zero score for other eligible titles. Its weights are not similarity. |
| `plain-mf` | Two-epoch, four-factor train-only MF, with graph lambda zero. |
| `positive-pair-mf` | The same MF hyperparameters and seed, with the current positive pair-preference attraction and lambda 0.01. |
| `unit-positive-pair-mf` | Same positive edge membership and lambda, but unit attraction per edge. |
| `shrunk-positive-pair-mf` | Same positive edge membership and lambda, with each pair-preference weight multiplied by `support/(support+2)`. |
| `hybrid-default-0.5` | Browser weighted rank fusion of completed `v1-positive-pair-graph` and `positive-pair-mf` lists at the current product default 0.5. |

Use the fixed `graph-two-epochs` candidate's factors, epochs, learning rate, bias/factor regularization, graph sample rate, and seed for **all** MF variants. The graph variants differ only in the declared edge-weight rule. Do not tune those rules or the similarity shrinkage after inspecting this validation report. The positive-pair graph and graph MF regularizer must never use the absolute value of a negative v1 pair-preference edge. Audit both legacy and compact edge loaders plus the shared trainer, and test a negative edge's exclusion from attractive updates.

## Report and acceptance

The primary descriptive comparison is equal-case mean **displayed** NDCG@10 over all measurable user-prefix cases. Also report Hit@10, Recall@10, per-prefix NDCG@10, eligible cases/positive labels, common candidate counts or hashes, nonzero signal coverage, and any fallback/selector effects needed to interpret the result. Methods with no signal still rank the common universe by the declared neutral/tie rule; do not silently substitute another engine. No test score, model promotion, quality gate, or winner is inferred from this small authored validation cohort. Decision 0017/0020 final-report boundaries remain in force.

Verify input-hash refusal, identical pre-selector eligible IDs and labels across every method, browser preference/eligibility/final-selector use, train-only statistics and model invariance under a refreshed held-out fit-score perturbation with fixed train membership, validation-label score/rank isolation, meaningful similarity support/sign cases, and graph negative-edge exclusion. Run focused and full fixture/mock gates and fresh PR CI before checking M5.6. If any comparison cannot meet the common-candidate contract, leave M5.6 unchecked and record the precise gap.
