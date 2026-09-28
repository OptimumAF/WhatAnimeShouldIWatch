# 0024 — Experiment, final refit, and model promotion boundaries

**Status:** M5.9 protocol fixed on 2026-09-28 before a final-refit run. The provider source/use holds and M5 exit gate remain active. M5.9 stays unchecked until its experiment and promotion paths pass.

## Separate stages

The existing `ml/eval_gnn_lightgcn.py` and `ml/build_content_features.py` are legacy experiments. The former splits and evaluates a prebuilt full graph; the latter requests Jikan metadata. Their metrics or feature cache cannot select a release engine. Any supported LightGCN or content candidate must consume a validated raw split and fixed, permitted metadata, fit learned values on training rows only, and use the same disjoint new-user validation cases, centrally eligible candidate universe, browser mapping/scorer, final selector, and metric definitions as decisions 0018, 0021, and 0022. If that adapter is absent, label the method unevaluated and do not promote it. Do not invoke Jikan or provider-derived training to test this boundary.

After a candidate is frozen by validation and its one-use final report has been produced, a separate final-fit command may refit the **unchanged configuration** on train plus validation. It must validate the same raw snapshot and split manifest, the exact frozen selection, selected train-only model fingerprint, and final-report link before fitting. Test rows remain excluded. This refit is a new model with its own input membership, fit/model/archive hashes and counts; its record includes only the final report's digest, never its test metric. An optional permitted production-snapshot fit needs a separate reviewed source/use path and is outside routine development.

For the invented MF fixture, require fixed train and validation membership, test-score perturbation invariance of every refitted parameter, a validation-score positive control, stale/tampered selection or final-report refusal, private unused output paths, numeric-only NPZ plus hash-bound JSON sidecar, and no public/release write. The official reserved new-user final cohort remains unscored by this check. A temporary test fixture may exercise the existing one-use warm-user final-report command.

## Release boundary

Retrain uploads and experimental outputs are not release candidates. Model promotion needs a predeclared quality/latency/coverage gate against the best simple baseline on a permitted, leakage-checked serving path; a frozen final report; a separately identified final refit; exact model/graph/catalog mapping and dataset compatibility; reviewed source/use and owner approval for publication and deployment; and an immutable previous compatible bundle for rollback. A web model's self-declared source digest cannot prove these requirements. The current `data-latest` release and full-snapshot retrain workflow do not satisfy them.

Later M5.9 work must prevent the existing optional-model publication routes from bypassing a promotion record, then test approval, compatibility, corruption, and rollback cases on invented artifacts. M8.1–M8.4 and M8.7 still own the full immutable release manifest, atomic installation, deployment trigger, and practiced recovery. No job dispatch, release update, Pages deployment, or promotion is authorized by this decision.
