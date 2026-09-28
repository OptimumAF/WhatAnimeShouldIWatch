# ML Recommender (Graph-Regularized Matrix Factorization)

This folder trains a recommendation model from:

- `data/anonymized-ratings.compact.json` (preferred) or `data/anonymized-ratings.json`
- `data/graph.compact.json` (preferred) or `data/graph.json`

The model is matrix factorization with an extra graph regularization term on anime embeddings.

## Evaluation status

The legacy MF training and LightGCN commands below remain research compatibility paths. They load already normalized ratings and a graph built from the full ratings snapshot before splitting, so their metrics are **not** valid M5 release evidence. The Optuna command now uses the restricted split-first validation route described below. Provider-derived training and publication remain held by [decision 0001](../docs/decisions/0001-provider-data-permissions.md).

M5.1's restricted raw split manifest is reproducible with invented data:

```bash
npm run split:fixture:check
python ml/raw_interaction_split.py --input fixtures/synthetic-split-input.json --out data/synthetic-split-local.json
```

The second command creates a new ignored local file and refuses an existing path. The manifest contains hashed interaction IDs, not scores or plain user IDs; it is still restricted data if generated from a real snapshot. `partition_snapshot` validates the exact manifest and returns unnormalized train/validation/test rows. Present exports lack verified rating/viewing event times, so their policy is a seeded per-user holdout. See [decision 0014](../docs/decisions/0014-raw-interaction-splits.md) for format, duplicate, timestamp, and leakage limits.

M5.2's split-first fitting route uses a separate, invented metadata snapshot and only the validated training rows. It computes train-only user centering, popularity, and exact pair weights/support through the existing TypeScript pair core, then fits MF with fixed seed. It prints fit/model hashes and counts, without reading held-out labels for training or reporting a quality metric:

```bash
npm run ml:train:split:fixture
```

The metadata file allows only anime IDs and titles, a source label, and snapshot time; it cannot carry rating-derived fields. The pinned synthetic file is `fixtures/synthetic-anime-metadata.json`. [Decision 0015](../docs/decisions/0015-train-only-preprocessing.md) defines the isolation boundary. [Decision 0016](../docs/decisions/0016-fixed-split-leakage-regression.md) and `ml/tests/test_split_first_leakage.py` verify exact train graph, hash, and MF parameter invariance after held-out score and interaction-ID edits under fixed train membership. An added interaction changes the derived train split and requires a new experiment. The fit-only command reports no quality metric; the legacy MF training and LightGCN commands below remain incompatible with M5 release metrics.

The initial M5.4 selection path uses the fixed three-candidate invented grid in `fixtures/synthetic-mf-candidates.json`. Selection scores **validation only**, freezes one configuration in an ignored local file, then a separate command produces one final synthetic test report:

```bash
python ml/split_first_selection.py select --raw-ratings fixtures/synthetic-split-input.json --split-manifest fixtures/synthetic-split-manifest.json --metadata fixtures/synthetic-anime-metadata.json --candidates fixtures/synthetic-mf-candidates.json --out-selection data/synthetic-selection-local.json --out-test-report data/synthetic-final-test-local.json
python ml/split_first_selection.py report-test --raw-ratings fixtures/synthetic-split-input.json --split-manifest fixtures/synthetic-split-manifest.json --metadata fixtures/synthetic-anime-metadata.json --selection data/synthetic-selection-local.json
```

The second command refuses a repeat using a one-use marker beside the selection file. These ignored local reports are restricted evaluation records, and their tiny warm-user metrics do not establish ranking quality. [Decision 0017](../docs/decisions/0017-validation-test-boundary.md) defines the read boundary. M5.4's supported MF-grid, Optuna, and browser hybrid selectors now use validation-only choice with frozen one-use final reporting. The reserved new-user final cohort remains unscored; M5.6-M5.9 and the milestone exit gate remain open.

M5.5's [separate new-user fixture](../docs/decisions/0018-new-user-split-first-evaluation.md) has an invented fit-only raw snapshot with a pinned 72/24/24 split and four disjoint evaluation users. The local Python adapter exports only train-fitted item factors/biases, a train-present catalog, and positive train pairs to the TypeScript evaluator; it never exports fitted user factors or evaluation ratings. The TypeScript runner calls the browser's native-score mapper, signed preference-to-vector scorer, candidate eligibility policy, and final-list selector. It reports 1/3/5/10 supplied-rating model validation separately from the fit users' warm validation metric and labels any graph/catalog fallback:

```bash
npm run eval:new-user:fixture
```

The command uses only invented files and writes no model or report artifact. The fit-user test IDs remain unscored here. These small authored results are an isolation and serving-path check, not a release-quality estimate.

Decision [0020](../docs/decisions/0020-hybrid-validation-boundary.md) adds a predeclared five-weight hybrid selector over the same train-only model and browser eligibility/final-list path:

```bash
npm run eval:hybrid:split:select
```

This one-time command writes `data/synthetic-hybrid-selection-local.json` and refuses an existing selection or report. It only hashes the separate invented final cohort; it does not score it. The separate `report-test` command in `web/bench/split-first-hybrid-selection.ts` is guarded by a one-use marker and is reserved until later M5 baseline/reporting choices are fixed. The selected fixture weight is not promoted to the product.

[Decision 0021](../docs/decisions/0021-fair-synthetic-baselines.md) compares ten predeclared simple, similarity, content, graph, MF, and hybrid methods on the same centrally eligible invented validation titles:

```bash
npm run eval:baselines:fixture
```

The exporter derives popularity, support-shrunk adjusted cosine, pair edges, and four two-epoch MF variants from the validated training rows only. The browser evaluator gives missing method signal a neutral zero, applies the same candidate policy and final selector, and reports each method's NDCG@10 and signal coverage. Filling sparse lists makes this a common-candidate comparison; graph/content/hybrid rows are not a replay of the browser's usual candidate generation or fallback. The fixed metadata-score baseline uses authored fixture metadata; v1 pair-preference graph weights are not item similarity. The reserved final cohort stays unscored, and these authored validation numbers do not select a product model.

## Install

```bash
python -m pip install -r ml/requirements.txt
```

## Train

```bash
npm run data:fetch:release
python ml/train_graph_mf.py
```

Common options:

```bash
python ml/train_graph_mf.py \
  --factors 96 \
  --epochs 15 \
  --lr 0.015 \
  --graph-lambda 0.02 \
  --top-k 20
```

Artifacts are written to `models/graph_mf/`:

- `model.npz`
- `metrics.json`

## Hyperparameter Search (Optuna)

`npm run ml:search` visits each predeclared candidate in `fixtures/synthetic-mf-candidates.json`
once with an in-memory Optuna grid. The default inputs are the invented raw
snapshot, validated split manifest, and fixed metadata snapshot. It scores only
validation rows, freezes one ignored local selection file, and emits no test
metric. See [decision 0019](../docs/decisions/0019-split-first-optuna-selection.md).

```bash
npm run ml:search
python ml/split_first_selection.py report-test --raw-ratings fixtures/synthetic-split-input.json --split-manifest fixtures/synthetic-split-manifest.json --metadata fixtures/synthetic-anime-metadata.json --selection data/synthetic-optuna-selection-local.json
```

The separate report command checks the exact frozen snapshot and model, then
creates a one-use marker before scoring test. A broader search requires a
predeclared candidate JSON and private output paths via
`--candidates`, `--out-selection`, and `--out-test-report`. The legacy
`ml/train_graph_mf.py` training/evaluation route is still outside the M5
validation boundary.

## Periodic Retraining Workflow

GitHub Actions workflow:

- `.github/workflows/ml-retrain.yml`

It runs weekly (Monday 08:00 UTC) and can also be triggered manually.
Each run:

1. Installs Node + Python dependencies
2. Trains graph MF into `models/graph_mf_ci/`
3. Exports web model JSON
4. Uploads artifacts (`model.npz`, `metrics.json`, exported model JSON)

## Export Model For Website

The web app can switch between graph and ML recommendations when this file exists:

- `data/model-mf-web.compact.json` (preferred)
- `data/model-mf-web.json` (legacy)

Export it from a trained model:

```bash
python ml/export_model_web.py --model models/graph_mf/model.npz --out data/model-mf-web.compact.json
```

Then sync web data:

```bash
npm run sync:web
```

By default this creates `web/public/data/model-mf-web.compact.json.gz` (compressed).

## Recommend

Use watched anime IDs with optional weights:

```bash
python ml/recommend_graph_mf.py --watched "1535,9253:1.5,5114:0.7" --top-n 12
```

Use a trained user ID from the dataset:

```bash
python ml/recommend_graph_mf.py --user-id "0074dec428ed8832c68bcca6" --top-n 12
```

JSON output:

```bash
python ml/recommend_graph_mf.py --watched "1535,9253" --top-n 10 --json
```

## Cold-Start Mitigation (Content Features)

Build content features from Jikan metadata:

```bash
python ml/build_content_features.py --model models/graph_mf/model.npz --out models/graph_mf/content-features.json
```

Then recommendations will auto-blend MF with content similarity for sparse watched lists:

```bash
python ml/recommend_graph_mf.py --watched "1535,9253" --content-features models/graph_mf/content-features.json
```

Control blend behavior:

```bash
python ml/recommend_graph_mf.py --watched "1535,9253" --content-blend 0.35
```

## GNN Ranking Evaluation (LightGCN)

Evaluate a graph neural ranking baseline:

```bash
python ml/eval_gnn_lightgcn.py --epochs 12 --layers 3 --factors 64 --top-k 20
```

Or via npm:

```bash
npm run ml:gnn:eval -- --epochs 12 --layers 3
```

Outputs (default `models/gnn_lightgcn/`):

- `model-lightgcn.npz`
- `metrics-lightgcn.json`
