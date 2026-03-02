# ML Recommender (Graph-Regularized Matrix Factorization)

This folder trains a recommendation model from:

- `data/anonymized-ratings.compact.json` (preferred) or `data/anonymized-ratings.json`
- `data/graph.compact.json` (preferred) or `data/graph.json`

The model is matrix factorization with an extra graph regularization term on anime embeddings.

## Install

```bash
python -m pip install -r ml/requirements.txt
```

## Train

```bash
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

Run Optuna search against the same train/test split and graph data:

```bash
python ml/search_graph_mf_optuna.py --trials 40 --metric ndcg_at_k --top-k 20
```

Or through npm:

```bash
npm run ml:search -- --trials 40 --metric ndcg_at_k
```

Search artifacts are written to `models/graph_mf_search/` by default:

- `<study-name>.sqlite3` (Optuna study DB)
- `<study-name>.best.json` (best params + best metrics)
- `<study-name>.trials.json` (top completed trials)

Use the best parameters to run a full training pass:

```bash
python ml/train_graph_mf.py --factors 96 --epochs 14 --lr 0.012 --reg 0.008 --reg-bias 0.004 --graph-lambda 0.02 --graph-sample-rate 0.7 --graph-min-abs-weight 0.35
```

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
