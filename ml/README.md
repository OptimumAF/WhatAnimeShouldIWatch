# ML Recommender (Graph-Regularized Matrix Factorization)

This folder trains a recommendation model from:

- `data/anonymized-ratings.json` (user-anime normalized ratings)
- `data/graph.json` (anime-anime similarity edges)

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

## Export Model For Website

The web app can switch between graph and ML recommendations when this file exists:

- `data/model-mf-web.json`

Export it from a trained model:

```bash
python ml/export_model_web.py --model models/graph_mf/model.npz --out data/model-mf-web.json
```

Then sync web data:

```bash
npm run sync:web
```

By default this creates `web/public/data/model-mf-web.json.gz` (compressed).

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
