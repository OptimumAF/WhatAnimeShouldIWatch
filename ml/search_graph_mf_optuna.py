#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import optuna

import train_graph_mf as trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for graph-regularized MF."
    )
    parser.add_argument(
        "--ratings",
        default="data/anonymized-ratings.compact.json",
        help="Path to ratings dataset JSON (compact or legacy)",
    )
    parser.add_argument(
        "--graph",
        default="data/graph.compact.json",
        help="Path to graph JSON (compact or legacy)",
    )
    parser.add_argument(
        "--study-dir",
        default="models/graph_mf_search",
        help="Output directory for study DB and reports",
    )
    parser.add_argument(
        "--study-name",
        default="graph_mf_optuna",
        help="Optuna study name",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Stop search after this many seconds (0 disables timeout)",
    )
    parser.add_argument(
        "--metric",
        choices=["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"],
        default="ndcg_at_k",
        help="Ranking metric to maximize",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="K for ranking metrics",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Per-user held-out ratio for evaluation",
    )
    parser.add_argument(
        "--min-ratings-for-test",
        type=int,
        default=8,
        help="Minimum ratings needed for a user to get held-out test ratings",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Held-out rating threshold considered positive for ranking metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split and Optuna sampler",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Optuna workers (n_jobs). Use 1 for deterministic runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ratings_path = trainer.resolve_with_fallback(
        Path(args.ratings),
        [Path("data/anonymized-ratings.json")],
    )
    graph_path = trainer.resolve_with_fallback(
        Path(args.graph),
        [Path("data/graph.json")],
    )
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    study_dir = Path(args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    study_db_path = study_dir / f"{args.study_name}.sqlite3"
    storage = f"sqlite:///{study_db_path.resolve().as_posix()}"

    print("Loading dataset and graph...")
    dataset = trainer.load_dataset(ratings_path)
    anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(dataset.anime_ids)}
    split = trainer.split_train_test(
        dataset=dataset,
        test_ratio=args.test_ratio,
        min_ratings_for_test=args.min_ratings_for_test,
        positive_threshold=args.positive_threshold,
        seed=args.seed,
    )
    base_graph_edges = trainer.load_anime_graph_edges(
        graph_path=graph_path,
        anime_id_to_idx=anime_id_to_idx,
        min_abs_weight=0.0,
    )
    print(
        f"Loaded users={len(dataset.user_ids)} anime={len(dataset.anime_ids)} "
        f"interactions={len(dataset.interactions)} graph_edges={base_graph_edges.shape[0]}"
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        factors = trial.suggest_int("factors", 32, 128, step=16)
        epochs = trial.suggest_int("epochs", 6, 18)
        lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
        reg = trial.suggest_float("reg", 1e-4, 5e-2, log=True)
        reg_bias = trial.suggest_float("reg_bias", 1e-4, 2e-2, log=True)
        graph_lambda = trial.suggest_float("graph_lambda", 1e-4, 1e-1, log=True)
        graph_sample_rate = trial.suggest_float("graph_sample_rate", 0.25, 1.0)
        graph_min_abs_weight = trial.suggest_float("graph_min_abs_weight", 0.0, 1.5)

        if graph_min_abs_weight > 0.0 and base_graph_edges.shape[0] > 0:
            edge_batch = base_graph_edges[base_graph_edges[:, 2] >= graph_min_abs_weight]
        else:
            edge_batch = base_graph_edges

        model = trainer.train_graph_mf(
            split=split,
            n_users=len(dataset.user_ids),
            n_items=len(dataset.anime_ids),
            graph_edges=edge_batch,
            factors=factors,
            epochs=epochs,
            lr=lr,
            reg=reg,
            reg_bias=reg_bias,
            graph_lambda=graph_lambda,
            graph_sample_rate=graph_sample_rate,
            seed=args.seed + trial.number,
        )

        metrics = trainer.evaluate_ranking(model=model, split=split, top_k=args.top_k)
        score = float(metrics[args.metric])
        for key, value in metrics.items():
            trial.set_user_attr(key, float(value))
        trial.set_user_attr("graph_edges_used", int(edge_batch.shape[0]))
        trial.set_user_attr("objective", args.metric)
        return score

    timeout = args.timeout_sec if args.timeout_sec > 0 else None
    started_at = time.time()
    study.optimize(
        objective,
        n_trials=max(1, args.trials),
        timeout=timeout,
        n_jobs=max(1, args.workers),
        gc_after_trial=True,
        show_progress_bar=False,
    )
    elapsed = time.time() - started_at

    report_path = study_dir / f"{args.study_name}.best.json"
    trials_path = study_dir / f"{args.study_name}.trials.json"
    best = study.best_trial
    best_report: Dict[str, Any] = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "studyName": args.study_name,
        "storage": str(study_db_path),
        "objective": args.metric,
        "bestValue": float(best.value),
        "bestParams": best.params,
        "bestMetrics": {
            "precisionAtK": _trial_attr_float(best, "precision_at_k"),
            "recallAtK": _trial_attr_float(best, "recall_at_k"),
            "hitRateAtK": _trial_attr_float(best, "hit_rate_at_k"),
            "ndcgAtK": _trial_attr_float(best, "ndcg_at_k"),
        },
        "data": {
            "ratingsPath": str(ratings_path),
            "graphPath": str(graph_path),
            "users": len(dataset.user_ids),
            "anime": len(dataset.anime_ids),
            "interactions": len(dataset.interactions),
            "trainInteractions": int(split.train_r.shape[0]),
            "usersWithHeldout": split.users_with_test,
        },
        "search": {
            "trialsRequested": args.trials,
            "trialsCompleted": len(study.trials),
            "timeoutSec": args.timeout_sec,
            "elapsedSec": elapsed,
            "seed": args.seed,
            "topK": args.top_k,
        },
    }
    report_path.write_text(json.dumps(best_report, indent=2), encoding="utf-8")

    ranked_trials = sorted(
        (
            trial
            for trial in study.trials
            if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE
        ),
        key=lambda trial: float(trial.value),
        reverse=True,
    )
    top_trials = [
        {
            "number": trial.number,
            "value": float(trial.value),
            "params": trial.params,
            "precisionAtK": _trial_attr_float(trial, "precision_at_k"),
            "recallAtK": _trial_attr_float(trial, "recall_at_k"),
            "hitRateAtK": _trial_attr_float(trial, "hit_rate_at_k"),
            "ndcgAtK": _trial_attr_float(trial, "ndcg_at_k"),
            "graphEdgesUsed": _trial_attr_int(trial, "graph_edges_used"),
        }
        for trial in ranked_trials[: min(25, len(ranked_trials))]
    ]
    trials_path.write_text(json.dumps(top_trials, indent=2), encoding="utf-8")

    print(
        f"Search complete. best_{args.metric}={float(best.value):.6f} "
        f"trial={best.number} elapsed={elapsed:.1f}s"
    )
    print(f"Saved best report -> {report_path}")
    print(f"Saved ranked trials -> {trials_path}")


def _trial_attr_float(trial: optuna.Trial, key: str) -> float:
    value = trial.user_attrs.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _trial_attr_int(trial: optuna.Trial, key: str) -> int:
    value = trial.user_attrs.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
