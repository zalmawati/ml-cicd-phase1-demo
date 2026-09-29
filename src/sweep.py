"""Grid search where every combination is its own tracked run.

Structure in the MLflow UI:
    sweep-<n>                (parent run: holds the winner)
      ├── n50_d3             (child run per combination)
      ├── n50_d6
      └── ...
Comparing children (parallel-coordinates plot, sort by roc_auc) is the whole
point: you can see which hyperparameters mattered instead of guessing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging

import mlflow

from src.data import generate_synthetic_data, load_csv, validate_data
from src.params import load_params
from src.tracking import configure_tracking, lineage_tags
from src.train import EXPERIMENT_NAME, fit_and_evaluate, split

logger = logging.getLogger(__name__)

GRID = {"n_estimators": [50, 100, 200, 300], "max_depth": [3, 6, 10]}


def run_sweep(params_path: str = "params.yaml", data_path: str | None = None, grid=None) -> dict:
    grid = grid or GRID
    params = load_params(params_path)
    tp = params["train"]

    if data_path:
        df = load_csv(data_path)
    else:
        cfg = params["dataset"]
        df = generate_synthetic_data(
            n_samples=cfg["n_samples"], random_state=cfg["seed"], label_noise=cfg["label_noise"]
        )
    validate_data(df)
    X_train, X_test, y_train, y_test = split(df, tp["test_size"], tp["random_state"])

    configure_tracking(EXPERIMENT_NAME)
    combos = list(itertools.product(grid["n_estimators"], grid["max_depth"]))

    with mlflow.start_run(run_name="sweep") as parent:
        mlflow.set_tags({**lineage_tags(data_path), "run_type": "sweep"})
        mlflow.log_param("n_combinations", len(combos))

        best = {"roc_auc": -1.0}
        for n_estimators, max_depth in combos:
            with mlflow.start_run(run_name=f"n{n_estimators}_d{max_depth}", nested=True) as child:
                mlflow.log_params(
                    {"train.n_estimators": n_estimators, "train.max_depth": max_depth}
                )
                _, metrics = fit_and_evaluate(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=tp["random_state"],
                )
                mlflow.log_metrics(metrics)
                if metrics["roc_auc"] > best["roc_auc"]:
                    best = {
                        **metrics,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "run_id": child.info.run_id,
                    }

        mlflow.log_metrics({f"best_{k}": v for k, v in best.items() if k in ("roc_auc", "f1")})
        mlflow.log_dict(best, "best_params.json")
        logger.info("Best combination: %s", json.dumps(best))
        return {"parent_run_id": parent.info.run_id, "best": best}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run a tracked hyperparameter sweep.")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()
    result = run_sweep(args.params, args.data_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
