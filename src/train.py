"""Training entrypoint, now with experiment tracking.

Same job as Phase 1 (train, evaluate, write model + metrics), plus every run
is recorded in MLflow: parameters, metrics, the model itself, and lineage tags.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import generate_synthetic_data, load_csv, validate_data
from src.params import load_params
from src.tracking import configure_tracking, lineage_tags

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "churn-model"

# MLflow 3.x saves sklearn models with `skops` (safer than pickle) and refuses to
# save types it hasn't been told to trust. We trained this forest ourselves, so we
# trust exactly the one type a RandomForest needs - not "everything".
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def split(df: pd.DataFrame, test_size: float, random_state: int):
    X = df.drop(columns=["churn"])
    y = df["churn"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def fit_and_evaluate(
    X_train, X_test, y_train, y_test, n_estimators: int, max_depth: int, random_state: int
):
    """Pure ML logic, no tracking. Shared by `train` and the hyperparameter sweep."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }
    return model, metrics


def train(
    output_dir: str = "artifacts",
    params_path: str = "params.yaml",
    data_path: str | None = None,
    register_as: str | None = None,
) -> dict:
    """Train, track in MLflow, write model + metrics to output_dir.

    Returns {"run_id": ..., "metrics": {...}, "model_version": int | None}.
    """
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
    with mlflow.start_run() as run:
        logger.info("MLflow run %s", run.info.run_id)
        mlflow.log_params({f"train.{k}": v for k, v in tp.items()})
        if not data_path:
            mlflow.log_params({f"dataset.{k}": v for k, v in params["dataset"].items()})
        mlflow.set_tags(lineage_tags(data_path))

        with warnings.catch_warnings():
            # MLflow warns that a local file path is "ambiguous" between two identical
            # source classes. Harmless, but it would print on every DVC-driven run.
            warnings.filterwarnings("ignore", message=".*interpreted in multiple ways.*")
            dataset = mlflow.data.from_pandas(df, source=data_path, name="churn", targets="churn")
        mlflow.log_input(dataset, context="training")

        model, metrics = fit_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            n_estimators=tp["n_estimators"],
            max_depth=tp["max_depth"],
            random_state=tp["random_state"],
        )
        logger.info("Metrics: %s", metrics)
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({"n_train": len(X_train), "n_test": len(X_test)})

        importances = dict(zip(X_train.columns, model.feature_importances_.round(4).tolist()))
        mlflow.log_dict(importances, "feature_importances.json")

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
            skops_trusted_types=TRUSTED_TYPES,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

        version = None
        if register_as:
            from src.registry import register_candidate

            version = register_candidate(model_info.model_uri, register_as)

        return {"run_id": run.info.run_id, "metrics": metrics, "model_version": version}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model and track it in MLflow.")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data-path", default=None, help="CSV to train on (default: synthetic)")
    parser.add_argument("--register-as", default=None, help="Registered model name")
    args = parser.parse_args()
    result = train(args.output_dir, args.params, args.data_path, args.register_as)
    logger.info("Done: %s", json.dumps(result))


if __name__ == "__main__":
    main()
