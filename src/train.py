"""Training entrypoint.

This is the script your CI pipeline lints, tests, containerizes, pushes to
a registry, and eventually runs from Airflow instead of a human typing
`python train.py` on their laptop.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import generate_synthetic_data, validate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train(output_dir: str = "artifacts", random_state: int = 42) -> dict:
    """Train a churn classifier and write the model + metrics to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Generating and validating data...")
    df = generate_synthetic_data(random_state=random_state)
    validate_data(df)

    X = df.drop(columns=["churn"])
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    logger.info("Training model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    logger.info("Metrics: %s", metrics)

    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument("--output-dir", default="artifacts", help="Where to write model + metrics")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(output_dir=args.output_dir, random_state=args.random_state)


if __name__ == "__main__":
    main()
