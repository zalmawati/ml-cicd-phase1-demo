"""Synthetic data generation and validation utilities.

In a real project this module would pull from a warehouse or feature store.
Here we generate synthetic churn-like data so the whole pipeline is
self-contained, reproducible, and runnable with zero external dependencies -
useful for a CI environment that shouldn't need a database connection.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification

REQUIRED_COLUMNS = ["tenure_months", "monthly_spend", "support_tickets", "churn"]


def load_csv(path: str) -> pd.DataFrame:
    """Load a dataset produced by `src.make_dataset` (or any CSV with our schema)."""
    return pd.read_csv(path)


class DataValidationError(Exception):
    """Raised when the input data fails a quality check."""


def generate_synthetic_data(
    n_samples: int = 2000, random_state: int = 42, label_noise: float = 0.0
) -> pd.DataFrame:
    """Generate a synthetic churn dataset.

    Stands in for "pull last 90 days of customer data from the warehouse"
    in a real pipeline. `label_noise` randomly flips that fraction of labels,
    which is how we simulate a "messier" version of the dataset in Phase 2.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],
        flip_y=label_noise,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=["tenure_months", "monthly_spend", "support_tickets"])

    # Rescale raw feature space into plausible business ranges.
    def _rescale(col: pd.Series, low: float, high: float) -> pd.Series:
        return (col - col.min()) / (col.max() - col.min()) * (high - low) + low

    df["tenure_months"] = _rescale(df["tenure_months"], 0, 72).round(1)
    df["monthly_spend"] = _rescale(df["monthly_spend"], 10, 210).round(2)
    df["support_tickets"] = _rescale(df["support_tickets"], 0, 10).round().astype(int)
    df["churn"] = y
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Run schema and sanity checks on the dataframe.

    This is a lightweight, dependency-free stand-in for a tool like Great
    Expectations (that's covered properly in Phase 4). The point here is the
    *pattern*: validation is a function that raises on failure, so it can be
    dropped into a pipeline or a CI job as a hard gate.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    if df.empty:
        raise DataValidationError("Dataframe is empty.")

    if df["tenure_months"].lt(0).any() or df["tenure_months"].gt(120).any():
        raise DataValidationError("tenure_months out of expected range [0, 120].")

    if df["monthly_spend"].lt(0).any():
        raise DataValidationError("monthly_spend cannot be negative.")

    if not set(df["churn"].unique()).issubset({0, 1}):
        raise DataValidationError("churn must be binary (0 or 1).")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        bad = null_counts[null_counts > 0].to_dict()
        raise DataValidationError(f"Null values found: {bad}")
