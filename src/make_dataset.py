"""Materialize the dataset as a CSV file so DVC can version it.

In a real project this stage would be "export from the warehouse". Here it
generates data deterministically from params.yaml, which lets us demo data
versioning without any external data source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import generate_synthetic_data, validate_data
from src.params import load_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the churn dataset to a CSV file.")
    parser.add_argument("--out", default="data/churn.csv")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    cfg = load_params(args.params)["dataset"]
    df = generate_synthetic_data(
        n_samples=cfg["n_samples"], random_state=cfg["seed"], label_noise=cfg["label_noise"]
    )
    validate_data(df)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
