"""What a *consumer* of the model does.

Notice there is no version number, run id or file path here. The consumer
only knows the registered name and the alias. When you promote a better
model, this code picks it up without any change.
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.tracking import resolve_tracking_uri


def load_champion(name: str, alias: str = "champion"):
    mlflow.set_tracking_uri(resolve_tracking_uri())
    return mlflow.pyfunc.load_model(f"models:/{name}@{alias}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict churn with the champion model.")
    parser.add_argument("--name", default="churn-model")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--tenure-months", type=float, default=12.0)
    parser.add_argument("--monthly-spend", type=float, default=80.0)
    parser.add_argument("--support-tickets", type=int, default=3)
    args = parser.parse_args()

    model = load_champion(args.name, args.alias)
    row = pd.DataFrame(
        [
            {
                "tenure_months": float(args.tenure_months),
                "monthly_spend": float(args.monthly_spend),
                "support_tickets": int(args.support_tickets),
            }
        ]
    )
    print("churn prediction (1 = will churn):", int(model.predict(row)[0]))


if __name__ == "__main__":
    main()
