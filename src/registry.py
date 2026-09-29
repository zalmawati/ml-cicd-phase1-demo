"""Model Registry helpers.

Concepts:
  * A *registered model* is a named lineage ("churn-model").
  * Each registration creates a numbered *version* (1, 2, 3, ...).
  * *Aliases* are movable labels pointing at a version. We use two:
        candidate  -> the newest model that has not been vetted
        champion   -> the model consumers should actually use
    Consumers load "models:/churn-model@champion" and never care which
    version number that currently is. Promotion = moving the alias.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.tracking import resolve_tracking_uri

logger = logging.getLogger(__name__)

CANDIDATE = "candidate"
CHAMPION = "champion"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(resolve_tracking_uri())
    return MlflowClient()


def register_candidate(model_uri: str, name: str) -> int:
    """Register a logged model as a new version and label it `candidate`."""
    mlflow.set_tracking_uri(resolve_tracking_uri())
    mv = mlflow.register_model(model_uri, name)
    _client().set_registered_model_alias(name, CANDIDATE, mv.version)
    logger.info("Registered %s version %s as '%s'", name, mv.version, CANDIDATE)
    return int(mv.version)


def _metric_of_version(client: MlflowClient, name: str, version: str, metric: str) -> float:
    mv = client.get_model_version(name, version)
    return client.get_run(mv.run_id).data.metrics[metric]


def promote_if_better(name: str, metric: str = "roc_auc", min_gain: float = 0.0) -> bool:
    """Move the `champion` alias to `candidate` only if it beats the current champion.

    Returns True if a promotion happened. With no champion yet, the candidate wins.
    """
    client = _client()
    cand = client.get_model_version_by_alias(name, CANDIDATE)
    cand_score = _metric_of_version(client, name, cand.version, metric)

    try:
        champ = client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        client.set_registered_model_alias(name, CHAMPION, cand.version)
        logger.info("No champion yet: v%s (%s=%.4f) is the first", cand.version, metric, cand_score)
        return True

    if champ.version == cand.version:
        logger.info("Candidate v%s is already champion", cand.version)
        return False

    champ_score = _metric_of_version(client, name, champ.version, metric)
    if cand_score > champ_score + min_gain:
        client.set_registered_model_alias(name, CHAMPION, cand.version)
        logger.info(
            "Promoted v%s (%s=%.4f) over v%s (%.4f)",
            cand.version,
            metric,
            cand_score,
            champ.version,
            champ_score,
        )
        return True

    logger.info(
        "Kept v%s (%s=%.4f); candidate v%s (%.4f) did not beat it",
        champ.version,
        metric,
        champ_score,
        cand.version,
        cand_score,
    )
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Model registry helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    promo = sub.add_parser("promote", help="Promote candidate to champion if it is better")
    promo.add_argument("--name", required=True)
    promo.add_argument("--metric", default="roc_auc")
    promo.add_argument("--min-gain", type=float, default=0.0)
    args = parser.parse_args()
    if args.cmd == "promote":
        promote_if_better(args.name, args.metric, args.min_gain)


if __name__ == "__main__":
    main()
