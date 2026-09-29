"""MLflow setup and lineage helpers shared by training, sweeps and the registry.

The idea: every run should be able to answer "which code, which data, which
image produced this model?" without anyone remembering. We do that by stamping
each run with tags (git commit, image tag, data hash).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
import requests

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def resolve_tracking_uri() -> str:
    """Server URL from MLFLOW_TRACKING_URI, else a local SQLite file."""
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def configure_tracking(experiment: str) -> str:
    """Point MLflow at the tracking server and select the experiment.

    For http(s) servers we do a quick health check first. Without it, MLflow
    silently retries for minutes when the server is down, and your Airflow task
    just looks hung instead of failing with a useful message.
    """
    uri = resolve_tracking_uri()
    if uri.startswith(("http://", "https://")):
        try:
            requests.get(uri.rstrip("/") + "/health", timeout=5).raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach MLflow at {uri} ({exc.__class__.__name__}). "
                "Is the server running, and is MLFLOW_TRACKING_URI correct? "
                "From inside a Docker container, 127.0.0.1 is the container itself - "
                "use http://host.docker.internal:5000 instead."
            ) from exc
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def git_sha() -> str:
    """Commit that produced this run: baked into the image (GIT_SHA), else asked from git."""
    env = os.getenv("GIT_SHA")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_dirty() -> str:
    """ "true" if the working tree had uncommitted changes when the run started.

    A git_sha tag is only trustworthy for reproducing a run if this is "false":
    otherwise the code that ran is NOT exactly what that commit contains.
    Images built by CI come from a clean checkout, so they are always "false".
    """
    if os.getenv("GIT_SHA"):
        return "false"
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=5
        )
        return "true" if out.stdout.strip() else "false"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def file_sha256(path: str) -> str:
    """Content hash of a data file, so a run records exactly which bytes it trained on."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lineage_tags(data_path: str | None = None) -> dict[str, str]:
    tags = {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "image_tag": os.getenv("IMAGE_TAG", "local"),
        "triggered_by": os.getenv("TRIGGERED_BY", "manual"),
    }
    if data_path:
        tags["data_path"] = data_path
        tags["data_sha256"] = file_sha256(data_path)
    else:
        tags["data_path"] = "synthetic (generated in code)"
    return tags
