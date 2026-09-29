"""Load params.yaml, falling back to safe defaults for anything missing.

The defaults matter: the Docker image and the unit tests both run without a
params.yaml next to them, and must still behave sensibly.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_PARAMS = {
    "dataset": {"n_samples": 2000, "seed": 42, "label_noise": 0.05},
    "train": {"n_estimators": 200, "max_depth": 6, "test_size": 0.2, "random_state": 42},
}


def load_params(path: str = "params.yaml") -> dict:
    params = copy.deepcopy(DEFAULT_PARAMS)
    p = Path(path)
    if p.exists():
        loaded = yaml.safe_load(p.read_text()) or {}
        for section, values in loaded.items():
            params.setdefault(section, {}).update(values or {})
    return params
