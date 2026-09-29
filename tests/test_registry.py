import pytest

from src.registry import CANDIDATE, CHAMPION, promote_if_better
from src.train import train

NAME = "test-churn-model"


@pytest.fixture
def weak_params(tracking_env):
    p = tracking_env / "weak.yaml"
    p.write_text("train: {n_estimators: 2, max_depth: 1}\n")
    return str(p)


def _alias_version(name, alias):
    from mlflow import MlflowClient

    return int(MlflowClient().get_model_version_by_alias(name, alias).version)


def test_first_candidate_becomes_champion(tracking_env, weak_params):
    train(output_dir=str(tracking_env / "o1"), params_path=weak_params, register_as=NAME)
    assert _alias_version(NAME, CANDIDATE) == 1
    assert promote_if_better(NAME) is True
    assert _alias_version(NAME, CHAMPION) == 1


def test_better_model_replaces_champion_and_worse_does_not(tracking_env, weak_params):
    train(output_dir=str(tracking_env / "o1"), params_path=weak_params, register_as=NAME)
    promote_if_better(NAME)  # v1 (weak) is champion

    train(output_dir=str(tracking_env / "o2"), register_as=NAME)  # v2, default params: stronger
    assert promote_if_better(NAME) is True
    assert _alias_version(NAME, CHAMPION) == 2

    train(output_dir=str(tracking_env / "o3"), params_path=weak_params, register_as=NAME)  # v3
    assert promote_if_better(NAME) is False  # weak candidate must NOT dethrone v2
    assert _alias_version(NAME, CHAMPION) == 2
