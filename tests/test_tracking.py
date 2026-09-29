import hashlib

import pytest
from mlflow import MlflowClient

from src import tracking
from src.data import generate_synthetic_data
from src.train import EXPERIMENT_NAME, train


def test_git_sha_prefers_env(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "deadbeef")
    assert tracking.git_sha() == "deadbeef"


def test_git_dirty_is_false_for_ci_built_images(monkeypatch):
    monkeypatch.setenv("GIT_SHA", "deadbeef")  # set at image build time => clean checkout
    assert tracking.git_dirty() == "false"


def test_file_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"a,b\n1,2\n")
    assert tracking.file_sha256(str(f)) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_lineage_tags_include_data_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_TAG", "v1.2.3")
    f = tmp_path / "d.csv"
    f.write_text("x\n1\n")
    tags = tracking.lineage_tags(str(f))
    assert tags["image_tag"] == "v1.2.3"
    assert len(tags["data_sha256"]) == 64


def test_configure_tracking_fails_fast_when_server_is_down(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # nothing listens here
    with pytest.raises(RuntimeError, match="Cannot reach MLflow"):
        tracking.configure_tracking("whatever")


def test_run_records_params_metrics_tags_and_model(tracking_env):
    csv = tracking_env / "churn.csv"
    generate_synthetic_data(n_samples=400).to_csv(csv, index=False)
    result = train(output_dir=str(tracking_env / "out"), data_path=str(csv))

    client = MlflowClient()
    run = client.get_run(result["run_id"])

    assert run.data.params["train.n_estimators"] == "200"
    assert run.data.metrics["roc_auc"] == result["metrics"]["roc_auc"]
    assert run.data.tags["git_sha"] == "testsha123"
    assert run.data.tags["image_tag"] == "v0.0.test"
    assert run.data.tags["data_sha256"] == tracking.file_sha256(str(csv))

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    models = client.search_logged_models(experiment_ids=[exp.experiment_id])
    assert len(models) == 1
