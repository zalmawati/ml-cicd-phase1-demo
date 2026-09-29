import pytest


@pytest.fixture
def tracking_env(tmp_path, monkeypatch):
    """Give each test its own throwaway MLflow (SQLite file + artifacts) in a temp dir.

    CI never needs a running tracking server; the tracking *code* is still fully exercised.

    The URI must be absolute and unique per test: MLflow caches stores by URI string, so a
    relative "sqlite:///mlflow.db" would silently share one database across every test.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("GIT_SHA", "testsha123")
    monkeypatch.setenv("IMAGE_TAG", "v0.0.test")
    return tmp_path
