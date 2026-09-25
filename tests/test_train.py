import json

from src.train import train


def test_train_produces_artifacts(tmp_path):
    output_dir = tmp_path / "artifacts"
    metrics = train(output_dir=str(output_dir))

    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metrics.json").exists()

    saved_metrics = json.loads((output_dir / "metrics.json").read_text())
    assert saved_metrics == metrics


def test_train_meets_minimum_quality_bar(tmp_path):
    """A preview of the Phase 4 idea: fail the build if the model is unusably bad."""
    metrics = train(output_dir=str(tmp_path / "artifacts"))
    assert metrics["roc_auc"] > 0.6, "Model ROC-AUC dropped below acceptable threshold"
