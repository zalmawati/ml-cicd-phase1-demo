import json

from src.data import generate_synthetic_data
from src.train import train


def test_train_produces_artifacts(tracking_env):
    out = tracking_env / "artifacts"
    result = train(output_dir=str(out))

    assert (out / "model.joblib").exists()
    assert (out / "metrics.json").exists()
    assert json.loads((out / "metrics.json").read_text()) == result["metrics"]
    assert result["run_id"]


def test_train_meets_minimum_quality_bar(tracking_env):
    """A preview of the Phase 4 idea: fail the build if the model is unusably bad."""
    result = train(output_dir=str(tracking_env / "artifacts"))
    assert result["metrics"]["roc_auc"] > 0.6, "Model ROC-AUC dropped below acceptable threshold"


def test_train_from_csv_file(tracking_env):
    csv = tracking_env / "data" / "churn.csv"
    csv.parent.mkdir()
    generate_synthetic_data(n_samples=500).to_csv(csv, index=False)

    result = train(output_dir=str(tracking_env / "out"), data_path=str(csv))
    assert result["metrics"]["roc_auc"] > 0.6
