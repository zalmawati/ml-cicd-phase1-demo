import pytest

from src.data import DataValidationError, generate_synthetic_data, validate_data


def test_generate_synthetic_data_shape():
    df = generate_synthetic_data(n_samples=100)
    assert len(df) == 100
    expected_cols = {"tenure_months", "monthly_spend", "support_tickets", "churn"}
    assert expected_cols.issubset(df.columns)


def test_validate_data_passes_on_good_data():
    df = generate_synthetic_data(n_samples=50)
    validate_data(df)  # should not raise


def test_validate_data_missing_column():
    df = generate_synthetic_data(n_samples=10).drop(columns=["churn"])
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_data(df)


def test_validate_data_negative_spend():
    df = generate_synthetic_data(n_samples=10)
    df.loc[0, "monthly_spend"] = -5
    with pytest.raises(DataValidationError, match="monthly_spend cannot be negative"):
        validate_data(df)


def test_validate_data_invalid_churn_label():
    df = generate_synthetic_data(n_samples=10)
    df.loc[0, "churn"] = 2
    with pytest.raises(DataValidationError, match="churn must be binary"):
        validate_data(df)


def test_validate_data_empty_dataframe():
    df = generate_synthetic_data(n_samples=10).iloc[0:0]
    with pytest.raises(DataValidationError, match="empty"):
        validate_data(df)


def test_label_noise_changes_the_labels():
    clean = generate_synthetic_data(n_samples=500, label_noise=0.0)
    noisy = generate_synthetic_data(n_samples=500, label_noise=0.2)
    assert (clean["churn"] != noisy["churn"]).any()


def test_load_csv_roundtrip(tmp_path):
    from src.data import load_csv

    df = generate_synthetic_data(n_samples=20)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    validate_data(load_csv(str(path)))
