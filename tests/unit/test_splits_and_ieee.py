"""Unit tests for chronological and stratified dataset splitting, and IEEE-CIS pipeline."""

import numpy as np
import pandas as pd

from scripts.run_ieee_cis_experiment import (
    downcast_dtypes,
    generate_mock_ieee_dataset,
    run_experiment,
)
from src.data.loader import DataLoader


def test_stratified_vs_chronological_splits(synthetic_df: pd.DataFrame):
    loader = DataLoader()

    # 1. Stratified Split
    x_tr_s, y_tr_s, x_va_s, y_va_s, x_te_s, y_te_s = loader.split_data(
        synthetic_df, strategy="stratified"
    )
    assert len(x_tr_s) + len(x_va_s) + len(x_te_s) == len(synthetic_df)
    assert y_tr_s.sum() > 0
    assert y_va_s.sum() > 0
    assert y_te_s.sum() > 0

    # 2. Chronological Split
    x_tr_c, y_tr_c, x_va_c, y_va_c, x_te_c, y_te_c = loader.split_data(
        synthetic_df, strategy="chronological"
    )
    assert len(x_tr_c) + len(x_va_c) + len(x_te_c) == len(synthetic_df)
    # Check temporal ordering: max Time in train <= min Time in val
    assert x_tr_c["Time"].max() <= x_va_c["Time"].min()
    assert x_va_c["Time"].max() <= x_te_c["Time"].min()


def test_ieee_mock_experiment_execution():
    df_mock = generate_mock_ieee_dataset(n_samples=600)
    assert len(df_mock) == 600
    assert "TransactionDT" in df_mock.columns
    assert "isFraud" in df_mock.columns

    # Test downcasting
    df_downcast = downcast_dtypes(df_mock)
    assert df_downcast["TransactionAmt"].dtype == np.float32

    # Test experiment running
    # Convert string columns to categorical codes for HistGradientBoosting
    for col in df_downcast.select_dtypes(include=["object", "string"]).columns:
        df_downcast[col] = df_downcast[col].astype("category").cat.codes

    results = run_experiment(df_downcast)
    assert results["dataset"] == "IEEE-CIS Fraud Detection"
    assert "dummy_baseline" in results["models"]
    assert "hist_gradient_boosting" in results["models"]

    hgb_metrics = results["models"]["hist_gradient_boosting"]
    assert 0.0 <= hgb_metrics["pr_auc"] <= 1.0
    assert 0.0 <= hgb_metrics["roc_auc"] <= 1.0
    assert "optimal_threshold" in hgb_metrics
