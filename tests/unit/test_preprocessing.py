"""Unit tests for feature preprocessing and leakage safeguards."""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import FraudFeaturePreprocessor


def test_preprocessor_fit_transform(synthetic_df: pd.DataFrame):
    X = synthetic_df.drop(columns=["Class"])
    preprocessor = FraudFeaturePreprocessor()
    X_proc = preprocessor.fit_transform(X)

    assert preprocessor.is_fitted_ is True
    assert X_proc.shape == X.shape
    assert not np.isnan(X_proc).any()
    assert not np.isinf(X_proc).any()


def test_transform_before_fit_raises_runtime_error(synthetic_df: pd.DataFrame):
    X = synthetic_df.drop(columns=["Class"])
    preprocessor = FraudFeaturePreprocessor()
    with pytest.raises(RuntimeError, match="Preprocessor has not been fitted yet"):
        preprocessor.transform(X)


def test_missing_column_at_inference_raises_value_error(synthetic_df: pd.DataFrame):
    X = synthetic_df.drop(columns=["Class"])
    preprocessor = FraudFeaturePreprocessor()
    preprocessor.fit(X)

    # Missing V3
    bad_payload = X.drop(columns=["V3"])
    with pytest.raises(ValueError, match="Missing expected feature columns"):
        preprocessor.transform(bad_payload)


def test_single_dict_and_list_transform(synthetic_df: pd.DataFrame, valid_transaction_dict: dict):
    X = synthetic_df.drop(columns=["Class"])
    preprocessor = FraudFeaturePreprocessor()
    preprocessor.fit(X)

    # Single dict
    proc_single = preprocessor.transform(valid_transaction_dict)
    assert proc_single.shape == (1, X.shape[1])

    # List of dicts
    proc_list = preprocessor.transform([valid_transaction_dict, valid_transaction_dict])
    assert proc_list.shape == (2, X.shape[1])
