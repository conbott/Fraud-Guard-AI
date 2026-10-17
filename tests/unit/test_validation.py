"""Unit tests for data validation logic."""

import numpy as np
import pandas as pd

from src.data.validator import DataValidator


def test_valid_synthetic_data_passes(synthetic_df: pd.DataFrame):
    validator = DataValidator()
    result = validator.validate(synthetic_df, is_training=True)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.row_count == len(synthetic_df)


def test_missing_required_column_fails(synthetic_df: pd.DataFrame):
    df_missing = synthetic_df.drop(columns=["V14"])
    validator = DataValidator()
    result = validator.validate(df_missing, is_training=True)
    assert result.is_valid is False
    assert any(err.category == "MISSING_COLUMNS" for err in result.errors)


def test_missing_values_fail(synthetic_df: pd.DataFrame):
    df_nan = synthetic_df.copy()
    df_nan.loc[0, "Amount"] = np.nan
    df_nan.loc[5, "V1"] = np.nan
    validator = DataValidator()
    result = validator.validate(df_nan, is_training=True)
    assert result.is_valid is False
    assert any(err.category == "NULL_VALUES" for err in result.errors)


def test_infinite_values_fail(synthetic_df: pd.DataFrame):
    df_inf = synthetic_df.copy()
    df_inf.loc[2, "V5"] = np.inf
    validator = DataValidator()
    result = validator.validate(df_inf, is_training=True)
    assert result.is_valid is False
    assert any(err.category == "INFINITE_VALUES" for err in result.errors)


def test_negative_amount_fails(synthetic_df: pd.DataFrame):
    df_neg = synthetic_df.copy()
    df_neg.loc[10, "Amount"] = -25.50
    validator = DataValidator()
    result = validator.validate(df_neg, is_training=True)
    assert result.is_valid is False
    assert any(err.category == "INVALID_AMOUNT" for err in result.errors)


def test_invalid_target_labels_fail(synthetic_df: pd.DataFrame):
    df_bad_target = synthetic_df.copy()
    df_bad_target.loc[0, "Class"] = 2  # Invalid class
    validator = DataValidator()
    result = validator.validate(df_bad_target, is_training=True)
    assert result.is_valid is False
    assert any(err.category == "INVALID_TARGET" for err in result.errors)


def test_duplicate_warning(synthetic_df: pd.DataFrame):
    df_dup = pd.concat([synthetic_df, synthetic_df.iloc[:5]], ignore_index=True)
    validator = DataValidator()
    result = validator.validate(df_dup, is_training=True)
    # Duplicate is flagged as warning, not hard error (per spec)
    assert result.is_valid is True
    assert any(warn.category == "DUPLICATES" for warn in result.warnings)


def test_inference_mode_allows_missing_target(synthetic_df: pd.DataFrame):
    df_inference = synthetic_df.drop(columns=["Class"])
    validator = DataValidator()
    result = validator.validate(df_inference, is_training=False)
    assert result.is_valid is True
    assert len(result.errors) == 0
