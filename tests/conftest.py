"""Shared pytest fixtures for FraudGuard AI tests."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.synthetic import generate_synthetic_transactions


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Session-scoped reproducible synthetic dataframe."""
    return generate_synthetic_transactions(n_samples=400, fraud_rate=0.05, random_seed=42)


@pytest.fixture
def valid_transaction_dict() -> dict:
    """A single valid transaction dictionary for inference."""
    data = {"Time": 100.0, "Amount": 49.99}
    for i in range(1, 29):
        data[f"V{i}"] = 0.05 * (i % 5 - 2)
    return data


@pytest.fixture
def sample_csv_file(tmp_path: Path, synthetic_df: pd.DataFrame) -> Path:
    """A temporary CSV file for testing validation and loading."""
    csv_path = tmp_path / "test_transactions.csv"
    synthetic_df.to_csv(csv_path, index=False)
    return csv_path
