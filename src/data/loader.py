"""Data loading and leakage-safe dataset splitting for FraudGuard AI."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ModelConfig, load_model_config
from src.data.validator import DataValidator


class DataLoader:
    """Handles dataset loading, validation, and stratified splitting."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.validator = DataValidator(self.config)

    def load_raw_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Load and validate raw data from CSV."""
        path = file_path or self.config.raw_data_path
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Please place creditcard.csv in data/raw/ "
                f"or generate synthetic data for testing."
            )
        df = pd.read_csv(path)
        val_result = self.validator.validate(df, is_training=True)
        if not val_result.is_valid:
            error_msgs = [e.message for e in val_result.errors]
            raise ValueError(f"Dataset validation failed: {error_msgs}")
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Perform leakage-safe splitting into Train, Validation, and Test sets.

        CRITICAL LEAKAGE SAFEGUARD:
        Splitting is performed strictly before any preprocessing, fitting, or scaling.

        Supports:
          - "stratified" (default): Stratifies across minority class labels.
          - "chronological": Sorts strictly by Time to simulate real-world production deployment.

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        target_col = self.config.target_column
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not present in dataframe.")

        split_strat = strategy or self.config.split.strategy
        test_size = self.config.split.test_size
        val_size = self.config.split.val_size

        if split_strat == "chronological":
            time_col = self.config.time_column
            if time_col not in df.columns:
                raise KeyError(f"Time column '{time_col}' required for chronological splitting.")
            df_sorted = df.sort_values(time_col).reset_index(drop=True)
            n_total = len(df_sorted)
            n_test = int(round(n_total * test_size))
            n_val = int(round(n_total * val_size))
            n_train = n_total - n_test - n_val

            train_df = df_sorted.iloc[:n_train]
            val_df = df_sorted.iloc[n_train:n_train + n_val]
            test_df = df_sorted.iloc[n_train + n_val:]

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_val = val_df.drop(columns=[target_col])
            y_val = val_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            return X_train, y_train, X_val, y_val, X_test, y_test

        # Default: Stratified Split
        X = df.drop(columns=[target_col])
        y = df[target_col]

        random_seed = self.config.random_seed
        stratify_test = y if self.config.split.stratify else None

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=stratify_test,
            random_state=random_seed,
            shuffle=self.config.split.shuffle,
        )

        val_size_adjusted = val_size / (1.0 - test_size)
        stratify_val = y_temp if self.config.split.stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_val,
            random_state=random_seed,
            shuffle=self.config.split.shuffle,
        )

        return X_train, y_train, X_val, y_val, X_test, y_test
