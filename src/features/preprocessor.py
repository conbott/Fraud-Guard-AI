"""Leakage-safe feature preprocessing pipeline for FraudGuard AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import ModelConfig, load_model_config


class FraudFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessing transformer fitted ONLY on training data.

    Applies:
      1. Verification and alignment of exact feature column schema and order.
      2. Log1p transformation on transaction Amount (to tame extreme skewness).
      3. RobustScaler (or StandardScaler) fitted on training partition only.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.amount_col = self.config.amount_column
        self.time_col = self.config.time_column
        self.pca_cols = self.config.pca_features
        self.log_transform_amount = self.config.preprocessing.log_transform_amount
        self.drop_time = self.config.preprocessing.drop_time
        self.scaler_type = self.config.preprocessing.scaler

        # Scaler instance
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _determine_feature_columns(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        if not self.drop_time and self.time_col in df.columns:
            cols.append(self.time_col)
        for col in self.pca_cols:
            if col in df.columns:
                cols.append(col)
        if self.amount_col in df.columns:
            cols.append(self.amount_col)
        return cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudFeaturePreprocessor":
        """Fit preprocessor strictly on training data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_ = self._determine_feature_columns(X)

        # Work on a copy with the aligned columns
        X_work = X[self.feature_names_].copy()

        if self.log_transform_amount and self.amount_col in X_work.columns:
            # np.log1p avoids log(0) issues
            X_work[self.amount_col] = np.log1p(np.maximum(X_work[self.amount_col], 0.0))

        # Fit the scaler on the transformed feature matrix
        self.scaler.fit(X_work.values)
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """Transform features using the fitted parameters.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If required feature columns are missing.
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, list):
            X_df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_)

        missing_cols = [c for c in self.feature_names_ if c not in X_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing expected feature columns for preprocessing: {missing_cols}"
            )

        X_aligned = X_df[self.feature_names_].copy()

        if self.log_transform_amount and self.amount_col in X_aligned.columns:
            X_aligned[self.amount_col] = np.log1p(
                np.maximum(X_aligned[self.amount_col].astype(float), 0.0)
            )

        return self.scaler.transform(X_aligned.values)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: Path) -> None:
        """Save fitted preprocessor artifact to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "FraudFeaturePreprocessor":
        """Load preprocessor artifact from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
        return joblib.load(filepath)
