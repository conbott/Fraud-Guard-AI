"""Inference engine producing calibrated fraud risk estimates and categorizations."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.data.validator import DataValidator
from src.inference.loader import LoadedModel, ModelLoader


@dataclass
class PredictionOutput:
    is_fraud: bool
    risk_score: float
    threshold: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    model_version: str
    algorithm: str
    cautionary_notice: str = (
        "Statistical risk estimate only. This prediction is not a verified legal or financial "
        "determination of fraud and should be evaluated alongside secondary verification rules."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FraudPredictor:
    """Production predictor executing validated, schema-safe inferences."""

    def __init__(self, loaded_model: Optional[LoadedModel] = None):
        self.model_bundle = loaded_model or ModelLoader.load_champion()
        self.validator = DataValidator()

    def _determine_risk_level(self, score: float, threshold: float) -> str:
        if score >= threshold:
            return "HIGH"
        elif score >= (threshold * 0.5):
            return "MEDIUM"
        return "LOW"

    def predict_single(self, transaction: Dict[str, Any]) -> PredictionOutput:
        """Run inference on a single transaction payload."""
        df = pd.DataFrame([transaction])
        val_result = self.validator.validate(df, is_training=False, min_rows=1)
        if not val_result.is_valid:
            error_msgs = [e.message for e in val_result.errors]
            raise ValueError(f"Inference input validation error: {error_msgs}")

        # Preprocess features
        X_proc = self.model_bundle.preprocessor.transform(df)

        # Predict probability
        model = self.model_bundle.model
        if hasattr(model, "predict_proba"):
            risk_score = float(model.predict_proba(X_proc)[0, 1])
        else:
            risk_score = float(model.predict(X_proc)[0])

        threshold = self.model_bundle.optimal_threshold
        is_fraud = bool(risk_score >= threshold)
        risk_level = self._determine_risk_level(risk_score, threshold)

        return PredictionOutput(
            is_fraud=is_fraud,
            risk_score=round(risk_score, 4),
            threshold=round(threshold, 4),
            risk_level=risk_level,
            model_version=self.model_bundle.version,
            algorithm=self.model_bundle.algorithm,
        )

    def predict_batch(
        self, transactions: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> List[PredictionOutput]:
        """Run batch inference with validation checks."""
        if isinstance(transactions, list):
            df = pd.DataFrame(transactions)
        else:
            df = transactions.copy()

        val_result = self.validator.validate(df, is_training=False, min_rows=1)
        if not val_result.is_valid:
            error_msgs = [e.message for e in val_result.errors]
            raise ValueError(f"Batch inference validation error: {error_msgs}")

        X_proc = self.model_bundle.preprocessor.transform(df)
        model = self.model_bundle.model

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_proc)[:, 1]
        else:
            scores = model.predict(X_proc).astype(float)

        threshold = self.model_bundle.optimal_threshold
        results: List[PredictionOutput] = []

        for score in scores:
            s = float(score)
            is_fraud = bool(s >= threshold)
            risk_level = self._determine_risk_level(s, threshold)
            results.append(
                PredictionOutput(
                    is_fraud=is_fraud,
                    risk_score=round(s, 4),
                    threshold=round(threshold, 4),
                    risk_level=risk_level,
                    model_version=self.model_bundle.version,
                    algorithm=self.model_bundle.algorithm,
                )
            )

        return results
