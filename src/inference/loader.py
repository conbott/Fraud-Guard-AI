"""Secure model loader and artifact verification for FraudGuard AI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.config import load_model_config
from src.features.preprocessor import FraudFeaturePreprocessor


@dataclass
class LoadedModel:
    model: Any
    preprocessor: FraudFeaturePreprocessor
    algorithm: str
    version: str
    optimal_threshold: float
    feature_names: List[str]
    metadata: Dict[str, Any]


class ModelLoader:
    """Safe model artifact loader validating integrity and schema compatibility."""

    REQUIRED_KEYS = {
        "model",
        "preprocessor",
        "algorithm",
        "version",
        "optimal_threshold",
        "feature_names",
    }

    @classmethod
    def load_model_from_file(cls, model_path: Path) -> LoadedModel:
        """Load and strictly validate a model artifact bundle from disk.

        CRITICAL SECURITY RULE:
        Joblib deserialization executes arbitrary code if given untrusted inputs.
        This loader MUST only be pointed at artifacts generated and managed directly
        by the FraudGuard training pipeline, never user-uploaded files.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                f"Please run the training pipeline to produce a champion model."
            )

        try:
            bundle = joblib.load(model_path)
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable model artifact at {model_path}: {e}")

        if not isinstance(bundle, dict):
            raise TypeError(
                f"Invalid artifact format. Expected dictionary bundle, got {type(bundle)}"
            )

        missing_keys = cls.REQUIRED_KEYS - set(bundle.keys())
        if missing_keys:
            raise KeyError(
                f"Model artifact bundle at {model_path} is missing required components: {missing_keys}"
            )

        # Verify preprocessor type and fitted state
        preprocessor = bundle["preprocessor"]
        if not hasattr(preprocessor, "transform") or not getattr(preprocessor, "is_fitted_", False):
            raise ValueError("Artifact preprocessor is not properly fitted or lacks transform().")

        metadata = {k: v for k, v in bundle.items() if k not in ("model", "preprocessor")}

        return LoadedModel(
            model=bundle["model"],
            preprocessor=bundle["preprocessor"],
            algorithm=bundle["algorithm"],
            version=bundle["version"],
            optimal_threshold=float(bundle["optimal_threshold"]),
            feature_names=list(bundle["feature_names"]),
            metadata=metadata,
        )

    @classmethod
    def load_champion(cls, champion_dir: Optional[Path] = None) -> LoadedModel:
        """Load the active production champion model."""
        config = load_model_config()
        target_dir = champion_dir or config.champion_dir
        model_file = target_dir / "model.joblib"
        try:
            return cls.load_model_from_file(model_file)
        except Exception as e:
            # If artifact is missing and raw data is available locally, auto-train champion
            if config.raw_data_path.exists():
                from src.training.train import ModelTrainer
                trainer = ModelTrainer(config)
                trainer.train_pipeline(is_champion=True)
                return cls.load_model_from_file(model_file)
            raise RuntimeError(
                f"Active champion model at {model_file} could not be loaded: {e}"
            ) from e
