"""Unit tests for ModelTrainer, ModelLoader, and FraudPredictor."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import ModelConfig
from src.inference.loader import ModelLoader
from src.inference.predictor import FraudPredictor
from src.training.train import ModelTrainer


def test_training_pipeline_and_inference(tmp_path: Path, sample_csv_file: Path):
    # Setup custom temporary configuration
    cfg = ModelConfig(
        raw_data_path=sample_csv_file,
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        champion_dir=tmp_path / "models" / "champion",
        candidates_dir=tmp_path / "models" / "candidates",
        reports_dir=tmp_path / "models" / "reports",
    )

    trainer = ModelTrainer(config=cfg)
    result = trainer.train_pipeline(data_path=sample_csv_file, is_champion=True)

    assert result["version"].startswith("model_v")
    assert result["optimal_threshold"] > 0.0
    assert Path(result["artifact_path"]).exists()
    assert Path(result["metadata_path"]).exists()
    assert "logistic_regression" in result["comparison"]
    assert "random_forest" in result["comparison"]

    # Test loading champion model
    loaded = ModelLoader.load_champion(champion_dir=cfg.champion_dir)
    assert loaded.version == result["version"]
    assert loaded.algorithm == result["algorithm"]
    assert len(loaded.feature_names) == 30  # Time + 28 PCA + Amount

    # Test predictor on single transaction
    predictor = FraudPredictor(loaded_model=loaded)
    df = pd.read_csv(sample_csv_file)
    single_txn = df.iloc[0].drop("Class").to_dict()

    pred_single = predictor.predict_single(single_txn)
    assert isinstance(pred_single.is_fraud, bool)
    assert 0.0 <= pred_single.risk_score <= 1.0
    assert pred_single.risk_level in ("LOW", "MEDIUM", "HIGH")
    assert pred_single.model_version == loaded.version

    # Test batch predictions
    batch_txns = df.iloc[:10].drop(columns=["Class"])
    pred_batch = predictor.predict_batch(batch_txns)
    assert len(pred_batch) == 10
    for p in pred_batch:
        assert isinstance(p.is_fraud, bool)
        assert 0.0 <= p.risk_score <= 1.0


def test_model_loader_rejects_missing_file(tmp_path: Path):
    non_existent = tmp_path / "missing_model.joblib"
    with pytest.raises(FileNotFoundError, match="not found"):
        ModelLoader.load_model_from_file(non_existent)
