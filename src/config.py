"""Configuration loader and management for FraudGuard AI."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PreprocessingConfig:
    scaler: str = "robust"
    log_transform_amount: bool = True
    drop_time: bool = False


@dataclass
class SplitConfig:
    test_size: float = 0.20
    val_size: float = 0.20
    stratify: bool = True
    shuffle: bool = True


@dataclass
class QualityGatesConfig:
    min_pr_auc: float = 0.70
    min_recall: float = 0.75
    min_precision: float = 0.60
    max_fpr: float = 0.02


@dataclass
class ModelConfig:
    project_name: str = "FraudGuard AI"
    version: str = "0.1.0"
    random_seed: int = 42
    target_column: str = "Class"
    amount_column: str = "Amount"
    time_column: str = "Time"
    pca_features: List[str] = field(
        default_factory=lambda: [f"V{i}" for i in range(1, 29)]
    )
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    synthetic_path: Path = (
        PROJECT_ROOT / "data" / "synthetic" / "synthetic_transactions.csv"
    )
    fixture_path: Path = (
        PROJECT_ROOT / "data" / "synthetic" / "test_fixture.csv"
    )
    models_dir: Path = PROJECT_ROOT / "models"
    champion_dir: Path = PROJECT_ROOT / "models" / "champion"
    candidates_dir: Path = PROJECT_ROOT / "models" / "candidates"
    reports_dir: Path = PROJECT_ROOT / "models" / "reports"
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality_gates: QualityGatesConfig = field(default_factory=QualityGatesConfig)
    threshold_metric: str = "f1"

    @property
    def feature_columns(self) -> List[str]:
        cols = []
        if not self.preprocessing.drop_time:
            cols.append(self.time_column)
        cols.extend(self.pca_features)
        cols.append(self.amount_column)
        return cols

    @property
    def all_expected_columns(self) -> List[str]:
        return [self.time_column] + self.pca_features + [self.amount_column, self.target_column]


def load_model_config(config_path: Optional[Path] = None) -> ModelConfig:
    """Load model configuration from YAML or fallback to robust defaults."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "model_config.yaml"

    if not config_path.exists():
        return ModelConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    data_cfg = raw_cfg.get("data", {})
    split_cfg = raw_cfg.get("split", {})
    prep_cfg = raw_cfg.get("preprocessing", {})
    qg_cfg = raw_cfg.get("quality_gates", {})
    art_cfg = raw_cfg.get("artifacts", {})

    return ModelConfig(
        project_name=raw_cfg.get("project", {}).get("name", "FraudGuard AI"),
        version=raw_cfg.get("project", {}).get("version", "0.1.0"),
        random_seed=raw_cfg.get("project", {}).get("random_seed", 42),
        target_column=data_cfg.get("target_column", "Class"),
        amount_column=data_cfg.get("amount_column", "Amount"),
        time_column=data_cfg.get("time_column", "Time"),
        pca_features=data_cfg.get("pca_features", [f"V{i}" for i in range(1, 29)]),
        raw_data_path=PROJECT_ROOT / data_cfg.get("raw_path", "data/raw/creditcard.csv"),
        processed_dir=PROJECT_ROOT / data_cfg.get("processed_dir", "data/processed"),
        synthetic_path=PROJECT_ROOT / data_cfg.get("synthetic_path", "data/synthetic/synthetic_transactions.csv"),
        fixture_path=PROJECT_ROOT / data_cfg.get("fixture_path", "data/synthetic/test_fixture.csv"),
        models_dir=PROJECT_ROOT / art_cfg.get("models_dir", "models"),
        champion_dir=PROJECT_ROOT / art_cfg.get("champion_dir", "models/champion"),
        candidates_dir=PROJECT_ROOT / art_cfg.get("candidates_dir", "models/candidates"),
        reports_dir=PROJECT_ROOT / art_cfg.get("reports_dir", "models/reports"),
        split=SplitConfig(
            test_size=split_cfg.get("test_size", 0.20),
            val_size=split_cfg.get("val_size", 0.20),
            stratify=split_cfg.get("stratify", True),
            shuffle=split_cfg.get("shuffle", True),
        ),
        preprocessing=PreprocessingConfig(
            scaler=prep_cfg.get("scaler", "robust"),
            log_transform_amount=prep_cfg.get("log_transform_amount", True),
            drop_time=prep_cfg.get("drop_time", False),
        ),
        quality_gates=QualityGatesConfig(
            min_pr_auc=qg_cfg.get("min_pr_auc", 0.70),
            min_recall=qg_cfg.get("min_recall", 0.75),
            min_precision=qg_cfg.get("min_precision", 0.60),
            max_fpr=qg_cfg.get("max_fpr", 0.02),
        ),
        threshold_metric=raw_cfg.get("threshold_tuning", {}).get("metric", "f1"),
    )
