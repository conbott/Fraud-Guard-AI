"""Unit tests for model registry quality gates, promotion, rejection, and rollback."""

import json
from pathlib import Path

from src.config import ModelConfig
from src.training.registry import ModelRegistry


def test_quality_gates_evaluation(tmp_path: Path):
    cfg = ModelConfig(
        models_dir=tmp_path / "models",
        champion_dir=tmp_path / "models" / "champion",
        candidates_dir=tmp_path / "models" / "candidates",
        reports_dir=tmp_path / "models" / "reports",
    )
    registry = ModelRegistry(cfg)

    # 1. High-performing candidate metrics
    good_metrics = {
        "pr_auc": 0.88,
        "recall": 0.82,
        "precision": 0.79,
        "false_positive_rate": 0.005,
    }
    passed, summary = registry.evaluate_quality_gates(good_metrics)
    assert passed is True
    assert summary["overall_passed"] is True

    # 2. Poor candidate metrics (fails recall and PR-AUC)
    bad_metrics = {
        "pr_auc": 0.45,  # Below 0.70 gate
        "recall": 0.50,  # Below 0.75 gate
        "precision": 0.70,
        "false_positive_rate": 0.04,  # Above 0.02 gate
    }
    passed_bad, summary_bad = registry.evaluate_quality_gates(bad_metrics)
    assert passed_bad is False
    assert summary_bad["gates"]["pr_auc_gate"]["passed"] is False
    assert summary_bad["gates"]["recall_gate"]["passed"] is False
    assert summary_bad["gates"]["fpr_gate"]["passed"] is False


def test_promotion_and_rollback_flow(tmp_path: Path):
    cfg = ModelConfig(
        models_dir=tmp_path / "models",
        champion_dir=tmp_path / "models" / "champion",
        candidates_dir=tmp_path / "models" / "candidates",
        reports_dir=tmp_path / "models" / "reports",
    )
    registry = ModelRegistry(cfg)

    # Setup initial champion v1
    v1_meta = {"version": "v1.0", "algorithm": "logistic_regression"}
    with open(cfg.champion_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(v1_meta, f)
    with open(cfg.champion_dir / "model.joblib", "wb") as f:
        f.write(b"model_v1_bytes")

    # Setup candidate v2
    cand_v2_dir = cfg.candidates_dir / "v2.0"
    cand_v2_dir.mkdir(parents=True, exist_ok=True)
    v2_meta = {"version": "v2.0", "algorithm": "random_forest"}
    with open(cand_v2_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(v2_meta, f)
    with open(cand_v2_dir / "model.joblib", "wb") as f:
        f.write(b"model_v2_bytes")

    # Promote candidate v2
    promoted = registry.promote_candidate("v2.0", reason="Performance benchmark beat v1.0")
    assert promoted is True

    champ = registry.get_champion_metadata()
    assert champ["version"] == "v2.0"
    with open(cfg.champion_dir / "model.joblib", "rb") as f:
        assert f.read() == b"model_v2_bytes"

    # Rollback to v1
    rolled_back = registry.rollback_to_previous_champion()
    assert rolled_back is True

    champ_restored = registry.get_champion_metadata()
    assert champ_restored["version"] == "v1.0"
    with open(cfg.champion_dir / "model.joblib", "rb") as f:
        assert f.read() == b"model_v1_bytes"
