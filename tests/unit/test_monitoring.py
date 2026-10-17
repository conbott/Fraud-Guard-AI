"""Unit tests for drift monitoring, PSI calculation, and audit logging."""

from pathlib import Path

import numpy as np

from src.monitoring.drift import (
    calculate_psi,
    simulate_synthetic_drift,
)
from src.monitoring.metrics import AuditLogger


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    curr = rng.normal(0, 1, 1000)

    psi = calculate_psi(ref, curr)
    # Identical distributions should have very small PSI (< 0.10)
    assert psi < 0.10


def test_psi_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    curr = rng.normal(3.0, 1, 1000)  # Heavy shift

    psi = calculate_psi(ref, curr)
    # Substantial shift should yield PSI >= 0.25
    assert psi >= 0.25


def test_synthetic_drift_simulation(synthetic_df):
    drifted_df = simulate_synthetic_drift(
        synthetic_df, drift_type="amount_spike", severity=5.0
    )
    assert drifted_df["Amount"].mean() > synthetic_df["Amount"].mean() * 4.0


def test_audit_logger(tmp_path: Path):
    db_file = tmp_path / "test_audit.db"
    logger = AuditLogger(db_path=db_file)

    logger.log_inference(
        request_id="req-101",
        model_version="model_v1",
        risk_score=0.88,
        is_fraud=True,
        risk_level="HIGH",
        latency_ms=12.4,
    )
    logger.log_inference(
        request_id="req-102",
        model_version="model_v1",
        risk_score=0.04,
        is_fraud=False,
        risk_level="LOW",
        latency_ms=8.1,
    )

    metrics = logger.get_summary_metrics()
    assert metrics["total_logged_requests"] == 2
    assert metrics["predicted_fraud_count"] == 1
    assert metrics["predicted_fraud_rate"] == 0.5
    assert metrics["avg_latency_ms"] > 0.0
