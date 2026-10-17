"""Unit tests for imbalanced evaluation and threshold tuning."""

import numpy as np

from src.training.evaluate import (
    compute_curve_points,
    evaluate_predictions,
    tune_threshold_on_validation,
)


def test_evaluate_predictions_metrics():
    # Ground truth: 10 normal, 2 fraud
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    # Model probabilities
    y_prob = np.array([0.01, 0.05, 0.02, 0.10, 0.04, 0.02, 0.03, 0.05, 0.08, 0.12, 0.85, 0.92])

    metrics = evaluate_predictions(y_true, y_prob, threshold=0.50, dataset_type="validation")

    assert metrics.dataset_type == "validation"
    assert metrics.total_samples == 12
    assert metrics.fraud_samples == 2
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.true_negatives == 10
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0
    assert metrics.pr_auc > 0.95
    assert metrics.roc_auc == 1.0
    assert metrics.false_positive_rate == 0.0


def test_tune_threshold_selects_optimal():
    # Fraud cases with slightly lower probabilities around 0.35-0.45
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.15, 0.20, 0.38, 0.42])

    best_thresh, summary = tune_threshold_on_validation(
        y_val=y_true,
        y_prob_val=y_prob,
        metric="f1",
        min_threshold=0.10,
        max_threshold=0.90,
        step=0.05,
    )

    # With default 0.50 threshold, recall would be 0! Tuned threshold should be <= 0.35
    assert best_thresh <= 0.40
    assert summary["best_score"] > 0.0


def test_compute_curve_points():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    curves = compute_curve_points(y_true, y_prob)

    assert "pr_curve" in curves
    assert "roc_curve" in curves
    assert len(curves["pr_curve"]["precision"]) > 0
    assert len(curves["roc_curve"]["fpr"]) > 0
