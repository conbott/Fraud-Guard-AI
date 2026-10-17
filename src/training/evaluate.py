"""Model evaluation and decision threshold tuning for imbalanced fraud classification."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationMetrics:
    dataset_type: str  # "validation" or "test"
    threshold: float
    total_samples: int
    fraud_samples: int
    prevalence: float
    precision: float
    recall: float
    f1_score: float
    pr_auc: float
    roc_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float  # Stored for baseline context, marked as non-primary

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    dataset_type: str = "validation",
) -> EvaluationMetrics:
    """Compute rigorous imbalanced classification metrics at a given decision threshold.

    Why PR-AUC is preferred over ROC-AUC and Accuracy:
    In datasets with ~0.17% fraud prevalence, an algorithm that predicts 0 for all
    transactions achieves 99.83% accuracy and may have an ROC-AUC above 0.90 while
    detecting 0 frauds (Recall=0%). Precision-Recall AUC focuses specifically on
    the minority fraud class, measuring how cleanly true frauds are separated from
    false alarms without being inflated by the overwhelming number of true negatives.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    total_samples = len(y_true_arr)
    fraud_samples = int(np.sum(y_true_arr == 1))
    prevalence = float(fraud_samples / total_samples) if total_samples > 0 else 0.0

    # Binary prediction at specified decision threshold
    y_pred = (y_prob_arr >= threshold).astype(int)

    # Confusion matrix elements: tn, fp, fn, tp
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    # Area under curves (computed on continuous risk probabilities)
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        roc_auc = 0.5

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    accuracy = float((tp + tn) / total_samples) if total_samples > 0 else 0.0

    return EvaluationMetrics(
        dataset_type=dataset_type,
        threshold=round(float(threshold), 4),
        total_samples=total_samples,
        fraud_samples=fraud_samples,
        prevalence=round(prevalence, 6),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        false_positive_rate=round(fpr, 6),
        false_negative_rate=round(fnr, 6),
        accuracy=round(accuracy, 6),
    )


def tune_threshold_on_validation(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, Dict[str, Any]]:
    """Tune the decision threshold strictly on the validation partition.

    CRITICAL SAFEGUARD:
    The threshold is NEVER tuned on the test set. Tuning on test introduces
    optimistic bias and data leakage into final reported figures.
    """
    y_val_arr = np.asarray(y_val, dtype=int)
    y_prob_arr = np.asarray(y_prob_val, dtype=float)

    thresholds = np.arange(min_threshold, max_threshold + step, step)
    best_threshold = 0.50
    best_score = -1.0
    history: List[Dict[str, float]] = []

    for thresh in thresholds:
        metrics = evaluate_predictions(
            y_val_arr, y_prob_arr, threshold=thresh, dataset_type="validation"
        )
        if metric == "f1":
            score = metrics.f1_score
        elif metric == "recall":
            score = metrics.recall
        elif metric == "precision":
            score = metrics.precision
        else:
            score = metrics.f1_score

        history.append({
            "threshold": round(float(thresh), 3),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1_score,
            "fpr": metrics.false_positive_rate,
        })

        if score > best_score:
            best_score = score
            best_threshold = float(thresh)

    summary = {
        "metric_optimized": metric,
        "best_threshold": round(best_threshold, 4),
        "best_score": round(best_score, 4),
        "search_history": history,
    }
    return best_threshold, summary


def compute_curve_points(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Any]:
    """Compute PR and ROC curve coordinates for dashboard rendering."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    # Precision-Recall Curve
    p, r, pr_thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_prob_arr)

    # Subsample if large to keep serializable / dashboard fast
    max_points = 200
    if len(p) > max_points:
        idx_pr = np.linspace(0, len(p) - 1, max_points, dtype=int)
        p, r = p[idx_pr], r[idx_pr]
    if len(fpr) > max_points:
        idx_roc = np.linspace(0, len(fpr) - 1, max_points, dtype=int)
        fpr, tpr = fpr[idx_roc], tpr[idx_roc]

    return {
        "pr_curve": {"precision": p.tolist(), "recall": r.tolist()},
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
