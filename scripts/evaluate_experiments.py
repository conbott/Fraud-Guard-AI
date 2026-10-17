"""Comprehensive reproducible evaluation script for FraudGuard AI on MLG-ULB dataset.

Evaluates:
  1. Stratified Split (60% Train, 20% Val, 20% Test)
  2. Chronological Split (Ordered by Time: First 60% Train, Next 20% Val, Last 20% Test)

Models:
  - DummyClassifier (Stratified Baseline)
  - Logistic Regression (Balanced, C=0.1, L-BFGS)
  - Random Forest (100 Trees, Max Depth=10, Balanced)

Outputs:
  - models/reports/ulb_stratified_evaluation.json
  - models/reports/ulb_chronological_evaluation.json
  - models/reports/ulb_benchmark_comparison.csv
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_model_config
from src.data.loader import DataLoader
from src.features.preprocessor import FraudFeaturePreprocessor
from src.training.evaluate import (
    compute_curve_points,
    evaluate_predictions,
    tune_threshold_on_validation,
)


def run_split_experiment(
    df: pd.DataFrame,
    strategy: str = "stratified",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete leakage-safe training and evaluation for a given split strategy."""
    config = load_model_config()
    loader = DataLoader(config)

    print(f"\n{'='*70}")
    print(f"RUNNING EVALUATION: {strategy.upper()} SPLIT (MLG-ULB Credit Card Dataset)")
    print(f"{'='*70}")

    # 1. Leakage-safe split
    X_train, y_train, X_val, y_val, X_test, y_test = loader.split_data(df, strategy=strategy)

    total_samples = len(df)
    train_frauds = int(y_train.sum())
    val_frauds = int(y_val.sum())
    test_frauds = int(y_test.sum())

    print(f"Dataset Size : {total_samples:,} transactions")
    print(f"Train Split  : {len(X_train):,} samples | {train_frauds} frauds ({train_frauds/len(X_train)*100:.4f}%)")
    print(f"Val Split    : {len(X_val):,} samples | {val_frauds} frauds ({val_frauds/len(X_val)*100:.4f}%)")
    print(f"Test Split   : {len(X_test):,} samples | {test_frauds} frauds ({test_frauds/len(X_test)*100:.4f}%)")
    print(f"{'-'*70}")

    # 2. Fit preprocessor strictly on training data
    preprocessor = FraudFeaturePreprocessor(config)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # 3. Candidate models
    models = {
        "dummy_baseline": DummyClassifier(strategy="stratified", random_state=random_seed),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.1,
            solver="lbfgs",
            random_state=random_seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_seed,
        ),
    }

    results: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_proc, y_train.values)
        fit_duration = round(time.time() - t0, 2)

        # Predict probabilities on Validation
        if hasattr(model, "predict_proba"):
            y_val_prob = model.predict_proba(X_val_proc)[:, 1]
            y_test_prob = model.predict_proba(X_test_proc)[:, 1]
        else:
            y_val_prob = model.predict(X_val_proc).astype(float)
            y_test_prob = model.predict(X_test_proc).astype(float)

        # Tune threshold strictly on validation partition
        best_threshold, thresh_summary = tune_threshold_on_validation(
            y_val.values, y_val_prob, metric="f1"
        )

        val_metrics = evaluate_predictions(
            y_val.values, y_val_prob, threshold=best_threshold, dataset_type="validation"
        )
        test_metrics = evaluate_predictions(
            y_test.values, y_test_prob, threshold=best_threshold, dataset_type="test"
        )

        curves = compute_curve_points(y_test.values, y_test_prob)

        results[name] = {
            "fit_duration_seconds": fit_duration,
            "optimal_threshold": best_threshold,
            "val_metrics": val_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
            "curve_points": curves,
        }

        table_rows.append({
            "split_strategy": strategy,
            "model": name,
            "fit_time_s": fit_duration,
            "threshold": best_threshold,
            "val_pr_auc": val_metrics.pr_auc,
            "val_roc_auc": val_metrics.roc_auc,
            "val_f1": val_metrics.f1_score,
            "test_pr_auc": test_metrics.pr_auc,
            "test_roc_auc": test_metrics.roc_auc,
            "test_precision": test_metrics.precision,
            "test_recall": test_metrics.recall,
            "test_f1": test_metrics.f1_score,
            "test_fpr": test_metrics.false_positive_rate,
            "true_positives": test_metrics.true_positives,
            "false_positives": test_metrics.false_positives,
            "true_negatives": test_metrics.true_negatives,
            "false_negatives": test_metrics.false_negatives,
        })

        print(
            f"[{name:<19}] Fit: {fit_duration:>5.2f}s | "
            f"Thresh: {best_threshold:.2f} | "
            f"Test PR-AUC: {test_metrics.pr_auc:.4f} | "
            f"ROC-AUC: {test_metrics.roc_auc:.4f} | "
            f"Recall: {test_metrics.recall:.4f} | "
            f"Precision: {test_metrics.precision:.4f} | "
            f"F1: {test_metrics.f1_score:.4f}"
        )
        print(
            f"  Confusion Matrix : TP={test_metrics.true_positives:<3} | "
            f"FP={test_metrics.false_positives:<4} | "
            f"TN={test_metrics.true_negatives:<6} | "
            f"FN={test_metrics.false_negatives:<3} | "
            f"FPR={test_metrics.false_positive_rate:.6f}"
        )

    experiment_report = {
        "dataset": "MLG-ULB Credit Card Fraud Detection",
        "split_strategy": strategy,
        "total_samples": total_samples,
        "total_frauds": int(df["Class"].sum()),
        "fraud_prevalence": round(float(df["Class"].mean()), 6),
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "models": results,
    }

    return experiment_report, table_rows


def main():
    config = load_model_config()
    raw_path = config.raw_data_path

    if not raw_path.exists():
        print(f"[ERROR] Real dataset not found at {raw_path}.")
        print("Please place creditcard.csv in data/raw/ before running this benchmark.")
        sys.exit(1)

    print(f"[INFO] Loading MLG-ULB dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    config.reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified evaluation
    strat_report, strat_rows = run_split_experiment(df, strategy="stratified")
    strat_file = config.reports_dir / "ulb_stratified_evaluation.json"
    with open(strat_file, "w", encoding="utf-8") as f:
        json.dump(strat_report, f, indent=2)
    print(f"\n[SAVED] Stratified report -> {strat_file}")

    # 2. Chronological evaluation
    chrono_report, chrono_rows = run_split_experiment(df, strategy="chronological")
    chrono_file = config.reports_dir / "ulb_chronological_evaluation.json"
    with open(chrono_file, "w", encoding="utf-8") as f:
        json.dump(chrono_report, f, indent=2)
    print(f"[SAVED] Chronological report -> {chrono_file}")

    # 3. Consolidated CSV summary table
    all_rows = strat_rows + chrono_rows
    summary_df = pd.DataFrame(all_rows)
    csv_file = config.reports_dir / "ulb_benchmark_comparison.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"[SAVED] Consolidated CSV comparison -> {csv_file}")

    print("\n" + "=" * 70)
    print("CONSOLIDATED BENCHMARK SUMMARY (HELD-OUT TEST SET):")
    print("=" * 70)
    cols_to_print = [
        "split_strategy",
        "model",
        "threshold",
        "test_pr_auc",
        "test_roc_auc",
        "test_recall",
        "test_precision",
        "test_f1",
        "false_positives",
    ]
    print(summary_df[cols_to_print].to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
