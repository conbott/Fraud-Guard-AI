"""IEEE-CIS Fraud Detection - Advanced Experiment Pipeline.

This module provides a standalone, memory-guarded experimentation pipeline
for the Kaggle IEEE-CIS Fraud Detection dataset. It is strictly separated from
the MLG-ULB baseline and live inference service to protect the production
Streamlit application and prevent memory exhaustion.

Competition: https://www.kaggle.com/competitions/ieee-fraud-detection
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

# Avoid deprecated Windows 11 WMIC calls in joblib/loky
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IEEE_CIS_Experiment")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "models" / "reports"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ieee_cis"


def print_ieee_cis_specifications() -> None:
    """Print comprehensive technical specifications and instructions for IEEE-CIS."""
    specs = """
================================================================================
IEEE-CIS FRAUD DETECTION DATASET SPECIFICATIONS & FEASIBILITY ASSESSMENT
================================================================================

1. Dataset Overview & Size:
   - Competition: IEEE-CIS Fraud Detection (Vesta Corporation)
   - Compressed archive: ~500 MB (.zip)
   - Uncompressed size: ~1.43 GB (across train and test tables)
   - Total rows: 590,540 (train_transaction) + 144,233 (train_identity)
   - Features: 434 total columns (TransactionAmt, ProductCD, card1-card6,
     addr1-addr2, dist1-dist2, P/R_emaildomain, C1-C14, D1-D15, M1-M9, V1-V339,
     DeviceType, DeviceInfo, id_01-id_38)
   - Class imbalance: ~3.5% fraud prevalence (20,663 frauds out of 590,540 transactions)

2. Label Availability & Evaluation Philosophy:
   - Only 'train_transaction.csv' contains ground-truth labels ('isFraud').
   - 'test_transaction.csv' (506,691 rows) is completely UNLABELED (competition private/public leaderboard).
   - Therefore, rigorous offline evaluation REQUIRES an internal chronological hold-out split
     on the labeled training data using 'TransactionDT' (timestep in seconds).
   - Random/stratified splitting on IEEE-CIS introduces severe data leakage because
     identity features, card numbers, and IP addresses cluster chronologically.

3. Memory Constraints & Windows-Native Execution:
   - Naive loading with standard pandas float64/object types requires 8-12 GB RAM during
     merges and one-hot encoding, posing Out-Of-Memory (OOM) risks on 16 GB machines.
   - Mitigation: Downcast float64 to float32, int64 to int32, encode high-cardinality
     categoricals using frequency or ordinal encoding, and select key feature subsets.

4. Manual Download Instructions:
   Step 1: Visit https://www.kaggle.com/competitions/ieee-fraud-detection/data
   Step 2: Sign in to Kaggle and accept the competition rules.
   Step 3: Download 'train_transaction.csv' and 'train_identity.csv'.
   Step 4: Create directory: data/raw/ieee_cis/
   Step 5: Place 'train_transaction.csv' and 'train_identity.csv' in data/raw/ieee_cis/

5. Execution Options:
   - Run on real data: python scripts/run_ieee_cis_experiment.py
   - Run mock verification: python scripts/run_ieee_cis_experiment.py --mock
================================================================================
"""
    print(specs)


def generate_mock_ieee_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """Generate a realistic mock IEEE-CIS dataset for pipeline verification.

    Args:
        n_samples: Number of synthetic transactions to generate.

    Returns:
        DataFrame mirroring key IEEE-CIS schema and temporal distribution.
    """
    logger.info("Generating %d synthetic IEEE-CIS records for mock verification...", n_samples)
    rng = np.random.default_rng(42)

    # Simulate chronological time (TransactionDT in seconds)
    transaction_dt = np.sort(rng.integers(100_000, 10_000_000, size=n_samples))
    transaction_amt = np.round(rng.exponential(scale=120.0, size=n_samples), 2)
    product_cd = rng.choice(["W", "C", "R", "H", "S"], size=n_samples, p=[0.7, 0.15, 0.05, 0.05, 0.05])
    card1 = rng.integers(1000, 20000, size=n_samples)
    card4 = rng.choice(["visa", "mastercard", "discover", "american express"], size=n_samples)

    # 3.5% fraud rate
    is_fraud = rng.choice([0, 1], size=n_samples, p=[0.965, 0.035])

    data: Dict[str, Any] = {
        "TransactionID": np.arange(3000000, 3000000 + n_samples),
        "isFraud": is_fraud,
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": product_cd,
        "card1": card1,
        "card4": card4,
    }

    # Add numeric columns (C1-C5, D1-D3, V1-V20)
    for c_idx in range(1, 6):
        data[f"C{c_idx}"] = rng.poisson(lam=1.5, size=n_samples)
    for d_idx in range(1, 4):
        data[f"D{d_idx}"] = rng.exponential(scale=50.0, size=n_samples)
    for v_idx in range(1, 21):
        data[f"V{v_idx}"] = np.round(rng.normal(loc=0.0, scale=1.0, size=n_samples), 3)

    df = pd.DataFrame(data)
    # Inject signal into fraud cases for demonstration
    fraud_mask = df["isFraud"] == 1
    df.loc[fraud_mask, "TransactionAmt"] *= 2.5
    df.loc[fraud_mask, "C1"] += 4
    return df


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 to float32 and int64 to int32 to conserve RAM.

    Args:
        df: Input DataFrame.

    Returns:
        Memory-optimized DataFrame.
    """
    initial_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = df[col].astype("float32")
        elif col_type == "int64" and col != "TransactionID":
            df[col] = df[col].astype("int32")
    final_mem = df.memory_usage().sum() / 1024**2
    logger.info("Memory reduced from %.2f MB to %.2f MB (%.1f%% reduction)",
                initial_mem, final_mem, (1 - final_mem / initial_mem) * 100)
    return df


def load_and_preprocess_ieee(raw_dir: Path, mock: bool = False, max_rows: int = 100_000) -> pd.DataFrame:
    """Load and preprocess IEEE-CIS dataset with memory safeguards.

    Args:
        raw_dir: Directory containing raw IEEE-CIS CSV files.
        mock: If True, generate mock data rather than loading raw files.
        max_rows: Maximum records to load to prevent memory exhaustion.

    Returns:
        Prepared DataFrame ready for feature engineering and splitting.
    """
    if mock:
        df = generate_mock_ieee_dataset(n_samples=5000)
    else:
        trans_file = raw_dir / "train_transaction.csv"
        ident_file = raw_dir / "train_identity.csv"

        if not trans_file.exists():
            raise FileNotFoundError(
                f"Missing required file: {trans_file}\n"
                "Please download the dataset from Kaggle or run with '--mock'."
            )

        logger.info("Loading transaction table (up to %d rows)...", max_rows)
        df_trans = pd.read_csv(trans_file, nrows=max_rows)
        df_trans = downcast_dtypes(df_trans)

        if ident_file.exists():
            logger.info("Loading and joining identity table...")
            df_ident = pd.read_csv(ident_file, nrows=max_rows)
            df_ident = downcast_dtypes(df_ident)
            df = df_trans.merge(df_ident, on="TransactionID", how="left")
        else:
            logger.info("Identity table not found; proceeding with transaction features only.")
            df = df_trans

    # Convert object and string columns to categorical codes
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category").cat.codes

    # Fill NaN values with medians for numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def run_experiment(df: pd.DataFrame, is_mock: bool = True) -> Dict[str, Any]:
    """Execute chronological split, threshold tuning, and model evaluation.

    Args:
        df: Prepared DataFrame with 'TransactionDT' and 'isFraud'.
        is_mock: Whether the dataset was generated from mock fixture.

    Returns:
        Dictionary of experiment results, metrics, and parameters.
    """
    logger.info("Sorting transactions chronologically by 'TransactionDT'...")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    target_col = "isFraud"
    feature_cols = [c for c in df.columns if c not in ["isFraud", "TransactionID", "TransactionDT"]]

    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    x_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy()
    x_val, y_val = val_df[feature_cols], val_df[target_col].to_numpy()
    x_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy()

    train_frauds = int(y_train.sum())
    val_frauds = int(y_val.sum())
    test_frauds = int(y_test.sum())
    test_legit = int((y_test == 0).sum())

    logger.info("Chronological Split Breakdown:")
    logger.info("  Train: %d rows (Frauds: %d, Rate: %.3f%%)",
                len(train_df), train_frauds, float(y_train.mean() * 100))
    logger.info("  Val:   %d rows (Frauds: %d, Rate: %.3f%%)",
                len(val_df), val_frauds, float(y_val.mean() * 100))
    logger.info("  Test:  %d rows (Frauds: %d, Rate: %.3f%%)",
                len(test_df), test_frauds, float(y_test.mean() * 100))

    results: Dict[str, Any] = {
        "dataset": "IEEE-CIS Fraud Detection (Synthetic Mock Fixture)" if is_mock else "IEEE-CIS Fraud Detection",
        "provenance": "synthetic_mock_fixture" if is_mock else "real_kaggle_dataset",
        "is_synthetic": is_mock,
        "sample_size": n_total,
        "feature_count": len(feature_cols),
        "split_method": "Chronological (TransactionDT)",
        "split_counts": {
            "train": {"total": len(train_df), "frauds": train_frauds, "prevalence": round(float(y_train.mean()), 6)},
            "val": {"total": len(val_df), "frauds": val_frauds, "prevalence": round(float(y_val.mean()), 6)},
            "test": {
                "total": len(test_df),
                "frauds": test_frauds,
                "legitimate": test_legit,
                "prevalence": round(float(y_test.mean()), 6),
            },
        },
        "verification_note": (
            "Evaluated on a synthetic mock fixture generated with IEEE-CIS schema and chronological "
            "timesteps. The official 1.43 GB Kaggle IEEE-CIS dataset was not downloaded locally."
            if is_mock
            else "Evaluated on genuine Kaggle IEEE-CIS dataset."
        ),
        "models": {},
    }

    # 1. Dummy Classifier Baseline
    logger.info("Evaluating Dummy Classifier Baseline...")
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(x_train, y_train)
    dummy_probs = dummy.predict_proba(x_test)[:, 1]
    dummy_preds = (dummy_probs >= 0.5).astype(int)

    results["models"]["dummy_baseline"] = {
        "pr_auc": float(average_precision_score(y_test, dummy_probs)),
        "roc_auc": float(roc_auc_score(y_test, dummy_probs)),
        "f1": float(f1_score(y_test, dummy_preds, zero_division=0)),
        "recall": float(recall_score(y_test, dummy_preds, zero_division=0)),
        "precision": float(precision_score(y_test, dummy_preds, zero_division=0)),
    }

    # 2. Scalable Tree Model (HistGradientBoosting / RandomForest)
    logger.info("Training HistGradientBoostingClassifier (Memory-efficient LightGBM-style)...")
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)

    # Threshold tuning on Validation Partition
    val_probs = model.predict_proba(x_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = np.where((precisions + recalls) > 0, 2 * (precisions * recalls) / (precisions + recalls), 0)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50

    logger.info("Optimal threshold tuned on Validation set: %.4f (Val F1: %.4f)",
                optimal_threshold, f1_scores[best_idx])

    # Final Evaluation on Held-Out Test Partition
    test_probs = model.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= optimal_threshold).astype(int)

    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    results["models"]["hist_gradient_boosting"] = {
        "optimal_threshold": optimal_threshold,
        "pr_auc": float(average_precision_score(y_test, test_probs)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "false_positive_rate": fpr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    return results


def main() -> None:
    """CLI entrypoint for IEEE-CIS experimentation."""
    parser = argparse.ArgumentParser(description="IEEE-CIS Fraud Detection Experiment")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic IEEE-CIS schema for verification")
    parser.add_argument("--max-rows", type=int, default=100_000, help="Maximum rows to load from raw CSV")
    parser.add_argument("--specs-only", action="store_true", help="Print dataset specifications and exit")
    args = parser.parse_args()

    print_ieee_cis_specifications()
    if args.specs_only:
        return

    trans_file = RAW_DATA_DIR / "train_transaction.csv"
    if not trans_file.exists() and not args.mock:
        logger.warning("Local raw IEEE-CIS files not found in '%s'.", RAW_DATA_DIR)
        logger.info("To test the pipeline logic immediately, re-run with: python scripts/run_ieee_cis_experiment.py --mock")
        return

    df = load_and_preprocess_ieee(raw_dir=RAW_DATA_DIR, mock=args.mock, max_rows=args.max_rows)
    results = run_experiment(df, is_mock=args.mock)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "ieee_cis_evaluation.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation report saved to: %s", report_file)
    logger.info("Experiment summary:")
    for model_name, metrics in results["models"].items():
        logger.info("  [%s] PR-AUC: %.4f | ROC-AUC: %.4f | F1: %.4f",
                    model_name, metrics["pr_auc"], metrics["roc_auc"], metrics["f1"])


if __name__ == "__main__":
    main()
