"""CLI script to evaluate an existing model artifact against a test dataset."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.loader import ModelLoader
from src.training.evaluate import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a FraudGuard model artifact on a dataset."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/champion/model.joblib",
        help="Path to model artifact (.joblib).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation dataset with target Class column.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom threshold override. If not set, uses model's tuned threshold.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_path = Path(args.data_path)

    print(f"[INFO] Loading model from {model_path}...")
    loaded = ModelLoader.load_model_from_file(model_path)
    threshold = args.threshold if args.threshold is not None else loaded.optimal_threshold

    print(f"[INFO] Loading evaluation data from {data_path}...")
    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        print("[ERROR] Dataset must contain 'Class' column for ground-truth evaluation.")
        sys.exit(1)

    y_true = df["Class"].values
    X_proc = loaded.preprocessor.transform(df)

    if hasattr(loaded.model, "predict_proba"):
        y_prob = loaded.model.predict_proba(X_proc)[:, 1]
    else:
        y_prob = loaded.model.predict(X_proc).astype(float)

    metrics = evaluate_predictions(
        y_true, y_prob, threshold=threshold, dataset_type="standalone_evaluation"
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {loaded.algorithm} ({loaded.version})")
    print("=" * 60)
    print(f"Decision Threshold: {metrics.threshold}")
    print(f"Total Transactions: {metrics.total_samples:,}")
    print(f"True Fraud Count  : {metrics.fraud_samples:,} ({metrics.prevalence*100:.3f}%)")
    print("-" * 60)
    print(f"PR-AUC (Avg Prec) : {metrics.pr_auc:.4f}")
    print(f"ROC-AUC           : {metrics.roc_auc:.4f}")
    print(f"Precision         : {metrics.precision:.4f}")
    print(f"Recall            : {metrics.recall:.4f}")
    print(f"F1-Score          : {metrics.f1_score:.4f}")
    print(f"False Pos. Rate   : {metrics.false_positive_rate:.6f}")
    print(f"False Neg. Rate   : {metrics.false_negative_rate:.6f}")
    print(f"Confusion Matrix  : TP={metrics.true_positives} | FP={metrics.false_positives} | "
          f"TN={metrics.true_negatives} | FN={metrics.false_negatives}")
    print("=" * 60)


if __name__ == "__main__":
    main()
