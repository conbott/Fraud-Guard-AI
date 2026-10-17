"""CLI script to run model training, threshold tuning, and artifact generation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import ModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and save FraudGuard AI models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training CSV. Defaults to raw Kaggle dataset if available, otherwise synthetic.",
    )
    parser.add_argument(
        "--champion",
        action="store_true",
        help="Directly export and promote this trained model as the active Champion.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["stratified", "chronological"],
        default="stratified",
        help="Data partitioning strategy: stratified (default) or chronological (ordered by Time).",
    )
    args = parser.parse_args()

    trainer = ModelTrainer()
    data_path = Path(args.data_path) if args.data_path else None

    print("\n[INFO] Starting FraudGuard AI Training Pipeline...")
    res = trainer.train_pipeline(
        data_path=data_path,
        save_candidate=not args.champion,
        is_champion=args.champion,
        split_strategy=args.split_strategy,
    )

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Selected Champion Model : {res['algorithm']}")
    print(f"Model Version Tag       : {res['version']}")
    print(f"Tuned Decision Threshold: {res['optimal_threshold']:.4f}")
    print(f"Artifact Saved At       : {res['artifact_path']}")
    print(f"Duration                : {res['duration']}s")
    print("-" * 60)
    print("MODEL COMPARISON (VALIDATION SET):")
    for mod_name, metrics in res["comparison"].items():
        print(
            f" - {mod_name:<20}: PR-AUC={metrics['val_pr_auc']:.4f} | "
            f"F1={metrics['val_f1_tuned']:.4f} | "
            f"Optimal Threshold={metrics['optimal_threshold']:.2f}"
        )
    print("-" * 60)
    print("TEST EVALUATION (HELD-OUT PARTITION):")
    t_m = res["test_metrics"]
    print(f" - Test PR-AUC      : {t_m['pr_auc']:.4f}")
    print(f" - Test ROC-AUC     : {t_m['roc_auc']:.4f}")
    print(f" - Test Precision   : {t_m['precision']:.4f}")
    print(f" - Test Recall      : {t_m['recall']:.4f}")
    print(f" - Test F1-Score    : {t_m['f1_score']:.4f}")
    print(f" - False Pos. Rate  : {t_m['false_positive_rate']:.6f}")
    print(f" - Confusion Matrix : TP={t_m['true_positives']} | FP={t_m['false_positives']} | "
          f"TN={t_m['true_negatives']} | FN={t_m['false_negatives']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
