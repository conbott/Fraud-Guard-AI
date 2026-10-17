"""CLI workflow for candidate retraining, quality gating, automated promotion, and rollback."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validator import validate_file
from src.training.registry import ModelRegistry
from src.training.train import ModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="FraudGuard AI Automated Retraining and Model Promotion Engine."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic/synthetic_transactions.csv",
        help="Path to retraining dataset.",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="Scheduled automated retraining batch",
        help="Reason for retraining / promotion attempt.",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback the champion model to the immediately preceding champion backup.",
    )
    args = parser.parse_args()

    registry = ModelRegistry()

    if args.rollback:
        print("[INFO] Initiating rollback to previous champion...")
        try:
            registry.rollback_to_previous_champion()
            champ_meta = registry.get_champion_metadata()
            print("[SUCCESS] Champion rollback complete!")
            print(f"Current Active Version: {champ_meta.get('version')}")
            print(f"Restored Algorithm    : {champ_meta.get('algorithm')}")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Rollback failed: {e}")
            sys.exit(1)

    data_path = Path(args.data_path)
    print(f"\n[STEP 1/5] Validating candidate retraining dataset: {data_path}...")
    val_res = validate_file(data_path, is_training=True)
    if not val_res.is_valid:
        print(f"[ERROR] Dataset validation failed with {len(val_res.errors)} errors:")
        for err in val_res.errors:
            print(f" - {err.message}")
        sys.exit(1)
    print(" -> Dataset passed schema and quality validation.")

    print("\n[STEP 2/5] Training candidate model...")
    trainer = ModelTrainer()
    train_result = trainer.train_pipeline(
        data_path=data_path,
        save_candidate=True,
        is_champion=False,
    )
    candidate_ver = train_result["version"]
    candidate_metrics = train_result["test_metrics"]
    print(f" -> Candidate trained: {candidate_ver} ({train_result['algorithm']})")

    print("\n[STEP 3/5] Evaluating candidate against production quality gates...")
    champ_meta = registry.get_champion_metadata()
    passed, qg_summary = registry.evaluate_quality_gates(
        candidate_metrics=candidate_metrics,
        champion_metrics=champ_meta,
    )

    # Save quality gate report
    report_file = registry.reports_dir / f"promotion_gate_{candidate_ver}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(qg_summary, f, indent=2)

    print("-" * 60)
    for gate_name, gate_info in qg_summary["gates"].items():
        status = "PASSED" if gate_info["passed"] else "FAILED"
        print(f" - {gate_name:<25}: {status} (Actual: {gate_info.get('actual', gate_info.get('delta'))})")
    print("-" * 60)

    if not passed:
        print(f"\n[REJECTED] Candidate {candidate_ver} failed quality gates.")
        print("Production champion remains unchanged.")
        print(f"Audit report saved at: {report_file}")
        sys.exit(1)

    print(f"\n[STEP 4/5] Candidate {candidate_ver} passed all gates! Promoting to Champion...")
    registry.promote_candidate(
        candidate_version=candidate_ver,
        reason=args.reason,
    )

    print(f"[STEP 5/5] Champion successfully updated to version: {candidate_ver}!")
    print(f"Audit report: {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
