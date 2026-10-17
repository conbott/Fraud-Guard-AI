"""CLI script to compute drift metrics between incoming transactions and baseline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.predictor import FraudPredictor
from src.monitoring.drift import DriftMonitor, simulate_synthetic_drift


def main():
    parser = argparse.ArgumentParser(
        description="Run feature and prediction drift monitoring for FraudGuard AI."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/synthetic/synthetic_transactions.csv",
        help="Path to current transaction batch CSV.",
    )
    parser.add_argument(
        "--simulate-drift",
        choices=["none", "amount_spike", "feature_shift", "mixed"],
        default="none",
        help="Inject controlled synthetic drift for testing monitoring alarms.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/reports/drift",
        help="Directory to save JSON drift report.",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"[ERROR] Transaction batch not found at: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)

    if args.simulate_drift != "none":
        print(f"[SIMULATION] Injecting '{args.simulate_drift}' drift into batch...")
        df = simulate_synthetic_drift(df, drift_type=args.simulate_drift, severity=3.5)

    # Compute risk scores using active champion predictor
    print("[INFO] Computing model predictions for score distribution drift...")
    predictor = FraudPredictor()
    predictions = predictor.predict_batch(df.drop(columns=["Class"], errors="ignore"))
    scores = np.array([p.risk_score for p in predictions], dtype=float)

    # Run Drift Analysis
    print("[INFO] Evaluating feature PSI and Kolmogorov-Smirnov statistics...")
    monitor = DriftMonitor()
    report = monitor.evaluate_drift(current_df=df, risk_scores=scores)

    # Save JSON report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"drift_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    print("\n" + "=" * 65)
    print("FRAUDGUARD AI - DRIFT MONITORING REPORT")
    print("=" * 65)
    print(f"Overall Drift Status : {report.overall_drift_status}")
    print(f"Reference Samples    : {report.reference_sample_count:,}")
    print(f"Current Samples      : {report.current_sample_count:,}")
    print(f"Drifted Features     : {report.drifted_features_count} / {report.total_monitored_features}")
    print("-" * 65)
    print(f"{'Feature':<10} | {'PSI':<8} | {'KS p-val':<10} | {'Status':<18}")
    print("-" * 65)
    for feat in report.feature_reports:
        print(f"{feat.feature_name:<10} | {feat.psi:<8.4f} | {feat.ks_p_value:<10.6f} | {feat.drift_status:<18}")
    print("-" * 65)
    if report.prediction_drift:
        pd_stat = report.prediction_drift
        print("PREDICTION DRIFT SUMMARY:")
        print(f" - Mean Risk Score   : {pd_stat['mean_score']}")
        print(f" - P95 Risk Score    : {pd_stat['p95_score']}")
        print(f" - High-Risk Pct     : {pd_stat['high_risk_percentage']}%")
    print("=" * 65)
    print(f"Report saved to: {report_file}\n")


if __name__ == "__main__":
    main()
