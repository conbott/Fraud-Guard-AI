"""CLI script to generate reproducible synthetic transaction datasets for testing."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import create_test_fixtures, generate_synthetic_transactions


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data for FraudGuard AI."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2500,
        help="Number of synthetic transaction records to generate.",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.015,
        help="Proportion of fraudulent transactions (e.g. 0.015 = 1.5%%).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/synthetic_transactions.csv",
        help="Target output CSV file path.",
    )
    parser.add_argument(
        "--create-fixtures",
        action="store_true",
        help="Generate standard test fixtures in data/synthetic/.",
    )
    args = parser.parse_args()

    if args.create_fixtures:
        fixture_p, synth_p = create_test_fixtures()
        print(f"[SUCCESS] Test fixture created: {fixture_p}")
        print(f"[SUCCESS] Synthetic dataset created: {synth_p}")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_transactions(
        n_samples=args.samples,
        fraud_rate=args.fraud_rate,
        random_seed=42,
    )
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Generated {len(df)} transactions ({args.fraud_rate*100:.2f}% fraud) -> {out_path}")


if __name__ == "__main__":
    main()
