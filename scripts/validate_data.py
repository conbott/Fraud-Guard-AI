"""CLI script to validate data files against the FraudGuard schema and quality criteria."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validator import validate_file


def main():
    parser = argparse.ArgumentParser(
        description="Validate transaction CSV schema and quality."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to CSV dataset to validate.",
    )
    parser.add_argument(
        "--is-inference",
        action="store_true",
        help="Validate for inference (Class target column is optional).",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output raw machine-readable JSON report.",
    )
    args = parser.parse_args()

    file_path = Path(args.data_path)
    result = validate_file(file_path, is_training=not args.is_inference)

    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.is_valid else 1)

    print("=" * 60)
    print(f"FRAUDGUARD AI - DATA VALIDATION REPORT: {file_path.name}")
    print("=" * 60)
    print(f"Status        : {'PASSED (VALID)' if result.is_valid else 'FAILED (INVALID)'}")
    print(f"Row Count     : {result.row_count:,}")
    print(f"Column Count  : {result.column_count}")
    print(f"Errors Found  : {len(result.errors)}")
    print(f"Warnings Found: {len(result.warnings)}")

    if "class_distribution" in result.summary_stats:
        cd = result.summary_stats["class_distribution"]
        print(f"Fraud Count   : {cd['fraud_count']:,} ({cd['fraud_rate']*100:.3f}%)")
        print(f"Normal Count  : {cd['normal_count']:,}")

    if result.errors:
        print("\n[ERRORS]")
        for err in result.errors:
            print(f" - [{err.category}] {err.message}")

    if result.warnings:
        print("\n[WARNINGS]")
        for warn in result.warnings:
            print(f" - [{warn.category}] {warn.message}")

    print("=" * 60)
    sys.exit(0 if result.is_valid else 1)


if __name__ == "__main__":
    main()
