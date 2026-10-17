"""Data schema and quality validation module for FraudGuard AI."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import ModelConfig, load_model_config


@dataclass
class ValidationIssue:
    severity: str  # "ERROR" or "WARNING"
    category: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    row_count: int
    column_count: int
    issues: List[ValidationIssue] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(i) for i in self.issues],
            "summary_stats": self.summary_stats,
        }


class DataValidator:
    """Validates transaction data against the FraudGuard schema and quality rules."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.expected_features = self.config.pca_features
        self.amount_col = self.config.amount_column
        self.time_col = self.config.time_column
        self.target_col = self.config.target_column

    def validate(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
        min_rows: int = 1,
    ) -> ValidationResult:
        """Execute complete schema and data quality validation.

        Args:
            df: DataFrame to validate.
            is_training: If True, target column (Class) is required.
                         If False (inference), target column is optional.
            min_rows: Minimum required number of rows.
        """
        issues: List[ValidationIssue] = []
        summary: Dict[str, Any] = {}

        # 1. Check basic dimensions
        row_count, col_count = df.shape
        summary["row_count"] = row_count
        summary["column_count"] = col_count

        if row_count < min_rows:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    category="ROW_COUNT",
                    message=f"Row count {row_count} is less than required minimum {min_rows}.",
                    details={"row_count": row_count, "min_rows": min_rows},
                )
            )

        # 2. Check required and unexpected columns
        required_cols = [self.time_col] + self.expected_features + [self.amount_col]
        if is_training:
            required_cols.append(self.target_col)

        existing_cols = set(df.columns)
        missing_cols = [c for c in required_cols if c not in existing_cols]
        unexpected_cols = [c for c in df.columns if c not in set(required_cols)]

        if missing_cols:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    category="MISSING_COLUMNS",
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing_columns": missing_cols},
                )
            )

        if unexpected_cols:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    category="UNEXPECTED_COLUMNS",
                    message=f"Found unexpected columns: {unexpected_cols}",
                    details={"unexpected_columns": unexpected_cols},
                )
            )

        # 3. Check data types and numerical consistency
        for col in df.columns:
            if col in required_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            category="INVALID_TYPE",
                            message=f"Column '{col}' must be numeric, got dtype {df[col].dtype}",
                            details={"column": col, "dtype": str(df[col].dtype)},
                        )
                    )

        # 4. Check Missing and Infinite Values
        nan_counts = df.isna().sum().to_dict()
        cols_with_nan = {k: int(v) for k, v in nan_counts.items() if v > 0}
        if cols_with_nan:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    category="NULL_VALUES",
                    message=f"Columns contain null/missing values: {cols_with_nan}",
                    details={"null_counts": cols_with_nan},
                )
            )

        # Check for inf / -inf in numeric columns
        inf_counts: Dict[str, int] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            num_infs = int(np.isinf(df[col]).sum())
            if num_infs > 0:
                inf_counts[col] = num_infs

        if inf_counts:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    category="INFINITE_VALUES",
                    message=f"Columns contain infinite values: {inf_counts}",
                    details={"infinite_counts": inf_counts},
                )
            )

        # 5. Check Amount & Time value ranges
        if self.amount_col in df.columns and pd.api.types.is_numeric_dtype(df[self.amount_col]):
            negative_amounts = int((df[self.amount_col] < 0).sum())
            if negative_amounts > 0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        category="INVALID_AMOUNT",
                        message=f"Found {negative_amounts} records with negative Amount.",
                        details={"negative_amounts_count": negative_amounts},
                    )
                )
            summary["amount_min"] = float(df[self.amount_col].min()) if row_count > 0 else 0.0
            summary["amount_max"] = float(df[self.amount_col].max()) if row_count > 0 else 0.0
            summary["amount_median"] = float(df[self.amount_col].median()) if row_count > 0 else 0.0

        if self.time_col in df.columns and pd.api.types.is_numeric_dtype(df[self.time_col]):
            negative_times = int((df[self.time_col] < 0).sum())
            if negative_times > 0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        category="INVALID_TIME",
                        message=f"Found {negative_times} records with negative Time.",
                        details={"negative_times_count": negative_times},
                    )
                )

        # 6. Target label validation for training datasets
        if is_training and self.target_col in df.columns:
            unique_classes = set(df[self.target_col].dropna().unique())
            valid_classes = {0, 1}
            if not unique_classes.issubset(valid_classes):
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        category="INVALID_TARGET",
                        message=f"Target column '{self.target_col}' must contain only binary {{0, 1}}, got {unique_classes}",
                        details={"unique_classes": list(unique_classes)},
                    )
                )
            else:
                fraud_count = int((df[self.target_col] == 1).sum())
                normal_count = int((df[self.target_col] == 0).sum())
                fraud_rate = float(fraud_count / row_count) if row_count > 0 else 0.0
                summary["class_distribution"] = {
                    "normal_count": normal_count,
                    "fraud_count": fraud_count,
                    "fraud_rate": fraud_rate,
                }
                if fraud_count == 0:
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            category="NO_FRAUD_EXAMPLES",
                            message="Target column contains zero fraud cases (Class=1). Cannot train a classifier.",
                            details={"fraud_count": 0},
                        )
                    )

        # 7. Check for Duplicate Records
        # Note: In financial transaction streams, duplicate rows can happen legitimately
        # (e.g. repeated small subscription attempts) or as replay anomalies. We warn rather
        # than silently discarding.
        duplicate_count = int(df.duplicated().sum())
        summary["duplicate_count"] = duplicate_count
        if duplicate_count > 0:
            dup_pct = round(duplicate_count / max(row_count, 1) * 100, 2)
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    category="DUPLICATES",
                    message=f"Detected {duplicate_count} ({dup_pct}%) duplicate rows. "
                    "In financial transactions, evaluate whether these represent recurring charges, "
                    "replay attacks, or telemetry duplicates before discarding.",
                    details={"duplicate_count": duplicate_count, "duplicate_pct": dup_pct},
                )
            )

        is_valid = len([i for i in issues if i.severity == "ERROR"]) == 0
        return ValidationResult(
            is_valid=is_valid,
            row_count=row_count,
            column_count=col_count,
            issues=issues,
            summary_stats=summary,
        )


def validate_file(file_path: Path, is_training: bool = True) -> ValidationResult:
    """Convenience helper to read CSV and run full validation."""
    if not file_path.exists():
        return ValidationResult(
            is_valid=False,
            row_count=0,
            column_count=0,
            issues=[
                ValidationIssue(
                    severity="ERROR",
                    category="FILE_NOT_FOUND",
                    message=f"File not found: {file_path}",
                    details={"path": str(file_path)},
                )
            ],
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            row_count=0,
            column_count=0,
            issues=[
                ValidationIssue(
                    severity="ERROR",
                    category="PARSE_ERROR",
                    message=f"Failed to read CSV: {str(e)}",
                    details={"error": str(e)},
                )
            ],
        )

    validator = DataValidator()
    return validator.validate(df, is_training=is_training)
