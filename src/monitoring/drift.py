"""Data and prediction drift monitoring engine using PSI and KS-tests."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import PROJECT_ROOT


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate the Population Stability Index (PSI) between reference and current samples.

    Threshold Standards:
      - PSI < 0.10: Insignificant shift (Feature is stable).
      - 0.10 <= PSI < 0.25: Moderate shift (Monitor closely).
      - PSI >= 0.25: Significant distributional shift (Action/Investigation required).
    """
    ref_clean = np.asarray(reference[~np.isnan(reference)], dtype=float)
    curr_clean = np.asarray(current[~np.isnan(current)], dtype=float)

    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0

    # Determine quantile bin edges from the reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    # Ensure strictly monotonic bin edges
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-5

    # Bucket counts
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    # Proportions with Laplace smoothing
    ref_pct = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * num_buckets)
    curr_pct = (curr_counts + epsilon) / (np.sum(curr_counts) + epsilon * num_buckets)

    # PSI Formula: sum((Actual - Expected) * ln(Actual / Expected))
    psi_values = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    return float(np.sum(psi_values))


@dataclass
class FeatureDriftReport:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    drift_status: str  # "NO_DRIFT", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"
    missing_rate_ref: float
    missing_rate_curr: float
    ref_mean: float
    curr_mean: float
    ref_std: float
    curr_std: float


@dataclass
class DatasetDriftReport:
    evaluated_at: str
    reference_sample_count: int
    current_sample_count: int
    overall_drift_status: str
    drifted_features_count: int
    total_monitored_features: int
    feature_reports: List[FeatureDriftReport] = field(default_factory=list)
    prediction_drift: Dict[str, Any] = field(default_factory=dict)
    conceptual_notes: str = (
        "Data Drift (P(X) shift) indicates feature inputs have changed. It does not "
        "automatically mean the model is inaccurate, but warrants review. Concept Drift "
        "(P(Y|X) shift) or Performance Degradation requires ground-truth labels to measure."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DriftMonitor:
    """Computes distribution shift metrics comparing incoming batches to training baseline."""

    def __init__(self, reference_data_path: Optional[Path] = None):
        self.ref_path = (
            reference_data_path
            or (PROJECT_ROOT / "data" / "processed" / "reference_features.parquet")
        )
        self._reference_df: Optional[pd.DataFrame] = None

    @property
    def reference_data(self) -> pd.DataFrame:
        if self._reference_df is None:
            if not self.ref_path.exists():
                self._generate_reference_cache()
            self._reference_df = pd.read_parquet(self.ref_path)
        return self._reference_df

    def _generate_reference_cache(self) -> None:
        """Generate baseline reference features without modifying champion model."""
        from src.config import load_model_config
        from src.data.synthetic import create_test_fixtures
        from src.inference.loader import ModelLoader

        cfg = load_model_config()
        champion = ModelLoader.load_champion()

        if cfg.raw_data_path.exists():
            df_raw = pd.read_csv(cfg.raw_data_path, nrows=2000)
        else:
            src_file = cfg.synthetic_path if cfg.synthetic_path.exists() else cfg.fixture_path
            if not src_file.exists():
                create_test_fixtures()
                src_file = cfg.fixture_path
            df_raw = pd.read_csv(src_file)

        proc_feats = champion.preprocessor.transform(df_raw)
        ref_df = pd.DataFrame(proc_feats, columns=champion.preprocessor.feature_names_)
        if cfg.amount_column in df_raw.columns:
            ref_df["Amount_original"] = df_raw[cfg.amount_column].values

        self.ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_df.to_parquet(self.ref_path, index=False)

    def evaluate_drift(
        self,
        current_df: pd.DataFrame,
        features: Optional[List[str]] = None,
        risk_scores: Optional[np.ndarray] = None,
    ) -> DatasetDriftReport:
        """Run drift analysis across specified feature columns."""
        ref_df = self.reference_data
        monitored_features = features or [
            c for c in ["Amount", "V1", "V2", "V3", "V4", "V10", "V12", "V14", "V17"]
            if c in current_df.columns and c in ref_df.columns
        ]

        feature_reports: List[FeatureDriftReport] = []
        drifted_count = 0

        for feat in monitored_features:
            ref_vals = ref_df[feat].dropna().values.astype(float)
            curr_vals = current_df[feat].dropna().values.astype(float)

            psi = calculate_psi(ref_vals, curr_vals)

            try:
                ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
            except Exception:
                ks_stat, p_val = 0.0, 1.0

            if psi >= 0.25 or p_val < 0.01:
                status = "SIGNIFICANT_DRIFT"
                drifted_count += 1
            elif psi >= 0.10 or p_val < 0.05:
                status = "MODERATE_DRIFT"
            else:
                status = "NO_DRIFT"

            report = FeatureDriftReport(
                feature_name=feat,
                psi=round(psi, 4),
                ks_statistic=round(float(ks_stat), 4),
                ks_p_value=round(float(p_val), 6),
                drift_status=status,
                missing_rate_ref=round(float(ref_df[feat].isna().mean()), 4),
                missing_rate_curr=round(float(current_df[feat].isna().mean()), 4),
                ref_mean=round(float(np.mean(ref_vals)), 4) if len(ref_vals) else 0.0,
                curr_mean=round(float(np.mean(curr_vals)), 4) if len(curr_vals) else 0.0,
                ref_std=round(float(np.std(ref_vals)), 4) if len(ref_vals) else 0.0,
                curr_std=round(float(np.std(curr_vals)), 4) if len(curr_vals) else 0.0,
            )
            feature_reports.append(report)

        if drifted_count >= 2:
            overall_status = "CRITICAL_DRIFT"
        elif drifted_count == 1:
            overall_status = "WARNING_DRIFT"
        else:
            overall_status = "STABLE"

        pred_drift = {}
        if risk_scores is not None and len(risk_scores) > 0:
            scores_arr = np.asarray(risk_scores, dtype=float)
            pred_drift = {
                "sample_count": len(scores_arr),
                "mean_score": round(float(np.mean(scores_arr)), 4),
                "median_score": round(float(np.median(scores_arr)), 4),
                "p95_score": round(float(np.percentile(scores_arr, 95)), 4),
                "high_risk_percentage": round(float((scores_arr >= 0.50).mean() * 100), 2),
            }

        return DatasetDriftReport(
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            reference_sample_count=len(ref_df),
            current_sample_count=len(current_df),
            overall_drift_status=overall_status,
            drifted_features_count=drifted_count,
            total_monitored_features=len(monitored_features),
            feature_reports=feature_reports,
            prediction_drift=pred_drift,
        )


def simulate_synthetic_drift(
    base_df: pd.DataFrame,
    drift_type: str = "amount_spike",
    severity: float = 3.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate controlled synthetic drift for testing and dashboard demonstration.

    Simulations:
      - 'amount_spike': Extreme increase in transaction amounts (e.g. inflation / attack).
      - 'feature_shift': Substantial negative shift in key PCA component V14.
      - 'mixed': Simultaneous shift in multiple features.
    """
    df_drifted = base_df.copy()
    rng = np.random.default_rng(random_seed)

    if drift_type == "amount_spike":
        # Multiply amounts by severe multiplier
        df_drifted["Amount"] = df_drifted["Amount"] * severity + rng.uniform(50, 200, size=len(df_drifted))
    elif drift_type == "feature_shift":
        df_drifted["V14"] = df_drifted["V14"] - (severity * 2.0)
    elif drift_type == "mixed":
        df_drifted["Amount"] = df_drifted["Amount"] * severity
        df_drifted["V14"] = df_drifted["V14"] - (severity * 2.0)
        df_drifted["V10"] = df_drifted["V10"] - (severity * 1.5)

    return df_drifted
