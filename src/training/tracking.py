"""Local experiment tracking and dataset fingerprinting for FraudGuard AI."""

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import PROJECT_ROOT


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 fingerprint for a dataset based on shape and sample values."""
    hasher = hashlib.sha256()
    # Hash shape and column names
    meta_str = f"{df.shape}_{list(df.columns)}"
    hasher.update(meta_str.encode("utf-8"))

    # Hash head, tail, and summary statistics to ensure fast deterministic fingerprinting
    head_bytes = df.head(10).to_csv(index=False).encode("utf-8")
    tail_bytes = df.tail(10).to_csv(index=False).encode("utf-8")
    hasher.update(head_bytes)
    hasher.update(tail_bytes)
    return hasher.hexdigest()[:16]


def get_git_commit_hash() -> str:
    """Retrieve active git commit hash if in a git repository, else return fallback."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "local-no-git"


class LocalExperimentTracker:
    """Manages local MLflow experiment tracking without requiring a remote server."""

    def __init__(self, experiment_name: str = "FraudGuard-Fraud-Detection"):
        self.experiment_name = experiment_name
        self._mlflow = None
        self._active_run = None
        self.enabled = False

        try:
            import mlflow
            self._mlflow = mlflow
            # Ensure local tracking URI points to workspace mlruns
            tracking_dir = PROJECT_ROOT / "mlruns"
            tracking_dir.mkdir(parents=True, exist_ok=True)
            self._mlflow.set_tracking_uri(f"file:///{tracking_dir.as_posix()}")
            self._mlflow.set_experiment(self.experiment_name)
            self.enabled = True
        except Exception as e:
            # Fallback gracefully if MLflow cannot initialize
            self.enabled = False
            self.init_error = str(e)

    def start_run(self, run_name: Optional[str] = None) -> Any:
        if self.enabled and self._mlflow:
            self._active_run = self._mlflow.start_run(run_name=run_name)
            return self._active_run
        return None

    def log_params(self, params: Dict[str, Any]) -> None:
        if self.enabled and self._mlflow and self._active_run:
            safe_params = {str(k): str(v)[:250] for k, v in params.items()}
            self._mlflow.log_params(safe_params)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._mlflow and self._active_run:
            numeric_metrics = {
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, bool))
            }
            self._mlflow.log_metrics(numeric_metrics, step=step)

    def log_artifact(self, local_path: Path, artifact_path: Optional[str] = None) -> None:
        if self.enabled and self._mlflow and self._active_run and local_path.exists():
            self._mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def end_run(self) -> None:
        if self.enabled and self._mlflow and self._active_run:
            self._mlflow.end_run()
            self._active_run = None
