"""Model training, baseline comparison, threshold tuning, and artifact serialization."""

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import ModelConfig, load_model_config
from src.data.loader import DataLoader
from src.features.preprocessor import FraudFeaturePreprocessor
from src.training.evaluate import (
    EvaluationMetrics,
    compute_curve_points,
    evaluate_predictions,
    tune_threshold_on_validation,
)
from src.training.tracking import (
    LocalExperimentTracker,
    compute_dataset_fingerprint,
    get_git_commit_hash,
)


class ModelTrainer:
    """Orchestrates end-to-end model training, model comparison, and artifact export."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.loader = DataLoader(self.config)
        self.tracker = LocalExperimentTracker()

    def train_pipeline(
        self,
        data_path: Optional[Path] = None,
        save_candidate: bool = True,
        is_champion: bool = False,
    ) -> Dict[str, Any]:
        """Execute the end-to-end training pipeline."""
        start_time = time.time()
        timestamp_str = datetime.now(timezone.utc).isoformat()

        # 1. Load dataset
        path = data_path or (
            self.config.raw_data_path
            if self.config.raw_data_path.exists()
            else self.config.synthetic_path
        )
        if not path.exists():
            from src.data.synthetic import create_test_fixtures
            path, _ = create_test_fixtures()

        df = pd.read_csv(path)
        dataset_fingerprint = compute_dataset_fingerprint(df)

        # 2. Leakage-safe split
        X_train, y_train, X_val, y_val, X_test, y_test = self.loader.split_data(df)

        # 3. Fit preprocessor ONLY on training partition
        preprocessor = FraudFeaturePreprocessor(self.config)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        X_test_proc = preprocessor.transform(X_test)

        # Save reference features for drift monitoring
        ref_df = pd.DataFrame(X_train_proc, columns=preprocessor.feature_names_)
        ref_df["Amount_original"] = X_train[self.config.amount_column].values
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        ref_path = self.config.processed_dir / "reference_features.parquet"
        ref_df.to_parquet(ref_path, index=False)

        # 4. Define candidate models
        seed = self.config.random_seed
        models = {
            "dummy_baseline": DummyClassifier(strategy="stratified", random_state=seed),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.1,
                solver="lbfgs",
                random_state=seed,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        }

        comparison_results: Dict[str, Dict[str, Any]] = {}
        best_model_name = ""
        best_pr_auc = -1.0
        best_model_obj = None
        best_threshold = 0.50
        best_val_metrics: Optional[EvaluationMetrics] = None

        # Start MLflow run
        run_name = f"training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.tracker.start_run(run_name=run_name)
        self.tracker.log_params({
            "dataset_path": str(path),
            "dataset_fingerprint": dataset_fingerprint,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "random_seed": seed,
            "git_commit": get_git_commit_hash(),
        })

        # 5. Train each model, tune threshold on validation, select champion
        for name, model in models.items():
            model.fit(X_train_proc, y_train)

            # Predict probabilities on validation partition
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_val_proc)[:, 1]
            else:
                y_val_prob = model.predict(X_val_proc).astype(float)

            # Tune decision threshold on validation set
            tuned_thresh, thresh_history = tune_threshold_on_validation(
                y_val=y_val.values,
                y_prob_val=y_val_prob,
                metric=self.config.threshold_metric,
            )

            # Evaluate on validation at default threshold (0.50) and tuned threshold
            val_metrics_default = evaluate_predictions(
                y_val.values, y_val_prob, threshold=0.50, dataset_type="validation"
            )
            val_metrics_tuned = evaluate_predictions(
                y_val.values, y_val_prob, threshold=tuned_thresh, dataset_type="validation"
            )

            comparison_results[name] = {
                "val_pr_auc": val_metrics_tuned.pr_auc,
                "val_roc_auc": val_metrics_tuned.roc_auc,
                "val_f1_default": val_metrics_default.f1_score,
                "val_f1_tuned": val_metrics_tuned.f1_score,
                "val_recall_tuned": val_metrics_tuned.recall,
                "val_precision_tuned": val_metrics_tuned.precision,
                "optimal_threshold": tuned_thresh,
            }

            # Select based on validation PR-AUC (key imbalanced metric)
            if val_metrics_tuned.pr_auc > best_pr_auc:
                best_pr_auc = val_metrics_tuned.pr_auc
                best_model_name = name
                best_model_obj = model
                best_threshold = tuned_thresh
                best_val_metrics = val_metrics_tuned

        # 6. Evaluate selected champion on held-out TEST set at tuned threshold
        if hasattr(best_model_obj, "predict_proba"):
            y_test_prob = best_model_obj.predict_proba(X_test_proc)[:, 1]
        else:
            y_test_prob = best_model_obj.predict(X_test_proc).astype(float)

        test_metrics = evaluate_predictions(
            y_test.values,
            y_test_prob,
            threshold=best_threshold,
            dataset_type="test",
        )

        # 7. Compute curves for evaluation visualization
        curve_data = compute_curve_points(y_test.values, y_test_prob)

        duration = round(time.time() - start_time, 2)
        version_id = f"model_v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # 8. Build artifact bundle
        artifact_bundle = {
            "model": best_model_obj,
            "preprocessor": preprocessor,
            "algorithm": best_model_name,
            "version": version_id,
            "timestamp": timestamp_str,
            "duration_seconds": duration,
            "dataset_fingerprint": dataset_fingerprint,
            "random_seed": seed,
            "feature_names": preprocessor.feature_names_,
            "optimal_threshold": best_threshold,
            "val_metrics": best_val_metrics.to_dict() if best_val_metrics else {},
            "test_metrics": test_metrics.to_dict(),
            "model_comparison": comparison_results,
            "curve_points": curve_data,
            "environment": {
                "python_version": sys.version,
                "scikit_learn_version": sklearn.__version__,
                "platform": platform.platform(),
            },
        }

        # 9. Save artifacts
        if is_champion:
            save_dir = self.config.champion_dir
        elif save_candidate:
            save_dir = self.config.candidates_dir / version_id
        else:
            save_dir = self.config.candidates_dir / "latest_run"

        save_dir.mkdir(parents=True, exist_ok=True)
        model_file = save_dir / "model.joblib"
        metadata_file = save_dir / "metadata.json"

        joblib.dump(artifact_bundle, model_file)

        # Exclude unpickled model object for JSON metadata
        meta_dict = {k: v for k, v in artifact_bundle.items() if k not in ("model", "preprocessor")}
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

        # 10. Log to MLflow
        self.tracker.log_params({
            "selected_model": best_model_name,
            "optimal_threshold": best_threshold,
            "version": version_id,
        })
        self.tracker.log_metrics({
            "test_pr_auc": test_metrics.pr_auc,
            "test_roc_auc": test_metrics.roc_auc,
            "test_f1": test_metrics.f1_score,
            "test_recall": test_metrics.recall,
            "test_precision": test_metrics.precision,
            "test_fpr": test_metrics.false_positive_rate,
            "duration_seconds": duration,
        })
        self.tracker.log_artifact(metadata_file)
        self.tracker.end_run()

        return {
            "version": version_id,
            "algorithm": best_model_name,
            "optimal_threshold": best_threshold,
            "artifact_path": str(model_file),
            "metadata_path": str(metadata_file),
            "val_metrics": best_val_metrics.to_dict() if best_val_metrics else {},
            "test_metrics": test_metrics.to_dict(),
            "comparison": comparison_results,
            "duration": duration,
        }
