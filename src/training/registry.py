"""File-based model registry, quality gates, candidate promotion, and rollback engine."""

import json
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.config import ModelConfig, load_model_config


class ModelRegistry:
    """Manages versioned model artifacts, candidate promotion gates, and rollback.

    NOTE ON DESIGN:
    This is a repository-local, file-based model registry implemented with JSON metadata
    and Joblib artifacts. It provides explicit quality gating, automated backups of previous
    champions, and deterministic rollback without requiring paid external registry services.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or load_model_config()
        self.models_dir = self.config.models_dir
        self.champion_dir = self.config.champion_dir
        self.candidates_dir = self.config.candidates_dir
        self.history_dir = self.config.models_dir / "history"
        self.reports_dir = self.config.reports_dir

        self.champion_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_champion_metadata(self) -> Optional[Dict[str, Any]]:
        """Read the active champion model's metadata."""
        meta_file = self.champion_dir / "metadata.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_candidates(self) -> List[Dict[str, Any]]:
        """List all candidate versions currently stored in models/candidates/."""
        candidates = []
        for p in self.candidates_dir.iterdir():
            if p.is_dir():
                meta_file = p / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            candidates.append(meta)
                    except Exception:
                        pass
        return sorted(candidates, key=lambda x: x.get("timestamp", ""), reverse=True)

    def evaluate_quality_gates(
        self,
        candidate_metrics: Dict[str, Any],
        champion_metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a candidate model against absolute quality gates and champion benchmarks."""
        qg = self.config.quality_gates
        gates_results = {}
        all_passed = True

        # 1. Absolute Quality Gates (Test metrics)
        pr_auc = float(candidate_metrics.get("pr_auc", 0.0))
        recall = float(candidate_metrics.get("recall", 0.0))
        precision = float(candidate_metrics.get("precision", 0.0))
        fpr = float(candidate_metrics.get("false_positive_rate", 1.0))

        gates_results["pr_auc_gate"] = {
            "required": f">= {qg.min_pr_auc}",
            "actual": pr_auc,
            "passed": pr_auc >= qg.min_pr_auc,
        }
        gates_results["recall_gate"] = {
            "required": f">= {qg.min_recall}",
            "actual": recall,
            "passed": recall >= qg.min_recall,
        }
        gates_results["precision_gate"] = {
            "required": f">= {qg.min_precision}",
            "actual": precision,
            "passed": precision >= qg.min_precision,
        }
        gates_results["fpr_gate"] = {
            "required": f"<= {qg.max_fpr}",
            "actual": fpr,
            "passed": fpr <= qg.max_fpr,
        }

        for gate in gates_results.values():
            if not gate["passed"]:
                all_passed = False

        # 2. Benchmark comparison vs active champion (if one exists)
        if champion_metrics:
            champ_test = champion_metrics.get("test_metrics", {})
            champ_pr_auc = float(champ_test.get("pr_auc", 0.0))
            # Candidate should not degrade PR-AUC below champion (within 0.01 margin)
            beats_champion = (pr_auc - champ_pr_auc) >= -0.01
            gates_results["champion_benchmark_gate"] = {
                "candidate_pr_auc": pr_auc,
                "champion_pr_auc": champ_pr_auc,
                "delta": round(pr_auc - champ_pr_auc, 4),
                "passed": beats_champion,
            }
            if not beats_champion:
                all_passed = False

        summary = {
            "overall_passed": all_passed,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "gates": gates_results,
        }
        return all_passed, summary

    def promote_candidate(
        self,
        candidate_version: str,
        reason: str = "Passed automated retraining quality gates",
    ) -> bool:
        """Promote a candidate model to active Champion with automatic rollback backup."""
        cand_dir = self.candidates_dir / candidate_version
        if not cand_dir.exists():
            raise FileNotFoundError(f"Candidate version directory not found: {cand_dir}")

        cand_model = cand_dir / "model.joblib"
        cand_meta = cand_dir / "metadata.json"

        if not cand_model.exists() or not cand_meta.exists():
            raise FileNotFoundError(f"Candidate {candidate_version} lacks model or metadata file.")

        with open(cand_meta, "r", encoding="utf-8") as f:
            cand_meta_dict = json.load(f)

        # 1. Backup current champion to history if exists
        champ_model = self.champion_dir / "model.joblib"
        champ_meta = self.champion_dir / "metadata.json"

        if champ_model.exists() and champ_meta.exists():
            with open(champ_meta, "r", encoding="utf-8") as f:
                current_champ_meta = json.load(f)
            prev_version = current_champ_meta.get("version", "unknown_prev")
            backup_dir = self.history_dir / f"backup_{prev_version}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(champ_model, backup_dir / "model.joblib")
            shutil.copy2(champ_meta, backup_dir / "metadata.json")

            # Also maintain quick-rollback pointer
            quick_rollback_dir = self.models_dir / "previous_champion"
            quick_rollback_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(champ_model, quick_rollback_dir / "model.joblib")
            shutil.copy2(champ_meta, quick_rollback_dir / "metadata.json")

        # 2. Copy candidate to champion directory
        shutil.copy2(cand_model, self.champion_dir / "model.joblib")

        # Update metadata with promotion info
        cand_meta_dict["promotion_info"] = {
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "candidate_source": candidate_version,
        }
        with open(self.champion_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(cand_meta_dict, f, indent=2)

        # 3. Save promotion log record
        log_file = self.reports_dir / "promotion_history.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(cand_meta_dict["promotion_info"]) + "\n")

        return True

    def rollback_to_previous_champion(self) -> bool:
        """Rollback the active champion to the immediately preceding champion."""
        prev_dir = self.models_dir / "previous_champion"
        prev_model = prev_dir / "model.joblib"
        prev_meta = prev_dir / "metadata.json"

        if not prev_model.exists() or not prev_meta.exists():
            raise FileNotFoundError("No previous champion backup found for rollback.")

        # Read previous metadata
        with open(prev_meta, "r", encoding="utf-8") as f:
            prev_meta_dict = json.load(f)

        # Copy back to champion dir
        shutil.copy2(prev_model, self.champion_dir / "model.joblib")

        prev_meta_dict["rollback_info"] = {
            "rolled_back_at": datetime.now(timezone.utc).isoformat(),
            "restored_version": prev_meta_dict.get("version"),
        }
        with open(self.champion_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(prev_meta_dict, f, indent=2)

        return True
