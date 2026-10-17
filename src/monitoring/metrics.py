"""Inference audit logging and system metrics storage via SQLite."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import PROJECT_ROOT


class AuditLogger:
    """Lightweight SQLite audit store for inference telemetry and latency tracking."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (PROJECT_ROOT / "data" / "processed" / "predictions_audit.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS inference_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        risk_score REAL NOT NULL,
                        is_fraud INTEGER NOT NULL,
                        risk_level TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        client_ip TEXT
                    )
                """)
        finally:
            conn.close()

    def log_inference(
        self,
        request_id: str,
        model_version: str,
        risk_score: float,
        is_fraud: bool,
        risk_level: str,
        latency_ms: float,
        client_ip: Optional[str] = None,
    ) -> None:
        """Insert a single prediction audit record."""
        ts = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO inference_logs (
                        request_id, timestamp, model_version, risk_score, is_fraud, risk_level, latency_ms, client_ip
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        ts,
                        model_version,
                        risk_score,
                        1 if is_fraud else 0,
                        risk_level,
                        round(latency_ms, 2),
                        client_ip or "local",
                    ),
                )
        finally:
            conn.close()

    def get_summary_metrics(self, limit: int = 1000) -> Dict[str, Any]:
        """Compute aggregated operational metrics from recent logs."""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM inference_logs ORDER BY id DESC LIMIT {limit}",
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            return {
                "total_logged_requests": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "predicted_fraud_count": 0,
                "predicted_fraud_rate": 0.0,
            }

        total = len(df)
        fraud_count = int(df["is_fraud"].sum())
        return {
            "total_logged_requests": total,
            "avg_latency_ms": round(float(df["latency_ms"].mean()), 2),
            "p95_latency_ms": round(float(df["latency_ms"].quantile(0.95)), 2),
            "predicted_fraud_count": fraud_count,
            "predicted_fraud_rate": round(float(fraud_count / total), 4),
        }
