"""Pydantic request and response schemas for FraudGuard AI API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TransactionPayload(BaseModel):
    """Schema for a single credit card transaction inference request."""

    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset", ge=0.0)
    Amount: float = Field(..., description="Transaction amount (must be non-negative)", ge=0.0)
    V1: float = Field(..., description="PCA transformed feature V1")
    V2: float = Field(..., description="PCA transformed feature V2")
    V3: float = Field(..., description="PCA transformed feature V3")
    V4: float = Field(..., description="PCA transformed feature V4")
    V5: float = Field(..., description="PCA transformed feature V5")
    V6: float = Field(..., description="PCA transformed feature V6")
    V7: float = Field(..., description="PCA transformed feature V7")
    V8: float = Field(..., description="PCA transformed feature V8")
    V9: float = Field(..., description="PCA transformed feature V9")
    V10: float = Field(..., description="PCA transformed feature V10")
    V11: float = Field(..., description="PCA transformed feature V11")
    V12: float = Field(..., description="PCA transformed feature V12")
    V13: float = Field(..., description="PCA transformed feature V13")
    V14: float = Field(..., description="PCA transformed feature V14")
    V15: float = Field(..., description="PCA transformed feature V15")
    V16: float = Field(..., description="PCA transformed feature V16")
    V17: float = Field(..., description="PCA transformed feature V17")
    V18: float = Field(..., description="PCA transformed feature V18")
    V19: float = Field(..., description="PCA transformed feature V19")
    V20: float = Field(..., description="PCA transformed feature V20")
    V21: float = Field(..., description="PCA transformed feature V21")
    V22: float = Field(..., description="PCA transformed feature V22")
    V23: float = Field(..., description="PCA transformed feature V23")
    V24: float = Field(..., description="PCA transformed feature V24")
    V25: float = Field(..., description="PCA transformed feature V25")
    V26: float = Field(..., description="PCA transformed feature V26")
    V27: float = Field(..., description="PCA transformed feature V27")
    V28: float = Field(..., description="PCA transformed feature V28")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Time": 406.0,
                "Amount": 149.62,
                **{f"V{i}": 0.05 * (i % 4 - 2) for i in range(1, 29)},
            }
        },
    )


class BatchTransactionPayload(BaseModel):
    """Schema for batch inference request with maximum batch size limit."""

    transactions: List[TransactionPayload] = Field(
        ..., min_length=1, max_length=1000, description="List of transactions (max 1000)"
    )


class SinglePredictionResponse(BaseModel):
    """Schema for single inference response."""

    request_id: str = Field(..., description="Unique UUID tracking the inference request")
    is_fraud: bool = Field(..., description="Binary classification at decision threshold")
    risk_score: float = Field(..., description="Calibrated fraud risk probability (0.0 to 1.0)")
    decision_threshold: float = Field(..., description="Optimal decision threshold tuned on validation")
    risk_level: str = Field(..., description="Categorical risk tier: LOW, MEDIUM, or HIGH")
    model_version: str = Field(..., description="Champion model artifact version")
    algorithm: str = Field(..., description="Champion classifier algorithm name")
    latency_ms: float = Field(..., description="Server inference latency in milliseconds")
    cautionary_notice: str = Field(
        default=(
            "Statistical risk estimate only. This prediction is not a verified legal or financial "
            "determination of fraud and should be evaluated alongside secondary verification rules."
        )
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch inference response."""

    batch_id: str
    total_processed: int
    predicted_fraud_count: int
    predictions: List[SinglePredictionResponse]
    batch_latency_ms: float


class HealthResponse(BaseModel):
    """Schema for API health status."""

    status: str
    timestamp: str
    model_loaded: bool
    champion_version: Optional[str] = None
    champion_algorithm: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Schema for detailed champion model metadata."""

    version: str
    algorithm: str
    timestamp: str
    decision_threshold: float
    feature_names: List[str]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    environment: Dict[str, Any]


class SystemMetricsResponse(BaseModel):
    """Schema for operational telemetry and audit summary."""

    total_logged_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    predicted_fraud_count: int
    predicted_fraud_rate: float
