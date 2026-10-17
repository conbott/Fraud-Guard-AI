"""API route endpoints for FraudGuard AI."""

import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Response

from src.api.schemas import (
    BatchPredictionResponse,
    BatchTransactionPayload,
    HealthResponse,
    ModelInfoResponse,
    SinglePredictionResponse,
    SystemMetricsResponse,
    TransactionPayload,
)
from src.inference.predictor import FraudPredictor
from src.monitoring.metrics import AuditLogger

router = APIRouter()

# Global predictor and audit instances
_predictor: FraudPredictor = None
_audit_logger = AuditLogger()


def get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = FraudPredictor()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Champion model is not currently available: {str(e)}",
            )
    return _predictor


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """System health check endpoint verifying API and model readiness."""
    now_str = datetime.now(timezone.utc).isoformat()
    try:
        pred = get_predictor()
        return HealthResponse(
            status="healthy",
            timestamp=now_str,
            model_loaded=True,
            champion_version=pred.model_bundle.version,
            champion_algorithm=pred.model_bundle.algorithm,
        )
    except HTTPException:
        return HealthResponse(
            status="degraded",
            timestamp=now_str,
            model_loaded=False,
            champion_version=None,
            champion_algorithm=None,
        )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    """Retrieve detailed metadata and evaluation metrics for active champion."""
    pred = get_predictor()
    meta = pred.model_bundle.metadata
    return ModelInfoResponse(
        version=pred.model_bundle.version,
        algorithm=pred.model_bundle.algorithm,
        timestamp=meta.get("timestamp", "unknown"),
        decision_threshold=pred.model_bundle.optimal_threshold,
        feature_names=pred.model_bundle.feature_names,
        val_metrics=meta.get("val_metrics", {}),
        test_metrics=meta.get("test_metrics", {}),
        environment=meta.get("environment", {}),
    )


@router.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(
    payload: TransactionPayload,
    request: Request,
    response: Response,
) -> SinglePredictionResponse:
    """Predict calibrated fraud risk probability for a single transaction."""
    start_time = time.perf_counter()
    req_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = req_id

    pred = get_predictor()
    txn_dict = payload.model_dump()

    try:
        result = pred.predict_single(txn_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

    latency = (time.perf_counter() - start_time) * 1000.0

    # Log inference telemetry asynchronously to SQLite audit store
    try:
        client_ip = request.client.host if request.client else "unknown"
        _audit_logger.log_inference(
            request_id=req_id,
            model_version=result.model_version,
            risk_score=result.risk_score,
            is_fraud=result.is_fraud,
            risk_level=result.risk_level,
            latency_ms=latency,
            client_ip=client_ip,
        )
    except Exception:
        pass  # Telemetry failures must never drop user inference requests

    return SinglePredictionResponse(
        request_id=req_id,
        is_fraud=result.is_fraud,
        risk_score=result.risk_score,
        decision_threshold=result.threshold,
        risk_level=result.risk_level,
        model_version=result.model_version,
        algorithm=result.algorithm,
        latency_ms=round(latency, 2),
        cautionary_notice=result.cautionary_notice,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(
    payload: BatchTransactionPayload,
    response: Response,
) -> BatchPredictionResponse:
    """Batch inference with maximum batch size enforcement (up to 1,000 transactions)."""
    start_time = time.perf_counter()
    batch_id = str(uuid.uuid4())
    response.headers["X-Batch-ID"] = batch_id

    pred = get_predictor()
    df = pd.DataFrame([t.model_dump() for t in payload.transactions])

    try:
        outputs = pred.predict_batch(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch inference error: {str(e)}")

    total_latency = (time.perf_counter() - start_time) * 1000.0
    item_latency = total_latency / max(len(outputs), 1)

    predictions_resp: list[SinglePredictionResponse] = []
    fraud_count = 0

    for out in outputs:
        if out.is_fraud:
            fraud_count += 1
        predictions_resp.append(
            SinglePredictionResponse(
                request_id=str(uuid.uuid4()),
                is_fraud=out.is_fraud,
                risk_score=out.risk_score,
                decision_threshold=out.threshold,
                risk_level=out.risk_level,
                model_version=out.model_version,
                algorithm=out.algorithm,
                latency_ms=round(item_latency, 2),
                cautionary_notice=out.cautionary_notice,
            )
        )

    return BatchPredictionResponse(
        batch_id=batch_id,
        total_processed=len(predictions_resp),
        predicted_fraud_count=fraud_count,
        predictions=predictions_resp,
        batch_latency_ms=round(total_latency, 2),
    )


@router.get("/metrics", response_model=SystemMetricsResponse, tags=["System"])
def system_metrics() -> SystemMetricsResponse:
    """Retrieve operational telemetry and latency distribution."""
    metrics = _audit_logger.get_summary_metrics()
    return SystemMetricsResponse(**metrics)
