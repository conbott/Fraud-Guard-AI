"""Integration tests for the FastAPI service endpoints."""

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["champion_version"] is not None


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "algorithm" in data
    assert "decision_threshold" in data
    assert len(data["feature_names"]) == 30


def test_predict_single_valid(valid_transaction_dict):
    response = client.post("/predict", json=valid_transaction_dict)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert isinstance(data["is_fraud"], bool)
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "cautionary_notice" in data
    assert "X-Request-ID" in response.headers


def test_predict_single_invalid_missing_feature(valid_transaction_dict):
    bad_payload = valid_transaction_dict.copy()
    del bad_payload["V14"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_single_negative_amount(valid_transaction_dict):
    bad_payload = valid_transaction_dict.copy()
    bad_payload["Amount"] = -10.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_single_extra_forbidden_field(valid_transaction_dict):
    bad_payload = valid_transaction_dict.copy()
    bad_payload["card_number"] = "4111111111111111"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_batch_valid(valid_transaction_dict):
    batch_payload = {"transactions": [valid_transaction_dict] * 5}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 5
    assert len(data["predictions"]) == 5
    assert "batch_id" in data
    assert "X-Batch-ID" in response.headers


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_logged_requests" in data
    assert "avg_latency_ms" in data
