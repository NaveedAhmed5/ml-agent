"""Automated Test Suite for FastAPI Model Inference Microservice.

Tests Kubernetes liveness (/healthz), readiness (/readyz), release version auditing (/version),
and real-time classification inference (/predict).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, model_holder


@pytest.fixture(autouse=True)
def ensure_model_loaded():
    """Fixture ensuring the model artifact is loaded prior to each test."""
    model_holder.load_model()
    yield
    model_holder.load_model()


def test_healthz_liveness_probe():
    """Verify that /healthz returns 200 OK and status 'alive'."""
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data


def test_readyz_readiness_probe_success():
    """Verify that /readyz returns 200 and reports model ready when artifact is present."""
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert "model_name" in data
        assert data["expected_features"] > 0


def test_readyz_readiness_probe_unloaded():
    """Verify that /readyz returns 503 when model artifact is missing or not ready."""
    with TestClient(app) as client:
        original_model = model_holder.model
        try:
            model_holder.model = None
            response = client.get("/readyz")
            assert response.status_code == 503
            assert "not ready" in response.json()["detail"].lower()
        finally:
            model_holder.model = original_model


def test_version_metadata():
    """Verify that /version returns accurate deployment, model, and DVC dataset metadata."""
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert data["app_version"] == "1.2.0"
        assert "model_version" in data
        assert "dvc_dataset_version" in data
        assert data["environment"] == "mlops-prod"
        assert data["model_loaded"] is True


def test_predict_list_features():
    """Verify /predict computes valid prediction and probability distribution from array."""
    with TestClient(app) as client:
        payload = {"features": [1.5, -0.2, 2.1, 0.8, -0.5, 1.2]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["confidence"] <= 1.0
        assert len(data["probabilities"]) >= 2
        assert sum(data["probabilities"]) == pytest.approx(1.0, abs=1e-3)
        assert data["model_name"] == "mlops-classifier"
        assert data["app_version"] == "1.2.0"
        assert "timestamp" in data


def test_predict_named_features():
    """Verify /predict parses explicit named feature fields."""
    with TestClient(app) as client:
        payload = {
            "feature_1": 0.5,
            "feature_2": -1.2,
            "feature_3": 1.8,
            "feature_4": 0.2,
            "feature_5": 0.0,
            "feature_6": 1.1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["confidence"] <= 1.0


def test_predict_iris_features():
    """Verify /predict parses iris feature schema."""
    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]


def test_predict_unloaded_model_error():
    """Verify /predict returns 503 when model is not available in memory."""
    with TestClient(app) as client:
        original_model = model_holder.model
        try:
            model_holder.model = None
            response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
            assert response.status_code == 503
            assert "not loaded" in response.json()["detail"].lower()
        finally:
            model_holder.model = original_model

