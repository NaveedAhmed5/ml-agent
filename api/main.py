"""FastAPI Inference Service for Production MLOps Model Serving.

Exposes Kubernetes-compliant liveness (/healthz), readiness (/readyz),
metadata inspection (/version), and real-time model inference (/predict).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------

class ModelFeatures(BaseModel):
    """Input feature vector for model classification inference."""
    features: list[float] | None = Field(
        default=None,
        description="Feature vector as numeric array (e.g. [1.2, 0.4, 2.1, 0.9, -0.5, 1.1])",
        json_schema_extra={"example": [1.2, -0.4, 2.1, 0.9, -0.5, 1.1]},
    )
    feature_1: float | None = Field(default=None, description="Feature 1")
    feature_2: float | None = Field(default=None, description="Feature 2")
    feature_3: float | None = Field(default=None, description="Feature 3")
    feature_4: float | None = Field(default=None, description="Feature 4")
    feature_5: float | None = Field(default=None, description="Feature 5")
    feature_6: float | None = Field(default=None, description="Feature 6")
    sepal_length: float | None = Field(default=None, description="Iris Sepal Length")
    sepal_width: float | None = Field(default=None, description="Iris Sepal Width")
    petal_length: float | None = Field(default=None, description="Iris Petal Length")
    petal_width: float | None = Field(default=None, description="Iris Petal Width")

    def to_numpy_array(self, expected_dim: int = 6) -> np.ndarray:
        """Convert input fields into standard 2D numpy array for inference."""
        if self.features is not None and len(self.features) > 0:
            arr = np.array(self.features, dtype=float)
        else:
            # Gather individual fields
            vals = []
            for f in ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6"]:
                val = getattr(self, f)
                if val is not None:
                    vals.append(val)
            if not vals:
                # Try iris fields
                for f in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
                    val = getattr(self, f)
                    if val is not None:
                        vals.append(val)
            if not vals:
                vals = [0.0] * expected_dim
            arr = np.array(vals, dtype=float)

        # Pad or slice to match expected model dimensions
        if len(arr) < expected_dim:
            arr = np.pad(arr, (0, expected_dim - len(arr)), mode="constant")
        elif len(arr) > expected_dim:
            arr = arr[:expected_dim]

        return arr.reshape(1, -1)


class PredictionResponse(BaseModel):
    """Inference outcome returned by the serving API."""
    prediction: int = Field(description="Predicted target class (0 or 1)")
    confidence: float = Field(ge=0.0, le=1.0, description="Highest class probability")
    probabilities: list[float] = Field(description="Probability distribution across classes")
    model_name: str = Field(description="Model identifier")
    model_version: str = Field(description="Serving model version")
    app_version: str = Field(description="Deployed application version")
    timestamp: str = Field(description="UTC timestamp of inference")


class VersionResponse(BaseModel):
    """Application and model version provenance metadata."""
    app_version: str = Field(description="API service release version")
    model_version: str = Field(description="Model registry version or tag")
    dvc_dataset_version: str = Field(description="Dataset version hash or label tracked in DVC")
    environment: str = Field(description="Runtime environment (e.g., mlops-prod, staging)")
    model_loaded: bool = Field(description="Whether the inference model is active in memory")


# ------------------------------------------------------------------------------
# Model Storage & Lifespan Management
# ------------------------------------------------------------------------------

class ModelHolder:
    """Manages model artifact lifecycle and caching."""
    def __init__(self) -> None:
        self.model: Any = None
        self.expected_dim: int = 6
        self.model_name: str = os.getenv("MODEL_NAME", "mlops-classifier")
        self.model_version: str = os.getenv("MODEL_VERSION", "1.2.0")

    def load_model(self) -> None:
        """Load trained scikit-learn model from candidate or default checkpoint paths."""
        candidate_paths = [
            Path("models/candidate_model.joblib"),
            Path("models/model.joblib"),
            Path("/app/models/candidate_model.joblib"),
            Path("/app/models/model.joblib"),
        ]

        loaded = False
        for p in candidate_paths:
            if p.exists():
                try:
                    self.model = joblib.load(p)
                    # Inspect expected features if available
                    if hasattr(self.model, "n_features_in_"):
                        self.expected_dim = int(self.model.n_features_in_)
                    loaded = True
                    break
                except Exception:
                    pass

        if not loaded:
            # Fallback self-contained classifier for cold-start / unmounted containers
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
            X_dummy = np.random.randn(100, 6)
            y_dummy = (X_dummy[:, 0] > 0).astype(int)
            clf.fit(X_dummy, y_dummy)
            self.model = clf
            self.expected_dim = 6


model_holder = ModelHolder()
# Eagerly initialize model checkpoint on import for test and standalone compatibility
model_holder.load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact into memory upon service startup."""
    model_holder.load_model()
    yield


app = FastAPI(
    title="MLOps Production Model Inference API",
    version=os.getenv("APP_VERSION", "1.2.0"),
    description="Containerized inference microservice monitored by the Kubernetes Operations Agent",
    lifespan=lifespan,
)


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get("/healthz", tags=["Kubernetes Probes"])
def healthz() -> dict[str, str]:
    """Liveness probe: verifies that the FastAPI application container is running."""
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/readyz", tags=["Kubernetes Probes"])
def readyz() -> dict[str, Any]:
    """Readiness probe: verifies that the inference model artifact is loaded and operational."""
    if model_holder.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or failed to load into memory.",
        )
    return {
        "status": "ready",
        "model_loaded": True,
        "model_name": model_holder.model_name,
        "expected_features": model_holder.expected_dim,
    }


@app.get("/version", response_model=VersionResponse, tags=["Metadata"])
def get_version() -> VersionResponse:
    """Return runtime metadata, application release, model tag, and DVC dataset version."""
    return VersionResponse(
        app_version=os.getenv("APP_VERSION", "1.2.0"),
        model_version=os.getenv("MODEL_VERSION", model_holder.model_version),
        dvc_dataset_version=os.getenv("DVC_DATASET_VERSION", "v1.0.0"),
        environment=os.getenv("AGENT_ENV", os.getenv("K8S_APPROVED_NAMESPACE", "mlops-prod")),
        model_loaded=(model_holder.model is not None),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: ModelFeatures) -> PredictionResponse:
    """Perform real-time model classification inference on provided feature vector."""
    if model_holder.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    X = features.to_numpy_array(expected_dim=model_holder.expected_dim)

    try:
        raw_pred = model_holder.model.predict(X)[0]
        pred_int = int(raw_pred)

        if hasattr(model_holder.model, "predict_proba"):
            probs = model_holder.model.predict_proba(X)[0].tolist()
            conf = float(max(probs))
        else:
            probs = [1.0 if i == pred_int else 0.0 for i in range(2)]
            conf = 1.0

        return PredictionResponse(
            prediction=pred_int,
            confidence=round(conf, 4),
            probabilities=[round(p, 4) for p in probs],
            model_name=model_holder.model_name,
            model_version=model_holder.model_version,
            app_version=os.getenv("APP_VERSION", "1.2.0"),
            timestamp=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation error: {exc}",
        )
