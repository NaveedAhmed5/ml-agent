"""FastAPI Inference Service for Iris Classification Model.

Provides health checks, prediction endpoints, and model metadata inspection.
"""

from __future__ import annotations

import os
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="MLOps Iris Model Serving API",
    version="1.0.0",
    description="Production inference API monitored and managed by the MLOps AI Agent",
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0.0, description="Sepal length in cm", example=5.1)
    sepal_width: float = Field(..., ge=0.0, description="Sepal width in cm", example=3.5)
    petal_length: float = Field(..., ge=0.0, description="Petal length in cm", example=1.4)
    petal_width: float = Field(..., ge=0.0, description="Petal width in cm", example=0.2)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Target class index (0: Setosa, 1: Versicolour, 2: Virginica)")
    class_name: str = Field(..., description="Human-readable class name")
    probabilities: list[float] = Field(..., description="Prediction probability distribution")
    model_version: str = Field(default="v1", description="Serving model version")


CLASS_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.get("/health", tags=["Monitoring"])
def health() -> dict[str, str]:
    """Liveness and readiness health probe."""
    return {"status": "ok", "service": "iris-api"}


@app.get("/metadata", tags=["Monitoring"])
def metadata() -> dict[str, Any]:
    """Returns model metadata and environment configuration."""
    return {
        "model_name": "iris-classifier",
        "environment": os.getenv("AGENT_ENV", "production"),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: IrisFeatures) -> PredictionResponse:
    """Run inference using the loaded model artifact."""
    # Stub inference logic; model loading from MLflow / joblib will be integrated
    return PredictionResponse(
        prediction=0,
        class_name="setosa",
        probabilities=[0.98, 0.01, 0.01],
        model_version="1",
    )

