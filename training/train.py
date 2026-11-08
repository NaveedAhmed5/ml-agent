"""Classification Model Training Script with MLflow Tracking and DVC Versioning.

Loads tabular dataset or Iris dataset, trains a RandomForestClassifier model,
computes evaluation metrics, writes metrics.json for DVC, and logs parameters,
metrics, and model artifacts to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def is_mlflow_server_up(uri: str, timeout: float = 1.0) -> bool:
    """Quick check if remote MLflow HTTP server is reachable."""
    if not uri.startswith("http"):
        return True
    try:
        req = urllib.request.Request(f"{uri.rstrip('/')}/health", headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 404)  # Server is alive
    except Exception:
        return False


def train(
    data_path: str = "data/dataset.csv",
    n_estimators: int = 100,
    max_depth: int = 4,
    random_state: int = 42,
    experiment_name: str = "classification-pipeline",
    metrics_path: str = "metrics.json",
    model_output_path: str = "models/model.joblib",
) -> dict[str, float]:
    """Train the model, save metrics.json and model artifact, and log to MLflow."""
    # 1. Load data
    data_file = Path(data_path)
    if data_file.exists():
        df = pd.read_csv(data_file)
        feature_cols = [c for c in df.columns if c not in ("target", "dataset_version", "generated_at")]
        X = df[feature_cols].values
        y = df["target"].values
        print(f"[train] Loaded {len(df)} samples from {data_file}")
    else:
        iris = load_iris()
        X = iris.data
        y = iris.target
        print(f"[train] {data_file} not found; fallback to Iris dataset ({len(y)} samples)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 2. Train model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    # 3. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
    }

    # 4. Save local model artifact
    model_file = Path(model_output_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_file)

    # 5. Write metrics.json for DVC tracking
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] Saved model to {model_file} and metrics to {metrics_file}")
    print(f"[train] Metrics: {metrics}")

    # 6. MLflow logging (fallback to local directory tracking if remote server is offline)
    try:
        import mlflow
        import mlflow.sklearn

        configured_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        if configured_uri.startswith("http") and not is_mlflow_server_up(configured_uri):
            tracking_uri = "file:./mlruns"
            print(f"[train] Remote server {configured_uri} offline; tracking to local '{tracking_uri}'")
        else:
            tracking_uri = configured_uri

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="dvc-pipeline-train") as run:
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
                "dataset": str(data_path),
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
            )
            print(f"[train] Logged run to MLflow: {run.info.run_id}")
    except Exception as exc:
        print(f"[train] MLflow logging note: {exc}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--data", type=str, default="data/dataset.csv")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--metrics", type=str, default="metrics.json")
    parser.add_argument("--output", type=str, default="models/model.joblib")
    args = parser.parse_args()

    train(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        metrics_path=args.metrics,
        model_output_path=args.output,
    )


if __name__ == "__main__":
    main()
