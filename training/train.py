"""Production-Ready Model Training Pipeline with MLflow & DVC Provenance Tracking.

Supports configurable scikit-learn models (RandomForest, GradientBoosting, LogisticRegression),
modular data validation, preprocessing, evaluation with ROC-AUC, quality gate checks,
and comprehensive provenance logging to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# 1. Provenance Helpers
# ------------------------------------------------------------------------------

def get_git_commit() -> str:
    """Retrieve current Git commit hash safely."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


def get_dvc_md5(data_path: str = "data/dataset.csv") -> str:
    """Retrieve MD5 content hash from associated .dvc pointer file."""
    dvc_file = Path(data_path).with_suffix(Path(data_path).suffix + ".dvc")
    if dvc_file.exists():
        try:
            with open(dvc_file, encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
            outs = content.get("outs", [])
            if outs:
                return str(outs[0].get("md5", "unknown"))
        except Exception:
            pass
    return "untracked"


def resolve_mlflow_tracking_uri() -> str:
    """Resolve an active MLflow tracking URI, falling back to SQLite if HTTP server is offline."""
    configured_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if configured_uri.startswith("http"):
        try:
            req = urllib.request.Request(
                f"{configured_uri.rstrip('/')}/health",
                headers={"User-Agent": "mlflow-healthcheck"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status in (200, 404):
                    return configured_uri
        except Exception:
            # HTTP server not reachable; use local sqlite database
            return "sqlite:///mlflow.db"
    return configured_uri


# ------------------------------------------------------------------------------
# 2. Modular Pipeline Stages
# ------------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validate tabular dataset schema, missingness, and class balance."""
    if df.empty:
        raise ValueError("Data validation failed: DataFrame is empty.")

    if "target" not in df.columns:
        raise ValueError("Data validation failed: Required 'target' column is missing.")

    missing_count = int(df.isnull().sum().sum())
    total_samples = len(df)
    class_counts = df["target"].value_counts().to_dict()
    minority_ratio = min(class_counts.values()) / total_samples

    if total_samples < 50:
        raise ValueError(f"Dataset too small: only {total_samples} records found.")

    if minority_ratio < 0.05:
        raise ValueError(f"Extreme class imbalance: minority ratio is {minority_ratio:.4f}")

    validation_summary = {
        "status": "VALID",
        "total_samples": total_samples,
        "missing_values": missing_count,
        "class_distribution": class_counts,
        "minority_class_ratio": round(minority_ratio, 4),
    }
    return validation_summary


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Separate features, stratify, and split into train and test partitions."""
    ignore_cols = {"target", "dataset_version", "generated_at"}
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    **hyperparams: Any,
) -> Any:
    """Instantiate and train specified scikit-learn model."""
    model_type_clean = model_type.lower().strip()

    if model_type_clean in ("rf", "random_forest", "randomforest"):
        n_estimators = int(hyperparams.get("n_estimators", 100))
        max_depth = hyperparams.get("max_depth", 4)
        if max_depth is not None:
            max_depth = int(max_depth)
        random_state = int(hyperparams.get("random_state", 42))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    elif model_type_clean in ("gb", "gradient_boosting", "gradientboosting"):
        n_estimators = int(hyperparams.get("n_estimators", 80))
        learning_rate = float(hyperparams.get("learning_rate", 0.1))
        max_depth = int(hyperparams.get("max_depth", 3))
        random_state = int(hyperparams.get("random_state", 42))

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
    elif model_type_clean in ("lr", "logistic_regression", "logisticregression"):
        C = float(hyperparams.get("c_param", 1.0))
        max_iter = int(hyperparams.get("max_iter", 200))
        random_state = int(hyperparams.get("random_state", 42))

        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")

    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute comprehensive evaluation metrics including train-test overfitting gap."""
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    # Compute ROC-AUC score
    roc_auc = 0.5
    try:
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)
            if probs.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, probs[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, probs, multi_class="ovr")
    except Exception:
        pass

    # Train score to detect overfitting
    train_f1 = f1
    overfit_gap = 0.0
    if X_train is not None and y_train is not None:
        train_preds = clf.predict(X_train)
        train_f1 = f1_score(y_train, train_preds, average="weighted", zero_division=0)
        overfit_gap = max(0.0, float(train_f1 - f1))

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "train_f1_score": round(float(train_f1), 4),
        "overfitting_gap": round(float(overfit_gap), 4),
    }


def check_quality_gate(
    metrics: dict[str, float],
    min_f1: float = 0.75,
    max_overfit_gap: float = 0.20,
) -> tuple[bool, str]:
    """Evaluate whether run satisfies production-readiness quality gates."""
    f1 = metrics.get("f1_score", 0.0)
    gap = metrics.get("overfitting_gap", 0.0)

    reasons: list[str] = []
    if f1 < min_f1:
        reasons.append(f"F1 score ({f1:.4f}) below required threshold ({min_f1:.4f})")
    if gap > max_overfit_gap:
        reasons.append(f"Overfitting gap ({gap:.4f}) exceeds allowable tolerance ({max_overfit_gap:.4f})")

    if not reasons:
        return True, f"Passed quality gate (F1: {f1:.4f} >= {min_f1}, Overfit gap: {gap:.4f} <= {max_overfit_gap})"
    return False, "; ".join(reasons)


# ------------------------------------------------------------------------------
# 3. Main Training Execution with MLflow Tracking
# ------------------------------------------------------------------------------

def run_training_pipeline(
    data_path: str = "data/dataset.csv",
    model_type: str = "random_forest",
    experiment_name: str = "iris-classification-pipeline",
    registered_model_name: str = "iris-classifier",
    min_f1: float = 0.75,
    output_model_path: str = "models/model.joblib",
    output_metrics_path: str = "metrics.json",
    **hyperparams: Any,
) -> dict[str, Any]:
    """Execute end-to-end training pipeline and record run to MLflow."""
    # 1. Load data
    data_file = Path(data_path)
    if data_file.exists():
        df = pd.read_csv(data_file)
        version_label = str(df["dataset_version"].iloc[0]) if "dataset_version" in df.columns else "v1.0.0"
        print(f"[pipeline] Loaded {len(df)} records from {data_file} (version: {version_label})")
    else:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=[f"feature_{i+1}" for i in range(iris.data.shape[1])])
        df["target"] = (iris.target == 0).astype(int)  # Binary classification
        version_label = "iris-fallback"
        print(f"[pipeline] {data_file} not found; fallback to synthetic binary iris dataset ({len(df)} records)")

    # 2. Validate data
    validate_data(df)

    # 3. Preprocess
    random_state = int(hyperparams.get("random_state", 42))
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df, test_size=0.2, random_state=random_state
    )

    # 4. Train model
    clf = train_model(X_train, y_train, model_type=model_type, **hyperparams)

    # 5. Evaluate
    metrics = evaluate_model(clf, X_test, y_test, X_train=X_train, y_train=y_train)

    # 6. Quality Gate
    passed_gate, gate_reason = check_quality_gate(metrics, min_f1=min_f1)

    # 7. Save local artifacts
    model_out = Path(output_model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)

    metrics_out = Path(output_metrics_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 8. Provenance Metadata
    git_commit = get_git_commit()
    dvc_md5 = get_dvc_md5(data_path)

    # 9. MLflow Tracking
    import mlflow
    import mlflow.sklearn

    tracking_uri = resolve_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"{model_type}_s{hyperparams.get('n_estimators', hyperparams.get('c_param', 'def'))}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log parameters
        logged_params = {
            "model_type": model_type,
            "dataset_path": str(data_path),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": len(feature_names),
            **{k: v for k, v in hyperparams.items() if v is not None},
        }
        mlflow.log_params(logged_params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log provenance and quality tags
        tags = {
            "dataset_version": version_label,
            "dvc_md5": dvc_md5,
            "git_commit": git_commit,
            "quality_gate_passed": "true" if passed_gate else "false",
            "quality_gate_reason": gate_reason,
            "model_type": model_type,
        }
        mlflow.set_tags(tags)

        # Log model artifact & register if passed quality gate
        reg_model_name = registered_model_name if passed_gate else None
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name=reg_model_name,
        )

    print(f"\n[pipeline] Run completed: {run_id}")
    print(f"  Model Type:     {model_type}")
    print(f"  Metrics:        {metrics}")
    print(f"  Quality Gate:   {'PASSED' if passed_gate else 'FAILED'} ({gate_reason})")
    print(f"  Provenance:     dataset={version_label}, git={git_commit}, dvc_md5={dvc_md5[:8] if dvc_md5 else 'N/A'}")
    print(f"  MLflow URI:     {tracking_uri}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "passed_gate": passed_gate,
        "gate_reason": gate_reason,
        "model_type": model_type,
        "tracking_uri": tracking_uri,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classification model with MLflow & DVC tracking")
    parser.add_argument("--data", type=str, default="data/dataset.csv")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "gradient_boosting", "logistic_regression"])
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--c-param", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-f1", type=float, default=0.75)
    parser.add_argument("--experiment-name", type=str, default="iris-classification-pipeline")
    parser.add_argument("--registered-model-name", type=str, default="iris-classifier")
    parser.add_argument("--output", type=str, default="models/model.joblib")
    parser.add_argument("--metrics", type=str, default="metrics.json")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        model_type=args.model_type,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        min_f1=args.min_f1,
        output_model_path=args.output,
        output_metrics_path=args.metrics,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        c_param=args.c_param,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
