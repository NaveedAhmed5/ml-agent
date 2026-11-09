"""Apache Airflow DAG: mlops_model_training_pipeline.

Defines a 5-stage production MLOps training workflow in strict sequential dependency:
validation >> preprocessing >> training >> evaluation_mlflow >> quality_gate

Features:
- Configured default_args with retries=2 and retry_delay=timedelta(minutes=1).
- Strict data validation that fails early on schema mismatch, high null ratio, or missing data.
- Feature preprocessing and artifact partitioning.
- Model training integrated with scikit-learn.
- Comprehensive metric evaluation (F1, Accuracy, Precision, Recall, ROC-AUC) and MLflow tracking.
- Quality gate enforcement blocking models with F1 < 0.75 or Accuracy < 0.75 from registry promotion.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Configure task logger
logger = logging.getLogger("airflow.task.mlops_training_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


# ------------------------------------------------------------------------------
# Modular Task Callables (Executable independently or inside Airflow operators)
# ------------------------------------------------------------------------------

def task_validate_data(
    data_path: str = "data/dataset.csv",
    dvc_file_path: str = "data/dataset.csv.dvc",
    **context: Any,
) -> dict[str, Any]:
    """Stage 1: Verify dataset presence, row count, schema, nulls, and record DVC MD5 hash.

    Raises:
        ValueError: If data file is missing, empty, or fails schema validation.
    """
    logger.info("Executing Stage 1: Data Validation...")
    csv_file = Path(data_path)
    if not csv_file.exists():
        raise ValueError(f"Stage 1 Validation Failed: Dataset file '{data_path}' does not exist.")

    df = pd.read_csv(csv_file)
    total_rows = len(df)
    if total_rows < 50:
        raise ValueError(f"Stage 1 Validation Failed: Dataset contains only {total_rows} rows (minimum 50 required).")

    if "target" not in df.columns:
        raise ValueError("Stage 1 Validation Failed: Required column 'target' is missing.")

    null_count = int(df.isnull().sum().sum())
    total_cells = df.shape[0] * df.shape[1]
    null_ratio = null_count / total_cells if total_cells > 0 else 0.0

    if null_ratio > 0.05:
        raise ValueError(f"Stage 1 Validation Failed: Excessive null ratio {null_ratio:.4f} > 0.05.")

    # Record DVC hash
    dvc_md5 = "untracked"
    dvc_p = Path(dvc_file_path)
    if dvc_p.exists():
        try:
            with open(dvc_p, encoding="utf-8") as f:
                dvc_yaml = yaml.safe_load(f) or {}
            outs = dvc_yaml.get("outs", [])
            if outs:
                dvc_md5 = outs[0].get("md5", "unknown")
        except Exception as exc:
            logger.warning(f"Failed parsing .dvc file: {exc}")

    report = {
        "status": "VALIDATED",
        "total_rows": total_rows,
        "feature_count": len([c for c in df.columns if c not in ("target", "dataset_version", "generated_at")]),
        "null_count": null_count,
        "null_ratio": round(null_ratio, 4),
        "dvc_md5": dvc_md5,
        "dataset_version": str(df["dataset_version"].iloc[0]) if "dataset_version" in df.columns else "v1.0.0",
        "validated_at": datetime.now(UTC).isoformat(),
    }
    logger.info(f"Stage 1 Completed Successfully: {report}")
    return report


def task_preprocess_data(
    data_path: str = "data/dataset.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
    **context: Any,
) -> dict[str, Any]:
    """Stage 2: Split features/labels with stratification and save processed partitions."""
    logger.info("Executing Stage 2: Data Preprocessing...")
    csv_file = Path(data_path)
    df = pd.read_csv(csv_file)

    ignore_cols = {"target", "dataset_version", "generated_at"}
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    np.save(out_p / "X_train.npy", X_train)
    np.save(out_p / "X_test.npy", X_test)
    np.save(out_p / "y_train.npy", y_train)
    np.save(out_p / "y_test.npy", y_test)

    summary = {
        "status": "PREPROCESSED",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_names": feature_cols,
        "output_directory": str(out_p.resolve()),
    }
    logger.info(f"Stage 2 Completed Successfully: {summary}")
    return summary


def task_train_model(
    processed_dir: str = "data/processed",
    candidate_model_path: str = "models/candidate_model.joblib",
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
    **context: Any,
) -> dict[str, Any]:
    """Stage 3: Train candidate classifier and store model checkpoint."""
    logger.info("Executing Stage 3: Model Training...")
    proc_p = Path(processed_dir)
    X_train = np.load(proc_p / "X_train.npy")
    y_train = np.load(proc_p / "y_train.npy")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    out_file = Path(candidate_model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_file)

    summary = {
        "status": "TRAINED",
        "model_type": "RandomForestClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "model_path": str(out_file.resolve()),
    }
    logger.info(f"Stage 3 Completed Successfully: {summary}")
    return summary


def task_evaluate_and_log_mlflow(
    processed_dir: str = "data/processed",
    candidate_model_path: str = "models/candidate_model.joblib",
    experiment_name: str = "iris-classification-pipeline",
    **context: Any,
) -> dict[str, Any]:
    """Stage 4: Compute evaluation metrics and log run artifacts to MLflow."""
    logger.info("Executing Stage 4: Evaluation & MLflow Logging...")
    proc_p = Path(processed_dir)
    X_train = np.load(proc_p / "X_train.npy")
    y_train = np.load(proc_p / "y_train.npy")
    X_test = np.load(proc_p / "X_test.npy")
    y_test = np.load(proc_p / "y_test.npy")

    clf = joblib.load(candidate_model_path)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    roc_auc = 0.5
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)
        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, probs[:, 1])

    train_preds = clf.predict(X_train)
    train_f1 = f1_score(y_train, train_preds, average="weighted", zero_division=0)
    overfit_gap = max(0.0, float(train_f1 - f1))

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "train_f1_score": round(float(train_f1), 4),
        "overfitting_gap": round(float(overfit_gap), 4),
    }

    # Log to MLflow
    import mlflow
    import mlflow.sklearn

    from training.train import get_git_commit, resolve_mlflow_tracking_uri

    tracking_uri = resolve_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="airflow_pipeline_run") as run:
        run_id = run.info.run_id
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "orchestrator": "Apache Airflow",
            "pipeline_dag": "mlops_model_training_pipeline",
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "git_commit": get_git_commit(),
            "airflow_stage": "evaluate_and_log_mlflow",
        })
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    # Cache metrics locally for downstream quality gate
    metrics_cache = Path("data/processed/latest_evaluation.json")
    with open(metrics_cache, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "run_id": run_id}, f, indent=2)

    summary = {
        "status": "EVALUATED",
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "tracking_uri": tracking_uri,
    }
    logger.info(f"Stage 4 Completed Successfully: {summary}")
    return summary


def task_quality_gate(
    evaluation_cache: str = "data/processed/latest_evaluation.json",
    min_f1: float = 0.75,
    min_accuracy: float = 0.75,
    registered_model_name: str = "iris-classifier",
    **context: Any,
) -> dict[str, Any]:
    """Stage 5: Assert production quality criteria. Block on failure, promote to Model Registry on success.

    Raises:
        RuntimeError: If candidate model fails minimum production metrics.
    """
    logger.info("Executing Stage 5: Quality Gate Enforcement...")
    cache_p = Path(evaluation_cache)
    if not cache_p.exists():
        raise RuntimeError("Stage 5 Quality Gate Failed: Evaluation cache not found.")

    with open(cache_p, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    run_id = data.get("run_id")
    f1 = metrics.get("f1_score", 0.0)
    acc = metrics.get("accuracy", 0.0)

    failures: list[str] = []
    if f1 < min_f1:
        failures.append(f"F1 score {f1:.4f} < {min_f1:.4f}")
    if acc < min_accuracy:
        failures.append(f"Accuracy {acc:.4f} < {min_accuracy:.4f}")

    if failures:
        reason = f"Quality gate rejected candidate model: {', '.join(failures)}."
        logger.error(reason)
        raise RuntimeError(reason)

    # Quality Gate Passed: Promote / Register model in MLflow
    from mlflow.tracking import MlflowClient

    from training.train import resolve_mlflow_tracking_uri

    client = MlflowClient(tracking_uri=resolve_mlflow_tracking_uri())
    model_uri = f"runs:/{run_id}/model"
    model_version = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Promoted by Airflow DAG mlops_model_training_pipeline (F1: {f1:.4f}, Acc: {acc:.4f})",
    )

    decision = {
        "status": "QUALITY_GATE_PASSED",
        "decision": "PROMOTED_TO_REGISTRY",
        "registered_model": registered_model_name,
        "model_version": model_version.version,
        "metrics_achieved": {"f1_score": f1, "accuracy": acc},
        "promoted_at": datetime.now(UTC).isoformat(),
    }
    logger.info(f"Stage 5 Completed Successfully: {decision}")
    return decision


# ------------------------------------------------------------------------------
# Airflow DAG Definition
# ------------------------------------------------------------------------------

default_args = {
    "owner": "mlops-agent",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

# Construct DAG conditionally to allow clean standalone Python testing
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    with DAG(
        dag_id="mlops_model_training_pipeline",
        default_args=default_args,
        description="Airflow production pipeline: validation >> preprocessing >> training >> evaluation >> quality_gate",
        schedule_interval=None,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "production", "agent"],
    ) as dag:

        validation = PythonOperator(
            task_id="validate_data",
            python_callable=task_validate_data,
        )

        preprocessing = PythonOperator(
            task_id="preprocess_data",
            python_callable=task_preprocess_data,
        )

        training = PythonOperator(
            task_id="train_model",
            python_callable=task_train_model,
        )

        evaluation_mlflow = PythonOperator(
            task_id="evaluate_and_log_mlflow",
            python_callable=task_evaluate_and_log_mlflow,
        )

        quality_gate = PythonOperator(
            task_id="quality_gate",
            python_callable=task_quality_gate,
        )

        # Strict sequential dependencies
        validation >> preprocessing >> training >> evaluation_mlflow >> quality_gate

except ImportError:
    # Airflow not installed in local environment; task callables remain fully operational
    dag = None
