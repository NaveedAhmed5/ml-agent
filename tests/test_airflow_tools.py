"""Unit tests for Airflow Training DAG and Workflow Agent Tools (Task 3 — 15 Marks)."""

import pytest

from agent.schemas import ToolExecutionStatus
from agent.tools.airflow_tools import (
    APPROVED_DAGS,
    get_dag_run_status,
    get_task_diagnostics,
    identify_failed_tasks,
    list_available_dags,
    trigger_training_dag,
)
from dags.training_pipeline import (
    task_evaluate_and_log_mlflow,
    task_preprocess_data,
    task_quality_gate,
    task_train_model,
    task_validate_data,
)

# ------------------------------------------------------------------------------
# 1. Tests for DAG Task Callables (Standalone Verification)
# ------------------------------------------------------------------------------

def test_dag_tasks_end_to_end(tmp_path):
    """Verify that all 5 DAG stages execute successfully in order."""
    # Stage 1: Validation
    res_val = task_validate_data(data_path="data/dataset.csv")
    assert res_val["status"] == "VALIDATED"
    assert res_val["total_rows"] >= 50

    # Stage 2: Preprocessing
    proc_dir = str(tmp_path / "processed")
    res_prep = task_preprocess_data(data_path="data/dataset.csv", output_dir=proc_dir)
    assert res_prep["status"] == "PREPROCESSED"
    assert res_prep["train_samples"] > 0

    # Stage 3: Training
    model_file = str(tmp_path / "candidate.joblib")
    res_train = task_train_model(processed_dir=proc_dir, candidate_model_path=model_file)
    assert res_train["status"] == "TRAINED"

    # Stage 4: Evaluation & MLflow
    res_eval = task_evaluate_and_log_mlflow(
        processed_dir=proc_dir,
        candidate_model_path=model_file,
    )
    assert res_eval["status"] == "EVALUATED"
    assert "f1_score" in res_eval["metrics"]

    # Stage 5: Quality Gate
    res_gate = task_quality_gate(min_f1=0.70, min_accuracy=0.70)
    assert res_gate["status"] == "QUALITY_GATE_PASSED"
    assert res_gate["decision"] == "PROMOTED_TO_REGISTRY"


def test_dag_validation_stage_failure(tmp_path):
    """Verify Stage 1 raises ValueError when dataset is missing or corrupt."""
    non_existent = str(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="does not exist"):
        task_validate_data(data_path=non_existent)


def test_dag_quality_gate_failure(tmp_path):
    """Verify Stage 5 raises RuntimeError when quality gate metric threshold fails."""
    # Write synthetic evaluation cache with sub-threshold F1
    cache_file = tmp_path / "eval_cache.json"
    cache_file.write_text('{"metrics": {"f1_score": 0.40, "accuracy": 0.50}, "run_id": "test_run"}')

    with pytest.raises(RuntimeError, match="Quality gate rejected candidate model"):
        task_quality_gate(evaluation_cache=str(cache_file), min_f1=0.75, min_accuracy=0.75)


# ------------------------------------------------------------------------------
# 2. Tests for Airflow Tools: Security & Guardrails
# ------------------------------------------------------------------------------

def test_trigger_unauthorized_dag_blocked():
    """Verify that triggering an unapproved DAG is immediately BLOCKED by guardrails."""
    res = trigger_training_dag("malicious_or_unapproved_dag")
    assert res.status == ToolExecutionStatus.BLOCKED
    assert "blocked" in str(res.error).lower()
    assert "malicious_or_unapproved_dag" in str(res.error)


def test_approved_dag_allowlist():
    """Verify that mlops_model_training_pipeline is in the approved allowlist."""
    assert "mlops_model_training_pipeline" in APPROVED_DAGS


# ------------------------------------------------------------------------------
# 3. Tests for Airflow Tools: Graceful Handling of Unreachable Service
# ------------------------------------------------------------------------------

def test_unreachable_airflow_service_error_handling(monkeypatch):
    """Verify that when Airflow service is offline, tools return ERROR without crashing."""
    # Point to a guaranteed unreachable port
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://127.0.0.1:59999/api/v2")
    monkeypatch.setenv("AIRFLOW_TIMEOUT_SECONDS", "0.5")

    res_list = list_available_dags()
    assert res_list.status == ToolExecutionStatus.ERROR
    assert "Airflow" in str(res_list.error)

    res_status = get_dag_run_status("mlops_model_training_pipeline")
    assert res_status.status == ToolExecutionStatus.ERROR

    res_trigger = trigger_training_dag("mlops_model_training_pipeline")
    assert res_trigger.status == ToolExecutionStatus.ERROR

    res_failed = identify_failed_tasks("mlops_model_training_pipeline", "run_123")
    assert res_failed.status == ToolExecutionStatus.ERROR

    res_diag = get_task_diagnostics("mlops_model_training_pipeline", "run_123", "validate_data")
    assert res_diag.status == ToolExecutionStatus.ERROR


# ------------------------------------------------------------------------------
# 4. Tests for Airflow Tools: Mocked REST API Responses
# ------------------------------------------------------------------------------

class MockResponse:
    def __init__(self, json_data, status_code=200, text=""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text or str(json_data)

    def json(self):
        return self._json_data


class MockSession:
    def __init__(self, routes):
        self.routes = routes

    def request(self, method, url, **kwargs):
        for pattern, response in self.routes.items():
            if pattern in url:
                return response
        return MockResponse({"detail": "Not Found"}, status_code=404)


def test_list_available_dags_mocked():
    """Verify list_available_dags parses API responses correctly."""
    routes = {
        "dags": MockResponse({
            "dags": [
                {
                    "dag_id": "mlops_model_training_pipeline",
                    "is_paused": False,
                    "is_active": True,
                    "description": "MLOps DAG",
                    "schedule_interval": "@daily",
                },
                {
                    "dag_id": "other_dag",
                    "is_paused": True,
                    "is_active": False,
                    "description": "Other DAG",
                    "schedule_interval": None,
                },
            ],
            "total_entries": 2,
        })
    }
    session = MockSession(routes)
    res = list_available_dags(session=session)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data["total_entries"] == 2
    dags = res.data["dags"]
    assert dags[0]["dag_id"] == "mlops_model_training_pipeline"
    assert dags[0]["is_approved_for_agent"] is True
    assert dags[1]["is_approved_for_agent"] is False


def test_get_dag_run_status_mocked():
    """Verify get_dag_run_status retrieves execution states."""
    routes = {
        "dags/mlops_model_training_pipeline/dagRuns/manual__123": MockResponse({
            "dag_id": "mlops_model_training_pipeline",
            "dag_run_id": "manual__123",
            "state": "success",
            "start_date": "2026-11-09T03:00:00Z",
            "end_date": "2026-11-09T03:02:15Z",
        })
    }
    session = MockSession(routes)
    res = get_dag_run_status("mlops_model_training_pipeline", dag_run_id="manual__123", session=session)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data["state"] == "success"
    assert res.data["dag_run_id"] == "manual__123"


def test_trigger_training_dag_mocked():
    """Verify trigger_training_dag posts to Airflow API."""
    routes = {
        "dags/mlops_model_training_pipeline/dagRuns": MockResponse({
            "dag_id": "mlops_model_training_pipeline",
            "dag_run_id": "manual__999",
            "state": "queued",
            "logical_date": "2026-11-09T03:10:00Z",
        }, status_code=201)
    }
    session = MockSession(routes)
    res = trigger_training_dag("mlops_model_training_pipeline", conf={"learning_rate": 0.05}, session=session)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data["state"] == "queued"
    assert res.data["dag_run_id"] == "manual__999"


def test_identify_failed_tasks_mocked():
    """Verify identify_failed_tasks filters failed tasks."""
    routes = {
        "taskInstances": MockResponse({
            "task_instances": [
                {"task_id": "validate_data", "state": "success", "duration": 1.2},
                {"task_id": "preprocess_data", "state": "success", "duration": 2.1},
                {"task_id": "train_model", "state": "failed", "try_number": 2, "duration": 5.4},
                {"task_id": "evaluate_and_log_mlflow", "state": "upstream_failed"},
            ]
        })
    }
    session = MockSession(routes)
    res = identify_failed_tasks("mlops_model_training_pipeline", "run_fail", session=session)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data["total_failed"] == 2
    failed_ids = [t["task_id"] for t in res.data["failed_tasks"]]
    assert "train_model" in failed_ids
    assert "evaluate_and_log_mlflow" in failed_ids


def test_get_task_diagnostics_mocked():
    """Verify get_task_diagnostics extracts error patterns and offers remediation."""
    log_content = """
    [2026-11-09 03:00:10] INFO Starting task
    [2026-11-09 03:00:12] ERROR ValueError: Stage 1 Validation Failed: Dataset file 'data/dataset.csv' does not exist.
    Traceback (most recent call last):
      File "training_pipeline.py", line 45, in task_validate_data
        raise ValueError("Dataset file does not exist")
    """
    routes = {
        "logs": MockResponse({"content": log_content})
    }
    session = MockSession(routes)
    res = get_task_diagnostics("mlops_model_training_pipeline", "run_fail", "validate_data", session=session)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data["error_message"] == "Dataset validation error"
    assert len(res.data["recommendations"]) >= 1
    assert any("data/dataset.csv" in rec for rec in res.data["recommendations"])

