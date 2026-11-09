"""Airflow REST API Tools for MLOps AI Agent (Task 3 — 15 Marks).

Provides authenticated HTTP integration with Airflow 3 / 2 REST API.
Enforces strict DAG authorization allowlists, tracks DAG execution states,
identifies failed task instances, and extracts diagnostics and error logs.
Every call returns a standardized ToolExecutionResult and logs structured JSON.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from agent.logger import get_logger, log_tool_execution
from agent.schemas import (
    AirflowGetDagRunStatusOutput,
    AirflowIdentifyFailedTasksOutput,
    AirflowListDagsOutput,
    AirflowTaskDiagnosticsOutput,
    AirflowTriggerDagOutput,
    ToolExecutionResult,
    ToolExecutionStatus,
)

logger = get_logger("airflow_tools")

# ------------------------------------------------------------------------------
# Security: Strict DAG Authorization Allowlist
# ------------------------------------------------------------------------------

APPROVED_DAGS = {
    "mlops_model_training_pipeline",
}


def get_airflow_client_config() -> dict[str, Any]:
    """Retrieve and normalize Airflow REST API configuration from environment."""
    base_url = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080/api/v2").rstrip("/")
    token = os.getenv("AIRFLOW_API_TOKEN")
    username = os.getenv("AIRFLOW_USERNAME", "admin")
    password = os.getenv("AIRFLOW_PASSWORD", "admin")
    timeout = float(os.getenv("AIRFLOW_TIMEOUT_SECONDS", "5.0"))

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    auth: tuple[str, str] | None = None

    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username and password:
        auth = (username, password)

    return {
        "base_url": base_url,
        "headers": headers,
        "auth": auth,
        "timeout": timeout,
    }


def make_airflow_request(
    method: str,
    endpoint: str,
    json_data: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
) -> tuple[int, Any]:
    """Execute authenticated HTTP request against Airflow REST API with strict timeouts.

    Returns:
        tuple[status_code, response_data]
    """
    cfg = get_airflow_client_config()
    url = f"{cfg['base_url']}/{endpoint.lstrip('/')}"
    requester = session or requests

    resp = requester.request(
        method=method.upper(),
        url=url,
        headers=cfg["headers"],
        auth=cfg["auth"],
        json=json_data,
        params=params,
        timeout=cfg["timeout"],
    )

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    return resp.status_code, data


# ------------------------------------------------------------------------------
# Task 3 Tool 1: list_available_dags
# ------------------------------------------------------------------------------

def list_available_dags(
    only_active: bool = False,
    session: requests.Session | None = None,
    request_id: str = "req_airflow_list_dags",
) -> ToolExecutionResult:
    """Query /dags to list available Airflow DAGs, their pause status, and schedules."""
    start_time = time.perf_counter()
    tool_name = "list_available_dags"

    try:
        params = {"only_active": str(only_active).lower()} if only_active else {}
        status_code, response = make_airflow_request("GET", "dags", params=params, session=session)

        if status_code != 200:
            raise RuntimeError(f"Airflow API error (status {status_code}): {response}")

        raw_dags = response.get("dags", []) if isinstance(response, dict) else []
        dag_items: list[dict[str, Any]] = []

        for d in raw_dags:
            dag_items.append({
                "dag_id": d.get("dag_id"),
                "is_paused": d.get("is_paused"),
                "is_active": d.get("is_active", True),
                "description": d.get("description") or "",
                "timetable_summary": d.get("timetable_summary") or d.get("schedule_interval") or "None",
                "is_approved_for_agent": d.get("dag_id") in APPROVED_DAGS,
            })

        output = AirflowListDagsOutput(
            total_entries=len(dag_items),
            dags=dag_items,
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Listed {len(dag_items)} DAGs from Airflow API.",
            result.data,
        )
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=f"Airflow connection/execution failed: {exc}",
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed listing DAGs", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 3 Tool 2: get_dag_run_status
# ------------------------------------------------------------------------------

def get_dag_run_status(
    dag_id: str,
    dag_run_id: str | None = None,
    session: requests.Session | None = None,
    request_id: str = "req_airflow_dag_run_status",
) -> ToolExecutionResult:
    """Query /dags/{dag_id}/dagRuns to inspect execution state and timestamps."""
    start_time = time.perf_counter()
    tool_name = "get_dag_run_status"

    try:
        if dag_run_id:
            endpoint = f"dags/{dag_id}/dagRuns/{dag_run_id}"
            status_code, response = make_airflow_request("GET", endpoint, session=session)
            if status_code != 200:
                raise RuntimeError(f"Airflow API error (status {status_code}): {response}")

            run_state = response.get("state", "unknown")
            output = AirflowGetDagRunStatusOutput(
                dag_id=dag_id,
                dag_run_id=dag_run_id,
                state=run_state,
                start_date=response.get("start_date"),
                end_date=response.get("end_date"),
                recent_runs=[response],
            )
        else:
            endpoint = f"dags/{dag_id}/dagRuns"
            status_code, response = make_airflow_request("GET", endpoint, session=session)
            if status_code != 200:
                raise RuntimeError(f"Airflow API error (status {status_code}): {response}")

            raw_runs = response.get("dag_runs", []) if isinstance(response, dict) else []
            if not raw_runs:
                output = AirflowGetDagRunStatusOutput(
                    dag_id=dag_id,
                    dag_run_id="none",
                    state="none",
                    start_date=None,
                    end_date=None,
                    recent_runs=[],
                )
            else:
                # Latest run
                latest = raw_runs[0]
                output = AirflowGetDagRunStatusOutput(
                    dag_id=dag_id,
                    dag_run_id=latest.get("dag_run_id", "unknown"),
                    state=latest.get("state", "unknown"),
                    start_date=latest.get("start_date"),
                    end_date=latest.get("end_date"),
                    recent_runs=raw_runs[:5],
                )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"DAG '{dag_id}' status: {output.state} (run: {output.dag_run_id})",
            result.data,
        )
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=f"Failed retrieving DAG run status: {exc}",
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed getting DAG run status", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 3 Tool 3: trigger_training_dag
# ------------------------------------------------------------------------------

def trigger_training_dag(
    dag_id: str = "mlops_model_training_pipeline",
    conf: dict[str, Any] | None = None,
    session: requests.Session | None = None,
    request_id: str = "req_airflow_trigger_dag",
) -> ToolExecutionResult:
    """Verify DAG is in APPROVED_DAGS allowlist and trigger new run via POST /dags/{dag_id}/dagRuns."""
    start_time = time.perf_counter()
    tool_name = "trigger_training_dag"

    # Guardrail: Check allowlist
    if dag_id not in APPROVED_DAGS:
        duration = (time.perf_counter() - start_time) * 1000.0
        msg = f"Triggering blocked: DAG '{dag_id}' is not in approved allowlist {sorted(APPROVED_DAGS)}."
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.BLOCKED,
            duration_ms=duration,
            error=msg,
        )
        log_tool_execution(logger, request_id, tool_name, "BLOCKED", duration, msg, error=msg)
        return result

    try:
        payload = {
            "conf": conf or {},
        }
        endpoint = f"dags/{dag_id}/dagRuns"
        status_code, response = make_airflow_request("POST", endpoint, json_data=payload, session=session)

        if status_code not in (200, 201):
            raise RuntimeError(f"Airflow API error (status {status_code}): {response}")

        output = AirflowTriggerDagOutput(
            dag_id=dag_id,
            dag_run_id=response.get("dag_run_id", f"manual__{int(time.time())}"),
            state=response.get("state", "queued"),
            logical_date=response.get("logical_date") or response.get("execution_date"),
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Triggered DAG '{dag_id}' (run_id: {output.dag_run_id}, state: {output.state})",
            result.data,
        )
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=f"Failed triggering DAG '{dag_id}': {exc}",
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed triggering DAG", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 3 Tool 4: identify_failed_tasks
# ------------------------------------------------------------------------------

def identify_failed_tasks(
    dag_id: str,
    dag_run_id: str,
    session: requests.Session | None = None,
    request_id: str = "req_airflow_failed_tasks",
) -> ToolExecutionResult:
    """Query /taskInstances to identify tasks in 'failed' or 'upstream_failed' states."""
    start_time = time.perf_counter()
    tool_name = "identify_failed_tasks"

    try:
        endpoint = f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
        status_code, response = make_airflow_request("GET", endpoint, session=session)

        if status_code != 200:
            raise RuntimeError(f"Airflow API error (status {status_code}): {response}")

        all_instances = response.get("task_instances", []) if isinstance(response, dict) else []
        failed_tasks: list[dict[str, Any]] = []

        for ti in all_instances:
            state = str(ti.get("state", "")).lower()
            if state in ("failed", "upstream_failed", "shutdown"):
                failed_tasks.append({
                    "task_id": ti.get("task_id"),
                    "state": state,
                    "try_number": ti.get("try_number", 1),
                    "start_date": ti.get("start_date"),
                    "end_date": ti.get("end_date"),
                    "duration": ti.get("duration"),
                })

        summary = (
            f"All {len(all_instances)} task instances succeeded or are executing normally."
            if not failed_tasks
            else f"Identified {len(failed_tasks)} failed or upstream_failed tasks in DAG run '{dag_run_id}': "
                 + ", ".join([f"{t['task_id']} ({t['state']})" for t in failed_tasks])
        )

        output = AirflowIdentifyFailedTasksOutput(
            dag_id=dag_id,
            dag_run_id=dag_run_id,
            failed_tasks=failed_tasks,
            total_failed=len(failed_tasks),
            summary=summary,
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Found {len(failed_tasks)} failed tasks for DAG '{dag_id}'.",
            result.data,
        )
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=f"Failed querying task instances: {exc}",
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed identifying failed tasks", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 3 Tool 5: get_task_diagnostics
# ------------------------------------------------------------------------------

def get_task_diagnostics(
    dag_id: str,
    dag_run_id: str,
    task_id: str,
    try_number: int = 1,
    session: requests.Session | None = None,
    request_id: str = "req_airflow_task_diagnostics",
) -> ToolExecutionResult:
    """Retrieve task execution metadata and error logs to generate actionable remediation steps."""
    start_time = time.perf_counter()
    tool_name = "get_task_diagnostics"

    try:
        endpoint = f"dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}"
        status_code, response = make_airflow_request("GET", endpoint, session=session)

        # Parse logs
        if isinstance(response, dict):
            raw_logs = response.get("content", str(response))
        else:
            raw_logs = str(response)

        # Diagnose error patterns
        recommendations: list[str] = []
        error_snippet: str | None = None

        lower_logs = raw_logs.lower()
        if "valueerror" in lower_logs or "validation failed" in lower_logs:
            error_snippet = "Dataset validation error"
            recommendations.append("Check input dataset schema and missingness: ensure 'data/dataset.csv' has valid target column and null ratio <= 0.05.")
            recommendations.append("Verify DVC tracking status using agent tool `dvc_status`.")
        elif "quality gate rejected" in lower_logs or "runtimeerror" in lower_logs:
            error_snippet = "Quality gate rejection"
            recommendations.append("Candidate model F1 or accuracy is below 0.75 threshold.")
            recommendations.append("Tune model hyperparameters (e.g. increase n_estimators or adjust max_depth) and rerun training pipeline.")
        elif "connection refused" in lower_logs or "http" in lower_logs:
            error_snippet = "Downstream service connectivity issue"
            recommendations.append("Check MLflow tracking server status or verify MLFLOW_TRACKING_URI configuration.")
        else:
            recommendations.append("Inspect task stacktrace and verify dependencies and environment variables.")

        output = AirflowTaskDiagnosticsOutput(
            dag_id=dag_id,
            dag_run_id=dag_run_id,
            task_id=task_id,
            try_number=try_number,
            state="failed" if error_snippet else "inspected",
            error_message=error_snippet,
            logs=raw_logs[-2000:] if len(raw_logs) > 2000 else raw_logs,
            recommendations=recommendations,
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Diagnosed task '{task_id}' (attempt {try_number})",
            result.data,
        )
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=f"Failed getting task diagnostics for '{task_id}': {exc}",
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed getting task diagnostics", error=str(exc))
        return result
