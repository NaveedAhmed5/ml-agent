"""Airflow Tools for MLOps AI Agent.

Handles DAG triggering, DAG status monitoring, and DAG inventory retrieval
via Airflow REST API.
"""

from __future__ import annotations

import time
from agent.logger import get_logger, log_tool_execution
from agent.schemas import (
    AirflowGetDagRunStatusInput,
    AirflowGetDagRunStatusOutput,
    AirflowListDagsInput,
    AirflowListDagsOutput,
    AirflowTriggerDagInput,
    AirflowTriggerDagOutput,
    ToolExecutionResult,
    ToolExecutionStatus,
)

logger = get_logger("airflow_tools")


def trigger_dag(params: AirflowTriggerDagInput, request_id: str = "req_airflow_trigger") -> ToolExecutionResult:
    """Trigger a DAG run in Airflow."""
    start_time = time.perf_counter()
    tool_name = "airflow_trigger_dag"
    try:
        # Stub logic
        output = AirflowTriggerDagOutput(
            dag_id=params.dag_id,
            dag_run_id=params.dag_run_id or f"manual__{int(time.time())}",
            state="queued",
            logical_date=None,
        )
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, "DAG triggered", result.data)
        return result
    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=str(exc),
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed to trigger DAG", error=str(exc))
        return result

