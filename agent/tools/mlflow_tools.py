"""MLflow Tools for MLOps AI Agent.

Handles experiment tracking, metric and artifact logging, model registration,
and stage transitions.
"""

from __future__ import annotations

import time
from agent.logger import get_logger, log_tool_execution
from agent.schemas import (
    MLflowCreateExperimentInput,
    MLflowCreateExperimentOutput,
    MLflowGetLatestModelInput,
    MLflowGetLatestModelOutput,
    MLflowGetRunInput,
    MLflowGetRunOutput,
    MLflowLogRunInput,
    MLflowLogRunOutput,
    MLflowRegisterModelInput,
    MLflowRegisterModelOutput,
    MLflowTransitionStageInput,
    MLflowTransitionStageOutput,
    ToolExecutionResult,
    ToolExecutionStatus,
)

logger = get_logger("mlflow_tools")


def get_latest_model(params: MLflowGetLatestModelInput, request_id: str = "req_mlflow_latest") -> ToolExecutionResult:
    """Fetch latest registered model version for specified stage."""
    start_time = time.perf_counter()
    tool_name = "mlflow_get_latest_model"
    try:
        # Stub logic
        output = MLflowGetLatestModelOutput(
            model_name=params.model_name,
            version="1",
            current_stage="Production",
            source=f"models:/{params.model_name}/Production",
        )
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, "Latest model retrieved", result.data)
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "MLflow lookup failed", error=str(exc))
        return result

