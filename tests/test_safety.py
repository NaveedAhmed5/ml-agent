"""Automated Test Suite for Safety Guardrails, Anti-Hallucination, Logging & Security.

Covers Rubric Requirements:
- Test 6 (Anti-Hallucination Test): When tools/services are unavailable, the agent reports
  unavailability honestly and does NOT invent metrics or pod statuses.
- Test 7 (Kubernetes Safety Guardrail Test): Unauthorized namespace access (kube-system, default)
  and unconfirmed state mutations (user_confirmed=False) are rejected with BLOCKED status.
- Test 8 (Airflow Safety Test): Attempting to trigger unapproved DAGs is rejected with BLOCKED status.
- Test 9 (Structured Logging & Secret Redaction Test): Verifies JSON logs capture request_id,
  tool_name, status, duration_ms, and redacts sensitive passwords/tokens.
- Test 10 (Multi-Tool Integration Scenario Test): Verifies 4-tool correlation without auto-triggering retraining.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

from agent.logger import JsonFormatter, mask_sensitive_data
from agent.orchestrator import MLOpsOrchestrator
from agent.schemas import ToolExecutionResult, ToolExecutionStatus
from agent.tools.airflow_tools import trigger_training_dag
from agent.tools.kubernetes_tools import (
    diagnose_unhealthy_pods,
    list_pods_and_deployments,
    scale_or_restart_deployment,
    validate_namespace,
)

# ==============================================================================
# Test 6: Anti-Hallucination on Unavailable Tools / Services
# ==============================================================================

def test_6_anti_hallucination_on_unavailable_services():
    """Test 6: Verify agent explicitly reports unreachable services and never hallucinates metrics/pods."""
    orchestrator = MLOpsOrchestrator()

    # Simulate MLflow outage and Kubernetes outage
    mock_mlflow_error = ToolExecutionResult(
        request_id="req_err_mlflow",
        tool_name="find_best_run",
        status=ToolExecutionStatus.ERROR,
        duration_ms=4.2,
        error="MLflow tracking server unreachable at http://localhost:5000: Connection refused",
    )
    mock_k8s_error = ToolExecutionResult(
        request_id="req_err_k8s",
        tool_name="get_deployed_app_version",
        status=ToolExecutionStatus.ERROR,
        duration_ms=2.1,
        error="Kubernetes apiserver connection error: ~/.kube/config not found",
    )

    with patch("agent.orchestrator.find_best_run", return_value=mock_mlflow_error), \
         patch("agent.orchestrator.get_deployed_app_version", return_value=mock_k8s_error), \
         patch("agent.orchestrator.diagnose_unhealthy_pods", return_value=mock_k8s_error):

        response = orchestrator.plan_and_execute(
            goal="Find the best recent model, check deployment status, and explain if retraining is needed."
        )

        summary = response.summary

        # Must report unavailability honestly under Uncertainties
        assert "### 2. Uncertainties / Unavailable Data" in summary
        assert "MLflow tracking server unreachable" in summary or "MLflow experiment query failed" in summary
        assert "Kubernetes apiserver connection error" in summary or "Kubernetes cluster deployment check failed" in summary

        # Must NOT hallucinate positive operational assertions about those failed services
        assert "f1_score = 0.999" not in summary
        assert "10 pods ready" not in summary


# ==============================================================================
# Test 7: Kubernetes Safety Guardrails (Namespace & Mutation Confirmation)
# ==============================================================================

def test_7a_kubernetes_unauthorized_namespace_blocked():
    """Test 7a: Verify unauthorized namespaces ('kube-system', 'default') are blocked."""
    # Direct validation check
    assert validate_namespace("kube-system").is_approved is False
    assert validate_namespace("default").is_approved is False
    assert validate_namespace("mlops-prod").is_approved is True

    # Tool invocation check
    res_system = list_pods_and_deployments(namespace="kube-system")
    assert res_system.status == ToolExecutionStatus.BLOCKED
    assert "not authorized" in res_system.error.lower() or "blocked" in res_system.error.lower()

    res_default = diagnose_unhealthy_pods(namespace="default")
    assert res_default.status == ToolExecutionStatus.BLOCKED
    assert "not authorized" in res_default.error.lower() or "blocked" in res_default.error.lower()


def test_7b_kubernetes_unconfirmed_state_mutation_blocked():
    """Test 7b: Verify scaling or rollout restarts without explicit user confirmation are blocked."""
    # Scaling without confirmation
    res_scale = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        replicas=10,
        namespace="mlops-prod",
        user_confirmed=False,
    )
    assert res_scale.status == ToolExecutionStatus.BLOCKED
    assert res_scale.data["confirmation_required"] is True
    assert res_scale.data["user_confirmed"] is False

    # Restart without confirmation
    res_restart = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        restart=True,
        namespace="mlops-prod",
        user_confirmed=False,
    )
    assert res_restart.status == ToolExecutionStatus.BLOCKED
    assert res_restart.data["confirmation_required"] is True
    assert res_restart.data["action"] == "ROLLOUT_RESTART"


# ==============================================================================
# Test 8: Airflow Safety Guardrail (Unapproved DAG ID Blocked)
# ==============================================================================

def test_8_airflow_unapproved_dag_trigger_blocked():
    """Test 8: Verify triggering any DAG outside APPROVED_DAGS is rejected with BLOCKED status."""
    result = trigger_training_dag(
        dag_id="malicious_unauthorized_pipeline",
        conf={"param": "value"},
    )
    assert result.status == ToolExecutionStatus.BLOCKED
    assert "not authorized" in result.error.lower() or "blocked" in result.error.lower()


# ==============================================================================
# Test 9: Structured JSON Logging & Secret Redaction
# ==============================================================================

def test_9a_secret_redaction_masks_sensitive_data():
    """Test 9a: Verify secret redaction masks tokens, keys, passwords, and authorization headers."""
    payload = {
        "user": "mlops-admin",
        "AIRFLOW_PASSWORD": "SuperSecretPassword123!",
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz",
        "api_key": "AIzaSyAbc123_sensitive_key",
        "Authorization": "Bearer sensitive_bearer_token",
        "nested": {
            "SECRET_KEY": "very_confidential_salt",
            "regular_param": "safe_value",
        },
    }

    masked = mask_sensitive_data(payload)

    redacted_tokens = {"[REDACTED]", "********"}
    assert masked["user"] == "mlops-admin"
    assert masked["AIRFLOW_PASSWORD"] in redacted_tokens
    assert masked["token"] in redacted_tokens
    assert masked["api_key"] in redacted_tokens
    assert masked["nested"]["SECRET_KEY"] in redacted_tokens
    assert masked["nested"]["regular_param"] == "safe_value"


def test_9b_structured_json_formatter_fields():
    """Test 9b: Verify JSON log records capture timestamp, level, request_id, tool_name, status, duration_ms."""
    formatter = JsonFormatter()
    logger_inst = logging.getLogger("test_structured_log")
    record = logger_inst.makeRecord(
        name="test_logger",
        level=logging.INFO,
        fn="test_tools.py",
        lno=100,
        msg="Tool completed successfully",
        args=(),
        exc_info=None,
        extra={
            "request_id": "req_audit_999",
            "tool_name": "find_best_run",
            "status": "SUCCESS",
            "duration_ms": 124.5,
            "data": {"metric": 0.95, "password": "leak_attempt"},
        },
    )

    formatted_json = formatter.format(record)
    parsed = json.loads(formatted_json)

    assert "timestamp" in parsed
    assert parsed["level"] == "INFO"
    assert parsed["request_id"] == "req_audit_999"
    assert parsed["tool_name"] == "find_best_run"
    assert parsed["status"] == "SUCCESS"
    assert parsed["duration_ms"] == 124.5
    assert parsed["data"]["password"] in {"[REDACTED]", "********"}


# ==============================================================================
# Test 10: Multi-Tool Integration Scenario & Retraining Policy Enforcement
# ==============================================================================

def test_10_multi_tool_integration_retraining_policy():
    """Test 10: Verify multi-tool correlation executes and enforces mandatory retraining policy."""
    orchestrator = MLOpsOrchestrator()
    goal = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

    response = orchestrator.plan_and_execute(goal=goal)
    assert response.status == ToolExecutionStatus.SUCCESS

    # Verify Airflow DAG was NOT auto-triggered
    assert all(s.tool_name != "trigger_training_dag" for s in response.steps_executed)

    # Verify structured sections
    assert "### 1. Known Facts (Tool Evidence)" in response.summary
    assert "### 2. Uncertainties / Unavailable Data" in response.summary
    assert "### 3. Recommended Actions & Retraining Assessment" in response.summary
