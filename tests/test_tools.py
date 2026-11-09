"""Automated Test Suite for Tool Execution, Reliability, and Error Handling.

Covers Rubric Requirements:
- Tests 1-4: Successful tool execution across DVC, MLflow, Airflow, and Kubernetes (with mocks for external servers).
- Test 5: Tool timeouts, invalid requests, and HTTP 500 / connection refused service failures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from agent.schemas import ToolExecutionStatus
from agent.tools.airflow_tools import (
    get_dag_run_status,
    list_available_dags,
)
from agent.tools.dvc_tools import (
    list_dvc_tracked_datasets,
)
from agent.tools.kubernetes_tools import (
    get_deployed_app_version,
    list_pods_and_deployments,
    scale_or_restart_deployment,
    set_k8s_clients,
)
from agent.tools.mlflow_tools import (
    find_best_run,
)

# ==============================================================================
# Tests 1-4: Successful Tool Execution Across 4 Domains
# ==============================================================================

def test_1_dvc_tool_execution():
    """Test 1: Verify successful DVC tool execution (dataset tracking)."""
    result = list_dvc_tracked_datasets()
    assert result.status == ToolExecutionStatus.SUCCESS
    assert "tracked_datasets" in result.data
    assert result.duration_ms >= 0


def test_2_mlflow_tool_execution():
    """Test 2: Verify successful MLflow tool execution (experiment listing and best run)."""
    result = find_best_run(experiment_name="iris-classification-pipeline")
    assert result.status == ToolExecutionStatus.SUCCESS
    assert "best_run_id" in result.data
    assert "metric_value" in result.data
    assert result.data["target_metric"] == "f1_score"


def test_3_airflow_tool_execution_mocked():
    """Test 3: Verify successful Airflow tool execution via mocked REST API."""
    mock_payload = {
        "dags": [
            {
                "dag_id": "mlops_model_training_pipeline",
                "is_paused": False,
                "description": "Production model training pipeline",
            }
        ],
        "total_entries": 1,
    }

    with patch("agent.tools.airflow_tools.make_airflow_request", return_value=(200, mock_payload)):
        result = list_available_dags()
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.data["total_entries"] == 1
        assert result.data["dags"][0]["dag_id"] == "mlops_model_training_pipeline"


def test_4_kubernetes_tool_execution_mocked():
    """Test 4: Verify successful Kubernetes tool execution via mock API clients."""
    mock_core = MagicMock()
    mock_apps = MagicMock()

    mock_dep = MagicMock()
    mock_dep.metadata.name = "mlops-inference-deployment"
    mock_dep.spec.replicas = 2
    mock_dep.status.ready_replicas = 2
    mock_dep.status.available_replicas = 2
    mock_dep.status.updated_replicas = 2
    mock_dep.spec.template.spec.containers = [MagicMock(image="mlops-inference-api:1.2.0")]

    mock_pod = MagicMock()
    mock_pod.metadata.name = "mlops-pod-1"
    mock_pod.status.phase = "Running"
    mock_c = MagicMock(ready=True, restart_count=0)
    mock_pod.status.container_statuses = [mock_c]

    mock_apps.list_namespaced_deployment.return_value.items = [mock_dep]
    mock_core.list_namespaced_pod.return_value.items = [mock_pod]

    set_k8s_clients(core_api=mock_core, apps_api=mock_apps)
    try:
        result = list_pods_and_deployments(namespace="mlops-prod")
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.data["total_deployments"] == 1
        assert result.data["total_pods"] == 1
        assert result.data["deployments"][0]["name"] == "mlops-inference-deployment"
    finally:
        set_k8s_clients(None, None)


# ==============================================================================
# Test 5: Tool Timeouts, Invalid Requests, and Service Failures
# ==============================================================================

def test_5a_airflow_connection_refused_failure():
    """Test 5a: Verify Airflow tool handles connection refused gracefully without crashing."""
    with patch(
        "agent.tools.airflow_tools.make_airflow_request",
        side_effect=requests.exceptions.ConnectionError("Connection refused on port 8080"),
    ):
        result = list_available_dags()
        assert result.status == ToolExecutionStatus.ERROR
        assert "connection" in result.error.lower() or "offline" in result.error.lower()
        assert result.duration_ms >= 0


def test_5b_airflow_http_500_server_error():
    """Test 5b: Verify Airflow tool handles HTTP 500 internal server error cleanly."""
    with patch(
        "agent.tools.airflow_tools.make_airflow_request",
        side_effect=requests.exceptions.HTTPError("500 Server Error: Internal Server Error"),
    ):
        result = get_dag_run_status(dag_id="mlops_model_training_pipeline")
        assert result.status == ToolExecutionStatus.ERROR
        assert "500" in result.error or "error" in result.error.lower()


def test_5c_airflow_timeout_handling():
    """Test 5c: Verify Airflow tool handles request timeout cleanly."""
    with patch(
        "agent.tools.airflow_tools.make_airflow_request",
        side_effect=requests.exceptions.Timeout("Request timed out after 5.0s"),
    ):
        result = get_dag_run_status(dag_id="mlops_model_training_pipeline")
        assert result.status == ToolExecutionStatus.ERROR
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


def test_5d_kubernetes_invalid_state_change_request():
    """Test 5d: Verify Kubernetes scale tool rejects invalid request (neither replicas nor restart specified)."""
    result = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        replicas=None,
        restart=False,
        namespace="mlops-prod",
        user_confirmed=True,
    )
    assert result.status == ToolExecutionStatus.ERROR
    assert "either 'replicas'" in result.error.lower()


def test_5e_kubernetes_cluster_unreachable_exception():
    """Test 5e: Verify Kubernetes tools report cluster unreachable cleanly."""
    set_k8s_clients(None, None)
    with patch("kubernetes.config.load_incluster_config", side_effect=RuntimeError("No in-cluster token")), \
         patch("kubernetes.config.load_kube_config", side_effect=RuntimeError("Kubeconfig not found")):
        result = get_deployed_app_version(namespace="mlops-prod")
        assert result.status == ToolExecutionStatus.ERROR
        assert "unavailable" in result.error.lower() or "kubeconfig" in result.error.lower()
