"""Automated Test Suite for MLOps AI Agent Orchestrator & CLI.

Tests:
1. Tool registry completeness across DVC, MLflow, Airflow, and Kubernetes (21 tools).
2. End-to-end execution of the mandatory integration scenario.
3. Strict enforcement of the Mandatory Retraining Policy:
   - NEVER automatically triggers Airflow retraining when assessing retraining need.
   - Recommends NO retraining when model passes quality gates, dataset is clean, and cluster is healthy.
   - Recommends retraining when dataset drift or quality gate failure is detected.
4. Grounding and non-hallucination when services are unavailable.
5. Structured three-section output compliance.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from agent.orchestrator import MLOpsOrchestrator, ToolRegistry
from agent.schemas import ToolExecutionResult, ToolExecutionStatus


@pytest.fixture
def orchestrator():
    """Create orchestrator instance using local registry and rule router."""
    return MLOpsOrchestrator(model_name="test-model")


# ==============================================================================
# Tool Registry Tests
# ==============================================================================

def test_tool_registry_registration():
    """Verify all 21 required tools across 4 domains are registered with valid schemas."""
    registry = ToolRegistry()
    tools = registry.list_all()
    assert len(tools) == 21

    domains = set(t.domain for t in tools)
    assert domains == {"dvc", "mlflow", "airflow", "kubernetes"}

    expected_tool_names = [
        # DVC (5)
        "list_dvc_tracked_datasets",
        "get_current_dataset_version",
        "compare_dataset_versions",
        "detect_data_changes",
        "explain_dataset_reproduction",
        # MLflow (5)
        "list_experiments_and_runs",
        "get_run_details",
        "find_best_run",
        "compare_runs",
        "get_model_registry_info",
        # Airflow (5)
        "list_available_dags",
        "get_dag_run_status",
        "trigger_training_dag",
        "identify_failed_tasks",
        "get_task_diagnostics",
        # Kubernetes (6)
        "list_pods_and_deployments",
        "get_pod_status_and_logs",
        "inspect_services",
        "diagnose_unhealthy_pods",
        "get_deployed_app_version",
        "scale_or_restart_deployment",
    ]

    for name in expected_tool_names:
        defn = registry.get(name)
        assert defn is not None, f"Tool '{name}' was not found in registry."
        assert callable(defn.fn)
        assert "type" in defn.parameters_schema


# ==============================================================================
# Mandatory Integration Scenario Execution Tests
# ==============================================================================

def test_mandatory_integration_scenario_execution(orchestrator):
    """Verify orchestrator coordinates the multi-tool chain and structures output."""
    goal = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    response = orchestrator.plan_and_execute(goal=goal)

    assert response.status == ToolExecutionStatus.SUCCESS
    assert len(response.steps_executed) >= 4

    step_names = [s.tool_name for s in response.steps_executed]
    assert "find_best_run" in step_names
    assert "get_current_dataset_version" in step_names
    assert "detect_data_changes" in step_names

    # Verify structured sections
    summary = response.summary
    assert "### 1. Known Facts (Tool Evidence)" in summary
    assert "### 2. Uncertainties / Unavailable Data" in summary
    assert "### 3. Recommended Actions & Retraining Assessment" in summary


# ==============================================================================
# Mandatory Retraining Policy Verification
# ==============================================================================

def test_retraining_never_auto_triggered(orchestrator):
    """CRITICAL SAFETY TEST: Verify retraining is NEVER automatically triggered when assessing need."""
    goal = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    response = orchestrator.plan_and_execute(goal=goal)

    for step in response.steps_executed:
        assert step.tool_name != "trigger_training_dag", (
            "SAFETY VIOLATION: 'trigger_training_dag' was invoked automatically during an exploratory evaluation!"
        )

    assert "NOT automatically triggered" in response.summary or "NOT necessary" in response.summary


def test_retraining_recommended_on_dataset_changes():
    """Verify agent recommends retraining when dataset drift/changes are detected."""
    orchestrator = MLOpsOrchestrator()

    # Mock tool results
    mock_best_run = ToolExecutionResult(
        request_id="r1",
        tool_name="find_best_run",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={
            "best_run_id": "mock_run_123",
            "metric_value": 0.92,
            "target_metric": "f1_score",
            "is_production_ready": True,
        },
    )
    mock_dvc_ver = ToolExecutionResult(
        request_id="r2",
        tool_name="get_current_dataset_version",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={"md5_hash": "20dddf13", "dataset_version_label": "v1.0.0", "git_commit": "4e81700"},
    )
    mock_dvc_diff = ToolExecutionResult(
        request_id="r3",
        tool_name="detect_data_changes",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={"has_changes": True, "status_summary": "data/dataset.csv modified (150 new rows added)"},
    )

    with patch("agent.orchestrator.find_best_run", return_value=mock_best_run), \
         patch("agent.orchestrator.get_current_dataset_version", return_value=mock_dvc_ver), \
         patch("agent.orchestrator.detect_data_changes", return_value=mock_dvc_diff):

        response = orchestrator.plan_and_execute(
            goal="Find best model, check dataset version and deployment, explain if retraining is necessary."
        )

        assert "Retraining is RECOMMENDED" in response.summary
        assert "New/modified dataset detected" in response.summary
        assert "NOT automatically triggered" in response.summary


def test_no_retraining_when_model_passes_and_data_clean():
    """Verify agent recommends NO retraining when quality gate passes, data is unmodified, and deployment is healthy."""
    orchestrator = MLOpsOrchestrator()

    mock_best_run = ToolExecutionResult(
        request_id="r1",
        tool_name="find_best_run",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={
            "best_run_id": "mock_run_123",
            "metric_value": 0.92,
            "target_metric": "f1_score",
            "is_production_ready": True,
        },
    )
    mock_dvc_ver = ToolExecutionResult(
        request_id="r2",
        tool_name="get_current_dataset_version",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={"md5_hash": "20dddf13", "dataset_version_label": "v1.0.0", "git_commit": "4e81700"},
    )
    mock_dvc_diff = ToolExecutionResult(
        request_id="r3",
        tool_name="detect_data_changes",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={"has_changes": False, "status_summary": "No uncommitted data changes"},
    )
    mock_k8s_ver = ToolExecutionResult(
        request_id="r4",
        tool_name="get_deployed_app_version",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={"app_name": "mlops-inference-deployment", "app_version": "1.2.0", "model_version": "1.2.0", "container_image": "mlops-inference-api:1.2.0"},
    )
    mock_k8s_diag = ToolExecutionResult(
        request_id="r5",
        tool_name="diagnose_unhealthy_pods",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={"total_unhealthy": 0, "diagnosis_summary": "All pods in namespace 'mlops-prod' are healthy."},
    )
    mock_airflow = ToolExecutionResult(
        request_id="r6",
        tool_name="get_dag_run_status",
        status=ToolExecutionStatus.SUCCESS,
        duration_ms=10.0,
        data={"dag_id": "mlops_model_training_pipeline", "dag_run_id": "run_1", "state": "success"},
    )

    with patch("agent.orchestrator.find_best_run", return_value=mock_best_run), \
         patch("agent.orchestrator.get_current_dataset_version", return_value=mock_dvc_ver), \
         patch("agent.orchestrator.detect_data_changes", return_value=mock_dvc_diff), \
         patch("agent.orchestrator.get_deployed_app_version", return_value=mock_k8s_ver), \
         patch("agent.orchestrator.diagnose_unhealthy_pods", return_value=mock_k8s_diag), \
         patch("agent.orchestrator.get_dag_run_status", return_value=mock_airflow):

        response = orchestrator.plan_and_execute(
            goal="Find best model, check dataset version and deployment, explain if retraining is necessary."
        )

        assert "Retraining is NOT necessary at this time." in response.summary
        assert "Dataset has not changed" in response.summary
        assert "satisfies quality threshold" in response.summary


# ==============================================================================
# Grounding & Non-Hallucination on Unavailable Services Tests
# ==============================================================================

def test_grounding_reports_offline_services_without_hallucinating():
    """Verify that when Airflow and K8s are unreachable, errors are explicitly reported in Uncertainties."""
    orchestrator = MLOpsOrchestrator()

    mock_k8s_error = ToolExecutionResult(
        request_id="r4",
        tool_name="get_deployed_app_version",
        status=ToolExecutionStatus.ERROR,
        duration_ms=5.0,
        error="Cluster connection refused: Connection error to kube-apiserver",
    )
    mock_airflow_error = ToolExecutionResult(
        request_id="r6",
        tool_name="get_dag_run_status",
        status=ToolExecutionStatus.ERROR,
        duration_ms=5.0,
        error="Airflow REST API service offline: Connection refused on port 8080",
    )

    with patch("agent.orchestrator.get_deployed_app_version", return_value=mock_k8s_error), \
         patch("agent.orchestrator.diagnose_unhealthy_pods", return_value=mock_k8s_error), \
         patch("agent.orchestrator.get_dag_run_status", return_value=mock_airflow_error):

        response = orchestrator.plan_and_execute(
            goal="Find best model, check dataset version and deployment, explain if retraining is necessary."
        )

        summary = response.summary
        assert "Uncertainties / Unavailable Data" in summary
        assert "Cluster connection refused" in summary or "Kubernetes cluster deployment check failed" in summary
        assert "Airflow REST API service offline" in summary


# ==============================================================================
# Specialized Evaluation Scenarios Tests
# ==============================================================================

def test_dvc_scenario_routing(orchestrator):
    """Verify DVC scenario routes to DVC tools."""
    response = orchestrator.plan_and_execute(goal="List all DVC tracked datasets and check for uncommitted changes")
    assert response.status == ToolExecutionStatus.SUCCESS
    tool_names = [s.tool_name for s in response.steps_executed]
    assert "list_dvc_tracked_datasets" in tool_names


def test_mlflow_scenario_routing(orchestrator):
    """Verify MLflow scenario routes to MLflow tools."""
    response = orchestrator.plan_and_execute(goal="Query MLflow experiment runs and find the best candidate model")
    assert response.status == ToolExecutionStatus.SUCCESS
    tool_names = [s.tool_name for s in response.steps_executed]
    assert "find_best_run" in tool_names


def test_airflow_scenario_routing(orchestrator):
    """Verify Airflow scenario routes to Airflow tools."""
    response = orchestrator.plan_and_execute(goal="List available Airflow DAGs and inspect pipeline status")
    assert response.status == ToolExecutionStatus.SUCCESS
    tool_names = [s.tool_name for s in response.steps_executed]
    assert "list_available_dags" in tool_names


def test_kubernetes_scenario_routing(orchestrator):
    """Verify Kubernetes scenario routes to Kubernetes tools."""
    response = orchestrator.plan_and_execute(goal="Inspect Kubernetes pods and deployments in mlops-prod")
    assert response.status == ToolExecutionStatus.SUCCESS
    tool_names = [s.tool_name for s in response.steps_executed]
    assert "list_pods_and_deployments" in tool_names

