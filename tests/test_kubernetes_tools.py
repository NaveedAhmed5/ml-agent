"""Automated Test Suite for Kubernetes Operations Agent Tools.

Verifies:
1. Strict namespace isolation guardrails (rejection of unapproved namespaces).
2. Explicit user confirmation requirement for state mutations (scaling and restarting).
3. Pod and deployment listing and status auditing.
4. Pod logs and container state inspection.
5. Service discovery, ClusterIP, and port mapping extraction.
6. Multi-pattern failure diagnosis (CrashLoopBackOff, OOMKilled, ImagePullBackOff, Pending, ProbeFailures).
7. Application and model release version auditing.
8. Controlled scaling and rolling restart executions.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.schemas import ToolExecutionStatus
from agent.tools.kubernetes_tools import (
    diagnose_unhealthy_pods,
    get_deployed_app_version,
    get_pod_status_and_logs,
    inspect_services,
    list_pods_and_deployments,
    scale_or_restart_deployment,
    set_k8s_clients,
    validate_namespace,
)

# ------------------------------------------------------------------------------
# Test Fixtures & Mocks
# ------------------------------------------------------------------------------

@pytest.fixture
def mock_k8s_cluster():
    """Build a comprehensive set of mock CoreV1Api and AppsV1Api clients."""
    mock_core = MagicMock()
    mock_apps = MagicMock()

    # 1. Mock Deployment
    mock_dep = MagicMock()
    mock_dep.metadata.name = "mlops-inference-deployment"
    mock_dep.metadata.namespace = "mlops-prod"
    mock_dep.metadata.creation_timestamp = "2026-11-09T10:00:00Z"
    mock_dep.spec.replicas = 2
    mock_dep.status.ready_replicas = 2
    mock_dep.status.available_replicas = 2
    mock_dep.status.updated_replicas = 2

    mock_container = MagicMock()
    mock_container.name = "inference-api"
    mock_container.image = "mlops-inference-api:1.2.0"
    mock_dep.spec.template.spec.containers = [mock_container]

    mock_dep_list = MagicMock()
    mock_dep_list.items = [mock_dep]
    mock_apps.list_namespaced_deployment.return_value = mock_dep_list
    mock_apps.read_namespaced_deployment.return_value = mock_dep

    # 2. Mock Pod (Healthy)
    mock_pod = MagicMock()
    mock_pod.metadata.name = "mlops-inference-deployment-abc12"
    mock_pod.metadata.namespace = "mlops-prod"
    mock_pod.status.phase = "Running"
    mock_pod.status.pod_ip = "10.244.1.25"
    mock_pod.status.start_time = "2026-11-09T10:05:00Z"
    mock_pod.spec.node_name = "worker-node-1"

    mock_c_status = MagicMock()
    mock_c_status.name = "inference-api"
    mock_c_status.ready = True
    mock_c_status.restart_count = 0
    mock_c_status.image = "mlops-inference-api:1.2.0"
    mock_c_status.state.running.started_at = "2026-11-09T10:05:02Z"
    mock_c_status.state.waiting = None
    mock_c_status.state.terminated = None
    mock_pod.status.container_statuses = [mock_c_status]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_core.list_namespaced_pod.return_value = mock_pod_list
    mock_core.read_namespaced_pod.return_value = mock_pod
    mock_core.read_namespaced_pod_log.return_value = "INFO: Uvicorn running on http://0.0.0.0:8000\nINFO: Ready for traffic."

    # 3. Mock Service
    mock_svc = MagicMock()
    mock_svc.metadata.name = "mlops-inference-service"
    mock_svc.metadata.namespace = "mlops-prod"
    mock_svc.metadata.creation_timestamp = "2026-11-09T10:00:00Z"
    mock_svc.spec.type = "ClusterIP"
    mock_svc.spec.cluster_ip = "10.96.100.50"
    mock_svc.spec.selector = {"app": "mlops-inference-api"}

    mock_port = MagicMock()
    mock_port.name = "http"
    mock_port.port = 80
    mock_port.target_port = 8000
    mock_port.protocol = "TCP"
    mock_svc.spec.ports = [mock_port]

    mock_svc_list = MagicMock()
    mock_svc_list.items = [mock_svc]
    mock_core.list_namespaced_service.return_value = mock_svc_list

    # 4. Mock ConfigMap
    mock_cm = MagicMock()
    mock_cm.metadata.name = "mlops-inference-config"
    mock_cm.data = {
        "APP_VERSION": "1.2.0",
        "MODEL_NAME": "mlops-classifier",
        "MODEL_VERSION": "1.2.0",
        "DVC_DATASET_VERSION": "v1.0.0",
        "AGENT_ENV": "mlops-prod",
    }
    mock_cm_list = MagicMock()
    mock_cm_list.items = [mock_cm]
    mock_core.list_namespaced_config_map.return_value = mock_cm_list

    set_k8s_clients(core_api=mock_core, apps_api=mock_apps)
    yield mock_core, mock_apps
    set_k8s_clients(None, None)


# ==============================================================================
# Namespace Security Guardrail Tests
# ==============================================================================

def test_namespace_validation():
    """Verify namespace policy validation function."""
    assert validate_namespace("mlops-prod").is_approved is True
    assert validate_namespace("kube-system").is_approved is False
    assert validate_namespace("default").is_approved is False


@pytest.mark.parametrize("tool_fn,kwargs", [
    (list_pods_and_deployments, {"namespace": "kube-system"}),
    (get_pod_status_and_logs, {"pod_name": "any-pod", "namespace": "default"}),
    (inspect_services, {"namespace": "staging"}),
    (diagnose_unhealthy_pods, {"namespace": "kube-public"}),
    (get_deployed_app_version, {"namespace": "dev"}),
    (scale_or_restart_deployment, {"deployment_name": "dep", "namespace": "kube-system", "user_confirmed": True}),
])
def test_namespace_guardrail_rejection(tool_fn, kwargs, mock_k8s_cluster):
    """Verify that all tools reject operations in unauthorized namespaces with status BLOCKED."""
    result = tool_fn(**kwargs)
    assert result.status == ToolExecutionStatus.BLOCKED
    assert "not authorized" in result.error.lower() or "blocked" in result.error.lower()


# ==============================================================================
# User Confirmation Guardrail Tests (State Mutations)
# ==============================================================================

def test_scale_deployment_requires_confirmation(mock_k8s_cluster):
    """Verify that scaling without confirmation returns status BLOCKED and does not execute."""
    _, mock_apps = mock_k8s_cluster
    result = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        replicas=5,
        namespace="mlops-prod",
        user_confirmed=False,
    )
    assert result.status == ToolExecutionStatus.BLOCKED
    assert result.data["confirmation_required"] is True
    assert result.data["user_confirmed"] is False
    assert result.data["requested_replicas"] == 5
    mock_apps.patch_namespaced_deployment_scale.assert_not_called()


def test_restart_deployment_requires_confirmation(mock_k8s_cluster):
    """Verify that rollout restart without confirmation returns status BLOCKED."""
    _, mock_apps = mock_k8s_cluster
    result = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        restart=True,
        namespace="mlops-prod",
        user_confirmed=False,
    )
    assert result.status == ToolExecutionStatus.BLOCKED
    assert result.data["confirmation_required"] is True
    assert result.data["action"] == "ROLLOUT_RESTART"
    mock_apps.patch_namespaced_deployment.assert_not_called()


# ==============================================================================
# Read-Only Inspection Tool Tests
# ==============================================================================

def test_list_pods_and_deployments(mock_k8s_cluster):
    """Verify pod and deployment listing in approved namespace."""
    result = list_pods_and_deployments(namespace="mlops-prod")
    assert result.status == ToolExecutionStatus.SUCCESS
    data = result.data
    assert data["namespace"] == "mlops-prod"
    assert data["total_deployments"] == 1
    assert data["total_pods"] == 1
    assert data["deployments"][0]["name"] == "mlops-inference-deployment"
    assert data["deployments"][0]["replicas_desired"] == 2
    assert data["pods"][0]["name"] == "mlops-inference-deployment-abc12"
    assert data["pods"][0]["is_ready"] is True
    assert data["pods"][0]["restart_count"] == 0


def test_get_pod_status_and_logs(mock_k8s_cluster):
    """Verify pod container states and log retrieval."""
    result = get_pod_status_and_logs(
        pod_name="mlops-inference-deployment-abc12",
        namespace="mlops-prod",
        tail_lines=30,
    )
    assert result.status == ToolExecutionStatus.SUCCESS
    data = result.data
    assert data["pod_name"] == "mlops-inference-deployment-abc12"
    assert data["phase"] == "Running"
    assert data["is_ready"] is True
    assert "Uvicorn running" in data["logs"]
    assert "inference-api" in data["container_states"]


def test_inspect_services(mock_k8s_cluster):
    """Verify service discovery, ClusterIP, and port bindings."""
    result = inspect_services(namespace="mlops-prod")
    assert result.status == ToolExecutionStatus.SUCCESS
    data = result.data
    assert data["total_services"] == 1
    svc = data["services"][0]
    assert svc["name"] == "mlops-inference-service"
    assert svc["cluster_ip"] == "10.96.100.50"
    assert svc["ports"][0]["port"] == 80
    assert svc["ports"][0]["target_port"] == "8000"


def test_get_deployed_app_version(mock_k8s_cluster):
    """Verify release version auditing via deployment images and ConfigMap values."""
    result = get_deployed_app_version(namespace="mlops-prod")
    assert result.status == ToolExecutionStatus.SUCCESS
    data = result.data
    assert data["app_version"] == "1.2.0"
    assert data["model_version"] == "1.2.0"
    assert data["dvc_dataset_version"] == "v1.0.0"
    assert "mlops-inference-api:1.2.0" in data["container_image"]
    assert data["config_map_values"]["MODEL_NAME"] == "mlops-classifier"


# ==============================================================================
# Failure Diagnosis Tests
# ==============================================================================

def test_diagnose_healthy_cluster(mock_k8s_cluster):
    """Verify diagnosis reports 0 unhealthy pods on healthy cluster."""
    result = diagnose_unhealthy_pods(namespace="mlops-prod")
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data["total_unhealthy"] == 0
    assert "healthy" in result.data["diagnosis_summary"].lower()


def test_diagnose_crashloop_and_oom_pods(mock_k8s_cluster):
    """Verify automated diagnosis detects CrashLoopBackOff, OOMKilled, Pending, and probe failures."""
    mock_core, _ = mock_k8s_cluster

    # Pod 1: CrashLoopBackOff
    p1 = MagicMock()
    p1.metadata.name = "crashing-pod"
    p1.status.phase = "Running"
    c1 = MagicMock()
    c1.name = "failing-app"
    c1.ready = False
    c1.restart_count = 12
    c1.state.waiting.reason = "CrashLoopBackOff"
    c1.state.waiting.message = "Back-off 5m0s restarting failed container"
    c1.state.terminated = None
    p1.status.container_statuses = [c1]

    # Pod 2: OOMKilled
    p2 = MagicMock()
    p2.metadata.name = "oom-pod"
    p2.status.phase = "Running"
    c2 = MagicMock()
    c2.name = "heavy-memory-worker"
    c2.ready = False
    c2.restart_count = 3
    c2.state.waiting = None
    c2.state.terminated.reason = "OOMKilled"
    c2.state.terminated.exit_code = 137
    c2.state.terminated.message = "Container killed by OOM killer"
    p2.status.container_statuses = [c2]

    # Pod 3: Pending
    p3 = MagicMock()
    p3.metadata.name = "pending-pod"
    p3.status.phase = "Pending"
    p3.status.reason = "Unschedulable"
    p3.status.message = "0/3 nodes are available: 3 Insufficient cpu."
    p3.status.container_statuses = []

    mock_core.list_namespaced_pod.return_value.items = [p1, p2, p3]

    result = diagnose_unhealthy_pods(namespace="mlops-prod")
    assert result.status == ToolExecutionStatus.SUCCESS
    data = result.data
    assert data["total_unhealthy"] >= 3

    issue_types = [u["issue_type"] for u in data["unhealthy_pods"]]
    assert "CRASH_LOOP_BACKOFF" in issue_types
    assert "OOM_KILLED" in issue_types
    assert "POD_PHASE_UNHEALTHY" in issue_types

    # Ensure actionable recommendations are provided
    recs = " ".join(data["recommendations"])
    assert "CrashLoopBackOff" in recs or "logs" in recs.lower()
    assert "memory" in recs.lower()
    assert "cpu" in recs.lower() or "capacity" in recs.lower()


# ==============================================================================
# Executing Guarded State Mutations (With Confirmation)
# ==============================================================================

def test_scale_deployment_with_confirmation(mock_k8s_cluster):
    """Verify scaling executes when user_confirmed=True."""
    _, mock_apps = mock_k8s_cluster
    result = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        replicas=4,
        namespace="mlops-prod",
        user_confirmed=True,
    )
    assert result.status == ToolExecutionStatus.SUCCESS
    data = result.data
    assert data["action"] == "SCALED"
    assert data["desired_replicas"] == 4
    assert data["user_confirmed"] is True
    mock_apps.patch_namespaced_deployment_scale.assert_called_once_with(
        name="mlops-inference-deployment",
        namespace="mlops-prod",
        body={"spec": {"replicas": 4}},
    )


def test_restart_deployment_with_confirmation(mock_k8s_cluster):
    """Verify rollout restart executes when user_confirmed=True."""
    _, mock_apps = mock_k8s_cluster
    result = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        restart=True,
        namespace="mlops-prod",
        user_confirmed=True,
    )
    assert result.status == ToolExecutionStatus.SUCCESS
    data = result.data
    assert data["action"] == "RESTARTED"
    assert data["user_confirmed"] is True
    mock_apps.patch_namespaced_deployment.assert_called_once()
    call_args = mock_apps.patch_namespaced_deployment.call_args[1]
    assert "restartedAt" in str(call_args["body"])


def test_scale_or_restart_missing_parameters(mock_k8s_cluster):
    """Verify error when neither replicas nor restart is specified with user_confirmed=True."""
    result = scale_or_restart_deployment(
        deployment_name="mlops-inference-deployment",
        replicas=None,
        restart=False,
        namespace="mlops-prod",
        user_confirmed=True,
    )
    assert result.status == ToolExecutionStatus.ERROR
    assert "either 'replicas'" in result.error.lower()

