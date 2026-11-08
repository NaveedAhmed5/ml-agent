"""Kubernetes Tools for MLOps AI Agent.

Handles manifest deployment, namespace guardrails, pod health checks,
and deployment rollback.
"""

from __future__ import annotations

import os
import time
from agent.logger import get_logger, log_tool_execution
from agent.schemas import (
    K8sDeployManifestInput,
    K8sDeployManifestOutput,
    K8sGetPodStatusInput,
    K8sGetPodStatusOutput,
    K8sListResourcesInput,
    K8sListResourcesOutput,
    K8sNamespaceGuardInput,
    K8sNamespaceGuardOutput,
    K8sRollbackDeploymentInput,
    K8sRollbackDeploymentOutput,
    ToolExecutionResult,
    ToolExecutionStatus,
)

logger = get_logger("kubernetes_tools")


def validate_namespace(target_namespace: str) -> K8sNamespaceGuardOutput:
    """Guardrail to verify deployment target namespace against allowed production policy."""
    approved_namespace = os.getenv("K8S_APPROVED_NAMESPACE", "mlops-prod")
    is_approved = (target_namespace == approved_namespace)
    message = (
        f"Namespace '{target_namespace}' is approved."
        if is_approved
        else f"Blocked: Target namespace '{target_namespace}' does not match approved '{approved_namespace}'."
    )
    return K8sNamespaceGuardOutput(
        is_approved=is_approved,
        target_namespace=target_namespace,
        approved_namespace=approved_namespace,
        message=message,
    )


def deploy_manifest(params: K8sDeployManifestInput, request_id: str = "req_k8s_deploy") -> ToolExecutionResult:
    """Deploy Kubernetes YAML manifest to the approved namespace."""
    start_time = time.perf_counter()
    tool_name = "k8s_deploy_manifest"

    guard = validate_namespace(params.namespace)
    if not guard.is_approved:
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.BLOCKED,
            duration_ms=duration,
            error=guard.message,
        )
        log_tool_execution(logger, request_id, tool_name, "BLOCKED", duration, guard.message, error=guard.message)
        return result

    try:
        # Stub logic
        output = K8sDeployManifestOutput(
            namespace=params.namespace,
            resources_applied=[params.manifest_path],
            success=True,
        )
        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, "Manifest deployed", result.data)
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Deployment failed", error=str(exc))
        return result

