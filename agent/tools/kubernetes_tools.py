"""Kubernetes Operations Agent Tools for Production MLOps Monitoring and Orchestration.

Provides safe inspection, automated failure diagnosis, version auditing,
and guarded scaling/restart operations on Kubernetes clusters using the official
kubernetes Python client (CoreV1Api, AppsV1Api).

Safety Guardrails:
1. Strict Namespace Isolation: Only the pre-approved production namespace
   (default: 'mlops-prod', configurable via K8S_APPROVED_NAMESPACE) is accessible.
   All requests targeting unapproved namespaces ('default', 'kube-system', etc.)
   are rejected with status BLOCKED.
2. Safe Read vs State-Changing Operations: Read-only inspection (pods, deployments,
   services, logs, metrics) executes safely. Destructive or state-changing operations
   (scaling replicas, triggering rollout restarts) require explicit confirmation
   (`user_confirmed: bool = True`). Unconfirmed requests return a structured confirmation
   request payload without mutating cluster state.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from typing import Any

from kubernetes.client.rest import ApiException

from agent.logger import get_logger, log_tool_execution
from agent.schemas import (
    K8sDeployedAppVersionOutput,
    K8sDiagnoseUnhealthyPodsOutput,
    K8sInspectServicesOutput,
    K8sListPodsDeploymentsOutput,
    K8sNamespaceGuardOutput,
    K8sPodStatusLogsOutput,
    K8sScaleRestartOutput,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from kubernetes import client, config

logger = get_logger("kubernetes_tools")

# Module-level client overrides for testing or custom client injection
_core_api_override: Any = None
_apps_api_override: Any = None
_client_lock = threading.Lock()


def get_approved_namespace() -> str:
    """Retrieve the authorized Kubernetes namespace from environment."""
    return os.getenv("K8S_APPROVED_NAMESPACE", "mlops-prod")


def validate_namespace(target_namespace: str) -> K8sNamespaceGuardOutput:
    """Verify that the target namespace conforms to the strict approved namespace policy."""
    approved_namespace = get_approved_namespace()
    is_approved = (target_namespace == approved_namespace)
    message = (
        f"Namespace '{target_namespace}' is authorized."
        if is_approved
        else f"Access BLOCKED: Namespace '{target_namespace}' is not authorized. "
             f"Strict security policy restricts operations solely to '{approved_namespace}'."
    )
    return K8sNamespaceGuardOutput(
        is_approved=is_approved,
        target_namespace=target_namespace,
        approved_namespace=approved_namespace,
        message=message,
    )


def set_k8s_clients(core_api: Any = None, apps_api: Any = None) -> None:
    """Inject mock or custom Kubernetes API clients (primarily for automated testing)."""
    global _core_api_override, _apps_api_override
    with _client_lock:
        _core_api_override = core_api
        _apps_api_override = apps_api


def get_k8s_clients(
    core_client: Any = None,
    apps_client: Any = None,
) -> tuple[client.CoreV1Api, client.AppsV1Api]:
    """Obtain active Kubernetes CoreV1Api and AppsV1Api clients.

    Prioritizes explicit arguments, then module overrides (set_k8s_clients),
    then standard in-cluster / kubeconfig discovery.
    """
    global _core_api_override, _apps_api_override

    c_api = core_client or _core_api_override
    a_api = apps_client or _apps_api_override

    if c_api is not None and a_api is not None:
        return c_api, a_api

    # Attempt cluster configuration
    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception as exc:
            raise RuntimeError(
                f"Kubernetes cluster connection unavailable: {exc}. "
                "Ensure ~/.kube/config is present or inject mock clients via set_k8s_clients()."
            ) from exc

    final_core = c_api or client.CoreV1Api()
    final_apps = a_api or client.AppsV1Api()
    return final_core, final_apps


# ==============================================================================
# Tool 1: List Pods and Deployments
# ==============================================================================

def list_pods_and_deployments(
    namespace: str = "mlops-prod",
    core_client: Any = None,
    apps_client: Any = None,
    request_id: str = "req_k8s_list",
) -> ToolExecutionResult:
    """List all running pods, their status, restart counts, and deployment replica health.

    Args:
        namespace: Target Kubernetes namespace (must match approved namespace).
        core_client: Optional injected CoreV1Api instance.
        apps_client: Optional injected AppsV1Api instance.
        request_id: Tracing identifier for structured audit logging.

    Returns:
        ToolExecutionResult with K8sListPodsDeploymentsOutput.
    """
    start_time = time.perf_counter()
    tool_name = "list_pods_and_deployments"

    guard = validate_namespace(namespace)
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
        core_api, apps_api = get_k8s_clients(core_client, apps_client)

        # 1. Inspect Deployments
        dep_resp = apps_api.list_namespaced_deployment(namespace)
        deployments_summary: list[dict[str, Any]] = []
        for dep in getattr(dep_resp, "items", []):
            spec = getattr(dep, "spec", None)
            status = getattr(dep, "status", None)
            meta = getattr(dep, "metadata", None)

            containers = getattr(getattr(spec, "template", None), "spec", None)
            container_list = getattr(containers, "containers", []) if containers else []
            images = [getattr(c, "image", "") for c in container_list]

            deployments_summary.append({
                "name": getattr(meta, "name", "unknown"),
                "namespace": getattr(meta, "namespace", namespace),
                "replicas_desired": getattr(spec, "replicas", 0) or 0,
                "replicas_ready": getattr(status, "ready_replicas", 0) or 0,
                "replicas_available": getattr(status, "available_replicas", 0) or 0,
                "replicas_updated": getattr(status, "updated_replicas", 0) or 0,
                "container_images": images,
                "created_at": str(getattr(meta, "creation_timestamp", "")),
            })

        # 2. Inspect Pods
        pod_resp = core_api.list_namespaced_pod(namespace)
        pods_summary: list[dict[str, Any]] = []
        for pod in getattr(pod_resp, "items", []):
            pmeta = getattr(pod, "metadata", None)
            pstatus = getattr(pod, "status", None)
            pspec = getattr(pod, "spec", None)

            c_statuses = getattr(pstatus, "container_statuses", []) or []
            total_restarts = sum(getattr(cs, "restart_count", 0) for cs in c_statuses)
            all_ready = all(getattr(cs, "ready", False) for cs in c_statuses) if c_statuses else False

            pods_summary.append({
                "name": getattr(pmeta, "name", "unknown"),
                "phase": getattr(pstatus, "phase", "Unknown"),
                "is_ready": all_ready,
                "restart_count": total_restarts,
                "pod_ip": getattr(pstatus, "pod_ip", None),
                "node_name": getattr(pspec, "node_name", None),
                "start_time": str(getattr(pstatus, "start_time", "")),
            })

        output = K8sListPodsDeploymentsOutput(
            namespace=namespace,
            deployments=deployments_summary,
            pods=pods_summary,
            total_deployments=len(deployments_summary),
            total_pods=len(pods_summary),
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        msg = f"Retrieved {len(deployments_summary)} deployments and {len(pods_summary)} pods in '{namespace}'."
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, msg, result.data)
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to list pods and deployments in '{namespace}': {exc}"
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=err_msg,
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, err_msg, error=str(exc))
        return result


# ==============================================================================
# Tool 2: Get Pod Status and Logs
# ==============================================================================

def get_pod_status_and_logs(
    pod_name: str,
    namespace: str = "mlops-prod",
    tail_lines: int = 50,
    core_client: Any = None,
    apps_client: Any = None,
    request_id: str = "req_k8s_pod_logs",
) -> ToolExecutionResult:
    """Return detailed container states and recent log output for troubleshooting a specific pod.

    Args:
        pod_name: Target pod name.
        namespace: Approved namespace.
        tail_lines: Number of recent log lines to tail.
        core_client: Optional injected CoreV1Api instance.
        apps_client: Optional injected AppsV1Api instance.
        request_id: Tracing identifier.

    Returns:
        ToolExecutionResult with K8sPodStatusLogsOutput.
    """
    start_time = time.perf_counter()
    tool_name = "get_pod_status_and_logs"

    guard = validate_namespace(namespace)
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
        core_api, _ = get_k8s_clients(core_client, apps_client)

        # 1. Fetch Pod Object / Status
        pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        pstatus = getattr(pod, "status", None)
        phase = getattr(pstatus, "phase", "Unknown")

        c_statuses = getattr(pstatus, "container_statuses", []) or []
        total_restarts = sum(getattr(cs, "restart_count", 0) for cs in c_statuses)
        all_ready = all(getattr(cs, "ready", False) for cs in c_statuses) if c_statuses else False

        container_states: dict[str, Any] = {}
        for cs in c_statuses:
            cname = getattr(cs, "name", "container")
            state_obj = getattr(cs, "state", None)
            state_dict: dict[str, Any] = {}

            if state_obj:
                if getattr(state_obj, "running", None):
                    state_dict["state"] = "running"
                    state_dict["started_at"] = str(getattr(state_obj.running, "started_at", ""))
                elif getattr(state_obj, "waiting", None):
                    state_dict["state"] = "waiting"
                    state_dict["reason"] = getattr(state_obj.waiting, "reason", "Unknown")
                    state_dict["message"] = getattr(state_obj.waiting, "message", "")
                elif getattr(state_obj, "terminated", None):
                    state_dict["state"] = "terminated"
                    state_dict["exit_code"] = getattr(state_obj.terminated, "exit_code", -1)
                    state_dict["reason"] = getattr(state_obj.terminated, "reason", "Unknown")
                    state_dict["message"] = getattr(state_obj.terminated, "message", "")

            container_states[cname] = {
                "ready": getattr(cs, "ready", False),
                "restart_count": getattr(cs, "restart_count", 0),
                "image": getattr(cs, "image", ""),
                "state_detail": state_dict,
            }

        # 2. Fetch Container Logs
        logs_str = ""
        try:
            logs_str = core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail_lines,
            )
            if logs_str is None:
                logs_str = ""
        except ApiException as api_exc:
            logs_str = f"[LOG FETCH NOTICE: {api_exc.reason} - Container may not have generated logs yet]"
        except Exception as log_exc:
            logs_str = f"[LOG FETCH NOTICE: Unable to retrieve logs: {log_exc}]"

        output = K8sPodStatusLogsOutput(
            namespace=namespace,
            pod_name=pod_name,
            phase=phase,
            is_ready=all_ready,
            restart_count=total_restarts,
            container_states=container_states,
            logs=logs_str,
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        msg = f"Retrieved status and logs for pod '{pod_name}' in namespace '{namespace}'."
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, msg, result.data)
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to get status and logs for pod '{pod_name}': {exc}"
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=err_msg,
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, err_msg, error=str(exc))
        return result


# ==============================================================================
# Tool 3: Inspect Services
# ==============================================================================

def inspect_services(
    namespace: str = "mlops-prod",
    core_client: Any = None,
    apps_client: Any = None,
    request_id: str = "req_k8s_services",
) -> ToolExecutionResult:
    """List Kubernetes services, ClusterIP addresses, port mappings, and selectors.

    Args:
        namespace: Approved namespace.
        core_client: Optional injected CoreV1Api.
        apps_client: Optional injected AppsV1Api.
        request_id: Tracing identifier.

    Returns:
        ToolExecutionResult with K8sInspectServicesOutput.
    """
    start_time = time.perf_counter()
    tool_name = "inspect_services"

    guard = validate_namespace(namespace)
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
        core_api, _ = get_k8s_clients(core_client, apps_client)
        svc_resp = core_api.list_namespaced_service(namespace)

        services_list: list[dict[str, Any]] = []
        for svc in getattr(svc_resp, "items", []):
            smeta = getattr(svc, "metadata", None)
            sspec = getattr(svc, "spec", None)

            ports_info: list[dict[str, Any]] = []
            for p in getattr(sspec, "ports", []) or []:
                ports_info.append({
                    "name": getattr(p, "name", None),
                    "port": getattr(p, "port", None),
                    "target_port": str(getattr(p, "target_port", "")),
                    "protocol": getattr(p, "protocol", "TCP"),
                })

            services_list.append({
                "name": getattr(smeta, "name", "unknown"),
                "namespace": getattr(smeta, "namespace", namespace),
                "type": getattr(sspec, "type", "ClusterIP"),
                "cluster_ip": getattr(sspec, "cluster_ip", None),
                "ports": ports_info,
                "selector": getattr(sspec, "selector", {}) or {},
                "created_at": str(getattr(smeta, "creation_timestamp", "")),
            })

        output = K8sInspectServicesOutput(
            namespace=namespace,
            services=services_list,
            total_services=len(services_list),
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        msg = f"Retrieved {len(services_list)} services in namespace '{namespace}'."
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, msg, result.data)
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to inspect services in '{namespace}': {exc}"
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=err_msg,
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, err_msg, error=str(exc))
        return result


# ==============================================================================
# Tool 4: Diagnose Unhealthy Pods
# ==============================================================================

def diagnose_unhealthy_pods(
    namespace: str = "mlops-prod",
    core_client: Any = None,
    apps_client: Any = None,
    request_id: str = "req_k8s_diagnose",
) -> ToolExecutionResult:
    """Scan pods for failure patterns (CrashLoopBackOff, OOMKilled, Pending, ProbeFailures).

    Extracts termination codes, container waiting reasons, and returns automated
    root-cause remediation advice.

    Args:
        namespace: Approved namespace.
        core_client: Optional injected CoreV1Api.
        apps_client: Optional injected AppsV1Api.
        request_id: Tracing identifier.

    Returns:
        ToolExecutionResult with K8sDiagnoseUnhealthyPodsOutput.
    """
    start_time = time.perf_counter()
    tool_name = "diagnose_unhealthy_pods"

    guard = validate_namespace(namespace)
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
        core_api, _ = get_k8s_clients(core_client, apps_client)
        pod_resp = core_api.list_namespaced_pod(namespace)

        unhealthy_pods: list[dict[str, Any]] = []
        actionable_recommendations: list[str] = []

        for pod in getattr(pod_resp, "items", []):
            pmeta = getattr(pod, "metadata", None)
            pname = getattr(pmeta, "name", "unknown")
            pstatus = getattr(pod, "status", None)
            phase = getattr(pstatus, "phase", "Unknown")

            c_statuses = getattr(pstatus, "container_statuses", []) or []

            # Check 1: Pod Phase is Pending or Failed
            if phase in ["Pending", "Failed", "Unknown"]:
                reason = getattr(pstatus, "reason", "PhaseIssue")
                msg = getattr(pstatus, "message", "Pod is not in Running state.")
                rec = (
                    f"Pod '{pname}' is in phase '{phase}'. Check node resource capacity (CPU/memory requests), "
                    "taint tolerations, and persistent volume claims using 'kubectl describe pod'."
                )
                unhealthy_pods.append({
                    "pod_name": pname,
                    "phase": phase,
                    "issue_type": "POD_PHASE_UNHEALTHY",
                    "reason": reason,
                    "message": msg,
                    "recommendation": rec,
                })
                if rec not in actionable_recommendations:
                    actionable_recommendations.append(rec)

            # Check 2: Inspect individual container statuses for failures
            for cs in c_statuses:
                cname = getattr(cs, "name", "container")
                restarts = getattr(cs, "restart_count", 0)
                is_ready = getattr(cs, "ready", False)
                state_obj = getattr(cs, "state", None)

                # Check Waiting states (CrashLoopBackOff, ImagePullBackOff, etc.)
                waiting = getattr(state_obj, "waiting", None) if state_obj else None
                if waiting:
                    w_reason = getattr(waiting, "reason", "Waiting")
                    w_msg = getattr(waiting, "message", "")

                    if "CrashLoopBackOff" in w_reason:
                        rec = (
                            f"Container '{cname}' in pod '{pname}' is in CrashLoopBackOff (restarts: {restarts}). "
                            "Examine container logs with 'get_pod_status_and_logs()'. "
                            "Inspect application startup failures, unhandled exceptions, and missing environment variables."
                        )
                        unhealthy_pods.append({
                            "pod_name": pname,
                            "container": cname,
                            "phase": phase,
                            "issue_type": "CRASH_LOOP_BACKOFF",
                            "reason": w_reason,
                            "message": w_msg,
                            "restart_count": restarts,
                            "recommendation": rec,
                        })
                        if rec not in actionable_recommendations:
                            actionable_recommendations.append(rec)

                    elif "ImagePull" in w_reason or "ErrImage" in w_reason:
                        rec = (
                            f"Container '{cname}' in pod '{pname}' failed to pull image (reason: {w_reason}). "
                            "Verify image name/tag in deployment.yaml and check registry pull secret permissions."
                        )
                        unhealthy_pods.append({
                            "pod_name": pname,
                            "container": cname,
                            "phase": phase,
                            "issue_type": "IMAGE_PULL_FAILURE",
                            "reason": w_reason,
                            "message": w_msg,
                            "recommendation": rec,
                        })
                        if rec not in actionable_recommendations:
                            actionable_recommendations.append(rec)

                    elif "Config" in w_reason:
                        rec = (
                            f"Container '{cname}' in pod '{pname}' encountered config error ({w_reason}). "
                            "Ensure referenced ConfigMaps and Secrets exist in namespace 'mlops-prod'."
                        )
                        unhealthy_pods.append({
                            "pod_name": pname,
                            "container": cname,
                            "phase": phase,
                            "issue_type": "CONFIG_ERROR",
                            "reason": w_reason,
                            "message": w_msg,
                            "recommendation": rec,
                        })
                        if rec not in actionable_recommendations:
                            actionable_recommendations.append(rec)

                # Check Terminated states (OOMKilled, Error)
                terminated = getattr(state_obj, "terminated", None) if state_obj else None
                if terminated:
                    t_reason = getattr(terminated, "reason", "Terminated")
                    t_exit = getattr(terminated, "exit_code", -1)
                    t_msg = getattr(terminated, "message", "")

                    if t_reason == "OOMKilled" or t_exit == 137:
                        rec = (
                            f"Container '{cname}' in pod '{pname}' was OOMKilled (Exit Code 137). "
                            "The container exceeded its memory limit. Increase 'resources.limits.memory' "
                            "in kubernetes/deployment.yaml or optimize model memory usage."
                        )
                        unhealthy_pods.append({
                            "pod_name": pname,
                            "container": cname,
                            "phase": phase,
                            "issue_type": "OOM_KILLED",
                            "reason": t_reason,
                            "exit_code": t_exit,
                            "message": t_msg,
                            "recommendation": rec,
                        })
                        if rec not in actionable_recommendations:
                            actionable_recommendations.append(rec)

                    elif t_exit != 0 and t_reason != "Completed":
                        rec = (
                            f"Container '{cname}' in pod '{pname}' terminated with non-zero exit code ({t_exit}: {t_reason}). "
                            "Check application logs for traceback or exit signals."
                        )
                        unhealthy_pods.append({
                            "pod_name": pname,
                            "container": cname,
                            "phase": phase,
                            "issue_type": "NON_ZERO_EXIT",
                            "reason": t_reason,
                            "exit_code": t_exit,
                            "message": t_msg,
                            "recommendation": rec,
                        })
                        if rec not in actionable_recommendations:
                            actionable_recommendations.append(rec)

                # Check Readiness failures while in Running phase
                if phase == "Running" and not is_ready:
                    # Only flag if not already captured by waiting/terminated
                    already_flagged = any(u["pod_name"] == pname and u.get("container") == cname for u in unhealthy_pods)
                    if not already_flagged:
                        rec = (
                            f"Container '{cname}' in pod '{pname}' is Running but not Ready. "
                            "Verify readiness probe (/readyz) is succeeding and model checkpoint is loaded."
                        )
                        unhealthy_pods.append({
                            "pod_name": pname,
                            "container": cname,
                            "phase": phase,
                            "issue_type": "READINESS_PROBE_FAILING",
                            "reason": "ContainerNotReady",
                            "restart_count": restarts,
                            "recommendation": rec,
                        })
                        if rec not in actionable_recommendations:
                            actionable_recommendations.append(rec)

        if unhealthy_pods:
            summary = (
                f"Detected {len(unhealthy_pods)} unhealthy pod issue(s) across namespace '{namespace}'. "
                f"Issues: {', '.join(set(u['issue_type'] for u in unhealthy_pods))}."
            )
        else:
            summary = f"All pods in namespace '{namespace}' are healthy and ready."

        output = K8sDiagnoseUnhealthyPodsOutput(
            namespace=namespace,
            unhealthy_pods=unhealthy_pods,
            total_unhealthy=len(unhealthy_pods),
            diagnosis_summary=summary,
            recommendations=actionable_recommendations,
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, summary, result.data)
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to diagnose pods in '{namespace}': {exc}"
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=err_msg,
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, err_msg, error=str(exc))
        return result


# ==============================================================================
# Tool 5: Get Deployed App Version
# ==============================================================================

def get_deployed_app_version(
    namespace: str = "mlops-prod",
    core_client: Any = None,
    apps_client: Any = None,
    request_id: str = "req_k8s_version",
) -> ToolExecutionResult:
    """Inspect active container images, ConfigMaps, and environment vars to audit release version.

    Args:
        namespace: Approved namespace.
        core_client: Optional injected CoreV1Api.
        apps_client: Optional injected AppsV1Api.
        request_id: Tracing identifier.

    Returns:
        ToolExecutionResult with K8sDeployedAppVersionOutput.
    """
    start_time = time.perf_counter()
    tool_name = "get_deployed_app_version"

    guard = validate_namespace(namespace)
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
        core_api, apps_api = get_k8s_clients(core_client, apps_client)

        # 1. Inspect Deployments
        dep_resp = apps_api.list_namespaced_deployment(namespace)
        target_dep = None
        for dep in getattr(dep_resp, "items", []):
            dname = getattr(getattr(dep, "metadata", None), "name", "")
            if "inference" in dname or "api" in dname or "mlops" in dname:
                target_dep = dep
                break
        if target_dep is None and getattr(dep_resp, "items", []):
            target_dep = dep_resp.items[0]

        app_name = "mlops-inference-deployment"
        container_image = "unknown"
        image_tag = "unknown"

        if target_dep:
            app_name = getattr(getattr(target_dep, "metadata", None), "name", app_name)
            spec = getattr(target_dep, "spec", None)
            template = getattr(spec, "template", None) if spec else None
            tspec = getattr(template, "spec", None) if template else None
            containers = getattr(tspec, "containers", []) if tspec else []
            if containers:
                container_image = getattr(containers[0], "image", "unknown")
                if ":" in container_image:
                    image_tag = container_image.split(":")[-1]

        # 2. Inspect ConfigMap
        cm_values: dict[str, str] = {}
        try:
            cm_resp = core_api.list_namespaced_config_map(namespace)
            for cm in getattr(cm_resp, "items", []):
                cm_data = getattr(cm, "data", {}) or {}
                if "APP_VERSION" in cm_data or "MODEL_NAME" in cm_data:
                    cm_values.update(cm_data)
                    break
        except Exception:
            pass

        app_version = cm_values.get("APP_VERSION", image_tag if image_tag != "unknown" else "1.2.0")
        model_version = cm_values.get("MODEL_VERSION", "1.2.0")
        dvc_dataset_version = cm_values.get("DVC_DATASET_VERSION", "v1.0.0")

        output = K8sDeployedAppVersionOutput(
            namespace=namespace,
            app_name=app_name,
            app_version=app_version,
            model_version=model_version,
            container_image=container_image,
            config_map_values=cm_values,
            dvc_dataset_version=dvc_dataset_version,
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        msg = f"Audited deployed app version '{app_version}' (model: '{model_version}') in '{namespace}'."
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, msg, result.data)
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to get deployed app version in '{namespace}': {exc}"
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=err_msg,
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, err_msg, error=str(exc))
        return result


# ==============================================================================
# Tool 6: Scale or Restart Deployment (Guarded State Mutation)
# ==============================================================================

def scale_or_restart_deployment(
    deployment_name: str,
    replicas: int | None = None,
    restart: bool = False,
    namespace: str = "mlops-prod",
    user_confirmed: bool = False,
    core_client: Any = None,
    apps_client: Any = None,
    request_id: str = "req_k8s_scale_restart",
) -> ToolExecutionResult:
    """Safely scale replica count or trigger rollout restart of a Kubernetes deployment.

    Safety Enforcement:
    - Namespace check: Rejects any namespace other than approved production namespace.
    - Explicit Confirmation Check: `user_confirmed` MUST be set to True.
      If `user_confirmed=False`, returns status BLOCKED with confirmation parameters.

    Args:
        deployment_name: Target deployment resource.
        replicas: Desired replica count (when scaling).
        restart: When True, triggers a rolling restart by updating pod annotations.
        namespace: Approved namespace.
        user_confirmed: Explicit safety flag required for mutation execution.
        core_client: Optional injected CoreV1Api.
        apps_client: Optional injected AppsV1Api.
        request_id: Tracing identifier.

    Returns:
        ToolExecutionResult with K8sScaleRestartOutput.
    """
    start_time = time.perf_counter()
    tool_name = "scale_or_restart_deployment"

    # Guardrail 1: Namespace verification
    guard = validate_namespace(namespace)
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

    # Guardrail 2: User Confirmation Check
    action_type = "ROLLOUT_RESTART" if restart else f"SCALE_TO_{replicas}_REPLICAS"
    if not user_confirmed:
        duration = (time.perf_counter() - start_time) * 1000.0
        prompt_text = (
            f"Safety Gate: State-changing operation '{action_type}' on deployment '{deployment_name}' "
            f"in namespace '{namespace}' requires explicit user confirmation. "
            "Please confirm by calling this tool with `user_confirmed=True`."
        )
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.BLOCKED,
            duration_ms=duration,
            data={
                "confirmation_required": True,
                "action": action_type,
                "deployment_name": deployment_name,
                "namespace": namespace,
                "requested_replicas": replicas,
                "restart": restart,
                "user_confirmed": False,
            },
            error=prompt_text,
        )
        log_tool_execution(logger, request_id, tool_name, "BLOCKED", duration, prompt_text, result.data)
        return result

    if replicas is None and not restart:
        duration = (time.perf_counter() - start_time) * 1000.0
        err_msg = "Invalid parameters: Either 'replicas' (int) or 'restart=True' must be provided."
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=err_msg,
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, err_msg, error=err_msg)
        return result

    try:
        _, apps_api = get_k8s_clients(core_client, apps_client)

        # Read current deployment
        curr_dep = apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        prev_replicas = getattr(getattr(curr_dep, "spec", None), "replicas", 0) or 0

        if restart:
            now_iso = datetime.now(UTC).isoformat()
            patch_body = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": now_iso,
                            }
                        }
                    }
                }
            }
            apps_api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=patch_body,
            )
            msg = f"Triggered rolling restart for deployment '{deployment_name}' at {now_iso}."
            output = K8sScaleRestartOutput(
                namespace=namespace,
                deployment_name=deployment_name,
                desired_replicas=prev_replicas,
                previous_replicas=prev_replicas,
                user_confirmed=True,
                action="RESTARTED",
                message=msg,
            )
        else:
            assert replicas is not None
            patch_body = {"spec": {"replicas": replicas}}
            apps_api.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=patch_body,
            )
            msg = f"Scaled deployment '{deployment_name}' from {prev_replicas} to {replicas} replicas."
            output = K8sScaleRestartOutput(
                namespace=namespace,
                deployment_name=deployment_name,
                desired_replicas=replicas,
                previous_replicas=prev_replicas,
                user_confirmed=True,
                action="SCALED",
                message=msg,
            )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(logger, request_id, tool_name, "SUCCESS", duration, msg, result.data)
        return result

    except Exception as exc:
        duration = (time.perf_counter() - start_time) * 1000.0
        err_msg = f"Failed to modify deployment '{deployment_name}': {exc}"
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.ERROR,
            duration_ms=duration,
            error=err_msg,
        )
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, err_msg, error=str(exc))
        return result
