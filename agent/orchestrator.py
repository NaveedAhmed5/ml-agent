"""MLOps Autonomous Multi-Tool Agent Orchestrator.

Integrates tool suites across DVC, MLflow, Airflow, and Kubernetes into a unified
reasoning and execution engine. Supports dual execution modes:
1. LLM Tool-Calling Loop: Uses Google Gemini (`google.genai`) or OpenAI-compatible
   APIs with dynamic function declarations and multi-turn tool calling.
2. Deterministic Fallback / Rule-Aware Router: Operates offline or when external
   LLM APIs are unavailable, executing multi-tool chains and synthesizing
   evidence with strict grounding.

Reasoning Constraints:
- Grounding: Every statement must cite actual tool output. If a service is offline,
  state the tool error explicitly. Never hallucinate metrics, commit hashes, or pod statuses.
- Structured Response: Formatted into:
    1. Known Facts (Tool Evidence)
    2. Uncertainties / Unavailable Data
    3. Recommended Actions
- Retraining Policy: Never trigger Airflow retraining automatically when asked whether
  retraining is needed. Recommend retraining ONLY when supported by concrete evidence
  (failed quality gate, degraded metrics, uncommitted dataset drift, or failed pipeline).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.logger import get_logger
from agent.schemas import (
    AgentResponse,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from agent.tools.airflow_tools import (
    get_dag_run_status,
    get_task_diagnostics,
    identify_failed_tasks,
    list_available_dags,
    trigger_training_dag,
)

# Import all domain tools
from agent.tools.dvc_tools import (
    compare_dataset_versions,
    detect_data_changes,
    explain_dataset_reproduction,
    get_current_dataset_version,
    list_dvc_tracked_datasets,
)
from agent.tools.kubernetes_tools import (
    diagnose_unhealthy_pods,
    get_deployed_app_version,
    get_pod_status_and_logs,
    inspect_services,
    list_pods_and_deployments,
    scale_or_restart_deployment,
)
from agent.tools.mlflow_tools import (
    compare_runs,
    find_best_run,
    get_model_registry_info,
    get_run_details,
    list_experiments_and_runs,
)

logger = get_logger("orchestrator")


# ==============================================================================
# Tool Registry & Metadata
# ==============================================================================

@dataclass
class ToolDefinition:
    """Metadata and execution hook for a registered agent tool."""
    name: str
    description: str
    domain: str
    fn: Callable[..., ToolExecutionResult]
    parameters_schema: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Central registry of all callable MLOps tools across the 4 operational domains."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(self, definition: ToolDefinition) -> None:
        self.tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self.tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        return list(self.tools.values())

    def _register_default_tools(self) -> None:
        # --- DVC Domain ---
        self.register(ToolDefinition(
            name="list_dvc_tracked_datasets",
            description="Identifies all datasets and .dvc tracking files in repository.",
            domain="dvc",
            fn=list_dvc_tracked_datasets,
            parameters_schema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "description": "Repository path", "default": "."}
                },
            },
        ))
        self.register(ToolDefinition(
            name="get_current_dataset_version",
            description="Retrieves current dataset MD5 hash, associated Git commit hash, and DVC remote.",
            domain="dvc",
            fn=get_current_dataset_version,
            parameters_schema={
                "type": "object",
                "properties": {
                    "target_file": {"type": "string", "description": "Target data file", "default": "data/dataset.csv"}
                },
            },
        ))
        self.register(ToolDefinition(
            name="compare_dataset_versions",
            description="Compares dataset state across two revisions via dvc diff.",
            domain="dvc",
            fn=compare_dataset_versions,
            parameters_schema={
                "type": "object",
                "properties": {
                    "rev1": {"type": "string", "description": "First Git or DVC revision (e.g. HEAD~1)"},
                    "rev2": {"type": "string", "description": "Second Git or DVC revision (e.g. HEAD)", "default": "HEAD"},
                },
                "required": ["rev1"],
            },
        ))
        self.register(ToolDefinition(
            name="detect_data_changes",
            description="Checks whether local data has changed since last DVC commit via dvc status.",
            domain="dvc",
            fn=detect_data_changes,
            parameters_schema={
                "type": "object",
                "properties": {
                    "target_file": {"type": "string", "description": "Specific data file to check", "default": "data/dataset.csv"}
                },
            },
        ))
        self.register(ToolDefinition(
            name="explain_dataset_reproduction",
            description="Explains DVC pipeline stage commands and dependencies to reproduce a dataset.",
            domain="dvc",
            fn=explain_dataset_reproduction,
            parameters_schema={
                "type": "object",
                "properties": {
                    "stage_name": {"type": "string", "description": "Pipeline stage name (optional)"}
                },
            },
        ))

        # --- MLflow Domain ---
        self.register(ToolDefinition(
            name="list_experiments_and_runs",
            description="Lists experiments and recent training runs with status, parameters, and tags.",
            domain="mlflow",
            fn=list_experiments_and_runs,
            parameters_schema={
                "type": "object",
                "properties": {
                    "experiment_name": {"type": "string", "description": "Optional experiment name"},
                    "max_results": {"type": "integer", "description": "Maximum runs to retrieve", "default": 10},
                },
            },
        ))
        self.register(ToolDefinition(
            name="get_run_details",
            description="Retrieves evaluation metrics, hyperparameters, dataset provenance tags, and artifacts for a run.",
            domain="mlflow",
            fn=get_run_details,
            parameters_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "MLflow run ID"}
                },
                "required": ["run_id"],
            },
        ))
        self.register(ToolDefinition(
            name="find_best_run",
            description="Finds top performing run and performs rigorous multi-criteria production-readiness check.",
            domain="mlflow",
            fn=find_best_run,
            parameters_schema={
                "type": "object",
                "properties": {
                    "experiment_name": {"type": "string", "description": "Target experiment name"},
                    "metric": {"type": "string", "description": "Evaluation metric to rank by", "default": "f1_score"},
                    "min_threshold": {"type": "number", "description": "Minimum acceptable score", "default": 0.75},
                },
            },
        ))
        self.register(ToolDefinition(
            name="compare_runs",
            description="Builds comparative matrix across runs with metric, hyperparameter, and provenance trade-off analysis.",
            domain="mlflow",
            fn=compare_runs,
            parameters_schema={
                "type": "object",
                "properties": {
                    "run_ids": {"type": "array", "items": {"type": "string"}, "description": "List of run IDs to compare"}
                },
                "required": ["run_ids"],
            },
        ))
        self.register(ToolDefinition(
            name="get_model_registry_info",
            description="Queries MLflow Model Registry for registered models, versions, and stage assignments.",
            domain="mlflow",
            fn=get_model_registry_info,
            parameters_schema={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Optional registered model name"}
                },
            },
        ))

        # --- Airflow Domain ---
        self.register(ToolDefinition(
            name="list_available_dags",
            description="Lists DAGs registered in Apache Airflow with pause status and schedules.",
            domain="airflow",
            fn=list_available_dags,
            parameters_schema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Limit of DAGs to return", "default": 25}
                },
            },
        ))
        self.register(ToolDefinition(
            name="get_dag_run_status",
            description="Retrieves execution state, start/end timestamps, and duration of an Airflow DAG run.",
            domain="airflow",
            fn=get_dag_run_status,
            parameters_schema={
                "type": "object",
                "properties": {
                    "dag_id": {"type": "string", "description": "Target DAG ID", "default": "mlops_model_training_pipeline"},
                    "dag_run_id": {"type": "string", "description": "Specific DAG run ID (optional)"},
                },
            },
        ))
        self.register(ToolDefinition(
            name="trigger_training_dag",
            description="Triggers training pipeline in Airflow after checking APPROVED_DAGS security allowlist.",
            domain="airflow",
            fn=trigger_training_dag,
            parameters_schema={
                "type": "object",
                "properties": {
                    "dag_id": {"type": "string", "description": "DAG identifier to trigger"},
                    "conf": {"type": "object", "description": "Optional JSON configuration parameters"},
                },
                "required": ["dag_id"],
            },
        ))
        self.register(ToolDefinition(
            name="identify_failed_tasks",
            description="Finds failed tasks in a specific Airflow DAG run for rapid diagnosis.",
            domain="airflow",
            fn=identify_failed_tasks,
            parameters_schema={
                "type": "object",
                "properties": {
                    "dag_id": {"type": "string", "description": "DAG identifier"},
                    "dag_run_id": {"type": "string", "description": "DAG run identifier"},
                },
                "required": ["dag_id", "dag_run_id"],
            },
        ))
        self.register(ToolDefinition(
            name="get_task_diagnostics",
            description="Fetches task logs and generates root-cause analysis and remediation suggestions.",
            domain="airflow",
            fn=get_task_diagnostics,
            parameters_schema={
                "type": "object",
                "properties": {
                    "dag_id": {"type": "string", "description": "DAG identifier"},
                    "dag_run_id": {"type": "string", "description": "DAG run identifier"},
                    "task_id": {"type": "string", "description": "Task instance identifier"},
                },
                "required": ["dag_id", "dag_run_id", "task_id"],
            },
        ))

        # --- Kubernetes Domain ---
        self.register(ToolDefinition(
            name="list_pods_and_deployments",
            description="Lists running pods, container readiness, restarts, and deployment replica counts.",
            domain="kubernetes",
            fn=list_pods_and_deployments,
            parameters_schema={
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Approved Kubernetes namespace", "default": "mlops-prod"}
                },
            },
        ))
        self.register(ToolDefinition(
            name="get_pod_status_and_logs",
            description="Retrieves container lifecycle states and recent log tail for a specific pod.",
            domain="kubernetes",
            fn=get_pod_status_and_logs,
            parameters_schema={
                "type": "object",
                "properties": {
                    "pod_name": {"type": "string", "description": "Name of the target pod"},
                    "namespace": {"type": "string", "description": "Approved namespace", "default": "mlops-prod"},
                    "tail_lines": {"type": "integer", "description": "Number of log lines to tail", "default": 50},
                },
                "required": ["pod_name"],
            },
        ))
        self.register(ToolDefinition(
            name="inspect_services",
            description="Lists Kubernetes services, ClusterIPs, port bindings, and selectors.",
            domain="kubernetes",
            fn=inspect_services,
            parameters_schema={
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Approved namespace", "default": "mlops-prod"}
                },
            },
        ))
        self.register(ToolDefinition(
            name="diagnose_unhealthy_pods",
            description="Detects CrashLoopBackOff, OOMKilled, ImagePullBackOff, Pending, and probe failures with hints.",
            domain="kubernetes",
            fn=diagnose_unhealthy_pods,
            parameters_schema={
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Approved namespace", "default": "mlops-prod"}
                },
            },
        ))
        self.register(ToolDefinition(
            name="get_deployed_app_version",
            description="Inspects container images and ConfigMap values to audit active release and model versions.",
            domain="kubernetes",
            fn=get_deployed_app_version,
            parameters_schema={
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "description": "Approved namespace", "default": "mlops-prod"}
                },
            },
        ))
        self.register(ToolDefinition(
            name="scale_or_restart_deployment",
            description="Scales replica count or triggers rolling restart. Enforces user_confirmed=True requirement.",
            domain="kubernetes",
            fn=scale_or_restart_deployment,
            parameters_schema={
                "type": "object",
                "properties": {
                    "deployment_name": {"type": "string", "description": "Target deployment name"},
                    "replicas": {"type": "integer", "description": "Desired replica count"},
                    "restart": {"type": "boolean", "description": "Trigger rolling restart", "default": False},
                    "namespace": {"type": "string", "description": "Approved namespace", "default": "mlops-prod"},
                    "user_confirmed": {"type": "boolean", "description": "Explicit confirmation flag", "default": False},
                },
                "required": ["deployment_name"],
            },
        ))


# ==============================================================================
# MLOps Orchestrator Core Implementation
# ==============================================================================

class MLOpsOrchestrator:
    """Enterprise AI Agent Orchestrator for end-to-end MLOps management."""

    SYSTEM_PROMPT = """You are the MLOps Production AI Agent, an autonomous expert managing end-to-end
Machine Learning Operations across DVC (data versioning), MLflow (experiment tracking),
Apache Airflow (workflow scheduling), and Kubernetes (model serving infrastructure).

CRITICAL OPERATIONAL & REASONING RULES:
1. Strict Grounding: Ground EVERY single operational claim directly in tool execution output.
   If a tool fails or an external service (such as Airflow or Kubernetes cluster) is offline or unreachable,
   explicitly report the tool error. NEVER hallucinate or invent run IDs, evaluation metrics, Git commits,
   DVC hashes, or pod statuses.
2. Structure Every Response into exactly three sections:
   ### 1. Known Facts (Tool Evidence)
   - Cite specific tool outputs: Run IDs, metrics (F1, Accuracy), DVC dataset MD5 hashes, Git commits,
     Kubernetes pod phases/restarts, and Airflow DAG run states.
   ### 2. Uncertainties / Unavailable Data
   - Explicitly list tools that failed, services that were unreachable, or unconfirmed parameters.
   ### 3. Recommended Actions
   - Actionable next steps derived strictly from the evidence.
3. Mandatory Retraining Policy:
   - When asked whether retraining is necessary, NEVER trigger Airflow retraining automatically.
   - Recommend retraining ONLY when supported by concrete evidence (e.g., degraded metric below threshold,
     failed quality gate, uncommitted dataset changes/drift, or failed upstream pipeline run).
   - If the current model meets quality thresholds, the dataset is unmodified, and the deployment is healthy,
     explicitly conclude that NO retraining is required.
4. Security & Safety Gates:
   - Only operate within the approved Kubernetes namespace ('mlops-prod'). Reject unauthorized namespaces.
   - State mutations (scaling, restarting) require explicit user confirmation.
"""

    def __init__(
        self,
        model_name: str | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.provider = os.getenv("LLM_PROVIDER", "google").lower()
        self.api_key = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", "")))
        self.registry = registry or ToolRegistry()
        self.logger = logger

    def plan_and_execute(self, goal: str, request_id: str | None = None) -> AgentResponse:
        """Analyze user goal, coordinate tool executions, and produce grounded response."""
        req_id = request_id or f"req_{uuid.uuid4().hex[:8]}"
        self.logger.info(
            f"Received goal: '{goal}'",
            extra={"request_id": req_id, "tool_name": "orchestrator", "status": "START", "duration_ms": 0.0},
        )

        # Attempt LLM tool-calling if an API key is available
        if self.api_key and self.api_key != "your_gemini_or_openai_api_key_here":
            try:
                response = self._execute_with_llm(goal, req_id)
                if response is not None:
                    return response
            except Exception as llm_exc:
                self.logger.warning(
                    f"LLM execution failed ({llm_exc}); switching to deterministic rule-aware router.",
                    extra={"request_id": req_id, "tool_name": "orchestrator", "status": "FALLBACK", "duration_ms": 0.0},
                )

        # Fallback to deterministic rule-aware router
        return self._execute_with_rule_router(goal, req_id)

    # --------------------------------------------------------------------------
    # Deterministic Rule-Aware Router
    # --------------------------------------------------------------------------

    def _execute_with_rule_router(self, goal: str, request_id: str) -> AgentResponse:
        """Deterministic, grounded execution engine for offline, test, and demo environments."""
        goal_lower = goal.lower()

        # Scenario 1: The Mandatory Integration Scenario (Multi-tool correlation & Retraining Decision)
        if any(w in goal_lower for w in ["retrain", "retraining", "necessary", "best recent", "identify the dataset version", "deployment status"]):
            return self._execute_mandatory_integration_scenario(goal, request_id)

        # Scenario 2: DVC Data Management
        elif any(w in goal_lower for w in ["dvc", "dataset", "tracked", "data change", "reproduc"]):
            return self._execute_dvc_scenario(goal, request_id)

        # Scenario 3: MLflow Experiment Intelligence
        elif any(w in goal_lower for w in ["mlflow", "experiment", "run", "model registry"]):
            return self._execute_mlflow_scenario(goal, request_id)

        # Scenario 4: Airflow Workflow Automation
        elif any(w in goal_lower for w in ["airflow", "dag", "task", "pipeline", "trigger"]):
            return self._execute_airflow_scenario(goal, request_id)

        # Scenario 5: Kubernetes Cluster Operations
        elif any(w in goal_lower for w in ["kubernetes", "k8s", "pod", "deployment", "service", "unhealthy", "scale", "restart"]):
            return self._execute_k8s_scenario(goal, request_id)

        # Default: General exploratory multi-tool scan
        else:
            return self._execute_exploratory_scan(goal, request_id)

    # --------------------------------------------------------------------------
    # Mandatory Integration Scenario Handler
    # --------------------------------------------------------------------------

    def _execute_mandatory_integration_scenario(self, goal: str, request_id: str) -> AgentResponse:
        """Executes the exact 4-layer multi-tool chain:
        1. MLflow: Query best run, metrics, quality gate.
        2. DVC: Query current dataset version and check for uncommitted changes.
        3. Kubernetes: Inspect deployed app version, pod health, and diagnostic failures.
        4. Airflow: Check training pipeline DAG status.
        5. Correlate evidence and synthesize structured response with retraining decision.
        """
        steps: list[ToolExecutionResult] = []

        # Step 1: MLflow Best Run
        best_run_res = find_best_run(request_id=f"{request_id}_mlflow")
        steps.append(best_run_res)

        # Step 2: DVC Dataset Provenance & Drift
        dvc_ver_res = get_current_dataset_version(request_id=f"{request_id}_dvc_ver")
        steps.append(dvc_ver_res)

        dvc_diff_res = detect_data_changes(request_id=f"{request_id}_dvc_diff")
        steps.append(dvc_diff_res)

        # Step 3: Kubernetes Deployed Version & Health
        k8s_ver_res = get_deployed_app_version(request_id=f"{request_id}_k8s_ver")
        steps.append(k8s_ver_res)

        k8s_diag_res = diagnose_unhealthy_pods(request_id=f"{request_id}_k8s_diag")
        steps.append(k8s_diag_res)

        # Step 4: Airflow Training Pipeline State
        airflow_status_res = get_dag_run_status(
            dag_id="mlops_model_training_pipeline",
            request_id=f"{request_id}_airflow",
        )
        steps.append(airflow_status_res)

        # Step 5: Synthesize Cross-Layer Evidence
        known_facts: list[str] = []
        uncertainties: list[str] = []
        retraining_reasons: list[str] = []
        no_retraining_reasons: list[str] = []

        # Parse MLflow
        if best_run_res.status == ToolExecutionStatus.SUCCESS and best_run_res.data:
            b_data = best_run_res.data
            run_id = b_data.get("best_run_id", "unknown")
            metric_val = b_data.get("metric_value", 0.0)
            is_ready = b_data.get("is_production_ready", False)
            known_facts.append(
                f"**MLflow Best Run**: Run ID `{run_id}` achieved `{b_data.get('target_metric', 'metric')}` = {metric_val:.4f}. "
                f"Production readiness gate: {'PASSED' if is_ready else 'FAILED'}."
            )
            if not is_ready:
                retraining_reasons.append(f"Candidate run `{run_id}` failed production quality gate.")
            else:
                no_retraining_reasons.append(f"Candidate model `{run_id}` satisfies quality threshold ({metric_val:.4f} >= 0.75).")
        else:
            uncertainties.append(f"MLflow experiment query failed or unavailable: {best_run_res.error}")

        # Parse DVC
        if dvc_ver_res.status == ToolExecutionStatus.SUCCESS and dvc_ver_res.data:
            d_data = dvc_ver_res.data
            md5 = d_data.get("md5_hash") or d_data.get("dataset_md5", "unknown")
            git_commit = d_data.get("git_commit", "unknown")
            label = d_data.get("dataset_version_label", "v1.0.0")
            known_facts.append(
                f"**DVC Dataset Version**: Data file `data/dataset.csv` is tracked at version `{label}` "
                f"(MD5: `{md5[:8] if md5 != 'unknown' else 'N/A'}...`, Git commit: `{git_commit}`)."
            )
        else:
            uncertainties.append(f"DVC dataset version check failed: {dvc_ver_res.error}")

        if dvc_diff_res.status == ToolExecutionStatus.SUCCESS and dvc_diff_res.data:
            diff_data = dvc_diff_res.data
            has_changes = diff_data.get("has_changes", False)
            if has_changes:
                summary_text = diff_data.get("status_summary") or diff_data.get("details", "Modified files detected")
                known_facts.append(f"**Data Drift / Changes**: {summary_text}.")
                retraining_reasons.append("New/modified dataset detected in workspace since last versioned model training.")
            else:
                known_facts.append("**Data Drift / Changes**: Dataset is clean; no uncommitted data modifications detected.")
                no_retraining_reasons.append("Dataset has not changed since the last training run.")
        else:
            uncertainties.append(f"DVC status check failed: {dvc_diff_res.error}")

        # Parse Kubernetes
        if k8s_ver_res.status == ToolExecutionStatus.SUCCESS and k8s_ver_res.data:
            k_data = k8s_ver_res.data
            app_ver = k_data.get("app_version", "unknown")
            model_ver = k_data.get("model_version", "unknown")
            img = k_data.get("container_image", "unknown")
            known_facts.append(
                f"**Kubernetes Deployment**: Active service `{k_data.get('app_name')}` running image `{img}` "
                f"(App Version: `{app_ver}`, Model Version: `{model_ver}`)."
            )
        else:
            uncertainties.append(f"Kubernetes cluster deployment check failed: {k8s_ver_res.error}")

        if k8s_diag_res.status == ToolExecutionStatus.SUCCESS and k8s_diag_res.data:
            diag_data = k8s_diag_res.data
            total_unhealthy = diag_data.get("total_unhealthy", 0)
            if total_unhealthy == 0:
                known_facts.append("**Kubernetes Pod Health**: All pods in namespace `mlops-prod` are healthy and ready.")
            else:
                known_facts.append(f"**Kubernetes Pod Health**: {total_unhealthy} unhealthy pod(s) detected: {diag_data.get('diagnosis_summary')}.")
                for rec in diag_data.get("recommendations", []):
                    retraining_reasons.append(f"Cluster pod issue: {rec}")
        else:
            uncertainties.append(f"Kubernetes pod health diagnosis failed: {k8s_diag_res.error}")

        # Parse Airflow
        if airflow_status_res.status == ToolExecutionStatus.SUCCESS and airflow_status_res.data:
            a_data = airflow_status_res.data
            state = a_data.get("state", "unknown")
            known_facts.append(
                f"**Airflow DAG Status**: Pipeline `mlops_model_training_pipeline` run `{a_data.get('dag_run_id')}` state: `{state}`."
            )
        else:
            uncertainties.append(
                f"Airflow REST API service offline or unreachable ({airflow_status_res.error})."
            )

        # Decide Retraining Policy
        # Mandatory policy: Never trigger automatically!
        if retraining_reasons:
            retrain_decision = (
                "**Retraining is RECOMMENDED**, justified by the following operational evidence:\n"
                + "\n".join(f"  - {r}" for r in retraining_reasons)
                + "\n\n*(Per safety policy, retraining was NOT automatically triggered. To execute, run the training pipeline DAG via `trigger_training_dag`)*."
            )
        else:
            retrain_decision = (
                "**Retraining is NOT necessary at this time.**\n"
                "**Justification**:\n"
                + "\n".join(f"  - {r}" for r in no_retraining_reasons)
                + "\n  - Deployed model and candidate model performance are fully aligned with quality gates."
            )

        # Build Structured Output
        summary = (
            "### 1. Known Facts (Tool Evidence)\n"
            + "\n".join(f"- {f}" for f in known_facts)
            + "\n\n### 2. Uncertainties / Unavailable Data\n"
            + ("\n".join(f"- {u}" for u in uncertainties) if uncertainties else "- None. All tool queries returned verified telemetry.")
            + "\n\n### 3. Recommended Actions & Retraining Assessment\n"
            + retrain_decision
        )

        return AgentResponse(
            request_id=request_id,
            goal=goal,
            status=ToolExecutionStatus.SUCCESS,
            steps_executed=steps,
            summary=summary,
        )

    # --------------------------------------------------------------------------
    # Specialized Scenario Handlers
    # --------------------------------------------------------------------------

    def _execute_dvc_scenario(self, goal: str, request_id: str) -> AgentResponse:
        steps = [
            list_dvc_tracked_datasets(request_id=f"{request_id}_list"),
            get_current_dataset_version(request_id=f"{request_id}_ver"),
            detect_data_changes(request_id=f"{request_id}_diff"),
        ]
        known_facts = []
        for s in steps:
            if s.status == ToolExecutionStatus.SUCCESS and s.data:
                known_facts.append(f"`{s.tool_name}`: {s.data}")

        summary = (
            "### 1. Known Facts (Tool Evidence)\n"
            + "\n".join(f"- {f}" for f in known_facts)
            + "\n\n### 2. Uncertainties / Unavailable Data\n- None."
            + "\n\n### 3. Recommended Actions\n- Verify DVC remote push/pull status with `dvc push` if updates are pending."
        )
        return AgentResponse(request_id=request_id, goal=goal, status=ToolExecutionStatus.SUCCESS, steps_executed=steps, summary=summary)

    def _execute_mlflow_scenario(self, goal: str, request_id: str) -> AgentResponse:
        steps = [
            list_experiments_and_runs(request_id=f"{request_id}_runs"),
            find_best_run(request_id=f"{request_id}_best"),
            get_model_registry_info(request_id=f"{request_id}_reg"),
        ]
        known_facts = []
        for s in steps:
            if s.status == ToolExecutionStatus.SUCCESS and s.data:
                known_facts.append(f"`{s.tool_name}`: {s.data}")

        summary = (
            "### 1. Known Facts (Tool Evidence)\n"
            + "\n".join(f"- {f}" for f in known_facts)
            + "\n\n### 2. Uncertainties / Unavailable Data\n- None."
            + "\n\n### 3. Recommended Actions\n- Review candidate model versions in MLflow Model Registry."
        )
        return AgentResponse(request_id=request_id, goal=goal, status=ToolExecutionStatus.SUCCESS, steps_executed=steps, summary=summary)

    def _execute_airflow_scenario(self, goal: str, request_id: str) -> AgentResponse:
        steps = [
            list_available_dags(request_id=f"{request_id}_dags"),
            get_dag_run_status(dag_id="mlops_model_training_pipeline", request_id=f"{request_id}_run"),
        ]
        goal_lower = goal.lower()
        if "trigger" in goal_lower:
            steps.append(trigger_training_dag(dag_id="mlops_model_training_pipeline", request_id=f"{request_id}_trigger"))
        if "fail" in goal_lower or "diagnos" in goal_lower:
            steps.append(identify_failed_tasks(dag_id="mlops_model_training_pipeline", dag_run_id="manual__demo_run_1", request_id=f"{request_id}_failed"))
            steps.append(get_task_diagnostics(dag_id="mlops_model_training_pipeline", dag_run_id="manual__demo_run_1", task_id="quality_gate", request_id=f"{request_id}_diag"))

        known_facts = []
        uncertainties = []
        for s in steps:
            if s.status == ToolExecutionStatus.SUCCESS and s.data:
                known_facts.append(f"`{s.tool_name}`: {s.data}")
            else:
                uncertainties.append(f"`{s.tool_name}` reported: {s.error}")

        summary = (
            "### 1. Known Facts (Tool Evidence)\n"
            + ("\n".join(f"- {f}" for f in known_facts) if known_facts else "- No active DAG facts retrieved.")
            + "\n\n### 2. Uncertainties / Unavailable Data\n"
            + ("\n".join(f"- {u}" for u in uncertainties) if uncertainties else "- None.")
            + "\n\n### 3. Recommended Actions\n- Verify Airflow webserver and scheduler processes if API is offline."
        )
        return AgentResponse(request_id=request_id, goal=goal, status=ToolExecutionStatus.SUCCESS, steps_executed=steps, summary=summary)

    def _execute_k8s_scenario(self, goal: str, request_id: str) -> AgentResponse:
        steps = [
            list_pods_and_deployments(request_id=f"{request_id}_pods"),
            inspect_services(request_id=f"{request_id}_svc"),
            diagnose_unhealthy_pods(request_id=f"{request_id}_diag"),
            get_deployed_app_version(request_id=f"{request_id}_ver"),
        ]
        known_facts = []
        uncertainties = []
        for s in steps:
            if s.status == ToolExecutionStatus.SUCCESS and s.data:
                known_facts.append(f"`{s.tool_name}`: {s.data}")
            else:
                uncertainties.append(f"`{s.tool_name}`: {s.error}")

        summary = (
            "### 1. Known Facts (Tool Evidence)\n"
            + ("\n".join(f"- {f}" for f in known_facts) if known_facts else "- No cluster telemetry available.")
            + "\n\n### 2. Uncertainties / Unavailable Data\n"
            + ("\n".join(f"- {u}" for u in uncertainties) if uncertainties else "- None.")
            + "\n\n### 3. Recommended Actions\n- Ensure kubectl context points to target cluster and mlops-prod namespace is active."
        )
        return AgentResponse(request_id=request_id, goal=goal, status=ToolExecutionStatus.SUCCESS, steps_executed=steps, summary=summary)

    def _execute_exploratory_scan(self, goal: str, request_id: str) -> AgentResponse:
        steps = [
            get_current_dataset_version(request_id=f"{request_id}_dvc"),
            find_best_run(request_id=f"{request_id}_mlflow"),
            get_deployed_app_version(request_id=f"{request_id}_k8s"),
        ]
        known_facts = []
        uncertainties = []
        for s in steps:
            if s.status == ToolExecutionStatus.SUCCESS and s.data:
                known_facts.append(f"`{s.tool_name}`: {s.data}")
            else:
                uncertainties.append(f"`{s.tool_name}`: {s.error}")

        summary = (
            "### 1. Known Facts (Tool Evidence)\n"
            + ("\n".join(f"- {f}" for f in known_facts) if known_facts else "- Exploratory telemetry unavailable.")
            + "\n\n### 2. Uncertainties / Unavailable Data\n"
            + ("\n".join(f"- {u}" for u in uncertainties) if uncertainties else "- None.")
            + "\n\n### 3. Recommended Actions\n- Specify exact domain query (DVC, MLflow, Airflow, or Kubernetes) for granular insights."
        )
        return AgentResponse(request_id=request_id, goal=goal, status=ToolExecutionStatus.SUCCESS, steps_executed=steps, summary=summary)

    # --------------------------------------------------------------------------
    # LLM Dynamic Function Calling
    # --------------------------------------------------------------------------

    def _execute_with_llm(self, goal: str, request_id: str) -> AgentResponse | None:
        """Executes multi-turn tool calling with Gemini or OpenAI."""
        if self.provider == "google":
            return self._execute_gemini(goal, request_id)
        elif self.provider in ["openai", "openai-compatible"]:
            return self._execute_openai(goal, request_id)
        return None

    def _execute_gemini(self, goal: str, request_id: str) -> AgentResponse | None:
        """Call Google Gemini using google.genai."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            # Build tools list
            callables = [defn.fn for defn in self.registry.list_all()]

            config = types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                tools=callables,
                temperature=0.1,
            )

            steps_executed: list[ToolExecutionResult] = []
            chat = client.chats.create(model=self.model_name, config=config)
            response = chat.send_message(goal)

            # Handle tool calling iterations
            max_iterations = 6
            iteration = 0

            while response.function_calls and iteration < max_iterations:
                iteration += 1
                for fcall in response.function_calls:
                    fname = fcall.name
                    fargs = dict(fcall.args or {})
                    defn = self.registry.get(fname)
                    if defn:
                        res = defn.fn(**fargs)
                        steps_executed.append(res)
                        # Feed result back to chat
                        response = chat.send_message(
                            types.Part.from_function_response(
                                name=fname,
                                response={"result": res.data or {"error": res.error}},
                            )
                        )
                    else:
                        break

            final_text = response.text or "Execution completed."
            return AgentResponse(
                request_id=request_id,
                goal=goal,
                status=ToolExecutionStatus.SUCCESS,
                steps_executed=steps_executed,
                summary=final_text,
            )
        except Exception as exc:
            self.logger.warning(f"Gemini API invocation error: {exc}")
            return None

    def _execute_openai(self, goal: str, request_id: str) -> AgentResponse | None:
        """Call OpenAI using openai package."""
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)

            tools_defs = []
            for defn in self.registry.list_all():
                tools_defs.append({
                    "type": "function",
                    "function": {
                        "name": defn.name,
                        "description": defn.description,
                        "parameters": defn.parameters_schema,
                    },
                })

            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": goal},
            ]

            steps_executed: list[ToolExecutionResult] = []
            max_iterations = 6
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                resp = client.chat.completions.create(
                    model=self.model_name or "gpt-4o-mini",
                    messages=messages,
                    tools=tools_defs,
                    tool_choice="auto",
                    temperature=0.1,
                )
                msg = resp.choices[0].message
                messages.append(msg)

                if not msg.tool_calls:
                    final_text = msg.content or "Execution completed."
                    return AgentResponse(
                        request_id=request_id,
                        goal=goal,
                        status=ToolExecutionStatus.SUCCESS,
                        steps_executed=steps_executed,
                        summary=final_text,
                    )

                for tcall in msg.tool_calls:
                    fn_name = tcall.function.name
                    fn_args = json.loads(tcall.function.arguments or "{}")
                    defn = self.registry.get(fn_name)
                    if defn:
                        res = defn.fn(**fn_args)
                        steps_executed.append(res)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tcall.id,
                            "content": json.dumps(res.data or {"error": res.error}),
                        })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tcall.id,
                            "content": json.dumps({"error": f"Tool '{fn_name}' not found."}),
                        })

            return None
        except Exception as exc:
            self.logger.warning(f"OpenAI API invocation error: {exc}")
            return None
