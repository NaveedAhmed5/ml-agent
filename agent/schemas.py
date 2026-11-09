"""Pydantic v2 Core Schemas for MLOps AI Agent.

Defines structured inputs, outputs, and standard execution results across
all 4 tool domains (DVC, MLflow, Airflow, Kubernetes) plus agent-level contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# Common & Execution Status Schemas
# ==============================================================================

class ToolExecutionStatus(str, Enum):
    """Execution status outcomes for agent tools."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    BLOCKED = "BLOCKED"


class ToolExecutionResult(BaseModel):
    """Standardized result wrapper returned by all MLOps tools."""
    model_config = ConfigDict(frozen=False)

    request_id: str = Field(description="Unique tracing identifier for the request/execution")
    tool_name: str = Field(description="Name of the tool executed")
    status: ToolExecutionStatus = Field(description="Status: SUCCESS, ERROR, or BLOCKED")
    duration_ms: float = Field(ge=0.0, description="Duration of tool execution in milliseconds")
    data: dict[str, Any] | list[Any] | None = Field(
        default=None, description="Structured payload returned upon successful execution"
    )
    error: str | None = Field(
        default=None, description="Descriptive error message if status is ERROR or BLOCKED"
    )


# ==============================================================================
# 1. DVC Tool Schemas
# ==============================================================================

class DVCStatusInput(BaseModel):
    """Input parameters for checking DVC repository status."""
    targets: list[str] = Field(default_factory=list, description="Specific targets or files to check")
    check_remote: bool = Field(default=False, description="Whether to compare against remote storage")


class DVCStatusOutput(BaseModel):
    """Output for DVC repository status check."""
    is_clean: bool = Field(description="True if tracked files and pipeline stages are up-to-date")
    changed_files: list[str] = Field(default_factory=list, description="Files with untracked or modified state")
    details: dict[str, Any] = Field(default_factory=dict, description="Raw status details")


class DVCListDatasetsOutput(BaseModel):
    """Output schema for listing tracked datasets."""
    tracked_datasets: list[dict[str, Any]] = Field(default_factory=list, description="List of tracked datasets")
    total_datasets: int = Field(ge=0, description="Total number of tracked datasets found")


class DVCDatasetVersionOutput(BaseModel):
    """Output schema for retrieving current dataset version and provenance."""
    dataset_path: str = Field(description="Path to tracked dataset file")
    dvc_file: str = Field(description="Associated .dvc file")
    md5_hash: str = Field(description="MD5 content hash recorded in DVC")
    size_bytes: int = Field(ge=0, description="File size in bytes")
    dataset_version_label: str = Field(description="Semantic version marker if present (e.g. v1.0.0)")
    git_commit: str = Field(description="Current Git commit hash")
    dvc_remote: str = Field(description="Configured default DVC remote")


class DVCCompareVersionsInput(BaseModel):
    """Input schema for comparing two dataset revisions or states."""
    rev1: str = Field(description="First revision, branch, or commit hash")
    rev2: str = Field(default="HEAD", description="Second revision, branch, or commit hash")


class DVCCompareVersionsOutput(BaseModel):
    """Output schema for comparing dataset versions."""
    rev1: str = Field(description="First revision identifier")
    rev2: str = Field(description="Second revision identifier")
    hash_rev1: str | None = Field(default=None, description="MD5 hash at rev1")
    hash_rev2: str | None = Field(default=None, description="MD5 hash at rev2")
    size_delta_bytes: int | None = Field(default=None, description="Change in file size")
    is_identical: bool = Field(description="True if hashes match")
    diff_summary: str = Field(description="Summary of detected differences")


class DVCDetectChangesOutput(BaseModel):
    """Output schema for detecting data modifications."""
    has_changes: bool = Field(description="True if data or pipeline stages have changed")
    changed_items: list[str] = Field(default_factory=list, description="List of modified or untracked items")
    status_summary: str = Field(description="Human-readable explanation of workspace state")


class DVCReproductionStepsOutput(BaseModel):
    """Output schema containing verified step-by-step reproduction instructions."""
    target_revision: str = Field(description="Target commit hash or version identifier")
    dataset_path: str = Field(description="Path to target dataset")
    commands: list[str] = Field(description="Ordered shell commands to execute reproduction")
    provenance_details: dict[str, Any] = Field(default_factory=dict, description="Metadata and remote provenance")



class DVCPushInput(BaseModel):
    """Input parameters for pushing data/models to remote DVC storage."""
    remote: str | None = Field(default=None, description="Target remote name; defaults to default remote")
    targets: list[str] = Field(default_factory=list, description="Specific files/directories to push")


class DVCPushOutput(BaseModel):
    """Output confirmation of DVC push operation."""
    remote: str = Field(description="Remote name used")
    pushed_files_count: int = Field(ge=0, description="Number of files pushed")
    success: bool = Field(description="Whether push completed successfully")


class DVCPullInput(BaseModel):
    """Input parameters for pulling data/models from remote DVC storage."""
    remote: str | None = Field(default=None, description="Remote name; defaults to default remote")
    targets: list[str] = Field(default_factory=list, description="Specific files/directories to pull")
    force: bool = Field(default=False, description="Force overwrite existing local files")


class DVCPullOutput(BaseModel):
    """Output confirmation of DVC pull operation."""
    pulled_files_count: int = Field(ge=0, description="Number of files pulled")
    success: bool = Field(description="Whether pull completed successfully")


class DVCReproInput(BaseModel):
    """Input parameters for reproducing DVC pipeline stages."""
    targets: list[str] = Field(default_factory=list, description="Specific stages to run; empty runs full pipeline")
    force: bool = Field(default=False, description="Force re-running stages even if hash matches")


class DVCReproOutput(BaseModel):
    """Output from reproducing DVC pipeline stages."""
    executed_stages: list[str] = Field(default_factory=list, description="Stages executed")
    success: bool = Field(description="Whether pipeline reproduction succeeded")


class DVCMetricsInput(BaseModel):
    """Input parameters for retrieving DVC metrics."""
    targets: list[str] = Field(default_factory=list, description="Target metrics files")
    all_branches: bool = Field(default=False, description="Compare metrics across Git branches")


class DVCMetricsOutput(BaseModel):
    """Output containing parsed DVC metrics."""
    metrics: dict[str, Any] = Field(default_factory=dict, description="Extracted metrics dictionary")


# ==============================================================================
# 2. MLflow Tool Schemas
# ==============================================================================

class MLflowCreateExperimentInput(BaseModel):
    """Input parameters to create or get an MLflow experiment."""
    experiment_name: str = Field(min_length=1, description="Name of the experiment to create")
    artifact_location: str | None = Field(default=None, description="Storage location for artifacts")


class MLflowCreateExperimentOutput(BaseModel):
    """Output result of MLflow experiment creation."""
    experiment_id: str = Field(description="Unique ID of the experiment")
    experiment_name: str = Field(description="Name of the experiment")


class MLflowLogRunInput(BaseModel):
    """Input parameters for logging an MLflow training run."""
    experiment_name: str = Field(description="Target experiment name")
    run_name: str | None = Field(default=None, description="Optional human-readable run name")
    params: dict[str, Any] = Field(default_factory=dict, description="Hyperparameters or configuration")
    metrics: dict[str, float] = Field(default_factory=dict, description="Scalar metrics (accuracy, loss, etc.)")
    tags: dict[str, str] = Field(default_factory=dict, description="Metadata tags")
    artifact_paths: list[str] = Field(default_factory=list, description="Local files/directories to upload as artifacts")


class MLflowLogRunOutput(BaseModel):
    """Output confirmation of logged MLflow run."""
    run_id: str = Field(description="Unique MLflow run ID")
    experiment_id: str = Field(description="Associated experiment ID")
    artifact_uri: str = Field(description="Root URI where artifacts were logged")


class MLflowGetRunInput(BaseModel):
    """Input parameters to fetch details of a specific MLflow run."""
    run_id: str = Field(min_length=1, description="MLflow run ID to inspect")


class MLflowGetRunOutput(BaseModel):
    """Output details of an MLflow run."""
    run_id: str = Field(description="Run ID")
    status: str = Field(description="Run status: FINISHED, FAILED, RUNNING, etc.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Logged metrics")
    params: dict[str, str] = Field(default_factory=dict, description="Logged parameters")
    tags: dict[str, str] = Field(default_factory=dict, description="Logged tags")


class MLflowRegisterModelInput(BaseModel):
    """Input parameters to register a model from an MLflow run."""
    model_uri: str = Field(description="URI pointing to model artifact (e.g., runs:/<run_id>/model)")
    model_name: str = Field(min_length=1, description="Registered model name in MLflow Model Registry")
    description: str | None = Field(default=None, description="Model version description")


class MLflowRegisterModelOutput(BaseModel):
    """Output result of model registration."""
    model_name: str = Field(description="Registered model name")
    version: str = Field(description="Registered version number")
    status: str = Field(description="Registration status")


class MLflowTransitionStageInput(BaseModel):
    """Input parameters to transition a model version stage or alias."""
    model_name: str = Field(min_length=1, description="Registered model name")
    version: str = Field(description="Version string or number")
    stage: str = Field(description="Target stage (e.g. Staging, Production, Archived) or alias")
    archive_existing_versions: bool = Field(default=True, description="Move existing stage versions to Archived")


class MLflowTransitionStageOutput(BaseModel):
    """Output confirmation of model stage transition."""
    model_name: str = Field(description="Model name")
    version: str = Field(description="Version transitioned")
    stage: str = Field(description="New assigned stage")
    success: bool = Field(description="Whether transition was applied")


class MLflowGetLatestModelInput(BaseModel):
    """Input parameters to query the latest model version for a given stage."""
    model_name: str = Field(min_length=1, description="Registered model name")
    stages: list[str] = Field(default_factory=lambda: ["Production", "Staging"], description="Allowed stages")


class MLflowGetLatestModelOutput(BaseModel):
    """Output details of the latest model version."""
    model_name: str = Field(description="Model name")
    version: str = Field(description="Version string")
    current_stage: str = Field(description="Current stage")
    source: str = Field(description="Artifact source path/URI")


class MLflowListExperimentsRunsOutput(BaseModel):
    """Output schema for listing experiments and runs."""
    experiments: list[dict[str, Any]] = Field(default_factory=list, description="Available experiments")
    runs: list[dict[str, Any]] = Field(default_factory=list, description="Recent runs across experiments")
    total_runs: int = Field(ge=0, description="Total runs retrieved")


class MLflowRunDetailsOutput(BaseModel):
    """Output schema for detailed run inspection."""
    run_id: str = Field(description="Unique run ID")
    experiment_id: str = Field(description="Experiment ID")
    status: str = Field(description="Run status: FINISHED, FAILED, RUNNING, etc.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Logged evaluation metrics")
    params: dict[str, Any] = Field(default_factory=dict, description="Hyperparameters logged")
    tags: dict[str, str] = Field(default_factory=dict, description="Metadata and provenance tags")
    provenance: dict[str, Any] = Field(default_factory=dict, description="Parsed dataset and git provenance")
    artifact_paths: list[str] = Field(default_factory=list, description="Artifact directory contents")
    start_time: str | int | None = Field(default=None, description="Run start time")


class MLflowBestRunOutput(BaseModel):
    """Output schema for finding the best run with production-readiness check."""
    best_run_id: str = Field(description="ID of the highest-performing run")
    experiment_name: str = Field(description="Experiment evaluated")
    target_metric: str = Field(description="Metric used for ranking")
    metric_value: float = Field(description="Achieved score on target metric")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters of best run")
    metrics: dict[str, float] = Field(default_factory=dict, description="All metrics of best run")
    is_production_ready: bool = Field(description="Whether run satisfies production-readiness criteria")
    readiness_evaluation: dict[str, Any] = Field(
        default_factory=dict,
        description="Checklist results: quality gate, overfitting gap, provenance completeness",
    )
    evaluation_notes: str = Field(
        description="Detailed assessment emphasizing that metric score alone does not guarantee production readiness",
    )


class MLflowCompareRunsOutput(BaseModel):
    """Output schema for side-by-side run comparisons."""
    run_ids: list[str] = Field(description="List of compared run IDs")
    comparison_table: list[dict[str, Any]] = Field(default_factory=list, description="Structured matrix of compared runs")
    tradeoffs_summary: str = Field(description="Executive analysis of model trade-offs")


class MLflowModelRegistryOutput(BaseModel):
    """Output schema for MLflow Model Registry state."""
    registered_models: list[dict[str, Any]] = Field(default_factory=list, description="List of registered models")
    total_models: int = Field(ge=0, description="Count of registered models")



# ==============================================================================
# 3. Airflow Tool Schemas (Airflow 3 / 2 REST API compatible)
# ==============================================================================

class AirflowTriggerDagInput(BaseModel):
    """Input parameters to trigger an Airflow DAG execution."""
    dag_id: str = Field(min_length=1, description="DAG identifier to trigger")
    conf: dict[str, Any] = Field(default_factory=dict, description="Runtime DAG configuration parameters")
    dag_run_id: str | None = Field(default=None, description="Optional custom DAG run ID")


class AirflowTriggerDagOutput(BaseModel):
    """Output confirmation of triggered DAG run."""
    dag_id: str = Field(description="Triggered DAG ID")
    dag_run_id: str = Field(description="Generated or provided DAG Run ID")
    state: str = Field(description="Initial state (e.g. queued, running)")
    logical_date: str | None = Field(default=None, description="Logical execution date timestamp")


class AirflowGetDagRunStatusInput(BaseModel):
    """Input parameters to query an Airflow DAG run state."""
    dag_id: str = Field(min_length=1, description="DAG identifier")
    dag_run_id: str = Field(min_length=1, description="DAG run ID to check")


class AirflowGetDagRunStatusOutput(BaseModel):
    """Output state of an Airflow DAG run."""
    dag_id: str = Field(description="DAG identifier")
    dag_run_id: str = Field(description="DAG run ID")
    state: str = Field(description="Current state: queued, running, success, failed")
    start_date: str | None = Field(default=None, description="DAG run start time")
    end_date: str | None = Field(default=None, description="DAG run end time")
    recent_runs: list[dict[str, Any]] = Field(default_factory=list, description="Recent execution runs for this DAG")


class AirflowListDagsInput(BaseModel):
    """Input parameters to list available DAGs in the Airflow environment."""
    limit: int = Field(default=50, ge=1, le=100, description="Max DAGs to return")
    only_active: bool = Field(default=True, description="Filter for unpaused/active DAGs only")


class AirflowListDagsOutput(BaseModel):
    """Output list of DAGs from Airflow."""
    total_entries: int = Field(description="Total number of DAGs available")
    dags: list[dict[str, Any]] = Field(default_factory=list, description="List of DAG summaries")


class AirflowIdentifyFailedTasksOutput(BaseModel):
    """Output schema for failed task identification in a DAG run."""
    dag_id: str = Field(description="DAG identifier")
    dag_run_id: str = Field(description="DAG run identifier")
    failed_tasks: list[dict[str, Any]] = Field(default_factory=list, description="Tasks in failed or upstream_failed state")
    total_failed: int = Field(ge=0, description="Count of failed task instances")
    summary: str = Field(description="Human-readable diagnosis summary")


class AirflowTaskDiagnosticsOutput(BaseModel):
    """Output schema for in-depth task instance diagnostics and logs."""
    dag_id: str = Field(description="DAG identifier")
    dag_run_id: str = Field(description="DAG run identifier")
    task_id: str = Field(description="Task identifier inspected")
    try_number: int = Field(ge=1, description="Attempt number")
    state: str = Field(description="Task state")
    error_message: str | None = Field(default=None, description="Extracted error reason or exception")
    logs: str = Field(description="Task log snippet or diagnosis")
    recommendations: list[str] = Field(default_factory=list, description="Actionable remediation steps")



# ==============================================================================
# 4. Kubernetes Tool Schemas
# ==============================================================================

class K8sDeployManifestInput(BaseModel):
    """Input parameters to deploy a manifest to Kubernetes."""
    manifest_path: str = Field(min_length=1, description="Path to the Kubernetes YAML manifest")
    namespace: str = Field(description="Target Kubernetes namespace")


class K8sDeployManifestOutput(BaseModel):
    """Output confirmation of Kubernetes resource deployment."""
    namespace: str = Field(description="Deployed namespace")
    resources_applied: list[str] = Field(default_factory=list, description="Names of applied resources")
    success: bool = Field(description="Whether all resources were successfully applied")


class K8sGetPodStatusInput(BaseModel):
    """Input parameters to query pod status in a namespace."""
    namespace: str = Field(description="Target Kubernetes namespace")
    label_selector: str | None = Field(default=None, description="Optional label selector query (e.g., app=ml-api)")


class K8sGetPodStatusOutput(BaseModel):
    """Output status of pods within a namespace."""
    namespace: str = Field(description="Inspected namespace")
    pods: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of pod summaries including name, phase, ready condition, restarts",
    )


class K8sRollbackDeploymentInput(BaseModel):
    """Input parameters to rollback a Kubernetes deployment."""
    deployment_name: str = Field(min_length=1, description="Deployment resource name")
    namespace: str = Field(description="Target Kubernetes namespace")
    revision: int | None = Field(default=None, description="Optional target revision number; 0/None rolls back to previous")


class K8sRollbackDeploymentOutput(BaseModel):
    """Output confirmation of Kubernetes rollback."""
    deployment_name: str = Field(description="Deployment name")
    namespace: str = Field(description="Namespace")
    success: bool = Field(description="Whether rollback was executed")
    message: str = Field(description="Rollback outcome message")


class K8sListResourcesInput(BaseModel):
    """Input parameters to list resources in a namespace."""
    namespace: str = Field(description="Target namespace")
    resource_kind: Literal["pods", "deployments", "services", "configmaps"] = Field(
        default="deployments", description="Resource kind to inspect"
    )


class K8sListResourcesOutput(BaseModel):
    """Output list of Kubernetes resources."""
    namespace: str = Field(description="Namespace")
    resource_kind: str = Field(description="Kind inspected")
    items: list[dict[str, Any]] = Field(default_factory=list, description="Resource summary items")


class K8sListPodsDeploymentsOutput(BaseModel):
    """Output schema for listing Pods and Deployments in a namespace."""
    namespace: str = Field(description="Kubernetes namespace")
    deployments: list[dict[str, Any]] = Field(default_factory=list, description="Deployments with replica status")
    pods: list[dict[str, Any]] = Field(default_factory=list, description="Pods with phase and container readiness")
    total_deployments: int = Field(ge=0, description="Total deployments found")
    total_pods: int = Field(ge=0, description="Total pods found")


class K8sPodStatusLogsOutput(BaseModel):
    """Output schema for pod status and recent container logs."""
    namespace: str = Field(description="Kubernetes namespace")
    pod_name: str = Field(description="Pod name")
    phase: str = Field(description="Pod lifecycle phase: Running, Pending, Failed, etc.")
    is_ready: bool = Field(description="Whether all containers are ready")
    restart_count: int = Field(ge=0, description="Cumulative restart count across containers")
    container_states: dict[str, Any] = Field(default_factory=dict, description="Detailed container state dictionary")
    logs: str = Field(description="Recent container log tail")


class K8sInspectServicesOutput(BaseModel):
    """Output schema for inspected Services in a namespace."""
    namespace: str = Field(description="Kubernetes namespace")
    services: list[dict[str, Any]] = Field(default_factory=list, description="List of inspected service details")
    total_services: int = Field(ge=0, description="Total services count")


class K8sDiagnoseUnhealthyPodsOutput(BaseModel):
    """Output schema for diagnosing unhealthy pods in a namespace."""
    namespace: str = Field(description="Kubernetes namespace")
    unhealthy_pods: list[dict[str, Any]] = Field(default_factory=list, description="Diagnosed unhealthy pods")
    total_unhealthy: int = Field(ge=0, description="Count of unhealthy pods found")
    diagnosis_summary: str = Field(description="Executive diagnosis summary")
    recommendations: list[str] = Field(default_factory=list, description="Recommended remediation actions")


class K8sDeployedAppVersionOutput(BaseModel):
    """Output schema for currently deployed application and model version."""
    namespace: str = Field(description="Kubernetes namespace")
    app_name: str = Field(description="Application or deployment name")
    app_version: str = Field(description="Application version tag")
    model_version: str = Field(description="Model version tag or label")
    container_image: str = Field(description="Full container image reference")
    config_map_values: dict[str, str] = Field(default_factory=dict, description="Active ConfigMap key-value pairs")
    dvc_dataset_version: str | None = Field(default=None, description="Linked DVC dataset version")


class K8sScaleRestartOutput(BaseModel):
    """Output schema for scaling or restarting a deployment."""
    namespace: str = Field(description="Target namespace")
    deployment_name: str = Field(description="Deployment resource modified")
    desired_replicas: int = Field(ge=0, description="Target replica count requested")
    previous_replicas: int = Field(ge=0, description="Replica count prior to operation")
    user_confirmed: bool = Field(description="Confirmation status")
    action: str = Field(description="Action executed: SCALED or RESTARTED")
    message: str = Field(description="Operational outcome message")



class K8sNamespaceGuardInput(BaseModel):
    """Guard schema to validate whether a target namespace is pre-approved."""
    target_namespace: str = Field(min_length=1, description="Requested namespace")
    approved_namespace: str = Field(min_length=1, description="Approved production namespace from policy")


class K8sNamespaceGuardOutput(BaseModel):
    """Result of namespace authorization validation."""
    is_approved: bool = Field(description="True if target namespace equals approved namespace")
    target_namespace: str = Field(description="Requested namespace")
    approved_namespace: str = Field(description="Permitted namespace")
    message: str = Field(description="Validation decision explanation")


# ==============================================================================
# Agent Workflow & Request Schemas
# ==============================================================================

class AgentAction(BaseModel):
    """A planned or executed step by the orchestrator agent."""
    step_id: int = Field(ge=1, description="Sequence index of the step")
    tool_name: str = Field(description="Name of the tool to invoke")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Arguments to provide")
    reasoning: str = Field(description="Reasoning or justification for this action")


class AgentExecutionPlan(BaseModel):
    """Plan generated by the AI agent to fulfill a user goal."""
    goal: str = Field(description="High-level goal description")
    steps: list[AgentAction] = Field(default_factory=list, description="Ordered action steps")


class AgentResponse(BaseModel):
    """Final unified response returned by the MLOps AI Agent."""
    request_id: str = Field(description="Tracing ID")
    goal: str = Field(description="Original goal requested")
    status: ToolExecutionStatus = Field(description="Overall workflow status")
    steps_executed: list[ToolExecutionResult] = Field(
        default_factory=list, description="Individual results of tool executions"
    )
    summary: str = Field(description="Natural language summary of operations and outcomes")

