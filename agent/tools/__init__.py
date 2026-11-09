"""Tool modules for MLOps AI Agent across DVC, MLflow, Airflow, and Kubernetes."""

from agent.tools.airflow_tools import (
    get_dag_run_status,
    get_task_diagnostics,
    identify_failed_tasks,
    list_available_dags,
    trigger_training_dag,
)
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
    validate_namespace,
)
from agent.tools.mlflow_tools import (
    compare_runs,
    find_best_run,
    get_model_registry_info,
    get_run_details,
    list_experiments_and_runs,
)

__all__ = [
    # DVC Tools
    "list_dvc_tracked_datasets",
    "get_current_dataset_version",
    "compare_dataset_versions",
    "detect_data_changes",
    "explain_dataset_reproduction",
    # MLflow Tools
    "list_experiments_and_runs",
    "get_run_details",
    "find_best_run",
    "compare_runs",
    "get_model_registry_info",
    # Airflow Tools
    "list_available_dags",
    "get_dag_run_status",
    "trigger_training_dag",
    "identify_failed_tasks",
    "get_task_diagnostics",
    # Kubernetes Tools
    "list_pods_and_deployments",
    "get_pod_status_and_logs",
    "inspect_services",
    "diagnose_unhealthy_pods",
    "get_deployed_app_version",
    "scale_or_restart_deployment",
    "validate_namespace",
]
