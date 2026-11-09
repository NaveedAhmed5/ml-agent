"""Deliberate Failure Injection & Automated Agent Diagnosis Script.

Simulates two realistic, high-impact production MLOps failures and demonstrates
how the MLOps AI Agent autonomously investigates, identifies root causes, and provides
actionable remediation steps:

1. Failure 1 (Data & Pipeline Layer):
   A corrupted dataset / severe data quality degradation causing the Airflow pipeline
   `validate_data` or `quality_gate` task to fail.
   Agent Tools: `identify_failed_tasks` and `get_task_diagnostics`.

2. Failure 2 (Infrastructure & Kubernetes Layer):
   A Kubernetes Pod entering `CrashLoopBackOff` / `OOMKilled` due to an invalid model
   checkpoint path in ConfigMap and container memory limits exceeded.
   Agent Tools: `diagnose_unhealthy_pods` and `get_pod_status_and_logs`.

Usage:
  python training/inject_failures.py --failure 1
  python training/inject_failures.py --failure 2
  python training/inject_failures.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.tools.airflow_tools import (
    get_task_diagnostics,
    identify_failed_tasks,
)
from agent.tools.kubernetes_tools import (
    diagnose_unhealthy_pods,
    get_pod_status_and_logs,
    set_k8s_clients,
)


def run_failure_1_data_pipeline_demo() -> None:
    """Simulate and diagnose Failure 1: Data / Pipeline Quality Gate Failure."""
    print("\n" + "=" * 80)
    print("🚨 SIMULATING FAILURE 1: Data / Pipeline Layer — Quality Gate Failure")
    print("=" * 80)
    print("Symptom: Airflow DAG run 'run_prod_training_20261109' marked as FAILED.")
    print("Context: New batch of incoming training data had corrupted labels and 40% missing features.")
    print("Agent Action: Query Airflow REST API to isolate failed tasks and diagnose root cause.\n")

    # Mock Airflow REST API responses simulating task failure
    mock_tasks_payload = {
        "task_instances": [
            {"task_id": "validate_data", "state": "success", "duration": 1.2},
            {"task_id": "preprocess_data", "state": "success", "duration": 2.5},
            {"task_id": "train_model", "state": "success", "duration": 8.1},
            {"task_id": "evaluate_and_log_mlflow", "state": "success", "duration": 3.4},
            {
                "task_id": "quality_gate",
                "state": "failed",
                "duration": 0.8,
                "try_number": 3,
                "max_tries": 3,
            },
        ],
        "total_entries": 5,
    }

    mock_log_text = """
*** Reading local log from /opt/airflow/logs/dag_id=mlops_model_training_pipeline/task_id=quality_gate/attempt=3.log
[2026-11-09T14:32:10.123+0000] {quality_gate.py:42} INFO - Evaluating candidate model metrics:
  - Accuracy: 0.5420 (threshold >= 0.7500) -> FAIL
  - F1 Score: 0.4180 (threshold >= 0.7500) -> FAIL
  - Precision: 0.5100
  - Recall: 0.3540
[2026-11-09T14:32:10.125+0000] {quality_gate.py:65} ERROR - Quality gate rejected candidate model:
  Metric criteria failed: F1 score (0.4180) and Accuracy (0.5420) are below required minimum 0.7500.
  Deployment and Model Registry promotion BLOCKED.
RuntimeError: Quality gate criteria not met.
"""

    with patch("agent.tools.airflow_tools.make_airflow_request") as mock_req:
        mock_req.side_effect = [
            (200, mock_tasks_payload),  # for identify_failed_tasks
            (200, mock_log_text),        # for get_task_diagnostics
        ]

        print("🔍 [STEP 1] Agent calling `identify_failed_tasks`...")
        failed_res = identify_failed_tasks(
            dag_id="mlops_model_training_pipeline",
            dag_run_id="run_prod_training_20261109",
        )
        print(f"   Status: {failed_res.status.value} (took {failed_res.duration_ms:.1f}ms)")
        print(f"   Failed Tasks Detected: {failed_res.data.get('failed_tasks')}")

        print("\n🔍 [STEP 2] Agent calling `get_task_diagnostics` for task 'quality_gate'...")
        diag_res = get_task_diagnostics(
            dag_id="mlops_model_training_pipeline",
            dag_run_id="run_prod_training_20261109",
            task_id="quality_gate",
        )
        print(f"   Status: {diag_res.status.value} (took {diag_res.duration_ms:.1f}ms)")
        d_data = diag_res.data
        print(f"   Identified Error: {d_data.get('error_message')}")
        print("\n   Actionable Remediation Advice:")
        for rec in d_data.get("recommendations", []):
            print(f"     • {rec}")

    print("\n" + "—" * 80)
    print("✅ FAILURE 1 ROOT CAUSE ANALYSIS & RESOLUTION:")
    print("   • Root Cause: Sub-threshold model performance (F1=0.4180 < 0.75) caused by noisy data.")
    print("   • Prevention: Quality gate successfully blocked bad model deployment to Kubernetes.")
    print("   • Resolution: Review incoming dataset quality, clean corrupt labels, and re-run pipeline.")
    print("—" * 80 + "\n")


def run_failure_2_kubernetes_crash_demo() -> None:
    """Simulate and diagnose Failure 2: Infrastructure CrashLoopBackOff & OOMKilled."""
    print("\n" + "=" * 80)
    print("🚨 SIMULATING FAILURE 2: Infrastructure / Kubernetes Layer — CrashLoopBackOff")
    print("=" * 80)
    print("Symptom: Deployment 'mlops-inference-deployment' in 'mlops-prod' has 0/2 pods ready.")
    print("Context: Invalid ConfigMap path mounted and memory limit configured below model footprint.")
    print("Agent Action: Scan pods with `diagnose_unhealthy_pods` and tail logs via `get_pod_status_and_logs`.\n")

    # Mock Kubernetes API objects
    mock_core = MagicMock()
    mock_apps = MagicMock()

    # Unhealthy Pod: CrashLoopBackOff with OOM
    mock_pod = MagicMock()
    mock_pod.metadata.name = "mlops-inference-deployment-7bb9fd4d-8k9m2"
    mock_pod.metadata.namespace = "mlops-prod"
    mock_pod.status.phase = "Running"
    mock_pod.status.pod_ip = "10.244.2.45"
    mock_pod.status.start_time = "2026-11-09T14:00:00Z"
    mock_pod.spec.node_name = "k8s-worker-node-2"

    mock_c = MagicMock()
    mock_c.name = "inference-api"
    mock_c.ready = False
    mock_c.restart_count = 8
    mock_c.image = "mlops-inference-api:1.2.0"
    mock_c.state.running = None
    mock_c.state.waiting.reason = "CrashLoopBackOff"
    mock_c.state.waiting.message = "Back-off 5m0s restarting failed container=inference-api pod=mlops-inference-deployment-..."
    mock_c.state.terminated = None
    mock_pod.status.container_statuses = [mock_c]

    mock_pod_list = MagicMock()
    mock_pod_list.items = [mock_pod]

    mock_core.list_namespaced_pod.return_value = mock_pod_list
    mock_core.read_namespaced_pod.return_value = mock_pod
    mock_core.read_namespaced_pod_log.return_value = """
2026-11-09 14:15:02,102 [INFO] uvicorn.error: Started server process [1]
2026-11-09 14:15:02,103 [INFO] uvicorn.error: Waiting for application startup.
2026-11-09 14:15:02,105 [INFO] api.main: Attempting to load model from: /app/models/corrupted_model.joblib
2026-11-09 14:15:02,110 [ERROR] api.main: FileNotFoundError: [Errno 2] No such file or directory: '/app/models/corrupted_model.joblib'
2026-11-09 14:15:02,112 [CRITICAL] api.main: Unhandled startup exception: Application aborted.
Killed (Out of memory / OOMKilled exit code 137)
"""

    set_k8s_clients(core_api=mock_core, apps_api=mock_apps)

    try:
        print("🔍 [STEP 1] Agent calling `diagnose_unhealthy_pods` in namespace 'mlops-prod'...")
        diag_res = diagnose_unhealthy_pods(namespace="mlops-prod")
        print(f"   Status: {diag_res.status.value} (took {diag_res.duration_ms:.1f}ms)")
        diag_data = diag_res.data
        print(f"   Diagnosis Summary: {diag_data.get('diagnosis_summary')}")
        print(f"   Unhealthy Pods Count: {diag_data.get('total_unhealthy')}")
        for unh in diag_data.get("unhealthy_pods", []):
            print(f"     • Pod: {unh.get('pod_name')} | Issue: {unh.get('issue_type')} | Restarts: {unh.get('restart_count')}")

        print("\n🔍 [STEP 2] Agent calling `get_pod_status_and_logs` for pod 'mlops-inference-deployment-7bb9fd4d-8k9m2'...")
        pod_res = get_pod_status_and_logs(
            pod_name="mlops-inference-deployment-7bb9fd4d-8k9m2",
            namespace="mlops-prod",
            tail_lines=20,
        )
        print(f"   Status: {pod_res.status.value} (took {pod_res.duration_ms:.1f}ms)")
        p_data = pod_res.data
        print(f"   Phase: {p_data.get('phase')} | Is Ready: {p_data.get('is_ready')} | Restarts: {p_data.get('restart_count')}")
        print("\n   Extracted Container Log Tail:")
        for line in p_data.get("logs", "").strip().split("\n")[-6:]:
            print(f"     {line}")

    finally:
        set_k8s_clients(None, None)

    print("\n" + "—" * 80)
    print("✅ FAILURE 2 ROOT CAUSE ANALYSIS & RESOLUTION:")
    print("   • Root Cause 1: Container crashed with FileNotFoundError looking for non-existent model path.")
    print("   • Root Cause 2: Process exceeded memory limit on initialization triggering OOMKilled.")
    print("   • Resolution: Correct MODEL_NAME / checkpoint in kubernetes/configmap.yaml and raise resources.limits.memory.")
    print("—" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate MLOps Failure Injections and Agent Diagnosis")
    parser.add_argument("--failure", type=int, choices=[1, 2], help="Run specific failure demo (1 or 2)")
    parser.add_argument("--all", action="store_true", help="Run both failure scenarios sequentially")

    args = parser.parse_args()

    if args.failure == 1:
        run_failure_1_data_pipeline_demo()
    elif args.failure == 2:
        run_failure_2_kubernetes_crash_demo()
    else:
        run_failure_1_data_pipeline_demo()
        run_failure_2_kubernetes_crash_demo()


if __name__ == "__main__":
    main()
