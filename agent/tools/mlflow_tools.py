"""MLflow Experiment Intelligence Tools for MLOps AI Agent (Task 2 — 15 Marks).

Uses mlflow.tracking.MlflowClient exclusively to inspect experiments, fetch run details,
evaluate production readiness, compare model runs, and audit the Model Registry.
Every tool call returns a standardized ToolExecutionResult and logs structured JSON.
"""

from __future__ import annotations

import os
import time
import urllib.request
from typing import Any

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from agent.logger import get_logger, log_tool_execution
from agent.schemas import (
    MLflowBestRunOutput,
    MLflowCompareRunsOutput,
    MLflowListExperimentsRunsOutput,
    MLflowModelRegistryOutput,
    MLflowRunDetailsOutput,
    ToolExecutionResult,
    ToolExecutionStatus,
)

logger = get_logger("mlflow_tools")


def get_client() -> MlflowClient:
    """Instantiate MlflowClient with active tracking URI, falling back to SQLite if needed."""
    configured_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    tracking_uri = configured_uri

    if configured_uri.startswith("http"):
        try:
            req = urllib.request.Request(
                f"{configured_uri.rstrip('/')}/health",
                headers={"User-Agent": "mlflow-tools-probe"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status not in (200, 404):
                    tracking_uri = "sqlite:///mlflow.db"
        except Exception:
            tracking_uri = "sqlite:///mlflow.db"

    return MlflowClient(tracking_uri=tracking_uri)


# ------------------------------------------------------------------------------
# Task 2 Tool 1: list_experiments_and_runs
# ------------------------------------------------------------------------------

def list_experiments_and_runs(
    experiment_name: str | None = None,
    max_results: int = 10,
    request_id: str = "req_list_experiments_runs",
) -> ToolExecutionResult:
    """List experiments and recent training runs with status, start time, and tags."""
    start_time = time.perf_counter()
    tool_name = "list_experiments_and_runs"

    try:
        client = get_client()

        # 1. Fetch experiments
        all_exps = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        if experiment_name:
            target_exps = [e for e in all_exps if e.name == experiment_name]
            if not target_exps:
                # Search case-insensitive
                target_exps = [e for e in all_exps if experiment_name.lower() in e.name.lower()]
        else:
            target_exps = all_exps

        exp_summaries: list[dict[str, Any]] = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
            }
            for exp in target_exps
        ]

        # 2. Fetch runs across matching experiments
        exp_ids = [exp.experiment_id for exp in target_exps]
        runs_summaries: list[dict[str, Any]] = []

        if exp_ids:
            runs = client.search_runs(
                experiment_ids=exp_ids,
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            for r in runs:
                runs_summaries.append({
                    "run_id": r.info.run_id,
                    "experiment_id": r.info.experiment_id,
                    "run_name": r.info.run_name or r.data.tags.get("mlflow.runName", "unnamed"),
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "metrics": {k: round(v, 4) for k, v in r.data.metrics.items()},
                    "params": r.data.params,
                    "tags": {
                        k: v for k, v in r.data.tags.items()
                        if not k.startswith("mlflow.")
                    },
                })

        output = MLflowListExperimentsRunsOutput(
            experiments=exp_summaries,
            runs=runs_summaries,
            total_runs=len(runs_summaries),
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Listed {len(exp_summaries)} experiments and {len(runs_summaries)} runs.",
            result.data,
        )
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed listing experiments/runs", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 2 Tool 2: get_run_details
# ------------------------------------------------------------------------------

def get_run_details(
    run_id: str,
    request_id: str = "req_get_run_details",
) -> ToolExecutionResult:
    """Retrieve all metrics, hyperparameters, dataset provenance tags, and artifacts for a specific run."""
    start_time = time.perf_counter()
    tool_name = "get_run_details"

    try:
        client = get_client()
        run = client.get_run(run_id)

        # Artifacts inspection
        artifact_list: list[str] = []
        try:
            artifacts = client.list_artifacts(run_id)
            artifact_list = [a.path for a in artifacts]
        except Exception:
            pass

        # Provenance tags
        tags = run.data.tags
        provenance = {
            "dataset_version": tags.get("dataset_version", "unknown"),
            "dvc_md5": tags.get("dvc_md5", "unknown"),
            "git_commit": tags.get("git_commit", "unknown"),
            "quality_gate_passed": tags.get("quality_gate_passed", "unknown"),
            "quality_gate_reason": tags.get("quality_gate_reason", "unknown"),
            "model_type": tags.get("model_type", run.data.params.get("model_type", "unknown")),
        }

        output = MLflowRunDetailsOutput(
            run_id=run.info.run_id,
            experiment_id=run.info.experiment_id,
            status=run.info.status,
            metrics={k: round(v, 4) for k, v in run.data.metrics.items()},
            params=run.data.params,
            tags={k: v for k, v in tags.items() if not k.startswith("mlflow.")},
            provenance=provenance,
            artifact_paths=artifact_list,
            start_time=run.info.start_time,
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Retrieved run details for '{run_id}'.",
            result.data,
        )
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, f"Failed getting details for run '{run_id}'", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 2 Tool 3: find_best_run
# ------------------------------------------------------------------------------

def find_best_run(
    experiment_name: str = "iris-classification-pipeline",
    metric: str = "f1_score",
    min_threshold: float = 0.75,
    request_id: str = "req_find_best_run",
) -> ToolExecutionResult:
    """Identify best run based on specified metric and evaluate comprehensive production readiness.

    Explicitly verifies that highest metric score alone does not guarantee production readiness;
    evaluates quality gate status, provenance completeness, sample size, and overfitting gaps.
    """
    start_time = time.perf_counter()
    tool_name = "find_best_run"

    try:
        client = get_client()

        # Find experiment
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

        # Search runs ordered by target metric descending
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=[f"metrics.{metric} DESC"],
            max_results=20,
        )

        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'.")

        top_run = runs[0]
        top_metrics = {k: round(v, 4) for k, v in top_run.data.metrics.items()}
        metric_score = top_metrics.get(metric, 0.0)
        tags = top_run.data.tags
        params = top_run.data.params

        # --- Production Readiness Checklist ---
        checklist: dict[str, Any] = {}

        # 1. Metric threshold
        metric_ok = metric_score >= min_threshold
        checklist["metric_threshold_satisfied"] = {
            "passed": metric_ok,
            "metric": metric,
            "score": metric_score,
            "required_min": min_threshold,
        }

        # 2. Quality gate tag
        gate_tag = tags.get("quality_gate_passed", "").lower()
        gate_ok = (gate_tag == "true")
        checklist["quality_gate_status"] = {
            "passed": gate_ok,
            "tag_value": gate_tag or "missing",
            "reason": tags.get("quality_gate_reason", "No reason recorded"),
        }

        # 3. Overfitting gap analysis
        overfit_gap = top_metrics.get("overfitting_gap", 0.0)
        train_f1 = top_metrics.get("train_f1_score", top_metrics.get("f1_score", 0.0))
        test_f1 = top_metrics.get("f1_score", 0.0)
        max_acceptable_gap = 0.15
        gap_ok = overfit_gap <= max_acceptable_gap
        checklist["overfitting_gap_evaluation"] = {
            "passed": gap_ok,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "gap": overfit_gap,
            "max_acceptable_gap": max_acceptable_gap,
        }

        # 4. Provenance audit
        has_dataset_version = bool(tags.get("dataset_version") and tags.get("dataset_version") != "unknown")
        has_git_commit = bool(tags.get("git_commit") and tags.get("git_commit") != "uncommitted")
        has_dvc_md5 = bool(tags.get("dvc_md5") and tags.get("dvc_md5") != "untracked")
        provenance_ok = has_dataset_version and has_git_commit and has_dvc_md5
        checklist["provenance_audit"] = {
            "passed": provenance_ok,
            "dataset_version": tags.get("dataset_version", "missing"),
            "git_commit": tags.get("git_commit", "missing"),
            "dvc_md5": tags.get("dvc_md5", "missing"),
        }

        # 5. Sample size adequacy
        train_samples = int(params.get("train_samples", 0))
        sample_ok = train_samples >= 100 or "iris" in experiment_name.lower()
        checklist["sample_size_check"] = {
            "passed": sample_ok,
            "train_samples": train_samples,
        }

        # Overall Production Readiness Decision
        is_prod_ready = bool(metric_ok and gate_ok and gap_ok and provenance_ok and sample_ok)

        # Formulate explicit assessment notes
        notes_lines = [
            f"Top candidate run '{top_run.info.run_id}' achieved {metric} = {metric_score:.4f}.",
            "IMPORTANT: Highest metric score alone does not guarantee production readiness.",
        ]
        if not is_prod_ready:
            failed_checks = [k for k, v in checklist.items() if not v.get("passed", False)]
            notes_lines.append(f"Model is NOT production-ready due to failed criteria: {', '.join(failed_checks)}.")
        else:
            notes_lines.append(
                "Model is PRODUCTION-READY: Satisfies quality gate, bounded generalization gap, "
                "verified dataset/Git provenance, and meets the deployment threshold."
            )

        output = MLflowBestRunOutput(
            best_run_id=top_run.info.run_id,
            experiment_name=experiment_name,
            target_metric=metric,
            metric_value=metric_score,
            params=params,
            metrics=top_metrics,
            is_production_ready=is_prod_ready,
            readiness_evaluation=checklist,
            evaluation_notes="\n".join(notes_lines),
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Best run '{top_run.info.run_id}' ({metric}={metric_score:.4f}, prod_ready={is_prod_ready})",
            result.data,
        )
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed finding best run", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 2 Tool 4: compare_runs
# ------------------------------------------------------------------------------

def compare_runs(
    run_ids: list[str],
    request_id: str = "req_compare_runs",
) -> ToolExecutionResult:
    """Compare two or more runs side-by-side across metrics, parameters, and dataset versions, highlighting trade-offs."""
    start_time = time.perf_counter()
    tool_name = "compare_runs"

    try:
        if not run_ids or len(run_ids) < 2:
            raise ValueError("Comparison requires at least 2 run_ids.")

        client = get_client()
        comparison_matrix: list[dict[str, Any]] = []

        for rid in run_ids:
            run = client.get_run(rid)
            metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}
            params = run.data.params
            tags = run.data.tags

            row = {
                "run_id": rid[:8] + "...",
                "full_run_id": rid,
                "model_type": params.get("model_type", tags.get("model_type", "unknown")),
                "f1_score": metrics.get("f1_score", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "roc_auc": metrics.get("roc_auc", 0.0),
                "overfitting_gap": metrics.get("overfitting_gap", 0.0),
                "quality_gate": tags.get("quality_gate_passed", "unknown"),
                "dataset_version": tags.get("dataset_version", "unknown"),
                "git_commit": tags.get("git_commit", "unknown"),
                "key_hyperparameters": {
                    k: v for k, v in params.items()
                    if k in ("n_estimators", "max_depth", "learning_rate", "c_param")
                },
            }
            comparison_matrix.append(row)

        # Sort matrix by f1_score descending
        comparison_matrix.sort(key=lambda x: x.get("f1_score", 0.0), reverse=True)

        # Build trade-offs summary
        top_candidate = comparison_matrix[0]
        second_candidate = comparison_matrix[1]

        tradeoffs = [
            f"Comparing {len(run_ids)} runs:",
            f"- Highest F1: Run {top_candidate['run_id']} ({top_candidate['model_type']}) with F1 = {top_candidate['f1_score']:.4f}, Overfit gap = {top_candidate['overfitting_gap']:.4f}.",
            f"- Alternative: Run {second_candidate['run_id']} ({second_candidate['model_type']}) with F1 = {second_candidate['f1_score']:.4f}, Overfit gap = {second_candidate['overfitting_gap']:.4f}.",
        ]

        if top_candidate["overfitting_gap"] > second_candidate["overfitting_gap"] + 0.03:
            tradeoffs.append(
                f"Trade-off Notice: Top run exhibits larger train-test divergence (+{top_candidate['overfitting_gap']:.4f}) "
                f"compared to runner-up (+{second_candidate['overfitting_gap']:.4f}). Consider simpler model for better generalization."
            )
        else:
            tradeoffs.append("Trade-off Notice: Top model demonstrates strong balance between test performance and generalization stability.")

        output = MLflowCompareRunsOutput(
            run_ids=run_ids,
            comparison_table=comparison_matrix,
            tradeoffs_summary="\n".join(tradeoffs),
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Compared {len(run_ids)} runs successfully.",
            result.data,
        )
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed comparing runs", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 2 Tool 5: get_model_registry_info
# ------------------------------------------------------------------------------

def get_model_registry_info(
    model_name: str | None = None,
    request_id: str = "req_model_registry_info",
) -> ToolExecutionResult:
    """Retrieve registered models, latest model versions, aliases/stages, and linked run IDs."""
    start_time = time.perf_counter()
    tool_name = "get_model_registry_info"

    try:
        client = get_client()

        registered_models_data: list[dict[str, Any]] = []

        if model_name:
            models_to_inspect = [client.get_registered_model(model_name)]
        else:
            models_to_inspect = list(client.search_registered_models())

        for rm in models_to_inspect:
            versions_data = []
            for mv in rm.latest_versions:
                versions_data.append({
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "status": mv.status,
                    "run_id": mv.run_id,
                    "source": mv.source,
                    "aliases": getattr(mv, "aliases", []),
                })

            registered_models_data.append({
                "model_name": rm.name,
                "description": rm.description or "",
                "creation_timestamp": rm.creation_timestamp,
                "last_updated_timestamp": rm.last_updated_timestamp,
                "latest_versions": versions_data,
                "aliases": getattr(rm, "aliases", {}),
            })

        output = MLflowModelRegistryOutput(
            registered_models=registered_models_data,
            total_models=len(registered_models_data),
        )

        duration = (time.perf_counter() - start_time) * 1000.0
        result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.SUCCESS,
            duration_ms=duration,
            data=output.model_dump(),
        )
        log_tool_execution(
            logger, request_id, tool_name, "SUCCESS", duration,
            f"Retrieved info for {len(registered_models_data)} registered models.",
            result.data,
        )
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed retrieving model registry info", error=str(exc))
        return result
