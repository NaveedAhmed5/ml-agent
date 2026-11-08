"""Unit tests for MLflow Experiment Intelligence Tools (Task 2 — 15 Marks)."""

import pytest
from agent.schemas import ToolExecutionStatus
from agent.tools.mlflow_tools import (
    compare_runs,
    find_best_run,
    get_model_registry_info,
    get_run_details,
    list_experiments_and_runs,
)


@pytest.fixture(scope="module")
def experiment_runs():
    """Fixture to ensure runs exist in iris-classification-pipeline."""
    res = list_experiments_and_runs("iris-classification-pipeline")
    assert res.status == ToolExecutionStatus.SUCCESS
    runs = res.data["runs"]
    assert len(runs) >= 5, f"Expected at least 5 runs, found {len(runs)}"
    return runs


def test_list_experiments_and_runs(experiment_runs):
    """Test listing active experiments and recent runs."""
    res = list_experiments_and_runs()
    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data is not None
    assert len(res.data["experiments"]) >= 1
    assert any("iris-classification-pipeline" in exp["name"] for exp in res.data["experiments"])
    assert len(res.data["runs"]) >= 5


def test_get_run_details(experiment_runs):
    """Test fetching comprehensive metrics, parameters, and provenance tags."""
    target_run_id = experiment_runs[0]["run_id"]
    res = get_run_details(target_run_id)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data is not None
    assert res.data["run_id"] == target_run_id
    assert "f1_score" in res.data["metrics"]
    assert "accuracy" in res.data["metrics"]

    provenance = res.data["provenance"]
    assert "dataset_version" in provenance
    assert "git_commit" in provenance
    assert "dvc_md5" in provenance
    assert "quality_gate_passed" in provenance


def test_find_best_run():
    """Test identifying top run and evaluating multi-criteria production readiness."""
    res = find_best_run("iris-classification-pipeline", metric="f1_score", min_threshold=0.75)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data is not None
    assert len(res.data["best_run_id"]) > 0
    assert res.data["target_metric"] == "f1_score"
    assert res.data["metric_value"] >= 0.75

    # Production-readiness criteria audit
    readiness = res.data["readiness_evaluation"]
    assert "metric_threshold_satisfied" in readiness
    assert "quality_gate_status" in readiness
    assert "overfitting_gap_evaluation" in readiness
    assert "provenance_audit" in readiness

    # Notes must explicitly mention that highest metric alone is insufficient
    notes = res.data["evaluation_notes"]
    assert "Highest metric score alone does not guarantee production readiness" in notes


def test_compare_runs(experiment_runs):
    """Test side-by-side run comparison and trade-offs summary."""
    run_ids = [r["run_id"] for r in experiment_runs[:3]]
    res = compare_runs(run_ids)

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data is not None
    assert len(res.data["comparison_table"]) == len(run_ids)

    table = res.data["comparison_table"]
    for row in table:
        assert "f1_score" in row
        assert "overfitting_gap" in row
        assert "model_type" in row
        assert "dataset_version" in row

    assert "Trade-off" in res.data["tradeoffs_summary"]


def test_compare_runs_invalid_input():
    """Test error handling when fewer than 2 run IDs are provided."""
    res = compare_runs(["single_run_id"])
    assert res.status == ToolExecutionStatus.ERROR
    assert "at least 2 run_ids" in str(res.error)


def test_get_model_registry_info():
    """Test inspecting registered models in MLflow Model Registry."""
    res = get_model_registry_info("iris-classifier")

    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data is not None
    assert res.data["total_models"] >= 1
    model_entry = res.data["registered_models"][0]
    assert model_entry["model_name"] == "iris-classifier"
    assert len(model_entry["latest_versions"]) >= 1

