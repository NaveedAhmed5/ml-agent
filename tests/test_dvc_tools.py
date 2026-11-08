"""Unit tests for DVC Tools (Task 1: DVC Data Management Agent)."""

import pytest
from agent.schemas import ToolExecutionStatus
from agent.tools.dvc_tools import (
    compare_dataset_versions,
    detect_data_changes,
    explain_dataset_reproduction,
    get_current_dataset_version,
    list_dvc_tracked_datasets,
    validate_command_allowlist,
)


def test_command_allowlist_security():
    """Verify that only strictly permitted commands can execute."""
    # Valid commands
    assert "dvc" in validate_command_allowlist(["dvc", "status"])[0]
    assert "git" in validate_command_allowlist(["git", "rev-parse", "HEAD"])[0]

    # Blocked root commands
    with pytest.raises(PermissionError, match="not in allowed root commands"):
        validate_command_allowlist(["rm", "-rf", "/"])

    with pytest.raises(PermissionError, match="not in allowed root commands"):
        validate_command_allowlist(["curl", "https://malicious.com"])

    # Blocked subcommands
    with pytest.raises(PermissionError, match="subcommand .* is not allowed"):
        validate_command_allowlist(["git", "commit", "-m", "bad"])

    with pytest.raises(PermissionError, match="subcommand .* is not allowed"):
        validate_command_allowlist(["dvc", "destroy"])

    # Injection prevention
    with pytest.raises(ValueError, match="illegal characters"):
        validate_command_allowlist(["dvc", "status", ";", "calc.exe"])

    with pytest.raises(ValueError, match="illegal characters"):
        validate_command_allowlist(["git", "log", "|", "cat"])


def test_list_dvc_tracked_datasets():
    """Verify listing of DVC tracked datasets."""
    result = list_dvc_tracked_datasets()
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data is not None
    assert result.data["total_datasets"] >= 1
    paths = [d["dataset_path"] for d in result.data["tracked_datasets"]]
    assert any("data/dataset.csv" in p for p in paths)


def test_get_current_dataset_version():
    """Verify retrieving current dataset version and metadata."""
    result = get_current_dataset_version("data/dataset.csv")
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data is not None
    assert result.data["dataset_path"] == "data/dataset.csv"
    assert len(result.data["md5_hash"]) > 0
    assert result.data["dvc_remote"] == "local_remote"


def test_compare_dataset_versions():
    """Verify version comparison functionality."""
    result = compare_dataset_versions("HEAD", "HEAD")
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data is not None
    assert result.data["is_identical"] is True


def test_detect_data_changes():
    """Verify detecting DVC workspace modification state."""
    result = detect_data_changes()
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data is not None
    assert isinstance(result.data["has_changes"], bool)


def test_explain_dataset_reproduction():
    """Verify generation of exact reproduction command sequence."""
    result = explain_dataset_reproduction("f32efd9")
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data is not None
    commands = result.data["commands"]
    assert len(commands) == 4
    assert "git checkout f32efd9 -- data/dataset.csv.dvc" in commands[0]
    assert "dvc checkout data/dataset.csv.dvc" in commands[1]
    assert "dvc pull" in commands[2]
    assert "dvc repro" in commands[3]

