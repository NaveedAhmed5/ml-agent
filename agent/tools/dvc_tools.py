"""DVC Tools for MLOps AI Agent (Task 1: DVC Data Management Agent).

Provides safe, allowlisted subprocess interactions with DVC and Git to inspect,
track, compare, and reproduce dataset versions. Every call returns a standardized
ToolExecutionResult and logs structured JSON with credential redaction.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import yaml

from agent.logger import get_logger, log_tool_execution
from agent.schemas import (
    DVCCompareVersionsOutput,
    DVCDatasetVersionOutput,
    DVCDetectChangesOutput,
    DVCListDatasetsOutput,
    DVCReproductionStepsOutput,
    ToolExecutionResult,
    ToolExecutionStatus,
)

logger = get_logger("dvc_tools")

# ------------------------------------------------------------------------------
# Security: Strict Subprocess Command Allowlisting
# ------------------------------------------------------------------------------

ALLOWED_ROOT_COMMANDS = {"dvc", "git"}
ALLOWED_DVC_SUBCOMMANDS = {
    "status", "list", "diff", "metrics", "params", "remote",
    "checkout", "pull", "push", "repro", "dag", "version",
}
ALLOWED_GIT_SUBCOMMANDS = {
    "rev-parse", "log", "status", "show", "diff",
}
FORBIDDEN_CHARACTERS = re.compile(r"[;&|`$><\n\r]")


def resolve_executable(name: str) -> str:
    """Find the full executable path, checking virtualenv first."""
    # Check venv Scripts or bin
    venv_dir = os.environ.get("VIRTUAL_ENV") or str(Path(".venv").resolve())
    for candidate in [
        Path(venv_dir) / "Scripts" / f"{name}.exe",
        Path(venv_dir) / "bin" / name,
        Path(venv_dir) / f"{name}.exe",
    ]:
        if candidate.is_file():
            return str(candidate)

    # Fallback to system PATH
    found = shutil.which(name)
    if found:
        return found
    return name


def validate_command_allowlist(cmd_args: list[str]) -> list[str]:
    """Validate that the requested command is strictly permitted.

    Raises:
        PermissionError: If command or subcommand violates security allowlist.
        ValueError: If arguments contain shell injection characters.
    """
    if not cmd_args:
        raise ValueError("Command arguments cannot be empty.")

    raw_binary = Path(cmd_args[0]).stem.lower()
    if raw_binary not in ALLOWED_ROOT_COMMANDS:
        raise PermissionError(
            f"Command execution blocked: '{raw_binary}' is not in allowed root commands: {ALLOWED_ROOT_COMMANDS}"
        )

    if len(cmd_args) > 1:
        subcmd = cmd_args[1].lower().strip()
        if raw_binary == "dvc" and subcmd not in ALLOWED_DVC_SUBCOMMANDS:
            raise PermissionError(
                f"DVC subcommand '{subcmd}' is not allowed. Permitted: {sorted(ALLOWED_DVC_SUBCOMMANDS)}"
            )
        elif raw_binary == "git" and subcmd not in ALLOWED_GIT_SUBCOMMANDS:
            raise PermissionError(
                f"Git subcommand '{subcmd}' is not allowed. Permitted: {sorted(ALLOWED_GIT_SUBCOMMANDS)}"
            )

    # Validate against shell injection characters in arguments
    for arg in cmd_args:
        if FORBIDDEN_CHARACTERS.search(arg):
            raise ValueError(f"Security validation failed: argument '{arg}' contains illegal characters.")

    # Substitute resolved executable path for index 0
    resolved_args = list(cmd_args)
    resolved_args[0] = resolve_executable(raw_binary)
    return resolved_args


def run_safe_subprocess(cmd_args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Execute an allowlisted command safely without shell=True."""
    safe_args = validate_command_allowlist(cmd_args)
    return subprocess.run(
        safe_args,
        capture_output=True,
        text=True,
        shell=False,
        timeout=timeout,
        check=False,
    )


# ------------------------------------------------------------------------------
# Task 1 Tool 1: list_dvc_tracked_datasets
# ------------------------------------------------------------------------------

def list_dvc_tracked_datasets(request_id: str = "req_list_datasets") -> ToolExecutionResult:
    """Identify all datasets and .dvc files tracked in the repository."""
    start_time = time.perf_counter()
    tool_name = "list_dvc_tracked_datasets"

    try:
        tracked_list: list[dict[str, Any]] = []

        # 1. Scan for *.dvc files in the repository
        for dvc_file in Path(".").glob("**/*.dvc"):
            # Avoid directories and .dvc internal folder
            if not dvc_file.is_file() or ".dvc" in dvc_file.parts[:-1] or ".venv" in dvc_file.parts:
                continue

            try:
                with open(dvc_file, encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                outs = content.get("outs", [])
                for out in outs:
                    rel_data_path = (dvc_file.parent / out.get("path", "")).as_posix()
                    tracked_list.append({
                        "dataset_path": rel_data_path,
                        "dvc_file": dvc_file.as_posix(),
                        "md5": out.get("md5"),
                        "size_bytes": out.get("size"),
                        "type": "dvc_tracked_file",
                    })
            except Exception as exc:
                logger.warning(f"Failed parsing .dvc file {dvc_file}: {exc}")

        # 2. Check dvc.yaml stages for input datasets
        dvc_yaml_path = Path("dvc.yaml")
        if dvc_yaml_path.exists():
            try:
                with open(dvc_yaml_path, encoding="utf-8") as f:
                    dvc_yaml = yaml.safe_load(f) or {}
                for stage_name, stage_info in dvc_yaml.get("stages", {}).items():
                    for dep in stage_info.get("deps", []):
                        if str(dep).endswith((".csv", ".parquet", ".json", ".tsv")):
                            tracked_list.append({
                                "dataset_path": str(dep),
                                "stage": stage_name,
                                "type": "pipeline_dependency",
                            })
            except Exception as exc:
                logger.warning(f"Failed parsing dvc.yaml: {exc}")

        # Deduplicate by dataset_path
        unique_datasets: dict[str, dict[str, Any]] = {}
        for item in tracked_list:
            path = item.get("dataset_path", "")
            if path and path not in unique_datasets:
                unique_datasets[path] = item

        output = DVCListDatasetsOutput(
            tracked_datasets=list(unique_datasets.values()),
            total_datasets=len(unique_datasets),
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
            f"Found {output.total_datasets} tracked datasets in repository.",
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed listing DVC datasets", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 1 Tool 2: get_current_dataset_version
# ------------------------------------------------------------------------------

def get_current_dataset_version(
    dataset_path: str = "data/dataset.csv",
    request_id: str = "req_dataset_version",
) -> ToolExecutionResult:
    """Retrieve current dataset version, MD5 hash, associated Git commit, and DVC remote."""
    start_time = time.perf_counter()
    tool_name = "get_current_dataset_version"

    try:
        data_p = Path(dataset_path)
        dvc_file = data_p.with_suffix(data_p.suffix + ".dvc")

        md5_hash = "N/A"
        size_bytes = 0
        if dvc_file.exists():
            with open(dvc_file, encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
            outs = content.get("outs", [])
            if outs:
                md5_hash = outs[0].get("md5", "N/A")
                size_bytes = outs[0].get("size", 0)

        # Inspect dataset semantic version marker if available
        version_label = "unknown"
        if data_p.exists():
            try:
                import pandas as pd
                df_head = pd.read_csv(data_p, nrows=5)
                if "dataset_version" in df_head.columns:
                    version_label = str(df_head["dataset_version"].iloc[0])
            except Exception:
                pass

        # Query Git commit hash safely
        git_commit = "uncommitted"
        proc_git = run_safe_subprocess(["git", "rev-parse", "--short", "HEAD"])
        if proc_git.returncode == 0 and proc_git.stdout.strip():
            git_commit = proc_git.stdout.strip()

        # Query configured DVC remote
        dvc_remote = "local_remote"
        proc_remote = run_safe_subprocess(["dvc", "remote", "list"])
        if proc_remote.returncode == 0 and proc_remote.stdout.strip():
            lines = proc_remote.stdout.strip().splitlines()
            for line in lines:
                parts = line.split()
                if parts:
                    dvc_remote = parts[0]
                    break

        output = DVCDatasetVersionOutput(
            dataset_path=data_p.as_posix(),
            dvc_file=dvc_file.as_posix() if dvc_file.exists() else "none",
            md5_hash=md5_hash,
            size_bytes=size_bytes,
            dataset_version_label=version_label,
            git_commit=git_commit,
            dvc_remote=dvc_remote,
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
            f"Dataset '{dataset_path}' version: {version_label} (md5: {md5_hash[:8]}..., git: {git_commit})",
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed getting dataset version", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 1 Tool 3: compare_dataset_versions
# ------------------------------------------------------------------------------

def compare_dataset_versions(
    rev1: str,
    rev2: str = "HEAD",
    dataset_path: str = "data/dataset.csv",
    request_id: str = "req_compare_versions",
) -> ToolExecutionResult:
    """Compare dataset versions between two Git revisions or states using dvc diff / dvc.lock inspection."""
    start_time = time.perf_counter()
    tool_name = "compare_dataset_versions"

    try:
        # Check dvc diff using safe allowlisted subprocess
        proc = run_safe_subprocess(["dvc", "diff", rev1, rev2])
        diff_stdout = proc.stdout.strip()

        # Extract hash at rev1 vs rev2 using git show on the .dvc file
        dvc_file_rel = Path(dataset_path).with_suffix(Path(dataset_path).suffix + ".dvc").as_posix()

        def get_dvc_meta_at_rev(rev: str) -> tuple[str | None, int | None]:
            # Try git show <rev>:<dvc_file>
            p = run_safe_subprocess(["git", "show", f"{rev}:{dvc_file_rel}"])
            if p.returncode == 0:
                try:
                    data = yaml.safe_load(p.stdout)
                    outs = data.get("outs", [])
                    if outs:
                        return outs[0].get("md5"), outs[0].get("size")
                except Exception:
                    pass
            # Fallback to local .dvc file if rev == "HEAD" or "working"
            if rev in ("HEAD", "working", "current") and Path(dvc_file_rel).exists():
                try:
                    with open(dvc_file_rel, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    outs = data.get("outs", [])
                    if outs:
                        return outs[0].get("md5"), outs[0].get("size")
                except Exception:
                    pass
            return None, None

        hash1, size1 = get_dvc_meta_at_rev(rev1)
        hash2, size2 = get_dvc_meta_at_rev(rev2)

        is_identical = (hash1 == hash2) if (hash1 and hash2) else (diff_stdout == "")
        size_delta = (size2 - size1) if (size1 is not None and size2 is not None) else None

        summary_parts = []
        if is_identical:
            summary_parts.append(f"Revisions '{rev1}' and '{rev2}' have identical dataset hashes.")
        else:
            summary_parts.append(f"Dataset changed between '{rev1}' ({hash1 or 'unknown'}) and '{rev2}' ({hash2 or 'unknown'}).")
            if size_delta is not None:
                summary_parts.append(f"Size difference: {size_delta:+d} bytes.")

        if diff_stdout:
            summary_parts.append(f"DVC Diff Details:\n{diff_stdout}")

        output = DVCCompareVersionsOutput(
            rev1=rev1,
            rev2=rev2,
            hash_rev1=hash1,
            hash_rev2=hash2,
            size_delta_bytes=size_delta,
            is_identical=is_identical,
            diff_summary="\n".join(summary_parts),
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
            f"Compared versions '{rev1}' vs '{rev2}': identical={is_identical}",
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed comparing dataset versions", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 1 Tool 4: detect_data_changes
# ------------------------------------------------------------------------------

def detect_data_changes(request_id: str = "req_detect_changes") -> ToolExecutionResult:
    """Run dvc status to detect whether tracked data or pipeline dependencies have changed."""
    start_time = time.perf_counter()
    tool_name = "detect_data_changes"

    try:
        proc = run_safe_subprocess(["dvc", "status"])
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # DVC returns exit code 0 if clean, or status table if modified
        # "Data and pipelines are up to date." indicates clean state
        is_clean = (
            "Data and pipelines are up to date." in stdout
            or "Data and pipelines are up to date." in stderr
            or (proc.returncode == 0 and not stdout and not stderr)
        )

        changed_items: list[str] = []
        if not is_clean:
            for line in stdout.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith(("Stage", "---", "+--")):
                    changed_items.append(stripped)

        has_changes = not is_clean
        summary = (
            "Data and pipeline stages are completely up to date."
            if not has_changes
            else f"Detected {len(changed_items)} modified or untracked stages/files in DVC workspace."
        )

        output = DVCDetectChangesOutput(
            has_changes=has_changes,
            changed_items=changed_items,
            status_summary=summary,
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
            f"DVC status checked: has_changes={has_changes}",
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed checking DVC status", error=str(exc))
        return result


# ------------------------------------------------------------------------------
# Task 1 Tool 5: explain_dataset_reproduction
# ------------------------------------------------------------------------------

def explain_dataset_reproduction(
    target_version_or_commit: str,
    dataset_path: str = "data/dataset.csv",
    request_id: str = "req_explain_repro",
) -> ToolExecutionResult:
    """Return exact, verified step-by-step commands and provenance details to reproduce a previous dataset version."""
    start_time = time.perf_counter()
    tool_name = "explain_dataset_reproduction"

    try:
        data_p = Path(dataset_path)
        dvc_file = data_p.with_suffix(data_p.suffix + ".dvc").as_posix()

        # Build exact, deterministic sequence of shell commands
        commands = [
            f"git checkout {target_version_or_commit} -- {dvc_file}",
            f"dvc checkout {dvc_file}",
            "dvc pull",
            "dvc repro",
        ]

        # Inspect remote provenance
        configured_remote = "local_remote"
        proc_remotes = run_safe_subprocess(["dvc", "remote", "list"])
        if proc_remotes.returncode == 0 and proc_remotes.stdout.strip():
            first_line = proc_remotes.stdout.strip().splitlines()[0]
            parts = first_line.split()
            if parts:
                configured_remote = parts[0]

        provenance = {
            "target_revision": target_version_or_commit,
            "dataset_file": dataset_path,
            "dvc_pointer_file": dvc_file,
            "remote_storage": configured_remote,
            "verification_command": f"dvc status {dvc_file}",
            "repro_pipeline_command": "dvc repro train",
        }

        output = DVCReproductionStepsOutput(
            target_revision=target_version_or_commit,
            dataset_path=dataset_path,
            commands=commands,
            provenance_details=provenance,
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
            f"Reproduction steps generated for revision '{target_version_or_commit}'",
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
        log_tool_execution(logger, request_id, tool_name, "ERROR", duration, "Failed explaining dataset reproduction", error=str(exc))
        return result
