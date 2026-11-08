"""Structured JSON Logger with Secret Redaction for MLOps AI Agent.

This module provides structured JSON logging with built-in credential
masking across all log records, tool executions, and arbitrary payload dictionaries.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

# Sensitive key patterns to redact
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i)(token|key|password|secret|credential|auth|bearer|private_key)"
)

# Common regex patterns to detect and mask secrets in unstructured strings
SECRET_IN_STRING_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-\._~\+\/]+=*)"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)([A-Za-z0-9\-\._~]+)(['\"]?)"),
    re.compile(r"(?i)(password\s*[:=]\s*['\"]?)([^\s'\"]+)(['\"]?)"),
    re.compile(r"(?i)(token\s*[:=]\s*['\"]?)([A-Za-z0-9\-\._~]+)(['\"]?)"),
    re.compile(r"(?i)(secret\s*[:=]\s*['\"]?)([A-Za-z0-9\-\._~]+)(['\"]?)"),
]

MASK_VALUE = "********"


def redact_sensitive_data(data: Any) -> Any:
    """Recursively mask sensitive values in dictionaries, lists, and strings.

    Any dictionary key matching `TOKEN`, `KEY`, `PASSWORD`, or `SECRET`
    (case-insensitive) will have its value replaced with `********`.
    Strings are also scanned for common secret patterns.
    """
    if isinstance(data, Mapping):
        redacted_dict: dict[str, Any] = {}
        for key, value in data.items():
            str_key = str(key)
            if SENSITIVE_KEY_PATTERN.search(str_key):
                redacted_dict[key] = MASK_VALUE
            else:
                redacted_dict[key] = redact_sensitive_data(value)
        return redacted_dict

    if isinstance(data, (list, tuple, set)):
        redacted_list = [redact_sensitive_data(item) for item in data]
        return type(data)(redacted_list) if not isinstance(data, (set, tuple)) else list(redacted_list)

    if isinstance(data, str):
        sanitized = data
        for pattern in SECRET_IN_STRING_PATTERNS:
            # Replace capture group 2 with MASK_VALUE while preserving capture group 1 and 3
            if pattern.groups == 3:
                sanitized = pattern.sub(rf"\g<1>{MASK_VALUE}\g<3>", sanitized)
            elif pattern.groups == 2:
                sanitized = pattern.sub(rf"\g<1>{MASK_VALUE}", sanitized)
        return sanitized

    return data


class SecretRedactionFilter(logging.Filter):
    """Logging filter that redacts secrets from LogRecord messages and attributes."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_data(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_sensitive_data(record.args)
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(redact_sensitive_data(arg) for arg in record.args)

        # Inspect extra attributes attached to record
        for key in list(record.__dict__.keys()):
            if SENSITIVE_KEY_PATTERN.search(key):
                setattr(record, key, MASK_VALUE)
            elif key in ("data", "payload", "details", "extra", "params", "result"):
                val = getattr(record, key)
                setattr(record, key, redact_sensitive_data(val))

        return True


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Standard structured fields
        log_entry: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "N/A"),
            "tool_name": getattr(record, "tool_name", "N/A"),
            "status": getattr(record, "status", "N/A"),
            "duration_ms": getattr(record, "duration_ms", 0.0),
            "message": record.getMessage(),
        }

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include optional metadata if present
        for optional_field in ("data", "error", "metadata", "details"):
            if hasattr(record, optional_field):
                val = getattr(record, optional_field)
                if val is not None:
                    log_entry[optional_field] = redact_sensitive_data(val)

        return json.dumps(log_entry, default=str)


def get_logger(
    name: str = "mlops_agent",
    level: int = logging.INFO,
    stream: Any = sys.stdout,
) -> logging.Logger:
    """Configures and returns a structured JSON logger with secret redaction."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(JSONFormatter())
        handler.addFilter(SecretRedactionFilter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def log_tool_execution(
    logger: logging.Logger,
    request_id: str,
    tool_name: str,
    status: str,
    duration_ms: float,
    message: str,
    data: Any = None,
    error: str | None = None,
) -> None:
    """Helper to emit structured tool execution log entries."""
    extra = {
        "request_id": request_id,
        "tool_name": tool_name,
        "status": status,
        "duration_ms": duration_ms,
        "data": redact_sensitive_data(data),
        "error": error,
    }

    if status.upper() == "SUCCESS":
        logger.info(message, extra=extra)
    elif status.upper() == "BLOCKED":
        logger.warning(message, extra=extra)
    else:
        logger.error(message, extra=extra)

