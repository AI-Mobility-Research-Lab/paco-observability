"""Capture a small, non-sensitive receipt for an analysis runtime.

The receipt intentionally uses a fixed allow-list of system properties.  In
particular, it never serializes the process environment; the only environment
variable consulted indirectly is ``SOURCE_DATE_EPOCH`` for a reproducible
timestamp.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform as platform_module
import re
import socket
import tempfile
from typing import Any

from .release_manifest import (
    executable_version,
    git_state,
    package_versions,
    utc_now,
    validate_numeric_producer_declaration,
)


RUNTIME_PACKAGE_NAMES = ("numpy", "pandas", "pyarrow")
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


def _normalize_declared_source_commit(value: str | None) -> str:
    if value is None:
        raise ValueError(
            "declared_source_commit is required when source_tree_without_git is true"
        )
    commit = value.strip().lower()
    if _FULL_GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("declared_source_commit must be a full 40-character hexadecimal commit")
    return commit


def linux_cpu_model(
    *,
    system_name: str | None = None,
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
) -> str | None:
    """Return the first Linux CPU model in ``/proc/cpuinfo``, if available.

    ``model name`` is used on x86 systems.  The conservative fallbacks cover
    common ARM ``/proc/cpuinfo`` layouts without exposing serial numbers or
    any other host identifiers.
    """

    system = platform_module.system() if system_name is None else system_name
    if system != "Linux":
        return None
    try:
        contents = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    fields: dict[str, str] = {}
    for line in contents.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_value and normalized_key not in fields:
            fields[normalized_key] = normalized_value
    for key in ("model name", "hardware"):
        if key in fields:
            return fields[key]
    processor = fields.get("processor")
    if processor is not None and not processor.isdecimal():
        return processor
    return None


def build_runtime_environment_receipt(
    *,
    label: str,
    repository: Path,
    generated_at_utc: str | None = None,
    source_tree_without_git: bool = False,
    declared_source_commit: str | None = None,
    execution_mode: str = "external_python",
    numeric_producer_config: Path | None = None,
) -> dict[str, Any]:
    """Build a runtime receipt from an explicit, non-sensitive allow-list.

    The execution checkout and the frozen numerical producer are deliberately
    separate concepts.  ``git_*`` describes the checkout that invoked this
    receipt writer, including its actual HEAD and dirty state.  When supplied,
    ``numeric_producer_config`` is independently validated byte-for-byte
    against its declared historical commit and stored under
    ``numeric_producer``.
    """

    normalized_label = label.strip()
    if not normalized_label:
        raise ValueError("label must not be empty")
    if execution_mode not in {"uv_frozen", "external_python"}:
        raise ValueError("execution_mode must be 'uv_frozen' or 'external_python'")
    if source_tree_without_git and numeric_producer_config is not None:
        raise ValueError(
            "numeric_producer_config requires a Git repository for byte verification"
        )

    versions = package_versions(RUNTIME_PACKAGE_NAMES)
    if source_tree_without_git:
        declared_commit = _normalize_declared_source_commit(declared_source_commit)
        revision: dict[str, Any] = {
            "commit": None,
            "branch": None,
            "dirty": None,
        }
        repository_available = False
        provenance_status = "declared_unverified_source_tree_without_git"
        state_timing = "source_tree_without_git_declaration"
    else:
        if declared_source_commit is not None:
            raise ValueError(
                "declared_source_commit is only valid when source_tree_without_git is true"
            )
        revision = git_state(repository)
        declared_commit = None
        repository_available = True
        provenance_status = "repository_verified"
        state_timing = "before_receipt_write"
    numeric_producer = (
        validate_numeric_producer_declaration(
            base_dir=repository,
            config_path=numeric_producer_config,
        )
        if numeric_producer_config is not None
        else None
    )
    if numeric_producer_config is not None and numeric_producer is None:
        raise ValueError(
            "numeric_producer_config does not declare release.numeric_producer_commit "
            "and release.numeric_producer_files"
        )
    return {
        "schema_version": "1.1",
        "label": normalized_label,
        "generated_at_utc": generated_at_utc or utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform_module.platform(),
        "machine": platform_module.machine() or None,
        "cpu_model": linux_cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "python_implementation": platform_module.python_implementation(),
        "python_version": platform_module.python_version(),
        "numpy_version": versions["numpy"],
        "pandas_version": versions["pandas"],
        "pyarrow_version": versions["pyarrow"],
        "execution_mode": execution_mode,
        "uv_version": executable_version("uv"),
        "git_repository_available": repository_available,
        "git_provenance_status": provenance_status,
        "declared_source_commit": declared_commit,
        "git_commit": revision["commit"],
        "git_branch": revision["branch"],
        "git_dirty": revision["dirty"],
        "git_state_timing": state_timing,
        "git_commit_role": "execution_source_tree_at_receipt_capture",
        "numeric_producer": numeric_producer,
    }


def write_runtime_environment_receipt(
    *,
    label: str,
    repository: Path,
    output_path: Path,
    generated_at_utc: str | None = None,
    source_tree_without_git: bool = False,
    declared_source_commit: str | None = None,
    execution_mode: str = "external_python",
    numeric_producer_config: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically write a runtime receipt as deterministic JSON."""

    receipt = build_runtime_environment_receipt(
        label=label,
        repository=repository,
        generated_at_utc=generated_at_utc,
        source_tree_without_git=source_tree_without_git,
        declared_source_commit=declared_source_commit,
        execution_mode=execution_mode,
        numeric_producer_config=numeric_producer_config,
    )
    output = output_path.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(receipt, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return receipt
