"""Byte-level input/output receipts for deterministic post-processing stages."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def stable_file_record(path: Path, *, path_base: Path | None = None) -> dict[str, Any]:
    """Hash one regular file and reject replacement or mutation during the read."""

    candidate = path.resolve(strict=True)
    if path.is_symlink() or not candidate.is_file():
        raise ValueError(f"receipt input must be a regular non-symlink file: {path}")
    before = candidate.stat()
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    after = candidate.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"file changed while derivation receipt was hashing it: {path}")
    display_path = (
        Path(os.path.relpath(candidate, path_base.resolve(strict=False))).as_posix()
        if path_base is not None
        else str(candidate)
    )
    return {
        "path": display_path,
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def snapshot_named_files(
    files: Mapping[str, Path], *, path_base: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Return sorted stable records for an explicitly named file mapping."""

    if not files:
        raise ValueError("at least one named file is required")
    return {
        name: stable_file_record(files[name], path_base=path_base)
        for name in sorted(files)
    }


def require_named_files_unchanged(
    records: Mapping[str, Mapping[str, Any]],
    files: Mapping[str, Path],
    *,
    path_base: Path | None = None,
) -> None:
    """Reject a derivation if any snapshotted input changed during processing."""

    current = snapshot_named_files(files, path_base=path_base)
    if dict(records) != current:
        raise ValueError("a derivation input changed while the output was being produced")


def write_derivation_receipt(
    *,
    receipt_path: Path,
    derivation: str,
    input_records: Mapping[str, Mapping[str, Any]],
    output_files: Mapping[str, Path],
) -> dict[str, Any]:
    """Atomically bind validated input snapshots to newly produced output bytes."""

    normalized = derivation.strip()
    if not normalized:
        raise ValueError("derivation must not be empty")
    output = receipt_path.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "1.0",
        "derivation": normalized,
        "status": "VERIFIED",
        "inputs": {name: dict(input_records[name]) for name in sorted(input_records)},
        "outputs": snapshot_named_files(output_files, path_base=output.parent),
    }
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


__all__ = [
    "require_named_files_unchanged",
    "snapshot_named_files",
    "stable_file_record",
    "write_derivation_receipt",
]
