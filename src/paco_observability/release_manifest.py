"""Deterministic inventory manifests for canonical release artifacts.

Artifact paths are always relative to an explicit base directory.  Input
directories are traversed recursively, overlapping requests are deduplicated,
and symlinks are rejected so that the same manifest cannot silently name a
different file on another host.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any


DEFAULT_PACKAGE_NAMES = (
    "paco-observability",
    "numpy",
    "pandas",
    "pyarrow",
    "matplotlib",
    "ipykernel",
    "ipython",
    "jupyter-client",
    "jupyter-core",
    "nbclient",
    "nbformat",
)


def _run_git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip() or "unknown git error"
        raise ValueError(f"cannot record Git provenance for {repository}: {detail}")
    return process.stdout.strip()


def _run_git_bytes(repository: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
    )
    if process.returncode:
        detail = (
            process.stderr.decode("utf-8", errors="replace").strip()
            or process.stdout.decode("utf-8", errors="replace").strip()
            or "unknown git error"
        )
        raise ValueError(f"cannot read Git object in {repository}: {detail}")
    return process.stdout


def git_state(base_dir: Path) -> dict[str, Any]:
    """Return the revision and pre-generation worktree state for ``base_dir``."""

    repository = Path(_run_git(base_dir, "rev-parse", "--show-toplevel")).resolve()
    return {
        "commit": _run_git(repository, "rev-parse", "HEAD"),
        "branch": _run_git(repository, "branch", "--show-current") or None,
        "dirty": bool(_run_git(repository, "status", "--porcelain=v1")),
        "state_timing": "before_manifest_write",
    }


def package_versions(package_names: Iterable[str] = DEFAULT_PACKAGE_NAMES) -> dict[str, str | None]:
    """Return a stable, sorted mapping of distribution names to versions."""

    versions: dict[str, str | None] = {}
    for name in sorted({str(value).strip() for value in package_names if str(value).strip()}):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def executable_version(name: str) -> str | None:
    """Return the first line of ``name --version``, or ``None`` if unavailable."""

    executable = shutil.which(name)
    if executable is None:
        return None
    process = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        return None
    output = (process.stdout.strip() or process.stderr.strip()).splitlines()
    return output[0] if output else None


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp, honoring ``SOURCE_DATE_EPOCH``.

    ``SOURCE_DATE_EPOCH`` makes byte-identical manifests possible when all
    inputs, Git state, and package versions are unchanged.
    """

    source_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_epoch is None:
        instant = datetime.now(timezone.utc)
    else:
        try:
            epoch = int(source_epoch)
        except ValueError as exc:
            raise ValueError("SOURCE_DATE_EPOCH must be an integer Unix timestamp") from exc
        instant = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return instant.isoformat(timespec="seconds").replace("+00:00", "Z")


def _within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _resolve_input(base_dir: Path, requested: str | Path) -> Path:
    value = Path(requested)
    candidate = value if value.is_absolute() else base_dir / value
    if not candidate.exists():
        raise ValueError(f"release input does not exist: {requested}")
    if candidate.is_symlink():
        raise ValueError(f"release inputs must not be symlinks: {requested}")
    resolved = candidate.resolve(strict=True)
    if not _within(resolved, base_dir):
        raise ValueError(f"release input falls outside base directory: {requested}")
    if not resolved.is_file() and not resolved.is_dir():
        raise ValueError(f"release input is not a regular file or directory: {requested}")
    return resolved


def resolve_release_inputs(
    *,
    base_dir: Path,
    requested_inputs: Sequence[str | Path],
    output_path: Path | None = None,
) -> tuple[list[dict[str, str]], list[Path]]:
    """Resolve explicit files/directories into a sorted, deduplicated file list.

    The output path is rejected if it equals an input file or falls anywhere
    below a requested input directory, even when the output does not exist yet.
    This prevents a manifest from hashing a previous copy of itself on reruns.
    """

    base = base_dir.resolve(strict=True)
    if not base.is_dir():
        raise ValueError("base_dir must be an existing directory")
    if not requested_inputs:
        raise ValueError("at least one explicit release input is required")

    resolved_output = None
    if output_path is not None:
        resolved_output = output_path if output_path.is_absolute() else base / output_path
        resolved_output = resolved_output.resolve(strict=False)
    requests: dict[tuple[str, str], dict[str, str]] = {}
    artifacts: dict[str, Path] = {}

    for requested in requested_inputs:
        source = _resolve_input(base, requested)
        relative_source = source.relative_to(base).as_posix()
        kind = "file" if source.is_file() else "directory"
        requests[(relative_source, kind)] = {"path": relative_source, "kind": kind}

        if resolved_output is not None:
            selects_output = source == resolved_output if source.is_file() else _within(
                resolved_output, source
            )
            if selects_output:
                raise ValueError(
                    "output manifest must not be an input or lie inside a recursively "
                    "selected input directory"
                )

        candidates = [source] if source.is_file() else source.rglob("*")
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"release input tree contains a symlink: {candidate}")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ValueError(f"release input tree contains a non-regular file: {candidate}")
            resolved_candidate = candidate.resolve(strict=True)
            if not _within(resolved_candidate, base):
                raise ValueError(f"release artifact falls outside base directory: {candidate}")
            if resolved_output is not None and resolved_candidate == resolved_output:
                raise ValueError("output manifest must not hash itself")
            relative = resolved_candidate.relative_to(base).as_posix()
            artifacts[relative] = resolved_candidate

    if not artifacts:
        raise ValueError("explicit release inputs contain no regular files")
    requested_records = [requests[key] for key in sorted(requests)]
    ordered_artifacts = [artifacts[key] for key in sorted(artifacts)]
    return requested_records, ordered_artifacts


def hash_stable_file(path: Path) -> tuple[int, str]:
    """Return size and SHA-256, rejecting a file modified during hashing."""

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise ValueError(f"release artifact changed while it was being hashed: {path}")
    return int(after.st_size), digest.hexdigest()


def validate_numeric_producer_declaration(
    *,
    base_dir: Path,
    config_path: Path,
) -> dict[str, Any] | None:
    """Validate the optional frozen numeric-producer declaration in a config.

    When ``release.numeric_producer_commit`` and
    ``release.numeric_producer_files`` are absent, the declaration is optional
    and this function returns ``None``.  If either field is present, both are
    required.  Every declared current file must be byte-identical to the blob
    at the declared commit; checking out or rewriting files is never attempted.
    """

    base = base_dir.resolve(strict=True)
    repository = Path(_run_git(base, "rev-parse", "--show-toplevel")).resolve()
    config = _resolve_input(base, config_path)
    if not config.is_file():
        raise ValueError("numeric-producer config must be a regular file")
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"numeric-producer config is not readable JSON: {config}") from exc
    if not isinstance(payload, dict):
        raise ValueError("numeric-producer config must contain a JSON object")
    release = payload.get("release")
    if release is None:
        return None
    if not isinstance(release, dict):
        raise ValueError("config.release must be a JSON object")
    commit_value = release.get("numeric_producer_commit")
    files_value = release.get("numeric_producer_files")
    if commit_value is None and files_value is None:
        return None
    if commit_value is None or files_value is None:
        raise ValueError(
            "release.numeric_producer_commit and release.numeric_producer_files "
            "must either both be present or both be absent"
        )
    commit = str(commit_value).strip()
    if not commit:
        raise ValueError("release.numeric_producer_commit must not be empty")
    resolved_commit = _run_git(repository, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if not isinstance(files_value, list) or not files_value:
        raise ValueError("release.numeric_producer_files must be a non-empty JSON array")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in files_value:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("release.numeric_producer_files entries must be non-empty strings")
        requested = Path(raw_path)
        if requested.is_absolute() or ".." in requested.parts:
            raise ValueError("numeric-producer file paths must be safe repository-relative paths")
        unresolved_candidate = repository / requested
        if unresolved_candidate.is_symlink():
            raise ValueError(f"numeric-producer path must not be a symlink: {raw_path}")
        candidate = unresolved_candidate.resolve(strict=True)
        if not _within(candidate, repository) or not candidate.is_file():
            raise ValueError(f"numeric-producer path is not a regular repository file: {raw_path}")
        relative = candidate.relative_to(repository).as_posix()
        if relative in seen:
            raise ValueError(f"numeric-producer file is declared more than once: {relative}")
        seen.add(relative)
        current_bytes = candidate.read_bytes()
        committed_bytes = _run_git_bytes(repository, "cat-file", "blob", f"{resolved_commit}:{relative}")
        if current_bytes != committed_bytes:
            raise ValueError(
                "numeric-producer file differs from declared commit "
                f"{resolved_commit}: {relative}"
            )
        records.append(
            {
                "path": relative,
                "size_bytes": len(current_bytes),
                "sha256": hashlib.sha256(current_bytes).hexdigest(),
                "git_blob": _run_git(repository, "rev-parse", f"{resolved_commit}:{relative}"),
            }
        )

    historical_record: dict[str, Any] | None = None
    historical = release.get("historical_full_record_entrypoint")
    if historical is not None:
        if not isinstance(historical, dict):
            raise ValueError("release.historical_full_record_entrypoint must be a JSON object")
        historical_path_value = historical.get("path")
        historical_commit_value = str(historical.get("commit", "")).strip()
        if not isinstance(historical_path_value, str) or not historical_path_value.strip():
            raise ValueError("historical full-record entrypoint path must be nonempty")
        historical_path = Path(historical_path_value)
        if historical_path.is_absolute() or ".." in historical_path.parts:
            raise ValueError("historical full-record entrypoint path must be repository-relative")
        historical_commit = _run_git(
            repository,
            "rev-parse",
            "--verify",
            f"{historical_commit_value}^{{commit}}",
        )
        if historical_commit != resolved_commit:
            raise ValueError(
                "historical full-record entrypoint commit must equal numeric_producer_commit"
            )
        relative_historical = historical_path.as_posix()
        committed_bytes = _run_git_bytes(
            repository,
            "cat-file",
            "blob",
            f"{historical_commit}:{relative_historical}",
        )
        committed_sha256 = hashlib.sha256(committed_bytes).hexdigest()
        committed_blob = _run_git(
            repository,
            "rev-parse",
            f"{historical_commit}:{relative_historical}",
        )
        if historical.get("sha256") != committed_sha256:
            raise ValueError("historical full-record entrypoint SHA-256 declaration is incorrect")
        if historical.get("git_blob_sha1") != committed_blob:
            raise ValueError("historical full-record entrypoint Git-blob declaration is incorrect")
        unresolved_current_path = repository / historical_path
        if unresolved_current_path.is_symlink():
            raise ValueError("historical full-record entrypoint must not be a symlink")
        current_path = unresolved_current_path.resolve(strict=True)
        if not _within(current_path, repository) or not current_path.is_file():
            raise ValueError("historical full-record entrypoint current path is not a regular file")
        current_bytes = current_path.read_bytes()
        difference_scope = historical.get("current_difference_scope")
        if current_bytes != committed_bytes and (
            not isinstance(difference_scope, str) or not difference_scope.strip()
        ):
            raise ValueError(
                "historical full-record entrypoint drift requires current_difference_scope"
            )
        historical_record = {
            "path": relative_historical,
            "commit": historical_commit,
            "size_bytes": len(committed_bytes),
            "sha256": committed_sha256,
            "git_blob_sha1": committed_blob,
            "current_file_sha256": hashlib.sha256(current_bytes).hexdigest(),
            "current_matches_historical_blob": current_bytes == committed_bytes,
            "current_difference_scope": difference_scope,
            "status": "VERIFIED_HISTORICAL_BLOB",
        }

    config_size, config_sha256 = hash_stable_file(config)
    return {
        "status": "VERIFIED",
        "declaration_source": config.relative_to(base).as_posix(),
        "declaration_source_size_bytes": config_size,
        "declaration_source_sha256": config_sha256,
        "commit": resolved_commit,
        "files": sorted(records, key=lambda record: record["path"]),
        "historical_full_record_entrypoint": historical_record,
    }


def build_release_manifest(
    *,
    base_dir: Path,
    requested_inputs: Sequence[str | Path],
    output_path: Path | None = None,
    generated_at_utc: str | None = None,
    package_names: Iterable[str] = DEFAULT_PACKAGE_NAMES,
    require_clean_git: bool = False,
    numeric_producer_config: Path | None = None,
    require_numeric_producer: bool = False,
) -> dict[str, Any]:
    """Build a deterministic canonical release-artifact manifest in memory."""

    base = base_dir.resolve(strict=True)
    initial_git = git_state(base)
    if require_clean_git and initial_git["dirty"]:
        repository = Path(_run_git(base, "rev-parse", "--show-toplevel")).resolve()
        dirty_paths = _run_git(repository, "status", "--porcelain=v1")
        raise ValueError(f"Git worktree must be clean for release:\n{dirty_paths}")
    if require_numeric_producer and numeric_producer_config is None:
        raise ValueError("a numeric-producer config is required for canonical release")
    numeric_producer = (
        validate_numeric_producer_declaration(
            base_dir=base,
            config_path=numeric_producer_config,
        )
        if numeric_producer_config is not None
        else None
    )
    if require_numeric_producer and numeric_producer is None:
        raise ValueError(
            "canonical release requires release.numeric_producer_commit and "
            "release.numeric_producer_files"
        )
    resolved_output = None
    if output_path is not None:
        resolved_output = output_path if output_path.is_absolute() else base / output_path
        resolved_output = resolved_output.resolve(strict=False)
    requests, files = resolve_release_inputs(
        base_dir=base,
        requested_inputs=requested_inputs,
        output_path=resolved_output,
    )
    artifacts: list[dict[str, Any]] = []
    for path in files:
        size, sha256 = hash_stable_file(path)
        artifacts.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size_bytes": size,
                "sha256": sha256,
            }
        )
    rechecked_requests, rechecked_files = resolve_release_inputs(
        base_dir=base,
        requested_inputs=requested_inputs,
        output_path=resolved_output,
    )
    initial_paths = [path.relative_to(base).as_posix() for path in files]
    rechecked_paths = [path.relative_to(base).as_posix() for path in rechecked_files]
    if requests != rechecked_requests or initial_paths != rechecked_paths:
        raise ValueError("release input tree changed while the manifest was being built")
    rehashed_artifacts: list[dict[str, Any]] = []
    for path in rechecked_files:
        size, sha256 = hash_stable_file(path)
        rehashed_artifacts.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size_bytes": size,
                "sha256": sha256,
            }
        )
    if artifacts != rehashed_artifacts:
        raise ValueError("release artifact bytes changed while the manifest was being built")

    rechecked_numeric_producer = (
        validate_numeric_producer_declaration(
            base_dir=base,
            config_path=numeric_producer_config,
        )
        if numeric_producer_config is not None
        else None
    )
    if numeric_producer != rechecked_numeric_producer:
        raise ValueError("numeric-producer declaration changed while the manifest was being built")
    final_git = git_state(base)
    if initial_git != final_git:
        raise ValueError("Git state changed while the release manifest was being built")
    if require_clean_git and final_git["dirty"]:
        raise ValueError("Git worktree became dirty while the release manifest was being built")

    return {
        "schema_version": "1.0",
        "generated_at_utc": generated_at_utc or utc_now(),
        "path_base": ".",
        "requested_inputs": requests,
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(record["size_bytes"] for record in artifacts),
        "artifacts": artifacts,
        "git": initial_git,
        "release_policy": {
            "require_clean_git": bool(require_clean_git),
            "require_numeric_producer": bool(require_numeric_producer),
        },
        "numeric_producer": numeric_producer,
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "packages": package_versions(package_names),
            "tools": {"uv": executable_version("uv")},
        },
    }


def write_release_manifest(
    *,
    base_dir: Path,
    requested_inputs: Sequence[str | Path],
    output_path: Path,
    generated_at_utc: str | None = None,
    package_names: Iterable[str] = DEFAULT_PACKAGE_NAMES,
    require_clean_git: bool = False,
    numeric_producer_config: Path | None = None,
    require_numeric_producer: bool = False,
) -> dict[str, Any]:
    """Build and atomically write a canonical release-artifact manifest."""

    base = base_dir.resolve(strict=True)
    output = output_path if output_path.is_absolute() else base / output_path
    output = output.resolve(strict=False)
    manifest = build_release_manifest(
        base_dir=base,
        requested_inputs=requested_inputs,
        output_path=output,
        generated_at_utc=generated_at_utc,
        package_names=package_names,
        require_clean_git=require_clean_git,
        numeric_producer_config=numeric_producer_config,
        require_numeric_producer=require_numeric_producer,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest
