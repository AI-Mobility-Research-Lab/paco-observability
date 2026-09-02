"""Retroactive, byte-exact lineage receipts for completed validation stages.

The 200-frame and 25-frame validation formats predate embedded source hashes.
This module does not rewrite those artifacts.  Instead, it validates their
frozen contracts and records an explicit retroactive binding to the canonical
inputs, provenance manifest, and declared producer commit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import pyarrow.parquet as pq

from .derivation_receipt import snapshot_named_files


_COMMIT = re.compile(r"[0-9a-f]{40}")
_STAGE_CONTRACTS = {
    "validation_main_200_final": {
        "frame_count": 200,
        "ray_grids": [9, 17, 33],
        "z_modes": ["raw", "ground_anchored"],
    },
    "validation_convergence_25_final": {
        "frame_count": 25,
        "ray_grids": [17, 33, 65],
        "z_modes": ["ground_anchored"],
    },
}


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(a, b) for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_commit(repository: Path, commit: str) -> str:
    normalized = commit.strip().lower()
    if _COMMIT.fullmatch(normalized) is None:
        raise ValueError("producer_commit must be a full 40-character hexadecimal commit")
    process = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{normalized}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise ValueError(f"declared producer commit is unavailable: {normalized}")
    return normalized


def build_validation_lineage_receipt(
    *,
    stage: str,
    repository: Path,
    producer_commit: str,
    tracks: Path,
    frame_index: Path,
    input_provenance: Path,
    summary: Path,
    config: Path,
    decisions: Path,
    timings: Path,
    exclusions: Path,
    excluded_source_frames: Sequence[int],
    main_sample_summary: Path | None = None,
    path_base: Path | None = None,
) -> dict[str, Any]:
    """Validate and bind one immutable validation-stage artifact set."""

    if stage not in _STAGE_CONTRACTS:
        raise ValueError(f"unknown validation lineage stage: {stage}")
    contract = _STAGE_CONTRACTS[stage]
    commit = _verify_commit(repository.resolve(strict=True), producer_commit)
    source_files = {
        "tracks": tracks,
        "frame_index": frame_index,
        "input_provenance": input_provenance,
    }
    artifact_files = {
        "summary": summary,
        "config": config,
        "decisions": decisions,
        "timings": timings,
        "exclusions": exclusions,
    }
    if stage == "validation_convergence_25_final":
        _require(
            main_sample_summary is not None,
            "convergence lineage requires the main validation summary",
        )
        source_files["main_sample_summary"] = main_sample_summary
    elif main_sample_summary is not None:
        raise ValueError("main_sample_summary is only valid for convergence lineage")
    record_base = repository.resolve(strict=True) if path_base is None else path_base
    sources = snapshot_named_files(source_files, path_base=record_base)
    artifacts = snapshot_named_files(artifact_files, path_base=record_base)
    provenance_payload = _load_json(input_provenance, "input provenance")
    summary_payload = _load_json(summary, "validation summary")
    config_payload = _load_json(config, "validation config")
    provenance_tracks = provenance_payload.get("tracks")
    provenance_frame_index = provenance_payload.get("frame_index")
    _require(isinstance(provenance_tracks, Mapping), "provenance.tracks must be an object")
    _require(
        isinstance(provenance_frame_index, Mapping),
        "provenance.frame_index must be an object",
    )
    _require(
        provenance_tracks.get("sha256") == sources["tracks"]["sha256"],
        "canonical tracks bytes disagree with input provenance",
    )
    _require(
        provenance_frame_index.get("sha256") == sources["frame_index"]["sha256"],
        "canonical frame-index bytes disagree with input provenance",
    )
    _require(summary_payload.get("config") == config_payload, "summary/config mismatch")
    _require(
        _equivalent(summary_payload.get("input_audit"), provenance_tracks.get("audit")),
        "validation input audit disagrees with canonical tracks audit",
    )
    _require(
        summary_payload.get("sampled_frame_count") == contract["frame_count"],
        "validation sampled-frame count violates the frozen stage contract",
    )
    _require(
        config_payload.get("ray_grids") == contract["ray_grids"],
        "validation ray grids violate the frozen stage contract",
    )
    _require(
        config_payload.get("z_modes") == contract["z_modes"],
        "validation z modes violate the frozen stage contract",
    )
    _require(
        config_payload.get("fov_degrees") == [120.0, 360.0],
        "validation FOVs violate the frozen stage contract",
    )
    _require(
        sorted(int(value) for value in summary_payload.get("excluded_source_frames", []))
        == sorted(int(value) for value in excluded_source_frames),
        "validation source exclusions violate the frozen stage contract",
    )
    sampled_frames = summary_payload.get("sampled_frames")
    _require(
        isinstance(sampled_frames, list)
        and len(sampled_frames) == contract["frame_count"]
        and len(set(sampled_frames)) == contract["frame_count"],
        "validation sampled-frame list is incomplete or non-unique",
    )
    decision_frames = set(
        pq.read_table(decisions, columns=["frame_idx"]).column("frame_idx").to_pylist()
    )
    _require(decision_frames == set(sampled_frames), "decision frames disagree with summary")
    if stage == "validation_convergence_25_final":
        main_payload = _load_json(Path(main_sample_summary), "main validation summary")
        main_frames = main_payload.get("sampled_frames")
        _require(
            isinstance(main_frames, list)
            and len(main_frames) == 200
            and len(set(main_frames)) == 200,
            "main validation summary does not contain a unique 200-frame sample",
        )
        _require(
            set(sampled_frames).issubset(set(main_frames)),
            "convergence frames are not a subset of the main validation sample",
        )
    _require(pq.ParquetFile(timings).metadata.num_rows > 0, "validation timings are empty")
    pq.ParquetFile(exclusions)
    # Rehash after semantic reads so a concurrent replacement cannot be blessed.
    _require(
        sources == snapshot_named_files(source_files, path_base=record_base),
        "lineage source changed during audit",
    )
    _require(
        artifacts == snapshot_named_files(artifact_files, path_base=record_base),
        "validation artifact changed during lineage audit",
    )
    checks = {
        "canonical_source_hashes_match_input_provenance": True,
        "summary_input_audit_matches_canonical_tracks_audit": True,
        "summary_config_and_sample_contract_verified": True,
        "decision_frame_set_matches_summary": True,
        "all_source_and_artifact_bytes_stable_across_audit": True,
    }
    if stage == "validation_convergence_25_final":
        checks["convergence_is_main_sample_subset"] = True
    return {
        "schema_version": "1.0",
        "receipt": "cacie_validation_source_lineage",
        "status": "VERIFIED",
        "binding_mode": "retroactive_byte_exact_no_artifact_rewrite",
        "stage": stage,
        "declared_numeric_producer_commit": commit,
        "sources": sources,
        "artifacts": artifacts,
        "checks": checks,
    }


def write_validation_lineage_receipt(*, output_path: Path, **kwargs: Any) -> dict[str, Any]:
    output = output_path.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_validation_lineage_receipt(path_base=output.parent, **kwargs)
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


def verify_validation_lineage_receipt(
    *, output_path: Path, producer_commit: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Rebuild a validation receipt and reject stale sources or artifacts."""

    stored = _load_json(output_path, "validation lineage receipt")
    declared = stored.get("declared_numeric_producer_commit")
    expected_commit = declared if producer_commit is None else producer_commit
    if declared != expected_commit:
        raise ValueError("validation lineage producer commit disagrees with required commit")
    current = build_validation_lineage_receipt(
        producer_commit=str(expected_commit),
        path_base=output_path.resolve(strict=True).parent,
        **kwargs,
    )
    if stored != current:
        raise ValueError("validation lineage receipt is stale or does not match current bytes")
    return current


__all__ = [
    "build_validation_lineage_receipt",
    "verify_validation_lineage_receipt",
    "write_validation_lineage_receipt",
]
