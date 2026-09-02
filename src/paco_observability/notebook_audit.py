"""Fail-closed audit and immutable execution receipt for CACIE artifacts.

The notebook and the post-execution CLI both call this module.  The notebook
therefore cannot silently skip an absent stage, while the CLI binds the exact
executed notebook bytes to the exact bytes of every artifact inspected here.
No visibility geometry or bootstrap resampling is recomputed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import pyarrow.parquet as pq

from .cacie_source_contract import (
    EXPECTED_ALLOWED_MISSING_FRAMES,
    EXPECTED_FOVS,
    EXPECTED_FRAME_RANGE,
    EXPECTED_METHODS,
    EXPECTED_OBSERVED_EXCLUSIONS,
    EXPECTED_QUALITY_EXCLUDED_FRAMES,
    EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
)
from .release_manifest import validate_numeric_producer_declaration
from .validation_lineage import verify_validation_lineage_receipt


PASS_MARKER = "CACIE_NOTEBOOK_AUDIT_PASS_V1"
REQUIRED_STAGES = (
    "provenance",
    "main_validation",
    "sparse_calibration",
    "validation_strata",
    "grid_convergence",
    "full_record_primary",
    "primary_bootstrap",
    "block_sensitivity",
    "residual_hotspot_ranking",
    "z_mode_sensitivity",
    "partial_scan_sensitivity",
    "sensitivity_summary",
    "validation_figure",
    "full_results_figure",
    "detector_validation",
    "runtime_environment",
)
EXPECTED_SEED = 20260902
EXPECTED_MAIN_FRAMES = 200
EXPECTED_CONVERGENCE_FRAMES = 25
EXPECTED_BOOTSTRAP_REPLICATES = 5_000
EXPECTED_PRIMARY_FRAMES = 40_779
EXPECTED_STEP10_FRAMES = 4_078
EXPECTED_PARTIAL_OBSERVED_FRAMES = 15
EXPECTED_EGO_COUNT = 60


class NotebookAuditError(RuntimeError):
    """Raised when a required canonical artifact or contract is not verified."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NotebookAuditError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise NotebookAuditError(f"{name} must be a JSON array")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise NotebookAuditError(f"{name} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise NotebookAuditError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise NotebookAuditError(f"{name} must be an integer")
    return int(numeric)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise NotebookAuditError(f"{name} must be finite numeric data")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise NotebookAuditError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(numeric):
        raise NotebookAuditError(f"{name} must be finite numeric data")
    return numeric


def _same_number(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(
        _number(left, "left numeric value"),
        _number(right, "right numeric value"),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _stable_file_record(path: Path, display_path: str) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise NotebookAuditError(f"artifact changed while being hashed: {display_path}")
    return {
        "path": display_path,
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


class _AuditContext:
    def __init__(self, repo_root: Path, output_root: Path) -> None:
        self.repo_root = repo_root.resolve(strict=True)
        self.output_root = output_root.resolve(strict=True)
        if not self.repo_root.is_dir() or not self.output_root.is_dir():
            raise NotebookAuditError("repository and output roots must be existing directories")
        self.paths: dict[str, Path] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self.checks: list[dict[str, str]] = []
        self.stages: list[dict[str, str]] = []

    def display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path)

    def _register(self, name: str, path: Path, *, allowed_root: Path) -> Path:
        if name in self.paths and self.paths[name] != path:
            raise NotebookAuditError(f"artifact alias {name!r} names two paths")
        if not path.exists():
            raise NotebookAuditError(f"required artifact is missing: {self.display_path(path)}")
        if path.is_symlink() or not path.is_file():
            raise NotebookAuditError(
                f"required artifact is not a regular non-symlink file: {self.display_path(path)}"
            )
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise NotebookAuditError(
                f"required artifact resolves outside its allowed root: {self.display_path(path)}"
            ) from exc
        self.paths[name] = resolved
        self.records[name] = {
            "name": name,
            **_stable_file_record(resolved, self.display_path(resolved)),
        }
        return resolved

    def artifact(self, name: str, relative: str | Path) -> Path:
        return self._register(
            name,
            self.output_root / relative,
            allowed_root=self.output_root,
        )

    def repository_file(self, name: str, relative: str | Path) -> Path:
        return self._register(
            name,
            self.repo_root / relative,
            allowed_root=self.repo_root,
        )

    def json_artifact(self, name: str, relative: str | Path) -> dict[str, Any]:
        return self._load_json(self.artifact(name, relative), name)

    def repository_json(self, name: str, relative: str | Path) -> dict[str, Any]:
        return self._load_json(self.repository_file(name, relative), name)

    def _load_json(self, path: Path, name: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise NotebookAuditError(f"{name} is not readable JSON: {self.display_path(path)}") from exc
        if not isinstance(value, dict):
            raise NotebookAuditError(f"{name} must contain a JSON object")
        return value

    def parquet(self, name: str, relative: str | Path) -> pq.ParquetFile:
        path = self.artifact(name, relative)
        try:
            return pq.ParquetFile(path)
        except Exception as exc:
            raise NotebookAuditError(f"{name} is not a readable Parquet file") from exc

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if not bool(condition):
            suffix = f" ({detail})" if detail else ""
            raise NotebookAuditError(f"FAILED: {name}{suffix}")
        self.checks.append({"check": name, "status": "PASS", "detail": detail})

    def stage(self, name: str, detail: str) -> None:
        if name not in REQUIRED_STAGES:
            raise NotebookAuditError(f"unknown audit stage: {name}")
        if any(row["stage"] == name for row in self.stages):
            raise NotebookAuditError(f"audit stage recorded more than once: {name}")
        self.stages.append({"stage": name, "status": "VERIFIED", "detail": detail})

    def snapshot(self, name: str) -> dict[str, Any]:
        if name not in self.paths:
            raise NotebookAuditError(f"artifact alias was not registered: {name}")
        if name not in self.records:
            path = self.paths[name]
            self.records[name] = {
                "name": name,
                **_stable_file_record(path, self.display_path(path)),
            }
        return self.records[name]

    def discover_artifact_tree(self, prefix: str, relative: str | Path) -> None:
        root = self.output_root / relative
        if not root.is_dir() or root.is_symlink():
            raise NotebookAuditError(f"required artifact directory is missing: {self.display_path(root)}")
        registered = 0
        for path in sorted(root.rglob("*")):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise NotebookAuditError(
                    f"artifact directory contains a symlink: {self.display_path(path)}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                raise NotebookAuditError(
                    f"artifact directory contains a non-regular file: {self.display_path(path)}"
                )
            resolved = path.resolve(strict=True)
            if resolved in self.paths.values():
                continue
            alias = f"{prefix}:{path.relative_to(root).as_posix()}"
            self._register(alias, path, allowed_root=self.output_root)
            registered += 1
        if registered == 0 and not any(
            root == existing.parent or root in existing.parents for existing in self.paths.values()
        ):
            raise NotebookAuditError(f"artifact directory contains no files: {self.display_path(root)}")


def _parquet_rows(file: pq.ParquetFile) -> int:
    return int(file.metadata.num_rows)


def _parquet_unique_values(path: Path, column: str) -> set[Any]:
    try:
        values = pq.read_table(path, columns=[column]).column(column).to_pylist()
    except Exception as exc:
        raise NotebookAuditError(f"cannot read required Parquet column {column!r}: {path}") from exc
    return set(values)


def _parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        return pq.read_table(path).to_pylist()
    except Exception as exc:
        raise NotebookAuditError(f"cannot read required Parquet records: {path}") from exc


def _audit_derivation_receipt(
    context: _AuditContext,
    *,
    alias: str,
    relative: str,
    derivation: str,
    input_aliases: Mapping[str, str],
    output_aliases: Mapping[str, str],
) -> None:
    receipt = context.json_artifact(alias, relative)
    context.check(f"{derivation} receipt schema", receipt.get("schema_version") == "1.0")
    context.check(f"{derivation} receipt kind", receipt.get("derivation") == derivation)
    context.check(f"{derivation} receipt status", receipt.get("status") == "VERIFIED")
    inputs = _mapping(receipt.get("inputs"), f"{derivation}.inputs")
    outputs = _mapping(receipt.get("outputs"), f"{derivation}.outputs")
    context.check(f"{derivation} exact input roles", set(inputs) == set(input_aliases))
    context.check(f"{derivation} exact output roles", set(outputs) == set(output_aliases))
    for role, source_alias in input_aliases.items():
        record = _mapping(inputs.get(role), f"{derivation}.inputs.{role}")
        current = context.snapshot(source_alias)
        context.check(
            f"{derivation} {role} input SHA lineage",
            record.get("sha256") == current["sha256"],
        )
        context.check(
            f"{derivation} {role} input size lineage",
            record.get("size_bytes") == current["size_bytes"],
        )
    for role, output_alias in output_aliases.items():
        record = _mapping(outputs.get(role), f"{derivation}.outputs.{role}")
        current = context.snapshot(output_alias)
        context.check(
            f"{derivation} {role} output SHA lineage",
            record.get("sha256") == current["sha256"],
        )
        context.check(
            f"{derivation} {role} output size lineage",
            record.get("size_bytes") == current["size_bytes"],
        )


def _check_confusion(context: _AuditContext, name: str, metrics: Mapping[str, Any]) -> None:
    n = _integer(metrics.get("n"), f"{name}.n")
    tp = _integer(metrics.get("tp"), f"{name}.tp")
    tn = _integer(metrics.get("tn"), f"{name}.tn")
    fp = _integer(metrics.get("fp"), f"{name}.fp")
    fn = _integer(metrics.get("fn"), f"{name}.fn")
    context.check(f"{name} confusion total", n == tp + tn + fp + fn)
    if "accuracy" in metrics:
        context.check(
            f"{name} accuracy",
            n > 0 and _same_number(metrics["accuracy"], (tp + tn) / n),
        )


def _audit_validation_source_lineage(
    context: _AuditContext,
    *,
    stage: str,
    directory: str,
    prefix: str,
    main_sample_summary: Path | None = None,
) -> None:
    receipt_path = context.artifact(
        f"{prefix}_source_lineage_receipt",
        f"{directory}/source_lineage_receipt.json",
    )
    try:
        verified = verify_validation_lineage_receipt(
            output_path=receipt_path,
            stage=stage,
            repository=context.repo_root,
            tracks=context.paths["canonical_tracks"],
            frame_index=context.paths["canonical_frame_index"],
            input_provenance=context.paths["input_provenance"],
            summary=context.paths[f"{prefix}_summary"],
            config=context.paths[f"{prefix}_config"],
            decisions=context.paths[f"{prefix}_decisions"],
            timings=context.paths[f"{prefix}_timings"],
            exclusions=context.paths[f"{prefix}_ego_exclusions"],
            excluded_source_frames=EXPECTED_QUALITY_EXCLUDED_FRAMES,
            main_sample_summary=main_sample_summary,
        )
    except ValueError as exc:
        raise NotebookAuditError(f"{prefix} source-lineage receipt failed: {exc}") from exc
    context.check(f"{prefix} source-lineage status", verified.get("status") == "VERIFIED")
    context.check(
        f"{prefix} source-lineage binding is transparent",
        verified.get("binding_mode") == "retroactive_byte_exact_no_artifact_rewrite",
    )


def _audit_provenance(context: _AuditContext) -> None:
    config = context.repository_json("analysis_config", "configs/cacie_v1.1.json")
    context.repository_file("canonical_tracks", "data/inputs/canonical/tracks.parquet")
    context.repository_file("canonical_frame_index", "data/inputs/canonical/frame_index.parquet")
    provenance = context.json_artifact("input_provenance", "input_provenance.json")
    context.json_artifact("upstream_manifest", "upstream/manifest.json")
    tracks = _mapping(provenance.get("tracks"), "provenance.tracks")
    frame_index = _mapping(provenance.get("frame_index"), "provenance.frame_index")
    track_audit = _mapping(tracks.get("audit"), "provenance.tracks.audit")
    frame_audit = _mapping(frame_index.get("audit"), "provenance.frame_index.audit")
    context.check(
        "track provenance quality gate",
        _mapping(track_audit.get("quality_gate"), "track quality gate").get("passed") is True,
    )
    context.check(
        "frame-index provenance quality gate",
        _mapping(frame_audit.get("quality_gate"), "frame quality gate").get("passed") is True,
    )
    parameters = _mapping(provenance.get("parameters"), "provenance.parameters")
    context.check(
        "provenance frame range",
        tuple(parameters.get("expected_frame_range", ())) == EXPECTED_FRAME_RANGE,
    )
    context.check(
        "provenance missing-frame exception",
        tuple(parameters.get("allowed_missing_frames", ())) == EXPECTED_ALLOWED_MISSING_FRAMES,
    )
    context.check(
        "provenance partial-scan exclusions",
        tuple(parameters.get("excluded_partial_frames", ())) == EXPECTED_QUALITY_EXCLUDED_FRAMES,
    )
    context.check(
        "detected partial scans equal declared exclusions",
        tuple(frame_audit.get("detected_partial_frames", ())) == EXPECTED_QUALITY_EXCLUDED_FRAMES,
    )
    context.check("no unexcluded partial scan", frame_audit.get("unexcluded_partial_frames") == [])
    configured_upstream = _mapping(config.get("upstream"), "config.upstream")
    context.check("track SHA agrees with config", tracks.get("sha256") == configured_upstream.get("tracks_sha256"))
    context.check(
        "track SHA agrees with canonical input bytes",
        tracks.get("sha256") == context.snapshot("canonical_tracks")["sha256"],
    )
    context.check(
        "frame-index SHA agrees with config",
        frame_index.get("sha256") == configured_upstream.get("frame_index_sha256"),
    )
    context.check(
        "frame-index SHA agrees with canonical input bytes",
        frame_index.get("sha256") == context.snapshot("canonical_frame_index")["sha256"],
    )
    upstream_record = _mapping(provenance.get("upstream_manifest"), "provenance.upstream_manifest")
    context.check(
        "upstream manifest bytes agree with provenance",
        upstream_record.get("sha256") == context.snapshot("upstream_manifest")["sha256"],
    )
    context.stage("provenance", "input gates, frozen exclusions, and source hashes verified")


def _audit_main_validation(context: _AuditContext) -> None:
    config = context.json_artifact("main_config", "validation_main/validation_config.json")
    summary = context.json_artifact("main_summary", "validation_main/validation_summary.json")
    decisions = context.parquet("main_decisions", "validation_main/validation_decisions.parquet")
    timings = context.parquet("main_timings", "validation_main/validation_timings.parquet")
    context.artifact("main_ego_exclusions", "validation_main/ego_exclusions.parquet")
    context.check("main grids are 9/17/33", config.get("ray_grids") == [9, 17, 33])
    context.check("main z modes", config.get("z_modes") == ["raw", "ground_anchored"])
    context.check("main FOVs", config.get("fov_degrees") == list(EXPECTED_FOVS))
    context.check("main summary/config agreement", summary.get("config") == config)
    context.check("main sampled frames", summary.get("sampled_frame_count") == EXPECTED_MAIN_FRAMES)
    context.check(
        "main partial exclusions",
        tuple(summary.get("excluded_source_frames", ())) == EXPECTED_QUALITY_EXCLUDED_FRAMES,
    )
    context.check("main decisions are nonempty", _parquet_rows(decisions) > 0)
    context.check("main timings are nonempty", _parquet_rows(timings) > 0)
    frame_values = _parquet_unique_values(context.paths["main_decisions"], "frame_idx")
    context.check("main decisions contain 200 frames", len(frame_values) == EXPECTED_MAIN_FRAMES)
    for index, raw_group in enumerate(_list(summary.get("groups"), "main groups")):
        group = _mapping(raw_group, f"main.groups[{index}]")
        agreements = _mapping(
            group.get("agreement_with_exact_among_covered"),
            f"main.groups[{index}].agreement",
        )
        for method in ("legacy_planar", "center_top", "sparse_multiray"):
            _check_confusion(
                context,
                f"main {group.get('z_mode')}/{group.get('fov_deg')}/{method}",
                _mapping(agreements.get(method), f"main agreement {method}"),
            )
    _audit_validation_source_lineage(
        context,
        stage="validation_main_200_final",
        directory="validation_main",
        prefix="main",
    )
    context.stage(
        "main_validation",
        "200-frame raw/ground 9/17/33 validation and stored confusion counts verified",
    )


def _audit_calibration(context: _AuditContext) -> None:
    payload = context.json_artifact(
        "calibration_summary", "sparse_calibration/sparse_threshold_calibration.json"
    )
    decisions = context.parquet(
        "calibrated_decisions", "sparse_calibration/validation_decisions_calibrated.parquet"
    )
    split = _mapping(payload.get("split"), "calibration.split")
    calibration_frames = tuple(int(value) for value in split.get("calibration_frames", ()))
    held_out_frames = tuple(int(value) for value in split.get("held_out_frames", ()))
    context.check("calibration frame count", len(calibration_frames) == 100)
    context.check("held-out frame count", len(held_out_frames) == 100)
    context.check(
        "calibration/held-out split disjoint",
        set(calibration_frames).isdisjoint(held_out_frames),
    )
    context.check(
        "calibration split covers main sample",
        len(set(calibration_frames) | set(held_out_frames)) == EXPECTED_MAIN_FRAMES,
    )
    candidates = _list(payload.get("candidates"), "calibration.candidates")
    thresholds = sorted(_integer(row.get("min_visible_rays"), "candidate threshold") for row in candidates)
    context.check("all 1..15 candidate thresholds", thresholds == list(range(1, 16)))
    selection = _mapping(payload.get("selection"), "calibration.selection")
    context.check(
        "post-development sparse threshold is 3/15",
        _integer(selection.get("selected_min_visible_rays"), "selected threshold")
        == EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
    )
    context.check(
        "calibration development status is explicit",
        _mapping(payload.get("protocol"), "calibration.protocol").get("development_status")
        == "post-development calibration with held-out evaluation",
    )
    _check_confusion(
        context,
        "calibration selected held-out overall",
        _mapping(_mapping(selection.get("held_out"), "selection.held_out").get("overall"), "held-out overall"),
    )
    input_record = _mapping(payload.get("input"), "calibration.input")
    output_record = _mapping(
        _mapping(payload.get("output"), "calibration.output").get("calibrated_decisions"),
        "calibration.output.calibrated_decisions",
    )
    context.check(
        "calibration input is current main decisions",
        input_record.get("sha256") == context.snapshot("main_decisions")["sha256"],
    )
    context.check(
        "calibrated decision bytes match summary",
        output_record.get("sha256") == context.snapshot("calibrated_decisions")["sha256"],
    )
    context.check(
        "calibrated decision row count",
        _parquet_rows(decisions) == _integer(output_record.get("rows"), "calibrated rows"),
    )
    columns = set(decisions.schema.names)
    context.check(
        "calibrated threshold column present",
        "sparse_multiray_min_visible_rays" in columns,
    )
    values = _parquet_unique_values(
        context.paths["calibrated_decisions"], "sparse_multiray_min_visible_rays"
    )
    context.check("every calibrated row stores threshold 3", values == {3})
    context.stage(
        "sparse_calibration",
        "3/15 selected on 100 frames and evaluated on 100 disjoint held-out frames",
    )


def _audit_strata(context: _AuditContext) -> None:
    summary = context.json_artifact(
        "strata_summary", "validation_strata_calibrated/validation_strata_summary.json"
    )
    metrics = context.parquet(
        "strata_metrics", "validation_strata_calibrated/validation_strata_metrics.parquet"
    )
    blockers = context.parquet(
        "strata_blockers", "validation_strata_calibrated/validation_blocker_classes.parquet"
    )
    timings = context.parquet(
        "strata_timings", "validation_strata_calibrated/validation_timing_summary.parquet"
    )
    scope = _mapping(summary.get("study_scope"), "strata.study_scope")
    context.check("strata sample declaration", scope.get("declared_sampled_frames") == 200)
    context.check("strata observed frames", scope.get("observed_sampled_frames") == 200)
    context.check("strata sample matches declaration", scope.get("sample_size_matches_declaration") is True)
    context.check("strata method set", tuple(summary.get("methods", ())) == EXPECTED_METHODS)
    metric_rows = _list(summary.get("metrics"), "strata.metrics")
    blocker_rows = _list(summary.get("blocker_classes"), "strata.blocker_classes")
    timing_rows = _list(summary.get("timings"), "strata.timings")
    context.check("strata metric Parquet rows", _parquet_rows(metrics) == len(metric_rows))
    context.check("strata blocker Parquet rows", _parquet_rows(blockers) == len(blocker_rows))
    context.check("strata timing Parquet rows", _parquet_rows(timings) == len(timing_rows))
    context.check(
        "strata JSON metrics equal Parquet records",
        metric_rows == _parquet_records(context.paths["strata_metrics"]),
    )
    context.check(
        "strata JSON blocker rows equal Parquet records",
        blocker_rows == _parquet_records(context.paths["strata_blockers"]),
    )
    context.check(
        "strata JSON timing rows equal Parquet records",
        timing_rows == _parquet_records(context.paths["strata_timings"]),
    )
    for index, raw_row in enumerate(metric_rows):
        _check_confusion(context, f"strata metric row {index}", _mapping(raw_row, "strata metric"))
    _audit_derivation_receipt(
        context,
        alias="strata_source_lineage_receipt",
        relative="validation_strata_calibrated/source_lineage_receipt.json",
        derivation="validation_strata_calibrated",
        input_aliases={
            "calibrated_decisions": "calibrated_decisions",
            "main_validation_timings": "main_timings",
        },
        output_aliases={
            "summary": "strata_summary",
            "metrics": "strata_metrics",
            "timings": "strata_timings",
            "blockers": "strata_blockers",
        },
    )
    context.stage("validation_strata", "calibrated class/range/density/blocker strata verified")


def _audit_convergence(context: _AuditContext) -> None:
    config = context.json_artifact(
        "convergence_config", "validation_convergence/validation_config.json"
    )
    summary = context.json_artifact(
        "convergence_summary", "validation_convergence/validation_summary.json"
    )
    decisions = context.parquet(
        "convergence_decisions", "validation_convergence/validation_decisions.parquet"
    )
    context.parquet("convergence_timings", "validation_convergence/validation_timings.parquet")
    context.artifact("convergence_ego_exclusions", "validation_convergence/ego_exclusions.parquet")
    context.check("convergence grids", config.get("ray_grids") == [17, 33, 65])
    context.check("convergence ground z mode", config.get("z_modes") == ["ground_anchored"])
    context.check("convergence FOVs", config.get("fov_degrees") == list(EXPECTED_FOVS))
    context.check("convergence summary/config agreement", summary.get("config") == config)
    context.check(
        "convergence sampled frames",
        summary.get("sampled_frame_count") == EXPECTED_CONVERGENCE_FRAMES,
    )
    frames = _parquet_unique_values(context.paths["convergence_decisions"], "frame_idx")
    context.check("convergence decisions contain 25 frames", len(frames) == 25)
    context.check("convergence decisions are nonempty", _parquet_rows(decisions) > 0)
    main_summary = context._load_json(context.paths["main_summary"], "main summary")
    convergence_sample = set(_list(summary.get("sampled_frames"), "convergence sampled frames"))
    main_sample = set(_list(main_summary.get("sampled_frames"), "main sampled frames"))
    context.check(
        "convergence is a 25-frame subset of the main validation sample",
        len(convergence_sample) == 25 and convergence_sample.issubset(main_sample),
    )
    _audit_validation_source_lineage(
        context,
        stage="validation_convergence_25_final",
        directory="validation_convergence",
        prefix="convergence",
        main_sample_summary=context.paths["main_summary"],
    )
    context.stage("grid_convergence", "25-frame main-sample subset 17/33/65 grid check verified")


def _check_full_groups(
    context: _AuditContext,
    summary: Mapping[str, Any],
    name: str,
    expected_z_modes: Sequence[str],
) -> None:
    groups = _list(summary.get("groups"), f"{name}.groups")
    expected = {(mode, fov) for mode in expected_z_modes for fov in EXPECTED_FOVS}
    observed: set[tuple[str, float]] = set()
    visible_columns = {
        "legacy_planar": "legacy_visible_count",
        "center_top": "center_top_visible_count",
        "sparse_multiray": "sparse_multiray_visible_count",
    }
    for index, raw_group in enumerate(groups):
        group = _mapping(raw_group, f"{name}.groups[{index}]")
        key = (str(group.get("z_mode")), _number(group.get("fov_deg"), "group FOV"))
        context.check(f"{name} unique group {key}", key not in observed)
        observed.add(key)
        counts = _mapping(group.get("counts"), f"{name}.counts")
        ratios = _mapping(group.get("observability_ratio_of_sums"), f"{name}.ratios")
        denominator = _integer(counts.get("total_count"), f"{name}.total_count")
        context.check(f"{name} positive denominator {key}", denominator > 0)
        for method, column in visible_columns.items():
            numerator = _integer(counts.get(column), f"{name}.{column}")
            context.check(
                f"{name} ratio-of-sums {key}/{method}",
                0 <= numerator <= denominator
                and _same_number(ratios.get(method), numerator / denominator),
            )
    context.check(f"{name} complete z/FOV groups", observed == expected)


def _audit_full_run(
    context: _AuditContext,
    *,
    directory: str,
    prefix: str,
    stage: str,
    expected_frame_step: int,
    expected_frame_count: int,
    expected_z_modes: Sequence[str],
    selection: str,
) -> dict[str, Any]:
    summary = context.json_artifact(f"{prefix}_summary", f"{directory}/full_record_summary.json")
    config = context.json_artifact(f"{prefix}_config", f"{directory}/full_record_config.json")
    input_audit = context.json_artifact(
        f"{prefix}_input_audit", f"{directory}/full_record_input_audit.json"
    )
    counts = context.parquet(
        f"{prefix}_counts", f"{directory}/full_record_observability.parquet"
    )
    contributions = context.parquet(
        f"{prefix}_contributions", f"{directory}/observability_contributions.parquet"
    )
    exclusions = context.parquet(f"{prefix}_exclusions", f"{directory}/ego_exclusions.parquet")
    timings = context.parquet(f"{prefix}_timings", f"{directory}/full_record_timings.parquet")
    partial_files = list((context.output_root / directory).glob("*.partial"))
    context.check(f"{prefix} has no interrupted partial output", not partial_files)
    context.check(
        f"{prefix} input gate",
        _mapping(input_audit.get("full_record_quality_gate"), f"{prefix} quality gate").get("passed")
        is True,
    )
    context.check(
        f"{prefix} completed every selected frame",
        summary.get("processed_frame_count") == summary.get("selected_frame_count"),
    )
    context.check(f"{prefix} selected frame count", summary.get("selected_frame_count") == expected_frame_count)
    context.check(f"{prefix} ego-position count", summary.get("ego_positions") == EXPECTED_EGO_COUNT)
    context.check(f"{prefix} frame step", summary.get("frame_step") == expected_frame_step)
    context.check(f"{prefix} is not truncated", summary.get("max_frames") is None)
    summary_config = _mapping(summary.get("config"), f"{prefix}.config")
    context.check(
        f"{prefix} sparse rule",
        summary_config.get("sparse_min_visible_rays") == EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
    )
    context.check(f"{prefix} z modes", summary_config.get("z_modes") == list(expected_z_modes))
    context.check(f"{prefix} FOVs", summary_config.get("fov_degrees") == list(EXPECTED_FOVS))
    file_geometry = _mapping(config.get("geometry"), f"{prefix} file geometry")
    file_selection = _mapping(config.get("selection"), f"{prefix} file selection")
    context.check(
        f"{prefix} file sparse rule",
        file_geometry.get("sparse_min_visible_rays") == EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
    )
    context.check(f"{prefix} file z modes", file_geometry.get("z_modes") == list(expected_z_modes))
    context.check(f"{prefix} file frame step", file_selection.get("frame_step") == expected_frame_step)
    context.check(f"{prefix} file selected frames", file_selection.get("selected_frame_count") == expected_frame_count)
    input_metadata = _mapping(summary.get("input"), f"{prefix}.input")
    context.check(f"{prefix} source quality gate", input_metadata.get("quality_gate_passed") is True)
    provenance = context._load_json(context.paths["input_provenance"], "input provenance")
    canonical_track_sha = _mapping(provenance.get("tracks"), "provenance.tracks").get("sha256")
    context.check(
        f"{prefix} summary source SHA joins canonical provenance",
        input_metadata.get("sha256") == canonical_track_sha,
    )
    input_audit_source = _mapping(input_audit.get("source"), f"{prefix}.input_audit.source")
    context.check(
        f"{prefix} input-audit source SHA joins canonical provenance",
        input_audit_source.get("sha256") == canonical_track_sha,
    )
    for output_name, parquet_file in (
        ("counts", counts),
        ("contributions", contributions),
        ("exclusions", exclusions),
        ("timings", timings),
    ):
        metadata = parquet_file.schema_arrow.metadata or {}
        encoded_source_sha = metadata.get(b"paco.input_sha256")
        context.check(
            f"{prefix} {output_name} Parquet source SHA joins canonical provenance",
            encoded_source_sha is not None
            and encoded_source_sha.decode("ascii", errors="strict") == canonical_track_sha,
        )
    quality_excluded = _mapping(
        input_metadata.get("quality_excluded_frames"), f"{prefix}.quality_excluded"
    )
    only_frames = _mapping(input_metadata.get("only_frames"), f"{prefix}.only_frames")
    if selection == "exclude_partial":
        context.check(
            f"{prefix} requested all partial exclusions",
            tuple(quality_excluded.get("requested", ())) == EXPECTED_QUALITY_EXCLUDED_FRAMES,
        )
        context.check(
            f"{prefix} removed observed partial exclusions",
            tuple(quality_excluded.get("observed_and_removed", ()))
            == EXPECTED_OBSERVED_EXCLUSIONS,
        )
        context.check(f"{prefix} is not an only-frame run", only_frames.get("requested") == [])
    elif selection == "partial_only":
        context.check(f"{prefix} excludes no frames", quality_excluded.get("requested") == [])
        context.check(
            f"{prefix} requests frozen partial frames",
            tuple(only_frames.get("requested", ())) == EXPECTED_QUALITY_EXCLUDED_FRAMES,
        )
        context.check(
            f"{prefix} contains 15 observed partial frames",
            tuple(only_frames.get("observed", ())) == EXPECTED_OBSERVED_EXCLUSIONS,
        )
        context.check(
            f"{prefix} records missing frame 6747",
            tuple(only_frames.get("absent_from_track_table", ()))
            == EXPECTED_ALLOWED_MISSING_FRAMES,
        )
    else:
        raise NotebookAuditError(f"unknown full-run selection contract: {selection}")
    outputs = _mapping(summary.get("outputs"), f"{prefix}.outputs")
    for output_name, parquet_file in (
        ("counts", counts),
        ("contributions", contributions),
        ("exclusions", exclusions),
        ("timings", timings),
    ):
        record = _mapping(outputs.get(output_name), f"{prefix}.outputs.{output_name}")
        context.check(
            f"{prefix} {output_name} row count",
            _parquet_rows(parquet_file) == _integer(record.get("rows"), f"{prefix} rows"),
        )
    _check_full_groups(context, summary, prefix, expected_z_modes)
    context.stage(
        stage,
        f"{expected_frame_count:,} selected frames; step={expected_frame_step}; 3/15 ratios verified",
    )
    return summary


def _check_source_lineage(
    context: _AuditContext,
    payload: Mapping[str, Any],
    *,
    name: str,
    input_alias: str,
    summary_alias: str,
    table_kind: str,
    frame_step: int,
) -> None:
    input_record = context.snapshot(input_alias)
    summary_record = context.snapshot(summary_alias)
    provenance = context._load_json(context.paths["input_provenance"], "input provenance")
    canonical_track_sha = _mapping(provenance.get("tracks"), "provenance.tracks").get("sha256")
    context.check(
        f"{name} input Parquet SHA lineage",
        payload.get("input_parquet_sha256") == input_record["sha256"],
    )
    context.check(
        f"{name} source-summary SHA lineage",
        payload.get("source_summary_sha256") == summary_record["sha256"],
    )
    context.check(
        f"{name} canonical source-input SHA lineage",
        payload.get("source_input_sha256") == canonical_track_sha,
    )
    contract = _mapping(payload.get("source_contract"), f"{name}.source_contract")
    context.check(f"{name} source table kind", contract.get("table_kind") == table_kind)
    context.check(
        f"{name} nested input SHA lineage",
        contract.get("input_parquet_sha256") == input_record["sha256"],
    )
    context.check(
        f"{name} nested summary SHA lineage",
        contract.get("source_summary_sha256") == summary_record["sha256"],
    )
    context.check(
        f"{name} nested canonical source-input SHA lineage",
        contract.get("source_input_sha256") == canonical_track_sha,
    )
    context.check(
        f"{name} source sparse rule",
        contract.get("sparse_min_visible_rays") == EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
    )
    frame_selection = _mapping(contract.get("frame_selection"), f"{name}.frame_selection")
    context.check(f"{name} source frame step", frame_selection.get("frame_step") == frame_step)
    context.check(f"{name} exact source frame set", frame_selection.get("exact_frame_set_validated") is True)
    context.check(f"{name} complete source group frames", frame_selection.get("group_frame_sets_complete") is True)


def _audit_bootstrap(context: _AuditContext, *, primary: bool) -> dict[str, Any]:
    if primary:
        prefix = "primary_bootstrap"
        directory = "bootstrap_primary"
        source_prefix = "primary_full"
        stage = "primary_bootstrap"
        rate = 10.0
        block_frames = 1_200
        frame_step = 1
    else:
        prefix = "z_bootstrap"
        directory = "z_mode_sensitivity_step10/bootstrap_120s"
        source_prefix = "z_full"
        stage = "z_mode_sensitivity"
        rate = 1.0
        block_frames = 120
        frame_step = 10
    summary = context.json_artifact(f"{prefix}_summary", f"{directory}/observability_bootstrap.json")
    replicates = context.parquet(
        f"{prefix}_replicates", f"{directory}/observability_bootstrap_replicates.parquet"
    )
    _check_source_lineage(
        context,
        summary,
        name=prefix,
        input_alias=f"{source_prefix}_contributions",
        summary_alias=f"{source_prefix}_summary",
        table_kind="contributions",
        frame_step=frame_step,
    )
    estimand = _mapping(summary.get("estimand"), f"{prefix}.estimand")
    design = _mapping(summary.get("bootstrap"), f"{prefix}.bootstrap")
    context.check(f"{prefix} ratio-of-sums estimand", estimand.get("name") == "ratio_of_sums")
    context.check(f"{prefix} circular moving blocks", design.get("circular") is True)
    context.check(f"{prefix} frame rate", _same_number(design.get("frame_rate_hz"), rate))
    context.check(f"{prefix} 120-second duration", _same_number(design.get("block_seconds"), 120.0))
    context.check(f"{prefix} block frame count", design.get("block_frame_count") == block_frames)
    context.check(
        f"{prefix} replicate count",
        design.get("n_resamples") == EXPECTED_BOOTSTRAP_REPLICATES,
    )
    context.check(f"{prefix} seed", design.get("seed") == EXPECTED_SEED)
    context.check(
        f"{prefix} replicate row count",
        _parquet_rows(replicates) == _integer(summary.get("replicate_row_count"), "replicate rows"),
    )
    for index, raw_group in enumerate(_list(summary.get("groups"), f"{prefix}.groups")):
        group = _mapping(raw_group, f"{prefix}.groups[{index}]")
        numerator = _number(group.get("numerator_sum"), "bootstrap numerator")
        denominator = _number(group.get("denominator_sum"), "bootstrap denominator")
        context.check(
            f"{prefix} ratio group {index}",
            denominator > 0 and _same_number(group.get("estimate"), numerator / denominator),
        )
    if primary:
        context.stage(stage, "10 Hz circular 120-second bootstrap with 5,000 replicates verified")
    return summary


def _audit_block_sensitivity(context: _AuditContext) -> None:
    payload = context.json_artifact(
        "block_sensitivity_summary",
        "bootstrap_block_sensitivity/observability_block_sensitivity.json",
    )
    _check_source_lineage(
        context,
        payload,
        name="block_sensitivity",
        input_alias="primary_full_contributions",
        summary_alias="primary_full_summary",
        table_kind="contributions",
        frame_step=1,
    )
    context.check(
        "block sensitivity ratio-of-sums",
        _mapping(payload.get("estimand"), "block estimand").get("name") == "ratio_of_sums",
    )
    design = _mapping(payload.get("bootstrap"), "block bootstrap")
    durations = _list(design.get("block_durations"), "block durations")
    context.check(
        "block sensitivity durations",
        [float(_mapping(item, "block duration").get("seconds")) for item in durations]
        == [30.0, 60.0, 120.0, 240.0],
    )
    context.check("block sensitivity reference", _same_number(design.get("reference_block_seconds"), 120.0))
    context.check("block sensitivity replicates", design.get("n_resamples") == 5_000)
    context.check("block sensitivity seed", design.get("seed") == EXPECTED_SEED)
    context.check("block sensitivity has groups", len(_list(payload.get("groups"), "block groups")) > 0)
    context.stage("block_sensitivity", "30/60/120/240-second dependence sensitivity verified")


def _audit_ranking(context: _AuditContext) -> None:
    payload = context.json_artifact(
        "residual_ranking_summary", "residual_demand/residual_hotspot_ranks.json"
    )
    ranks = context.parquet(
        "residual_ranking_table", "residual_demand/residual_hotspot_ranks.parquet"
    )
    _check_source_lineage(
        context,
        payload,
        name="residual_ranking",
        input_alias="primary_full_counts",
        summary_alias="primary_full_summary",
        table_kind="counts",
        frame_step=1,
    )
    scope = _mapping(payload.get("scope"), "ranking.scope")
    design = _mapping(payload.get("bootstrap"), "ranking.bootstrap")
    context.check("ranking ground z mode", scope.get("z_mode") == "ground_anchored")
    context.check("ranking calibrated sparse method", scope.get("method") == "sparse_multiray")
    context.check("ranking FOVs", scope.get("fov_degrees") == list(EXPECTED_FOVS))
    context.check("ranking 1,200-frame clusters", design.get("block_width_frames") == 1_200)
    context.check("ranking 5,000 bootstraps", design.get("n_resamples") == 5_000)
    context.check("ranking seed", design.get("seed") == EXPECTED_SEED)
    context.check("ranking table is nonempty", _parquet_rows(ranks) > 0)
    context.stage("residual_hotspot_ranking", "uniform and VRU-weighted residual rankings verified")


def _audit_partial_and_sensitivity(context: _AuditContext) -> None:
    _audit_full_run(
        context,
        directory="partial_scan_only",
        prefix="partial_full",
        stage="partial_scan_sensitivity",
        expected_frame_step=1,
        expected_frame_count=EXPECTED_PARTIAL_OBSERVED_FRAMES,
        expected_z_modes=("ground_anchored",),
        selection="partial_only",
    )
    payload = context.json_artifact(
        "sensitivity_summary", "sensitivity_summary/cacie_sensitivity_summary.json"
    )
    contracts = _mapping(payload.get("contracts"), "sensitivity.contracts")
    context.check(
        "sensitivity sparse rule",
        contracts.get("sparse_min_visible_rays") == EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
    )
    context.check("sensitivity primary step", contracts.get("primary_frame_step") == 1)
    context.check("sensitivity z step", contracts.get("z_mode_sensitivity_frame_step") == 10)
    expected_inputs = {
        "primary_summary": "primary_full_summary",
        "primary_bootstrap": "primary_bootstrap_summary",
        "z_summary": "z_full_summary",
        "z_bootstrap_replicates": "z_bootstrap_replicates",
        "partial_summary": "partial_full_summary",
    }
    inputs = _mapping(payload.get("inputs"), "sensitivity.inputs")
    for key, alias in expected_inputs.items():
        receipt = _mapping(inputs.get(key), f"sensitivity.inputs.{key}")
        current = context.snapshot(alias)
        context.check(f"sensitivity {key} SHA lineage", receipt.get("sha256") == current["sha256"])
        context.check(
            f"sensitivity {key} size lineage",
            receipt.get("size_bytes") == current["size_bytes"],
        )
    for section in ("z_mode_step10", "decimation_step10_vs_full", "partial_scan_inclusion"):
        section_value = _mapping(payload.get(section), f"sensitivity.{section}")
        context.check(
            f"sensitivity {section} comparisons present",
            len(_list(section_value.get("comparisons"), f"{section}.comparisons")) > 0,
        )
    context.stage(
        "sensitivity_summary",
        "raw/ground, decimation, and partial-scan comparisons plus all input hashes verified",
    )


def _audit_figures(context: _AuditContext) -> None:
    for stage, stem in (
        ("validation_figure", "cacie_validation"),
        ("full_results_figure", "cacie_full_results"),
    ):
        pdf_alias = f"{stem}_pdf"
        png_alias = f"{stem}_png"
        pdf = context.artifact(pdf_alias, f"figures/{stem}.pdf")
        png = context.artifact(png_alias, f"figures/{stem}.png")
        context.check(f"{stem} PDF signature", pdf.read_bytes()[:5] == b"%PDF-")
        context.check(f"{stem} PNG signature", png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
        context.check(f"{stem} figure files nonempty", pdf.stat().st_size > 100 and png.stat().st_size > 100)
        if stem == "cacie_validation":
            input_aliases = {
                "main_summary": "main_summary",
                "strata_summary": "strata_summary",
                "main_decisions": "main_decisions",
                "calibration_summary": "calibration_summary",
                "convergence_summary": "convergence_summary",
            }
            derivation = "cacie_validation_figure"
        else:
            input_aliases = {
                "full_summary": "primary_full_summary",
                "primary_bootstrap": "primary_bootstrap_summary",
                "residual_summary": "residual_ranking_summary",
                "residual_ranking": "residual_ranking_table",
                "sensitivity_summary": "sensitivity_summary",
                "block_sensitivity": "block_sensitivity_summary",
            }
            derivation = "cacie_full_results_figure"
        _audit_derivation_receipt(
            context,
            alias=f"{stem}_source_lineage_receipt",
            relative=f"figures/{stem}.source_lineage_receipt.json",
            derivation=derivation,
            input_aliases=input_aliases,
            output_aliases={"pdf": pdf_alias, "png": png_alias},
        )
        context.stage(stage, "publication PDF and PNG signatures verified")


def _audit_detector_validation(context: _AuditContext) -> None:
    evaluation = context.json_artifact(
        "detector_primary_evaluation",
        "detector_validation/full_holdout/cacie_detector_validation_full_holdout_excluding_partial_scans.json",
    )
    geometry = context.json_artifact(
        "detector_matched_geometry",
        "detector_validation/full_holdout/matched_box_geometry.json",
    )
    canonical = context.json_artifact(
        "detector_canonical_receipt",
        "detector_validation/source_receipts/canonical_artifact_counts.json",
    )
    human = context.json_artifact(
        "detector_human_label_receipt",
        "detector_validation/source_receipts/human_label_source.json",
    )
    split = context.json_artifact(
        "detector_split_receipt",
        "detector_validation/source_receipts/model_split_manifest.json",
    )
    overlap = context.json_artifact(
        "detector_overlap_receipt",
        "detector_validation/source_receipts/split_overlap_audit.json",
    )
    context.check("detector holdout window", evaluation.get("window") == [6401, 7200])
    context.check("detector match threshold", _same_number(evaluation.get("match_thresh_m"), 1.5))
    runs = _list(evaluation.get("runs"), "detector runs")
    context.check("one canonical detector run", len(runs) == 1)
    run = _mapping(runs[0], "detector run")
    context.check("canonical detector run ID", run.get("run_id") == "run-cacie-canonical-full-20260902")
    class_metrics = _list(evaluation.get("class_metrics"), "detector class metrics")
    context.check(
        "detector class set",
        {str(_mapping(row, "class metric").get("class_name")) for row in class_metrics}
        == {"bicycle", "car", "pedestrian", "truck"},
    )
    for row in class_metrics:
        metric = _mapping(row, "detector class metric")
        tp = _integer(metric.get("tp"), "detector tp")
        fp = _integer(metric.get("fp"), "detector fp")
        fn = _integer(metric.get("fn"), "detector fn")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_name = str(metric.get("class_name"))
        context.check(f"detector {class_name} precision", _same_number(metric.get("precision"), precision))
        context.check(f"detector {class_name} recall", _same_number(metric.get("recall"), recall))
        context.check(f"detector {class_name} F1", _same_number(metric.get("f1"), f1))
    geometry_scope = _mapping(geometry.get("scope"), "detector geometry scope")
    context.check("detector geometry is not visibility ground truth", geometry_scope.get("not_visibility_ground_truth") is True)
    identity = _mapping(geometry.get("detection_track_identity"), "detector identity")
    context.check("detector/track identity gate", identity.get("passed") is True)
    context.check("detector matched boxes positive", _integer(geometry.get("matched_boxes"), "matched boxes") > 0)
    provenance = context._load_json(context.paths["input_provenance"], "input provenance")
    provenance_track_sha = _mapping(provenance.get("tracks"), "provenance tracks").get("sha256")
    canonical_tracks = _mapping(
        _mapping(canonical.get("canonical_artifacts"), "canonical detector artifacts").get("tracks"),
        "canonical tracks",
    )
    context.check("detector track SHA joins observability provenance", canonical_tracks.get("sha256") == provenance_track_sha)
    geometry_inputs = _mapping(geometry.get("inputs"), "detector geometry inputs")
    context.check("matched geometry track SHA joins provenance", geometry_inputs.get("tracks_sha256") == provenance_track_sha)
    human_sha = str(human.get("sha256", ""))
    context.check("human-label receipt SHA format", len(human_sha) == 64 and all(char in "0123456789abcdef" for char in human_sha))
    context.check("detector split has 800 validation frames", split.get("num_val_frames") == 800)
    context.check("detector validation split range", split.get("val_start") == 6401 and split.get("val_end") == 7200)
    context.check("detector train/validation overlap is zero", overlap.get("train_validation_token_intersection_count") == 0)
    validation_info = _mapping(overlap.get("validation_info"), "detector validation split info")
    context.check("detector validation split exact range", validation_info.get("is_exact_integer_range_6401_7200") is True)
    context.discover_artifact_tree("detector_validation", "detector_validation")
    context.stage(
        "detector_validation",
        "training-isolated 6401-7200 internal detector/box validation and source receipts verified",
    )


def _runtime_producer_binding_mode(
    receipt: Mapping[str, Any],
    *,
    expected_declaration: Mapping[str, Any],
) -> str:
    """Validate new dual-layer or preserved historical runtime provenance.

    Schema-1.1 runner receipts bind the historical numeric producer with a
    byte-verified declaration while keeping the actual execution checkout in
    ``git_*``.  The two schema-1.0 historical forms remain admissible only
    when their sole commit evidence equals the frozen producer: a repository
    receipt captured at that commit, or an explicitly unverified no-Git source
    copy declaration.
    """

    expected_commit = str(expected_declaration.get("commit", ""))
    embedded = receipt.get("numeric_producer")
    if embedded is not None:
        if not isinstance(embedded, Mapping):
            raise NotebookAuditError("runtime numeric_producer must be a JSON object")
        if dict(embedded) != dict(expected_declaration):
            raise NotebookAuditError(
                "runtime byte-verified numeric producer declaration is stale or inconsistent"
            )
        return "byte_verified_numeric_producer"

    repository_available = receipt.get("git_repository_available")
    if repository_available is None and isinstance(receipt.get("git_commit"), str):
        repository_available = True
    if repository_available:
        if receipt.get("git_commit") != expected_commit:
            raise NotebookAuditError(
                "legacy repository runtime receipt does not identify the frozen numeric producer"
            )
        return "legacy_execution_commit_equals_numeric_producer"
    if receipt.get("declared_source_commit") != expected_commit:
        raise NotebookAuditError(
            "legacy no-Git runtime receipt does not declare the frozen numeric producer"
        )
    if receipt.get("git_provenance_status") != "declared_unverified_source_tree_without_git":
        raise NotebookAuditError(
            "legacy no-Git runtime receipt lacks the required unverified declaration status"
        )
    return "legacy_unverified_no_git_numeric_producer_declaration"


def _audit_runtime_environment(context: _AuditContext) -> None:
    receipts = {
        "validation_main_200_final": context.json_artifact(
            "validation_main_runtime_receipt", "validation_main/runtime_environment.json"
        ),
        "validation_convergence_25_final": context.json_artifact(
            "validation_convergence_runtime_receipt",
            "validation_convergence/runtime_environment.json",
        ),
        "full_record_primary_40779": context.json_artifact(
            "full_record_primary_runtime_receipt",
            "full_record_primary/runtime_environment.json",
        ),
    }
    config = context._load_json(context.paths["analysis_config"], "analysis config")
    release = _mapping(config.get("release"), "analysis_config.release")
    declared_commit = str(release.get("numeric_producer_commit", ""))
    context.check(
        "numeric producer commit declaration is a full SHA",
        len(declared_commit) == 40
        and all(character in "0123456789abcdef" for character in declared_commit),
    )
    producer_files = release.get("numeric_producer_files")
    context.check(
        "numeric producer file declaration is nonempty",
        isinstance(producer_files, list)
        and bool(producer_files)
        and all(isinstance(item, str) and bool(item.strip()) for item in producer_files),
    )
    try:
        producer_declaration = validate_numeric_producer_declaration(
            base_dir=context.repo_root,
            config_path=context.paths["analysis_config"],
        )
    except ValueError as exc:
        raise NotebookAuditError(f"numeric producer declaration failed: {exc}") from exc
    context.check(
        "numeric producer files match declared Git blobs",
        producer_declaration is not None and producer_declaration.get("status") == "VERIFIED",
    )

    execution_commits: set[str] = set()
    dirty_labels: list[str] = []
    legacy_runtime_labels: list[str] = []
    producer_binding_modes: dict[str, str] = {}
    for expected_label, receipt in receipts.items():
        schema_version = receipt.get("schema_version")
        context.check(f"{expected_label} runtime schema", schema_version in {"1.0", "1.1"})
        context.check(f"{expected_label} runtime label", receipt.get("label") == expected_label)
        if schema_version == "1.1":
            context.check(
                f"{expected_label} execution Git role is explicit",
                receipt.get("git_commit_role") == "execution_source_tree_at_receipt_capture",
            )
        repository_available = receipt.get("git_repository_available")
        if repository_available is None and isinstance(receipt.get("git_commit"), str):
            repository_available = True
            legacy_runtime_labels.append(expected_label)
        context.check(
            f"{expected_label} runtime repository availability is explicit",
            isinstance(repository_available, bool),
        )
        if repository_available:
            commit = str(receipt.get("git_commit", ""))
            context.check(
                f"{expected_label} runtime verified Git commit",
                len(commit) == 40
                and all(character in "0123456789abcdef" for character in commit),
            )
            context.check(
                f"{expected_label} runtime repository provenance status",
                receipt.get("git_provenance_status") in {None, "repository_verified"},
            )
            context.check(
                f"{expected_label} runtime has no declared source override",
                receipt.get("declared_source_commit") is None,
            )
            context.check(
                f"{expected_label} runtime Git dirty flag is boolean",
                isinstance(receipt.get("git_dirty"), bool),
            )
            if receipt.get("git_dirty") is True:
                dirty_labels.append(expected_label)
            execution_commits.add(commit)
        else:
            source_copy_commit = str(receipt.get("declared_source_commit", ""))
            context.check(
                f"{expected_label} runtime declared source commit",
                len(source_copy_commit) == 40
                and all(character in "0123456789abcdef" for character in source_copy_commit),
            )
            context.check(
                f"{expected_label} runtime unverified declaration status",
                receipt.get("git_provenance_status")
                == "declared_unverified_source_tree_without_git",
            )
            context.check(f"{expected_label} runtime Git commit is unavailable", receipt.get("git_commit") is None)
            context.check(f"{expected_label} runtime Git branch is unavailable", receipt.get("git_branch") is None)
            context.check(f"{expected_label} runtime dirty state is unavailable", receipt.get("git_dirty") is None)
        for package in ("numpy_version", "pandas_version", "pyarrow_version"):
            context.check(f"{expected_label} runtime {package}", bool(receipt.get(package)))
        execution_mode = receipt.get("execution_mode")
        context.check(
            f"{expected_label} runtime execution-mode value",
            execution_mode in {None, "uv_frozen", "external_python"},
        )
        if execution_mode == "uv_frozen":
            context.check(f"{expected_label} runtime uv version", bool(receipt.get("uv_version")))
        assert producer_declaration is not None
        producer_binding_modes[expected_label] = _runtime_producer_binding_mode(
            receipt,
            expected_declaration=producer_declaration,
        )
        context.check(
            f"{expected_label} runtime binds the frozen numeric producer",
            True,
            producer_binding_modes[expected_label],
        )
    for alias in ("main_source_lineage_receipt", "convergence_source_lineage_receipt"):
        lineage = context._load_json(context.paths[alias], alias)
        context.check(
            f"{alias} producer commit equals frozen numeric producer commit",
            lineage.get("declared_numeric_producer_commit") == declared_commit,
        )
    context.stage(
        "runtime_environment",
        "main-validation, convergence, and full-record runtime receipts verified; "
        f"execution commits disclosed as {sorted(execution_commits) or 'unavailable'}; "
        f"numeric-producer bindings {producer_binding_modes}; "
        f"historical dirty state disclosed for {dirty_labels or 'none'}; "
        f"legacy receipt schema disclosed for {legacy_runtime_labels or 'none'}",
    )


def audit_canonical_artifacts(*, repo_root: Path, output_root: Path) -> dict[str, Any]:
    """Audit every required CACIE stage and return a deterministic PASS payload."""

    context = _AuditContext(repo_root, output_root)
    _audit_provenance(context)
    _audit_main_validation(context)
    _audit_calibration(context)
    _audit_strata(context)
    _audit_convergence(context)
    _audit_full_run(
        context,
        directory="full_record_primary",
        prefix="primary_full",
        stage="full_record_primary",
        expected_frame_step=1,
        expected_frame_count=EXPECTED_PRIMARY_FRAMES,
        expected_z_modes=("ground_anchored",),
        selection="exclude_partial",
    )
    _audit_bootstrap(context, primary=True)
    _audit_block_sensitivity(context)
    _audit_ranking(context)
    _audit_full_run(
        context,
        directory="z_mode_sensitivity_step10",
        prefix="z_full",
        stage="z_mode_sensitivity",
        expected_frame_step=10,
        expected_frame_count=EXPECTED_STEP10_FRAMES,
        expected_z_modes=("ground_anchored", "raw"),
        selection="exclude_partial",
    )
    _audit_bootstrap(context, primary=False)
    _audit_partial_and_sensitivity(context)
    _audit_figures(context)
    _audit_detector_validation(context)
    _audit_runtime_environment(context)

    stage_names = tuple(row["stage"] for row in context.stages)
    context.check("every required audit stage is present", set(stage_names) == set(REQUIRED_STAGES))
    context.check(
        "every required audit stage is VERIFIED",
        all(row["status"] == "VERIFIED" for row in context.stages),
    )
    for name in sorted(context.paths):
        path = context.paths[name]
        current = {
            "name": name,
            **_stable_file_record(path, context.display_path(path)),
        }
        if current != context.records[name]:
            raise NotebookAuditError(
                f"artifact changed between validation and final audit hash: {current['path']}"
            )
    return {
        "schema_version": "1.0",
        "audit": "cacie_notebook_artifacts",
        "status": "PASS",
        "pass_marker": PASS_MARKER,
        "output_root": context.display_path(context.output_root),
        "required_stages": list(REQUIRED_STAGES),
        "stages": context.stages,
        "check_count": len(context.checks),
        "checks": context.checks,
        "audited_files": [context.records[name] for name in sorted(context.records)],
    }


def _notebook_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    return ""


def validate_executed_notebook(*, source_notebook: Path, executed_notebook: Path) -> None:
    """Require an exact-source, fully executed notebook with one PASS output."""

    try:
        source = json.loads(source_notebook.read_text(encoding="utf-8"))
        executed = json.loads(executed_notebook.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NotebookAuditError("source or executed notebook is not readable JSON") from exc
    source_cells = _list(_mapping(source, "source notebook").get("cells"), "source cells")
    executed_cells = _list(_mapping(executed, "executed notebook").get("cells"), "executed cells")
    if len(source_cells) != len(executed_cells):
        raise NotebookAuditError("executed notebook cell count differs from source notebook")
    for index, (source_cell_raw, executed_cell_raw) in enumerate(
        zip(source_cells, executed_cells, strict=True)
    ):
        source_cell = _mapping(source_cell_raw, f"source cell {index}")
        executed_cell = _mapping(executed_cell_raw, f"executed cell {index}")
        if source_cell.get("cell_type") != executed_cell.get("cell_type"):
            raise NotebookAuditError(f"executed notebook cell type changed at index {index}")
        if _notebook_text(source_cell.get("source")) != _notebook_text(
            executed_cell.get("source")
        ):
            raise NotebookAuditError(f"executed notebook source changed at cell index {index}")
    code_cells = [
        (index, _mapping(cell, f"executed cell {index}"))
        for index, cell in enumerate(executed_cells)
        if _mapping(cell, f"executed cell {index}").get("cell_type") == "code"
    ]
    if not code_cells:
        raise NotebookAuditError("executed notebook contains no code cells")
    missing = [index for index, cell in code_cells if cell.get("execution_count") is None]
    if missing:
        raise NotebookAuditError(f"executed notebook has unexecuted code cells: {missing}")
    counts = [cell.get("execution_count") for _, cell in code_cells]
    if any(isinstance(count, bool) or not isinstance(count, int) or count <= 0 for count in counts):
        raise NotebookAuditError("executed notebook code-cell counts must be positive integers")
    if counts != sorted(counts) or len(set(counts)) != len(counts):
        raise NotebookAuditError("executed notebook code-cell counts must be strictly increasing")
    final_code_index, final_code_cell = code_cells[-1]
    final_source = _notebook_text(final_code_cell.get("source"))
    if "print(PASS_MARKER)" not in final_source:
        raise NotebookAuditError("the final code cell must be the PASS-marker enforcement cell")
    errors = []
    marker_occurrences_by_cell: dict[int, int] = {}
    for index, cell in code_cells:
        occurrences = 0
        for raw_output in _list(cell.get("outputs"), f"executed cell {index} outputs"):
            output = _mapping(raw_output, f"executed cell {index} output")
            if output.get("output_type") == "error":
                errors.append((index, output.get("ename"), output.get("evalue")))
            text = _notebook_text(output.get("text"))
            data = output.get("data")
            if isinstance(data, Mapping):
                text += _notebook_text(data.get("text/plain"))
                text += _notebook_text(data.get("text/markdown"))
            occurrences += text.count(PASS_MARKER)
        marker_occurrences_by_cell[index] = occurrences
    if errors:
        raise NotebookAuditError(f"executed notebook contains error outputs: {errors}")
    nonfinal_marker_cells = [
        index
        for index, occurrences in marker_occurrences_by_cell.items()
        if index != final_code_index and occurrences
    ]
    if nonfinal_marker_cells:
        raise NotebookAuditError(
            f"PASS marker appeared outside the final enforcement cell: {nonfinal_marker_cells}"
        )
    final_occurrences = marker_occurrences_by_cell[final_code_index]
    if final_occurrences != 1:
        raise NotebookAuditError(
            f"final enforcement cell must emit {PASS_MARKER} exactly once; "
            f"found {final_occurrences}"
        )


def build_notebook_audit_receipt(
    *,
    source_notebook: Path,
    executed_notebook: Path,
    artifact_audit: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Bind an artifact PASS audit to immutable source/executed notebook bytes."""

    validate_executed_notebook(
        source_notebook=source_notebook,
        executed_notebook=executed_notebook,
    )
    audit = dict(artifact_audit)
    if audit.get("status") != "PASS" or audit.get("pass_marker") != PASS_MARKER:
        raise NotebookAuditError("artifact audit did not produce the required PASS contract")
    stages = _list(audit.get("stages"), "artifact audit stages")
    actual_stage_names = {str(_mapping(row, "stage row").get("stage")) for row in stages}
    if len(stages) != len(REQUIRED_STAGES) or actual_stage_names != set(REQUIRED_STAGES):
        raise NotebookAuditError("artifact audit does not contain every required stage")
    if any(_mapping(row, "stage row").get("status") != "VERIFIED" for row in stages):
        raise NotebookAuditError("artifact audit contains a non-VERIFIED required stage")
    root = repo_root.resolve(strict=True)

    def display(path: Path) -> str:
        resolved = path.resolve(strict=True)
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return str(resolved)

    return {
        "schema_version": "1.0",
        "receipt": "cacie_notebook_audit",
        "status": "PASS",
        "pass_marker": PASS_MARKER,
        "source_notebook": _stable_file_record(source_notebook, display(source_notebook)),
        "executed_notebook": _stable_file_record(executed_notebook, display(executed_notebook)),
        "required_stages": list(REQUIRED_STAGES),
        "stages": stages,
        "check_count": _integer(audit.get("check_count"), "artifact audit check_count"),
        "checks": _list(audit.get("checks"), "artifact audit checks"),
        "audited_files": _list(audit.get("audited_files"), "artifact audit audited_files"),
    }


def write_notebook_audit_receipt(
    *,
    receipt_path: Path,
    source_notebook: Path,
    executed_notebook: Path,
    artifact_audit: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Atomically write a notebook audit receipt."""

    receipt = build_notebook_audit_receipt(
        source_notebook=source_notebook,
        executed_notebook=executed_notebook,
        artifact_audit=artifact_audit,
        repo_root=repo_root,
    )
    output = receipt_path.resolve(strict=False)
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


def verify_notebook_audit_receipt(
    *,
    receipt_path: Path,
    source_notebook: Path,
    executed_notebook: Path,
    artifact_audit: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Recompute the complete receipt and reject any stale byte or status."""

    try:
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NotebookAuditError(f"notebook audit receipt is not readable JSON: {receipt_path}") from exc
    if not isinstance(stored, dict):
        raise NotebookAuditError("notebook audit receipt must contain a JSON object")
    current = build_notebook_audit_receipt(
        source_notebook=source_notebook,
        executed_notebook=executed_notebook,
        artifact_audit=artifact_audit,
        repo_root=repo_root,
    )
    if stored != current:
        raise NotebookAuditError(
            "notebook audit receipt is stale: source notebook, executed notebook, "
            "stage status, checks, or audited artifact bytes changed"
        )
    return current


__all__ = [
    "NotebookAuditError",
    "PASS_MARKER",
    "REQUIRED_STAGES",
    "audit_canonical_artifacts",
    "build_notebook_audit_receipt",
    "validate_executed_notebook",
    "verify_notebook_audit_receipt",
    "write_notebook_audit_receipt",
]
