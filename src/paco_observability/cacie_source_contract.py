"""Fail-closed lineage checks for canonical CACIE full-record consumers.

The numerical analysis functions intentionally remain usable with synthetic
``DataFrame`` inputs.  Canonical command-line consumers call this module to bind
their Parquet input to a completed ``full_record_summary.json``, verify the
declared frame-selection contract, and retain content hashes in downstream
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .derivation_receipt import (
    require_named_files_unchanged,
    snapshot_named_files,
)


EXPECTED_FRAME_RANGE = (1, 40_795)
EXPECTED_ALLOWED_MISSING_FRAMES = (6_747,)
EXPECTED_QUALITY_EXCLUDED_FRAMES = (
    6_674,
    6_675,
    6_747,
    7_025,
    7_435,
    11_077,
    11_155,
    11_166,
    11_178,
    11_189,
    11_201,
    11_212,
    11_213,
    11_235,
    11_247,
    40_795,
)
EXPECTED_OBSERVED_EXCLUSIONS = tuple(
    frame
    for frame in EXPECTED_QUALITY_EXCLUDED_FRAMES
    if frame not in EXPECTED_ALLOWED_MISSING_FRAMES
)
EXPECTED_EXCLUSION_REASON = "partial_scan_point_count_ratio_below_0.6_of_local_median"
EXPECTED_METHODS = ("legacy_planar", "center_top", "sparse_multiray")
EXPECTED_FOVS = (120.0, 360.0)
EXPECTED_EGO_COUNT = 60
EXPECTED_SPARSE_MIN_VISIBLE_RAYS = 3
NOMINAL_SOURCE_FRAME_RATE_HZ = 10.0


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite numeric data")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric data")
    return result


def _integer(value: Any, name: str) -> int:
    result = _finite(value, name)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result)


def _integers(value: Any, name: str) -> tuple[int, ...]:
    result = tuple(_integer(item, name) for item in _list(value, name))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate values")
    return result


def _floats(value: Any, name: str) -> tuple[float, ...]:
    result = tuple(_finite(item, name) for item in _list(value, name))
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate values")
    return result


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"source summary does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"source summary is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("source summary must contain a JSON object")
    return value


def _source_files(
    input_path: Path, source_summary_path: Path
) -> dict[str, Path]:
    return {
        "input_parquet": input_path,
        "source_summary": source_summary_path,
    }


def snapshot_cacie_full_record_source(
    input_path: Path, source_summary_path: Path
) -> dict[str, dict[str, Any]]:
    """Snapshot both canonical source files before a consumer loads either one."""

    return snapshot_named_files(_source_files(input_path, source_summary_path))


def require_cacie_full_record_source_unchanged(
    source_snapshot: Mapping[str, Mapping[str, Any]],
    input_path: Path,
    source_summary_path: Path,
) -> None:
    """Reject replacement or mutation of either source since ``source_snapshot``."""

    expected_roles = {"input_parquet", "source_summary"}
    if set(source_snapshot) != expected_roles:
        raise ValueError(
            "CACIE source snapshot must contain exactly input_parquet and source_summary"
        )
    try:
        require_named_files_unchanged(
            source_snapshot,
            _source_files(input_path, source_summary_path),
        )
    except ValueError as exc:
        raise ValueError(
            "CACIE source Parquet or summary changed after the pre-load snapshot"
        ) from exc


def _expected_selected_frames(frame_step: int) -> tuple[int, ...]:
    observed = [
        frame
        for frame in range(EXPECTED_FRAME_RANGE[0], EXPECTED_FRAME_RANGE[1] + 1)
        if frame not in EXPECTED_ALLOWED_MISSING_FRAMES
        and frame not in EXPECTED_OBSERVED_EXCLUSIONS
    ]
    return tuple(observed[::frame_step])


def _delta_counts(frame_ids: Sequence[int]) -> dict[str, int]:
    if len(frame_ids) < 2:
        return {}
    unique, counts = np.unique(np.diff(np.asarray(frame_ids, dtype=np.int64)), return_counts=True)
    return {str(int(delta)): int(count) for delta, count in zip(unique, counts, strict=True)}


def _metadata(schema_metadata: Mapping[bytes, bytes] | None) -> dict[str, str]:
    return {
        key.decode("utf-8"): value.decode("utf-8")
        for key, value in (schema_metadata or {}).items()
    }


def _metadata_json(metadata: Mapping[str, str], key: str) -> Any:
    if key not in metadata:
        raise ValueError(f"input Parquet schema is missing required metadata {key!r}")
    try:
        return json.loads(metadata[key])
    except json.JSONDecodeError as exc:
        raise ValueError(f"input Parquet metadata {key!r} is not valid JSON") from exc


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    allowed_frame_steps: Sequence[int],
    expected_ego_count: int,
) -> tuple[int, tuple[int, ...], tuple[str, ...], tuple[float, ...]]:
    if summary.get("schema_version") != "1.0":
        raise ValueError("source summary schema_version must be '1.0'")
    processed = _integer(summary.get("processed_frame_count"), "processed_frame_count")
    selected = _integer(summary.get("selected_frame_count"), "selected_frame_count")
    if processed != selected or processed <= 0:
        raise ValueError("source summary is not a completed selected-frame run")
    if summary.get("max_frames") is not None:
        raise ValueError("canonical source summary must not be max-frame truncated")
    frame_step = _integer(summary.get("frame_step"), "frame_step")
    allowed_steps = tuple(int(value) for value in allowed_frame_steps)
    if frame_step not in allowed_steps:
        raise ValueError(
            f"source frame_step={frame_step} is incompatible with this consumer; "
            f"allowed={allowed_steps}"
        )

    input_metadata = _mapping(summary.get("input"), "source summary input")
    if input_metadata.get("quality_gate_passed") is not True:
        raise ValueError("source summary input quality gate did not pass")
    source_input_sha = str(input_metadata.get("sha256", ""))
    if len(source_input_sha) != 64:
        raise ValueError("source summary input SHA-256 is missing or malformed")
    expected_range = _integers(
        input_metadata.get("expected_frame_range"), "input.expected_frame_range"
    )
    if expected_range != EXPECTED_FRAME_RANGE:
        raise ValueError(f"expected frame range must be {EXPECTED_FRAME_RANGE}")
    allowed_missing = _integers(
        input_metadata.get("allowed_missing_frames"), "input.allowed_missing_frames"
    )
    if allowed_missing != EXPECTED_ALLOWED_MISSING_FRAMES:
        raise ValueError(
            f"allowed missing frames must be {EXPECTED_ALLOWED_MISSING_FRAMES}"
        )

    exclusions = _mapping(
        input_metadata.get("quality_excluded_frames"), "input.quality_excluded_frames"
    )
    requested = _integers(exclusions.get("requested"), "excluded.requested")
    observed = _integers(
        exclusions.get("observed_and_removed"), "excluded.observed_and_removed"
    )
    absent = _integers(
        exclusions.get("absent_from_track_table"), "excluded.absent_from_track_table"
    )
    if requested != EXPECTED_QUALITY_EXCLUDED_FRAMES:
        raise ValueError("source summary does not contain the 16 declared exclusions")
    if observed != EXPECTED_OBSERVED_EXCLUSIONS or absent != EXPECTED_ALLOWED_MISSING_FRAMES:
        raise ValueError("source exclusions must contain 15 observed frames and absent frame 6747")
    if exclusions.get("reason") != EXPECTED_EXCLUSION_REASON:
        raise ValueError("source frame-exclusion reason does not match the CACIE contract")
    only_frames = _mapping(input_metadata.get("only_frames"), "input.only_frames")
    for key in ("requested", "observed", "absent_from_track_table"):
        if _integers(only_frames.get(key), f"only_frames.{key}"):
            raise ValueError("canonical bootstrap/rank sources must not be only-frame runs")

    config = _mapping(summary.get("config"), "source summary config")
    threshold = _integer(
        config.get("sparse_min_visible_rays"), "config.sparse_min_visible_rays"
    )
    if threshold != EXPECTED_SPARSE_MIN_VISIBLE_RAYS:
        raise ValueError("source summary must use the post-development-calibrated 3-of-15 rule")
    methods = tuple(_mapping(config.get("method_definitions"), "config.method_definitions"))
    if set(methods) != set(EXPECTED_METHODS):
        raise ValueError(f"source method set must be exactly {EXPECTED_METHODS}")
    sparse_definition = str(config["method_definitions"]["sparse_multiray"]).lower()
    if "3 of 15 rays" not in sparse_definition:
        raise ValueError("source summary does not document the 3-of-15 sparse rule")
    fovs = tuple(sorted(_floats(config.get("fov_degrees"), "config.fov_degrees")))
    if fovs != EXPECTED_FOVS:
        raise ValueError(f"source FOVs must be exactly {EXPECTED_FOVS}")
    z_modes = tuple(sorted(str(value) for value in _list(config.get("z_modes"), "config.z_modes")))
    expected_modes = (
        ("ground_anchored",)
        if frame_step == 1
        else tuple(sorted(("ground_anchored", "raw")))
    )
    if z_modes != expected_modes:
        raise ValueError(
            f"source z modes for frame_step={frame_step} must be {expected_modes}"
        )

    expected_frames = _expected_selected_frames(frame_step)
    if selected != len(expected_frames):
        raise ValueError(
            f"selected frame count is {selected}, expected {len(expected_frames)} "
            "from the declared missing/exclusion/step contract"
        )
    if _integer(summary.get("processed_frame_min"), "processed_frame_min") != expected_frames[0]:
        raise ValueError("source processed_frame_min disagrees with the frame contract")
    if _integer(summary.get("processed_frame_max"), "processed_frame_max") != expected_frames[-1]:
        raise ValueError("source processed_frame_max disagrees with the frame contract")
    expected_observed_count = (
        EXPECTED_FRAME_RANGE[1]
        - EXPECTED_FRAME_RANGE[0]
        + 1
        - len(EXPECTED_ALLOWED_MISSING_FRAMES)
    )
    if _integer(summary.get("input_observed_frame_count"), "input_observed_frame_count") != (
        expected_observed_count
    ):
        raise ValueError("source input_observed_frame_count disagrees with the frame contract")

    ego_count = _integer(summary.get("ego_positions"), "ego_positions")
    if ego_count != expected_ego_count:
        raise ValueError(f"source summary must declare exactly {expected_ego_count} ego positions")
    requested_scenes = _integer(summary.get("ego_scenes_requested"), "ego_scenes_requested")
    excluded_scenes = _integer(summary.get("ego_scenes_excluded"), "ego_scenes_excluded")
    evaluated_scenes = _integer(summary.get("ego_scenes_evaluated"), "ego_scenes_evaluated")
    if requested_scenes != selected * ego_count:
        raise ValueError("source ego_scenes_requested is inconsistent with frames × ego positions")
    if excluded_scenes < 0 or evaluated_scenes <= 0 or excluded_scenes + evaluated_scenes != requested_scenes:
        raise ValueError("source evaluated/excluded ego-scene partition is inconsistent")
    exclusions_summary = _mapping(summary.get("exclusions"), "source summary exclusions")
    if _integer(exclusions_summary.get("total"), "exclusions.total") != excluded_scenes:
        raise ValueError("source exclusion total disagrees with ego_scenes_excluded")

    expected_wide_rows = evaluated_scenes * len(z_modes) * len(fovs)
    wide_rows = _integer(summary.get("wide_count_rows"), "wide_count_rows")
    if wide_rows != expected_wide_rows:
        raise ValueError("source wide_count_rows is inconsistent with completed scene groups")
    outputs = _mapping(summary.get("outputs"), "source summary outputs")
    counts_output = _mapping(outputs.get("counts"), "outputs.counts")
    contribution_output = _mapping(outputs.get("contributions"), "outputs.contributions")
    exclusion_output = _mapping(outputs.get("exclusions"), "outputs.exclusions")
    timing_output = _mapping(outputs.get("timings"), "outputs.timings")
    if _integer(counts_output.get("rows"), "outputs.counts.rows") != wide_rows:
        raise ValueError("outputs.counts.rows disagrees with wide_count_rows")
    if _integer(contribution_output.get("rows"), "outputs.contributions.rows") != (
        wide_rows * len(EXPECTED_METHODS)
    ):
        raise ValueError("outputs.contributions.rows is not three rows per wide count row")
    if _integer(exclusion_output.get("rows"), "outputs.exclusions.rows") != excluded_scenes:
        raise ValueError("outputs.exclusions.rows disagrees with ego_scenes_excluded")
    if _integer(timing_output.get("rows"), "outputs.timings.rows") != selected:
        raise ValueError("outputs.timings.rows disagrees with selected frame count")

    groups = _list(summary.get("groups"), "source summary groups")
    expected_group_keys = {(mode, fov) for mode in z_modes for fov in fovs}
    actual_group_keys: set[tuple[str, float]] = set()
    for index, value in enumerate(groups):
        group = _mapping(value, f"groups[{index}]")
        key = (str(group.get("z_mode")), _finite(group.get("fov_deg"), "group.fov_deg"))
        if key in actual_group_keys:
            raise ValueError(f"source summary contains duplicate group {key}")
        actual_group_keys.add(key)
        if _integer(group.get("ego_frame_rows"), "group.ego_frame_rows") != evaluated_scenes:
            raise ValueError(f"source summary group {key} has incomplete ego-frame rows")
    if actual_group_keys != expected_group_keys:
        raise ValueError("source summary does not contain the complete z-mode × FOV product")
    return frame_step, expected_frames, z_modes, fovs


def _validate_schema_metadata(
    input_path: Path,
    summary: Mapping[str, Any],
) -> None:
    metadata = _metadata(pq.ParquetFile(input_path).schema_arrow.metadata)
    if metadata.get("paco.schema_version") != "1.0":
        raise ValueError("input Parquet paco.schema_version must be '1.0'")
    summary_input = _mapping(summary["input"], "source summary input")
    expected_pairs = {
        "paco.expected_frame_range": summary_input["expected_frame_range"],
        "paco.allowed_missing_frames": summary_input["allowed_missing_frames"],
        "paco.quality_excluded_frames": summary_input["quality_excluded_frames"]["requested"],
        "paco.only_frames_requested": summary_input["only_frames"]["requested"],
    }
    for key, expected in expected_pairs.items():
        if _metadata_json(metadata, key) != expected:
            raise ValueError(f"input Parquet metadata {key!r} disagrees with source summary")
    if metadata.get("paco.input_sha256") != summary_input["sha256"]:
        raise ValueError("input Parquet source-input SHA disagrees with source summary")
    if _integer(
        metadata.get("paco.sparse_min_visible_rays"),
        "Parquet paco.sparse_min_visible_rays",
    ) != EXPECTED_SPARSE_MIN_VISIBLE_RAYS:
        raise ValueError("input Parquet metadata must declare the 3-of-15 sparse threshold")
    if metadata.get("paco.frame_exclusion_reason") != EXPECTED_EXCLUSION_REASON:
        raise ValueError("input Parquet frame-exclusion reason disagrees with the contract")


def _validate_frame_groups(
    data: pd.DataFrame,
    *,
    table_kind: str,
    frame_column: str,
    frame_order_column: str,
    z_mode_column: str,
    fov_column: str,
    method_column: str,
    ego_column: str,
    expected_frames: tuple[int, ...],
    z_modes: tuple[str, ...],
    fovs: tuple[float, ...],
    expected_ego_count: int,
) -> None:
    group_columns = [z_mode_column, fov_column]
    required = {frame_column, frame_order_column, z_mode_column, fov_column, ego_column}
    if table_kind == "contributions":
        required.add(method_column)
        group_columns.append(method_column)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"canonical {table_kind} table is missing columns: {missing}")
    if data.loc[:, list(required)].isna().any().any():
        raise ValueError(f"canonical {table_kind} grouping columns contain missing values")

    frame_values = pd.to_numeric(data[frame_column], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(frame_values)) or not np.array_equal(frame_values, np.rint(frame_values)):
        raise ValueError("canonical frame labels must be finite integers")
    actual_frames = tuple(sorted(set(frame_values.astype(np.int64).tolist())))
    if actual_frames != expected_frames:
        missing_frames = sorted(set(expected_frames).difference(actual_frames))[:20]
        extra_frames = sorted(set(actual_frames).difference(expected_frames))[:20]
        raise ValueError(
            "input Parquet frame set disagrees with the source selection contract; "
            f"missing={missing_frames}, extra={extra_frames}"
        )

    frame_orders = pd.to_numeric(data[frame_order_column], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(frame_orders)) or not np.array_equal(
        frame_orders, np.rint(frame_orders)
    ):
        raise ValueError("canonical frame_order values must be finite integers")
    frame_order_table = pd.DataFrame(
        {"frame": frame_values.astype(np.int64), "order": frame_orders.astype(np.int64)}
    ).drop_duplicates()
    if len(frame_order_table) != len(expected_frames):
        raise ValueError("each selected frame must map to exactly one frame_order")
    observed_order = tuple(
        frame_order_table.sort_values("frame", kind="stable")["order"].tolist()
    )
    if observed_order != tuple(range(len(expected_frames))):
        raise ValueError("frame_order must be zero-based and follow the selected frame sequence")

    work = data.loc[:, group_columns + [frame_column]].copy()
    work[z_mode_column] = work[z_mode_column].astype(str)
    work[fov_column] = pd.to_numeric(work[fov_column], errors="raise").astype(float)
    expected_groups: set[tuple[Any, ...]]
    if table_kind == "contributions":
        work[method_column] = work[method_column].astype(str)
        expected_groups = {
            (mode, fov, method)
            for mode in z_modes
            for fov in fovs
            for method in EXPECTED_METHODS
        }
    else:
        expected_groups = {(mode, fov) for mode in z_modes for fov in fovs}
    actual_groups = set(
        work[group_columns].drop_duplicates().itertuples(index=False, name=None)
    )
    if actual_groups != expected_groups:
        raise ValueError(
            f"canonical {table_kind} groups do not match the source z-mode/FOV/method product"
        )
    presence = work.drop_duplicates(group_columns + [frame_column])
    group_frame_counts = presence.groupby(group_columns, sort=True, observed=True)[
        frame_column
    ].nunique()
    if not group_frame_counts.eq(len(expected_frames)).all():
        bad = group_frame_counts[group_frame_counts != len(expected_frames)]
        raise ValueError(f"canonical groups do not share the complete frame set: {bad.to_dict()}")

    ego_ids = tuple(sorted(data[ego_column].astype(str).unique().tolist()))
    if len(ego_ids) != expected_ego_count:
        raise ValueError(
            f"canonical {table_kind} table has {len(ego_ids)} ego IDs; "
            f"expected {expected_ego_count}"
        )
    if table_kind == "contributions":
        base = [frame_column, ego_column, z_mode_column, fov_column]
        method_counts = data.groupby(base, sort=False, observed=True)[method_column].agg(
            ["size", "nunique"]
        )
        if not (
            method_counts["size"].eq(len(EXPECTED_METHODS)).all()
            and method_counts["nunique"].eq(len(EXPECTED_METHODS)).all()
        ):
            raise ValueError("each contribution ego-scene group must contain exactly three methods")
    else:
        keys = [frame_column, ego_column, z_mode_column, fov_column]
        if data.duplicated(keys).any():
            raise ValueError("wide count rows must be unique by frame/ego/z-mode/FOV")


@dataclass(frozen=True, slots=True)
class CacieSourceContract:
    """Validated lineage receipt for one canonical full-record consumer."""

    input_parquet: str
    input_parquet_sha256: str
    input_parquet_rows: int
    source_summary: str
    source_summary_sha256: str
    source_input_sha256: str
    sparse_min_visible_rays: int
    table_kind: str
    frame_step: int
    nominal_source_frame_rate_hz: float
    effective_frame_rate_hz: float
    selected_frame_count: int
    selected_frame_first: int
    selected_frame_last: int
    selected_frame_delta_counts: dict[str, int]
    group_frame_sets_complete: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_parquet": self.input_parquet,
            "input_parquet_sha256": self.input_parquet_sha256,
            "input_parquet_rows": self.input_parquet_rows,
            "source_summary": self.source_summary,
            "source_summary_sha256": self.source_summary_sha256,
            "source_input_sha256": self.source_input_sha256,
            "sparse_min_visible_rays": self.sparse_min_visible_rays,
            "table_kind": self.table_kind,
            "frame_selection": {
                "semantics": (
                    "sorted observed frames, declared quality exclusions removed, then "
                    "zero-based [::frame_step] selection"
                ),
                "frame_step": self.frame_step,
                "nominal_source_frame_rate_hz": self.nominal_source_frame_rate_hz,
                "effective_frame_rate_hz": self.effective_frame_rate_hz,
                "selected_frame_count": self.selected_frame_count,
                "selected_frame_first": self.selected_frame_first,
                "selected_frame_last": self.selected_frame_last,
                "selected_frame_delta_counts": self.selected_frame_delta_counts,
                "exact_frame_set_validated": True,
                "group_frame_sets_complete": self.group_frame_sets_complete,
            },
        }


def validate_cacie_full_record_source(
    input_path: Path,
    source_summary_path: Path,
    data: pd.DataFrame,
    *,
    source_snapshot: Mapping[str, Mapping[str, Any]],
    table_kind: str,
    allowed_frame_steps: Sequence[int],
    expected_ego_count: int = EXPECTED_EGO_COUNT,
    frame_column: str = "frame_idx",
    frame_order_column: str = "frame_order",
    z_mode_column: str = "z_mode",
    fov_column: str = "fov_deg",
    method_column: str = "method",
    ego_column: str = "ego_id",
) -> CacieSourceContract:
    """Validate a canonical Parquet input against its completed source summary."""

    if table_kind not in {"counts", "contributions"}:
        raise ValueError("table_kind must be 'counts' or 'contributions'")
    if not input_path.is_file():
        raise ValueError(f"input Parquet does not exist: {input_path}")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    # The caller takes this snapshot before loading ``data``.  Checking it here
    # prevents a path replacement between the load and semantic validation from
    # being blessed with the replacement file's hash.
    require_cacie_full_record_source_unchanged(
        source_snapshot,
        input_path,
        source_summary_path,
    )
    summary = _load_summary(source_summary_path)
    frame_step, expected_frames, z_modes, fovs = _validate_summary(
        summary,
        allowed_frame_steps=allowed_frame_steps,
        expected_ego_count=expected_ego_count,
    )
    parquet_file = pq.ParquetFile(input_path)
    parquet_rows = int(parquet_file.metadata.num_rows)
    if parquet_rows != len(data):
        raise ValueError("loaded input row count disagrees with Parquet footer")
    output_record = _mapping(
        _mapping(summary["outputs"], "source summary outputs").get(table_kind),
        f"outputs.{table_kind}",
    )
    expected_rows = _integer(output_record.get("rows"), f"outputs.{table_kind}.rows")
    if parquet_rows != expected_rows:
        raise ValueError(
            f"input Parquet has {parquet_rows} rows; source summary declares {expected_rows}"
        )
    declared_name = Path(str(output_record.get("file", ""))).name
    if declared_name != input_path.name:
        raise ValueError(
            f"input file name {input_path.name!r} disagrees with source summary {declared_name!r}"
        )
    wide_rows = _integer(summary.get("wide_count_rows"), "wide_count_rows")
    if table_kind == "counts" and parquet_rows != wide_rows:
        raise ValueError("wide input row count disagrees with source wide_count_rows")
    if table_kind == "contributions" and parquet_rows != wide_rows * len(EXPECTED_METHODS):
        raise ValueError("contribution input must contain three rows per source wide row")

    _validate_schema_metadata(input_path, summary)
    _validate_frame_groups(
        data,
        table_kind=table_kind,
        frame_column=frame_column,
        frame_order_column=frame_order_column,
        z_mode_column=z_mode_column,
        fov_column=fov_column,
        method_column=method_column,
        ego_column=ego_column,
        expected_frames=expected_frames,
        z_modes=z_modes,
        fovs=fovs,
        expected_ego_count=expected_ego_count,
    )
    # Cover all summary, footer, metadata, and frame-group reads performed by
    # this validator.  Consumers repeat this check after writing temporary
    # outputs, immediately before publishing them.
    require_cacie_full_record_source_unchanged(
        source_snapshot,
        input_path,
        source_summary_path,
    )

    input_record = _mapping(source_snapshot["input_parquet"], "source snapshot input")
    summary_record = _mapping(source_snapshot["source_summary"], "source snapshot summary")

    return CacieSourceContract(
        input_parquet=str(input_path.resolve()),
        input_parquet_sha256=str(input_record["sha256"]),
        input_parquet_rows=parquet_rows,
        source_summary=str(source_summary_path.resolve()),
        source_summary_sha256=str(summary_record["sha256"]),
        source_input_sha256=str(summary["input"]["sha256"]),
        sparse_min_visible_rays=EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
        table_kind=table_kind,
        frame_step=frame_step,
        nominal_source_frame_rate_hz=NOMINAL_SOURCE_FRAME_RATE_HZ,
        effective_frame_rate_hz=NOMINAL_SOURCE_FRAME_RATE_HZ / frame_step,
        selected_frame_count=len(expected_frames),
        selected_frame_first=expected_frames[0],
        selected_frame_last=expected_frames[-1],
        selected_frame_delta_counts=_delta_counts(expected_frames),
    )


def attach_source_contract(output: dict[str, Any], contract: CacieSourceContract) -> None:
    """Attach explicit top-level hashes plus the complete validated receipt."""

    output["input_parquet_sha256"] = contract.input_parquet_sha256
    output["source_summary"] = contract.source_summary
    output["source_summary_sha256"] = contract.source_summary_sha256
    output["source_input_sha256"] = contract.source_input_sha256
    output["sparse_min_visible_rays"] = contract.sparse_min_visible_rays
    output["source_contract"] = contract.as_dict()


def require_effective_frame_rate(
    frame_rate_hz: float,
    contract: CacieSourceContract,
) -> None:
    """Reject a CLI rate that treats a decimated source as contiguous 10 Hz."""

    rate = _finite(frame_rate_hz, "frame_rate_hz")
    if not math.isclose(
        rate,
        contract.effective_frame_rate_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"frame_rate_hz={rate:g} disagrees with source frame_step={contract.frame_step}; "
            f"expected {contract.effective_frame_rate_hz:g} Hz"
        )


__all__ = [
    "CacieSourceContract",
    "EXPECTED_ALLOWED_MISSING_FRAMES",
    "EXPECTED_EGO_COUNT",
    "EXPECTED_EXCLUSION_REASON",
    "EXPECTED_FOVS",
    "EXPECTED_FRAME_RANGE",
    "EXPECTED_METHODS",
    "EXPECTED_OBSERVED_EXCLUSIONS",
    "EXPECTED_QUALITY_EXCLUDED_FRAMES",
    "EXPECTED_SPARSE_MIN_VISIBLE_RAYS",
    "NOMINAL_SOURCE_FRAME_RATE_HZ",
    "attach_source_contract",
    "require_cacie_full_record_source_unchanged",
    "require_effective_frame_rate",
    "snapshot_cacie_full_record_source",
    "validate_cacie_full_record_source",
]
