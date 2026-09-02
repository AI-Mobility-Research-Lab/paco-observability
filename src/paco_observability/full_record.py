"""Scalable full-record PACO observability computation.

This module reduces per-object visibility decisions to one count row per
``frame x ego x z-mode x FOV``.  The rows retain numerator and denominator
components needed by a ratio-of-sums estimator; individual ego rows must not be
treated as statistically independent observations.

Three methods are reported side by side:

``legacy_planar``
    The manuscript-era angular-wedge model at a fixed 0.25 degree resolution.
``center_top``
    The inexpensive center/top two-ray 3D approximation.
``sparse_multiray``
    A 15-ray approximation using the box center, six face centers, and eight
    vertices.  The canonical decision requires at least three clear rays, a
    threshold selected on a disjoint calibration subset because an any-ray
    rule was measurably optimistic.  It is included because two rays can
    under-resolve ground-anchored boxes.

No near-miss labels, ideal infrastructure coverage, or ``V2I = 1`` assumption
enters this computation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import islice
import math
import time
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from .geometry import GroundPlane, OrientedBox
from .legacy import planar_angular_wedge_visibility, theta_bins_for_resolution
from .metrics import ratio_of_sums
from .occlusion import center_coverage_flags, center_top_visibility, multi_ray_visibility
from .provenance import audit_tracks
from .validation import EgoPose, default_ego_grid


Record: TypeAlias = dict[str, Any]

METHOD_COLUMNS: Mapping[str, str] = {
    "legacy_planar": "legacy_visible_count",
    "center_top": "center_top_visible_count",
    "sparse_multiray": "sparse_multiray_visible_count",
}

TRACK_COLUMNS = (
    "frame_idx",
    "track_id",
    "class_name",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_center_z",
    "bbox_dx",
    "bbox_dy",
    "bbox_dz",
    "bbox_yaw",
)

COUNT_COLUMNS = (
    "frame_idx",
    "frame_order",
    "ego_index",
    "ego_id",
    "approach",
    "ego_x",
    "ego_y",
    "ego_heading_deg",
    "sensor_x",
    "sensor_y",
    "sensor_z",
    "z_mode",
    "fov_deg",
    "legacy_theta_bins",
    "total_count",
    "covered_count",
    "legacy_visible_count",
    "center_top_visible_count",
    "sparse_multiray_visible_count",
    "out_of_range_count",
    "out_of_fov_count",
    "both_out_of_range_and_fov_count",
    "legacy_occluded_count",
    "center_top_occluded_count",
    "sparse_multiray_occluded_count",
    "vru_count",
    "weighted_total_demand",
    "legacy_residual_demand",
    "center_top_residual_demand",
    "sparse_multiray_residual_demand",
    "vru_legacy_residual_count",
    "vru_center_top_residual_count",
    "vru_sparse_multiray_residual_count",
)

CONTRIBUTION_COLUMNS = (
    "frame_idx",
    "frame_order",
    "ego_index",
    "ego_id",
    "approach",
    "z_mode",
    "fov_deg",
    "method",
    "numerator",
    "denominator",
)

EXCLUSION_COLUMNS = (
    "frame_idx",
    "frame_order",
    "ego_index",
    "ego_id",
    "approach",
    "ego_x",
    "ego_y",
    "reason",
    "nearest_track_id",
    "nearest_distance_m",
    "clearance_m",
)

TIMING_COLUMNS = (
    "frame_idx",
    "frame_order",
    "objects",
    "ego_scenes_requested",
    "ego_scenes_evaluated",
    "ego_scenes_excluded",
    "count_rows",
    "box_build_ms",
    "legacy_planar_ms",
    "center_top_ms",
    "sparse_multiray_ms",
    "frame_total_ms",
)

_VRU_ALIASES = {"bike": "bicycle", "cyclist": "bicycle", "person": "pedestrian"}


def _normalize_class_name(value: Any) -> str:
    name = str(value).strip().lower()
    return _VRU_ALIASES.get(name, name)


@dataclass(frozen=True, slots=True)
class FullRecordConfig:
    """Geometry, exclusion, and demand parameters for the full-record run."""

    fov_degrees: tuple[float, ...] = (120.0, 360.0)
    z_modes: tuple[str, ...] = ("raw", "ground_anchored")
    range_m: float = 35.0
    near_blind_m: float = 1.5
    sensor_forward_offset_m: float = 1.5
    sensor_height_m: float = 1.8
    ego_clearance_m: float = 4.0
    angular_resolution_deg: float = 0.25
    occlusion_clearance_m: float = 0.0
    sparse_min_visible_rays: int = 3
    occluder_classes: frozenset[str] = frozenset({"car", "truck"})
    vru_classes: frozenset[str] = frozenset({"bicycle", "pedestrian"})
    non_vru_weight: float = 1.0
    vru_weight: float = 3.0

    def __post_init__(self) -> None:
        fovs = tuple(sorted(float(value) for value in self.fov_degrees))
        if not fovs or len(set(fovs)) != len(fovs):
            raise ValueError("fov_degrees must be nonempty and unique")
        if any(value not in {120.0, 360.0} for value in fovs):
            raise ValueError("full-record FOVs must be selected from 120 and 360 degrees")
        modes = tuple(sorted(str(value) for value in self.z_modes))
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("z_modes must be nonempty and unique")
        if any(value not in {"raw", "ground_anchored"} for value in modes):
            raise ValueError("z_modes must contain only raw or ground_anchored")
        if not math.isfinite(self.range_m) or self.range_m <= 0.0:
            raise ValueError("range_m must be positive and finite")
        if not 0.0 <= self.near_blind_m < self.range_m:
            raise ValueError("range limits must satisfy 0 <= near_blind_m < range_m")
        if not math.isfinite(self.sensor_forward_offset_m):
            raise ValueError("sensor_forward_offset_m must be finite")
        if not math.isfinite(self.sensor_height_m) or self.sensor_height_m <= 0.0:
            raise ValueError("sensor_height_m must be positive and finite")
        if not math.isfinite(self.ego_clearance_m) or self.ego_clearance_m < 0.0:
            raise ValueError("ego_clearance_m must be nonnegative and finite")
        if not math.isclose(self.angular_resolution_deg, 0.25, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("the full-record legacy comparison fixes resolution at 0.25 degrees")
        if not math.isfinite(self.occlusion_clearance_m) or self.occlusion_clearance_m < 0:
            raise ValueError("occlusion_clearance_m must be nonnegative and finite")
        if (
            isinstance(self.sparse_min_visible_rays, bool)
            or int(self.sparse_min_visible_rays) != self.sparse_min_visible_rays
            or not 1 <= self.sparse_min_visible_rays <= 15
        ):
            raise ValueError("sparse_min_visible_rays must be an integer in [1, 15]")
        if not math.isfinite(self.non_vru_weight) or self.non_vru_weight < 0.0:
            raise ValueError("non_vru_weight must be nonnegative and finite")
        if not math.isfinite(self.vru_weight) or self.vru_weight < 0.0:
            raise ValueError("vru_weight must be nonnegative and finite")
        if self.non_vru_weight == 0.0 and self.vru_weight == 0.0:
            raise ValueError("at least one demand weight must be positive")
        occluders = frozenset(_normalize_class_name(value) for value in self.occluder_classes)
        vrus = frozenset(_normalize_class_name(value) for value in self.vru_classes)
        if not occluders or "" in occluders or not vrus or "" in vrus:
            raise ValueError("occluder_classes and vru_classes must contain nonempty labels")
        object.__setattr__(self, "fov_degrees", fovs)
        object.__setattr__(self, "z_modes", modes)
        object.__setattr__(self, "occluder_classes", occluders)
        object.__setattr__(self, "vru_classes", vrus)

    def class_weight(self, class_name: str | None) -> float:
        """Return the configured demand weight for an object class."""

        normalized = _normalize_class_name(class_name)
        return self.vru_weight if normalized in self.vru_classes else self.non_vru_weight


def config_as_dict(config: FullRecordConfig) -> dict[str, Any]:
    """Return a JSON-safe representation of a full-record configuration."""

    values = asdict(config)
    values["occluder_classes"] = sorted(config.occluder_classes)
    values["vru_classes"] = sorted(config.vru_classes)
    values["method_definitions"] = {
        "legacy_planar": "0.25-degree angular-wedge center classification",
        "center_top": "center and top-center rays; visible if either is clear",
        "sparse_multiray": (
            "center, six face centers, and eight vertices; visible when at least "
            f"{config.sparse_min_visible_rays} of 15 rays are clear"
        ),
    }
    values["residual_demand_policy"] = (
        "sum configured class weights over objects not visible by the named method"
    )
    return values


def audit_full_record_tracks(
    tracks: pd.DataFrame,
    *,
    ground_plane: GroundPlane,
    expected_frame_range: tuple[int, int] | None = None,
    allowed_missing_frames: Iterable[int] = (),
) -> dict[str, Any]:
    """Run the shared input audit plus stricter full-record identity checks.

    Missing frames are fatal unless explicitly listed in
    ``allowed_missing_frames``.  Passing ``expected_frame_range`` also detects
    missing leading or trailing frames that cannot be inferred from the track
    table alone.
    """

    report = audit_tracks(
        tracks,
        ground_normal=tuple(float(value) for value in ground_plane.normal),
        ground_offset=float(ground_plane.d),
        expected_frame_range=expected_frame_range,
        allowed_missing_frames=allowed_missing_frames,
    )
    frame_numeric = pd.to_numeric(tracks["frame_idx"], errors="coerce").to_numpy(float)
    track_numeric = pd.to_numeric(tracks["track_id"], errors="coerce").to_numpy(float)
    noninteger_frames = int(
        np.sum(np.isfinite(frame_numeric) & (frame_numeric != np.floor(frame_numeric)))
    )
    noninteger_tracks = int(
        np.sum(np.isfinite(track_numeric) & (track_numeric != np.floor(track_numeric)))
    )
    empty_class_names = int(
        tracks["class_name"].fillna("").astype(str).str.strip().eq("").sum()
    )
    additional = {
        "noninteger_frame_rows": noninteger_frames,
        "noninteger_track_id_rows": noninteger_tracks,
        "empty_class_name_rows": empty_class_names,
    }
    shared_passed = bool(report["quality_gate"]["passed"])
    report["full_record_quality_gate"] = {
        "passed": shared_passed and not any(additional.values()),
        "shared_audit_passed": shared_passed,
        "additional_fatal_counts": additional,
        "policy": "fail closed before any paper-facing full-record computation",
    }
    return report


def require_passing_audit(audit: Mapping[str, Any]) -> None:
    """Raise when a full-record input audit did not pass."""

    gate = audit.get("full_record_quality_gate")
    if not isinstance(gate, Mapping) or not bool(gate.get("passed", False)):
        raise ValueError(f"input failed full-record quality gate: {gate!r}")


def select_frame_ids(
    tracks: pd.DataFrame,
    *,
    frame_step: int = 1,
    max_frames: int | None = None,
    exclude_frames: Iterable[int] = (),
    only_frames: Iterable[int] | None = None,
) -> np.ndarray:
    """Select sorted observed frame IDs for a full run or deterministic smoke.

    If ``only_frames`` is supplied, its observed intersection is selected
    first. Explicit frame exclusions are applied next, ``frame_step`` is then
    applied to the remaining sorted unique frame IDs, followed by the optional
    ``max_frames`` prefix. IDs absent from the track table are harmless and
    should still be retained in run metadata by the caller. No randomness is
    involved.
    """

    if isinstance(frame_step, bool) or int(frame_step) != frame_step or frame_step < 1:
        raise ValueError("frame_step must be a positive integer")
    if max_frames is not None and (
        isinstance(max_frames, bool) or int(max_frames) != max_frames or max_frames < 1
    ):
        raise ValueError("max_frames must be a positive integer when supplied")
    if "frame_idx" not in tracks or tracks.empty:
        raise ValueError("tracks must be nonempty and contain frame_idx")
    frames = np.sort(tracks["frame_idx"].astype(np.int64).unique())

    if only_frames is not None:
        requested: set[int] = set()
        for value in only_frames:
            if isinstance(value, bool) or int(value) != value:
                raise ValueError("only_frames must contain only integer frame IDs")
            requested.add(int(value))
        if not requested:
            raise ValueError("only_frames must be nonempty when supplied")
        frames = frames[np.isin(frames, np.asarray(sorted(requested), dtype=np.int64))]

    excluded: set[int] = set()
    for value in exclude_frames:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError("exclude_frames must contain only integer frame IDs")
        excluded.add(int(value))
    if excluded:
        frames = frames[~np.isin(frames, np.asarray(sorted(excluded), dtype=np.int64))]
    selected = frames[:: int(frame_step)]
    if max_frames is not None:
        selected = selected[: int(max_frames)]
    if not len(selected):
        raise ValueError("frame selection is empty")
    return selected.astype(np.int64, copy=False)


def _sensor_origin(
    pose: EgoPose,
    ground_plane: GroundPlane,
    config: FullRecordConfig,
) -> np.ndarray:
    x = pose.x + config.sensor_forward_offset_m * math.cos(pose.heading_rad)
    y = pose.y + config.sensor_forward_offset_m * math.sin(pose.heading_rad)
    z = float(ground_plane.height_at(x, y)) + config.sensor_height_m
    return np.asarray((x, y, z), dtype=np.float64)


def _nearest_observed_vehicle(
    pose: EgoPose,
    boxes: Sequence[OrientedBox],
    config: FullRecordConfig,
) -> tuple[int | str | None, float | None]:
    candidates: list[tuple[float, str, int | str | None]] = []
    for box in boxes:
        if _normalize_class_name(box.class_name) not in config.occluder_classes:
            continue
        distance = math.hypot(box.center[0] - pose.x, box.center[1] - pose.y)
        candidates.append((distance, str(box.object_id), box.object_id))
    if not candidates:
        return None, None
    distance, _, object_id = min(candidates)
    return object_id, float(distance)


def sparse_target_points(target: OrientedBox) -> dict[str, np.ndarray]:
    """Return center, six face centers, and eight vertices for a 15-ray test."""

    hx, hy, hz = target.half_size
    local_faces = (
        ("face_x_positive", (hx, 0.0, 0.0)),
        ("face_x_negative", (-hx, 0.0, 0.0)),
        ("face_y_positive", (0.0, hy, 0.0)),
        ("face_y_negative", (0.0, -hy, 0.0)),
        ("face_z_positive", (0.0, 0.0, hz)),
        ("face_z_negative", (0.0, 0.0, -hz)),
    )
    points: dict[str, np.ndarray] = {"center": target.center_array}
    for label, local in local_faces:
        points[label] = target.local_to_world(np.asarray(local, dtype=np.float64))
    for index, corner in enumerate(target.corners()):
        points[f"vertex_{index:02d}"] = np.asarray(corner, dtype=np.float64)
    return points


@dataclass(frozen=True, slots=True)
class FrameAggregateResult:
    """Stable, serializable output from one frame worker."""

    frame_idx: int
    frame_order: int
    count_rows: tuple[Record, ...]
    exclusion_rows: tuple[Record, ...]
    timing_rows: tuple[Record, ...]


def _empty_statistics(total_count: int, total_weight: float, vru_count: int) -> Record:
    return {
        "total_count": int(total_count),
        "covered_count": 0,
        "legacy_visible_count": 0,
        "center_top_visible_count": 0,
        "sparse_multiray_visible_count": 0,
        "out_of_range_count": 0,
        "out_of_fov_count": 0,
        "both_out_of_range_and_fov_count": 0,
        "legacy_occluded_count": 0,
        "center_top_occluded_count": 0,
        "sparse_multiray_occluded_count": 0,
        "vru_count": int(vru_count),
        "weighted_total_demand": float(total_weight),
        "legacy_residual_demand": 0.0,
        "center_top_residual_demand": 0.0,
        "sparse_multiray_residual_demand": 0.0,
        "vru_legacy_residual_count": 0,
        "vru_center_top_residual_count": 0,
        "vru_sparse_multiray_residual_count": 0,
    }


def _validate_count_row(row: Mapping[str, Any]) -> None:
    total = int(row["total_count"])
    covered = int(row["covered_count"])
    if total <= 0 or not 0 <= covered <= total:
        raise RuntimeError(f"invalid total/covered counts: {row!r}")
    for method, visible_column in METHOD_COLUMNS.items():
        visible = int(row[visible_column])
        occluded = int(row[f"{method.removesuffix('_planar')}_occluded_count"])
        if not 0 <= visible <= covered or visible + occluded != covered:
            raise RuntimeError(f"invalid {method} visibility partition: {row!r}")
    for column in (
        "out_of_range_count",
        "out_of_fov_count",
        "both_out_of_range_and_fov_count",
    ):
        if not 0 <= int(row[column]) <= total:
            raise RuntimeError(f"invalid nonexclusive failure count {column}: {row!r}")
    if int(row["both_out_of_range_and_fov_count"]) > min(
        int(row["out_of_range_count"]), int(row["out_of_fov_count"])
    ):
        raise RuntimeError(f"invalid joint failure count: {row!r}")
    total_weight = float(row["weighted_total_demand"])
    for method in METHOD_COLUMNS:
        residual = float(row[f"{method.removesuffix('_planar')}_residual_demand"])
        if residual < -1e-12 or residual > total_weight + 1e-12:
            raise RuntimeError(f"invalid {method} residual demand: {row!r}")


def _evaluate_frame_records(
    frame_order: int,
    frame_idx: int,
    records: Sequence[Record],
    *,
    ego_poses: Sequence[EgoPose],
    ground_plane: GroundPlane,
    config: FullRecordConfig,
) -> FrameAggregateResult:
    frame_started = time.perf_counter()
    build_started = time.perf_counter()
    normalized_records: list[Record] = []
    for source in records:
        row = dict(source)
        row["frame_idx"] = int(row["frame_idx"])
        row["track_id"] = int(row["track_id"])
        row["class_name"] = _normalize_class_name(row["class_name"])
        normalized_records.append(row)
    normalized_records.sort(key=lambda row: int(row["track_id"]))
    if not normalized_records:
        raise RuntimeError(f"selected frame {frame_idx} has no objects")
    if any(int(row["frame_idx"]) != frame_idx for row in normalized_records):
        raise RuntimeError("frame worker received records from multiple frames")

    raw_boxes = [OrientedBox.from_row(row, z_mode="raw") for row in normalized_records]
    boxes_by_mode: dict[str, list[OrientedBox]] = {}
    for z_mode in config.z_modes:
        if z_mode == "raw":
            boxes_by_mode[z_mode] = raw_boxes
        else:
            boxes_by_mode[z_mode] = [
                OrientedBox.from_row(
                    row,
                    z_mode="ground_anchored",
                    ground_plane=ground_plane,
                )
                for row in normalized_records
            ]
    box_build_ms = (time.perf_counter() - build_started) * 1000.0

    count_rows: list[Record] = []
    exclusion_rows: list[Record] = []
    legacy_seconds = 0.0
    center_top_seconds = 0.0
    sparse_seconds = 0.0

    for ego_index, pose in enumerate(ego_poses):
        nearest_id, nearest_distance = _nearest_observed_vehicle(pose, raw_boxes, config)
        if nearest_distance is not None and nearest_distance < config.ego_clearance_m:
            exclusion_rows.append(
                {
                    "frame_idx": frame_idx,
                    "frame_order": frame_order,
                    "ego_index": ego_index,
                    "ego_id": pose.ego_id,
                    "approach": pose.approach,
                    "ego_x": float(pose.x),
                    "ego_y": float(pose.y),
                    "reason": "hypothetical_ego_within_clearance_of_observed_vehicle",
                    "nearest_track_id": nearest_id,
                    "nearest_distance_m": nearest_distance,
                    "clearance_m": float(config.ego_clearance_m),
                }
            )
            continue

        origin = _sensor_origin(pose, ground_plane, config)
        legacy_by_fov: dict[float, Mapping[Any, Any]] = {}
        for fov_deg in config.fov_degrees:
            started = time.perf_counter()
            legacy_by_fov[fov_deg] = planar_angular_wedge_visibility(
                sensor_xy=(float(origin[0]), float(origin[1])),
                heading_rad=pose.heading_rad,
                boxes=raw_boxes,
                fov_deg=fov_deg,
                range_m=config.range_m,
                near_blind_m=config.near_blind_m,
                angular_resolution_deg=config.angular_resolution_deg,
                clearance_m=config.occlusion_clearance_m,
                occluder_classes=config.occluder_classes,
            )
            legacy_seconds += time.perf_counter() - started

        for z_mode in config.z_modes:
            boxes = boxes_by_mode[z_mode]
            blockers = [
                box
                for box in boxes
                if _normalize_class_name(box.class_name) in config.occluder_classes
            ]
            weights = [config.class_weight(box.class_name) for box in boxes]
            total_weight = float(math.fsum(weights))
            vru_count = sum(
                _normalize_class_name(box.class_name) in config.vru_classes for box in boxes
            )
            statistics = {
                fov: _empty_statistics(len(boxes), total_weight, vru_count)
                for fov in config.fov_degrees
            }

            for target, weight in zip(boxes, weights, strict=True):
                flags_by_fov = {
                    fov: center_coverage_flags(
                        origin,
                        target,
                        heading_rad=pose.heading_rad,
                        fov_rad=math.radians(fov),
                        near_range_m=config.near_blind_m,
                        far_range_m=config.range_m,
                    )
                    for fov in config.fov_degrees
                }
                center_top = None
                sparse = None
                if any(flags.covered for flags in flags_by_fov.values()):
                    started = time.perf_counter()
                    center_top = center_top_visibility(
                        origin,
                        target,
                        blockers,
                        criterion="any",
                        clearance_m=config.occlusion_clearance_m,
                    )
                    center_top_seconds += time.perf_counter() - started

                    started = time.perf_counter()
                    sparse = multi_ray_visibility(
                        origin,
                        target,
                        blockers,
                        target_points=sparse_target_points(target),
                        criterion="any",
                        clearance_m=config.occlusion_clearance_m,
                    )
                    sparse_seconds += time.perf_counter() - started

                is_vru = _normalize_class_name(target.class_name) in config.vru_classes
                for fov_deg, flags in flags_by_fov.items():
                    entry = statistics[fov_deg]
                    if flags.covered:
                        entry["covered_count"] += 1
                    if not flags.in_range:
                        entry["out_of_range_count"] += 1
                    if not flags.in_fov:
                        entry["out_of_fov_count"] += 1
                    if not flags.in_range and not flags.in_fov:
                        entry["both_out_of_range_and_fov_count"] += 1

                    legacy = legacy_by_fov[fov_deg][target.object_id]
                    legacy_visible = bool(legacy.visible)
                    center_top_visible = bool(
                        flags.covered and center_top is not None and center_top.visible
                    )
                    sparse_visible = bool(
                        flags.covered
                        and sparse is not None
                        and sparse.visible_rays >= config.sparse_min_visible_rays
                    )
                    if legacy_visible:
                        entry["legacy_visible_count"] += 1
                    else:
                        entry["legacy_residual_demand"] += weight
                        if is_vru:
                            entry["vru_legacy_residual_count"] += 1
                    if center_top_visible:
                        entry["center_top_visible_count"] += 1
                    else:
                        entry["center_top_residual_demand"] += weight
                        if is_vru:
                            entry["vru_center_top_residual_count"] += 1
                    if sparse_visible:
                        entry["sparse_multiray_visible_count"] += 1
                    else:
                        entry["sparse_multiray_residual_demand"] += weight
                        if is_vru:
                            entry["vru_sparse_multiray_residual_count"] += 1

                    if flags.covered and not legacy_visible:
                        entry["legacy_occluded_count"] += 1
                    if flags.covered and not center_top_visible:
                        entry["center_top_occluded_count"] += 1
                    if flags.covered and not sparse_visible:
                        entry["sparse_multiray_occluded_count"] += 1

            for fov_deg in config.fov_degrees:
                row: Record = {
                    "frame_idx": frame_idx,
                    "frame_order": frame_order,
                    "ego_index": ego_index,
                    "ego_id": pose.ego_id,
                    "approach": pose.approach,
                    "ego_x": float(pose.x),
                    "ego_y": float(pose.y),
                    "ego_heading_deg": float(math.degrees(pose.heading_rad)),
                    "sensor_x": float(origin[0]),
                    "sensor_y": float(origin[1]),
                    "sensor_z": float(origin[2]),
                    "z_mode": z_mode,
                    "fov_deg": float(fov_deg),
                    "legacy_theta_bins": theta_bins_for_resolution(
                        fov_deg, config.angular_resolution_deg
                    ),
                    **statistics[fov_deg],
                }
                _validate_count_row(row)
                count_rows.append(row)

    timing_rows = (
        {
            "frame_idx": frame_idx,
            "frame_order": frame_order,
            "objects": len(raw_boxes),
            "ego_scenes_requested": len(ego_poses),
            "ego_scenes_evaluated": len(ego_poses) - len(exclusion_rows),
            "ego_scenes_excluded": len(exclusion_rows),
            "count_rows": len(count_rows),
            "box_build_ms": float(box_build_ms),
            "legacy_planar_ms": float(legacy_seconds * 1000.0),
            "center_top_ms": float(center_top_seconds * 1000.0),
            "sparse_multiray_ms": float(sparse_seconds * 1000.0),
            "frame_total_ms": float((time.perf_counter() - frame_started) * 1000.0),
        },
    )
    expected_rows = (len(ego_poses) - len(exclusion_rows)) * len(config.z_modes) * len(
        config.fov_degrees
    )
    if len(count_rows) != expected_rows:
        raise RuntimeError(
            f"frame {frame_idx} produced {len(count_rows)} rows; expected {expected_rows}"
        )
    return FrameAggregateResult(
        frame_idx=frame_idx,
        frame_order=frame_order,
        count_rows=tuple(count_rows),
        exclusion_rows=tuple(exclusion_rows),
        timing_rows=timing_rows,
    )


def evaluate_frame_counts(
    frame_tracks: pd.DataFrame,
    *,
    ego_poses: Sequence[EgoPose],
    ground_plane: GroundPlane,
    config: FullRecordConfig,
    frame_order: int = 0,
) -> FrameAggregateResult:
    """Evaluate one nonempty frame into stable ego-frame count rows."""

    if frame_tracks.empty or "frame_idx" not in frame_tracks:
        raise ValueError("frame_tracks must be nonempty and contain frame_idx")
    frame_values = frame_tracks["frame_idx"].astype(np.int64).unique()
    if len(frame_values) != 1:
        raise ValueError("evaluate_frame_counts expects exactly one frame")
    return _evaluate_frame_records(
        int(frame_order),
        int(frame_values[0]),
        frame_tracks.to_dict("records"),
        ego_poses=ego_poses,
        ground_plane=ground_plane,
        config=config,
    )


_WORKER_EGOS: tuple[EgoPose, ...] | None = None
_WORKER_PLANE: GroundPlane | None = None
_WORKER_CONFIG: FullRecordConfig | None = None


def _initialize_worker(
    ego_poses: tuple[EgoPose, ...],
    ground_plane: GroundPlane,
    config: FullRecordConfig,
) -> None:
    global _WORKER_EGOS, _WORKER_PLANE, _WORKER_CONFIG
    _WORKER_EGOS = ego_poses
    _WORKER_PLANE = ground_plane
    _WORKER_CONFIG = config


def _worker_evaluate(payload: tuple[int, int, tuple[Record, ...]]) -> FrameAggregateResult:
    if _WORKER_EGOS is None or _WORKER_PLANE is None or _WORKER_CONFIG is None:
        raise RuntimeError("full-record process worker was not initialized")
    frame_order, frame_idx, records = payload
    return _evaluate_frame_records(
        frame_order,
        frame_idx,
        records,
        ego_poses=_WORKER_EGOS,
        ground_plane=_WORKER_PLANE,
        config=_WORKER_CONFIG,
    )


def _validate_ego_poses(ego_poses: Sequence[EgoPose]) -> tuple[EgoPose, ...]:
    poses = tuple(ego_poses)
    if not poses:
        raise ValueError("at least one ego pose is required")
    if len({pose.ego_id for pose in poses}) != len(poses):
        raise ValueError("ego pose IDs must be unique")
    for pose in poses:
        values = (pose.x, pose.y, pose.heading_rad)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"ego pose contains nonfinite values: {pose!r}")
    return poses


def _frame_payloads(
    tracks: pd.DataFrame,
    frame_ids: Sequence[int],
) -> Iterator[tuple[int, int, tuple[Record, ...]]]:
    wanted = {int(value): index for index, value in enumerate(frame_ids)}
    subset = tracks.loc[
        tracks["frame_idx"].astype(np.int64).isin(wanted), TRACK_COLUMNS
    ]
    yielded: set[int] = set()
    for frame_idx, group in subset.groupby("frame_idx", sort=True):
        frame = int(frame_idx)
        if frame not in wanted:
            continue
        yielded.add(frame)
        yield wanted[frame], frame, tuple(group.to_dict("records"))
    missing = sorted(set(wanted).difference(yielded))
    if missing:
        raise ValueError(f"selected frame IDs are absent from tracks: {missing[:20]}")


def iter_full_record(
    tracks: pd.DataFrame,
    *,
    ground_plane: GroundPlane,
    config: FullRecordConfig,
    frame_ids: Sequence[int] | None = None,
    ego_poses: Sequence[EgoPose] | None = None,
    n_processes: int = 1,
    chunksize: int = 8,
    input_audit: Mapping[str, Any] | None = None,
) -> Iterator[FrameAggregateResult]:
    """Yield full-record frame results in deterministic frame order.

    Multiprocessing uses bounded batches so the parent does not queue the whole
    40k-frame record or retain all ego rows in memory.  ``executor.map`` keeps
    output order identical across worker counts.
    """

    if isinstance(n_processes, bool) or int(n_processes) != n_processes or n_processes < 1:
        raise ValueError("n_processes must be a positive integer")
    if isinstance(chunksize, bool) or int(chunksize) != chunksize or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    audit = (
        dict(input_audit)
        if input_audit is not None
        else audit_full_record_tracks(tracks, ground_plane=ground_plane)
    )
    require_passing_audit(audit)
    poses = _validate_ego_poses(ego_poses or default_ego_grid())
    if frame_ids is None:
        selected = select_frame_ids(tracks)
    else:
        selected = np.asarray(frame_ids, dtype=np.int64)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("frame_ids must be a nonempty one-dimensional sequence")
        if np.any(selected[1:] <= selected[:-1]):
            raise ValueError("frame_ids must be strictly increasing and unique")

    payloads = _frame_payloads(tracks, selected)
    if int(n_processes) == 1:
        for payload in payloads:
            yield _evaluate_frame_records(
                payload[0],
                payload[1],
                payload[2],
                ego_poses=poses,
                ground_plane=ground_plane,
                config=config,
            )
        return

    batch_size = max(int(n_processes) * int(chunksize) * 4, int(chunksize))
    with ProcessPoolExecutor(
        max_workers=int(n_processes),
        initializer=_initialize_worker,
        initargs=(poses, ground_plane, config),
    ) as executor:
        while True:
            batch = list(islice(payloads, batch_size))
            if not batch:
                break
            yield from executor.map(_worker_evaluate, batch, chunksize=int(chunksize))


def contribution_rows(count_rows: Iterable[Mapping[str, Any]]) -> Iterator[Record]:
    """Convert wide ego-frame count rows to bootstrap-ready long form."""

    for row in count_rows:
        denominator = int(row["total_count"])
        if denominator <= 0:
            raise ValueError("contribution denominator must be positive")
        common = {
            "frame_idx": int(row["frame_idx"]),
            "frame_order": int(row["frame_order"]),
            "ego_index": int(row["ego_index"]),
            "ego_id": str(row["ego_id"]),
            "approach": str(row["approach"]),
            "z_mode": str(row["z_mode"]),
            "fov_deg": float(row["fov_deg"]),
        }
        for method, numerator_column in METHOD_COLUMNS.items():
            numerator = int(row[numerator_column])
            if not 0 <= numerator <= denominator:
                raise ValueError(f"invalid contribution for {method}: {numerator}/{denominator}")
            yield {
                **common,
                "method": method,
                "numerator": numerator,
                "denominator": denominator,
            }


class FullRecordSummaryAccumulator:
    """Streaming ratio-of-sums, exclusion, and runtime aggregation."""

    _COUNT_SUM_COLUMNS = tuple(
        column
        for column in COUNT_COLUMNS
        if column.endswith("_count")
        or column.endswith("_demand")
        or column == "weighted_total_demand"
    )

    def __init__(self) -> None:
        self._groups: dict[tuple[str, float], defaultdict[str, float]] = {}
        self._approaches: dict[tuple[str, float, str], defaultdict[str, float]] = {}
        self._exclusions: Counter[str] = Counter()
        self._timings: list[float] = []
        self._timing_sums: Counter[str] = Counter()
        self._frames: list[int] = []
        self._count_rows = 0

    @property
    def processed_frame_count(self) -> int:
        """Number of frame results incorporated so far."""

        return len(self._frames)

    @staticmethod
    def _add_row(target: defaultdict[str, float], row: Mapping[str, Any]) -> None:
        for column in FullRecordSummaryAccumulator._COUNT_SUM_COLUMNS:
            target[column] += float(row[column])
        target["ego_frame_rows"] += 1.0

    def update(self, result: FrameAggregateResult) -> None:
        """Add one frame result, checking monotonically increasing order."""

        if self._frames and result.frame_idx <= self._frames[-1]:
            raise ValueError("frame results must arrive in strictly increasing order")
        self._frames.append(int(result.frame_idx))
        self._count_rows += len(result.count_rows)
        for row in result.count_rows:
            group_key = (str(row["z_mode"]), float(row["fov_deg"]))
            approach_key = (*group_key, str(row["approach"]))
            group = self._groups.setdefault(group_key, defaultdict(float))
            approach = self._approaches.setdefault(approach_key, defaultdict(float))
            self._add_row(group, row)
            self._add_row(approach, row)
        for row in result.exclusion_rows:
            self._exclusions[str(row["approach"])] += 1
        for row in result.timing_rows:
            self._timings.append(float(row["frame_total_ms"]))
            for column in (
                "box_build_ms",
                "legacy_planar_ms",
                "center_top_ms",
                "sparse_multiray_ms",
                "frame_total_ms",
            ):
                self._timing_sums[column] += float(row[column])

    @staticmethod
    def _group_summary(
        key: tuple[Any, ...],
        values: Mapping[str, float],
        *,
        include_approach: bool,
    ) -> Record:
        total = int(values["total_count"])
        if total <= 0:
            raise ValueError(f"summary group has zero denominator: {key!r}")
        weighted_total = float(values["weighted_total_demand"])
        entry: Record = {
            "z_mode": str(key[0]),
            "fov_deg": float(key[1]),
            "ego_frame_rows": int(values["ego_frame_rows"]),
            "counts": {
                column: int(values[column])
                for column in FullRecordSummaryAccumulator._COUNT_SUM_COLUMNS
                if column.endswith("_count")
            },
            "weighted_demand": {
                column: float(values[column])
                for column in FullRecordSummaryAccumulator._COUNT_SUM_COLUMNS
                if column.endswith("_demand") or column == "weighted_total_demand"
            },
            "observability_ratio_of_sums": {
                method: ratio_of_sums([values[column]], [total])
                for method, column in METHOD_COLUMNS.items()
            },
            "coverage_ratio_of_sums": ratio_of_sums([values["covered_count"]], [total]),
            "residual_demand_fraction": {
                method: (
                    float(values[f"{method.removesuffix('_planar')}_residual_demand"])
                    / weighted_total
                    if weighted_total > 0.0
                    else None
                )
                for method in METHOD_COLUMNS
            },
        }
        if include_approach:
            entry["approach"] = str(key[2])
        return entry

    def finalize(
        self,
        *,
        config: FullRecordConfig,
        input_frame_count: int,
        selected_frame_count: int,
        frame_step: int,
        max_frames: int | None,
        ego_positions: int,
        wall_elapsed_s: float,
    ) -> Record:
        """Return a JSON-safe summary after all frames have been streamed."""

        if not self._frames or not self._groups:
            raise ValueError("cannot summarize an empty full-record run")
        timing_values = np.asarray(self._timings, dtype=np.float64)
        excluded = int(sum(self._exclusions.values()))
        requested_ego_scenes = len(self._frames) * int(ego_positions)
        return {
            "schema_version": "1.0",
            "estimand": "ratio of summed visible counts to summed target counts",
            "processed_frame_count": len(self._frames),
            "processed_frame_min": self._frames[0],
            "processed_frame_max": self._frames[-1],
            "input_observed_frame_count": int(input_frame_count),
            "selected_frame_count": int(selected_frame_count),
            "frame_step": int(frame_step),
            "max_frames": None if max_frames is None else int(max_frames),
            "ego_positions": int(ego_positions),
            "ego_scenes_requested": requested_ego_scenes,
            "ego_scenes_excluded": excluded,
            "ego_scenes_evaluated": requested_ego_scenes - excluded,
            "wide_count_rows": self._count_rows,
            "groups": [
                self._group_summary(key, values, include_approach=False)
                for key, values in sorted(self._groups.items())
            ],
            "by_approach": [
                self._group_summary(key, values, include_approach=True)
                for key, values in sorted(self._approaches.items())
            ],
            "exclusions": {
                "total": excluded,
                "fraction_of_requested_ego_scenes": excluded / requested_ego_scenes,
                "by_approach": dict(sorted(self._exclusions.items())),
                "rule": (
                    "exclude hypothetical ego-scene when ego center is strictly within "
                    f"{config.ego_clearance_m:g} m of an observed car/truck center"
                ),
            },
            "runtime": {
                "wall_elapsed_s": float(wall_elapsed_s),
                "frame_worker_total_s": float(self._timing_sums["frame_total_ms"] / 1000.0),
                "mean_frame_ms": float(np.mean(timing_values)),
                "median_frame_ms": float(np.median(timing_values)),
                "p95_frame_ms": float(np.quantile(timing_values, 0.95)),
                "component_sums_s": {
                    column.removesuffix("_ms"): float(value / 1000.0)
                    for column, value in sorted(self._timing_sums.items())
                    if column != "frame_total_ms"
                },
            },
            "config": config_as_dict(config),
            "scope": {
                "near_miss_labels_used": False,
                "ideal_v2i_assumption_used": False,
                "static_scene_mesh_used": False,
                "visibility_geometry": "supplied upright dynamic OBBs only",
            },
        }
