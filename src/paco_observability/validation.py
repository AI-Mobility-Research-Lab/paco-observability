"""Deterministic scene sampling and model-reference validation.

The high-resolution ray model is a geometric reference *within the same OBB
scene abstraction*.  It is intentionally not called physical ground truth.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .geometry import GroundPlane, OrientedBox
from .legacy import planar_angular_wedge_visibility
from .metrics import binary_agreement_metrics, ratio_of_sums
from .occlusion import (
    center_coverage_flags,
    center_top_visibility,
    multi_ray_visibility,
    projected_visible_fraction,
)


@dataclass(frozen=True, slots=True)
class EgoPose:
    """One hypothetical ego-vehicle center and heading."""

    ego_id: str
    approach: str
    x: float
    y: float
    heading_rad: float


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Frozen geometry and validation parameters."""

    fov_degrees: tuple[float, ...] = (120.0, 360.0)
    range_m: float = 35.0
    near_blind_m: float = 1.5
    sensor_forward_offset_m: float = 1.5
    sensor_height_m: float = 1.8
    ego_clearance_m: float = 4.0
    angular_resolution_deg: float = 0.25
    occlusion_clearance_m: float = 0.0
    visible_fraction_threshold: float = 0.05
    ray_grids: tuple[int, ...] = (17,)
    z_modes: tuple[str, ...] = ("raw", "ground_anchored")
    occluder_classes: frozenset[str] = frozenset({"car", "truck"})

    def __post_init__(self) -> None:
        if not self.fov_degrees or any(not 0 < value <= 360 for value in self.fov_degrees):
            raise ValueError("fov_degrees must contain values in (0, 360]")
        if not 0 <= self.near_blind_m < self.range_m:
            raise ValueError("range limits must satisfy 0 <= near_blind_m < range_m")
        if self.sensor_height_m <= 0 or self.ego_clearance_m < 0:
            raise ValueError("sensor height must be positive and clearance nonnegative")
        if not 0 <= self.visible_fraction_threshold <= 1:
            raise ValueError("visible_fraction_threshold must be in [0, 1]")
        if not self.ray_grids or any(int(value) != value or value < 1 for value in self.ray_grids):
            raise ValueError("ray_grids must contain positive integers")
        if any(value not in {"raw", "ground_anchored"} for value in self.z_modes):
            raise ValueError("z_modes must contain only raw or ground_anchored")

    @property
    def primary_grid(self) -> int:
        return max(self.ray_grids)


def default_ego_grid(n_per_approach: int = 15) -> list[EgoPose]:
    """Return the four-approach, endpoint-inclusive ego grid used by PACO."""

    if n_per_approach < 1:
        raise ValueError("n_per_approach must be positive")
    definitions = (
        ("north", 7.0, 35.0, 7.0, 8.0, -math.pi / 2),
        ("south", 8.0, -15.0, 8.0, 8.0, math.pi / 2),
        ("east", 35.0, 8.0, 8.0, 8.0, math.pi),
        ("west", -20.0, 8.0, 8.0, 8.0, 0.0),
    )
    poses: list[EgoPose] = []
    for approach, x0, y0, x1, y1, heading in definitions:
        for index in range(n_per_approach):
            fraction = index / max(n_per_approach - 1, 1)
            poses.append(
                EgoPose(
                    ego_id=f"{approach}_{index:02d}",
                    approach=approach,
                    x=x0 + fraction * (x1 - x0),
                    y=y0 + fraction * (y1 - y0),
                    heading_rad=heading,
                )
            )
    return poses


def balanced_stratified_frames(
    tracks: pd.DataFrame,
    n_frames: int,
    *,
    seed: int = 20260902,
    time_strata: int = 10,
    density_strata: int = 4,
) -> np.ndarray:
    """Sample frames across time and object-count strata without replacement."""

    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    if "frame_idx" not in tracks:
        raise ValueError("tracks must contain frame_idx")
    counts = tracks.groupby("frame_idx", sort=True).size().rename("object_count").reset_index()
    if counts.empty:
        raise ValueError("tracks contains no frames")
    if n_frames >= len(counts):
        return counts["frame_idx"].to_numpy(dtype=int)

    counts["time_stratum"] = np.minimum(
        int(time_strata) - 1,
        np.floor(np.arange(len(counts)) * int(time_strata) / len(counts)).astype(int),
    )
    n_density = min(int(density_strata), int(counts["object_count"].nunique()))
    if n_density <= 1:
        counts["density_stratum"] = 0
    else:
        counts["density_stratum"] = pd.qcut(
            counts["object_count"].rank(method="first"),
            q=n_density,
            labels=False,
            duplicates="drop",
        ).astype(int)

    rng = np.random.default_rng(seed)
    pools: list[list[int]] = []
    for _, group in counts.groupby(["time_stratum", "density_stratum"], sort=True):
        values = group["frame_idx"].astype(int).to_numpy(copy=True)
        rng.shuffle(values)
        pools.append(values.tolist())
    selected: list[int] = []
    while len(selected) < n_frames:
        progressed = False
        for pool in pools:
            if pool and len(selected) < n_frames:
                selected.append(pool.pop())
                progressed = True
        if not progressed:
            break
    return np.sort(np.asarray(selected, dtype=int))


def _sensor_origin(pose: EgoPose, plane: GroundPlane, config: ValidationConfig) -> np.ndarray:
    x = pose.x + config.sensor_forward_offset_m * math.cos(pose.heading_rad)
    y = pose.y + config.sensor_forward_offset_m * math.sin(pose.heading_rad)
    z = float(plane.height_at(x, y)) + config.sensor_height_m
    return np.asarray((x, y, z), dtype=float)


def _scene_is_clear(
    pose: EgoPose,
    boxes: Sequence[OrientedBox],
    config: ValidationConfig,
) -> tuple[bool, int | str | None, float | None]:
    nearest_id: int | str | None = None
    nearest_distance = math.inf
    for box in boxes:
        if box.class_name not in config.occluder_classes:
            continue
        distance = math.hypot(box.center[0] - pose.x, box.center[1] - pose.y)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_id = box.object_id
    return (
        nearest_distance >= config.ego_clearance_m,
        nearest_id,
        None if not math.isfinite(nearest_distance) else float(nearest_distance),
    )


def visible_at_fraction(fraction: float | np.ndarray, threshold: float) -> bool | np.ndarray:
    """Apply a partial-visibility threshold, with zero denoting any visible ray."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    values = np.asarray(fraction, dtype=float)
    visible = values > 0.0 if threshold == 0 else values >= threshold
    if values.ndim == 0:
        return bool(visible)
    return visible


def sparse_target_points(target: OrientedBox) -> dict[str, np.ndarray]:
    """Return center, face-center, and vertex samples for a 15-ray surrogate."""

    hx, hy, hz = target.half_size
    local: dict[str, tuple[float, float, float]] = {
        "center": (0.0, 0.0, 0.0),
        "face_pos_x": (hx, 0.0, 0.0),
        "face_neg_x": (-hx, 0.0, 0.0),
        "face_pos_y": (0.0, hy, 0.0),
        "face_neg_y": (0.0, -hy, 0.0),
        "face_top": (0.0, 0.0, hz),
        "face_bottom": (0.0, 0.0, -hz),
    }
    for z_name, z in (("bottom", -hz), ("top", hz)):
        for x_name, x in (("negx", -hx), ("posx", hx)):
            for y_name, y in (("negy", -hy), ("posy", hy)):
                local[f"vertex_{z_name}_{x_name}_{y_name}"] = (x, y, z)
    return {name: target.local_to_world(point) for name, point in local.items()}


def evaluate_frame(
    frame_tracks: pd.DataFrame,
    *,
    ego_poses: Sequence[EgoPose],
    ground_plane: GroundPlane,
    config: ValidationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all ego poses for one frame.

    Returns decision rows, explicit ego-placement exclusions, and timing rows.
    """

    if frame_tracks.empty:
        return [], [], []
    frame_values = frame_tracks["frame_idx"].astype(int).unique()
    if len(frame_values) != 1:
        raise ValueError("evaluate_frame expects exactly one frame")
    frame_idx = int(frame_values[0])

    raw_boxes = [OrientedBox.from_row(row, z_mode="raw") for row in frame_tracks.to_dict("records")]
    boxes_by_mode: dict[str, list[OrientedBox]] = {"raw": raw_boxes}
    if "ground_anchored" in config.z_modes:
        boxes_by_mode["ground_anchored"] = [
            OrientedBox.from_row(row, z_mode="ground_anchored", ground_plane=ground_plane)
            for row in frame_tracks.to_dict("records")
        ]

    decisions: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for pose in ego_poses:
        scene_clear, nearest_id, nearest_distance = _scene_is_clear(pose, raw_boxes, config)
        if not scene_clear:
            exclusions.append(
                {
                    "frame_idx": frame_idx,
                    "ego_id": pose.ego_id,
                    "approach": pose.approach,
                    "reason": "hypothetical_ego_overlaps_observed_vehicle",
                    "nearest_track_id": nearest_id,
                    "nearest_distance_m": nearest_distance,
                    "clearance_m": config.ego_clearance_m,
                }
            )
            continue

        origin = _sensor_origin(pose, ground_plane, config)
        legacy_by_fov: dict[float, dict[Any, Any]] = {}
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
            timings.append(
                {
                    "frame_idx": frame_idx,
                    "ego_id": pose.ego_id,
                    "z_mode": "planar",
                    "target_id": None,
                    "method": "legacy_planar_scene",
                    "grid": None,
                    "fov_deg": fov_deg,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "hit_rays": None,
                }
            )

        for z_mode in config.z_modes:
            boxes = boxes_by_mode[z_mode]
            blockers = [box for box in boxes if box.class_name in config.occluder_classes]
            for target in boxes:
                # Ray models are independent of horizontal FOV. Compute once
                # and gate the same result for each requested FOV.
                maximum_flags = center_coverage_flags(
                    origin,
                    target,
                    heading_rad=pose.heading_rad,
                    fov_rad=2.0 * math.pi,
                    near_range_m=config.near_blind_m,
                    far_range_m=config.range_m,
                )
                low_cost = None
                sparse = None
                exact: dict[int, Any] = {}
                if maximum_flags.in_range:
                    started = time.perf_counter()
                    low_cost = center_top_visibility(
                        origin,
                        target,
                        blockers,
                        criterion="any",
                        clearance_m=config.occlusion_clearance_m,
                    )
                    timings.append(
                        {
                            "frame_idx": frame_idx,
                            "ego_id": pose.ego_id,
                            "z_mode": z_mode,
                            "target_id": target.object_id,
                            "method": "center_top",
                            "grid": 2,
                            "fov_deg": None,
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                            "hit_rays": low_cost.hit_rays,
                        }
                    )
                    started = time.perf_counter()
                    sparse = multi_ray_visibility(
                        origin,
                        target,
                        blockers,
                        target_points=sparse_target_points(target),
                        criterion="any",
                        clearance_m=config.occlusion_clearance_m,
                    )
                    timings.append(
                        {
                            "frame_idx": frame_idx,
                            "ego_id": pose.ego_id,
                            "z_mode": z_mode,
                            "target_id": target.object_id,
                            "method": "sparse_multiray",
                            "grid": sparse.hit_rays,
                            "fov_deg": None,
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                            "hit_rays": sparse.hit_rays,
                        }
                    )
                    for grid in config.ray_grids:
                        started = time.perf_counter()
                        value = projected_visible_fraction(
                            origin,
                            target,
                            blockers,
                            azimuth_samples=grid,
                            elevation_samples=grid,
                            clearance_m=config.occlusion_clearance_m,
                        )
                        exact[grid] = value
                        timings.append(
                            {
                                "frame_idx": frame_idx,
                                "ego_id": pose.ego_id,
                                "z_mode": z_mode,
                                "target_id": target.object_id,
                                "method": "exact_angular_grid",
                                "grid": grid,
                                "fov_deg": None,
                                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                                "hit_rays": value.hit_rays,
                            }
                        )

                for fov_deg in config.fov_degrees:
                    flags = center_coverage_flags(
                        origin,
                        target,
                        heading_rad=pose.heading_rad,
                        fov_rad=math.radians(fov_deg),
                        near_range_m=config.near_blind_m,
                        far_range_m=config.range_m,
                    )
                    legacy = legacy_by_fov[fov_deg][target.object_id]
                    primary = exact.get(config.primary_grid)
                    exact_fraction = primary.score if primary is not None and flags.covered else 0.0
                    exact_visible = bool(
                        flags.covered
                        and visible_at_fraction(exact_fraction, config.visible_fraction_threshold)
                    )
                    low_cost_visible = bool(flags.covered and low_cost is not None and low_cost.visible)
                    sparse_visible = bool(flags.covered and sparse is not None and sparse.visible)
                    row: dict[str, Any] = {
                        "frame_idx": frame_idx,
                        "ego_id": pose.ego_id,
                        "approach": pose.approach,
                        "ego_x": pose.x,
                        "ego_y": pose.y,
                        "ego_heading_deg": math.degrees(pose.heading_rad),
                        "sensor_x": float(origin[0]),
                        "sensor_y": float(origin[1]),
                        "sensor_z": float(origin[2]),
                        "target_id": target.object_id,
                        "class_name": target.class_name,
                        "z_mode": z_mode,
                        "fov_deg": fov_deg,
                        "range_m": flags.range_m,
                        "in_range": flags.in_range,
                        "in_fov": flags.in_fov,
                        "covered": flags.covered,
                        "legacy_visible": bool(legacy.visible),
                        "legacy_reason": legacy.reason,
                        "legacy_occluded_by": legacy.occluded_by,
                        "center_top_visible": low_cost_visible,
                        "center_top_fraction": low_cost.score if low_cost is not None else 0.0,
                        "center_top_occluded_by": low_cost.nearest_blocker
                        if low_cost is not None
                        else None,
                        "sparse_multiray_visible": sparse_visible,
                        "sparse_multiray_fraction": sparse.score if sparse is not None else 0.0,
                        "sparse_multiray_occluded_by": sparse.nearest_blocker
                        if sparse is not None
                        else None,
                        "exact_visible": exact_visible,
                        "exact_visible_fraction": exact_fraction,
                        "exact_threshold": config.visible_fraction_threshold,
                        "exact_occluded_by": primary.nearest_blocker if primary is not None else None,
                    }
                    for grid, result in exact.items():
                        row[f"exact_fraction_{grid}"] = result.score if flags.covered else 0.0
                        row[f"exact_hit_rays_{grid}"] = result.hit_rays if flags.covered else 0
                    decisions.append(row)
    return decisions, exclusions, timings


def summarize_validation(
    decisions: pd.DataFrame,
    timings: pd.DataFrame,
    *,
    sensitivity_thresholds: Sequence[float] = (0.0, 0.01, 0.05, 0.10),
) -> dict[str, Any]:
    """Summarize observability, reference agreement, and runtime."""

    if decisions.empty:
        raise ValueError("decisions table is empty")
    thresholds = tuple(float(value) for value in sensitivity_thresholds)
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("sensitivity_thresholds must contain values in [0, 1]")
    grid_columns = sorted(
        (column for column in decisions.columns if column.startswith("exact_fraction_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    summaries: list[dict[str, Any]] = []
    for (z_mode, fov_deg), group in decisions.groupby(["z_mode", "fov_deg"], sort=True):
        reference = group["exact_visible"].astype(bool)
        entry: dict[str, Any] = {
            "z_mode": str(z_mode),
            "fov_deg": float(fov_deg),
            "decisions": int(len(group)),
            "covered_decisions": int(group["covered"].sum()),
            "observability": {
                method: ratio_of_sums(group[column].astype(int), np.ones(len(group), dtype=int))
                for method, column in (
                    ("legacy_planar", "legacy_visible"),
                    ("center_top", "center_top_visible"),
                    ("sparse_multiray", "sparse_multiray_visible"),
                    ("exact_reference", "exact_visible"),
                )
            },
            "failure_flags": {
                "out_of_range": int((~group["in_range"].astype(bool)).sum()),
                "out_of_fov": int((~group["in_fov"].astype(bool)).sum()),
                "both_range_and_fov": int(
                    ((~group["in_range"].astype(bool)) & (~group["in_fov"].astype(bool))).sum()
                ),
                "exact_occluded_among_covered": int(
                    (group["covered"].astype(bool) & ~reference).sum()
                ),
            },
            "exact_threshold_sensitivity": {
                f"{threshold:g}": float(
                    np.mean(
                        group["covered"].to_numpy(bool)
                        & np.asarray(
                            visible_at_fraction(
                                group["exact_visible_fraction"].to_numpy(float), threshold
                            ),
                            dtype=bool,
                        )
                    )
                )
                for threshold in thresholds
            },
        }
        covered = group[group["covered"].astype(bool)]
        if len(covered):
            entry["agreement_with_exact_among_covered"] = {
                "legacy_planar": binary_agreement_metrics(
                    covered["legacy_visible"], covered["exact_visible"]
                ),
                "center_top": binary_agreement_metrics(
                    covered["center_top_visible"], covered["exact_visible"]
                ),
                "sparse_multiray": binary_agreement_metrics(
                    covered["sparse_multiray_visible"], covered["exact_visible"]
                ),
            }
            entry["per_class"] = {}
            for class_name, class_group in group.groupby("class_name", sort=True):
                class_covered = class_group[class_group["covered"].astype(bool)]
                class_entry: dict[str, Any] = {
                    "decisions": int(len(class_group)),
                    "exact_observability": float(class_group["exact_visible"].mean()),
                }
                if len(class_covered):
                    class_entry["center_top_agreement"] = binary_agreement_metrics(
                        class_covered["center_top_visible"], class_covered["exact_visible"]
                    )
                    class_entry["sparse_multiray_agreement"] = binary_agreement_metrics(
                        class_covered["sparse_multiray_visible"], class_covered["exact_visible"]
                    )
                entry["per_class"][str(class_name)] = class_entry

            primary_fraction = covered["exact_visible_fraction"].to_numpy(float)
            entry["grid_convergence"] = {}
            for column in grid_columns:
                grid = int(column.rsplit("_", 1)[1])
                values = covered[column].to_numpy(float)
                grid_visible = np.asarray(
                    visible_at_fraction(values, float(covered["exact_threshold"].iloc[0])),
                    dtype=bool,
                )
                entry["grid_convergence"][str(grid)] = {
                    "mean_visible_fraction": float(np.mean(values)),
                    "mean_absolute_fraction_difference_from_primary": float(
                        np.mean(np.abs(values - primary_fraction))
                    ),
                    "binary_agreement_with_primary": binary_agreement_metrics(
                        grid_visible, covered["exact_visible"]
                    ),
                }
        summaries.append(entry)

    runtime: list[dict[str, Any]] = []
    if not timings.empty:
        for (method, grid, z_mode), group in timings.assign(
            grid=timings["grid"].fillna(-1), z_mode=timings["z_mode"].fillna("none")
        ).groupby(["method", "grid", "z_mode"], sort=True):
            values = group["elapsed_ms"].to_numpy(float)
            runtime.append(
                {
                    "method": str(method),
                    "grid": None if float(grid) < 0 else int(grid),
                    "z_mode": str(z_mode),
                    "n": int(len(values)),
                    "mean_ms": float(np.mean(values)),
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.quantile(values, 0.95)),
                }
            )
    return {"groups": summaries, "runtime": runtime}


def config_as_dict(config: ValidationConfig) -> dict[str, Any]:
    """Serialize a frozen config, including immutable set fields."""

    result = asdict(config)
    result["occluder_classes"] = sorted(config.occluder_classes)
    return result
