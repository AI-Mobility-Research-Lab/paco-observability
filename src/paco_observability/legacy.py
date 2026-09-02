"""Reproducible implementation of the manuscript-era planar occlusion model.

This module preserves the angular-wedge approximation as a comparison method,
while fixing one experimental-design issue: different fields of view use a
common angular resolution instead of a common number of bins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import OrientedBox


@dataclass(frozen=True)
class PlanarDecision:
    """Exclusive decision plus non-exclusive sensor-limit flags."""

    visible: bool
    reason: str
    range_m: float
    bearing_relative_rad: float
    in_range: bool
    in_fov: bool
    occluded: bool
    occluded_by: int | str | None = None


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Normalize an angle or array to ``[-pi, pi)``."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def theta_bins_for_resolution(fov_deg: float, resolution_deg: float = 0.25) -> int:
    """Return the bin count needed to keep angular resolution fixed."""

    if not (0 < fov_deg <= 360):
        raise ValueError("fov_deg must be in (0, 360]")
    if resolution_deg <= 0:
        raise ValueError("resolution_deg must be positive")
    return max(1, int(math.ceil(fov_deg / resolution_deg)))


def box_corners_xy(box: OrientedBox) -> np.ndarray:
    """Return the four BEV corners of an oriented box."""

    half = np.array([box.size[0] / 2.0, box.size[1] / 2.0], dtype=float)
    local = np.array(
        [
            [half[0], half[1]],
            [half[0], -half[1]],
            [-half[0], -half[1]],
            [-half[0], half[1]],
        ]
    )
    cosine, sine = math.cos(box.yaw), math.sin(box.yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + box.center[:2]


def _exclusive_reason(in_range: bool, in_fov: bool, occluded: bool) -> str:
    # A fixed priority makes simultaneous coverage failures reproducible while
    # the separate flags retain the non-exclusive decomposition.
    if not in_range:
        return "out_of_range"
    if not in_fov:
        return "out_of_fov"
    if occluded:
        return "occluded"
    return "visible"


def planar_angular_wedge_visibility(
    *,
    sensor_xy: tuple[float, float],
    heading_rad: float,
    boxes: Iterable[OrientedBox],
    fov_deg: float = 120.0,
    range_m: float = 35.0,
    near_blind_m: float = 1.5,
    angular_resolution_deg: float = 0.25,
    clearance_m: float = 0.5,
    occluder_classes: frozenset[str] = frozenset({"car", "truck"}),
) -> dict[int | str, PlanarDecision]:
    """Classify boxes with the legacy angular-wedge center-ray approximation.

    Boxes are processed from near to far. A vehicle marks all angular bins in
    its BEV silhouette; a target is hidden when its center bin was marked by a
    nearer vehicle. The approximation is retained for ablation, not used as the
    high-resolution validation reference.
    """

    if range_m <= 0 or near_blind_m < 0 or near_blind_m >= range_m:
        raise ValueError("range limits must satisfy 0 <= near_blind_m < range_m")
    if clearance_m < 0:
        raise ValueError("clearance_m cannot be negative")

    half_fov = math.radians(fov_deg) / 2.0
    full_circle = fov_deg >= 360.0 - 1e-9
    n_bins = theta_bins_for_resolution(fov_deg, angular_resolution_deg)
    bin_width = 2.0 * half_fov / n_bins
    sensor = np.asarray(sensor_xy, dtype=float)
    if sensor.shape != (2,) or not np.all(np.isfinite(sensor)):
        raise ValueError("sensor_xy must contain two finite coordinates")

    records: list[tuple[float, float, OrientedBox]] = []
    for box in boxes:
        delta = box.center[:2] - sensor
        distance = float(np.linalg.norm(delta))
        bearing = float(normalize_angle(math.atan2(delta[1], delta[0]) - heading_rad))
        records.append((distance, bearing, box))
    records.sort(key=lambda item: (item[0], str(item[2].object_id)))

    min_blocker_range = np.full(n_bins, np.inf, dtype=float)
    blocker_ids: list[int | str | None] = [None] * n_bins
    decisions: dict[int | str, PlanarDecision] = {}

    def bin_index(relative_bearing: float) -> int:
        raw = int(math.floor((relative_bearing + half_fov) / bin_width))
        return min(n_bins - 1, max(0, raw))

    for distance, bearing, box in records:
        in_fov = full_circle or abs(bearing) <= half_fov
        in_range = near_blind_m < distance <= range_m
        center_bin = bin_index(bearing) if in_fov else None
        blocked = bool(
            center_bin is not None
            and distance > min_blocker_range[center_bin] + clearance_m
        )
        reason = _exclusive_reason(in_range, in_fov, blocked)
        decisions[box.object_id] = PlanarDecision(
            visible=reason == "visible",
            reason=reason,
            range_m=distance,
            bearing_relative_rad=bearing,
            in_range=in_range,
            in_fov=in_fov,
            occluded=blocked,
            occluded_by=blocker_ids[center_bin] if blocked and center_bin is not None else None,
        )

        if box.class_name not in occluder_classes:
            continue
        corners = box_corners_xy(box) - sensor
        relative = normalize_angle(np.arctan2(corners[:, 1], corners[:, 0]) - heading_rad)
        corner_ranges = np.linalg.norm(corners, axis=1)
        near_range = float(corner_ranges.min())

        valid_bins = sorted(
            {
                bin_index(float(angle))
                for angle in relative
                if full_circle or -half_fov <= float(angle) <= half_fov
            }
        )
        if not valid_bins:
            continue
        if len(valid_bins) == 1:
            fill_bins: Iterable[int] = valid_bins
        else:
            low, high = valid_bins[0], valid_bins[-1]
            if full_circle and high - low > n_bins // 2:
                fill_bins = (*range(high, n_bins), *range(0, low + 1))
            else:
                fill_bins = range(low, high + 1)
        for index in fill_bins:
            if near_range < min_blocker_range[index]:
                min_blocker_range[index] = near_range
                blocker_ids[index] = box.object_id

    return decisions
