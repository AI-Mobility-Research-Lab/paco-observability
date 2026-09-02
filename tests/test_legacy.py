from __future__ import annotations

import math

from paco_observability.geometry import OrientedBox
from paco_observability.legacy import (
    planar_angular_wedge_visibility,
    theta_bins_for_resolution,
)


def box(identifier: int, x: float, y: float, cls: str = "car") -> OrientedBox:
    return OrientedBox(
        object_id=identifier,
        class_name=cls,
        center=(x, y, 0.75),
        size=(4.0, 2.0, 1.5),
        yaw=0.0,
    )


def test_fair_angular_resolution_scales_with_field_of_view() -> None:
    assert theta_bins_for_resolution(120.0, 0.25) == 480
    assert theta_bins_for_resolution(360.0, 0.25) == 1440


def test_planar_model_separates_coverage_flags_from_reason_priority() -> None:
    decisions = planar_angular_wedge_visibility(
        sensor_xy=(0.0, 0.0),
        heading_rad=0.0,
        boxes=[box(1, -50.0, 0.0)],
        fov_deg=120.0,
        range_m=35.0,
    )
    decision = decisions[1]
    assert not decision.in_range
    assert not decision.in_fov
    assert decision.reason == "out_of_range"


def test_planar_model_blocks_aligned_far_target() -> None:
    decisions = planar_angular_wedge_visibility(
        sensor_xy=(0.0, 0.0),
        heading_rad=0.0,
        boxes=[box(1, 8.0, 0.0), box(2, 16.0, 0.0, "bicycle")],
    )
    assert decisions[1].visible
    assert decisions[2].reason == "occluded"
    assert decisions[2].occluded_by == 1


def test_full_circle_wrap_does_not_occlude_opposite_half_plane() -> None:
    decisions = planar_angular_wedge_visibility(
        sensor_xy=(0.0, 0.0),
        heading_rad=math.pi,
        boxes=[box(1, -8.0, 0.0), box(2, 8.0, 0.0, "bicycle")],
        fov_deg=360.0,
    )
    assert decisions[2].visible
