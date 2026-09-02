from __future__ import annotations

import math

import pytest

from paco_observability.geometry import OrientedBox
from paco_observability.occlusion import (
    center_coverage_flags,
    center_top_visibility,
    projected_visible_fraction,
)


def box(identifier: int, center, size=(2.0, 2.0, 2.0), cls="car") -> OrientedBox:
    return OrientedBox(center=center, size=size, object_id=identifier, class_name=cls)


def test_fov_wrap_boundary_is_included_for_full_circle() -> None:
    target = box(1, (-10.0, 0.0, 0.0))
    flags = center_coverage_flags(
        (0, 0, 1.8), target, heading_rad=0.0, fov_rad=2 * math.pi, far_range_m=35
    )
    assert flags.in_fov and flags.in_range
    assert abs(abs(flags.relative_bearing_rad) - math.pi) < 1e-12


def test_exact_grid_fully_blocks_aligned_target() -> None:
    target = box(2, (10.0, 0.0, 1.0), size=(2, 2, 2), cls="bicycle")
    blocker = box(1, (5.0, 0.0, 1.0), size=(2, 4, 4))
    result = projected_visible_fraction(
        (0, 0, 1.8), target, [blocker], azimuth_samples=9, elevation_samples=9
    )
    assert result.hit_rays > 0
    assert result.visible_rays == 0
    assert result.nearest_blocker == 1


def test_exact_grid_ignores_off_axis_blocker() -> None:
    target = box(2, (10.0, 0.0, 1.0), cls="bicycle")
    blocker = box(1, (5.0, 5.0, 1.0))
    result = projected_visible_fraction(
        (0, 0, 1.8), target, [blocker], azimuth_samples=7, elevation_samples=7
    )
    assert result.score == pytest.approx(1.0)


def test_center_top_recovers_target_top_above_short_blocker() -> None:
    target = box(2, (10.0, 0.0, 2.0), size=(2, 2, 4), cls="truck")
    blocker = box(1, (5.0, 0.0, 0.5), size=(2, 3, 1), cls="car")
    result = center_top_visibility((0, 0, 1.8), target, [blocker], criterion="any")
    assert result.visible
    assert "top" in result.visible_labels


def test_partial_occlusion_produces_fraction_strictly_between_zero_and_one() -> None:
    target = box(2, (10.0, 0.0, 1.0), size=(3, 4, 2), cls="truck")
    blocker = box(1, (5.0, -0.8, 1.0), size=(2, 1.8, 2), cls="car")
    result = projected_visible_fraction(
        (0, 0, 1.8), target, [blocker], azimuth_samples=17, elevation_samples=17
    )
    assert 0.0 < result.score < 1.0
