from __future__ import annotations

import math

import numpy as np
import pytest

from paco_observability.geometry import GroundPlane, OrientedBox, ray_obb_intersection


def test_axis_aligned_ray_has_metric_entry_and_exit() -> None:
    target = OrientedBox(center=(5.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    hit = ray_obb_intersection((0, 0, 0), (10, 0, 0), target)
    assert hit is not None
    assert hit.distance == pytest.approx(4.0)
    assert hit.exit_distance == pytest.approx(6.0)


def test_rotated_box_intersection_is_yaw_aware() -> None:
    target = OrientedBox(
        center=(5.0, 0.0, 0.0), size=(4.0, 2.0, 2.0), yaw=math.pi / 2
    )
    hit = ray_obb_intersection((0, 0, 0), (1, 0, 0), target)
    assert hit is not None
    # After 90-degree rotation the box's 2 m local-y extent lies along world x.
    assert hit.distance == pytest.approx(4.0)


def test_parallel_ray_outside_slab_misses() -> None:
    target = OrientedBox(center=(5.0, 5.0, 0.0), size=(2.0, 2.0, 2.0))
    assert ray_obb_intersection((0, 0, 0), (1, 0, 0), target) is None


def test_origin_inside_and_zero_direction_are_defined() -> None:
    target = OrientedBox(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    moving = ray_obb_intersection((0, 0, 0), (1, 0, 0), target)
    stationary = ray_obb_intersection((0, 0, 0), (0, 0, 0), target)
    assert moving is not None and moving.started_inside and moving.distance == 0
    assert stationary is not None and stationary.exit_distance == 0
    assert ray_obb_intersection((3, 0, 0), (0, 0, 0), target) is None


def test_ground_anchored_box_preserves_dimensions_and_places_bottom_on_plane() -> None:
    plane = GroundPlane(normal=(-0.1, 0.0, 1.0), d=0.0)
    row = {
        "track_id": 4,
        "class_name": "car",
        "bbox_center_x": 10.0,
        "bbox_center_y": 2.0,
        "bbox_center_z": 99.0,
        "bbox_dx": 4.0,
        "bbox_dy": 2.0,
        "bbox_dz": 1.6,
        "bbox_yaw": 0.2,
    }
    box = OrientedBox.from_row(row, z_mode="ground_anchored", ground_plane=plane)
    assert box.bottom_center[2] == pytest.approx(plane.height_at(10.0, 2.0))
    np.testing.assert_allclose(box.size, (4.0, 2.0, 1.6))
