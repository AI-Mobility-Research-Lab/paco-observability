from __future__ import annotations

import pandas as pd

from paco_observability.geometry import GroundPlane
from paco_observability.validation import (
    EgoPose,
    ValidationConfig,
    balanced_stratified_frames,
    default_ego_grid,
    evaluate_frame,
    sparse_target_points,
    visible_at_fraction,
)


def track(frame: int, identifier: int, x: float, y: float) -> dict:
    return {
        "frame_idx": frame,
        "track_id": identifier,
        "class_name": "car",
        "bbox_center_x": x,
        "bbox_center_y": y,
        "bbox_center_z": 0.75,
        "bbox_dx": 4.0,
        "bbox_dy": 2.0,
        "bbox_dz": 1.5,
        "bbox_yaw": 0.0,
    }


def test_default_ego_grid_has_endpoint_inclusive_four_approach_layout() -> None:
    grid = default_ego_grid(3)
    assert len(grid) == 12
    north = [pose for pose in grid if pose.approach == "north"]
    assert [(pose.x, pose.y) for pose in north] == [(7.0, 35.0), (7.0, 21.5), (7.0, 8.0)]


def test_balanced_sample_is_deterministic_and_unique() -> None:
    data = pd.DataFrame(
        [track(frame, frame * 10 + index, float(index), 0.0) for frame in range(1, 41) for index in range(frame % 7 + 1)]
    )
    first = balanced_stratified_frames(data, 12, seed=4)
    second = balanced_stratified_frames(data, 12, seed=4)
    assert first.tolist() == second.tolist()
    assert len(set(first.tolist())) == 12


def test_evaluate_frame_records_overlap_exclusion_instead_of_silent_bias() -> None:
    data = pd.DataFrame([track(1, 1, 0.0, 0.0), track(1, 2, 10.0, 0.0)])
    poses = [EgoPose("overlap", "test", 0.0, 0.0, 0.0)]
    config = ValidationConfig(
        fov_degrees=(120.0,), ray_grids=(3,), z_modes=("raw",), ego_clearance_m=4.0
    )
    rows, exclusions, timings = evaluate_frame(
        data, ego_poses=poses, ground_plane=GroundPlane(), config=config
    )
    assert rows == []
    assert timings == []
    assert exclusions[0]["nearest_track_id"] == 1


def test_zero_visibility_threshold_means_any_positive_fraction() -> None:
    assert not visible_at_fraction(0.0, 0.0)
    assert visible_at_fraction(1e-12, 0.0)
    assert not visible_at_fraction(0.049, 0.05)
    assert visible_at_fraction(0.05, 0.05)


def test_sparse_surrogate_has_center_six_faces_and_eight_vertices() -> None:
    from paco_observability.geometry import OrientedBox

    points = sparse_target_points(OrientedBox(center=(0, 0, 0), size=(4, 2, 2)))
    assert len(points) == 15
    assert set(("center", "face_top", "face_bottom")).issubset(points)
