from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from parametric_ego_sweep import (  # noqa: E402
    ApproachLane,
    SensorConfig,
    compute_ego_occlusion,
)


def objects(*rows: dict) -> pd.DataFrame:
    defaults = {
        "class_name": "car",
        "bbox_dx": 4.5,
        "bbox_dy": 2.0,
        "bbox_yaw": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_forward_visibility_and_field_of_view() -> None:
    frame = objects(
        {"track_id": 1, "bbox_center_x": 10.0, "bbox_center_y": 0.0},
        {"track_id": 2, "bbox_center_x": -10.0, "bbox_center_y": 0.0},
    )
    result = compute_ego_occlusion(0.0, 0.0, 0.0, frame, SensorConfig())
    assert result[1]["reason"] == "visible"
    assert result[2]["reason"] == "out_of_fov"


def test_nearer_vehicle_occludes_aligned_target() -> None:
    frame = objects(
        {"track_id": 10, "bbox_center_x": 8.0, "bbox_center_y": 0.0},
        {"track_id": 20, "bbox_center_x": 16.0, "bbox_center_y": 0.0},
    )
    result = compute_ego_occlusion(0.0, 0.0, 0.0, frame, SensorConfig())
    assert result[10]["reason"] == "visible"
    assert result[20]["reason"] == "occluded"
    assert result[20]["occluded_by"] == 10


def test_range_limit() -> None:
    frame = objects({"track_id": 1, "bbox_center_x": 50.0, "bbox_center_y": 0.0})
    result = compute_ego_occlusion(0.0, 0.0, 0.0, frame, SensorConfig(range_m=35.0))
    assert result[1]["reason"] == "out_of_range"


def test_approach_sampling_includes_endpoints() -> None:
    lane = ApproachLane("test", 0.0, 0.0, 10.0, 0.0, math.pi / 2, n_positions=3)
    assert lane.sample_positions() == [
        (0.0, 0.0, math.pi / 2),
        (5.0, 0.0, math.pi / 2),
        (10.0, 0.0, math.pi / 2),
    ]

