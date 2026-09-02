from __future__ import annotations

import pandas as pd
import pytest

from paco_observability.full_record import (
    FullRecordConfig,
    FullRecordSummaryAccumulator,
    audit_full_record_tracks,
    contribution_rows,
    evaluate_frame_counts,
    require_passing_audit,
    select_frame_ids,
)
from paco_observability.geometry import GroundPlane
from paco_observability.validation import EgoPose


def scene(frame_idx: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_idx": [frame_idx, frame_idx],
            "track_id": [1, 2],
            "class_name": ["car", "pedestrian"],
            "bbox_center_x": [5.0, 10.0],
            "bbox_center_y": [0.0, 0.0],
            "bbox_center_z": [1.0, 0.9],
            "bbox_dx": [4.0, 0.8],
            "bbox_dy": [2.0, 0.8],
            "bbox_dz": [2.0, 1.8],
            "bbox_yaw": [0.0, 0.0],
        }
    )


def test_frame_counts_partition_visibility_and_weighted_residual() -> None:
    config = FullRecordConfig(
        fov_degrees=(120.0, 360.0),
        z_modes=("ground_anchored",),
        sensor_forward_offset_m=0.0,
    )
    result = evaluate_frame_counts(
        scene(),
        ego_poses=[EgoPose("ego", "eastbound", 0.0, 0.0, 0.0)],
        ground_plane=GroundPlane(),
        config=config,
    )
    assert len(result.count_rows) == 2
    assert not result.exclusion_rows
    by_fov = {row["fov_deg"]: row for row in result.count_rows}
    assert by_fov[120.0]["legacy_theta_bins"] == 480
    assert by_fov[360.0]["legacy_theta_bins"] == 1440
    for row in result.count_rows:
        assert row["total_count"] == 2
        assert row["covered_count"] == 2
        assert row["sparse_multiray_visible_count"] == 1
        assert row["sparse_multiray_occluded_count"] == 1
        assert row["weighted_total_demand"] == pytest.approx(4.0)
        assert row["sparse_multiray_residual_demand"] == pytest.approx(3.0)
        assert row["vru_sparse_multiray_residual_count"] == 1

    long_rows = list(contribution_rows(result.count_rows))
    assert len(long_rows) == 6
    assert {row["method"] for row in long_rows} == {
        "legacy_planar",
        "center_top",
        "sparse_multiray",
    }
    assert all(row["denominator"] == 2 for row in long_rows)


def test_ego_vehicle_clearance_is_an_explicit_exclusion() -> None:
    result = evaluate_frame_counts(
        scene(),
        ego_poses=[EgoPose("overlap", "eastbound", 4.0, 0.0, 0.0)],
        ground_plane=GroundPlane(),
        config=FullRecordConfig(
            fov_degrees=(120.0,),
            z_modes=("ground_anchored",),
            sensor_forward_offset_m=0.0,
        ),
    )
    assert not result.count_rows
    assert len(result.exclusion_rows) == 1
    assert result.exclusion_rows[0]["nearest_track_id"] == 1
    assert result.exclusion_rows[0]["nearest_distance_m"] == pytest.approx(1.0)


def test_frame_selection_applies_only_then_exclude_then_step_then_limit() -> None:
    tracks = pd.DataFrame({"frame_idx": [1, 2, 3, 4, 5, 6]})
    selected = select_frame_ids(
        tracks,
        only_frames=[1, 2, 3, 4, 5, 99],
        exclude_frames=[2, 99],
        frame_step=2,
        max_frames=2,
    )
    assert selected.tolist() == [1, 4]


def test_full_record_audit_fails_closed_and_summary_uses_ratio_of_sums() -> None:
    plane = GroundPlane()
    audit = audit_full_record_tracks(scene(), ground_plane=plane)
    assert audit["full_record_quality_gate"]["passed"]
    require_passing_audit(audit)

    bad = audit_full_record_tracks(scene().assign(frame_idx=[10, 12]), ground_plane=plane)
    assert not bad["full_record_quality_gate"]["passed"]
    with pytest.raises(ValueError, match="quality gate"):
        require_passing_audit(bad)

    config = FullRecordConfig(
        fov_degrees=(120.0,),
        z_modes=("ground_anchored",),
        sensor_forward_offset_m=0.0,
    )
    first = evaluate_frame_counts(
        scene(10),
        ego_poses=[EgoPose("ego", "eastbound", 0.0, 0.0, 0.0)],
        ground_plane=plane,
        config=config,
        frame_order=0,
    )
    second_scene = scene(11).iloc[[0]].copy()
    second = evaluate_frame_counts(
        second_scene,
        ego_poses=[EgoPose("ego", "eastbound", 0.0, 0.0, 0.0)],
        ground_plane=plane,
        config=config,
        frame_order=1,
    )
    accumulator = FullRecordSummaryAccumulator()
    accumulator.update(first)
    accumulator.update(second)
    summary = accumulator.finalize(
        config=config,
        input_frame_count=2,
        selected_frame_count=2,
        frame_step=1,
        max_frames=None,
        ego_positions=1,
        wall_elapsed_s=1.0,
    )
    group = summary["groups"][0]
    # Sparse visibility is 1/2 in the first frame and 1/1 in the second:
    # ratio-of-sums is 2/3, not the unweighted mean 3/4.
    assert group["observability_ratio_of_sums"]["sparse_multiray"] == pytest.approx(2 / 3)
