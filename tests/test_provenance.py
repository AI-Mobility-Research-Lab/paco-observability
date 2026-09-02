from __future__ import annotations

import pandas as pd
import pytest

from paco_observability.provenance import audit_frame_index, audit_tracks, detect_partial_scans


def valid_tracks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_idx": [1, 2],
            "track_id": [10, 10],
            "class_name": ["car", "car"],
            "bbox_center_x": [0.0, 0.1],
            "bbox_center_y": [0.0, 0.0],
            "bbox_center_z": [0.75, 0.75],
            "bbox_dx": [4.0, 4.0],
            "bbox_dy": [2.0, 2.0],
            "bbox_dz": [1.5, 1.5],
            "bbox_yaw": [0.0, 0.0],
        }
    )


def test_audit_passes_complete_unique_boxes() -> None:
    report = audit_tracks(valid_tracks(), ground_normal=(0.0, 0.0, 1.0), ground_offset=0.0)
    assert report["quality_gate"]["passed"]
    assert report["frames"] == 2
    assert report["ground_plane"]["box_bottom_residual_m"]["q50"] == pytest.approx(0.0)


def test_audit_fails_duplicate_frame_track_and_bad_dimension() -> None:
    data = valid_tracks()
    data.loc[1, "frame_idx"] = 1
    data.loc[1, "bbox_dx"] = -1.0
    report = audit_tracks(data)
    assert not report["quality_gate"]["passed"]
    assert report["duplicate_frame_track_rows"] == 1
    assert report["invalid_dimension_rows"] == 1


def test_audit_rejects_missing_3d_field() -> None:
    with pytest.raises(ValueError, match="bbox_center_z"):
        audit_tracks(valid_tracks().drop(columns="bbox_center_z"))


def test_missing_frame_must_be_explicitly_allowlisted() -> None:
    data = pd.concat(
        [valid_tracks().iloc[[0]], valid_tracks().iloc[[1]].assign(frame_idx=3)],
        ignore_index=True,
    )
    failed = audit_tracks(data, expected_frame_range=(1, 3))
    assert not failed["quality_gate"]["passed"]
    assert failed["unapproved_missing_frames"] == [2]
    passed = audit_tracks(
        data, expected_frame_range=(1, 3), allowed_missing_frames=[2]
    )
    assert passed["quality_gate"]["passed"]


def test_partial_scan_detection_uses_adjacent_point_count_median() -> None:
    index = pd.DataFrame(
        {
            "frame_idx": range(1, 8),
            "num_points": [100, 101, 99, 20, 102, 98, 100],
        }
    )
    partial = detect_partial_scans(index, neighbor_radius=3, point_ratio_threshold=0.6)
    assert partial["frame_idx"].tolist() == [4]
    assert partial.iloc[0]["neighbor_median_points"] == pytest.approx(100.0)
    assert partial.iloc[0]["point_count_ratio"] == pytest.approx(0.2)


def test_frame_index_gate_requires_detected_partial_scan_exclusion() -> None:
    index = pd.DataFrame(
        {
            "frame_idx": range(1, 8),
            "num_points": [100, 101, 99, 20, 102, 98, 100],
        }
    )
    failed = audit_frame_index(index, expected_frame_range=(1, 7))
    assert not failed["quality_gate"]["passed"]
    assert failed["unexcluded_partial_frames"] == [4]
    passed = audit_frame_index(
        index,
        expected_frame_range=(1, 7),
        excluded_partial_frames=[4],
    )
    assert passed["quality_gate"]["passed"]
    assert passed["partial_scan_evidence"][0]["reason"] == "partial_scan_low_point_ratio"
