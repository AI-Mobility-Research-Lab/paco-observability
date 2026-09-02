from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paco_observability.sparse_calibration import (
    calibrate_sparse_threshold,
    confusion_metrics,
    frame_strata,
    select_candidate_threshold,
    stratified_frame_split,
    visible_ray_counts,
)


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_sparse_threshold.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("calibrate_sparse_threshold_cli", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)


def synthetic_decisions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_decile in range(10):
        for density in range(1, 5):
            for replicate in range(5):
                frame_idx = time_decile * 20 + (density - 1) * 5 + replicate + 1
                for target_id in range(1, density + 1):
                    exact_visible = target_id % 2 == 0
                    ray_count = 3 if exact_visible else 2
                    for fov_deg in (120.0, 360.0):
                        rows.append(
                            {
                                "frame_idx": frame_idx,
                                "ego_id": "ego_00",
                                "target_id": target_id,
                                "z_mode": "ground_anchored",
                                "fov_deg": fov_deg,
                                "covered": True,
                                "exact_visible": exact_visible,
                                "sparse_multiray_fraction": ray_count / 15,
                                "sparse_multiray_visible": True,
                            }
                        )
    return pd.DataFrame(rows)


def test_visible_ray_count_recovery_is_strict() -> None:
    fractions = pd.Series([0.0, 1 / 15, 2 / 15, 1.0], index=[4, 5, 6, 7])
    counts = visible_ray_counts(fractions)
    assert counts.to_dict() == {4: 0, 5: 1, 6: 2, 7: 15}
    assert counts.dtype == np.int8

    with pytest.raises(ValueError, match="integer-valued"):
        visible_ray_counts(pd.Series([0.1]))
    with pytest.raises(ValueError, match="finite"):
        visible_ray_counts(pd.Series([np.nan]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        visible_ray_counts(pd.Series([1.1]))


def test_confusion_metrics_include_balanced_accuracy_fpr_and_bias() -> None:
    metrics = confusion_metrics(
        predicted_visible=[True, True, False, False],
        reference_visible=[True, False, True, False],
    )
    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (1, 1, 1, 1)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["prediction_reference_rate_bias"] == pytest.approx(0.0)


def test_frame_split_is_deterministic_disjoint_and_nearly_half_per_stratum() -> None:
    strata = frame_strata(synthetic_decisions())
    cell_sizes = strata.groupby(["time_decile", "frame_density_quartile"]).size()
    assert len(strata) == 200
    assert len(cell_sizes) == 40
    assert cell_sizes.eq(5).all()

    first_cal, first_eval, first_assignments = stratified_frame_split(
        strata, calibration_frame_count=100, seed=20260902
    )
    second_cal, second_eval, second_assignments = stratified_frame_split(
        strata, calibration_frame_count=100, seed=20260902
    )
    other_cal, _, _ = stratified_frame_split(
        strata, calibration_frame_count=100, seed=20260903
    )

    assert first_cal == second_cal
    assert first_eval == second_eval
    pd.testing.assert_frame_equal(first_assignments, second_assignments)
    assert first_cal != other_cal
    assert len(first_cal) == len(first_eval) == 100
    assert set(first_cal).isdisjoint(first_eval)
    assert set(first_cal).union(first_eval) == set(range(1, 201))
    per_cell = first_assignments.groupby(
        ["time_decile", "frame_density_quartile", "split"]
    ).size().unstack(fill_value=0)
    assert (per_cell["calibration"] - per_cell["held_out"]).abs().eq(1).all()


def _candidate(threshold: int, correct: int, predicted: int, reference: int) -> dict:
    return {
        "min_visible_rays": threshold,
        "calibration": {
            "overall": {
                "correct_n": correct,
                "predicted_visible_n": predicted,
                "reference_visible_n": reference,
            }
        },
    }


def test_selection_tie_breaks_by_rate_bias_then_higher_threshold() -> None:
    selected, accuracy_ties, bias_ties = select_candidate_threshold(
        [
            _candidate(1, correct=10, predicted=8, reference=8),
            _candidate(2, correct=10, predicted=6, reference=8),
            _candidate(3, correct=9, predicted=8, reference=8),
        ]
    )
    assert selected == 1
    assert accuracy_ties == [1, 2]
    assert bias_ties == [1]

    selected, _, bias_ties = select_candidate_threshold(
        [
            _candidate(4, correct=10, predicted=8, reference=8),
            _candidate(5, correct=10, predicted=8, reference=8),
        ]
    )
    assert selected == 5
    assert bias_ties == [4, 5]


def test_end_to_end_calibration_selects_three_without_held_out_leakage() -> None:
    source = synthetic_decisions()
    summary, calibrated = calibrate_sparse_threshold(source)

    assert summary["selection"]["selected_min_visible_rays"] == 3
    assert summary["selection"]["matches_expected_selection"] is True
    assert summary["protocol"]["expected_selected_min_visible_rays"] == 3
    assert "pre_registered_expected_selected_min_visible_rays" not in summary["protocol"]
    assert summary["split"]["calibration_frame_count"] == 100
    assert summary["split"]["held_out_frame_count"] == 100
    assert summary["split"]["disjoint"] is True
    assert summary["split"]["covers_all_sampled_frames"] is True
    assert len(summary["candidates"]) == 15
    assert set(summary["selection"]["calibration"]["by_fov"]) == {"120", "360"}
    assert summary["selection"]["calibration"]["overall"]["accuracy"] == 1.0
    assert summary["selection"]["held_out"]["overall"]["accuracy"] == 1.0
    assert all(row["split_count_difference"] == 1 for row in summary["strata"])

    assert "sparse_multiray_visible_any" not in source.columns
    assert calibrated["sparse_multiray_visible_input"].all()
    assert calibrated["sparse_multiray_visible_any"].all()
    assert calibrated["sparse_multiray_min_visible_rays"].eq(3).all()
    assert calibrated["sparse_multiray_visible"].equals(calibrated["exact_visible"])
    assert not calibrated["sparse_calibration_split"].isna().any()
    assert set(calibrated["sparse_calibration_split"]) == {"calibration", "held_out"}
    calibration_frames = set(summary["split"]["calibration_frames"])
    held_out_frames = set(summary["split"]["held_out_frames"])
    assert calibrated.loc[
        calibrated["sparse_calibration_split"].eq("calibration"), "frame_idx"
    ].isin(calibration_frames).all()
    assert calibrated.loc[
        calibrated["sparse_calibration_split"].eq("held_out"), "frame_idx"
    ].isin(held_out_frames).all()


def test_inconsistent_original_any_ray_semantics_fail_closed() -> None:
    decisions = synthetic_decisions()
    decisions.loc[0, "sparse_multiray_visible"] = False
    with pytest.raises(ValueError, match="any-ray semantics"):
        calibrate_sparse_threshold(decisions)


def test_new_schema_uses_explicit_any_column_and_preserves_input_decision() -> None:
    decisions = synthetic_decisions()
    decisions["sparse_multiray_visible_any"] = decisions["sparse_multiray_visible"]
    decisions["sparse_multiray_visible"] = decisions["exact_visible"]

    summary, calibrated = calibrate_sparse_threshold(decisions)

    assert summary["selection"]["selected_min_visible_rays"] == 3
    assert calibrated["sparse_multiray_visible_input"].equals(decisions["exact_visible"])
    assert calibrated["sparse_multiray_visible_any"].all()
    assert calibrated["sparse_multiray_visible"].equals(decisions["exact_visible"])


def test_explicit_any_column_is_validated_even_when_input_decision_differs() -> None:
    decisions = synthetic_decisions()
    decisions["sparse_multiray_visible_any"] = decisions["sparse_multiray_visible"]
    decisions["sparse_multiray_visible"] = decisions["exact_visible"]
    decisions.loc[0, "sparse_multiray_visible_any"] = False

    with pytest.raises(ValueError, match="sparse_multiray_visible_any"):
        calibrate_sparse_threshold(decisions)


def test_existing_visible_ray_column_must_agree_with_fraction() -> None:
    decisions = synthetic_decisions()
    decisions["sparse_multiray_visible_rays"] = 2
    decisions.loc[decisions["exact_visible"], "sparse_multiray_visible_rays"] = 4
    with pytest.raises(ValueError, match=r"disagrees with fraction \* 15"):
        calibrate_sparse_threshold(decisions)


def test_duplicate_decision_key_and_missing_fov_fail_closed() -> None:
    decisions = synthetic_decisions()
    duplicate = pd.concat([decisions, decisions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        calibrate_sparse_threshold(duplicate)

    one_fov = decisions.loc[decisions["fov_deg"].eq(120.0)].copy()
    with pytest.raises(ValueError, match="do not match expected"):
        calibrate_sparse_threshold(one_fov)


def test_cli_writes_fixed_artifact_names(tmp_path: Path) -> None:
    decisions_path = tmp_path / "validation_decisions.parquet"
    out_dir = tmp_path / "calibration"
    synthetic_decisions().to_parquet(decisions_path, index=False)

    assert (
        _SCRIPT.main(
            [
                "--decisions",
                str(decisions_path),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    summary_path = out_dir / "sparse_threshold_calibration.json"
    calibrated_path = out_dir / "validation_decisions_calibrated.parquet"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibrated = pd.read_parquet(calibrated_path)
    assert summary["selection"]["selected_min_visible_rays"] == 3
    assert summary["output"]["calibrated_decisions"]["rows"] == len(calibrated)
    assert summary["output"]["calibrated_decisions"]["sha256"]
    assert "sparse_calibration_split" in summary["output"]["calibrated_decisions"][
        "columns"
    ]
    assert not calibrated["sparse_calibration_split"].isna().any()
    assert calibrated_path.exists()


def test_cli_expected_selection_is_a_fail_closed_regression_guard(tmp_path: Path) -> None:
    decisions = synthetic_decisions()
    decisions["exact_visible"] = False
    decisions["sparse_multiray_fraction"] = 0.0
    decisions["sparse_multiray_visible"] = False
    decisions_path = tmp_path / "validation_decisions.parquet"
    decisions.to_parquet(decisions_path, index=False)

    guarded_dir = tmp_path / "guarded"
    with pytest.raises(ValueError, match="does not match"):
        _SCRIPT.main(
            [
                "--decisions",
                str(decisions_path),
                "--out-dir",
                str(guarded_dir),
            ]
        )
    assert not guarded_dir.exists()

    allowed_dir = tmp_path / "allowed"
    assert (
        _SCRIPT.main(
            [
                "--decisions",
                str(decisions_path),
                "--out-dir",
                str(allowed_dir),
                "--allow-unexpected-selection",
            ]
        )
        == 0
    )
    summary = json.loads(
        (allowed_dir / "sparse_threshold_calibration.json").read_text(encoding="utf-8")
    )
    assert summary["selection"]["selected_min_visible_rays"] == 15
    assert summary["selection"]["matches_expected_selection"] is False
