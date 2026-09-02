from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_block_sensitivity.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_block_sensitivity_cli", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
analyze_block_sensitivity = _SCRIPT.analyze_block_sensitivity
influence_acf_summary = _SCRIPT.influence_acf_summary
main = _SCRIPT.main


def test_influence_acf_reports_decay_and_first_zero_crossing() -> None:
    # p=1/2 and influence=[1/2, 1/2, -1/2, -1/2]. With fixed lag-zero
    # normalization the ACF is [1, 1/4, -1/2, -1/4].
    result = influence_acf_summary(
        numerator=[1, 1, 0, 0],
        denominator=[1, 1, 1, 1],
        frame_rate_hz=10.0,
        max_lag=3,
    )

    assert result["status"] == "ok"
    np.testing.assert_allclose(
        [entry["value"] for entry in result["acf"]],
        [1.0, 0.25, -0.5, -0.25],
        atol=1e-12,
    )
    assert result["first_below_one_over_e_lag_frames"] == 1
    assert result["first_below_one_over_e_lag_seconds"] == pytest.approx(0.1)
    assert result["first_zero_crossing_lag_frames"] == 2
    assert result["first_zero_crossing_lag_seconds"] == pytest.approx(0.2)


def test_zero_variance_influence_acf_is_explicitly_undefined() -> None:
    result = influence_acf_summary(
        numerator=[1, 2, 3],
        denominator=[2, 4, 6],
        frame_rate_hz=10.0,
        max_lag=20,
    )

    assert result["status"] == "undefined_zero_variance"
    assert result["lag_truncated_to_record"] is True
    assert result["effective_max_lag_frames"] == 2
    assert result["acf"] == []
    assert result["first_below_one_over_e_lag_frames"] is None
    assert result["first_zero_crossing_lag_frames"] is None


def test_sensitivity_uses_ratio_of_sums_and_skips_overlong_blocks() -> None:
    # The frame ratios [1, 0, 1, 0] average to 0.5. Ratio-of-sums is 2/20=0.1.
    data = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3],
            "z_mode": ["raw"] * 4,
            "fov_deg": [120.0] * 4,
            "method": ["exact"] * 4,
            "numerator": [1, 0, 1, 0],
            "denominator": [1, 9, 1, 9],
        }
    )
    result = analyze_block_sensitivity(
        data,
        block_seconds_values=(0.1, 0.2, 0.5),
        reference_block_seconds=0.2,
        frame_rate_hz=10.0,
        n_resamples=64,
        seed=13,
        acf_max_lag=3,
        paired_fovs=None,
    )

    group = result["groups"][0]
    assert result["estimand"]["name"] == "ratio_of_sums"
    assert group["point_estimate_ratio_of_sums"] == pytest.approx(0.1)
    first, reference, overlong = group["blocks"]
    assert first["status"] == "completed"
    assert first["estimate"] == pytest.approx(0.1)
    assert reference["status"] == "completed"
    assert reference["block_frame_count"] == 2
    assert reference["estimate_change_vs_reference"] == pytest.approx(0.0)
    assert reference["ci_width_change_vs_reference"] == pytest.approx(0.0)
    assert overlong["status"] == "skipped"
    assert overlong["skip_reason"] == "block_frame_count_exceeds_available_frames"
    assert overlong["block_frame_count"] == 5
    assert overlong["available_frame_count"] == 4
    assert overlong["ci_low"] is None
    assert overlong["ci_high"] is None
    assert overlong["ci_width"] is None
    assert overlong["relative_change_status"] == "current_block_skipped"


def test_analysis_is_seed_reproducible_across_all_block_durations() -> None:
    data = pd.DataFrame(
        {
            "frame_idx": np.arange(8),
            "z_mode": ["raw"] * 8,
            "fov_deg": [120.0] * 8,
            "method": ["center_top"] * 8,
            "numerator": [0, 0, 1, 1, 0, 1, 0, 1],
            "denominator": [1] * 8,
        }
    )
    options = {
        "block_seconds_values": (0.1, 0.2, 0.4),
        "reference_block_seconds": 0.2,
        "frame_rate_hz": 10.0,
        "n_resamples": 48,
        "seed": 91,
        "acf_max_lag": 4,
        "paired_fovs": None,
    }

    first = analyze_block_sensitivity(data, **options)
    repeated = analyze_block_sensitivity(data, **options)

    assert first == repeated
    assert first["bootstrap"]["seed"] == 91
    assert first["bootstrap"]["same_seed_for_all_durations"] is True
    assert [entry["block_frame_count"] for entry in first["groups"][0]["blocks"]] == [
        1,
        2,
        4,
    ]


def test_relative_changes_are_not_fabricated_when_reference_block_is_skipped() -> None:
    data = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3],
            "z_mode": ["raw"] * 4,
            "fov_deg": [120.0] * 4,
            "method": ["exact"] * 4,
            "numerator": [0, 1, 0, 1],
            "denominator": [1, 1, 1, 1],
        }
    )
    result = analyze_block_sensitivity(
        data,
        block_seconds_values=(0.1, 0.5),
        reference_block_seconds=0.5,
        frame_rate_hz=10.0,
        n_resamples=16,
        seed=2,
        acf_max_lag=2,
        paired_fovs=None,
    )

    shorter, reference = result["groups"][0]["blocks"]
    assert shorter["status"] == "completed"
    assert shorter["relative_change_status"] == "reference_block_skipped"
    assert shorter["estimate_change_vs_reference"] is None
    assert shorter["ci_width_relative_change_vs_reference"] is None
    assert reference["status"] == "skipped"
    assert reference["relative_change_status"] == "current_block_skipped"


def test_cli_writes_one_json_with_paired_delta_and_reference_changes(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    numerator_120 = [0, 1, 0, 1]
    numerator_360 = [1, 2, 1, 2]
    for frame, (value_120, value_360) in enumerate(zip(numerator_120, numerator_360)):
        for fov, numerator in ((120.0, value_120), (360.0, value_360)):
            rows.append(
                {
                    "frame_idx": frame,
                    "z_mode": "ground_anchored",
                    "fov_deg": fov,
                    "method": "exact",
                    "numerator": numerator,
                    "denominator": 2,
                }
            )
    input_path = tmp_path / "contributions.parquet"
    output_path = tmp_path / "result" / "block_sensitivity.json"
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--block-seconds",
            "0.1",
            "0.2",
            "0.5",
            "--reference-block-seconds",
            "0.2",
            "--n-resamples",
            "32",
            "--acf-max-lag",
            "3",
            "--seed",
            "7",
        ]
    )

    result = json.loads(output_path.read_text())
    assert result["bootstrap"]["overlong_block_policy"].startswith("skip and report")
    assert result["bootstrap"]["reference_block_seconds"] == pytest.approx(0.2)
    paired = result["paired_fov_deltas"][0]
    assert paired["delta_direction"] == "360 - 120"
    assert paired["point_estimate_delta"] == pytest.approx(0.5)
    assert paired["blocks"][1]["estimate_change_vs_reference"] == pytest.approx(0.0)
    assert paired["blocks"][2]["status"] == "skipped"
    assert list(output_path.parent.iterdir()) == [output_path]
