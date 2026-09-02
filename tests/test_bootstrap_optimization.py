from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paco_observability.bootstrap import (
    moving_block_bootstrap_ratio,
    paired_moving_block_bootstrap_ratio,
)
from paco_observability.optimization import (
    greedy_weighted_maximum_coverage,
    summarize_hotspot_rank_stability,
)


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_observability.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("bootstrap_observability_cli", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
aggregate_frame_contributions = _SCRIPT.aggregate_frame_contributions
bootstrap_observability_table = _SCRIPT.bootstrap_observability_table
main = _SCRIPT.main


def test_moving_block_seed_is_reproducible() -> None:
    numerator = np.array([0, 0, 1, 1, 0, 1, 1, 1], dtype=float)
    denominator = np.ones_like(numerator)
    first = moving_block_bootstrap_ratio(
        numerator,
        denominator,
        block_length=3,
        n_resamples=128,
        seed=17,
    )
    repeated = moving_block_bootstrap_ratio(
        numerator,
        denominator,
        block_length=3,
        n_resamples=128,
        seed=17,
    )
    other_seed = moving_block_bootstrap_ratio(
        numerator,
        denominator,
        block_length=3,
        n_resamples=128,
        seed=18,
    )

    np.testing.assert_array_equal(first.samples, repeated.samples)
    assert not np.array_equal(first.samples, other_seed.samples)


def test_paired_delta_is_fov_360_minus_fov_120() -> None:
    numerator_120 = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    numerator_360 = numerator_120 + 1.0
    denominator = np.full(6, 2.0)

    result = paired_moving_block_bootstrap_ratio(
        numerator_120,
        denominator,
        numerator_360,
        denominator,
        block_length=2,
        n_resamples=64,
        seed=9,
    )

    assert result.estimate_delta == pytest.approx(0.5)
    np.testing.assert_allclose(result.samples_delta, 0.5)
    np.testing.assert_allclose(result.samples_delta, result.samples_b - result.samples_a)


def test_ego_rows_are_clustered_by_frame_before_resampling() -> None:
    # Each frame contains two perfectly dependent ego rows. Frame-level
    # resampling can produce ratios 0, 1/2, or 1, but never 1/4 or 3/4.
    raw = pd.DataFrame(
        {
            "frame_idx": [10, 10, 11, 11],
            "z_mode": ["raw"] * 4,
            "fov_deg": [120.0] * 4,
            "method": ["exact"] * 4,
            "numerator": [1, 1, 0, 0],
            "denominator": [1, 1, 1, 1],
        }
    )
    aggregated = aggregate_frame_contributions(raw)

    assert aggregated["frame_idx"].tolist() == [10, 11]
    assert aggregated["numerator"].tolist() == [2, 0]
    assert aggregated["denominator"].tolist() == [2, 2]
    result = moving_block_bootstrap_ratio(
        aggregated["numerator"],
        aggregated["denominator"],
        block_length=1,
        n_resamples=512,
        seed=4,
    )
    observed = set(np.round(result.samples, decimals=12))
    assert observed == {0.0, 0.5, 1.0}
    assert 0.25 not in observed
    assert 0.75 not in observed


def test_group_estimate_is_ratio_of_sums_not_mean_of_frame_ratios() -> None:
    # Per-frame ratios are 1 and 0, whose unweighted mean is 0.5. The intended
    # full-record estimand is (1 + 0) / (1 + 9) = 0.1.
    raw = pd.DataFrame(
        {
            "frame_idx": [1, 2],
            "z_mode": ["raw", "raw"],
            "fov_deg": [120.0, 120.0],
            "method": ["exact", "exact"],
            "numerator": [1.0, 0.0],
            "denominator": [1.0, 9.0],
        }
    )
    summary, _ = bootstrap_observability_table(
        raw,
        frame_rate_hz=10.0,
        block_seconds=0.1,
        n_resamples=32,
        seed=3,
        paired_fovs=None,
    )

    assert summary["estimand"]["name"] == "ratio_of_sums"
    assert summary["groups"][0]["estimate"] == pytest.approx(0.1)
    assert summary["groups"][0]["estimate"] != pytest.approx(0.5)


def test_cli_writes_explicit_metadata_and_per_replicate_parquet(tmp_path) -> None:
    rows: list[dict[str, object]] = []
    for frame in range(4):
        for ego in range(2):
            rows.append(
                {
                    "frame_idx": frame,
                    "z_mode": "ground_anchored",
                    "fov_deg": 120.0,
                    "method": "exact",
                    "numerator": int(frame % 2 == 1),
                    "denominator": 1,
                    "ego": ego,
                }
            )
            rows.append(
                {
                    "frame_idx": frame,
                    "z_mode": "ground_anchored",
                    "fov_deg": 360.0,
                    "method": "exact",
                    "numerator": 1,
                    "denominator": 1,
                    "ego": ego,
                }
            )
    input_path = tmp_path / "ego_frames.parquet"
    out_dir = tmp_path / "bootstrap"
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    main(
        [
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--block-seconds",
            "0.2",
            "--n-resamples",
            "24",
            "--seed",
            "77",
        ]
    )

    summary = json.loads((out_dir / "observability_bootstrap.json").read_text())
    draws = pd.read_parquet(out_dir / "observability_bootstrap_replicates.parquet")
    assert summary["bootstrap"]["block_frame_count"] == 2
    assert summary["estimand"]["formula"] == "sum(numerator) / sum(denominator)"
    assert summary["paired_fov_contrast"]["delta_definition"] == "fov_b - fov_a"
    assert summary["paired_fov_contrast"]["delta_direction"] == "360 - 120"
    assert summary["paired_fov_deltas"][0]["estimate_delta_b_minus_a"] == pytest.approx(
        0.5
    )
    assert len(draws) == 3 * 24  # two single-FOV distributions plus one paired distribution
    paired = draws[draws["analysis_type"] == "paired_fov_delta"]
    assert paired["delta_direction"].unique().tolist() == ["360 - 120"]
    np.testing.assert_allclose(paired["delta_b_minus_a"], 0.5)


def test_weighted_maximum_coverage_ties_and_submodular_trajectory() -> None:
    # Rows 0 and 1 initially tie at gain 3, so row 0 wins by original index.
    # Once row 0 is selected, row 1 has marginal gain 1: the trajectory records
    # the expected diminishing return.
    coverage = np.array(
        [
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=bool,
    )
    result = greedy_weighted_maximum_coverage(
        coverage,
        weights=np.ones(4),
        budget=2,
        candidate_ids=("first", "second", "third"),
    )

    assert result.selected_indices == (0, 1)
    assert result.selected_ids == ("first", "second")
    assert result.marginal_gains == pytest.approx((3.0, 1.0))
    assert result.cumulative_gains == pytest.approx((3.0, 4.0))
    assert result.coverage_fraction == pytest.approx(1.0)
    assert result.trajectory[0].newly_covered_indices == (0, 1, 2)
    assert result.trajectory[1].newly_covered_indices == (3,)


def test_hotspot_rank_stability_uses_deterministic_ties_and_top_k_frequency() -> None:
    scores = np.array(
        [
            [0.9, 0.8, 0.1],
            [0.8, 0.9, 0.1],
            [0.9, 0.8, 0.1],
            [0.9, 0.8, 0.1],
        ]
    )
    result = summarize_hotspot_rank_stability(
        scores,
        point_scores=[0.85, 0.85, 0.1],
        hotspot_ids=("A", "B", "C"),
        top_k=1,
        seed=5,
    )

    assert result.ordered_hotspot_ids == ("A", "B", "C")
    np.testing.assert_allclose(result.top_k_probability, [0.75, 0.25, 0.0])
    np.testing.assert_allclose(result.median_rank, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.bootstrap_ranks[:, 2], [3, 3, 3, 3])
