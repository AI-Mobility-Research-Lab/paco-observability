from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_cacie_sensitivities.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "summarize_cacie_sensitivities_cli", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)

EXPECTED_PARTIAL_FRAMES = _SCRIPT.EXPECTED_PARTIAL_FRAMES
EXPECTED_OBSERVED_PARTIAL_FRAMES = _SCRIPT.EXPECTED_OBSERVED_PARTIAL_FRAMES
METHOD_NUMERATOR_COLUMNS = _SCRIPT.METHOD_NUMERATOR_COLUMNS
main = _SCRIPT.main
summarize_cacie_sensitivities = _SCRIPT.summarize_cacie_sensitivities


_METHOD_DEFINITIONS = {
    "legacy_planar": "planar reference",
    "center_top": "center and top-center rays",
    "sparse_multiray": "visible when at least 3 of 15 rays are clear",
}
_PRIMARY_NUMERATORS = {
    120.0: {"legacy_planar": 40, "center_top": 45, "sparse_multiray": 50},
    360.0: {"legacy_planar": 70, "center_top": 75, "sparse_multiray": 80},
}
_STEP10_NUMERATORS = {
    ("ground_anchored", 120.0): {
        "legacy_planar": 8,
        "center_top": 9,
        "sparse_multiray": 12,
    },
    ("ground_anchored", 360.0): {
        "legacy_planar": 14,
        "center_top": 15,
        "sparse_multiray": 16,
    },
    ("raw", 120.0): {
        "legacy_planar": 9,
        "center_top": 8,
        "sparse_multiray": 13,
    },
    ("raw", 360.0): {
        "legacy_planar": 15,
        "center_top": 14,
        "sparse_multiray": 17,
    },
}
_PARTIAL_NUMERATORS = {
    120.0: {"legacy_planar": 5, "center_top": 6, "sparse_multiray": 7},
    360.0: {"legacy_planar": 7, "center_top": 8, "sparse_multiray": 9},
}


def _selection(*, partial_only: bool) -> dict[str, object]:
    empty = {
        "requested": [],
        "observed": [],
        "absent_from_track_table": [],
    }
    if partial_only:
        only_frames = {
            "requested": list(EXPECTED_PARTIAL_FRAMES),
            "observed": list(EXPECTED_OBSERVED_PARTIAL_FRAMES),
            "absent_from_track_table": [6747],
        }
        exclusions = {
            "requested": [],
            "observed_and_removed": [],
            "absent_from_track_table": [],
        }
    else:
        only_frames = empty
        exclusions = {
            "requested": list(EXPECTED_PARTIAL_FRAMES),
            "observed_and_removed": list(EXPECTED_OBSERVED_PARTIAL_FRAMES),
            "absent_from_track_table": [6747],
        }
    return {
        "quality_gate_passed": True,
        "only_frames": only_frames,
        "quality_excluded_frames": exclusions,
    }


def _group(
    z_mode: str,
    fov: float,
    denominator: int,
    numerators: dict[str, int],
) -> dict[str, object]:
    counts: dict[str, int] = {"total_count": denominator}
    for method, column in METHOD_NUMERATOR_COLUMNS.items():
        counts[column] = numerators[method]
    return {
        "z_mode": z_mode,
        "fov_deg": fov,
        "counts": counts,
        "observability_ratio_of_sums": {
            method: numerator / denominator for method, numerator in numerators.items()
        },
    }


def _summary(role: str) -> dict[str, object]:
    if role == "primary":
        z_modes = ["ground_anchored"]
        frame_step = 1
        selected = 1000
        groups = [
            _group("ground_anchored", fov, 100, _PRIMARY_NUMERATORS[fov]) for fov in (120.0, 360.0)
        ]
    elif role == "step10":
        z_modes = ["ground_anchored", "raw"]
        frame_step = 10
        selected = 100
        groups = [
            _group(z_mode, fov, 20, _STEP10_NUMERATORS[(z_mode, fov)])
            for z_mode in z_modes
            for fov in (120.0, 360.0)
        ]
    elif role == "partial":
        z_modes = ["ground_anchored"]
        frame_step = 1
        selected = 15
        groups = [
            _group("ground_anchored", fov, 10, _PARTIAL_NUMERATORS[fov]) for fov in (120.0, 360.0)
        ]
    else:
        raise AssertionError(role)
    return {
        "config": {
            "sparse_min_visible_rays": 3,
            "method_definitions": dict(_METHOD_DEFINITIONS),
            "z_modes": z_modes,
            "fov_degrees": [120.0, 360.0],
        },
        "frame_step": frame_step,
        "selected_frame_count": selected,
        "processed_frame_count": selected,
        "input": _selection(partial_only=role == "partial"),
        "groups": groups,
    }


def _z_replicates() -> tuple[pd.DataFrame, np.ndarray]:
    deltas = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(METHOD_NUMERATOR_COLUMNS):
        for fov_index, fov in enumerate((120.0, 360.0)):
            ground = 0.35 + 0.05 * method_index + 0.1 * fov_index
            for replicate, delta in enumerate(deltas):
                rows.extend(
                    [
                        {
                            "analysis_type": "single_ratio",
                            "estimand": "ratio_of_sums",
                            "z_mode": "ground_anchored",
                            "method": method,
                            "fov_deg": fov,
                            "replicate": replicate,
                            "ratio": ground + 0.005 * replicate,
                            "seed": 23,
                            "block_frame_count": 12,
                        },
                        {
                            "analysis_type": "single_ratio",
                            "estimand": "ratio_of_sums",
                            "z_mode": "raw",
                            "method": method,
                            "fov_deg": fov,
                            "replicate": replicate,
                            "ratio": ground + 0.005 * replicate + delta,
                            "seed": 23,
                            "block_frame_count": 12,
                        },
                    ]
                )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7), deltas


def _primary_bootstrap() -> dict[str, object]:
    rows = []
    for fov, numerators in _PRIMARY_NUMERATORS.items():
        for method, numerator in numerators.items():
            estimate = numerator / 100
            rows.append(
                {
                    "z_mode": "ground_anchored",
                    "fov_deg": fov,
                    "method": method,
                    "numerator_sum": numerator,
                    "denominator_sum": 100,
                    "estimate": estimate,
                    "ci_low": max(0.0, estimate - 0.05),
                    "ci_high": min(1.0, estimate + 0.05),
                }
            )
    return {
        "estimand": {"name": "ratio_of_sums"},
        "bootstrap": {"circular": True, "block_seconds": 120.0},
        "groups": rows,
    }


def _comparison(
    result: dict[str, object], section: str, fov: float, method: str
) -> dict[str, object]:
    section_value = result[section]
    assert isinstance(section_value, dict)
    comparisons = section_value["comparisons"]
    assert isinstance(comparisons, list)
    return next(row for row in comparisons if row["fov_deg"] == fov and row["method"] == method)


def test_summary_reconstructs_all_point_estimates_and_pairs_draws_by_id() -> None:
    replicates, deltas = _z_replicates()

    result = summarize_cacie_sensitivities(
        _summary("primary"),
        _summary("step10"),
        replicates,
        _summary("partial"),
        primary_bootstrap=_primary_bootstrap(),
    )

    assert result["analysis"] == "cacie_frozen_sensitivity_summary"
    z_row = _comparison(result, "z_mode_step10", 120.0, "sparse_multiray")
    assert z_row["delta_raw_minus_ground_anchored"] == pytest.approx(13 / 20 - 12 / 20)
    expected_ci = np.quantile(deltas, [0.025, 0.975])
    assert z_row["bootstrap"]["ci_low"] == pytest.approx(expected_ci[0])
    assert z_row["bootstrap"]["ci_high"] == pytest.approx(expected_ci[1])
    assert z_row["bootstrap"]["n_paired_replicates"] == len(deltas)

    decimation = _comparison(result, "decimation_step10_vs_full", 120.0, "sparse_multiray")
    assert decimation["delta_step10_minus_full_primary"] == pytest.approx(12 / 20 - 50 / 100)
    assert decimation["full_primary_bootstrap_ci"] == {
        "ci_low": pytest.approx(0.45),
        "ci_high": pytest.approx(0.55),
    }

    partial = _comparison(result, "partial_scan_inclusion", 120.0, "sparse_multiray")
    assert partial["include_partial"] == {
        "numerator": 57,
        "denominator": 110,
        "ratio": pytest.approx(57 / 110),
    }
    assert partial["delta_include_partial_minus_primary"] == pytest.approx(57 / 110 - 0.5)
    assert result["contracts"]["partial_frames_absent"] == [6747]


def test_replicate_collections_must_match_exactly_across_every_group() -> None:
    replicates, _ = _z_replicates()
    missing = replicates.loc[
        ~(
            replicates["z_mode"].eq("raw")
            & replicates["method"].eq("sparse_multiray")
            & replicates["fov_deg"].eq(360.0)
            & replicates["replicate"].eq(4)
        )
    ]

    with pytest.raises(ValueError, match="replicate ID collections must be identical"):
        summarize_cacie_sensitivities(
            _summary("primary"),
            _summary("step10"),
            missing,
            _summary("partial"),
        )


def test_calibrated_rule_method_and_partial_selection_contracts_fail_closed() -> None:
    replicates, _ = _z_replicates()
    wrong_rule = _summary("primary")
    wrong_rule["config"]["sparse_min_visible_rays"] = 2
    with pytest.raises(ValueError, match="3-of-15"):
        summarize_cacie_sensitivities(
            wrong_rule,
            _summary("step10"),
            replicates,
            _summary("partial"),
        )

    wrong_partial = _summary("partial")
    wrong_partial["input"]["only_frames"]["observed"].pop()
    with pytest.raises(ValueError, match="exactly the 15 observed partial frames"):
        summarize_cacie_sensitivities(
            _summary("primary"),
            _summary("step10"),
            replicates,
            wrong_partial,
        )

    wrong_exclusions = _summary("primary")
    wrong_exclusions["input"]["quality_excluded_frames"]["requested"].pop()
    with pytest.raises(ValueError, match="all 16 declared partial-frame exclusions"):
        summarize_cacie_sensitivities(
            wrong_exclusions,
            _summary("step10"),
            replicates,
            _summary("partial"),
        )

    wrong_methods = _summary("partial")
    wrong_methods["config"]["method_definitions"].pop("center_top")
    with pytest.raises(ValueError, match="method definitions must be exactly"):
        summarize_cacie_sensitivities(
            _summary("primary"),
            _summary("step10"),
            replicates,
            wrong_methods,
        )


def test_fov_sets_and_same_frame_denominators_must_match() -> None:
    replicates, _ = _z_replicates()
    partial_one_fov = _summary("partial")
    partial_one_fov["config"]["fov_degrees"] = [120.0]
    partial_one_fov["groups"] = [
        row for row in partial_one_fov["groups"] if row["fov_deg"] == 120.0
    ]
    with pytest.raises(ValueError, match="identical FOV sets"):
        summarize_cacie_sensitivities(
            _summary("primary"),
            _summary("step10"),
            replicates,
            partial_one_fov,
        )

    unequal_rows = _summary("step10")
    raw_120 = next(
        row for row in unequal_rows["groups"] if row["z_mode"] == "raw" and row["fov_deg"] == 120.0
    )
    raw_120["counts"]["total_count"] = 21
    raw_120["observability_ratio_of_sums"] = {
        method: raw_120["counts"][column] / 21
        for method, column in METHOD_NUMERATOR_COLUMNS.items()
    }
    with pytest.raises(ValueError, match="must describe the same rows"):
        summarize_cacie_sensitivities(
            _summary("primary"),
            unequal_rows,
            replicates,
            _summary("partial"),
        )


def test_cli_writes_json_with_input_receipts_using_only_synthetic_inputs(
    tmp_path: Path,
) -> None:
    primary_path = tmp_path / "primary.json"
    z_path = tmp_path / "step10.json"
    partial_path = tmp_path / "partial.json"
    replicates_path = tmp_path / "replicates.parquet"
    output_path = tmp_path / "sensitivity.json"
    for path, payload in (
        (primary_path, _summary("primary")),
        (z_path, _summary("step10")),
        (partial_path, _summary("partial")),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    replicates, _ = _z_replicates()
    replicates.to_parquet(replicates_path, index=False)

    assert (
        main(
            [
                "--primary-summary",
                str(primary_path),
                "--z-summary",
                str(z_path),
                "--z-bootstrap-replicates",
                str(replicates_path),
                "--partial-summary",
                str(partial_path),
                "--out",
                str(output_path),
            ]
        )
        == 0
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["decimation_step10_vs_full"]["primary_bootstrap_status"] == "not supplied"
    assert written["inputs"]["primary_summary"]["sha256"]
    assert written["inputs"]["primary_bootstrap"] is None
    assert len(written["z_mode_step10"]["comparisons"]) == 6
