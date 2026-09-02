from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_validation_strata.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("summarize_validation_strata_cli", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)

confusion_metrics = _SCRIPT.confusion_metrics
summarize_validation_strata = _SCRIPT.summarize_validation_strata
main = _SCRIPT.main


def _synthetic_decisions() -> pd.DataFrame:
    ranges = [2.0, 9.99, 10.0, 19.99, 20.0, 25.0, 34.0, 35.0, 5.0, 15.0]
    frame_ids = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    exact = [True, False, True, False, True, False, True, False, True, False]
    legacy = [True, True, False, False, True, True, False, False, True, False]
    center_top = exact.copy()
    rows = []
    for index, (frame_idx, range_m) in enumerate(zip(frame_ids, ranges), start=1):
        rows.append(
            {
                "frame_idx": frame_idx,
                "ego_id": "north_0",
                "target_id": index,
                "class_name": "car" if index % 2 else "truck",
                "approach": "north" if frame_idx <= 2 else "south",
                "z_mode": "ground_anchored",
                "fov_deg": 120.0,
                "range_m": range_m,
                "covered": True,
                "exact_visible": exact[index - 1],
                "legacy_visible": legacy[index - 1],
                "center_top_visible": center_top[index - 1],
            }
        )
    return pd.DataFrame(rows)


def _synthetic_timings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["legacy_planar_scene", "center_top", "center_top"],
            "grid": [None, 2, 2],
            "z_mode": ["planar", "ground_anchored", "ground_anchored"],
            "fov_deg": [120.0, None, None],
            "elapsed_ms": [2.0, 1.0, 3.0],
            "hit_rays": [None, 2, 1],
        }
    )


def test_confusion_metrics_report_both_classes_and_balanced_accuracy() -> None:
    result = confusion_metrics(
        predicted_visible=[True, False, True, False],
        exact_visible=[True, True, False, False],
    )

    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 1, 1)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["f1_visible"] == pytest.approx(0.5)
    assert result["f1_hidden"] == pytest.approx(0.5)
    assert result["false_visible_rate_given_exact_hidden"] == pytest.approx(0.5)


def test_summary_uses_declared_range_bands_and_frame_density_quartiles() -> None:
    summary, metrics, timing_summary, blockers = summarize_validation_strata(
        _synthetic_decisions(), _synthetic_timings(), declared_sampled_frames=200
    )

    legacy_ranges = metrics.loc[
        metrics["stratum"].eq("range_band") & metrics["method"].eq("legacy_planar")
    ].set_index("stratum_value")
    assert legacy_ranges["n"].to_dict() == {"[1.5,10)": 3, "[10,20)": 3, "[20,35]": 4}

    legacy_density = metrics.loc[
        metrics["stratum"].eq("frame_density_quartile")
        & metrics["method"].eq("legacy_planar")
    ].set_index("stratum_value")
    assert legacy_density["n"].to_dict() == {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    assert summary["metric_definition"]["range_bands_m"] == [
        "[1.5,10)",
        "[10,20)",
        "[20,35]",
    ]
    assert summary["study_scope"]["declared_sampled_frames"] == 200
    assert summary["study_scope"]["observed_sampled_frames"] == 4
    assert "Descriptive validation on 200 sampled frames" in summary["study_scope"][
        "interpretation"
    ]
    assert "not physical visibility ground truth" in summary["study_scope"]["reference"]
    assert not timing_summary.empty
    assert blockers.empty


def test_blocker_ids_are_joined_to_target_class_within_frame() -> None:
    decisions = _synthetic_decisions()
    decisions["exact_occluded_by"] = None
    # Frame 2 target 3 is a car and blocks the exact-hidden target 2.
    decisions.loc[decisions["target_id"].eq(2), "exact_occluded_by"] = 3

    _, _, _, blockers = summarize_validation_strata(
        decisions, _synthetic_timings(), declared_sampled_frames=200
    )

    linked = blockers.loc[
        blockers["model"].eq("exact_reference") & blockers["blocker_class"].eq("car")
    ]
    assert linked["blocked_rows"].sum() == 1


def test_cli_writes_json_and_parquet_without_recomputing_geometry(tmp_path: Path) -> None:
    decisions_path = tmp_path / "validation_decisions.parquet"
    timings_path = tmp_path / "validation_timings.parquet"
    output_dir = tmp_path / "summary"
    _synthetic_decisions().to_parquet(decisions_path, index=False)
    _synthetic_timings().to_parquet(timings_path, index=False)

    assert (
        main(
            [
                "--decisions",
                str(decisions_path),
                "--timings",
                str(timings_path),
                "--out-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    written = pd.read_parquet(output_dir / "validation_strata_metrics.parquet")
    payload = json.loads((output_dir / "validation_strata_summary.json").read_text())
    assert not written.empty
    assert payload["covered_decision_rows"] == 10
    assert (output_dir / "validation_timing_summary.parquet").exists()
    written_blockers = pd.read_parquet(output_dir / "validation_blocker_classes.parquet")
    assert written_blockers.empty
    lineage = json.loads((output_dir / "source_lineage_receipt.json").read_text())
    assert lineage["status"] == "VERIFIED"
    assert lineage["derivation"] == "validation_strata_calibrated"
    assert set(lineage["inputs"]) == {
        "calibrated_decisions",
        "main_validation_timings",
    }
    assert set(lineage["outputs"]) == {"summary", "metrics", "timings", "blockers"}
    assert all(not Path(record["path"]).is_absolute() for record in lineage["inputs"].values())
