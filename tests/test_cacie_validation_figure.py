from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "generate_cacie_validation_figure.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "generate_cacie_validation_figure_cli", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)

FigureInputError = _SCRIPT.FigureInputError
prepare_figure_data = _SCRIPT.prepare_figure_data
create_figure = _SCRIPT.create_figure
write_figure = _SCRIPT.write_figure
main = _SCRIPT.main


def _validation_group(fov: float, grids: tuple[int, ...]) -> dict:
    primary = max(grids)
    return {
        "z_mode": "ground_anchored",
        "fov_deg": fov,
        "agreement_with_exact_among_covered": {
            "legacy_planar": {"accuracy": 0.80},
            "center_top": {"accuracy": 0.90},
            # Deliberately wrong/legacy value: the figure must replace this
            # with a 3/15 recomputation from decision fractions.
            "sparse_multiray": {"accuracy": 0.10},
        },
        "observability": {
            "legacy_planar": 0.40,
            "center_top": 0.45,
            "sparse_multiray": 0.95,
            "exact_reference": 0.50,
        },
        "grid_convergence": {
            str(grid): {
                "mean_absolute_fraction_difference_from_primary": (
                    0.0 if grid == primary else (primary - grid) / 100.0
                )
            }
            for grid in grids
        },
    }


def _main_summary() -> dict:
    return {
        "sampled_frame_count": 200,
        "groups": [
            _validation_group(120.0, (9, 17, 33)),
            _validation_group(360.0, (9, 17, 33)),
        ],
        "runtime": [
            {
                "method": "center_top",
                "grid": 2,
                "z_mode": "ground_anchored",
                "median_ms": 0.02,
            },
            {
                "method": "sparse_multiray",
                "grid": 15,
                "z_mode": "ground_anchored",
                "median_ms": 0.08,
            },
            {
                "method": "exact_angular_grid",
                "grid": 33,
                "z_mode": "ground_anchored",
                "median_ms": 1.7,
            },
        ],
    }


def _strata_summary() -> dict:
    metrics = []
    for fov in (120.0, 360.0):
        for category in ("bicycle", "car", "pedestrian", "truck"):
            metrics.append(
                {
                    "stratum": "target_class",
                    "stratum_value": category,
                    "z_mode": "ground_anchored",
                    "fov_deg": fov,
                    "method": "sparse_multiray",
                    "accuracy": 0.10,
                }
            )
        for category in ("[1.5,10)", "[10,20)", "[20,35]"):
            metrics.append(
                {
                    "stratum": "range_band",
                    "stratum_value": category,
                    "z_mode": "ground_anchored",
                    "fov_deg": fov,
                    "method": "sparse_multiray",
                    "accuracy": 0.10,
                }
            )
    return {
        "study_scope": {
            "declared_sampled_frames": 200,
            "observed_sampled_frames": 200,
            "sample_size_matches_declaration": True,
            "reference": (
                "Analytic ray--OBB visibility within the detected dynamic OBB abstraction; "
                "not physical visibility ground truth."
            ),
        },
        "metrics": metrics,
    }


def _decisions() -> pd.DataFrame:
    rows = []
    # The four repeating cases make the calibrated >=3/15 prediction equal to
    # the exact label, while an any-ray rule would fail on the 1/15 case.
    fractions = (1 / 15, 3 / 15, 0.0, 15 / 15)
    exact = (False, True, False, True)
    classes = ("bicycle", "car", "pedestrian", "truck")
    ranges = (5.0, 15.0, 25.0)
    for frame_idx in range(1, 201):
        case = (frame_idx - 1) % 4
        for fov in (120.0, 360.0):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "ego_id": "north_00",
                    "target_id": frame_idx,
                    "class_name": classes[case],
                    "z_mode": "ground_anchored",
                    "fov_deg": fov,
                    "range_m": ranges[(frame_idx - 1) % len(ranges)],
                    "covered": True,
                    "exact_visible": exact[case],
                    # Deliberately wrong on calibration frames and correct on
                    # held-out frames, so the primary panel reveals leakage.
                    "legacy_visible": (not exact[case] if frame_idx <= 100 else exact[case]),
                    "center_top_visible": exact[case],
                    "sparse_multiray_fraction": fractions[case],
                    "sparse_multiray_visible": fractions[case] > 0.0,
                }
            )
    return pd.DataFrame(rows)


def _calibration_summary() -> dict:
    return {
        "split": {
            "calibration_frames": list(range(1, 101)),
            "held_out_frames": list(range(101, 201)),
        },
        "selection": {"selected_min_visible_rays": 3},
    }


def _convergence_summary() -> dict:
    return {
        "sampled_frame_count": 25,
        "groups": [
            _validation_group(120.0, (17, 33, 65)),
            _validation_group(360.0, (17, 33, 65)),
        ],
    }


def test_calibrated_sparse_values_replace_legacy_any_ray_summary() -> None:
    data = prepare_figure_data(
        _main_summary(),
        _strata_summary(),
        _decisions(),
        _calibration_summary(),
        _convergence_summary(),
    )

    assert data["agreement"]["sparse_multiray"] == pytest.approx([1.0, 1.0])
    assert data["agreement"]["legacy_planar"] == pytest.approx([1.0, 1.0])
    assert data["sparse_rule"] == {
        "minimum_visible_rays": 3,
        "total_rays": 15,
        "fraction_threshold": pytest.approx(0.2),
        "source": "recomputed from main validation decisions",
    }
    assert data["convergence"]["grids"] == [17, 33, 65]
    assert data["convergence"]["reference_grid"] == 65
    assert data["convergence"]["source"] == "subset"


def test_main_summary_convergence_is_used_when_subset_summary_is_omitted() -> None:
    data = prepare_figure_data(
        _main_summary(), _strata_summary(), _decisions(), _calibration_summary()
    )

    assert data["convergence"]["grids"] == [9, 17, 33]
    assert data["convergence"]["reference_grid"] == 33
    assert data["convergence"]["source"] == "main"


def test_missing_runtime_and_incomplete_convergence_fail_clearly() -> None:
    missing_runtime = _main_summary()
    missing_runtime["runtime"] = missing_runtime["runtime"][:-1]
    with pytest.raises(FigureInputError, match="exact_angular_grid.*grid=33; found 0"):
        prepare_figure_data(
            missing_runtime, _strata_summary(), _decisions(), _calibration_summary()
        )

    wrong_sample_size = _convergence_summary()
    wrong_sample_size["sampled_frame_count"] = 24
    with pytest.raises(FigureInputError, match="sampled_frame_count must be 25"):
        prepare_figure_data(
            _main_summary(),
            _strata_summary(),
            _decisions(),
            _calibration_summary(),
            wrong_sample_size,
        )

    incomplete_convergence = _convergence_summary()
    del incomplete_convergence["groups"][0]["grid_convergence"]["65"]
    with pytest.raises(FigureInputError, match="grid_convergence.*65"):
        prepare_figure_data(
            _main_summary(),
            _strata_summary(),
            _decisions(),
            _calibration_summary(),
            incomplete_convergence,
        )


def test_decisions_without_sparse_fraction_fail_instead_of_using_old_label() -> None:
    decisions = _decisions().drop(columns="sparse_multiray_fraction")
    with pytest.raises(FigureInputError, match="sparse_multiray_fraction"):
        prepare_figure_data(_main_summary(), _strata_summary(), decisions, _calibration_summary())


def test_calibration_json_must_strictly_partition_frames_and_select_three() -> None:
    wrong_selection = _calibration_summary()
    wrong_selection["selection"]["selected_min_visible_rays"] = 1
    with pytest.raises(FigureInputError, match="selected_min_visible_rays=3"):
        prepare_figure_data(_main_summary(), _strata_summary(), _decisions(), wrong_selection)

    overlap = _calibration_summary()
    overlap["split"]["held_out_frames"][0] = 100
    with pytest.raises(FigureInputError, match="must be disjoint"):
        prepare_figure_data(_main_summary(), _strata_summary(), _decisions(), overlap)


def test_publication_layout_and_pdf_png_writer(tmp_path: Path) -> None:
    data = prepare_figure_data(
        _main_summary(),
        _strata_summary(),
        _decisions(),
        _calibration_summary(),
        _convergence_summary(),
    )
    figure = create_figure(data)
    try:
        assert len(figure.axes) == 6
        assert "dynamic-box abstraction" in _SCRIPT.REFERENCE_SCOPE_LABEL
        # Use a compact figure for an inexpensive real PDF/PNG export test.
        figure.set_size_inches(2.0, 1.2)
        pdf_path, png_path = write_figure(figure, tmp_path / "validation")
    finally:
        _SCRIPT.plt.close(figure)

    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0


def test_cli_reads_all_sources_and_writes_both_formats(tmp_path: Path, monkeypatch) -> None:
    summary_path = tmp_path / "main.json"
    strata_path = tmp_path / "strata.json"
    convergence_path = tmp_path / "convergence.json"
    calibration_path = tmp_path / "calibration.json"
    decisions_path = tmp_path / "decisions.parquet"
    summary_path.write_text(json.dumps(_main_summary()), encoding="utf-8")
    strata_path.write_text(json.dumps(_strata_summary()), encoding="utf-8")
    convergence_path.write_text(json.dumps(_convergence_summary()), encoding="utf-8")
    calibration_path.write_text(json.dumps(_calibration_summary()), encoding="utf-8")
    _decisions().to_parquet(decisions_path, index=False)

    def lightweight_writer(figure, output_base):
        del figure
        pdf_path = output_base.with_suffix(".pdf")
        png_path = output_base.with_suffix(".png")
        pdf_path.write_bytes(b"pdf")
        png_path.write_bytes(b"png")
        return pdf_path, png_path

    monkeypatch.setattr(_SCRIPT, "write_figure", lightweight_writer)
    output = tmp_path / "figure"
    assert (
        main(
            [
                "--summary",
                str(summary_path),
                "--strata-summary",
                str(strata_path),
                "--decisions",
                str(decisions_path),
                "--calibration-summary",
                str(calibration_path),
                "--convergence-summary",
                str(convergence_path),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert output.with_suffix(".pdf").read_bytes() == b"pdf"
    assert output.with_suffix(".png").read_bytes() == b"png"
    lineage = json.loads(
        output.with_suffix(".source_lineage_receipt.json").read_text(encoding="utf-8")
    )
    assert lineage["status"] == "VERIFIED"
    assert lineage["derivation"] == "cacie_validation_figure"
    assert set(lineage["inputs"]) == {
        "main_summary",
        "strata_summary",
        "main_decisions",
        "calibration_summary",
        "convergence_summary",
    }
    assert set(lineage["outputs"]) == {"pdf", "png"}
    assert all(not Path(record["path"]).is_absolute() for record in lineage["inputs"].values())


def test_missing_json_path_reports_the_source_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="main validation summary"):
        main(
            [
                "--summary",
                str(tmp_path / "missing.json"),
                "--strata-summary",
                str(tmp_path / "also-missing.json"),
                "--decisions",
                str(tmp_path / "missing.parquet"),
                "--out",
                str(tmp_path / "figure"),
            ]
        )
