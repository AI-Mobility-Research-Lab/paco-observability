from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_cacie_full_results_figure.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "generate_cacie_full_results_figure_cli", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)

FigureInputError = _SCRIPT.FigureInputError
create_figure = _SCRIPT.create_figure
main = _SCRIPT.main
prepare_figure_data = _SCRIPT.prepare_figure_data
write_figure = _SCRIPT.write_figure


METHODS = ("legacy_planar", "center_top", "sparse_multiray")
FOVS = (120.0, 360.0)
NUMERATOR_COLUMNS = {
    "legacy_planar": "legacy_visible_count",
    "center_top": "center_top_visible_count",
    "sparse_multiray": "sparse_multiray_visible_count",
}
NUMERATORS = {
    120.0: {"legacy_planar": 3_000, "center_top": 3_600, "sparse_multiray": 4_200},
    360.0: {"legacy_planar": 4_200, "center_top": 4_800, "sparse_multiray": 5_400},
}
DENOMINATOR = 6_000


def _triplet(numerator: int, denominator: int) -> dict[str, int | float]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": numerator / denominator,
    }


def _full_summary() -> dict:
    groups = []
    for fov in FOVS:
        counts = {"total_count": DENOMINATOR}
        for method, numerator in NUMERATORS[fov].items():
            counts[NUMERATOR_COLUMNS[method]] = numerator
        groups.append(
            {
                "z_mode": "ground_anchored",
                "fov_deg": fov,
                "counts": counts,
                "observability_ratio_of_sums": {
                    method: numerator / DENOMINATOR
                    for method, numerator in NUMERATORS[fov].items()
                },
            }
        )
    return {
        "schema_version": "1.0",
        "config": {
            "sparse_min_visible_rays": 3,
            "method_definitions": {
                "legacy_planar": "planar angular wedge",
                "center_top": "center and top-center rays",
                "sparse_multiray": "visible when at least 3 of 15 rays are clear",
            },
            "fov_degrees": list(FOVS),
            "z_modes": ["ground_anchored"],
        },
        "frame_step": 1,
        "ego_positions": 60,
        "selected_frame_count": 1_000,
        "processed_frame_count": 1_000,
        "input": {"quality_gate_passed": True},
        "groups": groups,
    }


def _primary_bootstrap() -> dict:
    groups = []
    for fov in FOVS:
        for method in METHODS:
            estimate = NUMERATORS[fov][method] / DENOMINATOR
            groups.append(
                {
                    "z_mode": "ground_anchored",
                    "fov_deg": fov,
                    "method": method,
                    "n_frames": 1_000,
                    "numerator_sum": NUMERATORS[fov][method],
                    "denominator_sum": DENOMINATOR,
                    "estimate": estimate,
                    "ci_low": estimate - 0.02,
                    "ci_high": estimate + 0.02,
                }
            )
    return {
        "analysis": "frame_moving_block_bootstrap",
        "estimand": {
            "name": "ratio_of_sums",
            "formula": "sum(numerator) / sum(denominator)",
        },
        "bootstrap": {
            "n_resamples": 5_000,
            "block_seconds": 120.0,
            "block_frame_count": 1_200,
            "frame_rate_hz": 10.0,
            "confidence_level": 0.95,
            "circular": True,
            "method": "circular moving block",
            "unit": "frame",
        },
        "groups": groups,
    }


def _residual_inputs() -> tuple[dict, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    fov_results = []
    for fov in FOVS:
        scale = 1.0 if fov == 120.0 else 1.0 / 3.0
        fov_rows = []
        for index in range(60):
            rank = index + 1
            residual = (59.5 - index) * scale
            estimate = residual / 100.0
            row = {
                "demand_mode": "uniform",
                "analysis_role": "primary",
                "fov_deg": fov,
                "z_mode": "ground_anchored",
                "method": "sparse_multiray",
                "ego_id": f"ego_{index:02d}",
                "approach": ("east", "north", "south", "west")[index // 15],
                "ego_x": float((index % 15) * 2),
                "ego_y": float((index // 15) * 8),
                "point_rank": rank,
                "point_estimate": estimate,
                "residual_demand_sum": residual,
                "total_demand_sum": 100.0,
                "score_ci_low": max(0.0, estimate - 0.01),
                "score_ci_high": min(1.0, estimate + 0.01),
                "median_rank": float(rank),
                "rank_ci_low": float(max(1, rank - 2)),
                "rank_ci_high": float(min(60, rank + 2)),
                "top_k_probability": 0.9 if rank <= 10 else 0.1,
                "valid_fraction": 1.0,
                "time_block_count": 10,
                "block_width_frames": 1_200,
                "top_k": 10,
                "n_bootstrap": 5_000,
                "seed": 23,
            }
            rows.append(row)
            fov_rows.append(dict(row))
        fov_results.append(
            {
                "demand_mode": "uniform",
                "analysis_role": "primary",
                "fov_deg": fov,
                "n_ego_positions": 60,
                "n_time_blocks": 10,
                "time_blocks": [{"time_block_id": value} for value in range(10)],
                "ranking": fov_rows,
            }
        )
    summary = {
        "schema_version": "1.0",
        "analysis": "residual_demand_hotspot_rank_stability",
        "scope": {
            "z_mode": "ground_anchored",
            "method": "sparse_multiray",
            "fov_degrees": list(FOVS),
            "primary_demand_mode": "uniform",
            "sensitivity_demand_mode": "vru_weighted",
        },
        "score": {
            "name": "residual_demand_ratio_of_sums",
            "formulas": {
                "uniform": (
                    "sum(total_count - sparse_multiray_visible_count) / sum(total_count)"
                )
            },
            "higher_is_hotter": True,
        },
        "bootstrap": {
            "block_width_frames": 1_200,
            "n_resamples": 5_000,
            "confidence_level": 0.95,
            "top_k": 10,
        },
        "expected_ego_count": 60,
        "fov_results": fov_results,
        "interpretation_guardrail": (
            "Detected dynamic-box residual-demand hotspot only; this is not crash risk and an "
            "ego-grid coordinate is not a deployment-optimal pole or surveyed site."
        ),
    }
    return summary, pd.DataFrame(rows)


def _sensitivity_summary() -> dict:
    z_rows = []
    decimation_rows = []
    partial_rows = []
    primary_bootstrap = {
        (row["fov_deg"], row["method"]): row for row in _primary_bootstrap()["groups"]
    }
    for fov in FOVS:
        fov_step = 0.01 if fov == 120.0 else -0.01
        for method in METHODS:
            full = _triplet(NUMERATORS[fov][method], DENOMINATOR)
            step_numerator = int((float(full["ratio"]) + fov_step) * 1_000)
            step = _triplet(step_numerator, 1_000)
            raw = _triplet(step_numerator + 5, 1_000)
            z_delta = float(raw["ratio"]) - float(step["ratio"])
            z_rows.append(
                {
                    "fov_deg": fov,
                    "method": method,
                    "raw": raw,
                    "ground_anchored": step,
                    "delta_raw_minus_ground_anchored": z_delta,
                    "delta_percentage_points": z_delta * 100.0,
                    "bootstrap": {
                        "n_paired_replicates": 5_000,
                        "confidence_level": 0.95,
                        "ci_low": z_delta - 0.002,
                        "ci_high": z_delta + 0.002,
                        "ci_low_percentage_points": (z_delta - 0.002) * 100.0,
                        "ci_high_percentage_points": (z_delta + 0.002) * 100.0,
                    },
                }
            )
            decimation_delta = float(step["ratio"]) - float(full["ratio"])
            primary_ci = primary_bootstrap[(fov, method)]
            decimation_rows.append(
                {
                    "fov_deg": fov,
                    "method": method,
                    "step10_ground_anchored": step,
                    "full_primary_ground_anchored": full,
                    "delta_step10_minus_full_primary": decimation_delta,
                    "delta_percentage_points": decimation_delta * 100.0,
                    "full_primary_bootstrap_ci": {
                        "ci_low": primary_ci["ci_low"],
                        "ci_high": primary_ci["ci_high"],
                    },
                }
            )
            partial_numerator = int((float(full["ratio"]) - 0.1) * 100)
            partial_only = _triplet(partial_numerator, 100)
            included = _triplet(
                int(full["numerator"]) + partial_numerator,
                int(full["denominator"]) + 100,
            )
            partial_delta = float(included["ratio"]) - float(full["ratio"])
            partial_rows.append(
                {
                    "fov_deg": fov,
                    "method": method,
                    "primary_excluding_partial": full,
                    "partial_only": partial_only,
                    "include_partial": included,
                    "delta_include_partial_minus_primary": partial_delta,
                    "delta_percentage_points": partial_delta * 100.0,
                }
            )
    return {
        "schema_version": "1.0",
        "analysis": "cacie_frozen_sensitivity_summary",
        "contracts": {
            "sparse_min_visible_rays": 3,
            "methods": list(METHODS),
            "fov_degrees": list(FOVS),
            "primary_frame_step": 1,
            "z_mode_sensitivity_frame_step": 10,
            "point_estimand": "ratio_of_sums = sum(numerator) / sum(denominator)",
        },
        "z_mode_step10": {
            "delta_direction": "raw - ground_anchored",
            "bootstrap_replicate_ids": {
                "count": 5_000,
                "first": 0,
                "last": 4_999,
                "identical_across_all_groups": True,
            },
            "comparisons": z_rows,
        },
        "decimation_step10_vs_full": {
            "delta_direction": "step10 ground_anchored - full primary ground_anchored",
            "primary_bootstrap_status": "validated",
            "comparisons": decimation_rows,
        },
        "partial_scan_inclusion": {
            "delta_direction": "include partial scans - primary excluding partial scans",
            "comparisons": partial_rows,
        },
    }


def _block_sensitivity() -> dict:
    widths = {30.0: 0.030, 60.0: 0.035, 120.0: 0.040, 240.0: 0.050}
    groups = []
    for fov in FOVS:
        for method in METHODS:
            point = NUMERATORS[fov][method] / DENOMINATOR
            blocks = []
            for seconds, width in widths.items():
                blocks.append(
                    {
                        "block_seconds": seconds,
                        "block_frame_count": int(seconds * 10),
                        "status": "completed",
                        "estimate": point,
                        "ci_low": point - width / 2.0,
                        "ci_high": point + width / 2.0,
                        "ci_width": width,
                    }
                )
            groups.append(
                {
                    "z_mode": "ground_anchored",
                    "fov_deg": fov,
                    "method": method,
                    "n_frames": 1_000,
                    "numerator_sum": NUMERATORS[fov][method],
                    "denominator_sum": DENOMINATOR,
                    "point_estimate_ratio_of_sums": point,
                    "blocks": blocks,
                }
            )
    return {
        "schema_version": "1.0",
        "analysis": "moving_block_duration_sensitivity",
        "estimand": {"name": "ratio_of_sums"},
        "bootstrap": {
            "method": "circular moving block",
            "frame_rate_hz": 10.0,
            "block_durations": [
                {"seconds": seconds, "frame_count": int(seconds * 10)}
                for seconds in (30.0, 60.0, 120.0, 240.0)
            ],
            "reference_block_seconds": 120.0,
            "n_resamples": 5_000,
            "confidence_level": 0.95,
            "same_seed_for_all_durations": True,
        },
        "groups": groups,
    }


def _prepared_data() -> dict:
    residual_summary, residual_ranking = _residual_inputs()
    return prepare_figure_data(
        _full_summary(),
        _primary_bootstrap(),
        residual_summary,
        residual_ranking,
        _sensitivity_summary(),
        _block_sensitivity(),
    )


def test_prepare_data_recomputes_and_cross_validates_all_panels() -> None:
    data = _prepared_data()

    assert data["protocol"] == {
        "z_mode": "ground_anchored",
        "sparse_min_visible_rays": 3,
        "sparse_ray_count": 15,
        "bootstrap_resamples": 5_000,
        "primary_block_seconds": 120.0,
        "confidence_level": 0.95,
        "ego_positions": 60,
        "point_estimand": "ratio_of_sums",
    }
    assert data["primary"][(120.0, "sparse_multiray")]["estimate"] == pytest.approx(0.7)
    assert set(data["residual_by_fov"]) == set(FOVS)
    assert all(len(data["residual_by_fov"][fov]) == 60 for fov in FOVS)
    assert data["residual_by_fov"][120.0]["residual_demand_sum"].sum() == pytest.approx(
        1_800
    )
    assert len(data["sensitivity"]) == 18
    assert {row["category"] for row in data["sensitivity"]} == {
        "raw_minus_ground",
        "include_partial_minus_primary",
        "step10_minus_full",
    }
    assert data["block_stability"]["durations_seconds"] == [30.0, 60.0, 120.0, 240.0]
    assert data["block_stability"]["ci_width_pp"][(360.0, "center_top")][2] == (
        pytest.approx(4.0)
    )


def test_frozen_threshold_and_primary_bootstrap_contracts_fail_closed() -> None:
    residual_summary, residual_ranking = _residual_inputs()
    full = _full_summary()
    full["config"]["sparse_min_visible_rays"] = 1
    with pytest.raises(FigureInputError, match="threshold 3 of 15"):
        prepare_figure_data(
            full,
            _primary_bootstrap(),
            residual_summary,
            residual_ranking,
            _sensitivity_summary(),
            _block_sensitivity(),
        )

    bootstrap = _primary_bootstrap()
    bootstrap["bootstrap"]["n_resamples"] = 4_999
    with pytest.raises(FigureInputError, match="5000 resamples"):
        prepare_figure_data(
            _full_summary(),
            bootstrap,
            residual_summary,
            residual_ranking,
            _sensitivity_summary(),
            _block_sensitivity(),
        )


def test_bootstrap_count_mismatch_fails_before_plotting() -> None:
    residual_summary, residual_ranking = _residual_inputs()
    bootstrap = _primary_bootstrap()
    bootstrap["groups"][0]["numerator_sum"] += 1
    with pytest.raises(FigureInputError, match="numerator disagrees with full summary"):
        prepare_figure_data(
            _full_summary(),
            bootstrap,
            residual_summary,
            residual_ranking,
            _sensitivity_summary(),
            _block_sensitivity(),
        )


def test_residual_count_ratio_and_json_parquet_agreement_fail_closed() -> None:
    residual_summary, residual_ranking = _residual_inputs()
    broken_ratio = residual_ranking.copy()
    broken_ratio.loc[0, "point_estimate"] += 0.01
    with pytest.raises(FigureInputError, match="point estimates are not count ratios"):
        prepare_figure_data(
            _full_summary(),
            _primary_bootstrap(),
            residual_summary,
            broken_ratio,
            _sensitivity_summary(),
            _block_sensitivity(),
        )

    broken_summary = deepcopy(residual_summary)
    broken_summary["fov_results"][0]["ranking"][0]["ego_x"] += 1.0
    with pytest.raises(FigureInputError, match="JSON and Parquet disagree"):
        prepare_figure_data(
            _full_summary(),
            _primary_bootstrap(),
            broken_summary,
            residual_ranking,
            _sensitivity_summary(),
            _block_sensitivity(),
        )


def test_sensitivity_and_120_second_ci_cross_checks_fail_closed() -> None:
    residual_summary, residual_ranking = _residual_inputs()
    sensitivity = _sensitivity_summary()
    sensitivity["z_mode_step10"]["comparisons"][0]["delta_percentage_points"] += 1.0
    with pytest.raises(FigureInputError, match="delta is not count-derived"):
        prepare_figure_data(
            _full_summary(),
            _primary_bootstrap(),
            residual_summary,
            residual_ranking,
            sensitivity,
            _block_sensitivity(),
        )

    block = _block_sensitivity()
    reference = block["groups"][0]["blocks"][2]
    reference["ci_low"] -= 0.001
    reference["ci_width"] += 0.001
    with pytest.raises(FigureInputError, match="120-second CI disagrees"):
        prepare_figure_data(
            _full_summary(),
            _primary_bootstrap(),
            residual_summary,
            residual_ranking,
            _sensitivity_summary(),
            block,
        )


def test_publication_layout_guardrail_and_pdf_png_writer(tmp_path: Path) -> None:
    figure = create_figure(_prepared_data())
    try:
        assert len(figure.axes) == 6
        titles = " ".join(
            axis.get_title(loc=location)
            for axis in figure.axes
            for location in ("left", "center", "right")
        )
        assert "(a)" in titles and "(b)" in titles and "(c)" in titles
        assert "(d)" in titles and "(e)" in titles
        all_text = " ".join(text.get_text() for text in figure.texts).lower()
        assert "modeled ego-viewpoint" in all_text
        assert "not pole or site recommendations" in all_text
        figure.set_size_inches(2.4, 1.6)
        pdf_path, png_path = write_figure(figure, tmp_path / "full_results")
    finally:
        _SCRIPT.plt.close(figure)

    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert png_path.is_file() and png_path.stat().st_size > 0
    assert pdf_path.stem == png_path.stem == "full_results"


def test_cli_reads_all_six_sources_and_writes_same_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    residual_summary, residual_ranking = _residual_inputs()
    json_inputs = {
        "full": _full_summary(),
        "primary": _primary_bootstrap(),
        "residual": residual_summary,
        "sensitivity": _sensitivity_summary(),
        "blocks": _block_sensitivity(),
    }
    paths: dict[str, Path] = {}
    for name, value in json_inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    ranking_path = tmp_path / "residual.parquet"
    residual_ranking.to_parquet(ranking_path, index=False)

    def lightweight_writer(figure, output_base):
        del figure
        output_base.parent.mkdir(parents=True, exist_ok=True)
        pdf_path = output_base.with_suffix(".pdf")
        png_path = output_base.with_suffix(".png")
        pdf_path.write_bytes(b"pdf")
        png_path.write_bytes(b"png-600dpi")
        return pdf_path, png_path

    monkeypatch.setattr(_SCRIPT, "write_figure", lightweight_writer)
    output = tmp_path / "publication" / "cacie_full_results"
    assert (
        main(
            [
                "--full-summary",
                str(paths["full"]),
                "--primary-bootstrap",
                str(paths["primary"]),
                "--residual-summary",
                str(paths["residual"]),
                "--residual-ranking",
                str(ranking_path),
                "--sensitivity-summary",
                str(paths["sensitivity"]),
                "--block-sensitivity",
                str(paths["blocks"]),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert output.with_suffix(".pdf").read_bytes() == b"pdf"
    assert output.with_suffix(".png").read_bytes() == b"png-600dpi"
    lineage = json.loads(
        output.with_suffix(".source_lineage_receipt.json").read_text(encoding="utf-8")
    )
    assert lineage["status"] == "VERIFIED"
    assert lineage["derivation"] == "cacie_full_results_figure"
    assert set(lineage["outputs"]) == {"pdf", "png"}
    assert all(not Path(record["path"]).is_absolute() for record in lineage["inputs"].values())


def test_output_stem_must_not_have_a_suffix(tmp_path: Path) -> None:
    figure = create_figure(_prepared_data())
    try:
        with pytest.raises(FigureInputError, match="must have no suffix"):
            write_figure(figure, tmp_path / "bad.pdf")
    finally:
        _SCRIPT.plt.close(figure)
