from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rank_residual_demand.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("rank_residual_demand_cli", _SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)
aggregate_residual_demand_blocks = _SCRIPT.aggregate_residual_demand_blocks
rank_residual_demand = _SCRIPT.rank_residual_demand
main = _SCRIPT.main


def _row(
    frame: int,
    ego_id: str,
    *,
    fov: float = 120.0,
    total: int,
    visible: int,
    weighted_total: float | None = None,
    weighted_residual: float | None = None,
    z_mode: str = "ground_anchored",
) -> dict[str, object]:
    index = 0 if ego_id == "A" else 1
    total_weight = float(total if weighted_total is None else weighted_total)
    residual_weight = float(total - visible if weighted_residual is None else weighted_residual)
    return {
        "frame_idx": frame,
        "ego_id": ego_id,
        "approach": "north" if ego_id == "A" else "south",
        "ego_x": float(index),
        "ego_y": float(index + 10),
        "z_mode": z_mode,
        "fov_deg": fov,
        "total_count": total,
        "sparse_multiray_visible_count": visible,
        "sparse_multiray_residual_demand": residual_weight,
        "weighted_total_demand": total_weight,
    }


def test_original_frame_width_blocks_are_aggregated_before_bootstrap() -> None:
    data = pd.DataFrame(
        [
            _row(100, "A", total=10, visible=6, weighted_total=12, weighted_residual=5),
            _row(1299, "A", total=20, visible=15, weighted_total=25, weighted_residual=6),
            _row(1300, "A", total=30, visible=20, weighted_total=40, weighted_residual=15),
            # A non-primary vertical mode must not enter either demand mode.
            _row(100, "A", total=99, visible=0, z_mode="raw"),
        ]
    )
    blocks, metadata = aggregate_residual_demand_blocks(
        data,
        fov_degrees=(120.0,),
        block_width_frames=1200,
        expected_ego_count=1,
    )

    assert metadata["ego_id"].tolist() == ["A"]
    uniform = blocks[blocks["demand_mode"] == "uniform"].reset_index(drop=True)
    assert uniform["time_block_id"].tolist() == [0, 1]
    assert uniform["time_block_start_frame"].tolist() == [100, 1300]
    assert uniform["time_block_end_frame"].tolist() == [1299, 2499]
    assert uniform["residual_demand"].tolist() == [9, 10]
    assert uniform["total_demand"].tolist() == [30, 30]
    weighted = blocks[blocks["demand_mode"] == "vru_weighted"].reset_index(drop=True)
    assert weighted["residual_demand"].tolist() == [11, 15]
    assert weighted["total_demand"].tolist() == [37, 40]


def test_hotspot_score_is_ratio_of_sums_not_mean_of_block_ratios() -> None:
    # Ego A: block ratios 1 and 0, but ratio-of-sums is 1/(1+9)=0.1.
    # Ego B: ratio-of-sums is 2/(1+9)=0.2, so B must rank above A.
    data = pd.DataFrame(
        [
            _row(0, "A", total=1, visible=0),
            _row(10, "A", total=9, visible=9),
            _row(0, "B", total=1, visible=1),
            _row(10, "B", total=9, visible=7),
        ]
    )
    _, ranking = rank_residual_demand(
        data,
        fov_degrees=(120.0,),
        block_width_frames=10,
        n_bootstrap=64,
        top_k=1,
        seed=4,
        expected_ego_count=2,
    )

    uniform = ranking[ranking["demand_mode"] == "uniform"].set_index("ego_id")
    assert uniform.loc["A", "point_estimate"] == pytest.approx(0.1)
    assert uniform.loc["B", "point_estimate"] == pytest.approx(0.2)
    assert uniform.loc["B", "point_rank"] == 1
    assert uniform.loc["A", "point_rank"] == 2


def test_equal_scores_have_stable_ego_id_order_and_top_k_probability() -> None:
    # Deliberately place B first. The implementation sorts hotspot IDs, so the
    # exact tie is deterministically resolved as A before B in every draw.
    data = pd.DataFrame(
        [
            _row(0, "B", total=2, visible=1),
            _row(10, "B", total=2, visible=1),
            _row(0, "A", total=2, visible=1),
            _row(10, "A", total=2, visible=1),
        ]
    )
    _, ranking = rank_residual_demand(
        data,
        fov_degrees=(120.0,),
        block_width_frames=10,
        n_bootstrap=48,
        top_k=1,
        seed=8,
        expected_ego_count=2,
    )

    for demand_mode in ("uniform", "vru_weighted"):
        mode = ranking[ranking["demand_mode"] == demand_mode].set_index("ego_id")
        assert mode.loc["A", "point_rank"] == 1
        assert mode.loc["B", "point_rank"] == 2
        assert mode.loc["A", "top_k_probability"] == pytest.approx(1.0)
        assert mode.loc["B", "top_k_probability"] == pytest.approx(0.0)


def test_cli_writes_both_demand_modes_and_fovs_with_guardrail(
    tmp_path, stub_cacie_source_validator, monkeypatch
) -> None:
    rows: list[dict[str, object]] = []
    for fov in (120.0, 360.0):
        for frame in range(4):
            rows.extend(
                [
                    _row(
                        frame,
                        "A",
                        fov=fov,
                        total=2,
                        visible=1,
                        weighted_total=4,
                        weighted_residual=3,
                    ),
                    _row(
                        frame,
                        "B",
                        fov=fov,
                        total=2,
                        visible=2 if frame % 2 else 1,
                        weighted_total=4,
                        weighted_residual=0 if frame % 2 else 1,
                    ),
                ]
            )
    input_path = tmp_path / "full_record_observability.parquet"
    source_summary_path = tmp_path / "full_record_summary.json"
    out_dir = tmp_path / "ranks"
    pd.DataFrame(rows).to_parquet(input_path, index=False)
    source_summary_path.write_text("{}\n", encoding="utf-8")
    stub_cacie_source_validator(_SCRIPT)
    monkeypatch.setattr(_SCRIPT, "EXPECTED_EGO_COUNT", 2)

    main(
        [
            "--input",
            str(input_path),
            "--source-summary",
            str(source_summary_path),
            "--out-dir",
            str(out_dir),
            "--block-width-frames",
            "2",
            "--n-bootstrap",
            "32",
            "--top-k",
            "1",
            "--expected-ego-count",
            "2",
            "--seed",
            "19",
        ]
    )

    summary = json.loads((out_dir / "residual_hotspot_ranks.json").read_text())
    ranking = pd.read_parquet(out_dir / "residual_hotspot_ranks.parquet")
    assert summary["source_input_sha256"] == "c" * 64
    assert summary["source_contract"]["table_kind"] == "counts"
    assert summary["scope"]["primary_demand_mode"] == "uniform"
    assert summary["scope"]["sensitivity_demand_mode"] == "vru_weighted"
    assert len(summary["fov_results"]) == 4
    assert set(ranking["demand_mode"]) == {"uniform", "vru_weighted"}
    assert set(ranking["fov_deg"]) == {120.0, 360.0}
    assert len(ranking) == 8
    assert set(ranking["analysis_role"]) == {"primary", "sensitivity"}
    assert {
        "approach",
        "ego_x",
        "ego_y",
        "point_estimate",
        "score_ci_low",
        "score_ci_high",
        "median_rank",
        "rank_ci_low",
        "rank_ci_high",
        "top_k_probability",
        "valid_fraction",
    }.issubset(ranking.columns)
    guardrail = summary["interpretation_guardrail"].lower()
    assert "detected dynamic-box" in guardrail
    assert "not crash risk" in guardrail
    assert "not a deployment-optimal pole" in guardrail
