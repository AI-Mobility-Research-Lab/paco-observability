from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import paco_observability.cacie_source_contract as contract_module
from paco_observability.cacie_source_contract import (
    CacieSourceContract,
    attach_source_contract,
    require_cacie_full_record_source_unchanged,
    require_effective_frame_rate,
    snapshot_cacie_full_record_source,
    validate_cacie_full_record_source,
)


def _patch_small_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract_module, "EXPECTED_FRAME_RANGE", (1, 4))
    monkeypatch.setattr(contract_module, "EXPECTED_ALLOWED_MISSING_FRAMES", (3,))
    monkeypatch.setattr(contract_module, "EXPECTED_QUALITY_EXCLUDED_FRAMES", (2, 3))
    monkeypatch.setattr(contract_module, "EXPECTED_OBSERVED_EXCLUSIONS", (2,))


def _counts() -> pd.DataFrame:
    rows = []
    # Ego B is legitimately absent at frame 4, representing one excluded
    # ego-scene. Every analysis group must nevertheless contain both frames.
    for frame, frame_order, ego_ids in ((1, 0, ("A", "B")), (4, 1, ("A",))):
        for ego_id in ego_ids:
            for fov in (120.0, 360.0):
                rows.append(
                    {
                        "frame_idx": frame,
                        "frame_order": frame_order,
                        "ego_id": ego_id,
                        "z_mode": "ground_anchored",
                        "fov_deg": fov,
                    }
                )
    return pd.DataFrame(rows)


def _contributions() -> pd.DataFrame:
    rows = []
    for record in _counts().to_dict(orient="records"):
        for method in contract_module.EXPECTED_METHODS:
            rows.append({**record, "method": method, "numerator": 1, "denominator": 1})
    return pd.DataFrame(rows)


def _summary() -> dict[str, object]:
    input_sha = "d" * 64
    return {
        "schema_version": "1.0",
        "processed_frame_count": 2,
        "selected_frame_count": 2,
        "processed_frame_min": 1,
        "processed_frame_max": 4,
        "input_observed_frame_count": 3,
        "frame_step": 1,
        "max_frames": None,
        "ego_positions": 2,
        "ego_scenes_requested": 4,
        "ego_scenes_excluded": 1,
        "ego_scenes_evaluated": 3,
        "wide_count_rows": 6,
        "exclusions": {"total": 1},
        "input": {
            "sha256": input_sha,
            "quality_gate_passed": True,
            "expected_frame_range": [1, 4],
            "allowed_missing_frames": [3],
            "quality_excluded_frames": {
                "requested": [2, 3],
                "observed_and_removed": [2],
                "absent_from_track_table": [3],
                "reason": contract_module.EXPECTED_EXCLUSION_REASON,
            },
            "only_frames": {
                "requested": [],
                "observed": [],
                "absent_from_track_table": [],
            },
        },
        "config": {
            "sparse_min_visible_rays": 3,
            "method_definitions": {
                "legacy_planar": "legacy",
                "center_top": "center top",
                "sparse_multiray": "at least 3 of 15 rays clear",
            },
            "fov_degrees": [120.0, 360.0],
            "z_modes": ["ground_anchored"],
        },
        "outputs": {
            "counts": {"file": "full_record_observability.parquet", "rows": 6},
            "contributions": {
                "file": "observability_contributions.parquet",
                "rows": 18,
            },
            "exclusions": {"file": "ego_exclusions.parquet", "rows": 1},
            "timings": {"file": "full_record_timings.parquet", "rows": 2},
        },
        "groups": [
            {"z_mode": "ground_anchored", "fov_deg": 120.0, "ego_frame_rows": 3},
            {"z_mode": "ground_anchored", "fov_deg": 360.0, "ego_frame_rows": 3},
        ],
    }


def _write_source(
    tmp_path: Path,
    *,
    table_kind: str,
    data: pd.DataFrame | None = None,
    summary: dict[str, object] | None = None,
) -> tuple[Path, Path, pd.DataFrame]:
    table = (data if data is not None else (_counts() if table_kind == "counts" else _contributions()))
    source_summary = summary if summary is not None else _summary()
    output = source_summary["outputs"][table_kind]  # type: ignore[index]
    input_path = tmp_path / output["file"]  # type: ignore[index]
    summary_path = tmp_path / "full_record_summary.json"
    metadata = {
        b"paco.schema_version": b"1.0",
        b"paco.input_sha256": ("d" * 64).encode(),
        b"paco.expected_frame_range": b"[1, 4]",
        b"paco.allowed_missing_frames": b"[3]",
        b"paco.quality_excluded_frames": b"[2, 3]",
        b"paco.only_frames_requested": b"[]",
        b"paco.frame_exclusion_reason": contract_module.EXPECTED_EXCLUSION_REASON.encode(),
        b"paco.sparse_min_visible_rays": b"3",
    }
    arrow_table = pa.Table.from_pandas(table, preserve_index=False).replace_schema_metadata(metadata)
    pq.write_table(arrow_table, input_path)
    summary_path.write_text(
        json.dumps(source_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return input_path, summary_path, table


@pytest.mark.parametrize("table_kind", ["counts", "contributions"])
def test_contract_validates_exact_frames_groups_rows_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table_kind: str
) -> None:
    _patch_small_contract(monkeypatch)
    input_path, summary_path, data = _write_source(tmp_path, table_kind=table_kind)

    receipt = validate_cacie_full_record_source(
        input_path,
        summary_path,
        data,
        source_snapshot=snapshot_cacie_full_record_source(input_path, summary_path),
        table_kind=table_kind,
        allowed_frame_steps=(1,),
        expected_ego_count=2,
    )

    assert receipt.input_parquet_rows == len(data)
    assert len(receipt.input_parquet_sha256) == 64
    assert len(receipt.source_summary_sha256) == 64
    assert receipt.selected_frame_delta_counts == {"3": 1}
    artifact: dict[str, object] = {}
    attach_source_contract(artifact, receipt)
    assert artifact["sparse_min_visible_rays"] == 3
    assert artifact["source_contract"]["frame_selection"]["exact_frame_set_validated"] is True  # type: ignore[index]


def test_contract_rejects_summary_row_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    source_summary = _summary()
    source_summary["outputs"]["counts"]["rows"] = 5  # type: ignore[index]
    input_path, summary_path, data = _write_source(
        tmp_path, table_kind="counts", summary=source_summary
    )

    with pytest.raises(ValueError, match="outputs.counts.rows disagrees"):
        validate_cacie_full_record_source(
            input_path,
            summary_path,
            data,
            source_snapshot=snapshot_cacie_full_record_source(input_path, summary_path),
            table_kind="counts",
            allowed_frame_steps=(1,),
            expected_ego_count=2,
        )


def test_contract_rejects_group_missing_one_selected_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    data = _counts()
    bad_index = data.index[(data["frame_idx"] == 4) & (data["fov_deg"] == 360.0)][0]
    data.loc[bad_index, "frame_idx"] = 1
    data.loc[bad_index, "frame_order"] = 0
    input_path, summary_path, data = _write_source(
        tmp_path, table_kind="counts", data=data
    )

    with pytest.raises(ValueError, match="complete frame set"):
        validate_cacie_full_record_source(
            input_path,
            summary_path,
            data,
            source_snapshot=snapshot_cacie_full_record_source(input_path, summary_path),
            table_kind="counts",
            allowed_frame_steps=(1,),
            expected_ego_count=2,
        )


def test_contract_rejects_wrong_sparse_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    source_summary = _summary()
    source_summary["config"]["sparse_min_visible_rays"] = 2  # type: ignore[index]
    input_path, summary_path, data = _write_source(
        tmp_path, table_kind="counts", summary=source_summary
    )

    with pytest.raises(ValueError, match="3-of-15"):
        validate_cacie_full_record_source(
            input_path,
            summary_path,
            data,
            source_snapshot=snapshot_cacie_full_record_source(input_path, summary_path),
            table_kind="counts",
            allowed_frame_steps=(1,),
            expected_ego_count=2,
        )


@pytest.mark.parametrize("replaced_role", ["input_parquet", "source_summary"])
def test_preload_snapshot_rejects_source_replaced_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replaced_role: str,
) -> None:
    _patch_small_contract(monkeypatch)
    input_path, summary_path, data = _write_source(tmp_path, table_kind="counts")
    source_snapshot = snapshot_cacie_full_record_source(input_path, summary_path)

    # Simulate a consumer completing pd.read_parquet() and then observing a
    # replacement at either source path before it validates or publishes.
    replaced_path = input_path if replaced_role == "input_parquet" else summary_path
    replaced_path.write_bytes(replaced_path.read_bytes() + b"\nreplacement")

    with pytest.raises(ValueError, match="changed after the pre-load snapshot"):
        validate_cacie_full_record_source(
            input_path,
            summary_path,
            data,
            source_snapshot=source_snapshot,
            table_kind="counts",
            allowed_frame_steps=(1,),
            expected_ego_count=2,
        )


def test_postload_guard_rejects_source_replaced_before_output_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_small_contract(monkeypatch)
    input_path, summary_path, _ = _write_source(tmp_path, table_kind="counts")
    source_snapshot = snapshot_cacie_full_record_source(input_path, summary_path)
    summary_path.write_bytes(summary_path.read_bytes() + b"\nreplacement")

    with pytest.raises(ValueError, match="changed after the pre-load snapshot"):
        require_cacie_full_record_source_unchanged(
            source_snapshot,
            input_path,
            summary_path,
        )


def test_effective_rate_rejects_treating_step10_as_contiguous_10hz() -> None:
    receipt = CacieSourceContract(
        input_parquet="input.parquet",
        input_parquet_sha256="a" * 64,
        input_parquet_rows=1,
        source_summary="full_record_summary.json",
        source_summary_sha256="b" * 64,
        source_input_sha256="c" * 64,
        sparse_min_visible_rays=3,
        table_kind="contributions",
        frame_step=10,
        nominal_source_frame_rate_hz=10.0,
        effective_frame_rate_hz=1.0,
        selected_frame_count=1,
        selected_frame_first=1,
        selected_frame_last=1,
        selected_frame_delta_counts={},
    )

    require_effective_frame_rate(1.0, receipt)
    with pytest.raises(ValueError, match="expected 1 Hz"):
        require_effective_frame_rate(10.0, receipt)
