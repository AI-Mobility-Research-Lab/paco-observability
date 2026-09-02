from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from paco_observability.validation_lineage import (
    verify_validation_lineage_receipt,
    write_validation_lineage_receipt,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _fixture(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    _git(repo, "config", "user.name", "Lineage Test")
    _git(repo, "config", "user.email", "lineage@example.invalid")
    (repo / "README").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    tracks = tmp_path / "tracks.parquet"
    frame_index = tmp_path / "frame_index.parquet"
    tracks.write_bytes(b"canonical tracks fixture")
    frame_index.write_bytes(b"canonical frame index fixture")
    audit = {"rows": 2, "quality_gate": {"passed": True}}
    provenance = tmp_path / "input_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "tracks": {
                    "sha256": hashlib.sha256(tracks.read_bytes()).hexdigest(),
                    "audit": audit,
                },
                "frame_index": {
                    "sha256": hashlib.sha256(frame_index.read_bytes()).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    config_payload = {
        "ray_grids": [9, 17, 33],
        "z_modes": ["raw", "ground_anchored"],
        "fov_degrees": [120.0, 360.0],
    }
    config = tmp_path / "validation_config.json"
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    summary = tmp_path / "validation_summary.json"
    summary.write_text(
        json.dumps(
            {
                "config": config_payload,
                "input_audit": audit,
                "sampled_frame_count": 200,
                "sampled_frames": list(range(1, 201)),
                "excluded_source_frames": [10, 20],
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "validation_decisions.parquet"
    timings = tmp_path / "validation_timings.parquet"
    exclusions = tmp_path / "ego_exclusions.parquet"
    pd.DataFrame({"frame_idx": list(range(1, 201))}).to_parquet(decisions, index=False)
    pd.DataFrame({"elapsed_ms": [1.0]}).to_parquet(timings, index=False)
    pd.DataFrame({"frame_idx": pd.Series([], dtype="int64")}).to_parquet(
        exclusions, index=False
    )
    return {
        "stage": "validation_main_200_final",
        "repository": repo,
        "producer_commit": commit,
        "tracks": tracks,
        "frame_index": frame_index,
        "input_provenance": provenance,
        "summary": summary,
        "config": config,
        "decisions": decisions,
        "timings": timings,
        "exclusions": exclusions,
        "excluded_source_frames": [10, 20],
    }


def test_validation_lineage_binds_exact_sources_outputs_and_commit(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    output = tmp_path / "source_lineage_receipt.json"
    receipt = write_validation_lineage_receipt(output_path=output, **arguments)

    assert receipt["status"] == "VERIFIED"
    assert receipt["binding_mode"] == "retroactive_byte_exact_no_artifact_rewrite"
    assert receipt["declared_numeric_producer_commit"] == arguments["producer_commit"]
    assert receipt["sources"]["tracks"]["sha256"]
    assert not Path(receipt["sources"]["tracks"]["path"]).is_absolute()
    assert set(receipt["artifacts"]) == {
        "summary",
        "config",
        "decisions",
        "timings",
        "exclusions",
    }
    assert all(not Path(record["path"]).is_absolute() for record in receipt["artifacts"].values())
    assert verify_validation_lineage_receipt(output_path=output, **arguments) == receipt

    arguments["config"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary/config mismatch"):
        verify_validation_lineage_receipt(output_path=output, **arguments)


def test_validation_lineage_rejects_wrong_canonical_track_bytes(tmp_path: Path) -> None:
    arguments = _fixture(tmp_path)
    arguments["tracks"].write_bytes(b"different tracks")
    with pytest.raises(ValueError, match="canonical tracks bytes disagree"):
        write_validation_lineage_receipt(
            output_path=tmp_path / "source_lineage_receipt.json",
            **arguments,
        )
