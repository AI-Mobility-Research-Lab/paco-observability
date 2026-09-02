from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = REPOSITORY / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_toctou_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("script_name", "output_arguments", "published_names"),
    [
        (
            "bootstrap_observability",
            lambda root: ["--out-dir", str(root / "bootstrap")],
            ("bootstrap/observability_bootstrap.json", "bootstrap/observability_bootstrap_replicates.parquet"),
        ),
        (
            "bootstrap_block_sensitivity",
            lambda root: ["--output", str(root / "block.json")],
            ("block.json",),
        ),
        (
            "rank_residual_demand",
            lambda root: ["--out-dir", str(root / "ranking")],
            ("ranking/residual_hotspot_ranks.json", "ranking/residual_hotspot_ranks.parquet"),
        ),
    ],
)
def test_consumer_rejects_source_replaced_immediately_after_parquet_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    output_arguments,
    published_names: tuple[str, ...],
) -> None:
    module = _load_script(script_name)
    input_path = tmp_path / "source.parquet"
    summary_path = tmp_path / "full_record_summary.json"
    input_path.write_bytes(b"snapshot fixture")
    summary_path.write_text("{}\n", encoding="utf-8")

    def replace_summary_after_read(path: Path) -> pd.DataFrame:
        assert Path(path) == input_path
        summary_path.write_text('{"replacement": true}\n', encoding="utf-8")
        return pd.DataFrame()

    monkeypatch.setattr(module.pd, "read_parquet", replace_summary_after_read)
    argv = [
        "--input",
        str(input_path),
        "--source-summary",
        str(summary_path),
        *output_arguments(tmp_path),
    ]
    with pytest.raises(ValueError, match="changed after the pre-load snapshot"):
        module.main(argv)

    assert all(not (tmp_path / name).exists() for name in published_names)
