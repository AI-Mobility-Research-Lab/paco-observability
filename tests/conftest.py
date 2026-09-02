from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from paco_observability.cacie_source_contract import CacieSourceContract


@pytest.fixture
def stub_cacie_source_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Any], None]:
    """Stub only the canonical lineage gate in small numerical CLI tests.

    The gate itself has dedicated tests against real Parquet metadata.  These
    existing CLI tests intentionally retain tiny synthetic frames so they stay
    focused on bootstrap/ranking output semantics.
    """

    def install(script_module: Any) -> None:
        def validate(
            input_path: Path,
            source_summary_path: Path,
            data: pd.DataFrame,
            *,
            table_kind: str,
            **_: Any,
        ) -> CacieSourceContract:
            frames = sorted(data["frame_idx"].astype(int).unique().tolist())
            return CacieSourceContract(
                input_parquet=str(input_path.resolve()),
                input_parquet_sha256="a" * 64,
                input_parquet_rows=len(data),
                source_summary=str(source_summary_path.resolve()),
                source_summary_sha256="b" * 64,
                source_input_sha256="c" * 64,
                sparse_min_visible_rays=3,
                table_kind=table_kind,
                frame_step=1,
                nominal_source_frame_rate_hz=10.0,
                effective_frame_rate_hz=10.0,
                selected_frame_count=len(frames),
                selected_frame_first=frames[0],
                selected_frame_last=frames[-1],
                selected_frame_delta_counts={"1": max(0, len(frames) - 1)},
            )

        monkeypatch.setattr(script_module, "validate_cacie_full_record_source", validate)

    return install
