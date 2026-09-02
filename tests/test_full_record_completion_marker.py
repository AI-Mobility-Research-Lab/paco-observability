from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "compute_full_record_observability.py"
)
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "compute_full_record_observability_cli", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT)


def test_overwrite_atomically_quarantines_old_completion_marker(tmp_path: Path) -> None:
    summary_path = tmp_path / "full_record_summary.json"
    summary_path.write_text('{"old_complete": true}\n', encoding="utf-8")

    stale_path = _SCRIPT._invalidate_existing_completion_marker(summary_path)

    assert stale_path is not None
    assert not summary_path.exists()
    assert stale_path.name.startswith("full_record_summary.json.")
    assert stale_path.suffix == ".stale"
    assert stale_path.read_text(encoding="utf-8") == '{"old_complete": true}\n'
