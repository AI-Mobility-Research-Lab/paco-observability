from __future__ import annotations

import json
from pathlib import Path

import pytest

from paco_observability.notebook_audit import (
    NotebookAuditError,
    PASS_MARKER,
    REQUIRED_STAGES,
    _runtime_producer_binding_mode,
    build_notebook_audit_receipt,
    validate_executed_notebook,
    verify_notebook_audit_receipt,
    write_notebook_audit_receipt,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPOSITORY / "notebooks" / "cacie_validation.ipynb"


def notebook(*, execution_count: int | None, output_text: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": execution_count,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": output_text,
                    }
                ],
                "source": ["print(PASS_MARKER)\n"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def fake_audit() -> dict:
    return {
        "status": "PASS",
        "pass_marker": PASS_MARKER,
        "stages": [
            {"stage": stage, "status": "VERIFIED", "detail": "fixture"}
            for stage in REQUIRED_STAGES
        ],
        "check_count": 1,
        "checks": [{"check": "fixture", "status": "PASS", "detail": ""}],
        "audited_files": [
            {"name": "fixture", "path": "artifact.bin", "size_bytes": 1, "sha256": "a" * 64}
        ],
    }


def write_notebooks(tmp_path: Path, output_text: str = f"{PASS_MARKER}\n") -> tuple[Path, Path]:
    source = tmp_path / "source.ipynb"
    executed = tmp_path / "executed.ipynb"
    source.write_text(
        json.dumps(notebook(execution_count=None, output_text="")),
        encoding="utf-8",
    )
    executed_payload = notebook(execution_count=1, output_text=output_text)
    # nbformat is allowed to serialize a cell source as one string rather than
    # the source notebook's list of lines; semantic source content must match.
    executed_payload["cells"][0]["source"] = "print(PASS_MARKER)\n"
    executed.write_text(json.dumps(executed_payload), encoding="utf-8")
    return source, executed


def test_source_notebook_is_unexecuted_fail_closed_and_parameterized() -> None:
    payload = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    all_source = SOURCE_NOTEBOOK.read_text(encoding="utf-8").lower()
    assert "assert" not in all_source
    assert "skipp" not in all_source
    assert "CACIE_OUTPUT_ROOT" in code
    assert "audit_canonical_artifacts" in code
    assert "set(observed_stages) != set(REQUIRED_STAGES)" in code
    assert "nonverified_stages" in code
    assert "print(PASS_MARKER)" in code
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)


def test_executed_notebook_requires_exact_source_and_one_pass_marker(tmp_path: Path) -> None:
    source, executed = write_notebooks(tmp_path)
    validate_executed_notebook(source_notebook=source, executed_notebook=executed)

    executed.write_text(
        json.dumps(notebook(execution_count=1, output_text=f"{PASS_MARKER}\n{PASS_MARKER}\n")),
        encoding="utf-8",
    )
    with pytest.raises(NotebookAuditError, match="exactly once"):
        validate_executed_notebook(source_notebook=source, executed_notebook=executed)

    executed.write_text(
        json.dumps(notebook(execution_count=None, output_text=f"{PASS_MARKER}\n")),
        encoding="utf-8",
    )
    with pytest.raises(NotebookAuditError, match="unexecuted code cells"):
        validate_executed_notebook(source_notebook=source, executed_notebook=executed)


def test_pass_marker_must_come_from_final_monotonic_enforcement_cell(tmp_path: Path) -> None:
    first = notebook(execution_count=None, output_text="")["cells"][0]
    first["source"] = ["print('ordinary cell')\n"]
    final = notebook(execution_count=None, output_text="")["cells"][0]
    source_payload = {
        "cells": [first, final],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    source = tmp_path / "source.ipynb"
    executed = tmp_path / "executed.ipynb"
    source.write_text(json.dumps(source_payload), encoding="utf-8")

    forged = json.loads(json.dumps(source_payload))
    forged["cells"][0]["execution_count"] = 1
    forged["cells"][0]["outputs"] = [
        {"name": "stdout", "output_type": "stream", "text": f"{PASS_MARKER}\n"}
    ]
    forged["cells"][1]["execution_count"] = 2
    forged["cells"][1]["outputs"] = []
    executed.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(NotebookAuditError, match="outside the final"):
        validate_executed_notebook(source_notebook=source, executed_notebook=executed)

    forged["cells"][0]["outputs"] = []
    forged["cells"][0]["execution_count"] = 2
    forged["cells"][1]["outputs"] = [
        {"name": "stdout", "output_type": "stream", "text": f"{PASS_MARKER}\n"}
    ]
    executed.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(NotebookAuditError, match="strictly increasing"):
        validate_executed_notebook(source_notebook=source, executed_notebook=executed)


def test_notebook_receipt_revalidates_current_notebook_and_artifact_snapshot(
    tmp_path: Path,
) -> None:
    source, executed = write_notebooks(tmp_path)
    receipt_path = tmp_path / "notebook_audit_receipt.json"
    audit = fake_audit()
    written = write_notebook_audit_receipt(
        receipt_path=receipt_path,
        source_notebook=source,
        executed_notebook=executed,
        artifact_audit=audit,
        repo_root=tmp_path,
    )
    assert written["status"] == "PASS"
    assert written["executed_notebook"]["sha256"]
    assert verify_notebook_audit_receipt(
        receipt_path=receipt_path,
        source_notebook=source,
        executed_notebook=executed,
        artifact_audit=audit,
        repo_root=tmp_path,
    ) == written

    changed_audit = fake_audit()
    changed_audit["audited_files"][0]["sha256"] = "b" * 64
    with pytest.raises(NotebookAuditError, match="receipt is stale"):
        verify_notebook_audit_receipt(
            receipt_path=receipt_path,
            source_notebook=source,
            executed_notebook=executed,
            artifact_audit=changed_audit,
            repo_root=tmp_path,
        )


def test_receipt_rejects_nonverified_or_incomplete_stage_set(tmp_path: Path) -> None:
    source, executed = write_notebooks(tmp_path)
    audit = fake_audit()
    audit["stages"][0]["status"] = "FAILED"
    with pytest.raises(NotebookAuditError, match="non-VERIFIED"):
        build_notebook_audit_receipt(
            source_notebook=source,
            executed_notebook=executed,
            artifact_audit=audit,
            repo_root=tmp_path,
        )

    audit = fake_audit()
    audit["stages"].pop()
    with pytest.raises(NotebookAuditError, match="every required stage"):
        build_notebook_audit_receipt(
            source_notebook=source,
            executed_notebook=executed,
            artifact_audit=audit,
            repo_root=tmp_path,
        )


def test_runtime_producer_binding_accepts_dual_layer_and_historical_receipts() -> None:
    producer_commit = "7" * 40
    declaration = {
        "status": "VERIFIED",
        "commit": producer_commit,
        "files": [{"path": "numeric.py", "sha256": "a" * 64}],
    }

    fresh = {
        "git_repository_available": True,
        "git_commit": "8" * 40,
        "git_dirty": False,
        "numeric_producer": declaration,
    }
    assert _runtime_producer_binding_mode(
        fresh,
        expected_declaration=declaration,
    ) == "byte_verified_numeric_producer"

    historical_local_dirty = {
        "git_commit": producer_commit,
        "git_dirty": True,
    }
    assert _runtime_producer_binding_mode(
        historical_local_dirty,
        expected_declaration=declaration,
    ) == "legacy_execution_commit_equals_numeric_producer"

    historical_remote_no_git = {
        "git_repository_available": False,
        "git_provenance_status": "declared_unverified_source_tree_without_git",
        "declared_source_commit": producer_commit,
        "git_commit": None,
        "git_dirty": None,
    }
    assert _runtime_producer_binding_mode(
        historical_remote_no_git,
        expected_declaration=declaration,
    ) == "legacy_unverified_no_git_numeric_producer_declaration"

    with pytest.raises(NotebookAuditError, match="legacy repository runtime receipt"):
        _runtime_producer_binding_mode(
            {"git_repository_available": True, "git_commit": "8" * 40},
            expected_declaration=declaration,
        )
