from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import paco_observability.runtime_environment as runtime_environment
from paco_observability.runtime_environment import (
    build_runtime_environment_receipt,
    linux_cpu_model,
    write_runtime_environment_receipt,
)


FIXED_UTC = "2026-09-02T18:00:00Z"


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
    ).strip()


def initialized_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repository"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    git(repo, "config", "user.name", "Runtime Receipt Test")
    git(repo, "config", "user.email", "runtime@example.invalid")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "--quiet", "-m", "fixture")
    return repo, git(repo, "rev-parse", "HEAD")


def test_linux_cpu_model_uses_allow_list_and_handles_absence(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor : 0\nmodel name : Example CPU 9000\nSerial : secret-value\n",
        encoding="utf-8",
    )

    assert linux_cpu_model(system_name="Linux", cpuinfo_path=cpuinfo) == "Example CPU 9000"
    assert linux_cpu_model(system_name="Darwin", cpuinfo_path=cpuinfo) is None
    assert linux_cpu_model(system_name="Linux", cpuinfo_path=tmp_path / "missing") is None

    cpuinfo.write_text("processor : 0\nSerial : secret-value\n", encoding="utf-8")
    assert linux_cpu_model(system_name="Linux", cpuinfo_path=cpuinfo) is None


def test_build_receipt_has_explicit_stable_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_environment.socket, "gethostname", lambda: "analysis-host")
    monkeypatch.setattr(runtime_environment.platform_module, "platform", lambda: "TestOS-1")
    monkeypatch.setattr(runtime_environment.platform_module, "machine", lambda: "test-machine")
    monkeypatch.setattr(
        runtime_environment.platform_module,
        "python_implementation",
        lambda: "TestPython",
    )
    monkeypatch.setattr(runtime_environment.platform_module, "python_version", lambda: "3.99.1")
    monkeypatch.setattr(runtime_environment, "linux_cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(runtime_environment.os, "cpu_count", lambda: 48)
    monkeypatch.setattr(
        runtime_environment,
        "package_versions",
        lambda names: {"numpy": "2.0", "pandas": "3.0", "pyarrow": "23.0"},
    )
    monkeypatch.setattr(runtime_environment, "executable_version", lambda name: "uv 9.9.9")
    monkeypatch.setattr(
        runtime_environment,
        "git_state",
        lambda repository: {
            "commit": "a" * 40,
            "branch": None,
            "dirty": False,
            "state_timing": "before_manifest_write",
        },
    )

    receipt = build_runtime_environment_receipt(
        label=" validation ",
        repository=Path("unused"),
        generated_at_utc=FIXED_UTC,
    )

    assert receipt == {
        "schema_version": "1.1",
        "label": "validation",
        "generated_at_utc": FIXED_UTC,
        "hostname": "analysis-host",
        "platform": "TestOS-1",
        "machine": "test-machine",
        "cpu_model": "Test CPU",
        "logical_cpu_count": 48,
        "python_implementation": "TestPython",
        "python_version": "3.99.1",
        "numpy_version": "2.0",
        "pandas_version": "3.0",
        "pyarrow_version": "23.0",
        "execution_mode": "external_python",
        "uv_version": "uv 9.9.9",
        "git_repository_available": True,
        "git_provenance_status": "repository_verified",
        "declared_source_commit": None,
        "git_commit": "a" * 40,
        "git_branch": None,
        "git_dirty": False,
        "git_state_timing": "before_receipt_write",
        "git_commit_role": "execution_source_tree_at_receipt_capture",
        "numeric_producer": None,
    }


def test_source_tree_without_git_is_explicitly_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_environment,
        "git_state",
        lambda repository: (_ for _ in ()).throw(AssertionError("git_state must not be called")),
    )

    receipt = build_runtime_environment_receipt(
        label="full-record",
        repository=Path("source-copy-without-git"),
        source_tree_without_git=True,
        declared_source_commit="A" * 40,
        generated_at_utc=FIXED_UTC,
    )

    assert receipt["git_repository_available"] is False
    assert receipt["git_provenance_status"] == "declared_unverified_source_tree_without_git"
    assert receipt["declared_source_commit"] == "a" * 40
    assert receipt["git_commit"] is None
    assert receipt["git_branch"] is None
    assert receipt["git_dirty"] is None
    assert receipt["git_state_timing"] == "source_tree_without_git_declaration"
    assert receipt["execution_mode"] == "external_python"


@pytest.mark.parametrize("declared_commit", [None, "", "abc", "g" * 40, "a" * 39])
def test_source_tree_without_git_requires_full_commit(declared_commit: str | None) -> None:
    with pytest.raises(ValueError, match="declared_source_commit"):
        build_runtime_environment_receipt(
            label="full-record",
            repository=Path("source-copy-without-git"),
            source_tree_without_git=True,
            declared_source_commit=declared_commit,
        )


def test_declared_commit_is_rejected_in_repository_mode(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    with pytest.raises(ValueError, match="only valid"):
        build_runtime_environment_receipt(
            label="validation",
            repository=repo,
            declared_source_commit="a" * 40,
        )


def test_source_date_epoch_controls_receipt_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = initialized_repository(tmp_path)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    receipt = build_runtime_environment_receipt(label="test", repository=repo)

    assert receipt["generated_at_utc"] == "1970-01-01T00:00:00Z"


def test_writer_atomically_replaces_output_and_does_not_capture_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, commit = initialized_repository(tmp_path)
    output = tmp_path / "receipts" / "runtime.json"
    output.parent.mkdir()
    output.write_text("old, incomplete contents", encoding="utf-8")
    monkeypatch.setenv("CACIE_SECRET_TEST_TOKEN", "must-not-appear")

    receipt = write_runtime_environment_receipt(
        label="full-record",
        repository=repo,
        output_path=output,
        generated_at_utc=FIXED_UTC,
    )

    serialized = output.read_text(encoding="utf-8")
    assert json.loads(serialized) == receipt
    assert receipt["git_commit"] == commit
    assert receipt["git_dirty"] is False
    assert "CACIE_SECRET_TEST_TOKEN" not in serialized
    assert "must-not-appear" not in serialized
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_fresh_receipt_separates_execution_head_from_byte_verified_producer(
    tmp_path: Path,
) -> None:
    repo, producer_commit = initialized_repository(tmp_path)
    (repo / "execution-only.txt").write_text("new runner\n", encoding="utf-8")
    git(repo, "add", "execution-only.txt")
    git(repo, "commit", "--quiet", "-m", "new execution head")
    execution_commit = git(repo, "rev-parse", "HEAD")
    assert execution_commit != producer_commit

    config = repo / "numeric-producer.json"
    config.write_text(
        json.dumps(
            {
                "release": {
                    "numeric_producer_commit": producer_commit,
                    "numeric_producer_files": ["tracked.txt"],
                }
            }
        ),
        encoding="utf-8",
    )

    receipt = build_runtime_environment_receipt(
        label="fresh-after-runner-change",
        repository=repo,
        numeric_producer_config=config,
        generated_at_utc=FIXED_UTC,
    )

    assert receipt["git_commit"] == execution_commit
    assert receipt["git_dirty"] is True
    assert receipt["git_commit_role"] == "execution_source_tree_at_receipt_capture"
    assert receipt["numeric_producer"]["commit"] == producer_commit
    assert receipt["numeric_producer"]["status"] == "VERIFIED"
    assert receipt["numeric_producer"]["files"][0]["path"] == "tracked.txt"


def test_numeric_producer_config_fails_closed_without_a_declaration(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    config = repo / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not declare"):
        build_runtime_environment_receipt(
            label="fresh",
            repository=repo,
            numeric_producer_config=config,
        )


def test_empty_label_is_rejected(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    with pytest.raises(ValueError, match="label must not be empty"):
        build_runtime_environment_receipt(label="  ", repository=repo)


def test_cli_writes_receipt_with_reproducible_timestamp(tmp_path: Path) -> None:
    repo, commit = initialized_repository(tmp_path)
    output = tmp_path / "cli-runtime.json"
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["SOURCE_DATE_EPOCH"] = "1788372000"

    process = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "capture_runtime_environment.py"),
            "--label",
            "validation",
            "--repository",
            str(repo),
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(process.stdout) == receipt
    assert receipt["label"] == "validation"
    assert receipt["generated_at_utc"] == FIXED_UTC
    assert receipt["git_commit"] == commit
    assert receipt["git_dirty"] is False


def test_cli_records_unverified_source_tree_without_git(tmp_path: Path) -> None:
    source_tree = tmp_path / "source-copy"
    source_tree.mkdir()
    output = tmp_path / "cli-runtime-no-git.json"
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "capture_runtime_environment.py"),
            "--label",
            "full_record_primary_final",
            "--repository",
            str(source_tree),
            "--source-tree-without-git",
            "--declared-source-commit",
            "b" * 40,
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["git_repository_available"] is False
    assert receipt["git_provenance_status"] == "declared_unverified_source_tree_without_git"
    assert receipt["declared_source_commit"] == "b" * 40
    assert receipt["git_commit"] is None
    assert receipt["git_dirty"] is None
