from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from paco_observability.release_manifest import (
    DEFAULT_PACKAGE_NAMES,
    build_release_manifest,
    resolve_release_inputs,
    utc_now,
    validate_numeric_producer_declaration,
    write_release_manifest,
)


FIXED_UTC = "2026-09-02T18:00:00Z"


def test_default_packages_cover_numerical_plotting_and_notebook_stack() -> None:
    assert {
        "paco-observability",
        "numpy",
        "pandas",
        "pyarrow",
        "matplotlib",
        "ipykernel",
        "ipython",
        "jupyter-client",
        "jupyter-core",
        "nbclient",
        "nbformat",
    }.issubset(DEFAULT_PACKAGE_NAMES)


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
    ).strip()


def initialized_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repository"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    git(repo, "config", "user.name", "Manifest Test")
    git(repo, "config", "user.email", "manifest@example.invalid")

    artifacts = repo / "artifacts"
    (artifacts / "nested").mkdir(parents=True)
    (artifacts / "a.txt").write_text("alpha\n", encoding="utf-8")
    (artifacts / "nested" / "b.bin").write_bytes(b"\x00\x01\x02")
    (repo / "tracked.txt").write_text("clean\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "--quiet", "-m", "fixture")
    return repo, git(repo, "rev-parse", "HEAD")


def test_recursive_inventory_is_sorted_relative_and_deduplicated(tmp_path: Path) -> None:
    repo, commit = initialized_repository(tmp_path)
    manifest = build_release_manifest(
        base_dir=repo,
        requested_inputs=[Path("artifacts/nested/b.bin"), Path("artifacts")],
        output_path=tmp_path / "release.json",
        generated_at_utc=FIXED_UTC,
        package_names=["package-that-does-not-exist-cacie-test"],
    )

    assert manifest["generated_at_utc"] == FIXED_UTC
    assert manifest["git"] == {
        "commit": commit,
        "branch": git(repo, "branch", "--show-current"),
        "dirty": False,
        "state_timing": "before_manifest_write",
    }
    assert manifest["requested_inputs"] == [
        {"path": "artifacts", "kind": "directory"},
        {"path": "artifacts/nested/b.bin", "kind": "file"},
    ]
    assert manifest["artifacts"] == [
        {
            "path": "artifacts/a.txt",
            "size_bytes": 6,
            "sha256": hashlib.sha256(b"alpha\n").hexdigest(),
        },
        {
            "path": "artifacts/nested/b.bin",
            "size_bytes": 3,
            "sha256": hashlib.sha256(b"\x00\x01\x02").hexdigest(),
        },
    ]
    assert manifest["artifact_count"] == 2
    assert manifest["total_size_bytes"] == 9
    assert manifest["runtime"]["packages"] == {
        "package-that-does-not-exist-cacie-test": None
    }


def test_writer_is_byte_deterministic_for_fixed_provenance(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    output = tmp_path / "release.json"
    arguments = {
        "base_dir": repo,
        "requested_inputs": [Path("artifacts")],
        "output_path": output,
        "generated_at_utc": FIXED_UTC,
        "package_names": ["package-that-does-not-exist-cacie-test"],
    }
    first = write_release_manifest(**arguments)
    first_bytes = output.read_bytes()
    second = write_release_manifest(**arguments)

    assert first == second
    assert output.read_bytes() == first_bytes
    assert json.loads(first_bytes) == first


def test_git_dirty_state_is_recorded(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    manifest = build_release_manifest(
        base_dir=repo,
        requested_inputs=[Path("artifacts/a.txt")],
        generated_at_utc=FIXED_UTC,
        package_names=[],
    )

    assert manifest["git"]["dirty"] is True


def test_require_clean_git_rejects_dirty_repository(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Git worktree must be clean"):
        build_release_manifest(
            base_dir=repo,
            requested_inputs=[Path("artifacts/a.txt")],
            generated_at_utc=FIXED_UTC,
            package_names=[],
            require_clean_git=True,
        )


def test_second_hash_pass_rejects_bytes_changed_after_first_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = initialized_repository(tmp_path)
    import paco_observability.release_manifest as module

    original = module.hash_stable_file
    calls = 0

    def mutate_after_first_hash(path: Path) -> tuple[int, str]:
        nonlocal calls
        result = original(path)
        if path.name == "a.txt":
            calls += 1
            if calls == 1:
                path.write_text("changed after first pass\n", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "hash_stable_file", mutate_after_first_hash)
    with pytest.raises(ValueError, match="artifact bytes changed"):
        build_release_manifest(
            base_dir=repo,
            requested_inputs=[Path("artifacts/a.txt")],
            generated_at_utc=FIXED_UTC,
            package_names=[],
        )


def test_numeric_producer_declaration_is_optional_and_byte_exact(tmp_path: Path) -> None:
    repo, commit = initialized_repository(tmp_path)
    config = repo / "config.json"
    config.write_text(
        json.dumps(
            {
                "release": {
                    "numeric_producer_commit": commit,
                    "numeric_producer_files": ["tracked.txt"],
                }
            }
        ),
        encoding="utf-8",
    )

    declaration = validate_numeric_producer_declaration(
        base_dir=repo,
        config_path=Path("config.json"),
    )
    assert declaration is not None
    assert declaration["status"] == "VERIFIED"
    assert declaration["commit"] == commit
    assert declaration["files"][0]["path"] == "tracked.txt"
    assert declaration["files"][0]["sha256"] == hashlib.sha256(b"clean\n").hexdigest()

    manifest = build_release_manifest(
        base_dir=repo,
        requested_inputs=[Path("artifacts/a.txt"), Path("config.json")],
        generated_at_utc=FIXED_UTC,
        package_names=[],
        numeric_producer_config=Path("config.json"),
    )
    assert manifest["numeric_producer"] == declaration

    (repo / "tracked.txt").write_text("drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from declared commit"):
        validate_numeric_producer_declaration(
            base_dir=repo,
            config_path=Path("config.json"),
        )


def test_numeric_producer_half_declaration_fails_closed(tmp_path: Path) -> None:
    repo, commit = initialized_repository(tmp_path)
    config = repo / "config.json"
    config.write_text(
        json.dumps({"release": {"numeric_producer_commit": commit}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must either both be present"):
        validate_numeric_producer_declaration(
            base_dir=repo,
            config_path=Path("config.json"),
        )


def test_historical_full_record_entrypoint_is_verified_without_rewriting(
    tmp_path: Path,
) -> None:
    repo, commit = initialized_repository(tmp_path)
    entrypoint = repo / "historical.py"
    entrypoint.write_text("print('historical')\n", encoding="utf-8")
    git(repo, "add", "historical.py")
    git(repo, "commit", "--quiet", "-m", "historical entrypoint")
    commit = git(repo, "rev-parse", "HEAD")
    historical_bytes = entrypoint.read_bytes()
    historical_sha256 = hashlib.sha256(historical_bytes).hexdigest()
    historical_blob = git(repo, "rev-parse", f"{commit}:historical.py")
    entrypoint.write_text("print('current guard only')\n", encoding="utf-8")
    config = repo / "config.json"
    config.write_text(
        json.dumps(
            {
                "release": {
                    "numeric_producer_commit": commit,
                    "numeric_producer_files": ["tracked.txt"],
                    "historical_full_record_entrypoint": {
                        "path": "historical.py",
                        "commit": commit,
                        "git_blob_sha1": historical_blob,
                        "sha256": historical_sha256,
                        "current_difference_scope": "non-numerical guard only",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    declaration = validate_numeric_producer_declaration(
        base_dir=repo,
        config_path=Path("config.json"),
    )
    historical = declaration["historical_full_record_entrypoint"]
    assert historical["status"] == "VERIFIED_HISTORICAL_BLOB"
    assert historical["sha256"] == historical_sha256
    assert historical["current_matches_historical_blob"] is False

    config_payload = json.loads(config.read_text())
    config_payload["release"]["historical_full_record_entrypoint"]["sha256"] = "0" * 64
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 declaration is incorrect"):
        validate_numeric_producer_declaration(
            base_dir=repo,
            config_path=Path("config.json"),
        )


def test_canonical_release_can_require_numeric_producer_declaration(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    config = repo / "config.json"
    config.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="canonical release requires"):
        build_release_manifest(
            base_dir=repo,
            requested_inputs=[Path("artifacts/a.txt")],
            generated_at_utc=FIXED_UTC,
            package_names=[],
            numeric_producer_config=Path("config.json"),
            require_numeric_producer=True,
        )


def test_output_inside_recursive_input_is_rejected_before_write(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    output = repo / "artifacts" / "release.json"

    with pytest.raises(ValueError, match="must not be an input"):
        write_release_manifest(
            base_dir=repo,
            requested_inputs=[Path("artifacts")],
            output_path=output,
            generated_at_utc=FIXED_UTC,
            package_names=[],
        )

    assert not output.exists()


def test_explicit_output_file_cannot_hash_itself(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    output = repo / "release.json"
    output.write_text("old manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be an input"):
        write_release_manifest(
            base_dir=repo,
            requested_inputs=[Path("release.json")],
            output_path=output,
            generated_at_utc=FIXED_UTC,
            package_names=[],
        )


def test_inputs_outside_base_and_symlinks_are_rejected(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside base directory"):
        resolve_release_inputs(base_dir=repo, requested_inputs=[outside])

    (repo / "artifacts" / "link.txt").symlink_to(repo / "artifacts" / "a.txt")
    with pytest.raises(ValueError, match="contains a symlink"):
        resolve_release_inputs(base_dir=repo, requested_inputs=[Path("artifacts")])


def test_source_date_epoch_controls_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    assert utc_now() == "1970-01-01T00:00:00Z"

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-an-integer")
    with pytest.raises(ValueError, match="must be an integer"):
        utc_now()


def test_empty_input_directory_is_rejected(tmp_path: Path) -> None:
    repo, _ = initialized_repository(tmp_path)
    empty = repo / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contain no regular files"):
        resolve_release_inputs(base_dir=repo, requested_inputs=[Path("empty")])


def test_cli_writes_manifest_and_machine_readable_receipt(tmp_path: Path) -> None:
    repo, commit = initialized_repository(tmp_path)
    output = tmp_path / "cli-release.json"
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["SOURCE_DATE_EPOCH"] = "1788372000"

    process = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_release_artifact_manifest.py"),
            "artifacts",
            "--base-dir",
            str(repo),
            "--out",
            str(output),
            "--package",
            "package-that-does-not-exist-cacie-test",
            "--require-clean-git",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    receipt = json.loads(process.stdout)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["artifact_count"] == 2
    assert receipt["git_commit"] == commit
    assert receipt["git_dirty"] is False
    assert receipt["require_clean_git"] is True
    assert receipt["output"] == str(output)
    assert manifest["generated_at_utc"] == "2026-09-02T18:00:00Z"
    assert manifest["release_policy"] == {
        "require_clean_git": True,
        "require_numeric_producer": False,
    }
    assert "uv" in manifest["runtime"]["tools"]
    assert [row["path"] for row in manifest["artifacts"]] == [
        "artifacts/a.txt",
        "artifacts/nested/b.bin",
    ]
