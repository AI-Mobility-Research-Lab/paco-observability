from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "scripts" / "run_canonical.sh"


def test_run_canonical_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], cwd=REPOSITORY, check=True)


def test_run_canonical_help_does_not_start_computation() -> None:
    completed = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Stage switches" in completed.stdout
    assert "post-development calibrated threshold" in completed.stdout
    assert completed.stderr == ""


def test_run_canonical_uses_cacie_stages_not_legacy_event_claims() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "parametric_ego_sweep.py",
        "sensitivity_sweep.py",
        "range_sweep.py",
        "ego_near_miss_sweep.py",
        "--near-misses",
        "--allow-quality-fail",
    )
    for token in forbidden:
        assert token not in source

    ordered_stages = (
        "scripts/audit_input_provenance.py",
        "scripts/validate_occlusion_models.py",
        "scripts/calibrate_sparse_threshold.py",
        "scripts/compute_full_record_observability.py",
        "scripts/bootstrap_observability.py",
        "scripts/bootstrap_block_sensitivity.py",
        "scripts/rank_residual_demand.py",
        "scripts/summarize_cacie_sensitivities.py",
        "scripts/generate_cacie_validation_figure.py",
        "jupyter execute",
        "scripts/audit_cacie_notebook.py",
        "scripts/build_release_artifact_manifest.py",
    )
    offsets = [source.index(stage) for stage in ordered_stages]
    assert offsets == sorted(offsets)
    assert 'RUN_FULL_FIGURE="${RUN_FULL_FIGURE:-1}"' in source
    assert "scripts/generate_cacie_full_results_figure.py" in source
    assert source.count("--source-summary") == 4
    full_figure_stage = source.index('if [[ "$RUN_FULL_FIGURE" == 1 ]]')
    assert source.index("scripts/generate_cacie_validation_figure.py") < full_figure_stage
    assert full_figure_stage < source.index("jupyter execute")
    assert "--inplace" not in source
    assert 'CACIE_OUTPUT_ROOT="$OUTPUT_ROOT_ABSOLUTE"' in source
    assert "cacie_validation.executed.ipynb" in source
    assert "notebook_audit_receipt.json" in source
    assert "--write-receipt" in source
    assert "--verify-receipt" in source
    assert "--require-clean-git" in source
    assert "--numeric-producer-config" in source
    assert "--require-numeric-producer" in source
    assert 'PYTHON_CMD=("$UV_BIN" run --frozen python)' in source
    assert "canonical release mode forbids PYTHON_BIN" in source
    assert "scripts/write_validation_lineage_receipt.py" in source
    assert source.count("--verify") >= 3
    assert "source_lineage_receipt.json" in source
    assert "src/paco_observability scripts" in source
    assert "DETECTOR_VALIDATION_DIR" in source
    assert "MAIN_RUNTIME_RECEIPT" in source
    assert "CONVERGENCE_RUNTIME_RECEIPT" in source
    assert "FULL_RUNTIME_RECEIPT" in source
    assert 'NUMERIC_PRODUCER_COMMIT="$("${PYTHON_CMD[@]}" - "$CONFIG"' in source
    assert 'git -C "$REPO_ROOT" rev-parse HEAD' not in source
    assert source.count('--numeric-producer-config "$CONFIG"') >= 3
    assert "NUMERIC_PRODUCER_COMMIT disagrees with the byte-verified frozen release declaration" in source
