#!/usr/bin/env bash
# Run the fail-closed PACO/CACIE v1.1 analysis from governed canonical inputs.
#
# Every RUN_* switch accepts only 0 or 1.  Setting a stage to 0 means "reuse a
# previously completed artifact", not "ignore the stage": downstream inputs are
# still required and checked.  Full-record stages intentionally do not overwrite
# existing output unless their corresponding OVERWRITE_* switch is set to 1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_canonical.sh

Run the complete fail-closed CACIE v1.1 workflow. The script accepts no
positional arguments; configure it with environment variables.

Stage switches (default 1):
  RUN_AUDIT RUN_MAIN_VALIDATION RUN_CALIBRATION RUN_CONVERGENCE
  RUN_VALIDATION_STRATA RUN_FULL_RECORD RUN_BOOTSTRAP
  RUN_BLOCK_SENSITIVITY RUN_RANKING RUN_Z_MODE_SENSITIVITY
  RUN_PARTIAL_SCAN_SENSITIVITY RUN_SENSITIVITY_SUMMARY
  RUN_VALIDATION_FIGURE RUN_FULL_FIGURE RUN_NOTEBOOK
  RUN_RELEASE_MANIFEST

Setting an analytical stage to 0 reuses its completed artifact; required
downstream files and frozen contracts are still checked. Expensive output is
never overwritten unless OVERWRITE_FULL_RECORD=1 or OVERWRITE_SENSITIVITY=1.

Supported runtime/layout overrides:
  OUTPUT_ROOT UV_BIN VALIDATION_N_PROCS FULL_N_PROCESSES
  FULL_CHUNKSIZE FULL_BUFFER_ROWS FULL_PROGRESS_EVERY

Canonical release uses `uv run --frozen`; PYTHON_BIN is accepted only for
non-release diagnostics. Canonical input/stage names are fixed so computation
and notebook audit cannot silently use different layouts.

Notebook outputs (defaults below OUTPUT_ROOT):
  EXECUTED_NOTEBOOK NOTEBOOK_AUDIT_RECEIPT

The scientific settings (frame cohort, exclusions, seed, ray grids, 3/15
post-development calibrated threshold, and bootstrap design) are frozen and
are checked against configs/cacie_v1.1.json before computation.
EOF
}

if (( $# > 0 )); then
  if [[ "$1" == "--help" || "$1" == "-h" ]] && (( $# == 1 )); then
    usage
    exit 0
  fi
  usage >&2
  printf '[CACIE] ERROR: unexpected positional arguments\n' >&2
  exit 2
fi

UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-}"
CONFIG="${CONFIG:-configs/cacie_v1.1.json}"
TRACKS="${TRACKS:-data/inputs/canonical/tracks.parquet}"
FRAME_INDEX="${FRAME_INDEX:-data/inputs/canonical/frame_index.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/canonical}"
UPSTREAM_MANIFEST="${UPSTREAM_MANIFEST:-${OUTPUT_ROOT}/upstream/manifest.json}"

PROVENANCE_MANIFEST="${PROVENANCE_MANIFEST:-${OUTPUT_ROOT}/input_provenance.json}"
MAIN_DIR="${MAIN_DIR:-${OUTPUT_ROOT}/validation_main}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${OUTPUT_ROOT}/sparse_calibration}"
CONVERGENCE_DIR="${CONVERGENCE_DIR:-${OUTPUT_ROOT}/validation_convergence}"
STRATA_DIR="${STRATA_DIR:-${OUTPUT_ROOT}/validation_strata_calibrated}"
FULL_DIR="${FULL_DIR:-${OUTPUT_ROOT}/full_record_primary}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${OUTPUT_ROOT}/bootstrap_primary}"
BLOCK_SENSITIVITY_DIR="${BLOCK_SENSITIVITY_DIR:-${OUTPUT_ROOT}/bootstrap_block_sensitivity}"
RANK_DIR="${RANK_DIR:-${OUTPUT_ROOT}/residual_demand}"
Z_SENSITIVITY_DIR="${Z_SENSITIVITY_DIR:-${OUTPUT_ROOT}/z_mode_sensitivity_step10}"
PARTIAL_SENSITIVITY_DIR="${PARTIAL_SENSITIVITY_DIR:-${OUTPUT_ROOT}/partial_scan_only}"
SENSITIVITY_DIR="${SENSITIVITY_DIR:-${OUTPUT_ROOT}/sensitivity_summary}"
FIGURE_DIR="${FIGURE_DIR:-${OUTPUT_ROOT}/figures}"
NOTEBOOK="${NOTEBOOK:-notebooks/cacie_validation.ipynb}"
NOTEBOOK_DIR="${NOTEBOOK_DIR:-${OUTPUT_ROOT}/notebook}"
EXECUTED_NOTEBOOK="${EXECUTED_NOTEBOOK:-${NOTEBOOK_DIR}/cacie_validation.executed.ipynb}"
NOTEBOOK_AUDIT_RECEIPT="${NOTEBOOK_AUDIT_RECEIPT:-${NOTEBOOK_DIR}/notebook_audit_receipt.json}"
DETECTOR_VALIDATION_DIR="${DETECTOR_VALIDATION_DIR:-${OUTPUT_ROOT}/detector_validation}"
MAIN_RUNTIME_RECEIPT="${MAIN_RUNTIME_RECEIPT:-${MAIN_DIR}/runtime_environment.json}"
CONVERGENCE_RUNTIME_RECEIPT="${CONVERGENCE_RUNTIME_RECEIPT:-${CONVERGENCE_DIR}/runtime_environment.json}"
FULL_RUNTIME_RECEIPT="${FULL_RUNTIME_RECEIPT:-${FULL_DIR}/runtime_environment.json}"
MAIN_SOURCE_LINEAGE_RECEIPT="${MAIN_SOURCE_LINEAGE_RECEIPT:-${MAIN_DIR}/source_lineage_receipt.json}"
CONVERGENCE_SOURCE_LINEAGE_RECEIPT="${CONVERGENCE_SOURCE_LINEAGE_RECEIPT:-${CONVERGENCE_DIR}/source_lineage_receipt.json}"
RELEASE_MANIFEST="${RELEASE_MANIFEST:-outputs/cacie_v1.1_release_manifest.json}"

VALIDATION_N_PROCS="${VALIDATION_N_PROCS:-28}"
FULL_N_PROCESSES="${FULL_N_PROCESSES:-28}"
FULL_CHUNKSIZE="${FULL_CHUNKSIZE:-8}"
FULL_BUFFER_ROWS="${FULL_BUFFER_ROWS:-50000}"
FULL_PROGRESS_EVERY="${FULL_PROGRESS_EVERY:-250}"

RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_MAIN_VALIDATION="${RUN_MAIN_VALIDATION:-1}"
RUN_CALIBRATION="${RUN_CALIBRATION:-1}"
RUN_CONVERGENCE="${RUN_CONVERGENCE:-1}"
RUN_VALIDATION_STRATA="${RUN_VALIDATION_STRATA:-1}"
RUN_FULL_RECORD="${RUN_FULL_RECORD:-1}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-1}"
RUN_BLOCK_SENSITIVITY="${RUN_BLOCK_SENSITIVITY:-1}"
RUN_RANKING="${RUN_RANKING:-1}"
RUN_Z_MODE_SENSITIVITY="${RUN_Z_MODE_SENSITIVITY:-1}"
RUN_PARTIAL_SCAN_SENSITIVITY="${RUN_PARTIAL_SCAN_SENSITIVITY:-1}"
RUN_SENSITIVITY_SUMMARY="${RUN_SENSITIVITY_SUMMARY:-1}"
RUN_VALIDATION_FIGURE="${RUN_VALIDATION_FIGURE:-1}"
RUN_FULL_FIGURE="${RUN_FULL_FIGURE:-1}"
RUN_NOTEBOOK="${RUN_NOTEBOOK:-1}"
RUN_RELEASE_MANIFEST="${RUN_RELEASE_MANIFEST:-1}"
OVERWRITE_FULL_RECORD="${OVERWRITE_FULL_RECORD:-0}"
OVERWRITE_SENSITIVITY="${OVERWRITE_SENSITIVITY:-0}"

FULL_FIGURE_SCRIPT="${FULL_FIGURE_SCRIPT:-scripts/generate_cacie_full_results_figure.py}"
VALIDATION_FIGURE_BASE="${VALIDATION_FIGURE_BASE:-${FIGURE_DIR}/cacie_validation}"
FULL_FIGURE_BASE="${FULL_FIGURE_BASE:-${FIGURE_DIR}/cacie_full_results}"
SENSITIVITY_SUMMARY="${SENSITIVITY_SUMMARY:-${SENSITIVITY_DIR}/cacie_sensitivity_summary.json}"
BLOCK_SENSITIVITY_JSON="${BLOCK_SENSITIVITY_JSON:-${BLOCK_SENSITIVITY_DIR}/observability_block_sensitivity.json}"
NUMERIC_PRODUCER_COMMIT="${NUMERIC_PRODUCER_COMMIT:-}"

readonly EXPECTED_FIRST_FRAME=1
readonly EXPECTED_LAST_FRAME=40795
readonly ALLOWED_MISSING_FRAME=6747
readonly SPARSE_MIN_VISIBLE_RAYS=3
readonly SEED=20260902
readonly -a GROUND_PLANE=(-0.0015611966 -0.0601932594 0.9981855209 2.907646656)
readonly -a PARTIAL_FRAMES=(
  6674 6675 6747 7025 7435 11077 11155 11166
  11178 11189 11201 11212 11213 11235 11247 40795
)

log() {
  printf '[CACIE] %s\n' "$*" >&2
}

die() {
  printf '[CACIE] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "required file is missing: $path"
}

require_bool() {
  local name="$1"
  local value="${!name}"
  [[ "$value" == 0 || "$value" == 1 ]] || die "$name must be 0 or 1 (got: $value)"
}

for switch in \
  RUN_AUDIT RUN_MAIN_VALIDATION RUN_CALIBRATION RUN_CONVERGENCE \
  RUN_VALIDATION_STRATA RUN_FULL_RECORD RUN_BOOTSTRAP RUN_BLOCK_SENSITIVITY \
  RUN_RANKING RUN_Z_MODE_SENSITIVITY RUN_PARTIAL_SCAN_SENSITIVITY \
  RUN_SENSITIVITY_SUMMARY RUN_VALIDATION_FIGURE RUN_FULL_FIGURE RUN_NOTEBOOK \
  RUN_RELEASE_MANIFEST OVERWRITE_FULL_RECORD OVERWRITE_SENSITIVITY; do
  require_bool "$switch"
done

command -v "$UV_BIN" >/dev/null 2>&1 || die "uv executable not found: $UV_BIN"
if [[ -n "$PYTHON_BIN" ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
  PYTHON_CMD=("$PYTHON_BIN")
  PYTHON_EXECUTION_MODE=external_python
else
  PYTHON_CMD=("$UV_BIN" run --frozen python)
  PYTHON_EXECUTION_MODE=uv_frozen
fi
if [[ -z "$NUMERIC_PRODUCER_COMMIT" ]]; then
  NUMERIC_PRODUCER_COMMIT="$("${PYTHON_CMD[@]}" - "$CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
release = config.get("release")
if not isinstance(release, dict):
    raise SystemExit("config.release must declare the frozen numeric producer")
commit = release.get("numeric_producer_commit")
if not isinstance(commit, str) or not commit.strip():
    raise SystemExit("config.release.numeric_producer_commit must be nonempty")
print(commit.strip())
PY
)"
fi
if [[ ! "$NUMERIC_PRODUCER_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  die "NUMERIC_PRODUCER_COMMIT must be a full lowercase 40-character Git SHA"
fi
if [[ "$RUN_RELEASE_MANIFEST" == 1 && -n "$PYTHON_BIN" ]]; then
  die "canonical release mode forbids PYTHON_BIN; use the uv --frozen environment"
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for input in "$CONFIG" "$TRACKS" "$FRAME_INDEX" "$UPSTREAM_MANIFEST"; do
  require_file "$input"
done

# The notebook audit intentionally has one canonical layout. Reject path
# overrides that would compute successfully but become unauditable at stage 15.
"${PYTHON_CMD[@]}" - \
  "$REPO_ROOT" "$OUTPUT_ROOT" "$CONFIG" "$TRACKS" "$FRAME_INDEX" \
  "$UPSTREAM_MANIFEST" "$PROVENANCE_MANIFEST" "$MAIN_DIR" "$CALIBRATION_DIR" \
  "$CONVERGENCE_DIR" "$STRATA_DIR" "$FULL_DIR" "$BOOTSTRAP_DIR" \
  "$BLOCK_SENSITIVITY_DIR" "$RANK_DIR" "$Z_SENSITIVITY_DIR" \
  "$PARTIAL_SENSITIVITY_DIR" "$SENSITIVITY_DIR" "$FIGURE_DIR" \
  "$DETECTOR_VALIDATION_DIR" "$VALIDATION_FIGURE_BASE" "$FULL_FIGURE_BASE" \
  "$RUN_RELEASE_MANIFEST" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
output = (root / sys.argv[2]).resolve() if not Path(sys.argv[2]).is_absolute() else Path(sys.argv[2]).resolve()
actual = [Path(value) for value in sys.argv[3:22]]
expected = [
    root / "configs/cacie_v1.1.json",
    root / "data/inputs/canonical/tracks.parquet",
    root / "data/inputs/canonical/frame_index.parquet",
    output / "upstream/manifest.json",
    output / "input_provenance.json",
    output / "validation_main",
    output / "sparse_calibration",
    output / "validation_convergence",
    output / "validation_strata_calibrated",
    output / "full_record_primary",
    output / "bootstrap_primary",
    output / "bootstrap_block_sensitivity",
    output / "residual_demand",
    output / "z_mode_sensitivity_step10",
    output / "partial_scan_only",
    output / "sensitivity_summary",
    output / "figures",
    output / "detector_validation",
    output / "figures/cacie_validation",
]
# The final figure base is the remaining path argument before the release flag.
actual.append(Path(sys.argv[22]))
expected.append(output / "figures/cacie_full_results")
for supplied, required in zip(actual, expected, strict=True):
    resolved = (root / supplied).resolve() if not supplied.is_absolute() else supplied.resolve()
    if resolved != required.resolve():
        raise SystemExit(
            f"canonical notebook layout forbids path override: {resolved} != {required.resolve()}"
        )
if sys.argv[23] == "1":
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise SystemExit("canonical release OUTPUT_ROOT must be inside the repository") from exc
PY

# Refuse protocol drift before any expensive work.  These are the frozen v1.1
# analysis settings; the 3/15 rule is a post-development calibration result,
# guarded here only after that selection and held-out evaluation were completed.
"${PYTHON_CMD[@]}" - "$CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(f"frozen config check failed: {message}")

require(config["seed"] == 20260902, "seed must be 20260902")
require(config["upstream"]["source_frame_range"] == [1, 40795], "source frame range")
require(config["source_quality_exclusions"]["partial_scan_frames"] == [
    6674, 6675, 6747, 7025, 7435, 11077, 11155, 11166,
    11178, 11189, 11201, 11212, 11213, 11235, 11247, 40795,
], "partial-scan declarations")
require(config["source_quality_exclusions"]["neighbor_radius_frames"] == 3,
        "partial-scan neighbor radius")
require(config["source_quality_exclusions"]["point_count_ratio_threshold"] == 0.6,
        "partial-scan point-ratio threshold")
require(config["ground_plane"]["normal"] == [-0.0015611966, -0.0601932594, 0.9981855209],
        "ground-plane normal")
require(config["ground_plane"]["d"] == 2.907646656, "ground-plane offset")
require(config["onboard_sensor"]["fov_degrees"] == [120.0, 360.0], "FOVs")
require(config["validation"]["balanced_time_density_frames"] == 200,
        "main validation frame count")
require(config["validation"]["primary_grid_comparison"] == [9, 17, 33],
        "main ray grids")
require(config["validation"]["convergence_frames"] == 25,
        "convergence frame count")
require(config["validation"]["convergence_sample_relationship"] ==
        "deterministic 25-frame subset of the 200 main validation frames; not an independent sample",
        "convergence sample relationship")
require(config["validation"]["convergence_grids"] == [17, 33, 65],
        "convergence grids")
calibration = config["validation"]["threshold_calibration"]
require(calibration["calibration_frames"] == 100, "calibration frame count")
require(calibration["held_out_evaluation_frames"] == 100, "held-out frame count")
require(calibration["candidate_min_visible_rays"] == list(range(1, 16)),
        "calibration candidates")
require(calibration["selected_min_visible_rays"] == 3, "selected sparse threshold")
require(config["occlusion"]["sparse_min_visible_rays"] == 3, "full-record sparse threshold")
bootstrap = config["bootstrap"]
require(bootstrap["frame_rate_hz"] == 10.0, "frame rate")
require(bootstrap["primary_block_seconds"] == 120.0, "primary block duration")
require(bootstrap["block_sensitivity_seconds"] == [30.0, 60.0, 120.0, 240.0],
        "block-duration sensitivity")
require(bootstrap["replicates"] == 5000, "bootstrap replicate count")
PY

# If the config declares the commit and files that produced the numerical
# artifacts, reject drift before any expensive stage. A half declaration or
# any byte mismatch is always fatal; release mode additionally requires it.
"${PYTHON_CMD[@]}" - "$REPO_ROOT" "$CONFIG" "$NUMERIC_PRODUCER_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

from paco_observability.release_manifest import validate_numeric_producer_declaration

declaration = validate_numeric_producer_declaration(
    base_dir=Path(sys.argv[1]),
    config_path=Path(sys.argv[2]),
)
if declaration is not None:
    if declaration["commit"] != sys.argv[3]:
        raise SystemExit(
            "NUMERIC_PRODUCER_COMMIT disagrees with the byte-verified frozen release declaration"
        )
    print(json.dumps({
        "numeric_producer_commit": declaration["commit"],
        "numeric_producer_file_count": len(declaration["files"]),
        "status": declaration["status"],
    }, sort_keys=True))
PY

if [[ "$RUN_RELEASE_MANIFEST" == 1 ]]; then
  "${PYTHON_CMD[@]}" - "$REPO_ROOT" "$CONFIG" "$NUMERIC_PRODUCER_COMMIT" <<'PY'
import sys
from pathlib import Path

from paco_observability.release_manifest import git_state
from paco_observability.release_manifest import validate_numeric_producer_declaration

state = git_state(Path(sys.argv[1]))
if state["dirty"]:
    raise SystemExit(
        "canonical release mode requires a clean Git worktree before numerical work"
    )
declaration = validate_numeric_producer_declaration(
    base_dir=Path(sys.argv[1]),
    config_path=Path(sys.argv[2]),
)
if declaration is None:
    raise SystemExit(
        "canonical release mode requires release.numeric_producer_commit and "
        "release.numeric_producer_files"
    )
if declaration["commit"] != sys.argv[3]:
    raise SystemExit("NUMERIC_PRODUCER_COMMIT disagrees with the frozen release declaration")
PY
fi

AUDIT_PARTIAL_ARGS=()
ANALYSIS_EXCLUDE_ARGS=()
ONLY_PARTIAL_ARGS=()
for frame in "${PARTIAL_FRAMES[@]}"; do
  AUDIT_PARTIAL_ARGS+=(--exclude-partial-frame "$frame")
  ANALYSIS_EXCLUDE_ARGS+=(--exclude-frame "$frame")
  ONLY_PARTIAL_ARGS+=(--only-frame "$frame")
done

COMMON_VALIDATION_ARGS=(
  --tracks "$TRACKS"
  --expected-frame-range "$EXPECTED_FIRST_FRAME" "$EXPECTED_LAST_FRAME"
  --allow-missing-frame "$ALLOWED_MISSING_FRAME"
  "${ANALYSIS_EXCLUDE_ARGS[@]}"
  --ground-plane "${GROUND_PLANE[@]}"
  --n-ego-per-approach 15
  --fov-deg 120 360
  --sparse-min-visible-rays "$SPARSE_MIN_VISIBLE_RAYS"
  --seed "$SEED"
  --n-procs "$VALIDATION_N_PROCS"
)

COMMON_FULL_ARGS=(
  --tracks "$TRACKS"
  --ground-plane "${GROUND_PLANE[@]}"
  --expected-frame-range "$EXPECTED_FIRST_FRAME" "$EXPECTED_LAST_FRAME"
  --allow-missing-frame "$ALLOWED_MISSING_FRAME"
  --n-ego-per-approach 15
  --fov-deg 120 360
  --sparse-min-visible-rays "$SPARSE_MIN_VISIBLE_RAYS"
  --vru-weight 3
  --non-vru-weight 1
  --n-processes "$FULL_N_PROCESSES"
  --chunksize "$FULL_CHUNKSIZE"
  --buffer-rows "$FULL_BUFFER_ROWS"
  --progress-every "$FULL_PROGRESS_EVERY"
)

FULL_OVERWRITE_ARGS=()
SENSITIVITY_OVERWRITE_ARGS=()
if [[ "$OVERWRITE_FULL_RECORD" == 1 ]]; then
  FULL_OVERWRITE_ARGS+=(--overwrite)
fi
if [[ "$OVERWRITE_SENSITIVITY" == 1 ]]; then
  SENSITIVITY_OVERWRITE_ARGS+=(--overwrite)
fi

if [[ "$RUN_AUDIT" == 1 ]]; then
  log "1/16 auditing canonical tracks and source-frame provenance"
  "${PYTHON_CMD[@]}" scripts/audit_input_provenance.py \
    --tracks "$TRACKS" \
    --frame-index "$FRAME_INDEX" \
    --upstream-manifest "$UPSTREAM_MANIFEST" \
    --output "$PROVENANCE_MANIFEST" \
    --expected-frame-range "$EXPECTED_FIRST_FRAME" "$EXPECTED_LAST_FRAME" \
    --allow-missing-frame "$ALLOWED_MISSING_FRAME" \
    "${AUDIT_PARTIAL_ARGS[@]}" \
    --ground-plane "${GROUND_PLANE[@]}"
else
  log "1/16 reusing provenance manifest (RUN_AUDIT=0)"
fi
require_file "$PROVENANCE_MANIFEST"

# Revalidate a reused manifest against the current bytes and frozen exclusions.
"${PYTHON_CMD[@]}" - "$PROVENANCE_MANIFEST" "$TRACKS" "$FRAME_INDEX" "$UPSTREAM_MANIFEST" <<'PY'
import hashlib
import json
import sys

manifest_path, tracks_path, frame_index_path, upstream_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(f"provenance contract failed: {message}")

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

expected_partial = [
    6674, 6675, 6747, 7025, 7435, 11077, 11155, 11166,
    11178, 11189, 11201, 11212, 11213, 11235, 11247, 40795,
]
require(manifest["tracks"]["audit"]["quality_gate"]["passed"] is True,
        "track gate did not pass")
require(manifest["frame_index"]["audit"]["quality_gate"]["passed"] is True,
        "frame-index gate did not pass")
require(manifest["tracks"]["sha256"] == sha256(tracks_path), "track bytes changed")
require(manifest["frame_index"]["sha256"] == sha256(frame_index_path),
        "frame-index bytes changed")
require(manifest["upstream_manifest"]["sha256"] == sha256(upstream_path),
        "upstream manifest bytes changed")
require(manifest["parameters"]["expected_frame_range"] == [1, 40795],
        "source frame range")
require(manifest["parameters"]["allowed_missing_frames"] == [6747],
        "allowed missing frames")
require(manifest["parameters"]["excluded_partial_frames"] == expected_partial,
        "declared partial frames")
require(manifest["frame_index"]["audit"]["detected_partial_frames"] == expected_partial,
        "detected partial frames")
require(manifest["frame_index"]["audit"]["unexcluded_partial_frames"] == [],
        "an unexcluded partial frame remains")
PY

if [[ "$RUN_MAIN_VALIDATION" == 1 ]]; then
  log "2/16 running the 200-frame 9/17/33 geometric validation"
  "${PYTHON_CMD[@]}" scripts/validate_occlusion_models.py \
    "${COMMON_VALIDATION_ARGS[@]}" \
    --out-dir "$MAIN_DIR" \
    --n-frames 200 \
    --ray-grids 9 17 33 \
    --z-modes raw ground_anchored
else
  log "2/16 reusing main validation (RUN_MAIN_VALIDATION=0)"
fi
for artifact in validation_config.json validation_summary.json validation_decisions.parquet validation_timings.parquet ego_exclusions.parquet; do
  require_file "${MAIN_DIR}/${artifact}"
done

"${PYTHON_CMD[@]}" - "${MAIN_DIR}/validation_config.json" "${MAIN_DIR}/validation_summary.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
summary = json.load(open(sys.argv[2], encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(f"main-validation contract failed: {message}")

expected_partial = [
    6674, 6675, 6747, 7025, 7435, 11077, 11155, 11166,
    11178, 11189, 11201, 11212, 11213, 11235, 11247, 40795,
]
require(config["ray_grids"] == [9, 17, 33], "ray grids are not 9/17/33")
require(config["z_modes"] == ["raw", "ground_anchored"], "z modes changed")
require(config["fov_degrees"] == [120.0, 360.0], "FOVs changed")
require(summary["sampled_frame_count"] == 200, "sample does not contain 200 frames")
require(summary["excluded_source_frames"] == expected_partial, "partial exclusions changed")
require(summary["config"] == config, "summary/config mismatch")
PY

if [[ "$RUN_MAIN_VALIDATION" == 1 ]]; then
  log "recording the 200-frame validation runtime environment"
  "${PYTHON_CMD[@]}" scripts/capture_runtime_environment.py \
    --label validation_main_200_final \
    --execution-mode "$PYTHON_EXECUTION_MODE" \
    --repository "$REPO_ROOT" \
    --numeric-producer-config "$CONFIG" \
    --out "$MAIN_RUNTIME_RECEIPT"
else
  log "reusing the 200-frame validation runtime receipt"
fi
require_file "$MAIN_RUNTIME_RECEIPT"

MAIN_LINEAGE_ARGS=(
  --stage validation_main_200_final
  --repository "$REPO_ROOT"
  --producer-commit "$NUMERIC_PRODUCER_COMMIT"
  --tracks "$TRACKS"
  --frame-index "$FRAME_INDEX"
  --input-provenance "$PROVENANCE_MANIFEST"
  --summary "${MAIN_DIR}/validation_summary.json"
  --config "${MAIN_DIR}/validation_config.json"
  --decisions "${MAIN_DIR}/validation_decisions.parquet"
  --timings "${MAIN_DIR}/validation_timings.parquet"
  --exclusions "${MAIN_DIR}/ego_exclusions.parquet"
  --out "$MAIN_SOURCE_LINEAGE_RECEIPT"
)
for frame in "${PARTIAL_FRAMES[@]}"; do
  MAIN_LINEAGE_ARGS+=(--exclude-source-frame "$frame")
done
if [[ "$RUN_MAIN_VALIDATION" == 1 ]]; then
  "${PYTHON_CMD[@]}" scripts/write_validation_lineage_receipt.py \
    "${MAIN_LINEAGE_ARGS[@]}" --write
fi
require_file "$MAIN_SOURCE_LINEAGE_RECEIPT"
"${PYTHON_CMD[@]}" scripts/write_validation_lineage_receipt.py \
  "${MAIN_LINEAGE_ARGS[@]}" --verify

if [[ "$RUN_CALIBRATION" == 1 ]]; then
  log "3/16 calibrating 1..15 sparse thresholds on 100 frames; evaluating once on 100 disjoint held-out frames"
  "${PYTHON_CMD[@]}" scripts/calibrate_sparse_threshold.py \
    --decisions "${MAIN_DIR}/validation_decisions.parquet" \
    --out-dir "$CALIBRATION_DIR" \
    --seed "$SEED" \
    --expected-frame-count 200 \
    --calibration-frame-count 100 \
    --fov-deg 120 360 \
    --expected-selected-min-rays "$SPARSE_MIN_VISIBLE_RAYS"
else
  log "3/16 reusing post-development calibration (RUN_CALIBRATION=0)"
fi
require_file "${CALIBRATION_DIR}/sparse_threshold_calibration.json"
require_file "${CALIBRATION_DIR}/validation_decisions_calibrated.parquet"

"${PYTHON_CMD[@]}" - \
  "${CALIBRATION_DIR}/sparse_threshold_calibration.json" \
  "${MAIN_DIR}/validation_decisions.parquet" \
  "${CALIBRATION_DIR}/validation_decisions_calibrated.parquet" <<'PY'
import hashlib
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
split = payload["split"]
calibration = [int(value) for value in split["calibration_frames"]]
held_out = [int(value) for value in split["held_out_frames"]]

def require(condition, message):
    if not condition:
        raise SystemExit(f"calibration contract failed: {message}")

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

require(len(calibration) == 100 and len(held_out) == 100, "expected a 100/100 split")
require(set(calibration).isdisjoint(held_out), "calibration and held-out frames overlap")
require(len(set(calibration) | set(held_out)) == 200, "split does not cover 200 frames")
require(int(payload["selection"]["selected_min_visible_rays"]) == 3,
        "selected threshold is not 3/15")
require(payload["selection"]["calibration"]["overall"]["n"] > 0,
        "calibration decision set is empty")
require(payload["selection"]["held_out"]["overall"]["n"] > 0,
        "held-out decision set is empty")
require(payload["protocol"]["development_status"] ==
        "post-development calibration with held-out evaluation",
        "development status is not explicit")
require(payload["input"]["sha256"] == sha256(sys.argv[2]),
        "main decisions changed after calibration")
require(payload["output"]["calibrated_decisions"]["sha256"] == sha256(sys.argv[3]),
        "calibrated decisions changed after calibration")
PY

if [[ "$RUN_CONVERGENCE" == 1 ]]; then
  log "4/16 running the separate 25-frame subset 17/33/65 convergence check"
  "${PYTHON_CMD[@]}" scripts/validate_occlusion_models.py \
    "${COMMON_VALIDATION_ARGS[@]}" \
    --out-dir "$CONVERGENCE_DIR" \
    --n-frames 25 \
    --ray-grids 17 33 65 \
    --z-modes ground_anchored
else
  log "4/16 reusing convergence analysis (RUN_CONVERGENCE=0)"
fi
for artifact in validation_config.json validation_summary.json validation_decisions.parquet validation_timings.parquet ego_exclusions.parquet; do
  require_file "${CONVERGENCE_DIR}/${artifact}"
done


"${PYTHON_CMD[@]}" - "${CONVERGENCE_DIR}/validation_config.json" "${CONVERGENCE_DIR}/validation_summary.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
summary = json.load(open(sys.argv[2], encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise SystemExit(f"convergence contract failed: {message}")

require(config["ray_grids"] == [17, 33, 65], "ray grids are not 17/33/65")
require(config["z_modes"] == ["ground_anchored"], "z mode is not ground_anchored")
require(config["fov_degrees"] == [120.0, 360.0], "FOVs changed")
require(summary["sampled_frame_count"] == 25, "sample does not contain 25 frames")
require(summary["config"] == config, "summary/config mismatch")
PY

if [[ "$RUN_CONVERGENCE" == 1 ]]; then
  log "recording the 25-frame convergence runtime environment"
  "${PYTHON_CMD[@]}" scripts/capture_runtime_environment.py \
    --label validation_convergence_25_final \
    --execution-mode "$PYTHON_EXECUTION_MODE" \
    --repository "$REPO_ROOT" \
    --numeric-producer-config "$CONFIG" \
    --out "$CONVERGENCE_RUNTIME_RECEIPT"
else
  log "reusing the 25-frame convergence runtime receipt"
fi
require_file "$CONVERGENCE_RUNTIME_RECEIPT"

CONVERGENCE_LINEAGE_ARGS=(
  --stage validation_convergence_25_final
  --repository "$REPO_ROOT"
  --producer-commit "$NUMERIC_PRODUCER_COMMIT"
  --tracks "$TRACKS"
  --frame-index "$FRAME_INDEX"
  --input-provenance "$PROVENANCE_MANIFEST"
  --summary "${CONVERGENCE_DIR}/validation_summary.json"
  --config "${CONVERGENCE_DIR}/validation_config.json"
  --decisions "${CONVERGENCE_DIR}/validation_decisions.parquet"
  --timings "${CONVERGENCE_DIR}/validation_timings.parquet"
  --exclusions "${CONVERGENCE_DIR}/ego_exclusions.parquet"
  --main-sample-summary "${MAIN_DIR}/validation_summary.json"
  --out "$CONVERGENCE_SOURCE_LINEAGE_RECEIPT"
)
for frame in "${PARTIAL_FRAMES[@]}"; do
  CONVERGENCE_LINEAGE_ARGS+=(--exclude-source-frame "$frame")
done
if [[ "$RUN_CONVERGENCE" == 1 ]]; then
  "${PYTHON_CMD[@]}" scripts/write_validation_lineage_receipt.py \
    "${CONVERGENCE_LINEAGE_ARGS[@]}" --write
fi
require_file "$CONVERGENCE_SOURCE_LINEAGE_RECEIPT"
"${PYTHON_CMD[@]}" scripts/write_validation_lineage_receipt.py \
  "${CONVERGENCE_LINEAGE_ARGS[@]}" --verify

if [[ "$RUN_VALIDATION_STRATA" == 1 ]]; then
  log "5/16 summarizing calibrated validation decisions without retracing rays"
  "${PYTHON_CMD[@]}" scripts/summarize_validation_strata.py \
    --decisions "${CALIBRATION_DIR}/validation_decisions_calibrated.parquet" \
    --timings "${MAIN_DIR}/validation_timings.parquet" \
    --out-dir "$STRATA_DIR" \
    --declared-sampled-frames 200
else
  log "5/16 reusing calibrated validation strata (RUN_VALIDATION_STRATA=0)"
fi
for artifact in validation_strata_summary.json validation_strata_metrics.parquet validation_timing_summary.parquet validation_blocker_classes.parquet source_lineage_receipt.json; do
  require_file "${STRATA_DIR}/${artifact}"
done

if [[ "$RUN_FULL_RECORD" == 1 ]]; then
  log "6/16 running all non-excluded frames with the frozen 3-of-15 rule"
  "${PYTHON_CMD[@]}" scripts/compute_full_record_observability.py \
    "${COMMON_FULL_ARGS[@]}" \
    "${ANALYSIS_EXCLUDE_ARGS[@]}" \
    "${FULL_OVERWRITE_ARGS[@]}" \
    --out-dir "$FULL_DIR" \
    --frame-exclusion-reason partial_scan_point_count_ratio_below_0.6_of_local_median \
    --z-modes ground_anchored
else
  log "6/16 reusing full-record output (RUN_FULL_RECORD=0)"
fi
for artifact in full_record_summary.json full_record_config.json full_record_input_audit.json full_record_observability.parquet observability_contributions.parquet ego_exclusions.parquet full_record_timings.parquet; do
  require_file "${FULL_DIR}/${artifact}"
done
if find "$FULL_DIR" -maxdepth 1 -type f -name '*.partial' -print -quit | grep -q .; then
  die "full-record output still contains an interrupted .partial file: $FULL_DIR"
fi

"${PYTHON_CMD[@]}" - "${FULL_DIR}/full_record_summary.json" "${FULL_DIR}/full_record_config.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
expected_partial = [
    6674, 6675, 6747, 7025, 7435, 11077, 11155, 11166,
    11178, 11189, 11201, 11212, 11213, 11235, 11247, 40795,
]

def require(condition, message):
    if not condition:
        raise SystemExit(f"full-record contract failed: {message}")

require(summary["processed_frame_count"] == summary["selected_frame_count"],
        "not all selected frames completed")
require(summary["frame_step"] == 1 and summary["max_frames"] is None,
        "primary run is decimated or truncated")
require(summary["input"]["quality_excluded_frames"]["requested"] == expected_partial,
        "partial-frame exclusion list changed")
require(config["geometry"]["sparse_min_visible_rays"] == 3,
        "sparse threshold is not 3/15")
require(config["geometry"]["z_modes"] == ["ground_anchored"],
        "primary z mode is not ground_anchored")
require(config["selection"]["frame_step"] == 1, "primary frame step is not one")
require(config["selection"]["max_frames"] is None, "primary run is max-frame truncated")
require(config["selection"]["only_frames"]["requested"] == [],
        "primary run selected only a frame subset")
require(config["selection"]["selected_frame_count"] == 40779,
        "primary selected-frame count is not 40,779")
PY

if [[ "$RUN_FULL_RECORD" == 1 ]]; then
  log "recording the full-record runtime environment"
  "${PYTHON_CMD[@]}" scripts/capture_runtime_environment.py \
    --label full_record_primary_40779 \
    --execution-mode "$PYTHON_EXECUTION_MODE" \
    --repository "$REPO_ROOT" \
    --numeric-producer-config "$CONFIG" \
    --out "$FULL_RUNTIME_RECEIPT"
else
  log "reusing full-record runtime receipt"
fi
require_file "$FULL_RUNTIME_RECEIPT"

if [[ "$RUN_BOOTSTRAP" == 1 ]]; then
  log "7/16 running the nominal 120-second (1,200 retained observations) circular moving-block bootstrap"
  "${PYTHON_CMD[@]}" scripts/bootstrap_observability.py \
    --input "${FULL_DIR}/observability_contributions.parquet" \
    --source-summary "${FULL_DIR}/full_record_summary.json" \
    --out-dir "$BOOTSTRAP_DIR" \
    --frame-rate-hz 10 \
    --block-seconds 120 \
    --n-resamples 5000 \
    --seed "$SEED" \
    --paired-fovs 120 360
else
  log "7/16 reusing primary bootstrap (RUN_BOOTSTRAP=0)"
fi
require_file "${BOOTSTRAP_DIR}/observability_bootstrap.json"
require_file "${BOOTSTRAP_DIR}/observability_bootstrap_replicates.parquet"

if [[ "$RUN_BLOCK_SENSITIVITY" == 1 ]]; then
  log "8/16 running 30/60/120/240-second block-duration sensitivity"
  mkdir -p "$BLOCK_SENSITIVITY_DIR"
  "${PYTHON_CMD[@]}" scripts/bootstrap_block_sensitivity.py \
    --input "${FULL_DIR}/observability_contributions.parquet" \
    --source-summary "${FULL_DIR}/full_record_summary.json" \
    --output "$BLOCK_SENSITIVITY_JSON" \
    --block-seconds 30 60 120 240 \
    --reference-block-seconds 120 \
    --frame-rate-hz 10 \
    --n-resamples 5000 \
    --seed "$SEED" \
    --paired-fovs 120 360
else
  log "8/16 reusing block-duration sensitivity (RUN_BLOCK_SENSITIVITY=0)"
fi
require_file "$BLOCK_SENSITIVITY_JSON"

if [[ "$RUN_RANKING" == 1 ]]; then
  log "9/16 ranking residual-demand hotspots with 1,200-frame clusters"
  "${PYTHON_CMD[@]}" scripts/rank_residual_demand.py \
    --input "${FULL_DIR}/full_record_observability.parquet" \
    --source-summary "${FULL_DIR}/full_record_summary.json" \
    --out-dir "$RANK_DIR" \
    --fov-deg 120 360 \
    --block-width-frames 1200 \
    --n-bootstrap 5000 \
    --seed "$SEED" \
    --expected-ego-count 60
else
  log "9/16 reusing residual-demand ranking (RUN_RANKING=0)"
fi
require_file "${RANK_DIR}/residual_hotspot_ranks.json"
require_file "${RANK_DIR}/residual_hotspot_ranks.parquet"

# These two compact recomputations supply the declared vertical-coordinate and
# partial-scan sensitivities.  They remain detected-dynamic-box analyses; they
# do not add human review, a static-scene mesh, or surveyed infrastructure.
if [[ "$RUN_Z_MODE_SENSITIVITY" == 1 ]]; then
  log "10/16 running the deterministic every-10th-frame raw/ground z sensitivity"
  "${PYTHON_CMD[@]}" scripts/compute_full_record_observability.py \
    "${COMMON_FULL_ARGS[@]}" \
    "${ANALYSIS_EXCLUDE_ARGS[@]}" \
    "${SENSITIVITY_OVERWRITE_ARGS[@]}" \
    --out-dir "$Z_SENSITIVITY_DIR" \
    --frame-exclusion-reason partial_scan_point_count_ratio_below_0.6_of_local_median \
    --frame-step 10 \
    --z-modes ground_anchored raw
  "${PYTHON_CMD[@]}" scripts/bootstrap_observability.py \
    --input "${Z_SENSITIVITY_DIR}/observability_contributions.parquet" \
    --source-summary "${Z_SENSITIVITY_DIR}/full_record_summary.json" \
    --out-dir "${Z_SENSITIVITY_DIR}/bootstrap_120s" \
    --frame-rate-hz 1 \
    --block-seconds 120 \
    --n-resamples 5000 \
    --seed "$SEED" \
    --paired-fovs 120 360
else
  log "10/16 reusing raw/ground z sensitivity (RUN_Z_MODE_SENSITIVITY=0)"
fi
require_file "${Z_SENSITIVITY_DIR}/full_record_summary.json"
require_file "${Z_SENSITIVITY_DIR}/bootstrap_120s/observability_bootstrap_replicates.parquet"

if [[ "$RUN_PARTIAL_SCAN_SENSITIVITY" == 1 ]]; then
  log "11/16 evaluating the 15 observed partial scans (frame 6747 has no track row)"
  "${PYTHON_CMD[@]}" scripts/compute_full_record_observability.py \
    "${COMMON_FULL_ARGS[@]}" \
    "${ONLY_PARTIAL_ARGS[@]}" \
    "${SENSITIVITY_OVERWRITE_ARGS[@]}" \
    --out-dir "$PARTIAL_SENSITIVITY_DIR" \
    --z-modes ground_anchored
else
  log "11/16 reusing partial-scan sensitivity (RUN_PARTIAL_SCAN_SENSITIVITY=0)"
fi
require_file "${PARTIAL_SENSITIVITY_DIR}/full_record_summary.json"

if [[ "$RUN_SENSITIVITY_SUMMARY" == 1 ]]; then
  log "12/16 combining primary, z-mode, and partial-scan sensitivity evidence"
  require_file scripts/summarize_cacie_sensitivities.py
  "${PYTHON_CMD[@]}" scripts/summarize_cacie_sensitivities.py \
    --primary-summary "${FULL_DIR}/full_record_summary.json" \
    --primary-bootstrap "${BOOTSTRAP_DIR}/observability_bootstrap.json" \
    --z-summary "${Z_SENSITIVITY_DIR}/full_record_summary.json" \
    --z-bootstrap-replicates "${Z_SENSITIVITY_DIR}/bootstrap_120s/observability_bootstrap_replicates.parquet" \
    --partial-summary "${PARTIAL_SENSITIVITY_DIR}/full_record_summary.json" \
    --out "$SENSITIVITY_SUMMARY"
else
  log "12/16 reusing sensitivity summary when required downstream (RUN_SENSITIVITY_SUMMARY=0)"
fi
if [[ "$RUN_SENSITIVITY_SUMMARY" == 1 || "$RUN_FULL_FIGURE" == 1 || "$RUN_RELEASE_MANIFEST" == 1 ]]; then
  require_file "$SENSITIVITY_SUMMARY"
fi

if [[ "$RUN_VALIDATION_FIGURE" == 1 ]]; then
  log "13/16 generating the source-backed held-out validation figure"
  "${PYTHON_CMD[@]}" scripts/generate_cacie_validation_figure.py \
    --summary "${MAIN_DIR}/validation_summary.json" \
    --strata-summary "${STRATA_DIR}/validation_strata_summary.json" \
    --decisions "${MAIN_DIR}/validation_decisions.parquet" \
    --calibration-summary "${CALIBRATION_DIR}/sparse_threshold_calibration.json" \
    --convergence-summary "${CONVERGENCE_DIR}/validation_summary.json" \
    --out "$VALIDATION_FIGURE_BASE" \
    --z-mode ground_anchored
else
  log "13/16 reusing validation figure (RUN_VALIDATION_FIGURE=0)"
fi
require_file "${VALIDATION_FIGURE_BASE}.pdf"
require_file "${VALIDATION_FIGURE_BASE}.png"
require_file "${VALIDATION_FIGURE_BASE}.source_lineage_receipt.json"

if [[ "$RUN_FULL_FIGURE" == 1 ]]; then
  log "14/16 generating the source-audited full-record results figure"
  require_file "$FULL_FIGURE_SCRIPT"
  "${PYTHON_CMD[@]}" "$FULL_FIGURE_SCRIPT" \
    --full-summary "${FULL_DIR}/full_record_summary.json" \
    --primary-bootstrap "${BOOTSTRAP_DIR}/observability_bootstrap.json" \
    --residual-summary "${RANK_DIR}/residual_hotspot_ranks.json" \
    --residual-ranking "${RANK_DIR}/residual_hotspot_ranks.parquet" \
    --sensitivity-summary "$SENSITIVITY_SUMMARY" \
    --block-sensitivity "$BLOCK_SENSITIVITY_JSON" \
    --out "$FULL_FIGURE_BASE"
else
  log "14/16 reusing full-record results figure (RUN_FULL_FIGURE=0)"
fi
require_file "${FULL_FIGURE_BASE}.pdf"
require_file "${FULL_FIGURE_BASE}.png"
require_file "${FULL_FIGURE_BASE}.source_lineage_receipt.json"

if [[ "$RUN_NOTEBOOK" == 1 ]]; then
  log "15/16 executing the fail-closed notebook audit after all numerical artifacts"
  command -v "$UV_BIN" >/dev/null 2>&1 || die "uv executable not found: $UV_BIN"
  require_file "$NOTEBOOK"
  mkdir -p "$(dirname -- "$EXECUTED_NOTEBOOK")"
  if [[ "$OUTPUT_ROOT" == /* ]]; then
    OUTPUT_ROOT_ABSOLUTE="$OUTPUT_ROOT"
  else
    OUTPUT_ROOT_ABSOLUTE="${REPO_ROOT}/${OUTPUT_ROOT}"
  fi
  if [[ "$EXECUTED_NOTEBOOK" == /* ]]; then
    EXECUTED_NOTEBOOK_ABSOLUTE="$EXECUTED_NOTEBOOK"
  else
    EXECUTED_NOTEBOOK_ABSOLUTE="${REPO_ROOT}/${EXECUTED_NOTEBOOK}"
  fi
  CACIE_OUTPUT_ROOT="$OUTPUT_ROOT_ABSOLUTE" \
    "$UV_BIN" run --frozen --extra dev jupyter execute "$NOTEBOOK" \
    --output="$EXECUTED_NOTEBOOK_ABSOLUTE"
  "${PYTHON_CMD[@]}" scripts/audit_cacie_notebook.py \
    --repo-root "$REPO_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --source-notebook "$NOTEBOOK" \
    --executed-notebook "$EXECUTED_NOTEBOOK" \
    --receipt "$NOTEBOOK_AUDIT_RECEIPT" \
    --write-receipt
else
  log "15/16 reusing the immutable executed notebook and audit receipt"
fi
require_file "$NOTEBOOK"
require_file "$EXECUTED_NOTEBOOK"
require_file "$NOTEBOOK_AUDIT_RECEIPT"
# This recomputes every audited SHA even immediately after creation.  In
# RUN_NOTEBOOK=0 mode it is what prevents a stale PASS notebook/receipt pair
# from being reused after any source or numerical artifact changed.
"${PYTHON_CMD[@]}" scripts/audit_cacie_notebook.py \
  --repo-root "$REPO_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --source-notebook "$NOTEBOOK" \
  --executed-notebook "$EXECUTED_NOTEBOOK" \
  --receipt "$NOTEBOOK_AUDIT_RECEIPT" \
  --verify-receipt

if [[ "$RUN_RELEASE_MANIFEST" == 1 ]]; then
  log "16/16 hashing the explicit release artifact set"
  RELEASE_INPUTS=(
    README.md
    pyproject.toml
    uv.lock
    docs
    "$CONFIG"
    "$TRACKS"
    "$FRAME_INDEX"
    "$PROVENANCE_MANIFEST"
    "$UPSTREAM_MANIFEST"
    "$MAIN_DIR"
    "$CALIBRATION_DIR"
    "$CONVERGENCE_DIR"
    "$STRATA_DIR"
    "$FULL_DIR"
    "$BOOTSTRAP_DIR"
    "$BLOCK_SENSITIVITY_JSON"
    "$RANK_DIR"
    "$Z_SENSITIVITY_DIR"
    "$PARTIAL_SENSITIVITY_DIR"
    "$SENSITIVITY_SUMMARY"
    "${VALIDATION_FIGURE_BASE}.pdf"
    "${VALIDATION_FIGURE_BASE}.png"
    "${VALIDATION_FIGURE_BASE}.source_lineage_receipt.json"
    "${FULL_FIGURE_BASE}.pdf"
    "${FULL_FIGURE_BASE}.png"
    "${FULL_FIGURE_BASE}.source_lineage_receipt.json"
    "$NOTEBOOK"
    "$EXECUTED_NOTEBOOK"
    "$NOTEBOOK_AUDIT_RECEIPT"
    "$MAIN_RUNTIME_RECEIPT"
    "$CONVERGENCE_RUNTIME_RECEIPT"
    "$FULL_RUNTIME_RECEIPT"
    "${DETECTOR_VALIDATION_DIR}/README.md"
    "${DETECTOR_VALIDATION_DIR}/evaluate_matched_box_geometry.py"
    "${DETECTOR_VALIDATION_DIR}/full_holdout"
    "${DETECTOR_VALIDATION_DIR}/requested_window"
    "${DETECTOR_VALIDATION_DIR}/sensitivity"
    "${DETECTOR_VALIDATION_DIR}/source_receipts"
  )
  # Add every producing source/script file explicitly so transient Python cache
  # files are never promoted into the release inventory.
  while IFS= read -r -d '' producer_file; do
    RELEASE_INPUTS+=("$producer_file")
  done < <(
    find src/paco_observability scripts -type f \
      ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name '*.pyo' -print0
  )
  command -v "$UV_BIN" >/dev/null 2>&1 || die "uv executable not found: $UV_BIN"
  "$UV_BIN" run --frozen --extra dev python scripts/build_release_artifact_manifest.py \
    --base-dir "$REPO_ROOT" \
    --out "$RELEASE_MANIFEST" \
    --require-clean-git \
    --numeric-producer-config "$CONFIG" \
    --require-numeric-producer \
    "${RELEASE_INPUTS[@]}"
else
  log "16/16 release manifest disabled (RUN_RELEASE_MANIFEST=0)"
fi

if [[ "$RUN_RELEASE_MANIFEST" == 1 ]]; then
  require_file "$RELEASE_MANIFEST"
fi
log "canonical CACIE v1.1 workflow completed"
