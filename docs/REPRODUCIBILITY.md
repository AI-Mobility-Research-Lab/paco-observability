# Reproducibility map

## Evidence tiers

The repository supports two deliberately separate activities:

- **Archival v1.0 reproduction:** `scripts/download_dataverse.py` downloads
  deposited derived outputs for supported historical figures. These files are
  legacy artifacts and cannot support new v1.1 conclusions.
- **PACO v1.1 CACIE recomputation:** a governed `tracks.parquet` first passes the
  provenance gate, after which validation, full-record analysis, bootstrap
  inference, and residual-demand hotspot ranking are run under frozen parameters.

Do not use an archival figure matching an old manuscript value as validation of
the v1.1 implementation.

## v1.1 script map

| CACIE stage | Script | Principal outputs or role |
|---|---|---|
| Provenance gate | `scripts/audit_input_provenance.py` | Input audit and SHA-256/Git/runtime manifest |
| Validation sampling and model comparison | `scripts/validate_occlusion_models.py` | Per-target decisions, exclusions, timings, configuration, and summary |
| Post-development sparse calibration | `scripts/calibrate_sparse_threshold.py` | Deterministic 100/100 calibration/held-out split, 1--15 threshold search, and calibrated decisions |
| Validation result re-summary | `scripts/summarize_validation_strata.py` | Covered-row confusion metrics and declared strata; no ray recomputation |
| Validation figure | `scripts/generate_cacie_validation_figure.py` | Source-backed accuracy, observability, runtime, and convergence panels |
| Full-record low-cost sweep | `scripts/compute_full_record_observability.py` | Bounded-memory, ordered Parquet outputs plus bootstrap-ready contributions |
| Moving-block inference | `scripts/bootstrap_observability.py` | Bootstrap summary and replicate table from frame-level contributions |
| Block-duration sensitivity | `scripts/bootstrap_block_sensitivity.py` | 30/60/120/240-second interval sensitivity from the same frame contributions |
| Residual-demand hotspot ranking | `scripts/rank_residual_demand.py` | Ground-anchored calibrated 3-of-15-ray hotspot scores and block-bootstrap rank stability |
| Sensitivity synthesis | `scripts/summarize_cacie_sensitivities.py` | Strict post-processing of primary, vertical-coordinate, decimation, and partial-scan artifacts |
| Full-results figure | `scripts/generate_cacie_full_results_figure.py` | Source-audited primary intervals, residual-demand hotspots, and declared sensitivities |
| Executed artifact audit | `notebooks/cacie_validation.ipynb`, `scripts/audit_cacie_notebook.py` | Immutable source, separate executed copy, unique PASS marker, and current-byte receipt after all stages finish |
| Runtime receipts | `scripts/capture_runtime_environment.py` | Three stage-local allow-listed hardware/package/Git receipts, including explicit unverified no-Git source-tree declarations |
| Lineage receipts | `scripts/write_validation_lineage_receipt.py`, figure/strata producers | Portable relative-path SHA-256 bindings for reused validation artifacts and every cheap derived output |
| Release inventory | `scripts/build_release_artifact_manifest.py` | Clean-Git enforcement, producing code/docs, detector evidence, double SHA-256 pass, and runtime versions |
| Future illustrative external coverage | `scripts/optimize_external_coverage.py` | Non-canonical scenario tool; hypothetical candidates and greedy coverage only |
| Unit verification | `pytest` | Synthetic geometry, provenance, validation, bootstrap, and optimization tests |

`scripts/run_canonical.sh` is the authoritative v1.1 orchestration entry point.
It contains no conflict-event or review-ledger stage and does not infer human
review, static-scene geometry, or surveyed infrastructure from unavailable data.

## Frozen paper-facing defaults

These are method parameters, not final findings.
`configs/cacie_v1.1.json` is the frozen machine-readable analysis protocol and
run configuration; the formal commands must override smoke-test CLI defaults
where the two differ. It is not evidence of preregistration. In particular, the
3/15 threshold and deterministic calibration split were added post-development
after the any-ray diagnostic was found to be optimistic.

| Parameter | Frozen/default design |
|---|---|
| Ground plane, `n dot p + d = 0` | `(-0.001561196615141432, -0.06019325942675009, 0.9981855209252, 2.907646656036377)` |
| Vertical-coordinate modes | `raw` and `ground_anchored` |
| Onboard sensor height above local plane | 1.8 m |
| Ego positions | 15 per approach |
| Validation FOVs | 120 and 360 degrees |
| Maximum/near-blind range | 35 m / 1.5 m |
| Hypothetical-ego clearance | 4.0 m |
| Low-cost angular resolution | 0.25 degrees across each FOV |
| Occlusion clearance | 0.0 m |
| Visible-fraction threshold | 0.05 |
| Deterministic seed | `20260902` |
| Bootstrap | circular blocks of 1,200 consecutive retained-frame observations, nominally 120 seconds at 10 Hz (source-index gaps may extend elapsed time), 5,000 replicates, 95% percentile intervals |
| Partial-scan rule | exclude every frame with `num_points / median(available preceding/following 3 source rows) < 0.6`; boundary frames use only existing neighbors; fail closed if a detected frame is undeclared |

The 0.25-degree setting fixes angular resolution rather than fixing one bin
count for every FOV.

For the frozen upstream run, the manifest's stage-level `tracking_method` and
tracking parameters are authoritative because they were written by the stage
that produced `tracks.parquet`: `sort_bev`, center reference, 6 m association
distance, and maximum age 3.  The nested
`effective_run_config.tracking.track_method=bev_iou_hungarian_baseline` is an
earlier orchestration default retained in the manifest and was superseded by
that separately executed tracking stage.

## Geometric-reference and convergence design

The validation hierarchy and frozen grid design are:

1. the planar angular-wedge approximation at 0.25-degree resolution;
2. low-cost center/top ray checks;
3. a **15-ray sparse surrogate** consisting of the target center, six face
   centers, and eight vertices;
4. **9 x 9, 17 x 17, and 33 x 33 projected angular grids** on the frozen
   200-frame main validation sample, with 33 x 33 supplying the primary
   within-abstraction reference; and
5. a separate **17 x 17, 33 x 33, and 65 x 65** convergence run on a frozen
   25-frame subset of the 200 main frames (not an independent sample).

The sparse binary rule is **at least 3 of 15 rays visible**, not an any-ray
rule. After the any-ray diagnostic appeared optimistic, the 200 main frames
were split once by deterministic time-by-density stratification into 100
calibration frames and 100 held-out frames. This threshold search and split are
post-development, not preregistered or prespecified. Candidate thresholds 1
through 15 are compared only on the calibration half by covered-row accuracy;
ties are resolved first by smaller observability-ratio bias and then in the
more conservative direction. The held-out half is used once to report
generalization. The selected 3/15 rule is then frozen for the full-record run.

The formal main run explicitly passes `--n-frames 200 --ray-grids 9 17 33`; the
convergence run explicitly passes `--n-frames 25 --ray-grids 17 33 65`. The
CLI's single-grid default of 17 is only a smoke default, not the frozen
paper configuration. Agreement and runtime are reported by method,
vertical-coordinate mode, FOV, class, and range without calling any ray-grid
output physical ground truth.

Every ray uses analytic ray--oriented-box intersection. “Exact” means exact for
that ray against the supplied oriented boxes. The scene may still omit objects
or contain detector, tracker, calibration, alignment, and box-shape errors.

## Required execution order

1. Audit the governed trajectory input and source-frame index with
   `scripts/audit_input_provenance.py`; retain its manifest and fail closed.
   Partial scans are detected from point counts, not from a zero timestamp.
   Every detected partial frame must appear in the declared exclusion list.
   `--allow-missing-frame` only justifies an expected frame that has no track
   row; it does not remove an observed partial frame from an analysis.
2. Run `scripts/validate_occlusion_models.py` on the frozen stratified frame
   sample with both vertical-coordinate modes. Use the explicit 200-frame
   9/17/33 main comparison and the separate 25-frame-subset 17/33/65 convergence run
   described above.
3. Run the post-development `scripts/calibrate_sparse_threshold.py` once on the
   saved 200-frame decisions. Freeze its 100/100 calibration/held-out assignment
   and selected 3-of-15 rule; do not select a threshold from the held-out scores
   or describe this process as preregistered.
4. Run `scripts/summarize_validation_strata.py` on
   `validation_decisions_calibrated.parquet` and the saved main timings. This
   produces covered-row confusion counts and descriptive strata without
   tracing any new rays or treating repeated rows as independent.
5. Run `scripts/compute_full_record_observability.py` over every non-excluded
   frame in the gate-passed record. The completion marker is
   `full_record_summary.json`; a directory containing only uniquely named
   `.partial` files is an interrupted run, not a usable result. The script
   writes both `full_record_observability.parquet` and the bootstrap-ready
   `observability_contributions.parquet`; no ad hoc conversion is needed.
6. Run `scripts/bootstrap_observability.py` on
   `observability_contributions.parquet`. It first aggregates all repeated
   ego rows within a frame and then resamples circular blocks of 1,200
   consecutive retained-frame observations with seed `20260902` (nominally
   120 seconds at 10 Hz; source-index gaps may extend elapsed time). The
   360-minus-120 FOV comparison uses identical aligned blocks.
7. Run `scripts/rank_residual_demand.py` on
   `full_record_observability.parquet`. The primary ranking is the
   ground-anchored, calibrated 3-of-15-ray, uniform residual-demand ratio; VRU
   weighting is a sensitivity analysis. Its source-index blocks of width 1,200
   preserve the frozen nominal 120-second dependence scale.
8. Run `scripts/bootstrap_block_sensitivity.py`, the declared vertical-coordinate
   and partial-scan sensitivity computations, and
   `scripts/summarize_cacie_sensitivities.py`. The synthesis must rebuild point
   estimates from saved counts and use paired bootstrap draws; it must not fill
   missing inputs with inferred values.
9. Generate the validation and full-results figures, then execute
   `notebooks/cacie_validation.ipynb` after every numerical artifact exists.
   Keep that tracked source notebook unmodified: write the executed copy and
   machine-readable receipt below `outputs/canonical/notebook/`. Every required
   stage must be `VERIFIED`, and the executed output must contain exactly one
   `CACIE_NOTEBOOK_AUDIT_PASS_V1` marker. Reuse re-hashes all current bytes.
   Finish by building the explicit release manifest; any missing, stale, or
   lineage-mismatched artifact stops the run.
10. Treat `scripts/optimize_external_coverage.py` as an **illustrative future
   extension**, not a current canonical CACIE result. It cannot turn an
   ego-grid hotspot into a surveyed or deployment-optimal pole location.

Each published table or figure should identify the input manifest, producing
command/configuration, analysis Git revision, and output checksum.

## Bootstrap estimand

The full-record estimand is a ratio of sums:

```text
sum(frame numerator) / sum(frame denominator)
```

It is not an unweighted mean of frame ratios, and ego positions from one frame
are not treated as independent samples. The default block contains 1,200
consecutive retained-frame observations, nominally 120 seconds at 10 Hz;
source-index gaps can extend the elapsed span. Paired comparisons reuse the
same sampled blocks.

## Canonical command skeleton

For a complete run, use:

```bash
bash scripts/run_canonical.sh
```

All stages run by default. Every stage has a `RUN_*` environment switch. Setting
a switch to `0` reuses an existing artifact; the runner still checks that
artifact before allowing downstream work. For
example, after an interrupted downstream analysis, preserve the completed full
record and use `RUN_FULL_RECORD=0`. Existing full-record directories are not
overwritten unless `OVERWRITE_FULL_RECORD=1` (or
`OVERWRITE_SENSITIVITY=1` for sensitivity runs) is explicitly set. Paths and
worker counts and the output root can be configured with variables defined at
the top of the script. The canonical stage names below that root are fixed so
the notebook cannot compute into one layout and audit another.

`RUN_RELEASE_MANIFEST=1` additionally requires a clean worktree both before
numerical work and during the two manifest hash passes. The release inventory
contains the numerical artifacts, internal detector-validation evidence,
runtime receipts, producing package/scripts, lockfile, README/protocol docs,
notebook source, executed notebook, notebook receipt, and all stage-lineage
receipts. The required
`release.numeric_producer_commit`/`release.numeric_producer_files` declaration
binds current numerical core files to an existing Git commit. The historical
full-record entrypoint is checked directly against its declared Git blob; none
of these checks modifies the older input-provenance record.

The following Bash arrays make the frame-quality contract explicit. The
frozen list is also recorded in `configs/cacie_v1.1.json` and must match the
input-provenance manifest.

```bash
PARTIAL_FRAMES=(
  6674 6675 6747 7025 7435 11077 11155 11166
  11178 11189 11201 11212 11213 11235 11247 40795
)
AUDIT_PARTIAL_ARGS=()
ANALYSIS_EXCLUDE_ARGS=()
for frame in "${PARTIAL_FRAMES[@]}"; do
  AUDIT_PARTIAL_ARGS+=(--exclude-partial-frame "$frame")
  ANALYSIS_EXCLUDE_ARGS+=(--exclude-frame "$frame")
done

python scripts/audit_input_provenance.py \
  --tracks data/inputs/canonical/tracks.parquet \
  --frame-index data/inputs/canonical/frame_index.parquet \
  --upstream-manifest outputs/canonical/upstream/manifest.json \
  --output outputs/canonical/input_provenance.json \
  --expected-frame-range 1 40795 \
  --allow-missing-frame 6747 \
  "${AUDIT_PARTIAL_ARGS[@]}" \
  --ground-plane -0.0015611966 -0.0601932594 0.9981855209 2.907646656
```

The main and convergence validation runs differ in both sample size and grids:

```bash
COMMON_VALIDATION_ARGS=(
  --tracks data/inputs/canonical/tracks.parquet
  --expected-frame-range 1 40795
  --allow-missing-frame 6747
  "${ANALYSIS_EXCLUDE_ARGS[@]}"
  --ground-plane -0.0015611966 -0.0601932594 0.9981855209 2.907646656
  --n-ego-per-approach 15 --fov-deg 120 360
  --sparse-min-visible-rays 3 --seed 20260902
)

python scripts/validate_occlusion_models.py \
  "${COMMON_VALIDATION_ARGS[@]}" \
  --out-dir outputs/canonical/validation_main \
  --n-frames 200 --ray-grids 9 17 33 \
  --z-modes raw ground_anchored

python scripts/calibrate_sparse_threshold.py \
  --decisions outputs/canonical/validation_main/validation_decisions.parquet \
  --out-dir outputs/canonical/sparse_calibration \
  --seed 20260902 --expected-frame-count 200 \
  --calibration-frame-count 100 --fov-deg 120 360 \
  --expected-selected-min-rays 3

python scripts/validate_occlusion_models.py \
  "${COMMON_VALIDATION_ARGS[@]}" \
  --out-dir outputs/canonical/validation_convergence \
  --n-frames 25 --ray-grids 17 33 65 \
  --z-modes ground_anchored

python scripts/summarize_validation_strata.py \
  --decisions outputs/canonical/sparse_calibration/validation_decisions_calibrated.parquet \
  --timings outputs/canonical/validation_main/validation_timings.parquet \
  --out-dir outputs/canonical/validation_strata_calibrated \
  --declared-sampled-frames 200
```

The full-record script directly emits the two tables consumed downstream:

```bash
python scripts/compute_full_record_observability.py \
  --tracks data/inputs/canonical/tracks.parquet \
  --out-dir outputs/canonical/full_record_primary \
  --ground-plane -0.0015611966 -0.0601932594 0.9981855209 2.907646656 \
  --expected-frame-range 1 40795 --allow-missing-frame 6747 \
  "${ANALYSIS_EXCLUDE_ARGS[@]}" \
  --z-modes ground_anchored --fov-deg 120 360 \
  --sparse-min-visible-rays 3 \
  --n-ego-per-approach 15 --vru-weight 3 --non-vru-weight 1

python scripts/bootstrap_observability.py \
  --input outputs/canonical/full_record_primary/observability_contributions.parquet \
  --source-summary outputs/canonical/full_record_primary/full_record_summary.json \
  --out-dir outputs/canonical/bootstrap_primary \
  --frame-rate-hz 10 --block-seconds 120 --n-resamples 5000 \
  --seed 20260902 --paired-fovs 120 360

python scripts/bootstrap_block_sensitivity.py \
  --input outputs/canonical/full_record_primary/observability_contributions.parquet \
  --source-summary outputs/canonical/full_record_primary/full_record_summary.json \
  --output outputs/canonical/bootstrap_block_sensitivity/observability_block_sensitivity.json \
  --block-seconds 30 60 120 240 --reference-block-seconds 120 \
  --frame-rate-hz 10 --n-resamples 5000 --seed 20260902 \
  --paired-fovs 120 360

python scripts/rank_residual_demand.py \
  --input outputs/canonical/full_record_primary/full_record_observability.parquet \
  --source-summary outputs/canonical/full_record_primary/full_record_summary.json \
  --out-dir outputs/canonical/residual_demand \
  --fov-deg 120 360 --block-width-frames 1200 \
  --n-bootstrap 5000 --seed 20260902 --expected-ego-count 60
```

The runner additionally produces the every-tenth-frame raw/ground sensitivity,
its 1 Hz paired 120-second bootstrap, and the declared partial-scan-only run,
then combines them without recomputing visibility:

```bash
python scripts/summarize_cacie_sensitivities.py \
  --primary-summary outputs/canonical/full_record_primary/full_record_summary.json \
  --primary-bootstrap outputs/canonical/bootstrap_primary/observability_bootstrap.json \
  --z-summary outputs/canonical/z_mode_sensitivity_step10/full_record_summary.json \
  --z-bootstrap-replicates outputs/canonical/z_mode_sensitivity_step10/bootstrap_120s/observability_bootstrap_replicates.parquet \
  --partial-summary outputs/canonical/partial_scan_only/full_record_summary.json \
  --out outputs/canonical/sensitivity_summary/cacie_sensitivity_summary.json

python scripts/generate_cacie_validation_figure.py \
  --summary outputs/canonical/validation_main/validation_summary.json \
  --strata-summary outputs/canonical/validation_strata_calibrated/validation_strata_summary.json \
  --decisions outputs/canonical/validation_main/validation_decisions.parquet \
  --calibration-summary outputs/canonical/sparse_calibration/sparse_threshold_calibration.json \
  --convergence-summary outputs/canonical/validation_convergence/validation_summary.json \
  --out outputs/canonical/figures/cacie_validation \
  --z-mode ground_anchored

python scripts/generate_cacie_full_results_figure.py \
  --full-summary outputs/canonical/full_record_primary/full_record_summary.json \
  --primary-bootstrap outputs/canonical/bootstrap_primary/observability_bootstrap.json \
  --residual-summary outputs/canonical/residual_demand/residual_hotspot_ranks.json \
  --residual-ranking outputs/canonical/residual_demand/residual_hotspot_ranks.parquet \
  --sensitivity-summary outputs/canonical/sensitivity_summary/cacie_sensitivity_summary.json \
  --block-sensitivity outputs/canonical/bootstrap_block_sensitivity/observability_block_sensitivity.json \
  --out outputs/canonical/figures/cacie_full_results

mkdir -p outputs/canonical/notebook
CACIE_OUTPUT_ROOT="$PWD/outputs/canonical" \
  uv run --frozen --extra dev jupyter execute \
  notebooks/cacie_validation.ipynb \
  --output="$PWD/outputs/canonical/notebook/cacie_validation.executed.ipynb"

python scripts/audit_cacie_notebook.py \
  --repo-root "$PWD" \
  --output-root outputs/canonical \
  --source-notebook notebooks/cacie_validation.ipynb \
  --executed-notebook outputs/canonical/notebook/cacie_validation.executed.ipynb \
  --receipt outputs/canonical/notebook/notebook_audit_receipt.json \
  --write-receipt

python scripts/audit_cacie_notebook.py \
  --repo-root "$PWD" \
  --output-root outputs/canonical \
  --source-notebook notebooks/cacie_validation.ipynb \
  --executed-notebook outputs/canonical/notebook/cacie_validation.executed.ipynb \
  --receipt outputs/canonical/notebook/notebook_audit_receipt.json \
  --verify-receipt
```

The notebook is deliberately last among the analytical checks. The source file
remains unexecuted and reviewable; only the output copy contains results. The
release manifest is then built from the explicit artifact list in
`scripts/run_canonical.sh`; do not recursively include the manifest itself.

## Residual demand and future illustrative coverage

`scripts/rank_residual_demand.py` is the current canonical decision-layer
analysis. It ranks the fixed ego grid by a ratio of summed residual demand to
summed detected-box demand and reports block-bootstrap rank stability. A high
score identifies a modeled residual-demand hotspot, not crash risk or an
infrastructure recommendation.

`scripts/optimize_external_coverage.py` evaluates finite hypothetical external
viewpoints against residual onboard demand and records a deterministic greedy
maximum-coverage trace. It is retained only as an illustrative/future
extension. Its default coordinates are computational stress-test points, not
surveyed pole sites, and its output is not required for canonical CACIE claims.

The stage does not model all static occluders, communication reliability,
latency, packet loss, detector performance, installation feasibility, or life
cycle cost. Its outputs support transparent prioritization logic only; they do
not establish operational V2I performance or an optimal deployment.

The separate theoretical cooperative ceiling assigns visibility one by
definition. It is useful for decomposing residual demand but must never be
reported as measured V2I coverage.

## Archival legacy figure map

The following scripts remain useful for historical v1.0 figure reproduction
when paired with the deposited legacy derived outputs:

| Legacy item | Producing script | Status |
|---|---|---|
| Methodology schematic | `scripts/generate_methodology_figure.py` | Data-free schematic; not validation |
| Object decomposition | `scripts/generate_object_counts_figure.py` | Legacy derived-output reproduction only |
| Candidate ego-partner summary | `scripts/generate_ego_near_miss_figure.py` | Legacy and not evidence of human review |
| FOV/range sensitivity | `scripts/generate_sensitivity_figure.py` | Legacy derived-output reproduction only |
| First-order planar/vertical comparison | `scripts/compute_3d_occlusion_comparison.py` | Legacy diagnostic; not the v1.1 exact reference |
| Confidence filtering | `scripts/compute_confidence_robustness.py` | Legacy proxy analysis; not onboard/RSU operational calibration |

Matching these figures does not pass the v1.1 provenance or validation gate.
