# CACIE validation protocol

## Purpose and claim boundary

This protocol records how PACO v1.1 evaluates low-cost observability models,
quantifies full-record uncertainty, and ranks residual-demand hotspots. It also
sets boundaries for a future illustrative finite-viewpoint extension. It
freezes the analysis before final result values are inserted into the revised
manuscript. It is not a preregistration: after the any-ray sparse diagnostic was
found to be optimistic, the threshold-selection rule and deterministic split
were added as post-development calibration with a disjoint held-out evaluation.
The corresponding machine-readable analysis configuration is
`configs/cacie_v1.1.json`.

The high-resolution reference uses analytic ray--oriented-box (ray--OBB)
intersection. It is exact only within the detected, tracked oriented-box scene
abstraction. It is not physical line-of-sight ground truth, detector ground
truth, or an external field-validation measurement.

## 1. Provenance gate

All paper-facing stages begin with `scripts/audit_input_provenance.py`.

The input must contain finite 3-D box centers, positive box dimensions, yaw,
class, frame, and track identifiers with no duplicate `(frame_idx, track_id)`
rows. The gate writes the trajectory SHA-256 digest, input diagnostics,
upstream-manifest digest/content, analysis Git revision and dirty state,
parameters, and runtime versions. A failed gate stops the analysis.

The upstream manifest must establish an active paper-facing detector/tracker and
verified alignment lineage. A legacy-only or alignment-ambiguous artifact cannot
be promoted by passing only the table-shape checks.

Source-frame completeness is audited from the frame index over the declared
range. A partial scan is a frame whose point count is below 0.6 times the
median of the available preceding and following three source rows; boundary
frames therefore use only the neighbors that exist. Timestamp zero is
not an exclusion criterion. The gate fails closed when any detected partial
scan is missing from the declared exclusion list. An allowed missing track
frame and an excluded observed partial frame are distinct: the former explains
why an expected source frame has no tracked boxes, while the latter must be
removed explicitly from validation and full-record selection.

## 2. Ground and vertical-coordinate design

The paper-facing reference plane uses `n dot p + d = 0`:

```text
n = (-0.001561196615141432,
     -0.06019325942675009,
      0.9981855209252)
d =  2.907646656036377
```

This plane is an empirical calibration fit, not a surveyed installation value.
The coefficients are supplied explicitly to every validation and coverage run.
The modeled onboard sensor origin is 1.8 m above the local plane.

Validation is run in both modes:

- `raw`: retain `bbox_center_z` from the governed trajectory artifact; and
- `ground_anchored`: place each box bottom on the fitted plane while preserving
  its horizontal center, dimensions, and yaw.

The difference is a vertical-coordinate sensitivity analysis, not a correction
that makes either representation ground truth.

## 3. Deterministic sample and ego design

`scripts/validate_occlusion_models.py` draws balanced, deterministic frame
samples with seed `20260902`. The main validation sample contains 200 frozen
frames. A 25-frame subset of those 200 frames is frozen for the
higher-resolution convergence run; it is not an independent sample. Fifteen
hypothetical ego positions are used on each approach.
A pose is explicitly excluded when it lies within 4.0 m of an observed vehicle;
exclusions and reason codes are written as an output rather than silently
discarded.

The default modeled onboard envelope uses FOVs of 120 and 360 degrees, a 35 m
maximum range, a 1.5 m near-blind radius, and a 1.8 m sensor height.

## 4. Low-cost model

The planar angular-wedge model uses a fixed angular resolution of 0.25 degrees
across every requested FOV. Boxes are processed from near to far, and car/truck
boxes update the blocking map. Coverage and occlusion reasons are mutually
exclusive under the implementation's frozen attribution rule.

The validation run also records the center/top two-ray check and a 15-ray
sparse approximation consisting of the target center, six face centers, and
eight vertices. The frozen sparse binary rule requires at least **3 of 15**
rays to be visible; an any-ray rule is not the primary CACIE method. No
low-cost method supplies reference labels.

## 5. Ray--OBB hierarchy and convergence

The main validation sample compares these target-sampling levels:

1. **15-ray sparse surrogate:** one target-center ray, six face-center rays,
   and eight vertex rays.
2. **9 x 9 projected angular grid:** coarse projected-grid comparison.
3. **17 x 17 projected angular grid:** intermediate comparison.
4. **33 x 33 projected angular grid:** primary within-abstraction reference.

Before reporting sparse-method accuracy, the frozen 200-frame set is divided
once by deterministic time-by-density stratification into 100 calibration
frames and 100 held-out frames. This split and the threshold search are a
post-development response to the optimistic any-ray diagnostic, not a
preregistered or prespecified analysis. Thresholds from 1 through 15 visible
rays are evaluated only on the calibration frames. The selection criterion is
covered-row accuracy; ties are resolved by smaller observability-ratio bias and
then in the more conservative direction. The resulting 3/15 threshold is
evaluated once on the untouched held-out frames and then frozen for the
full-record computation. Calibration and held-out results must remain
separately labeled.

The frozen main command explicitly includes `--n-frames 200 --ray-grids 9 17
33`. A separate frozen 25-frame subset run uses `--n-frames 25 --ray-grids 17 33 65`
to test 17 x 17/33 x 33/65 x 65 convergence. The CLI default of 17 is only a
smoke-test default. The primary binary decision uses the 33 x 33 projected
fraction and a visible-fraction threshold of 0.05. Sensitivity to threshold
choice is reported separately from grid convergence.

Every sampled direction first intersects the target OBB; only target-hitting
rays enter the projected-fraction denominator. A target ray is blocked when a
candidate blocker has an earlier valid ray--OBB entry. Self-intersection is
excluded. This calculation is analytic for the supplied boxes but remains
conditional on the scene abstraction and upstream provenance.

## 6. Validation outputs

`scripts/validate_occlusion_models.py` writes:

- `validation_decisions.parquet` with per-target method decisions and projected
  fractions;
- `ego_exclusions.parquet` with explicit pose-exclusion reasons;
- `validation_timings.parquet` with method/grid timing records;
- `validation_config.json` with frozen parameters; and
- `validation_summary.json` with grouped agreement and runtime summaries.

`scripts/summarize_validation_strata.py` consumes the saved decisions and
timings and emits covered-row confusion counts plus class, range, approach,
and frame-density strata. It performs no ray tracing. These 200-frame metrics
are descriptive repeated-decision summaries; an independence-based confidence
interval over target/ego rows is not valid and is not reported.

`scripts/calibrate_sparse_threshold.py` consumes the saved 200-frame decisions
and writes `outputs/canonical/sparse_calibration/sparse_threshold_calibration.json`
plus `validation_decisions_calibrated.parquet`. The JSON is authoritative for
the calibration/held-out assignment, all 1--15 candidates, the selected
`selected_min_visible_rays = 3`, and output checksums. The original main
validation file retains useful ray-count and geometric evidence, but any
pre-calibration any-ray label is not a primary sparse-method result.

Agreement is stratified by method, vertical-coordinate mode, FOV, target class,
range, and other declared scene factors. Runtime comparisons use the same
sample and hardware context. No final values are specified in this protocol.

## 7. Full-record analysis and dependence-aware inference

The full-record low-cost sweep is performed by
`scripts/compute_full_record_observability.py` only after the input gate
passes. It streams a wide `full_record_observability.parquet` table and a long
`observability_contributions.parquet` table in stable frame order. The
`full_record_summary.json` file is published last and is the completion marker;
temporary `.partial` files are not valid results. Primary v1.1 estimates are
scene-observability quantities that do not depend on a conflict-event review
claim.

Frame-level numerator and denominator contributions are passed to
`scripts/bootstrap_observability.py`. All ego positions within a frame are
aggregated before resampling. The frozen bootstrap design is:

- circular moving blocks;
- 1,200 consecutive retained-frame observations per block, nominally 120
  seconds at 10 Hz (source-index gaps can extend the elapsed span);
- 5,000 replicates;
- 95% percentile intervals; and
- deterministic seed `20260902`.

The paired 360-minus-120 comparison uses the same aligned frames and the same
sampled blocks. The estimand is the ratio of summed numerators to summed
denominators, not the unweighted mean of per-frame ratios.

## 8. Conflict-event boundary

No versioned human-review ledger is included in this repository. Candidate
events may therefore be used only for explicitly exploratory software checks
unless an external ledger is supplied, versioned, and joined to the same
gate-passed trajectory run and coordinate frame.

Directory names, clip viewers, or candidate flags do not establish review.
Without a qualifying ledger, omit human-reviewed, verified-event, and
conflict-weighted primary claims.

## 9. Residual demand and future illustrative coverage

`scripts/rank_residual_demand.py` is the current decision-layer analysis. It
uses the ground-anchored 3/15 sparse method and reports, separately for 120 and
360 degree FOVs, uniform residual-demand hotspot scores and VRU-weighted
sensitivity scores. Both are ratios of summed contributions. Rank stability is
estimated by resampling 1,200-frame intervals on the original frame axis with
5,000 replicates and seed `20260902`.

`scripts/optimize_external_coverage.py` evaluates finite hypothetical external
viewpoints with analytic ray--OBB tests and a greedy maximum-coverage trace. It
is an illustrative **future extension**, not a required canonical CACIE result.

Candidate coordinates and budgets are illustrative scenario inputs. They are
not surveyed sites, and the model omits important operational factors including
static occluders, communication reliability, latency, packet loss, detector
performance, permitting, utilities, and cost validation. Consequently, the
output is a transparent screening exercise rather than a deployment design.

## 10. Cooperative-ceiling boundary

The theoretical cooperative ceiling assigns visibility one to every
quality-gated target by definition. It is a diagnostic decomposition of onboard
residual demand. It must not be described as measured V2I coverage,
availability, reliability, or operational performance.

Only finite-viewpoint scenario results may be compared with that ceiling, and
even those results remain geometric information-availability estimates rather
than detection, communication, control, or safety outcomes.

## 11. Release checklist

Before any v1.1 value appears in a paper or figure, archive:

- the upstream and CACIE provenance manifests;
- input and output checksums;
- validation sample identifiers and exclusion table;
- the calibration/held-out frame assignment, all 1--15 candidate scores, and
  the frozen 3/15 sparse-rule decision;
- frozen commands and configuration files;
- validation decisions, timing records, and convergence outputs;
- full-record frame contributions and bootstrap replicates;
- stage-local main-validation, subset-convergence, and full-record
  runtime-environment receipts, preserving disclosed dirty/no-Git states;
- portable byte-lineage receipts for main validation, convergence, calibrated
  strata, and both figures;
- the training-isolated internal detector/box validation and its human-label,
  model-split, overlap, and canonical-input source receipts;
- the immutable notebook source, separately executed notebook, and
  `notebook_audit_receipt.json`; and
- the producing `src/paco_observability` and `scripts` files, `pyproject.toml`,
  `uv.lock`, README, and protocol documentation.

The release manifest is built only from a clean worktree. The configuration
must declare `release.numeric_producer_commit` and
`release.numeric_producer_files`; the commit must exist and every listed
current numerical producer must match its Git blob byte-for-byte. The declared
historical full-record entrypoint is independently checked against its stored
Git blob and SHA-256. The release builder inventories and hashes all selected
files twice before atomic output; it does not update historical provenance to
mask a code-revision mismatch.

If the future external-viewpoint scenario is reported separately, also archive
its illustrative candidate definitions, weights, and greedy trajectory; those
files are not prerequisites for the canonical CACIE results.

The manuscript must preserve the within-abstraction, conflict-review, and
theoretical-ceiling limitations stated above.
