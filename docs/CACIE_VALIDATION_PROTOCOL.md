# CACIE validation protocol

## Purpose and claim boundary

This protocol freezes how PACO v1.1 evaluates low-cost observability models,
quantifies full-record uncertainty, and illustrates finite external-viewpoint
coverage. It specifies the design before final result values are inserted into
any manuscript. The corresponding machine-readable preregistration intent is
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
frames. A separate 25-frame sample is frozen for the higher-resolution
convergence run. Fifteen hypothetical ego positions are used on each approach.
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

The validation run also records the center/top two-ray check as a separate
low-cost three-dimensional approximation. Neither low-cost method supplies
reference labels.

## 5. Ray--OBB hierarchy and convergence

The main validation sample compares these target-sampling levels:

1. **15-ray sparse surrogate:** one target-center ray, six face-center rays,
   and eight vertex rays.
2. **9 x 9 projected angular grid:** coarse projected-grid comparison.
3. **17 x 17 projected angular grid:** intermediate comparison.
4. **33 x 33 projected angular grid:** primary within-abstraction reference.

The frozen main command explicitly includes `--n-frames 200 --ray-grids 9 17
33`. A separate frozen 25-frame run uses `--n-frames 25 --ray-grids 17 33 65`
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

Agreement is stratified by method, vertical-coordinate mode, FOV, target class,
range, and other prespecified scene factors. Runtime comparisons use the same
sample and hardware context. No final values are specified in this protocol.

## 7. Full-record analysis and dependence-aware inference

The full-record low-cost sweep is performed by
`scripts/parametric_ego_sweep.py` only after the input gate passes. Primary
v1.1 estimates are scene-observability quantities that do not depend on a
conflict-event review claim.

Frame-level numerator and denominator contributions are passed to
`scripts/bootstrap_observability.py`. All ego positions within a frame are
aggregated before resampling. The frozen bootstrap design is:

- circular moving blocks;
- 120 seconds per block, equal to 1,200 frames at 10 Hz;
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

## 9. Residual demand and illustrative coverage

`scripts/optimize_external_coverage.py` computes residual demand from onboard
decisions and evaluates finite hypothetical external viewpoints with analytic
ray--OBB tests. It records candidate definitions, class weights, grid settings,
greedy marginal gains, uncovered demand, and block-bootstrap rank stability
using seed `20260902`.

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
- frozen commands and configuration files;
- validation decisions, timing records, and convergence outputs;
- full-record frame contributions and bootstrap replicates; and
- illustrative candidate definitions, weights, and greedy trajectory.

The manuscript must preserve the within-abstraction, conflict-review, and
theoretical-ceiling limitations stated above.
