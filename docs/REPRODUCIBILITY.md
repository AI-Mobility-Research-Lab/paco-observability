# Reproducibility map

## Evidence tiers

The repository supports two deliberately separate activities:

- **Archival v1.0 reproduction:** `scripts/download_dataverse.py` downloads
  deposited derived outputs for supported historical figures. These files are
  legacy artifacts and cannot support new v1.1 conclusions.
- **PACO v1.1 CACIE recomputation:** a governed `tracks.parquet` first passes the
  provenance gate, after which validation, full-record analysis, bootstrap
  inference, and illustrative coverage are run under frozen parameters.

Do not use an archival figure matching an old manuscript value as validation of
the v1.1 implementation.

## v1.1 script map

| CACIE stage | Script | Principal outputs or role |
|---|---|---|
| Provenance gate | `scripts/audit_input_provenance.py` | Input audit and SHA-256/Git/runtime manifest |
| Validation sampling and model comparison | `scripts/validate_occlusion_models.py` | Per-target decisions, exclusions, timings, configuration, and summary |
| Validation figure | `scripts/generate_cacie_validation_figure.py` | Source-backed accuracy, observability, runtime, and convergence panels |
| Full-record low-cost sweep | `scripts/parametric_ego_sweep.py` | Full-record ego-grid observability outputs |
| Moving-block inference | `scripts/bootstrap_observability.py` | Bootstrap summary and replicate table from frame-level contributions |
| Residual demand and illustrative finite-viewpoint coverage | `scripts/optimize_external_coverage.py` | Residual-demand table, candidate visibility, greedy trace, and rank stability |
| Unit verification | `pytest` | Synthetic geometry, provenance, validation, bootstrap, and optimization tests |

`scripts/run_canonical.sh` records the earlier analysis sequence and is not the
authoritative v1.1 CACIE driver. In particular, its conflict-event stages must
not be represented as reviewed analyses without a qualifying ledger.

## Frozen paper-facing defaults

These are method parameters, not final findings.
`configs/cacie_v1.1.json` is the machine-readable preregistration intent; the
formal commands must override smoke-test CLI defaults where the two differ.

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
| Bootstrap | circular 120-second blocks at 10 Hz, 5,000 replicates, 95% percentile intervals |

The 0.25-degree setting fixes angular resolution rather than fixing one bin
count for every FOV.

## Geometric-reference and convergence design

The validation hierarchy and preregistered grid design are:

1. the planar angular-wedge approximation at 0.25-degree resolution;
2. low-cost center/top ray checks;
3. a **15-ray sparse surrogate** consisting of the target center, six face
   centers, and eight vertices;
4. **9 x 9, 17 x 17, and 33 x 33 projected angular grids** on the frozen
   200-frame main validation sample, with 33 x 33 supplying the primary
   within-abstraction reference; and
5. a separate **17 x 17, 33 x 33, and 65 x 65** convergence run on a frozen
   25-frame sample.

The formal main run explicitly passes `--n-frames 200 --ray-grids 9 17 33`; the
convergence run explicitly passes `--n-frames 25 --ray-grids 17 33 65`. The
CLI's single-grid default of 17 is only a smoke default, not the preregistered
paper configuration. Agreement and runtime are reported by method,
vertical-coordinate mode, FOV, class, and range without calling any ray-grid
output physical ground truth.

Every ray uses analytic ray--oriented-box intersection. “Exact” means exact for
that ray against the supplied oriented boxes. The scene may still omit objects
or contain detector, tracker, calibration, alignment, and box-shape errors.

## Required execution order

1. Audit the governed trajectory input with
   `scripts/audit_input_provenance.py`; retain its manifest and fail closed.
2. Run `scripts/validate_occlusion_models.py` on the frozen stratified frame
   sample with both vertical-coordinate modes. Use the explicit 200-frame
   9/17/33 main comparison and the separate 25-frame 17/33/65 convergence run
   described above.
3. Run `scripts/parametric_ego_sweep.py` over the full gate-passed record for the
   low-cost scene-observability analysis. Any candidate-event columns are
   excluded from primary results unless a separate review ledger passes its
   provenance join.
4. Convert full-record outputs into a long-form contribution table containing
   `frame_idx`, `z_mode`, `fov_deg`, `method`, `numerator`, and `denominator`.
5. Run `scripts/bootstrap_observability.py`. It first aggregates all repeated
   ego rows within a frame and then resamples contiguous 120-second circular
   blocks with seed `20260902`. The 360-minus-120 FOV comparison uses identical
   aligned blocks.
6. Run `scripts/optimize_external_coverage.py` only as an illustrative residual
   demand exercise. Preserve its candidate definitions, weights, bootstrap
   settings, greedy trajectory, and interpretation guardrail.

Each published table or figure should identify the input manifest, producing
command/configuration, analysis Git revision, and output checksum.

## Bootstrap estimand

The full-record estimand is a ratio of sums:

```text
sum(frame numerator) / sum(frame denominator)
```

It is not an unweighted mean of frame ratios, and ego positions from one frame
are not treated as independent samples. The default 120-second block equals
1,200 frames at 10 Hz. Paired comparisons reuse the same sampled blocks.

## Residual demand and illustrative coverage

`scripts/optimize_external_coverage.py` evaluates finite hypothetical external
viewpoints against residual onboard demand and records a deterministic greedy
maximum-coverage trace plus block-bootstrap rank stability. Its default
coordinates are computational stress-test points, not surveyed pole sites.

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
