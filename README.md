# PACO observability analysis (v1.1 CACIE)

PACO (Parametric Assessment of Cooperative Observability) is a geometry-based
framework for auditing field-of-view, range, and occlusion limits of modeled
onboard perception using roadside-LiDAR trajectories. Version 1.1 adds the
CACIE provenance, geometric-reference validation, dependence-aware inference,
and illustrative external-viewpoint coverage pipeline.

## Evidence boundary

The Harvard Dataverse record at <https://doi.org/10.7910/DVN/7T4DWL> contains
derived outputs from the earlier PACO analysis. Those files are retained as an
**archival/legacy v1.0 reproduction package**. They can reproduce supported
legacy figures, but they are not governed v1.1 inputs and must not be used to
support new CACIE conclusions, validation claims, or operational V2I claims.

PACO v1.1 starts from a trajectory table that passes the fail-closed provenance
gate. No final v1.1 result values are asserted in this repository documentation;
they must come from a frozen, manifested run.

The machine-readable preregistration intent is recorded in
[`configs/cacie_v1.1.json`](configs/cacie_v1.1.json). Null result hashes in that
file remain null until the canonical run is frozen.

## CACIE pipeline

| Stage | Script | Purpose |
|---|---|---|
| Input provenance gate | `scripts/audit_input_provenance.py` | Audit the 3-D box schema and write an input/code manifest |
| Geometric validation | `scripts/validate_occlusion_models.py` | Compare low-cost models with ray--oriented-box references |
| Validation figure | `scripts/generate_cacie_validation_figure.py` | Plot only source-backed validation summaries |
| Full-record sweep | `scripts/parametric_ego_sweep.py` | Produce full-record low-cost observability outputs on gate-passed inputs |
| Dependence-aware inference | `scripts/bootstrap_observability.py` | Apply grouped circular moving-block bootstrap inference |
| Residual demand and illustrative coverage | `scripts/optimize_external_coverage.py` | Evaluate finite hypothetical viewpoints and greedy coverage |

The exact ray--oriented-box calculation is an exact geometric reference only
within the supplied oriented-box scene abstraction. It is not physical
visibility ground truth, detector ground truth, or field validation.

Likewise, a cooperative visibility value fixed to one is a theoretical ceiling
by definition. It is not measured V2I availability or operational performance.
The finite-viewpoint coverage stage is an illustrative geometry exercise, not a
deployment or pole-siting recommendation.

## Installation

PACO supports Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Inputs and provenance gate

Place an authorized trajectory input at `data/inputs/tracks.parquet` and retain
its upstream run manifest. The paper-facing reference ground plane uses
coefficients `(nx, ny, nz, d)` in `n dot p + d = 0` form:

```text
(-0.001561196615141432,
 -0.06019325942675009,
  0.9981855209252,
  2.907646656036377)
```

Run the gate before any v1.1 analysis:

```bash
python scripts/audit_input_provenance.py \
  --tracks data/inputs/tracks.parquet \
  --upstream-manifest path/to/upstream_manifest.json \
  --ground-plane -0.001561196615141432 -0.06019325942675009 \
                 0.9981855209252 2.907646656036377 \
  --output outputs/cacie/input_manifest.json
```

Do not use `--allow-quality-fail` for paper-facing computation. See
[docs/DATA.md](docs/DATA.md) for schema and provenance requirements.

## Validation protocol

The frozen design uses a 0.25-degree low-cost angular resolution and
deterministic seed `20260902`. It compares a 15-ray sparse surrogate with 9 x 9,
17 x 17, and 33 x 33 projected grids on the main validation sample, uses
33 x 33 as the primary within-abstraction reference, and runs a separate
17 x 17/33 x 33/65 x 65 convergence check on a frozen 25-frame sample.
Full-record uncertainty uses a 120-second circular moving block. The complete
design and interpretation guardrails are in
[docs/CACIE_VALIDATION_PROTOCOL.md](docs/CACIE_VALIDATION_PROTOCOL.md).

## Archival legacy reproduction

Legacy Dataverse files may still be downloaded for historical figure
reproduction:

```bash
python scripts/download_dataverse.py
```

Files are written to `data/derived/` and checked against the MD5 digests
reported by Harvard Dataverse. This command does not promote them to v1.1
inputs. The legacy figure map is retained in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Conflict-event inputs

This repository does not contain a review ledger that substantiates a
human-reviewed conflict-event set. Candidate-event files may be used for
software checks or explicitly labeled exploratory analyses, but they must not
be described as reviewed events or used for primary v1.1 conclusions without a
versioned ledger that joins to the gate-passed trajectory run.

## Tests

```bash
pytest
```

The tests use synthetic road-user geometry and do not require governed data.

## License and citation

The software is released under the MIT License. See [LICENSE](LICENSE) and
[CITATION.cff](CITATION.cff). The Harvard Dataverse data retain their own CC BY
4.0 terms.
