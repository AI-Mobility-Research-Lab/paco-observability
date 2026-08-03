# PACO observability analysis

PACO (Parametric Assessment of Cooperative Observability) is a geometry-based
framework for auditing field-of-view, range, and occlusion limits of modeled
onboard perception using real roadside-LiDAR trajectories.

This repository is the software release accompanying:

> **Roadside LiDAR reveals distinct limits to scene-wide and ego-centric
> perception at urban intersections**

## Reproducibility scope

The public Harvard Dataverse record contains the derived observability outputs
used by the manuscript:

- Data DOI: <https://doi.org/10.7910/DVN/7T4DWL>
- License: CC BY 4.0

This repository provides the analysis algorithms, manuscript figure generators,
configuration defaults, dependency specification, and tests. Two levels of
reproduction are intentionally distinguished:

1. **Derived-output reproduction.** Download the public Dataverse files and
   regenerate figures supported by those deposited outputs.
2. **End-to-end recomputation.** Recompute PACO outputs from `tracks.parquet`
   and `near_misses.parquet`. Those trajectory inputs are governed by the
   companion data release and are not redistributed in this code repository.

The code does not claim that the public derived outputs alone reproduce every
trajectory-level analysis step.

## Installation

PACO supports Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Download the public derived data

```bash
python scripts/download_dataverse.py
```

Files are downloaded to `data/derived/` and checked against the MD5 digests
reported by Harvard Dataverse.

## Run the public-data figure reproduction

Figure 3 can be regenerated directly from the public conflict-observability
summary:

```bash
python scripts/generate_ego_near_miss_figure.py \
  --data data/derived/ego_near_miss_observability.json \
  --out figures/ego_near_miss_observability
```

The range-sensitivity figure can be generated from the public JSON:

```bash
python scripts/generate_range_figure.py \
  --data data/derived/range_sensitivity.json \
  --out figures/range_sensitivity
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the manuscript figure
map and the inputs required by each analysis.

## End-to-end analysis

Place authorized inputs at:

```text
data/inputs/tracks.parquet
data/inputs/near_misses.parquet
```

Then run a smoke analysis:

```bash
python scripts/parametric_ego_sweep.py \
  --tracks data/inputs/tracks.parquet \
  --near-misses data/inputs/near_misses.parquet \
  --out-dir outputs/smoke \
  --max-frames 20 \
  --n-procs 1
```

For the full manuscript configuration, use `scripts/run_canonical.sh` after
reviewing its compute requirements.

## Tests

```bash
pytest
```

The tests use synthetic road-user geometry and do not require restricted data.

## License and citation

The software is released under the MIT License. See [LICENSE](LICENSE) and
[CITATION.cff](CITATION.cff). The Harvard Dataverse data retain their own CC BY
4.0 terms.

