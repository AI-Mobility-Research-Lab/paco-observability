#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. These analyses require the authorized
# trajectory and reviewed conflict-event inputs documented in docs/DATA.md.
python scripts/parametric_ego_sweep.py \
  --tracks data/inputs/tracks.parquet \
  --near-misses data/inputs/near_misses.parquet \
  --out-dir outputs/fov120 \
  --fov-deg 120 \
  --range-m 35

python scripts/parametric_ego_sweep.py \
  --tracks data/inputs/tracks.parquet \
  --near-misses data/inputs/near_misses.parquet \
  --out-dir outputs/fov360 \
  --fov-deg 360 \
  --range-m 35

python scripts/sensitivity_sweep.py \
  --tracks data/inputs/tracks.parquet \
  --near-misses data/inputs/near_misses.parquet \
  --out-dir outputs/sensitivity

python scripts/range_sweep.py \
  --tracks data/inputs/tracks.parquet \
  --near-misses data/inputs/near_misses.parquet \
  --out outputs/range_sensitivity.json

python scripts/ego_near_miss_sweep.py \
  --tracks data/inputs/tracks.parquet \
  --near-misses data/inputs/near_misses.parquet \
  --out outputs/ego_near_miss_observability.json \
  --with-pet
