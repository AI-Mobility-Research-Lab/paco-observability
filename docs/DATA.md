# Data inputs

## Public derived outputs

The manuscript's public derived observability outputs are archived at Harvard
Dataverse: <https://doi.org/10.7910/DVN/7T4DWL>.

Run `python scripts/download_dataverse.py` to download and verify the published
files into `data/derived/`.

## End-to-end inputs

The core analysis expects two Parquet files.

### `tracks.parquet`

Required columns used across the PACO analyses include:

- `frame_idx`, `track_id`, `class_name`
- `bbox_center_x`, `bbox_center_y`
- `bbox_dx`, `bbox_dy`, `bbox_dz`, `bbox_yaw`
- detector confidence where required by the confidence sensitivity analysis

Coordinates and dimensions are in metres; yaw is in radians; frames are sampled
at 10 Hz in the manuscript analysis.

### `near_misses.parquet`

The conflict-event input contains frame-aligned interacting track identifiers,
class labels, and the reviewed closest-approach event fields consumed by
`parametric_ego_sweep.py` and `ego_near_miss_sweep.py`.

The trajectory inputs are governed by the companion release and are not bundled
with this software repository. Do not substitute a different detection run
without verifying frame indices and track identifiers.

