# Reproducibility map

The table distinguishes public derived-output reproduction from analyses that
require the authorized trajectory input.

| Manuscript item | Producing script | Required input |
|---|---|---|
| Figure 1 methodology geometry | `generate_methodology_figure.py` | none; schematic parameters are embedded |
| Figure 2 object decomposition | `generate_object_counts_figure.py` | 120° and 360° ego-sweep Parquet outputs |
| Figure 3 ego-partner observability | `generate_ego_near_miss_figure.py` | public `ego_near_miss_observability.json` |
| Figure 4 FOV/range sensitivity | `generate_sensitivity_figure.py` | sensitivity-sweep Parquet plus public range JSON |
| Figure 5 planar/vertical correction | `compute_3d_occlusion_comparison.py` | `tracks.parquet` |
| Figure 6 confidence robustness | `compute_confidence_robustness.py` | `tracks.parquet` with confidence scores |
| Main 120° observability estimates | `parametric_ego_sweep.py` | tracks and reviewed conflict events |
| FOV sweep | `sensitivity_sweep.py` | tracks and reviewed conflict events |
| Range sweep | `range_sweep.py` | tracks and reviewed conflict events |
| Ego-partner/PET cross-check | `ego_near_miss_sweep.py` | tracks and reviewed conflict events |

## Canonical configuration

- Frames: 40,793 at 10 Hz
- Hypothetical ego placements: 15 per approach, 60 total per frame
- Forward field of view: 120°
- Surround field of view: 360°
- Maximum range: 35 m
- Near-blind radius: 1.5 m
- Sensor forward offset: 1.5 m
- Occlusion bins: 360
- Occlusion clearance: 0.5 m
- Vertical-correction sensor height: 1.8 m

The scripts expose the parameters needed for the reported sensitivity analyses.
`scripts/run_canonical.sh` records the end-to-end invocation sequence.

