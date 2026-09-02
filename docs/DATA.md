# Data and provenance

## Data classes and permitted use

PACO v1.1 distinguishes three data classes.

1. **Archival derived outputs.** The Harvard Dataverse files at
   <https://doi.org/10.7910/DVN/7T4DWL> are legacy v1.0 derived outputs. They may
   be used to reproduce supported historical figures, but not for new CACIE
   conclusions or validation.
2. **Governed trajectory inputs.** A v1.1 paper-facing run must begin from a
   trajectory table with an upstream run manifest and must pass
   `scripts/audit_input_provenance.py`.
3. **Candidate-event inputs.** A frame-aligned event table is optional and is
   not part of the primary scene-observability estimand. No review ledger is
   distributed here, so no bundled or referenced candidate-event file may be
   called human reviewed.

Do not silently combine artifacts from different detector, tracking, alignment,
or coordinate-system lineages.

## Governed `tracks.parquet` schema

The provenance gate requires these columns:

- `frame_idx`, `track_id`, `class_name`
- `bbox_center_x`, `bbox_center_y`, `bbox_center_z`
- `bbox_dx`, `bbox_dy`, `bbox_dz`, `bbox_yaw`

Coordinates and dimensions are in metres and yaw is in radians. A `run_id` and
detector confidence score are recommended when available, but a confidence
score is not interpreted as an onboard-detector score.

The gate checks, at minimum:

- all required fields are present;
- required numeric fields are finite;
- box dimensions are strictly positive;
- each `(frame_idx, track_id)` row is unique; and
- the optional ground-plane diagnostic can be evaluated.

The resulting manifest records the trajectory SHA-256 digest, input audit,
analysis Git revision and dirty state, parameters, runtime versions, and the
upstream manifest digest/content. A failed quality gate is a hard stop for
paper-facing analysis. `--allow-quality-fail` is only for writing diagnostic
artifacts while investigating a failure.

## Upstream provenance requirements

The upstream manifest should identify the source run, extraction configuration,
alignment transform, detector checkpoint/configuration, tracker configuration,
coordinate frame, frame extent, and checksums of the relevant artifacts. Inputs
marked legacy-only, diagnostic-only, or alignment-provenance-ambiguous by their
source registry are not valid substitutes for an active paper-facing run.

## Ground plane

The v1.1 paper-facing default/reference plane is represented as
`n dot p + d = 0` with:

```text
normal = (-0.001561196615141432,
          -0.06019325942675009,
           0.9981855209252)
d      =  2.907646656036377
```

It is an empirical fit from an aligned audited window, not an installation
survey or physical ground-truth measurement. The validation and illustrative
coverage CLIs require the four coefficients to be supplied explicitly. The
library's generic `GroundPlane()` fallback (`z = 0`) is useful for synthetic
tests but is not the CACIE paper-facing default.

Both raw vertical centers and ground-anchored boxes are evaluated in validation.
Ground anchoring places the box bottom on the fitted local plane; it does not
repair horizontal position, yaw, dimensions, missing objects, or detector
errors.

## Candidate-event inputs and review status

Legacy scripts accept a `near_misses.parquet`-style table containing
frame-aligned interacting track identifiers and candidate-event attributes.
The presence of such a file, a directory name containing `human_compare`, or a
visual clip browser does not establish review.

A future reviewed-event analysis would require a versioned ledger containing at
least an event identifier, source run and coordinate frame, joined track/frame
identifiers, reviewer identifier, decision, review timestamp/version, and any
inclusion or exclusion rationale. Until that ledger exists and passes the
provenance join, conflict-weighted or ego-partner analyses are exploratory and
must be omitted from primary v1.1 conclusions.

## Directory convention

```text
data/inputs/tracks.parquet          # governed input; not committed
data/inputs/upstream_manifest.json  # governed upstream provenance; not committed
data/derived/                       # archival Dataverse v1.0 outputs
outputs/cacie/                      # generated v1.1 manifests and results
```

Generated results must remain separated from archival derived outputs.
