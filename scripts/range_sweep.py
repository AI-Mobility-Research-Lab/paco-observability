#!/usr/bin/env python3
"""Detection-range sensitivity sweep for the Part C paper.

Runs the angular-wedge observability model over the FULL dataset (all frames,
~68 min) at FOV in {120, 360} deg and a range of maximum detection distances,
writing data/derived/range_sensitivity.json (consumed by
generate_range_figure.py). Uses the frame-parallel run_sweep.

The point of the figure: beyond ~50 m, increasing range yields negligible
additional observability -- the gap is set by field of view and occlusion, not
detection distance -- so the default 35 m is a conservative, non-limiting
assumption.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import parametric_ego_sweep as P  # noqa: E402

TRACKS = Path("data/inputs/tracks.parquet")
NM = Path("data/inputs/near_misses.parquet")
OUT = REPO_ROOT / "outputs" / "range_sensitivity.json"
FOVS = [120, 360]
RANGES = [35, 50, 75, 100, 150, 200]
NPROCS = max(1, (os.cpu_count() or 2) - 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=TRACKS)
    parser.add_argument("--near-misses", type=Path, default=NM)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-procs", type=int, default=NPROCS)
    args = parser.parse_args()

    results = []
    for fov in FOVS:
        for rng in RANGES:
            df = P.run_sweep(
                tracks_path=args.tracks,
                near_miss_path=args.near_misses,
                out_dir=args.out.parent,
                sensor=P.SensorConfig(fov_deg=fov, range_m=rng),
                approaches=P.DEFAULT_APPROACHES, occluder_classes={"truck", "car"},
                frame_start=0, frame_end=0, n_procs=args.n_procs,
            )
            results.append({
                "fov": fov, "range_m": rng,
                "int_wide": float(df["visibility_ratio_onboard"].mean()),
                "nearby": float(df["nearby_visibility_ratio"].mean()),
                "conflict": float(df["conflict_zone_visibility_ratio"].mean()),
                "total": float(df["n_objects_total"].mean()),
            })
            print(f"FOV={fov:3d} R={rng:3d}m | int={results[-1]['int_wide']:.3f} "
                  f"near={results[-1]['nearby']:.3f} cz={results[-1]['conflict']:.3f}",
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
