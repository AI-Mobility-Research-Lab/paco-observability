#!/usr/bin/env python3
"""Sensitivity analysis: sweep AV sensor parameters (FOV, range) and compute
cooperative perception benefit for each configuration.

Runs the parametric ego sweep at multiple FOV and range settings to produce
a parameter-space map of cooperative benefit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from parametric_ego_sweep import (
    ApproachLane,
    SensorConfig,
    build_ego_grid,
    run_sweep,
)


def main():
    parser = argparse.ArgumentParser(description="Sensitivity sweep over FOV and range.")
    parser.add_argument(
        "--tracks", type=Path,
        default=Path("data/inputs/tracks.parquet"),
    )
    parser.add_argument(
        "--near-misses", type=Path,
        default=Path("data/inputs/near_misses.parquet"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(
            "outputs/sensitivity"
        ),
    )
    parser.add_argument("--frame-step", type=int, default=50, help="Frame subsampling for speed")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--n-ego-per-approach", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    fov_values = [60, 90, 120, 150, 180]
    range_values = [20, 35, 50, 80]

    approaches = [
        ApproachLane("north", 7.0, 35.0, 7.0, 8.0, math.radians(-90), n_positions=args.n_ego_per_approach),
        ApproachLane("south", 8.0, -15.0, 8.0, 8.0, math.radians(90), n_positions=args.n_ego_per_approach),
        ApproachLane("east", 35.0, 8.0, 8.0, 8.0, math.radians(180), n_positions=args.n_ego_per_approach),
        ApproachLane("west", -20.0, 8.0, 8.0, 8.0, math.radians(0), n_positions=args.n_ego_per_approach),
    ]
    occluder_classes = {"truck", "car"}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    total_configs = len(fov_values) * len(range_values)
    config_idx = 0

    for fov in fov_values:
        for rng in range_values:
            config_idx += 1
            print(f"\n{'='*60}")
            print(f"Config {config_idx}/{total_configs}: FOV={fov}°, Range={rng}m")
            print(f"{'='*60}")

            sensor = SensorConfig(fov_deg=fov, range_m=rng)

            df = run_sweep(
                tracks_path=args.tracks,
                near_miss_path=args.near_misses,
                out_dir=args.out_dir,
                sensor=sensor,
                approaches=approaches,
                occluder_classes=occluder_classes,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                verbose=args.verbose,
            )

            vis_mean = float(df["visibility_ratio_onboard"].mean())
            vis_median = float(df["visibility_ratio_onboard"].median())

            by_approach = {}
            for approach in sorted(df["approach"].unique()):
                sub = df[df["approach"] == approach]
                by_approach[approach] = float(sub["visibility_ratio_onboard"].mean())

            nm_detect_rate = 0.0
            if df["nm_total"].sum() > 0:
                nm_detect_rate = float(df["nm_detectable_onboard"].sum() / df["nm_total"].sum())

            summary = {
                "fov_deg": fov,
                "range_m": rng,
                "visibility_mean": vis_mean,
                "visibility_median": vis_median,
                "cooperative_benefit": 1.0 - vis_mean,
                "nm_detectability_onboard": nm_detect_rate,
                "nm_detectability_cooperative": 1.0,
                **{f"vis_{k}": v for k, v in by_approach.items()},
            }
            all_summaries.append(summary)
            print(f"  Visibility: {vis_mean:.3f}, Coop benefit: {1.0 - vis_mean:.3f}")

    # Save sensitivity results
    results_df = pd.DataFrame(all_summaries)
    out_path = args.out_dir / "sensitivity_results.parquet"
    results_df.to_parquet(out_path, index=False)

    out_json = args.out_dir / "sensitivity_results.json"
    out_json.write_text(json.dumps(all_summaries, indent=2) + "\n")

    print(f"\n\nWrote {len(all_summaries)} configurations to {out_path}")
    print(f"\n=== Sensitivity Matrix (mean onboard visibility) ===")
    pivot = results_df.pivot(index="fov_deg", columns="range_m", values="visibility_mean")
    print(pivot.round(3).to_string())

    print(f"\n=== Cooperative Benefit Matrix (1 - visibility) ===")
    pivot_cb = results_df.pivot(index="fov_deg", columns="range_m", values="cooperative_benefit")
    print(pivot_cb.round(3).to_string())


if __name__ == "__main__":
    main()
