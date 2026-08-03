#!/usr/bin/env python3
"""Quantify the ego-too-close-to-a-vehicle sampling artifact.

The uniform ego grid samples 15 equally-spaced positions per approach lane.
Some sampled positions can fall very close to (or on top of) a real vehicle,
which then subtends a large angular wedge and inflates the occluded/blind area
in a way a real AV (which keeps a following distance and cannot overlap another
vehicle's footprint) would never experience.

This script reuses the canonical per-(frame, ego) output
(outputs/ego_sweep_results.parquet) and joins it against the pinned tracks file
to measure, for every observation, the distance from the ego to the nearest
real vehicle (car/truck). It then reports:

  1. How observability and the occluded/out-of-FOV counts vary with that
     distance (binned).
  2. How much the headline intersection-wide visibility moves if we exclude
     ego positions that physically overlap a vehicle (clearance thresholds).

intersection_wide_visibility is defined in parametric_ego_sweep.py as the mean
of the per-observation visibility_ratio_onboard, so the corrected numbers here
use the identical definition.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SIM_DIR = Path(__file__).resolve().parents[1]
RESULTS = SIM_DIR / "outputs" / "ego_sweep_results.parquet"
TRACKS = Path("data/inputs/tracks.parquet")
OUT = SIM_DIR / "outputs" / "ego_proximity_bias.json"

# Match parametric_ego_sweep.SensorConfig defaults.
OFFSET_X = 1.5  # sensor forward offset from ego center (m)
OCCLUDER_CLASSES = {"car", "truck"}

# Distance bins (m) for the nearest real vehicle.
BIN_EDGES = [0.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, np.inf]
# Clearance thresholds (m) to test for the "exclude overlapping ego" correction.
CLEARANCE = [1.5, 2.0, 2.5, 3.0, 4.0]


def main() -> None:
    print("Loading canonical per-(frame, ego) results ...")
    ego = pd.read_parquet(
        RESULTS,
        columns=[
            "frame_idx", "ego_x", "ego_y", "ego_heading_deg",
            "n_objects_total", "n_visible_onboard", "n_occluded",
            "n_out_of_fov", "n_out_of_range", "visibility_ratio_onboard",
        ],
    )
    print(f"  {len(ego):,} observations across {ego['frame_idx'].nunique():,} frames")

    print("Loading tracks and extracting vehicle (car/truck) centers ...")
    tr = pd.read_parquet(TRACKS, columns=["frame_idx", "class_name", "bbox_center_x", "bbox_center_y"])
    veh = tr[tr["class_name"].isin(OCCLUDER_CLASSES)]
    veh_by_frame = {
        int(f): g[["bbox_center_x", "bbox_center_y"]].to_numpy(np.float64)
        for f, g in veh.groupby("frame_idx")
    }

    # Reconstruct the sensor position (occlusion is computed from here).
    head = np.radians(ego["ego_heading_deg"].to_numpy(np.float64))
    sx = ego["ego_x"].to_numpy(np.float64) + OFFSET_X * np.cos(head)
    sy = ego["ego_y"].to_numpy(np.float64) + OFFSET_X * np.sin(head)
    ex = ego["ego_x"].to_numpy(np.float64)
    ey = ego["ego_y"].to_numpy(np.float64)

    # Nearest-vehicle distance per observation, from both the sensor and the
    # ego center, computed per frame.
    n = len(ego)
    d_sensor = np.full(n, np.inf)
    d_center = np.full(n, np.inf)
    frame_arr = ego["frame_idx"].to_numpy()

    print("Computing nearest-vehicle distance per observation ...")
    order = np.argsort(frame_arr, kind="stable")
    f_sorted = frame_arr[order]
    # Boundaries of each frame block in the sorted order.
    uniq, starts = np.unique(f_sorted, return_index=True)
    ends = np.append(starts[1:], n)
    for f, s, e in zip(uniq, starts, ends):
        v = veh_by_frame.get(int(f))
        if v is None or len(v) == 0:
            continue  # no vehicles this frame -> distance stays inf
        idx = order[s:e]
        # sensor distances
        ds = np.hypot(sx[idx][:, None] - v[:, 0][None, :], sy[idx][:, None] - v[:, 1][None, :])
        d_sensor[idx] = ds.min(axis=1)
        dc = np.hypot(ex[idx][:, None] - v[:, 0][None, :], ey[idx][:, None] - v[:, 1][None, :])
        d_center[idx] = dc.min(axis=1)

    ego = ego.assign(d_sensor_nearest_veh=d_sensor, d_center_nearest_veh=d_center)

    vr = ego["visibility_ratio_onboard"]
    baseline = float(vr.mean())
    print(f"\nBaseline intersection-wide visibility (all obs): {baseline:.4f}")
    print(f"(canonical summary reports 0.3869 -- should match)\n")

    # ---- 1. Distribution / trend binned by sensor-to-nearest-vehicle ----
    labels = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        hi_s = "inf" if math.isinf(hi) else f"{hi:g}"
        labels.append(f"[{lo:g},{hi_s})")
    ego["dist_bin"] = pd.cut(ego["d_sensor_nearest_veh"], bins=BIN_EDGES, labels=labels, right=False)

    print("=== Observability vs. distance to nearest real vehicle (sensor-based) ===")
    rows = []
    grp = ego.groupby("dist_bin", observed=False)
    for lab, g in grp:
        if len(g) == 0:
            continue
        rows.append({
            "bin_m": str(lab),
            "n_obs": int(len(g)),
            "pct_obs": round(100 * len(g) / n, 3),
            "mean_visibility": round(float(g["visibility_ratio_onboard"].mean()), 4),
            "mean_occluded": round(float(g["n_occluded"].mean()), 3),
            "mean_out_of_fov": round(float(g["n_out_of_fov"].mean()), 3),
            "mean_total_obj": round(float(g["n_objects_total"].mean()), 3),
        })
    dist_table = pd.DataFrame(rows)
    print(dist_table.to_string(index=False))

    # ---- 2. Corrected headline excluding ego positions that overlap a vehicle ----
    print("\n=== Headline correction: exclude ego positions within clearance of a vehicle ===")
    print("(clearance measured ego-center to nearest vehicle center)\n")
    corr_rows = [{
        "rule": "baseline (keep all)",
        "clearance_m": 0.0,
        "n_kept": n,
        "pct_dropped": 0.0,
        "int_wide_visibility": round(baseline, 4),
        "delta": 0.0,
    }]
    for c in CLEARANCE:
        keep = ego["d_center_nearest_veh"] >= c
        kept = ego.loc[keep, "visibility_ratio_onboard"]
        v = float(kept.mean())
        corr_rows.append({
            "rule": f"drop ego center < {c:g} m from a vehicle",
            "clearance_m": c,
            "n_kept": int(keep.sum()),
            "pct_dropped": round(100 * (1 - keep.mean()), 3),
            "int_wide_visibility": round(v, 4),
            "delta": round(v - baseline, 4),
        })
    corr_table = pd.DataFrame(corr_rows)
    print(corr_table.to_string(index=False))

    OUT.write_text(json.dumps({
        "baseline_int_wide_visibility": baseline,
        "n_observations": n,
        "distance_trend_sensor": rows,
        "headline_correction_center": corr_rows,
    }, indent=2) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
