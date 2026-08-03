#!/usr/bin/env python3
"""Robustness variant: use detected vehicles as ego positions.

The headline analysis (parametric_ego_sweep.py) places ego sensors on a uniform
15-per-approach grid. A reviewer may object that equally-spaced grid points can
land on top of a real vehicle, inflating occlusion. This script answers that by
re-running the identical angular-wedge occlusion model with ego positions taken
from the *detected vehicles themselves* (cars/trucks), inheriting each vehicle's
measured yaw as the sensor heading, and excluding each ego vehicle from its own
observed/occluder set. Real vehicles keep realistic spacing and never overlap,
so this is free of the grid's overlap artifact.

It reuses compute_ego_occlusion and SensorConfig from parametric_ego_sweep and
the identical metric definitions (PROXIMITY_R, CONFLICT_ZONE_R, CONFLICT_CENTER,
visibility_ratio = n_visible / n_total), so the numbers are directly comparable
to outputs/ego_sweep_summary.json. The canonical pipeline is not modified.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

from parametric_ego_sweep import SensorConfig, compute_ego_occlusion

# Identical to parametric_ego_sweep.analyze_frame.
PROXIMITY_R = 15.0
CONFLICT_ZONE_R = 20.0
CONFLICT_CENTER = (7.5, 8.0)
OCCLUDER_CLASSES = frozenset({"truck", "car"})
EGO_CLASSES = frozenset({"truck", "car"})  # an AV is a car/truck

TRACKS = Path("data/inputs/tracks.parquet")
SIM_DIR = Path(__file__).resolve().parents[1]

_WCTX: dict = {}


def analyze_frame_vehicle_ego(frame_idx, frame_tracks, sensor, occluder_classes,
                              ego_classes, heading_mode="yaw"):
    """Per-frame analysis with each detected vehicle acting as the ego sensor.

    Mirrors parametric_ego_sweep.analyze_frame, except (a) ego positions are the
    car/truck detections in this frame, (b) the ego vehicle's own track is
    removed from the scene before computing occlusion (a sensor cannot detect or
    be occluded by itself).

    heading_mode:
      "yaw"    -- sensor points along the vehicle's measured orientation (the
                  realistic case: includes outbound and cross-street vehicles).
      "inward" -- sensor points at the intersection center (mimics the grid's
                  always-approaching assumption; isolates the position effect).
    """
    results = []
    ego_rows = frame_tracks[frame_tracks["class_name"].isin(ego_classes)]
    for ego in ego_rows.itertuples(index=False):
        ego_tid = int(ego.track_id)
        ego_x = float(ego.bbox_center_x)
        ego_y = float(ego.bbox_center_y)
        yaw = float(ego.bbox_yaw)
        bearing_to_center = math.atan2(CONFLICT_CENTER[1] - ego_y, CONFLICT_CENTER[0] - ego_x)
        # "Inbound" = facing roughly toward the intersection (within +-90 deg).
        ang = (bearing_to_center - yaw + math.pi) % (2 * math.pi) - math.pi
        is_inbound = abs(ang) < math.pi / 2
        ego_heading = bearing_to_center if heading_mode == "inward" else yaw

        # Self-exclusion: the ego cannot see/occlude itself.
        frame_objects = frame_tracks[frame_tracks["track_id"] != ego_tid]

        vis = compute_ego_occlusion(
            ego_x, ego_y, ego_heading, frame_objects, sensor, occluder_classes,
        )

        n_total = len(vis)
        n_visible = sum(1 for v in vis.values() if v["visible"])
        n_occluded = sum(1 for v in vis.values()
                         if not v["visible"] and v["reason"] == "occluded")
        n_out_fov = sum(1 for v in vis.values()
                        if not v["visible"] and v["reason"] == "out_of_fov")
        n_out_range = sum(1 for v in vis.values()
                          if not v["visible"] and v["reason"] == "out_of_range")

        # Nearby (within 15 m of ego) -- identical to canonical.
        n_nearby_total = sum(1 for v in vis.values() if v["range_m"] <= PROXIMITY_R)
        n_nearby_visible = sum(1 for v in vis.values()
                               if v["range_m"] <= PROXIMITY_R and v["visible"])

        # Conflict zone (within 20 m of intersection center), self excluded.
        n_cz_total = 0
        n_cz_visible = 0
        for row_obj in frame_objects.itertuples(index=False):
            d = math.hypot(float(row_obj.bbox_center_x) - CONFLICT_CENTER[0],
                           float(row_obj.bbox_center_y) - CONFLICT_CENTER[1])
            if d <= CONFLICT_ZONE_R:
                n_cz_total += 1
                tid = int(row_obj.track_id)
                if tid in vis and vis[tid]["visible"]:
                    n_cz_visible += 1

        results.append({
            "frame_idx": int(frame_idx),
            "ego_track_id": ego_tid,
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_heading_deg": math.degrees(ego_heading),
            "ego_dist_to_center": math.hypot(ego_x - CONFLICT_CENTER[0],
                                             ego_y - CONFLICT_CENTER[1]),
            "ego_inbound": bool(is_inbound),
            "n_objects_total": n_total,
            "n_visible_onboard": n_visible,
            "n_occluded": n_occluded,
            "n_out_of_fov": n_out_fov,
            "n_out_of_range": n_out_range,
            "visibility_ratio_onboard": (n_visible / n_total) if n_total > 0 else float("nan"),
            "n_nearby_total": n_nearby_total,
            "n_nearby_visible": n_nearby_visible,
            "nearby_visibility_ratio": (n_nearby_visible / n_nearby_total) if n_nearby_total > 0 else float("nan"),
            "n_conflict_zone_total": n_cz_total,
            "n_conflict_zone_visible": n_cz_visible,
            "conflict_zone_visibility_ratio": (n_cz_visible / n_cz_total) if n_cz_total > 0 else float("nan"),
        })
    return results


def _worker_init(sensor, occluder_classes, ego_classes, heading_mode):
    _WCTX["sensor"] = sensor
    _WCTX["occluder_classes"] = occluder_classes
    _WCTX["ego_classes"] = ego_classes
    _WCTX["heading_mode"] = heading_mode


def _worker_frame(task):
    fidx, frame_tracks = task
    return analyze_frame_vehicle_ego(
        fidx, frame_tracks, _WCTX["sensor"], _WCTX["occluder_classes"],
        _WCTX["ego_classes"], _WCTX["heading_mode"],
    )


def _summarize(df: pd.DataFrame) -> dict:
    nearby = df.dropna(subset=["nearby_visibility_ratio"])
    cz = df.dropna(subset=["conflict_zone_visibility_ratio"])
    return {
        "n_ego_observations": int(len(df)),
        "frames": int(df["frame_idx"].nunique()),
        "mean_ego_per_frame": round(len(df) / max(df["frame_idx"].nunique(), 1), 3),
        "intersection_wide_visibility": round(float(df["visibility_ratio_onboard"].mean()), 4),
        "nearby_visibility": round(float(nearby["nearby_visibility_ratio"].mean()), 4) if len(nearby) else None,
        "conflict_zone_visibility": round(float(cz["conflict_zone_visibility_ratio"].mean()), 4) if len(cz) else None,
        "mean_counts": {
            "total_objects": round(float(df["n_objects_total"].mean()), 3),
            "visible": round(float(df["n_visible_onboard"].mean()), 3),
            "occluded": round(float(df["n_occluded"].mean()), 3),
            "out_of_fov": round(float(df["n_out_of_fov"].mean()), 3),
            "out_of_range": round(float(df["n_out_of_range"].mean()), 3),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--fov-deg", type=float, default=120.0)
    ap.add_argument("--range-m", type=float, default=35.0)
    ap.add_argument("--region-radius-m", type=float, default=28.0,
                    help="Region-matched subset: egos within this radius of the intersection center "
                         "(matches the grid's ~27 m reach).")
    ap.add_argument("--heading-mode", choices=["yaw", "inward"], default="yaw",
                    help="yaw = realistic measured heading; inward = aim at intersection center "
                         "(isolates the position effect from the heading effect).")
    ap.add_argument("--n-procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        tag = f"fov{args.fov_deg:g}_{args.heading_mode}"
        args.out = SIM_DIR / "outputs" / f"vehicle_as_ego_{tag}.json"

    sensor = SensorConfig(fov_deg=args.fov_deg, range_m=args.range_m)
    print(f"Loading tracks from {args.tracks}")
    cols = ["frame_idx", "track_id", "class_name",
            "bbox_center_x", "bbox_center_y", "bbox_dx", "bbox_dy", "bbox_yaw"]
    tracks = pd.read_parquet(args.tracks, columns=cols)
    print(f"  {len(tracks):,} track rows; "
          f"{tracks['class_name'].isin(EGO_CLASSES).sum():,} car/truck ego candidates")

    grouped = tracks.groupby("frame_idx")
    tasks = ((int(f), g) for f, g in grouped)

    all_results = []
    if args.n_procs > 1:
        print(f"Parallelizing across {args.n_procs} processes (heading={args.heading_mode}) ...")
        with mp.Pool(args.n_procs, initializer=_worker_init,
                     initargs=(sensor, OCCLUDER_CLASSES, EGO_CLASSES, args.heading_mode)) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker_frame, tasks, chunksize=64)):
                all_results.extend(res)
                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1} frames")
    else:
        for f, g in tasks:
            all_results.extend(analyze_frame_vehicle_ego(
                f, g, sensor, OCCLUDER_CLASSES, EGO_CLASSES, args.heading_mode))

    df = pd.DataFrame(all_results)
    print(f"\n{len(df):,} ego-vehicle observations\n")

    all_summary = _summarize(df)
    region = df[df["ego_dist_to_center"] <= args.region_radius_m]
    region_summary = _summarize(region)
    region_inbound = region[region["ego_inbound"]]
    region_inbound_summary = _summarize(region_inbound)
    print(f"Inbound share within {args.region_radius_m:g} m: "
          f"{100 * region['ego_inbound'].mean():.1f}%")

    # Grid baseline for side-by-side comparison (matched to this FOV).
    ms = json.load(open(SIM_DIR / "outputs" / "multi_sensor_rigorous.json"))
    fov_key = f"{args.fov_deg:g}"
    g = ms["configs"].get(fov_key)
    if g is not None:
        grid_row = {
            "design": "uniform grid (60/frame)",
            "n_obs": g["frames"] * 60,
            "int_wide": g["int_wide"], "nearby": g["nearby"], "conflict": g["conflict"],
            "occluded": g["occluded"], "out_fov": g["out_fov"],
            "out_range": g["out_range"], "total_obj": g["total"],
        }
    else:
        grid_row = {"design": f"uniform grid (no {fov_key}-deg baseline)", "n_obs": None,
                    "int_wide": None, "nearby": None, "conflict": None,
                    "occluded": None, "out_fov": None, "out_range": None, "total_obj": None}

    def _row(name, s):
        c = s["mean_counts"]
        return {
            "design": name,
            "n_obs": s["n_ego_observations"],
            "int_wide": s["intersection_wide_visibility"],
            "nearby": s["nearby_visibility"],
            "conflict": s["conflict_zone_visibility"],
            "occluded": c["occluded"],
            "out_fov": c["out_of_fov"],
            "out_range": c["out_of_range"],
            "total_obj": c["total_objects"],
        }

    rr = f"{args.region_radius_m:g}"
    table = pd.DataFrame([
        grid_row,
        _row("vehicle-as-ego (all)", all_summary),
        _row(f"vehicle-as-ego (<{rr} m)", region_summary),
        _row(f"vehicle-as-ego (<{rr} m, inbound)", region_inbound_summary),
    ])
    print(f"=== Grid vs. vehicle-as-ego (FOV {args.fov_deg:g} deg, range {args.range_m:g} m, "
          f"heading={args.heading_mode}) ===")
    print(table.to_string(index=False))

    args.out.write_text(json.dumps({
        "sensor": {"fov_deg": args.fov_deg, "range_m": args.range_m},
        "heading_mode": args.heading_mode,
        "region_radius_m": args.region_radius_m,
        "grid_baseline": grid_row,
        "vehicle_as_ego_all": all_summary,
        "vehicle_as_ego_region": region_summary,
        "vehicle_as_ego_region_inbound": region_inbound_summary,
    }, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
