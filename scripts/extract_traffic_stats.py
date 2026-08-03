#!/usr/bin/env python3
"""Extract traffic statistics from tracked trajectory data for Part C manuscript tables.

Produces a JSON summary with class counts, speed distributions, dimension statistics,
arrival rates, and near-miss breakdowns suitable for manuscript tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def compute_speeds(tracks: pd.DataFrame, dt: float = 0.1) -> pd.Series:
    """Compute frame-to-frame speed (m/s) for each track row."""
    tracks = tracks.sort_values(["track_id", "frame_idx"])
    dx = tracks.groupby("track_id")["bbox_center_x"].diff()
    dy = tracks.groupby("track_id")["bbox_center_y"].diff()
    dframe = tracks.groupby("track_id")["frame_idx"].diff()
    speed = np.hypot(dx, dy) / (dframe * dt)
    return speed


def class_summary(tracks: pd.DataFrame) -> dict:
    """Per-class count and duration statistics."""
    rows_per_class = tracks.groupby("class_name").size().to_dict()
    tracks_per_class = tracks.groupby("class_name")["track_id"].nunique().to_dict()

    track_lengths = tracks.groupby(["class_name", "track_id"]).size().reset_index(name="n_frames")
    length_stats = (
        track_lengths.groupby("class_name")["n_frames"]
        .agg(["mean", "median", "std", "min", "max"])
        .round(1)
        .to_dict(orient="index")
    )

    return {
        "frame_level_rows": rows_per_class,
        "unique_tracks": tracks_per_class,
        "track_length_frames": length_stats,
    }


def speed_summary(tracks: pd.DataFrame) -> dict:
    """Per-class speed statistics (m/s)."""
    out = {}
    for cls, grp in tracks.groupby("class_name"):
        s = grp["speed_mps"].dropna()
        if len(s) == 0:
            continue
        out[cls] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2),
            "p05": round(float(s.quantile(0.05)), 2),
            "p95": round(float(s.quantile(0.95)), 2),
        }
    return out


def dimension_summary(tracks: pd.DataFrame) -> dict:
    """Per-class bounding box dimension statistics."""
    out = {}
    for cls, grp in tracks.groupby("class_name"):
        out[cls] = {}
        for dim in ["bbox_dx", "bbox_dy", "bbox_dz"]:
            vals = grp[dim].dropna()
            out[cls][dim] = {
                "mean": round(float(vals.mean()), 2),
                "median": round(float(vals.median()), 2),
                "std": round(float(vals.std()), 2),
            }
    return out


def near_miss_summary(nm: pd.DataFrame) -> dict:
    """Near-miss pair type breakdown and severity distribution."""
    # Normalize class names
    nm = nm.copy()
    nm["class_a"] = nm["class_a"].str.lower().str.replace("bike", "bicycle")
    nm["class_b"] = nm["class_b"].str.lower().str.replace("bike", "bicycle")
    nm["pair_type"] = nm.apply(
        lambda r: "-".join(sorted([r["class_a"], r["class_b"]])), axis=1
    )

    pair_counts = nm["pair_type"].value_counts().to_dict()
    unique_frames = int(nm["frame_idx"].nunique())

    ttc_valid = nm["ttc"].dropna()
    ttc_stats = {}
    if len(ttc_valid) > 0:
        ttc_stats = {
            "count": int(len(ttc_valid)),
            "mean": round(float(ttc_valid.mean()), 3),
            "median": round(float(ttc_valid.median()), 3),
            "std": round(float(ttc_valid.std()), 3),
            "min": round(float(ttc_valid.min()), 3),
            "p05": round(float(ttc_valid.quantile(0.05)), 3),
            "p95": round(float(ttc_valid.quantile(0.95)), 3),
        }

    sep_valid = nm["min_separation"].dropna()
    sep_stats = {}
    if len(sep_valid) > 0:
        sep_stats = {
            "count": int(len(sep_valid)),
            "mean": round(float(sep_valid.mean()), 3),
            "median": round(float(sep_valid.median()), 3),
            "std": round(float(sep_valid.std()), 3),
            "min": round(float(sep_valid.min()), 3),
        }

    return {
        "total_rows": int(len(nm)),
        "unique_frames": unique_frames,
        "pair_type_counts": pair_counts,
        "ttc_stats": ttc_stats,
        "separation_stats": sep_stats,
    }


def arrival_rates(tracks: pd.DataFrame, dt: float = 0.1) -> dict:
    """Estimate arrival rates (tracks per minute) by class."""
    total_frames = tracks["frame_idx"].nunique()
    duration_min = total_frames * dt / 60.0

    tracks_per_class = tracks.groupby("class_name")["track_id"].nunique()
    rates = (tracks_per_class / duration_min).round(2).to_dict()

    return {
        "total_frames": int(total_frames),
        "duration_minutes": round(duration_min, 1),
        "tracks_per_minute_by_class": rates,
        "total_tracks_per_minute": round(float(sum(rates.values())), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract traffic statistics for Part C manuscript.")
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path("data/inputs/tracks.parquet"),
    )
    parser.add_argument(
        "--near-misses",
        type=Path,
        default=Path("data/inputs/near_misses.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/derived/traffic_stats.json"
        ),
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Frame interval in seconds (10 Hz)")
    args = parser.parse_args()

    print(f"Loading tracks from {args.tracks}")
    tracks = pd.read_parquet(args.tracks)
    tracks["speed_mps"] = compute_speeds(tracks, dt=args.dt)
    print(f"  {len(tracks):,} rows, {tracks['track_id'].nunique():,} tracks")

    print(f"Loading near-misses from {args.near_misses}")
    nm = pd.read_parquet(args.near_misses)
    print(f"  {len(nm):,} rows")

    summary = {
        "data_overview": {
            "total_track_rows": int(len(tracks)),
            "unique_tracks": int(tracks["track_id"].nunique()),
            "frame_range": [int(tracks["frame_idx"].min()), int(tracks["frame_idx"].max())],
            "classes": sorted(tracks["class_name"].unique().tolist()),
        },
        "class_summary": class_summary(tracks),
        "speed_summary": speed_summary(tracks),
        "dimension_summary": dimension_summary(tracks),
        "arrival_rates": arrival_rates(tracks, dt=args.dt),
        "near_miss_summary": near_miss_summary(nm),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote summary to {args.out}")

    # Print key numbers for manuscript
    print("\n=== Key Numbers for Manuscript ===")
    ar = summary["arrival_rates"]
    print(f"Duration: {ar['duration_minutes']:.1f} min ({ar['total_frames']:,} frames at {1/args.dt:.0f} Hz)")
    print(f"Total tracks: {summary['data_overview']['unique_tracks']:,}")
    for cls in sorted(summary["class_summary"]["unique_tracks"].keys()):
        n = summary["class_summary"]["unique_tracks"][cls]
        rate = ar["tracks_per_minute_by_class"].get(cls, 0)
        print(f"  {cls}: {n:,} tracks ({rate:.1f}/min)")
    print(f"Near-miss events: {summary['near_miss_summary']['total_rows']:,} across {summary['near_miss_summary']['unique_frames']:,} frames")
    for pt, cnt in sorted(summary["near_miss_summary"]["pair_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {pt}: {cnt:,}")


if __name__ == "__main__":
    main()
