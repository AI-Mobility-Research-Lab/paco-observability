#!/usr/bin/env python3
"""How long do vehicles follow a lead truck? (context for the passing time benefit)

For each car track, mark frames where a truck is within LEAD_RANGE ahead (within
a forward cone), then measure consecutive-frame run durations. At a signalized
intersection these episodes are brief, so the confident-passing benefit is
chiefly safety (revealed hidden zone) rather than travel time at this site.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TRACKS_DEFAULT = "data/inputs/tracks.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=TRACKS_DEFAULT)
    ap.add_argument("--lead-range", type=float, default=15.0)
    ap.add_argument("--cone-deg", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=10.0)
    a = ap.parse_args()
    cone = math.radians(a.cone_deg)

    df = pd.read_parquet(a.tracks, columns=["frame_idx", "track_id", "class_name",
                                            "bbox_center_x", "bbox_center_y", "bbox_yaw"])
    cars = df[df.class_name == "car"]
    trucks = df[df.class_name == "truck"]
    trucks_by_f = {f: g[["bbox_center_x", "bbox_center_y"]].values for f, g in trucks.groupby("frame_idx")}

    runs = []
    for tid, g in cars.groupby("track_id"):
        g = g.sort_values("frame_idx")
        fs = g["frame_idx"].values; xs = g["bbox_center_x"].values
        ys = g["bbox_center_y"].values; yaw = g["bbox_yaw"].values
        foll = np.zeros(len(fs), bool)
        for i, f in enumerate(fs):
            tk = trucks_by_f.get(f)
            if tk is None:
                continue
            dx = tk[:, 0] - xs[i]; dy = tk[:, 1] - ys[i]; d = np.hypot(dx, dy)
            ang = np.abs((np.arctan2(dy, dx) - yaw[i] + np.pi) % (2 * np.pi) - np.pi)
            if np.any((d < a.lead_range) & (d > 0.5) & (ang < cone)):
                foll[i] = True
        i = 0
        while i < len(fs):
            if foll[i]:
                j = i
                while j + 1 < len(fs) and foll[j + 1] and fs[j + 1] == fs[j] + 1:
                    j += 1
                runs.append((j - i + 1) / a.hz); i = j + 1
            else:
                i += 1
    runs = np.array(runs)
    print(f"following-a-truck episodes: {len(runs)}")
    print(f"duration (s): mean {runs.mean():.2f}  median {np.median(runs):.2f}  "
          f"p90 {np.percentile(runs, 90):.1f}  max {runs.max():.1f}")
    print(f"episodes >=2s: {(runs >= 2).mean():.2f}   >=5s: {(runs >= 5).mean():.2f}")


if __name__ == "__main__":
    main()
