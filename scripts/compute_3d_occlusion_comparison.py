#!/usr/bin/env python3
"""Compare planar occlusion with a first-order vertical line-of-sight correction.

Planar model (current paper): a nearer car/truck blocks everything behind it in its
bearing -- purely planar.
Vertical-corrected model: an object behind an occluder is still VISIBLE if its top clears the
occluder along the sensor->target line of sight. With an elevated sensor (h_s)
and real object heights (bbox_dz), a tall target (truck) pokes over a car, and
the sensor sees over short vehicles at distance.

  ray height at the occluder's range r_o  =  h_s + (h_target - h_s) * (r_o / r_target)
  visible-over-top  iff  ray_height >= h_occluder_top

Azimuth occlusion logic is replicated verbatim from compute_ego_occlusion so the
Planar numbers match the paper; only the vertical test is added. Runs on a frame
sample for speed. Heights are taken as object height above ground (bbox_dz).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parametric_ego_sweep import (SensorConfig, DEFAULT_APPROACHES,
                                  _box_corners_xy, _normalize_angle)

TRACKS_DEFAULT = "data/inputs/tracks.parquet"
CLASSES = ["car", "truck", "pedestrian", "bicycle"]
OCCLUDERS = {"car", "truck"}


def classify_frame(frame_objects, ego_x, ego_y, ego_heading, sensor):
    """Return list of (class, status, h_target, r_occ, r_target, h_occ) per object.
    status in {'out','vis2d','occ2d'}. Azimuth logic mirrors compute_ego_occlusion."""
    half_fov = sensor.half_fov_rad
    sx = ego_x + sensor.offset_x * math.cos(ego_heading) - sensor.offset_y * math.sin(ego_heading)
    sy = ego_y + sensor.offset_x * math.sin(ego_heading) + sensor.offset_y * math.cos(ego_heading)

    objs = []
    for row in frame_objects.itertuples(index=False):
        dx = float(row.bbox_center_x) - sx
        dy = float(row.bbox_center_y) - sy
        rng = math.hypot(dx, dy)
        rel = float(_normalize_angle(np.array([math.atan2(dy, dx) - ego_heading]))[0])
        objs.append({"cls": str(row.class_name), "cx": float(row.bbox_center_x),
                     "cy": float(row.bbox_center_y), "dxb": float(row.bbox_dx),
                     "dyb": float(row.bbox_dy), "yaw": float(row.bbox_yaw),
                     "h": float(row.bbox_dz), "rng": rng, "rel": rel})
    objs.sort(key=lambda o: o["rng"])

    n = sensor.theta_bins
    fov_start, fov_end = -half_fov, half_fov
    bw = (fov_end - fov_start) / n
    min_r = np.full(n, np.inf)
    occ_h = np.zeros(n)

    out = []
    for o in objs:
        in_fov = abs(o["rel"]) <= half_fov
        in_range = sensor.near_blind_m < o["rng"] <= sensor.range_m
        corners = _box_corners_xy(o["cx"], o["cy"], o["dxb"], o["dyb"], o["yaw"])
        cdx = corners[:, 0] - sx
        cdy = corners[:, 1] - sy
        cbr = _normalize_angle(np.arctan2(cdy, cdx) - ego_heading)
        near = float(np.min(np.hypot(cdx, cdy)))

        if not (in_fov and in_range):
            out.append((o["cls"], "out", o["h"], 0.0, o["rng"], 0.0))
        else:
            cb = max(0, min(n - 1, int((o["rel"] - fov_start) / bw)))
            br = min_r[cb]
            if o["rng"] > br + 0.5:
                out.append((o["cls"], "occ2d", o["h"], float(br), o["rng"], float(occ_h[cb])))
            else:
                out.append((o["cls"], "vis2d", o["h"], 0.0, o["rng"], 0.0))

        if o["cls"] in OCCLUDERS:
            for cb in cbr:
                b = float(cb)
                if fov_start <= b <= fov_end:
                    bi = max(0, min(n - 1, int((b - fov_start) / bw)))
                    if near < min_r[bi]:
                        min_r[bi] = near; occ_h[bi] = o["h"]
            valid = sorted(max(0, min(n - 1, int((float(b) - fov_start) / bw)))
                           for b in cbr if fov_start <= float(b) <= fov_end)
            if len(valid) >= 2:
                lo, hi = valid[0], valid[-1]
                full = (fov_end - fov_start) >= 2 * math.pi - 1e-6
                fill = (list(range(hi, n)) + list(range(0, lo + 1))) if (full and (hi - lo) > n // 2) else range(lo, hi + 1)
                for bi in fill:
                    if near < min_r[bi]:
                        min_r[bi] = near; occ_h[bi] = o["h"]
    return out


HS_GRID = [1.5, 1.8, 2.1, 2.5, 3.0, 4.0]


def analyze(fov, by_frame, sample, egos, sensor_h):
    """Compare planar and vertical-corrected results at one field of view."""
    sensor = SensorConfig(fov_deg=fov)
    tot = {c: 0 for c in CLASSES}; v2 = {c: 0 for c in CLASSES}
    occ = {c: {"h": [], "ro": [], "rt": [], "ho": []} for c in CLASSES}

    for fi in sample:
        g = by_frame.get(fi)
        if g is None or g.empty:
            continue
        for (ex, ey, eh) in egos:
            for cls, st, h, ro, rt, ho in classify_frame(g, ex, ey, eh, sensor):
                if cls not in tot:
                    continue
                tot[cls] += 1
                if st == "vis2d":
                    v2[cls] += 1
                elif st == "occ2d":
                    occ[cls]["h"].append(h); occ[cls]["ro"].append(ro)
                    occ[cls]["rt"].append(rt); occ[cls]["ho"].append(ho)

    def recovered(cls, hs):
        h = np.array(occ[cls]["h"]); ro = np.array(occ[cls]["ro"])
        rt = np.array(occ[cls]["rt"]); ho = np.array(occ[cls]["ho"])
        if h.size == 0:
            return 0
        ray = hs + (h - hs) * (ro / rt)
        return int(np.sum(ray >= ho))

    rows = []
    T = sum(tot.values()); V2 = sum(v2.values())
    V3 = V2 + sum(recovered(c, sensor_h) for c in CLASSES)
    rows.append(("Overall", T, V2 / T, V3 / T))
    for c in CLASSES:
        if tot[c] == 0:
            continue
        rows.append((c.capitalize(), tot[c], v2[c] / tot[c],
                     (v2[c] + recovered(c, sensor_h)) / tot[c]))

    print(f"\n  === FOV {int(fov)}deg ===")
    print("  scope        N      planar  vertical-corrected(h_s=%.1f)   delta" % sensor_h)
    for name, N, r2, r3 in rows:
        print(f"  {name:11s} {N:9d}  {r2:.3f}   {r3:.3f}     +{r3-r2:.3f}")

    curve = [(V2 + sum(recovered(c, hs) for c in CLASSES)) / T for hs in HS_GRID]
    return rows, V2 / T, curve


def _panel_bars(ax, rows, fov, sensor_h, label):
    """Horizontal bars: class names sit on the y-axis at full length (no rotation
    or truncation) and value labels sit clear to the right of each bar, so
    nothing collides with a neighbouring bar."""
    names = [r[0] for r in rows]
    r2s = [r[2] for r in rows]; r3s = [r[3] for r in rows]
    y = np.arange(len(names)); h = 0.36
    ax.barh(y + h/2, r2s, h, label="Planar", color="#E69F00", edgecolor="black", linewidth=0.4)
    ax.barh(y - h/2, r3s, h, label=f"Vertical-corrected ($h_s$={sensor_h} m)", color="#0072B2", edgecolor="black", linewidth=0.4)
    for yi, (a2, a3) in enumerate(zip(r2s, r3s)):
        ax.text(a2 + 0.02, yi + h/2, f"{a2:.2f}", ha="left", va="center", fontsize=10.0)
        ax.text(a3 + 0.02, yi - h/2, f"{a3:.2f}", ha="left", va="center", fontsize=10.0)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10.0)
    # Overall at the top, plus a blank band at the bottom so the legend in panel
    # (a) never sits on top of the last row's bars or value labels.
    ax.set_ylim(len(names) + 1.05, -0.7)
    ax.set_xlabel("Observability", fontsize=10.0)
    ax.set_xlim(0, 1.42)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_title(f"({label}) FOV {int(fov)}\u00b0", fontsize=10.0)
    if label == "a":
        ax.legend(fontsize=10.0, loc="lower right", framealpha=0.95,
                  borderpad=0.25, labelspacing=0.25, handlelength=1.3, handletextpad=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="x", labelsize=10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=TRACKS_DEFAULT)
    ap.add_argument("--fovs", type=float, nargs="+", default=[120.0, 360.0],
                    help="One or more FOVs; with two, the figure gains a per-FOV bar panel each.")
    ap.add_argument("--n-frames", type=int, default=800)
    ap.add_argument("--sensor-h", type=float, default=1.8)
    ap.add_argument("--out", default="figures/occlusion_2d_vs_3d")
    a = ap.parse_args()

    egos = [p for lane in DEFAULT_APPROACHES for p in lane.sample_positions()]
    df = pd.read_parquet(a.tracks, columns=["frame_idx", "class_name", "bbox_center_x",
                                            "bbox_center_y", "bbox_dx", "bbox_dy", "bbox_dz", "bbox_yaw"])
    frames = np.sort(df["frame_idx"].unique())
    sample = frames[np.linspace(0, len(frames) - 1, min(a.n_frames, len(frames))).astype(int)]
    sub = df[df["frame_idx"].isin(sample)]
    by_frame = {f: g for f, g in sub.groupby("frame_idx")}
    print(f"tracks={len(df)} frames_total={len(frames)} sampled={len(sample)} "
          f"egos={len(egos)} fovs={a.fovs} h_s={a.sensor_h}")

    results = [(fov, *analyze(fov, by_frame, sample, egos, a.sensor_h)) for fov in a.fovs]

    # ---- figure: one bar panel per FOV, then a shared sensor-height panel ----
    n_bar = len(results)
    fig, axes = plt.subplots(1, n_bar + 1, figsize=(6.5, 3.40))
    axes = np.atleast_1d(axes)
    letters = "abcdefg"
    for i, (fov, rows, _r2, _curve) in enumerate(results):
        _panel_bars(axes[i], rows, fov, a.sensor_h, letters[i])

    axH = axes[n_bar]
    colors = ["#0072B2", "#009E73", "#CC79A7"]
    for i, (fov, _rows, r2, curve) in enumerate(results):
        c = colors[i % len(colors)]
        axH.axhline(r2, color=c, ls="--", lw=1.3, alpha=0.75)
        axH.plot(HS_GRID, curve, "-o", color=c, lw=1.8, ms=4.5, label=f"Vertical-corrected, {int(fov)}°")
        axH.annotate(f"Planar {int(fov)}° = {r2:.2f}", xy=(HS_GRID[0], r2), xytext=(0, -9),
                     textcoords="offset points", fontsize=10.0, color=c)
    axH.axvline(a.sensor_h, color="#999", ls=":", lw=1)
    axH.set_xlabel("Sensor height $h_s$ (m)", fontsize=10.0)
    axH.set_ylabel("Observability", fontsize=10.0)
    axH.set_title(f"({letters[n_bar]}) Sensor-height sensitivity", fontsize=10.0)
    axH.set_ylim(0, 1.05); axH.legend(fontsize=10.0, loc="lower right"); axH.grid(True, alpha=0.3)
    axH.tick_params(labelsize=10.0)

    fig.tight_layout(pad=0.4, w_pad=1.1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(f"{a.out}.{fmt}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
