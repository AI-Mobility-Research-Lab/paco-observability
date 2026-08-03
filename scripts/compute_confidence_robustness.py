#!/usr/bin/env python3
"""Robustness of observability to detection-confidence thresholds.

The onboard AV and the roadside unit (RSU) each accept a detection only if its
confidence exceeds a threshold. We use the roadside detector's per-object score
as a confidence proxy (the AV case is a proxy: detectability correlates with
size/point density). We then vary:
  - tau_AV : the onboard confidence threshold (gates onboard-visible objects)
  - tau_RSU: the RSU confidence threshold (gates cooperatively-added objects)

Onboard geometry (FOV + range + angular-wedge occlusion) is the exact paper
model (compute_ego_occlusion). Outputs:
  (a) observability vs a uniform threshold for 120 deg, 360 deg, and V2I;
  (b) cooperative observability over the (tau_AV, tau_RSU) grid.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parametric_ego_sweep import SensorConfig, DEFAULT_APPROACHES, compute_ego_occlusion

TRACKS_DEFAULT = "data/inputs/tracks.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=TRACKS_DEFAULT)
    ap.add_argument("--n-frames", type=int, default=800)
    ap.add_argument("--out", default="figures/confidence_robustness")
    a = ap.parse_args()

    s120 = SensorConfig(fov_deg=120.0)
    s360 = SensorConfig(fov_deg=360.0)
    egos = [p for lane in DEFAULT_APPROACHES for p in lane.sample_positions()]

    df = pd.read_parquet(a.tracks, columns=["frame_idx", "track_id", "class_name", "score",
                                            "bbox_center_x", "bbox_center_y", "bbox_dx",
                                            "bbox_dy", "bbox_yaw"])
    frames = np.sort(df["frame_idx"].unique())
    sample = frames[np.linspace(0, len(frames) - 1, min(a.n_frames, len(frames))).astype(int)]
    by_frame = {f: g for f, g in df[df["frame_idx"].isin(sample)].groupby("frame_idx")}
    print(f"frames sampled={len(sample)} egos={len(egos)}")

    scores, v120, v360 = [], [], []
    for fi in sample:
        g = by_frame.get(fi)
        if g is None or g.empty:
            continue
        for (ex, ey, eh) in egos:
            r120 = compute_ego_occlusion(ex, ey, eh, g, s120)
            r360 = compute_ego_occlusion(ex, ey, eh, g, s360)
            for row in g.itertuples(index=False):
                tid = int(row.track_id)
                scores.append(float(row.score))
                v120.append(bool(r120.get(tid, {}).get("visible", False)))
                v360.append(bool(r360.get(tid, {}).get("visible", False)))

    scores = np.array(scores); v120 = np.array(v120); v360 = np.array(v360)
    N = scores.size
    print(f"observations={N}  baseline(tau=0.3): 120={np.mean(v120):.3f} 360={np.mean(v360):.3f}")

    taus = np.round(np.arange(0.30, 0.96, 0.05), 2)
    obs120 = [np.mean(v120 & (scores >= t)) for t in taus]
    obs360 = [np.mean(v360 & (scores >= t)) for t in taus]
    obsV2I = [np.mean(scores >= t) for t in taus]   # ideal RSU coverage, gated by tau

    print("\n  tau   120     360     V2I")
    for t, a1, a2, a3 in zip(taus, obs120, obs360, obsV2I):
        print(f"  {t:.2f}  {a1:.3f}  {a2:.3f}  {a3:.3f}")

    # cooperative observability over (tau_AV, tau_RSU): onboard-360 confident OR RSU confident
    grid = np.round(np.arange(0.30, 0.96, 0.05), 2)
    H = np.zeros((len(grid), len(grid)))
    for i, ta in enumerate(grid):          # tau_AV (rows)
        av = v360 & (scores >= ta)
        for j, tr in enumerate(grid):      # tau_RSU (cols)
            H[i, j] = np.mean(av | (scores >= tr))

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 3.40))
    axA.plot(taus, obsV2I, "-o", color="#7B3294", lw=1.8, ms=4, label="V2I cooperative")
    axA.plot(taus, obs360, "-o", color="#0072B2", lw=1.8, ms=4, label="360° surround")
    axA.plot(taus, obs120, "-o", color="#E69F00", lw=1.8, ms=4, label="120° forward")
    axA.axvspan(0.30, 0.55, color="#eef3f8", zorder=0)
    axA.text(0.42, 0.04, "typical\noperating", ha="center", va="bottom", fontsize=10.0, color="#666")
    axA.set_xlabel("Detection-confidence threshold $\\tau$")
    axA.set_ylabel("Intersection-wide observability")
    axA.set_ylim(0, 1.02); axA.set_title("(a) Robustness to a uniform threshold", fontsize=10)
    axA.legend(fontsize=10.0, loc="upper right"); axA.grid(True, alpha=0.3)

    im = axB.imshow(H, origin="lower", aspect="auto", cmap="viridis", vmin=0.3, vmax=1.0,
                    extent=[grid[0], grid[-1], grid[0], grid[-1]])
    cs = axB.contour(grid, grid, H, levels=[0.7, 0.8, 0.9, 0.95], colors="white", linewidths=0.6)
    axB.clabel(cs, fmt="%.2f", fontsize=10.0)
    axB.set_xlabel("RSU confidence threshold $\\tau_{\\mathrm{RSU}}$")
    axB.set_ylabel("AV confidence threshold $\\tau_{\\mathrm{AV}}$")
    axB.set_title("(b) Cooperative observability", fontsize=10)
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(f"{a.out}.{fmt}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
