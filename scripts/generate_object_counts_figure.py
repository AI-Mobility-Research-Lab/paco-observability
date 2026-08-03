#!/usr/bin/env python3
"""Object-invisibility decomposition: 120 deg forward vs 360 deg surround.

Stacked per-frame object counts (visible / occluded / out-of-FOV / beyond range)
for the two onboard configurations, showing the limiting-factor reversal: the
out-of-FOV component (dominant at 120 deg) vanishes under a 360 deg surround,
while occlusion grows to become the binding residual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.0,
    "axes.titlesize": 10,
    "text.usetex": False,
    "savefig.dpi": 600,
})

C_VIS = "#4daf4a"
C_OCC = "#e41a1c"
C_FOV = "#984ea3"
C_RNG = "#ff7f00"


def counts(path):
    df = pd.read_parquet(path)
    return [df["n_visible_onboard"].mean(), df["n_occluded"].mean(),
            df["n_out_of_fov"].mean(), df["n_out_of_range"].mean()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p120", default="data/derived/ego_sweep_results.parquet")
    ap.add_argument("--p360", default="data/derived/sweep_360/ego_sweep_results.parquet")
    ap.add_argument("--out", default="figures/object_counts")
    a = ap.parse_args()

    data = [counts(a.p120), counts(a.p360)]
    labels = ["120° forward", "360° surround"]
    notes = ["FOV-limited", "occlusion-limited"]
    segs = [("Visible", C_VIS), ("Occluded", C_OCC), ("Outside FOV", C_FOV), ("Beyond range", C_RNG)]

    x = np.arange(2)
    w = 0.55
    fig, ax = plt.subplots(figsize=(5.58, 4.35))
    bottoms = [0.0, 0.0]
    for si, (name, col) in enumerate(segs):
        vals = [data[0][si], data[1][si]]
        ax.bar(x, vals, w, bottom=bottoms, label=name, color=col, alpha=0.88,
               edgecolor="white", linewidth=0.7)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.4:
                ax.text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center",
                        fontsize=10.0, color="white", fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for xi, note in enumerate(notes):
        ax.text(xi, bottoms[xi] + 0.25, note, ha="center", va="bottom",
                fontsize=10.0, style="italic", color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean objects per frame  (of \\textasciitilde11.8 total)".replace("\\textasciitilde", "~"))
    ax.set_ylim(0, 13.6)
    ax.set_title("Why objects are unobserved: 120° versus 360°")
    ax.legend(loc="lower center", ncol=2, fontsize=10.0, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.30))
    ax.grid(True, alpha=0.3, axis="y")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(f"{a.out}.{fmt}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", a.out)
    for lab, d in zip(labels, data):
        print(f"  {lab}: vis={d[0]:.2f} occ={d[1]:.2f} fov={d[2]:.2f} rng={d[3]:.2f}")


if __name__ == "__main__":
    main()
