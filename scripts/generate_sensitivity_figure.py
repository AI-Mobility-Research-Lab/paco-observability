#!/usr/bin/env python3
"""Merged sensitivity figure: FOV sweep (incl. 360 deg) + range sweep.

Replaces the earlier FOV-range heatmap (FOV capped at 180 deg, plus a redundant
1-visibility panel) and the separate range figure.

Panel (a): intersection-wide visibility vs FOV at 35 m, 60..360 deg.
Panel (b): intersection-wide and nearby visibility vs max detection range at
           120 deg and 360 deg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.0,
    "axes.titlesize": 10.0,
    "text.usetex": False,
    "savefig.dpi": 600,
})

C120 = "#E69F00"   # orange
C360 = "#0072B2"   # blue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", default="data/derived/sensitivity/sensitivity_results.parquet")
    ap.add_argument("--rng", default="data/derived/range_sensitivity.json")
    ap.add_argument("--out", default="figures/sensitivity_range")
    a = ap.parse_args()

    # ---- Panel A: FOV slice at 35 m (60..180 from sensitivity sweep, 360 from range sweep) ----
    sdf = pd.read_parquet(a.sens)
    s35 = sdf[sdf["range_m"] == 35].sort_values("fov_deg")
    fov = list(s35["fov_deg"])
    visA = list(s35["visibility_mean"])
    rng = json.load(open(a.rng))
    v360 = [r["int_wide"] for r in rng if r["fov"] == 360 and r["range_m"] == 35][0]
    fov.append(360.0)
    visA.append(v360)

    # ---- Panel B: range slice at 120 and 360 ----
    r120 = sorted([r for r in rng if r["fov"] == 120], key=lambda r: r["range_m"])
    r360 = sorted([r for r in rng if r["fov"] == 360], key=lambda r: r["range_m"])
    R = [r["range_m"] for r in r120]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 3.40))

    # Panel A
    axA.plot(fov, visA, "-o", color="#555", lw=1.8, ms=5, zorder=3)
    for f, v in zip(fov, visA):
        axA.annotate(f"{v:.2f}", (f, v), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=10.0)
    for f, c, lab in [(120, C120, "ADAS"), (360, C360, "AV")]:
        v = visA[fov.index(f)]
        axA.plot(f, v, "o", color=c, ms=10, zorder=5)
        axA.annotate(lab, (f, v), textcoords="offset points", xytext=(0, -14),
                     ha="center", fontsize=10.0, color=c, fontweight="bold")
    axA.set_xlabel("Sensor field of view (deg)")
    axA.set_ylabel("Intersection-wide visibility")
    axA.set_title("(a) FOV sweep (range = 35 m)")
    axA.set_xticks([60, 90, 120, 150, 180, 360])
    axA.set_ylim(0, 0.8)
    axA.grid(True, alpha=0.3)

    # Panel B
    axB.plot(R, [r["int_wide"] for r in r120], "-o", color=C120, lw=1.8, ms=5,
             label="120° — intersection-wide")
    axB.plot(R, [r["nearby"] for r in r120], "--s", color=C120, lw=1.4, ms=4, mfc="white",
             label="120° — nearby (15 m)")
    axB.plot(R, [r["int_wide"] for r in r360], "-o", color=C360, lw=1.8, ms=5,
             label="360° — intersection-wide")
    axB.plot(R, [r["nearby"] for r in r360], "--s", color=C360, lw=1.4, ms=4, mfc="white",
             label="360° — nearby (15 m)")
    axB.axvline(35, color="#999", ls=":", lw=1)
    axB.text(37, 0.04, "default 35 m", fontsize=10.0, va="bottom", ha="left", color="#666")
    axB.set_xlabel("Maximum detection range (m)")
    axB.set_ylabel("Visibility ratio")
    axB.set_title("(b) Range sweep (120° and 360°)")
    axB.set_ylim(0, 1.0)
    axB.grid(True, alpha=0.3)
    axB.legend(loc="lower right", fontsize=10.0, framealpha=0.95)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(f"{a.out}.{fmt}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", a.out)
    print("  FOV slice @35m:", [(int(f), round(v, 3)) for f, v in zip(fov, visA)])


if __name__ == "__main__":
    main()
