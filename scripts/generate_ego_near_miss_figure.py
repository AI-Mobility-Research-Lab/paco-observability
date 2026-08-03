#!/usr/bin/env python3
"""Figure: ego-involved surrogate-conflict partner observability.

Reads outputs/ego_near_miss_observability.json (ego_near_miss_sweep.py) and
draws a two-panel figure into figures/ego_near_miss_observability.{pdf,png}:

  (a) Partner-observability rate by sensor configuration (forward 120 deg,
      360 deg surround, cooperative V2I).
  (b) Why the partner is missed: visible / outside FOV / occluded / beyond range,
      for the 120 deg and 360 deg sensors -- showing field of view, not occlusion,
      is the binding constraint for the ego's own conflicts, and that a surround
      view nearly closes it.

Style matches generate_supplementary_figures.py (TR Part C / IEEE).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10.0,
    "axes.labelsize": 10.0,
    "axes.titlesize": 10.0,
    "legend.fontsize": 10.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "text.usetex": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

COL_VISIBLE = "#2ca02c"
COL_OUTFOV = "#984ea3"
COL_OCCLUDED = "#d62728"
COL_OUTRANGE = "#ff7f00"
COL_120 = "#2166ac"
COL_360 = "#4393c3"
COL_V2I = "#b2182b"

def _save(fig, output_stem: Path):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{fmt}"), format=fmt,
                    dpi=600 if fmt == "png" else None,
                    facecolor="white", edgecolor="none")
    print(f"  Saved: {output_stem}.pdf / .png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/derived/ego_near_miss_observability.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/ego_near_miss_observability"),
        help="Output path without extension; both PDF and PNG are written.",
    )
    args = parser.parse_args()

    d = json.loads(args.data.read_text())
    ttc = d["ttc"]
    s120, s360 = ttc["fov_120"], ttc["fov_360"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 3.60))

    # ---- Panel (a): observability rate by sensor config ----
    labels = ["Forward\n120°", "360°\nsurround", "Cooperative\n(V2I)"]
    vals = [s120["partner_visible_rate"], s360["partner_visible_rate"], 1.0]
    colors = [COL_120, COL_360, COL_V2I]
    bars = axA.bar(labels, vals, color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.88, width=0.62)
    for bar, v in zip(bars, vals):
        axA.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:.0%}", ha="center", va="bottom", fontsize=10.0, fontweight="bold")
    axA.set_ylabel("Conflict partner observable to ego")
    axA.set_ylim(0, 1.15)
    axA.set_yticks(np.arange(0, 1.01, 0.2))
    axA.grid(True, alpha=0.25, axis="y")
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.set_title("(a) Ego sees its own conflict partner", fontsize=10.0, fontweight="bold")

    # ---- Panel (b): cause decomposition, 120 vs 360 ----
    cats = ["visible", "out_of_fov", "occluded", "out_of_range"]
    cat_labels = ["Visible", "Outside FOV", "Occluded", "Beyond range"]
    cat_colors = [COL_VISIBLE, COL_OUTFOV, COL_OCCLUDED, COL_OUTRANGE]
    r120 = s120["reason_breakdown"]
    r360 = s360["reason_breakdown"]
    x = np.arange(2)
    width = 0.5
    bottoms = np.zeros(2)
    for cat, lab, col in zip(cats, cat_labels, cat_colors):
        heights = np.array([r120.get(cat, 0.0), r360.get(cat, 0.0)])
        axB.bar(x, heights, width, bottom=bottoms, label=lab, color=col,
                alpha=0.82, edgecolor="white", linewidth=0.4)
        for xi, (h, b0) in enumerate(zip(heights, bottoms)):
            if h >= 0.06:
                axB.text(xi, b0 + h / 2, f"{h:.0%}", ha="center", va="center",
                         fontsize=10.0, color="white", fontweight="bold")
        bottoms += heights
    axB.set_xticks(x)
    axB.set_xticklabels(["Forward 120°", "360° surround"])
    axB.set_ylabel("Fraction of ego-involved surrogate-conflict frames")
    axB.set_ylim(0, 1.05)
    axB.set_yticks(np.arange(0, 1.01, 0.2))
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4,
               fontsize=10.0, framealpha=0.9, edgecolor="#CCC", handlelength=1.2,
               columnspacing=1.0)
    axB.grid(True, alpha=0.25, axis="y")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.set_title("(b) Why the partner is missed", fontsize=10.0, fontweight="bold")

    plt.tight_layout(pad=0.4, w_pad=1.5)
    _save(fig, args.out)

    print("\n  Panel (a): 120={:.1%}  360={:.1%}  V2I=100%".format(
        s120["partner_visible_rate"], s360["partner_visible_rate"]))
    print("  Panel (b) 120 reasons:", r120)
    print("  Panel (b) 360 reasons:", r360)


if __name__ == "__main__":
    main()
