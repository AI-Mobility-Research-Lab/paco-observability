#!/usr/bin/env python3
"""Range sensitivity figure for the Part C paper.

Shows onboard observability vs. maximum detection range at FOV = 120 deg and
360 deg, for two spatial scopes (intersection-wide and nearby/15 m). The point:
beyond ~50 m, increasing range yields negligible additional observability --
the gap is set by field of view and occlusion, not detection distance. The
default 35 m (an adverse-weather operating point) is therefore a robust choice.

Reads: data/derived/range_sensitivity.json
Writes: figures/range_sensitivity.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "text.usetex": False,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

COL_120 = "#E65100"   # orange  -- forward 120 deg
COL_360 = "#1565C0"   # blue    -- 360 deg surround
DEFAULT_RANGE = 35.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/derived/range_sensitivity.json")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/range_sensitivity"),
        help="Output path without extension; both PDF and PNG are written.",
    )
    args = parser.parse_args()

    rows = json.loads(args.data.read_text())

    def series(fov, key):
        pts = sorted([r for r in rows if r["fov"] == fov], key=lambda r: r["range_m"])
        return [r["range_m"] for r in pts], [r[key] for r in pts]

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    fig.patch.set_facecolor("white")

    # Intersection-wide (solid, circle) and nearby/15 m (dashed, square)
    for fov, col in [(120, COL_120), (360, COL_360)]:
        x, y = series(fov, "int_wide")
        ax.plot(x, y, "-o", color=col, linewidth=1.4, markersize=4.5,
                label=f"{fov}° FOV  — intersection-wide")
        x, y = series(fov, "nearby")
        ax.plot(x, y, "--s", color=col, linewidth=1.2, markersize=3.8,
                markerfacecolor="white", alpha=0.85,
                label=f"{fov}° FOV  — nearby (15 m)")

    # Default operating point
    ax.axvline(DEFAULT_RANGE, color="#555", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(DEFAULT_RANGE + 2, 0.03, "default 35 m\n(adverse weather)",
            fontsize=6, color="#555", ha="left", va="bottom", fontstyle="italic")

    ax.set_xlabel("Maximum detection range (m)")
    ax.set_ylabel("Onboard observability (visibility ratio)")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(30, 205)
    ax.set_xticks([35, 50, 75, 100, 150, 200])
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center right", fontsize=6.6, framealpha=0.92, edgecolor="#CCC")

    ax.annotate(
        "Plateaus by ~50 m: the gap is set by\nfield of view and occlusion, not range",
        xy=(100, 0.42), xytext=(95, 0.27),
        fontsize=6.3, color="#333", ha="center", va="center", fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFDE7",
                  edgecolor="#F9A825", linewidth=0.4, alpha=0.95),
    )

    plt.tight_layout(pad=0.4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out = args.out.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=600 if fmt == "png" else None,
                    facecolor="white", edgecolor="none")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
