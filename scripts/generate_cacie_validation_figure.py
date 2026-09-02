#!/usr/bin/env python3
"""Generate a source-backed CACIE model-validation figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "legacy_planar": "#D55E00",
    "center_top": "#E69F00",
    "sparse_multiray": "#0072B2",
    "exact_reference": "#009E73",
}
LABELS = {
    "legacy_planar": "Planar wedge",
    "center_top": "Center + top (2 rays)",
    "sparse_multiray": "Sparse 3D (15 rays)",
    "exact_reference": "Ray–OBB reference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output path without suffix")
    parser.add_argument("--z-mode", default="ground_anchored")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text())
    groups = sorted(
        (group for group in payload["groups"] if group["z_mode"] == args.z_mode),
        key=lambda group: float(group["fov_deg"]),
    )
    if not groups:
        raise ValueError(f"summary contains no groups for z_mode={args.z_mode!r}")
    fovs = [float(group["fov_deg"]) for group in groups]
    validation_methods = ["legacy_planar", "center_top", "sparse_multiray"]
    all_methods = [*validation_methods, "exact_reference"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    ax_accuracy, ax_observability, ax_runtime, ax_convergence = axes.ravel()

    x = np.arange(len(fovs), dtype=float)
    width = 0.23
    for index, method in enumerate(validation_methods):
        values = [
            group["agreement_with_exact_among_covered"][method]["accuracy"] for group in groups
        ]
        positions = x + (index - 1) * width
        bars = ax_accuracy.bar(
            positions,
            values,
            width,
            label=LABELS[method],
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.35,
        )
        ax_accuracy.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax_accuracy.set_xticks(x, [f"{value:g}°" for value in fovs])
    ax_accuracy.set_ylim(0.70, 1.025)
    ax_accuracy.set_ylabel("Binary agreement")
    ax_accuracy.set_title("(a) Agreement with within-OBB reference")
    ax_accuracy.legend(fontsize=7, loc="lower right")
    ax_accuracy.grid(axis="y", alpha=0.25)

    width = 0.18
    for index, method in enumerate(all_methods):
        values = [group["observability"][method] for group in groups]
        positions = x + (index - 1.5) * width
        ax_observability.bar(
            positions,
            values,
            width,
            label=LABELS[method],
            color=COLORS[method],
            edgecolor="black",
            linewidth=0.35,
        )
    ax_observability.set_xticks(x, [f"{value:g}°" for value in fovs])
    ax_observability.set_ylim(0.0, 1.0)
    ax_observability.set_ylabel("Observable / all decisions")
    ax_observability.set_title("(b) Consequence for observability")
    ax_observability.grid(axis="y", alpha=0.25)

    runtime_methods = ["center_top", "sparse_multiray", "exact_angular_grid"]
    runtime_labels = ["2 rays", "15 rays", "Projected grid"]
    runtime_values: list[float] = []
    runtime_grids: list[str] = []
    runtime_rows = [
        row
        for row in payload.get("runtime", [])
        if row["z_mode"] == args.z_mode and row["method"] in runtime_methods
    ]
    for method in runtime_methods:
        matches = [row for row in runtime_rows if row["method"] == method]
        if not matches:
            runtime_values.append(np.nan)
            runtime_grids.append("")
            continue
        # For the reference, plot the largest supplied grid; low-cost methods
        # each have only one row.
        row = max(matches, key=lambda item: item["grid"] or -1)
        runtime_values.append(float(row["median_ms"]))
        runtime_grids.append("" if method != "exact_angular_grid" else f"{row['grid']}×{row['grid']}")
    bars = ax_runtime.bar(
        np.arange(len(runtime_methods)),
        runtime_values,
        color=[COLORS["center_top"], COLORS["sparse_multiray"], COLORS["exact_reference"]],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_runtime.set_yscale("log")
    ax_runtime.set_xticks(
        np.arange(len(runtime_methods)),
        [f"{label}\n{grid}".strip() for label, grid in zip(runtime_labels, runtime_grids)],
    )
    ax_runtime.set_ylabel("Median compute time (ms/target)")
    ax_runtime.set_title("(c) Accuracy–cost trade-off")
    ax_runtime.grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, runtime_values):
        if np.isfinite(value):
            ax_runtime.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.2,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    convergence: dict[int, list[float]] = {}
    for group in groups:
        for grid, values in group.get("grid_convergence", {}).items():
            convergence.setdefault(int(grid), []).append(
                values["mean_absolute_fraction_difference_from_primary"]
            )
    if convergence:
        grids = sorted(convergence)
        mean_values = [float(np.mean(convergence[grid])) for grid in grids]
        ax_convergence.plot(grids, mean_values, "-o", color="#6A3D9A", lw=1.8)
        ax_convergence.set_xticks(grids)
        ax_convergence.set_xlabel("Angular samples per axis")
        ax_convergence.set_ylabel("Mean |fraction − primary|")
    else:
        ax_convergence.text(0.5, 0.5, "No convergence grids in input", ha="center", va="center")
        ax_convergence.set_xticks([])
        ax_convergence.set_yticks([])
    ax_convergence.set_title("(d) Projected-grid convergence")
    ax_convergence.grid(alpha=0.25)

    fig.suptitle(
        "PACO geometry validation (ground-anchored boxes)"
        if args.z_mode == "ground_anchored"
        else f"PACO geometry validation ({args.z_mode})",
        fontsize=11,
    )
    fig.tight_layout(pad=0.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(args.out.with_suffix(suffix), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out.with_suffix('.pdf')} and {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
