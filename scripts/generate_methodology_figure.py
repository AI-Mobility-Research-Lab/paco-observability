#!/usr/bin/env python3
"""Generate publication-quality methodology figure for the Part C paper.

Two-panel figure:
  (a) Intersection Observability Geometry — BEV of a 4-way intersection
      showing ego vehicle, RSU, onboard FOV sector, and the three spatial
      scopes used for observability analysis (intersection-wide, nearby,
      conflict zone).
  (b) Angular-Wedge Occlusion Model — geometric illustration of the
      occlusion mechanism with angular bins.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Wedge,
    Circle,
    Arc,
)
from matplotlib.collections import PatchCollection

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
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
        "savefig.pad_inches": 0.02,
    }
)

# Colours
COL_VISIBLE = "#2ca02c"       # green
COL_OCCLUDED = "#d62728"      # red
COL_OUTSIDE = "#888888"       # gray
COL_FOV = "#2196F3"           # blue — FOV sector fill
COL_FOV_EDGE = "#1565C0"      # darker blue for FOV edges
COL_NEARBY = "#1976D2"        # blue dashed circle
COL_CONFLICT = "#E65100"      # orange dashed circle
COL_ROAD = "#E8E8E8"          # light gray road
COL_ROAD_MARK = "#CCCCCC"     # lane markings
COL_RSU = "#7B1FA2"           # purple for RSU
COL_EGO = "#0D47A1"           # dark blue ego vehicle
COL_WEDGE = "#FFCDD2"         # light red for occluded wedge
COL_WEDGE_EDGE = "#D62728"    # red edge
COL_SENSOR_ORIGIN = "#FF6F00" # amber for sensor origin
COL_WIDE = "#90A4AE"          # blue-gray fill for intersection-wide scope
COL_WIDE_EDGE = "#455A64"     # slate edge for intersection-wide scope


def _draw_vehicle(ax, cx, cy, w, h, angle_deg, color, label=None, zorder=5):
    """Draw a vehicle as a rounded rectangle rotated to *angle_deg*."""
    rect = FancyBboxPatch(
        (-w / 2, -h / 2),
        w,
        h,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor="k",
        linewidth=0.6,
        zorder=zorder,
    )
    t = (
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg(angle_deg)
        .translate(cx, cy)
        + ax.transData
    )
    rect.set_transform(t)
    ax.add_patch(rect)
    if label:
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=10.0,
            color="white",
            fontweight="bold",
            zorder=zorder + 1,
        )


def _draw_truck(ax, cx, cy, w, h, angle_deg, color, zorder=5):
    """Draw a truck as a larger rounded rectangle."""
    rect = FancyBboxPatch(
        (-w / 2, -h / 2),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="k",
        linewidth=0.6,
        zorder=zorder,
    )
    t = (
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg(angle_deg)
        .translate(cx, cy)
        + ax.transData
    )
    rect.set_transform(t)
    ax.add_patch(rect)


def _draw_pedestrian(ax, cx, cy, color, zorder=5, size=0.7):
    """Draw pedestrian as a small filled circle."""
    c = Circle((cx, cy), size, facecolor=color, edgecolor="k", linewidth=0.5, zorder=zorder)
    ax.add_patch(c)


def _draw_bicycle(ax, cx, cy, color, angle_deg=0, zorder=5):
    """Draw a bicycle as a small diamond."""
    s = 0.9
    pts = np.array([
        [0, s],
        [s * 0.6, 0],
        [0, -s],
        [-s * 0.6, 0],
    ])
    rad = math.radians(angle_deg)
    rot = np.array([[math.cos(rad), -math.sin(rad)],
                    [math.sin(rad),  math.cos(rad)]])
    pts = pts @ rot.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    poly = Polygon(pts, closed=True, facecolor=color, edgecolor="k", linewidth=0.5, zorder=zorder)
    ax.add_patch(poly)


def _draw_rsu(ax, cx, cy, zorder=10):
    """Draw the roadside LiDAR unit as a star-like marker with radiating lines."""
    ax.plot(cx, cy, marker="*", markersize=10, color=COL_RSU,
            markeredgecolor="k", markeredgewidth=0.4, zorder=zorder)
    # Small radiating dashes
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        dx, dy = 1.2 * math.cos(ang), 1.2 * math.sin(ang)
        ax.plot(
            [cx + 0.6 * math.cos(ang), cx + dx],
            [cy + 0.6 * math.sin(ang), cy + dy],
            color=COL_RSU,
            linewidth=0.5,
            alpha=0.6,
            zorder=zorder - 1,
        )


# ===================================================================
# Panel (a) — Intersection Observability Geometry
# ===================================================================
def draw_panel_a(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-48, 48)
    ax.set_ylim(-48, 48)
    ax.axis("off")

    # --- Roads ---
    road_w = 12  # total road width
    hw = road_w / 2

    # Horizontal road
    ax.fill_between([-48, 48], -hw, hw, color=COL_ROAD, zorder=0)
    # Vertical road
    ax.fill_betweenx([-48, 48], -hw, hw, color=COL_ROAD, zorder=0)

    # Center-line dashes
    for start in np.arange(-48, 48, 4):
        ax.plot([start, start + 2], [0, 0], color=COL_ROAD_MARK, linewidth=0.5, zorder=1)
        ax.plot([0, 0], [start, start + 2], color=COL_ROAD_MARK, linewidth=0.5, zorder=1)

    # Lane edge lines
    for offset in [-hw, hw]:
        ax.plot([-48, -hw], [offset, offset], color="#AAAAAA", linewidth=0.4, zorder=1)
        ax.plot([hw, 48], [offset, offset], color="#AAAAAA", linewidth=0.4, zorder=1)
        ax.plot([offset, offset], [-48, -hw], color="#AAAAAA", linewidth=0.4, zorder=1)
        ax.plot([offset, offset], [hw, 48], color="#AAAAAA", linewidth=0.4, zorder=1)

    # Crosswalk hatching at intersection edges
    for pos in [-hw, hw]:
        for off in np.arange(-hw + 1, hw, 1.5):
            ax.plot([pos - 0.8, pos + 0.8], [off, off], color="#BBBBBB", linewidth=0.4, zorder=1)
            ax.plot([off, off], [pos - 0.8, pos + 0.8], color="#BBBBBB", linewidth=0.4, zorder=1)

    # --- Intersection center marker ---
    ax.plot(0, 0, "+", color="#999999", markersize=6, markeredgewidth=0.5, zorder=2)

    # --- Ego vehicle (approaching from west on right lane) ---
    ego_x, ego_y = -22, -3
    ego_heading = 0  # facing east
    _draw_vehicle(ax, ego_x, ego_y, 4.5, 2.0, ego_heading, COL_EGO, zorder=8)
    # Heading arrow
    ax.annotate(
        "",
        xy=(ego_x + 4.5, ego_y),
        xytext=(ego_x + 2.5, ego_y),
        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.2",
                        color=COL_EGO, lw=1.0),
        zorder=9,
    )
    ax.text(ego_x, ego_y - 2.8, "Ego AV", ha="center", va="top", fontsize=10.0,
            fontweight="bold", color=COL_EGO, zorder=10)

    # --- Onboard sensor FOV sector ---
    fov_deg = 120
    sensor_range = 35
    sensor_x = ego_x + 1.5  # offset from center
    sensor_y = ego_y
    heading_deg = 0  # east
    theta1 = heading_deg - fov_deg / 2
    theta2 = heading_deg + fov_deg / 2

    wedge = Wedge(
        (sensor_x, sensor_y),
        sensor_range,
        theta1,
        theta2,
        facecolor=COL_FOV,
        edgecolor=COL_FOV_EDGE,
        alpha=0.08,
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(wedge)

    # FOV edge lines
    for ang in [theta1, theta2]:
        rad = math.radians(ang)
        ax.plot(
            [sensor_x, sensor_x + sensor_range * math.cos(rad)],
            [sensor_y, sensor_y + sensor_range * math.sin(rad)],
            color=COL_FOV_EDGE,
            linewidth=0.8,
            linestyle="-",
            alpha=0.5,
            zorder=3,
        )

    # FOV label — place along upper FOV edge, away from RSU
    fov_label_ang = math.radians(theta2 - 8)
    fov_label_r = sensor_range * 0.62
    ax.text(
        sensor_x + fov_label_r * math.cos(fov_label_ang),
        sensor_y + fov_label_r * math.sin(fov_label_ang) + 2,
        "Onboard Sensor FOV\n(120°, 35 m range)",
        fontsize=10.0,
        color=COL_FOV_EDGE,
        ha="center",
        va="bottom",
        fontstyle="italic",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=COL_FOV_EDGE,
                  linewidth=0.3, alpha=0.85),
    )

    # Range arc
    range_arc = Arc(
        (sensor_x, sensor_y),
        2 * sensor_range,
        2 * sensor_range,
        angle=0,
        theta1=theta1,
        theta2=theta2,
        color=COL_FOV_EDGE,
        linewidth=0.6,
        linestyle="--",
        alpha=0.4,
        zorder=3,
    )
    ax.add_patch(range_arc)

    # --- Three spatial scopes for observability analysis ---
    # Scope 1: Intersection-wide (every road user present in the scene)
    wide_r = 38
    ax.add_patch(Circle((0, 0), wide_r, facecolor=COL_WIDE, edgecolor="none",
                        alpha=0.05, zorder=1))
    ax.add_patch(Circle((0, 0), wide_r, facecolor="none", edgecolor=COL_WIDE_EDGE,
                        linewidth=1.0, linestyle=(0, (6, 3)), zorder=4))
    ax.text(
        0, wide_r - 3.5,
        "Intersection-wide scope (all road users)",
        fontsize=10.0, color=COL_WIDE_EDGE, ha="center", va="center",
        fontweight="bold", zorder=10,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor=COL_WIDE_EDGE, linewidth=0.3, alpha=0.85),
    )

    # Scope 2: Conflict zone (20 m around intersection center)
    conflict_r = 20
    ax.add_patch(Circle((0, 0), conflict_r, facecolor=COL_CONFLICT, edgecolor="none",
                        alpha=0.07, zorder=1))
    ax.add_patch(Circle((0, 0), conflict_r, facecolor="none", edgecolor=COL_CONFLICT,
                        linewidth=1.1, linestyle=(0, (5, 3)), zorder=4))
    ax.text(
        conflict_r + 1.5, -conflict_r + 2.5,
        "Conflict zone\n(20 m from center)",
        fontsize=10.0, color=COL_CONFLICT, ha="left", va="center",
        fontweight="bold", zorder=10,
    )

    # Scope 3: Nearby zone (15 m around ego)
    nearby_r = 15
    ax.add_patch(Circle((ego_x, ego_y), nearby_r, facecolor=COL_NEARBY, edgecolor="none",
                        alpha=0.07, zorder=1))
    ax.add_patch(Circle((ego_x, ego_y), nearby_r, facecolor="none", edgecolor=COL_NEARBY,
                        linewidth=1.1, linestyle=(0, (5, 3)), zorder=4))
    ax.text(
        ego_x, ego_y - nearby_r - 1.5,
        "Nearby zone\n(15 m from ego)",
        fontsize=10.0, color=COL_NEARBY, ha="center", va="top",
        fontweight="bold", zorder=10,
    )

    # --- RSU ---
    rsu_x, rsu_y = 8, 14
    _draw_rsu(ax, rsu_x, rsu_y, zorder=10)
    ax.text(
        rsu_x + 2.5,
        rsu_y + 1.0,
        "Roadside LiDAR (RSU)",
        ha="left",
        va="center",
        fontsize=10.0,
        color=COL_RSU,
        fontweight="bold",
        zorder=10,
    )

    ax.set_title("(a) Intersection Observability Geometry", fontsize=10.0, fontweight="bold",
                 pad=6)


# ===================================================================
# Panel (b) — Angular-Wedge Occlusion Model
# ===================================================================
def draw_panel_b(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-4, 30)
    ax.set_ylim(-14, 20)
    ax.axis("off")

    # --- Sensor origin ---
    origin_x, origin_y = 0, 0
    ax.plot(origin_x, origin_y, "o", color=COL_SENSOR_ORIGIN, markersize=7,
            markeredgecolor="k", markeredgewidth=0.5, zorder=10)
    ax.text(origin_x - 0.5, origin_y - 2.0, "Sensor\norigin", ha="center", va="top",
            fontsize=10.0, fontweight="bold", color=COL_SENSOR_ORIGIN, zorder=10)

    # --- FOV hint (light sector) ---
    fov_range = 28
    fov_half = 42  # ±42 degrees from heading (east)
    wedge_bg = Wedge(
        (origin_x, origin_y),
        fov_range,
        -fov_half,
        fov_half,
        facecolor=COL_FOV,
        edgecolor=COL_FOV_EDGE,
        alpha=0.05,
        linewidth=0.6,
        zorder=1,
    )
    ax.add_patch(wedge_bg)

    # --- Angular bin grid lines (subtle) ---
    n_bins_shown = 18
    for i in range(n_bins_shown + 1):
        ang = -fov_half + i * (2 * fov_half / n_bins_shown)
        rad = math.radians(ang)
        ax.plot(
            [origin_x, origin_x + fov_range * math.cos(rad)],
            [origin_y, origin_y + fov_range * math.sin(rad)],
            color="#DDDDDD",
            linewidth=0.25,
            alpha=0.6,
            zorder=1,
        )

    # Highlight bin boundaries near the occluder
    for ang in [-8, -2, 4, 10]:
        rad = math.radians(ang)
        ax.plot(
            [origin_x, origin_x + fov_range * math.cos(rad)],
            [origin_y, origin_y + fov_range * math.sin(rad)],
            color="#BBBBBB",
            linewidth=0.35,
            alpha=0.7,
            zorder=1,
        )

    # Range arcs
    for r in [9, 18, 27]:
        arc = Arc(
            (origin_x, origin_y),
            2 * r,
            2 * r,
            angle=0,
            theta1=-fov_half,
            theta2=fov_half,
            color="#E0E0E0",
            linewidth=0.25,
            zorder=1,
        )
        ax.add_patch(arc)

    # --- Occluder truck ---
    truck_cx, truck_cy = 11, 1
    truck_w, truck_h = 5.5, 2.8
    _draw_truck(ax, truck_cx, truck_cy, truck_w, truck_h, 0, "#555555", zorder=7)
    ax.text(truck_cx, truck_cy, "Truck", ha="center", va="center",
            fontsize=10.0, color="white", fontweight="bold", zorder=8)

    # Label "Occluder"
    ax.annotate(
        "Occluder",
        xy=(truck_cx, truck_cy + 2.0),
        xytext=(truck_cx - 2, truck_cy + 5.5),
        fontsize=10.0,
        fontweight="bold",
        color="#333333",
        ha="center",
        arrowprops=dict(arrowstyle="->,head_width=0.15", color="#555555", lw=0.7),
        zorder=10,
    )

    # --- Occluded wedge behind truck ---
    truck_corners = [
        (truck_cx - truck_w / 2, truck_cy - truck_h / 2),
        (truck_cx + truck_w / 2, truck_cy - truck_h / 2),
        (truck_cx + truck_w / 2, truck_cy + truck_h / 2),
        (truck_cx - truck_w / 2, truck_cy + truck_h / 2),
    ]
    angles = [math.degrees(math.atan2(cy - origin_y, cx - origin_x)) for cx, cy in truck_corners]
    ang_min, ang_max = min(angles), max(angles)

    # Occluded region wedge
    occ_wedge = Wedge(
        (origin_x, origin_y),
        fov_range,
        ang_min,
        ang_max,
        facecolor=COL_WEDGE,
        edgecolor=COL_WEDGE_EDGE,
        alpha=0.18,
        linewidth=0.7,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(occ_wedge)

    # "Blocked region" label
    mid_ang = math.radians((ang_min + ang_max) / 2)
    shadow_label_r = 22
    ax.text(
        origin_x + shadow_label_r * math.cos(mid_ang),
        origin_y + shadow_label_r * math.sin(mid_ang),
        "Blocked\nregion",
        ha="center",
        va="center",
        fontsize=10.0,
        color=COL_WEDGE_EDGE,
        fontstyle="italic",
        zorder=10,
    )

    # --- Visible target (car above, clearly outside the wedge) ---
    vis_x, vis_y = 18, 9
    _draw_vehicle(ax, vis_x, vis_y, 3.5, 1.6, 0, COL_VISIBLE, zorder=6)
    ax.text(
        vis_x, vis_y + 2.2,
        "Visible target",
        fontsize=10.0,
        fontweight="bold",
        color=COL_VISIBLE,
        ha="center",
        va="bottom",
        zorder=10,
    )
    # LOS line to visible target (solid green)
    ax.plot(
        [origin_x, vis_x],
        [origin_y, vis_y],
        color=COL_VISIBLE,
        linewidth=0.9,
        linestyle="-",
        alpha=0.45,
        zorder=3,
    )
    # Checkmark near visible target (use sans-serif for symbol availability)
    ax.text(vis_x + 2.5, vis_y, "\u2713", fontsize=10.0, color=COL_VISIBLE,
            fontweight="bold", ha="center", va="center", zorder=10,
            fontfamily="sans-serif")

    # --- Occluded target (pedestrian behind truck, inside wedge) ---
    occ_target_x, occ_target_y = 21, 0.5
    _draw_pedestrian(ax, occ_target_x, occ_target_y, COL_OCCLUDED, zorder=6, size=0.9)
    ax.text(
        occ_target_x, occ_target_y - 2.5,
        "Occluded target",
        fontsize=10.0,
        fontweight="bold",
        color=COL_OCCLUDED,
        ha="center",
        va="top",
        zorder=10,
    )
    # Blocked LOS line: solid to truck face, then dashed behind
    # Compute intersection with truck front face
    truck_front_x = truck_cx - truck_w / 2
    t_param = truck_front_x / occ_target_x  # parametric position along line
    block_y = occ_target_y * t_param
    ax.plot(
        [origin_x, truck_front_x],
        [origin_y, block_y],
        color=COL_OCCLUDED,
        linewidth=0.9,
        linestyle="-",
        alpha=0.45,
        zorder=3,
    )
    ax.plot(
        [truck_front_x, occ_target_x],
        [block_y, occ_target_y],
        color=COL_OCCLUDED,
        linewidth=0.9,
        linestyle="--",
        alpha=0.35,
        zorder=3,
    )
    # X mark where LOS is blocked
    ax.plot(truck_front_x, block_y, "x", color=COL_OCCLUDED,
            markersize=7, markeredgewidth=1.8, zorder=8)

    # --- Angular wedge annotation ---
    arc_r = 5.5
    ang_arc = Arc(
        (origin_x, origin_y),
        2 * arc_r,
        2 * arc_r,
        angle=0,
        theta1=ang_min,
        theta2=ang_max,
        color=COL_WEDGE_EDGE,
        linewidth=1.2,
        zorder=5,
    )
    ax.add_patch(ang_arc)

    # Wedge label. Anchored below the wedge rather than on the mid-angle: at the
    # panel's aspect ratio the mid-angle position falls on the occluder truck.
    ax.annotate(
        "Angular\nwedge",
        xy=(origin_x + arc_r * math.cos(mid_ang), origin_y + arc_r * math.sin(mid_ang)),
        xytext=(8.5, -6.5),
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=COL_WEDGE_EDGE,
        arrowprops=dict(arrowstyle="->,head_width=0.15", color=COL_WEDGE_EDGE, lw=0.7),
        zorder=10,
    )

    # --- "Angular bins" label (top right) ---
    ax.text(
        24, 18,
        "Angular bins\n(360 bins over FOV)",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#777777",
        fontstyle="italic",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#BBBBBB",
                  linewidth=0.4, alpha=0.9),
    )

    # --- Nearest-object blocking rule (bottom right) ---
    ax.text(
        22, -11,
        "Rule: In each angular bin,\nthe nearest object blocks\nall objects behind it.",
        ha="center",
        va="center",
        fontsize=10.0,
        color="#555555",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825",
                  linewidth=0.5, alpha=0.95),
    )

    ax.set_title("(b) Angular-Wedge Occlusion Model", fontsize=10.0, fontweight="bold",
                 pad=6)


# ===================================================================
# Main — assemble figure
# ===================================================================
def main():
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(6.5, 4.30),  # IEEE double-column width ~7.16 in
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    fig.patch.set_facecolor("white")

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    plt.tight_layout(pad=0.5, w_pad=1.5)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save in multiple formats
    for fmt in ["pdf", "png", "svg"]:
        out_path = out_dir / f"methodology_overview.{fmt}"
        dpi = 600 if fmt == "png" else None
        fig.savefig(out_path, format=fmt, dpi=dpi, facecolor="white", edgecolor="none")
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
