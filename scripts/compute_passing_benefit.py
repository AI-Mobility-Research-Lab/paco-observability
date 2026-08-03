#!/usr/bin/env python3
"""Quantify the cooperative-perception benefit for confidently passing large vehicles.

When the ego follows a large vehicle, that vehicle casts a long occlusion shadow:
the onboard sensor cannot see the road users beyond it that a passing maneuver
must clear (oncoming / cross traffic). V2I cooperation reveals them. We quantify:

  1. Occlusion shadow per occluder: how many road users a single car vs a single
     truck hides from the onboard sensor.
  2. Following-a-truck scenario: when a truck is the near-forward lead vehicle,
     the onboard observability and the number of road users hidden behind it that
     V2I recovers (the "confident passing" benefit).

Uses the exact angular-wedge model (compute_ego_occlusion), forward 120 deg.
"""
from __future__ import annotations

import argparse
import math
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
LEAD_RANGE = 25.0    # m: a lead vehicle within this forward range
LEAD_BEAR = 25.0     # deg: within this of straight ahead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default=TRACKS_DEFAULT)
    ap.add_argument("--n-frames", type=int, default=800)
    ap.add_argument("--out", default="figures/passing_benefit")
    a = ap.parse_args()

    sensor = SensorConfig(fov_deg=120.0)
    egos = [p for lane in DEFAULT_APPROACHES for p in lane.sample_positions()]
    df = pd.read_parquet(a.tracks, columns=["frame_idx", "track_id", "class_name",
                                            "bbox_center_x", "bbox_center_y", "bbox_dx",
                                            "bbox_dy", "bbox_yaw"])
    frames = np.sort(df["frame_idx"].unique())
    sample = frames[np.linspace(0, len(frames) - 1, min(a.n_frames, len(frames))).astype(int)]
    by_frame = {f: g for f, g in df[df["frame_idx"].isin(sample)].groupby("frame_idx")}
    print(f"frames sampled={len(sample)} egos={len(egos)}")

    sum_shadow = {"car": 0, "truck": 0}      # total road users hidden by occluders of this class
    n_occ = {"car": 0, "truck": 0}           # number of distinct occluders of this class that hid >=1
    occ_by = {"car": 0, "truck": 0}          # occlusions attributed to this occluder class
    n_eval = 0
    obs_all = 0.0
    # following-a-truck scenario
    n_lead_truck = 0
    obs_lead_truck = 0.0
    hidden_by_lead_truck = 0.0   # road users occluded by trucks while following a truck (V2I recovers)
    vis_lead_truck = 0.0
    tot_lead_truck = 0.0
    n_lead_car = 0
    obs_lead_car = 0.0

    for fi in sample:
        g = by_frame.get(fi)
        if g is None or g.empty:
            continue
        cls = {int(r.track_id): str(r.class_name) for r in g.itertuples(index=False)}
        for (ex, ey, eh) in egos:
            res = compute_ego_occlusion(ex, ey, eh, g, sensor)
            if not res:
                continue
            n_eval += 1
            total = len(res)
            visible = sum(1 for v in res.values() if v["visible"])
            obs_all += visible / total

            shadow = {}  # occluder track -> count hidden
            occ_truck = 0
            for tid, v in res.items():
                if v["reason"] == "occluded" and v["occluded_by"] is not None:
                    b = v["occluded_by"]; bclass = cls.get(b, "car")
                    if bclass in occ_by:
                        occ_by[bclass] += 1
                    shadow[b] = shadow.get(b, 0) + 1
                    if bclass == "truck":
                        occ_truck += 1
            for b, cnt in shadow.items():
                bclass = cls.get(b, "car")
                if bclass in n_occ:
                    n_occ[bclass] += 1; sum_shadow[bclass] += cnt

            # lead vehicle: visible vehicle near-forward and close
            lead = None
            for tid, v in res.items():
                if v["class_name"] in ("car", "truck") and v["visible"] \
                   and abs(v["bearing_rel_deg"]) < LEAD_BEAR and v["range_m"] < LEAD_RANGE:
                    if lead is None or v["range_m"] < lead[1]:
                        lead = (v["class_name"], v["range_m"])
            if lead and lead[0] == "truck":
                n_lead_truck += 1
                obs_lead_truck += visible / total
                hidden_by_lead_truck += occ_truck
                vis_lead_truck += visible
                tot_lead_truck += total
            elif lead and lead[0] == "car":
                n_lead_car += 1
                obs_lead_car += visible / total

    truck_shadow = sum_shadow["truck"] / max(n_occ["truck"], 1)
    car_shadow = sum_shadow["car"] / max(n_occ["car"], 1)
    print(f"\nObservations: {n_eval}")
    print(f"Occlusion shadow per occluder:  car={car_shadow:.2f}  truck={truck_shadow:.2f}  (truck/car={truck_shadow/car_shadow:.1f}x)")
    tot_occ = occ_by["car"] + occ_by["truck"]
    print(f"Share of occlusions caused by trucks: {occ_by['truck']/max(tot_occ,1):.3f}  (trucks ~1.7% of road users)")
    print(f"Overall onboard observability: {obs_all/n_eval:.3f}")
    print(f"\nFollowing a truck:  P={n_lead_truck/n_eval:.3f}  onboard_obs={obs_lead_truck/max(n_lead_truck,1):.3f}"
          f"  hidden_behind_truck(=V2I recovers)={hidden_by_lead_truck/max(n_lead_truck,1):.2f} per frame"
          f"  onboard_visible={vis_lead_truck/max(n_lead_truck,1):.2f}")
    print(f"Following a car:    P={n_lead_car/n_eval:.3f}  onboard_obs={obs_lead_car/max(n_lead_car,1):.3f}")

    # ---- figure ----
    obs_lt = obs_lead_truck / max(n_lead_truck, 1)
    vis_lt = vis_lead_truck / max(n_lead_truck, 1)
    hid_lt = hidden_by_lead_truck / max(n_lead_truck, 1)
    tot_lt = tot_lead_truck / max(n_lead_truck, 1)
    other_lt = max(tot_lt - vis_lt - hid_lt, 0.0)
    print(f"  (following a truck, per frame: visible {vis_lt:.2f} + truck-shadow {hid_lt:.2f} + other-hidden {other_lt:.2f} = {tot_lt:.2f})")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    axA.bar(["Car", "Truck"], [car_shadow, truck_shadow], color=["#9ecae1", "#D55E00"],
            edgecolor="black", linewidth=0.5, width=0.55)
    for i, val in enumerate([car_shadow, truck_shadow]):
        axA.text(i, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axA.set_ylabel("Road users hidden per occluding vehicle")
    axA.set_title("(a) Occlusion shadow by vehicle type", fontsize=10)
    axA.set_ylim(0, max(car_shadow, truck_shadow) * 1.3)
    axA.grid(True, axis="y", alpha=0.3)

    axB.bar([0], [vis_lt], color="#009E73", edgecolor="black", linewidth=0.5, width=0.5, label="onboard-visible")
    axB.bar([0], [hid_lt], bottom=[vis_lt], color="#D55E00", edgecolor="black", linewidth=0.5, width=0.5,
            label="hidden in truck shadow")
    axB.bar([0], [other_lt], bottom=[vis_lt + hid_lt], color="#C0C0C0", edgecolor="black", linewidth=0.5, width=0.5,
            label="otherwise hidden")
    axB.text(0, vis_lt / 2, f"{vis_lt:.1f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    axB.text(0, vis_lt + hid_lt / 2, f"{hid_lt:.1f}", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    axB.text(0, vis_lt + hid_lt + other_lt / 2, f"{other_lt:.1f}", ha="center", va="center", fontsize=8, color="#333", fontweight="bold")
    axB.annotate("V2I reveals\nall hidden", xy=(0.27, vis_lt + (hid_lt + other_lt) / 2),
                 xytext=(0.95, vis_lt + (hid_lt + other_lt) / 2), fontsize=7.5, va="center",
                 arrowprops=dict(arrowstyle="->", color="#7B3294", lw=1.2), color="#7B3294")
    axB.set_xticks([0]); axB.set_xticklabels(["following a truck"])
    axB.set_ylabel("Road users per frame")
    axB.set_title(f"(b) Confident passing: onboard {obs_lt:.0%} $\\rightarrow$ V2I 100%", fontsize=10)
    axB.set_xlim(-0.7, 2.0)
    axB.legend(fontsize=7.2, loc="upper right", framealpha=0.95)
    axB.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(f"{a.out}.{fmt}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
