#!/usr/bin/env python3
"""Ego-involved near-miss observability.

Section 5.6 (parametric_ego_sweep.py) measures the observability of near-miss
interactions occurring among *third parties* in the scene. This script answers
the complementary, more safety-critical question: when the ego AV is itself a
party to a near-miss, can its onboard sensor observe the *conflict partner* --
the road user it is about to nearly collide with?

This requires the ego to be a real moving agent, so we adopt the same
vehicle-as-ego construction as vehicle_as_ego_sweep.py: each detected car/truck
acts as the ego, inheriting its measured position and heading, and is excluded
from its own observed/occluder set. For every near-miss event in which a vehicle
is a party, that vehicle is treated as the ego and its counterpart as the
conflict partner. We then evaluate whether the partner is visible from the ego
sensor (in-FOV, in-range, unoccluded) at the near-miss frame, under a forward
120-deg sensor and a 360-deg surround, plus the cooperative V2I upper bound.

Conflict definition:
  * PRIMARY -- the canonical TTC-based near-miss set (near_misses.parquet),
    human-validated and identical to the events used in Section 5.6. Re-scoping
    those events to the ego-involved subset keeps the two near-miss metrics
    directly comparable.
  * CROSS-CHECK -- post-encroachment time (PET) computed independently on the
    full trajectories (--with-pet). PET is the canonical surrogate for crossing
    conflicts and does not rely on instantaneous closing velocity; the check
    confirms the partner-observability conclusion is not an artifact of the TTC
    definition.

Outputs outputs/ego_near_miss_observability.json. The occlusion model, sensor
config, and self-exclusion are reused verbatim from parametric_ego_sweep.py /
vehicle_as_ego_sweep.py, so numbers are directly comparable to the other Part C
results. The canonical near-miss window (frames 6001-8000) is the only interval
with annotated events, so -- like Section 5.6 -- this analysis is scoped to it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from parametric_ego_sweep import SensorConfig, compute_ego_occlusion

SIM_DIR = Path(__file__).resolve().parents[1]
TRACKS = Path("data/inputs/tracks.parquet")
NEAR_MISSES = Path("data/inputs/near_misses.parquet")

EGO_CLASSES = frozenset({"car", "truck"})       # an AV is a car/truck
OCCLUDER_CLASSES = frozenset({"truck", "car"})
CONFLICT_CENTER = (7.5, 8.0)
PROXIMITY_R = 15.0
FRAME_PERIOD_S = 0.1                              # 10 Hz


def _load_tracks() -> pd.DataFrame:
    cols = ["frame_idx", "track_id", "class_name",
            "bbox_center_x", "bbox_center_y", "bbox_dx", "bbox_dy", "bbox_yaw"]
    tracks = pd.read_parquet(TRACKS, columns=cols)
    tracks["class_name"] = (tracks["class_name"].str.lower()
                            .str.replace("bike", "bicycle"))
    return tracks


def _load_near_misses() -> pd.DataFrame:
    nm = pd.read_parquet(NEAR_MISSES)
    for col in ("class_a", "class_b"):
        nm[col] = nm[col].str.lower().str.replace("bike", "bicycle")
    return nm


# ---------------------------------------------------------------------------
# Partner observability for one ego-perspective at one frame
# ---------------------------------------------------------------------------

def _partner_status(frame_tracks, ego_tid, partner_tid, sensor, heading_mode):
    """Return the partner's visibility dict from the ego's sensor, or None.

    The ego is removed from the scene (a sensor cannot see/occlude itself); the
    sensor points along the ego's measured yaw ("yaw") or at the intersection
    center ("inward").
    """
    ego_row = frame_tracks[frame_tracks["track_id"] == ego_tid]
    if ego_row.empty:
        return None
    ego_row = ego_row.iloc[0]
    ego_x = float(ego_row.bbox_center_x)
    ego_y = float(ego_row.bbox_center_y)
    if heading_mode == "inward":
        heading = math.atan2(CONFLICT_CENTER[1] - ego_y, CONFLICT_CENTER[0] - ego_x)
    else:
        heading = float(ego_row.bbox_yaw)
    scene = frame_tracks[frame_tracks["track_id"] != ego_tid]
    vis = compute_ego_occlusion(ego_x, ego_y, heading, scene, sensor, OCCLUDER_CLASSES)
    return vis.get(int(partner_tid))


def _ego_perspectives(nm_row):
    """Yield (ego_tid, partner_tid) for each vehicle party in a near-miss row."""
    out = []
    if nm_row.class_a in EGO_CLASSES:
        out.append((int(nm_row.track_id_a), int(nm_row.track_id_b)))
    if nm_row.class_b in EGO_CLASSES:
        out.append((int(nm_row.track_id_b), int(nm_row.track_id_a)))
    return out


# ---------------------------------------------------------------------------
# TTC-based ego-involved near-miss observability (PRIMARY)
# ---------------------------------------------------------------------------

def run_ttc(tracks, nm, fov_deg, heading_mode="yaw"):
    sensor = SensorConfig(fov_deg=fov_deg, range_m=35.0)
    tbf = {int(f): g for f, g in tracks.groupby("frame_idx")}
    rows = []
    for r in nm.itertuples(index=False):
        ft = tbf.get(int(r.frame_idx))
        if ft is None:
            continue
        for ego_tid, partner_tid in _ego_perspectives(r):
            st = _partner_status(ft, ego_tid, partner_tid, sensor, heading_mode)
            if st is None:
                continue
            rows.append({
                "frame": int(r.frame_idx),
                "pair": tuple(sorted((ego_tid, partner_tid))),
                "ego_tid": ego_tid,
                "partner_tid": partner_tid,
                "visible": bool(st["visible"]),
                "reason": st["reason"],
                "partner_range": float(st["range_m"]),
                "partner_bearing_abs_deg": abs(float(st["bearing_rel_deg"])),
            })
    return pd.DataFrame(rows)


def _summarize_obs(df):
    """Per-event-frame observability summary (the frame-by-frame metric used
    throughout the paper)."""
    n = len(df)
    if n == 0:
        return {}
    reason = df["reason"].value_counts(normalize=True).to_dict()
    # per-pair: partner visible at its closest-approach frame (the conflict
    # instant), and partner ever visible during the event (temporal memory).
    per_pair_closest = (df.sort_values("partner_range")
                        .groupby("pair").first()["visible"])
    per_pair_ever = df.groupby("pair")["visible"].max()
    return {
        "n_event_frames": int(n),
        "n_unique_pairs": int(df["pair"].nunique()),
        "partner_visible_rate": round(float(df["visible"].mean()), 4),
        "reason_breakdown": {k: round(float(reason.get(k, 0.0)), 4)
                             for k in ("visible", "out_of_fov", "occluded", "out_of_range")},
        "partner_visible_at_closest_frame": round(float(per_pair_closest.mean()), 4),
        "partner_visible_ever_in_event": round(float(per_pair_ever.mean()), 4),
        "median_partner_range_m": round(float(df["partner_range"].median()), 2),
        "median_partner_bearing_deg": round(float(df["partner_bearing_abs_deg"].median()), 1),
        "frac_partner_outside_120fov": round(float((df["partner_bearing_abs_deg"] > 60).mean()), 4),
        "frac_partner_behind_ego": round(float((df["partner_bearing_abs_deg"] > 120).mean()), 4),
    }


# ---------------------------------------------------------------------------
# PET-based ego-involved conflicts (CROSS-CHECK)
# ---------------------------------------------------------------------------

def _mine_pet_conflicts(tracks, frame_lo, frame_hi, d_cross_m=2.0, pet_thresh_s=5.0,
                        prox_gate_m=30.0):
    """Independently detect ego-involved conflicts by post-encroachment time.

    For each (ego vehicle, other road user) pair whose paths cross -- i.e. come
    within d_cross_m at some point, evaluated over the two trajectories without
    requiring simultaneity -- PET is the time gap between the two occupying that
    crossing point. A pair is a conflict if it crosses and PET <= pet_thresh_s.
    A cheap proximity gate (ever within prox_gate_m at a shared frame) prunes the
    pair list before the O(n_a * n_b) crossing test.

    Returns a list of dicts: {ego_tid, partner_tid, pet_s, eval_frame}, where
    eval_frame is the co-present frame of minimum center separation (the conflict
    instant at which partner observability is evaluated -- the moment the AV most
    needs to see the partner). PET sets *which pairs* are conflicts; observability
    is then evaluated identically to the TTC path, so the two are comparable.
    """
    win = tracks[(tracks["frame_idx"] >= frame_lo) & (tracks["frame_idx"] <= frame_hi)]
    by_track = {int(t): g.sort_values("frame_idx") for t, g in win.groupby("track_id")}
    cls = {int(t): g["class_name"].iloc[0] for t, g in win.groupby("track_id")}
    arr = {t: g[["frame_idx", "bbox_center_x", "bbox_center_y"]].to_numpy()
           for t, g in by_track.items()}

    # Proximity gate: candidate pairs ever within prox_gate_m at a shared frame.
    cand = set()
    for f, g in win.groupby("frame_idx"):
        pos = g[["track_id", "bbox_center_x", "bbox_center_y", "class_name"]].to_numpy()
        for i in range(len(pos)):
            ti, xi, yi, ci = int(pos[i, 0]), pos[i, 1], pos[i, 2], pos[i, 3]
            if ci not in EGO_CLASSES:
                continue
            for j in range(len(pos)):
                tj, xj, yj = int(pos[j, 0]), pos[j, 1], pos[j, 2]
                if ti == tj:
                    continue
                if (xi - xj) ** 2 + (yi - yj) ** 2 <= prox_gate_m ** 2:
                    cand.add((ti, tj))  # ordered: ego first

    conflicts = []
    for ego_tid, partner_tid in cand:
        a = arr[ego_tid]
        b = arr[partner_tid]
        # Path crossing: min spatial distance over all (a-point, b-point) pairs.
        d = np.hypot(a[:, 1][:, None] - b[:, 1][None, :],
                     a[:, 2][:, None] - b[:, 2][None, :])
        ia, ib = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[ia, ib] > d_cross_m:
            continue  # paths never share a conflict point
        pet_s = abs(a[ia, 0] - b[ib, 0]) * FRAME_PERIOD_S
        if pet_s > pet_thresh_s:
            continue
        # Conflict instant = co-present frame of min center separation.
        fa, fb = set(a[:, 0].astype(int)), set(b[:, 0].astype(int))
        shared = sorted(fa & fb)
        if not shared:
            continue
        amap = {int(f): (x, y) for f, x, y in a}
        bmap = {int(f): (x, y) for f, x, y in b}
        best_f, best_d = None, 1e18
        for f in shared:
            ax, ay = amap[f]
            bx, by = bmap[f]
            dd = (ax - bx) ** 2 + (ay - by) ** 2
            if dd < best_d:
                best_d, best_f = dd, f
        conflicts.append({
            "ego_tid": ego_tid, "partner_tid": partner_tid,
            "partner_class": cls[partner_tid],
            "pet_s": round(float(pet_s), 3),
            "min_sep_m": round(float(best_d ** 0.5), 2),
            "eval_frame": int(best_f),
        })
    return conflicts


def run_pet(tracks, frame_lo, frame_hi, fov_deg, heading_mode="yaw",
            d_cross_m=2.0, pet_thresh_s=5.0):
    sensor = SensorConfig(fov_deg=fov_deg, range_m=35.0)
    tbf = {int(f): g for f, g in tracks.groupby("frame_idx")}
    conflicts = _mine_pet_conflicts(tracks, frame_lo, frame_hi,
                                    d_cross_m=d_cross_m, pet_thresh_s=pet_thresh_s)
    rows = []
    for c in conflicts:
        ft = tbf.get(c["eval_frame"])
        if ft is None:
            continue
        st = _partner_status(ft, c["ego_tid"], c["partner_tid"], sensor, heading_mode)
        if st is None:
            continue
        rows.append({
            "pair": tuple(sorted((c["ego_tid"], c["partner_tid"]))),
            "pet_s": c["pet_s"], "min_sep_m": c["min_sep_m"],
            "visible": bool(st["visible"]), "reason": st["reason"],
            "partner_range": float(st["range_m"]),
        })
    return pd.DataFrame(rows)


def _summarize_pet(df):
    if len(df) == 0:
        return {"n_conflict_pairs": 0}
    reason = df["reason"].value_counts(normalize=True).to_dict()
    return {
        "n_conflict_pairs": int(len(df)),
        "partner_visible_rate": round(float(df["visible"].mean()), 4),
        "reason_breakdown": {k: round(float(reason.get(k, 0.0)), 4)
                             for k in ("visible", "out_of_fov", "occluded", "out_of_range")},
        "median_pet_s": round(float(df["pet_s"].median()), 2),
        "median_min_sep_m": round(float(df["min_sep_m"].median()), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TRACKS, NEAR_MISSES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks", type=Path, default=TRACKS)
    ap.add_argument("--near-misses", type=Path, default=NEAR_MISSES)
    ap.add_argument("--heading-mode", choices=["yaw", "inward"], default="yaw")
    ap.add_argument("--with-pet", action="store_true",
                    help="Also run the PET cross-check (slower).")
    ap.add_argument("--pet-thresh-s", type=float, default=5.0)
    ap.add_argument("--pet-cross-m", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=SIM_DIR / "outputs" / "ego_near_miss_observability.json")
    args = ap.parse_args()
    TRACKS = args.tracks
    NEAR_MISSES = args.near_misses

    print(f"Loading tracks    {TRACKS}")
    tracks = _load_tracks()
    print(f"  {len(tracks):,} track rows")
    print(f"Loading near-miss {NEAR_MISSES}")
    nm = _load_near_misses()
    frame_lo, frame_hi = int(nm["frame_idx"].min()), int(nm["frame_idx"].max())
    print(f"  {len(nm):,} near-miss event-frames over frames {frame_lo}-{frame_hi}")
    n_ego_inv = int((nm["class_a"].isin(EGO_CLASSES) | nm["class_b"].isin(EGO_CLASSES)).sum())
    print(f"  {n_ego_inv:,} involve a vehicle (ego-involvable)\n")

    out = {
        "_source": "ego_near_miss_sweep.py",
        "window": [frame_lo, frame_hi],
        "heading_mode": args.heading_mode,
        "ego_classes": sorted(EGO_CLASSES),
        "ttc": {},
        "pet_cross_check": {},
    }

    print("=== TTC-based ego-involved near-miss partner observability ===")
    for fov in (120.0, 360.0):
        df = run_ttc(tracks, nm, fov, args.heading_mode)
        s = _summarize_obs(df)
        out["ttc"][f"fov_{int(fov)}"] = s
        print(f"\nFOV {int(fov)} deg (heading={args.heading_mode}):")
        print(f"  event-frames={s['n_event_frames']}  unique pairs={s['n_unique_pairs']}")
        print(f"  partner VISIBLE (per event-frame): {s['partner_visible_rate']:.3f}")
        print(f"  reasons: {s['reason_breakdown']}")
        print(f"  partner visible at closest frame (per pair): {s['partner_visible_at_closest_frame']:.3f}")
        print(f"  partner visible ever in event (per pair):    {s['partner_visible_ever_in_event']:.3f}")
        print(f"  median partner range {s['median_partner_range_m']} m, "
              f"bearing {s['median_partner_bearing_deg']} deg, "
              f"outside-120FOV {s['frac_partner_outside_120fov']:.2f}, "
              f"behind-ego {s['frac_partner_behind_ego']:.2f}")
    out["ttc"]["cooperative_v2i"] = 1.0
    print("\n  Cooperative V2I: partner observable = 1.000 (infrastructure sees partner)")

    if args.with_pet:
        print("\n=== PET cross-check (independent conflict definition) ===")
        out["pet_cross_check"]["params"] = {
            "pet_thresh_s": args.pet_thresh_s, "cross_dist_m": args.pet_cross_m,
            "eval": "partner observability at co-present frame of min center separation",
        }
        for fov in (120.0, 360.0):
            df = run_pet(tracks, frame_lo, frame_hi, fov, args.heading_mode,
                         d_cross_m=args.pet_cross_m, pet_thresh_s=args.pet_thresh_s)
            s = _summarize_pet(df)
            out["pet_cross_check"][f"fov_{int(fov)}"] = s
            print(f"\nFOV {int(fov)} deg: conflict pairs={s['n_conflict_pairs']}  "
                  f"partner VISIBLE={s.get('partner_visible_rate')}")
            print(f"  reasons: {s.get('reason_breakdown')}  "
                  f"median PET {s.get('median_pet_s')} s")
        out["pet_cross_check"]["cooperative_v2i"] = 1.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
