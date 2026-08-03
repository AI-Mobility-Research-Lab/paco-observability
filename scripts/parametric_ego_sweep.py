#!/usr/bin/env python3
"""Parametric ego-position sweep for cooperative perception benefit analysis.

For each frame in the dataset, places hypothetical AV ego sensors at sampled
positions along intersection approach lanes and computes:
- Which road users are visible vs. occluded from the ego sensor
- Which near-miss interactions are detectable (both parties visible)
- Risk-weighted observability metrics comparing onboard-only vs. cooperative

Extends the angular-wedge occlusion model from compute_observability.py.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """Onboard AV sensor parameters."""
    fov_deg: float = 120.0      # Horizontal field of view (degrees)
    range_m: float = 35.0       # Maximum detection range (meters)
    near_blind_m: float = 1.5   # Near-blind radius (meters)
    offset_x: float = 1.5       # Sensor offset from vehicle center, forward (meters)
    offset_y: float = 0.0       # Sensor offset lateral (meters)
    theta_bins: int = 360       # Angular resolution for occlusion grid

    @property
    def fov_rad(self) -> float:
        return math.radians(self.fov_deg)

    @property
    def half_fov_rad(self) -> float:
        return self.fov_rad / 2.0


@dataclass
class ApproachLane:
    """A parameterized approach lane for ego sampling."""
    name: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    heading_rad: float  # Vehicle heading on this approach
    n_positions: int = 15

    def sample_positions(self) -> list[tuple[float, float, float]]:
        """Return (x, y, heading) tuples along the approach."""
        positions = []
        for i in range(self.n_positions):
            t = i / max(self.n_positions - 1, 1)
            x = self.start_x + t * (self.end_x - self.start_x)
            y = self.start_y + t * (self.end_y - self.start_y)
            positions.append((x, y, self.heading_rad))
        return positions


# Default approach lanes derived from trajectory data analysis:
# North approach: vehicles at x~[5,9], y from ~35 down to ~8 (heading ~-90deg = south)
# South approach: vehicles at x~[6,9], y from ~-15 up to ~8 (heading ~+90deg = north)
# East approach:  vehicles at x~35, y~[6,10] heading west (~180deg)
# West approach:  vehicles at x~-20, y~[6,10] heading east (~0deg)
DEFAULT_APPROACHES = [
    ApproachLane("north", 7.0, 35.0, 7.0, 8.0, math.radians(-90), n_positions=15),
    ApproachLane("south", 8.0, -15.0, 8.0, 8.0, math.radians(90), n_positions=15),
    ApproachLane("east", 35.0, 8.0, 8.0, 8.0, math.radians(180), n_positions=15),
    ApproachLane("west", -20.0, 8.0, 8.0, 8.0, math.radians(0), n_positions=15),
]


# ---------------------------------------------------------------------------
# Occlusion computation (ego-centric)
# ---------------------------------------------------------------------------

def _box_corners_xy(cx: float, cy: float, dx: float, dy: float, yaw: float) -> np.ndarray:
    """Compute 2D BEV corners of a bounding box."""
    half_dx, half_dy = dx / 2.0, dy / 2.0
    corners = np.array([
        [half_dx, half_dy], [half_dx, -half_dy],
        [-half_dx, -half_dy], [-half_dx, half_dy],
    ], dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return corners @ rot.T + np.array([cx, cy])


def _normalize_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _angular_span(angles: np.ndarray) -> tuple[float, float]:
    """Return (start, end) of the minimal arc covering angles in [-pi, pi]."""
    if angles.size == 0:
        return (0.0, 0.0)
    sorted_a = np.sort(_normalize_angle(angles))
    if sorted_a.size == 1:
        return (float(sorted_a[0]), float(sorted_a[0]))
    wrapped = np.concatenate([sorted_a, [sorted_a[0] + 2.0 * math.pi]])
    gaps = np.diff(wrapped)
    max_idx = int(np.argmax(gaps))
    start = float(wrapped[max_idx + 1])
    end = float(wrapped[max_idx] + 2.0 * math.pi)
    start = (start + math.pi) % (2.0 * math.pi) - math.pi
    end = (end + math.pi) % (2.0 * math.pi) - math.pi
    return (start, end)


def compute_ego_occlusion(
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    frame_objects: pd.DataFrame,
    sensor: SensorConfig,
    occluder_classes: set[str] = frozenset({"truck", "car"}),
) -> dict[int, dict]:
    """Compute visibility of each object from the ego sensor.

    Returns a dict mapping track_id -> {visible, range_m, bearing_deg, occluded_by}.
    """
    half_fov = sensor.half_fov_rad
    results = {}

    # Transform object positions to ego-centric coordinates
    if frame_objects.empty:
        return results

    # Sensor position (offset from vehicle center)
    sensor_x = ego_x + sensor.offset_x * math.cos(ego_heading) - sensor.offset_y * math.sin(ego_heading)
    sensor_y = ego_y + sensor.offset_x * math.sin(ego_heading) + sensor.offset_y * math.cos(ego_heading)

    objects = []
    for row in frame_objects.itertuples(index=False):
        dx = float(row.bbox_center_x) - sensor_x
        dy = float(row.bbox_center_y) - sensor_y
        rng = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        # Bearing relative to ego heading
        rel_bearing = _normalize_angle(np.array([bearing - ego_heading]))[0]

        objects.append({
            "track_id": int(row.track_id),
            "class_name": str(row.class_name),
            "cx": float(row.bbox_center_x),
            "cy": float(row.bbox_center_y),
            "dx": float(row.bbox_dx),
            "dy": float(row.bbox_dy),
            "yaw": float(row.bbox_yaw),
            "range_m": rng,
            "bearing_abs": bearing,
            "bearing_rel": float(rel_bearing),
        })

    # Sort by range (nearest first) for occlusion computation
    objects.sort(key=lambda o: o["range_m"])

    # Build occlusion map: angular bins within FOV
    n_bins = sensor.theta_bins
    fov_start = -half_fov
    fov_end = half_fov
    bin_width = (fov_end - fov_start) / n_bins
    # min_occluder_range per bin: the nearest occluder range blocking that angle
    min_occluder_range = np.full(n_bins, np.inf)
    # Track which object created each occlusion
    occluder_track_ids = [None] * n_bins

    for obj in objects:
        tid = obj["track_id"]
        rng = obj["range_m"]
        rel_bearing = obj["bearing_rel"]

        # Check if in FOV at all
        in_fov = abs(rel_bearing) <= half_fov
        in_range = sensor.near_blind_m < rng <= sensor.range_m

        # Compute angular extent of this object from ego
        corners = _box_corners_xy(obj["cx"], obj["cy"], obj["dx"], obj["dy"], obj["yaw"])
        corner_dx = corners[:, 0] - sensor_x
        corner_dy = corners[:, 1] - sensor_y
        corner_bearings_abs = np.arctan2(corner_dy, corner_dx)
        corner_bearings_rel = _normalize_angle(corner_bearings_abs - ego_heading)
        corner_ranges = np.hypot(corner_dx, corner_dy)
        near_corner_range = float(np.min(corner_ranges))

        # Is this object visible?
        # Check: (a) center in FOV and range, (b) not fully occluded by nearer objects
        if not (in_fov and in_range):
            results[tid] = {
                "visible": False,
                "reason": "out_of_fov" if not in_fov else "out_of_range",
                "range_m": rng,
                "bearing_rel_deg": math.degrees(rel_bearing),
                "class_name": obj["class_name"],
                "occluded_by": None,
            }
        else:
            # Check occlusion: is the center angle blocked by a nearer object?
            center_bin = int((rel_bearing - fov_start) / bin_width)
            center_bin = max(0, min(n_bins - 1, center_bin))
            blocker_range = min_occluder_range[center_bin]
            occluded = rng > blocker_range + 0.5  # 0.5m margin
            blocker_id = occluder_track_ids[center_bin] if occluded else None

            results[tid] = {
                "visible": not occluded,
                "reason": "occluded" if occluded else "visible",
                "range_m": rng,
                "bearing_rel_deg": math.degrees(rel_bearing),
                "class_name": obj["class_name"],
                "occluded_by": blocker_id,
            }

        # If this object is an occluder class, update the occlusion map
        if obj["class_name"] in occluder_classes:
            # Mark angular bins covered by this object
            for cb in corner_bearings_rel:
                b = float(cb)
                if fov_start <= b <= fov_end:
                    bin_idx = int((b - fov_start) / bin_width)
                    bin_idx = max(0, min(n_bins - 1, bin_idx))
                    if near_corner_range < min_occluder_range[bin_idx]:
                        min_occluder_range[bin_idx] = near_corner_range
                        occluder_track_ids[bin_idx] = tid

            # Fill the angular arc spanned by the object's corners. A convex
            # object viewed from outside subtends < 180 deg, so we fill the
            # shorter arc. At a full 360-deg FOV an object straddling the
            # +-180-deg wrap boundary has its min/max corner bins at opposite
            # ends of the array; filling min..max directly would spuriously
            # occlude the entire circle, so in that case fill the complement
            # (the short arc through the wrap). At sub-360 FOV this branch is
            # inert -- objects behind the ego are excluded from the FOV window --
            # so 120-deg results are unchanged.
            valid_bins = sorted(
                max(0, min(n_bins - 1, int((float(b) - fov_start) / bin_width)))
                for b in corner_bearings_rel
                if fov_start <= float(b) <= fov_end
            )
            if len(valid_bins) >= 2:
                b_lo, b_hi = valid_bins[0], valid_bins[-1]
                is_full_circle = (fov_end - fov_start) >= 2 * math.pi - 1e-6
                if is_full_circle and (b_hi - b_lo) > n_bins // 2:
                    fill_bins = list(range(b_hi, n_bins)) + list(range(0, b_lo + 1))
                else:
                    fill_bins = range(b_lo, b_hi + 1)
                for bi in fill_bins:
                    if near_corner_range < min_occluder_range[bi]:
                        min_occluder_range[bi] = near_corner_range
                        occluder_track_ids[bi] = tid

    return results


# ---------------------------------------------------------------------------
# Frame-level analysis
# ---------------------------------------------------------------------------

def analyze_frame(
    frame_idx: int,
    frame_tracks: pd.DataFrame,
    frame_near_misses: pd.DataFrame,
    ego_positions: list[tuple[str, float, float, float]],
    sensor: SensorConfig,
    occluder_classes: set[str],
) -> list[dict]:
    """Analyze one frame across all ego positions.

    Returns list of per-ego-position result dicts.
    """
    results = []
    for approach_name, ego_x, ego_y, ego_heading in ego_positions:
        # Skip if ego is too close to its own position in the scene
        # (the ego vehicle wouldn't be detected by itself)
        vis = compute_ego_occlusion(
            ego_x, ego_y, ego_heading,
            frame_tracks, sensor, occluder_classes,
        )

        n_total = len(vis)
        n_visible = sum(1 for v in vis.values() if v["visible"])
        n_occluded = sum(1 for v in vis.values() if not v["visible"] and v["reason"] == "occluded")
        n_out_fov = sum(1 for v in vis.values() if not v["visible"] and v["reason"] == "out_of_fov")
        n_out_range = sum(1 for v in vis.values() if not v["visible"] and v["reason"] == "out_of_range")

        # Per-class visibility
        class_vis = {}
        for v in vis.values():
            cls = v["class_name"]
            if cls not in class_vis:
                class_vis[cls] = {"visible": 0, "total": 0}
            class_vis[cls]["total"] += 1
            if v["visible"]:
                class_vis[cls]["visible"] += 1

        # --- Proximity-based metrics ---
        # "Nearby" = within 15m of ego center; "conflict zone" = within 20m of
        # intersection center (7.5, 8.0)
        PROXIMITY_R = 15.0
        CONFLICT_ZONE_R = 20.0
        CONFLICT_CENTER = (7.5, 8.0)

        n_nearby_total = 0
        n_nearby_visible = 0
        n_cz_total = 0
        n_cz_visible = 0
        for v in vis.values():
            rng = v["range_m"]
            if rng <= PROXIMITY_R:
                n_nearby_total += 1
                if v["visible"]:
                    n_nearby_visible += 1
            # Conflict zone: distance from object to intersection center
            # (we need the original position; approximate via ego + range * bearing)
            # More precise: use stored object positions from the loop above

        # Second pass for conflict-zone using original object positions
        sensor_x = ego_x + sensor.offset_x * math.cos(ego_heading)
        sensor_y = ego_y + sensor.offset_x * math.sin(ego_heading)
        for row_obj in frame_tracks.itertuples(index=False):
            tid = int(row_obj.track_id)
            obj_dist_to_center = math.hypot(
                float(row_obj.bbox_center_x) - CONFLICT_CENTER[0],
                float(row_obj.bbox_center_y) - CONFLICT_CENTER[1],
            )
            if obj_dist_to_center <= CONFLICT_ZONE_R:
                n_cz_total += 1
                if tid in vis and vis[tid]["visible"]:
                    n_cz_visible += 1

        nearby_vis_ratio = n_nearby_visible / n_nearby_total if n_nearby_total > 0 else float("nan")
        cz_vis_ratio = n_cz_visible / n_cz_total if n_cz_total > 0 else float("nan")

        # --- Near-miss detectability (multiple criteria) ---
        nm_total = 0
        nm_both_visible = 0        # strict: both parties visible
        nm_at_least_one = 0        # relaxed: at least one party visible (ego-awareness)
        nm_ego_relevant = 0        # ego-relevant: at least one party within 15m of ego
        nm_ego_relevant_aware = 0  # ego-relevant AND at least one party visible
        nm_detectable_cooperative = 0
        for _, nm_row in frame_near_misses.iterrows():
            tid_a = int(nm_row["track_id_a"])
            tid_b = int(nm_row["track_id_b"])
            if tid_a not in vis or tid_b not in vis:
                continue
            nm_total += 1
            a_vis = vis[tid_a]["visible"]
            b_vis = vis[tid_b]["visible"]
            a_rng = vis[tid_a]["range_m"]
            b_rng = vis[tid_b]["range_m"]

            if a_vis and b_vis:
                nm_both_visible += 1
            if a_vis or b_vis:
                nm_at_least_one += 1

            # Is this near-miss ego-relevant? (at least one party nearby)
            ego_relevant = (a_rng <= PROXIMITY_R) or (b_rng <= PROXIMITY_R)
            if ego_relevant:
                nm_ego_relevant += 1
                if a_vis or b_vis:
                    nm_ego_relevant_aware += 1

            # Cooperative: infrastructure sees everything in its FOV
            nm_detectable_cooperative += 1

        vis_ratio = n_visible / n_total if n_total > 0 else float("nan")

        results.append({
            "frame_idx": frame_idx,
            "approach": approach_name,
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_heading_deg": math.degrees(ego_heading),
            # --- Absolute counts ---
            "n_objects_total": n_total,
            "n_visible_onboard": n_visible,
            "n_occluded": n_occluded,
            "n_out_of_fov": n_out_fov,
            "n_out_of_range": n_out_range,
            # --- Global intersection-wide ratio (denominator = all objects) ---
            "visibility_ratio_onboard": vis_ratio,
            "visibility_ratio_cooperative": 1.0,
            # --- Proximity-based metrics ---
            "n_nearby_total": n_nearby_total,
            "n_nearby_visible": n_nearby_visible,
            "nearby_visibility_ratio": nearby_vis_ratio,
            "n_conflict_zone_total": n_cz_total,
            "n_conflict_zone_visible": n_cz_visible,
            "conflict_zone_visibility_ratio": cz_vis_ratio,
            # --- Near-miss metrics (multiple criteria) ---
            "nm_total": nm_total,
            "nm_both_visible": nm_both_visible,
            "nm_at_least_one_visible": nm_at_least_one,
            "nm_ego_relevant": nm_ego_relevant,
            "nm_ego_relevant_aware": nm_ego_relevant_aware,
            "nm_detectable_onboard": nm_both_visible,  # backwards compat
            "nm_detectable_cooperative": nm_detectable_cooperative,
            "class_visibility": json.dumps(class_vis),
        })

    return results


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def build_ego_grid(approaches: list[ApproachLane]) -> list[tuple[str, float, float, float]]:
    """Build flat list of (approach_name, x, y, heading) for all ego positions."""
    grid = []
    for ap in approaches:
        for x, y, h in ap.sample_positions():
            grid.append((ap.name, x, y, h))
    return grid


# Worker globals for multiprocessing: set once per process via the initializer
# so the ego grid / sensor config are not re-pickled for every frame.
_WCTX: dict = {}


def _worker_init(ego_grid, sensor, occluder_classes):
    _WCTX["ego_grid"] = ego_grid
    _WCTX["sensor"] = sensor
    _WCTX["occluder_classes"] = occluder_classes


def _worker_frame(task):
    """Analyze one frame in a worker process. Frames are fully independent, so
    results are numerically identical to the serial path."""
    fidx, frame_tracks, frame_nm = task
    return analyze_frame(
        int(fidx), frame_tracks, frame_nm,
        _WCTX["ego_grid"], _WCTX["sensor"], _WCTX["occluder_classes"],
    )


def run_sweep(
    tracks_path: Path,
    near_miss_path: Path,
    out_dir: Path,
    sensor: SensorConfig,
    approaches: list[ApproachLane],
    occluder_classes: set[str],
    frame_start: int = 0,
    frame_end: int = 0,
    frame_step: int = 10,
    max_frames: int = 0,
    n_procs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the full parametric ego sweep."""
    print(f"Loading tracks from {tracks_path}")
    tracks = pd.read_parquet(tracks_path)
    print(f"  {len(tracks):,} rows")

    print(f"Loading near-misses from {near_miss_path}")
    near_misses = pd.read_parquet(near_miss_path)
    # Normalize class names in near-misses
    for col in ["class_a", "class_b"]:
        if col in near_misses.columns:
            near_misses[col] = near_misses[col].str.lower().str.replace("bike", "bicycle")
    print(f"  {len(near_misses):,} rows")

    # Frame selection
    all_frames = np.sort(tracks["frame_idx"].unique())
    if frame_start > 0:
        all_frames = all_frames[all_frames >= frame_start]
    if frame_end > 0:
        all_frames = all_frames[all_frames <= frame_end]
    # Subsample frames
    if frame_step > 1:
        all_frames = all_frames[::frame_step]
    if max_frames > 0:
        all_frames = all_frames[:max_frames]

    print(f"Analyzing {len(all_frames)} frames (step={frame_step})")

    ego_grid = build_ego_grid(approaches)
    print(f"Ego grid: {len(ego_grid)} positions across {len(approaches)} approaches")

    # Pre-group data
    tracks_grouped = tracks.groupby("frame_idx")
    nm_grouped = near_misses.groupby("frame_idx") if len(near_misses) > 0 else {}

    def _nm_for(fidx):
        if isinstance(nm_grouped, dict):
            return pd.DataFrame()
        if fidx in nm_grouped.groups:
            return nm_grouped.get_group(fidx)
        return pd.DataFrame()

    frames = [int(f) for f in all_frames if f in tracks_grouped.groups]
    tasks = ((f, tracks_grouped.get_group(f), _nm_for(f)) for f in frames)

    all_results = []
    if n_procs and n_procs > 1:
        print(f"Parallelizing across {n_procs} processes")
        with mp.Pool(n_procs, initializer=_worker_init,
                     initargs=(ego_grid, sensor, occluder_classes)) as pool:
            for i, results in enumerate(pool.imap_unordered(_worker_frame, tasks, chunksize=16)):
                all_results.extend(results)
                if verbose and (i + 1) % 500 == 0:
                    print(f"  processed {i + 1}/{len(frames)} frames")
    else:
        for i, (f, ft, fnm) in enumerate(tasks):
            all_results.extend(analyze_frame(f, ft, fnm, ego_grid, sensor, occluder_classes))
            if verbose and (i + 1) % 100 == 0:
                print(f"  processed {i + 1}/{len(frames)} frames")

    df = pd.DataFrame(all_results)
    # Deterministic row order (imap_unordered returns frames out of order).
    sort_cols = [c for c in ("frame_idx", "approach") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Parametric ego sweep for cooperative perception analysis.")
    parser.add_argument(
        "--tracks", type=Path,
        default=Path("data/inputs/tracks.parquet"),
    )
    parser.add_argument(
        # Canonical near-miss file, frame-aligned to the run-20260214 tracks
        # over the 6001-8000 analysis window (the run-20260127 file is from a
        # different detection run and does NOT share track IDs/frames).
        "--near-misses", type=Path,
        default=Path("data/inputs/near_misses.parquet"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(
            "outputs"
        ),
    )
    parser.add_argument("--fov-deg", type=float, default=120.0)
    parser.add_argument("--range-m", type=float, default=35.0)
    # Canonical analysis = the full record (all frames, ~68 min at 10 Hz).
    # frame-start/end = 0 means "no bound". Near-miss metrics auto-scope to the
    # 6001-8000 window, where near-miss events are annotated.
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame within [frame-start, frame-end]")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--n-ego-per-approach", type=int, default=15)
    parser.add_argument("--n-procs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="Worker processes for frame-parallel analysis (1 = serial)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sensor = SensorConfig(
        fov_deg=args.fov_deg,
        range_m=args.range_m,
    )

    approaches = [
        ApproachLane("north", 7.0, 35.0, 7.0, 8.0, math.radians(-90), n_positions=args.n_ego_per_approach),
        ApproachLane("south", 8.0, -15.0, 8.0, 8.0, math.radians(90), n_positions=args.n_ego_per_approach),
        ApproachLane("east", 35.0, 8.0, 8.0, 8.0, math.radians(180), n_positions=args.n_ego_per_approach),
        ApproachLane("west", -20.0, 8.0, 8.0, 8.0, math.radians(0), n_positions=args.n_ego_per_approach),
    ]

    occluder_classes = {"truck", "car"}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = run_sweep(
        tracks_path=args.tracks,
        near_miss_path=args.near_misses,
        out_dir=args.out_dir,
        sensor=sensor,
        approaches=approaches,
        occluder_classes=occluder_classes,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        n_procs=args.n_procs,
        verbose=args.verbose,
    )

    # Save results
    out_path = args.out_dir / "ego_sweep_results.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df):,} rows to {out_path}")

    # Summary statistics
    print("\n=== Summary ===")
    print(f"Frames analyzed: {df['frame_idx'].nunique()}")
    print(f"Ego positions: {len(df) // df['frame_idx'].nunique() if df['frame_idx'].nunique() > 0 else 0}")

    print(f"\n--- Intersection-Wide Observability (all objects) ---")
    print(f"Overall: mean={df['visibility_ratio_onboard'].mean():.3f}")
    for approach in df["approach"].unique():
        sub = df[df["approach"] == approach]
        print(f"  {approach}: mean={sub['visibility_ratio_onboard'].mean():.3f}")

    print(f"\n--- Proximity-Based Observability (objects within 15m of ego) ---")
    nearby_valid = df.dropna(subset=["nearby_visibility_ratio"])
    if len(nearby_valid) > 0:
        print(f"Nearby mean visibility: {nearby_valid['nearby_visibility_ratio'].mean():.3f}")
        print(f"Mean nearby objects: {nearby_valid['n_nearby_total'].mean():.1f}")

    print(f"\n--- Conflict-Zone Observability (within 20m of center) ---")
    cz_valid = df.dropna(subset=["conflict_zone_visibility_ratio"])
    if len(cz_valid) > 0:
        print(f"Conflict-zone mean visibility: {cz_valid['conflict_zone_visibility_ratio'].mean():.3f}")
        print(f"Mean conflict-zone objects: {cz_valid['n_conflict_zone_total'].mean():.1f}")

    print(f"\n--- Absolute Counts (mean per ego-frame) ---")
    print(f"Total objects:   {df['n_objects_total'].mean():.1f}")
    print(f"Visible:         {df['n_visible_onboard'].mean():.1f}")
    print(f"Occluded:        {df['n_occluded'].mean():.1f}")
    print(f"Out of FOV:      {df['n_out_of_fov'].mean():.1f}")
    print(f"Out of range:    {df['n_out_of_range'].mean():.1f}")

    if df["nm_total"].sum() > 0:
        nm_both = df["nm_both_visible"].sum() / df["nm_total"].sum()
        nm_one = df["nm_at_least_one_visible"].sum() / df["nm_total"].sum()
        print(f"\n--- Near-Miss Detectability ---")
        print(f"Both parties visible (strict):      {nm_both:.3f}")
        print(f"At least one party visible:         {nm_one:.3f}")
        if df["nm_ego_relevant"].sum() > 0:
            nm_ego_aware = df["nm_ego_relevant_aware"].sum() / df["nm_ego_relevant"].sum()
            print(f"Ego-relevant events:                {df['nm_ego_relevant'].sum()}")
            print(f"Ego-relevant awareness rate:        {nm_ego_aware:.3f}")
        print(f"Cooperative (both visible):          1.000")

    # Save summary
    nearby_valid = df.dropna(subset=["nearby_visibility_ratio"])
    cz_valid = df.dropna(subset=["conflict_zone_visibility_ratio"])
    summary = {
        "sensor_config": {
            "fov_deg": sensor.fov_deg,
            "range_m": sensor.range_m,
            "near_blind_m": sensor.near_blind_m,
        },
        "frames_analyzed": int(df["frame_idx"].nunique()),
        "ego_positions_per_frame": len(build_ego_grid(approaches)),
        "total_observations": int(len(df)),
        "intersection_wide_visibility": float(df["visibility_ratio_onboard"].mean()),
        "nearby_visibility": float(nearby_valid["nearby_visibility_ratio"].mean()) if len(nearby_valid) > 0 else None,
        "conflict_zone_visibility": float(cz_valid["conflict_zone_visibility_ratio"].mean()) if len(cz_valid) > 0 else None,
        "mean_counts": {
            "total_objects": float(df["n_objects_total"].mean()),
            "visible": float(df["n_visible_onboard"].mean()),
            "occluded": float(df["n_occluded"].mean()),
            "out_of_fov": float(df["n_out_of_fov"].mean()),
            "out_of_range": float(df["n_out_of_range"].mean()),
            "nearby_total": float(df["n_nearby_total"].mean()),
            "nearby_visible": float(df["n_nearby_visible"].mean()),
        },
        "near_miss_detectability": {},
        "by_approach": {},
    }
    if df["nm_total"].sum() > 0:
        summary["near_miss_detectability"] = {
            "both_visible_rate": float(df["nm_both_visible"].sum() / df["nm_total"].sum()),
            "at_least_one_visible_rate": float(df["nm_at_least_one_visible"].sum() / df["nm_total"].sum()),
            "ego_relevant_awareness_rate": float(df["nm_ego_relevant_aware"].sum() / df["nm_ego_relevant"].sum()) if df["nm_ego_relevant"].sum() > 0 else None,
        }
    for approach in sorted(df["approach"].unique()):
        sub = df[df["approach"] == approach]
        sub_nearby = sub.dropna(subset=["nearby_visibility_ratio"])
        summary["by_approach"][approach] = {
            "mean_visibility_onboard": float(sub["visibility_ratio_onboard"].mean()),
            "median_visibility_onboard": float(sub["visibility_ratio_onboard"].median()),
            "std_visibility_onboard": float(sub["visibility_ratio_onboard"].std()),
            "mean_nearby_visibility": float(sub_nearby["nearby_visibility_ratio"].mean()) if len(sub_nearby) > 0 else None,
        }

    summary_path = args.out_dir / "ego_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
