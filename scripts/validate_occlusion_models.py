#!/usr/bin/env python3
"""Validate low-cost observability models against analytic ray--OBB geometry."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from paco_observability.geometry import GroundPlane
from paco_observability.provenance import audit_tracks
from paco_observability.validation import (
    ValidationConfig,
    balanced_stratified_frames,
    config_as_dict,
    default_ego_grid,
    evaluate_frame,
    summarize_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--expected-frame-range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--allow-missing-frame", type=int, action="append", default=[])
    parser.add_argument(
        "--exclude-frame",
        type=int,
        action="append",
        default=[],
        help="Source-quality exclusion applied before frame sampling",
    )
    parser.add_argument("--n-ego-per-approach", type=int, default=15)
    parser.add_argument("--fov-deg", type=float, nargs="+", default=[120.0, 360.0])
    parser.add_argument("--range-m", type=float, default=35.0)
    parser.add_argument("--near-blind-m", type=float, default=1.5)
    parser.add_argument("--sensor-height-m", type=float, default=1.8)
    parser.add_argument("--ego-clearance-m", type=float, default=4.0)
    parser.add_argument("--angular-resolution-deg", type=float, default=0.25)
    parser.add_argument("--occlusion-clearance-m", type=float, default=0.0)
    parser.add_argument("--visible-fraction-threshold", type=float, default=0.05)
    parser.add_argument(
        "--sparse-min-visible-rays",
        type=int,
        default=3,
        help=(
            "Minimum clear rays among the 15-point sparse sampler. The canonical value "
            "3 was selected on a disjoint 100-frame calibration subset."
        ),
    )
    parser.add_argument("--ray-grids", type=int, nargs="+", default=[17])
    parser.add_argument(
        "--z-modes", choices=["raw", "ground_anchored"], nargs="+", default=["raw", "ground_anchored"]
    )
    parser.add_argument(
        "--ground-plane",
        type=float,
        nargs=4,
        metavar=("NX", "NY", "NZ", "D"),
        required=True,
    )
    return parser.parse_args()


def _run(payload: tuple[pd.DataFrame, list[Any], GroundPlane, ValidationConfig]):
    frame, egos, plane, config = payload
    return evaluate_frame(frame, ego_poses=egos, ground_plane=plane, config=config)


def main() -> None:
    args = parse_args()
    tracks = pd.read_parquet(args.tracks)
    plane = GroundPlane.from_coefficients(args.ground_plane)
    audit = audit_tracks(
        tracks,
        ground_normal=tuple(args.ground_plane[:3]),
        ground_offset=args.ground_plane[3],
        expected_frame_range=tuple(args.expected_frame_range)
        if args.expected_frame_range
        else None,
        allowed_missing_frames=args.allow_missing_frame,
    )
    if not audit["quality_gate"]["passed"]:
        raise SystemExit(f"input failed quality gate: {audit['quality_gate']}")
    excluded_source_frames = sorted(set(args.exclude_frame))
    available_frames = set(tracks["frame_idx"].astype(int).unique())
    unknown_exclusions = sorted(set(excluded_source_frames) - available_frames)
    # A documented partial scan can have no detections (and therefore no track
    # rows) while still being a valid source-frame exclusion.
    unknown_exclusions = sorted(set(unknown_exclusions) - set(args.allow_missing_frame))
    if unknown_exclusions:
        raise SystemExit(f"excluded frames absent from input: {unknown_exclusions}")
    tracks_for_sampling = tracks[~tracks["frame_idx"].isin(excluded_source_frames)].copy()
    config = ValidationConfig(
        fov_degrees=tuple(args.fov_deg),
        range_m=args.range_m,
        near_blind_m=args.near_blind_m,
        sensor_height_m=args.sensor_height_m,
        ego_clearance_m=args.ego_clearance_m,
        angular_resolution_deg=args.angular_resolution_deg,
        occlusion_clearance_m=args.occlusion_clearance_m,
        visible_fraction_threshold=args.visible_fraction_threshold,
        sparse_min_visible_rays=args.sparse_min_visible_rays,
        ray_grids=tuple(sorted(set(args.ray_grids))),
        z_modes=tuple(args.z_modes),
    )
    selected = balanced_stratified_frames(tracks_for_sampling, args.n_frames, seed=args.seed)
    subset = tracks_for_sampling[tracks_for_sampling["frame_idx"].isin(selected)]
    frames = [group.copy() for _, group in subset.groupby("frame_idx", sort=True)]
    egos = default_ego_grid(args.n_ego_per_approach)
    payloads = [(frame, egos, plane, config) for frame in frames]

    if args.n_procs > 1:
        with ProcessPoolExecutor(max_workers=args.n_procs) as executor:
            results = list(executor.map(_run, payloads, chunksize=1))
    else:
        results = [_run(payload) for payload in payloads]
    decision_rows = [row for result in results for row in result[0]]
    exclusion_rows = [row for result in results for row in result[1]]
    timing_rows = [row for result in results for row in result[2]]
    decisions = pd.DataFrame(decision_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    timings = pd.DataFrame(timing_rows)
    summary = summarize_validation(decisions, timings)
    summary.update(
        {
            "schema_version": "1.0",
            "tracks_path": str(args.tracks.resolve()),
            "sampled_frames": selected.astype(int).tolist(),
            "sampled_frame_count": int(len(selected)),
            "excluded_source_frames": excluded_source_frames,
            "ego_positions": len(egos),
            "excluded_ego_scenes": int(len(exclusions)),
            "config": config_as_dict(config),
            "input_audit": audit,
            "reference_scope": (
                "analytic ray--oriented-box reference within the detected dynamic-box scene; "
                "not physical visibility ground truth"
            ),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decisions.sort_values(["frame_idx", "ego_id", "z_mode", "fov_deg", "target_id"]).to_parquet(
        args.out_dir / "validation_decisions.parquet", index=False
    )
    exclusions.to_parquet(args.out_dir / "ego_exclusions.parquet", index=False)
    timings.to_parquet(args.out_dir / "validation_timings.parquet", index=False)
    (args.out_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "validation_config.json").write_text(
        json.dumps(config_as_dict(config), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary["groups"], indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
