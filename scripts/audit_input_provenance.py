#!/usr/bin/env python3
"""Audit a tracked-box input before paper-facing observability analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from paco_observability.provenance import (
    audit_frame_index,
    audit_tracks,
    build_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path)
    parser.add_argument("--frame-index", type=Path)
    parser.add_argument(
        "--expected-frame-range", type=int, nargs=2, metavar=("START", "END")
    )
    parser.add_argument(
        "--exclude-partial-frame",
        type=int,
        action="append",
        default=[],
        help="Declare a low-point partial source frame excluded from all analyses",
    )
    parser.add_argument("--partial-neighbor-radius", type=int, default=3)
    parser.add_argument("--partial-point-ratio", type=float, default=0.6)
    parser.add_argument(
        "--allow-missing-frame",
        type=int,
        action="append",
        default=[],
        help="Explicitly allow a documented source frame with no tracked objects",
    )
    parser.add_argument(
        "--ground-plane",
        type=float,
        nargs=4,
        metavar=("NX", "NY", "NZ", "D"),
        help="Plane coefficients for n dot p + d = 0",
    )
    parser.add_argument(
        "--allow-quality-fail",
        action="store_true",
        help="Write diagnostics even if the fail-closed gate detects fatal defects",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks = pd.read_parquet(args.tracks)
    plane = args.ground_plane
    audit = audit_tracks(
        tracks,
        ground_normal=tuple(plane[:3]) if plane else None,
        ground_offset=plane[3] if plane else None,
        expected_frame_range=tuple(args.expected_frame_range)
        if args.expected_frame_range
        else None,
        allowed_missing_frames=args.allow_missing_frame,
    )
    frame_audit = None
    if args.frame_index:
        frame_audit = audit_frame_index(
            pd.read_parquet(args.frame_index),
            expected_frame_range=tuple(args.expected_frame_range)
            if args.expected_frame_range
            else None,
            excluded_partial_frames=args.exclude_partial_frame,
            neighbor_radius=args.partial_neighbor_radius,
            point_ratio_threshold=args.partial_point_ratio,
        )
    manifest = build_run_manifest(
        tracks_path=args.tracks,
        analysis_repository=Path(__file__).resolve().parents[1],
        parameters={
            "ground_plane": plane,
            "expected_frame_range": args.expected_frame_range,
            "allowed_missing_frames": args.allow_missing_frame,
            "excluded_partial_frames": args.exclude_partial_frame,
            "partial_neighbor_radius": args.partial_neighbor_radius,
            "partial_point_ratio": args.partial_point_ratio,
        },
        input_audit=audit,
        upstream_manifest=args.upstream_manifest,
        frame_index_path=args.frame_index,
        frame_index_audit=frame_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    gates = {"tracks": audit["quality_gate"]}
    if frame_audit is not None:
        gates["frame_index"] = frame_audit["quality_gate"]
    print(json.dumps(gates, indent=2))
    print(f"wrote {args.output}")
    if not audit["quality_gate"]["passed"] and not args.allow_quality_fail:
        raise SystemExit("input failed the paper-facing quality gate")
    if frame_audit is not None and not frame_audit["quality_gate"]["passed"]:
        if not args.allow_quality_fail:
            raise SystemExit("frame index failed the paper-facing quality gate")


if __name__ == "__main__":
    main()
