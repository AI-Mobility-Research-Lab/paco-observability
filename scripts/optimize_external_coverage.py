#!/usr/bin/env python3
"""Illustrative finite-viewpoint coverage of residual onboard demand.

The default coordinates are computational stress-test viewpoints, not surveyed
or deployment-ready pole locations.  Results therefore support prioritization
logic, not a site-design recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paco_observability.geometry import GroundPlane, OrientedBox
from paco_observability.occlusion import center_coverage_flags, projected_visible_fraction
from paco_observability.optimization import (
    bootstrap_hotspot_rank_stability,
    greedy_weighted_maximum_coverage,
)
from paco_observability.provenance import audit_tracks
from paco_observability.validation import visible_at_fraction


@dataclass(frozen=True)
class CandidateViewpoint:
    candidate_id: str
    x: float
    y: float
    height_m: float = 3.0
    range_m: float = 35.0


DEFAULT_CANDIDATES = (
    CandidateViewpoint("north", 7.0, 40.0),
    CandidateViewpoint("south", 8.0, -20.0),
    CandidateViewpoint("east", 40.0, 8.0),
    CandidateViewpoint("west", -25.0, 8.0),
    CandidateViewpoint("northwest", -5.0, 25.0),
    CandidateViewpoint("northeast", 20.0, 25.0),
    CandidateViewpoint("southeast", 20.0, -5.0),
    CandidateViewpoint("southwest", -5.0, -5.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--z-mode", default="ground_anchored")
    parser.add_argument("--onboard-fov-deg", type=float, default=120.0)
    parser.add_argument("--candidate-grid", type=int, default=9)
    parser.add_argument("--visible-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--max-budget", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-block-frames", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--expected-frame-range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--allow-missing-frame", type=int, action="append", default=[])
    parser.add_argument(
        "--class-weights-json",
        default='{"car":1.0,"truck":1.0,"pedestrian":3.0,"bicycle":3.0}',
    )
    parser.add_argument(
        "--candidates-json",
        type=Path,
        help="Optional list of {candidate_id,x,y,height_m,range_m}; otherwise illustrative defaults",
    )
    parser.add_argument(
        "--ground-plane", type=float, nargs=4, metavar=("NX", "NY", "NZ", "D"), required=True
    )
    return parser.parse_args()


def load_candidates(path: Path | None) -> tuple[CandidateViewpoint, ...]:
    if path is None:
        return DEFAULT_CANDIDATES
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise ValueError("candidates JSON must be a non-empty list")
    result = tuple(CandidateViewpoint(**item) for item in values)
    ids = [candidate.candidate_id for candidate in result]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be unique")
    return result


def main() -> None:
    args = parse_args()
    decisions = pd.read_parquet(args.decisions)
    tracks = pd.read_parquet(args.tracks)
    required = {
        "frame_idx",
        "ego_id",
        "target_id",
        "class_name",
        "z_mode",
        "fov_deg",
        "exact_visible",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"decisions table is missing columns: {missing}")
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
        raise SystemExit(f"tracks failed quality gate: {audit['quality_gate']}")
    plane = GroundPlane.from_coefficients(args.ground_plane)
    candidates = load_candidates(args.candidates_json)
    class_weights = {str(key): float(value) for key, value in json.loads(args.class_weights_json).items()}
    if any(not math.isfinite(value) or value < 0 for value in class_weights.values()):
        raise ValueError("class weights must be finite and nonnegative")

    scope = decisions[
        (decisions["z_mode"] == args.z_mode)
        & np.isclose(decisions["fov_deg"].astype(float), args.onboard_fov_deg)
    ].copy()
    if scope.empty:
        raise ValueError("no validation decisions match the requested z_mode/FOV")
    key_columns = ["frame_idx", "ego_id", "target_id"]
    if scope.duplicated(key_columns).any():
        raise ValueError("validation decisions are not unique by frame/ego/target")
    demand = scope[~scope["exact_visible"].astype(bool)].copy().reset_index(drop=True)
    if demand.empty:
        raise ValueError("the selected scope has no residual demand elements")
    demand["weight"] = demand["class_name"].map(class_weights).fillna(1.0).astype(float)
    if float(demand["weight"].sum()) <= 0:
        raise ValueError("residual demand has zero total weight")

    selected_frames = np.sort(demand["frame_idx"].astype(int).unique())
    tracks = tracks[tracks["frame_idx"].isin(selected_frames)].copy()
    by_frame = {int(frame): group for frame, group in tracks.groupby("frame_idx", sort=True)}
    visibility: dict[tuple[str, int, Any], bool] = {}
    visibility_rows: list[dict[str, Any]] = []
    missing_targets = 0
    for frame_idx in selected_frames:
        frame = by_frame.get(int(frame_idx))
        if frame is None:
            continue
        boxes = [
            OrientedBox.from_row(row, z_mode=args.z_mode, ground_plane=plane)
            for row in frame.to_dict("records")
        ]
        by_id = {box.object_id: box for box in boxes}
        blockers = [box for box in boxes if box.class_name in {"car", "truck"}]
        target_ids = demand.loc[demand["frame_idx"] == frame_idx, "target_id"].unique()
        for candidate in candidates:
            origin = np.asarray(
                (
                    candidate.x,
                    candidate.y,
                    float(plane.height_at(candidate.x, candidate.y)) + candidate.height_m,
                ),
                dtype=float,
            )
            for target_id in target_ids:
                target = by_id.get(target_id)
                if target is None:
                    missing_targets += 1
                    visible = False
                    fraction = 0.0
                    in_range = False
                else:
                    flags = center_coverage_flags(
                        origin,
                        target,
                        heading_rad=0.0,
                        fov_rad=2 * math.pi,
                        near_range_m=0.0,
                        far_range_m=candidate.range_m,
                    )
                    in_range = flags.in_range
                    if in_range:
                        result = projected_visible_fraction(
                            origin,
                            target,
                            blockers,
                            azimuth_samples=args.candidate_grid,
                            elevation_samples=args.candidate_grid,
                        )
                        fraction = result.score
                        visible = bool(
                            visible_at_fraction(fraction, args.visible_fraction_threshold)
                        )
                    else:
                        fraction = 0.0
                        visible = False
                visibility[(candidate.candidate_id, int(frame_idx), target_id)] = visible
                visibility_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "frame_idx": int(frame_idx),
                        "target_id": target_id,
                        "in_range": in_range,
                        "visible_fraction": fraction,
                        "visible": visible,
                    }
                )

    coverage = np.zeros((len(candidates), len(demand)), dtype=bool)
    for candidate_index, candidate in enumerate(candidates):
        coverage[candidate_index] = [
            visibility.get((candidate.candidate_id, int(row.frame_idx), row.target_id), False)
            for row in demand.itertuples(index=False)
        ]
        demand[f"covered_by_{candidate.candidate_id}"] = coverage[candidate_index]

    result = greedy_weighted_maximum_coverage(
        coverage,
        weights=demand["weight"].to_numpy(float),
        budget=min(args.max_budget, len(candidates)),
        candidate_ids=[candidate.candidate_id for candidate in candidates],
    )

    # Aggregate dependent observations into contiguous time blocks before the
    # ordinary cluster bootstrap used for candidate-rank stability.
    minimum_frame = int(demand["frame_idx"].min())
    demand["time_block"] = (
        (demand["frame_idx"].astype(int) - minimum_frame) // args.bootstrap_block_frames
    ).astype(int)
    blocks = np.sort(demand["time_block"].unique())
    numerator = np.zeros((len(blocks), len(candidates)), dtype=float)
    denominator = np.zeros_like(numerator)
    for block_index, block in enumerate(blocks):
        mask = demand["time_block"].to_numpy(int) == int(block)
        weights = demand.loc[mask, "weight"].to_numpy(float)
        denominator[block_index, :] = float(weights.sum())
        numerator[block_index, :] = coverage[:, mask] @ weights
    rank = bootstrap_hotspot_rank_stability(
        numerator,
        denominator,
        hotspot_ids=[candidate.candidate_id for candidate in candidates],
        n_resamples=args.n_bootstrap,
        top_k=min(3, len(candidates)),
        seed=args.seed,
    )

    summary = {
        "schema_version": "1.0",
        "scope": {
            "z_mode": args.z_mode,
            "onboard_fov_deg": args.onboard_fov_deg,
            "onboard_visibility_source": "exact within-OBB validation reference",
            "residual_demand_elements": int(len(demand)),
            "unique_frames": int(demand["frame_idx"].nunique()),
            "unique_ego_poses": int(demand["ego_id"].nunique()),
            "weighted_demand": float(demand["weight"].sum()),
        },
        "candidate_model": {
            "candidates": [asdict(candidate) for candidate in candidates],
            "ray_grid": [args.candidate_grid, args.candidate_grid],
            "visible_fraction_threshold": args.visible_fraction_threshold,
            "class_weights": class_weights,
            "static_occluders_included": False,
        },
        "greedy_coverage": {
            "selected_ids": list(result.selected_ids),
            "coverage_fraction": result.coverage_fraction,
            "covered_weight": result.covered_weight,
            "total_weight": result.total_weight,
            "trajectory": [
                {
                    "step": step.step,
                    "candidate_index": step.candidate_index,
                    "candidate_id": step.candidate_id,
                    "marginal_gain": step.marginal_gain,
                    "cumulative_covered_weight": step.cumulative_covered_weight,
                    "coverage_fraction": step.coverage_fraction,
                    "newly_covered_count": step.newly_covered_count,
                }
                for step in result.trajectory
            ],
        },
        "rank_stability": rank.records(),
        "bootstrap": {
            "clusters": int(len(blocks)),
            "block_frames": args.bootstrap_block_frames,
            "n_resamples": args.n_bootstrap,
            "seed": args.seed,
        },
        "quality": {"missing_target_lookups": int(missing_targets), "input_audit": audit},
        "interpretation_guardrail": (
            "Illustrative finite viewpoints only; coordinates are not surveyed, static scene "
            "occluders and communication reliability are absent, and results are not a deployment design."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(args.out_dir / "residual_demand_coverage.parquet", index=False)
    pd.DataFrame(visibility_rows).to_parquet(
        args.out_dir / "candidate_target_visibility.parquet", index=False
    )
    pd.DataFrame(
        rank.bootstrap_scores, columns=[candidate.candidate_id for candidate in candidates]
    ).to_parquet(args.out_dir / "candidate_rank_bootstrap_scores.parquet", index=False)
    (args.out_dir / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps(summary["greedy_coverage"], indent=2, default=str))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
