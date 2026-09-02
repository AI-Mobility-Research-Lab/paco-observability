#!/usr/bin/env python3
"""Rank detected dynamic-box residual-demand hotspots over ego-grid poses.

The primary scope is fixed to ground-anchored boxes and the sparse 15-ray
method. For each FOV, wide-table residual and total demand contributions are
first summed into non-overlapping intervals of the original ``frame_idx``
axis. Those time blocks, not ego-frame rows, are then resampled as clusters.

Two demand modes are reported separately. The primary ``uniform`` score is
``sum(total_count - sparse_multiray_visible_count) / sum(total_count)``. The
``vru_weighted`` sensitivity score is ``sum(sparse_multiray_residual_demand) /
sum(weighted_total_demand)``. Both are hotspot measures for residual demand
among detected dynamic boxes. Neither is a crash-risk estimate nor evidence
that an ego-grid coordinate is a deployment-optimal infrastructure pole.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from paco_observability.optimization import bootstrap_hotspot_rank_stability


PRIMARY_Z_MODE = "ground_anchored"
PRIMARY_METHOD = "sparse_multiray"
UNIFORM_NUMERATOR_SOURCE = "total_count - sparse_multiray_visible_count"
UNIFORM_DENOMINATOR_COLUMN = "total_count"
WEIGHTED_NUMERATOR_COLUMN = "sparse_multiray_residual_demand"
WEIGHTED_DENOMINATOR_COLUMN = "weighted_total_demand"
DEMAND_MODES = ("uniform", "vru_weighted")
DEFAULT_FOVS = (120.0, 360.0)
DEFAULT_BLOCK_WIDTH_FRAMES = 1_200
DEFAULT_N_BOOTSTRAP = 5_000
DEFAULT_TOP_K = 10
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SEED = 20260902
DEFAULT_EXPECTED_EGO_COUNT = 60
DEFAULT_SUMMARY_NAME = "residual_hotspot_ranks.json"
DEFAULT_RANKING_NAME = "residual_hotspot_ranks.parquet"

REQUIRED_COLUMNS = (
    "frame_idx",
    "ego_id",
    "approach",
    "ego_x",
    "ego_y",
    "z_mode",
    "fov_deg",
    "total_count",
    "sparse_multiray_visible_count",
    WEIGHTED_NUMERATOR_COLUMN,
    WEIGHTED_DENOMINATOR_COLUMN,
)

INTERPRETATION_GUARDRAIL = (
    "Detected dynamic-box residual-demand hotspot only; this is not crash risk and an "
    "ego-grid coordinate is not a deployment-optimal pole or surveyed site."
)


def _positive_integer(name: str, value: int) -> int:
    """Return a validated positive integer."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(value)


def _integer(name: str, value: int) -> int:
    """Return a validated integer."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _finite_numeric(data: pd.DataFrame, column: str) -> pd.Series:
    """Convert one column to finite float values."""

    try:
        values = pd.to_numeric(data[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} must contain only numeric values") from exc
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError(f"column {column!r} must contain only finite values")
    return values


def _validate_fovs(fov_degrees: Sequence[float]) -> tuple[float, ...]:
    """Return unique finite requested FOV values."""

    try:
        count = len(fov_degrees)
    except TypeError as exc:
        raise ValueError("fov_degrees must be a sequence") from exc
    if count == 0:
        raise ValueError("at least one FOV is required")
    result: list[float] = []
    for raw_value in fov_degrees:
        if isinstance(raw_value, (bool, np.bool_)):
            raise ValueError("FOV values must be finite positive numbers")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("FOV values must be finite positive numbers") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("FOV values must be finite positive numbers")
        if any(math.isclose(value, prior, rel_tol=0.0, abs_tol=1e-9) for prior in result):
            raise ValueError(f"FOV values must be unique; duplicate={value}")
        result.append(value)
    return tuple(result)


def _primary_scope(
    data: pd.DataFrame,
    *,
    fov_degrees: Sequence[float],
    expected_ego_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, ...]]:
    """Validate and select primary wide-table rows plus static ego metadata."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("input table is empty")
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"input table is missing required columns: {missing}")
    requested_fovs = _validate_fovs(fov_degrees)

    work = data.loc[:, list(REQUIRED_COLUMNS) + (["method"] if "method" in data else [])].copy()
    if work.loc[:, ["frame_idx", "ego_id", "approach", "ego_x", "ego_y"]].isna().any().any():
        raise ValueError("frame and ego metadata columns must not contain missing values")
    frame_values = _finite_numeric(work, "frame_idx")
    rounded_frames = np.rint(frame_values.to_numpy(dtype=float))
    if not np.array_equal(frame_values.to_numpy(dtype=float), rounded_frames):
        raise ValueError("frame_idx values must be integers")
    work["frame_idx"] = rounded_frames.astype(np.int64)
    work["fov_deg"] = _finite_numeric(work, "fov_deg")
    work["ego_x"] = _finite_numeric(work, "ego_x")
    work["ego_y"] = _finite_numeric(work, "ego_y")
    work["total_count"] = _finite_numeric(work, "total_count")
    work["sparse_multiray_visible_count"] = _finite_numeric(
        work, "sparse_multiray_visible_count"
    )
    work[WEIGHTED_NUMERATOR_COLUMN] = _finite_numeric(work, WEIGHTED_NUMERATOR_COLUMN)
    work[WEIGHTED_DENOMINATOR_COLUMN] = _finite_numeric(work, WEIGHTED_DENOMINATOR_COLUMN)
    work["ego_id"] = work["ego_id"].astype(str)
    work["approach"] = work["approach"].astype(str)
    work["z_mode"] = work["z_mode"].astype(str)

    total_count = work["total_count"].to_numpy(dtype=float)
    visible_count = work["sparse_multiray_visible_count"].to_numpy(dtype=float)
    count_tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(total_count)) * 16.0
    if np.any(total_count < 0.0) or np.any(visible_count < 0.0):
        raise ValueError("total and sparse_multiray visible counts must be nonnegative")
    if np.any(visible_count > total_count + count_tolerance):
        raise ValueError("sparse_multiray visible count must not exceed total_count")
    if not np.array_equal(total_count, np.rint(total_count)) or not np.array_equal(
        visible_count, np.rint(visible_count)
    ):
        raise ValueError("total and sparse_multiray visible counts must be integers")

    weighted_numerator = work[WEIGHTED_NUMERATOR_COLUMN].to_numpy(dtype=float)
    weighted_denominator = work[WEIGHTED_DENOMINATOR_COLUMN].to_numpy(dtype=float)
    weighted_tolerance = (
        np.finfo(float).eps * np.maximum(1.0, np.abs(weighted_denominator)) * 16.0
    )
    if np.any(weighted_numerator < 0.0) or np.any(weighted_denominator < 0.0):
        raise ValueError("weighted residual and total demand contributions must be nonnegative")
    if np.any(weighted_numerator > weighted_denominator + weighted_tolerance):
        raise ValueError("sparse_multiray residual demand must not exceed weighted total demand")

    fov_mask = np.zeros(len(work), dtype=bool)
    for fov in requested_fovs:
        fov_mask |= np.isclose(
            work["fov_deg"].to_numpy(dtype=float), fov, rtol=0.0, atol=1e-9
        )
    scope_mask = (work["z_mode"] == PRIMARY_Z_MODE).to_numpy() & fov_mask
    if "method" in work:
        scope_mask &= (work["method"].astype(str) == PRIMARY_METHOD).to_numpy()
    scope = work.loc[scope_mask].copy()
    if scope.empty:
        raise ValueError(
            f"no rows match z_mode={PRIMARY_Z_MODE!r}, method={PRIMARY_METHOD!r}, "
            f"FOVs={requested_fovs}"
        )
    for fov in requested_fovs:
        if not np.any(np.isclose(scope["fov_deg"], fov, rtol=0.0, atol=1e-9)):
            raise ValueError(f"primary scope is missing requested FOV {fov:g}")

    key_columns = ["frame_idx", "ego_id", "fov_deg"]
    if bool(scope.duplicated(key_columns, keep=False).any()):
        examples = scope.loc[scope.duplicated(key_columns, keep=False), key_columns].head()
        raise ValueError(
            "wide primary rows must be unique by frame_idx/ego_id/fov_deg; "
            f"examples={examples.to_dict(orient='records')}"
        )

    metadata_columns = ["ego_id", "approach", "ego_x", "ego_y"]
    for column in metadata_columns[1:]:
        counts = scope.groupby("ego_id", sort=True, observed=True)[column].nunique(dropna=False)
        inconsistent = counts[counts != 1]
        if not inconsistent.empty:
            raise ValueError(
                f"each ego_id must map to one {column}; bad_ego_ids={list(inconsistent.index)}"
            )
    metadata = (
        scope.loc[:, metadata_columns]
        .drop_duplicates("ego_id")
        .sort_values("ego_id", kind="stable")
        .reset_index(drop=True)
    )
    ego_ids = tuple(metadata["ego_id"])
    for fov in requested_fovs:
        fov_scope = scope[np.isclose(scope["fov_deg"], fov, rtol=0.0, atol=1e-9)]
        if set(fov_scope["ego_id"]) != set(ego_ids):
            raise ValueError(f"FOV {fov:g} does not contain the common ego_id set")
    if expected_ego_count is not None:
        expected = _positive_integer("expected_ego_count", expected_ego_count)
        if len(ego_ids) != expected:
            raise ValueError(f"expected {expected} ego_id values, found {len(ego_ids)}")
    return scope, metadata, requested_fovs


def aggregate_residual_demand_blocks(
    data: pd.DataFrame,
    *,
    fov_degrees: Sequence[float] = DEFAULT_FOVS,
    block_width_frames: int = DEFAULT_BLOCK_WIDTH_FRAMES,
    block_origin_frame: int | None = None,
    expected_ego_count: int | None = DEFAULT_EXPECTED_EGO_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate primary residual-demand contributions by raw-frame interval.

    ``time_block_id = floor((frame_idx - block_origin_frame) /
    block_width_frames)``. The default origin is the first selected primary
    frame. Consequently a block spans a fixed width on the original
    frame-number axis even when some frames are absent from the input.

    Returns:
        ``(block_contributions, ego_metadata)``. Contributions contain one row
        per observed ``FOV × block × ego_id``; missing block/ego combinations
        are filled with zero only when constructing bootstrap matrices.
    """

    width = _positive_integer("block_width_frames", block_width_frames)
    scope, metadata, _ = _primary_scope(
        data,
        fov_degrees=fov_degrees,
        expected_ego_count=expected_ego_count,
    )
    origin = (
        int(scope["frame_idx"].min())
        if block_origin_frame is None
        else _integer("block_origin_frame", block_origin_frame)
    )
    uniform = scope.copy()
    uniform["demand_mode"] = "uniform"
    uniform["residual_demand"] = (
        uniform["total_count"] - uniform["sparse_multiray_visible_count"]
    )
    uniform["total_demand"] = uniform["total_count"]
    weighted = scope.copy()
    weighted["demand_mode"] = "vru_weighted"
    weighted["residual_demand"] = weighted[WEIGHTED_NUMERATOR_COLUMN]
    weighted["total_demand"] = weighted[WEIGHTED_DENOMINATOR_COLUMN]
    contributions = pd.concat((uniform, weighted), ignore_index=True)
    contributions["time_block_id"] = np.floor_divide(
        contributions["frame_idx"].to_numpy(dtype=np.int64) - origin, width
    ).astype(np.int64)
    contributions["time_block_start_frame"] = origin + contributions["time_block_id"] * width
    contributions["time_block_end_frame"] = (
        contributions["time_block_start_frame"] + width - 1
    )
    grouping = [
        "demand_mode",
        "fov_deg",
        "time_block_id",
        "time_block_start_frame",
        "time_block_end_frame",
        "ego_id",
    ]
    blocks = (
        contributions.groupby(grouping, as_index=False, sort=True, observed=True)
        .agg(
            residual_demand=("residual_demand", "sum"),
            total_demand=("total_demand", "sum"),
            observed_frame_count=("frame_idx", "nunique"),
            first_observed_frame=("frame_idx", "min"),
            last_observed_frame=("frame_idx", "max"),
        )
        .sort_values(
            ["demand_mode", "fov_deg", "time_block_id", "ego_id"], kind="stable"
        )
        .reset_index(drop=True)
    )
    return blocks, metadata


def _point_ranks(point_order: tuple[int, ...], n_hotspots: int) -> np.ndarray:
    """Convert an ordered index tuple to one-based integer ranks."""

    ranks = np.empty(n_hotspots, dtype=np.int64)
    for rank, index in enumerate(point_order, start=1):
        ranks[index] = rank
    return ranks


def rank_residual_demand(
    data: pd.DataFrame,
    *,
    fov_degrees: Sequence[float] = DEFAULT_FOVS,
    block_width_frames: int = DEFAULT_BLOCK_WIDTH_FRAMES,
    block_origin_frame: int | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    top_k: int = DEFAULT_TOP_K,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
    expected_ego_count: int | None = DEFAULT_EXPECTED_EGO_COUNT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return JSON-ready metadata and per-FOV residual-hotspot ranks."""

    width = _positive_integer("block_width_frames", block_width_frames)
    requested_origin = (
        None
        if block_origin_frame is None
        else _integer("block_origin_frame", block_origin_frame)
    )
    resamples = _positive_integer("n_bootstrap", n_bootstrap)
    normalized_top_k = _positive_integer("top_k", top_k)
    normalized_seed = _integer("seed", seed)
    if normalized_seed < 0:
        raise ValueError("seed must be nonnegative")
    try:
        confidence = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be strictly between 0 and 1") from exc
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    blocks, metadata = aggregate_residual_demand_blocks(
        data,
        fov_degrees=fov_degrees,
        block_width_frames=width,
        block_origin_frame=requested_origin,
        expected_ego_count=expected_ego_count,
    )
    first_block = blocks.iloc[0]
    origin = int(
        first_block["time_block_start_frame"]
        - first_block["time_block_id"] * width
    )
    requested_fovs = _validate_fovs(fov_degrees)
    ego_ids = tuple(metadata["ego_id"].astype(str))
    if normalized_top_k > len(ego_ids):
        raise ValueError(f"top_k={normalized_top_k} exceeds {len(ego_ids)} ego positions")
    metadata_by_ego = metadata.set_index("ego_id")

    ranking_frames: list[pd.DataFrame] = []
    fov_summaries: list[dict[str, Any]] = []
    for demand_mode in DEMAND_MODES:
        for fov in requested_fovs:
            selected = blocks[
                (blocks["demand_mode"] == demand_mode)
                & np.isclose(blocks["fov_deg"], fov, rtol=0.0, atol=1e-9)
            ]
            if selected.empty:
                raise RuntimeError(f"no block contributions for {demand_mode}, FOV {fov:g}")
            block_ids = np.sort(selected["time_block_id"].unique())
            numerator = (
                selected.pivot(
                    index="time_block_id", columns="ego_id", values="residual_demand"
                )
                .reindex(index=block_ids, columns=ego_ids, fill_value=0.0)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            denominator = (
                selected.pivot(index="time_block_id", columns="ego_id", values="total_demand")
                .reindex(index=block_ids, columns=ego_ids, fill_value=0.0)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            rank = bootstrap_hotspot_rank_stability(
                numerator,
                denominator,
                hotspot_ids=ego_ids,
                n_resamples=resamples,
                top_k=normalized_top_k,
                confidence_level=confidence,
                higher_is_hotter=True,
                seed=normalized_seed,
            )
            point_ranks = _point_ranks(rank.point_order, len(ego_ids))
            numerator_total = np.sum(numerator, axis=0, dtype=np.float64)
            denominator_total = np.sum(denominator, axis=0, dtype=np.float64)
            rows: list[dict[str, Any]] = []
            for index, ego_id in enumerate(ego_ids):
                ego = metadata_by_ego.loc[ego_id]
                rows.append(
                    {
                        "demand_mode": demand_mode,
                        "analysis_role": (
                            "primary" if demand_mode == "uniform" else "sensitivity"
                        ),
                        "fov_deg": fov,
                        "z_mode": PRIMARY_Z_MODE,
                        "method": PRIMARY_METHOD,
                        "ego_id": ego_id,
                        "approach": str(ego["approach"]),
                        "ego_x": float(ego["ego_x"]),
                        "ego_y": float(ego["ego_y"]),
                        "point_rank": int(point_ranks[index]),
                        "point_estimate": float(rank.point_scores[index]),
                        "residual_demand_sum": float(numerator_total[index]),
                        "total_demand_sum": float(denominator_total[index]),
                        "score_ci_low": float(rank.score_ci_low[index]),
                        "score_ci_high": float(rank.score_ci_high[index]),
                        "median_rank": float(rank.median_rank[index]),
                        "rank_ci_low": float(rank.rank_ci_low[index]),
                        "rank_ci_high": float(rank.rank_ci_high[index]),
                        "top_k_probability": float(rank.top_k_probability[index]),
                        "valid_fraction": float(rank.valid_fraction[index]),
                        "time_block_count": int(len(block_ids)),
                        "block_width_frames": width,
                        "top_k": normalized_top_k,
                        "n_bootstrap": resamples,
                        "seed": normalized_seed,
                    }
                )
            ranking = (
                pd.DataFrame(rows)
                .sort_values("point_rank", kind="stable")
                .reset_index(drop=True)
            )
            ranking_frames.append(ranking)

            block_metadata = (
                selected.groupby(
                    ["time_block_id", "time_block_start_frame", "time_block_end_frame"],
                    as_index=False,
                    sort=True,
                    observed=True,
                )
                .agg(
                    first_observed_frame=("first_observed_frame", "min"),
                    last_observed_frame=("last_observed_frame", "max"),
                    maximum_observed_frame_count=("observed_frame_count", "max"),
                )
            )
            fov_summaries.append(
                {
                    "demand_mode": demand_mode,
                    "analysis_role": (
                        "primary" if demand_mode == "uniform" else "sensitivity"
                    ),
                    "fov_deg": fov,
                    "n_ego_positions": len(ego_ids),
                    "n_time_blocks": int(len(block_ids)),
                    "time_blocks": [
                        {key: int(value) for key, value in record.items()}
                        for record in block_metadata.to_dict(orient="records")
                    ],
                    "ranking": ranking.to_dict(orient="records"),
                }
            )

    rankings = pd.concat(ranking_frames, ignore_index=True)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis": "residual_demand_hotspot_rank_stability",
        "scope": {
            "z_mode": PRIMARY_Z_MODE,
            "method": PRIMARY_METHOD,
            "fov_degrees": list(requested_fovs),
            "demand_universe": "quality-gated detected dynamic boxes",
            "primary_demand_mode": "uniform",
            "sensitivity_demand_mode": "vru_weighted",
        },
        "score": {
            "name": "residual_demand_ratio_of_sums",
            "formulas": {
                "uniform": (
                    "sum(total_count - sparse_multiray_visible_count) / sum(total_count)"
                ),
                "vru_weighted": (
                    "sum(sparse_multiray_residual_demand) / sum(weighted_total_demand)"
                ),
            },
            "not_used": "unweighted mean of ego-frame or time-block ratios",
            "higher_is_hotter": True,
        },
        "bootstrap": {
            "helper": "optimization.bootstrap_hotspot_rank_stability",
            "cluster": "non-overlapping intervals on original frame_idx axis",
            "block_width_frames": width,
            "block_origin_frame": origin,
            "n_resamples": resamples,
            "confidence_level": confidence,
            "interval": "two-sided equal-tail percentile",
            "top_k": normalized_top_k,
            "seed": normalized_seed,
        },
        "input_row_count": int(len(data)),
        "aggregated_block_contribution_row_count": int(len(blocks)),
        "expected_ego_count": (
            None if expected_ego_count is None else int(expected_ego_count)
        ),
        "fov_results": fov_summaries,
        "interpretation_guardrail": INTERPRETATION_GUARDRAIL,
    }
    return summary, rankings


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fov-deg", type=float, nargs="+", default=DEFAULT_FOVS)
    parser.add_argument("--block-width-frames", type=int, default=DEFAULT_BLOCK_WIDTH_FRAMES)
    parser.add_argument(
        "--block-origin-frame",
        type=int,
        help="Block origin on frame_idx axis (default: first selected primary frame)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-ego-count", type=int, default=DEFAULT_EXPECTED_EGO_COUNT)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--ranking-name", default=DEFAULT_RANKING_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Read the wide full-record table and write JSON plus Parquet ranks."""

    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"input Parquet does not exist: {args.input}")
    if Path(args.summary_name).name != args.summary_name:
        raise SystemExit("--summary-name must be a file name, not a path")
    if Path(args.ranking_name).name != args.ranking_name:
        raise SystemExit("--ranking-name must be a file name, not a path")
    data = pd.read_parquet(args.input)
    summary, rankings = rank_residual_demand(
        data,
        fov_degrees=args.fov_deg,
        block_width_frames=args.block_width_frames,
        block_origin_frame=args.block_origin_frame,
        n_bootstrap=args.n_bootstrap,
        top_k=args.top_k,
        confidence_level=args.confidence_level,
        seed=args.seed,
        expected_ego_count=args.expected_ego_count,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / args.summary_name
    ranking_path = args.out_dir / args.ranking_name
    rankings.to_parquet(ranking_path, index=False)
    summary["input_parquet"] = str(args.input.resolve())
    summary["outputs"] = {
        "summary_json": str(summary_path.resolve()),
        "ranking_parquet": str(ranking_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")
    print(f"wrote {ranking_path}")


if __name__ == "__main__":
    main()
