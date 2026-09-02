#!/usr/bin/env python3
"""Bootstrap full-record observability while preserving temporal dependence.

The input is a long-form Parquet table containing numerator and denominator
contributions for ego-frame observations. Contributions are first summed within
``frame_idx × z_mode × fov_deg × method``. The resulting frame totals, rather
than individual ego rows, are the moving-block bootstrap units.

The estimand is always a ratio of sums::

    sum(frame numerator) / sum(frame denominator)

It is deliberately not the unweighted mean of per-frame ratios. By default the
script uses a circular block of 1,200 consecutive retained-frame observations
(nominally about 120 seconds at 10 Hz), 5,000 replicates, and a fixed seed.
Declared missing and quality-excluded frames can make the covered source-index
span slightly longer than that nominal duration. The paired FOV contrast is
``360 - 120`` and uses identical frame-aligned blocks for both FOVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence
import uuid

import numpy as np
import pandas as pd

from paco_observability.bootstrap import (
    moving_block_bootstrap_ratio,
    paired_moving_block_bootstrap_ratio,
)
from paco_observability.cacie_source_contract import (
    attach_source_contract,
    require_cacie_full_record_source_unchanged,
    require_effective_frame_rate,
    snapshot_cacie_full_record_source,
    validate_cacie_full_record_source,
)


DEFAULT_FRAME_RATE_HZ = 10.0
DEFAULT_BLOCK_SECONDS = 120.0
DEFAULT_N_RESAMPLES = 5_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SEED = 20260902
DEFAULT_SUMMARY_NAME = "observability_bootstrap.json"
DEFAULT_REPLICATES_NAME = "observability_bootstrap_replicates.parquet"


def block_frame_count(frame_rate_hz: float, block_seconds: float) -> int:
    """Convert a physical block duration to the nearest positive frame count."""

    try:
        rate = float(frame_rate_hz)
        duration = float(block_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_rate_hz and block_seconds must be numbers") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("frame_rate_hz must be finite and strictly positive")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("block_seconds must be finite and strictly positive")
    frames = int(math.floor(rate * duration + 0.5))
    if frames < 1:
        raise ValueError("frame_rate_hz * block_seconds must round to at least one frame")
    return frames


def _validate_numeric_column(data: pd.DataFrame, column: str) -> pd.Series:
    """Return a finite float conversion of one required numeric column."""

    try:
        values = pd.to_numeric(data[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} must contain only numeric values") from exc
    if not np.all(np.isfinite(values.to_numpy(dtype=float))):
        raise ValueError(f"column {column!r} must contain only finite values")
    return values


def aggregate_frame_contributions(
    data: pd.DataFrame,
    *,
    frame_column: str = "frame_idx",
    z_mode_column: str = "z_mode",
    fov_column: str = "fov_deg",
    method_column: str = "method",
    numerator_column: str = "numerator",
    denominator_column: str = "denominator",
) -> pd.DataFrame:
    """Sum ego-row contributions within each analysis group and frame.

    Args:
        data: Long-form ego-frame contribution table.
        frame_column: Ordered temporal label used as the bootstrap cluster.
        z_mode_column: Vertical-coordinate assumption label.
        fov_column: Horizontal field of view in degrees.
        method_column: Observability method label.
        numerator_column: Visible/observable contribution.
        denominator_column: Eligible target contribution.

    Returns:
        One row per ``z_mode × fov × method × frame``, sorted in that order,
        with standardized ``numerator`` and ``denominator`` output names.

    Raises:
        ValueError: If required fields are missing, non-finite, negative, or
            inconsistent with an observability fraction.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("input table is empty")
    source_columns = (
        frame_column,
        z_mode_column,
        fov_column,
        method_column,
        numerator_column,
        denominator_column,
    )
    if len(set(source_columns)) != len(source_columns):
        raise ValueError("input column arguments must identify six distinct columns")
    missing = [column for column in source_columns if column not in data.columns]
    if missing:
        raise ValueError(f"input table is missing required columns: {missing}")

    work = data.loc[:, source_columns].copy()
    key_columns = (frame_column, z_mode_column, fov_column, method_column)
    missing_keys = work.loc[:, key_columns].isna().any(axis=1)
    if bool(missing_keys.any()):
        bad_rows = work.index[missing_keys].tolist()[:5]
        raise ValueError(f"group and frame columns must not be missing; rows={bad_rows}")

    work[fov_column] = _validate_numeric_column(work, fov_column)
    work[numerator_column] = _validate_numeric_column(work, numerator_column)
    work[denominator_column] = _validate_numeric_column(work, denominator_column)
    work[z_mode_column] = work[z_mode_column].astype(str)
    work[method_column] = work[method_column].astype(str)

    numerator = work[numerator_column].to_numpy(dtype=float)
    denominator = work[denominator_column].to_numpy(dtype=float)
    if np.any(numerator < 0.0):
        raise ValueError(f"column {numerator_column!r} must be nonnegative")
    if np.any(denominator < 0.0):
        raise ValueError(f"column {denominator_column!r} must be nonnegative")
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(denominator)) * 16.0
    if np.any(numerator > denominator + tolerance):
        raise ValueError(
            f"observability contributions require {numerator_column!r} <= "
            f"{denominator_column!r}"
        )

    group_columns = [z_mode_column, fov_column, method_column, frame_column]
    try:
        aggregated = (
            work.groupby(group_columns, as_index=False, sort=True, observed=True)
            .agg(
                numerator=(numerator_column, "sum"),
                denominator=(denominator_column, "sum"),
            )
            .sort_values(group_columns, kind="stable")
            .reset_index(drop=True)
        )
    except TypeError as exc:
        raise ValueError("frame labels must be mutually sortable within the input table") from exc

    totals = aggregated.groupby(
        [z_mode_column, fov_column, method_column], sort=True, observed=True
    )["denominator"].sum()
    nonpositive = totals[totals <= 0.0]
    if not nonpositive.empty:
        raise ValueError(
            "every z_mode/fov/method group must have positive denominator total; "
            f"bad_groups={list(nonpositive.index)}"
        )
    return aggregated


def _python_scalar(value: Any) -> Any:
    """Convert pandas/NumPy scalar labels to JSON-safe Python values."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def _fov_mask(values: pd.Series, target: float) -> np.ndarray:
    """Match a requested FOV without accepting materially different values."""

    return np.isclose(values.to_numpy(dtype=float), target, rtol=0.0, atol=1e-9)


def _single_replicates(
    samples: np.ndarray,
    *,
    z_mode: Any,
    fov_deg: float,
    method: Any,
    block_frames: int,
    seed: int,
) -> pd.DataFrame:
    """Create tidy replicate rows for one single-FOV estimate."""

    return pd.DataFrame(
        {
            "analysis_type": "single_ratio",
            "estimand": "ratio_of_sums",
            "delta_direction": None,
            "z_mode": str(z_mode),
            "method": str(method),
            "fov_deg": float(fov_deg),
            "fov_a_deg": np.nan,
            "fov_b_deg": np.nan,
            "replicate": np.arange(len(samples), dtype=np.int64),
            "ratio": np.asarray(samples, dtype=float),
            "ratio_a": np.nan,
            "ratio_b": np.nan,
            "delta_b_minus_a": np.nan,
            "block_frame_count": block_frames,
            "seed": seed,
        }
    )


def _paired_replicates(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    samples_delta: np.ndarray,
    *,
    z_mode: Any,
    method: Any,
    fov_a: float,
    fov_b: float,
    block_frames: int,
    seed: int,
) -> pd.DataFrame:
    """Create tidy replicate rows for a paired ``B - A`` FOV contrast."""

    delta_direction = f"{fov_b:g} - {fov_a:g}"
    return pd.DataFrame(
        {
            "analysis_type": "paired_fov_delta",
            "estimand": "paired_difference_of_ratio_of_sums",
            "delta_direction": delta_direction,
            "z_mode": str(z_mode),
            "method": str(method),
            "fov_deg": np.nan,
            "fov_a_deg": fov_a,
            "fov_b_deg": fov_b,
            "replicate": np.arange(len(samples_delta), dtype=np.int64),
            "ratio": np.nan,
            "ratio_a": np.asarray(samples_a, dtype=float),
            "ratio_b": np.asarray(samples_b, dtype=float),
            "delta_b_minus_a": np.asarray(samples_delta, dtype=float),
            "block_frame_count": block_frames,
            "seed": seed,
        }
    )


def bootstrap_observability_table(
    data: pd.DataFrame,
    *,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    block_seconds: float = DEFAULT_BLOCK_SECONDS,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
    circular: bool = True,
    paired_fovs: tuple[float, float] | None = (120.0, 360.0),
    strict_pairing: bool = True,
    frame_column: str = "frame_idx",
    z_mode_column: str = "z_mode",
    fov_column: str = "fov_deg",
    method_column: str = "method",
    numerator_column: str = "numerator",
    denominator_column: str = "denominator",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run grouped moving-block and frame-aligned paired bootstraps.

    The same deterministic seed is applied to each group. This keeps repeated
    runs byte-for-byte reproducible and gives same-length groups the same block
    start sequence. A paired comparison additionally passes both FOV arrays to
    the paired bootstrap in one call, guaranteeing identical sampled blocks.
    """

    frames_per_block = block_frame_count(frame_rate_hz, block_seconds)
    if not isinstance(circular, (bool, np.bool_)):
        raise ValueError("circular must be boolean")
    if not isinstance(strict_pairing, (bool, np.bool_)):
        raise ValueError("strict_pairing must be boolean")

    aggregated = aggregate_frame_contributions(
        data,
        frame_column=frame_column,
        z_mode_column=z_mode_column,
        fov_column=fov_column,
        method_column=method_column,
        numerator_column=numerator_column,
        denominator_column=denominator_column,
    )

    group_summaries: list[dict[str, Any]] = []
    replicate_tables: list[pd.DataFrame] = []
    analysis_columns = [z_mode_column, fov_column, method_column]
    grouped = aggregated.groupby(analysis_columns, sort=True, observed=True)
    for (z_mode, fov_deg, method), group in grouped:
        ordered = group.sort_values(frame_column, kind="stable")
        n_frames = len(ordered)
        if frames_per_block > n_frames:
            raise ValueError(
                f"block_frame_count={frames_per_block} exceeds n_frames={n_frames} "
                f"for group {(z_mode, fov_deg, method)!r}"
            )
        estimate = moving_block_bootstrap_ratio(
            ordered["numerator"].to_numpy(dtype=float),
            ordered["denominator"].to_numpy(dtype=float),
            block_length=frames_per_block,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            circular=bool(circular),
            seed=seed,
        )
        group_summaries.append(
            {
                "z_mode": str(z_mode),
                "fov_deg": float(fov_deg),
                "method": str(method),
                "n_frames": n_frames,
                "frame_start": _python_scalar(ordered[frame_column].iloc[0]),
                "frame_end": _python_scalar(ordered[frame_column].iloc[-1]),
                "numerator_sum": float(ordered["numerator"].sum()),
                "denominator_sum": float(ordered["denominator"].sum()),
                "estimate": estimate.estimate,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
            }
        )
        replicate_tables.append(
            _single_replicates(
                estimate.samples,
                z_mode=z_mode,
                fov_deg=float(fov_deg),
                method=method,
                block_frames=frames_per_block,
                seed=seed,
            )
        )

    paired_summaries: list[dict[str, Any]] = []
    pairing_omissions: list[dict[str, Any]] = []
    if paired_fovs is not None:
        if len(paired_fovs) != 2:
            raise ValueError("paired_fovs must contain exactly two FOV values")
        fov_a, fov_b = (float(paired_fovs[0]), float(paired_fovs[1]))
        if not np.all(np.isfinite([fov_a, fov_b])) or fov_a == fov_b:
            raise ValueError("paired_fovs must contain two distinct finite values")

        requested = aggregated[
            _fov_mask(aggregated[fov_column], fov_a)
            | _fov_mask(aggregated[fov_column], fov_b)
        ]
        if requested.empty and bool(strict_pairing):
            raise ValueError(f"input contains neither requested paired FOV: {fov_a}, {fov_b}")
        pair_groups = requested.groupby(
            [z_mode_column, method_column], sort=True, observed=True
        )
        for (z_mode, method), group in pair_groups:
            a = group[_fov_mask(group[fov_column], fov_a)]
            b = group[_fov_mask(group[fov_column], fov_b)]
            if a.empty or b.empty:
                omission = {
                    "z_mode": str(z_mode),
                    "method": str(method),
                    "has_fov_a": not a.empty,
                    "has_fov_b": not b.empty,
                }
                if bool(strict_pairing):
                    raise ValueError(
                        f"paired FOV data missing for z_mode={z_mode!r}, method={method!r}: "
                        f"fov_a={fov_a}, fov_b={fov_b}"
                    )
                pairing_omissions.append(omission)
                continue

            paired = a.merge(
                b,
                on=frame_column,
                how="outer",
                suffixes=("_a", "_b"),
                indicator=True,
                validate="one_to_one",
            )
            if not bool((paired["_merge"] == "both").all()):
                a_only = int((paired["_merge"] == "left_only").sum())
                b_only = int((paired["_merge"] == "right_only").sum())
                raise ValueError(
                    "paired FOVs must contain exactly the same frame labels for "
                    f"z_mode={z_mode!r}, method={method!r}; "
                    f"only_fov_a={a_only}, only_fov_b={b_only}"
                )
            paired = paired.sort_values(frame_column, kind="stable")
            n_frames = len(paired)
            if frames_per_block > n_frames:
                raise ValueError(
                    f"block_frame_count={frames_per_block} exceeds paired n_frames={n_frames} "
                    f"for z_mode={z_mode!r}, method={method!r}"
                )
            estimate = paired_moving_block_bootstrap_ratio(
                paired["numerator_a"].to_numpy(dtype=float),
                paired["denominator_a"].to_numpy(dtype=float),
                paired["numerator_b"].to_numpy(dtype=float),
                paired["denominator_b"].to_numpy(dtype=float),
                block_length=frames_per_block,
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                circular=bool(circular),
                seed=seed,
            )
            paired_summaries.append(
                {
                    "z_mode": str(z_mode),
                    "method": str(method),
                    "fov_a_deg": fov_a,
                    "fov_b_deg": fov_b,
                    "delta_definition": "fov_b - fov_a",
                    "delta_direction": f"{fov_b:g} - {fov_a:g}",
                    "n_aligned_frames": n_frames,
                    "estimate_a": estimate.estimate_a,
                    "estimate_b": estimate.estimate_b,
                    "estimate_delta_b_minus_a": estimate.estimate_delta,
                    "ci_a": list(estimate.ci_a),
                    "ci_b": list(estimate.ci_b),
                    "ci_delta_b_minus_a": list(estimate.ci_delta),
                }
            )
            replicate_tables.append(
                _paired_replicates(
                    estimate.samples_a,
                    estimate.samples_b,
                    estimate.samples_delta,
                    z_mode=z_mode,
                    method=method,
                    fov_a=fov_a,
                    fov_b=fov_b,
                    block_frames=frames_per_block,
                    seed=seed,
                )
            )

    replicates = pd.concat(replicate_tables, ignore_index=True)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "analysis": "frame_moving_block_bootstrap",
        "estimand": {
            "name": "ratio_of_sums",
            "formula": "sum(numerator) / sum(denominator)",
            "not_used": "unweighted mean of per-frame ratios",
            "pre_resampling_aggregation": "sum contributions within frame",
        },
        "bootstrap": {
            "unit": "frame",
            "frame_unit_semantics": "consecutive retained-frame observation",
            "time_order": "ascending observed frame label",
            "duration_interpretation": (
                "nominal seconds computed from consecutive retained-frame observations; "
                "declared source-frame gaps can extend the source-index span"
            ),
            "method": "circular moving block" if bool(circular) else "moving block",
            "circular": bool(circular),
            "frame_rate_hz": float(frame_rate_hz),
            "block_seconds": float(block_seconds),
            "block_frame_count": frames_per_block,
            "n_resamples": int(n_resamples),
            "confidence_level": float(confidence_level),
            "interval": "two-sided equal-tail percentile",
            "seed": int(seed),
            "seed_policy": "same deterministic seed for every group",
        },
        "paired_fov_contrast": None
        if paired_fovs is None
        else {
            "fov_a_deg": float(paired_fovs[0]),
            "fov_b_deg": float(paired_fovs[1]),
            "delta_definition": "fov_b - fov_a",
            "delta_direction": f"{float(paired_fovs[1]):g} - {float(paired_fovs[0]):g}",
            "sampling": "same aligned frames and same moving blocks",
        },
        "columns": {
            "frame": frame_column,
            "z_mode": z_mode_column,
            "fov": fov_column,
            "method": method_column,
            "numerator": numerator_column,
            "denominator": denominator_column,
        },
        "input_row_count": int(len(data)),
        "aggregated_frame_row_count": int(len(aggregated)),
        "groups": group_summaries,
        "paired_fov_deltas": paired_summaries,
        "pairing_omissions": pairing_omissions,
        "replicate_row_count": int(len(replicates)),
        "replicate_indexing": "zero_based_within_analysis_group",
    }
    return summary, replicates


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Full-record Parquet table")
    parser.add_argument(
        "--source-summary",
        type=Path,
        required=True,
        help="Completed full_record_summary.json that produced --input",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--replicates-name", default=DEFAULT_REPLICATES_NAME)
    parser.add_argument("--frame-rate-hz", type=float, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--block-seconds", type=float, default=DEFAULT_BLOCK_SECONDS)
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--non-circular", action="store_true")
    parser.add_argument(
        "--paired-fovs",
        "--pair-fovs",
        type=float,
        nargs=2,
        metavar=("FOV_A", "FOV_B"),
        default=(120.0, 360.0),
        help="Paired B-A FOV contrast (default: 120 360, i.e. 360-120)",
    )
    parser.add_argument("--no-paired-fov-delta", action="store_true")
    parser.add_argument("--allow-missing-pairs", action="store_true")
    parser.add_argument("--frame-column", default="frame_idx")
    parser.add_argument("--z-mode-column", default="z_mode")
    parser.add_argument("--fov-column", default="fov_deg")
    parser.add_argument("--method-column", default="method")
    parser.add_argument("--numerator-column", default="numerator")
    parser.add_argument("--denominator-column", default="denominator")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Read input, run the bootstrap analyses, and write both artifacts."""

    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"input Parquet does not exist: {args.input}")
    if Path(args.summary_name).name != args.summary_name:
        raise SystemExit("--summary-name must be a file name, not a path")
    if Path(args.replicates_name).name != args.replicates_name:
        raise SystemExit("--replicates-name must be a file name, not a path")

    source_snapshot = snapshot_cacie_full_record_source(args.input, args.source_summary)
    data = pd.read_parquet(args.input)
    source_contract = validate_cacie_full_record_source(
        args.input,
        args.source_summary,
        data,
        source_snapshot=source_snapshot,
        table_kind="contributions",
        allowed_frame_steps=(1, 10),
        frame_column=args.frame_column,
        z_mode_column=args.z_mode_column,
        fov_column=args.fov_column,
        method_column=args.method_column,
    )
    require_effective_frame_rate(args.frame_rate_hz, source_contract)
    paired_fovs = None if args.no_paired_fov_delta else tuple(args.paired_fovs)
    summary, replicates = bootstrap_observability_table(
        data,
        frame_rate_hz=args.frame_rate_hz,
        block_seconds=args.block_seconds,
        n_resamples=args.n_resamples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        circular=not args.non_circular,
        paired_fovs=paired_fovs,
        strict_pairing=not args.allow_missing_pairs,
        frame_column=args.frame_column,
        z_mode_column=args.z_mode_column,
        fov_column=args.fov_column,
        method_column=args.method_column,
        numerator_column=args.numerator_column,
        denominator_column=args.denominator_column,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / args.summary_name
    replicates_path = args.out_dir / args.replicates_name
    token = uuid.uuid4().hex
    temporary_summary = summary_path.with_name(f".{summary_path.name}.{token}.tmp")
    temporary_replicates = replicates_path.with_name(
        f".{replicates_path.name}.{token}.tmp"
    )
    summary["input_parquet"] = str(args.input.resolve())
    attach_source_contract(summary, source_contract)
    summary["outputs"] = {
        "summary_json": str(summary_path.resolve()),
        "replicates_parquet": str(replicates_path.resolve()),
    }
    try:
        replicates.to_parquet(temporary_replicates, index=False)
        temporary_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        require_cacie_full_record_source_unchanged(
            source_snapshot,
            args.input,
            args.source_summary,
        )
        # The JSON summary is the completion marker and is therefore published
        # after the Parquet replicates.
        os.replace(temporary_replicates, replicates_path)
        os.replace(temporary_summary, summary_path)
    finally:
        temporary_replicates.unlink(missing_ok=True)
        temporary_summary.unlink(missing_ok=True)
    print(f"wrote {summary_path}")
    print(f"wrote {replicates_path}")


if __name__ == "__main__":
    main()
