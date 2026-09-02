#!/usr/bin/env python3
"""Assess observability-bootstrap sensitivity to temporal block duration.

The script reads the same long-form contribution table as
``bootstrap_observability.py``. Ego rows are summed within frame before any
analysis, and every point estimate remains the ratio of sums. The default block
durations are 30, 60, 120, and 240 seconds; all use the same seed and replicate
count. Blocks longer than a group's available frame record are reported as
skipped and never replaced by a shorter or synthetic block.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


_BOOTSTRAP_SCRIPT = Path(__file__).with_name("bootstrap_observability.py")
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "_paco_bootstrap_observability_cli", _BOOTSTRAP_SCRIPT
)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load sibling script: {_BOOTSTRAP_SCRIPT}")
_BOOTSTRAP_MODULE = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_BOOTSTRAP_MODULE)

aggregate_frame_contributions = _BOOTSTRAP_MODULE.aggregate_frame_contributions
block_frame_count = _BOOTSTRAP_MODULE.block_frame_count
bootstrap_observability_table = _BOOTSTRAP_MODULE.bootstrap_observability_table

DEFAULT_BLOCK_SECONDS = (30.0, 60.0, 120.0, 240.0)
DEFAULT_REFERENCE_BLOCK_SECONDS = 120.0
DEFAULT_ACF_MAX_LAG = 2_400
DEFAULT_OUTPUT_NAME = "observability_block_sensitivity.json"


def _validate_nonnegative_integer(name: str, value: int) -> int:
    """Return a validated nonnegative integer."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _validate_positive_integer(name: str, value: int) -> int:
    """Return a validated positive integer."""

    result = _validate_nonnegative_integer(name, value)
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


def _validate_seed(seed: int) -> int:
    """Return a deterministic nonnegative integer seed."""

    return _validate_nonnegative_integer("seed", seed)


def _validate_confidence(confidence_level: float) -> float:
    """Return a finite confidence level strictly between zero and one."""

    if isinstance(confidence_level, (bool, np.bool_)):
        raise ValueError("confidence_level must be strictly between 0 and 1")
    try:
        value = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be strictly between 0 and 1") from exc
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    return value


def _validate_block_durations(
    block_seconds_values: Sequence[float], reference_block_seconds: float
) -> tuple[tuple[float, ...], float]:
    """Validate unique positive durations and the requested reference block."""

    try:
        duration_count = len(block_seconds_values)
    except TypeError as exc:
        raise ValueError("block_seconds_values must be a sequence") from exc
    if duration_count == 0:
        raise ValueError("at least one block duration is required")
    durations: list[float] = []
    for raw_value in block_seconds_values:
        if isinstance(raw_value, (bool, np.bool_)):
            raise ValueError("block durations must be finite positive numbers")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("block durations must be finite positive numbers") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("block durations must be finite positive numbers")
        if any(math.isclose(value, prior, rel_tol=0.0, abs_tol=1e-12) for prior in durations):
            raise ValueError(f"block durations must be unique; duplicate={value}")
        durations.append(value)

    try:
        reference = float(reference_block_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference_block_seconds must be a finite positive number") from exc
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_block_seconds must be a finite positive number")
    if not any(math.isclose(reference, value, rel_tol=0.0, abs_tol=1e-12) for value in durations):
        raise ValueError("reference_block_seconds must be one of the requested block durations")
    return tuple(durations), reference


def influence_acf_summary(
    numerator: Sequence[float] | np.ndarray | pd.Series,
    denominator: Sequence[float] | np.ndarray | pd.Series,
    *,
    frame_rate_hz: float,
    max_lag: int,
) -> dict[str, Any]:
    """Compute the ACF of ``numerator - p * denominator`` efficiently.

    Here ``p = sum(numerator) / sum(denominator)``. Since this influence series
    has zero sample sum by construction, the reported ACF uses uncentered
    lagged cross-products normalized by the lag-zero energy. Integer lags refer
    to adjacent rows in the sorted, observed frame sequence.

    A constant influence series has zero lag-zero energy. Its ACF and crossing
    lags are undefined and are returned as an explicit status, not zeros.
    """

    try:
        num = np.asarray(numerator, dtype=np.float64)
        den = np.asarray(denominator, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("numerator and denominator must be numeric vectors") from exc
    if num.ndim != 1 or den.ndim != 1 or num.shape != den.shape or num.size == 0:
        raise ValueError("numerator and denominator must be nonempty vectors of equal length")
    if not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)):
        raise ValueError("numerator and denominator must contain only finite values")
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(den)) * 16.0
    if np.any(num < 0.0) or np.any(den < 0.0) or np.any(num > den + tolerance):
        raise ValueError("observability components must satisfy 0 <= numerator <= denominator")
    denominator_total = float(np.sum(den, dtype=np.float64))
    if denominator_total <= 0.0:
        raise ValueError("denominator must have a positive total")
    lag_limit = _validate_nonnegative_integer("max_lag", max_lag)
    frames_per_second = float(frame_rate_hz)
    if not math.isfinite(frames_per_second) or frames_per_second <= 0.0:
        raise ValueError("frame_rate_hz must be finite and strictly positive")

    point_estimate = float(np.sum(num, dtype=np.float64) / denominator_total)
    influence = num - point_estimate * den
    n_frames = len(influence)
    effective_max_lag = min(lag_limit, n_frames - 1)
    energy = float(np.dot(influence, influence))
    energy_scale = max(1.0, float(np.max(np.abs(influence))) ** 2 * n_frames)
    zero_tolerance = np.finfo(np.float64).eps * energy_scale * 16.0
    common = {
        "definition": "numerator_t - p * denominator_t",
        "p": point_estimate,
        "n_frames": n_frames,
        "requested_max_lag_frames": lag_limit,
        "effective_max_lag_frames": effective_max_lag,
        "lag_truncated_to_record": effective_max_lag < lag_limit,
        "normalization": "lagged cross-product divided by lag-zero energy",
        "influence_sum": float(np.sum(influence, dtype=np.float64)),
        "influence_mean": float(np.mean(influence)),
        "influence_standard_deviation": float(np.std(influence)),
        "one_over_e_threshold": float(math.exp(-1.0)),
    }
    if energy <= zero_tolerance:
        return {
            **common,
            "status": "undefined_zero_variance",
            "first_below_one_over_e_lag_frames": None,
            "first_below_one_over_e_lag_seconds": None,
            "first_zero_crossing_lag_frames": None,
            "first_zero_crossing_lag_seconds": None,
            "acf": [],
        }

    transform_length = 1 << (2 * n_frames - 1).bit_length()
    spectrum = np.fft.rfft(influence, n=transform_length)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=transform_length)
    acf = np.asarray(autocorrelation[: effective_max_lag + 1] / energy, dtype=float)
    acf[0] = 1.0
    one_over_e = math.exp(-1.0)
    below_lag = next(
        (lag for lag in range(1, len(acf)) if float(acf[lag]) < one_over_e), None
    )
    zero_lag = next((lag for lag in range(1, len(acf)) if float(acf[lag]) <= 0.0), None)

    return {
        **common,
        "status": "ok",
        "first_below_one_over_e_lag_frames": below_lag,
        "first_below_one_over_e_lag_seconds": None
        if below_lag is None
        else float(below_lag / frames_per_second),
        "first_zero_crossing_lag_frames": zero_lag,
        "first_zero_crossing_lag_seconds": None
        if zero_lag is None
        else float(zero_lag / frames_per_second),
        "acf": [
            {
                "lag_frames": lag,
                "lag_seconds": float(lag / frames_per_second),
                "value": float(value),
            }
            for lag, value in enumerate(acf)
        ],
    }


def _skipped_block(
    *,
    block_seconds: float,
    block_frames: int,
    n_frames: int,
    estimate: float,
) -> dict[str, Any]:
    """Represent an overlong block without inventing an interval."""

    return {
        "block_seconds": block_seconds,
        "block_frame_count": block_frames,
        "status": "skipped",
        "skip_reason": "block_frame_count_exceeds_available_frames",
        "available_frame_count": n_frames,
        "estimate": estimate,
        "ci_low": None,
        "ci_high": None,
        "ci_width": None,
    }


def _completed_block(
    group: pd.DataFrame,
    *,
    block_seconds: float,
    frame_rate_hz: float,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    circular: bool,
    frame_column: str,
    z_mode_column: str,
    fov_column: str,
    method_column: str,
) -> dict[str, Any]:
    """Run one eligible single-group block duration through the canonical helper."""

    summary, _ = bootstrap_observability_table(
        group,
        frame_rate_hz=frame_rate_hz,
        block_seconds=block_seconds,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        circular=circular,
        paired_fovs=None,
        frame_column=frame_column,
        z_mode_column=z_mode_column,
        fov_column=fov_column,
        method_column=method_column,
        numerator_column="numerator",
        denominator_column="denominator",
    )
    result = summary["groups"][0]
    ci_width = float(result["ci_high"] - result["ci_low"])
    return {
        "block_seconds": block_seconds,
        "block_frame_count": int(summary["bootstrap"]["block_frame_count"]),
        "status": "completed",
        "skip_reason": None,
        "available_frame_count": int(result["n_frames"]),
        "estimate": float(result["estimate"]),
        "ci_low": float(result["ci_low"]),
        "ci_high": float(result["ci_high"]),
        "ci_width": ci_width,
    }


def _completed_paired_block(
    pair: pd.DataFrame,
    *,
    paired_fovs: tuple[float, float],
    block_seconds: float,
    frame_rate_hz: float,
    n_resamples: int,
    confidence_level: float,
    seed: int,
    circular: bool,
    frame_column: str,
    z_mode_column: str,
    fov_column: str,
    method_column: str,
) -> dict[str, Any]:
    """Run one eligible paired-FOV block duration through the canonical helper."""

    summary, _ = bootstrap_observability_table(
        pair,
        frame_rate_hz=frame_rate_hz,
        block_seconds=block_seconds,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
        circular=circular,
        paired_fovs=paired_fovs,
        strict_pairing=True,
        frame_column=frame_column,
        z_mode_column=z_mode_column,
        fov_column=fov_column,
        method_column=method_column,
        numerator_column="numerator",
        denominator_column="denominator",
    )
    result = summary["paired_fov_deltas"][0]
    low, high = result["ci_delta_b_minus_a"]
    return {
        "block_seconds": block_seconds,
        "block_frame_count": int(summary["bootstrap"]["block_frame_count"]),
        "status": "completed",
        "skip_reason": None,
        "available_frame_count": int(result["n_aligned_frames"]),
        "estimate": float(result["estimate_delta_b_minus_a"]),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_width": float(high - low),
    }


def _add_reference_changes(
    blocks: list[dict[str, Any]], reference_block_seconds: float
) -> None:
    """Annotate block results with absolute and relative changes in place."""

    reference = next(
        block
        for block in blocks
        if math.isclose(
            float(block["block_seconds"]),
            reference_block_seconds,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    for block in blocks:
        block["reference_block_seconds"] = reference_block_seconds
        block["estimate_change_vs_reference"] = None
        block["estimate_relative_change_vs_reference"] = None
        block["ci_width_change_vs_reference"] = None
        block["ci_width_relative_change_vs_reference"] = None
        if block["status"] != "completed":
            block["relative_change_status"] = "current_block_skipped"
            continue
        if reference["status"] != "completed":
            block["relative_change_status"] = "reference_block_skipped"
            continue

        estimate_change = float(block["estimate"] - reference["estimate"])
        width_change = float(block["ci_width"] - reference["ci_width"])
        block["estimate_change_vs_reference"] = estimate_change
        block["ci_width_change_vs_reference"] = width_change
        estimate_relative: float | None = None
        width_relative: float | None = None
        if float(reference["estimate"]) != 0.0:
            estimate_relative = float(block["estimate"] / reference["estimate"] - 1.0)
        if float(reference["ci_width"]) != 0.0:
            width_relative = float(block["ci_width"] / reference["ci_width"] - 1.0)
        block["estimate_relative_change_vs_reference"] = estimate_relative
        block["ci_width_relative_change_vs_reference"] = width_relative
        if estimate_relative is None or width_relative is None:
            block["relative_change_status"] = "partly_undefined_zero_reference"
        else:
            block["relative_change_status"] = "available"


def analyze_block_sensitivity(
    data: pd.DataFrame,
    *,
    block_seconds_values: Sequence[float] = DEFAULT_BLOCK_SECONDS,
    reference_block_seconds: float = DEFAULT_REFERENCE_BLOCK_SECONDS,
    frame_rate_hz: float = 10.0,
    n_resamples: int = 5_000,
    confidence_level: float = 0.95,
    seed: int = 20260902,
    circular: bool = True,
    acf_max_lag: int = DEFAULT_ACF_MAX_LAG,
    paired_fovs: tuple[float, float] | None = (120.0, 360.0),
    strict_pairing: bool = True,
    frame_column: str = "frame_idx",
    z_mode_column: str = "z_mode",
    fov_column: str = "fov_deg",
    method_column: str = "method",
    numerator_column: str = "numerator",
    denominator_column: str = "denominator",
) -> dict[str, Any]:
    """Compute grouped block-duration sensitivity and influence-series ACFs."""

    durations, reference = _validate_block_durations(
        block_seconds_values, reference_block_seconds
    )
    resamples = _validate_positive_integer("n_resamples", n_resamples)
    normalized_seed = _validate_seed(seed)
    confidence = _validate_confidence(confidence_level)
    max_lag = _validate_nonnegative_integer("acf_max_lag", acf_max_lag)
    if not isinstance(circular, (bool, np.bool_)):
        raise ValueError("circular must be boolean")
    if not isinstance(strict_pairing, (bool, np.bool_)):
        raise ValueError("strict_pairing must be boolean")
    frame_counts = tuple(block_frame_count(frame_rate_hz, value) for value in durations)

    aggregated = aggregate_frame_contributions(
        data,
        frame_column=frame_column,
        z_mode_column=z_mode_column,
        fov_column=fov_column,
        method_column=method_column,
        numerator_column=numerator_column,
        denominator_column=denominator_column,
    )

    group_results: list[dict[str, Any]] = []
    group_columns = [z_mode_column, fov_column, method_column]
    for (z_mode, fov_deg, method), group in aggregated.groupby(
        group_columns, sort=True, observed=True
    ):
        ordered = group.sort_values(frame_column, kind="stable")
        n_frames = len(ordered)
        point_estimate = float(ordered["numerator"].sum() / ordered["denominator"].sum())
        blocks: list[dict[str, Any]] = []
        for duration, frames in zip(durations, frame_counts, strict=True):
            if frames > n_frames:
                blocks.append(
                    _skipped_block(
                        block_seconds=duration,
                        block_frames=frames,
                        n_frames=n_frames,
                        estimate=point_estimate,
                    )
                )
            else:
                blocks.append(
                    _completed_block(
                        ordered,
                        block_seconds=duration,
                        frame_rate_hz=frame_rate_hz,
                        n_resamples=resamples,
                        confidence_level=confidence,
                        seed=normalized_seed,
                        circular=bool(circular),
                        frame_column=frame_column,
                        z_mode_column=z_mode_column,
                        fov_column=fov_column,
                        method_column=method_column,
                    )
                )
        _add_reference_changes(blocks, reference)
        group_results.append(
            {
                "z_mode": str(z_mode),
                "fov_deg": float(fov_deg),
                "method": str(method),
                "n_frames": n_frames,
                "numerator_sum": float(ordered["numerator"].sum()),
                "denominator_sum": float(ordered["denominator"].sum()),
                "point_estimate_ratio_of_sums": point_estimate,
                "influence_acf": influence_acf_summary(
                    ordered["numerator"],
                    ordered["denominator"],
                    frame_rate_hz=frame_rate_hz,
                    max_lag=max_lag,
                ),
                "blocks": blocks,
            }
        )

    paired_results: list[dict[str, Any]] = []
    pairing_omissions: list[dict[str, Any]] = []
    if paired_fovs is not None:
        if len(paired_fovs) != 2:
            raise ValueError("paired_fovs must contain exactly two values")
        fov_a, fov_b = float(paired_fovs[0]), float(paired_fovs[1])
        if not np.all(np.isfinite([fov_a, fov_b])) or fov_a == fov_b:
            raise ValueError("paired_fovs must contain two distinct finite values")
        fov_values = aggregated[fov_column].to_numpy(dtype=float)
        requested = aggregated[
            np.isclose(fov_values, fov_a, rtol=0.0, atol=1e-9)
            | np.isclose(fov_values, fov_b, rtol=0.0, atol=1e-9)
        ]
        if requested.empty and bool(strict_pairing):
            raise ValueError(f"input contains neither requested paired FOV: {fov_a}, {fov_b}")
        for (z_mode, method), pair in requested.groupby(
            [z_mode_column, method_column], sort=True, observed=True
        ):
            pair_fov = pair[fov_column].to_numpy(dtype=float)
            a = pair[np.isclose(pair_fov, fov_a, rtol=0.0, atol=1e-9)]
            b = pair[np.isclose(pair_fov, fov_b, rtol=0.0, atol=1e-9)]
            if a.empty or b.empty:
                omission = {
                    "z_mode": str(z_mode),
                    "method": str(method),
                    "has_fov_a": not a.empty,
                    "has_fov_b": not b.empty,
                }
                if bool(strict_pairing):
                    raise ValueError(
                        f"paired FOV data missing for z_mode={z_mode!r}, method={method!r}"
                    )
                pairing_omissions.append(omission)
                continue

            frame_check = a[[frame_column]].merge(
                b[[frame_column]],
                on=frame_column,
                how="outer",
                indicator=True,
                validate="one_to_one",
            )
            if not bool((frame_check["_merge"] == "both").all()):
                raise ValueError(
                    "paired FOVs must contain exactly the same frame labels for "
                    f"z_mode={z_mode!r}, method={method!r}"
                )
            n_frames = len(frame_check)
            estimate_a = float(a["numerator"].sum() / a["denominator"].sum())
            estimate_b = float(b["numerator"].sum() / b["denominator"].sum())
            point_delta = estimate_b - estimate_a
            blocks = []
            for duration, frames in zip(durations, frame_counts, strict=True):
                if frames > n_frames:
                    blocks.append(
                        _skipped_block(
                            block_seconds=duration,
                            block_frames=frames,
                            n_frames=n_frames,
                            estimate=point_delta,
                        )
                    )
                else:
                    blocks.append(
                        _completed_paired_block(
                            pair,
                            paired_fovs=(fov_a, fov_b),
                            block_seconds=duration,
                            frame_rate_hz=frame_rate_hz,
                            n_resamples=resamples,
                            confidence_level=confidence,
                            seed=normalized_seed,
                            circular=bool(circular),
                            frame_column=frame_column,
                            z_mode_column=z_mode_column,
                            fov_column=fov_column,
                            method_column=method_column,
                        )
                    )
            _add_reference_changes(blocks, reference)
            paired_results.append(
                {
                    "z_mode": str(z_mode),
                    "method": str(method),
                    "fov_a_deg": fov_a,
                    "fov_b_deg": fov_b,
                    "delta_definition": "fov_b - fov_a",
                    "delta_direction": f"{fov_b:g} - {fov_a:g}",
                    "n_aligned_frames": n_frames,
                    "point_estimate_delta": point_delta,
                    "blocks": blocks,
                }
            )

    return {
        "schema_version": "1.0",
        "analysis": "moving_block_duration_sensitivity",
        "estimand": {
            "name": "ratio_of_sums",
            "formula": "sum(frame numerator) / sum(frame denominator)",
            "pre_resampling_aggregation": "sum contributions within frame",
            "not_used": "unweighted mean of per-frame ratios",
        },
        "bootstrap": {
            "unit": "frame",
            "method": "circular moving block" if bool(circular) else "moving block",
            "frame_rate_hz": float(frame_rate_hz),
            "block_durations": [
                {"seconds": duration, "frame_count": frames}
                for duration, frames in zip(durations, frame_counts, strict=True)
            ],
            "reference_block_seconds": reference,
            "n_resamples": resamples,
            "confidence_level": confidence,
            "seed": normalized_seed,
            "same_seed_for_all_durations": True,
            "overlong_block_policy": "skip and report; never shorten or fabricate",
        },
        "influence_acf": {
            "definition": "numerator_t - p * denominator_t",
            "p_definition": "ratio_of_sums point estimate",
            "requested_max_lag_frames": max_lag,
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
        "groups": group_results,
        "paired_fov_deltas": paired_results,
        "pairing_omissions": pairing_omissions,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        "--output-json",
        "--out",
        type=Path,
        required=True,
        help=f"Output JSON path (suggested name: {DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument(
        "--block-seconds",
        type=float,
        nargs="+",
        default=DEFAULT_BLOCK_SECONDS,
        help="Block durations in seconds (default: 30 60 120 240)",
    )
    parser.add_argument(
        "--reference-block-seconds", type=float, default=DEFAULT_REFERENCE_BLOCK_SECONDS
    )
    parser.add_argument("--frame-rate-hz", type=float, default=10.0)
    parser.add_argument("--n-resamples", type=int, default=5_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--acf-max-lag", type=int, default=DEFAULT_ACF_MAX_LAG)
    parser.add_argument("--non-circular", action="store_true")
    parser.add_argument(
        "--paired-fovs",
        type=float,
        nargs=2,
        metavar=("FOV_A", "FOV_B"),
        default=(120.0, 360.0),
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
    """Read the long-form Parquet and write one sensitivity JSON artifact."""

    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"input Parquet does not exist: {args.input}")
    data = pd.read_parquet(args.input)
    paired_fovs = None if args.no_paired_fov_delta else tuple(args.paired_fovs)
    result = analyze_block_sensitivity(
        data,
        block_seconds_values=args.block_seconds,
        reference_block_seconds=args.reference_block_seconds,
        frame_rate_hz=args.frame_rate_hz,
        n_resamples=args.n_resamples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        circular=not args.non_circular,
        acf_max_lag=args.acf_max_lag,
        paired_fovs=paired_fovs,
        strict_pairing=not args.allow_missing_pairs,
        frame_column=args.frame_column,
        z_mode_column=args.z_mode_column,
        fov_column=args.fov_column,
        method_column=args.method_column,
        numerator_column=args.numerator_column,
        denominator_column=args.denominator_column,
    )
    result["input_parquet"] = str(args.input.resolve())
    result["output_json"] = str(args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
