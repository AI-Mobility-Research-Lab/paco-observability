"""Deterministic held-out calibration for the 15-ray sparse visibility rule.

The calibration unit is a sampled source frame.  Threshold selection uses
only the calibration frames; held-out frames are evaluated once after the
selection.  The analytic ray--OBB decision remains a computational reference,
not physical visibility ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


SPARSE_RAY_COUNT = 15
DEFAULT_SEED = 20260902
DEFAULT_EXPECTED_FRAMES = 200
DEFAULT_CALIBRATION_FRAMES = 100
DEFAULT_FOV_DEGREES = (120.0, 360.0)
DEFAULT_EXPECTED_MIN_VISIBLE_RAYS = 3
CANDIDATE_MIN_VISIBLE_RAYS = tuple(range(1, SPARSE_RAY_COUNT + 1))

REQUIRED_COLUMNS = frozenset(
    {
        "frame_idx",
        "ego_id",
        "target_id",
        "z_mode",
        "fov_deg",
        "covered",
        "exact_visible",
        "sparse_multiray_fraction",
        "sparse_multiray_visible",
    }
)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def strict_bool(series: pd.Series, column: str) -> pd.Series:
    """Coerce only real booleans or numeric 0/1 values."""

    if series.isna().any():
        raise ValueError(f"{column} contains null labels")
    if is_bool_dtype(series.dtype):
        return series.astype(bool)
    if is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="raise")
        if numeric.isin([0, 1]).all():
            return numeric.astype(bool)
    raise ValueError(f"{column} must contain only booleans or numeric 0/1 labels")


def visible_ray_counts(
    fractions: pd.Series,
    *,
    ray_count: int = SPARSE_RAY_COUNT,
    integer_tolerance: float = 1.0e-9,
) -> pd.Series:
    """Recover integer visible-ray counts from a saved fraction, fail closed."""

    if ray_count <= 0:
        raise ValueError("ray_count must be positive")
    if not math.isfinite(integer_tolerance) or integer_tolerance < 0:
        raise ValueError("integer_tolerance must be finite and non-negative")
    numeric = pd.to_numeric(fractions, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("sparse_multiray_fraction must contain only finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("sparse_multiray_fraction must lie in [0, 1]")
    scaled = values * ray_count
    rounded = np.rint(scaled)
    residual = np.abs(scaled - rounded)
    if np.any(residual > integer_tolerance):
        bad_position = int(np.flatnonzero(residual > integer_tolerance)[0])
        bad_index = fractions.index[bad_position]
        raise ValueError(
            "sparse_multiray_fraction * 15 must be integer-valued; "
            f"row index {bad_index!r} gives {scaled[bad_position]!r}"
        )
    counts = rounded.astype(np.int8)
    if np.any(counts < 0) or np.any(counts > ray_count):
        raise ValueError("derived visible-ray count falls outside the sampler size")
    return pd.Series(counts, index=fractions.index, name="sparse_multiray_visible_rays")


def confusion_metrics(
    predicted_visible: Iterable[bool],
    reference_visible: Iterable[bool],
) -> dict[str, int | float | None]:
    """Return visible-positive confusion metrics and signed rate bias."""

    predicted = np.asarray(list(predicted_visible), dtype=bool)
    reference = np.asarray(list(reference_visible), dtype=bool)
    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference labels must have identical shapes")
    if predicted.size == 0:
        raise ValueError("at least one eligible decision row is required")

    tp = int(np.count_nonzero(predicted & reference))
    tn = int(np.count_nonzero(~predicted & ~reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    n = int(predicted.size)
    reference_visible_n = tp + fn
    reference_hidden_n = tn + fp
    predicted_visible_n = tp + fp
    true_positive_rate = _safe_ratio(tp, reference_visible_n)
    true_negative_rate = _safe_ratio(tn, reference_hidden_n)
    balanced_accuracy = (
        float((true_positive_rate + true_negative_rate) / 2.0)
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    reference_rate = float(reference_visible_n / n)
    prediction_rate = float(predicted_visible_n / n)
    bias = prediction_rate - reference_rate
    return {
        "n": n,
        "reference_visible_n": reference_visible_n,
        "reference_hidden_n": reference_hidden_n,
        "predicted_visible_n": predicted_visible_n,
        "predicted_hidden_n": tn + fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "correct_n": tp + tn,
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": _safe_ratio(fp, reference_hidden_n),
        "true_positive_rate": true_positive_rate,
        "true_negative_rate": true_negative_rate,
        "reference_visible_rate": reference_rate,
        "prediction_visible_rate": prediction_rate,
        "prediction_reference_rate_bias": bias,
        "absolute_prediction_reference_rate_bias": abs(bias),
    }


def frame_strata(decisions: pd.DataFrame) -> pd.DataFrame:
    """Assign frames to time deciles and deterministic density quartiles."""

    if decisions.empty:
        raise ValueError("decisions table is empty")
    if {"frame_idx", "target_id"}.difference(decisions.columns):
        raise ValueError("decisions require frame_idx and target_id for stratification")
    frame_numeric = pd.to_numeric(decisions["frame_idx"], errors="coerce")
    frame_values = frame_numeric.to_numpy(dtype=float)
    if not np.isfinite(frame_values).all() or not np.equal(frame_values, np.rint(frame_values)).all():
        raise ValueError("frame_idx must contain finite integers")
    if decisions["target_id"].isna().any():
        raise ValueError("target_id contains null values")

    work = decisions[["frame_idx", "target_id"]].copy()
    work["frame_idx"] = frame_numeric.astype(np.int64)
    density = work.groupby("frame_idx", sort=True)["target_id"].nunique().astype(int)
    if density.empty or (density <= 0).any():
        raise ValueError("every sampled frame must contain at least one target")
    frames = density.index.to_numpy(dtype=np.int64)

    time_groups = min(10, len(frames))
    time_rank = pd.Series(np.arange(1, len(frames) + 1), index=frames)
    time_codes = pd.qcut(time_rank, q=time_groups, labels=False).astype(int)
    density_groups = min(4, len(frames))
    density_codes = pd.qcut(
        density.rank(method="first"), q=density_groups, labels=False
    ).astype(int)

    result = pd.DataFrame(
        {
            "frame_idx": frames,
            "frame_density_unique_targets": density.to_numpy(dtype=int),
            "time_decile": [f"D{value + 1:02d}" for value in time_codes.to_numpy()],
            "frame_density_quartile": [
                f"Q{value + 1}" for value in density_codes.to_numpy()
            ],
        }
    )
    return result.sort_values("frame_idx", kind="stable").reset_index(drop=True)


def stratified_frame_split(
    strata: pd.DataFrame,
    *,
    calibration_frame_count: int,
    seed: int = DEFAULT_SEED,
) -> tuple[list[int], list[int], pd.DataFrame]:
    """Split frames nearly in half within every time-by-density stratum."""

    required = {"frame_idx", "time_decile", "frame_density_quartile"}
    missing = sorted(required.difference(strata.columns))
    if missing:
        raise ValueError(f"strata missing required columns: {missing}")
    if strata.empty or strata["frame_idx"].duplicated().any():
        raise ValueError("strata must contain one unique row per sampled frame")
    frame_count = int(len(strata))
    if calibration_frame_count not in {frame_count // 2, (frame_count + 1) // 2}:
        raise ValueError("calibration_frame_count must produce a near-half frame split")

    grouped = list(
        strata.sort_values(
            ["time_decile", "frame_density_quartile", "frame_idx"], kind="stable"
        ).groupby(["time_decile", "frame_density_quartile"], sort=True, observed=True)
    )
    base_counts = {key: len(group) // 2 for key, group in grouped}
    remaining = calibration_frame_count - sum(base_counts.values())
    odd_keys = [key for key, group in grouped if len(group) % 2]
    if remaining < 0 or remaining > len(odd_keys):
        raise ValueError("requested split cannot keep every stratum within one frame of half")

    rng = np.random.default_rng(seed)
    extra_keys: set[tuple[Any, Any]] = set()
    if remaining:
        choices = rng.permutation(len(odd_keys))[:remaining]
        extra_keys = {odd_keys[int(position)] for position in choices}

    assignments = strata.copy()
    assignments["split"] = "held_out"
    calibration_frames: list[int] = []
    for key, group in grouped:
        ordered_frames = group["frame_idx"].astype(int).to_numpy()
        selected_count = base_counts[key] + int(key in extra_keys)
        shuffled = rng.permutation(ordered_frames)
        selected = sorted(int(value) for value in shuffled[:selected_count])
        calibration_frames.extend(selected)
        assignments.loc[assignments["frame_idx"].isin(selected), "split"] = "calibration"

    calibration_frames = sorted(calibration_frames)
    held_out_frames = sorted(
        assignments.loc[assignments["split"].eq("held_out"), "frame_idx"]
        .astype(int)
        .tolist()
    )
    if len(calibration_frames) != calibration_frame_count:
        raise RuntimeError("internal error: calibration frame count mismatch")
    if set(calibration_frames).intersection(held_out_frames):
        raise RuntimeError("internal error: calibration and held-out frames overlap")
    if set(calibration_frames).union(held_out_frames) != set(
        assignments["frame_idx"].astype(int)
    ):
        raise RuntimeError("internal error: split does not cover all frames")
    return calibration_frames, held_out_frames, assignments.sort_values(
        "frame_idx", kind="stable"
    ).reset_index(drop=True)


def _fov_key(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else format(float(value), ".12g")


def _metrics_for_split(rows: pd.DataFrame, threshold: int) -> dict[str, Any]:
    prediction = rows["sparse_multiray_visible_rays"].to_numpy(dtype=int) >= threshold
    reference = rows["exact_visible"].to_numpy(dtype=bool)
    overall = confusion_metrics(prediction, reference)
    by_fov: dict[str, dict[str, int | float | None]] = {}
    for fov, group in rows.groupby("fov_deg", sort=True, observed=True):
        group_prediction = (
            group["sparse_multiray_visible_rays"].to_numpy(dtype=int) >= threshold
        )
        by_fov[_fov_key(float(fov))] = confusion_metrics(
            group_prediction, group["exact_visible"].to_numpy(dtype=bool)
        )
    return {"overall": overall, "by_fov": by_fov}


def _stratum_summary(assignments: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (time_decile, density_quartile), group in assignments.groupby(
        ["time_decile", "frame_density_quartile"], sort=True, observed=True
    ):
        calibration_count = int(group["split"].eq("calibration").sum())
        held_out_count = int(group["split"].eq("held_out").sum())
        density = group["frame_density_unique_targets"].astype(int)
        rows.append(
            {
                "time_decile": str(time_decile),
                "frame_density_quartile": str(density_quartile),
                "frame_count": int(len(group)),
                "calibration_frame_count": calibration_count,
                "held_out_frame_count": held_out_count,
                "split_count_difference": abs(calibration_count - held_out_count),
                "min_unique_targets": int(density.min()),
                "max_unique_targets": int(density.max()),
            }
        )
    return rows


def select_candidate_threshold(
    candidates: Sequence[dict[str, Any]],
) -> tuple[int, list[int], list[int]]:
    """Apply the pre-specified accuracy, rate-bias, then threshold ordering."""

    if not candidates:
        raise ValueError("at least one threshold candidate is required")
    try:
        all_thresholds = [int(row["min_visible_rays"]) for row in candidates]
        max_correct = max(
            int(row["calibration"]["overall"]["correct_n"]) for row in candidates
        )
        accuracy_ties = [
            row
            for row in candidates
            if int(row["calibration"]["overall"]["correct_n"]) == max_correct
        ]
        min_bias_count = min(
            abs(
                int(row["calibration"]["overall"]["predicted_visible_n"])
                - int(row["calibration"]["overall"]["reference_visible_n"])
            )
            for row in accuracy_ties
        )
        bias_ties = [
            row
            for row in accuracy_ties
            if abs(
                int(row["calibration"]["overall"]["predicted_visible_n"])
                - int(row["calibration"]["overall"]["reference_visible_n"])
            )
            == min_bias_count
        ]
        accuracy_thresholds = [int(row["min_visible_rays"]) for row in accuracy_ties]
        bias_thresholds = [int(row["min_visible_rays"]) for row in bias_ties]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate metrics do not follow the calibration schema") from exc
    if len(set(all_thresholds)) != len(all_thresholds):
        raise ValueError("candidate ray thresholds must be unique")
    selected = max(bias_thresholds)
    return selected, accuracy_thresholds, bias_thresholds


def calibrate_sparse_threshold(
    decisions: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    expected_frame_count: int = DEFAULT_EXPECTED_FRAMES,
    calibration_frame_count: int = DEFAULT_CALIBRATION_FRAMES,
    fov_degrees: Sequence[float] = DEFAULT_FOV_DEGREES,
    expected_selected_min_visible_rays: int = DEFAULT_EXPECTED_MIN_VISIBLE_RAYS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select the sparse-ray threshold and return summary plus calibrated rows."""

    if decisions.empty:
        raise ValueError("decisions table is empty")
    missing = sorted(REQUIRED_COLUMNS.difference(decisions.columns))
    if missing:
        raise ValueError(f"decisions missing required columns: {missing}")
    if expected_frame_count <= 1:
        raise ValueError("expected_frame_count must exceed one")
    expected_fovs = sorted({float(value) for value in fov_degrees})
    if not expected_fovs or not np.isfinite(expected_fovs).all():
        raise ValueError("fov_degrees must contain finite values")
    if not 1 <= expected_selected_min_visible_rays <= SPARSE_RAY_COUNT:
        raise ValueError("expected selected threshold must lie in [1, 15]")

    source = decisions.copy()
    source["covered"] = strict_bool(source["covered"], "covered")
    source["exact_visible"] = strict_bool(source["exact_visible"], "exact_visible")
    source["sparse_multiray_visible"] = strict_bool(
        source["sparse_multiray_visible"], "sparse_multiray_visible"
    )
    if source["z_mode"].isna().any():
        raise ValueError("z_mode contains null values")
    fov_numeric = pd.to_numeric(source["fov_deg"], errors="coerce")
    if not np.isfinite(fov_numeric.to_numpy(dtype=float)).all():
        raise ValueError("fov_deg must contain finite numeric values")
    source["fov_deg"] = fov_numeric.astype(float)

    rays = visible_ray_counts(source["sparse_multiray_fraction"])
    if "sparse_multiray_visible_rays" in source:
        existing = pd.to_numeric(source["sparse_multiray_visible_rays"], errors="coerce")
        if not np.isfinite(existing.to_numpy(dtype=float)).all():
            raise ValueError("existing sparse_multiray_visible_rays contains non-finite values")
        if not np.array_equal(existing.to_numpy(dtype=float), rays.to_numpy(dtype=float)):
            raise ValueError(
                "existing sparse_multiray_visible_rays disagrees with fraction * 15"
            )
    input_decision = source["sparse_multiray_visible"].copy()
    if "sparse_multiray_visible_any" in source:
        any_source_column = "sparse_multiray_visible_any"
        original_any = strict_bool(
            source["sparse_multiray_visible_any"], "sparse_multiray_visible_any"
        )
    else:
        # Backward compatibility for the canonical validation file produced
        # before the explicit any-ray column was added.
        any_source_column = "legacy sparse_multiray_visible"
        original_any = input_decision.copy()
    expected_any = source["covered"] & rays.ge(1)
    if not original_any.equals(expected_any):
        mismatch_count = int((original_any != expected_any).sum())
        raise ValueError(
            f"{any_source_column} does not have covered-and-any-ray semantics; "
            f"{mismatch_count} rows disagree"
        )

    decision_key = ["frame_idx", "ego_id", "target_id", "z_mode", "fov_deg"]
    if source[decision_key].isna().any().any():
        raise ValueError("decision-key columns contain null values")
    if source.duplicated(decision_key).any():
        raise ValueError("decisions contain duplicate frame/ego/target/z-mode/FOV keys")

    strata = frame_strata(source)
    if len(strata) != expected_frame_count:
        raise ValueError(
            f"expected {expected_frame_count} sampled frames, observed {len(strata)}"
        )
    calibration_frames, held_out_frames, assignments = stratified_frame_split(
        strata,
        calibration_frame_count=calibration_frame_count,
        seed=seed,
    )

    eligible = source.loc[
        source["z_mode"].astype(str).eq("ground_anchored")
        & source["covered"]
        & source["fov_deg"].isin(expected_fovs)
    ].copy()
    eligible["sparse_multiray_visible_rays"] = rays.loc[eligible.index].astype(np.int8)
    if eligible.empty:
        raise ValueError("no ground_anchored, covered decisions exist for calibration")
    observed_fovs = sorted(eligible["fov_deg"].unique().astype(float).tolist())
    if observed_fovs != expected_fovs:
        raise ValueError(f"eligible FOVs {observed_fovs} do not match expected {expected_fovs}")
    all_frames = set(assignments["frame_idx"].astype(int))
    for fov in expected_fovs:
        fov_frames = set(
            eligible.loc[eligible["fov_deg"].eq(fov), "frame_idx"].astype(int)
        )
        if fov_frames != all_frames:
            missing_frames = sorted(all_frames - fov_frames)
            raise ValueError(
                f"FOV {fov:g} lacks covered calibration rows for frames {missing_frames[:10]}"
            )

    calibration = eligible.loc[eligible["frame_idx"].isin(calibration_frames)].copy()
    held_out = eligible.loc[eligible["frame_idx"].isin(held_out_frames)].copy()
    if calibration.empty or held_out.empty:
        raise ValueError("calibration and held-out decision populations must both be non-empty")

    candidates: list[dict[str, Any]] = []
    for threshold in CANDIDATE_MIN_VISIBLE_RAYS:
        candidates.append(
            {
                "min_visible_rays": threshold,
                "calibration": _metrics_for_split(calibration, threshold),
            }
        )

    selected, accuracy_ties, bias_ties = select_candidate_threshold(candidates)
    # Deliberately evaluate held-out frames only after the calibration-only
    # selection is frozen. All candidate curves are retained for transparency,
    # but none of them can affect ``selected`` above.
    for candidate in candidates:
        candidate["held_out"] = _metrics_for_split(
            held_out, int(candidate["min_visible_rays"])
        )
    selected_candidate = next(
        row for row in candidates if int(row["min_visible_rays"]) == selected
    )

    calibrated = source.copy()
    calibrated["sparse_multiray_visible_input"] = input_decision.astype(bool)
    calibrated["sparse_multiray_visible_any"] = original_any.astype(bool)
    calibrated["sparse_multiray_visible_rays"] = rays.astype(np.int8)
    calibrated["sparse_multiray_min_visible_rays"] = np.int8(selected)
    calibrated["sparse_multiray_visible"] = (
        calibrated["covered"] & calibrated["sparse_multiray_visible_rays"].ge(selected)
    )
    split_by_frame = assignments.set_index("frame_idx")["split"].astype(str).to_dict()
    calibrated["sparse_calibration_split"] = (
        pd.to_numeric(calibrated["frame_idx"], errors="raise")
        .astype(np.int64)
        .map(split_by_frame)
    )
    if calibrated["sparse_calibration_split"].isna().any():
        raise RuntimeError("internal error: calibrated rows lack frame split assignments")
    observed_split_labels = set(calibrated["sparse_calibration_split"].astype(str))
    if observed_split_labels != {"calibration", "held_out"}:
        raise RuntimeError(
            "internal error: calibrated output does not contain both split labels"
        )

    frame_assignments = [
        {
            "frame_idx": int(row.frame_idx),
            "frame_density_unique_targets": int(row.frame_density_unique_targets),
            "time_decile": str(row.time_decile),
            "frame_density_quartile": str(row.frame_density_quartile),
            "split": str(row.split),
        }
        for row in assignments.itertuples(index=False)
    ]
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol": {
            "seed": int(seed),
            "ray_count": SPARSE_RAY_COUNT,
            "candidate_min_visible_rays": list(CANDIDATE_MIN_VISIBLE_RAYS),
            "calibration_filter": {
                "z_mode": "ground_anchored",
                "covered": True,
                "fov_degrees": expected_fovs,
                "fov_pooling_for_selection": "row-level decisions pooled across FOVs",
            },
            "frame_density": "per-frame unique target_id over the full decisions table",
            "time_strata": "equal-frequency deciles of sorted sampled frame_idx",
            "density_strata": (
                "equal-frequency quartiles of first-ranked frame density; ties may span quartiles"
            ),
            "split_rule": (
                "frame-disjoint near-half allocation within every time-decile by density-quartile "
                "stratum; odd-cell allocation and within-cell selection use the fixed seed"
            ),
            "selection_rule": (
                "maximize pooled calibration accuracy; break ties by minimum absolute "
                "prediction-reference visible-rate bias, then higher ray threshold"
            ),
            "held_out_use": "evaluated only after the calibration threshold is selected",
            "reference": (
                "analytic ray--OBB visibility within detected dynamic OBBs; not physical "
                "visibility ground truth"
            ),
            "development_status": "post-development calibration with held-out evaluation",
            "expected_selected_min_visible_rays": int(expected_selected_min_visible_rays),
            "selection_expectation_role": (
                "regression and fail-closed guard; not a preregistered threshold"
            ),
        },
        "split": {
            "sampled_frame_count": int(len(assignments)),
            "calibration_frame_count": len(calibration_frames),
            "held_out_frame_count": len(held_out_frames),
            "calibration_frames": calibration_frames,
            "held_out_frames": held_out_frames,
            "disjoint": not bool(set(calibration_frames).intersection(held_out_frames)),
            "covers_all_sampled_frames": set(calibration_frames).union(held_out_frames)
            == all_frames,
            "frame_assignments": frame_assignments,
        },
        "strata": _stratum_summary(assignments),
        "selection": {
            "selected_min_visible_rays": selected,
            "selected_visible_fraction_threshold": float(selected / SPARSE_RAY_COUNT),
            "matches_expected_selection": selected == expected_selected_min_visible_rays,
            "thresholds_tied_on_maximum_accuracy": accuracy_ties,
            "thresholds_tied_after_rate_bias": bias_ties,
            "calibration": selected_candidate["calibration"],
            "held_out": selected_candidate["held_out"],
        },
        "candidates": candidates,
        "row_counts": {
            "input": int(len(source)),
            "eligible_ground_anchored_covered": int(len(eligible)),
            "calibration": int(len(calibration)),
            "held_out": int(len(held_out)),
            "calibrated_output": int(len(calibrated)),
        },
    }
    return summary, calibrated
