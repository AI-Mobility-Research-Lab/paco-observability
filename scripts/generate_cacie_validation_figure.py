#!/usr/bin/env python3
"""Generate source-backed CACIE geometric-validation PDF and PNG figures.

The main validation summary supplies runtime and projected-grid convergence.
A separate strata summary supplies the declared target-class and range-band
structure.  Held-out agreement and the calibrated sparse result are always
reconstructed from saved decisions plus the frozen calibration split, so an
older ``sparse_multiray_visible`` any-ray label cannot be presented as the
calibrated method.  An optional separate-subset convergence summary can supply
the frozen 17/33/65 projected-grid comparison; otherwise the 9/17/33 comparison embedded
in the main validation summary is shown.  Missing values are errors rather
than blank panels or silently invented numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from paco_observability.derivation_receipt import (
    require_named_files_unchanged,
    snapshot_named_files,
    write_derivation_receipt,
)


VALIDATION_METHODS = ("legacy_planar", "center_top", "sparse_multiray")
FROZEN_FOVS = (120.0, 360.0)
FROZEN_MAIN_FRAMES = 200
MAIN_CONVERGENCE_GRIDS = (9, 17, 33)
SUBSET_CONVERGENCE_GRIDS = (17, 33, 65)
SUBSET_CONVERGENCE_FRAMES = 25
PRIMARY_REFERENCE_GRID = 33
STRATA_METHOD = "sparse_multiray"
SPARSE_VISIBLE_FRACTION_THRESHOLD = 3.0 / 15.0

COLORS = {
    "legacy_planar": "#D55E00",
    "center_top": "#E69F00",
    "sparse_multiray": "#0072B2",
    "exact_reference": "#009E73",
    "120": "#0072B2",
    "360": "#CC79A7",
}
LABELS = {
    "legacy_planar": "Planar wedge",
    "center_top": "Center + top (2 rays)",
    "sparse_multiray": "Sparse 3D (≥3/15 rays)",
    "exact_reference": "33×33 projected-grid reference",
}
REFERENCE_SCOPE_LABEL = (
    "Exact ray–OBB intersection reference within the dynamic-box abstraction; "
    "projected-grid visibility is not physical visibility ground truth."
)
CALIBRATION_SCOPE_LABEL = (
    "Primary agreement uses 100 held-out frames. Full-200 class/range panels are descriptive; "
    "the ≥3/15 sparse threshold was selected only on the other 100 calibration frames."
)
DEFAULT_CALIBRATION_SUMMARY = Path(
    "outputs/canonical/sparse_calibration/sparse_threshold_calibration.json"
)
PUBLICATION_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


class FigureInputError(ValueError):
    """Raised when a source summary cannot support the requested figure."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication validation figures from frozen main and strata summaries"
        )
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Main 200-frame validation_summary.json (9/17/33 grids)",
    )
    parser.add_argument(
        "--strata-summary",
        type=Path,
        required=True,
        help="validation_strata_summary.json produced from saved main-run decisions",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        required=True,
        help=(
            "Main validation_decisions.parquet; sparse labels are reconstructed as "
            "sparse_multiray_fraction >= 0.2"
        ),
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=DEFAULT_CALIBRATION_SUMMARY,
        help=(
            "Calibration JSON with split.calibration_frames, split.held_out_frames, and "
            "selection.selected_min_visible_rays (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--convergence-summary",
        type=Path,
        help=(
            "Optional separate 25-frame subset validation summary containing "
            "17/33/65 grids"
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="Output path without suffix")
    parser.add_argument("--z-mode", default="ground_anchored")
    return parser.parse_args(argv)


def _read_json(path: Path, source_name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} does not exist or is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FigureInputError(
            f"{source_name} is not valid JSON ({path}): line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FigureInputError(f"{source_name} must contain a JSON object: {path}")
    return payload


def _read_decisions(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"main validation decisions do not exist or are not a file: {path}")
    try:
        decisions = pd.read_parquet(path)
    except Exception as exc:
        raise FigureInputError(f"could not read main validation decisions ({path}): {exc}") from exc
    if decisions.empty:
        raise FigureInputError(f"main validation decisions are empty: {path}")
    return decisions


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureInputError(f"{location} must be an object")
    return value


def _required_mapping(container: Mapping[str, Any], key: str, location: str) -> Mapping[str, Any]:
    if key not in container:
        raise FigureInputError(f"{location} missing required object {key!r}")
    return _mapping(container[key], f"{location}.{key}")


def _required_records(
    container: Mapping[str, Any], key: str, location: str
) -> list[Mapping[str, Any]]:
    if key not in container:
        raise FigureInputError(f"{location} missing required array {key!r}")
    values = container[key]
    if not isinstance(values, list) or not values:
        raise FigureInputError(f"{location}.{key} must be a non-empty array")
    return [_mapping(value, f"{location}.{key}[{index}]") for index, value in enumerate(values)]


def _finite_number(
    value: Any,
    location: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise FigureInputError(f"{location} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(f"{location} must be a finite number") from exc
    if not math.isfinite(result):
        raise FigureInputError(f"{location} must be a finite number")
    if lower is not None and result < lower:
        raise FigureInputError(f"{location} must be at least {lower:g}; found {result:g}")
    if upper is not None and result > upper:
        raise FigureInputError(f"{location} must be at most {upper:g}; found {result:g}")
    return result


def _same_fov(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-9)


def _require_sampled_frame_count(
    payload: Mapping[str, Any], *, source_name: str, expected: int
) -> None:
    key = "sampled_frame_count"
    if key not in payload:
        raise FigureInputError(f"{source_name} missing required field {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise FigureInputError(f"{source_name}.{key} must be an integer")
    if value != expected:
        raise FigureInputError(
            f"{source_name}.{key} must be {expected} for the frozen publication run; found {value}"
        )


def _strict_bool(series: pd.Series, column: str) -> pd.Series:
    if series.isna().any():
        raise FigureInputError(f"main validation decisions column {column!r} contains null values")
    if is_bool_dtype(series.dtype):
        return series.astype(bool)
    if is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all() and numeric.isin([0, 1]).all():
            return numeric.astype(bool)
    raise FigureInputError(
        f"main validation decisions column {column!r} must contain booleans or numeric 0/1"
    )


def _prepare_decisions(decisions: pd.DataFrame, z_mode: str) -> pd.DataFrame:
    required = {
        "frame_idx",
        "ego_id",
        "target_id",
        "class_name",
        "z_mode",
        "fov_deg",
        "range_m",
        "covered",
        "exact_visible",
        "legacy_visible",
        "center_top_visible",
        "sparse_multiray_fraction",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise FigureInputError(f"main validation decisions missing required columns: {missing}")
    work = decisions.loc[:, sorted(required)].copy()
    grouping = ["frame_idx", "ego_id", "target_id", "class_name", "z_mode"]
    if work[grouping].isna().any().any():
        bad = work[grouping].columns[work[grouping].isna().any()].tolist()
        raise FigureInputError(f"main validation decisions contain null grouping values: {bad}")
    work["covered"] = _strict_bool(work["covered"], "covered")
    work["exact_visible"] = _strict_bool(work["exact_visible"], "exact_visible")
    work["legacy_visible"] = _strict_bool(work["legacy_visible"], "legacy_visible")
    work["center_top_visible"] = _strict_bool(work["center_top_visible"], "center_top_visible")
    for column in ("frame_idx", "fov_deg", "range_m", "sparse_multiray_fraction"):
        numeric = pd.to_numeric(work[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
            raise FigureInputError(
                f"main validation decisions column {column!r} must contain finite numbers"
            )
        work[column] = numeric.astype(float)
    if not np.array_equal(work["frame_idx"].to_numpy(float), np.rint(work["frame_idx"])):
        raise FigureInputError("main validation decisions frame_idx must contain integers")
    work["frame_idx"] = np.rint(work["frame_idx"]).astype(np.int64)
    fractions = work["sparse_multiray_fraction"]
    if fractions.lt(0.0).any() or fractions.gt(1.0).any():
        raise FigureInputError(
            "main validation decisions sparse_multiray_fraction must lie in [0, 1]"
        )
    work = work.loc[work["z_mode"].astype(str).eq(z_mode)].copy()
    if work.empty:
        raise FigureInputError(f"main validation decisions contain no rows for z_mode={z_mode!r}")
    found_fovs = set(work["fov_deg"].unique().tolist())
    missing_fovs = [
        fov for fov in FROZEN_FOVS if not any(_same_fov(fov, found) for found in found_fovs)
    ]
    extra_fovs = [
        found
        for found in found_fovs
        if not any(_same_fov(found, expected) for expected in FROZEN_FOVS)
    ]
    if missing_fovs or extra_fovs:
        raise FigureInputError(
            "main validation decisions must contain exactly the frozen FOVs "
            f"{[int(value) for value in FROZEN_FOVS]} for z_mode={z_mode!r}; "
            f"missing={missing_fovs}, extra={extra_fovs}"
        )
    decision_key = ["frame_idx", "ego_id", "target_id", "z_mode", "fov_deg"]
    if work.duplicated(decision_key).any():
        duplicate = work.loc[work.duplicated(decision_key, keep=False), decision_key].iloc[0]
        raise FigureInputError(
            f"main validation decisions contain duplicate decision key: {duplicate.to_dict()}"
        )
    work["calibrated_sparse_visible"] = work["covered"] & work["sparse_multiray_fraction"].ge(
        SPARSE_VISIBLE_FRACTION_THRESHOLD
    )
    work["class_name"] = work["class_name"].astype(str)
    work["range_band"] = pd.Series(pd.NA, index=work.index, dtype="string")
    for lower, upper, upper_inclusive, label in (
        (1.5, 10.0, False, "[1.5,10)"),
        (10.0, 20.0, False, "[10,20)"),
        (20.0, 35.0, True, "[20,35]"),
    ):
        selected = work["range_m"].ge(lower) & (
            work["range_m"].le(upper) if upper_inclusive else work["range_m"].lt(upper)
        )
        work.loc[selected, "range_band"] = label
    missing_range = work["covered"] & work["range_band"].isna()
    if missing_range.any():
        examples = sorted(work.loc[missing_range, "range_m"].unique().tolist())[:10]
        raise FigureInputError(
            "covered main validation decisions fall outside declared range bands "
            f"[1.5,35]: {examples}"
        )
    return work


def _group_by_fov(
    payload: Mapping[str, Any],
    *,
    z_mode: str,
    source_name: str,
) -> dict[float, Mapping[str, Any]]:
    records = _required_records(payload, "groups", source_name)
    selected: dict[float, Mapping[str, Any]] = {}
    for index, group in enumerate(records):
        location = f"{source_name}.groups[{index}]"
        if "z_mode" not in group:
            raise FigureInputError(f"{location} missing required field 'z_mode'")
        if str(group["z_mode"]) != z_mode:
            continue
        if "fov_deg" not in group:
            raise FigureInputError(f"{location} missing required field 'fov_deg'")
        fov = _finite_number(group["fov_deg"], f"{location}.fov_deg", lower=0.0)
        duplicate = next((value for value in selected if _same_fov(value, fov)), None)
        if duplicate is not None:
            raise FigureInputError(
                f"{source_name} has duplicate groups for z_mode={z_mode!r}, fov_deg={fov:g}"
            )
        selected[fov] = group
    if not selected:
        raise FigureInputError(f"{source_name} contains no groups for z_mode={z_mode!r}")

    missing = [
        expected for expected in FROZEN_FOVS if not any(_same_fov(expected, x) for x in selected)
    ]
    extras = [found for found in selected if not any(_same_fov(found, x) for x in FROZEN_FOVS)]
    if missing or extras:
        raise FigureInputError(
            f"{source_name} for z_mode={z_mode!r} must contain exactly the frozen FOVs "
            f"{[int(value) for value in FROZEN_FOVS]}; missing={missing}, extra={extras}"
        )
    return {
        expected: next(group for fov, group in selected.items() if _same_fov(fov, expected))
        for expected in FROZEN_FOVS
    }


def _runtime_values(payload: Mapping[str, Any], z_mode: str) -> dict[str, float]:
    rows = _required_records(payload, "runtime", "main validation summary")
    specifications = {
        "center_top": 2,
        "sparse_multiray": 15,
        "exact_angular_grid": PRIMARY_REFERENCE_GRID,
    }
    result: dict[str, float] = {}
    for method, expected_grid in specifications.items():
        matches: list[tuple[int, Mapping[str, Any]]] = []
        for index, row in enumerate(rows):
            if str(row.get("z_mode")) != z_mode or str(row.get("method")) != method:
                continue
            if "grid" not in row:
                raise FigureInputError(
                    f"main validation summary.runtime[{index}] missing required field 'grid'"
                )
            grid = _finite_number(
                row["grid"], f"main validation summary.runtime[{index}].grid", lower=1.0
            )
            if grid.is_integer() and int(grid) == expected_grid:
                matches.append((index, row))
        if len(matches) != 1:
            raise FigureInputError(
                "main validation summary requires exactly one runtime row for "
                f"z_mode={z_mode!r}, method={method!r}, grid={expected_grid}; "
                f"found {len(matches)}"
            )
        index, row = matches[0]
        if "median_ms" not in row:
            raise FigureInputError(
                f"main validation summary.runtime[{index}] missing required field 'median_ms'"
            )
        result[method] = _finite_number(
            row["median_ms"],
            f"main validation summary.runtime[{index}].median_ms",
            lower=np.finfo(float).tiny,
        )
    return result


def _ordered_categories(values: set[str], preferred: Sequence[str]) -> list[str]:
    preferred_values = [value for value in preferred if value in values]
    return [*preferred_values, *sorted(values.difference(preferred_values))]


def _stratum_values(
    metrics: list[Mapping[str, Any]],
    *,
    stratum: str,
    z_mode: str,
    method: str,
    preferred_order: Sequence[str],
) -> tuple[list[str], dict[float, list[float]]]:
    selected: dict[tuple[float, str], float] = {}
    for index, row in enumerate(metrics):
        if (
            str(row.get("stratum")) != stratum
            or str(row.get("z_mode")) != z_mode
            or str(row.get("method")) != method
        ):
            continue
        location = f"strata summary.metrics[{index}]"
        if "stratum_value" not in row or "fov_deg" not in row or "accuracy" not in row:
            raise FigureInputError(
                f"{location} requires stratum_value, fov_deg, and accuracy fields"
            )
        category = str(row["stratum_value"])
        fov = _finite_number(row["fov_deg"], f"{location}.fov_deg", lower=0.0)
        canonical_fov = next((value for value in FROZEN_FOVS if _same_fov(fov, value)), None)
        if canonical_fov is None:
            continue
        key = (canonical_fov, category)
        if key in selected:
            raise FigureInputError(
                f"strata summary has duplicate {stratum!r} metric for fov={fov:g}, "
                f"category={category!r}, method={method!r}, z_mode={z_mode!r}"
            )
        selected[key] = _finite_number(
            row["accuracy"], f"{location}.accuracy", lower=0.0, upper=1.0
        )
    if not selected:
        raise FigureInputError(
            f"strata summary contains no {stratum!r} accuracy rows for "
            f"z_mode={z_mode!r}, method={method!r}"
        )

    categories_by_fov = {
        fov: {category for row_fov, category in selected if _same_fov(row_fov, fov)}
        for fov in FROZEN_FOVS
    }
    if any(not categories for categories in categories_by_fov.values()):
        missing_fovs = [fov for fov, categories in categories_by_fov.items() if not categories]
        raise FigureInputError(
            f"strata summary {stratum!r} rows missing frozen FOVs: {missing_fovs}"
        )
    first_categories = categories_by_fov[FROZEN_FOVS[0]]
    for fov, categories in categories_by_fov.items():
        if categories != first_categories:
            raise FigureInputError(
                f"strata summary {stratum!r} categories differ by FOV: "
                f"{FROZEN_FOVS[0]:g}={sorted(first_categories)}, {fov:g}={sorted(categories)}"
            )
    ordered = _ordered_categories(first_categories, preferred_order)
    return ordered, {
        fov: [selected[(fov, category)] for category in ordered] for fov in FROZEN_FOVS
    }


def _strata_values(
    payload: Mapping[str, Any], z_mode: str
) -> dict[str, tuple[list[str], dict[float, list[float]]]]:
    study_scope = _required_mapping(payload, "study_scope", "strata summary")
    if study_scope.get("sample_size_matches_declaration") is not True:
        declared = study_scope.get("declared_sampled_frames", "missing")
        observed = study_scope.get("observed_sampled_frames", "missing")
        raise FigureInputError(
            "strata summary is not publication-complete: sampled-frame declaration does not "
            f"match observed frames (declared={declared}, observed={observed})"
        )
    declared_frames = study_scope.get("declared_sampled_frames")
    observed_frames = study_scope.get("observed_sampled_frames")
    if declared_frames != FROZEN_MAIN_FRAMES or observed_frames != FROZEN_MAIN_FRAMES:
        raise FigureInputError(
            "strata summary is not the frozen publication sample: expected "
            f"{FROZEN_MAIN_FRAMES} declared and observed frames, found "
            f"declared={declared_frames}, observed={observed_frames}"
        )
    metrics = _required_records(payload, "metrics", "strata summary")
    return {
        "target_class": _stratum_values(
            metrics,
            stratum="target_class",
            z_mode=z_mode,
            method=STRATA_METHOD,
            preferred_order=("bicycle", "car", "pedestrian", "truck"),
        ),
        "range_band": _stratum_values(
            metrics,
            stratum="range_band",
            z_mode=z_mode,
            method=STRATA_METHOD,
            preferred_order=("[1.5,10)", "[10,20)", "[20,35]"),
        ),
    }


def _strata_observed_frames(payload: Mapping[str, Any]) -> int:
    study_scope = _required_mapping(payload, "study_scope", "strata summary")
    if "observed_sampled_frames" not in study_scope:
        raise FigureInputError(
            "strata summary.study_scope missing required field 'observed_sampled_frames'"
        )
    value = _finite_number(
        study_scope["observed_sampled_frames"],
        "strata summary.study_scope.observed_sampled_frames",
        lower=1.0,
    )
    if not value.is_integer():
        raise FigureInputError(
            "strata summary.study_scope.observed_sampled_frames must be an integer"
        )
    return int(value)


def _frame_list(container: Mapping[str, Any], key: str, location: str) -> tuple[int, ...]:
    if key not in container:
        raise FigureInputError(f"{location} missing required array {key!r}")
    values = container[key]
    if not isinstance(values, list):
        raise FigureInputError(f"{location}.{key} must be an array")
    frames: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FigureInputError(f"{location}.{key}[{index}] must be an integer frame ID")
        frames.append(value)
    if len(frames) != len(set(frames)):
        raise FigureInputError(f"{location}.{key} contains duplicate frame IDs")
    return tuple(frames)


def _calibration_design(
    payload: Mapping[str, Any], decision_frames: set[int]
) -> dict[str, tuple[int, ...] | int]:
    split = _required_mapping(payload, "split", "calibration summary")
    selection = _required_mapping(payload, "selection", "calibration summary")
    calibration_frames = _frame_list(split, "calibration_frames", "calibration summary.split")
    held_out_frames = _frame_list(split, "held_out_frames", "calibration summary.split")
    if len(calibration_frames) != 100 or len(held_out_frames) != 100:
        raise FigureInputError(
            "calibration summary must contain exactly 100 calibration and 100 held-out frames; "
            f"found {len(calibration_frames)} and {len(held_out_frames)}"
        )
    overlap = sorted(set(calibration_frames).intersection(held_out_frames))
    if overlap:
        raise FigureInputError(
            f"calibration and held-out frame lists must be disjoint; overlap={overlap[:10]}"
        )
    split_frames = set(calibration_frames).union(held_out_frames)
    if split_frames != decision_frames:
        missing = sorted(decision_frames.difference(split_frames))
        extra = sorted(split_frames.difference(decision_frames))
        raise FigureInputError(
            "calibration split must partition the 200 decision frames exactly; "
            f"missing_from_split={missing[:10]}, absent_from_decisions={extra[:10]}"
        )
    key = "selected_min_visible_rays"
    if key not in selection:
        raise FigureInputError(f"calibration summary.selection missing required field {key!r}")
    selected = selection[key]
    if isinstance(selected, bool) or not isinstance(selected, int) or selected != 3:
        raise FigureInputError(
            "calibration summary must freeze selected_min_visible_rays=3 for this figure; "
            f"found {selected!r}"
        )
    return {
        "calibration_frames": calibration_frames,
        "held_out_frames": held_out_frames,
        "selected_min_visible_rays": selected,
    }


def _binary_accuracy(rows: pd.DataFrame, prediction_column: str, location: str) -> float:
    if rows.empty:
        raise FigureInputError(f"{location} contains no covered decision rows")
    return float(
        np.mean(rows[prediction_column].to_numpy(bool) == rows["exact_visible"].to_numpy(bool))
    )


def _decision_values(
    decisions: pd.DataFrame,
    calibration_summary: Mapping[str, Any],
    *,
    z_mode: str,
    strata: Mapping[str, tuple[list[str], dict[float, list[float]]]],
    expected_frames: int,
) -> dict[str, Any]:
    work = _prepare_decisions(decisions, z_mode)
    observed_frames = int(work["frame_idx"].nunique())
    if observed_frames != expected_frames:
        raise FigureInputError(
            "main validation decisions and strata summary disagree on sampled-frame count: "
            f"decisions={observed_frames}, strata={expected_frames}"
        )
    split = _calibration_design(
        calibration_summary, set(work["frame_idx"].astype(int).unique().tolist())
    )

    held_out_set = set(split["held_out_frames"])
    held_out = work.loc[work["frame_idx"].isin(held_out_set) & work["covered"]]
    predictions = {
        "legacy_planar": "legacy_visible",
        "center_top": "center_top_visible",
        "sparse_multiray": "calibrated_sparse_visible",
    }
    held_out_agreement: dict[str, list[float]] = {method: [] for method in VALIDATION_METHODS}
    for fov in FROZEN_FOVS:
        fov_rows = held_out.loc[np.isclose(held_out["fov_deg"], fov, rtol=0.0, atol=1.0e-9)]
        for method, column in predictions.items():
            held_out_agreement[method].append(
                _binary_accuracy(
                    fov_rows,
                    column,
                    f"held-out fov={fov:g}, z_mode={z_mode!r}, method={method!r}",
                )
            )

    descriptive_strata: dict[str, tuple[list[str], dict[float, list[float]]]] = {}
    stratum_columns = {"target_class": "class_name", "range_band": "range_band"}
    for stratum, column in stratum_columns.items():
        categories = strata[stratum][0]
        by_fov: dict[float, list[float]] = {}
        for fov in FROZEN_FOVS:
            fov_rows = work.loc[
                np.isclose(work["fov_deg"], fov, rtol=0.0, atol=1.0e-9) & work["covered"]
            ]
            by_fov[fov] = [
                _binary_accuracy(
                    fov_rows.loc[fov_rows[column].astype(str).eq(category)],
                    "calibrated_sparse_visible",
                    f"full-200 {stratum}={category!r}, fov={fov:g}, z_mode={z_mode!r}",
                )
                for category in categories
            ]
        descriptive_strata[stratum] = (categories, by_fov)

    curves: dict[str, list[float]] = {"calibration": [], "held_out": []}
    for split_name, frame_key in (
        ("calibration", "calibration_frames"),
        ("held_out", "held_out_frames"),
    ):
        split_rows = work.loc[work["frame_idx"].isin(set(split[frame_key])) & work["covered"]]
        if split_rows.empty:
            raise FigureInputError(f"{split_name} split contains no covered decision rows")
        reference = split_rows["exact_visible"].to_numpy(bool)
        fractions = split_rows["sparse_multiray_fraction"].to_numpy(float)
        for minimum_rays in range(1, 16):
            prediction = fractions >= minimum_rays / 15.0
            curves[split_name].append(float(np.mean(prediction == reference)))

    return {
        "held_out_agreement": held_out_agreement,
        "descriptive_strata": descriptive_strata,
        "calibration_curve": {
            "minimum_visible_rays": list(range(1, 16)),
            **curves,
            "selected_min_visible_rays": split["selected_min_visible_rays"],
        },
        "split": {
            "calibration_frame_count": len(split["calibration_frames"]),
            "held_out_frame_count": len(split["held_out_frames"]),
        },
    }


def _convergence_values(
    payload: Mapping[str, Any],
    *,
    z_mode: str,
    subset_run: bool,
) -> tuple[list[int], dict[float, list[float]], int]:
    source_name = "subset convergence summary" if subset_run else "main validation summary"
    grids = SUBSET_CONVERGENCE_GRIDS if subset_run else MAIN_CONVERGENCE_GRIDS
    groups = _group_by_fov(payload, z_mode=z_mode, source_name=source_name)
    values_by_fov: dict[float, list[float]] = {}
    for fov in FROZEN_FOVS:
        group = groups[fov]
        location = f"{source_name} group z_mode={z_mode!r}, fov={fov:g}"
        convergence = _required_mapping(group, "grid_convergence", location)
        values: list[float] = []
        for grid in grids:
            key = str(grid)
            grid_metrics = _required_mapping(convergence, key, f"{location}.grid_convergence")
            metric = "mean_absolute_fraction_difference_from_primary"
            if metric not in grid_metrics:
                raise FigureInputError(
                    f"{location}.grid_convergence.{key} missing required field {metric!r}"
                )
            values.append(
                _finite_number(
                    grid_metrics[metric],
                    f"{location}.grid_convergence.{key}.{metric}",
                    lower=0.0,
                    upper=1.0,
                )
            )
        values_by_fov[fov] = values
    return list(grids), values_by_fov, max(grids)


def prepare_figure_data(
    main_summary: Mapping[str, Any],
    strata_summary: Mapping[str, Any],
    decisions: pd.DataFrame,
    calibration_summary: Mapping[str, Any],
    convergence_summary: Mapping[str, Any] | None = None,
    *,
    z_mode: str = "ground_anchored",
) -> dict[str, Any]:
    """Validate source summaries and return only values that will be plotted."""

    _require_sampled_frame_count(
        main_summary,
        source_name="main validation summary",
        expected=FROZEN_MAIN_FRAMES,
    )
    if convergence_summary is not None:
        _require_sampled_frame_count(
            convergence_summary,
            source_name="subset convergence summary",
            expected=SUBSET_CONVERGENCE_FRAMES,
        )
    _group_by_fov(main_summary, z_mode=z_mode, source_name="main validation summary")
    runtime = _runtime_values(main_summary, z_mode)
    strata = _strata_values(strata_summary, z_mode)
    decision_values = _decision_values(
        decisions,
        calibration_summary,
        z_mode=z_mode,
        strata=strata,
        expected_frames=_strata_observed_frames(strata_summary),
    )
    convergence = _convergence_values(
        convergence_summary if convergence_summary is not None else main_summary,
        z_mode=z_mode,
        subset_run=convergence_summary is not None,
    )
    return {
        "agreement": decision_values["held_out_agreement"],
        "fovs": list(FROZEN_FOVS),
        "runtime": runtime,
        "strata": decision_values["descriptive_strata"],
        "calibration_curve": decision_values["calibration_curve"],
        "split": decision_values["split"],
        "convergence": {
            "grids": convergence[0],
            "values_by_fov": convergence[1],
            "reference_grid": convergence[2],
            "source": "subset" if convergence_summary is not None else "main",
        },
        "z_mode": z_mode,
        "sparse_rule": {
            "minimum_visible_rays": 3,
            "total_rays": 15,
            "fraction_threshold": SPARSE_VISIBLE_FRACTION_THRESHOLD,
            "source": "recomputed from main validation decisions",
        },
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)


def _plot_fov_series(
    axis: plt.Axes,
    categories: Sequence[str],
    values_by_fov: Mapping[float, Sequence[float]],
) -> None:
    positions = np.arange(len(categories), dtype=float)
    for fov, marker in zip(FROZEN_FOVS, ("o", "s")):
        axis.plot(
            positions,
            values_by_fov[fov],
            marker=marker,
            color=COLORS[f"{int(fov)}"],
            label=f"{fov:g}° FOV",
        )
    axis.set_xticks(positions, categories)
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("Binary agreement")
    _style_axis(axis)


def create_figure(data: Mapping[str, Any]) -> plt.Figure:
    """Create the publication layout from already validated figure data."""

    fovs = data["fovs"]
    x = np.arange(len(fovs), dtype=float)
    with plt.rc_context(PUBLICATION_STYLE):
        figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.4))
        (
            axis_agreement,
            axis_class,
            axis_range,
            axis_calibration,
            axis_runtime,
            axis_convergence,
        ) = axes.ravel()

        for method, marker in zip(VALIDATION_METHODS, ("o", "s", "^")):
            axis_agreement.plot(
                x,
                data["agreement"][method],
                marker=marker,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis_agreement.set_xticks(x, [f"{value:g}°" for value in fovs])
        axis_agreement.set_ylim(0.0, 1.02)
        axis_agreement.set_xlabel("Modeled onboard FOV")
        axis_agreement.set_ylabel("Binary agreement")
        axis_agreement.set_title("(a) Held-out agreement with 33×33 reference", loc="left")
        axis_agreement.legend(loc="lower right", frameon=False)
        _style_axis(axis_agreement)

        class_categories, class_values = data["strata"]["target_class"]
        _plot_fov_series(axis_class, [value.title() for value in class_categories], class_values)
        axis_class.tick_params(axis="x", rotation=25)
        axis_class.set_title("(b) 3/15 by class (full-200 descriptive)", loc="left")
        axis_class.legend(loc="lower right", frameon=False)

        range_categories, range_values = data["strata"]["range_band"]
        _plot_fov_series(axis_range, range_categories, range_values)
        axis_range.set_xlabel("Target-center range (m)")
        axis_range.set_title("(c) 3/15 by range (full-200 descriptive)", loc="left")

        curve = data["calibration_curve"]
        thresholds = curve["minimum_visible_rays"]
        axis_calibration.plot(
            thresholds,
            curve["calibration"],
            "-o",
            color="#7A5195",
            label=f"Calibration (n={data['split']['calibration_frame_count']} frames)",
        )
        axis_calibration.plot(
            thresholds,
            curve["held_out"],
            "--s",
            color="#2F4B7C",
            label=f"Held out (n={data['split']['held_out_frame_count']} frames)",
        )
        selected_threshold = curve["selected_min_visible_rays"]
        axis_calibration.axvline(
            selected_threshold,
            color="#D62728",
            linestyle=":",
            linewidth=1.2,
            label=f"Selected: {selected_threshold}/15",
        )
        axis_calibration.set_xticks((1, 3, 5, 7, 9, 11, 13, 15))
        axis_calibration.set_xlim(1, 15)
        axis_calibration.set_ylim(0.0, 1.02)
        axis_calibration.set_xlabel("Minimum clear rays (of 15)")
        axis_calibration.set_ylabel("Binary agreement")
        axis_calibration.set_title("(d) Sparse-threshold calibration", loc="left")
        axis_calibration.legend(loc="lower right", frameon=False)
        _style_axis(axis_calibration)

        runtime_methods = ("center_top", "sparse_multiray", "exact_angular_grid")
        runtime_labels = ("2 rays", "15 rays", "33×33 grid")
        runtime_colors = (
            COLORS["center_top"],
            COLORS["sparse_multiray"],
            COLORS["exact_reference"],
        )
        runtime_values = [data["runtime"][method] for method in runtime_methods]
        runtime_x = np.arange(len(runtime_methods), dtype=float)
        runtime_bars = axis_runtime.bar(
            runtime_x,
            runtime_values,
            color=runtime_colors,
            edgecolor="black",
            linewidth=0.35,
            width=0.62,
        )
        axis_runtime.set_yscale("log")
        axis_runtime.set_xticks(runtime_x, runtime_labels)
        axis_runtime.set_ylabel("Median compute time (ms/target)")
        axis_runtime.set_title("(e) Per-target computational cost", loc="left")
        axis_runtime.grid(axis="y", which="both", color="#D0D0D0", linewidth=0.55, alpha=0.8)
        axis_runtime.set_axisbelow(True)
        axis_runtime.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(runtime_bars, runtime_values):
            axis_runtime.annotate(
                f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )

        convergence = data["convergence"]
        grids = convergence["grids"]
        for fov, marker in zip(FROZEN_FOVS, ("o", "s")):
            axis_convergence.plot(
                grids,
                convergence["values_by_fov"][fov],
                marker=marker,
                color=COLORS[f"{int(fov)}"],
                label=f"{fov:g}° FOV",
            )
        axis_convergence.set_xticks(grids, [f"{grid}×{grid}" for grid in grids])
        axis_convergence.set_ylim(bottom=0.0)
        axis_convergence.set_xlabel("Projected angular grid")
        axis_convergence.set_ylabel(
            f"Mean |visible fraction − {convergence['reference_grid']}×"
            f"{convergence['reference_grid']}|"
        )
        source_label = (
            "25-frame subset run" if convergence["source"] == "subset" else "main run"
        )
        axis_convergence.set_title(f"(f) Grid convergence ({source_label})", loc="left")
        axis_convergence.legend(loc="upper right", frameon=False)
        _style_axis(axis_convergence)

        mode_label = str(data["z_mode"]).replace("_", "-")
        figure.suptitle(f"PACO geometric validation — {mode_label} boxes", fontsize=11.0, y=0.985)
        figure.text(0.5, 0.022, REFERENCE_SCOPE_LABEL, ha="center", va="bottom", fontsize=7.2)
        figure.text(0.5, 0.008, CALIBRATION_SCOPE_LABEL, ha="center", va="bottom", fontsize=7.2)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.96), pad=0.9, w_pad=1.0, h_pad=1.1)
        return figure


def write_figure(figure: plt.Figure, output_base: Path) -> tuple[Path, Path]:
    """Write vector PDF and high-resolution PNG outputs."""

    if output_base.suffix:
        raise FigureInputError(
            f"--out must be a path without a suffix so both PDF and PNG can be written: {output_base}"
        )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "PACO geometric validation",
            "Subject": f"{REFERENCE_SCOPE_LABEL} {CALIBRATION_SCOPE_LABEL}",
            "Creator": "PACO generate_cacie_validation_figure.py",
        },
    )
    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Software": "PACO",
            "Description": f"{REFERENCE_SCOPE_LABEL} {CALIBRATION_SCOPE_LABEL}",
        },
    )
    return pdf_path, png_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.summary.is_file():
        raise FileNotFoundError(
            f"main validation summary does not exist or is not a file: {args.summary}"
        )
    input_files = {
        "main_summary": args.summary,
        "strata_summary": args.strata_summary,
        "main_decisions": args.decisions,
        "calibration_summary": args.calibration_summary,
    }
    if args.convergence_summary is not None:
        input_files["convergence_summary"] = args.convergence_summary
    input_records = snapshot_named_files(input_files, path_base=args.out.parent)
    main_summary = _read_json(args.summary, "main validation summary")
    strata_summary = _read_json(args.strata_summary, "strata summary")
    decisions = _read_decisions(args.decisions)
    calibration_summary = _read_json(args.calibration_summary, "calibration summary")
    convergence_summary = (
            _read_json(args.convergence_summary, "subset convergence summary")
        if args.convergence_summary is not None
        else None
    )
    data = prepare_figure_data(
        main_summary,
        strata_summary,
        decisions,
        calibration_summary,
        convergence_summary,
        z_mode=args.z_mode,
    )
    figure = create_figure(data)
    try:
        pdf_path, png_path = write_figure(figure, args.out)
    finally:
        plt.close(figure)
    require_named_files_unchanged(input_records, input_files, path_base=args.out.parent)
    write_derivation_receipt(
        receipt_path=args.out.with_suffix(".source_lineage_receipt.json"),
        derivation="cacie_validation_figure",
        input_records=input_records,
        output_files={"pdf": pdf_path, "png": png_path},
    )
    print(f"wrote {pdf_path} and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
