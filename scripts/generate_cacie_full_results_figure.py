#!/usr/bin/env python3
"""Generate the source-audited CACIE full-results publication figure.

The figure is deliberately a strict post-processing step.  Every plotted
point estimate is reconstructed from saved numerator/denominator counts, and
redundant JSON/Parquet values are required to agree before plotting.  The
spatial panels describe modeled ego-viewpoint residual-demand hotspots among
detected dynamic boxes; they are not infrastructure-pole or site-selection
recommendations.
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
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from paco_observability.derivation_receipt import (
    require_named_files_unchanged,
    snapshot_named_files,
    write_derivation_receipt,
)


METHODS = ("legacy_planar", "center_top", "sparse_multiray")
FOVS = (120.0, 360.0)
GROUND_MODE = "ground_anchored"
SPARSE_MIN_VISIBLE_RAYS = 3
SPARSE_RAY_COUNT = 15
EXPECTED_EGO_COUNT = 60
EXPECTED_BOOTSTRAP_RESAMPLES = 5_000
PRIMARY_BLOCK_SECONDS = 120.0
PRIMARY_CONFIDENCE_LEVEL = 0.95
RESIDUAL_BLOCK_WIDTH_FRAMES = 1_200
RESIDUAL_TOP_K = 10
BLOCK_DURATIONS = (30.0, 60.0, 120.0, 240.0)

METHOD_NUMERATOR_COLUMNS = {
    "legacy_planar": "legacy_visible_count",
    "center_top": "center_top_visible_count",
    "sparse_multiray": "sparse_multiray_visible_count",
}
METHOD_LABELS = {
    "legacy_planar": "Planar wedge",
    "center_top": "Center + top",
    "sparse_multiray": "Sparse 3D (≥3/15)",
}
METHOD_COLORS = {
    "legacy_planar": "#D55E00",
    "center_top": "#E69F00",
    "sparse_multiray": "#0072B2",
}
METHOD_MARKERS = {
    "legacy_planar": "o",
    "center_top": "s",
    "sparse_multiray": "^",
}
FOV_COLORS = {120.0: "#0072B2", 360.0: "#CC79A7"}
FOV_LINESTYLES = {120.0: "-", 360.0: "--"}

VIEWPOINT_GUARDRAIL = (
    "Spatial panels show modeled ego-viewpoint residual-demand hotspots among detected "
    "dynamic boxes—not pole or site recommendations."
)
ESTIMAND_NOTE = (
    "Ground-anchored boxes; all point estimates are count ratios (sum numerator / sum "
    "denominator). Error bars are 95% moving-block intervals where available."
)

PUBLICATION_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


class FigureInputError(ValueError):
    """Raised when source artifacts cannot support the frozen figure."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-summary", type=Path, required=True)
    parser.add_argument("--primary-bootstrap", type=Path, required=True)
    parser.add_argument("--residual-summary", type=Path, required=True)
    parser.add_argument("--residual-ranking", type=Path, required=True)
    parser.add_argument("--sensitivity-summary", type=Path, required=True)
    parser.add_argument("--block-sensitivity", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without a suffix; writes the same stem as PDF and 600-dpi PNG",
    )
    return parser.parse_args(argv)


def _read_json(path: Path, source_name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} does not exist or is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FigureInputError(
            f"{source_name} is not valid JSON ({path}): line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, Mapping):
        raise FigureInputError(f"{source_name} must contain a JSON object: {path}")
    return value


def _read_ranking(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"residual ranking Parquet does not exist or is not a file: {path}"
        )
    try:
        ranking = pd.read_parquet(path)
    except Exception as exc:
        raise FigureInputError(f"could not read residual ranking Parquet ({path}): {exc}") from exc
    if ranking.empty:
        raise FigureInputError(f"residual ranking Parquet is empty: {path}")
    return ranking


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureInputError(f"{location} must be an object")
    return value


def _required_mapping(
    container: Mapping[str, Any], key: str, location: str
) -> Mapping[str, Any]:
    if key not in container:
        raise FigureInputError(f"{location} missing required object {key!r}")
    return _mapping(container[key], f"{location}.{key}")


def _records(container: Mapping[str, Any], key: str, location: str) -> list[Mapping[str, Any]]:
    if key not in container:
        raise FigureInputError(f"{location} missing required array {key!r}")
    values = container[key]
    if not isinstance(values, list) or not values:
        raise FigureInputError(f"{location}.{key} must be a non-empty array")
    return [
        _mapping(value, f"{location}.{key}[{index}]")
        for index, value in enumerate(values)
    ]


def _finite(
    value: Any,
    location: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
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


def _integer(
    value: Any,
    location: str,
    *,
    lower: int | None = None,
) -> int:
    number = _finite(value, location)
    if not number.is_integer():
        raise FigureInputError(f"{location} must be an integer")
    result = int(number)
    if lower is not None and result < lower:
        raise FigureInputError(f"{location} must be at least {lower}; found {result}")
    return result


def _close(first: float, second: float, *, atol: float = 1.0e-12) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=atol)


def _canonical_fov(value: Any, location: str) -> float:
    number = _finite(value, location, lower=np.finfo(float).tiny)
    matches = [fov for fov in FOVS if _close(number, fov, atol=1.0e-9)]
    if len(matches) != 1:
        raise FigureInputError(
            f"{location} must be one of {[int(fov) for fov in FOVS]}; found {number:g}"
        )
    return matches[0]


def _fov_list(value: Any, location: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise FigureInputError(f"{location} must be an array")
    fovs = tuple(_canonical_fov(item, f"{location}[{index}]") for index, item in enumerate(value))
    if len(fovs) != len(set(fovs)) or set(fovs) != set(FOVS):
        raise FigureInputError(
            f"{location} must contain each frozen FOV exactly once: "
            f"{[int(fov) for fov in FOVS]}"
        )
    return fovs


def _count_triplet(value: Any, location: str) -> dict[str, int | float]:
    row = _mapping(value, location)
    numerator = _integer(row.get("numerator"), f"{location}.numerator", lower=0)
    denominator = _integer(row.get("denominator"), f"{location}.denominator", lower=1)
    if numerator > denominator:
        raise FigureInputError(f"{location} must satisfy numerator <= denominator")
    ratio = numerator / denominator
    stored = _finite(row.get("ratio"), f"{location}.ratio", lower=0.0, upper=1.0)
    if not _close(ratio, stored):
        raise FigureInputError(f"{location}.ratio is not derived from its counts")
    return {"numerator": numerator, "denominator": denominator, "ratio": ratio}


def _assert_triplet_equal(
    observed: Mapping[str, int | float],
    expected: Mapping[str, int | float],
    location: str,
) -> None:
    if (
        int(observed["numerator"]) != int(expected["numerator"])
        or int(observed["denominator"]) != int(expected["denominator"])
        or not _close(float(observed["ratio"]), float(expected["ratio"]))
    ):
        raise FigureInputError(f"{location} disagrees with the full-record count ratio")


def _validate_full_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    location = "full summary"
    if payload.get("schema_version") != "1.0":
        raise FigureInputError("full summary.schema_version must be '1.0'")
    config = _required_mapping(payload, "config", location)
    threshold = _integer(
        config.get("sparse_min_visible_rays"),
        "full summary.config.sparse_min_visible_rays",
    )
    if threshold != SPARSE_MIN_VISIBLE_RAYS:
        raise FigureInputError("full summary must use the calibrated sparse threshold 3 of 15")
    definitions = _required_mapping(config, "method_definitions", "full summary.config")
    if set(definitions) != set(METHODS):
        raise FigureInputError(f"full summary method definitions must be exactly {list(METHODS)}")
    sparse_definition = str(definitions["sparse_multiray"]).lower()
    if "3 of 15 rays" not in sparse_definition:
        raise FigureInputError("full summary must document the calibrated 3-of-15 sparse rule")
    _fov_list(config.get("fov_degrees"), "full summary.config.fov_degrees")
    z_modes = config.get("z_modes")
    if not isinstance(z_modes, list) or [str(value) for value in z_modes] != [GROUND_MODE]:
        raise FigureInputError("full summary must contain only ground_anchored boxes")
    if _integer(payload.get("frame_step"), "full summary.frame_step") != 1:
        raise FigureInputError("full summary must be the full frame_step=1 primary run")
    if _integer(payload.get("ego_positions"), "full summary.ego_positions") != EXPECTED_EGO_COUNT:
        raise FigureInputError(f"full summary must contain {EXPECTED_EGO_COUNT} ego positions")
    selected = _integer(
        payload.get("selected_frame_count"), "full summary.selected_frame_count", lower=1
    )
    processed = _integer(
        payload.get("processed_frame_count"), "full summary.processed_frame_count", lower=1
    )
    if selected != processed:
        raise FigureInputError("full summary did not process every selected frame")
    input_metadata = _required_mapping(payload, "input", location)
    if input_metadata.get("quality_gate_passed") is not True:
        raise FigureInputError("full summary input quality gate did not pass")

    groups = _records(payload, "groups", location)
    if len(groups) != len(FOVS):
        raise FigureInputError("full summary must contain exactly one ground-anchored group per FOV")
    result: dict[tuple[float, str], dict[str, int | float]] = {}
    observed_fovs: set[float] = set()
    for index, group in enumerate(groups):
        group_location = f"full summary.groups[{index}]"
        if str(group.get("z_mode")) != GROUND_MODE:
            raise FigureInputError(f"{group_location}.z_mode must be {GROUND_MODE!r}")
        fov = _canonical_fov(group.get("fov_deg"), f"{group_location}.fov_deg")
        if fov in observed_fovs:
            raise FigureInputError(f"full summary duplicates FOV {fov:g}")
        observed_fovs.add(fov)
        counts = _required_mapping(group, "counts", group_location)
        stored = _required_mapping(group, "observability_ratio_of_sums", group_location)
        if set(stored) != set(METHODS):
            raise FigureInputError(f"{group_location} must report exactly {list(METHODS)}")
        denominator = _integer(
            counts.get("total_count"), f"{group_location}.counts.total_count", lower=1
        )
        for method in METHODS:
            column = METHOD_NUMERATOR_COLUMNS[method]
            numerator = _integer(
                counts.get(column), f"{group_location}.counts.{column}", lower=0
            )
            if numerator > denominator:
                raise FigureInputError(f"{group_location}.{method} numerator exceeds total_count")
            ratio = numerator / denominator
            stored_ratio = _finite(
                stored.get(method),
                f"{group_location}.observability_ratio_of_sums.{method}",
                lower=0.0,
                upper=1.0,
            )
            if not _close(ratio, stored_ratio):
                raise FigureInputError(
                    f"{group_location}.{method} observability is not count-derived"
                )
            result[(fov, method)] = {
                "numerator": numerator,
                "denominator": denominator,
                "ratio": ratio,
            }
    if observed_fovs != set(FOVS):
        raise FigureInputError("full summary does not contain both frozen FOVs")
    return {
        "groups": result,
        "processed_frame_count": processed,
        "sparse_min_visible_rays": threshold,
    }


def _validate_primary_bootstrap(
    payload: Mapping[str, Any],
    full_groups: Mapping[tuple[float, str], Mapping[str, int | float]],
) -> dict[str, Any]:
    location = "primary bootstrap"
    if payload.get("analysis") != "frame_moving_block_bootstrap":
        raise FigureInputError(
            "primary bootstrap.analysis must be 'frame_moving_block_bootstrap'"
        )
    estimand = _required_mapping(payload, "estimand", location)
    if estimand.get("name") != "ratio_of_sums":
        raise FigureInputError("primary bootstrap must use the ratio_of_sums estimand")
    design = _required_mapping(payload, "bootstrap", location)
    if _integer(design.get("n_resamples"), f"{location}.bootstrap.n_resamples") != (
        EXPECTED_BOOTSTRAP_RESAMPLES
    ):
        raise FigureInputError(
            f"primary bootstrap must use {EXPECTED_BOOTSTRAP_RESAMPLES} resamples"
        )
    if not _close(
        _finite(design.get("block_seconds"), f"{location}.bootstrap.block_seconds"),
        PRIMARY_BLOCK_SECONDS,
    ):
        raise FigureInputError("primary bootstrap must use nominal 120-second blocks")
    if not _close(
        _finite(
            design.get("confidence_level"),
            f"{location}.bootstrap.confidence_level",
        ),
        PRIMARY_CONFIDENCE_LEVEL,
    ):
        raise FigureInputError("primary bootstrap must use 95% confidence intervals")
    if design.get("circular") is not True or design.get("method") != "circular moving block":
        raise FigureInputError("primary bootstrap must use circular moving blocks")
    if design.get("unit") != "frame":
        raise FigureInputError("primary bootstrap resampling unit must be frame")
    frame_rate = _finite(
        design.get("frame_rate_hz"), f"{location}.bootstrap.frame_rate_hz", lower=0.0
    )
    block_frames = _integer(
        design.get("block_frame_count"),
        f"{location}.bootstrap.block_frame_count",
        lower=1,
    )
    if not _close(block_frames, frame_rate * PRIMARY_BLOCK_SECONDS, atol=1.0e-9):
        raise FigureInputError("primary bootstrap block seconds and frame count disagree")

    groups = _records(payload, "groups", location)
    if len(groups) != len(FOVS) * len(METHODS):
        raise FigureInputError("primary bootstrap must contain exactly six method/FOV groups")
    result: dict[tuple[float, str], dict[str, float]] = {}
    for index, row in enumerate(groups):
        row_location = f"primary bootstrap.groups[{index}]"
        if str(row.get("z_mode")) != GROUND_MODE:
            raise FigureInputError(f"{row_location}.z_mode must be {GROUND_MODE!r}")
        fov = _canonical_fov(row.get("fov_deg"), f"{row_location}.fov_deg")
        method = str(row.get("method"))
        key = (fov, method)
        if method not in METHODS or key in result:
            raise FigureInputError(f"primary bootstrap has invalid or duplicate group {key}")
        expected = full_groups[key]
        numerator = _finite(row.get("numerator_sum"), f"{row_location}.numerator_sum")
        denominator = _finite(row.get("denominator_sum"), f"{row_location}.denominator_sum")
        if not _close(numerator, float(expected["numerator"]), atol=1.0e-9):
            raise FigureInputError(f"{row_location} numerator disagrees with full summary")
        if not _close(denominator, float(expected["denominator"]), atol=1.0e-9):
            raise FigureInputError(f"{row_location} denominator disagrees with full summary")
        estimate = _finite(
            row.get("estimate"), f"{row_location}.estimate", lower=0.0, upper=1.0
        )
        if not _close(estimate, float(expected["ratio"])):
            raise FigureInputError(f"{row_location} estimate disagrees with the count ratio")
        ci_low = _finite(
            row.get("ci_low"), f"{row_location}.ci_low", lower=0.0, upper=1.0
        )
        ci_high = _finite(
            row.get("ci_high"), f"{row_location}.ci_high", lower=0.0, upper=1.0
        )
        if not ci_low <= estimate <= ci_high:
            raise FigureInputError(f"{row_location} confidence interval is invalid")
        result[key] = {
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    if set(result) != set(full_groups):
        raise FigureInputError("primary bootstrap method/FOV coverage is incomplete")
    return {
        "groups": result,
        "frame_rate_hz": frame_rate,
        "block_frame_count": block_frames,
    }


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(f"residual ranking column {column!r} must be numeric") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise FigureInputError(f"residual ranking column {column!r} contains non-finite values")
    return values


def _validate_embedded_residual_rows(
    summary: Mapping[str, Any],
    ranking: pd.DataFrame,
) -> None:
    selected_results: dict[float, Mapping[str, Any]] = {}
    for index, result in enumerate(_records(summary, "fov_results", "residual summary")):
        if str(result.get("demand_mode")) != "uniform":
            continue
        location = f"residual summary.fov_results[{index}]"
        if str(result.get("analysis_role")) != "primary":
            raise FigureInputError(f"{location} uniform result must have analysis_role='primary'")
        fov = _canonical_fov(result.get("fov_deg"), f"{location}.fov_deg")
        if fov in selected_results:
            raise FigureInputError(f"residual summary duplicates uniform FOV {fov:g}")
        if _integer(result.get("n_ego_positions"), f"{location}.n_ego_positions") != (
            EXPECTED_EGO_COUNT
        ):
            raise FigureInputError(f"{location} must contain {EXPECTED_EGO_COUNT} ego positions")
        selected_results[fov] = result
    if set(selected_results) != set(FOVS):
        raise FigureInputError("residual summary uniform results must cover both frozen FOVs")

    text_columns = ("approach", "z_mode", "method", "demand_mode", "analysis_role")
    numeric_columns = (
        "ego_x",
        "ego_y",
        "point_rank",
        "point_estimate",
        "residual_demand_sum",
        "total_demand_sum",
        "score_ci_low",
        "score_ci_high",
        "median_rank",
        "rank_ci_low",
        "rank_ci_high",
        "top_k_probability",
        "valid_fraction",
        "n_bootstrap",
    )
    for fov, result in selected_results.items():
        embedded = result.get("ranking")
        if not isinstance(embedded, list) or len(embedded) != EXPECTED_EGO_COUNT:
            raise FigureInputError(
                f"residual summary uniform FOV {fov:g} must embed {EXPECTED_EGO_COUNT} rows"
            )
        embedded_by_id: dict[str, Mapping[str, Any]] = {}
        for index, value in enumerate(embedded):
            row = _mapping(value, f"residual summary uniform FOV {fov:g}.ranking[{index}]")
            ego_id = str(row.get("ego_id"))
            if ego_id in embedded_by_id:
                raise FigureInputError(
                    f"residual summary uniform FOV {fov:g} duplicates ego_id={ego_id!r}"
                )
            embedded_by_id[ego_id] = row
        parquet_rows = ranking.loc[ranking["fov_deg"].eq(fov)].set_index("ego_id")
        if set(embedded_by_id) != set(parquet_rows.index.astype(str)):
            raise FigureInputError(
                f"residual JSON and Parquet ego IDs disagree for uniform FOV {fov:g}"
            )
        for ego_id, embedded_row in embedded_by_id.items():
            parquet_row = parquet_rows.loc[ego_id]
            for column in text_columns:
                if str(embedded_row.get(column)) != str(parquet_row[column]):
                    raise FigureInputError(
                        f"residual JSON and Parquet disagree for FOV {fov:g}, "
                        f"ego_id={ego_id!r}, column={column!r}"
                    )
            for column in numeric_columns:
                embedded_value = _finite(
                    embedded_row.get(column),
                    f"residual summary FOV {fov:g}, ego_id={ego_id!r}.{column}",
                )
                if not _close(embedded_value, float(parquet_row[column]), atol=1.0e-9):
                    raise FigureInputError(
                        f"residual JSON and Parquet disagree for FOV {fov:g}, "
                        f"ego_id={ego_id!r}, column={column!r}"
                    )


def _validate_residual_inputs(
    summary: Mapping[str, Any],
    ranking: pd.DataFrame,
    full_groups: Mapping[tuple[float, str], Mapping[str, int | float]],
) -> dict[float, pd.DataFrame]:
    if summary.get("analysis") != "residual_demand_hotspot_rank_stability":
        raise FigureInputError(
            "residual summary.analysis must be 'residual_demand_hotspot_rank_stability'"
        )
    scope = _required_mapping(summary, "scope", "residual summary")
    if scope.get("z_mode") != GROUND_MODE or scope.get("method") != "sparse_multiray":
        raise FigureInputError("residual summary must use ground_anchored sparse_multiray scope")
    _fov_list(scope.get("fov_degrees"), "residual summary.scope.fov_degrees")
    if scope.get("primary_demand_mode") != "uniform":
        raise FigureInputError("residual summary primary demand mode must be uniform")
    score = _required_mapping(summary, "score", "residual summary")
    if score.get("name") != "residual_demand_ratio_of_sums" or score.get(
        "higher_is_hotter"
    ) is not True:
        raise FigureInputError("residual summary score must be a higher-is-hotter ratio of sums")
    formulas = _required_mapping(score, "formulas", "residual summary.score")
    uniform_formula = str(formulas.get("uniform", "")).replace(" ", "")
    if uniform_formula != (
        "sum(total_count-sparse_multiray_visible_count)/sum(total_count)"
    ):
        raise FigureInputError("residual summary uniform score formula is not the frozen count ratio")
    design = _required_mapping(summary, "bootstrap", "residual summary")
    if _integer(design.get("n_resamples"), "residual summary.bootstrap.n_resamples") != (
        EXPECTED_BOOTSTRAP_RESAMPLES
    ):
        raise FigureInputError(
            f"residual ranking must use {EXPECTED_BOOTSTRAP_RESAMPLES} bootstrap resamples"
        )
    if not _close(
        _finite(
            design.get("confidence_level"), "residual summary.bootstrap.confidence_level"
        ),
        PRIMARY_CONFIDENCE_LEVEL,
    ):
        raise FigureInputError("residual ranking must use 95% intervals")
    if _integer(
        design.get("block_width_frames"),
        "residual summary.bootstrap.block_width_frames",
    ) != RESIDUAL_BLOCK_WIDTH_FRAMES:
        raise FigureInputError("residual ranking must use 1,200-frame time blocks")
    if _integer(design.get("top_k"), "residual summary.bootstrap.top_k") != RESIDUAL_TOP_K:
        raise FigureInputError("residual ranking must use top_k=10")
    if _integer(summary.get("expected_ego_count"), "residual summary.expected_ego_count") != (
        EXPECTED_EGO_COUNT
    ):
        raise FigureInputError(f"residual summary must expect {EXPECTED_EGO_COUNT} ego positions")
    guardrail = str(summary.get("interpretation_guardrail", "")).lower()
    for phrase in ("detected dynamic-box", "not crash risk", "not a deployment-optimal pole"):
        if phrase not in guardrail:
            raise FigureInputError(
                "residual summary is missing its modeled-viewpoint interpretation guardrail"
            )

    required_columns = {
        "demand_mode",
        "analysis_role",
        "fov_deg",
        "z_mode",
        "method",
        "ego_id",
        "approach",
        "ego_x",
        "ego_y",
        "point_rank",
        "point_estimate",
        "residual_demand_sum",
        "total_demand_sum",
        "score_ci_low",
        "score_ci_high",
        "median_rank",
        "rank_ci_low",
        "rank_ci_high",
        "top_k_probability",
        "valid_fraction",
        "n_bootstrap",
    }
    missing = sorted(required_columns.difference(ranking.columns))
    if missing:
        raise FigureInputError(f"residual ranking Parquet missing columns: {missing}")
    work = ranking.loc[
        ranking["demand_mode"].astype(str).eq("uniform"), sorted(required_columns)
    ].copy()
    if work.empty:
        raise FigureInputError("residual ranking Parquet contains no uniform rows")
    if work[list(required_columns)].isna().any().any():
        raise FigureInputError("residual ranking uniform rows contain null values")
    for column in required_columns.difference(
        {"demand_mode", "analysis_role", "z_mode", "method", "ego_id", "approach"}
    ):
        work[column] = _numeric_series(work, column)
    work["fov_deg"] = work["fov_deg"].map(
        lambda value: _canonical_fov(value, "residual ranking.fov_deg")
    )
    work["ego_id"] = work["ego_id"].astype(str)
    if set(work["z_mode"].astype(str)) != {GROUND_MODE}:
        raise FigureInputError("residual ranking uniform rows must be ground_anchored")
    if set(work["method"].astype(str)) != {"sparse_multiray"}:
        raise FigureInputError("residual ranking uniform rows must use sparse_multiray")
    if set(work["analysis_role"].astype(str)) != {"primary"}:
        raise FigureInputError("residual ranking uniform rows must have analysis_role='primary'")
    if set(work["fov_deg"]) != set(FOVS):
        raise FigureInputError("residual ranking uniform rows must cover both frozen FOVs")
    if work.duplicated(["fov_deg", "ego_id"]).any():
        raise FigureInputError("residual ranking contains duplicate uniform FOV/ego rows")
    if not work["n_bootstrap"].eq(EXPECTED_BOOTSTRAP_RESAMPLES).all():
        raise FigureInputError(
            f"every residual ranking row must use {EXPECTED_BOOTSTRAP_RESAMPLES} resamples"
        )

    by_fov: dict[float, pd.DataFrame] = {}
    ego_sets: list[set[str]] = []
    for fov in FOVS:
        selected = work.loc[work["fov_deg"].eq(fov)].copy()
        if len(selected) != EXPECTED_EGO_COUNT:
            raise FigureInputError(
                f"residual ranking FOV {fov:g} must contain {EXPECTED_EGO_COUNT} ego rows"
            )
        ego_sets.append(set(selected["ego_id"]))
        ranks = selected["point_rank"].to_numpy(float)
        if not np.equal(ranks, np.rint(ranks)).all() or set(ranks.astype(int)) != set(
            range(1, EXPECTED_EGO_COUNT + 1)
        ):
            raise FigureInputError(f"residual ranking FOV {fov:g} must contain ranks 1..60")
        numerator = selected["residual_demand_sum"].to_numpy(float)
        denominator = selected["total_demand_sum"].to_numpy(float)
        estimate = selected["point_estimate"].to_numpy(float)
        if np.any(denominator <= 0.0) or np.any(numerator < 0.0) or np.any(
            numerator > denominator
        ):
            raise FigureInputError(f"residual ranking FOV {fov:g} has invalid count totals")
        if not np.allclose(estimate, numerator / denominator, rtol=0.0, atol=1.0e-12):
            raise FigureInputError(
                f"residual ranking FOV {fov:g} point estimates are not count ratios"
            )
        low = selected["score_ci_low"].to_numpy(float)
        high = selected["score_ci_high"].to_numpy(float)
        if np.any(low < 0.0) or np.any(high > 1.0) or np.any(low > high):
            raise FigureInputError(f"residual ranking FOV {fov:g} score intervals are invalid")
        rank_low = selected["rank_ci_low"].to_numpy(float)
        rank_high = selected["rank_ci_high"].to_numpy(float)
        median_rank = selected["median_rank"].to_numpy(float)
        if (
            np.any(rank_low < 1.0)
            or np.any(rank_high > EXPECTED_EGO_COUNT)
            or np.any(rank_low > median_rank)
            or np.any(median_rank > rank_high)
        ):
            raise FigureInputError(f"residual ranking FOV {fov:g} rank intervals are invalid")
        for column in ("top_k_probability", "valid_fraction"):
            if not selected[column].between(0.0, 1.0).all():
                raise FigureInputError(
                    f"residual ranking FOV {fov:g} {column} values must lie in [0, 1]"
                )
        full = full_groups[(fov, "sparse_multiray")]
        total_sum = float(np.sum(denominator, dtype=np.float64))
        residual_sum = float(np.sum(numerator, dtype=np.float64))
        if not _close(total_sum, float(full["denominator"]), atol=1.0e-6):
            raise FigureInputError(
                f"residual ranking FOV {fov:g} total demand disagrees with full summary"
            )
        expected_residual = float(full["denominator"]) - float(full["numerator"])
        if not _close(residual_sum, expected_residual, atol=1.0e-6):
            raise FigureInputError(
                f"residual ranking FOV {fov:g} residual demand disagrees with full summary"
            )
        selected["point_rank"] = selected["point_rank"].astype(int)
        by_fov[fov] = selected.sort_values("point_rank", kind="stable").reset_index(drop=True)
    if ego_sets[0] != ego_sets[1]:
        raise FigureInputError("residual ranking ego sets differ between FOVs")
    metadata = work[["ego_id", "approach", "ego_x", "ego_y"]]
    for column in ("approach", "ego_x", "ego_y"):
        if metadata.groupby("ego_id", observed=True)[column].nunique(dropna=False).gt(1).any():
            raise FigureInputError(f"residual ranking {column} changes across FOVs")

    embedded = pd.concat(by_fov.values(), ignore_index=True)
    _validate_embedded_residual_rows(summary, embedded)
    return by_fov


def _comparison_map(
    section: Mapping[str, Any],
    location: str,
) -> dict[tuple[float, str], Mapping[str, Any]]:
    result: dict[tuple[float, str], Mapping[str, Any]] = {}
    for index, row in enumerate(_records(section, "comparisons", location)):
        row_location = f"{location}.comparisons[{index}]"
        fov = _canonical_fov(row.get("fov_deg"), f"{row_location}.fov_deg")
        method = str(row.get("method"))
        key = (fov, method)
        if method not in METHODS or key in result:
            raise FigureInputError(f"{location} has invalid or duplicate comparison {key}")
        result[key] = row
    expected = {(fov, method) for fov in FOVS for method in METHODS}
    if set(result) != expected:
        raise FigureInputError(f"{location} does not contain the complete method/FOV product")
    return result


def _validate_sensitivity_summary(
    payload: Mapping[str, Any],
    full_groups: Mapping[tuple[float, str], Mapping[str, int | float]],
    primary_bootstrap: Mapping[tuple[float, str], Mapping[str, float]],
) -> list[dict[str, Any]]:
    if payload.get("analysis") != "cacie_frozen_sensitivity_summary":
        raise FigureInputError(
            "sensitivity summary.analysis must be 'cacie_frozen_sensitivity_summary'"
        )
    contracts = _required_mapping(payload, "contracts", "sensitivity summary")
    if _integer(
        contracts.get("sparse_min_visible_rays"),
        "sensitivity summary.contracts.sparse_min_visible_rays",
    ) != SPARSE_MIN_VISIBLE_RAYS:
        raise FigureInputError("sensitivity summary must use the sparse threshold 3 of 15")
    methods = contracts.get("methods")
    if not isinstance(methods, list) or tuple(str(value) for value in methods) != METHODS:
        raise FigureInputError(f"sensitivity summary methods must be exactly {list(METHODS)}")
    _fov_list(contracts.get("fov_degrees"), "sensitivity summary.contracts.fov_degrees")
    if _integer(
        contracts.get("primary_frame_step"),
        "sensitivity summary.contracts.primary_frame_step",
    ) != 1:
        raise FigureInputError("sensitivity summary primary frame step must be 1")
    if _integer(
        contracts.get("z_mode_sensitivity_frame_step"),
        "sensitivity summary.contracts.z_mode_sensitivity_frame_step",
    ) != 10:
        raise FigureInputError("sensitivity summary z-mode frame step must be 10")
    if "ratio_of_sums" not in str(contracts.get("point_estimand", "")):
        raise FigureInputError("sensitivity summary point estimand must be ratio_of_sums")

    z_section = _required_mapping(payload, "z_mode_step10", "sensitivity summary")
    decimation_section = _required_mapping(
        payload, "decimation_step10_vs_full", "sensitivity summary"
    )
    partial_section = _required_mapping(
        payload, "partial_scan_inclusion", "sensitivity summary"
    )
    if z_section.get("delta_direction") != "raw - ground_anchored":
        raise FigureInputError("z-mode sensitivity delta direction must be raw - ground_anchored")
    if decimation_section.get("delta_direction") != (
        "step10 ground_anchored - full primary ground_anchored"
    ):
        raise FigureInputError("decimation sensitivity has the wrong delta direction")
    if partial_section.get("delta_direction") != (
        "include partial scans - primary excluding partial scans"
    ):
        raise FigureInputError("partial-scan sensitivity has the wrong delta direction")
    replicate_design = _required_mapping(
        z_section, "bootstrap_replicate_ids", "sensitivity summary.z_mode_step10"
    )
    if _integer(
        replicate_design.get("count"),
        "sensitivity summary.z_mode_step10.bootstrap_replicate_ids.count",
    ) != EXPECTED_BOOTSTRAP_RESAMPLES:
        raise FigureInputError(
            f"z-mode sensitivity must use {EXPECTED_BOOTSTRAP_RESAMPLES} paired replicates"
        )
    if replicate_design.get("identical_across_all_groups") is not True:
        raise FigureInputError("z-mode sensitivity replicate IDs must match across all groups")
    if decimation_section.get("primary_bootstrap_status") != "validated":
        raise FigureInputError("sensitivity summary must validate the primary bootstrap")

    z_rows = _comparison_map(z_section, "sensitivity summary.z_mode_step10")
    decimation_rows = _comparison_map(
        decimation_section, "sensitivity summary.decimation_step10_vs_full"
    )
    partial_rows = _comparison_map(
        partial_section, "sensitivity summary.partial_scan_inclusion"
    )
    plotted: list[dict[str, Any]] = []
    for fov in FOVS:
        for method in METHODS:
            key = (fov, method)
            z_row = z_rows[key]
            raw = _count_triplet(z_row.get("raw"), f"z sensitivity {key}.raw")
            anchored = _count_triplet(
                z_row.get("ground_anchored"), f"z sensitivity {key}.ground_anchored"
            )
            if raw["denominator"] != anchored["denominator"]:
                raise FigureInputError(f"z sensitivity {key} raw/anchored denominators differ")
            delta = float(raw["ratio"]) - float(anchored["ratio"])
            stored_delta = _finite(
                z_row.get("delta_raw_minus_ground_anchored"), f"z sensitivity {key}.delta"
            )
            stored_pp = _finite(
                z_row.get("delta_percentage_points"), f"z sensitivity {key}.delta_pp"
            )
            if not _close(delta, stored_delta) or not _close(delta * 100.0, stored_pp):
                raise FigureInputError(f"z sensitivity {key} delta is not count-derived")
            z_bootstrap = _required_mapping(z_row, "bootstrap", f"z sensitivity {key}")
            if _integer(
                z_bootstrap.get("n_paired_replicates"),
                f"z sensitivity {key}.bootstrap.n_paired_replicates",
            ) != EXPECTED_BOOTSTRAP_RESAMPLES:
                raise FigureInputError(
                    f"z sensitivity {key} must use {EXPECTED_BOOTSTRAP_RESAMPLES} replicates"
                )
            if not _close(
                _finite(
                    z_bootstrap.get("confidence_level"),
                    f"z sensitivity {key}.bootstrap.confidence_level",
                ),
                PRIMARY_CONFIDENCE_LEVEL,
            ):
                raise FigureInputError(f"z sensitivity {key} must use a 95% interval")
            ci_low = _finite(z_bootstrap.get("ci_low"), f"z sensitivity {key}.ci_low")
            ci_high = _finite(z_bootstrap.get("ci_high"), f"z sensitivity {key}.ci_high")
            ci_low_pp = _finite(
                z_bootstrap.get("ci_low_percentage_points"),
                f"z sensitivity {key}.ci_low_percentage_points",
            )
            ci_high_pp = _finite(
                z_bootstrap.get("ci_high_percentage_points"),
                f"z sensitivity {key}.ci_high_percentage_points",
            )
            if ci_low > ci_high or not _close(ci_low * 100.0, ci_low_pp) or not _close(
                ci_high * 100.0, ci_high_pp
            ):
                raise FigureInputError(f"z sensitivity {key} bootstrap interval is inconsistent")
            plotted.append(
                {
                    "category": "raw_minus_ground",
                    "fov_deg": fov,
                    "method": method,
                    "delta_percentage_points": delta * 100.0,
                    "ci_low_percentage_points": ci_low * 100.0,
                    "ci_high_percentage_points": ci_high * 100.0,
                }
            )

            decimation = decimation_rows[key]
            step10 = _count_triplet(
                decimation.get("step10_ground_anchored"), f"decimation sensitivity {key}.step10"
            )
            full = _count_triplet(
                decimation.get("full_primary_ground_anchored"),
                f"decimation sensitivity {key}.full_primary",
            )
            _assert_triplet_equal(full, full_groups[key], f"decimation sensitivity {key}")
            decimation_delta = float(step10["ratio"]) - float(full["ratio"])
            if not _close(
                decimation_delta,
                _finite(
                    decimation.get("delta_step10_minus_full_primary"),
                    f"decimation sensitivity {key}.delta",
                ),
            ) or not _close(
                decimation_delta * 100.0,
                _finite(
                    decimation.get("delta_percentage_points"),
                    f"decimation sensitivity {key}.delta_pp",
                ),
            ):
                raise FigureInputError(f"decimation sensitivity {key} delta is not count-derived")
            full_ci = _required_mapping(
                decimation, "full_primary_bootstrap_ci", f"decimation sensitivity {key}"
            )
            if not _close(
                _finite(full_ci.get("ci_low"), f"decimation sensitivity {key}.ci_low"),
                float(primary_bootstrap[key]["ci_low"]),
            ) or not _close(
                _finite(full_ci.get("ci_high"), f"decimation sensitivity {key}.ci_high"),
                float(primary_bootstrap[key]["ci_high"]),
            ):
                raise FigureInputError(
                    f"decimation sensitivity {key} primary CI disagrees with primary bootstrap"
                )
            plotted.append(
                {
                    "category": "step10_minus_full",
                    "fov_deg": fov,
                    "method": method,
                    "delta_percentage_points": decimation_delta * 100.0,
                    "ci_low_percentage_points": None,
                    "ci_high_percentage_points": None,
                }
            )

            partial = partial_rows[key]
            primary = _count_triplet(
                partial.get("primary_excluding_partial"), f"partial sensitivity {key}.primary"
            )
            _assert_triplet_equal(primary, full_groups[key], f"partial sensitivity {key}")
            partial_only = _count_triplet(
                partial.get("partial_only"), f"partial sensitivity {key}.partial_only"
            )
            included = _count_triplet(
                partial.get("include_partial"), f"partial sensitivity {key}.include_partial"
            )
            combined_numerator = int(primary["numerator"]) + int(partial_only["numerator"])
            combined_denominator = int(primary["denominator"]) + int(
                partial_only["denominator"]
            )
            if included["numerator"] != combined_numerator or included[
                "denominator"
            ] != combined_denominator:
                raise FigureInputError(
                    f"partial sensitivity {key} included counts are not the component sums"
                )
            partial_delta = float(included["ratio"]) - float(primary["ratio"])
            if not _close(
                partial_delta,
                _finite(
                    partial.get("delta_include_partial_minus_primary"),
                    f"partial sensitivity {key}.delta",
                ),
            ) or not _close(
                partial_delta * 100.0,
                _finite(
                    partial.get("delta_percentage_points"),
                    f"partial sensitivity {key}.delta_pp",
                ),
            ):
                raise FigureInputError(f"partial sensitivity {key} delta is not count-derived")
            plotted.append(
                {
                    "category": "include_partial_minus_primary",
                    "fov_deg": fov,
                    "method": method,
                    "delta_percentage_points": partial_delta * 100.0,
                    "ci_low_percentage_points": None,
                    "ci_high_percentage_points": None,
                }
            )
    return plotted


def _validate_block_sensitivity(
    payload: Mapping[str, Any],
    full_groups: Mapping[tuple[float, str], Mapping[str, int | float]],
    primary_bootstrap: Mapping[tuple[float, str], Mapping[str, float]],
) -> dict[str, Any]:
    if payload.get("analysis") != "moving_block_duration_sensitivity":
        raise FigureInputError(
            "block sensitivity.analysis must be 'moving_block_duration_sensitivity'"
        )
    estimand = _required_mapping(payload, "estimand", "block sensitivity")
    if estimand.get("name") != "ratio_of_sums":
        raise FigureInputError("block sensitivity must use the ratio_of_sums estimand")
    design = _required_mapping(payload, "bootstrap", "block sensitivity")
    if _integer(design.get("n_resamples"), "block sensitivity.bootstrap.n_resamples") != (
        EXPECTED_BOOTSTRAP_RESAMPLES
    ):
        raise FigureInputError(
            f"block sensitivity must use {EXPECTED_BOOTSTRAP_RESAMPLES} resamples"
        )
    if not _close(
        _finite(
            design.get("confidence_level"), "block sensitivity.bootstrap.confidence_level"
        ),
        PRIMARY_CONFIDENCE_LEVEL,
    ):
        raise FigureInputError("block sensitivity must use 95% intervals")
    if not _close(
        _finite(
            design.get("reference_block_seconds"),
            "block sensitivity.bootstrap.reference_block_seconds",
        ),
        PRIMARY_BLOCK_SECONDS,
    ):
        raise FigureInputError("block sensitivity reference duration must be nominally 120 seconds")
    if design.get("same_seed_for_all_durations") is not True:
        raise FigureInputError("block sensitivity must use one seed across durations")
    if design.get("method") != "circular moving block":
        raise FigureInputError("block sensitivity must use circular moving blocks")
    frame_rate = _finite(
        design.get("frame_rate_hz"), "block sensitivity.bootstrap.frame_rate_hz", lower=0.0
    )
    duration_records = design.get("block_durations")
    if not isinstance(duration_records, list) or len(duration_records) != len(BLOCK_DURATIONS):
        raise FigureInputError(
            "block sensitivity must report 30, 60, 120, and 240 second durations"
        )
    duration_frames: dict[float, int] = {}
    for index, value in enumerate(duration_records):
        row = _mapping(value, f"block sensitivity.bootstrap.block_durations[{index}]")
        seconds = _finite(
            row.get("seconds"),
            f"block sensitivity.bootstrap.block_durations[{index}].seconds",
            lower=0.0,
        )
        canonical = next(
            (duration for duration in BLOCK_DURATIONS if _close(duration, seconds)), None
        )
        if canonical is None or canonical in duration_frames:
            raise FigureInputError("block sensitivity duration set is not the frozen design")
        frames = _integer(
            row.get("frame_count"),
            f"block sensitivity.bootstrap.block_durations[{index}].frame_count",
            lower=1,
        )
        if not _close(frames, canonical * frame_rate, atol=1.0e-9):
            raise FigureInputError("block sensitivity duration seconds/frame counts disagree")
        duration_frames[canonical] = frames
    if set(duration_frames) != set(BLOCK_DURATIONS):
        raise FigureInputError("block sensitivity duration coverage is incomplete")

    groups = _records(payload, "groups", "block sensitivity")
    if len(groups) != len(FOVS) * len(METHODS):
        raise FigureInputError("block sensitivity must contain exactly six method/FOV groups")
    curves: dict[tuple[float, str], list[float]] = {}
    for index, group in enumerate(groups):
        location = f"block sensitivity.groups[{index}]"
        if str(group.get("z_mode")) != GROUND_MODE:
            raise FigureInputError(f"{location}.z_mode must be {GROUND_MODE!r}")
        fov = _canonical_fov(group.get("fov_deg"), f"{location}.fov_deg")
        method = str(group.get("method"))
        key = (fov, method)
        if method not in METHODS or key in curves:
            raise FigureInputError(f"block sensitivity has invalid or duplicate group {key}")
        expected = full_groups[key]
        numerator = _finite(group.get("numerator_sum"), f"{location}.numerator_sum")
        denominator = _finite(group.get("denominator_sum"), f"{location}.denominator_sum")
        point = _finite(
            group.get("point_estimate_ratio_of_sums"),
            f"{location}.point_estimate_ratio_of_sums",
            lower=0.0,
            upper=1.0,
        )
        if not _close(numerator, float(expected["numerator"]), atol=1.0e-9) or not _close(
            denominator, float(expected["denominator"]), atol=1.0e-9
        ):
            raise FigureInputError(f"{location} counts disagree with full summary")
        if not _close(point, float(expected["ratio"])):
            raise FigureInputError(f"{location} point estimate disagrees with full summary")
        blocks = group.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != len(BLOCK_DURATIONS):
            raise FigureInputError(f"{location} must report every frozen duration")
        widths: dict[float, float] = {}
        for block_index, value in enumerate(blocks):
            block = _mapping(value, f"{location}.blocks[{block_index}]")
            seconds = _finite(
                block.get("block_seconds"), f"{location}.blocks[{block_index}].block_seconds"
            )
            canonical = next(
                (duration for duration in BLOCK_DURATIONS if _close(duration, seconds)), None
            )
            if canonical is None or canonical in widths:
                raise FigureInputError(f"{location} has an invalid or duplicate block duration")
            if block.get("status") != "completed":
                raise FigureInputError(
                    f"{location} duration {canonical:g}s is not publication-complete"
                )
            if _integer(
                block.get("block_frame_count"),
                f"{location}.blocks[{block_index}].block_frame_count",
            ) != duration_frames[canonical]:
                raise FigureInputError(f"{location} duration frame count disagrees with design")
            estimate = _finite(block.get("estimate"), f"{location}.{canonical:g}s.estimate")
            low = _finite(
                block.get("ci_low"), f"{location}.{canonical:g}s.ci_low", lower=0.0, upper=1.0
            )
            high = _finite(
                block.get("ci_high"),
                f"{location}.{canonical:g}s.ci_high",
                lower=0.0,
                upper=1.0,
            )
            width = _finite(
                block.get("ci_width"), f"{location}.{canonical:g}s.ci_width", lower=0.0
            )
            if not _close(estimate, point) or low > estimate or estimate > high:
                raise FigureInputError(f"{location} duration {canonical:g}s interval is invalid")
            if not _close(width, high - low):
                raise FigureInputError(
                    f"{location} duration {canonical:g}s CI width is not endpoint-derived"
                )
            if _close(canonical, PRIMARY_BLOCK_SECONDS):
                primary = primary_bootstrap[key]
                if not _close(low, float(primary["ci_low"])) or not _close(
                    high, float(primary["ci_high"])
                ):
                    raise FigureInputError(
                        f"{location} nominal 120-second CI disagrees with primary bootstrap"
                    )
            widths[canonical] = width * 100.0
        curves[key] = [widths[duration] for duration in BLOCK_DURATIONS]
    if set(curves) != set(full_groups):
        raise FigureInputError("block sensitivity method/FOV coverage is incomplete")
    return {"durations_seconds": list(BLOCK_DURATIONS), "ci_width_pp": curves}


def prepare_figure_data(
    full_summary: Mapping[str, Any],
    primary_bootstrap: Mapping[str, Any],
    residual_summary: Mapping[str, Any],
    residual_ranking: pd.DataFrame,
    sensitivity_summary: Mapping[str, Any],
    block_sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-validate all source artifacts and return only plotted values."""

    full = _validate_full_summary(full_summary)
    bootstrap = _validate_primary_bootstrap(primary_bootstrap, full["groups"])
    residual = _validate_residual_inputs(residual_summary, residual_ranking, full["groups"])
    sensitivity = _validate_sensitivity_summary(
        sensitivity_summary,
        full["groups"],
        bootstrap["groups"],
    )
    block = _validate_block_sensitivity(
        block_sensitivity,
        full["groups"],
        bootstrap["groups"],
    )
    return {
        "methods": list(METHODS),
        "fovs": list(FOVS),
        "primary": bootstrap["groups"],
        "residual_by_fov": residual,
        "sensitivity": sensitivity,
        "block_stability": block,
        "protocol": {
            "z_mode": GROUND_MODE,
            "sparse_min_visible_rays": SPARSE_MIN_VISIBLE_RAYS,
            "sparse_ray_count": SPARSE_RAY_COUNT,
            "bootstrap_resamples": EXPECTED_BOOTSTRAP_RESAMPLES,
            "primary_block_seconds": PRIMARY_BLOCK_SECONDS,
            "confidence_level": PRIMARY_CONFIDENCE_LEVEL,
            "ego_positions": EXPECTED_EGO_COUNT,
            "point_estimand": "ratio_of_sums",
        },
    }


def _style_axis(axis: plt.Axes, *, x_grid: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(
        axis="x" if x_grid else "y",
        color="#D0D0D0",
        linewidth=0.55,
        alpha=0.8,
    )
    axis.set_axisbelow(True)


def _plot_primary(axis: plt.Axes, data: Mapping[str, Any]) -> None:
    centers = np.arange(len(FOVS), dtype=float)
    offsets = (-0.23, 0.0, 0.23)
    for method, offset in zip(METHODS, offsets, strict=True):
        values = [float(data["primary"][(fov, method)]["estimate"]) for fov in FOVS]
        lows = [
            value - float(data["primary"][(fov, method)]["ci_low"])
            for fov, value in zip(FOVS, values, strict=True)
        ]
        highs = [
            float(data["primary"][(fov, method)]["ci_high"]) - value
            for fov, value in zip(FOVS, values, strict=True)
        ]
        axis.errorbar(
            centers + offset,
            values,
            yerr=np.asarray([lows, highs]),
            fmt=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markerfacecolor="white",
            markeredgewidth=1.0,
            capsize=2.5,
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(centers, [f"{fov:g}°" for fov in FOVS])
    axis.set_xlim(-0.5, len(FOVS) - 0.5)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Modeled onboard FOV")
    axis.set_ylabel("Full-record observability ratio")
    axis.set_title("(a) Count ratios with nominal 120-s moving-block 95% CI", loc="left")
    axis.legend(frameon=False, loc="lower right")
    _style_axis(axis)


def _plot_residual_map(
    axis: plt.Axes,
    rows: pd.DataFrame,
    fov: float,
    normalization: Normalize,
) -> Any:
    ranks = rows["point_rank"].to_numpy(int)
    sizes = 18.0 + 62.0 * (EXPECTED_EGO_COUNT + 1 - ranks) / EXPECTED_EGO_COUNT
    top = ranks <= RESIDUAL_TOP_K
    edgecolors = np.where(top, "black", "white")
    widths = np.where(top, 0.8, 0.35)
    scatter = axis.scatter(
        rows["ego_x"],
        rows["ego_y"],
        c=rows["point_estimate"],
        s=sizes,
        cmap="magma",
        norm=normalization,
        edgecolors=edgecolors,
        linewidths=widths,
        alpha=0.92,
        zorder=3,
    )
    for row in rows.loc[rows["point_rank"].le(RESIDUAL_TOP_K)].itertuples(index=False):
        axis.annotate(
            str(int(row.point_rank)),
            (float(row.ego_x), float(row.ego_y)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="#202020",
        )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Modeled ego x (m)")
    axis.set_ylabel("Modeled ego y (m)")
    panel = "b" if _close(fov, 120.0) else "c"
    axis.set_title(
        f"({panel}) Uniform residual demand · {fov:g}° (60 viewpoints)\n"
        "Top-10 point ranks labeled; modeled-viewpoint hotspot only",
        loc="left",
    )
    axis.grid(color="#D8D8D8", linewidth=0.45, alpha=0.65)
    axis.set_axisbelow(True)
    return scatter


def _plot_sensitivity(axis: plt.Axes, records: Sequence[Mapping[str, Any]]) -> None:
    categories = (
        "raw_minus_ground",
        "include_partial_minus_primary",
        "step10_minus_full",
    )
    category_labels = (
        "Raw − anchored\n(step-10; 95% CI)",
        "Include partial − primary",
        "Step-10 − full primary",
    )
    base = {category: float(len(categories) - 1 - index) for index, category in enumerate(categories)}
    offsets = {
        (fov, method): -0.25 + index * 0.10
        for index, (fov, method) in enumerate(
            (fov, method) for fov in FOVS for method in METHODS
        )
    }
    for record in records:
        fov = float(record["fov_deg"])
        method = str(record["method"])
        category = str(record["category"])
        y = base[category] + offsets[(fov, method)]
        value = float(record["delta_percentage_points"])
        low = record["ci_low_percentage_points"]
        high = record["ci_high_percentage_points"]
        xerr = None
        if low is not None and high is not None:
            xerr = np.asarray([[value - float(low)], [float(high) - value]])
        axis.errorbar(
            value,
            y,
            xerr=xerr,
            fmt=METHOD_MARKERS[method],
            color=FOV_COLORS[fov],
            markerfacecolor=("white" if _close(fov, 120.0) else FOV_COLORS[fov]),
            markeredgewidth=0.9,
            capsize=2.0,
            label=f"{METHOD_LABELS[method]} · {fov:g}°",
        )
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axis.axvline(0.0, color="#666666", linestyle=":", linewidth=0.9)
    axis.set_yticks([base[value] for value in categories], category_labels)
    axis.set_ylim(-0.55, 2.55)
    axis.set_xlabel("Sensitivity delta (percentage points)")
    axis.set_title("(d) Frozen vertical, partial-scan, and decimation sensitivities", loc="left")
    axis.legend(unique.values(), unique.keys(), ncol=3, frameon=False, loc="lower center")
    _style_axis(axis, x_grid=True)


def _plot_block_stability(axis: plt.Axes, data: Mapping[str, Any]) -> None:
    durations = data["block_stability"]["durations_seconds"]
    curves = data["block_stability"]["ci_width_pp"]
    for fov in FOVS:
        for method in METHODS:
            axis.plot(
                durations,
                curves[(fov, method)],
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                linestyle=FOV_LINESTYLES[fov],
                markerfacecolor=("white" if _close(fov, 120.0) else METHOD_COLORS[method]),
                label=f"{METHOD_LABELS[method]} · {fov:g}°",
            )
    axis.axvline(
        PRIMARY_BLOCK_SECONDS,
        color="#666666",
        linestyle=":",
        linewidth=0.9,
        label="Primary 120 s",
    )
    axis.set_xticks(durations, [f"{value:g}" for value in durations])
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Nominal moving-block duration (s)")
    axis.set_ylabel("95% CI width (percentage points)")
    axis.set_title("(e) Temporal block-duration stability", loc="left")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(axis)


def create_figure(data: Mapping[str, Any]) -> plt.Figure:
    """Create the five-panel publication figure from validated values."""

    all_scores = np.concatenate(
        [
            data["residual_by_fov"][fov]["point_estimate"].to_numpy(float)
            for fov in FOVS
        ]
    )
    minimum = float(np.min(all_scores))
    maximum = float(np.max(all_scores))
    if _close(minimum, maximum):
        padding = max(0.01, abs(minimum) * 0.05)
        minimum = max(0.0, minimum - padding)
        maximum = min(1.0, maximum + padding)
    normalization = Normalize(vmin=minimum, vmax=maximum)

    with plt.rc_context(PUBLICATION_STYLE):
        figure = plt.figure(figsize=(12.0, 7.2))
        grid = figure.add_gridspec(
            2,
            3,
            height_ratios=(1.0, 0.92),
            width_ratios=(1.08, 1.0, 1.0),
            hspace=0.42,
            wspace=0.34,
        )
        axis_primary = figure.add_subplot(grid[0, 0])
        axis_120 = figure.add_subplot(grid[0, 1])
        axis_360 = figure.add_subplot(grid[0, 2])
        axis_sensitivity = figure.add_subplot(grid[1, :2])
        axis_blocks = figure.add_subplot(grid[1, 2])

        _plot_primary(axis_primary, data)
        scatter = _plot_residual_map(
            axis_120, data["residual_by_fov"][120.0], 120.0, normalization
        )
        _plot_residual_map(
            axis_360, data["residual_by_fov"][360.0], 360.0, normalization
        )
        figure.subplots_adjust(left=0.065, right=0.925, bottom=0.105, top=0.91)
        colorbar_axis = figure.add_axes((0.945, 0.565, 0.012, 0.265))
        colorbar = figure.colorbar(scatter, cax=colorbar_axis)
        colorbar.set_label("Uniform residual-demand count ratio")
        _plot_sensitivity(axis_sensitivity, data["sensitivity"])
        _plot_block_stability(axis_blocks, data)

        figure.suptitle(
            "CACIE full-record observability and robustness — ground-anchored, sparse ≥3/15",
            fontsize=11.2,
            y=0.985,
        )
        figure.text(0.5, 0.028, ESTIMAND_NOTE, ha="center", va="bottom", fontsize=7.1)
        figure.text(
            0.5,
            0.012,
            VIEWPOINT_GUARDRAIL,
            ha="center",
            va="bottom",
            fontsize=7.1,
            fontweight="bold",
        )
        return figure


def write_figure(figure: plt.Figure, output_base: Path) -> tuple[Path, Path]:
    """Write a vector PDF and a 600-dpi PNG with the same stem."""

    if output_base.suffix:
        raise FigureInputError(
            f"--out must have no suffix so PDF and PNG share one stem: {output_base}"
        )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    metadata_text = f"{ESTIMAND_NOTE} {VIEWPOINT_GUARDRAIL}"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "CACIE full-record observability and robustness",
            "Subject": metadata_text,
            "Creator": "PACO generate_cacie_full_results_figure.py",
        },
    )
    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "PACO", "Description": metadata_text},
    )
    return pdf_path, png_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_files = {
        "full_summary": args.full_summary,
        "primary_bootstrap": args.primary_bootstrap,
        "residual_summary": args.residual_summary,
        "residual_ranking": args.residual_ranking,
        "sensitivity_summary": args.sensitivity_summary,
        "block_sensitivity": args.block_sensitivity,
    }
    input_records = snapshot_named_files(input_files, path_base=args.out.parent)
    data = prepare_figure_data(
        _read_json(args.full_summary, "full summary"),
        _read_json(args.primary_bootstrap, "primary bootstrap"),
        _read_json(args.residual_summary, "residual summary"),
        _read_ranking(args.residual_ranking),
        _read_json(args.sensitivity_summary, "sensitivity summary"),
        _read_json(args.block_sensitivity, "block sensitivity"),
    )
    figure = create_figure(data)
    try:
        pdf_path, png_path = write_figure(figure, args.out)
    finally:
        plt.close(figure)
    require_named_files_unchanged(input_records, input_files, path_base=args.out.parent)
    write_derivation_receipt(
        receipt_path=args.out.with_suffix(".source_lineage_receipt.json"),
        derivation="cacie_full_results_figure",
        input_records=input_records,
        output_files={"pdf": pdf_path, "png": png_path},
    )
    print(f"wrote {pdf_path} and {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
