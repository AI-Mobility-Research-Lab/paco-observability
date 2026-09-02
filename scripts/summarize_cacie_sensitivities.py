#!/usr/bin/env python3
"""Combine declared CACIE z-mode, decimation, and partial-scan sensitivities.

This is a strict post-processing step.  It never reads trajectory rows or
recomputes visibility.  Every reported point estimate is rebuilt from saved
numerator/denominator counts, and the raw-minus-ground interval pairs existing
single-ratio bootstrap draws by their complete replicate key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


METHOD_NUMERATOR_COLUMNS: dict[str, str] = {
    "legacy_planar": "legacy_visible_count",
    "center_top": "center_top_visible_count",
    "sparse_multiray": "sparse_multiray_visible_count",
}
METHOD_ORDER = tuple(METHOD_NUMERATOR_COLUMNS)
GROUND_MODE = "ground_anchored"
RAW_MODE = "raw"
EXPECTED_SPARSE_MIN_VISIBLE_RAYS = 3
EXPECTED_PARTIAL_FRAMES = (
    6674,
    6675,
    6747,
    7025,
    7435,
    11077,
    11155,
    11166,
    11178,
    11189,
    11201,
    11212,
    11213,
    11235,
    11247,
    40795,
)
EXPECTED_ABSENT_PARTIAL_FRAME = 6747
EXPECTED_OBSERVED_PARTIAL_FRAMES = tuple(
    frame for frame in EXPECTED_PARTIAL_FRAMES if frame != EXPECTED_ABSENT_PARTIAL_FRAME
)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _integer(value: Any, name: str) -> int:
    numeric = _finite_float(value, name)
    if not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _frame_set(value: Any, name: str) -> tuple[int, ...]:
    frames = tuple(sorted(_integer(item, name) for item in _require_list(value, name)))
    if len(frames) != len(set(frames)):
        raise ValueError(f"{name} contains duplicate frame IDs")
    return frames


def _validate_run_selection(summary: Mapping[str, Any], role: str) -> None:
    input_metadata = _require_mapping(summary.get("input"), f"{role}.input")
    if input_metadata.get("quality_gate_passed") is not True:
        raise ValueError(f"{role} did not pass its input quality gate")
    processed = _integer(summary.get("processed_frame_count"), f"{role}.processed_frame_count")
    selected = _integer(summary.get("selected_frame_count"), f"{role}.selected_frame_count")
    if processed != selected:
        raise ValueError(f"{role} did not process every selected frame")


def _validate_primary_or_step10_exclusions(summary: Mapping[str, Any], role: str) -> None:
    input_metadata = _require_mapping(summary["input"], f"{role}.input")
    excluded = _require_mapping(
        input_metadata.get("quality_excluded_frames"), f"{role}.input.quality_excluded_frames"
    )
    requested = _frame_set(excluded.get("requested"), f"{role}.excluded.requested")
    observed = _frame_set(
        excluded.get("observed_and_removed"), f"{role}.excluded.observed_and_removed"
    )
    absent = _frame_set(
        excluded.get("absent_from_track_table"), f"{role}.excluded.absent_from_track_table"
    )
    if requested != EXPECTED_PARTIAL_FRAMES:
        raise ValueError(f"{role} must request all 16 declared partial-frame exclusions")
    if observed != EXPECTED_OBSERVED_PARTIAL_FRAMES or absent != (EXPECTED_ABSENT_PARTIAL_FRAME,):
        raise ValueError(
            f"{role} partial exclusions must contain 15 observed frames and absent frame 6747"
        )
    only_frames = _require_mapping(input_metadata.get("only_frames"), f"{role}.input.only_frames")
    only_requested = _frame_set(only_frames.get("requested"), f"{role}.only_frames.requested")
    only_observed = _frame_set(only_frames.get("observed"), f"{role}.only_frames.observed")
    only_absent = _frame_set(
        only_frames.get("absent_from_track_table"),
        f"{role}.only_frames.absent_from_track_table",
    )
    if only_requested or only_observed or only_absent:
        raise ValueError(f"{role} must not be an only-frame run")


def _validate_partial_only_selection(summary: Mapping[str, Any]) -> None:
    role = "partial_summary"
    input_metadata = _require_mapping(summary["input"], f"{role}.input")
    only_frames = _require_mapping(input_metadata.get("only_frames"), f"{role}.input.only_frames")
    requested = _frame_set(only_frames.get("requested"), f"{role}.only_frames.requested")
    observed = _frame_set(only_frames.get("observed"), f"{role}.only_frames.observed")
    absent = _frame_set(
        only_frames.get("absent_from_track_table"), f"{role}.only_frames.absent_from_track_table"
    )
    if requested != EXPECTED_PARTIAL_FRAMES:
        raise ValueError("partial-only run must request all 16 frozen partial frames")
    if observed != EXPECTED_OBSERVED_PARTIAL_FRAMES:
        raise ValueError("partial-only run must contain exactly the 15 observed partial frames")
    if absent != (EXPECTED_ABSENT_PARTIAL_FRAME,):
        raise ValueError("partial-only run must record frame 6747 as absent")
    excluded = _require_mapping(
        input_metadata.get("quality_excluded_frames"), f"{role}.input.quality_excluded_frames"
    )
    excluded_requested = _frame_set(excluded.get("requested"), f"{role}.excluded.requested")
    excluded_observed = _frame_set(
        excluded.get("observed_and_removed"), f"{role}.excluded.observed_and_removed"
    )
    excluded_absent = _frame_set(
        excluded.get("absent_from_track_table"), f"{role}.excluded.absent_from_track_table"
    )
    if excluded_requested or excluded_observed or excluded_absent:
        raise ValueError("partial-only run must select partial frames, not exclude them")
    if _integer(summary.get("selected_frame_count"), f"{role}.selected_frame_count") != 15:
        raise ValueError("partial-only run must select 15 observed frames")
    if _integer(summary.get("frame_step"), f"{role}.frame_step") != 1:
        raise ValueError("partial-only run must use frame_step=1")


def _extract_groups(
    summary: Mapping[str, Any],
    role: str,
    *,
    expected_z_modes: set[str],
) -> tuple[dict[tuple[str, float, str], dict[str, int | float]], tuple[float, ...]]:
    """Validate one full-record summary and return count-derived method groups."""

    _validate_run_selection(summary, role)
    config = _require_mapping(summary.get("config"), f"{role}.config")
    threshold = _integer(
        config.get("sparse_min_visible_rays"), f"{role}.config.sparse_min_visible_rays"
    )
    if threshold != EXPECTED_SPARSE_MIN_VISIBLE_RAYS:
        raise ValueError(f"{role} must use the post-development-calibrated 3-of-15 sparse rule")
    method_definitions = _require_mapping(
        config.get("method_definitions"), f"{role}.config.method_definitions"
    )
    if set(method_definitions) != set(METHOD_ORDER):
        raise ValueError(f"{role} method definitions must be exactly {list(METHOD_ORDER)}")
    sparse_definition = str(method_definitions["sparse_multiray"]).lower()
    if "3 of 15 rays" not in sparse_definition:
        raise ValueError(
            f"{role} does not document the post-development-calibrated 3-of-15 sparse rule"
        )

    z_modes = {str(value) for value in _require_list(config.get("z_modes"), f"{role}.z_modes")}
    if z_modes != expected_z_modes:
        raise ValueError(
            f"{role} z modes are {sorted(z_modes)}, expected {sorted(expected_z_modes)}"
        )
    fovs = tuple(
        sorted(
            _finite_float(value, f"{role}.config.fov_degrees")
            for value in _require_list(config.get("fov_degrees"), f"{role}.fov_degrees")
        )
    )
    if not fovs or len(fovs) != len(set(fovs)):
        raise ValueError(f"{role} must contain unique FOV values")

    groups = _require_list(summary.get("groups"), f"{role}.groups")
    extracted: dict[tuple[str, float, str], dict[str, int | float]] = {}
    observed_z_fov: set[tuple[str, float]] = set()
    for index, group_value in enumerate(groups):
        group = _require_mapping(group_value, f"{role}.groups[{index}]")
        z_mode = str(group.get("z_mode"))
        fov = _finite_float(group.get("fov_deg"), f"{role}.groups[{index}].fov_deg")
        if z_mode not in expected_z_modes or fov not in fovs:
            raise ValueError(f"{role} contains an unexpected z-mode/FOV group: {(z_mode, fov)}")
        if (z_mode, fov) in observed_z_fov:
            raise ValueError(f"{role} contains duplicate z-mode/FOV group {(z_mode, fov)}")
        observed_z_fov.add((z_mode, fov))
        counts = _require_mapping(group.get("counts"), f"{role}.groups[{index}].counts")
        stored_ratios = _require_mapping(
            group.get("observability_ratio_of_sums"),
            f"{role}.groups[{index}].observability_ratio_of_sums",
        )
        if set(stored_ratios) != set(METHOD_ORDER):
            raise ValueError(f"{role} method set must be exactly {list(METHOD_ORDER)}")
        denominator = _integer(counts.get("total_count"), f"{role}.{z_mode}.{fov}.total_count")
        if denominator <= 0:
            raise ValueError(f"{role}.{z_mode}.{fov} has a nonpositive denominator")
        for method, column in METHOD_NUMERATOR_COLUMNS.items():
            numerator = _integer(counts.get(column), f"{role}.{z_mode}.{fov}.{column}")
            if not 0 <= numerator <= denominator:
                raise ValueError(f"{role}.{z_mode}.{fov}.{method} has invalid counts")
            ratio = numerator / denominator
            stored = _finite_float(
                stored_ratios[method], f"{role}.{z_mode}.{fov}.{method}.stored_ratio"
            )
            if not math.isclose(ratio, stored, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{role}.{z_mode}.{fov}.{method} ratio is not count-derived")
            extracted[(z_mode, fov, method)] = {
                "numerator": numerator,
                "denominator": denominator,
                "ratio": ratio,
            }

    expected_z_fov = {(z_mode, fov) for z_mode in expected_z_modes for fov in fovs}
    if observed_z_fov != expected_z_fov:
        raise ValueError(f"{role} does not contain the complete z-mode by FOV product")
    return extracted, fovs


def _validate_bootstrap_replicates(
    replicates: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    fovs: tuple[float, ...],
) -> tuple[pd.DataFrame, int, int, tuple[int, ...]]:
    required = (
        "analysis_type",
        "estimand",
        "z_mode",
        "method",
        "fov_deg",
        "replicate",
        "ratio",
        "seed",
        "block_frame_count",
    )
    missing = sorted(set(required) - set(replicates.columns))
    if missing:
        raise ValueError(f"z bootstrap replicates missing columns: {missing}")
    single = replicates.loc[replicates["analysis_type"].eq("single_ratio"), list(required)].copy()
    if single.empty:
        raise ValueError("z bootstrap replicates contain no single_ratio rows")
    if single.isna().any().any():
        raise ValueError("single_ratio bootstrap rows contain null required values")
    if set(single["estimand"].astype(str)) != {"ratio_of_sums"}:
        raise ValueError("single_ratio rows must use the ratio_of_sums estimand")

    single["z_mode"] = single["z_mode"].astype(str)
    single["method"] = single["method"].astype(str)
    single["fov_deg"] = pd.to_numeric(single["fov_deg"], errors="raise").astype(float)
    single["replicate"] = pd.to_numeric(single["replicate"], errors="raise")
    single["ratio"] = pd.to_numeric(single["ratio"], errors="raise").astype(float)
    single["seed"] = pd.to_numeric(single["seed"], errors="raise")
    single["block_frame_count"] = pd.to_numeric(single["block_frame_count"], errors="raise")
    numeric_columns = ["fov_deg", "replicate", "ratio", "seed", "block_frame_count"]
    if not np.isfinite(single[numeric_columns].to_numpy(float)).all():
        raise ValueError("single_ratio bootstrap rows contain non-finite numeric values")
    for column in ("replicate", "seed", "block_frame_count"):
        if not np.equal(single[column], np.floor(single[column])).all():
            raise ValueError(f"single_ratio {column} values must be integers")
        single[column] = single[column].astype(np.int64)
    if not single["ratio"].between(0.0, 1.0).all():
        raise ValueError("single_ratio values must lie in [0, 1]")

    expected_groups = {
        (z_mode, method, fov)
        for z_mode in (GROUND_MODE, RAW_MODE)
        for method in methods
        for fov in fovs
    }
    actual_groups = set(single[["z_mode", "method", "fov_deg"]].itertuples(index=False, name=None))
    if actual_groups != expected_groups:
        raise ValueError("single_ratio z-mode/method/FOV groups do not match the summaries")
    if single.duplicated(["z_mode", "method", "fov_deg", "replicate"]).any():
        raise ValueError("single_ratio bootstrap keys are not unique")

    seeds = tuple(sorted(single["seed"].unique().tolist()))
    blocks = tuple(sorted(single["block_frame_count"].unique().tolist()))
    if len(seeds) != 1:
        raise ValueError("all paired z-mode bootstrap groups must use the same seed")
    if len(blocks) != 1 or blocks[0] <= 0:
        raise ValueError("all paired z-mode bootstrap groups must use one positive block length")

    replicate_sets: dict[tuple[str, str, float], tuple[int, ...]] = {}
    for key, group in single.groupby(["z_mode", "method", "fov_deg"], sort=True):
        replicate_sets[key] = tuple(sorted(group["replicate"].tolist()))
    collections = list(replicate_sets.values())
    if not collections[0] or any(values != collections[0] for values in collections[1:]):
        raise ValueError(
            "single_ratio replicate ID collections must be identical across z-mode/method/FOV"
        )
    return single, int(seeds[0]), int(blocks[0]), collections[0]


def _validate_primary_bootstrap(
    bootstrap: Mapping[str, Any],
    primary_groups: Mapping[tuple[str, float, str], Mapping[str, int | float]],
    *,
    fovs: tuple[float, ...],
) -> dict[tuple[float, str], dict[str, float]]:
    estimand = _require_mapping(bootstrap.get("estimand"), "primary_bootstrap.estimand")
    if estimand.get("name") != "ratio_of_sums":
        raise ValueError("primary bootstrap must use ratio_of_sums")
    design = _require_mapping(bootstrap.get("bootstrap"), "primary_bootstrap.bootstrap")
    if design.get("circular") is not True or not math.isclose(
        _finite_float(design.get("block_seconds"), "primary_bootstrap.block_seconds"),
        120.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("primary bootstrap must use circular nominal 120-second blocks")

    rows = _require_list(bootstrap.get("groups"), "primary_bootstrap.groups")
    intervals: dict[tuple[float, str], dict[str, float]] = {}
    for index, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"primary_bootstrap.groups[{index}]")
        z_mode = str(row.get("z_mode"))
        fov = _finite_float(row.get("fov_deg"), f"primary_bootstrap.groups[{index}].fov")
        method = str(row.get("method"))
        key = (z_mode, fov, method)
        if key not in primary_groups:
            raise ValueError(f"primary bootstrap has unexpected group {key}")
        if (fov, method) in intervals:
            raise ValueError(f"primary bootstrap duplicates group {(fov, method)}")
        numerator_sum = _finite_float(row.get("numerator_sum"), f"{key}.numerator_sum")
        denominator_sum = _finite_float(row.get("denominator_sum"), f"{key}.denominator_sum")
        primary = primary_groups[key]
        if not math.isclose(
            numerator_sum,
            float(primary["numerator"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"primary bootstrap numerator disagrees with summary for {key}")
        if not math.isclose(
            denominator_sum,
            float(primary["denominator"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"primary bootstrap denominator disagrees with summary for {key}")
        estimate = _finite_float(row.get("estimate"), f"{key}.estimate")
        if not math.isclose(estimate, float(primary["ratio"]), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"primary bootstrap estimate disagrees with summary for {key}")
        ci_low = _finite_float(row.get("ci_low"), f"{key}.ci_low")
        ci_high = _finite_float(row.get("ci_high"), f"{key}.ci_high")
        if not 0.0 <= ci_low <= estimate <= ci_high <= 1.0:
            raise ValueError(f"primary bootstrap interval is invalid for {key}")
        intervals[(fov, method)] = {"ci_low": ci_low, "ci_high": ci_high}
    expected = {(fov, method) for fov in fovs for method in METHOD_ORDER}
    if set(intervals) != expected:
        raise ValueError("primary bootstrap method/FOV set does not match primary summary")
    return intervals


def summarize_cacie_sensitivities(
    primary_summary: Mapping[str, Any],
    z_summary: Mapping[str, Any],
    z_bootstrap_replicates: pd.DataFrame,
    partial_summary: Mapping[str, Any],
    *,
    primary_bootstrap: Mapping[str, Any] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Build the count-derived, strictly paired CACIE sensitivity summary."""

    confidence = _finite_float(confidence_level, "confidence_level")
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the sensitivity interval must use 95% confidence")

    primary_groups, primary_fovs = _extract_groups(
        primary_summary, "primary_summary", expected_z_modes={GROUND_MODE}
    )
    z_groups, z_fovs = _extract_groups(
        z_summary, "z_summary", expected_z_modes={GROUND_MODE, RAW_MODE}
    )
    partial_groups, partial_fovs = _extract_groups(
        partial_summary, "partial_summary", expected_z_modes={GROUND_MODE}
    )
    if not (primary_fovs == z_fovs == partial_fovs):
        raise ValueError("primary, z-mode, and partial summaries must have identical FOV sets")
    if _integer(primary_summary.get("frame_step"), "primary_summary.frame_step") != 1:
        raise ValueError("primary full-record summary must use frame_step=1")
    if _integer(z_summary.get("frame_step"), "z_summary.frame_step") != 10:
        raise ValueError("z-mode sensitivity summary must use frame_step=10")
    _validate_primary_or_step10_exclusions(primary_summary, "primary_summary")
    _validate_primary_or_step10_exclusions(z_summary, "z_summary")
    _validate_partial_only_selection(partial_summary)

    single, bootstrap_seed, block_frames, replicate_ids = _validate_bootstrap_replicates(
        z_bootstrap_replicates,
        methods=METHOD_ORDER,
        fovs=primary_fovs,
    )
    alpha = (1.0 - confidence) / 2.0

    z_comparisons: list[dict[str, Any]] = []
    decimation_comparisons: list[dict[str, Any]] = []
    partial_comparisons: list[dict[str, Any]] = []
    primary_intervals = (
        _validate_primary_bootstrap(
            primary_bootstrap,
            primary_groups,
            fovs=primary_fovs,
        )
        if primary_bootstrap is not None
        else {}
    )

    for fov in primary_fovs:
        for method in METHOD_ORDER:
            raw = z_groups[(RAW_MODE, fov, method)]
            step_ground = z_groups[(GROUND_MODE, fov, method)]
            full_ground = primary_groups[(GROUND_MODE, fov, method)]
            partial = partial_groups[(GROUND_MODE, fov, method)]
            if raw["denominator"] != step_ground["denominator"]:
                raise ValueError(
                    "raw and ground-anchored step10 counts must describe the same rows "
                    f"for {(method, fov)}"
                )

            raw_draws = single.loc[
                single["z_mode"].eq(RAW_MODE)
                & single["method"].eq(method)
                & single["fov_deg"].eq(fov),
                ["replicate", "ratio"],
            ].rename(columns={"ratio": "raw_ratio"})
            ground_draws = single.loc[
                single["z_mode"].eq(GROUND_MODE)
                & single["method"].eq(method)
                & single["fov_deg"].eq(fov),
                ["replicate", "ratio"],
            ].rename(columns={"ratio": "ground_ratio"})
            paired = raw_draws.merge(
                ground_draws,
                on="replicate",
                how="inner",
                validate="one_to_one",
            ).sort_values("replicate", kind="stable")
            if tuple(paired["replicate"].tolist()) != replicate_ids:
                raise ValueError(f"raw/ground replicate pairing failed for {(method, fov)}")
            differences = paired["raw_ratio"].to_numpy() - paired["ground_ratio"].to_numpy()
            ci_low, ci_high = np.quantile(differences, [alpha, 1.0 - alpha])
            point_delta = float(raw["ratio"]) - float(step_ground["ratio"])
            z_comparisons.append(
                {
                    "fov_deg": fov,
                    "method": method,
                    "raw": dict(raw),
                    "ground_anchored": dict(step_ground),
                    "delta_raw_minus_ground_anchored": point_delta,
                    "delta_percentage_points": point_delta * 100.0,
                    "bootstrap": {
                        "pairing": "same method, FOV, seed, and replicate ID",
                        "n_paired_replicates": int(len(differences)),
                        "confidence_level": confidence,
                        "interval": "two-sided equal-tail percentile",
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "ci_low_percentage_points": float(ci_low) * 100.0,
                        "ci_high_percentage_points": float(ci_high) * 100.0,
                    },
                }
            )

            decimation_delta = float(step_ground["ratio"]) - float(full_ground["ratio"])
            decimation_entry: dict[str, Any] = {
                "fov_deg": fov,
                "method": method,
                "step10_ground_anchored": dict(step_ground),
                "full_primary_ground_anchored": dict(full_ground),
                "delta_step10_minus_full_primary": decimation_delta,
                "delta_percentage_points": decimation_delta * 100.0,
                "inference": "point comparison only; no paired decimation CI is constructed",
            }
            if (fov, method) in primary_intervals:
                decimation_entry["full_primary_bootstrap_ci"] = primary_intervals[(fov, method)]
            decimation_comparisons.append(decimation_entry)

            combined_numerator = int(full_ground["numerator"]) + int(partial["numerator"])
            combined_denominator = int(full_ground["denominator"]) + int(partial["denominator"])
            combined_ratio = combined_numerator / combined_denominator
            partial_delta = combined_ratio - float(full_ground["ratio"])
            partial_comparisons.append(
                {
                    "fov_deg": fov,
                    "method": method,
                    "primary_excluding_partial": dict(full_ground),
                    "partial_only": dict(partial),
                    "include_partial": {
                        "numerator": combined_numerator,
                        "denominator": combined_denominator,
                        "ratio": combined_ratio,
                    },
                    "delta_include_partial_minus_primary": partial_delta,
                    "delta_percentage_points": partial_delta * 100.0,
                    "inference": "deterministic count recombination; no independence-based CI",
                }
            )

    return {
        "schema_version": "1.0",
        "analysis": "cacie_frozen_sensitivity_summary",
        "contracts": {
            "sparse_rule": "at least 3 of 15 rays visible",
            "sparse_rule_selection_stage": "post-development calibration",
            "sparse_min_visible_rays": EXPECTED_SPARSE_MIN_VISIBLE_RAYS,
            "methods": list(METHOD_ORDER),
            "fov_degrees": list(primary_fovs),
            "primary_frame_step": 1,
            "z_mode_sensitivity_frame_step": 10,
            "partial_frames_requested": list(EXPECTED_PARTIAL_FRAMES),
            "partial_frames_observed": list(EXPECTED_OBSERVED_PARTIAL_FRAMES),
            "partial_frames_absent": [EXPECTED_ABSENT_PARTIAL_FRAME],
            "point_estimand": "ratio_of_sums = sum(numerator) / sum(denominator)",
        },
        "z_mode_step10": {
            "delta_direction": "raw - ground_anchored",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_block_frame_count": block_frames,
            "bootstrap_replicate_ids": {
                "count": len(replicate_ids),
                "first": replicate_ids[0],
                "last": replicate_ids[-1],
                "identical_across_all_groups": True,
            },
            "comparisons": z_comparisons,
        },
        "decimation_step10_vs_full": {
            "delta_direction": "step10 ground_anchored - full primary ground_anchored",
            "primary_bootstrap_status": "validated"
            if primary_bootstrap is not None
            else "not supplied",
            "comparisons": decimation_comparisons,
        },
        "partial_scan_inclusion": {
            "delta_direction": "include partial scans - primary excluding partial scans",
            "combination": "add primary and partial-only numerator/denominator counts before division",
            "comparisons": partial_comparisons,
        },
        "limitations": [
            "The raw/ground interval describes the declared every-10th-frame sensitivity run.",
            "The decimation comparison is reported without an invented paired confidence interval.",
            "Partial-scan inclusion is a deterministic count sensitivity, not a new independent sample.",
        ],
    }


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"{name} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must contain a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-summary", type=Path, required=True)
    parser.add_argument("--primary-bootstrap", type=Path)
    parser.add_argument("--z-summary", type=Path, required=True)
    parser.add_argument("--z-bootstrap-replicates", type=Path, required=True)
    parser.add_argument("--partial-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    primary_summary = _load_json(args.primary_summary, "--primary-summary")
    z_summary = _load_json(args.z_summary, "--z-summary")
    partial_summary = _load_json(args.partial_summary, "--partial-summary")
    if not args.z_bootstrap_replicates.is_file():
        raise SystemExit(f"--z-bootstrap-replicates does not exist: {args.z_bootstrap_replicates}")
    replicates = pd.read_parquet(args.z_bootstrap_replicates)
    primary_bootstrap = (
        _load_json(args.primary_bootstrap, "--primary-bootstrap")
        if args.primary_bootstrap is not None
        else None
    )
    result = summarize_cacie_sensitivities(
        primary_summary,
        z_summary,
        replicates,
        partial_summary,
        primary_bootstrap=primary_bootstrap,
    )
    result["inputs"] = {
        "primary_summary": _receipt(args.primary_summary),
        "primary_bootstrap": (
            _receipt(args.primary_bootstrap) if args.primary_bootstrap is not None else None
        ),
        "z_summary": _receipt(args.z_summary),
        "z_bootstrap_replicates": _receipt(args.z_bootstrap_replicates),
        "partial_summary": _receipt(args.partial_summary),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "z_mode_comparisons": len(result["z_mode_step10"]["comparisons"]),
                "decimation_comparisons": len(result["decimation_step10_vs_full"]["comparisons"]),
                "partial_scan_comparisons": len(result["partial_scan_inclusion"]["comparisons"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
