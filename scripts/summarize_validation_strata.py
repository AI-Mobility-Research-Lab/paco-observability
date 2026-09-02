#!/usr/bin/env python3
"""Re-summarize saved occlusion-validation decisions without tracing new rays.

All agreement metrics in this module are calculated from discrete covered-row
confusion counts.  The output is descriptive; it deliberately does not attach
independence-based confidence intervals to the repeated frame/ego decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from paco_observability.derivation_receipt import (
    require_named_files_unchanged,
    snapshot_named_files,
    write_derivation_receipt,
)


RANGE_BANDS: tuple[tuple[float, float, bool, str], ...] = (
    (1.5, 10.0, False, "[1.5,10)"),
    (10.0, 20.0, False, "[10,20)"),
    (20.0, 35.0, True, "[20,35]"),
)

METHOD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("legacy_planar", "legacy_visible"),
    ("center_top", "center_top_visible"),
    ("sparse_multiray", "sparse_multiray_visible"),
)

METRIC_COLUMNS: tuple[str, ...] = (
    "stratum",
    "stratum_value",
    "z_mode",
    "fov_deg",
    "method",
    "n",
    "exact_visible_n",
    "exact_hidden_n",
    "predicted_visible_n",
    "predicted_hidden_n",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "balanced_accuracy",
    "precision_visible",
    "recall_visible",
    "f1_visible",
    "precision_hidden",
    "recall_hidden",
    "f1_hidden",
    "false_visible_count_given_exact_hidden",
    "false_visible_rate_given_exact_hidden",
)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a count ratio, leaving an empty denominator explicitly undefined."""

    return float(numerator / denominator) if denominator else None


def confusion_metrics(
    predicted_visible: Iterable[bool], exact_visible: Iterable[bool]
) -> dict[str, int | float | None]:
    """Calculate two-class metrics directly from covered decision-row counts."""

    predicted = np.asarray(list(predicted_visible), dtype=bool)
    reference = np.asarray(list(exact_visible), dtype=bool)
    if predicted.shape != reference.shape:
        raise ValueError("predicted and exact labels must have identical shapes")
    if predicted.size == 0:
        raise ValueError("at least one covered decision is required")

    tp = int(np.count_nonzero(predicted & reference))
    tn = int(np.count_nonzero(~predicted & ~reference))
    fp = int(np.count_nonzero(predicted & ~reference))
    fn = int(np.count_nonzero(~predicted & reference))
    exact_visible_n = tp + fn
    exact_hidden_n = tn + fp

    precision_visible = _safe_ratio(tp, tp + fp)
    recall_visible = _safe_ratio(tp, exact_visible_n)
    precision_hidden = _safe_ratio(tn, tn + fn)
    recall_hidden = _safe_ratio(tn, exact_hidden_n)
    available_recalls = [value for value in (recall_visible, recall_hidden) if value is not None]

    return {
        "n": int(predicted.size),
        "exact_visible_n": exact_visible_n,
        "exact_hidden_n": exact_hidden_n,
        "predicted_visible_n": tp + fp,
        "predicted_hidden_n": tn + fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / predicted.size),
        "balanced_accuracy": (
            float(sum(available_recalls) / 2.0) if len(available_recalls) == 2 else None
        ),
        "precision_visible": precision_visible,
        "recall_visible": recall_visible,
        "f1_visible": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "precision_hidden": precision_hidden,
        "recall_hidden": recall_hidden,
        "f1_hidden": _safe_ratio(2 * tn, 2 * tn + fp + fn),
        "false_visible_count_given_exact_hidden": fp,
        "false_visible_rate_given_exact_hidden": _safe_ratio(fp, exact_hidden_n),
    }


def _as_strict_bool(series: pd.Series, column: str) -> pd.Series:
    """Coerce bool or 0/1 data without accepting truthy strings such as ``"False"``."""

    if series.isna().any():
        raise ValueError(f"{column} contains null labels")
    if is_bool_dtype(series.dtype):
        return series.astype(bool)
    if is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="raise")
        if numeric.isin([0, 1]).all():
            return numeric.astype(bool)
    raise ValueError(f"{column} must contain only booleans or numeric 0/1 labels")


def _require_columns(frame: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def _range_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    for lower, upper, upper_inclusive, label in RANGE_BANDS:
        selected = numeric.ge(lower) & (numeric.le(upper) if upper_inclusive else numeric.lt(upper))
        result.loc[selected] = label
    return result


def _density_quartiles(decisions: pd.DataFrame) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Assign observed frames to up to four equal-frequency density groups."""

    density = decisions.groupby("frame_idx", sort=True)["target_id"].nunique().astype(int)
    if density.empty:
        raise ValueError("decisions contain no observed frames")
    n_quantiles = min(4, len(density))
    # First-ranked ties make assignment deterministic and keep quartile sizes
    # balanced.  The underlying target count is retained in the output so a
    # tie split remains transparent.
    codes = pd.qcut(density.rank(method="first"), q=n_quantiles, labels=False)
    labels = codes.astype(int).map(lambda value: f"Q{value + 1}")
    metadata: list[dict[str, Any]] = []
    for label in sorted(labels.unique()):
        values = density.loc[labels.eq(label)]
        metadata.append(
            {
                "quartile": str(label),
                "frame_count": int(len(values)),
                "min_unique_targets": int(values.min()),
                "max_unique_targets": int(values.max()),
            }
        )
    return labels.astype("string"), metadata


def prepare_covered_decisions(
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str]], list[dict[str, Any]]]:
    """Validate the decision table, filter covered rows, and attach strata."""

    if decisions.empty:
        raise ValueError("decisions table is empty")
    required = {
        "frame_idx",
        "ego_id",
        "target_id",
        "class_name",
        "approach",
        "z_mode",
        "fov_deg",
        "range_m",
        "covered",
        "exact_visible",
        "legacy_visible",
        "center_top_visible",
    }
    _require_columns(decisions, required, "decisions")

    methods = [(method, column) for method, column in METHOD_COLUMNS if column in decisions]
    for column in ("covered", "exact_visible", *(column for _, column in methods)):
        decisions[column] = _as_strict_bool(decisions[column], column)

    group_columns = ["frame_idx", "ego_id", "target_id", "class_name", "approach", "z_mode"]
    if decisions[group_columns].isna().any().any():
        columns = decisions[group_columns].columns[decisions[group_columns].isna().any()].tolist()
        raise ValueError(f"decisions contain null grouping values in: {columns}")
    for column in ("fov_deg", "range_m"):
        numeric = pd.to_numeric(decisions[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError(f"{column} must contain finite numeric values")
        decisions[column] = numeric.astype(float)

    decision_key = ["frame_idx", "ego_id", "z_mode", "fov_deg", "target_id"]
    if decisions.duplicated(decision_key).any():
        duplicate = decisions.loc[decisions.duplicated(decision_key, keep=False), decision_key].head(1)
        raise ValueError(f"duplicate decision key: {duplicate.to_dict(orient='records')[0]}")

    density_labels, density_metadata = _density_quartiles(decisions)
    density_counts = decisions.groupby("frame_idx")["target_id"].nunique().astype(int)

    covered = decisions.loc[decisions["covered"]].copy()
    if covered.empty:
        raise ValueError("decisions contain no covered rows")
    covered["range_band"] = _range_band(covered["range_m"])
    if covered["range_band"].isna().any():
        bad = sorted(covered.loc[covered["range_band"].isna(), "range_m"].unique().tolist())
        raise ValueError(f"covered rows outside declared range bands [1.5,35]: {bad[:10]}")
    covered["frame_density_unique_targets"] = covered["frame_idx"].map(density_counts).astype(int)
    covered["frame_density_quartile"] = covered["frame_idx"].map(density_labels).astype("string")
    return covered, methods, density_metadata


def _append_metric_groups(
    rows: list[dict[str, Any]],
    covered: pd.DataFrame,
    methods: Sequence[tuple[str, str]],
    *,
    stratum: str,
    stratum_column: str | None,
) -> None:
    keys = ["z_mode", "fov_deg"] + ([stratum_column] if stratum_column else [])
    for group_key, group in covered.groupby(keys, sort=True, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        z_mode, fov_deg = group_key[:2]
        stratum_value = str(group_key[2]) if stratum_column else "all"
        for method, prediction_column in methods:
            metrics = confusion_metrics(group[prediction_column], group["exact_visible"])
            rows.append(
                {
                    "stratum": stratum,
                    "stratum_value": stratum_value,
                    "z_mode": str(z_mode),
                    "fov_deg": float(fov_deg),
                    "method": method,
                    **metrics,
                }
            )


def summarize_decisions(
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[tuple[str, str]]]:
    """Build overall/stratified metrics and optional blocker-class counts."""

    # Work on a copy because validation includes strict dtype normalization.
    source = decisions.copy()
    covered, methods, density_metadata = prepare_covered_decisions(source)
    rows: list[dict[str, Any]] = []
    _append_metric_groups(rows, covered, methods, stratum="overall", stratum_column=None)
    for name, column in (
        ("target_class", "class_name"),
        ("range_band", "range_band"),
        ("approach", "approach"),
        ("frame_density_quartile", "frame_density_quartile"),
    ):
        _append_metric_groups(rows, covered, methods, stratum=name, stratum_column=column)
    metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    metrics = metrics.sort_values(
        ["stratum", "stratum_value", "z_mode", "fov_deg", "method"], kind="stable"
    ).reset_index(drop=True)
    blockers = summarize_blocker_classes(source, covered)
    return metrics, blockers, density_metadata, methods


def _normalized_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value).is_integer():
            return str(int(value))
        return repr(float(value))
    return str(value)


def summarize_blocker_classes(decisions: pd.DataFrame, covered: pd.DataFrame) -> pd.DataFrame:
    """Resolve blocker IDs to per-frame target classes when those columns exist."""

    output_columns = ["model", "z_mode", "fov_deg", "blocker_class", "blocked_rows"]
    specifications = [
        ("legacy_planar", "legacy_visible", "legacy_occluded_by"),
        ("center_top", "center_top_visible", "center_top_occluded_by"),
        ("sparse_multiray", "sparse_multiray_visible", "sparse_multiray_occluded_by"),
        ("exact_reference", "exact_visible", "exact_occluded_by"),
    ]
    available = [item for item in specifications if {item[1], item[2]}.issubset(covered.columns)]
    if not available:
        return pd.DataFrame(columns=output_columns)

    lookup_source = decisions[["frame_idx", "target_id", "class_name"]].drop_duplicates()
    lookup_source["_target_key"] = lookup_source["target_id"].map(_normalized_id)
    ambiguity = lookup_source.groupby(["frame_idx", "_target_key"])["class_name"].nunique()
    ambiguous_keys = set(ambiguity.loc[ambiguity.gt(1)].index.tolist())
    lookup: dict[tuple[Any, str], str] = {}
    lookup_rows = lookup_source[["frame_idx", "_target_key", "class_name"]]
    for frame_idx, target_key, class_name in lookup_rows.itertuples(index=False, name=None):
        key = (frame_idx, target_key)
        if key not in ambiguous_keys:
            lookup[key] = str(class_name)

    rows: list[pd.DataFrame] = []
    for model, visible_column, blocker_column in available:
        hidden = covered.loc[
            ~covered[visible_column], ["frame_idx", "z_mode", "fov_deg", blocker_column]
        ].copy()
        if hidden.empty:
            continue

        def resolve(row: pd.Series) -> str:
            blocker_key = _normalized_id(row[blocker_column])
            if blocker_key is None:
                return "__missing_blocker_id__"
            return lookup.get((row["frame_idx"], blocker_key), "__unlinked__")

        hidden["blocker_class"] = hidden.apply(resolve, axis=1)
        counts = (
            hidden.groupby(["z_mode", "fov_deg", "blocker_class"], sort=True)
            .size()
            .rename("blocked_rows")
            .reset_index()
        )
        counts.insert(0, "model", model)
        rows.append(counts)
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(rows, ignore_index=True)[output_columns].sort_values(
        ["model", "z_mode", "fov_deg", "blocker_class"], kind="stable"
    ).reset_index(drop=True)


def summarize_timings(timings: pd.DataFrame) -> pd.DataFrame:
    """Summarize saved per-call runtimes; no geometry is re-evaluated."""

    if timings.empty:
        raise ValueError("timings table is empty")
    _require_columns(timings, {"method", "elapsed_ms"}, "timings")
    elapsed = pd.to_numeric(timings["elapsed_ms"], errors="coerce")
    if elapsed.isna().any() or not np.isfinite(elapsed.to_numpy(float)).all() or elapsed.lt(0).any():
        raise ValueError("elapsed_ms must contain finite non-negative values")
    work = timings.copy()
    work["elapsed_ms"] = elapsed.astype(float)
    grouping = [column for column in ("method", "grid", "z_mode", "fov_deg") if column in work]

    rows: list[dict[str, Any]] = []
    for group_key, group in work.groupby(grouping, sort=True, dropna=False, observed=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(grouping, group_key)}
        values = group["elapsed_ms"].to_numpy(float)
        row.update(
            {
                "n_calls": int(values.size),
                "total_ms": float(values.sum()),
                "mean_ms": float(values.mean()),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.quantile(values, 0.95)),
            }
        )
        if "hit_rays" in group and group["hit_rays"].notna().any():
            hit_rays = pd.to_numeric(group["hit_rays"], errors="coerce").dropna().to_numpy(float)
            row["hit_rays_total"] = float(hit_rays.sum())
            row["hit_rays_mean"] = float(hit_rays.mean())
        else:
            row["hit_rays_total"] = None
            row["hit_rays_mean"] = None
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(grouping, kind="stable", na_position="last").reset_index(drop=True)


def _json_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def summarize_validation_strata(
    decisions: pd.DataFrame,
    timings: pd.DataFrame,
    *,
    declared_sampled_frames: int = 200,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return JSON-ready metadata and all three reusable summary tables."""

    if declared_sampled_frames <= 0:
        raise ValueError("declared_sampled_frames must be positive")
    metrics, blockers, density_metadata, methods = summarize_decisions(decisions)
    timing_summary = summarize_timings(timings)
    observed_frames = int(decisions["frame_idx"].nunique())

    blocker_models = sorted(blockers["model"].unique().tolist()) if not blockers.empty else []
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "study_scope": {
            "declared_sampled_frames": int(declared_sampled_frames),
            "observed_sampled_frames": observed_frames,
            "sample_size_matches_declaration": observed_frames == declared_sampled_frames,
            "interpretation": (
                f"Descriptive validation on {declared_sampled_frames} sampled frames; repeated "
                "frame/ego/target decisions are not treated as independent observations and no "
                "independence-based confidence intervals are reported."
            ),
            "reference": (
                "Analytic ray--OBB visibility within the detected dynamic OBB abstraction; "
                "this is a computational reference, not physical visibility ground truth."
            ),
            "analysis_population": "Rows with covered == true only.",
        },
        "metric_definition": {
            "positive_class": "visible",
            "negative_class": "hidden",
            "calculation": "All ratios and confusion metrics are derived from decision-row counts.",
            "false_visible_rate_given_exact_hidden": "FP / (FP + TN)",
            "range_bands_m": [label for _, _, _, label in RANGE_BANDS],
            "frame_density": (
                "Per-frame unique target_id count; deterministic equal-frequency quartiles over "
                "observed frames (first-ranked tie handling)."
            ),
        },
        "methods": [method for method, _ in methods],
        "decision_rows": int(len(decisions)),
        "covered_decision_rows": int(_as_strict_bool(decisions["covered"], "covered").sum()),
        "frame_density_quartiles": density_metadata,
        "blocker_class_linkage": {
            "status": "available" if blocker_models else "unavailable",
            "models": blocker_models,
            "note": (
                "Classes are joined by frame_idx and blocker target ID. Missing or unresolved IDs "
                "are retained as explicit sentinel categories."
                if blocker_models
                else "Required blocker-ID columns were unavailable or contained no hidden rows."
            ),
        },
        "metrics": _records_for_json(metrics),
        "timings": _records_for_json(timing_summary),
        "blocker_classes": _records_for_json(blockers),
    }
    return summary, metrics, timing_summary, blockers


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-summarize saved validation decisions and timings without tracing rays"
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--declared-sampled-frames", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_files = {
        "calibrated_decisions": args.decisions,
        "main_validation_timings": args.timings,
    }
    input_records = snapshot_named_files(input_files, path_base=args.out_dir)
    decisions = pd.read_parquet(args.decisions)
    timings = pd.read_parquet(args.timings)
    summary, metrics, timing_summary, blockers = summarize_validation_strata(
        decisions,
        timings,
        declared_sampled_frames=args.declared_sampled_frames,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(args.out_dir / "validation_strata_metrics.parquet", index=False)
    timing_summary.to_parquet(args.out_dir / "validation_timing_summary.parquet", index=False)
    blockers.to_parquet(args.out_dir / "validation_blocker_classes.parquet", index=False)
    (args.out_dir / "validation_strata_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    require_named_files_unchanged(input_records, input_files, path_base=args.out_dir)
    write_derivation_receipt(
        receipt_path=args.out_dir / "source_lineage_receipt.json",
        derivation="validation_strata_calibrated",
        input_records=input_records,
        output_files={
            "summary": args.out_dir / "validation_strata_summary.json",
            "metrics": args.out_dir / "validation_strata_metrics.parquet",
            "timings": args.out_dir / "validation_timing_summary.parquet",
            "blockers": args.out_dir / "validation_blocker_classes.parquet",
        },
    )
    print(
        json.dumps(
            {
                "metrics_rows": int(len(metrics)),
                "timing_rows": int(len(timing_summary)),
                "blocker_rows": int(len(blockers)),
                "out_dir": str(args.out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
