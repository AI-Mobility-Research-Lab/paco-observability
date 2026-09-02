#!/usr/bin/env python3
"""Compute full-record PACO counts with bounded multiprocessing and streaming output.

The wide output contains one row per valid ``frame x ego x z-mode x FOV``.
``observability_contributions.parquet`` expands each wide row into the three
method-specific numerator/denominator rows consumed by
``bootstrap_observability.py``.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from paco_observability.full_record import (
    CONTRIBUTION_COLUMNS,
    COUNT_COLUMNS,
    EXCLUSION_COLUMNS,
    TIMING_COLUMNS,
    FullRecordConfig,
    FullRecordSummaryAccumulator,
    audit_full_record_tracks,
    config_as_dict,
    contribution_rows,
    iter_full_record,
    require_passing_audit,
    select_frame_ids,
)
from paco_observability.geometry import GroundPlane
from paco_observability.provenance import sha256_file
from paco_observability.validation import default_ego_grid


OUTPUT_NAMES = {
    "counts": "full_record_observability.parquet",
    "contributions": "observability_contributions.parquet",
    "exclusions": "ego_exclusions.parquet",
    "timings": "full_record_timings.parquet",
    "summary": "full_record_summary.json",
    "config": "full_record_config.json",
    "audit": "full_record_input_audit.json",
}


COUNT_SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int64()),
        ("frame_order", pa.int64()),
        ("ego_index", pa.int64()),
        ("ego_id", pa.string()),
        ("approach", pa.string()),
        ("ego_x", pa.float64()),
        ("ego_y", pa.float64()),
        ("ego_heading_deg", pa.float64()),
        ("sensor_x", pa.float64()),
        ("sensor_y", pa.float64()),
        ("sensor_z", pa.float64()),
        ("z_mode", pa.string()),
        ("fov_deg", pa.float64()),
        ("legacy_theta_bins", pa.int64()),
        ("total_count", pa.int64()),
        ("covered_count", pa.int64()),
        ("legacy_visible_count", pa.int64()),
        ("center_top_visible_count", pa.int64()),
        ("sparse_multiray_visible_count", pa.int64()),
        ("out_of_range_count", pa.int64()),
        ("out_of_fov_count", pa.int64()),
        ("both_out_of_range_and_fov_count", pa.int64()),
        ("legacy_occluded_count", pa.int64()),
        ("center_top_occluded_count", pa.int64()),
        ("sparse_multiray_occluded_count", pa.int64()),
        ("vru_count", pa.int64()),
        ("weighted_total_demand", pa.float64()),
        ("legacy_residual_demand", pa.float64()),
        ("center_top_residual_demand", pa.float64()),
        ("sparse_multiray_residual_demand", pa.float64()),
        ("vru_legacy_residual_count", pa.int64()),
        ("vru_center_top_residual_count", pa.int64()),
        ("vru_sparse_multiray_residual_count", pa.int64()),
    ]
)

CONTRIBUTION_SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int64()),
        ("frame_order", pa.int64()),
        ("ego_index", pa.int64()),
        ("ego_id", pa.string()),
        ("approach", pa.string()),
        ("z_mode", pa.string()),
        ("fov_deg", pa.float64()),
        ("method", pa.string()),
        ("numerator", pa.int64()),
        ("denominator", pa.int64()),
    ]
)

EXCLUSION_SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int64()),
        ("frame_order", pa.int64()),
        ("ego_index", pa.int64()),
        ("ego_id", pa.string()),
        ("approach", pa.string()),
        ("ego_x", pa.float64()),
        ("ego_y", pa.float64()),
        ("reason", pa.string()),
        ("nearest_track_id", pa.int64()),
        ("nearest_distance_m", pa.float64()),
        ("clearance_m", pa.float64()),
    ]
)

TIMING_SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int64()),
        ("frame_order", pa.int64()),
        ("objects", pa.int64()),
        ("ego_scenes_requested", pa.int64()),
        ("ego_scenes_evaluated", pa.int64()),
        ("ego_scenes_excluded", pa.int64()),
        ("count_rows", pa.int64()),
        ("box_build_ms", pa.float64()),
        ("legacy_planar_ms", pa.float64()),
        ("center_top_ms", pa.float64()),
        ("sparse_multiray_ms", pa.float64()),
        ("frame_total_ms", pa.float64()),
    ]
)


def _verify_schema_columns(schema: pa.Schema, expected: Sequence[str], *, name: str) -> None:
    if tuple(schema.names) != tuple(expected):
        raise RuntimeError(f"internal {name} schema is out of sync with the library columns")


class BufferedParquetWriter:
    """Append dictionaries to a fixed-schema Parquet file in bounded row groups."""

    def __init__(self, path: Path, schema: pa.Schema, *, buffer_rows: int) -> None:
        if buffer_rows < 1:
            raise ValueError("buffer_rows must be positive")
        self.path = path
        self.schema = schema
        self.buffer_rows = int(buffer_rows)
        self._rows: list[Mapping[str, Any]] = []
        self._writer = pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        self.rows_written = 0
        self.closed = False

    def append(self, rows: Iterable[Mapping[str, Any]]) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed Parquet writer")
        for row in rows:
            self._rows.append(row)
            if len(self._rows) >= self.buffer_rows:
                self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=self.schema)
        self._writer.write_table(table, row_group_size=len(table))
        self.rows_written += len(table)
        self._rows.clear()

    def close(self) -> None:
        if self.closed:
            return
        self.flush()
        self._writer.close()
        self.closed = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--ground-plane",
        type=float,
        nargs=4,
        metavar=("NX", "NY", "NZ", "D"),
        required=True,
        help="Plane coefficients satisfying n dot p + d = 0.",
    )
    parser.add_argument(
        "--expected-frame-range",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="Expected inclusive frame range; required to audit missing endpoints.",
    )
    parser.add_argument(
        "--allow-missing-frame",
        type=int,
        action="append",
        default=[],
        help="Explicitly justified missing frame; repeat for multiple frames.",
    )
    parser.add_argument(
        "--exclude-frame",
        type=int,
        action="append",
        default=[],
        help=(
            "Frame excluded from computation by a documented quality rule; repeat for "
            "multiple frames. This is distinct from allowing a missing track-table frame."
        ),
    )
    parser.add_argument(
        "--only-frame",
        type=int,
        action="append",
        default=[],
        help=(
            "After auditing the complete input, compute only this observed frame; repeat "
            "for a targeted sensitivity run."
        ),
    )
    parser.add_argument(
        "--frame-exclusion-reason",
        default="partial_scan_point_count_ratio_below_0.6_of_local_median",
        help="Audit label stored with every --exclude-frame ID.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--n-processes", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--buffer-rows", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--n-ego-per-approach", type=int, default=15)
    parser.add_argument(
        "--fov-deg",
        type=float,
        nargs="+",
        choices=(120.0, 360.0),
        default=(120.0, 360.0),
    )
    parser.add_argument(
        "--z-modes",
        choices=("raw", "ground_anchored"),
        nargs="+",
        default=("raw", "ground_anchored"),
    )
    parser.add_argument("--range-m", type=float, default=35.0)
    parser.add_argument("--near-blind-m", type=float, default=1.5)
    parser.add_argument("--sensor-forward-offset-m", type=float, default=1.5)
    parser.add_argument("--sensor-height-m", type=float, default=1.8)
    parser.add_argument("--ego-clearance-m", type=float, default=4.0)
    parser.add_argument("--occlusion-clearance-m", type=float, default=0.0)
    parser.add_argument(
        "--sparse-min-visible-rays",
        type=int,
        default=3,
        help=(
            "Minimum clear rays among the 15-point sparse sampler; canonical calibrated "
            "value is 3."
        ),
    )
    parser.add_argument("--vru-weight", type=float, default=3.0)
    parser.add_argument("--non-vru-weight", type=float, default=1.0)
    parser.add_argument(
        "--vru-classes",
        nargs="+",
        default=("bicycle", "pedestrian"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _temporary_paths(out_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    token = uuid.uuid4().hex
    final = {key: out_dir / name for key, name in OUTPUT_NAMES.items()}
    temporary = {
        key: path.with_name(f".{path.name}.{token}.partial") for key, path in final.items()
    }
    return final, temporary


def _invalidate_existing_completion_marker(summary_path: Path) -> Path | None:
    """Atomically quarantine an old completion marker before an overwrite run.

    A failed or interrupted overwrite must not leave the previous summary at the
    canonical path, where it could be mistaken for proof that the newly replaced
    Parquet files form a complete run.  The stale copy is kept until the new
    summary is published successfully.
    """

    if not summary_path.exists():
        return None
    stale_path = summary_path.with_name(
        f"{summary_path.name}.{uuid.uuid4().hex}.stale"
    )
    os.replace(summary_path, stale_path)
    return stale_path


def _schema_with_run_metadata(
    schema: pa.Schema,
    *,
    input_sha256: str,
    expected_frame_range: tuple[int, int] | None,
    allowed_missing_frames: Sequence[int],
    excluded_frames: Sequence[int],
    only_frames: Sequence[int],
    exclusion_reason: str,
    sparse_min_visible_rays: int,
) -> pa.Schema:
    """Attach the frame-quality contract directly to a Parquet schema."""

    metadata = dict(schema.metadata or {})
    values = {
        "paco.schema_version": "1.0",
        "paco.input_sha256": input_sha256,
        "paco.expected_frame_range": json.dumps(expected_frame_range),
        "paco.allowed_missing_frames": json.dumps(list(allowed_missing_frames)),
        "paco.quality_excluded_frames": json.dumps(list(excluded_frames)),
        "paco.only_frames_requested": json.dumps(list(only_frames)),
        "paco.frame_exclusion_reason": exclusion_reason,
        "paco.sparse_min_visible_rays": str(int(sparse_min_visible_rays)),
    }
    metadata.update({key.encode(): value.encode() for key, value in values.items()})
    return schema.with_metadata(metadata)


def main() -> None:
    args = parse_args()
    if not args.tracks.is_file():
        raise SystemExit(f"tracks file does not exist: {args.tracks}")
    if args.n_ego_per_approach < 1:
        raise SystemExit("--n-ego-per-approach must be positive")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be nonnegative")
    if args.exclude_frame and not str(args.frame_exclusion_reason).strip():
        raise SystemExit("--frame-exclusion-reason must be nonempty when frames are excluded")

    _verify_schema_columns(COUNT_SCHEMA, COUNT_COLUMNS, name="count")
    _verify_schema_columns(
        CONTRIBUTION_SCHEMA, CONTRIBUTION_COLUMNS, name="contribution"
    )
    _verify_schema_columns(EXCLUSION_SCHEMA, EXCLUSION_COLUMNS, name="exclusion")
    _verify_schema_columns(TIMING_SCHEMA, TIMING_COLUMNS, name="timing")

    tracks = pd.read_parquet(args.tracks)
    ground_plane = GroundPlane.from_coefficients(args.ground_plane)
    expected_range = (
        tuple(int(value) for value in args.expected_frame_range)
        if args.expected_frame_range is not None
        else None
    )
    audit = audit_full_record_tracks(
        tracks,
        ground_plane=ground_plane,
        expected_frame_range=expected_range,
        allowed_missing_frames=args.allow_missing_frame,
    )
    audit["source"] = {
        "path": str(args.tracks.resolve()),
        "sha256": sha256_file(args.tracks),
    }
    require_passing_audit(audit)

    config = FullRecordConfig(
        fov_degrees=tuple(args.fov_deg),
        z_modes=tuple(args.z_modes),
        range_m=args.range_m,
        near_blind_m=args.near_blind_m,
        sensor_forward_offset_m=args.sensor_forward_offset_m,
        sensor_height_m=args.sensor_height_m,
        ego_clearance_m=args.ego_clearance_m,
        angular_resolution_deg=0.25,
        occlusion_clearance_m=args.occlusion_clearance_m,
        sparse_min_visible_rays=args.sparse_min_visible_rays,
        vru_classes=frozenset(args.vru_classes),
        non_vru_weight=args.non_vru_weight,
        vru_weight=args.vru_weight,
    )
    frame_ids = select_frame_ids(
        tracks,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        exclude_frames=args.exclude_frame,
        only_frames=args.only_frame or None,
    )
    input_frame_count = int(tracks["frame_idx"].nunique())
    observed_frame_ids = set(tracks["frame_idx"].astype(int).unique())
    excluded_requested = sorted(set(int(value) for value in args.exclude_frame))
    excluded_observed = sorted(observed_frame_ids.intersection(excluded_requested))
    excluded_absent = sorted(set(excluded_requested).difference(observed_frame_ids))
    only_requested = sorted(set(int(value) for value in args.only_frame))
    only_observed = sorted(observed_frame_ids.intersection(only_requested))
    only_absent = sorted(set(only_requested).difference(observed_frame_ids))
    if expected_range is not None:
        first_expected, last_expected = expected_range
        outside_expected = [
            value
            for value in excluded_requested
            if value < first_expected or value > last_expected
        ]
        if outside_expected:
            raise SystemExit(
                "--exclude-frame contains IDs outside --expected-frame-range: "
                f"{outside_expected}"
            )
        only_outside_expected = [
            value for value in only_requested if value < first_expected or value > last_expected
        ]
        if only_outside_expected:
            raise SystemExit(
                "--only-frame contains IDs outside --expected-frame-range: "
                f"{only_outside_expected}"
            )
    ego_poses = default_ego_grid(args.n_ego_per_approach)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_paths, temporary_paths = _temporary_paths(args.out_dir)
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing outputs: {names}")
    stale_summary = (
        _invalidate_existing_completion_marker(final_paths["summary"])
        if args.overwrite
        else None
    )

    writers: dict[str, BufferedParquetWriter] = {}
    schema_arguments = {
        "input_sha256": str(audit["source"]["sha256"]),
        "expected_frame_range": expected_range,
        "allowed_missing_frames": sorted(set(args.allow_missing_frame)),
        "excluded_frames": excluded_requested,
        "only_frames": only_requested,
        "exclusion_reason": str(args.frame_exclusion_reason),
        "sparse_min_visible_rays": int(config.sparse_min_visible_rays),
    }
    started = time.perf_counter()
    summary_accumulator = FullRecordSummaryAccumulator()
    try:
        writers["counts"] = BufferedParquetWriter(
            temporary_paths["counts"],
            _schema_with_run_metadata(COUNT_SCHEMA, **schema_arguments),
            buffer_rows=args.buffer_rows,
        )
        writers["contributions"] = BufferedParquetWriter(
            temporary_paths["contributions"],
            _schema_with_run_metadata(CONTRIBUTION_SCHEMA, **schema_arguments),
            buffer_rows=args.buffer_rows,
        )
        writers["exclusions"] = BufferedParquetWriter(
            temporary_paths["exclusions"],
            _schema_with_run_metadata(EXCLUSION_SCHEMA, **schema_arguments),
            buffer_rows=args.buffer_rows,
        )
        writers["timings"] = BufferedParquetWriter(
            temporary_paths["timings"],
            _schema_with_run_metadata(TIMING_SCHEMA, **schema_arguments),
            buffer_rows=args.buffer_rows,
        )
        results = iter_full_record(
            tracks,
            ground_plane=ground_plane,
            config=config,
            frame_ids=frame_ids,
            ego_poses=ego_poses,
            n_processes=args.n_processes,
            chunksize=args.chunksize,
            input_audit=audit,
        )
        for completed, result in enumerate(results, start=1):
            expected_index = completed - 1
            if result.frame_order != expected_index or result.frame_idx != int(
                frame_ids[expected_index]
            ):
                raise RuntimeError("worker results violated deterministic frame ordering")
            writers["counts"].append(result.count_rows)
            writers["contributions"].append(contribution_rows(result.count_rows))
            writers["exclusions"].append(result.exclusion_rows)
            writers["timings"].append(result.timing_rows)
            summary_accumulator.update(result)
            if args.progress_every and (
                completed % args.progress_every == 0 or completed == len(frame_ids)
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"frames {completed}/{len(frame_ids)}; frame={result.frame_idx}; "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
        if summary_accumulator.processed_frame_count != len(frame_ids):
            raise RuntimeError("full-record iterator ended before all selected frames completed")
        for writer in writers.values():
            writer.close()

        wall_elapsed = time.perf_counter() - started
        summary = summary_accumulator.finalize(
            config=config,
            input_frame_count=input_frame_count,
            selected_frame_count=len(frame_ids),
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            ego_positions=len(ego_poses),
            wall_elapsed_s=wall_elapsed,
        )
        summary["input"] = {
            "path": str(args.tracks.resolve()),
            "sha256": audit["source"]["sha256"],
            "quality_gate_passed": True,
            "expected_frame_range": expected_range,
            "allowed_missing_frames": sorted(set(args.allow_missing_frame)),
            "quality_excluded_frames": {
                "requested": excluded_requested,
                "observed_and_removed": excluded_observed,
                "absent_from_track_table": excluded_absent,
                "reason": str(args.frame_exclusion_reason),
            },
            "only_frames": {
                "requested": only_requested,
                "observed": only_observed,
                "absent_from_track_table": only_absent,
            },
        }
        summary["execution"] = {
            "n_processes": int(args.n_processes),
            "chunksize": int(args.chunksize),
            "buffer_rows": int(args.buffer_rows),
            "stable_sort_key": ["frame_order", "ego_index", "z_mode", "fov_deg"],
        }
        summary["ground_plane"] = {
            "normal": list(ground_plane.normal),
            "d": float(ground_plane.d),
            "input_coefficients": [float(value) for value in args.ground_plane],
        }
        summary["outputs"] = {
            key: {
                "file": OUTPUT_NAMES[key],
                "rows": writers[key].rows_written,
            }
            for key in ("counts", "contributions", "exclusions", "timings")
        }
        run_config = {
            "schema_version": "1.0",
            "geometry": config_as_dict(config),
            "ground_plane": {
                "normal": list(ground_plane.normal),
                "d": float(ground_plane.d),
                "input_coefficients": [float(value) for value in args.ground_plane],
            },
            "selection": {
                "frame_step": int(args.frame_step),
                "max_frames": args.max_frames,
                "selected_frame_count": int(len(frame_ids)),
                "selected_frame_first": int(frame_ids[0]),
                "selected_frame_last": int(frame_ids[-1]),
                "quality_excluded_frames": {
                    "requested": excluded_requested,
                    "observed_and_removed": excluded_observed,
                    "absent_from_track_table": excluded_absent,
                    "reason": str(args.frame_exclusion_reason),
                },
                "only_frames": {
                    "requested": only_requested,
                    "observed": only_observed,
                    "absent_from_track_table": only_absent,
                },
            },
            "ego_grid": {
                "n_per_approach": int(args.n_ego_per_approach),
                "total": len(ego_poses),
            },
            "input_contract": {
                "expected_frame_range": expected_range,
                "allowed_missing_frames": sorted(set(args.allow_missing_frame)),
            },
        }
        _write_json(temporary_paths["summary"], summary)
        _write_json(temporary_paths["config"], run_config)
        _write_json(temporary_paths["audit"], audit)

        # Publish the completion marker (summary) last.  Until then, any
        # interrupted run leaves only uniquely named .partial files.
        for key in ("counts", "contributions", "exclusions", "timings", "config", "audit"):
            os.replace(temporary_paths[key], final_paths[key])
        os.replace(temporary_paths["summary"], final_paths["summary"])
    except BaseException:
        for writer in writers.values():
            with suppress(Exception):
                writer.close()
        for path in temporary_paths.values():
            with suppress(FileNotFoundError):
                path.unlink()
        raise
    if stale_summary is not None:
        with suppress(FileNotFoundError):
            stale_summary.unlink()

    print(json.dumps(summary["groups"], indent=2), flush=True)
    print(f"wrote {args.out_dir} in {summary['runtime']['wall_elapsed_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
