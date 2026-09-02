"""Input validation and provenance helpers for paper-facing analyses.

The functions in this module deliberately fail closed: an input missing the
3-D box fields needed by the reference model is rejected before a computation
can produce manuscript-facing numbers.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_TRACK_COLUMNS = frozenset(
    {
        "frame_idx",
        "track_id",
        "class_name",
        "bbox_center_x",
        "bbox_center_y",
        "bbox_center_z",
        "bbox_dx",
        "bbox_dy",
        "bbox_dz",
        "bbox_yaw",
    }
)

REQUIRED_FRAME_INDEX_COLUMNS = frozenset({"frame_idx", "num_points"})


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: pd.Series, probabilities: Iterable[float]) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {f"q{int(p * 100):02d}": float(clean.quantile(p)) for p in probabilities}


def detect_partial_scans(
    frame_index: pd.DataFrame,
    *,
    neighbor_radius: int = 3,
    point_ratio_threshold: float = 0.6,
) -> pd.DataFrame:
    """Detect low-point partial scans relative to adjacent source frames.

    For each frame, the denominator is the median point count among the
    preceding and following ``neighbor_radius`` rows, excluding the frame
    itself.  This matches the production alignment-suite diagnostic while
    avoiding the unreliable ``frame_timestamp_min == 0`` heuristic.
    """

    missing = sorted(REQUIRED_FRAME_INDEX_COLUMNS.difference(frame_index.columns))
    if missing:
        raise ValueError(f"frame index is missing required columns: {missing}")
    if frame_index.empty:
        raise ValueError("frame index is empty")
    if neighbor_radius < 1:
        raise ValueError("neighbor_radius must be positive")
    if not 0 < point_ratio_threshold < 1:
        raise ValueError("point_ratio_threshold must be in (0, 1)")

    ordered = frame_index.sort_values("frame_idx", kind="stable").reset_index(drop=True)
    frame_ids = pd.to_numeric(ordered["frame_idx"], errors="coerce").to_numpy(float)
    point_counts = pd.to_numeric(ordered["num_points"], errors="coerce").to_numpy(float)
    if not np.isfinite(frame_ids).all() or not np.isfinite(point_counts).all():
        raise ValueError("frame_idx and num_points must be finite")

    rows: list[dict[str, Any]] = []
    for index, count in enumerate(point_counts):
        before = point_counts[max(0, index - neighbor_radius) : index]
        after = point_counts[index + 1 : index + neighbor_radius + 1]
        neighbors = np.concatenate((before, after))
        finite_positive = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
        if finite_positive.size == 0:
            raise ValueError(f"frame {int(frame_ids[index])} has no valid neighboring counts")
        neighbor_median = float(np.median(finite_positive))
        ratio = float(count / neighbor_median)
        if ratio < point_ratio_threshold:
            rows.append(
                {
                    "frame_idx": int(frame_ids[index]),
                    "num_points": int(count),
                    "neighbor_median_points": neighbor_median,
                    "point_count_ratio": ratio,
                    "neighbor_radius": int(neighbor_radius),
                    "point_ratio_threshold": float(point_ratio_threshold),
                    "reason": "partial_scan_low_point_ratio",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "frame_idx",
            "num_points",
            "neighbor_median_points",
            "point_count_ratio",
            "neighbor_radius",
            "point_ratio_threshold",
            "reason",
        ],
    )


def audit_frame_index(
    frame_index: pd.DataFrame,
    *,
    expected_frame_range: tuple[int, int] | None = None,
    excluded_partial_frames: Iterable[int] = (),
    neighbor_radius: int = 3,
    point_ratio_threshold: float = 0.6,
) -> dict[str, Any]:
    """Fail-closed audit of source-frame completeness and exclusions."""

    missing = sorted(REQUIRED_FRAME_INDEX_COLUMNS.difference(frame_index.columns))
    if missing:
        raise ValueError(f"frame index is missing required columns: {missing}")
    if frame_index.empty:
        raise ValueError("frame index is empty")

    frame_values = pd.to_numeric(frame_index["frame_idx"], errors="coerce")
    point_values = pd.to_numeric(frame_index["num_points"], errors="coerce")
    nonfinite_frame_ids = int((~np.isfinite(frame_values.to_numpy(float))).sum())
    nonfinite_point_counts = int((~np.isfinite(point_values.to_numpy(float))).sum())
    duplicate_frames = int(frame_index.duplicated("frame_idx").sum())
    nonpositive_point_counts = int((point_values <= 0).sum())
    if nonfinite_frame_ids:
        raise ValueError("frame index contains non-finite frame_idx values")

    frames = np.sort(frame_values.astype(int).unique())
    if expected_frame_range is None:
        expected_start, expected_end = int(frames[0]), int(frames[-1])
    else:
        expected_start, expected_end = (int(value) for value in expected_frame_range)
        if expected_start > expected_end:
            raise ValueError("expected_frame_range start must not exceed end")
        if int(frames[0]) < expected_start or int(frames[-1]) > expected_end:
            raise ValueError("frame-index rows fall outside expected_frame_range")
    expected = np.arange(expected_start, expected_end + 1, dtype=int)
    missing_frames = np.setdiff1d(expected, frames).astype(int).tolist()

    partial = detect_partial_scans(
        frame_index,
        neighbor_radius=neighbor_radius,
        point_ratio_threshold=point_ratio_threshold,
    )
    detected = partial["frame_idx"].astype(int).tolist()
    declared = sorted({int(value) for value in excluded_partial_frames})
    detected_set = set(detected)
    declared_set = set(declared)
    unexcluded = sorted(detected_set - declared_set)
    declared_not_detected = sorted(declared_set - detected_set)
    declared_absent = sorted(declared_set - set(frames))

    fatal_counts = {
        "duplicate_frame_rows": duplicate_frames,
        "nonfinite_point_counts": nonfinite_point_counts,
        "nonpositive_point_counts": nonpositive_point_counts,
        "missing_frames": len(missing_frames),
        "unexcluded_partial_frames": len(unexcluded),
        "declared_exclusions_absent_from_frame_index": len(declared_absent),
    }
    return {
        "rows": int(len(frame_index)),
        "frame_min": int(frames[0]),
        "frame_max": int(frames[-1]),
        "expected_frame_min": expected_start,
        "expected_frame_max": expected_end,
        "duplicate_frame_rows": duplicate_frames,
        "nonfinite_point_counts": nonfinite_point_counts,
        "nonpositive_point_counts": nonpositive_point_counts,
        "missing_frames": missing_frames[:20],
        "partial_scan_policy": {
            "neighbor_radius": int(neighbor_radius),
            "point_ratio_threshold": float(point_ratio_threshold),
            "timestamp_zero_is_not_an_exclusion_criterion": True,
        },
        "detected_partial_frames": detected,
        "declared_excluded_partial_frames": declared,
        "unexcluded_partial_frames": unexcluded,
        "declared_exclusions_not_detected": declared_not_detected,
        "partial_scan_evidence": partial.to_dict(orient="records"),
        "quality_gate": {
            "passed": not any(fatal_counts.values()),
            "fatal_counts": fatal_counts,
            "policy": "fail closed for paper-facing computation",
        },
    }


def audit_tracks(
    tracks: pd.DataFrame,
    *,
    ground_normal: tuple[float, float, float] | None = None,
    ground_offset: float | None = None,
    expected_frame_range: tuple[int, int] | None = None,
    allowed_missing_frames: Iterable[int] = (),
) -> dict[str, Any]:
    """Validate a tracked-box table and return a JSON-serializable audit.

    ``ground_normal`` and ``ground_offset`` describe ``n dot p + d = 0``.
    When supplied, the report includes the detected-box bottom residual from
    that plane. This is a diagnostic rather than an automatic correction.
    """

    missing = sorted(REQUIRED_TRACK_COLUMNS.difference(tracks.columns))
    if missing:
        raise ValueError(f"tracks table is missing required columns: {missing}")
    if tracks.empty:
        raise ValueError("tracks table is empty")

    numeric_columns = [
        "frame_idx",
        "track_id",
        "bbox_center_x",
        "bbox_center_y",
        "bbox_center_z",
        "bbox_dx",
        "bbox_dy",
        "bbox_dz",
        "bbox_yaw",
    ]
    nonfinite: dict[str, int] = {}
    for column in numeric_columns:
        values = pd.to_numeric(tracks[column], errors="coerce").to_numpy(dtype=float)
        nonfinite[column] = int((~np.isfinite(values)).sum())

    invalid_dimensions = int(
        ((tracks[["bbox_dx", "bbox_dy", "bbox_dz"]].astype(float) <= 0).any(axis=1)).sum()
    )
    duplicate_frame_track = int(tracks.duplicated(["frame_idx", "track_id"]).sum())
    frames = np.sort(tracks["frame_idx"].astype(int).unique())
    if expected_frame_range is None:
        expected_start, expected_end = int(frames[0]), int(frames[-1])
    else:
        expected_start, expected_end = (int(value) for value in expected_frame_range)
        if expected_start > expected_end:
            raise ValueError("expected_frame_range start must not exceed end")
        if int(frames[0]) < expected_start or int(frames[-1]) > expected_end:
            raise ValueError("observed frames fall outside expected_frame_range")
    expected = np.arange(expected_start, expected_end + 1, dtype=int)
    missing_frames = np.setdiff1d(expected, frames)
    allowed_missing = sorted({int(value) for value in allowed_missing_frames})
    allowed_set = set(allowed_missing)
    unapproved_missing = [int(value) for value in missing_frames if int(value) not in allowed_set]

    report: dict[str, Any] = {
        "rows": int(len(tracks)),
        "frames": int(len(frames)),
        "frame_min": int(frames[0]),
        "frame_max": int(frames[-1]),
        "expected_frame_min": expected_start,
        "expected_frame_max": expected_end,
        "missing_frame_count": int(len(missing_frames)),
        "missing_frame_examples": missing_frames[:20].astype(int).tolist(),
        "allowed_missing_frames": allowed_missing,
        "unapproved_missing_frames": unapproved_missing[:20],
        "unique_tracks": int(tracks["track_id"].nunique()),
        "class_rows": {
            str(key): int(value)
            for key, value in tracks["class_name"].astype(str).value_counts().sort_index().items()
        },
        "duplicate_frame_track_rows": duplicate_frame_track,
        "invalid_dimension_rows": invalid_dimensions,
        "nonfinite_values": nonfinite,
        "dimension_quantiles_m": {
            column: _quantiles(tracks[column], (0.01, 0.05, 0.5, 0.95, 0.99))
            for column in ("bbox_dx", "bbox_dy", "bbox_dz")
        },
    }

    if "run_id" in tracks.columns:
        report["run_ids"] = sorted(tracks["run_id"].dropna().astype(str).unique().tolist())
    if "score" in tracks.columns:
        report["score_quantiles"] = _quantiles(tracks["score"], (0.01, 0.05, 0.5, 0.95, 0.99))

    if ground_normal is not None or ground_offset is not None:
        if ground_normal is None or ground_offset is None:
            raise ValueError("ground_normal and ground_offset must be supplied together")
        nx, ny, nz = (float(value) for value in ground_normal)
        if abs(nz) < 1e-12:
            raise ValueError("ground plane must have a non-zero z component")
        ground_z = -(
            nx * tracks["bbox_center_x"].to_numpy(float)
            + ny * tracks["bbox_center_y"].to_numpy(float)
            + float(ground_offset)
        ) / nz
        bottom = tracks["bbox_center_z"].to_numpy(float) - tracks["bbox_dz"].to_numpy(float) / 2
        residual = pd.Series(bottom - ground_z, index=tracks.index)
        report["ground_plane"] = {
            "normal": [nx, ny, nz],
            "offset": float(ground_offset),
            "box_bottom_residual_m": _quantiles(
                residual, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
            ),
            "by_class": {
                str(name): {
                    "count": int(len(group)),
                    "mean_m": float(residual.loc[group.index].mean()),
                    "median_m": float(residual.loc[group.index].median()),
                    "std_m": float(residual.loc[group.index].std(ddof=1))
                    if len(group) > 1
                    else 0.0,
                }
                for name, group in tracks.groupby(tracks["class_name"].astype(str), sort=True)
            },
        }

    fatal_counts = {
        "duplicate_frame_track_rows": duplicate_frame_track,
        "invalid_dimension_rows": invalid_dimensions,
        "nonfinite_required_values": int(sum(nonfinite.values())),
        "unapproved_missing_frames": len(unapproved_missing),
    }
    report["quality_gate"] = {
        "passed": not any(fatal_counts.values()),
        "fatal_counts": fatal_counts,
        "policy": "fail closed for paper-facing computation",
    }
    return report


def git_revision(repository: Path) -> dict[str, Any]:
    """Capture a repository revision without mutating it."""

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    return {
        "path": str(repository.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def build_run_manifest(
    *,
    tracks_path: Path,
    analysis_repository: Path,
    parameters: dict[str, Any],
    input_audit: dict[str, Any],
    upstream_manifest: Path | None = None,
    frame_index_path: Path | None = None,
    frame_index_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a provenance record for a deterministic analysis run."""

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "tracks": {
            "path": str(tracks_path.resolve()),
            "sha256": sha256_file(tracks_path),
            "audit": input_audit,
        },
        "analysis_code": git_revision(analysis_repository),
        "parameters": parameters,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    if upstream_manifest is not None:
        manifest["upstream_manifest"] = {
            "path": str(upstream_manifest.resolve()),
            "sha256": sha256_file(upstream_manifest),
            "content": json.loads(upstream_manifest.read_text()),
        }
    if frame_index_path is not None:
        if frame_index_audit is None:
            raise ValueError("frame_index_audit is required with frame_index_path")
        manifest["frame_index"] = {
            "path": str(frame_index_path.resolve()),
            "sha256": sha256_file(frame_index_path),
            "audit": frame_index_audit,
        }
    return manifest
