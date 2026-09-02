#!/usr/bin/env python3
"""Calibrate the 15-ray visibility threshold on frame-disjoint held-out data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from paco_observability.provenance import sha256_file
from paco_observability.sparse_calibration import (
    DEFAULT_CALIBRATION_FRAMES,
    DEFAULT_EXPECTED_FRAMES,
    DEFAULT_EXPECTED_MIN_VISIBLE_RAYS,
    DEFAULT_FOV_DEGREES,
    DEFAULT_SEED,
    calibrate_sparse_threshold,
)


DEFAULT_OUT_DIR = Path("outputs/canonical/sparse_calibration")
SUMMARY_NAME = "sparse_threshold_calibration.json"
CALIBRATED_DECISIONS_NAME = "validation_decisions_calibrated.parquet"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the minimum clear-ray count using only a stratified calibration half, "
            "then report performance once on the frame-disjoint held-out half."
        )
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--expected-frame-count", type=int, default=DEFAULT_EXPECTED_FRAMES
    )
    parser.add_argument(
        "--calibration-frame-count", type=int, default=DEFAULT_CALIBRATION_FRAMES
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        nargs="+",
        default=list(DEFAULT_FOV_DEGREES),
        help="FOVs pooled for calibration and reported separately (default: 120 360)",
    )
    parser.add_argument(
        "--expected-selected-min-rays",
        type=int,
        default=DEFAULT_EXPECTED_MIN_VISIBLE_RAYS,
        help="Expected winner used as a regression guard (default: 3)",
    )
    parser.add_argument(
        "--allow-unexpected-selection",
        action="store_true",
        help="Write results even when the data-driven winner differs from the expected guard",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_size = int(args.decisions.stat().st_size)
    input_sha256 = sha256_file(args.decisions)
    decisions = pd.read_parquet(args.decisions)
    summary, calibrated = calibrate_sparse_threshold(
        decisions,
        seed=args.seed,
        expected_frame_count=args.expected_frame_count,
        calibration_frame_count=args.calibration_frame_count,
        fov_degrees=args.fov_deg,
        expected_selected_min_visible_rays=args.expected_selected_min_rays,
    )
    if int(args.decisions.stat().st_size) != input_size or sha256_file(
        args.decisions
    ) != input_sha256:
        raise ValueError("decisions input changed while threshold calibration was running")
    if (
        not summary["selection"]["matches_expected_selection"]
        and not args.allow_unexpected_selection
    ):
        raise ValueError(
            "data-driven threshold does not match --expected-selected-min-rays; "
            "inspect the calibration input or pass --allow-unexpected-selection"
        )

    calibrated_path = args.out_dir / CALIBRATED_DECISIONS_NAME
    summary_path = args.out_dir / SUMMARY_NAME
    if calibrated_path.resolve() == args.decisions.resolve():
        raise ValueError("calibrated output must not overwrite the input decisions parquet")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(calibrated_path, index=False)

    summary["input"] = {
        "path": str(args.decisions.resolve()),
        "size_bytes": input_size,
        "sha256": input_sha256,
        "rows": int(len(decisions)),
        "unique_frames": int(decisions["frame_idx"].nunique()),
    }
    summary["output"] = {
        "calibrated_decisions": {
            "path": str(calibrated_path.resolve()),
            "size_bytes": int(calibrated_path.stat().st_size),
            "sha256": sha256_file(calibrated_path),
            "rows": int(len(calibrated)),
            "columns": calibrated.columns.astype(str).tolist(),
        },
        "summary": {"path": str(summary_path.resolve())},
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_min_visible_rays": summary["selection"][
                    "selected_min_visible_rays"
                ],
                "calibration_frame_count": summary["split"][
                    "calibration_frame_count"
                ],
                "held_out_frame_count": summary["split"]["held_out_frame_count"],
                "summary": str(summary_path.resolve()),
                "calibrated_decisions": str(calibrated_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
