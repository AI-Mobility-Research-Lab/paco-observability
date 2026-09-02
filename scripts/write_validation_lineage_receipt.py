#!/usr/bin/env python3
"""Write a byte-exact source-lineage receipt for a completed validation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paco_observability.validation_lineage import (
    verify_validation_lineage_receipt,
    write_validation_lineage_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validation_main_200_final", "validation_convergence_25_final"),
    )
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--input-provenance", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--exclude-source-frame", type=int, action="append", default=[])
    parser.add_argument(
        "--main-sample-summary",
        type=Path,
        help="Required for convergence: the 200-frame main summary containing its parent sample",
    )
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arguments = dict(
        stage=args.stage,
        repository=args.repository,
        producer_commit=args.producer_commit,
        tracks=args.tracks,
        frame_index=args.frame_index,
        input_provenance=args.input_provenance,
        summary=args.summary,
        config=args.config,
        decisions=args.decisions,
        timings=args.timings,
        exclusions=args.exclusions,
        excluded_source_frames=args.exclude_source_frame,
        main_sample_summary=args.main_sample_summary,
    )
    if args.write:
        receipt = write_validation_lineage_receipt(output_path=args.out, **arguments)
    else:
        receipt = verify_validation_lineage_receipt(output_path=args.out, **arguments)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
