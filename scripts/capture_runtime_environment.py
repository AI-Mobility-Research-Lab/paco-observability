#!/usr/bin/env python3
"""Capture a non-sensitive hardware/software receipt for an analysis run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paco_observability.runtime_environment import write_runtime_environment_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write an allow-listed runtime receipt containing host hardware, Python/package "
            "versions, and Git state. SOURCE_DATE_EPOCH pins the receipt timestamp."
        )
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Human-readable run label, for example validation or full-record",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path("."),
        help="Git repository whose commit, branch, and dirty state are recorded (default: cwd)",
    )
    parser.add_argument(
        "--source-tree-without-git",
        action="store_true",
        help=(
            "Record that the source tree has no Git metadata. This mode requires "
            "--declared-source-commit and marks the declaration as unverified."
        ),
    )
    parser.add_argument(
        "--declared-source-commit",
        help=(
            "Full 40-character source commit declaration for --source-tree-without-git; "
            "the receipt does not claim that this value was repository-verified"
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("uv_frozen", "external_python"),
        default="external_python",
        help="How the numerical stage's Python environment was selected",
    )
    parser.add_argument(
        "--numeric-producer-config",
        type=Path,
        help=(
            "JSON config whose release.numeric_producer_commit/files declaration is "
            "verified byte-for-byte against --repository and recorded separately from "
            "the execution checkout's Git state"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON path (written atomically)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = write_runtime_environment_receipt(
        label=args.label,
        repository=args.repository,
        output_path=args.out,
        source_tree_without_git=args.source_tree_without_git,
        declared_source_commit=args.declared_source_commit,
        execution_mode=args.execution_mode,
        numeric_producer_config=args.numeric_producer_config,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
