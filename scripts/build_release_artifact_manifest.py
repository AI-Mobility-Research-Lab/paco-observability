#!/usr/bin/env python3
"""Build a canonical release manifest from explicit files and directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paco_observability.release_manifest import (
    DEFAULT_PACKAGE_NAMES,
    write_release_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively hash explicit release files/directories. Paths in the JSON are "
            "relative to --base-dir; directory traversal and output are deterministic."
        ),
        epilog=(
            "Set SOURCE_DATE_EPOCH to pin generated_at_utc for byte-identical rebuilds. "
            "The output must be outside every selected file/directory tree. Canonical "
            "releases should also use --require-clean-git."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Explicit file or directory paths, interpreted relative to --base-dir",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base for resolving inputs and recording relative artifact paths (default: cwd)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("release_artifact_manifest.json"),
        help="Output JSON; relative paths are interpreted below --base-dir",
    )
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        help=(
            "Distribution to record (repeatable). Defaults to paco-observability and its "
            "core numerical, plotting, and notebook-execution dependencies."
        ),
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Fail unless the repository is clean before and after the two hash passes",
    )
    parser.add_argument(
        "--numeric-producer-config",
        type=Path,
        help=(
            "Optional JSON config containing release.numeric_producer_commit and "
            "release.numeric_producer_files; present declarations are verified byte-for-byte"
        ),
    )
    parser.add_argument(
        "--require-numeric-producer",
        action="store_true",
        help=(
            "Fail unless the supplied config has a complete, byte-verified numeric-producer "
            "commit/file declaration"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packages = args.packages if args.packages is not None else DEFAULT_PACKAGE_NAMES
    manifest = write_release_manifest(
        base_dir=args.base_dir,
        requested_inputs=args.inputs,
        output_path=args.out,
        package_names=packages,
        require_clean_git=args.require_clean_git,
        numeric_producer_config=args.numeric_producer_config,
        require_numeric_producer=args.require_numeric_producer,
    )
    output = args.out if args.out.is_absolute() else args.base_dir / args.out
    receipt = {
        "output": str(output.resolve()),
        "artifact_count": manifest["artifact_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "git_commit": manifest["git"]["commit"],
        "git_dirty": manifest["git"]["dirty"],
        "numeric_producer": manifest["numeric_producer"],
        "require_clean_git": manifest["release_policy"]["require_clean_git"],
        "require_numeric_producer": manifest["release_policy"]["require_numeric_producer"],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
