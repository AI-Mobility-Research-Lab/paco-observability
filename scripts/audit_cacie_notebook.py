#!/usr/bin/env python3
"""Create or verify the fail-closed CACIE executed-notebook audit receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from paco_observability.notebook_audit import (
    audit_canonical_artifacts,
    verify_notebook_audit_receipt,
    write_notebook_audit_receipt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-notebook", type=Path, required=True)
    parser.add_argument("--executed-notebook", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-receipt", action="store_true")
    mode.add_argument("--verify-receipt", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve(strict=True)
    output_root = (
        args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    )
    source_notebook = (
        args.source_notebook
        if args.source_notebook.is_absolute()
        else repo_root / args.source_notebook
    )
    executed_notebook = (
        args.executed_notebook
        if args.executed_notebook.is_absolute()
        else repo_root / args.executed_notebook
    )
    receipt_path = args.receipt if args.receipt.is_absolute() else repo_root / args.receipt
    artifact_audit = audit_canonical_artifacts(
        repo_root=repo_root,
        output_root=output_root,
    )
    if args.write_receipt:
        receipt = write_notebook_audit_receipt(
            receipt_path=receipt_path,
            source_notebook=source_notebook,
            executed_notebook=executed_notebook,
            artifact_audit=artifact_audit,
            repo_root=repo_root,
        )
        action = "written"
    else:
        receipt = verify_notebook_audit_receipt(
            receipt_path=receipt_path,
            source_notebook=source_notebook,
            executed_notebook=executed_notebook,
            artifact_audit=artifact_audit,
            repo_root=repo_root,
        )
        action = "verified"
    print(
        json.dumps(
            {
                "action": action,
                "receipt": str(receipt_path.resolve()),
                "status": receipt["status"],
                "pass_marker": receipt["pass_marker"],
                "stage_count": len(receipt["stages"]),
                "check_count": receipt["check_count"],
                "audited_file_count": len(receipt["audited_files"]),
                "executed_notebook_sha256": receipt["executed_notebook"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
