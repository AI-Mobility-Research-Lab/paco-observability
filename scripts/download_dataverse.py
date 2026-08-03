#!/usr/bin/env python3
"""Download and verify the published PACO derived-data files."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DATASET_DOI = "doi:10.7910/DVN/7T4DWL"
API = "https://dataverse.harvard.edu/api"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - Dataverse publishes MD5 file digests
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest() -> list[dict]:
    url = f"{API}/datasets/:persistentId/?persistentId={DATASET_DOI}"
    request = urllib.request.Request(url, headers={"User-Agent": "paco-observability/1.0"})
    with urllib.request.urlopen(request) as response:  # nosec B310 - fixed HTTPS host
        payload = json.load(response)
    if payload.get("status") != "OK":
        raise RuntimeError(f"Dataverse API returned {payload!r}")
    return payload["data"]["latestVersion"]["files"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/derived"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for entry in dataset_manifest():
        data_file = entry["dataFile"]
        file_id = data_file["id"]
        filename = entry["label"]
        expected = data_file["checksum"]["value"].lower()
        destination = args.out / filename

        if destination.exists() and not args.force and md5sum(destination) == expected:
            print(f"verified {destination}")
            continue

        url = f"{API}/access/datafile/{file_id}"
        request = urllib.request.Request(url, headers={"User-Agent": "paco-observability/1.0"})
        print(f"downloading {filename}")
        with urllib.request.urlopen(request) as response:  # nosec B310 - fixed HTTPS host
            content = response.read()
        destination.write_bytes(content)
        actual = md5sum(destination)
        if actual != expected:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {filename}: {actual} != {expected}")
        print(f"verified {destination}")


if __name__ == "__main__":
    main()

