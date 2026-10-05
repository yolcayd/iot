"""Download, verify and extract the Car-Hacking and ROAD datasets.

Usage:
    python -m src.data.download [car_hacking|road|all]

The command is idempotent and resumable: an archive whose checksum already matches is not
downloaded again, a partial download (``*.part``) is resumed with an HTTP Range request when
the server supports it, and extraction is skipped when the extracted files are already listed
in the manifest. Every archive and every extracted file is recorded with its size and SHA-256
in ``data/raw/manifest.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.csv"
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    archive_name: str
    md5: str | None  # published checksum, when the provider publishes one


SOURCES = {
    "car_hacking": DatasetSource(
        name="car_hacking",
        # Public Dropbox folder linked from https://ocslab.hksecurity.net/Datasets/car-hacking-dataset;
        # dl=1 makes Dropbox serve the whole folder as one zip.
        url=(
            "https://www.dropbox.com/scl/fo/9rwsf9pclhvv9xxloojom/AF7JeRW893grZkigkulkAHk"
            "?rlkey=3h6zamu3kc262lrnipu5qden8&st=3g52q9hd&dl=1"
        ),
        archive_name="car_hacking_dataset.zip",
        md5=None,
    ),
    "road": DatasetSource(
        name="road",
        url="https://zenodo.org/records/10462796/files/road.zip?download=1",
        archive_name="road.zip",
        md5="cab184cfc2fe12c0834bc46188c0f330",
    ),
}


def file_digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while block := f.read(CHUNK_BYTES):
            h.update(block)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download(source: DatasetSource, dest: Path) -> None:
    """Download ``source.url`` to ``dest`` through a ``.part`` file, resuming when possible."""
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(source.url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise RuntimeError(
                f"{source.name}: server returned an HTML page instead of the archive "
                "(login, form or consent page?). Download manually."
            )
        if offset and r.status_code != 206:
            log(f"{source.name}: server ignored the Range request, restarting from 0")
            offset = 0
        total = int(r.headers.get("Content-Length", 0)) + offset
        mode = "ab" if offset else "wb"
        done, last = offset, time.time()
        with part.open(mode) as f:
            for block in r.iter_content(CHUNK_BYTES):
                f.write(block)
                done += len(block)
                if time.time() - last > 10:
                    pct = f"{100 * done / total:.1f}%" if total else "?"
                    log(f"{source.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB ({pct})")
                    last = time.time()
    part.replace(dest)


def is_os_metadata(path: Path) -> bool:
    """macOS archive artifacts (``__MACOSX/``, ``.DS_Store``) are not part of the dataset."""
    return "__MACOSX" in path.parts or path.name == ".DS_Store"


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict[str, str]]) -> None:
    rows = sorted(rows, key=lambda r: (r["dataset"], r["kind"], r["path"]))
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "kind", "path", "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)


def manifest_row(dataset: str, kind: str, path: Path) -> dict[str, str]:
    return {
        "dataset": dataset,
        "kind": kind,
        "path": path.relative_to(RAW_DIR).as_posix(),
        "size_bytes": str(path.stat().st_size),
        "sha256": file_digest(path, "sha256"),
    }


def fetch(source: DatasetSource) -> None:
    target_dir = RAW_DIR / source.name
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / source.archive_name
    manifest = read_manifest()
    recorded = {r["path"]: r for r in manifest}

    archive_key = archive.relative_to(RAW_DIR).as_posix()
    if archive.exists() and archive_key in recorded and recorded[archive_key]["size_bytes"] == str(
        archive.stat().st_size
    ):
        log(f"{source.name}: archive already present, skipping download")
    else:
        log(f"{source.name}: downloading {source.url}")
        download(source, archive)
        if source.md5 is not None:
            md5 = file_digest(archive, "md5")
            if md5 != source.md5:
                raise RuntimeError(f"{source.name}: MD5 mismatch ({md5} != {source.md5})")
            log(f"{source.name}: MD5 verified ({md5})")
        manifest = [r for r in manifest if r["path"] != archive_key]
        manifest.append(manifest_row(source.name, "archive", archive))
        write_manifest(manifest)

    extracted = [r for r in manifest if r["dataset"] == source.name and r["kind"] == "extracted"]
    if extracted and all((RAW_DIR / r["path"]).exists() for r in extracted):
        log(f"{source.name}: already extracted ({len(extracted)} files), skipping")
        return
    log(f"{source.name}: extracting to {target_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)
    manifest = [r for r in manifest if not (r["dataset"] == source.name and r["kind"] == "extracted")]
    for path in sorted(p for p in target_dir.rglob("*") if p.is_file() and not is_os_metadata(p)):
        manifest.append(manifest_row(source.name, "extracted", path))
    write_manifest(manifest)
    log(f"{source.name}: done")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=[*SOURCES, "all"], nargs="?", default="all")
    args = parser.parse_args(argv)
    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    for name in names:
        fetch(SOURCES[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
