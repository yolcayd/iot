"""Readers for the Car-Hacking and ROAD datasets that convert raw CAN logs into a common schema.

Common schema (one row per CAN frame, one parquet file per source capture):

    timestamp       float64   seconds, as recorded in the raw log
    can_id          int32     arbitration ID
    dlc             int8      data length code (0..8)
    data            uint8[8]  payload, zero-padded when dlc < 8
    label           int8      1 = injected frame, 0 = benign frame
    attack_type     string    attack name for injected frames, "normal" otherwise
    source_file     string    raw file path relative to data/raw
    source_capture  string    capture the file was recorded as; a derived capture (ROAD
                              "_masquerade" logs) carries the id of the capture it was derived from

Labeling rules were verified against the raw files and metadata (docs/DATA.md, docs/DECISIONS.md):

* Car-Hacking: the per-row flag ("T" injected, "R" benign). Rows with dlc < 8 have fewer
  columns, so the flag sits at column 3 + dlc.
* ROAD targeted fabrication/masquerade: frame ID == ``injection_id`` and payload matches
  ``injection_data_str`` ("X" = wildcard nibble) and the frame lies inside ``injection_interval``
  (seconds relative to the first frame of the capture, both ends inclusive).
* ROAD fuzzing: payload == FF*8 at or after the injection start. That payload never occurs in
  ambient traffic; two injected frames trail the metadata end by < 6 ms.
* ROAD accelerator captures contain no injected frame (the vehicle was compromised before
  recording), so no frame-level label exists; they are converted with label 0 and marked as
  excluded in the index.

Usage:
    python -m src.data.loaders [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LINEAGE_TABLE = PROJECT_ROOT / "results" / "tables" / "capture_lineage.csv"
CHUNK_ROWS = 1_000_000
NORMAL = "normal"

_STR = pa.dictionary(pa.int32(), pa.string())
SCHEMA = pa.schema(
    [
        ("timestamp", pa.float64()),
        ("can_id", pa.int32()),
        ("dlc", pa.int8()),
        ("data", pa.list_(pa.uint8(), 8)),
        ("label", pa.int8()),
        ("attack_type", _STR),
        ("source_file", _STR),
        ("source_capture", _STR),
    ]
)

# ASCII code -> hex nibble value; -1 marks a non-hex character.
_NIBBLE = np.full(256, -1, dtype=np.int16)
for _i, _c in enumerate(b"0123456789abcdef"):
    _NIBBLE[_c] = _i
for _i, _c in enumerate(b"ABCDEF"):
    _NIBBLE[_c] = 10 + _i


def hex_to_uint8(strings: np.ndarray, n_bytes: int) -> np.ndarray:
    """Decode equal-width hex strings into an (N, n_bytes) uint8 array."""
    raw = np.asarray(strings, dtype=f"S{2 * n_bytes}")
    ascii_codes = np.frombuffer(raw.tobytes(), dtype=np.uint8).reshape(-1, 2 * n_bytes)
    nib = _NIBBLE[ascii_codes]
    if (nib < 0).any():
        raise ValueError("non-hex character in payload")
    return (nib[:, 0::2] * 16 + nib[:, 1::2]).astype(np.uint8)


def hex_ids(strings: pd.Series) -> np.ndarray:
    """Decode hex arbitration IDs through a lookup over the unique values."""
    uniq = strings.unique()
    lut = {u: int(u, 16) for u in uniq}
    return strings.map(lut).to_numpy(dtype=np.int32)


def to_table(
    timestamp: np.ndarray,
    can_id: np.ndarray,
    dlc: np.ndarray,
    data: np.ndarray,
    label: np.ndarray,
    attack_type: str,
    source_file: str,
    source_capture: str,
) -> pa.Table:
    n = len(timestamp)
    attack = np.where(label == 1, attack_type, NORMAL)
    return pa.Table.from_arrays(
        [
            pa.array(timestamp, pa.float64()),
            pa.array(can_id, pa.int32()),
            pa.array(dlc, pa.int8()),
            pa.FixedSizeListArray.from_arrays(pa.array(data.reshape(-1), pa.uint8()), 8),
            pa.array(label, pa.int8()),
            pa.array(attack).dictionary_encode(),
            pa.array(np.full(n, source_file)).dictionary_encode(),
            pa.array(np.full(n, source_capture)).dictionary_encode(),
        ],
        schema=SCHEMA,
    )


# --------------------------------------------------------------------------------------------
# Car-Hacking (HCRL)
# --------------------------------------------------------------------------------------------

CAR_HACKING_FILES = {
    # raw file (relative to data/raw/car_hacking) -> attack type
    "DoS_dataset.csv": "dos",
    "Fuzzy_dataset.csv": "fuzzy",
    "gear_dataset.csv": "gear_spoofing",
    "RPM_dataset.csv": "rpm_spoofing",
    "normal_run_data/normal_run_data.txt": None,
}


def byte_cells_to_uint8(cells: np.ndarray, dlc: np.ndarray) -> np.ndarray:
    """Decode an (N, 8) object array of 1-byte hex strings; cells at index >= dlc become 0."""
    cells = cells.copy()
    cells[np.arange(8)[None, :] >= dlc[:, None]] = "00"
    return np.stack(
        [hex_to_uint8(np.char.zfill(cells[:, j].astype(str), 2), 1)[:, 0] for j in range(8)], axis=1
    )


def parse_car_hacking_attack_chunk(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Parse rows ``ts, ID, DLC, b0..b{DLC-1}, flag`` (12 columns padded with NaN)."""
    dlc = df[2].astype(np.int8).to_numpy()
    if dlc.min() < 0 or dlc.max() > 8:
        raise ValueError("DLC out of range")
    cells = df[list(range(3, 12))].fillna("00").to_numpy(dtype=object)
    flag = cells[np.arange(len(df)), dlc]
    if not np.isin(flag, ["R", "T"]).all():
        raise ValueError(f"unexpected flag values {set(flag) - {'R', 'T'}}")
    return {
        "timestamp": df[0].astype(np.float64).to_numpy(),
        "can_id": hex_ids(df[1]),
        "dlc": dlc,
        "data": byte_cells_to_uint8(cells[:, :8], dlc),
        "label": (flag == "T").astype(np.int8),
    }


def parse_car_hacking_normal_chunk(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], int]:
    """Parse ``Timestamp: <ts> ID: <id> 000 DLC: <n> <b0> ...`` rows; drops truncated rows."""
    complete = df[6].notna()
    n_dropped = int((~complete).sum())
    df = df[complete]
    dlc = df[6].astype(np.int8).to_numpy()
    cells = df[list(range(7, 15))].fillna("00").to_numpy(dtype=object)
    data = byte_cells_to_uint8(cells, dlc)
    return (
        {
            "timestamp": df[1].astype(np.float64).to_numpy(),
            "can_id": hex_ids(df[3]),
            "dlc": dlc,
            "data": data,
            "label": np.zeros(len(df), dtype=np.int8),
        },
        n_dropped,
    )


def read_car_hacking(path: Path, attack_type: str | None, chunksize: int = CHUNK_ROWS) -> Iterator[pa.Table]:
    """Yield common-schema tables chunk by chunk for one Car-Hacking file."""
    source_file = path.relative_to(RAW_DIR).as_posix()
    capture = capture_id("car_hacking", path)
    if attack_type is None:
        reader = pd.read_csv(path, sep=r"\s+", header=None, names=range(15), dtype=str, chunksize=chunksize)
        dropped = 0
        for chunk in reader:
            cols, n = parse_car_hacking_normal_chunk(chunk)
            dropped += n
            yield to_table(**cols, attack_type=NORMAL, source_file=source_file, source_capture=capture)
        if dropped:
            log(f"{source_file}: dropped {dropped} truncated row(s)")
    else:
        reader = pd.read_csv(path, sep=",", header=None, names=range(12), dtype=str, chunksize=chunksize)
        for chunk in reader:
            cols = parse_car_hacking_attack_chunk(chunk)
            yield to_table(**cols, attack_type=attack_type, source_file=source_file, source_capture=capture)


# --------------------------------------------------------------------------------------------
# ROAD (ORNL)
# --------------------------------------------------------------------------------------------

ROAD_DIR = RAW_DIR / "road" / "road"
FUZZ_PAYLOAD = np.full(8, 0xFF, dtype=np.uint8)


@dataclass(frozen=True)
class RoadCapture:
    name: str
    folder: str  # "ambient" or "attacks"
    meta: dict

    @property
    def path(self) -> Path:
        return ROAD_DIR / self.folder / f"{self.name}.log"

    @property
    def is_masquerade(self) -> bool:
        return self.name.endswith("_masquerade")

    @property
    def source_name(self) -> str:
        """Capture this log was derived from (itself unless it is a masquerade log)."""
        return self.name.removesuffix("_masquerade")

    @property
    def kind(self) -> str:
        if self.folder == "ambient":
            return "ambient"
        if self.meta.get("injection_id") is None:
            return "unlabeled"  # accelerator: compromised before the capture, no injected frame
        if self.meta["injection_id"] == "XXX":
            return "fuzzing"
        return "targeted"

    @property
    def attack_type(self) -> str:
        if self.kind == "ambient" or self.kind == "unlabeled":
            return NORMAL
        base = self.source_name.split("_attack")[0]
        return f"{base}_masquerade" if self.is_masquerade else base


def road_captures() -> list[RoadCapture]:
    out = []
    for folder in ("ambient", "attacks"):
        meta = json.loads((ROAD_DIR / folder / "capture_metadata.json").read_text(encoding="utf-8"))
        out.extend(RoadCapture(name, folder, m) for name, m in sorted(meta.items()))
    return out


def parse_injection_pattern(pattern: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, value) uint8[8] arrays for an ``injection_data_str`` like ``XXXX0CXXXXXXXXXX``."""
    pattern = pattern.upper()
    if len(pattern) != 16:
        raise ValueError(f"unexpected pattern {pattern!r}")
    mask = np.zeros(8, dtype=np.uint8)
    value = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        for j, ch in enumerate(pattern[2 * i : 2 * i + 2]):
            shift = 4 * (1 - j)
            if ch != "X":
                mask[i] |= 0xF << shift
                value[i] |= int(ch, 16) << shift
    return mask, value


def road_labels(cap: RoadCapture, rel_us: np.ndarray, can_id: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Frame-level labels for one ROAD chunk; ``rel_us`` is microseconds since the capture start."""
    kind = cap.kind
    if kind in ("ambient", "unlabeled"):
        return np.zeros(len(rel_us), dtype=np.int8)
    start, end = (int(round(x * 1e6)) for x in cap.meta["injection_interval"])
    if kind == "fuzzing":
        return ((data == FUZZ_PAYLOAD).all(axis=1) & (rel_us >= start)).astype(np.int8)
    mask, value = parse_injection_pattern(cap.meta["injection_data_str"])
    target = int(cap.meta["injection_id"], 16)
    match = ((data & mask) == value).all(axis=1)
    inside = (rel_us >= start) & (rel_us <= end)
    return ((can_id == target) & match & inside).astype(np.int8)


def parse_road_chunk(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Parse SocketCAN lines ``(sec.usec) can0 ID#PAYLOAD``."""
    ts = df["ts"].str.slice(1, -1)
    parts = ts.str.split(".", n=1, expand=True)
    if not (parts[1].str.len() == 6).all():
        raise ValueError("ROAD timestamps are expected to have microsecond resolution")
    us = parts[0].astype(np.int64).to_numpy() * 1_000_000 + parts[1].astype(np.int64).to_numpy()
    frame = df["frame"].str.split("#", n=1, expand=True)
    payload = frame[1].fillna("")
    dlc = (payload.str.len() // 2).to_numpy(dtype=np.int8)
    padded = payload.str.ljust(16, "0").to_numpy(dtype=str)
    return {
        "us": us,
        "timestamp": ts.astype(np.float64).to_numpy(),
        "can_id": hex_ids(frame[0]),
        "dlc": dlc,
        "data": hex_to_uint8(padded, 8),
    }


# Every ROAD line observed is fixed width: "(SSSSSSSSSS.UUUUUU) can0 III#PPPPPPPPPPPPPPPP\n".
ROAD_LINE_BYTES = 46
_ROAD_LITERALS = {0: b"(", 11: b".", 18: b")", 19: b" ", 20: b"c", 21: b"a", 22: b"n", 23: b"0", 24: b" ", 28: b"#", 45: b"\n"}
_POW10 = 10 ** np.arange(9, -1, -1, dtype=np.int64)


def parse_road_fixed(buf: bytes) -> dict[str, np.ndarray] | None:
    """Vectorized parser for fixed-width ROAD lines; returns None if any line deviates."""
    if len(buf) % ROAD_LINE_BYTES:
        return None
    a = np.frombuffer(buf, dtype=np.uint8).reshape(-1, ROAD_LINE_BYTES)
    for pos, ch in _ROAD_LITERALS.items():
        if not (a[:, pos] == ch[0]).all():
            return None
    digits = a[:, np.r_[1:11, 12:18]].astype(np.int64) - 48
    if ((digits < 0) | (digits > 9)).any():
        return None
    sec = digits[:, :10] @ _POW10
    usec = digits[:, 10:] @ _POW10[4:]
    nib = _NIBBLE[a[:, np.r_[25:28, 29:45]]]  # 3 ID nibbles + 16 payload nibbles ('#' skipped)
    if (nib < 0).any():
        return None
    can_id = (nib[:, 0] * 256 + nib[:, 1] * 16 + nib[:, 2]).astype(np.int32)
    data = (nib[:, 3::2] * 16 + nib[:, 4::2]).astype(np.uint8)
    return {
        "us": sec * 1_000_000 + usec,
        "timestamp": sec.astype(np.float64) + usec / 1e6,
        "can_id": can_id,
        "dlc": np.full(len(a), 8, dtype=np.int8),
        "data": data,
    }


def iter_road_chunks(path: Path, chunksize: int) -> Iterator[dict[str, np.ndarray]]:
    """Parse a ROAD log with the fixed-width parser, falling back to pandas for irregular files."""
    size = path.stat().st_size
    if size % ROAD_LINE_BYTES == 0:
        parsed = []
        with path.open("rb") as f:
            while buf := f.read(chunksize * ROAD_LINE_BYTES):
                cols = parse_road_fixed(buf)
                if cols is None:
                    parsed = None
                    break
                parsed.append(cols)
        if parsed is not None:
            yield from parsed
            return
    reader = pd.read_csv(path, sep=" ", header=None, names=["ts", "iface", "frame"], dtype=str, chunksize=chunksize)
    for chunk in reader:
        yield parse_road_chunk(chunk)


def read_road(cap: RoadCapture, chunksize: int = CHUNK_ROWS) -> Iterator[pa.Table]:
    """Yield common-schema tables chunk by chunk for one ROAD capture."""
    source_file = cap.path.relative_to(RAW_DIR).as_posix()
    source_capture = f"road/{cap.source_name}"
    t0_us = None
    for cols in iter_road_chunks(cap.path, chunksize):
        if t0_us is None:
            t0_us = int(cols["us"][0])
        label = road_labels(cap, cols["us"] - t0_us, cols["can_id"], cols["data"])
        yield to_table(
            timestamp=cols["timestamp"],
            can_id=cols["can_id"],
            dlc=cols["dlc"],
            data=cols["data"],
            label=label,
            attack_type=cap.attack_type,
            source_file=source_file,
            source_capture=source_capture,
        )


def verify_masquerade_lineage(source: pa.Table, derived: pa.Table, meta: dict) -> dict[str, object]:
    """Check that ``derived`` equals ``source`` minus the conflicting legitimate frames.

    Frames are compared on (time since capture start, ID, payload); ROAD anonymizes the absolute
    start time of every log separately. Expected: every derived frame exists in the source, and
    every removed frame is a non-injected frame of the target ID inside the injection interval.
    """

    def keys(t: pa.Table) -> pd.Series:
        ts = t["timestamp"].to_numpy()
        rel_us = np.round((ts - ts[0]) * 1e6).astype(np.int64)
        payload = np.ascontiguousarray(data_matrix(t)).view(np.uint64)[:, 0]
        return (
            pd.Series(rel_us).astype(str)
            + ":" + pd.Series(t["can_id"].to_numpy()).astype(str)
            + ":" + pd.Series(payload).astype(str)
        )

    ks, kd = keys(source), keys(derived)
    in_source = kd.isin(set(ks))
    removed = ~ks.isin(set(kd))
    target = int(meta["injection_id"], 16)
    start, end = meta["injection_interval"]
    ts = source["timestamp"].to_numpy()
    rel = ts - ts[0]
    src_label = source["label"].to_numpy()
    src_id = source["can_id"].to_numpy()
    removed_ok = removed & (src_id == target) & (src_label == 0) & (rel >= start - 1e-6) & (rel <= end + 1e-6)
    return {
        "n_source": source.num_rows,
        "n_derived": derived.num_rows,
        "n_removed": int(removed.sum()),
        "derived_frames_in_source": float(in_source.mean()),
        "removed_conflicting_legit": int(removed_ok.sum()),
        "injected_frames_kept": int(derived["label"].to_numpy().sum()),
        "injected_frames_in_source": int(src_label.sum()),
        "verified": bool(in_source.all() and removed_ok.sum() == removed.sum()
                         and derived["label"].to_numpy().sum() == src_label.sum()),
    }


# --------------------------------------------------------------------------------------------
# Conversion driver and helpers
# --------------------------------------------------------------------------------------------


def capture_id(dataset: str, path: Path) -> str:
    stem = path.stem
    return f"{dataset}/{stem}"


def parquet_path(capture: str) -> Path:
    return PROCESSED_DIR / f"{capture}.parquet"


def read_processed(path: Path, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(path, columns=columns)


def data_matrix(table: pa.Table) -> np.ndarray:
    """Return the ``data`` column as an (N, 8) uint8 array."""
    col = table["data"].combine_chunks()
    return pc.list_flatten(col).to_numpy().reshape(-1, 8)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_parquet(tables: Iterator[pa.Table], out: Path) -> dict[str, object]:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    n = n_attack = 0
    attack_types: set[str] = set()
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for t in tables:
            writer.write_table(t)
            lab = t["label"].to_numpy()
            n += t.num_rows
            n_attack += int(lab.sum())
            attack_types |= set(pc.unique(t["attack_type"].combine_chunks().cast(pa.string())).to_pylist())
    tmp.replace(out)
    return {"n_frames": n, "n_attack": n_attack, "attack_types": ";".join(sorted(attack_types - {NORMAL}))}


INDEX_FIELDS = [
    "dataset", "capture", "source_file", "source_capture", "parquet", "kind",
    "n_frames", "n_attack", "attack_types", "excluded", "exclude_reason",
]


def convert_all(force: bool = False) -> list[dict[str, object]]:
    index_path = PROCESSED_DIR / "index.csv"
    previous = {}
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as f:
            previous = {r["capture"]: r for r in csv.DictReader(f)}

    jobs = []
    for rel, attack in CAR_HACKING_FILES.items():
        path = RAW_DIR / "car_hacking" / rel
        cap = capture_id("car_hacking", path)
        jobs.append(({"dataset": "car_hacking", "capture": cap, "source_file": path.relative_to(RAW_DIR).as_posix(),
                      "source_capture": cap, "kind": "attack" if attack else "ambient",
                      "excluded": False, "exclude_reason": ""},
                     lambda p=path, a=attack: read_car_hacking(p, a)))
    for rc in road_captures():
        cap = f"road/{rc.name}"
        unlabeled = rc.kind == "unlabeled"
        jobs.append(({"dataset": "road", "capture": cap, "source_file": rc.path.relative_to(RAW_DIR).as_posix(),
                      "source_capture": f"road/{rc.source_name}", "kind": rc.kind,
                      "excluded": unlabeled,
                      "exclude_reason": "no frame-level label (injection before capture start)" if unlabeled else ""},
                     lambda c=rc: read_road(c)))

    rows = []
    for info, make_reader in jobs:
        out = parquet_path(info["capture"])
        info["parquet"] = out.relative_to(PROJECT_ROOT).as_posix()
        if out.exists() and not force and info["capture"] in previous:
            prev = previous[info["capture"]]
            info.update({k: prev[k] for k in ("n_frames", "n_attack", "attack_types")})
            log(f"{info['capture']}: exists, skipping")
        else:
            log(f"{info['capture']}: converting {info['source_file']}")
            info.update(write_parquet(make_reader(), out))
            log(f"{info['capture']}: {info['n_frames']} frames, {info['n_attack']} injected")
        rows.append(info)
        write_index(rows, index_path)  # keep progress after each file
    write_lineage([r for r in rows if r["dataset"] == "road"])
    return rows


def write_index(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)


def write_lineage(road_rows: list[dict[str, object]]) -> None:
    """Verify every derived ROAD capture against its source and write the lineage table."""
    metas = {c.name: c.meta for c in road_captures()}
    out_rows = []
    for r in road_rows:
        name = str(r["capture"]).split("/", 1)[1]
        source = str(r["source_capture"])
        row = {"capture": r["capture"], "source_capture": source, "derived": source != r["capture"]}
        if row["derived"]:
            src_tab = read_processed(parquet_path(source))
            der_tab = read_processed(parquet_path(str(r["capture"])))
            check = verify_masquerade_lineage(src_tab, der_tab, metas[name])
            if not check["verified"]:
                raise RuntimeError(f"lineage check failed for {r['capture']}: {check}")
            row.update(check)
        out_rows.append(row)
    fields = ["capture", "source_capture", "derived", "n_source", "n_derived", "n_removed",
              "derived_frames_in_source", "removed_conflicting_legit", "injected_frames_kept",
              "injected_frames_in_source", "verified"]
    for path in (PROCESSED_DIR / "capture_lineage.csv", LINEAGE_TABLE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
    log(f"lineage: {sum(r['derived'] for r in out_rows)} derived captures verified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert raw CAN logs to the common parquet schema.")
    parser.add_argument("--force", action="store_true", help="reconvert files that already exist")
    args = parser.parse_args(argv)
    convert_all(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
