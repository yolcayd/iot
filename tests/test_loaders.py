"""Tests for src/data/loaders.py.

Synthetic tests always run. Tests marked ``real_data`` run against data/processed and data/raw and
are skipped when the datasets have not been downloaded and converted. Expected label counts are
re-derived from the raw files with an independent, line-by-line implementation.
"""

from __future__ import annotations

import csv
import io
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from src.data import loaders as L

INDEX = L.PROCESSED_DIR / "index.csv"
real_data = pytest.mark.skipif(not INDEX.exists(), reason="processed data not available")


# ------------------------------------------------------------------------------------------
# Synthetic: parsers and labeling rules
# ------------------------------------------------------------------------------------------

CAR_HACKING_SAMPLE = """\
1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R
1478198376.389636,0000,8,00,00,00,00,00,00,00,00,T
1478198376.389864,02b0,5,ff,7f,00,05,49,R
1478198376.390096,05f0,2,01,00,R
1478198376.390333,043f,8,01,45,60,ff,6b,00,00,00,T
"""

CAR_HACKING_NORMAL_SAMPLE = """\
Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2
Timestamp: 1479121434.851463        ID: 02b0    000    DLC: 5    ff 7f 00 05 49
Timestamp: 1479121434.851700        ID: 05f0    000    DLC: 2    01 00
Timestamp: 1479121941.287013
"""


def test_car_hacking_attack_rows_with_short_dlc():
    df = pd.read_csv(io.StringIO(CAR_HACKING_SAMPLE), header=None, names=range(12), dtype=str)
    cols = L.parse_car_hacking_attack_chunk(df)
    assert cols["dlc"].tolist() == [8, 8, 5, 2, 8]
    assert cols["label"].tolist() == [0, 1, 0, 0, 1]
    assert cols["can_id"].tolist() == [0x316, 0x000, 0x2B0, 0x5F0, 0x43F]
    assert cols["data"][2].tolist() == [0xFF, 0x7F, 0x00, 0x05, 0x49, 0, 0, 0]
    assert cols["data"][3].tolist() == [0x01, 0x00, 0, 0, 0, 0, 0, 0]
    assert cols["data"][0].tolist() == [0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6F]
    assert cols["timestamp"][0] == pytest.approx(1478198376.389427, abs=1e-6)


def test_car_hacking_attack_rejects_unknown_flag():
    df = pd.read_csv(io.StringIO("1.0,0316,2,01,02,X\n"), header=None, names=range(12), dtype=str)
    with pytest.raises(ValueError):
        L.parse_car_hacking_attack_chunk(df)


def test_car_hacking_normal_format_and_truncated_row():
    df = pd.read_csv(io.StringIO(CAR_HACKING_NORMAL_SAMPLE), sep=r"\s+", header=None, names=range(15), dtype=str)
    cols, dropped = L.parse_car_hacking_normal_chunk(df)
    assert dropped == 1
    assert cols["dlc"].tolist() == [8, 5, 2]
    assert cols["can_id"].tolist() == [0x350, 0x2B0, 0x5F0]
    assert cols["data"][1].tolist() == [0xFF, 0x7F, 0x00, 0x05, 0x49, 0, 0, 0]
    assert cols["label"].sum() == 0


ROAD_SAMPLE = (
    "(1030000000.000000) can0 0D0#0000040000000000\n"   # target ID, matches pattern, before interval
    "(1030000001.000000) can0 0D0#00000C0000000000\n"   # target ID, inside, wrong payload (legit)
    "(1030000001.500000) can0 0D0#1122041122334455\n"   # target ID, inside, matches -> injected
    "(1030000002.000000) can0 0D1#0000040000000000\n"   # other ID
    "(1030000003.000000) can0 0D0#0000040000000000\n"   # interval end, inclusive -> injected
    "(1030000003.000001) can0 0D0#0000040000000000\n"   # just after the interval
)


def road_cols(text: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(io.StringIO(text), sep=" ", header=None, names=["ts", "iface", "frame"], dtype=str)
    return L.parse_road_chunk(df)


def test_road_fixed_width_parser_matches_pandas_parser():
    fixed = L.parse_road_fixed(ROAD_SAMPLE.encode())
    slow = road_cols(ROAD_SAMPLE)
    assert fixed is not None
    for k in ("us", "timestamp", "can_id", "dlc", "data"):
        np.testing.assert_array_equal(fixed[k], slow[k])
    assert fixed["data"][2].tolist() == [0x11, 0x22, 0x04, 0x11, 0x22, 0x33, 0x44, 0x55]


def test_road_fixed_width_parser_rejects_irregular_lines():
    assert L.parse_road_fixed(b"(1030000000.000000) can0 0D0#00000400\n") is None
    bad = ROAD_SAMPLE.replace("can0", "can1", 1).encode()
    assert L.parse_road_fixed(bad) is None


def test_injection_pattern():
    mask, value = L.parse_injection_pattern("XXXX0CXXXXXXXXXX")
    assert mask.tolist() == [0, 0, 0xFF, 0, 0, 0, 0, 0]
    assert value.tolist() == [0, 0, 0x0C, 0, 0, 0, 0, 0]
    mask, value = L.parse_injection_pattern("XXXXXXXXXXFXXXXX")
    assert mask[5] == 0xF0 and value[5] == 0xF0


def targeted_capture(name: str = "reverse_light_off_attack_1") -> L.RoadCapture:
    meta = {"injection_id": "0xd0", "injection_data_str": "XXXX04XXXXXXXXXX", "injection_interval": [1.0, 3.0]}
    return L.RoadCapture(name, "attacks", meta)


def test_road_targeted_labels():
    cols = road_cols(ROAD_SAMPLE)
    labels = L.road_labels(targeted_capture(), cols["us"] - cols["us"][0], cols["can_id"], cols["data"])
    assert labels.tolist() == [0, 0, 1, 0, 1, 0]


def test_road_fuzzing_labels():
    text = (
        "(1000000000.000000) can0 48D#FFFFFFFFFFFFFFFF\n"   # before the injection start
        "(1000000001.000000) can0 48D#FFFFFFFFFFFFFFFF\n"
        "(1000000001.100000) can0 0CA#FFFFFFFFFFFFFFFE\n"
        "(1000000002.100000) can0 0CA#FFFFFFFFFFFFFFFF\n"   # trails the metadata end -> still injected
    )
    cap = L.RoadCapture("fuzzing_attack_1", "attacks",
                        {"injection_id": "XXX", "injection_data_str": "F" * 16, "injection_interval": [1.0, 2.0]})
    cols = road_cols(text)
    assert L.road_labels(cap, cols["us"] - cols["us"][0], cols["can_id"], cols["data"]).tolist() == [0, 1, 0, 1]


def test_road_capture_naming_and_lineage_ids():
    assert targeted_capture("max_speedometer_attack_2").attack_type == "max_speedometer"
    masq = targeted_capture("max_speedometer_attack_2_masquerade")
    assert masq.attack_type == "max_speedometer_masquerade"
    assert masq.source_name == "max_speedometer_attack_2"
    acc = L.RoadCapture("accelerator_attack_drive_1", "attacks", {"injection_id": None})
    assert acc.kind == "unlabeled" and acc.attack_type == L.NORMAL
    amb = L.RoadCapture("ambient_dyno_reverse", "ambient", {})
    assert amb.kind == "ambient" and amb.source_name == "ambient_dyno_reverse"


def synthetic_table(text: str, cap: L.RoadCapture) -> pa.Table:
    cols = road_cols(text)
    lab = L.road_labels(cap, cols["us"] - cols["us"][0], cols["can_id"], cols["data"])
    return L.to_table(cols["timestamp"], cols["can_id"], cols["dlc"], cols["data"], lab,
                      cap.attack_type, "x.log", f"road/{cap.source_name}")


def test_masquerade_lineage_verification():
    src_cap, masq_cap = targeted_capture(), targeted_capture("reverse_light_off_attack_1_masquerade")
    src = synthetic_table(ROAD_SAMPLE, src_cap)
    # Remove the legit conflicting frame (line 2) and shift the anonymized absolute start time.
    masq_lines = [ln for i, ln in enumerate(ROAD_SAMPLE.splitlines(keepends=True)) if i != 1]
    masq_text = "".join(ln.replace("(103", "(106", 1) for ln in masq_lines)
    check = L.verify_masquerade_lineage(src, synthetic_table(masq_text, masq_cap), src_cap.meta)
    assert check["verified"] and check["n_removed"] == 1
    # Removing an injected frame must fail verification.
    bad_lines = [ln for i, ln in enumerate(ROAD_SAMPLE.splitlines(keepends=True)) if i != 2]
    bad = L.verify_masquerade_lineage(src, synthetic_table("".join(bad_lines), masq_cap), src_cap.meta)
    assert not bad["verified"]


def test_to_table_schema():
    cols = road_cols(ROAD_SAMPLE)
    t = L.to_table(cols["timestamp"], cols["can_id"], cols["dlc"], cols["data"],
                   np.zeros(len(cols["us"]), np.int8), "x", "f", "c")
    assert t.schema.equals(L.SCHEMA)
    assert set(t["attack_type"].to_pylist()) == {L.NORMAL}
    assert L.data_matrix(t).shape == (len(cols["us"]), 8)


# ------------------------------------------------------------------------------------------
# Real data: schema and label consistency with the raw files / metadata
# ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def index_rows() -> list[dict[str, str]]:
    with INDEX.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def road_meta() -> dict[str, dict]:
    return {c.name: c.meta for c in L.road_captures()}


@real_data
def test_index_covers_every_raw_capture(index_rows, road_meta):
    captures = {r["capture"] for r in index_rows}
    expected = {L.capture_id("car_hacking", L.RAW_DIR / "car_hacking" / f) for f in L.CAR_HACKING_FILES}
    expected |= {f"road/{n}" for n in road_meta}
    assert captures == expected
    assert sum(r["dataset"] == "road" for r in index_rows) == 45  # 12 ambient + 33 attack logs


@real_data
def test_schema_and_label_invariants(index_rows):
    for r in index_rows:
        t = L.read_processed(L.PROJECT_ROOT / r["parquet"])
        assert t.schema.equals(L.SCHEMA), r["capture"]
        assert t.num_rows == int(r["n_frames"])
        label = t["label"].to_numpy()
        is_attack = pc.not_equal(t["attack_type"].cast(pa.string()), L.NORMAL).to_numpy(zero_copy_only=False)
        dlc = t["dlc"].to_numpy()
        data = L.data_matrix(t)
        assert set(np.unique(label)) <= {0, 1}
        assert int(label.sum()) == int(r["n_attack"])
        np.testing.assert_array_equal(label == 1, is_attack)
        assert dlc.min() >= 0 and dlc.max() <= 8
        assert not (data[np.arange(8)[None, :] >= dlc[:, None]]).any(), "bytes beyond DLC must be zero"
        assert (np.diff(t["timestamp"].to_numpy()) >= 0).all(), "timestamps must be non-decreasing"
        assert t["source_file"].unique().to_pylist() == [r["source_file"]]
        assert t["source_capture"].unique().to_pylist() == [r["source_capture"]]


def count_car_hacking_raw(path) -> tuple[int, int]:
    """Independent recount: complete rows and rows whose last field is 'T'."""
    rows = injected = 0
    with open(path, encoding="ascii") as f:
        for line in f:
            fields = line.strip().split(",")
            if len(fields) < 4:
                continue
            rows += 1
            injected += fields[-1] == "T"
    return rows, injected


@real_data
@pytest.mark.parametrize("name", [n for n, a in L.CAR_HACKING_FILES.items() if a])
def test_car_hacking_counts_match_raw(index_rows, name):
    row = next(r for r in index_rows if r["source_file"] == f"car_hacking/{name}")
    rows, injected = count_car_hacking_raw(L.RAW_DIR / "car_hacking" / name)
    assert int(row["n_frames"]) == rows
    assert int(row["n_attack"]) == injected


@real_data
def test_car_hacking_normal_counts_match_raw(index_rows):
    row = next(r for r in index_rows if r["capture"] == "car_hacking/normal_run_data")
    with open(L.RAW_DIR / "car_hacking/normal_run_data/normal_run_data.txt", encoding="ascii") as f:
        complete = sum("DLC:" in line for line in f)
    assert int(row["n_frames"]) == complete
    assert int(row["n_attack"]) == 0


def recount_road(name: str, meta: dict) -> int:
    """Independent recount of injected frames from the raw ROAD log using string operations."""
    path = L.ROAD_DIR / "attacks" / f"{name}.log"
    start, end = (round(x * 1e6) for x in meta["injection_interval"])
    if meta["injection_id"] == "XXX":
        target, regex = None, re.compile("F" * 16)
    else:
        target = int(meta["injection_id"], 16)
        regex = re.compile(meta["injection_data_str"].upper().replace("X", "."))
    t0 = None
    n = 0
    with open(path, encoding="ascii") as f:
        for line in f:
            ts, _, frame = line.split()
            us = int(ts[1:-1].replace(".", ""))
            t0 = us if t0 is None else t0
            rel = us - t0
            aid, payload = frame.split("#")
            if target is None:
                n += rel >= start and payload.upper() == "F" * 16
            else:
                n += start <= rel <= end and int(aid, 16) == target and bool(regex.fullmatch(payload.upper()))
    return n


@real_data
def test_road_counts_match_metadata_recount(index_rows, road_meta):
    for name, meta in road_meta.items():
        if "injection_id" not in meta:
            continue  # ambient
        row = next(r for r in index_rows if r["capture"] == f"road/{name}")
        if meta["injection_id"] is None:
            assert int(row["n_attack"]) == 0 and row["excluded"] == "True"
            continue
        assert int(row["n_attack"]) == recount_road(name, meta) > 0, name


@real_data
def test_road_injected_frames_respect_metadata(index_rows, road_meta):
    for r in index_rows:
        name = r["capture"].split("/", 1)[1]
        meta = road_meta.get(name, {})
        if r["dataset"] != "road" or meta.get("injection_id") in (None, "XXX"):
            continue
        t = L.read_processed(L.PROJECT_ROOT / r["parquet"])
        ts = t["timestamp"].to_numpy()
        rel = ts - ts[0]
        lab = t["label"].to_numpy() == 1
        cid = t["can_id"].to_numpy()
        start, end = meta["injection_interval"]
        target = int(meta["injection_id"], 16)
        inside = (rel >= start - 1e-6) & (rel <= end + 1e-6)
        assert (cid[lab] == target).all() and inside[lab].all(), name
        if name.endswith("_masquerade"):
            # Masquerade: no legitimate frame of the target ID is left inside the interval.
            assert not ((cid == target) & inside & ~lab).any(), name


@real_data
def test_source_capture_lineage(index_rows):
    with L.LINEAGE_TABLE.open(newline="", encoding="utf-8") as f:
        lineage = {r["capture"]: r for r in csv.DictReader(f)}
    for r in index_rows:
        name = r["capture"].split("/", 1)[1]
        if name.endswith("_masquerade"):
            assert r["source_capture"] == r["capture"].removesuffix("_masquerade")
            assert lineage[r["capture"]]["verified"] == "True"
            assert lineage[r["capture"]]["source_capture"] == r["source_capture"]
        else:
            assert r["source_capture"] == r["capture"]
    assert sum(v["derived"] == "True" for v in lineage.values()) == 13
