"""Leakage tests for the source-level splits, normalisation statistics and evaluation-window stride.

Plan F1 tests 1-4 plus the checks required by docs/DECISIONS.md K-015...K-022:

1. no CAN message occurs in two different splits: segments of one file do not overlap, and no
   32-frame sequence (IDs, payloads, inter-frame deltas; invariant to timestamp re-basing) is
   shared between splits; a positive control shows the check catches copied recordings
2. normalisation parameters were fitted on training segments only
3. validation/test stride equals the window size (no overlap)
4. every attack type has at least ``min_test_attack_windows`` test windows
5. every fabrication-masquerade pair (same source_capture) sits in one split
6. Car-Hacking temporal blocks are disjoint, time-ordered, cut in attack-free stretches and
   separated by at least ``min_block_gap_s`` (>= 10 x IAT cap); they match the block rule
7. the shared Car-Hacking tail is one recording, cut from every attack file and used once
8. excluded (accelerator) captures are in no split
9. the LSTM autoencoder subset holds training windows only, far from every injected frame
10. frames of every split segment are in recording (time) order: timestamps never decrease, so windows and
    per-ID features see the messages in the order they were sent (P10, K-072)

Tests run against data/processed and are skipped when the data has not been converted.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import features as F
from src.data import loaders as L
from src.data import sources as S
from src.data import splits as SP
from src.data import windows as W

INDEX = L.PROCESSED_DIR / "index.csv"
pytestmark = pytest.mark.skipif(not INDEX.exists(), reason="processed data not available")


@pytest.fixture(scope="module")
def cfg():
    return S.load_config()


@pytest.fixture(scope="module")
def views(cfg):
    return S.all_views(cfg)


@pytest.fixture(scope="module")
def segs(cfg, views):
    return SP.segments(cfg, views)


@pytest.fixture(scope="module")
def params():
    return F.load_params()


@pytest.fixture(scope="module")
def windows(cfg, segs):
    return SP.build_windows(segs, cfg)


@pytest.fixture(scope="module")
def summary(windows, segs, cfg):
    return SP.summarize(windows, segs, cfg)


# Message identity. ROAD timestamps are anonymised per capture and many captures start at the same
# round value (e.g. 1110000000.0), so a single (timestamp, ID, payload) key matches unrelated frames
# by chance, and a re-based copy of a recording (masquerade logs) has different timestamps altogether.
# A recorded message is therefore identified by the 32-frame sequence it starts: IDs, payloads and
# inter-frame time deltas in microseconds. This is invariant to timestamp re-basing; a copied
# recording shares thousands of such sequences, and independent recordings share none.
NGRAM = 32


def _frame_hashes(t) -> tuple[np.ndarray, np.ndarray]:
    """(content hash, content + delta-to-previous-frame hash) per frame."""
    ts = t["timestamp"].to_numpy()
    delta = np.r_[0, np.round(np.diff(ts) * 1e6).astype(np.int64)].view(np.uint64)
    cid = t["can_id"].to_numpy().astype(np.uint64)
    pay = np.ascontiguousarray(L.data_matrix(t)).view("<u8").ravel()
    with np.errstate(over="ignore"):
        content = cid * np.uint64(0xC2B2AE3D27D4EB4F)
        content ^= pay * np.uint64(0x165667B19E3779F9) + (content >> np.uint64(31))
        full = content ^ (delta * np.uint64(0x9E3779B97F4A7C15) + (content >> np.uint64(29)))
    return content, full


def sequence_keys(table) -> np.ndarray:
    """uint64 hash of every NGRAM-frame sequence; the first frame's delta points outside it and is left out."""
    content, full = _frame_hashes(table)
    n = len(full) - NGRAM + 1
    if n <= 0:
        return np.empty(0, np.uint64)
    with np.errstate(over="ignore"):
        g = content[:n].copy()
        for j in range(1, NGRAM):
            g = g * np.uint64(1099511628211) ^ full[j: j + n]
    return g


def sequence_rows(table, start: int) -> tuple:
    t = table.slice(start, NGRAM)
    ts = t["timestamp"].to_numpy()
    return (tuple(np.round(np.diff(ts) * 1e6).astype(np.int64).tolist()), tuple(t["can_id"].to_numpy().tolist()),
            bytes(L.data_matrix(t)))


SEQ_COLUMNS = ["timestamp", "can_id", "data"]


# 1 ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_no_message_shared_between_splits(segs, dataset):
    keys, where = {}, {}
    for split in SP.SPLITS:
        parts, owners = [], []
        for s in (x for x in segs if x.dataset == dataset and x.split == split):
            k = sequence_keys(s.read(SEQ_COLUMNS))
            parts.append(k)
            owners.append((s, k))
        keys[split] = np.unique(np.concatenate(parts))
        where[split] = owners
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        common = np.intersect1d(keys[a], keys[b], assume_unique=True)
        shared = []
        for key in common[:20]:  # confirm on the raw frames so a hash collision cannot fail the test
            seqs = []
            for split in (a, b):
                s, k = next((s, k) for s, k in where[split] if (k == key).any())
                seqs.append((s.id, sequence_rows(s.read(SEQ_COLUMNS), int(np.argmax(k == key)))))
            if seqs[0][1] == seqs[1][1]:
                shared.append((seqs[0][0], seqs[1][0]))
        assert not shared, f"{dataset}: {len(common)} {NGRAM}-frame sequences in both {a} and {b}, e.g. {shared[:3]}"


def test_sequence_check_detects_copied_recordings(cfg, views):
    """Positive control: the detector flags a re-based copy (masquerade) and the duplicated tail."""
    v = {x.capture: x for x in views}
    fab = sequence_keys(S.read_view(v["road/max_speedometer_attack_1"], SEQ_COLUMNS))
    masq = sequence_keys(S.read_view(v["road/max_speedometer_attack_1_masquerade"], SEQ_COLUMNS))
    assert len(np.intersect1d(fab, masq)) > 10_000
    tail = sequence_keys(S.read_view(v["car_hacking/shared_tail"], SEQ_COLUMNS))
    fuzzy = v["car_hacking/Fuzzy_dataset"]
    fuzzy_tail = L.read_processed(S.PROJECT_ROOT / fuzzy.parquet, columns=SEQ_COLUMNS).slice(fuzzy.stop)
    assert len(np.intersect1d(tail, sequence_keys(fuzzy_tail))) == len(np.unique(tail))


def test_segments_of_one_parquet_do_not_overlap(segs):
    by_file = defaultdict(list)
    for s in segs:
        by_file[s.parquet].append(s)
    for f, ss in by_file.items():
        ss = sorted(ss, key=lambda s: s.start)
        for a, b in zip(ss, ss[1:]):
            assert a.stop <= b.start, f"{f}: {a.id} overlaps {b.id}"


def test_timestamps_monotonic_within_segments(segs):
    bad = []
    for s in segs:
        ts = s.read(["timestamp"])["timestamp"].to_numpy()
        d = np.diff(ts)
        if (d < 0).any():
            bad.append((s.id, int((d < 0).sum()), float(d.min())))
    assert not bad, f"segments with decreasing timestamps (id, count, most negative step): {bad}"


# 2 ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_normalisation_fitted_on_training_segments_only(cfg, segs, params, dataset):
    p = params["datasets"][dataset]
    train = [s for s in segs if s.dataset == dataset and s.split == "train"]
    assert p["fitted_on_splits"] == ["train"]
    assert set(p["fitted_on_segments"]) == {s.id for s in train}
    held_out = {s.id for s in segs if s.dataset == dataset and s.split != "train"}
    assert not held_out & set(p["fitted_on_segments"])
    fit = F.fit_normalization(train, p["iat_floor_quantile"], p["iat_cap_quantile"])
    assert fit["iat_cap_s"] == pytest.approx(p["iat_cap_s"], rel=0, abs=1e-12)
    assert fit["iat_floor_s"] == pytest.approx(p["iat_floor_s"], rel=0, abs=1e-12)
    assert fit["n_iat_samples"] == p["n_iat_samples"]
    assert fit["id_vocab"] == p["id_vocab"] and p["n_ids"] == len(p["id_vocab"])
    assert p["iat_floor_quantile"] == cfg["features"]["iat"]["floor_quantile"]
    assert p["iat_cap_quantile"] == cfg["features"]["iat"]["cap_quantile"]
    with pytest.raises(ValueError):
        F.fit_normalization(train + [next(s for s in segs if s.dataset == dataset and s.split == "test")], 0.001, 0.999)


@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_id_vocabulary_from_benign_training_frames_only(segs, params, dataset):
    """No ID that appears only in validation/test (or only in injected frames) enters the vocabulary."""
    vocab = set(params["datasets"][dataset]["id_vocab"])
    benign_train = set()
    for s in segs:
        if s.dataset == dataset and s.split == "train":
            t = s.read(["can_id", "label"])
            benign_train |= set(np.unique(t["can_id"].to_numpy()[t["label"].to_numpy() == 0]).tolist())
    assert vocab == benign_train


def test_normalisation_params_match_config(cfg, params):
    current = F.config_hash({k: cfg[k] for k in ("features", "splits", "shared_segments")})
    assert params["config_sha256_16"] == current, "configs/data.yaml changed; re-run python -m src.data.features"


# 3 ---------------------------------------------------------------------------------------------


def test_eval_stride_equals_window_size(cfg, windows):
    size = cfg["windows"]["size"]
    assert cfg["windows"]["stride"]["val"] == size
    assert cfg["windows"]["stride"]["test"] == size
    for split in ("val", "test"):
        w = windows[windows.split == split]
        for seg, g in w.groupby("segment", observed=True):
            st = np.sort(g["start"].to_numpy())
            assert (np.diff(st) == size).all(), f"{split} {seg}: stride != window size"


def test_train_stride(cfg, windows):
    w = windows[windows.split == "train"]
    for seg, g in w.groupby("segment", observed=True):
        assert (np.diff(np.sort(g["start"].to_numpy())) == cfg["windows"]["stride"]["train"]).all(), seg


def test_windows_stay_inside_their_segment(cfg, segs, windows):
    size = cfg["windows"]["size"]
    bounds = {s.id: (s.start, s.stop) for s in segs}
    lo = windows["segment"].map(lambda k: bounds[k][0]).astype(np.int64)
    hi = windows["segment"].map(lambda k: bounds[k][1]).astype(np.int64)
    assert ((windows["start"] >= lo) & (windows["start"] + size <= hi)).all()


# 4 ---------------------------------------------------------------------------------------------


def test_every_attack_type_represented_in_test(cfg, summary):
    assert SP.check_test_coverage(summary, cfg) == []
    groups = yaml.safe_load(SP.GROUPS_PATH.read_text(encoding="utf-8"))
    tested = set(map(tuple, summary.loc[summary.split == "test", ["dataset", "attack_type"]].to_numpy()))
    assert {(d, a) for d, t in groups.items() for a in t} <= tested


def test_split_summary_table_is_current(windows, segs, cfg):
    on_disk = pd.read_csv(SP.SUMMARY_PATH).fillna({"group": ""})
    fresh = SP.summarize(windows, segs, cfg)
    pd.testing.assert_frame_equal(on_disk, fresh, check_dtype=False, check_categorical=False)


def test_validation_types(cfg, summary):
    """Every attack type except the declared ones has validation windows."""
    groups = yaml.safe_load(SP.GROUPS_PATH.read_text(encoding="utf-8"))
    val = set(map(tuple, summary.loc[summary.split == "val", ["dataset", "attack_type"]].to_numpy()))
    roles = cfg["splits"]["analysis_roles"]
    no_val = {tuple(x.split("/")) for x in roles["no_validation_windows"]}
    for d, types in groups.items():
        for a in types:
            assert ((d, a) in val) != ((d, a) in no_val), (d, a)
    unseen = {tuple(x.split("/")) for x in roles["unseen_in_training"]}
    train = set(map(tuple, summary.loc[summary.split == "train", ["dataset", "attack_type"]].to_numpy()))
    assert not unseen & train


# 5 ---------------------------------------------------------------------------------------------


def test_masquerade_pairs_share_a_split(segs, views):
    split_of_source = defaultdict(set)
    for s in segs:
        split_of_source[s.source_capture].add(s.split)
    for src, splits in split_of_source.items():
        if src.startswith("road/"):
            assert len(splits) == 1, f"{src} spread over {splits}"
    by_capture = {s.capture: s.split for s in segs if s.dataset == "road"}
    masq = [v for v in views if v.is_masquerade]
    assert len(masq) == 13
    for v in masq:
        assert v.source_capture != v.capture and v.source_capture in by_capture, v.capture
        assert by_capture[v.capture] == by_capture[v.source_capture], v.capture
    # in particular every masquerade test capture has its fabrication twin in test (K-013)
    test_masq = [v for v in masq if by_capture[v.capture] == "test"]
    assert test_masq and all(by_capture[v.source_capture] == "test" for v in test_masq)


# 6 ---------------------------------------------------------------------------------------------


def test_temporal_blocks_match_rule(cfg, views):
    assert SP.propose_car_hacking_blocks(cfg, views) == cfg["splits"]["car_hacking"]["blocks"]


def test_temporal_blocks_gap_and_order(cfg, segs, params):
    min_gap = cfg["splits"]["min_block_gap_s"]
    assert min_gap >= 10 * params["datasets"]["car_hacking"]["iat_cap_s"] - 1e-9
    burst_gap = cfg["splits"]["temporal_block_rule"]["burst_gap_s"]
    order = {"train": 0, "val": 1, "test": 2}
    for cap in cfg["splits"]["car_hacking"]["blocks"]:
        ss = sorted((s for s in segs if s.capture == cap), key=lambda s: s.start)
        assert [s.split for s in ss] == sorted((s.split for s in ss), key=order.get), cap
        t = L.read_processed(S.PROJECT_ROOT / ss[0].parquet, columns=["timestamp", "label"])
        ts, lab = t["timestamp"].to_numpy(), t["label"].to_numpy()
        inj = ts[lab == 1]
        for a, b in zip(ss, ss[1:]):
            gap = ts[b.start] - ts[a.stop - 1]
            assert gap >= min_gap, f"{cap}: gap {gap:.2f} s between {a.split} and {b.split}"
            for row in (a.stop - 1, b.start):  # cut points lie in attack-free stretches
                if len(inj):
                    assert np.min(np.abs(inj - ts[row])) > burst_gap, f"{cap}: cut at row {row} near an attack"


# 7 ---------------------------------------------------------------------------------------------


def test_shared_tail_is_one_recording_used_once(cfg, views, segs):
    seg = cfg["shared_segments"]["car_hacking/shared_tail"]
    ref = None
    for cap in seg["contained_in"]:
        v = next(x for x in views if x.capture == cap)
        t = L.read_processed(S.PROJECT_ROOT / v.parquet, columns=["timestamp", "can_id", "data", "label"])
        tail = t.slice(v.stop)
        assert v.stop < t.num_rows and tail["label"].to_numpy().sum() == 0
        cur = (tail["timestamp"].to_numpy(), tail["can_id"].to_numpy(), L.data_matrix(tail))
        if ref is None:
            ref = cur
        else:
            for x, y in zip(ref, cur):
                np.testing.assert_array_equal(x, y)
        # the view of the attack file stops before the tail; the gap before it is > 1 s
        ts = t["timestamp"].to_numpy()
        assert ts[v.stop] - ts[v.stop - 1] > 1.0
    tail_segs = [s for s in segs if s.capture == "car_hacking/shared_tail"]
    assert len(tail_segs) == 1 and tail_segs[0].n_frames == len(ref[0])


# 8 ---------------------------------------------------------------------------------------------


def test_excluded_captures_in_no_split(cfg, segs):
    idx = S.read_index()
    excluded = set(idx.loc[idx.excluded, "capture"])
    assert excluded == set(cfg["datasets"]["road"]["excluded_captures"])
    assert not excluded & {s.capture for s in segs}


def test_every_usable_capture_is_assigned(views, segs):
    assigned = {s.capture for s in segs}
    assert {v.capture for v in views if not v.excluded} == assigned


# 9 ---------------------------------------------------------------------------------------------


def test_autoencoder_subset(cfg, segs, windows):
    sub = windows[windows.normal_subset]
    assert len(sub) > 0
    assert set(sub.split.unique()) == {"train"}
    assert (sub.label == 0).all()
    masq = {s.id for s in segs if s.is_masquerade}
    assert not set(sub.segment.unique()) & masq
    size, margin = cfg["windows"]["size"], cfg["normal_subset"]["margin_s"]
    seg_by_id = {s.id: s for s in segs}
    for seg_id, g in sub.groupby("segment", observed=True):
        s = seg_by_id[seg_id]
        t = L.read_processed(S.PROJECT_ROOT / s.parquet, columns=["timestamp", "label"])
        ts, lab = t["timestamp"].to_numpy(), t["label"].to_numpy()
        starts = g["start"].to_numpy()
        assert W.margin_clean(ts, starts, size, ts[lab == 1], margin).all(), seg_id
