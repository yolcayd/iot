"""Tests for the extended per-frame features of the P10 ablation (docs/DECISIONS.md K-073)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import features as F
from src.data import features_ext as FX
from src.quantize import quantize_values

ECFG = FX.load_ext_config()
needs_ext = pytest.mark.skipif(not (FX.EXT_PARAMS_PATH.exists() and F.PARAMS_PATH.exists()),
                               reason="normalization parameters not generated")
IQ = {"dtype": "uint8", "scale": np.float32(1 / 255), "zero_point": 0}   # asymmetric INT8 input quantizer (K-044)


def _random_segment(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.choice([0x10, 0x20, 0x30, 0x55, 0x7E8], size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    ts = np.cumsum(rng.exponential(0.002, size=n))
    data = rng.integers(0, 256, size=(n, 8), dtype=np.uint8)
    return ts, ids, data


def test_sets_are_cumulative_prefixes_in_the_fixed_order():
    assert FX.set_columns("s0") == []
    assert FX.set_columns("s1") == ["bit_flips"]
    assert FX.set_columns("s2") == ["bit_flips"] + [f"dbyte_{i}" for i in range(8)]
    assert FX.set_columns("s3")[-1] == "id_rate"
    assert FX.set_columns("s4")[-2:] == ["id_share_w", "id_entropy_w"]
    assert [FX.n_features(s) for s in ("s0", "s1", "s2", "s3", "s4")] == [10, 11, 19, 20, 22]
    assert FX.feature_names("s4")[:10] == F.FEATURE_NAMES
    with pytest.raises(ValueError):
        FX.set_columns("s9")


def test_bit_flips_and_payload_delta_by_hand():
    # frames: id A, id B, id A (byte0 0x0F -> 0xF0: 8 bits, |240-15| = 225), id A (no change)
    ids = np.array([1, 2, 1, 1])
    data = np.zeros((4, 8), dtype=np.uint8)
    data[0, 0], data[2, 0], data[3, 0] = 0x0F, 0xF0, 0xF0
    data[1, 7] = 0xFF
    prev = FX.previous_same_id(ids)
    np.testing.assert_array_equal(prev, [-1, -1, 0, 2])
    np.testing.assert_allclose(FX.bit_flips(data, prev), [0, 0, 8 / 64, 0])
    dd = FX.payload_delta(data, prev)
    np.testing.assert_allclose(dd[:, 0], [0, 0, 225 / 255, 0])
    assert dd[1].sum() == 0          # first frame of an ID: no predecessor, 0


def test_short_dlc_padding_compares_padded_bytes():
    # dlc 2 frames of one ID: the zero-padded bytes 2..7 never flip
    ids = np.array([5, 5])
    data = np.array([[1, 2, 0, 0, 0, 0, 0, 0], [3, 2, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_allclose(FX.bit_flips(data, FX.previous_same_id(ids)), [0, 1 / 64])


def test_trailing_time_count_and_id_rate_by_hand():
    ts = np.array([0.0, 0.4, 0.9, 1.0, 1.3, 2.5])
    ids = np.array([1, 1, 1, 1, 2, 1])
    np.testing.assert_array_equal(FX.trailing_time_count(ts, ids, 1.0), [1, 2, 3, 3, 1, 1])  # (t - 1, t]
    v = FX.id_rate(ts, ids, np.array([1]), np.array([3.0]), 1.0, 3.0, 0.5, 1.0)
    # first second of the segment: neutral 0.5; id 2 has no reference: 1.0; t=2.5: r = 1/3
    assert v[0] == v[1] == v[2] == 0.5
    assert v[3] == pytest.approx(0.5)                       # r = 3/3
    assert v[4] == 1.0
    assert v[5] == pytest.approx((np.log2(1 / 3) + 3) / 6)
    big = FX.id_rate(np.array([0.0, 1.0]), np.array([1, 1]), np.array([1]), np.array([0.01]), 0.5, 3.0, 0.5, 1.0)
    assert big[1] == 1.0                                    # log2 clipped at +3


def test_id_window_matches_brute_force():
    ts, ids, _ = _random_segment()
    w = 100
    share, ent = FX.id_window(ids, w)
    for t in range(0, len(ids), 7):
        win = ids[max(0, t - w + 1):t + 1]
        _, c = np.unique(win, return_counts=True)
        p = c / len(win)
        assert ent[t] == pytest.approx(-(p * np.log2(p)).sum() / np.log2(w), abs=1e-9)
        assert share[t] == pytest.approx((win == ids[t]).sum() / w, abs=1e-12)


def test_columns_are_causal_and_in_unit_range():
    ts, ids, data = _random_segment()
    ref = {"ref_ids": [0x10, 0x20, 0x30, 0x55], "ref_counts": [200.0, 150.0, 70.0, 50.0]}
    x = FX.ext_columns(ts, ids, data, ref, ECFG)
    assert x.shape == (len(ts), len(FX.all_ext_columns(ECFG))) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    # changing the future does not change the past
    cut = 1500
    data2, ids2 = data.copy(), ids.copy()
    data2[cut:] = 0
    ids2[cut:] = 0x55
    y = FX.ext_columns(ts, ids2, data2, ref, ECFG)
    np.testing.assert_array_equal(x[:cut], y[:cut])


def test_columns_restart_at_a_segment_boundary():
    ts, ids, data = _random_segment()
    ref = {"ref_ids": [0x10], "ref_counts": [200.0]}
    whole = FX.ext_columns(ts, ids, data, ref, ECFG)
    tail = FX.ext_columns(ts[1000:], ids[1000:], data[1000:], ref, ECFG)
    names = FX.all_ext_columns(ECFG)
    assert tail[0, names.index("bit_flips")] == 0.0          # no predecessor across the boundary
    assert tail[0, names.index("id_share_w")] == pytest.approx(0.01)
    assert tail[0, names.index("id_rate")] == 0.5            # window of the new segment not yet full
    assert not np.array_equal(whole[1000:1100], tail[:100])


def test_reference_refuses_non_training_segments():
    class Seg:
        id, split = "x[0:1]", "val"
    with pytest.raises(ValueError, match="training"):
        FX.reference_counts([Seg()], ECFG)


def test_clean_frames_margin():
    ts = np.array([0.0, 4.0, 6.0, 20.0])
    np.testing.assert_array_equal(FX.clean_frames(ts, np.array([10.0]), 5.0), [True, True, False, True])


@pytest.mark.parametrize("name,n", [("bit_flips", 64), ("dbyte_0", 255), ("id_share_w", 100)])
def test_grid_columns_keep_every_value_on_its_own_int8_level(name, n):
    assert FX.column_kind(name, ECFG) == "grid"
    grid = np.arange(n + 1, dtype=np.float32) / np.float32(n)
    assert len(np.unique(quantize_values(grid, IQ))) == n + 1


def test_column_kinds():
    assert FX.column_kind("iat") == FX.column_kind("id_rate") == FX.column_kind("id_entropy_w") == "continuous"
    assert FX.column_kind("can_id") == FX.column_kind("byte_3") == "grid"


@needs_ext
def test_reference_set_keeps_the_base_hash_and_run_ids():
    from src import train as T
    from src.data import dataset as D

    base = F.load_params()["config_sha256_16"]
    assert D.feature_set_hash("s0") == base
    hashes = {D.feature_set_hash(s) for s in ("s1", "s2", "s3", "s4")}
    assert len(hashes) == 4 and base not in hashes
    hp = T.model_config("cnn1d", T.load_models_config())
    size = 100
    assert T.planned_run_id("cnn1d", "road", 0) == T.run_id("cnn1d", "road", 0, hp, base, size)
    assert T.planned_run_id("cnn1d", "road", 0, feature_set="s1") != T.planned_run_id("cnn1d", "road", 0)


@needs_ext
def test_ext_params_fitted_on_benign_training_segments_only():
    from src.data import splits as SP

    p = FX.load_ext_params()
    FX.check_ext_params_current(p, F.load_params(), ECFG)
    segs = {s.id: s for s in SP.segments()}
    for ds, d in p["datasets"].items():
        assert d["fitted_on_splits"] == ["train"]
        assert d["fitted_on_segments"] and all(segs[s].split == "train" and segs[s].dataset == ds
                                               and not segs[s].is_masquerade for s in d["fitted_on_segments"])
        assert d["ref_ids"] == sorted(d["ref_ids"]) and all(c > 0 for c in d["ref_counts"])
