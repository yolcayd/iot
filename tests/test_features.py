"""Tests for the per-message feature vector and its INT8 resolution."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import features as F
from src.data import loaders as L

PARAMS_AVAILABLE = F.PARAMS_PATH.exists()
needs_params = pytest.mark.skipif(not PARAMS_AVAILABLE, reason="normalization_params.json not generated")
VOCAB = np.array([0x2B0, 0x316, 0x5F0])


def test_known_message_vector_matches_hand_computation():
    # frames: (t, id) = (0.000, 0x316), (0.010, 0x2B0), (0.020, 0x316), (0.900, 0x316)
    # floor 1 ms, cap 0.5 s
    ts = np.array([0.000, 0.010, 0.020, 0.900])
    ids = np.array([0x316, 0x2B0, 0x316, 0x316])
    data = np.zeros((4, 8), dtype=np.uint8)
    data[2] = [0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6F]
    x = F.message_features(ts, ids, data, VOCAB, iat_floor=0.001, iat_cap=0.5)

    iat = (np.log10(0.020) - np.log10(0.001)) / (np.log10(0.5) - np.log10(0.001))
    expected = np.array([2 / 3, 5 / 255, 33 / 255, 104 / 255, 9 / 255, 33 / 255, 33 / 255, 0.0, 111 / 255, iat],
                        dtype=np.float32)
    np.testing.assert_allclose(x[2], expected, rtol=0, atol=1e-6)
    assert x.shape == (4, F.N_FEATURES) and x.dtype == np.float32
    # first frame of each ID gets 1.0; IAT 0.88 s is clipped to the cap
    np.testing.assert_allclose(x[:, 9], [1.0, 1.0, iat, 1.0], atol=1e-6)
    np.testing.assert_allclose(x[:, 0], [2 / 3, 1 / 3, 2 / 3, 2 / 3], atol=1e-7)


def test_short_dlc_payload_is_zero_padded_in_features():
    sample = "1478198376.389427,05f0,2,01,00,R\n1478198376.389864,02b0,5,ff,7f,00,05,49,R\n"
    df = pd.read_csv(io.StringIO(sample), header=None, names=range(12), dtype=str)
    cols = L.parse_car_hacking_attack_chunk(df)
    x = F.message_features(cols["timestamp"], cols["can_id"], cols["data"], VOCAB, 0.001, 1.0)
    np.testing.assert_allclose(x[0, 1:9], [1 / 255, 0, 0, 0, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(x[1, 1:9], [1.0, 127 / 255, 0, 5 / 255, 73 / 255, 0, 0, 0], atol=1e-7)


def test_id_encoding_vocabulary_and_reserved_code():
    vocab = np.array([0x000, 0x0D0, 0x7FF, 0xFFF])
    codes = F.encode_ids(np.array([0x0D0, 0xFFF, 0x123, 0x000, 0x7FF]), vocab)
    np.testing.assert_allclose(codes, [2 / 4, 4 / 4, 0.0, 1 / 4, 3 / 4])
    with pytest.raises(ValueError):
        F.encode_ids(np.array([1]), np.array([5, 3]))  # unsorted vocabulary


def test_iat_log_scale_floor_and_cap():
    v = F.scale_iat(np.array([1e-6, 1e-3, 1e-2, 1e-1, 1.0, 5.0, np.nan]), floor=1e-3, cap=1.0)
    np.testing.assert_allclose(v, [0.0, 0.0, 1 / 3, 2 / 3, 1.0, 1.0, 1.0], atol=1e-12)
    with pytest.raises(ValueError):
        F.scale_iat(np.array([0.1]), floor=1.0, cap=0.5)


def test_iat_is_per_id_and_resets_at_segment_start():
    ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ids = np.array([1, 2, 1, 2, 1, 2])
    iat = F.per_id_iat(ts, ids)
    np.testing.assert_allclose(iat, [np.nan, np.nan, 0.2, 0.2, 0.2, 0.2], equal_nan=True)
    # features of a later segment never look at frames before it
    x = F.message_features(ts[3:], ids[3:], np.zeros((3, 8), np.uint8), np.array([1, 2]), 0.01, 1.0)
    expected = (np.log10(0.2) - np.log10(0.01)) / (np.log10(1.0) - np.log10(0.01))
    np.testing.assert_allclose(x[:, 9], [1.0, 1.0, expected], atol=1e-6)


def test_features_are_in_unit_range():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.uniform(0, 10, 500))
    ids = rng.integers(0, 0x800, 500)
    data = rng.integers(0, 256, (500, 8)).astype(np.uint8)
    x = F.message_features(ts, ids, data, np.arange(0, 0x800, 7), 1e-4, 0.3)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_normalization_refuses_non_training_segments():
    seg = SimpleNamespace(split="test", id="road/x[0:10]")
    with pytest.raises(ValueError, match="training"):
        F.fit_normalization([seg], 0.001, 0.999)


def test_benign_iats_ignore_injected_frames():
    ts = np.array([0.0, 0.1, 0.15, 0.2])
    ids = np.array([5, 5, 5, 5])
    label = np.array([0, 0, 1, 0])
    np.testing.assert_allclose(F.benign_iats(ts, ids, label), [0.1, 0.05])


def test_int8_levels():
    np.testing.assert_array_equal(F.int8_levels(np.array([0.0, 1 / 255, 0.5, 1.0])), [0, 1, 128, 255])
    np.testing.assert_array_equal(F.int8_levels(np.array([0.0, 1.0]), "symmetric"), [0, 127])


# ------------------------------------------------------------------------------------------
# INT8 resolution of the fitted representation (K-024, K-025)
# ------------------------------------------------------------------------------------------


@needs_params
@pytest.mark.parametrize("scheme", list(F.INT8_SCHEMES))
@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_distinct_ids_never_share_an_int8_level(dataset, scheme):
    p = F.load_params()["datasets"][dataset]
    vocab = np.asarray(p["id_vocab"])
    assert len(vocab) + 1 <= F.INT8_SCHEMES[scheme] + 1, "vocabulary does not fit the INT8 range"
    # every vocabulary ID plus one ID outside the vocabulary (reserved code)
    outside = int(np.setdiff1d(np.arange(0x1000), vocab)[0])
    levels = F.int8_levels(F.encode_ids(np.r_[vocab, outside], vocab), scheme)
    assert len(np.unique(levels)) == len(vocab) + 1


@needs_params
@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_column_levels_under_asymmetric_int8(dataset):
    """Asymmetric per-tensor INT8 keeps every ID and every byte value apart and resolves IAT finely."""
    params = F.load_params()
    rows = pd.DataFrame(F.level_report(params))
    r = rows[(rows.dataset == dataset) & (rows.scheme == "asymmetric")].set_index("feature")
    p = params["datasets"][dataset]
    assert r.loc["can_id", "n_levels"] == p["n_ids"] + 1
    assert r.loc["byte (each of 8)", "n_levels"] == 256
    # 1 ms and 10 ms are separated by many levels (linear scaling with a 1 s cap gave ~2)
    assert r.loc["iat", "iat_level_10ms"] - r.loc["iat", "iat_level_1ms"] >= 20
    on_disk = pd.read_csv(F.LEVELS_PATH)
    pd.testing.assert_frame_equal(on_disk, rows, check_dtype=False, rtol=0, atol=1e-2)  # csv keeps 4 digits


@needs_params
@pytest.mark.skipif(not (L.PROCESSED_DIR / "index.csv").exists(), reason="processed data not available")
@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_real_data_ids_do_not_merge_under_int8(dataset):
    """On training data: as many INT8 ID levels as distinct IDs codes (all unseen IDs share code 0)."""
    from src.data import splits as SP

    params = F.load_params()
    vocab = np.asarray(params["datasets"][dataset]["id_vocab"])
    for seg in [s for s in SP.segments() if s.dataset == dataset and s.split == "train"][:4]:
        t = seg.read(["can_id"])
        cid = t["can_id"].to_numpy()
        codes = F.encode_ids(cid, vocab)
        for scheme in F.INT8_SCHEMES:
            lv = F.int8_levels(codes, scheme)
            pairs = np.unique(np.c_[codes, lv], axis=0)
            assert len(np.unique(pairs[:, 0])) == len(np.unique(pairs[:, 1])) == len(pairs), (seg.id, scheme)
