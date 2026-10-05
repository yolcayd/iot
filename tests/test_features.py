"""Tests for the per-message feature vector."""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import features as F
from src.data import loaders as L


def test_known_message_vector_matches_hand_computation():
    # frames: (t, id) = (0.000, 0x316), (0.010, 0x2B0), (0.020, 0x316), (0.900, 0x316); cap 0.5 s
    ts = np.array([0.000, 0.010, 0.020, 0.900])
    ids = np.array([0x316, 0x2B0, 0x316, 0x316])
    data = np.zeros((4, 8), dtype=np.uint8)
    data[2] = [0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6F]
    x = F.message_features(ts, ids, data, id_space=0x7FF, iat_cap=0.5)

    expected = np.array([0x316 / 0x7FF, 5 / 255, 33 / 255, 104 / 255, 9 / 255, 33 / 255, 33 / 255, 0.0, 111 / 255,
                         0.020 / 0.5], dtype=np.float32)
    np.testing.assert_allclose(x[2], expected, rtol=0, atol=1e-6)
    assert x.shape == (4, F.N_FEATURES) and x.dtype == np.float32
    # first frame of each ID gets the cap; IAT 0.88 s is clipped to the cap
    np.testing.assert_allclose(x[:, 9], [1.0, 1.0, 0.04, 1.0], atol=1e-6)


def test_short_dlc_payload_is_zero_padded_in_features():
    sample = "1478198376.389427,05f0,2,01,00,R\n1478198376.389864,02b0,5,ff,7f,00,05,49,R\n"
    df = pd.read_csv(io.StringIO(sample), header=None, names=range(12), dtype=str)
    cols = L.parse_car_hacking_attack_chunk(df)
    x = F.message_features(cols["timestamp"], cols["can_id"], cols["data"], 0x7FF, 1.0)
    np.testing.assert_allclose(x[0, 1:9], [1 / 255, 0, 0, 0, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(x[1, 1:9], [1.0, 127 / 255, 0, 5 / 255, 73 / 255, 0, 0, 0], atol=1e-7)


def test_id_space_per_dataset():
    ids = np.array([0x000, 0x7FF, 0xFFF])
    x = F.message_features(np.arange(3.0), ids, np.zeros((3, 8), np.uint8), id_space=0xFFF, iat_cap=1.0)
    np.testing.assert_allclose(x[:, 0], [0.0, 0x7FF / 0xFFF, 1.0], atol=1e-7)
    with pytest.raises(ValueError):  # 0xFFF does not fit the 11-bit space
        F.message_features(np.arange(3.0), ids, np.zeros((3, 8), np.uint8), id_space=0x7FF, iat_cap=1.0)


def test_iat_is_per_id_and_resets_at_segment_start():
    ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ids = np.array([1, 2, 1, 2, 1, 2])
    iat = F.per_id_iat(ts, ids)
    np.testing.assert_allclose(iat, [np.nan, np.nan, 0.2, 0.2, 0.2, 0.2], equal_nan=True)
    # features of a later segment never look at frames before it
    x = F.message_features(ts[3:], ids[3:], np.zeros((3, 8), np.uint8), 0x7FF, 1.0)
    np.testing.assert_allclose(x[:, 9], [1.0, 1.0, 0.2], atol=1e-7)


def test_features_are_in_unit_range():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.uniform(0, 10, 500))
    ids = rng.integers(0, 0x800, 500)
    data = rng.integers(0, 256, (500, 8)).astype(np.uint8)
    x = F.message_features(ts, ids, data, 0x7FF, 0.3)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_iat_cap_refuses_non_training_segments():
    seg = SimpleNamespace(split="test", id="road/x[0:10]")
    with pytest.raises(ValueError, match="training"):
        F.fit_iat_cap([seg], 0.999)


def test_benign_iats_ignore_injected_frames():
    ts = np.array([0.0, 0.1, 0.15, 0.2])
    ids = np.array([5, 5, 5, 5])
    label = np.array([0, 0, 1, 0])
    np.testing.assert_allclose(F.benign_iats(ts, ids, label), [0.1, 0.05])
