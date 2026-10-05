"""Tests for windowing and window labelling."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import windows as W


def test_window_starts_drop_incomplete_tail():
    np.testing.assert_array_equal(W.window_starts(250, 100, 50), [0, 50, 100, 150])
    np.testing.assert_array_equal(W.window_starts(250, 100, 100), [0, 100])
    assert len(W.window_starts(99, 100, 100)) == 0
    np.testing.assert_array_equal(W.window_starts(100, 100, 100), [0])
    with pytest.raises(ValueError):
        W.window_starts(100, 100, 0)


def test_window_label_is_one_if_any_frame_injected():
    label = np.zeros(250, dtype=np.int8)
    label[120] = 1
    starts = W.window_starts(250, 100, 50)
    np.testing.assert_array_equal(W.window_labels(label, starts, 100), [0, 1, 1, 0])
    label[149] = 1
    np.testing.assert_array_equal(W.injected_counts(label, starts, 100), [0, 2, 2, 0])


def test_dominant_attack_type():
    cats = ["normal", "a", "b"]
    codes = np.zeros(200, dtype=np.int64)
    label = np.zeros(200, dtype=np.int8)
    # window 0: 2 x b then 3 x a -> a; window 1: b first, tie 1:1 -> b
    for i, c in zip([1, 2, 3, 4, 5], [2, 2, 1, 1, 1]):
        codes[i], label[i] = c, 1
    for i, c in zip([110, 150], [2, 1]):
        codes[i], label[i] = c, 1
    starts = W.window_starts(200, 100, 100)
    assert W.dominant_attack_type(codes, label, starts, 100, cats).tolist() == ["a", "b"]
    assert W.dominant_attack_type(codes, np.zeros(200, np.int8), starts, 100, cats).tolist() == ["normal", "normal"]


def test_margin_clean_excludes_windows_near_injection():
    ts = np.arange(300) * 0.1          # 30 s, one frame every 100 ms
    starts = W.window_starts(300, 20, 20)  # 2 s windows
    clean = W.margin_clean(ts, starts, 20, np.array([15.0]), margin_s=5.0)
    t0, t1 = ts[starts], ts[starts + 19]
    expected = (t1 < 10.0) | (t0 > 20.0)
    np.testing.assert_array_equal(clean, expected)
    assert W.margin_clean(ts, starts, 20, np.array([]), 5.0).all()
