"""Windowing of message sequences; overlapping stride for training, non-overlapping stride for testing.

Windows are N consecutive frames of one split segment and never cross a segment boundary; a
trailing remainder shorter than N is dropped. They are represented by their start row, not copied.

    label        1 if the window contains at least one injected frame, else 0
    attack_type  most frequent attack type among the injected frames of the window ("normal" if
                 none; ties go to the type that occurs first in the window)
    normal_ok    no frame of the window lies within ``margin_s`` seconds of any injected frame of
                 the capture (used for the attack-free LSTM autoencoder training subset)
"""

from __future__ import annotations

import numpy as np

NORMAL = "normal"


def window_starts(n_frames: int, size: int, stride: int) -> np.ndarray:
    """Start offsets of all complete windows in a sequence of ``n_frames`` frames."""
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be positive")
    if n_frames < size:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_frames - size + 1, stride, dtype=np.int64)


def injected_counts(label: np.ndarray, starts: np.ndarray, size: int) -> np.ndarray:
    cs = np.r_[0, np.cumsum(label.astype(np.int64))]
    return cs[starts + size] - cs[starts]


def window_labels(label: np.ndarray, starts: np.ndarray, size: int) -> np.ndarray:
    return (injected_counts(label, starts, size) > 0).astype(np.int8)


def dominant_attack_type(type_codes: np.ndarray, label: np.ndarray, starts: np.ndarray, size: int,
                         categories: list[str]) -> np.ndarray:
    """Most frequent attack type among the injected frames of each window (object array of names)."""
    out = np.full(len(starts), NORMAL, dtype=object)
    att = np.where(injected_counts(label, starts, size) > 0)[0]
    if len(att) == 0:
        return out
    rows = starts[att, None] + np.arange(size)
    codes = type_codes[rows]
    inj = label[rows] == 1
    counts = np.zeros((len(att), len(categories)), dtype=np.int64)
    r, c = np.nonzero(inj)
    np.add.at(counts, (r, codes[r, c]), 1)
    # tie-break: first injected type in the window
    first = codes[np.arange(len(att)), np.argmax(inj, axis=1)]
    best = counts.max(axis=1, keepdims=True)
    tied = (counts == best).sum(axis=1) > 1
    choice = np.argmax(counts, axis=1)
    choice[tied] = first[tied]
    out[att] = np.asarray(categories, dtype=object)[choice]
    return out


def margin_clean(ts: np.ndarray, starts: np.ndarray, size: int, injected_ts: np.ndarray, margin_s: float) -> np.ndarray:
    """True where no frame of the window lies within ``margin_s`` of any injected timestamp."""
    if len(injected_ts) == 0:
        return np.ones(len(starts), dtype=bool)
    inj = np.sort(injected_ts)
    lo = np.searchsorted(inj, ts[starts] - margin_s, side="left")
    hi = np.searchsorted(inj, ts[starts + size - 1] + margin_s, side="right")
    return hi == lo
