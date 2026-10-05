"""Sanity-check helpers on hand-made tables (no training, no model output)."""

import numpy as np
import pandas as pd

from src import sanity as SA


def _win():
    rows = [
        ("road", "train", "fab_1", "s1", 0, 0, "normal"), ("road", "train", "fab_1", "s1", 100, 1, "max_speedometer"),
        ("road", "test", "fab_3", "s3", 0, 0, "normal"), ("road", "test", "fab_3", "s3", 100, 1, "max_speedometer"),
        ("road", "test", "amb", "s4", 0, 0, "normal"),
    ]
    return pd.DataFrame(rows, columns=["dataset", "split", "capture", "segment", "start", "label", "attack_type"])


def test_high_f1_selects_attack_type_rows_only():
    t = pd.DataFrame({"dataset": ["road"] * 3, "model": ["cnn1d"] * 3, "run_id": ["r"] * 3,
                      "level": ["attack_type", "attack_type", "pooled"], "name": ["a", "b", "primary_pool"],
                      "group": ["coarse", "stealthy", ""], "role": ["primary", "primary", "secondary"],
                      "f1": [0.999, 0.90, 0.999]})
    assert SA.high_f1(t)["name"].tolist() == ["a"]


def test_origins_and_type_subset_follow_type_captures():
    win = _win()
    o = SA.origins(win, "road", "max_speedometer")
    assert set(o["capture"]) == {"fab_1", "fab_3"} and set(o["split"]) == {"train", "test"}
    te = SA.type_subset(win, "road", "max_speedometer", "test")
    assert set(te["capture"]) == {"fab_3"} and len(te) == 2      # ambient capture not included


def test_diag_id_codes_are_vocabulary_ranks():
    vocab = [0x100, 0x7E8, 0x7E9]
    assert np.allclose(SA.id_codes([0x7E8, 0x7E9], vocab), [2 / 3, 1.0])
