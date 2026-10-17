"""P10 aggregation and diagnosis helpers (analysis/ext_aggregate.py, analysis/ext_diagnosis.py). Synthetic inputs only."""

import numpy as np
import pandas as pd
import pytest

from analysis import ext_aggregate as A
from analysis import ext_diagnosis as D


def _long():
    rows = []
    for fs, base in (("s0", 0.5), ("s3", 0.7)):
        for seed, eps in ((0, 0.0), (1, 0.1), (2, -0.1)):
            rows.append({"model": "cnn1d", "dataset": "road", "feature_set": fs, "n_features": 10 if fs == "s0" else 20,
                         "seed": seed, "split": "test", "level": "attack_type", "name": "x", "group": "stealthy",
                         "role": "primary", "metric": "recall", "value": base + eps + (0.05 * seed if fs == "s3" else 0)})
    return pd.DataFrame(rows)


def test_summary_mean_std_and_paired_delta_to_s0():
    s = A.summarize(_long()).set_index("feature_set")
    assert s.loc["s0", "mean"] == pytest.approx(0.5)
    assert s.loc["s0", "std"] == pytest.approx(0.1)
    assert s.loc["s0", "n_seeds"] == 3 and s.loc["s0", "delta_mean"] == 0
    # same-seed differences: 0.2, 0.25, 0.3
    assert s.loc["s3", "delta_mean"] == pytest.approx(0.25)
    assert (s.loc["s3", "delta_min"], s.loc["s3", "delta_max"]) == (pytest.approx(0.2), pytest.approx(0.3))


def test_type_sets_attack_windows_plus_same_capture_normals():
    win = pd.DataFrame({"segment": ["road/a_attack_1[0:9]"] * 3 + ["road/amb[0:9]"] * 2,
                        "start": [0, 100, 200, 0, 100], "label": [0, 1, 0, 0, 0],
                        "attack_type": ["normal", "a", "normal", "normal", "normal"]})
    s = A.type_sets(win, "road")
    assert list(s) == ["a"]
    assert sorted(s["a"]["index"].tolist()) == [0, 1, 2]           # ambient capture not in the type's set


@pytest.mark.parametrize("name,group", [("byte_3", "payload_bytes"), ("dbyte_0", "payload_delta"), ("can_id", "can_id"),
                                        ("id_entropy_w", "id_window"), ("id_rate", "id_rate"), ("iat", "iat")])
def test_channel_groups(name, group):
    assert D.channel_group(name) == group


def test_calibration_mask_requires_subset():
    val = pd.DataFrame({"segment": ["a", "a", "b"], "start": [0, 100, 0]})
    m = D._calib_mask(val, pd.DataFrame({"segment": ["a", "b"], "start": [100, 0]}))
    np.testing.assert_array_equal(m, [False, True, True])
    with pytest.raises(SystemExit):
        D._calib_mask(val, pd.DataFrame({"segment": ["c"], "start": [0]}))
