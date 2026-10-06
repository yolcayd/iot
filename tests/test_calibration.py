"""Decision threshold calibrated on validation normals (K-035)."""

import numpy as np
import pytest

from src import calibration as C


def test_threshold_gives_at_most_target_fpr_on_val_normals():
    rng = np.random.default_rng(0)
    s = rng.normal(size=10_000)
    y = np.zeros(10_000, dtype=int)
    d = C.calibrate_threshold(s, y, "val", 0.01)
    assert d["realized_val_fpr"] <= 0.01 and d["realized_val_fpr"] == pytest.approx(0.01, abs=2e-4)
    assert d["n_val_normal"] == 10_000 and d["fitted_on_split"] == "val"
    assert (C.predict(s, d)).mean() == d["realized_val_fpr"]


def test_attack_labels_and_scores_are_not_used():
    rng = np.random.default_rng(1)
    s = rng.normal(size=2000)
    y = (np.arange(2000) % 5 == 0).astype(int)
    d1 = C.calibrate_threshold(s, y, "val", 0.01)
    s2 = s.copy()
    s2[y == 1] = rng.normal(size=(y == 1).sum()) * 100      # change only the attack scores
    assert C.calibrate_threshold(s2, y, "val", 0.01)["threshold"] == d1["threshold"]
    assert d1["n_val_normal"] == (y == 0).sum()


def test_ties_never_exceed_target():
    s = np.r_[np.zeros(990), np.ones(10) * 5.0, np.ones(5) * 5.0]   # 1.5 % of normals tie at the top
    d = C.calibrate_threshold(s, np.zeros(len(s), dtype=int), "val", 0.01)
    assert d["realized_val_fpr"] <= 0.01


@pytest.mark.parametrize("split", ["train", "test"])
def test_only_validation_scores_accepted(split):
    with pytest.raises(ValueError):
        C.calibrate_threshold(np.arange(10.0), np.zeros(10), split, 0.01)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        C.calibrate_threshold(np.arange(10.0), np.ones(10), "val", 0.01)    # no normals
    with pytest.raises(ValueError):
        C.calibrate_threshold(np.arange(10.0), np.zeros(10), "val", 0.0)
    with pytest.raises(ValueError):
        C.predict(np.arange(3.0), {"rule": "probability >= threshold", "threshold": 0.5})
