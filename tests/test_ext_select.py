"""Selection rule of the P10 extension (analysis/ext_select.py; docs/DECISIONS.md K-075). Synthetic inputs only."""

import pandas as pd
import pytest

from analysis import ext_select as X

SEL = X.load_config()["selection"]
TARGETS = sorted(SEL["expected_road_targets"])
CH_TYPES = ["dos", "fuzzy", "gear_spoofing", "rpm_spoofing"]


def _means(road_auc: dict, ch_auc: dict | None = None, ch_rec: dict | None = None, model="cnn1d"):
    """road_auc: set -> target mean; ch_auc / ch_rec: set -> value on every CH type (default: equal to s0)."""
    rows = []
    for fs, v in road_auc.items():
        for t in TARGETS:
            rows.append({"model": model, "dataset": "road", "feature_set": fs, "name": t, "roc_auc": v,
                         "recall": 0.5, "n_seeds": 3})
        for t in CH_TYPES:
            rows.append({"model": model, "dataset": "car_hacking", "feature_set": fs, "name": t,
                         "roc_auc": (ch_auc or {}).get(fs, 0.99), "recall": (ch_rec or {}).get(fs, 0.95),
                         "n_seeds": 3})
    return pd.DataFrame(rows)


def test_smallest_set_within_tolerance_of_the_best():
    m = _means({"s0": 0.80, "s1": 0.80, "s2": 0.847, "s3": 0.85, "s4": 0.851})
    out = X.select_model(m, "cnn1d", TARGETS, SEL)
    assert out["g_star"] == pytest.approx(0.051)
    assert out["feature_set"] == "s2"            # 0.047 >= 0.051 - 0.005; s1 (0.0) is not


def test_reference_kept_without_meaningful_gain():
    m = _means({"s0": 0.80, "s1": 0.805, "s2": 0.808, "s3": 0.809, "s4": 0.801})
    out = X.select_model(m, "cnn1d", TARGETS, SEL)
    assert out["feature_set"] == "s0" and "min_gain" in out["reason"]


def test_car_hacking_guard_excludes_a_set():
    road = {"s0": 0.80, "s1": 0.80, "s2": 0.80, "s3": 0.90, "s4": 0.88}
    m = _means(road, ch_auc={"s0": 0.99, "s3": 0.98})              # s3 loses 0.01 AUC on Car-Hacking
    out = X.select_model(m, "cnn1d", TARGETS, SEL)
    assert not next(c for c in out["candidates"] if c["feature_set"] == "s3")["ch_guard_passed"]
    assert out["feature_set"] == "s4"
    m = _means(road, ch_rec={"s0": 0.95, "s3": 0.93, "s4": 0.93})  # recall guard
    assert X.select_model(m, "cnn1d", TARGETS, SEL)["feature_set"] == "s0"


def test_refuses_when_a_new_run_has_test_outputs(tmp_path):
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    X.check_no_test_results([tmp_path / "run_a", tmp_path / "run_b"])
    (tmp_path / "run_b" / "metrics_test.csv").write_text("x\n")
    with pytest.raises(SystemExit, match="test outputs"):
        X.check_no_test_results([tmp_path / "run_a", tmp_path / "run_b"])


def test_incomplete_seeds_refused():
    val = pd.DataFrame([{"level": "attack_type", "model": "cnn1d", "dataset": "road", "feature_set": "s1",
                         "name": TARGETS[0], "roc_auc": 0.9, "recall": 0.5, "seed": s} for s in (0, 1)])
    with pytest.raises(SystemExit, match="3 seeds"):
        X.seed_means(val)


def test_pre_registered_road_targets_match_the_group_file():
    assert X.road_targets(SEL, set(TARGETS) | {"correlated_signal", "max_speedometer"}) == TARGETS
