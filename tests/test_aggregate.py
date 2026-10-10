"""Tidy aggregation, completeness check and deltas of the P6 analysis (analysis/aggregate.py). Synthetic data."""

from fnmatch import fnmatch

import numpy as np
import pandas as pd
import pytest

from analysis import aggregate as A
from analysis import stats as X
from src import train as T

NO_VAL = {"road/fuzzing"}


def _metrics(run_id="cnn1d_road_s1_x", model="cnn1d", seed=1, quants=("q0", "q1", "q2"), recall=None):
    recall = recall or {"q0": 0.8, "q1": 0.8, "q2": 0.7}
    rows = []
    for q in quants:
        for mode in ("recalibrated", "q0_threshold"):
            for name, level, group, role in (("fuzzing", "attack_type", "coarse", "primary"),
                                             ("primary_pool", "pooled", np.nan, "secondary"),
                                             ("ambient", "attack_free", np.nan, "fpr_only")):
                r = {"run_id": run_id, "model": model, "dataset": "road", "seed": seed, "quantization": q,
                     "threshold_mode": mode, "split": "test", "level": level, "name": name, "group": group,
                     "role": role}
                for k in A.ATTACK_METRICS:
                    r[k] = np.nan if level == "attack_free" and k not in A.FREE_METRICS else 0.5
                if level != "attack_free":
                    r["recall"] = recall[q]
                rows.append(r)
    return pd.DataFrame(rows)


def test_tidy_metrics_keeps_level_metrics_and_groups():
    t = A.tidy_metrics(_metrics())
    assert set(t[t.level == "attack_free"]["metric"]) == set(A.FREE_METRICS)
    assert set(t[t.level == "attack_type"]["metric"]) == set(A.ATTACK_METRICS)
    assert set(t[t.level == "pooled"]["attack_group"]) == {"all"}
    assert set(t[t.level == "attack_free"]["attack_group"]) == {"none"}
    assert not t.duplicated(["run_id", "quantization", "threshold_mode", "split", "level", "attack_type",
                             "metric"]).any()


def test_delta_is_q0_minus_qk():
    d = A.delta_table(A.tidy_metrics(_metrics()))
    r = d[(d.metric == "recall") & (d.level == "attack_type") & (d.threshold_mode == "recalibrated")]
    assert r.set_index("quantization")["delta"].to_dict() == pytest.approx({"q1": 0.0, "q2": 0.1})
    assert set(d["quantization"]) == {"q1", "q2"}


def _cell(**kw):
    base = {"model": "cnn1d", "dataset": "road", "seed": 1, "quantization": "q0", "threshold_mode": "recalibrated",
            "split": "test", "level": "attack_type", "attack_type": "fuzzing", "metric": "recall", "status": "nan"}
    return {**base, **kw}


def test_structural_reasons():
    assert A.structural_reason(_cell(model="xgb", quantization="q2", status="absent"), NO_VAL)[1] == "K-042"
    assert A.structural_reason(_cell(level="model", metric="latency_median_ms_session1", quantization="q1",
                                     seed=0), NO_VAL)[1] == "K-043"
    assert A.structural_reason(_cell(level="model", metric="peak_rss_mb_session2", seed=2), NO_VAL)[1] == "K-052"
    assert A.structural_reason(_cell(split="val", status="absent"), NO_VAL)[1] == "K-018, K-019"
    assert "K-029" in A.structural_reason(_cell(metric="precision"), NO_VAL, flagged=0)[1]


def test_unexplained_gaps_are_not_excused():
    assert A.structural_reason(_cell(), NO_VAL) is None                          # NaN recall on test
    assert A.structural_reason(_cell(metric="precision"), NO_VAL, flagged=3) is None
    assert A.structural_reason(_cell(level="model", metric="size_mb", seed=2), NO_VAL) is None
    assert A.structural_reason(_cell(level="model", metric="latency_median_ms_session1", seed=0), NO_VAL) is None


def _runs():
    return [{"run_id": "cnn1d_road_s1_x", "model": "cnn1d", "dataset": "road", "seed": 1, "quantize": True,
             "evaluate_splits": ["test"]}]


CFG = {"pareto": {"sessions": [1, 2]}}
GROUPS = {"road": {"fuzzing": {"group": "coarse"}}}


def _with_resources(t):
    res = [{"run_id": "cnn1d_road_s1_x", "model": "cnn1d", "dataset": "road", "seed": 1, "quantization": q,
            "attack_type": "model", "attack_group": "none", "metric": m, "value": v, "split": A.NA,
            "threshold_mode": A.NA, "level": "model", "role": A.NA}
           for q in A.QUANT_LEVELS for m, v in [("size_mb", 0.1)] + [(f"{k}_session{s}", np.nan)
                                                                      for k in A.SESSION_METRICS for s in (1, 2)]]
    return pd.concat([t, pd.DataFrame(res)], ignore_index=True)


def test_missing_cells_lists_structural_gaps_only():
    t = _with_resources(A.tidy_metrics(_metrics()))
    miss = A.missing_cells(t, _runs(), GROUPS, CFG, NO_VAL)
    assert (miss["reason"] != "UNEXPLAINED").all()
    assert set(miss["decision"]) == {"K-043", "K-052"}           # Q1 (any seed), seed 1: no latency/memory


def test_missing_cells_flags_an_unexplained_nan():
    m = _metrics()
    m.loc[(m.quantization == "q2") & (m.level == "attack_type") & (m.threshold_mode == "recalibrated"), "f1"] = np.nan
    miss = A.missing_cells(_with_resources(A.tidy_metrics(m)), _runs(), GROUPS, CFG, NO_VAL)
    bad = miss[miss["reason"] == "UNEXPLAINED"]
    assert len(bad) == 1 and bad.iloc[0]["metric"] == "f1" and bad.iloc[0]["quantization"] == "q2"


def test_missing_cells_flags_an_absent_row():
    m = _metrics()
    m = m[~((m.quantization == "q1") & (m.level == "pooled") & (m.threshold_mode == "q0_threshold"))]
    miss = A.missing_cells(_with_resources(A.tidy_metrics(m)), _runs(), GROUPS, CFG, NO_VAL)
    bad = miss[miss["reason"] == "UNEXPLAINED"]
    assert len(bad) == len(A.ATTACK_METRICS) and set(bad["status"]) == {"absent"}


def test_cross_check_detects_a_differing_delta():
    d = A.delta_table(A.tidy_metrics(_metrics()))
    g = d[d.metric == "recall"].rename(columns={"quantization": "quant", "attack_type": "name",
                                                "delta": "d_recall"})
    g["split"] = "test"
    for k in ("f1", "roc_auc", "average_precision", "roc_tpr_at_fpr_0.01"):
        g[f"d_{k}"] = np.nan
    assert A.cross_check_grid(d[d.metric == "recall"], g) == len(g)
    g.loc[g.index[0], "d_recall"] += 0.01
    with pytest.raises(SystemExit):
        A.cross_check_grid(d[d.metric == "recall"], g)


def test_analysis_outputs_are_generated_outputs():
    def generated(path):
        rel = path.relative_to(T.S.PROJECT_ROOT).as_posix()
        return any(rel == p or rel.startswith(p + "/") or fnmatch(rel, p) for p in T.GENERATED_OUTPUTS)

    names = ["all_results.csv", "delta_f1.csv", "missing_cells.csv", "analysis_manifest.json", "stats.csv",
             "as1_pairs.csv", "per_type_delta.csv", "pareto.csv", "calibration_shift.csv", "decision_churn.csv"]
    assert all(generated(A.TABLES / n) for n in names)


def test_cache_directory_is_git_ignored_data():
    assert X.CACHE.relative_to(T.S.PROJECT_ROOT).as_posix().startswith("data/")
