"""Grid stages and the autoencoder feasibility protocol of K-033 (no training)."""

import pytest

from src import run_grid as G

MIN = 0.6
CH, RD = "car_hacking", "road"


def _auc(**kw):
    """kw: candidate=(ch_auc, road_auc)"""
    out = {}
    for c, (a, b) in kw.items():
        out[(c, CH)], out[(c, RD)] = a, b
    return out


def test_baseline_passing_both_stops_inspection():
    d = G.ae_decision(_auc(baseline=(0.8, 0.7)), MIN)
    assert d["inspected"] == ["baseline"] and d["selected"] == {CH: "baseline", RD: "baseline"}


def test_retry_pending_after_baseline_fails():
    d = G.ae_decision(_auc(baseline=(0.807, 0.394)), MIN)
    assert d["inspected"] == ["baseline", "retry"] and d["pending"] == ["retry"] and d["selected"] is None


def test_retry_passing_both_is_selected_and_conv_not_inspected():
    d = G.ae_decision(_auc(baseline=(0.807, 0.394), retry=(0.7, 0.65), conv_ae=(0.99, 0.99)), MIN)
    assert d["inspected"] == ["baseline", "retry"] and d["selected"] == {CH: "retry", RD: "retry"}


def test_conv_ae_inspected_only_after_retry_fails():
    d = G.ae_decision(_auc(baseline=(0.807, 0.394), retry=(0.8, 0.5), conv_ae=(0.7, 0.62)), MIN)
    assert d["inspected"] == ["baseline", "retry", "conv_ae"] and d["selected"] == {CH: "conv_ae", RD: "conv_ae"}


def test_exit_rule_road_excluded_ch_keeps_first_passing():
    d = G.ae_decision(_auc(baseline=(0.807, 0.394), retry=(0.5, 0.45), conv_ae=(0.65, 0.55)), MIN)
    assert d["selected"] == {CH: "baseline", RD: None}


def test_mixed_case_per_dataset_first_passing():
    d = G.ae_decision(_auc(baseline=(0.807, 0.394), retry=(0.5, 0.45), conv_ae=(0.55, 0.7)), MIN)
    assert d["selected"] == {CH: "baseline", RD: "conv_ae"}


def test_stage_runs():
    base = G.stage_runs("baseline")
    assert len(base) == 8 and {r["evaluate_splits"] == ["val", "test"] for r in base} == {True}
    ae = G.stage_runs("ae_feasibility")
    assert len(ae) == 4 and {r["evaluate_splits"] == ["val"] for r in ae} == {True}
    retry = [r for r in ae if r["model"] == "lstm_ae"]
    assert all(r["overrides"]["hidden"] == 128 for r in retry)
    base_ids = {r["run_id"] for r in base}
    assert not base_ids & {r["run_id"] for r in ae}
    assert len(base_ids) == 8
