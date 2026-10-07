"""Grid stages, the autoencoder feasibility protocol of K-033 and the full-grid runner (no training)."""

import json
import os
import sys

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


# --------------------------------------------------------------------------------------------
# Full stage (P5, K-053): plan, resumability, scheduler, lock
# --------------------------------------------------------------------------------------------


def test_full_stage_plan_reuses_seed0_run_ids():
    full = G.stage_runs("full")
    assert len(full) == 18
    assert {(r["model"], r["quantize"]) for r in full} == {("cnn1d", True), ("conv_ae", True), ("xgb", False)}
    base = {(r["model"], r["dataset"]): r["run_id"] for r in G.stage_runs("baseline")}
    ae = {(r["model"], r["dataset"]): r["run_id"] for r in G.stage_runs("ae_feasibility") if r["model"] == "conv_ae"}
    seed0 = {(r["model"], r["dataset"]): r["run_id"] for r in full if r["seed"] == 0}
    for k, rid in seed0.items():
        assert rid == (ae[k] if k[0] == "conv_ae" else base[k])
    assert len({r["run_id"] for r in full}) == 18


def _job(quantize=True, rid="cnn1d_x_s1_abc"):
    return {"stage": "full", "model": "cnn1d", "dataset": "road", "seed": 1, "run_id": rid,
            "evaluate_splits": ["val", "test"], "quantize": quantize, "overrides": {}}


def test_pending_steps_follow_the_outputs(tmp_path):
    r = _job()
    d = tmp_path / r["run_id"]
    assert G.pending_steps(r, tmp_path) == ["train", "evaluate", "quantize", "quant_eval"]
    assert G.job_state(r, tmp_path) == "todo"
    d.mkdir()
    (d / "checkpoint.pt").write_text("x")
    assert G.job_state(r, tmp_path) == "interrupted"
    (d / "DONE").write_text("x")
    (d / "metrics_val.csv").write_text("x")
    assert G.pending_steps(r, tmp_path) == ["evaluate", "quantize", "quant_eval"]
    (d / "metrics_test.csv").write_text("x")
    q = d / "quant"
    q.mkdir()
    (q / "quant_meta.json").write_text(json.dumps({"created_utc": "2026-10-07T10:00:00+00:00"}))
    assert G.pending_steps(r, tmp_path) == ["quant_eval"]
    assert G.job_state(r, tmp_path) == "partial"
    (q / "metrics.csv").write_text("x")
    assert G.pending_steps(r, tmp_path) == ["quant_eval"]          # evaluated_utc missing: half-finished
    (q / "quant_meta.json").write_text(json.dumps({"created_utc": "2026-10-07T10:00:00+00:00",
                                                   "evaluated_utc": "2026-10-07T10:01:00+00:00"}))
    assert G.pending_steps(r, tmp_path) == [] and G.job_state(r, tmp_path) == "done"
    assert G.pending_steps(_job(quantize=False), tmp_path) == []


def test_reference_durations_measured_before_seed0_record(tmp_path):
    r0 = {**_job(rid="a_s0"), "seed": 0}
    r1 = _job(rid="a_s1")
    d0 = tmp_path / "a_s0"
    (d0 / "quant").mkdir(parents=True)
    (d0 / "DONE").write_text("x")
    (d0 / "run_config.json").write_text(json.dumps({"duration_s": 100.0}))
    (d0 / "quant" / "quant_meta.json").write_text(json.dumps({"created_utc": "2026-10-07T10:00:00+00:00",
                                                              "evaluated_utc": "2026-10-07T10:00:30+00:00"}))
    sec, src = G.reference_durations([r0, r1], tmp_path)[("cnn1d", "road")]
    assert sec == 130.0 and src.startswith("seed0_record")
    (tmp_path / "a_s1").mkdir()
    (tmp_path / "a_s1" / G.JOB_FILE).write_text(json.dumps({"steps_run": ["train"], "wall_s": 200.0}))
    sec, src = G.reference_durations([r0, r1], tmp_path)[("cnn1d", "road")]
    assert sec == 200.0 and src.startswith("measured")


def test_makespan_lpt():
    assert G.makespan([], 2) == 0.0
    assert G.makespan([10, 10, 10], 1) == 30
    assert G.makespan([10, 6, 4], 2) == 10
    assert G.makespan([5, 5, 5, 5], 3) == 10


def _fake_jobs(n):
    return [{"stage": "full", "model": "m", "dataset": "d", "seed": i, "run_id": f"job{i}"} for i in range(n)]


def test_scheduler_failure_does_not_stop_the_grid(tmp_path):
    jobs = _fake_jobs(3)

    def cmd(r):
        code = 3 if r["run_id"] == "job1" else 0
        return [sys.executable, "-c", f"print('hello from {r['run_id']}'); raise SystemExit({code})"]

    glog = G.GridLog(tmp_path / "grid.log", echo=False)
    fails = G.run_jobs(jobs, 2, cmd, glog, {"job0": 1.0, "job1": 2.0, "job2": 3.0}, poll=0.05)
    assert [f["run_id"] for f in fails] == ["job1"] and fails[0]["exit_code"] == 3
    assert "hello from job1" in fails[0]["tail"]
    text = (tmp_path / "grid.log").read_text(encoding="utf-8")
    for j in ("job0", "job1", "job2"):
        assert f"[{j}] hello from {j}" in text
    assert "OK job0" in text and "OK job2" in text and "FAILED job1" in text
    assert "PROGRESS 3/3 finished (1 failed)" in text
    G.write_failures(fails, tmp_path / "f.csv")
    assert "job1" in (tmp_path / "f.csv").read_text(encoding="utf-8")


def test_scheduler_respects_worker_limit(tmp_path):
    marker = tmp_path / "running"
    marker.mkdir()
    code = ("import os, sys, time, pathlib; p = pathlib.Path(sys.argv[1]); f = p / str(os.getpid()); f.touch(); "
            "n = len(list(p.iterdir())); time.sleep(0.3); f.unlink(); print('max', n)")
    glog = G.GridLog(tmp_path / "grid.log", echo=False)
    fails = G.run_jobs(_fake_jobs(4), 2, lambda r: [sys.executable, "-c", code, str(marker)], glog, poll=0.02)
    assert not fails
    seen = [int(l.rsplit(" ", 1)[1]) for l in (tmp_path / "grid.log").read_text(encoding="utf-8").splitlines()
            if l.startswith("[job") and " max " in l]
    assert len(seen) == 4 and max(seen) <= 2


def test_lock_refuses_live_holder_and_replaces_stale(tmp_path):
    lock = tmp_path / "grid.lock"
    lock.write_text(json.dumps({"pid": 999999, "started": "x"}))
    with pytest.raises(SystemExit):
        G.acquire_lock(lock, alive=lambda pid: True)
    G.acquire_lock(lock, alive=lambda pid: False)
    assert json.loads(lock.read_text())["pid"] == os.getpid()
    G.release_lock(lock)
    assert not lock.exists()


def test_dry_run_writes_nothing(tmp_path, capsys, monkeypatch):
    import src.run_grid as RG

    monkeypatch.setattr(RG, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(RG, "LOCK", tmp_path / "logs" / "grid.lock")
    assert RG.main(["--stage", "full", "--dry-run", "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "18 runs" in out and "estimated duration" in out
    assert not (tmp_path / "logs").exists()
