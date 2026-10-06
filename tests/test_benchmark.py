"""Benchmark statistics, repeatability rule, environment record and a 20-run smoke (not a measurement)."""

import json

import joblib
import numpy as np
import pytest
import torch

from src import benchmark as B
from src import quantize as Q
from src.models.classical import summary_features
from src.models.cnn1d import CNN1D


def test_latency_stats_drop_warmup_and_require_1000():
    s = np.r_[np.full(100, 1e9), np.full(1000, 2e6)]          # slow warm-up must not count
    st = B.latency_stats(s, 100, 1000)
    assert st["n_warmup"] == 100 and st["n_measured"] == 1000
    assert st["mean_ms"] == pytest.approx(2.0) and st["p95_ms"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        B.latency_stats(s[:1099], 100, 1000)
    with pytest.raises(ValueError):
        B.latency_stats(s, 1100, 1)


def test_repeatability_tolerances():
    tol = {"median_rel_tol": 0.05, "p95_rel_tol": 0.10}

    def r(med, p95):
        return {"latency": {"median_ms": med, "p95_ms": p95}}
    assert B.repeatability(r(1.0, 2.0), r(1.04, 2.15), tol)["consistent"]
    assert not B.repeatability(r(1.0, 2.0), r(1.06, 2.0), tol)["consistent"]
    assert not B.repeatability(r(1.0, 2.0), r(1.0, 2.3), tol)["consistent"]


def test_environment_record_fields():
    env = B.environment_record("host", 4, B.load_qconfig())
    for k in ("cpu", "physical_cores", "logical_cores", "memory_total_mb", "cgroup", "threads", "power", "git",
              "versions", "profile_note", "cpu_features"):
        assert k in env
    assert env["threads"] == 4 and env["versions"]["onnxruntime"]


def test_smoke_run_outside_results(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.random((40, 100, 10)).astype(np.float32)
    np.save(tmp_path / "in.npy", x)
    Q.export_onnx(Q.ScoreModel(CNN1D().eval(), False), tmp_path / "m.onnx", 100, 10, 17)
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(summary_features(x), np.arange(40) % 2)
    joblib.dump(rf, tmp_path / "rf.joblib")
    targets = [{"id": "m:q0", "run_id": "m", "model": "cnn1d", "dataset": "d", "quant": "q0", "kind": "onnx",
                "model_path": str(tmp_path / "m.onnx"), "input_path": str(tmp_path / "in.npy"),
                "out_dir": str(tmp_path), "status": "measure"},
               {"id": "m:q1", "run_id": "m", "model": "cnn1d", "dataset": "d", "quant": "q1", "kind": "onnx",
                "model_path": str(tmp_path / "m.onnx"), "input_path": str(tmp_path / "in.npy"),
                "out_dir": str(tmp_path), "status": "cpu_unsupported"},
               {"id": "r:native", "run_id": "r", "model": "rf", "dataset": "d", "quant": "native", "kind": "rf",
                "model_path": str(tmp_path / "rf.joblib"), "input_path": str(tmp_path / "in.npy"),
                "out_dir": str(tmp_path), "status": "measure"}]
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"targets": targets}))
    out = tmp_path / "smoke"
    done = dict(B.run_all("host", 1, 2, B.load_qconfig(), smoke_out=out, manifest=man))
    assert done == {"m:q0": "measured", "m:q1": "cpu_unsupported", "r:native": "measured"}
    res = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
    assert len(res) == 2 and all(r["smoke"] for r in res)
    for r in res:
        assert r["latency"]["n_measured"] == B.SMOKE["repeats"] and r["batch_size"] == 1
        assert r["memory"]["peak_rss_mb"] >= r["memory"]["rss_before_load_mb"] > 0
        assert r["environment"]["thread_env"]["OMP_NUM_THREADS"] == "2"
    onnx_res = next(r for r in res if r["target"]["kind"] == "onnx")
    assert onnx_res["onnxruntime"]["providers_used"] == ["CPUExecutionProvider"]
    rf_res = next(r for r in res if r["target"]["kind"] == "rf")
    assert rf_res["runtime"]["n_jobs"] == B.RF_N_JOBS == 1                         # K-048
    for r in res:
        assert r["warnings_guard"]["warning_filters_intact"] and r["warnings_guard"]["n_warnings_shown_timed"] == 0


@pytest.fixture
def rf_target(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    x = np.random.default_rng(0).random((10, 100, 10)).astype(np.float32)
    np.save(tmp_path / "in.npy", x)
    joblib.dump(RandomForestClassifier(n_estimators=8, random_state=0).fit(summary_features(x), np.arange(10) % 2),
                tmp_path / "rf.joblib")
    return {"id": "r:native", "kind": "rf", "model_path": str(tmp_path / "rf.joblib"),
            "input_path": str(tmp_path / "in.npy")}


def _patched_features(monkeypatch, action, from_call):
    """summary_features that runs ``action`` from the ``from_call``-th call on (0-based)."""
    import src.models.classical as C

    calls = [0]
    orig = C.summary_features

    def f(x):
        if calls[0] >= from_call:
            action()
        calls[0] += 1
        return orig(x)
    monkeypatch.setattr(C, "summary_features", f)


def test_guard_rejects_a_warning_inside_the_timed_loop(monkeypatch, rf_target):
    import warnings

    _patched_features(monkeypatch, lambda: warnings.warn("noise", RuntimeWarning), from_call=3)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.raises(B.MeasurementInvalid):
            B.measure(rf_target, 1, warmup=2, repeats=4, min_repeats=4)


def test_guard_accepts_a_warning_during_warmup(monkeypatch, rf_target):
    import warnings

    calls = [0]

    def once():
        if calls[0] == 0:
            warnings.warn("first call only", RuntimeWarning)
        calls[0] += 1
    _patched_features(monkeypatch, once, from_call=0)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        r = B.measure(rf_target, 1, warmup=2, repeats=4, min_repeats=4)
    assert r["warnings_guard"]["n_warnings_shown"] == 1 and r["warnings_guard"]["n_warnings_shown_timed"] == 0


def test_guard_rejects_changed_warning_filters(monkeypatch, rf_target):
    import warnings

    # what the scikit-learn race did in host session 1: the process-wide filter list ends up replaced / emptied
    _patched_features(monkeypatch, warnings.resetwarnings, from_call=3)
    with warnings.catch_warnings():
        with pytest.raises(B.MeasurementInvalid):
            B.measure(rf_target, 1, warmup=2, repeats=4, min_repeats=4)


def test_only_the_sklearn_delayed_message_is_ignored(rf_target):
    import re
    import warnings

    with warnings.catch_warnings():
        B.measure(rf_target, 1, warmup=2, repeats=4, min_repeats=4)
        added = [f for f in warnings.filters if f[0] == "ignore" and f[1] is not None
                 and f[1].pattern == B.SKLEARN_DELAYED_WARNING]
        assert len(added) == 1 and added[0][2] is UserWarning
        msg = ("`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel` to make it "
               "possible to propagate the scikit-learn configuration of the current thread to the joblib workers.")
        assert re.match(B.SKLEARN_DELAYED_WARNING, msg)
        assert not any(f[0] == "ignore" and f[1] is None and f[2] in (UserWarning, Warning) for f in warnings.filters)


def test_invalid_results_are_kept_ignored_and_archived(tmp_path):
    p = B.result_path(tmp_path, "host", 1, "native")
    p.write_text(json.dumps({"latency": {"median_ms": 1.0}}))
    assert B.load_result(tmp_path, "host", 1, "native") is not None
    B.invalidate(p, "warning output inside the timed loop", "K-048")
    assert B.load_result(tmp_path, "host", 1, "native") is None
    assert json.loads(p.read_text())["invalid"]["decision"] == "K-048"
    kept = B.archive_invalid(p)
    assert kept.exists() and not p.exists() and ".invalid-" in kept.name
    assert json.loads(kept.read_text())["latency"]["median_ms"] == 1.0            # data preserved
    assert B.archive_invalid(p) is None


def test_smoke_must_not_write_under_results():
    with pytest.raises(SystemExit):
        B.main(["--smoke", "--out", str(B.PROJECT_ROOT / "results" / "smoke")])


def test_benchmark_module_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, src.benchmark, src.models.classical; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=B.PROJECT_ROOT).returncode == 0
    assert torch is not None
