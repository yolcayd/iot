"""Training plumbing: run id, configuration, window selectors and seeded reproducibility (synthetic data)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src import train as T
from src.data import dataset as D
from src.data import splits as SP


def test_run_id_is_deterministic_and_ignores_runtime():
    mcfg = T.load_models_config()
    hp = T.model_config("cnn1d", mcfg)
    a = T.run_id("cnn1d", "road", 0, hp, "abc", 100)
    assert a == T.run_id("cnn1d", "road", 0, dict(hp), "abc", 100)
    assert a.startswith("cnn1d_road_s0_")
    assert a != T.run_id("cnn1d", "road", 1, hp, "abc", 100)
    assert a != T.run_id("cnn1d", "road", 0, {**hp, "lr": 1.0}, "abc", 100)
    assert a != T.run_id("cnn1d", "road", 0, hp, "abd", 100)
    # runtime resources live in models.yaml `runtime`, outside the hyperparameters
    assert "threads" not in hp and "device" not in hp


def test_model_config_merges_common_block_for_deep_models_only():
    mcfg = T.load_models_config()
    ae = T.model_config("lstm_ae", mcfg)
    assert ae["batch_size"] == mcfg["models"]["lstm_ae"]["batch_size"]   # model value wins
    assert ae["max_epochs"] == mcfg["common"]["max_epochs"]
    assert "max_epochs" not in T.model_config("rf", mcfg)
    assert T.model_config("lstm_ae", mcfg, {"hidden": 128})["hidden"] == 128
    with pytest.raises(ValueError):
        T.model_config("nope", mcfg)


def _seg(capture, attack_type, split, start=0, stop=10_000):
    return SP.Segment("road", capture, capture, "x.parquet", attack_type, split, start, stop)


def _windows():
    segs = {s.id: s for s in [_seg("fab", "max_speedometer", "train"),
                              _seg("masq", "max_speedometer_masquerade", "train"),
                              _seg("amb", "normal", "train"), _seg("v", "normal", "val")]}
    rows = []
    for sid, s in segs.items():
        for k, lab in enumerate([0, 0, 1, 1] if s.attack_type != "normal" else [0, 0, 0, 0]):
            rows.append({"dataset": "road", "split": s.split, "capture": s.capture, "source_capture": s.capture,
                         "segment": sid, "start": k * 100, "label": lab,
                         "attack_type": s.attack_type if lab else "normal",
                         "normal_subset": s.split == "train" and lab == 0 and s.capture == "amb"})
    return pd.DataFrame(rows), segs


def test_supervised_train_drops_masquerade_normals_and_keeps_train_only():
    win, segs = _windows()
    tr = D.supervised_train(win, "road", segs)
    assert set(tr["split"]) == {"train"}
    masq = tr[tr["capture"] == "masq"]
    assert (masq["label"] == 1).all() and len(masq) == 2
    assert len(tr[tr["capture"] == "fab"]) == 4


def test_ae_train_is_normal_subset_only():
    win, _ = _windows()
    tr = D.ae_train(win, "road")
    assert set(tr["capture"]) == {"amb"} and (tr["label"] == 0).all() and set(tr["split"]) == {"train"}


def test_selector_split_guard():
    win, _ = _windows()
    with pytest.raises(ValueError):
        D._require_split(win, "train")
    with pytest.raises(ValueError):
        D.eval_windows(win, "road", "train")


def _store(n=3000, seed=0):
    frames = np.random.default_rng(seed).random((n, 10), dtype=np.float32)
    return D.FeatureStore("road", frames, {"seg": (0, 0)}, {}, "h")


def _hp(model):
    hp = T.model_config(model, T.load_models_config())
    hp.update(max_epochs=2, batch_size=16, eval_batch_size=64)
    if model == "cnn1d":
        hp.update(channels=[4, 8, 8], hidden=4)
    else:
        hp.update(hidden=8, latent=4)
    return hp


@pytest.mark.parametrize("model", ["cnn1d", "lstm_ae"])
def test_same_seed_same_training(model, tmp_path):
    store = _store()
    starts = np.arange(0, 2800, 50)
    win = pd.DataFrame({"segment": "seg", "start": starts, "label": (np.arange(len(starts)) % 4 == 0).astype(int)})
    if model == "lstm_ae":
        win["label"] = 0
    out = []
    for k in range(2):
        d = tmp_path / str(k)
        d.mkdir()
        T.seed_everything(0)
        res = T.train_deep(model, _hp(model), 0, store, win, win, 100, "cpu", d, log=lambda m: None)
        out.append((res, torch.load(d / "model.pt", weights_only=True)))
    (r0, w0), (r1, w1) = out
    assert [h["train_loss"] for h in r0["history"]] == [h["train_loss"] for h in r1["history"]]
    assert all(torch.equal(w0[k], w1[k]) for k in w0)
    assert "decision" not in r0["meta"]          # the threshold is calibrated afterwards on val normals (K-035)
    if model == "lstm_ae":
        assert r0["meta"]["early_stopping"] == "val_loss"
        assert r0["meta"]["train_score_quantiles"] == r1["meta"]["train_score_quantiles"]
    else:
        assert r0["meta"]["pos_weight"] == pytest.approx(3.0, rel=0.1)
        assert r0["meta"]["early_stopping"] == "val_average_precision"
        assert all("val_ap" in h for h in r0["history"])
        best = max(r0["history"], key=lambda h: h["val_ap"])
        assert r0["meta"]["best_epoch"] == best["epoch"] and r0["meta"]["best_monitor_value"] == best["val_ap"]
    json.dumps(r0["meta"])  # metadata is serialisable


@pytest.mark.parametrize("model", ["cnn1d", "lstm_ae"])
def test_resume_from_checkpoint_matches_uninterrupted_run(model, tmp_path):
    store = _store()
    starts = np.arange(0, 2800, 50)
    win = pd.DataFrame({"segment": "seg", "start": starts, "label": (np.arange(len(starts)) % 4 == 0).astype(int)})
    if model == "lstm_ae":
        win["label"] = 0
    hp = {**_hp(model), "max_epochs": 4, "patience": 100}
    full, part = tmp_path / "full", tmp_path / "part"
    full.mkdir(), part.mkdir()
    T.seed_everything(0)
    T.train_deep(model, hp, 0, store, win, win, 100, "cpu", full, log=lambda m: None)
    T.seed_everything(0)
    T.train_deep(model, {**hp, "max_epochs": 2}, 0, store, win, win, 100, "cpu", part, log=lambda m: None)
    assert (part / "checkpoint.pt").exists()          # left behind, as after an interruption
    T.seed_everything(0)
    res = T.train_deep(model, hp, 0, store, win, win, 100, "cpu", part, log=lambda m: None)
    assert res["meta"]["resumed_at_epochs"] == [2] and res["meta"]["epochs_run"] == 4
    wa, wb = torch.load(full / "model.pt", weights_only=True), torch.load(part / "model.pt", weights_only=True)
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


@pytest.mark.skipif(not SP.WINDOW_INDEX.exists(), reason="window index not built")
@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_real_selectors_draw_from_allowed_splits_only(dataset):
    from src.data import sources as S

    cfg = S.load_config()
    win = D.load_window_index()
    segs = {s.id: s for s in SP.segments(cfg) if s.dataset == dataset}
    sup = D.supervised_train(win, dataset, segs)
    assert set(sup["split"].astype(str)) == {"train"}
    assert not ((sup["label"] == 0) & sup["segment"].astype(str).map(lambda s: segs[s].is_masquerade)).any()
    ae = D.ae_train(win, dataset)
    assert set(ae["split"].astype(str)) == {"train"} and (ae["label"] == 0).all()
    assert len(ae) == int(win[(win["dataset"] == dataset) & win["normal_subset"]].shape[0])
    val = D.ae_val(win, dataset, segs, cfg)
    assert set(val["split"].astype(str)) == {"val"} and (val["label"] == 0).all() and len(val) > 0
    assert not val["segment"].astype(str).map(lambda s: segs[s].is_masquerade).any()
    for split in ("val", "test"):
        assert set(D.eval_windows(win, dataset, split)["split"].astype(str)) == {split}


def test_autoencoder_refuses_ap_early_stopping(tmp_path):
    store = _store()
    win = pd.DataFrame({"segment": "seg", "start": np.arange(0, 2800, 50), "label": 0})
    with pytest.raises(ValueError):
        T.train_deep("lstm_ae", {**_hp("lstm_ae"), "early_stopping": "val_average_precision"}, 0, store, win, win,
                     100, "cpu", tmp_path, log=lambda m: None)


def test_overrides_and_planned_run_id():
    mcfg = T.load_models_config()
    ov = T.resolve_overrides("ae_feasibility.retry", mcfg)
    assert ov == mcfg["ae_feasibility"]["retry"] and T.resolve_overrides(None, mcfg) == {}
    base, retry = T.planned_run_id("lstm_ae", "road", 0), T.planned_run_id("lstm_ae", "road", 0, ov)
    assert base != retry and base.startswith("lstm_ae_road_s0_")
    assert T.model_config("lstm_ae", mcfg, ov)["hidden"] == 128
    # the decision rule is not a hyperparameter: no model carries a threshold any more (K-035)
    for m in mcfg["models"]:
        hp = T.model_config(m, mcfg)
        assert "decision_threshold" not in hp and "threshold_percentile" not in hp


def test_git_info_fields():
    g = T.git_info()
    assert len(g["commit"]) == 40 and isinstance(g["dirty"], bool)


@pytest.mark.skipif(not SP.WINDOW_INDEX.exists(), reason="window index not built")
@pytest.mark.parametrize("dataset", ["car_hacking", "road"])
def test_calibration_windows_are_clean_validation_normals(dataset):
    from src.data import sources as S
    from src.data import windows as W

    cfg = S.load_config()
    win = D.load_window_index()
    segs = {s.id: s for s in SP.segments(cfg) if s.dataset == dataset}
    cal = D.calibration_windows(win, dataset, segs, cfg)
    assert cal.equals(D.ae_val(win, dataset, segs, cfg))                  # K-038: same subset as AE early stopping
    assert set(cal["split"].astype(str)) == {"val"} and (cal["label"] == 0).all()
    assert not cal["segment"].astype(str).map(lambda s: segs[s].is_masquerade).any()
    size, margin = cfg["windows"]["size"], cfg["normal_subset"]["margin_s"]
    for seg_id, g in cal.groupby(cal["segment"].astype(str), observed=True):   # +/-5 s from any injection
        seg = segs[seg_id]
        ts = seg.read(["timestamp"])["timestamp"].to_numpy()
        rel = g["start"].to_numpy() - seg.start
        assert W.margin_clean(ts, rel, size, SP.injected_timestamps(seg.parquet), margin).all()
    allnorm = win[(win["dataset"] == dataset) & (win["split"] == "val") & (win["label"] == 0)]
    assert 0 < len(cal) < len(allnorm)


def test_training_calibrates_on_calibration_windows():
    import inspect

    src = inspect.getsource(T.train)
    assert "D.calibration_windows(" in src and 'eval_windows(win, dataset, "val")' not in src


def _git_repo(root):
    import subprocess

    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=root, check=True,
                       capture_output=True)
    git("init", "-q")
    for rel in ("src/a.py", "results/tables/baseline_f1.csv", "results/tables/normalization_params.json",
                "results/tables/ae_feasibility.csv"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("v1\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")


@pytest.mark.parametrize("rel, dirty", [
    ("results/tables/baseline_f1.csv", False),          # result table rewritten by the grid
    ("results/tables/ae_feasibility.csv", False),
    ("results/raw/run_x/run_config.json", False),       # new run output (untracked)
    ("results/tables/normalization_params.json", True), # training input
    ("src/a.py", True),                                 # code
    ("configs/models.yaml", True),                      # new config file (untracked)
])
def test_dirty_check_ignores_generated_outputs_only(tmp_path, monkeypatch, rel, dirty):
    _git_repo(tmp_path)
    monkeypatch.setattr(T.S, "PROJECT_ROOT", tmp_path)
    assert T.git_info()["dirty"] is False
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text("v2\n")
    assert T.git_info()["dirty"] is dirty


def test_every_table_the_pipeline_writes_is_a_generated_output():
    from fnmatch import fnmatch

    from src import evaluate as E
    from src import run_grid as G
    from src import sanity as SA

    def generated(path):
        rel = path.relative_to(T.S.PROJECT_ROOT).as_posix()
        return any(rel == p or rel.startswith(p + "/") or fnmatch(rel, p) for p in T.GENERATED_OUTPUTS)

    written = [E.TABLES / "baseline_f1.csv", E.TABLES / "baseline_pairs.csv", SA.TABLES / "baseline_sanity.csv",
               SA.TABLES / "baseline_origins.csv", G.AE_TABLE, E.TABLES / "road_cnn_gap_by_epoch.csv",
               E.TABLES / "road_cnn_gap_val_loss.csv", E.TABLES / "road_cnn_gap_selected.csv", T.RAW_DIR / "x" / "meta.json"]
    assert all(generated(p) for p in written)
    inputs = ["normalization_params.json", "split_summary.csv", "feature_int8_levels.csv", "capture_lineage.csv"]
    assert not any(generated(E.TABLES / n) for n in inputs)
