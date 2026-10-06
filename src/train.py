"""Training entry point: trains a model for a given dataset and seed and stores artifacts under results/raw/<run_id>/.

The run id is derived deterministically from the model hyperparameters, the dataset, the seed and the
feature/split configuration hash; runtime resources (device, CPU threads) and the git commit are not
part of it but are written to run_config.json (CLAUDE.md rule 7, docs/DECISIONS.md K-031).

Artifacts: run_config.json, history.csv, training_curve.png, model weights (git-ignored), meta.json
(decision rule; autoencoder threshold fitted on training scores only), checkpoint.pt (deep models,
every epoch, so an interrupted run resumes) and a DONE marker.

Usage:
    python -m src.train --model cnn1d --dataset road --seed 0 [--threads 4] [--smoke --out DIR]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from src.data import dataset as D
from src.data import sources as S

MODELS_CONFIG = S.PROJECT_ROOT / "configs" / "models.yaml"
RAW_DIR = S.PROJECT_ROOT / "results" / "raw"
DEEP = {"cnn1d", "lstm_ae", "conv_ae"}
AUTOENCODERS = {"lstm_ae", "conv_ae"}
CLASSICAL = {"rf", "xgb"}
VERSIONED = ["numpy", "pandas", "pyarrow", "scikit-learn", "xgboost", "torch", "onnx", "onnxruntime", "PyYAML"]
SMOKE = {"max_windows": 4000, "max_epochs": 2, "n_estimators": 20}


# --------------------------------------------------------------------------------------------
# Configuration, run id, provenance
# --------------------------------------------------------------------------------------------


def load_models_config(path: Path = MODELS_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def model_config(name: str, mcfg: dict, overrides: dict | None = None) -> dict:
    """Hyperparameters of one model; deep models inherit the ``common`` block."""
    if name not in mcfg["models"]:
        raise ValueError(f"unknown model {name!r}")
    m = dict(mcfg["models"][name])
    if name in DEEP:
        m = {**mcfg["common"], **m}
    m.update(overrides or {})
    return m


def resolve_overrides(ref: str | dict | None, mcfg: dict) -> dict:
    """Overrides given inline or as a dotted path into configs/models.yaml (e.g. ``ae_feasibility.retry``)."""
    if ref is None or isinstance(ref, dict):
        return dict(ref or {})
    node = mcfg
    for key in ref.split("."):
        node = node[key]
    return dict(node)


def planned_run_id(model: str, dataset: str, seed: int, overrides: dict | None = None) -> str:
    """Run id that ``train`` will use for this configuration (without building the feature store)."""
    from src.data import features as F

    hp = model_config(model, load_models_config(), overrides)
    return run_id(model, dataset, seed, hp, F.load_params()["config_sha256_16"], S.load_config()["windows"]["size"])


def run_id(model: str, dataset: str, seed: int, hparams: dict, data_hash: str, window_size: int) -> str:
    key = {"model": model, "dataset": dataset, "seed": seed, "hparams": hparams, "data_hash": data_hash,
           "window_size": window_size}
    sha = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{model}_{dataset}_s{seed}_{sha}"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=S.PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_info() -> dict:
    # run outputs under results/raw do not make the tree dirty for the next run of a grid
    status = _git("status", "--porcelain", "--", ".", ":(exclude)results/raw")
    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(status), "dirty_files": status.splitlines()[:50]}


def environment(device: str, threads: int) -> dict:
    import torch

    env = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(), "threads": threads, "device": device,
           "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    env["versions"] = {}
    for p in VERSIONED:
        try:
            env["versions"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            env["versions"][p] = None
    return env


def set_threads(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch

    torch.set_num_threads(threads)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede the first cuBLAS call
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


def select(model: str, dataset: str, win, store: D.FeatureStore, cfg: dict):
    """(train windows, validation windows) for a model family."""
    if model in AUTOENCODERS:
        return D.ae_train(win, dataset), D.ae_val(win, dataset, store.segments, cfg)
    return D.supervised_train(win, dataset, store.segments), D.eval_windows(win, dataset, "val")


def _subsample(w, n: int, seed: int):
    if len(w) <= n:
        return w
    idx = np.sort(np.random.default_rng(seed).choice(len(w), n, replace=False))
    return w.iloc[idx].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Deep models
# --------------------------------------------------------------------------------------------


def build_deep(model: str, hp: dict):
    if model == "cnn1d":
        from src.models import cnn1d as M
    elif model == "lstm_ae":
        from src.models import lstm_autoencoder as M
    else:
        from src.models import conv_autoencoder as M
    return M.build(hp)


class GpuWindows:
    """Frames on the device; windows gathered per batch by global start row."""

    def __init__(self, store: D.FeatureStore, size: int, device: str):
        import torch

        self.torch = torch
        self.frames = torch.from_numpy(store.frames).to(device)
        self.offsets = torch.arange(size, device=device)
        self.device = device

    def batch(self, gstarts):
        return self.frames[gstarts[:, None] + self.offsets]


def score_deep(net, model: str, gw: GpuWindows, gstarts: np.ndarray, batch_size: int) -> np.ndarray:
    """Logit (classifier) or reconstruction MSE (autoencoder) per window; higher = more anomalous.

    Logits, not probabilities: sigmoid saturates to exactly 1.0 in float32, which would create ties at the
    validation-normal FPR threshold (K-035). The ranking is the same."""
    import torch

    from src.models.lstm_autoencoder import reconstruction_score

    net.eval()
    out = []
    g = torch.as_tensor(gstarts, device=gw.device)
    with torch.no_grad():
        for i in range(0, len(g), batch_size):
            x = gw.batch(g[i:i + batch_size])
            s = reconstruction_score(net, x) if model in AUTOENCODERS else net(x)
            out.append(s.float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def train_deep(model: str, hp: dict, seed: int, store: D.FeatureStore, tr, va, size: int, device: str,
               out: Path, log=print) -> dict:
    import torch

    gw = GpuWindows(store, size, device)
    g_tr = torch.as_tensor(store.global_starts(tr), device=device)
    g_va = torch.as_tensor(store.global_starts(va), device=device)
    y_tr = torch.as_tensor(tr["label"].to_numpy(np.float32), device=device)
    y_va = torch.as_tensor(va["label"].to_numpy(np.float32), device=device)
    is_ae = model in AUTOENCODERS
    extra = {}
    if is_ae:
        def loss_fn(x, y, net):
            return ((net(x) - x) ** 2).mean()
    else:
        from src.models.classical import pos_ratio

        pw = pos_ratio(tr["label"].to_numpy()) if hp["pos_weight"] == "auto" else float(hp["pos_weight"])
        extra["pos_weight"] = pw
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

        def loss_fn(x, y, net):
            return bce(net(x), y)

    net = build_deep(model, hp).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    gen = torch.Generator().manual_seed(seed)
    state = {"epoch": 0, "best_val": float("inf"), "best_epoch": -1, "bad": 0, "history": [], "best_state": None}
    ckpt = out / "checkpoint.pt"
    if ckpt.exists():
        # RNG states must stay CPU byte tensors; weights and optimizer state are moved by load_state_dict
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        gen.set_state(c["generator"])
        torch.set_rng_state(c["torch_rng"])
        if c.get("cuda_rng") is not None and device.startswith("cuda"):
            torch.cuda.set_rng_state_all(c["cuda_rng"])
        state = c["state"]
        state.setdefault("resumed_at_epochs", []).append(state["epoch"])
        log(f"resumed from epoch {state['epoch']}")

    def mean_loss(g, y):
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(g), hp["eval_batch_size"]):
                xb = gw.batch(g[i:i + hp["eval_batch_size"]])
                tot += float(loss_fn(xb, y[i:i + hp["eval_batch_size"]], net)) * len(xb)
        return tot / max(len(g), 1)

    # early-stopping monitor (lower is better): validation loss, or minus validation AP for the CNN (K-034)
    use_ap = hp.get("early_stopping", "val_loss") == "val_average_precision"
    if use_ap and is_ae:
        raise ValueError("autoencoders stop on validation loss (their validation set has no attacks)")
    y_va_np = va["label"].to_numpy()

    bs = hp["batch_size"]
    while state["epoch"] < hp["max_epochs"] and state["bad"] < hp["patience"]:
        t0 = time.time()
        net.train()
        perm = torch.randperm(len(g_tr), generator=gen).to(device)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = gw.batch(g_tr[idx])
            loss = loss_fn(xb, y_tr[idx], net)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        val = mean_loss(g_va, y_va)
        row = {}
        if use_ap:
            from sklearn.metrics import average_precision_score

            ap = float(average_precision_score(y_va_np, score_deep(net, model, gw, store.global_starts(va),
                                                                   hp["eval_batch_size"])))
            row["val_ap"] = ap
        monitor = -row["val_ap"] if use_ap else val
        state["epoch"] += 1
        if monitor < state["best_val"]:
            state.update(best_val=monitor, best_epoch=state["epoch"], bad=0,
                         best_state=copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()}))
        else:
            state["bad"] += 1
        state["history"].append({"epoch": state["epoch"], "train_loss": tot / len(perm), "val_loss": val, **row,
                                 "seconds": time.time() - t0})
        # written to a temporary file and renamed, so a kill during the write leaves the previous checkpoint
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "generator": gen.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                    "state": state}, tmp)
        os.replace(tmp, ckpt)
        ap_txt = f" val AP {row['val_ap']:.4f}" if use_ap else ""
        log(f"epoch {state['epoch']:3d} train {tot / len(perm):.6f} val {val:.6f}{ap_txt} ({time.time() - t0:.1f} s)")

    net.load_state_dict(state["best_state"])
    torch.save(net.state_dict(), out / "model.pt")
    meta = {"best_epoch": state["best_epoch"], "early_stopping": "val_average_precision" if use_ap else "val_loss",
            "best_monitor_value": -state["best_val"] if use_ap else state["best_val"], "epochs_run": state["epoch"],
            "stopped_early": state["bad"] >= hp["patience"],
            "resumed_at_epochs": state.get("resumed_at_epochs", []), **extra}
    if is_ae:   # descriptive only; the decision threshold comes from validation normals (K-035)
        scores = score_deep(net, model, gw, store.global_starts(tr), hp["eval_batch_size"])
        meta["train_score_quantiles"] = {str(q): float(np.quantile(scores, q)) for q in (0.5, 0.9, 0.99, 0.999)}
    return {"meta": meta, "history": state["history"]}


# --------------------------------------------------------------------------------------------
# Classical models
# --------------------------------------------------------------------------------------------


def summary_matrix(store: D.FeatureStore, w, size: int, chunk: int = 20000) -> np.ndarray:
    from src.models.classical import summary_features

    g = store.global_starts(w)
    return np.concatenate([summary_features(store.windows(g[i:i + chunk], size)) for i in range(0, len(g), chunk)])


def train_classical(model: str, hp: dict, seed: int, store: D.FeatureStore, tr, va, size: int, device: str,
                    threads: int, out: Path, log=print) -> dict:
    import joblib

    from src.models import classical as C

    x_tr, y_tr = summary_matrix(store, tr, size), tr["label"].to_numpy()
    x_va, y_va = summary_matrix(store, va, size), va["label"].to_numpy()
    log(f"summary features: train {x_tr.shape}, val {x_va.shape}")
    meta = {"summary_features": C.SUMMARY_NAMES}
    history = []
    if model == "rf":
        clf = C.build_rf(hp, seed, threads).fit(x_tr, y_tr)
        joblib.dump(clf, out / "model.joblib")
    else:
        spw = C.pos_ratio(y_tr) if hp["scale_pos_weight"] == "auto" else float(hp["scale_pos_weight"])
        meta["scale_pos_weight"] = spw
        dev = "cuda" if device.startswith("cuda") else "cpu"
        clf = C.build_xgb(hp, seed, threads, dev, spw)
        clf.fit(x_tr, y_tr, eval_set=[(x_tr, y_tr), (x_va, y_va)], verbose=False)
        ev = clf.evals_result()
        history = [{"epoch": i + 1, "train_loss": a, "val_loss": b}
                   for i, (a, b) in enumerate(zip(ev["validation_0"]["logloss"], ev["validation_1"]["logloss"]))]
        meta["best_iteration"] = int(clf.best_iteration)
        clf.save_model(out / "model.ubj")
    return {"meta": meta, "history": history}


# --------------------------------------------------------------------------------------------
# Loading a trained run (used by src.evaluate)
# --------------------------------------------------------------------------------------------


def score_windows(run_dir: Path, store: D.FeatureStore, w, device: str | None = None) -> np.ndarray:
    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    model, hp, size = rc["model"], rc["hparams"], rc["window_size"]
    device = device or rc["environment"]["device"]
    if model in DEEP:
        import torch

        net = build_deep(model, hp)
        net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
        net.to(device)
        return score_deep(net, model, GpuWindows(store, size, device), store.global_starts(w), hp["eval_batch_size"])
    x = summary_matrix(store, w, size)
    if model == "rf":
        import joblib

        clf = joblib.load(run_dir / "model.joblib")
    else:
        from xgboost import XGBClassifier

        clf = XGBClassifier()
        clf.load_model(run_dir / "model.ubj")
        # same device as in training (CPU and CUDA predictions differ by ~1e-7) and the early-stopping
        # best iteration stated explicitly rather than relying on the attribute restored by load_model
        # raw margin (log-odds), not probability: no saturation ties at the FPR threshold (K-035)
        clf.set_params(device="cuda" if device.startswith("cuda") else "cpu")
        best = int(rc["meta"]["best_iteration"])
        return clf.predict(x, output_margin=True, iteration_range=(0, best + 1)).astype(np.float64)
    return clf.predict_proba(x)[:, 1]


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def plot_history(history: list[dict], path: Path, title: str) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = {k: [r[k] for r in history] for k in ("epoch", "train_loss", "val_loss")}
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.plot(h["epoch"], h["train_loss"], marker="o", ms=2, label="train")
    ax.plot(h["epoch"], h["val_loss"], marker="s", ms=2, ls="--", label="validation")
    ax.set_xlabel("epoch / boosting round")
    ax.set_ylabel("loss")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(model: str, dataset: str, seed: int, threads: int | None = None, device: str | None = None,
          out_root: Path = RAW_DIR, smoke: bool = False, allow_dirty: bool = False, force: bool = False,
          overrides: dict | None = None, log=print) -> Path:
    mcfg = load_models_config()
    threads = threads or mcfg["runtime"]["threads"]
    device = device or mcfg["runtime"]["device"]
    set_threads(threads)
    seed_everything(seed)
    git = git_info()
    if git["dirty"] and not (allow_dirty or smoke):
        raise RuntimeError("working tree has uncommitted changes (CLAUDE.md rule 7); commit or pass --allow-dirty")
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    cfg = S.load_config()
    size = cfg["windows"]["size"]
    hp = model_config(model, mcfg, overrides)
    if smoke:
        hp.update({k: v for k, v in (("max_epochs", SMOKE["max_epochs"]), ("n_estimators", SMOKE["n_estimators"]))
                   if k in hp})
    store = D.FeatureStore.build(dataset, ("train", "val"), cfg, log=log)
    rid = run_id(model, dataset, seed, hp, store.params_hash, size) + ("_smoke" if smoke else "")
    out = out_root / rid
    if (out / "DONE").exists() and not force:
        log(f"{rid}: already done, skipping")
        return out
    out.mkdir(parents=True, exist_ok=True)

    win = D.load_window_index()
    tr, va = select(model, dataset, win, store, cfg)
    if smoke:
        tr, va = _subsample(tr, SMOKE["max_windows"], seed), _subsample(va, SMOKE["max_windows"], seed)
    started = datetime.now(timezone.utc)
    rc = {"run_id": rid, "model": model, "dataset": dataset, "seed": seed, "smoke": smoke, "hparams": hp,
          "overrides": overrides or {}, "decision_config": mcfg["decision"],
          "window_size": size, "data_hash": store.params_hash,
          "train_windows": {**D.describe(tr), "digest": D.selection_digest(tr)},
          "val_windows": {**D.describe(va), "digest": D.selection_digest(va)},
          "git": git, "environment": environment(device, threads),
          "started_utc": started.isoformat(timespec="seconds")}
    (out / "run_config.json").write_text(json.dumps(rc, indent=2) + "\n", encoding="utf-8")
    log(f"{rid}: train {rc['train_windows']}, val {rc['val_windows']}")

    if model in DEEP:
        res = train_deep(model, hp, seed, store, tr, va, size, device, out, log)
    elif model in CLASSICAL:
        res = train_classical(model, hp, seed, store, tr, va, size, device, threads, out, log)
    else:
        raise ValueError(f"unknown model {model!r}")

    finished = datetime.now(timezone.utc)
    rc.update(finished_utc=finished.isoformat(timespec="seconds"),
              duration_s=round((finished - started).total_seconds(), 1), meta=res["meta"])
    (out / "run_config.json").write_text(json.dumps(rc, indent=2) + "\n", encoding="utf-8")

    # decision threshold from ALL label-0 validation windows at the target FPR (K-035); attack labels unused
    from src.calibration import calibrate_threshold

    cal = D.eval_windows(win, dataset, "val")
    if smoke:
        cal = _subsample(cal, SMOKE["max_windows"], seed)
    scores = score_windows(out, store, cal, device)
    res["meta"]["decision"] = {**calibrate_threshold(scores, cal["label"].to_numpy(), "val",
                                                     mcfg["decision"]["target_fpr"]),
                               "calibration_windows_digest": D.selection_digest(cal)}
    d = res["meta"]["decision"]
    log(f"{rid}: threshold {d['threshold']:.6g} (val normal FPR {d['realized_val_fpr']:.4f}, n={d['n_val_normal']})")
    rc["meta"] = res["meta"]
    (out / "run_config.json").write_text(json.dumps(rc, indent=2) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(res["meta"], indent=2) + "\n", encoding="utf-8")
    if res["history"]:
        import pandas as pd

        pd.DataFrame(res["history"]).to_csv(out / "history.csv", index=False)
        plot_history(res["history"], out / "training_curve.png", rid)
    (out / "checkpoint.pt").unlink(missing_ok=True)
    (out / "DONE").write_text(finished.isoformat(timespec="seconds") + "\n", encoding="utf-8")
    log(f"{rid}: done in {rc['duration_s']} s")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(DEEP | CLASSICAL))
    ap.add_argument("--dataset", required=True, choices=["car_hacking", "road"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, help="CPU threads (default: configs/models.yaml runtime.threads)")
    ap.add_argument("--device", help="default: configs/models.yaml runtime.device")
    ap.add_argument("--out", type=Path, default=RAW_DIR)
    ap.add_argument("--smoke", action="store_true", help="tiny run (few windows, 2 epochs) for checking the pipeline")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain even if the run is marked DONE")
    ap.add_argument("--overrides", help="dotted path into configs/models.yaml, e.g. ae_feasibility.retry")
    a = ap.parse_args(argv)
    ov = resolve_overrides(a.overrides, load_models_config())
    train(a.model, a.dataset, a.seed, a.threads, a.device, a.out, a.smoke, a.allow_dirty, a.force, overrides=ov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
