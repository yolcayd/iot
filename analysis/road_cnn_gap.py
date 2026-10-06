"""Generalisation-gap analysis of the ROAD 1D-CNN baseline (P3).

Re-trains the baseline ROAD CNN with the SAME hyperparameters and seed into a scratch directory and keeps the
weights of every epoch (training is resumed epoch by epoch; resuming is bit-identical, tests/test_train.py).
Early stopping only decides when training stops, not the trajectory, so epochs 1..6 equal the baseline run;
the run is continued to ``--epochs`` for the analysis only. Epoch-1 weights are checked against the stored
baseline model.

For every epoch: per-attack-type and per-capture metrics on TRAIN and VAL windows (decision threshold 0.5,
as in the baseline), plus a decomposition of the weighted validation loss (pos_weight = N_neg / N_pos).
TEST metrics are reported for the epoch selected by early stopping only (the stored baseline model), so the
test set never informs an epoch choice (CLAUDE.md: no model selection on test data).

This analysis documents the P3 v1 baseline (early stopping on weighted validation loss, threshold 0.5),
which led to K-034 and K-035; it is pinned to that run with ``--run``.

Usage:
    python -m analysis.road_cnn_gap --scratch DIR [--epochs 12] [--run cnn1d_road_s0_e89b45da]
Writes results/tables/road_cnn_gap_by_epoch.csv, road_cnn_gap_val_loss.csv and road_cnn_gap_selected.csv.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src import evaluate as E
from src import train as T
from src.data import dataset as D
from src.data import sources as S
from src.data import splits as SP

TABLES = S.PROJECT_ROOT / "results" / "tables"


def logits(net, gw, gstarts: np.ndarray, batch: int = 4096) -> np.ndarray:
    net.eval()
    g = torch.as_tensor(gstarts, device=gw.device)
    out = []
    with torch.no_grad():
        for i in range(0, len(g), batch):
            out.append(net(gw.batch(g[i:i + batch])).float().cpu().numpy())
    return np.concatenate(out)


def bce(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-window binary cross-entropy from logits (numerically stable)."""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def loss_decomposition(z: np.ndarray, y: np.ndarray, types: np.ndarray, pw: float) -> dict:
    l = bce(z, y)
    pos, neg = y == 1, y == 0
    weighted = (pw * l[pos].sum() + l[neg].sum()) / len(y)      # = BCEWithLogitsLoss(pos_weight) mean
    pos_part = pw * l[pos].sum() / len(y)
    lp = np.sort(l[pos])[::-1]
    top = {k: float(pw * lp[:k].sum() / len(y) / weighted) for k in (10, 50)}
    top1 = max(1, int(round(0.01 * pos.sum())))
    p = 1 / (1 + np.exp(-z))
    out = {"weighted_loss": float(weighted), "unweighted_loss": float(l.mean()),
           "share_from_positives": float(pos_part / weighted),
           "share_top10_positives": top[10], "share_top50_positives": top[50],
           "share_top1pct_positives": float(pw * lp[:top1].sum() / len(y) / weighted),
           "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "median_pos_loss": float(np.median(l[pos])), "median_neg_loss": float(np.median(l[neg])),
           "roc_auc": float(roc_auc_score(y, z)), "average_precision": float(average_precision_score(y, z)),
           "pred_pos_rate": float((p >= 0.5).mean())}
    for t in sorted(set(types[pos])):
        out[f"share_pos_{t}"] = float(pw * l[pos & (types == t)].sum() / len(y) / weighted)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run", default="cnn1d_road_s0_e89b45da", help="P3 v1 ROAD CNN baseline run")
    a = ap.parse_args(argv)

    base = T.RAW_DIR / a.run
    rc = json.loads((base / "run_config.json").read_text(encoding="utf-8"))
    hp, seed, size = dict(rc["hparams"]), rc["seed"], rc["window_size"]
    T.set_threads(a.threads)
    cfg = S.load_config()
    groups = yaml.safe_load(SP.GROUPS_PATH.read_text(encoding="utf-8"))
    store = D.FeatureStore.build("road", SP.SPLITS, cfg)
    win = D.load_window_index()
    tr, va = T.select("cnn1d", "road", win, store, cfg)          # exactly the baseline selection
    if D.selection_digest(tr) != rc["train_windows"]["digest"] or D.selection_digest(va) != rc["val_windows"]["digest"]:
        raise RuntimeError("window selection differs from the baseline run")
    pw = rc["meta"]["pos_weight"]
    train_all = win[(win["dataset"] == "road") & (win["split"] == "train")].reset_index(drop=True)  # incl. masquerade normals
    gw = T.GpuWindows(store, size, "cuda")

    out = a.scratch / "road_cnn_gap"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    rows, loss_rows = [], []
    for ep in range(1, a.epochs + 1):
        T.seed_everything(seed)
        T.train_deep("cnn1d", {**hp, "max_epochs": ep, "patience": 10_000}, seed, store, tr, va, size, "cuda", out,
                     log=lambda m: None)
        ck = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        net = T.build_deep("cnn1d", hp)
        net.load_state_dict(ck["model"])
        net.to("cuda")
        if ep == 1:
            ref = torch.load(base / "model.pt", map_location="cpu", weights_only=True)
            same = all(torch.equal(ref[k], ck["model"][k].cpu()) for k in ref)
            print(f"epoch-1 weights identical to the baseline model: {same}", flush=True)
            if not same:
                raise RuntimeError("re-training does not reproduce the baseline epoch-1 model")
        for split, w in (("train", train_all), ("val", va)):
            z = logits(net, gw, store.global_starts(w))
            p = 1 / (1 + np.exp(-z))
            m = E.evaluate_windows(w, p, p >= 0.5, "road", groups, cfg)
            m.insert(0, "split", split)
            m.insert(0, "epoch", ep)
            rows.append(m)
            if split == "val":
                loss_rows.append({"epoch": ep, "logged_val_loss": ck["state"]["history"][-1]["val_loss"],
                                  **loss_decomposition(z, w["label"].to_numpy().astype(float),
                                                       w["attack_type"].astype(str).to_numpy(), pw)})
        print(f"epoch {ep}: val weighted loss {loss_rows[-1]['weighted_loss']:.3f} "
              f"(logged {loss_rows[-1]['logged_val_loss']:.3f}), val AUC {loss_rows[-1]['roc_auc']:.3f}", flush=True)

    by_epoch = pd.concat(rows, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    by_epoch.to_csv(TABLES / "road_cnn_gap_by_epoch.csv", index=False, float_format="%.4f")
    pd.DataFrame(loss_rows).to_csv(TABLES / "road_cnn_gap_val_loss.csv", index=False, float_format="%.4f")

    # selected model (epoch chosen by early stopping = stored baseline model): train / val / test side by side
    sel = rc["meta"]["best_epoch"]
    test = pd.read_csv(base / "metrics_test.csv")
    test.insert(0, "epoch", sel)
    cols = ["epoch", "split", "level", "name", "n_attack", "n_normal", "precision", "recall", "f1", "fpr", "roc_auc"]
    side = pd.concat([by_epoch[by_epoch["epoch"] == sel], test], ignore_index=True)[cols]
    side.to_csv(TABLES / "road_cnn_gap_selected.csv", index=False, float_format="%.4f")
    shutil.rmtree(out, ignore_errors=True)
    print(f"selected epoch {sel}; wrote road_cnn_gap_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
