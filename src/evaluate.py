"""Per-attack-type evaluation of a model version on the test set.

Definitions (docs/DECISIONS.md K-029):

* Per attack type t (primary): the attack windows of type t plus the normal windows of the SAME
  captures (the captures that contain type-t attack windows) -> precision, recall (TPR), F1, FPR,
  ROC-AUC, window counts. Recall depends on the attack windows only and is reported next to F1 for
  between-type comparisons.
* Role: ``unseen`` for types listed in configs/data.yaml analysis_roles.unseen_in_training (ROAD
  coolant pair), reported separately and kept out of the pooled metrics; else ``primary``.
* Attack-free captures: FPR on all attack-free captures together (``ambient``) and on every
  attack-free capture on its own.
* Pairs: fabrication vs masquerade capture of the same source capture (K-013).
* Pooled (secondary): every window of the primary pool.
* Decision: score > threshold, the threshold calibrated on validation normals at FPR 1% (K-035). The
  primary AS-1 measure is the per-type recall at that threshold (K-037); F1, ROC-AUC and AP are secondary.
* ``roc_tpr_at_fpr_0.01``: highest TPR on the evaluated set's own ROC curve at FPR <= 1%; descriptive only,
  same status as ROC-AUC, never used for a choice (K-040).

Usage:
    python -m src.evaluate --run <run_id> [--split test|val]   # metrics_<split>.csv, scores_<split>.parquet
    python -m src.evaluate --baseline-table                    # results/tables/baseline_f1.csv, baseline_pairs.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import calibration as C
from src.data import dataset as D
from src.data import sources as S
from src.data import splits as SP

TABLES = S.PROJECT_ROOT / "results" / "tables"
NORMAL = "normal"
ROC_FPR = 0.01     # operating point of the descriptive ROC measure (K-040)
PAIR_COLUMNS = ["source_capture", "fabrication", "masquerade", "role", "f1_fabrication", "f1_masquerade",
                "recall_fabrication", "recall_masquerade", "fpr_fabrication", "fpr_masquerade"]


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, fp = int((y & pred).sum()), int((~y & pred).sum())
    fn, tn = int((y & ~pred).sum()), int((~y & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    auc = ap = tpr_roc = np.nan
    if score is not None and 0 < y.sum() < len(y):
        from sklearn.metrics import average_precision_score, roc_auc_score

        auc = float(roc_auc_score(y, score))
        ap = float(average_precision_score(y, score))
        tpr_roc = roc_tpr_at_fpr(y, score, ROC_FPR)
    return {"n_attack": int(y.sum()), "n_normal": int((~y).sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "roc_auc": auc,
            "average_precision": ap, "roc_tpr_at_fpr_0.01": tpr_roc}


def roc_tpr_at_fpr(y: np.ndarray, score: np.ndarray, max_fpr: float) -> float:
    """Highest TPR on the ROC curve of THIS window set with FPR <= max_fpr (K-040).

    Descriptive only, same status as ROC-AUC: the operating point is read off the evaluated set's own labels,
    so it is never used for any threshold, model or epoch choice. The decision threshold stays the one
    calibrated on validation normals (K-035, K-038).
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(np.asarray(y).astype(bool), score)
    ok = fpr <= max_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def roles(cfg: dict) -> dict[str, set[str]]:
    r = cfg["splits"]["analysis_roles"]
    return {k: set(v) for k, v in r.items()}


def evaluate_windows(win: pd.DataFrame, score: np.ndarray, pred: np.ndarray, dataset: str, groups: dict,
                     cfg: dict) -> pd.DataFrame:
    """All metric rows of one model version on one split (see module docstring)."""
    win = win.reset_index(drop=True)
    cap = win["capture"].astype(str).to_numpy()
    typ = win["attack_type"].astype(str).to_numpy()
    y = win["label"].to_numpy() == 1
    unseen = roles(cfg)["unseen_in_training"]
    rows = []

    def add(level, name, mask, **kw):
        rows.append({"level": level, "name": name, **kw, **binary_metrics(y[mask], pred[mask], score[mask])})

    types = sorted(set(typ[y]))
    unseen_caps = set()
    for t in types:
        caps = set(cap[y & (typ == t)])
        role = "unseen" if f"{dataset}/{t}" in unseen else "primary"
        if role == "unseen":
            unseen_caps |= caps
        mask = np.isin(cap, list(caps)) & (~y | (typ == t))
        add("attack_type", t, mask, group=groups[dataset][t]["group"], role=role, captures=";".join(sorted(caps)))
    attack_caps = set(cap[y])
    free = sorted(set(cap) - attack_caps)
    if free:
        add("attack_free", "ambient", np.isin(cap, free), role="fpr_only", captures=";".join(free))
        for c in free:
            add("attack_free_capture", c, cap == c, role="fpr_only", captures=c)
    for c in sorted(attack_caps):
        t = sorted(set(typ[(cap == c) & y]))
        add("capture", c, cap == c, role="unseen" if c in unseen_caps else "primary", attack_types=";".join(t))
    add("pooled", "primary_pool", ~np.isin(cap, list(unseen_caps)), role="secondary")
    return pd.DataFrame(rows)


def pairs_table(metrics: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """Fabrication vs masquerade capture of the same source capture (K-013)."""
    c = metrics[metrics["level"] == "capture"].set_index("name")
    src = win.groupby(win["capture"].astype(str), observed=True)["source_capture"].first().astype(str)
    src = src[src.index.isin(c.index)]
    is_masq = c["attack_types"].astype(str).str.endswith("_masquerade")
    out = []
    for s in sorted(set(src)):
        caps = list(src.index[src == s])
        masq = [k for k in caps if is_masq[k]]
        fab = [k for k in caps if not is_masq[k]]
        if len(masq) == 1 and len(fab) == 1:
            f, m = c.loc[fab[0]], c.loc[masq[0]]
            out.append({"source_capture": s, "fabrication": fab[0], "masquerade": masq[0], "role": m["role"],
                        "f1_fabrication": f["f1"], "f1_masquerade": m["f1"],
                        "recall_fabrication": f["recall"], "recall_masquerade": m["recall"],
                        "fpr_fabrication": f["fpr"], "fpr_masquerade": m["fpr"]})
    return pd.DataFrame(out, columns=PAIR_COLUMNS)


def evaluate_run(run_dir: Path, split: str = "test", log=print) -> pd.DataFrame:
    from src.train import score_windows

    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = S.load_config()
    groups = yaml.safe_load(SP.GROUPS_PATH.read_text(encoding="utf-8"))
    store = D.FeatureStore.build(rc["dataset"], (split,), cfg, log=log, feature_set=rc.get("feature_set", "s0"))
    win = D.eval_windows(D.load_window_index(), rc["dataset"], split)
    score = score_windows(run_dir, store, win)
    pred = C.predict(score, meta["decision"])     # threshold calibrated on validation normals (K-035)
    m = evaluate_windows(win, score, pred, rc["dataset"], groups, cfg)
    m.insert(0, "split", split)
    for k in ("seed", "dataset", "model", "run_id"):
        m.insert(0, k, rc[k])
    m.to_csv(run_dir / f"metrics_{split}.csv", index=False)
    pairs_table(m, win).to_csv(run_dir / f"pairs_{split}.csv", index=False)
    pd.DataFrame({"segment": win["segment"].astype(str), "start": win["start"], "label": win["label"],
                  "attack_type": win["attack_type"].astype(str), "score": score}).to_parquet(
        run_dir / f"scores_{split}.parquet", index=False)
    return m


def selected_autoencoders() -> dict[str, str | None]:
    """dataset -> run id of the autoencoder selected by the K-033 protocol (None = excluded / not decided)."""
    from src import run_grid as G
    from src import train as T

    table, dec = G.ae_check(T.load_models_config()["ae_feasibility"]["min_val_roc_auc"])
    if dec["selected"] is None:
        return {d: None for d in G.DATASETS}
    out = {}
    for d, cand in dec["selected"].items():
        out[d] = None if cand is None else table.loc[(table.candidate == cand) & (table.dataset == d), "run"].iloc[0]
    return out


def baseline_table(run_dirs: list[Path] | None = None) -> pd.DataFrame:
    """Test metrics of the P3 baseline: the runs the CURRENT config plans for the baseline stage plus the
    autoencoder selected by K-033 (older runs with other run ids under results/raw are ignored).

    ``status``: ``baseline`` (supervised), ``selected_k033`` / ``rejected_k033`` (autoencoders).
    """
    from src import train as T

    sel = {}
    if run_dirs is None:
        from src.run_grid import planned_dirs

        sel = selected_autoencoders()
        run_dirs = planned_dirs("baseline") + [T.RAW_DIR / r for r in sel.values() if r]
    selected = {r for r in sel.values() if r}
    frames, pairs = [], []
    for d in dict.fromkeys(run_dirs):
        if not (d / "DONE").exists() or not (d / "metrics_test.csv").exists():
            continue
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        status = ("selected_k033" if rc["run_id"] in selected else "rejected_k033")             if rc["model"] in T.AUTOENCODERS else "baseline"
        m = pd.read_csv(d / "metrics_test.csv")
        m = m[m["level"].isin(["attack_type", "attack_free", "pooled"])].copy()
        m["status"] = status
        frames.append(m)
        p = pd.read_csv(d / "pairs_test.csv")
        if len(p):
            p.insert(0, "run_id", rc["run_id"])
            p.insert(1, "model", rc["model"])
            p.insert(2, "status", status)
            pairs.append(p)
    if not frames:
        raise RuntimeError("no finished, evaluated baseline runs under results/raw")
    out = pd.concat(frames, ignore_index=True)
    cols = ["dataset", "model", "status", "seed", "run_id", "level", "name", "group", "role", "n_attack", "n_normal",
            "precision", "recall", "f1", "fpr", "roc_auc", "average_precision", "roc_tpr_at_fpr_0.01"]
    out = out[cols].sort_values(["dataset", "level", "name", "model"]).reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "baseline_f1.csv", index=False, float_format="%.4f")
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(TABLES / "baseline_pairs.csv", index=False, float_format="%.4f")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", help="run id under results/raw")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--baseline-table", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        m = evaluate_run(S.PROJECT_ROOT / "results" / "raw" / a.run, a.split)
        print(m[m["level"] != "attack_free_capture"].to_string(index=False))
    if a.baseline_table:
        print(baseline_table().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
