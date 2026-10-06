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

from src.data import dataset as D
from src.data import sources as S
from src.data import splits as SP

TABLES = S.PROJECT_ROOT / "results" / "tables"
NORMAL = "normal"
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
    auc = np.nan
    if score is not None and 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, score))
    return {"n_attack": int(y.sum()), "n_normal": int((~y).sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "roc_auc": auc}


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
    store = D.FeatureStore.build(rc["dataset"], (split,), cfg, log=log)
    win = D.eval_windows(D.load_window_index(), rc["dataset"], split)
    score = score_windows(run_dir, store, win)
    thr = meta["decision"]["threshold"]
    pred = score > thr if rc["model"] in ("lstm_ae", "conv_ae") else score >= thr
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


def baseline_table(raw: Path = S.PROJECT_ROOT / "results" / "raw") -> pd.DataFrame:
    grid = yaml.safe_load((S.PROJECT_ROOT / "configs" / "grid.yaml").read_text(encoding="utf-8"))["baseline"]
    frames, pairs = [], []
    for d in sorted(raw.iterdir()):
        if not (d / "DONE").exists() or not (d / "metrics_test.csv").exists():
            continue
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        if rc["smoke"] or rc["model"] not in grid["models"] + ["conv_ae"] or rc["seed"] not in grid["seeds"]:
            continue
        m = pd.read_csv(d / "metrics_test.csv")
        frames.append(m[m["level"].isin(["attack_type", "attack_free", "pooled"])])
        p = pd.read_csv(d / "pairs_test.csv")
        if len(p):
            p.insert(0, "run_id", rc["run_id"])
            p.insert(1, "model", rc["model"])
            pairs.append(p)
    if not frames:
        raise RuntimeError("no finished, evaluated baseline runs under results/raw")
    out = pd.concat(frames, ignore_index=True)
    cols = ["dataset", "model", "seed", "run_id", "level", "name", "group", "role", "n_attack", "n_normal",
            "precision", "recall", "f1", "fpr", "roc_auc"]
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
