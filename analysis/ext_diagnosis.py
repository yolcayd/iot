"""Diagnosis of the low calibrated recall of the conv-AE on ROAD (P10, docs/DECISIONS.md K-076). Descriptive only.

For every feature set and seed of cnn1d and conv_ae on ROAD:

(a) threshold   realised FPR on the clean validation normals (calibration set) vs on the attack-free test captures;
                calibrated recall vs TPR at test FPR 1 % (K-040; test labels, descriptive only); threshold gap =
                TPR@1% - calibrated recall; score quantiles of the clean validation normals and of the attack-free
                test windows; position of the validation threshold in the attack-free test score distribution
(b) features    test ROC-AUC and TPR@1% of the conv-AE along s0 -> s4 (from ext_ablation_summary.csv)
(c) model       cnn1d - conv_ae at the same feature set (ROC-AUC, TPR@1%); this mixes architecture and the
                supervised/unsupervised paradigm and is not a pure architecture effect
(d) score       (validation only, conv-AE) share of every input channel group in the mean squared reconstruction
                error on the clean validation normals and on the validation attack windows, and the attack/normal
                ratio of every group's error

Outputs: results/tables/ext_diag_threshold.csv, ext_diag_model.csv, ext_diag_channels.csv.

Usage:
    python -m analysis.ext_diagnosis
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from analysis import ext_aggregate as A
from src import train as T
from src.data import dataset as D
from src.data import sources as S

TABLES = S.PROJECT_ROOT / "results" / "tables"
DATASET = "road"


def channel_group(name: str) -> str:
    if name.startswith("byte_"):
        return "payload_bytes"
    if name.startswith("dbyte_"):
        return "payload_delta"
    if name in ("id_share_w", "id_entropy_w"):
        return "id_window"
    return name                      # can_id, iat, bit_flips, id_rate


def _calib_mask(win_val: pd.DataFrame, cal: pd.DataFrame) -> np.ndarray:
    key = win_val[["segment", "start"]].astype({"segment": str}).reset_index().merge(
        cal[["segment", "start"]].astype({"segment": str}), on=["segment", "start"], how="inner")
    m = np.zeros(len(win_val), dtype=bool)
    m[key["index"].to_numpy()] = True
    if m.sum() != len(cal):
        raise SystemExit("calibration windows are not a subset of the validation windows")
    return m


def threshold_rows(runs: list[dict], cfg: dict, qs: list[float]) -> pd.DataFrame:
    dcfg = S.load_config()
    win = D.load_window_index()
    rows = []
    cal_cache = {}
    for r in runs:
        d = T.RAW_DIR / r["run_id"]
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))["decision"]
        mt = pd.read_csv(d / "metrics_test.csv")
        sv = pd.read_parquet(d / "scores_val.parquet")
        st = pd.read_parquet(d / "scores_test.parquet")
        if "cal" not in cal_cache:
            segs = {s.id: s for s in __import__("src.data.splits", fromlist=["segments"]).segments(dcfg)}
            cal_cache["cal"] = D.calibration_windows(win, DATASET, segs, dcfg)
        cmask = _calib_mask(sv, cal_cache["cal"])
        free = mt[mt["level"] == "attack_free_capture"]["name"].tolist()
        tcap = st["segment"].str.split("[").str[0]
        amb = st[tcap.isin(free)]["score"].to_numpy()
        cal_scores = sv["score"].to_numpy()[cmask]
        thr = float(meta["threshold"])
        pooled = mt[mt["level"] == "pooled"].iloc[0]
        row = {"model": r["model"], "feature_set": r["feature_set"], "seed": r["seed"], "threshold": thr,
               "val_calibration_fpr": float(meta["realized_val_fpr"]), "n_val_calibration": int(cmask.sum()),
               "test_attack_free_fpr": float((amb > thr).mean()), "n_test_attack_free": int(len(amb)),
               "fpr_transfer_ratio": float((amb > thr).mean() / max(meta["realized_val_fpr"], 1e-12)),
               "threshold_quantile_in_test_attack_free": float((amb <= thr).mean()),
               "pooled_recall": float(pooled["recall"]), "pooled_tpr_at_test_fpr_0.01": float(pooled["roc_tpr_at_fpr_0.01"]),
               "pooled_threshold_gap": float(pooled["roc_tpr_at_fpr_0.01"] - pooled["recall"]),
               "pooled_roc_auc": float(pooled["roc_auc"])}
        for q in qs:
            row[f"val_cal_q{q}"] = float(np.quantile(cal_scores, q))
            row[f"test_free_q{q}"] = float(np.quantile(amb, q))
        rows.append(row)
    return pd.DataFrame(rows)


def model_rows(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[(summary["dataset"] == DATASET) & (summary["split"] == "test")
                & summary["metric"].isin(["roc_auc", "roc_tpr_at_fpr_0.01", "recall"])
                & ((summary["level"] == "pooled") | (summary["level"] == "group")
                   | ((summary["level"] == "attack_type") & (summary["role"] == "primary")))]
    p = s.pivot_table(index=["feature_set", "level", "name", "metric"], columns="model", values="mean").reset_index()
    p["cnn_minus_ae"] = p["cnn1d"] - p["conv_ae"]
    return p


def channel_rows(runs: list[dict]) -> pd.DataFrame:
    import torch

    from src.quantize import load_trained

    dcfg = S.load_config()
    win = D.load_window_index()
    rows = []
    stores = {}
    for r in runs:
        d = T.RAW_DIR / r["run_id"]
        rc, module = load_trained(d)
        fs = rc.get("feature_set", "s0")
        if fs not in stores:
            stores.clear()
            stores[fs] = D.FeatureStore.build(DATASET, ("val",), dcfg, feature_set=fs, log=lambda *_: None)
        store = stores[fs]
        if store.params_hash != rc["data_hash"]:
            raise SystemExit(f"{r['run_id']}: feature store does not match the run")
        cal = D.calibration_windows(win, DATASET, store.segments, dcfg)
        att = D.eval_windows(win, DATASET, "val")
        att = att[att["label"] == 1]
        net = module.net.eval()
        names = list(store.feature_names)
        groups = sorted({channel_group(n) for n in names}, key=[channel_group(n) for n in names].index)
        res = {}
        for kind, w in (("val_clean_normal", cal), ("val_attack", att)):
            g = store.global_starts(w)
            err = np.zeros(len(names))
            with torch.no_grad():
                for i in range(0, len(g), 2048):
                    x = torch.from_numpy(store.windows(g[i:i + 2048], rc["window_size"]).astype(np.float32))
                    err += ((net(x) - x) ** 2).sum(dim=(0, 1)).double().numpy()
            res[kind] = err / (len(g) * rc["window_size"])        # mean squared error per channel
        for grp in groups:
            idx = [i for i, n in enumerate(names) if channel_group(n) == grp]
            n_e, a_e = res["val_clean_normal"][idx].sum(), res["val_attack"][idx].sum()
            rows.append({"model": r["model"], "feature_set": fs, "seed": r["seed"], "channel_group": grp,
                         "n_channels": len(idx),
                         "share_of_score_normal": float(n_e / res["val_clean_normal"].sum()),
                         "share_of_score_attack": float(a_e / res["val_attack"].sum()),
                         "mse_normal": float(n_e / len(names)), "mse_attack": float(a_e / len(names)),
                         "attack_over_normal": float(a_e / n_e) if n_e > 0 else np.nan})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    cfg = yaml.safe_load((S.PROJECT_ROOT / "configs" / "ext_analysis.yaml").read_text(encoding="utf-8"))
    runs = [r for r in A.all_runs() if r["dataset"] == DATASET]
    thr = threshold_rows(runs, cfg, cfg["diagnosis"]["score_quantiles"])
    thr.to_csv(TABLES / "ext_diag_threshold.csv", index=False, float_format="%.6g")
    summary = pd.read_csv(TABLES / "ext_ablation_summary.csv")
    model_rows(summary).to_csv(TABLES / "ext_diag_model.csv", index=False, float_format="%.6g")
    ch = channel_rows([r for r in runs if r["model"] == "conv_ae"])
    ch.to_csv(TABLES / "ext_diag_channels.csv", index=False, float_format="%.6g")
    print(thr.groupby(["model", "feature_set"])[["val_calibration_fpr", "test_attack_free_fpr", "pooled_recall",
                                                 "pooled_tpr_at_test_fpr_0.01", "pooled_threshold_gap",
                                                 "threshold_quantile_in_test_attack_free"]].mean().round(4).to_string())
    print(ch.groupby(["feature_set", "channel_group"], sort=False)[
        ["share_of_score_normal", "share_of_score_attack", "attack_over_normal"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
