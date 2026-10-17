"""Aggregation of the P10 feature ablation (docs/DECISIONS.md K-074) and of the quantization of the selected sets (K-077).

Inputs: the native (PyTorch) evaluations metrics_<split>.csv / scores_test.parquet of the s0 runs (full stage) and of
the extended-feature runs (ext_test stage); quant/metrics.csv of the quantized runs; results/tables/ext_selection.json.

Outputs (results/tables/):
    ext_ablation.csv            tidy: model, dataset, feature_set, n_features, seed, split, level, name, group, role,
                                metric, value; level ``group`` = mean over the primary types of a group
                                (stealthy_mean, coarse_mean) per seed
    ext_ablation_summary.csv    mean, std (ddof 1), n_seeds per cell; difference to s0 (same seed): mean, std, min, max
    ext_ablation_contrast.csv   selected set - s0 per model on the ROAD test set: recall, F1, ROC-AUC with a 95 %
                                hierarchical bootstrap interval (K-058 method; independent seed draws per set, shared
                                window weights)
    ext_quant.csv               Q0/Q1/Q2 x threshold mode (recalibrated / q0_threshold) on the test set for the selected
                                set of every model next to s0: recall, FPR (attack types, attack-free), seed mean +- std
                                and Q0 - Qk

Usage:
    python -m analysis.ext_aggregate
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from analysis import stats as X
from src import run_grid as G
from src import train as T
from src.data import sources as S

TABLES = S.PROJECT_ROOT / "results" / "tables"
CONFIG = S.PROJECT_ROOT / "configs" / "ext_analysis.yaml"
MODELS = ("cnn1d", "conv_ae")
SETS = ("s0", "s1", "s2", "s3", "s4")
METRICS = ("recall", "f1", "roc_auc", "average_precision", "roc_tpr_at_fpr_0.01", "fpr", "precision")
LEVELS = ("attack_type", "attack_free", "pooled")
BOOT_METRICS = ("recall", "f1", "roc_auc")


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def all_runs() -> list[dict]:
    ref = [{**r, "feature_set": "s0"} for r in G.stage_runs("full") if r["model"] in MODELS]
    ext = G.stage_runs("ext_test")
    return ref + ext


def tidy(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for r in runs:
        d = T.RAW_DIR / r["run_id"]
        rc = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
        for split in ("val", "test"):
            f = d / f"metrics_{split}.csv"
            if not f.exists():
                raise SystemExit(f"missing {f}")
            m = pd.read_csv(f)
            m = m[m["level"].isin(LEVELS)].copy()
            m["feature_set"] = r["feature_set"]
            m["n_features"] = T.run_n_features(rc)
            frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    keys = ["model", "dataset", "feature_set", "n_features", "seed", "split", "level", "name", "group", "role"]
    long = df.melt(id_vars=keys, value_vars=list(METRICS), var_name="metric", value_name="value")
    # group means over the primary attack types (per seed); fpr is per-type in-capture FPR
    t = long[(long["level"] == "attack_type") & (long["role"] == "primary")]
    g = t.groupby(["model", "dataset", "feature_set", "n_features", "seed", "split", "group", "metric"],
                  as_index=False)["value"].mean()
    g["level"], g["name"], g["role"] = "group", g["group"] + "_mean", "primary"
    return pd.concat([long, g[long.columns]], ignore_index=True)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "dataset", "feature_set", "n_features", "split", "level", "name", "group", "role", "metric"]
    s = long.groupby(keys, dropna=False, as_index=False)["value"].agg(mean="mean", std="std", n_seeds="count")
    base = long[long["feature_set"] == "s0"].drop(columns=["feature_set", "n_features"])
    j = long.merge(base, on=["model", "dataset", "seed", "split", "level", "name", "group", "role", "metric"],
                   suffixes=("", "_s0"), how="left")
    j["d"] = j["value"] - j["value_s0"]
    d = j.groupby(keys, dropna=False, as_index=False)["d"].agg(delta_mean="mean", delta_std="std", delta_min="min",
                                                               delta_max="max")
    return s.merge(d, on=keys, how="left").sort_values(keys)


# --------------------------------------------------------------------------------------------
# Bootstrap contrast: selected set - s0 (ROAD test)
# --------------------------------------------------------------------------------------------


def _scores(run_id: str) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    d = T.RAW_DIR / run_id
    w = pd.read_parquet(d / "scores_test.parquet")
    thr = float(json.loads((d / "meta.json").read_text(encoding="utf-8"))["decision"]["threshold"])
    m = pd.read_csv(d / "metrics_test.csv")
    return w, thr, m[m["level"] == "attack_type"].set_index("name")


def type_sets(win: pd.DataFrame, dataset: str) -> dict[str, dict]:
    """K-029 evaluation set of every attack type, split into (segment, label) strata in time order (as stats.py)."""
    w = win.assign(capture=win["segment"].str.split("[").str[0])
    y = w["label"].to_numpy() == 1
    cap, typ = w["capture"].to_numpy(), w["attack_type"].to_numpy()
    out = {}
    for t in sorted(set(typ[y])):
        caps = sorted(set(cap[y & (typ == t)]))
        mask = np.isin(cap, caps) & (~y | (typ == t))
        sub = w[mask].assign(_i=np.flatnonzero(mask)).sort_values(["segment", "label", "start"], kind="stable")
        strata = [(f"{dataset}|{seg}|{lab}", g["_i"].to_numpy()) for (seg, lab), g in sub.groupby(["segment", "label"])]
        out[t] = {"strata": strata, "index": np.concatenate([s[1] for s in strata])}
    return out


def contrast(model: str, chosen: str, dataset: str, cfg: dict, groups: dict, roles: dict) -> list[dict]:
    b = cfg["bootstrap"]
    runs = {fs: sorted([r for r in all_runs() if r["model"] == model and r["dataset"] == dataset
                        and r["feature_set"] == fs], key=lambda r: r["seed"]) for fs in ("s0", chosen)}
    data = {fs: [_scores(r["run_id"]) for r in rs] for fs, rs in runs.items()}
    win = data["s0"][0][0]
    for fs in data:
        for w, _, _ in data[fs]:
            if not (np.array_equal(w["segment"].to_numpy(), win["segment"].to_numpy())
                    and np.array_equal(w["start"].to_numpy(), win["start"].to_numpy())):
                raise SystemExit(f"{model} {fs}: test windows differ between runs")
    sets = type_sets(win, dataset)
    n_seeds = len(runs["s0"])
    draws = {fs: X.seed_draws(b["seed"], f"{model}|{fs}", b["n_boot"], n_seeds) for fs in data}
    rows, rep_by_type = [], {}
    for t, ts in sets.items():
        idx = ts["index"]
        y = win["label"].to_numpy()[idx] == 1
        gens = [(X.stratum_rng(b["seed"], X.block_length(len(ix), b["block_length"]), name),
                 X.block_length(len(ix), b["block_length"]), len(ix)) for name, ix in ts["strata"]]
        prep = {fs: [X.Prepared(w["score"].to_numpy(np.float64)[idx], y, thr) for w, thr, _ in data[fs]] for fs in data}
        theta = {(fs, k): np.empty((n_seeds, b["n_boot"])) for fs in data for k in BOOT_METRICS}
        point = {(fs, k): np.array([p.metrics(np.ones((1, len(idx))))[k][0] for p in prep[fs]])
                 for fs in data for k in BOOT_METRICS}
        for fs in data:                                   # recomputed full-sample values must equal metrics_test.csv
            for i, (_, _, m) in enumerate(data[fs]):
                for k in BOOT_METRICS:
                    if not abs(point[(fs, k)][i] - float(m.loc[t, k])) <= 1e-9:
                        raise SystemExit(f"{model} {fs} {t} {k}: recomputed {point[(fs, k)][i]} != {m.loc[t, k]}")
        chunk = 500
        for b0 in range(0, b["n_boot"], chunk):
            nb = min(chunk, b["n_boot"] - b0)
            C = np.concatenate([X.mbb_counts(n, L, nb, g) for g, L, n in gens], axis=1).astype(np.float32)
            for fs in data:
                for i, p in enumerate(prep[fs]):
                    res = p.metrics(C)
                    for k in BOOT_METRICS:
                        theta[(fs, k)][i, b0:b0 + nb] = res[k]
        rep_by_type[t] = {}
        for k in BOOT_METRICS:
            rep = X.hier(theta[(chosen, k)], draws[chosen]) - X.hier(theta[("s0", k)], draws["s0"])
            rep_by_type[t][k] = rep
            est = float(point[(chosen, k)].mean() - point[("s0", k)].mean())
            lo, hi = X.ci(rep, b["level"])
            rows.append({"model": model, "dataset": dataset, "selected": chosen, "name": t,
                         "group": groups[dataset][t]["group"], "role": roles.get(t, "primary"), "metric": k,
                         "s0_mean": float(point[("s0", k)].mean()), "selected_mean": float(point[(chosen, k)].mean()),
                         "delta": est, "ci_lo": lo, "ci_hi": hi,
                         "delta_by_seed_index": ";".join(f"{v:.4f}" for v in point[(chosen, k)] - point[("s0", k)])})
    for grp in ("stealthy", "coarse"):
        types = [t for t in sets if groups[dataset][t]["group"] == grp and roles.get(t, "primary") == "primary"]
        for k in BOOT_METRICS:
            sub = [r for r in rows if r["name"] in types and r["metric"] == k]
            rep = np.mean([rep_by_type[t][k] for t in types], axis=0)
            lo, hi = X.ci(rep, b["level"])
            rows.append({"model": model, "dataset": dataset, "selected": chosen, "name": f"{grp}_mean", "group": grp,
                         "role": "primary", "metric": k, "s0_mean": float(np.mean([r["s0_mean"] for r in sub])),
                         "selected_mean": float(np.mean([r["selected_mean"] for r in sub])),
                         "delta": float(np.mean([r["delta"] for r in sub])), "ci_lo": lo, "ci_hi": hi,
                         "delta_by_seed_index": ""})
    return rows


# --------------------------------------------------------------------------------------------
# Quantization of the selected sets (K-077)
# --------------------------------------------------------------------------------------------


def quant_table(selection: dict) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        chosen = selection[model]["feature_set"]
        sets = ["s0"] if chosen == "s0" else ["s0", chosen]
        for r in all_runs():
            if r["model"] != model or r["feature_set"] not in sets:
                continue
            f = T.RAW_DIR / r["run_id"] / "quant" / "metrics.csv"
            if not f.exists():
                raise SystemExit(f"missing {f} (ext_quant / full stage)")
            m = pd.read_csv(f)
            m = m[(m["split"] == "test") & m["level"].isin(["attack_type", "attack_free", "pooled"])].copy()
            m["feature_set"] = r["feature_set"]
            m["selected_set"] = chosen
            frames.append(m)
    q = pd.concat(frames, ignore_index=True)
    keys = ["model", "dataset", "feature_set", "selected_set", "threshold_mode", "level", "name", "group", "role"]
    cols = ["recall", "fpr", "f1", "roc_auc"]
    s = q.groupby(keys + ["quant"], dropna=False)[cols].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    base = q[q["quant"] == "q0"][keys + ["seed"] + cols]
    j = q.merge(base, on=keys + ["seed"], suffixes=("", "_q0"))
    for c in ("recall", "fpr"):
        j[f"d_{c}"] = j[f"{c}_q0"] - j[c]             # Q0 - Qk (K-045); for fpr negative = more false alarms
    d = j.groupby(keys + ["quant"], dropna=False)[["d_recall", "d_fpr"]].agg(["mean", "std", "min", "max"])
    d.columns = [f"{a}_{b}" for a, b in d.columns]
    return s.merge(d.reset_index(), on=keys + ["quant"]).sort_values(keys + ["quant"])


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    G.check_selection_committed()
    cfg = load_config()
    selection = G.load_selection()["selected"]
    groups = yaml.safe_load((S.PROJECT_ROOT / "configs" / "attack_groups.yaml").read_text(encoding="utf-8"))
    roles = {t.split("/", 1)[1]: "unseen" for t in S.load_config()["splits"]["analysis_roles"]["unseen_in_training"]}
    runs = all_runs()
    long = tidy(runs)
    summ = summarize(long)
    TABLES.mkdir(parents=True, exist_ok=True)
    long.to_csv(TABLES / "ext_ablation.csv", index=False, float_format="%.6g")
    summ.to_csv(TABLES / "ext_ablation_summary.csv", index=False, float_format="%.6g")
    rows = []
    for model in MODELS:
        chosen = selection[model]["feature_set"]
        if chosen != "s0":
            rows += contrast(model, chosen, "road", cfg, groups, roles)
    pd.DataFrame(rows).to_csv(TABLES / "ext_ablation_contrast.csv", index=False, float_format="%.6g")
    quant_table(selection).to_csv(TABLES / "ext_quant.csv", index=False, float_format="%.6g")
    print(f"{len(runs)} runs; wrote ext_ablation.csv ({len(long)} rows), ext_ablation_summary.csv, "
          f"ext_ablation_contrast.csv ({len(rows)} rows), ext_quant.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
