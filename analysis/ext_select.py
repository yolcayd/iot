"""Feature-set selection of the P10 extension: per model, on VALIDATION metrics only (docs/DECISIONS.md K-075).

Rule (configs/ext_analysis.yaml ``selection``), applied to every model on its own:

1. Gain G(S) = mean over the ROAD stealthy types that have validation windows of the 3-seed mean validation
   ROC-AUC of set S, minus the same quantity of the reference set s0.
2. Car-Hacking guard: on every Car-Hacking attack type, the 3-seed mean validation ROC-AUC of S is at least that of
   s0 minus ``ch_guard.roc_auc`` and the calibrated recall at least that of s0 minus ``ch_guard.recall``.
3. Among the sets that pass the guard, G* = max G. If G* < ``min_gain`` the reference s0 is kept; otherwise the
   selected set is the first one in ``candidate_order`` (the smallest) with G >= G* - ``near_best_tolerance``.

Only metrics_val.csv files are read. The script refuses to run if any extended-feature run already has a test
evaluation (metrics_test.csv / scores_test.parquet): the choice must be made, and committed, before any test result
of the new sets exists (K-074). Output: results/tables/ext_selection.json (inputs with sha256, every intermediate
value, git commit).

Usage:
    python -m analysis.ext_select
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import sources as S

CONFIG = S.PROJECT_ROOT / "configs" / "ext_analysis.yaml"
OUT = S.PROJECT_ROOT / "results" / "tables" / "ext_selection.json"
MODELS = ("cnn1d", "conv_ae")
TEST_FILES = ("metrics_test.csv", "scores_test.parquet", "pairs_test.csv")


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_no_test_results(run_dirs: list[Path]) -> None:
    found = [str(d / f) for d in run_dirs for f in TEST_FILES if (d / f).exists()]
    found += [str(d / "quant") for d in run_dirs if (d / "quant").exists()]
    if found:
        raise SystemExit(f"extended-feature runs already have test outputs; the selection must precede them (K-074): "
                         f"{found[:5]}")


def road_targets(cfg_sel: dict, val_types: set[str]) -> list[str]:
    groups = yaml.safe_load((S.PROJECT_ROOT / "configs" / "attack_groups.yaml").read_text(encoding="utf-8"))["road"]
    targets = sorted(t for t, g in groups.items() if g["group"] == cfg_sel["road_target_group"] and t in val_types)
    if targets != sorted(cfg_sel["expected_road_targets"]):
        raise SystemExit(f"ROAD target types {targets} differ from the pre-registered list "
                         f"{sorted(cfg_sel['expected_road_targets'])} (K-075)")
    return targets


def seed_means(val: pd.DataFrame) -> pd.DataFrame:
    """(model, dataset, feature_set, type) -> 3-seed mean of roc_auc and recall; refuses incomplete seed sets."""
    m = val[val["level"] == "attack_type"]
    g = m.groupby(["model", "dataset", "feature_set", "name"])
    out = g.agg(roc_auc=("roc_auc", "mean"), recall=("recall", "mean"), n_seeds=("seed", "nunique")).reset_index()
    if (out["n_seeds"] != 3).any():
        bad = out[out["n_seeds"] != 3][["model", "dataset", "feature_set", "name", "n_seeds"]]
        raise SystemExit(f"every cell needs 3 seeds (CLAUDE.md rule 5):\n{bad.to_string(index=False)}")
    return out


def select_model(means: pd.DataFrame, model: str, targets: list[str], sel: dict) -> dict:
    """The K-075 rule for one model; ``means`` from ``seed_means``."""
    ref = sel["reference_set"]
    m = means[means["model"] == model]
    road = m[(m["dataset"] == "road") & m["name"].isin(targets)]
    ch = m[m["dataset"] == "car_hacking"]
    if set(road["name"]) != set(targets):
        raise SystemExit(f"{model}: missing ROAD target types in validation metrics")

    def road_mean(fs):
        r = road[road["feature_set"] == fs]
        if len(r) != len(targets):
            raise SystemExit(f"{model} {fs}: {len(r)} of {len(targets)} ROAD target types")
        return float(r[sel["metric"]].mean())

    base_road = road_mean(ref)
    ch_ref = ch[ch["feature_set"] == ref].set_index("name")
    rows = []
    for fs in sel["candidate_order"]:
        c = ch[ch["feature_set"] == fs].set_index("name").reindex(ch_ref.index)
        d_auc = c["roc_auc"] - ch_ref["roc_auc"]
        d_rec = c["recall"] - ch_ref["recall"]
        guard = bool((d_auc >= -sel["ch_guard"]["roc_auc"] - 1e-12).all()
                     and (d_rec >= -sel["ch_guard"]["recall"] - 1e-12).all())
        rr = road_mean(fs)
        rows.append({"feature_set": fs, "road_target_mean": rr, "gain": rr - base_road, "ch_guard_passed": guard,
                     "ch_delta_roc_auc": {k: float(v) for k, v in d_auc.items()},
                     "ch_delta_recall": {k: float(v) for k, v in d_rec.items()}})
    adm = [r for r in rows if r["ch_guard_passed"]]
    g_star = max((r["gain"] for r in adm), default=None)
    if g_star is None or g_star < sel["min_gain"]:
        chosen = ref
        reason = ("no candidate passes the Car-Hacking guard" if g_star is None else
                  f"G* = {g_star:.4f} < min_gain {sel['min_gain']}: no meaningful validation gain, reference kept")
    else:
        chosen = next(r["feature_set"] for r in rows
                      if r["ch_guard_passed"] and r["gain"] >= g_star - sel["near_best_tolerance"] - 1e-12)
        reason = (f"smallest admissible set with G >= G* - {sel['near_best_tolerance']} "
                  f"(G* = {g_star:.4f})")
    return {"feature_set": chosen, "reason": reason, "road_target_mean_reference": base_road, "g_star": g_star,
            "candidates": rows,
            "ch_reference": {k: {"roc_auc": float(v["roc_auc"]), "recall": float(v["recall"])}
                             for k, v in ch_ref.iterrows()}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> tuple[pd.DataFrame, list[dict]]:
    from src import run_grid as G

    sel_cfg = load_config()["selection"]
    ext = [r for r in G.stage_runs("ext_ablation")]
    ref = [{**r, "feature_set": sel_cfg["reference_set"]} for r in G.stage_runs("full") if r["model"] in MODELS]
    check_no_test_results([G.T.RAW_DIR / r["run_id"] for r in ext])
    frames, inputs, missing = [], [], []
    for r in ref + ext:
        f = G.T.RAW_DIR / r["run_id"] / "metrics_val.csv"
        if not f.exists():
            missing.append(r["run_id"])
            continue
        m = pd.read_csv(f)
        m["feature_set"] = r["feature_set"]
        frames.append(m)
        inputs.append({"run_id": r["run_id"], "model": r["model"], "dataset": r["dataset"], "seed": r["seed"],
                       "feature_set": r["feature_set"], "file": f.relative_to(S.PROJECT_ROOT).as_posix(),
                       "sha256": _sha(f)})
    if missing:
        raise SystemExit(f"{len(missing)} runs without validation metrics: {missing[:5]}")
    return pd.concat(frames, ignore_index=True), inputs


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    from src import train as T

    sel_cfg = load_config()["selection"]
    val, inputs = collect()
    if set(val["split"]) != {"val"}:
        raise SystemExit("only validation metrics may enter the selection")
    targets = road_targets(sel_cfg, set(val.loc[(val["dataset"] == "road") & (val["level"] == "attack_type"), "name"]))
    means = seed_means(val)
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "decision": "K-075",
           "rule": sel_cfg, "road_targets": targets, "git": T.git_info(),
           "selected": {m: select_model(means, m, targets, sel_cfg) for m in MODELS}, "inputs": inputs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for m, s in out["selected"].items():
        print(f"{m}: selected {s['feature_set']} ({s['reason']})")
        for c in s["candidates"]:
            print(f"    {c['feature_set']}: G = {c['gain']:+.4f}, CH guard {'passed' if c['ch_guard_passed'] else 'FAILED'}")
    print(f"wrote {OUT.relative_to(S.PROJECT_ROOT)}; commit it before `python -m src.run_grid --stage ext_test`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
