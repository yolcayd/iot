"""Runs the experiment grid in configs/grid.yaml (P3: baseline and ae_feasibility stages; extended in F4).

Every run: train -> calibrate the decision threshold on validation normals (inside ``src.train``, K-035) ->
evaluate the stage's ``evaluate_splits``. Resumable: runs marked DONE are skipped and an interrupted
deep-model run continues from its last epoch checkpoint. A failing run is logged and the grid moves on.
Only the run ids that the CURRENT configuration plans are used (older runs under results/raw are ignored).

Usage:
    python -m src.run_grid --stage baseline ae_feasibility [--threads N]
    python -m src.run_grid --stage baseline ae_feasibility --status
    python -m src.run_grid --ae-check     # autoencoder protocol on validation ROC-AUC (K-030, K-033)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src import evaluate as E
from src import train as T
from src.data import sources as S

GRID = S.PROJECT_ROOT / "configs" / "grid.yaml"
AE_TABLE = S.PROJECT_ROOT / "results" / "tables" / "ae_feasibility.csv"
# K-033 candidate order: (name, model, overrides reference)
AE_CANDIDATES = [("baseline", "lstm_ae", None), ("retry", "lstm_ae", "ae_feasibility.retry"), ("conv_ae", "conv_ae", None)]
DATASETS = ("car_hacking", "road")


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def load_grid() -> dict:
    return yaml.safe_load(GRID.read_text(encoding="utf-8"))


def stage_runs(stage: str) -> list[dict]:
    g = load_grid()[stage]
    mcfg = T.load_models_config()
    entries = g.get("runs") or [{"model": m} for m in g["models"]]
    out = []
    for d in g["datasets"]:
        for e in entries:
            for s in g["seeds"]:
                ov = T.resolve_overrides(e.get("overrides"), mcfg)
                out.append({"stage": stage, "model": e["model"], "dataset": d, "seed": s, "overrides": ov,
                            "overrides_ref": e.get("overrides") or "",
                            "run_id": T.planned_run_id(e["model"], d, s, ov),
                            "evaluate_splits": g.get("evaluate_splits", ["val", "test"])})
    return out


def planned_dirs(stage: str) -> list[Path]:
    return [T.RAW_DIR / r["run_id"] for r in stage_runs(stage)]


def status(stages: list[str]) -> pd.DataFrame:
    rows = []
    for st in stages:
        for r in stage_runs(st):
            d = T.RAW_DIR / r["run_id"]
            state = "done" if (d / "DONE").exists() else "interrupted" if (d / "checkpoint.pt").exists() else \
                "started" if d.exists() else "missing"
            rows.append({"stage": st, "model": r["model"], "overrides": r["overrides_ref"], "dataset": r["dataset"],
                         "seed": r["seed"], "state": state, "run": r["run_id"],
                         "evaluated": ",".join(s for s in r["evaluate_splits"] if (d / f"metrics_{s}.csv").exists())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Autoencoder feasibility protocol (K-033)
# --------------------------------------------------------------------------------------------


def val_auc(run_dir: Path) -> float | None:
    f = run_dir / "metrics_val.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    return float(m.loc[m["level"] == "pooled", "roc_auc"].iloc[0])


def ae_decision(auc: dict[tuple[str, str], float | None], min_auc: float) -> dict:
    """Inspection order and selection of K-033 from validation ROC-AUC values (None = not available yet).

    candidate 2 is inspected only if candidate 1 fails on a dataset, candidate 3 only if candidate 2 fails on a
    dataset. Selected = the first inspected candidate that passes on BOTH datasets; if there is none, per dataset
    the first inspected candidate that passes on that dataset; a dataset with no passing candidate is excluded
    (exit rule: the autoencoder leaves the ROAD analyses, AS-3 on Car-Hacking only).
    """
    names = [c[0] for c in AE_CANDIDATES]

    def passes(c, d):
        return auc.get((c, d)) is not None and auc[(c, d)] >= min_auc

    inspected = []
    for c in names:
        inspected.append(c)
        if any(auc.get((c, d)) is None for d in DATASETS):
            return {"inspected": inspected, "pending": [c], "selected": None}
        if all(passes(c, d) for d in DATASETS):
            break                                  # later candidates are not inspected
    both = [c for c in inspected if all(passes(c, d) for d in DATASETS)]
    if both:
        sel = {d: both[0] for d in DATASETS}
    else:                                          # all three inspected
        sel = {d: next((c for c in inspected if passes(c, d)), None) for d in DATASETS}
    return {"inspected": inspected, "pending": [], "selected": sel}


def ae_check(min_auc: float) -> tuple[pd.DataFrame, dict]:
    mcfg = T.load_models_config()
    auc, rows = {}, {}
    for name, model, ref in AE_CANDIDATES:
        for d in DATASETS:
            rid = T.planned_run_id(model, d, 0, T.resolve_overrides(ref, mcfg))
            auc[(name, d)] = val_auc(T.RAW_DIR / rid)
            rows[(name, d)] = rid
    dec = ae_decision(auc, min_auc)
    shown = [{"candidate": c, "dataset": d, "run": rows[(c, d)], "val_roc_auc": auc[(c, d)],
              "passes": None if auc[(c, d)] is None else auc[(c, d)] >= min_auc}
             for c in dec["inspected"] for d in DATASETS]       # uninspected candidates are not shown (K-033)
    return pd.DataFrame(shown), dec


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", nargs="+", default=["baseline"])
    ap.add_argument("--threads", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--ae-check", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        print(status(a.stage).to_string(index=False))
        return 0
    if a.ae_check:
        t, dec = ae_check(T.load_models_config()["ae_feasibility"]["min_val_roc_auc"])
        print(t.to_string(index=False))
        print(json.dumps(dec, indent=2))
        AE_TABLE.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(AE_TABLE, index=False, float_format="%.4f")
        return 0
    failed = []
    for st in a.stage:
        for r in stage_runs(st):
            log(f"=== [{st}] {r['model']} {r['overrides_ref']} / {r['dataset']} / seed {r['seed']} ({r['run_id']})")
            try:
                run_dir = T.train(r["model"], r["dataset"], r["seed"], threads=a.threads, overrides=r["overrides"], log=log)
                if run_dir.name != r["run_id"]:
                    raise RuntimeError(f"run id mismatch: planned {r['run_id']}, trained {run_dir.name}")
                for split in r["evaluate_splits"]:
                    if not (run_dir / f"metrics_{split}.csv").exists():
                        E.evaluate_run(run_dir, split, log=log)
            except Exception:
                log(f"FAILED {r['model']} / {r['dataset']} / seed {r['seed']}\n{traceback.format_exc()}")
                failed.append((st, r["model"], r["dataset"], r["seed"]))
        if st == "baseline":
            try:
                E.baseline_table()
            except RuntimeError as e:
                log(str(e))
    log(f"finished; {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
