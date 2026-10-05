"""Runs the experiment grid in configs/grid.yaml (P3: baseline stage; extended in F4).

Resumable: runs marked DONE are skipped and an interrupted deep-model run continues from its last epoch
checkpoint. A failing run is logged and the grid moves on. After every run the model is evaluated on the
validation and the test split; at the end results/tables/baseline_f1.csv and baseline_pairs.csv are written.

Usage:
    python -m src.run_grid --stage baseline [--threads N]
    python -m src.run_grid --stage baseline --status
    python -m src.run_grid --stage baseline --ae-check     # LSTM-AE feasibility on validation ROC-AUC (K-030)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime

import pandas as pd
import yaml

from src import evaluate as E
from src import train as T
from src.data import sources as S

GRID = S.PROJECT_ROOT / "configs" / "grid.yaml"


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def stage_runs(stage: str) -> list[tuple[str, str, int]]:
    g = yaml.safe_load(GRID.read_text(encoding="utf-8"))[stage]
    return [(m, d, s) for d in g["datasets"] for m in g["models"] for s in g["seeds"]]


def find_run(model: str, dataset: str, seed: int):
    """Most recent non-smoke run directory of (model, dataset, seed), or None."""
    hits = sorted(T.RAW_DIR.glob(f"{model}_{dataset}_s{seed}_*"), key=lambda p: p.stat().st_mtime)
    hits = [h for h in hits if not h.name.endswith("_smoke")]
    return hits[-1] if hits else None


def status(stage: str) -> pd.DataFrame:
    rows = []
    for m, d, s in stage_runs(stage):
        r = find_run(m, d, s)
        state = "missing" if r is None else "done" if (r / "DONE").exists() else \
            "interrupted" if (r / "checkpoint.pt").exists() else "started"
        rows.append({"model": m, "dataset": d, "seed": s, "state": state, "run": r.name if r else "",
                     "evaluated": bool(r and (r / "metrics_test.csv").exists())})
    return pd.DataFrame(rows)


def ae_check(min_auc: float) -> pd.DataFrame:
    """Validation ROC-AUC (primary pool) of every finished autoencoder run; never uses test data."""
    rows = []
    for d in sorted(T.RAW_DIR.glob("*_ae_*")):
        f = d / "metrics_val.csv"
        if f.exists():
            m = pd.read_csv(f)
            auc = float(m.loc[(m["level"] == "pooled"), "roc_auc"].iloc[0])
            rows.append({"run": d.name, "val_roc_auc": auc, "passes": auc >= min_auc})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="baseline")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--ae-check", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        print(status(a.stage).to_string(index=False))
        return 0
    if a.ae_check:
        print(ae_check(T.load_models_config()["ae_feasibility"]["min_val_roc_auc"]).to_string(index=False))
        return 0
    failed = []
    for m, d, s in stage_runs(a.stage):
        log(f"=== {m} / {d} / seed {s}")
        try:
            run_dir = T.train(m, d, s, threads=a.threads, log=log)
            for split in ("val", "test"):
                if not (run_dir / f"metrics_{split}.csv").exists():
                    E.evaluate_run(run_dir, split, log=log)
        except Exception:
            log(f"FAILED {m} / {d} / seed {s}\n{traceback.format_exc()}")
            failed.append((m, d, s))
    try:
        E.baseline_table()
    except RuntimeError as e:
        log(str(e))
    log(f"finished; {len(failed)} failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
