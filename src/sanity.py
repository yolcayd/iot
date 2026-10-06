"""Sanity checks on the baseline results (P3): leakage suspicion for near-perfect F1, K-026 counter.

For every (model, dataset, attack type) with test F1 >= 0.995:

1. the leakage tests (tests/test_no_leakage.py) are run again;
2. the file origins of the training and test windows of that type are listed (capture, segment row
   range, split, window counts);
3. a one-feature decision stump is fitted on the TRAINING windows of the type's captures (window summary
   features, src/models/classical.py) and applied to the test windows of the type's captures. If a single
   traffic statistic already reaches F1 >= ``stump_f1``, the high F1 is explained by an obvious traffic
   signature (e.g. DoS: unseen ID 0x000), not by leakage.

K-026: for Car-Hacking fuzzy, the number of false-alarm (normal) test windows that contain the
normal-labelled diagnostic response IDs 0x7E8/0x7E9.

Usage:
    python -m src.sanity          # results/tables/baseline_sanity.csv, baseline_origins.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src import calibration as C
from src import train as T
from src.data import dataset as D
from src.data import features as F
from src.data import sources as S

TABLES = S.PROJECT_ROOT / "results" / "tables"
HIGH_F1 = 0.995
DIAG_IDS = (0x7E8, 0x7E9)


def high_f1(table: pd.DataFrame, threshold: float = HIGH_F1) -> pd.DataFrame:
    t = table[(table["level"] == "attack_type") & (table["f1"] >= threshold)]
    return t[["dataset", "model", "run_id", "name", "group", "role", "f1"]].reset_index(drop=True)


def type_captures(win: pd.DataFrame, dataset: str, attack_type: str) -> set[str]:
    w = win[(win["dataset"] == dataset) & (win["attack_type"].astype(str) == attack_type)]
    return set(w["capture"].astype(str))


def origins(win: pd.DataFrame, dataset: str, attack_type: str) -> pd.DataFrame:
    """Every segment that holds windows of the type's captures, per split."""
    caps = type_captures(win, dataset, attack_type)
    w = win[(win["dataset"] == dataset) & win["capture"].astype(str).isin(caps)]
    g = w.groupby([w["split"].astype(str), w["capture"].astype(str), w["segment"].astype(str)], observed=True)
    out = g.agg(n_windows=("start", "size"), n_attack=("label", "sum"),
                first_row=("start", "min"), last_row=("start", "max")).reset_index()
    out.columns = ["split", "capture", "segment", "n_windows", "n_attack", "first_row", "last_row"]
    out.insert(0, "attack_type", attack_type)
    out.insert(0, "dataset", dataset)
    return out


def type_subset(win: pd.DataFrame, dataset: str, attack_type: str, split: str) -> pd.DataFrame:
    caps = type_captures(win, dataset, attack_type)
    w = win[(win["dataset"] == dataset) & (win["split"] == split) & win["capture"].astype(str).isin(caps)]
    return w[(w["label"] == 0) | (w["attack_type"].astype(str) == attack_type)].reset_index(drop=True)


def stump_check(store: D.FeatureStore, win: pd.DataFrame, dataset: str, attack_type: str, size: int) -> dict:
    from sklearn.tree import DecisionTreeClassifier

    from src.evaluate import binary_metrics
    from src.models.classical import SUMMARY_NAMES

    tr, te = type_subset(win, dataset, attack_type, "train"), type_subset(win, dataset, attack_type, "test")
    if tr["label"].nunique() < 2:
        return {"stump_feature": None, "stump_f1": np.nan, "note": "type not in training"}
    stump = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=0)
    stump.fit(T.summary_matrix(store, tr, size), tr["label"].to_numpy())
    pred = stump.predict(T.summary_matrix(store, te, size))
    m = binary_metrics(te["label"].to_numpy(), pred)
    return {"stump_feature": SUMMARY_NAMES[int(stump.tree_.feature[0])], "stump_threshold": float(stump.tree_.threshold[0]),
            "stump_f1": m["f1"], "stump_recall": m["recall"], "stump_fpr": m["fpr"]}


def id_codes(ids, vocab) -> np.ndarray:
    return F.encode_ids(np.asarray(ids), np.asarray(vocab)).astype(np.float32)


def diag_false_alarms(run_dir, store: D.FeatureStore, params: dict, size: int) -> dict:
    """K-026: Car-Hacking fuzzy false alarms that contain the diagnostic response IDs."""
    s = pd.read_parquet(run_dir / "scores_test.parquet")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    alarm = C.predict(s["score"].to_numpy(), meta["decision"])
    fuzzy_segs = [k for k, seg in store.segments.items() if seg.attack_type == "fuzzy" and seg.split == "test"]
    fa = s[alarm & (s["label"] == 0) & s["segment"].isin(fuzzy_segs)]
    codes = id_codes(DIAG_IDS, params["datasets"]["car_hacking"]["id_vocab"])
    x = store.windows(store.global_starts(fa), size)[:, :, 0] if len(fa) else np.empty((0, size))
    with_diag = np.isin(x, codes).any(axis=1)
    return {"fuzzy_false_alarms": int(len(fa)), "with_diag_ids": int(with_diag.sum())}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    table = pd.read_csv(TABLES / "baseline_f1.csv")
    flagged = high_f1(table)
    cfg = S.load_config()
    size = cfg["windows"]["size"]
    win = D.load_window_index()
    rows, orig = [], []
    if len(flagged):
        res = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_no_leakage.py"], cwd=S.PROJECT_ROOT,
                             capture_output=True, text=True)
        leak_summary = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-200:]
        print(f"test_no_leakage: {leak_summary}")
        for dataset in sorted(flagged["dataset"].unique()):
            store = D.FeatureStore.build(dataset, ("train", "test"), cfg)
            for t in sorted(flagged.loc[flagged["dataset"] == dataset, "name"].unique()):
                orig.append(origins(win, dataset, t))
                st = stump_check(store, win, dataset, t, size)
                for r in flagged[(flagged["dataset"] == dataset) & (flagged["name"] == t)].itertuples():
                    rows.append({"dataset": dataset, "attack_type": t, "model": r.model, "group": r.group,
                                 "role": r.role, "f1": r.f1, "leakage_tests": leak_summary, **st})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "baseline_sanity.csv", index=False, float_format="%.4f")
    if orig:
        pd.concat(orig, ignore_index=True).to_csv(TABLES / "baseline_origins.csv", index=False)
    print(out.to_string(index=False) if len(out) else "no attack type with F1 >= 0.995")
    params = F.load_params()
    store = D.FeatureStore.build("car_hacking", ("test",), cfg)
    for run in [T.RAW_DIR / r for r in sorted(set(table["run_id"]))]:   # every run in baseline_f1.csv
        if not (run / "scores_test.parquet").exists():
            continue
        if json.loads((run / "run_config.json").read_text(encoding="utf-8"))["dataset"] == "car_hacking":
            print(f"K-026 {run.name}: {diag_false_alarms(run, store, params, size)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
