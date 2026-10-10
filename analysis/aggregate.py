"""Collect results/raw into tidy tables for the P6 analysis (plan F5; docs/DECISIONS.md K-057 ... K-062).

Reads only results/raw (nothing is trained, quantized or re-scored). The runs are the run ids that
configs/grid.yaml plans for the analysed stage (K-053: 18 runs); older runs under results/raw are ignored.

Outputs (results/tables/):

* ``all_results.csv`` -- tidy: run_id, model, dataset, seed, quantization, attack_type, attack_group, metric,
  value, plus the keys split, threshold_mode, level and role that make a row unique. Levels: ``attack_type``,
  ``pooled`` (primary pool), ``attack_free`` (all attack-free captures) and ``model`` (file size; seed-0 host
  latency and peak memory of both benchmark sessions, K-049/K-050/K-052). Quantization ``native`` and threshold
  mode ``calibrated`` mark the unquantized reference model (XGBoost, K-042).
* ``delta_f1.csv`` -- Q0 - Qk (positive = loss, K-037) of F1 and the other metrics per (model, dataset, seed,
  attack type, k, threshold mode) on the test split; Q0 is the Q0-ONNX evaluation (K-045).
* ``missing_cells.csv`` -- every expected cell that is absent or NaN with its structural reason. A missing cell
  without a structural reason makes the command fail.
* ``analysis_manifest.json`` -- provenance (CLAUDE.md rule 7): git commit, versions, configuration and the
  sha256 of every input file.

Usage:
    python -m analysis.aggregate
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import benchmark as B
from src import run_grid as G
from src import train as T
from src.data import sources as S
from src.data import splits as SP

ANALYSIS_CONFIG = S.PROJECT_ROOT / "configs" / "analysis.yaml"
TABLES = S.PROJECT_ROOT / "results" / "tables"
LEVELS = ("attack_type", "pooled", "attack_free")
ATTACK_METRICS = ("n_attack", "n_normal", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "fpr", "roc_auc",
                  "average_precision", "roc_tpr_at_fpr_0.01")
FREE_METRICS = ("n_normal", "fp", "tn", "fpr")
DELTA_METRICS = ("f1", "recall", "precision", "fpr", "roc_auc", "average_precision", "roc_tpr_at_fpr_0.01")
QUANT_LEVELS = ("q0", "q1", "q2")
NATIVE = "native"
CALIBRATED = "calibrated"          # threshold mode of the unquantized reference model (K-035/K-038)
NA = "na"
SESSION_METRICS = ("latency_median_ms", "latency_p95_ms", "peak_rss_mb")
VERSIONED = ["numpy", "pandas", "scipy", "pyarrow", "PyYAML"]


def load_analysis_config(path: Path = ANALYSIS_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    """Records every file read (sha256 goes into the manifest)."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def csv(self, path: Path) -> pd.DataFrame:
        self._add(path)
        return pd.read_csv(path)

    def json(self, path: Path) -> dict:
        self._add(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def _add(self, path: Path) -> None:
        path = Path(path)
        try:
            key = path.resolve().relative_to(S.PROJECT_ROOT).as_posix()
        except ValueError:
            key = str(path)
        self.files[key] = sha256(path)


# --------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------


def grid_runs(cfg: dict) -> list[dict]:
    runs = G.stage_runs(cfg["stage"])
    missing = [r["run_id"] for r in runs if G.pending_steps(r)]
    if missing:
        raise SystemExit(f"grid not complete, missing: {missing}")
    return runs


def run_metrics(r: dict, raw: Path, inp: Inputs) -> pd.DataFrame:
    """Metric rows (levels LEVELS, val and test) of one run in a common layout."""
    d = raw / r["run_id"]
    if r["quantize"]:
        m = inp.csv(d / "quant" / "metrics.csv").rename(columns={"quant": "quantization"})
    else:
        m = pd.concat([inp.csv(d / f"metrics_{s}.csv") for s in r["evaluate_splits"]], ignore_index=True)
        m["quantization"] = NATIVE
        m["threshold_mode"] = CALIBRATED
    return m[m["level"].isin(LEVELS)].reset_index(drop=True)


def tidy_metrics(m: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_id", "model", "dataset", "seed", "quantization", "threshold_mode", "split", "level", "name", "group",
            "role"]
    out = []
    for level, metrics in (("attack_type", ATTACK_METRICS), ("pooled", ATTACK_METRICS), ("attack_free", FREE_METRICS)):
        part = m[m["level"] == level]
        if len(part):
            out.append(part[keys + list(metrics)].melt(id_vars=keys, var_name="metric", value_name="value"))
    t = pd.concat(out, ignore_index=True).rename(columns={"name": "attack_type", "group": "attack_group"})
    t.loc[t["level"] == "pooled", "attack_group"] = "all"
    t.loc[t["level"] == "attack_free", "attack_group"] = "none"
    return t


def resource_rows(r: dict, raw: Path, inp: Inputs, cfg: dict) -> list[dict]:
    """File size of every version (all seeds); host latency and peak memory of both sessions (seed-0 runs only,
    K-052; Q1 is not measured, K-043). Results of another protocol or marked invalid are skipped (K-048/K-049)."""
    d = raw / r["run_id"]
    base = {"run_id": r["run_id"], "model": r["model"], "dataset": r["dataset"], "seed": r["seed"],
            "threshold_mode": NA, "split": NA, "level": "model", "attack_type": "model", "attack_group": "none",
            "role": NA}
    rows = []
    if r["quantize"]:
        meta = inp.json(d / "quant" / "quant_meta.json")
        sizes = {q: meta["levels"][q]["size_bytes"] for q in QUANT_LEVELS}
        bench_dir = d / "quant"
    else:
        model_file = next(p for p in sorted(d.glob("model.*")) if p.suffix in (".ubj", ".joblib"))
        sizes = {NATIVE: model_file.stat().st_size}
        bench_dir = d
    protocol = B.current_protocol()
    profile = cfg["pareto"]["profile"]
    for q, size in sizes.items():
        rows.append({**base, "quantization": q, "metric": "size_mb", "value": size / 2**20})
        for s in cfg["pareto"]["sessions"]:
            p = B.result_path(bench_dir, profile, s, q)
            res = B.load_result(bench_dir, profile, s, q, protocol) if p.exists() else None
            if res is not None:
                inp._add(p)
                if res["model_size_bytes"] != size:
                    raise SystemExit(f"{p}: benchmarked file size {res['model_size_bytes']} != {size}")
            vals = {"latency_median_ms": res["latency"]["median_ms"], "latency_p95_ms": res["latency"]["p95_ms"],
                    "peak_rss_mb": res["memory"]["peak_rss_mb"]} if res else {}
            for k in SESSION_METRICS:
                rows.append({**base, "quantization": q, "metric": f"{k}_session{s}", "value": vals.get(k, np.nan)})
    return rows


def collect(cfg: dict, raw: Path = T.RAW_DIR, runs: list[dict] | None = None,
            inp: Inputs | None = None) -> pd.DataFrame:
    inp = inp or Inputs()
    runs = runs if runs is not None else grid_runs(cfg)
    parts, res = [], []
    for r in runs:
        parts.append(tidy_metrics(run_metrics(r, raw, inp)))
        res.extend(resource_rows(r, raw, inp, cfg))
    t = pd.concat(parts + [pd.DataFrame(res)], ignore_index=True)
    cols = ["run_id", "model", "dataset", "seed", "quantization", "attack_type", "attack_group", "metric", "value",
            "split", "threshold_mode", "level", "role"]
    t = t[cols].sort_values(["dataset", "model", "seed", "quantization", "threshold_mode", "split", "level",
                             "attack_type", "metric"], kind="stable").reset_index(drop=True)
    dup = t.duplicated(["run_id", "quantization", "threshold_mode", "split", "level", "attack_type", "metric"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate cells in all_results")
    return t


# --------------------------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------------------------


def expected_cells(runs: list[dict], groups: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for r in runs:
        base = {"run_id": r["run_id"], "model": r["model"], "dataset": r["dataset"], "seed": r["seed"]}
        quants = QUANT_LEVELS if r["quantize"] else (NATIVE,)
        modes = ("recalibrated", "q0_threshold") if r["quantize"] else (CALIBRATED,)
        for q in quants:
            for mode in modes:
                for split in r["evaluate_splits"]:
                    # every type on both splits; absent val types are classified by structural_reason (K-018/K-019)
                    cells = [("attack_type", t, ATTACK_METRICS) for t in sorted(groups[r["dataset"]])]
                    cells += [("pooled", "primary_pool", ATTACK_METRICS), ("attack_free", "ambient", FREE_METRICS)]
                    for level, name, metrics in cells:
                        rows += [{**base, "quantization": q, "threshold_mode": mode, "split": split, "level": level,
                                  "attack_type": name, "metric": k} for k in metrics]
            rows.append({**base, "quantization": q, "threshold_mode": NA, "split": NA, "level": "model",
                         "attack_type": "model", "metric": "size_mb"})
            rows += [{**base, "quantization": q, "threshold_mode": NA, "split": NA, "level": "model",
                      "attack_type": "model", "metric": f"{k}_session{s}"}
                     for k in SESSION_METRICS for s in cfg["pareto"]["sessions"]]
        if not r["quantize"]:
            rows += [{**base, "quantization": q, "threshold_mode": "all", "split": "all", "level": "all",
                      "attack_type": "all", "metric": "all"} for q in ("q1", "q2")]
    return pd.DataFrame(rows)


def structural_reason(cell: dict, no_val: set[str], flagged: float | None = None) -> tuple[str, str] | None:
    """(reason, decision) of a cell that is missing by design; None if the absence is unexplained. ``flagged`` is
    tp + fp of the cell's evaluation set (precision is undefined when no window is flagged)."""
    if cell["metric"] == "precision" and cell["status"] == "nan" and flagged == 0:
        return "precision undefined: no window flagged as attack (tp + fp = 0)", "K-029 (src/evaluate.binary_metrics)"
    if cell["model"] in ("xgb", "rf") and cell["quantization"] in ("q1", "q2"):
        return "reference model is not quantized", "K-042"
    if cell["level"] == "model" and cell["metric"] != "size_mb":
        if cell["quantization"] == "q1":
            return "no fp16 kernels in the CPU provider; Q1 latency/memory not measured", "K-043"
        if cell["seed"] != 0:
            return "latency/memory measured on seed 0 only", "K-052"
    if cell["split"] == "val" and f"{cell['dataset']}/{cell['attack_type']}" in no_val:
        return "attack type has no validation windows", "K-018, K-019"
    return None


def missing_cells(t: pd.DataFrame, runs: list[dict], groups: dict, cfg: dict, no_val: set[str]) -> pd.DataFrame:
    keys = ["run_id", "quantization", "threshold_mode", "split", "level", "attack_type", "metric"]
    exp = expected_cells(runs, groups, cfg)
    j = exp.merge(t[keys + ["value"]], on=keys, how="left", indicator=True)
    j["status"] = np.where(j["_merge"] == "left_only", "absent", np.where(j["value"].isna(), "nan", "ok"))
    miss = j[j["status"] != "ok"].drop(columns=["_merge", "value"]).copy()
    cell = keys[:-1]
    pos = t[t["metric"].isin(["tp", "fp"])].groupby(cell)["value"].sum()
    flagged = [pos.get(tuple(c[k] for k in cell)) for c in miss.to_dict("records")]
    reasons = [structural_reason(c, no_val, f) for c, f in zip(miss.to_dict("records"), flagged)]
    miss["reason"] = [r[0] if r else "UNEXPLAINED" for r in reasons]
    miss["decision"] = [r[1] if r else "" for r in reasons]
    extra = t.merge(exp[keys], on=keys, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"]
    if len(extra):
        raise SystemExit(f"{len(extra)} cells in all_results outside the expected grid, e.g. "
                         f"{extra[keys].head(3).to_dict('records')}")
    return miss.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Deltas
# --------------------------------------------------------------------------------------------


def delta_table(t: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Q0 - Qk for every quantized run (both threshold modes); Q0 is the same version in both modes."""
    m = t[(t["quantization"].isin(QUANT_LEVELS)) & (t["split"] == split) & (t["level"].isin(LEVELS))
          & (t["metric"].isin(DELTA_METRICS))]
    keys = ["run_id", "threshold_mode", "level", "attack_type", "metric"]
    q0 = m[m["quantization"] == "q0"].set_index(keys)["value"].rename("q0_value")
    qk = m[m["quantization"] != "q0"].join(q0, on=keys).rename(columns={"value": "qk_value"})
    qk["delta"] = qk["q0_value"] - qk["qk_value"]
    cols = ["model", "dataset", "seed", "run_id", "attack_type", "attack_group", "role", "level", "threshold_mode",
            "quantization", "metric", "q0_value", "qk_value", "delta"]
    return qk[cols].sort_values(["dataset", "model", "threshold_mode", "level", "attack_type", "metric",
                                 "quantization", "seed"], kind="stable").reset_index(drop=True)


def cross_check_grid(delta: pd.DataFrame, grid_quant: pd.DataFrame) -> int:
    """Deltas must equal the d_* columns of results/tables/grid_quant.csv (P5 table, written with %.6g)."""
    g = grid_quant[grid_quant["split"] == "test"].rename(columns={"quant": "quantization", "name": "attack_type"})
    n = 0
    for k in ("recall", "f1", "roc_auc", "average_precision", "roc_tpr_at_fpr_0.01"):
        dk = delta[delta["metric"] == k].merge(
            g[["run_id", "threshold_mode", "level", "attack_type", "quantization", f"d_{k}"]],
            on=["run_id", "threshold_mode", "level", "attack_type", "quantization"], how="inner")
        ok = dk["delta"].isna() & dk[f"d_{k}"].isna() | np.isclose(dk["delta"], dk[f"d_{k}"], rtol=1e-5, atol=1e-6)
        if not ok.all():
            raise SystemExit(f"delta of {k} differs from grid_quant.csv in {int((~ok).sum())} rows")
        n += len(dk)
    return n


# --------------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------------


def versions(packages=VERSIONED) -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def write_manifest(name: str, cfg: dict, inputs: dict[str, str], outputs: list[Path], extra: dict | None = None,
                   path: Path | None = None) -> Path:
    """Merge one step's provenance into results/tables/analysis_manifest.json (rule 7)."""
    path = path or TABLES / "analysis_manifest.json"
    man = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    man[name] = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": T.git_info(),
                 "versions": versions(), "config": cfg, "n_inputs": len(inputs), "inputs_sha256": inputs,
                 "outputs_sha256": {p.relative_to(S.PROJECT_ROOT).as_posix(): sha256(p) for p in outputs},
                 **(extra or {})}
    path.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    cfg = load_analysis_config()
    groups = yaml.safe_load(SP.GROUPS_PATH.read_text(encoding="utf-8"))
    no_val = set(S.load_config()["splits"]["analysis_roles"]["no_validation_windows"])
    runs = grid_runs(cfg)
    inp = Inputs()
    t = collect(cfg, runs=runs, inp=inp)
    miss = missing_cells(t, runs, groups, cfg, no_val)
    delta = delta_table(t, cfg["split"])
    n_checked = cross_check_grid(delta, pd.read_csv(TABLES / "grid_quant.csv"))
    TABLES.mkdir(parents=True, exist_ok=True)
    outs = {"all_results.csv": t, "delta_f1.csv": delta, "missing_cells.csv": miss}
    for name, df in outs.items():
        df.to_csv(TABLES / name, index=False, float_format="%.10g")
    unexplained = miss[miss["reason"] == "UNEXPLAINED"]
    summary = (miss.groupby(["reason", "decision", "status"]).size().rename("cells").reset_index()
               if len(miss) else pd.DataFrame())
    write_manifest("aggregate", cfg, inp.files, [TABLES / n for n in outs],
                   {"runs": [r["run_id"] for r in runs], "rows_all_results": len(t), "rows_delta": len(delta),
                    "missing_by_reason": summary.to_dict("records"), "unexplained_missing": len(unexplained),
                    "delta_rows_checked_against_grid_quant": n_checked})
    print(f"all_results.csv: {len(t)} rows from {len(runs)} runs ({len(inp.files)} input files)")
    print(f"delta_f1.csv: {len(delta)} rows; {n_checked} checked against grid_quant.csv")
    print("missing cells by reason:")
    print(summary.to_string(index=False) if len(summary) else "  none")
    if len(unexplained):
        print(f"UNEXPLAINED missing cells: {len(unexplained)}")
        print(unexplained.head(20).to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
