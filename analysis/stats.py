"""Statistical analysis of the grid for RQ1-RQ3 (plan F5; docs/DECISIONS.md K-057 ... K-062).

Reads results/raw (window-level test scores and the validation-calibrated thresholds of every quantized version)
and the tables of ``analysis.aggregate``; nothing is trained or re-scored, no threshold is re-fitted (rule 3).

* Hierarchical bootstrap (K-058): seeds x windows, crossed. Every replicate draws the 3 seeds with replacement
  (independently per model) and one common resample of the test windows (moving blocks of ceil(n^(1/3)) windows
  inside every (segment, label) stratum, in time order), shared by every seed, quantization level and model, so
  that Q0-Qk, fabrication-masquerade and CNN-AE contrasts stay paired. The replicate value is the mean over the
  drawn seeds of the statistic on the drawn windows. Point estimates are 3-seed means on all windows; they are
  checked against metrics.csv before any interval is computed.
* AS-1 (K-057, exploratory): per (model, fabrication-masquerade pair) and k: dR_fab, dR_masq and
  D = dR_masq - dR_fab (D > 0: more loss on the stealthy side); mean of D over the primary units. No test
  (Wilcoxon cannot reach p < 0.25 with 3 units). iid windows (L = 1) as sensitivity.
* Group comparison (K-059, secondary): ROAD coarse vs stealthy types, exact Mann-Whitney U, Cliff's delta with
  bootstrap interval, Holm per family, minimum attainable p.
* AS-2 (K-060): Pareto front over latency (two host sessions, K-050), size and stealthy-attack recall; budgets.
* AS-3 (K-061): CNN vs conv-AE, d_t = |dAE_t| - |dCNN_t| on dROC-AUC (all primary types) and dRecall (types above
  the floor in both models); Wilcoxon + Holm and Cliff's delta secondary. Calibration shift (K-036 b) apart.
* Hypothesis verdicts (K-062).

Outputs (results/tables/): stats.csv (long format), as1_pairs.csv, per_type_delta.csv, pareto.csv,
calibration_shift.csv; provenance in analysis_manifest.json. Bootstrap replicates are cached per stratum set
under data/processed/analysis_cache/ (an interrupted run resumes; ``--force`` recomputes).

Usage:
    python -m analysis.stats [--n-boot B] [--threads T] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

from analysis import aggregate as A
from src import train as T
from src.data import sources as S

CACHE = S.PROJECT_ROOT / "data" / "processed" / "analysis_cache"
QK = ("q1", "q2")
BOOT_METRICS = ("recall", "f1", "roc_auc")


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------
# Small statistics
# --------------------------------------------------------------------------------------------


def cliffs_delta(x, y) -> float:
    """P(X > Y) - P(X < Y) over all pairs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sign(x[:, None] - y[None, :]).mean())


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p values (NaN entries are left out of the family)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


def exact_mwu(x, y) -> tuple[float, float]:
    """Mann-Whitney U of x (ties count 1/2) and its exact two-sided permutation p value (all relabelings)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    z = np.concatenate([x, y])
    n1 = len(x)

    def u(ix):
        a = z[list(ix)]
        b = np.delete(z, list(ix))
        return float((np.sign(a[:, None] - b[None, :]) + 1).sum() / 2)

    u_obs = u(range(n1))
    centre = n1 * len(y) / 2
    us = np.array([u(ix) for ix in itertools.combinations(range(len(z)), n1)])
    p = float((np.abs(us - centre) >= abs(u_obs - centre) - 1e-12).mean())
    return u_obs, p


def min_p_mwu(n1: int, n2: int) -> float:
    """Smallest two-sided p value an exact Mann-Whitney test can reach (no ties)."""
    return 2 / math.comb(n1 + n2, n1)


def min_p_wilcoxon(n: int) -> float:
    """Smallest two-sided p value an exact Wilcoxon signed-rank test can reach with n non-zero differences."""
    return 2 / 2**n


def ci(rep: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(rep, [a, 1 - a])
    # rounded: replicate means of exact zeros can leave +-1e-17, which must not count as excluding 0 (K-062)
    return round(float(lo), 12), round(float(hi), 12)


# --------------------------------------------------------------------------------------------
# Bootstrap machinery
# --------------------------------------------------------------------------------------------


def block_length(n: int, rule) -> int:
    if rule == "cube_root":
        return max(1, math.ceil(n ** (1 / 3) - 1e-9))
    return int(rule)


def mbb_counts(n: int, L: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap of n time-ordered items: (n_boot, n) counts, each row summing to n."""
    L = min(L, n)
    k = math.ceil(n / L)
    starts = rng.integers(0, n - L + 1, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(L)).reshape(n_boot, -1)[:, :n]
    flat = idx + (np.arange(n_boot) * n)[:, None]
    return np.bincount(flat.ravel(), minlength=n_boot * n).reshape(n_boot, n)


def _stable_int(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def stratum_rng(base_seed: int, L: int, name: str) -> np.random.Generator:
    return np.random.default_rng([base_seed, L, _stable_int(name)])


def seed_draws(base_seed: int, model: str, n_boot: int, n_seeds: int) -> np.ndarray:
    """(n_boot, n_seeds) seed indices drawn with replacement; independent per model (K-058, K-061)."""
    rng = np.random.default_rng([base_seed, _stable_int("seeds", model)])
    return rng.integers(0, n_seeds, size=(n_boot, n_seeds))


def hier(theta: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """theta (n_seeds, n_boot) -> replicate means over the drawn seeds (n_boot,)."""
    return theta[draws, np.arange(theta.shape[1])[:, None]].mean(axis=1)


class Prepared:
    """Sorted structure of one score vector on one evaluation set (for weighted recall, F1, ROC-AUC)."""

    def __init__(self, score: np.ndarray, y: np.ndarray, thr: float):
        y = y.astype(bool)
        pred = score > thr
        self.att_pred = (y & pred).astype(np.float32)
        self.nor_pred = (~y & pred).astype(np.float32)
        self.n_att = float(y.sum())
        self.n_nor = float((~y).sum())
        # ROC-AUC = P(score_attack > score_normal) + 1/2 P(tie): attack windows grouped by equal score; for each
        # group the number of normal windows strictly below (lo) and at most equal (hi) in sorted normal order
        ia, inn = np.flatnonzero(y), np.flatnonzero(~y)
        self.perm_a = ia[np.argsort(score[ia], kind="stable")]
        self.perm_n = inn[np.argsort(score[inn], kind="stable")]
        sa, sn = score[self.perm_a], score[self.perm_n]
        self.gstart = np.flatnonzero(np.r_[True, sa[1:] != sa[:-1]])
        gv = sa[self.gstart]
        self.lo = np.searchsorted(sn, gv, side="left")
        self.hi = np.searchsorted(sn, gv, side="right")

    def metrics(self, C: np.ndarray, auc: bool = True) -> dict[str, np.ndarray]:
        """Recall, F1 and ROC-AUC under integer window weights C (rows = replicates; the class totals are fixed by
        the stratified resampling). C may be float32: every partial sum is an integer below 2**24."""
        tp = (C @ self.att_pred).astype(np.float64)
        fp = (C @ self.nor_pred).astype(np.float64)
        fn = self.n_att - tp
        out = {"recall": tp / self.n_att, "f1": 2 * tp / (2 * tp + fp + fn)}
        if auc:
            gA = np.add.reduceat(C[:, self.perm_a], self.gstart, axis=1).astype(np.float64)
            cn = np.zeros((C.shape[0], len(self.perm_n) + 1), dtype=np.float64)
            np.cumsum(C[:, self.perm_n], axis=1, out=cn[:, 1:])
            below = cn[:, self.lo]
            ties = cn[:, self.hi] - below
            out["roc_auc"] = (gA * (below + 0.5 * ties)).sum(axis=1) / (self.n_att * self.n_nor)
        return out


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


class TestScores:
    """Test windows of one dataset and the scores / thresholds of every quantized version (all seeds)."""

    def __init__(self, dataset: str, runs: list[dict], mode: str, inp: A.Inputs, raw: Path = T.RAW_DIR):
        self.dataset = dataset
        self.models = sorted({r["model"] for r in runs})
        self.seeds = sorted({r["seed"] for r in runs})
        self.score, self.thr, self.metrics = {}, {}, {}
        self.win = None
        for r in runs:
            d = raw / r["run_id"] / "quant"
            m = inp.csv(d / "metrics.csv")
            # recalibrated thresholds at full precision (metrics.csv keeps 16 significant digits, which flips windows
            # whose fp16 score equals the threshold); q0_threshold mode = the Q0 threshold for every level (K-045)
            meta = inp.json(d / "quant_meta.json")["thresholds"]
            for q in A.QUANT_LEVELS:
                f = d / f"scores_{q}_test.parquet"
                inp._add(f)
                w = pd.read_parquet(f)
                if self.win is None:
                    self.win = w[["segment", "start", "label", "attack_type"]].copy()
                    self.win["capture"] = self.win["segment"].str.split("[").str[0]
                elif not (np.array_equal(w["segment"].to_numpy(), self.win["segment"].to_numpy())
                          and np.array_equal(w["start"].to_numpy(), self.win["start"].to_numpy())):
                    raise SystemExit(f"{f}: test windows differ from the other runs of {dataset}")
                thr = m.loc[(m["quant"] == q) & (m["threshold_mode"] == mode) & (m["split"] == "test"), "threshold"]
                if thr.nunique() != 1:
                    raise SystemExit(f"{d}: threshold of {q}/{mode} not unique")
                exact = float(meta[q if mode == "recalibrated" else "q0"]["threshold"])
                if not np.isclose(exact, float(thr.iloc[0]), rtol=1e-14, atol=0):
                    raise SystemExit(f"{d}: quant_meta threshold {exact!r} != metrics.csv {thr.iloc[0]!r}")
                key = (r["model"], r["seed"], q)
                self.score[key] = w["score"].to_numpy(dtype=np.float64)
                self.thr[key] = exact
                self.metrics[key] = m[(m["quant"] == q) & (m["threshold_mode"] == mode) & (m["split"] == "test")
                                      & (m["level"] == "attack_type")].set_index("name")

    def type_sets(self) -> dict[str, dict]:
        """K-029 evaluation set of every attack type: its attack windows plus the normal windows of the captures
        holding them; split into (segment, label) strata in time order."""
        w = self.win
        y = w["label"].to_numpy() == 1
        cap = w["capture"].to_numpy()
        typ = w["attack_type"].to_numpy()
        out = {}
        for t in sorted(set(typ[y])):
            caps = sorted(set(cap[y & (typ == t)]))
            mask = np.isin(cap, caps) & (~y | (typ == t))
            sub = w[mask].assign(_i=np.flatnonzero(mask)).sort_values(["segment", "label", "start"], kind="stable")
            strata = [(f"{self.dataset}|{seg}|{lab}", g["_i"].to_numpy())
                      for (seg, lab), g in sub.groupby(["segment", "label"], sort=True)]
            out[t] = {"captures": caps, "strata": strata, "index": np.concatenate([s[1] for s in strata])}
        return out


def type_replicates(ts: TestScores, t: str, tset: dict, n_boot: int, L_rule, base_seed: int, chunk: int,
                    auc: bool) -> dict:
    """theta[(model, q, metric)] -> (n_seeds, n_boot) replicate values and the full-sample values (n_seeds,)."""
    idx = tset["index"]
    y = ts.win["label"].to_numpy()[idx] == 1
    prep = {(m, s, q): Prepared(ts.score[(m, s, q)][idx], y, ts.thr[(m, s, q)])
            for m in ts.models for s in ts.seeds for q in A.QUANT_LEVELS}
    metrics = BOOT_METRICS if auc else ("recall", "f1")
    full = {k: p.metrics(np.ones((1, len(idx))), auc) for k, p in prep.items()}
    theta = {(m, q, k): np.empty((len(ts.seeds), n_boot)) for m in ts.models for q in A.QUANT_LEVELS
             for k in metrics}
    gens = [(stratum_rng(base_seed, block_length(len(ix), L_rule), name), block_length(len(ix), L_rule), len(ix))
            for name, ix in tset["strata"]]
    for b0 in range(0, n_boot, chunk):
        nb = min(chunk, n_boot - b0)
        C = np.concatenate([mbb_counts(n, L, nb, g) for g, L, n in gens], axis=1).astype(np.float32)
        for (m, s, q), p in prep.items():
            res = p.metrics(C, auc)
            for k in metrics:
                theta[(m, q, k)][ts.seeds.index(s), b0:b0 + nb] = res[k]
    point = {(m, q, k): np.array([full[(m, s, q)][k][0] for s in ts.seeds]) for m, q, k in theta}
    return {"theta": theta, "point": point, "n": len(idx), "n_attack": int(y.sum())}


def verify_point(ts: TestScores, t: str, point: dict, tol: float = 1e-9) -> None:
    """Recomputed per-seed recall, F1, ROC-AUC must equal metrics.csv (else the set definition is wrong)."""
    for (m, q, k), vals in point.items():
        for i, s in enumerate(ts.seeds):
            ref = float(ts.metrics[(m, s, q)].loc[t, k])
            if not abs(vals[i] - ref) <= tol:
                raise SystemExit(f"{ts.dataset}/{t} {m} s{s} {q} {k}: recomputed {vals[i]!r} != metrics.csv {ref!r}")


def cache_key(ts: TestScores, t: str, n_boot: int, L_rule, base_seed: int, auc: bool) -> str:
    h = hashlib.sha256()
    for k in sorted(ts.score):
        h.update(repr((k, ts.thr[k])).encode())
        h.update(ts.score[k].tobytes())
    h.update(repr((ts.dataset, t, n_boot, L_rule, base_seed, auc, 2)).encode())
    return h.hexdigest()[:24]


def replicates(ts: TestScores, t: str, tset: dict, cfg: dict, n_boot: int, L_rule, auc: bool,
               force: bool) -> dict:
    b = cfg["bootstrap"]
    key = cache_key(ts, t, n_boot, L_rule, b["rng_seed"], auc)
    f = CACHE / f"{ts.dataset}__{t}__L{L_rule}__B{n_boot}.npz"
    if f.exists() and not force:
        with np.load(f, allow_pickle=False) as z:      # closed before a stale file is replaced (Windows)
            if str(z["key"]) == key:
                names = json.loads(str(z["names"]))
                theta = {tuple(n): z[f"t{i}"] for i, n in enumerate(names)}
                point = {tuple(n): z[f"p{i}"] for i, n in enumerate(names)}
                return {"theta": theta, "point": point, "n": int(z["n"]), "n_attack": int(z["n_attack"]),
                        "cached": True}
    r = type_replicates(ts, t, tset, n_boot, L_rule, b["rng_seed"], b["chunk"], auc)
    verify_point(ts, t, r["point"])
    CACHE.mkdir(parents=True, exist_ok=True)
    names = list(r["theta"])
    arrays = {f"t{i}": r["theta"][n] for i, n in enumerate(names)}
    arrays.update({f"p{i}": r["point"][n] for i, n in enumerate(names)})
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez_compressed(tmp, key=key, names=json.dumps(names), n=r["n"], n_attack=r["n_attack"], **arrays)
    tmp.replace(f)
    r["cached"] = False
    return r


# --------------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------------


class Boot:
    """Replicates of every (dataset, type) and helpers to turn them into seed-mean contrasts."""

    def __init__(self, cfg: dict, n_boot: int, n_seeds: int, models: list[str]):
        self.cfg, self.n_boot = cfg, n_boot
        self.level = cfg["bootstrap"]["ci"]
        self.draws = {m: seed_draws(cfg["bootstrap"]["rng_seed"], m, n_boot, n_seeds) for m in models}
        self.rep: dict[tuple, dict] = {}          # (dataset, type, L) -> replicates

    def delta(self, ds: str, t: str, m: str, q: str, k: str, L="cube_root") -> tuple[float, np.ndarray, np.ndarray]:
        """Q0 - Qk of metric k: (3-seed mean, per-seed values, hierarchical replicates)."""
        r = self.rep[(ds, t, L)]
        seeds = r["point"][(m, "q0", k)] - r["point"][(m, q, k)]
        theta = r["theta"][(m, "q0", k)] - r["theta"][(m, q, k)]
        return float(seeds.mean()), seeds, hier(theta, self.draws[m])

    def value(self, ds: str, t: str, m: str, q: str, k: str, L="cube_root") -> tuple[float, np.ndarray, np.ndarray]:
        r = self.rep[(ds, t, L)]
        return float(r["point"][(m, q, k)].mean()), r["point"][(m, q, k)], hier(r["theta"][(m, q, k)], self.draws[m])


def row(analysis: str, **kw) -> dict:
    base = {"analysis": analysis, "model": "", "dataset": "", "quantization": "", "unit": "", "statistic": "",
            "estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_units": np.nan, "n_seeds": np.nan,
            "seed_values": "", "p_value": np.nan, "p_holm": np.nan, "holm_family": "", "min_attainable_p": np.nan,
            "block": "", "note": ""}
    base.update(kw)
    return base


def fmt_seeds(v) -> str:
    return ";".join(f"{x:.6g}" for x in np.asarray(v, dtype=float))


def q0_recall_means(t_all: pd.DataFrame, mode: str) -> pd.Series:
    sel = t_all[(t_all["split"] == "test") & (t_all["level"] == "attack_type") & (t_all["metric"] == "recall")
                & (t_all["quantization"].isin(["q0", A.NATIVE])) & (t_all["threshold_mode"].isin([mode, A.CALIBRATED]))]
    return sel.groupby(["dataset", "model", "attack_type"])["value"].mean()


def as1(boot: Boot, cfg: dict, t_all: pd.DataFrame, k041: pd.DataFrame, groups: dict) -> tuple[list, pd.DataFrame]:
    """K-057: per-unit dR_fab, dR_masq, D (and the F1 versions); mean D over the primary units."""
    floor = cfg["as1"]["floor"]
    q0r = q0_recall_means(t_all, cfg["primary_threshold_mode"])
    roles = t_all[(t_all["level"] == "attack_type")].groupby(["dataset", "attack_type"])["role"].first()
    rows, table = [], []
    ds = "road"
    fabs = [t for t, g in groups[ds].items() if f"{t}_masquerade" in groups[ds]]
    for m in cfg["quantized_models"] + cfg["reference_models"]:
        for tf in sorted(fabs):
            tm = f"{tf}_masquerade"
            rf, rm = float(q0r[(ds, m, tf)]), float(q0r[(ds, m, tm)])
            role = roles[(ds, tm)]
            in_floor = rf >= floor and rm >= floor
            ref = k041[(k041["dataset"] == ds) & (k041["model"] == m)
                       & k041["fabrication"].str.startswith(f"{ds}/{tf}_attack")]
            if len(ref) != 1:
                raise SystemExit(f"grid_k041_units.csv: {len(ref)} rows for {m}/{tf}")
            k_table = bool(ref["in_primary_test"].iloc[0])
            base = {"model": m, "dataset": ds, "pair": tf, "fabrication_type": tf, "masquerade_type": tm, "role": role,
                    "source_capture": ref["source_capture"].iloc[0], "q0_recall_fabrication_mean": rf,
                    "q0_recall_masquerade_mean": rm, "k041_in_primary_test_table": k_table}
            if m in cfg["reference_models"]:
                reason, eligible = "not quantized: no Qk, delta undefined (K-042, K-057)", False
            elif role != cfg["as1"]["primary_role"]:
                reason, eligible = "not in the primary pool (K-019)", False
            elif not in_floor:
                reason, eligible = f"floor: mean Q0 recall < {floor} (K-041)", False
            else:
                reason, eligible = "", True
            # the floor recomputed from all_results must reproduce the P5 unit table (K-041, K-053)
            if k_table != (role == cfg["as1"]["primary_role"] and in_floor):
                raise SystemExit(f"K-041 floor mismatch for {m}/{tf}: grid_k041_units.csv says {k_table}")
            if m in cfg["reference_models"]:
                table.append({**base, "quantization": "", "as1_eligible": False, "as1_excluded_reason": reason})
                continue
            for q in QK:
                rec = {**base, "quantization": q, "as1_eligible": eligible, "as1_excluded_reason": reason}
                for k, lab in (("recall", "R"), ("f1", "F1")):
                    for L, tag in (("cube_root", "block"), (cfg["bootstrap"]["sensitivity_block_length"], "iid")):
                        if (ds, tf, L) not in boot.rep:
                            continue
                        ef, sf, bf = boot.delta(ds, tf, m, q, k, L)
                        em, sm, bm = boot.delta(ds, tm, m, q, k, L)
                        sd, bd = sm - sf, bm - bf
                        for name, e, sv, bv in ((f"d{lab}_fab", ef, sf, bf), (f"d{lab}_masq", em, sm, bm),
                                                (f"D_{lab}", float(sd.mean()), sd, bd)):
                            lo, hi = ci(bv, boot.level)
                            if tag == "block":
                                rec[f"{name}_seeds"] = fmt_seeds(sv)
                                rec[f"{name}_mean"] = e
                                rec[f"{name}_std"] = float(np.std(sv, ddof=1))
                            rec[f"{name}_ci_low_{tag}"], rec[f"{name}_ci_high_{tag}"] = lo, hi
                            if name.startswith("D_") or tag == "block":
                                rows.append(row("as1_pair", model=m, dataset=ds, quantization=q, unit=tf,
                                                statistic=name, estimate=e, ci_low=lo, ci_high=hi, n_units=1,
                                                n_seeds=len(sv), seed_values=fmt_seeds(sv), block=tag,
                                                note="primary unit" if eligible else f"reported only: {reason}"))
                table.append(rec)
    tab = pd.DataFrame(table)
    # mean D over the primary units, per model and k
    prim = tab[tab["as1_eligible"] == True]  # noqa: E712
    for (m, q), g in prim.groupby(["model", "quantization"]):
        for lab in ("R", "F1"):
            k = "recall" if lab == "R" else "f1"
            for L, tag in (("cube_root", "block"), (cfg["bootstrap"]["sensitivity_block_length"], "iid")):
                parts = [(boot.delta(ds, u, m, q, k, L), boot.delta(ds, f"{u}_masquerade", m, q, k, L))
                         for u in g["pair"]]
                seeds = np.mean([pm[1] - pf[1] for pf, pm in parts], axis=0)
                rep = np.mean([pm[2] - pf[2] for pf, pm in parts], axis=0)
                lo, hi = ci(rep, boot.level)
                rows.append(row("as1_pooled", model=m, dataset=ds, quantization=q, unit=";".join(g["pair"]),
                                statistic=f"mean_D_{lab}", estimate=float(seeds.mean()), ci_low=lo, ci_high=hi,
                                n_units=len(g), n_seeds=len(seeds), seed_values=fmt_seeds(seeds),
                                min_attainable_p=min_p_wilcoxon(len(g)), block=tag,
                                note="exploratory; no test: Wilcoxon with n units cannot reach p below "
                                     "min_attainable_p (K-057)"))
    return rows, tab


def per_type(boot: Boot, cfg: dict, sets: dict, roles: pd.Series) -> tuple[list, pd.DataFrame]:
    rows, table = [], []
    for (ds, t, L), r in sorted(boot.rep.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if L != "cube_root":
            continue
        for m in cfg["quantized_models"]:
            for q in QK:
                for k in BOOT_METRICS:
                    e, sv, bv = boot.delta(ds, t, m, q, k)
                    lo, hi = ci(bv, boot.level)
                    rec = {"model": m, "dataset": ds, "attack_type": t, "attack_group": sets[ds][t]["group"],
                           "role": roles[(ds, t)], "quantization": q, "metric": k, "delta_mean": e,
                           "delta_std": float(np.std(sv, ddof=1)), "delta_seeds": fmt_seeds(sv), "ci_low": lo,
                           "ci_high": hi, "n_windows": r["n"], "n_attack": r["n_attack"]}
                    table.append(rec)
                    rows.append(row("per_type_delta", model=m, dataset=ds, quantization=q, unit=t,
                                    statistic=f"d_{k}", estimate=e, ci_low=lo, ci_high=hi, n_units=1,
                                    n_seeds=len(sv), seed_values=fmt_seeds(sv), block="block",
                                    note=f"group={rec['attack_group']}; role={rec['role']}"))
    return rows, pd.DataFrame(table)


def group_tests(boot: Boot, cfg: dict, sets: dict, roles: pd.Series) -> list:
    """K-059: ROAD coarse vs stealthy types; exact MWU, Cliff's delta with bootstrap interval; Holm per family."""
    rows = []
    ds = cfg["group_test"]["dataset"]
    add = cfg["group_test"]["coarse_sensitivity_add"]

    def types(d, grp):
        return sorted(t for t in sets[d] if sets[d][t]["group"] == grp and roles[(d, t)] == "primary")

    designs = {"road": ([(ds, t) for t in types(ds, "coarse")], [(ds, t) for t in types(ds, "stealthy")]),
               "road+car_hacking_coarse": ([(ds, t) for t in types(ds, "coarse")] + [(add, t) for t in types(add, "coarse")],
                                           [(ds, t) for t in types(ds, "stealthy")])}
    for design, (coarse, stealthy) in designs.items():
        for k in ("recall", "f1"):
            fam = []
            for m in cfg["quantized_models"]:
                for q in QK:
                    dc = [boot.delta(d, t, m, q, k) for d, t in coarse]
                    dsl = [boot.delta(d, t, m, q, k) for d, t in stealthy]
                    xc, xs = np.array([x[0] for x in dc]), np.array([x[0] for x in dsl])
                    u, p = exact_mwu(xs, xc)
                    delta = cliffs_delta(xs, xc)
                    rc, rs = np.stack([x[2] for x in dc], 1), np.stack([x[2] for x in dsl], 1)
                    reps = np.sign(rs[:, :, None] - rc[:, None, :]).mean(axis=(1, 2))
                    lo, hi = ci(reps, boot.level)
                    fam.append(row("group_mwu", model=m, dataset=design, quantization=q,
                                   unit=f"stealthy n={len(xs)} vs coarse n={len(xc)}", statistic=f"cliffs_delta_d{k}",
                                   estimate=delta, ci_low=lo, ci_high=hi, n_units=len(xs) + len(xc), n_seeds=3,
                                   p_value=p, holm_family=f"group_{design}_d{k}",
                                   min_attainable_p=min_p_mwu(len(xs), len(xc)), block="block",
                                   note=f"U={u:g}; mean d{k} stealthy={xs.mean():.6g}, coarse={xc.mean():.6g}; "
                                        f"{'primary' if design == 'road' else 'sensitivity (dataset confounded)'}; "
                                        f"{'secondary family' if k == 'f1' else 'primary family'} (K-059)"))
            adj = holm([r["p_value"] for r in fam])
            for r, a in zip(fam, adj):
                r["p_holm"] = a
                r["note"] += f"; Holm-adjusted min attainable p = {min(1.0, r['min_attainable_p'] * len(fam)):.4g}"
            rows += fam
    return rows


def as3(boot: Boot, cfg: dict, sets: dict, roles: pd.Series, q0r: pd.Series) -> list:
    """K-061: d_t = |d_AE| - |d_CNN| per (dataset, type); mean with hierarchical bootstrap interval."""
    rows, fam = [], []
    sup, uns = cfg["quantized_models"]          # [cnn1d, conv_ae]
    fl = cfg["as3"]["recall_floor"]
    for k in ("roc_auc", "recall"):
        units = [(ds, t) for ds in sorted(sets) for t in sorted(sets[ds]) if roles[(ds, t)] == "primary"]
        if k == "recall":
            units = [(ds, t) for ds, t in units if q0r[(ds, sup, t)] >= fl and q0r[(ds, uns, t)] >= fl]
        for q in QK:
            a = [boot.delta(ds, t, uns, q, k) for ds, t in units]
            c = [boot.delta(ds, t, sup, q, k) for ds, t in units]
            d = np.array([abs(x[0]) - abs(y[0]) for x, y in zip(a, c)])
            rep = np.mean([np.abs(x[2]) - np.abs(y[2]) for x, y in zip(a, c)], axis=0)
            lo, hi = ci(rep, boot.level)
            nz = int((d != 0).sum())
            try:
                p = float(st.wilcoxon(d, zero_method=cfg["as3"]["wilcoxon_zero_method"]).pvalue) if nz else np.nan
            except ValueError:
                p = np.nan
            delta = cliffs_delta([abs(x[0]) for x in a], [abs(y[0]) for y in c])
            note = (f"units={';'.join(f'{ds}/{t}' for ds, t in units)}; mean signed d: {uns}="
                    f"{np.mean([x[0] for x in a]):.6g}, {sup}={np.mean([y[0] for y in c]):.6g}; "
                    f"Cliff's delta(|d_{uns}| vs |d_{sup}|)={delta:.4g}; Wilcoxon zero_method="
                    f"{cfg['as3']['wilcoxon_zero_method']}, {nz} non-zero differences")
            r = row("as3", model=f"{uns}-{sup}", dataset="car_hacking+road", quantization=q,
                    unit=f"{len(units)} types", statistic=f"mean_abs_diff_d{k}", estimate=float(d.mean()),
                    ci_low=lo, ci_high=hi, n_units=len(units), n_seeds=3, p_value=p, holm_family="as3",
                    min_attainable_p=min_p_wilcoxon(len(units)), block="block",
                    note=("primary (K-061); " if k == "roc_auc" else f"floor {fl} in both models (K-061); ") + note)
            fam.append(r)
            # K-063 (a): signed contrast d_AE - d_CNN (smooth; no absolute-value bias)
            sd = np.array([x[0] - y[0] for x, y in zip(a, c)])
            srep = np.mean([x[2] - y[2] for x, y in zip(a, c)], axis=0)
            slo, shi = ci(srep, boot.level)
            rows.append(row("as3_signed_diff", model=f"{uns}-{sup}", dataset="car_hacking+road", quantization=q,
                            unit=f"{len(units)} types", statistic=f"mean_signed_diff_d{k}", estimate=float(sd.mean()),
                            ci_low=slo, ci_high=shi, n_units=len(units), n_seeds=3, block="block",
                            note="robustness check, written after the B=200 smoke run (K-063); positive = "
                                 f"{uns} loses more"))
            for m, xs in ((uns, a), (sup, c)):
                rep_m = np.mean([x[2] for x in xs], axis=0)
                lo_m, hi_m = ci(rep_m, boot.level)
                rows.append(row("as3_signed", model=m, dataset="car_hacking+road", quantization=q,
                                unit=f"{len(units)} types", statistic=f"mean_d{k}",
                                estimate=float(np.mean([x[0] for x in xs])), ci_low=lo_m, ci_high=hi_m,
                                n_units=len(units), n_seeds=3, block="block", note="descriptive (K-061)"))
    adj = holm([r["p_value"] for r in fam])
    for r, x in zip(fam, adj):
        r["p_holm"] = x
    return fam + rows


def churn(scores: dict[str, "TestScores"], unseen: set[str]) -> tuple[list, pd.DataFrame]:
    """K-063 (b): share of primary-pool test windows whose decision differs between Q0 and Qk (own recalibrated
    thresholds) and the Spearman correlation of the Q0 and Qk scores; per seed, descriptive."""
    table, rows = [], []
    for ds, ts in scores.items():
        w = ts.win
        y = w["label"].to_numpy() == 1
        bad = {c for c, t in zip(w["capture"], w["attack_type"]) if f"{ds}/{t}" in unseen}
        pool = ~w["capture"].isin(bad).to_numpy()
        for m in ts.models:
            for s in ts.seeds:
                s0 = ts.score[(m, s, "q0")][pool]
                p0 = s0 > ts.thr[(m, s, "q0")]
                for q in QK:
                    sk = ts.score[(m, s, q)][pool]
                    pk = sk > ts.thr[(m, s, q)]
                    flip = p0 != pk
                    table.append({"dataset": ds, "model": m, "seed": s, "quantization": q, "n_windows": int(pool.sum()),
                                  "churn_all": float(flip.mean()), "churn_attack": float(flip[y[pool]].mean()),
                                  "churn_normal": float(flip[~y[pool]].mean()),
                                  "spearman_q0_qk": float(st.spearmanr(s0, sk).statistic)})
    tab = pd.DataFrame(table)
    for (ds, m, q), g in tab.groupby(["dataset", "model", "quantization"]):
        for col in ("churn_all", "churn_attack", "churn_normal", "spearman_q0_qk"):
            v = g[col].to_numpy()
            rows.append(row("as3_churn", model=m, dataset=ds, quantization=q, unit="primary_pool", statistic=col,
                            estimate=float(v.mean()), ci_low=float(v.min()), ci_high=float(v.max()), n_seeds=len(v),
                            seed_values=fmt_seeds(v),
                            note="robustness check (K-063 b), descriptive; ci_low/ci_high = min/max over seeds; "
                                 f"std={np.std(v, ddof=1):.3g}"))
    return rows, tab


def calibration_shift(t_all: pd.DataFrame, delta: pd.DataFrame, cfg: dict) -> tuple[list, pd.DataFrame]:
    """K-036 b / K-061: Q0 threshold applied to Qk; ambient FPR (val, test) and recall changes, per seed."""
    rows, table = [], []
    deep = t_all[t_all["model"].isin(cfg["quantized_models"]) & (t_all["threshold_mode"] == "q0_threshold")]
    fpr = deep[(deep["level"] == "attack_free") & (deep["metric"] == "fpr")]
    rec = deep[(deep["level"] == "pooled") & (deep["metric"] == "recall") & (deep["split"] == "test")]
    for (ds, m, seed), g in fpr.groupby(["dataset", "model", "seed"]):
        for q in QK:
            r = {"dataset": ds, "model": m, "seed": seed, "quantization": q}
            for sp in ("val", "test"):
                v = g[g["split"] == sp].set_index("quantization")["value"]
                r[f"fpr_{sp}_q0"], r[f"fpr_{sp}_qk"] = float(v["q0"]), float(v[q])
            pr = rec[(rec["dataset"] == ds) & (rec["model"] == m) & (rec["seed"] == seed)].set_index("quantization")
            r["pooled_recall_q0"], r["pooled_recall_qk"] = float(pr.loc["q0", "value"]), float(pr.loc[q, "value"])
            dt = delta[(delta["dataset"] == ds) & (delta["model"] == m) & (delta["seed"] == seed)
                       & (delta["quantization"] == q) & (delta["threshold_mode"] == "q0_threshold")
                       & (delta["level"] == "attack_type") & (delta["role"] == "primary") & (delta["metric"] == "recall")]
            r["type_d_recall_min"], r["type_d_recall_max"] = float(dt["delta"].min()), float(dt["delta"].max())
            table.append(r)
    tab = pd.DataFrame(table)
    for (ds, m, q), g in tab.groupby(["dataset", "model", "quantization"]):
        for col in ("fpr_val_qk", "fpr_test_qk", "fpr_val_q0", "fpr_test_q0", "pooled_recall_qk", "pooled_recall_q0"):
            v = g[col].to_numpy()
            rows.append(row("calib_shift", model=m, dataset=ds, quantization=q, unit="ambient" if "fpr" in col
                            else "primary_pool", statistic=col, estimate=float(v.mean()), ci_low=float(v.min()),
                            ci_high=float(v.max()), n_seeds=len(v), seed_values=fmt_seeds(v),
                            note="Q0 threshold applied to Qk (K-036 b); ci_low/ci_high = min/max over seeds"))
    return rows, tab


def message_rates(path: Path) -> dict[str, float]:
    p = pd.read_csv(path)
    amb = p[(p["kind"] == "ambient") & ~p["excluded"].astype(str).str.lower().eq("true")]
    return amb.groupby("dataset")["msg_rate_per_s"].median().to_dict()


def pareto_mask(points: np.ndarray, senses: list[str]) -> np.ndarray:
    """True for non-dominated rows; senses 'min' or 'max' per column."""
    x = np.asarray(points, dtype=float) * np.array([1 if s == "min" else -1 for s in senses])
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dom = np.all(x <= x[i], axis=1) & np.any(x < x[i], axis=1)
        keep[i] = not dom.any()
    return keep


def pareto(t_all: pd.DataFrame, cfg: dict, boot: Boot, inp: A.Inputs) -> tuple[list, pd.DataFrame]:
    pc = cfg["pareto"]
    rates_path = S.PROJECT_ROOT / pc["budgets"]["message_rate_source"]
    inp._add(rates_path)
    rates = message_rates(rates_path)
    sessions = pc["sessions"]
    res = t_all[t_all["level"] == "model"]
    test = t_all[(t_all["split"] == "test") & (t_all["threshold_mode"].isin([cfg["primary_threshold_mode"], A.CALIBRATED]))]
    cands = [dict(c, latency=True) for c in pc["candidates"]] + [dict(c, latency=False) for c in pc["size_only_candidates"]]
    table = []
    for ds in sorted(t_all["dataset"].unique()):
        stealthy = sorted(t for t in test[(test["dataset"] == ds) & (test["attack_group"] == "stealthy")
                                          & (test["role"] == "primary")]["attack_type"].unique())
        for c in cands:
            m, q = c["model"], c["quantization"]
            r = {"dataset": ds, "model": m, "quantization": q, "latency_measured": c["latency"]}
            rr = res[(res["dataset"] == ds) & (res["model"] == m) & (res["quantization"] == q)]
            r["size_mb"] = float(rr[rr["metric"] == "size_mb"]["value"].mean())
            for s in sessions:
                v = rr[(rr["metric"] == f"latency_median_ms_session{s}") & (rr["seed"] == 0)]["value"]
                r[f"latency_median_ms_s{s}"] = float(v.iloc[0]) if len(v) else np.nan
                v = rr[(rr["metric"] == f"latency_p95_ms_session{s}") & (rr["seed"] == 0)]["value"]
                r[f"latency_p95_ms_s{s}"] = float(v.iloc[0]) if len(v) else np.nan
            tt = test[(test["dataset"] == ds) & (test["model"] == m) & (test["quantization"] == q)]
            for k in ("recall", "f1"):
                pool = tt[(tt["level"] == "pooled") & (tt["metric"] == k)]["value"]
                r[f"pooled_{k}_mean"], r[f"pooled_{k}_std"] = float(pool.mean()), float(pool.std(ddof=1))
                if stealthy:
                    sv = tt[(tt["level"] == "attack_type") & (tt["metric"] == k) & tt["attack_type"].isin(stealthy)]
                    per_seed = sv.groupby("seed")["value"].mean()
                    r[f"stealthy_{k}_mean"], r[f"stealthy_{k}_std"] = float(per_seed.mean()), float(per_seed.std(ddof=1))
                    if m in cfg["quantized_models"] and k == "recall":
                        rep = np.mean([boot.value(ds, t, m, q, k)[2] for t in stealthy], axis=0)
                        r["stealthy_recall_ci_low"], r["stealthy_recall_ci_high"] = ci(rep, boot.level)
            table.append(r)
    tab = pd.DataFrame(table)
    rows = []
    for ds, g in tab.groupby("dataset"):
        idx = g.index
        rate = rates[ds]
        win_budget = pc["budgets"]["window_messages"] / rate * 1e3
        strict_budget = pc["budgets"]["strict_messages"] / rate * 1e3
        tab.loc[idx, "msg_rate_per_s"] = rate
        tab.loc[idx, "budget_window_ms"] = win_budget
        tab.loc[idx, "budget_strict_ms"] = strict_budget
        worst = g[[f"latency_median_ms_s{s}" for s in sessions]].max(axis=1)
        tab.loc[idx, "meets_window_budget"] = np.where(g["latency_measured"], worst < win_budget, np.nan)
        tab.loc[idx, "meets_strict_budget"] = np.where(g["latency_measured"], worst < strict_budget, np.nan)
        ref = g[(g["model"] == pc["latency_reference"]["model"]) & (g["quantization"] == pc["latency_reference"]["quantization"])]
        for s in sessions:
            tab.loc[idx, f"latency_ratio_vs_ref_s{s}"] = g[f"latency_median_ms_s{s}"] / float(ref[f"latency_median_ms_s{s}"].iloc[0])
        ratios = tab.loc[idx, [f"latency_ratio_vs_ref_s{s}" for s in sessions]]
        tab.loc[idx, "latency_ratio_min"], tab.loc[idx, "latency_ratio_max"] = ratios.min(axis=1), ratios.max(axis=1)
        has_stealthy = "stealthy_recall_mean" in g and g["stealthy_recall_mean"].notna().any()
        rec_col = "stealthy_recall_mean" if has_stealthy else "pooled_recall_mean"     # K-060
        fronts = {}
        for rc in dict.fromkeys([rec_col, "pooled_recall_mean"]):
            lat = g[g["latency_measured"]]
            on = np.zeros(len(lat), dtype=bool)
            for s in sessions:
                f = pareto_mask(lat[[f"latency_median_ms_s{s}", "size_mb", rc]].to_numpy(), ["min", "min", "max"])
                tab.loc[lat.index, f"front_{rc}_s{s}"] = f
                on |= f
            tab.loc[lat.index, f"robust_front_{rc}"] = on
            fs = pareto_mask(g[["size_mb", rc]].to_numpy(), ["min", "max"])
            tab.loc[idx, f"size_front_{rc}"] = fs
            fronts[rc] = on
        for _, r in tab.loc[idx].iterrows():
            for rc in fronts:
                rows.append(row("pareto", model=r["model"], dataset=ds, quantization=r["quantization"],
                                unit=rc.replace("_mean", ""), statistic="robust_front" if r["latency_measured"]
                                else "size_front",
                                estimate=float(r[f"robust_front_{rc}"] if r["latency_measured"] else r[f"size_front_{rc}"]),
                                note=f"latency s1/s2 = {r['latency_median_ms_s1']:.3g}/{r['latency_median_ms_s2']:.3g} ms "
                                     f"(order of magnitude only, K-050); size {r['size_mb']:.4g} MB; "
                                     f"{rc} {r[rc]:.4g}; window budget {win_budget:.3g} ms, strict {strict_budget:.3g} ms"))
    return rows, tab


# --------------------------------------------------------------------------------------------
# Hypotheses (K-062)
# --------------------------------------------------------------------------------------------


def verdict(parts: list[bool], partial_max: bool = False) -> str:
    if all(parts):
        return "kısmen doğrulandı" if partial_max else "doğrulandı"
    if any(parts):
        return "kısmen doğrulandı"
    return "doğrulanmadı"


def hypotheses(stats: pd.DataFrame, t_all: pd.DataFrame, delta: pd.DataFrame, ptd: pd.DataFrame, cfg: dict) -> list:
    h = cfg["hypotheses"]
    rows = []
    deep = cfg["quantized_models"]
    # H1 (a): pooled dF1 at Q2, seed mean <= limit for every deep model x dataset
    pf = delta[(delta["level"] == "pooled") & (delta["metric"] == "f1") & (delta["quantization"] == "q2")
               & (delta["threshold_mode"] == cfg["primary_threshold_mode"])].groupby(["model", "dataset"])["delta"].mean()
    h1a = bool((pf <= h["h1_pooled_df1_max"]).all())
    # H1 (b): over all primary types of the deep models at Q2: >= 1 interval excluding 0 on the loss side and
    # >= 1 interval containing 0
    r2 = ptd[(ptd["quantization"] == "q2") & (ptd["metric"] == "recall") & (ptd["role"] == "primary")]
    n_loss = int((r2["ci_low"] > 0).sum())
    n_zero = int(((r2["ci_low"] <= 0) & (r2["ci_high"] >= 0)).sum())
    h1b = n_loss >= 1 and n_zero >= 1
    rows.append(row("hypothesis", unit="H1", statistic="verdict", estimate=float(h1a and h1b),
                    note=f"{verdict([h1a, h1b])}; (a) pooled dF1(Q2) seed means "
                         f"{'; '.join(f'{m}/{d}={v:.4g}' for (m, d), v in pf.items())} <= {h['h1_pooled_df1_max']}: {h1a}; "
                         f"(b) types with dRecall(Q2) interval > 0: {n_loss}, containing 0: {n_zero} of {len(r2)}: {h1b}"))
    # H2 (a): mean D over primary units at Q2, interval excludes 0 on the positive side
    d = stats[(stats["analysis"] == "as1_pooled") & (stats["statistic"] == "mean_D_R") & (stats["quantization"] == "q2")
              & (stats["block"] == "block")]
    h2a = bool(len(d)) and bool(((d["estimate"] > 0) & (d["ci_low"] > 0)).all())
    # H2 (b): Q2 recall of the coarse reference types >= threshold for every deep model
    rec = t_all[(t_all["split"] == "test") & (t_all["level"] == "attack_type") & (t_all["metric"] == "recall")
                & (t_all["quantization"] == "q2") & (t_all["threshold_mode"] == cfg["primary_threshold_mode"])]
    vals = {}
    for ds, types in h["h2_coarse_types"].items():
        for t in types:
            for m in deep:
                vals[(m, ds, t)] = float(rec[(rec["dataset"] == ds) & (rec["model"] == m) & (rec["attack_type"] == t)]["value"].mean())
    h2b = all(v >= h["h2_coarse_recall_min"] for v in vals.values())
    rows.append(row("hypothesis", unit="H2", statistic="verdict", estimate=float(h2a and h2b),
                    note=f"{verdict([h2a, h2b])}; (a) mean D_R(Q2) "
                         f"{'; '.join(f'{r.model}={r.estimate:.4g} [{r.ci_low:.4g}, {r.ci_high:.4g}]' for r in d.itertuples())}"
                         f": {h2a}; (b) Q2 recall {'; '.join(f'{m}/{ds}/{t}={v:.4g}' for (m, ds, t), v in vals.items())}"
                         f" >= {h['h2_coarse_recall_min']}: {h2b}"))
    # H3 (a): mean |dRecall| over primary types, Q1 <= Q2 per deep model x dataset; (b) Q1 size < Q0; (c) = H2 (a)
    pr = ptd[(ptd["metric"] == "recall") & (ptd["role"] == "primary")]
    mag = pr.assign(a=pr["delta_mean"].abs()).groupby(["model", "dataset", "quantization"])["a"].mean().unstack()
    h3a = bool((mag["q1"] <= mag["q2"]).all())
    size = t_all[(t_all["level"] == "model") & (t_all["metric"] == "size_mb")]
    sz = size.groupby(["model", "dataset", "quantization"])["value"].mean().unstack()
    sz = sz[sz.index.get_level_values("model").isin(deep)]
    h3b = bool((sz["q1"] < sz["q0"]).all())
    rows.append(row("hypothesis", unit="H3", statistic="verdict", estimate=float(h3a and h3b and h2a),
                    note=f"{verdict([h3a, h3b, h2a], partial_max=True)} (latency gain of Q1 not testable, K-043); "
                         f"(a) mean |dRecall| Q1 vs Q2 "
                         f"{'; '.join(f'{m}/{ds}={r.q1:.4g} vs {r.q2:.4g}' for (m, ds), r in mag.iterrows())}: {h3a}; "
                         f"(b) size Q1/Q0 {'; '.join(f'{m}/{ds}={r.q1 / r.q0:.3f}' for (m, ds), r in sz.iterrows())}: {h3b}; "
                         f"(c) INT8 stealthy loss = H2 (a): {h2a}"))
    # H4: AS-3 at Q2, dROC-AUC interval entirely > 0; dRecall in the same direction
    a3 = stats[(stats["analysis"] == "as3") & (stats["quantization"] == "q2")].set_index("statistic")
    auc_up = bool(a3.loc["mean_abs_diff_droc_auc", "ci_low"] > 0)
    rec_up = bool(a3.loc["mean_abs_diff_drecall", "ci_low"] > 0)
    v = "doğrulandı" if auc_up and rec_up else "kısmen doğrulandı" if auc_up or rec_up else "doğrulanmadı"
    sg = stats[(stats["analysis"] == "as3_signed_diff") & (stats["quantization"] == "q2")].set_index("statistic")
    s_auc = bool(sg.loc["mean_signed_diff_droc_auc", "ci_low"] > 0)
    s_rec = bool(sg.loc["mean_signed_diff_drecall", "ci_low"] > 0)
    v_signed = "doğrulandı" if s_auc and s_rec else "kısmen doğrulandı" if s_auc or s_rec else "doğrulanmadı"
    final = v if v == v_signed else "kullanıcı kararı bekliyor (K-063)"
    rows.append(row("hypothesis", unit="H4", statistic="verdict", estimate=float(auc_up and rec_up),
                    note=f"{final}; K-062 rule: {v}: mean(|dAUC_AE| - |dAUC_CNN|)(Q2) = {a3.loc['mean_abs_diff_droc_auc', 'estimate']:.4g} "
                         f"[{a3.loc['mean_abs_diff_droc_auc', 'ci_low']:.4g}, {a3.loc['mean_abs_diff_droc_auc', 'ci_high']:.4g}]: {auc_up}; "
                         f"mean(|dR_AE| - |dR_CNN|)(Q2) = {a3.loc['mean_abs_diff_drecall', 'estimate']:.4g} "
                         f"[{a3.loc['mean_abs_diff_drecall', 'ci_low']:.4g}, {a3.loc['mean_abs_diff_drecall', 'ci_high']:.4g}]: {rec_up}; "
                         f"K-063 signed check: {v_signed}: mean(dAUC_AE - dAUC_CNN)(Q2) = "
                         f"{sg.loc['mean_signed_diff_droc_auc', 'estimate']:.4g} [{sg.loc['mean_signed_diff_droc_auc', 'ci_low']:.4g}, "
                         f"{sg.loc['mean_signed_diff_droc_auc', 'ci_high']:.4g}]; mean(dR_AE - dR_CNN)(Q2) = "
                         f"{sg.loc['mean_signed_diff_drecall', 'estimate']:.4g} [{sg.loc['mean_signed_diff_drecall', 'ci_low']:.4g}, "
                         f"{sg.loc['mean_signed_diff_drecall', 'ci_high']:.4g}]"))
    return rows


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default: configs/analysis.yaml)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: configs/models.yaml runtime)")
    ap.add_argument("--force", action="store_true", help="recompute cached bootstrap replicates")
    ap.add_argument("--out", type=Path, default=A.TABLES, help="output directory of the tables")
    a = ap.parse_args(argv)

    from threadpoolctl import threadpool_limits

    threads = a.threads or T.load_models_config()["runtime"]["threads"]
    threadpool_limits(threads)
    cfg = A.load_analysis_config()
    n_boot = a.n_boot or cfg["bootstrap"]["n_boot"]
    t0 = time.time()
    inp = A.Inputs()
    t_all = inp.csv(A.TABLES / "all_results.csv")
    delta = inp.csv(A.TABLES / "delta_f1.csv")
    k041 = inp.csv(A.TABLES / "grid_k041_units.csv")
    import yaml

    from src.data import splits as SP

    groups = yaml.safe_load(SP.GROUPS_PATH.read_text(encoding="utf-8"))
    runs = [r for r in A.grid_runs(cfg) if r["model"] in cfg["quantized_models"]]
    roles = t_all[t_all["level"] == "attack_type"].groupby(["dataset", "attack_type"])["role"].first()
    q0r = q0_recall_means(t_all, cfg["primary_threshold_mode"])
    boot = Boot(cfg, n_boot, len({r["seed"] for r in runs}), cfg["quantized_models"])
    sets, scores = {}, {}
    log(f"bootstrap: B={n_boot}, threads={threads}, cache {CACHE.relative_to(S.PROJECT_ROOT)}")
    jobs = []
    for ds in sorted({r["dataset"] for r in runs}):
        ts = TestScores(ds, [r for r in runs if r["dataset"] == ds], cfg["primary_threshold_mode"], inp)
        scores[ds] = ts
        tsets = ts.type_sets()
        sets[ds] = {t: {"group": groups[ds][t]["group"], **v} for t, v in tsets.items()}
        for t in tsets:
            jobs.append((ts, t, "cube_root", True))
            if f"{t}_masquerade" in groups[ds] or t.endswith("_masquerade"):      # pair members (AS-1)
                jobs.append((ts, t, cfg["bootstrap"]["sensitivity_block_length"], False))
    for i, (ts, t, L, auc) in enumerate(jobs, 1):
        t1 = time.time()
        r = replicates(ts, t, sets[ts.dataset][t], cfg, n_boot, L, auc, a.force)
        boot.rep[(ts.dataset, t, L)] = r
        el = time.time() - t0
        log(f"PROGRESS {i}/{len(jobs)} {ts.dataset}/{t} L={L} n={r['n']} "
            f"{'cached' if r['cached'] else f'{time.time() - t1:.1f}s'}; elapsed {el:.0f}s")
    rows = []
    r1, as1_tab = as1(boot, cfg, t_all, k041, groups)
    r2, ptd = per_type(boot, cfg, sets, roles)
    rows += r1 + r2 + group_tests(boot, cfg, sets, roles) + as3(boot, cfg, sets, roles, q0r)
    r3, cs = calibration_shift(t_all, delta, cfg)
    r4, par = pareto(t_all, cfg, boot, inp)
    unseen = set(S.load_config()["splits"]["analysis_roles"]["unseen_in_training"])
    r5, ch = churn(scores, unseen)
    rows += r3 + r4 + r5
    stats = pd.DataFrame(rows)
    stats = pd.concat([stats, pd.DataFrame(hypotheses(stats, t_all, delta, ptd, cfg))], ignore_index=True)
    stats["n_boot"] = n_boot
    a.out.mkdir(parents=True, exist_ok=True)
    outs = {"stats.csv": stats, "as1_pairs.csv": as1_tab, "per_type_delta.csv": ptd, "pareto.csv": par,
            "calibration_shift.csv": cs, "decision_churn.csv": ch}
    for name, df in outs.items():
        df.to_csv(a.out / name, index=False, float_format="%.6g")
    if a.out == A.TABLES:
        A.write_manifest("stats", cfg, inp.files, [a.out / n for n in outs],
                         {"n_boot": n_boot, "threads": threads, "elapsed_s": round(time.time() - t0, 1)})
    log(f"done in {time.time() - t0:.0f}s: " + ", ".join(f"{n} ({len(d)} rows)" for n, d in outs.items()))
    for r in stats[stats["analysis"] == "hypothesis"].itertuples():
        log(f"{r.unit}: {r.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
