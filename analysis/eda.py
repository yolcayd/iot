"""Exploratory analysis of the converted CAN datasets (phase P1).

Outputs:
    results/tables/data_profile.csv        per-capture profile
    results/tables/iat_by_id.csv           per-ID inter-arrival statistics of attack-free traffic
    results/tables/periodicity_summary.csv share of periodic IDs per dataset
    results/tables/attack_signatures.csv   per-attack-type traffic metrics, grouping rule C1..C4
    results/figures/signature_<dataset>_<attack>.{pdf,png}
    results/figures/attack_signature_summary.{pdf,png}

The grouping rule and its thresholds are read from configs/eda.yaml (fixed before this script
was first run). Only traffic statistics are used; no model output.

Captures are read as views (src/data/sources.py): the attack-free tail that all four Car-Hacking
attack files share is cut from each file and profiled once as ``car_hacking/shared_tail``
(docs/DECISIONS.md K-016). It is not part of any attack file's normal period.

Usage:
    python -m analysis.eda
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.data import loaders as L
from src.data import sources as S

PROJECT_ROOT = L.PROJECT_ROOT
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
CONFIG = PROJECT_ROOT / "configs" / "eda.yaml"

# A-priori group from the project plan (docs/PLAN.md section 1.2): DoS/flooding and fuzzing are
# coarse; spoofing, malfunction (ROAD targeted fabrication) and masquerade are stealthy.
PLAN_GROUP = {
    "dos": "coarse",
    "fuzzy": "coarse",
    "fuzzing": "coarse",
}

COLOR_NORMAL = "#2a78d6"  # categorical slot 1
COLOR_ATTACK = "#eb6834"  # categorical slot 2
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"


def plan_group(attack_type: str) -> str:
    return PLAN_GROUP.get(attack_type, "stealthy")


# --------------------------------------------------------------------------------------------
# Loading and period definitions
# --------------------------------------------------------------------------------------------


@dataclass
class Capture:
    row: dict
    ts: np.ndarray
    can_id: np.ndarray
    label: np.ndarray
    data: np.ndarray
    attack_type: str
    # period id per frame: >= 0 attack segment k, <= -1 normal segment, NaN-like sentinel excluded
    period: np.ndarray = field(default=None)  # type: ignore[assignment]
    target_id: int | None = None

    @property
    def name(self) -> str:
        return self.row["capture"]


EXCLUDED = np.iinfo(np.int32).min
# ROAD uses 3-hex-digit IDs up to 0xFFF (anonymized), beyond the 11-bit 0x7FF range.
ID_SPACE = 4096


def load_capture(view: S.View) -> Capture:
    t = S.read_view(view, columns=["timestamp", "can_id", "label", "data", "attack_type"])
    row = asdict(view)
    types = sorted(set(t["attack_type"].unique().to_pylist()) - {L.NORMAL})
    return Capture(
        row=row,
        ts=t["timestamp"].to_numpy(),
        can_id=t["can_id"].to_numpy(),
        label=t["label"].to_numpy(),
        data=L.data_matrix(t),
        attack_type=types[0] if types else L.NORMAL,
    )


def segment_ids(mask: np.ndarray, start_index: int, negative: bool) -> np.ndarray:
    """Number the contiguous True runs of ``mask`` (frames order) starting at ``start_index``."""
    starts = mask & ~np.r_[False, mask[:-1]]
    ids = np.cumsum(starts) - 1 + start_index
    return -(ids + 1) if negative else ids


def assign_periods(cap: Capture, cfg: dict, road_meta: dict) -> None:
    """Label every frame as attack period (>= 0), normal period (< 0) or excluded margin."""
    ts = cap.ts
    n = len(ts)
    if cap.row["dataset"] == "car_hacking":
        gap = cfg["periods"]["car_hacking_burst_gap_s"]
        margin = cfg["periods"]["car_hacking_margin_s"]
        t_inj = ts[cap.label == 1]
        breaks = np.where(np.diff(t_inj) > gap)[0]
        starts = np.r_[t_inj[0], t_inj[breaks + 1]]
        ends = np.r_[t_inj[breaks], t_inj[-1]]
        k = np.searchsorted(starts, ts, side="right") - 1
        valid = k >= 0
        in_attack = np.zeros(n, bool)
        in_attack[valid] = ts[valid] <= ends[k[valid]]
        km = np.searchsorted(starts - margin, ts, side="right") - 1
        near = np.zeros(n, bool)
        vm = km >= 0
        near[vm] = ts[vm] <= ends[km[vm]] + margin
        normal = ~near
        ids = np.full(n, EXCLUDED, dtype=np.int64)
        ids[in_attack] = k[in_attack]
        ids[normal] = segment_ids(normal, 0, True)[normal]
        cap.period = ids
        tid = np.unique(cap.can_id[cap.label == 1])
        cap.target_id = int(tid[0]) if len(tid) == 1 else None
    else:
        meta = road_meta[cap.name.split("/", 1)[1]]
        rel = ts - ts[0]
        a, b = meta["injection_interval"]
        in_attack = (rel >= a - 1e-6) & (rel <= b + 1e-6)
        ids = np.where(in_attack, 0, segment_ids(~in_attack, 0, True))
        cap.period = ids.astype(np.int64)
        cap.target_id = None if meta["injection_id"] == "XXX" else int(meta["injection_id"], 16)


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def jsd(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits between two count vectors on the same support."""
    p = p_counts / p_counts.sum()
    q = q_counts / q_counts.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / m), 0.0).sum()
        kl_q = np.where(q > 0, q * np.log2(q / m), 0.0).sum()
    return float(0.5 * kl_p + 0.5 * kl_q)


def id_counts(ids: np.ndarray) -> np.ndarray:
    counts = np.bincount(ids, minlength=ID_SPACE)
    if len(counts) != ID_SPACE:
        raise ValueError("CAN ID outside 0..0xFFF")
    return counts.astype(np.float64)


def period_durations(ts: np.ndarray, period: np.ndarray, attack: bool, gap_s: float) -> float:
    """Active duration of the attack or normal period: sum of consecutive-frame intervals inside
    the same period segment, excluding recording gaps longer than ``gap_s``."""
    sel = period >= 0 if attack else (period < 0) & (period != EXCLUDED)
    t, p = ts[sel], period[sel]
    if len(t) < 2:
        return 0.0
    d = np.diff(t)
    keep = (p[1:] == p[:-1]) & (d <= gap_s)
    return float(d[keep].sum())


RECORDING_GAP_S = 1.0  # overwritten from configs/eda.yaml in main()


def target_iat(ts: np.ndarray, period: np.ndarray, can_id: np.ndarray, target: int, attack: bool) -> np.ndarray:
    """IATs of ``target`` inside one period segment, excluding intervals that span a recording gap."""
    sel = (can_id == target) & ((period >= 0) if attack else ((period < 0) & (period != EXCLUDED)))
    t, p = ts[sel], period[sel]
    d = np.diff(t)
    return d[(p[1:] == p[:-1]) & (d <= RECORDING_GAP_S)]


def bit_flip_rate(data: np.ndarray) -> float:
    """Mean fraction of payload bits that change between consecutive frames."""
    if len(data) < 2:
        return float("nan")
    x = np.bitwise_xor(data[1:], data[:-1])
    return float(np.unpackbits(x, axis=1).mean())


def attack_windows(cap: Capture, w: int) -> list[np.ndarray]:
    out = []
    for k in np.unique(cap.period[cap.period >= 0]):
        idx = np.where(cap.period == k)[0]
        if len(idx) < w:
            out.append(cap.can_id[idx])
        else:
            for s in range(0, len(idx) - w + 1, w):
                out.append(cap.can_id[idx[s : s + w]])
    return out


def null_jsd(reference: list[Capture], w: int, comparator: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    lengths = np.array([len(c.ts) for c in reference], dtype=float)
    out = []
    while len(out) < draws:
        c = reference[rng.choice(len(reference), p=lengths / lengths.sum())]
        n = len(c.ts)
        m = int(min(comparator, n - w))
        if m <= 0:
            continue
        # window and comparator are disjoint contiguous segments; order chosen at random
        span = w + m
        start = int(rng.integers(0, n - span + 1))
        if rng.random() < 0.5:
            win, comp = c.can_id[start : start + w], c.can_id[start + w : start + span]
        else:
            comp, win = c.can_id[start : start + m], c.can_id[start + m : start + span]
        out.append(jsd(id_counts(win), id_counts(comp)))
    return np.asarray(out)


def attack_type_metrics(caps: list[Capture], ref_ids: set[int], cfg: dict, null_q: float) -> dict:
    rule = cfg["grouping_rule"]
    w = rule["C2_id_distribution_jsd"]["window_frames"]
    att = np.concatenate([c.period >= 0 for c in caps])
    nor = np.concatenate([(c.period < 0) & (c.period != EXCLUDED) for c in caps])
    ids = np.concatenate([c.can_id for c in caps])
    unseen = ~np.isin(ids, list(ref_ids))
    c1 = float(unseen[att].mean())
    c1_normal = float(unseen[nor].mean()) if nor.any() else float("nan")

    window_jsd = []
    for c in caps:
        normal_counts = id_counts(c.can_id[(c.period < 0) & (c.period != EXCLUDED)])
        window_jsd += [jsd(id_counts(win), normal_counts) for win in attack_windows(c, w)]
    c2 = float(np.median(window_jsd))

    gap = cfg["periods"]["recording_gap_s"]
    dur_a = sum(period_durations(c.ts, c.period, True, gap) for c in caps)
    dur_n = sum(period_durations(c.ts, c.period, False, gap) for c in caps)
    bus_ratio = (att.sum() / dur_a) / (nor.sum() / dur_n)

    target = caps[0].target_id
    if target is not None:
        tgt = ids == target
        rate_a, rate_n = tgt[att].sum() / dur_a, tgt[nor].sum() / dur_n
        c3 = float(rate_a / rate_n) if rate_n > 0 else float("inf")
        iat_a = np.concatenate([target_iat(c.ts, c.period, c.can_id, target, True) for c in caps])
        iat_n = np.concatenate([target_iat(c.ts, c.period, c.can_id, target, False) for c in caps])
        c4 = float(np.median(iat_a) / np.median(iat_n)) if len(iat_n) and len(iat_a) else float("inf")
        pay_sel_a = [c.data[(c.can_id == target) & (c.period >= 0)] for c in caps]
        pay_sel_n = [c.data[(c.can_id == target) & (c.period < 0) & (c.period != EXCLUDED)] for c in caps]
    else:
        c3 = c4 = float("nan")
        pay_sel_a = [c.data[c.period >= 0] for c in caps]
        pay_sel_n = [c.data[(c.period < 0) & (c.period != EXCLUDED)] for c in caps]

    flips_a = np.nanmean([bit_flip_rate(d) for d in pay_sel_a if len(d) > 1])
    flips_n = np.nanmean([bit_flip_rate(d) for d in pay_sel_n if len(d) > 1])
    pa_, pn_ = np.concatenate(pay_sel_a), np.concatenate(pay_sel_n)
    std_a = pa_.astype(float).std(axis=0) if len(pa_) else np.full(8, np.nan)
    std_n = pn_.astype(float).std(axis=0) if len(pn_) else np.full(8, np.nan)
    q_a = np.quantile(pa_, [0.05, 0.5, 0.95], axis=0) if len(pa_) else np.full((3, 8), np.nan)
    q_n = np.quantile(pn_, [0.05, 0.5, 0.95], axis=0) if len(pn_) else np.full((3, 8), np.nan)
    # Largest per-byte shift of the median value, attack vs normal period (level change of a byte).
    median_shift = float(np.nanmax(np.abs(q_a[1] - q_n[1]))) if len(pa_) and len(pn_) else float("nan")

    lo3, hi3 = rule["C3_target_rate_ratio"]["band"]
    lo4, hi4 = rule["C4_target_median_iat_ratio"]["band"]
    triggered = []
    if c1 > rule["C1_unseen_id_share"]["threshold"]:
        triggered.append("C1")
    if c2 > null_q:
        triggered.append("C2")
    if target is not None and not (lo3 <= c3 <= hi3):
        triggered.append("C3")
    if target is not None and not (lo4 <= c4 <= hi4):
        triggered.append("C4")
    return {
        "n_captures": len(caps),
        "captures": ";".join(c.name for c in caps),
        "n_injected_frames": int(sum(c.label.sum() for c in caps)),
        "attack_period_s": round(dur_a, 3),
        "normal_period_s": round(dur_n, 3),
        "attack_period_frames": int(att.sum()),
        "normal_period_frames": int(nor.sum()),
        "target_id": f"0x{target:03X}" if target is not None else "",
        "C1_unseen_id_share": c1,
        "C1_unseen_id_share_normal_period": c1_normal,
        "C2_jsd_median": c2,
        "C2_null_q": null_q,
        "C2_n_windows": len(window_jsd),
        "C3_target_rate_ratio": c3,
        "C4_target_median_iat_ratio": c4,
        "bus_rate_ratio": float(bus_ratio),
        "payload_scope": "target_id" if target is not None else "all_frames",
        "payload_bitflip_normal": float(flips_n),
        "payload_bitflip_attack": float(flips_a),
        "payload_byte_std_normal": json.dumps(np.round(std_n, 2).tolist()),
        "payload_byte_std_attack": json.dumps(np.round(std_a, 2).tolist()),
        "payload_byte_q05_q50_q95_normal": json.dumps(np.round(q_n, 1).tolist()),
        "payload_byte_q05_q50_q95_attack": json.dumps(np.round(q_a, 1).tolist()),
        "payload_max_byte_median_shift": median_shift,
        "triggered": ",".join(triggered),
        "proposed_group": "coarse" if triggered else "stealthy",
    }


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100, "hatch.linewidth": 0.6,
    })


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(x[x > 0])
    return x, np.arange(1, len(x) + 1) / len(x)


def signature_figure(dataset: str, attack: str, caps: list[Capture], ref_ids: set[int], metrics: dict) -> None:
    att = np.concatenate([c.period >= 0 for c in caps])
    nor = np.concatenate([(c.period < 0) & (c.period != EXCLUDED) for c in caps])
    ids = np.concatenate([c.can_id for c in caps])
    target = caps[0].target_id

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.1), gridspec_kw={"width_ratios": [1.6, 1, 1]})

    # (a) ID distribution: 14 most frequent seen IDs + one bin for IDs unseen in reference traffic
    ax = axes[0]
    share_n = id_counts(ids[nor]) / nor.sum()
    share_a = id_counts(ids[att]) / att.sum()
    seen = np.array(sorted(ref_ids))
    top = seen[np.argsort(-np.maximum(share_n[seen], share_a[seen]))[:14]]
    top = np.sort(top)
    unseen_mask = np.ones(ID_SPACE, bool)
    unseen_mask[seen] = False
    vals_n = np.r_[share_n[top], share_n[unseen_mask].sum()]
    vals_a = np.r_[share_a[top], share_a[unseen_mask].sum()]
    labels = [f"{i:03X}" for i in top] + ["unseen"]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, vals_n, 0.38, color=COLOR_NORMAL, label="normal period")
    ax.bar(x + 0.2, vals_a, 0.38, color=COLOR_ATTACK, hatch="////", edgecolor="white", linewidth=0, label="attack period")
    ax.set_xticks(x, labels, rotation=90, fontsize=6)
    ax.set_ylabel("share of frames")
    ax.set_title("(a) ID distribution", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # (b) IAT ECDF: target ID if the attack has one, else the whole bus
    ax = axes[1]
    if target is not None:
        iat_n = np.concatenate([target_iat(c.ts, c.period, c.can_id, target, False) for c in caps])
        iat_a = np.concatenate([target_iat(c.ts, c.period, c.can_id, target, True) for c in caps])
        title = f"(b) IAT of ID {target:03X}"
    else:
        iat_n = np.concatenate([np.diff(c.ts)[((c.period < 0) & (c.period != EXCLUDED))[1:] & ((c.period < 0) & (c.period != EXCLUDED))[:-1]] for c in caps])
        iat_a = np.concatenate([np.diff(c.ts)[(c.period >= 0)[1:] & (c.period >= 0)[:-1]] for c in caps])
        title = "(b) IAT, whole bus"
    for vals, color, ls, lab in ((iat_n, COLOR_NORMAL, "-", "normal"), (iat_a, COLOR_ATTACK, "--", "attack")):
        if len(vals):
            xs, ys = ecdf(vals)
            ax.step(xs * 1e3, ys, where="post", color=color, linestyle=ls, linewidth=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("inter-arrival time (ms)")
    ax.set_ylabel("ECDF")
    ax.set_title(title, loc="left")

    # (c) payload variability per byte
    ax = axes[2]
    q_n = np.array(json.loads(metrics["payload_byte_q05_q50_q95_normal"]), dtype=float)
    q_a = np.array(json.loads(metrics["payload_byte_q05_q50_q95_attack"]), dtype=float)
    b = np.arange(8)
    for q, off, color, marker, ls in ((q_n, -0.17, COLOR_NORMAL, "o", "-"), (q_a, 0.17, COLOR_ATTACK, "s", "--")):
        if np.isnan(q).all():
            continue
        ax.vlines(b + off, q[0], q[2], color=color, linewidth=2.0, linestyles=ls)
        ax.scatter(b + off, q[1], color=color, marker=marker, s=12, zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_xticks(b)
    ax.set_ylim(-8, 263)
    ax.set_xlabel("payload byte")
    ax.set_ylabel("byte value (p5-p95, median)")
    scope = f"ID {target:03X}" if target is not None else "all frames"
    ax.set_title(f"(c) payload bytes, {scope}", loc="left")

    fig.suptitle(f"{dataset} / {attack}  (rule: {metrics['triggered'] or 'none'} -> {metrics['proposed_group']})",
                 x=0.01, ha="left", fontsize=8, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, f"signature_{dataset}_{attack}")


def summary_figure(sig: pd.DataFrame, cfg: dict, stem: str = "attack_signature_summary",
                   figsize: tuple[float, float] = (7.16, 3.6), note_size: float = 6, legend_size: float = 7,
                   save_fn=None, label_fn=None, styles: dict | None = None,
                   xticks: dict[int, list[float]] | None = None) -> None:
    """C1-C4 per attack type. The keyword arguments let analysis/figures.py redraw it in the paper style:
    ``label_fn(dataset, attack_type)`` row labels, ``styles`` scatter kwargs per group ("coarse", "stealthy"),
    ``xticks`` fixed ticks per panel index."""
    rule = cfg["grouping_rule"]
    sig = sig.sort_values(["dataset", "attack_type"]).reset_index(drop=True)
    label_fn = label_fn or (lambda d, a: f"{'CH' if d == 'car_hacking' else 'ROAD'}: {a}")
    labels = [label_fn(d, a) for d, a in zip(sig.dataset, sig.attack_type)]
    styles = styles or {"coarse": dict(marker="s", s=20, color=INK_2),
                        "stealthy": dict(marker="o", s=20, facecolors="none", edgecolors=INK, linewidths=1.0)}
    y = np.arange(len(sig))[::-1]
    coarse = sig.proposed_group.eq("coarse").to_numpy()

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    panels = [
        ("C1: unseen-ID\nframe share", sig.C1_unseen_id_share,
         ("line", rule["C1_unseen_id_share"]["threshold"]), "symlog"),
        ("C2: window JSD /\nnull q99", sig.C2_jsd_median / sig.C2_null_q, ("line", 1.0), "log"),
        ("C3: target-ID\nrate ratio", sig.C3_target_rate_ratio,
         ("band", rule["C3_target_rate_ratio"]["band"]), "log"),
        ("C4: target-ID\nmedian IAT ratio", sig.C4_target_median_iat_ratio,
         ("band", rule["C4_target_median_iat_ratio"]["band"]), "log"),
    ]
    for i_panel, (ax, (title, vals, (kind, thr), scale)) in enumerate(zip(axes, panels)):
        v = vals.to_numpy(dtype=float)
        if kind == "band":
            ax.axvspan(thr[0], thr[1], color=GRID, alpha=0.7, linewidth=0)
        else:
            ax.axvline(thr, color=INK_2, linewidth=0.8, linestyle=":")
        finite = np.isfinite(v)
        cap = np.nanmax(v[finite]) * 3 if finite.any() else 1.0
        plot_v = np.where(np.isinf(v), cap, v)
        ok = ~np.isnan(v)
        ax.scatter(plot_v[coarse & ok], y[coarse & ok], label="coarse (rule)", **styles["coarse"])
        ax.scatter(plot_v[~coarse & ok], y[~coarse & ok], label="stealthy (rule)", **styles["stealthy"])
        for yi, vi in zip(y[np.isinf(v)], plot_v[np.isinf(v)]):
            ax.annotate("inf", (vi, yi), xytext=(0, 5), textcoords="offset points", fontsize=note_size, color=INK_2, ha="center")
        for yi in y[np.isnan(v)]:
            ax.annotate("n/a", (0.5, yi), xycoords=("axes fraction", "data"), fontsize=note_size, color=INK_2,
                        ha="center", va="center")
        if scale == "symlog":
            ax.set_xscale("symlog", linthresh=1e-3)
        else:
            ax.set_xscale("log")
        if xticks and i_panel in xticks:
            ax.set_xticks(xticks[i_panel])
        ax.set_title(title, loc="left")
    axes[0].set_yticks(y, labels)
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, loc="lower center", ncol=2, fontsize=legend_size, bbox_to_anchor=(0.6, 0.0))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.05, 1, 1))
    (save_fn or save)(fig, stem)


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def profile_row(cap: Capture) -> dict:
    r = cap.row
    n = len(cap.ts)
    n_att = int(cap.label.sum())
    return {
        "dataset": r["dataset"],
        "capture": r["capture"],
        "source_capture": r["source_capture"],
        "kind": r["kind"],
        "attack_type": cap.attack_type if n_att else ("none" if r["kind"] != "unlabeled" else "unlabeled (accelerator)"),
        "n_messages": n,
        "n_normal": n - n_att,
        "n_attack": n_att,
        "attack_ratio": n_att / n,
        "n_unique_ids": int(np.unique(cap.can_id).size),
        "duration_s": float(cap.ts[-1] - cap.ts[0]),
        "msg_rate_per_s": n / float(cap.ts[-1] - cap.ts[0]),
        "excluded": r["excluded"],
    }


def iat_rows(dataset: str, refs: list[Capture], cfg: dict) -> list[dict]:
    cv_thr = cfg["periodicity"]["cv_threshold"]
    min_n = cfg["periodicity"]["min_frames"]
    gap = cfg["periods"]["recording_gap_s"]
    bus_gaps = []
    pieces = []
    for c in refs:
        d = np.diff(c.ts)
        bus_gaps.append(np.c_[c.ts[:-1][d > gap], c.ts[1:][d > gap]])
        df = pd.DataFrame({"id": c.can_id, "t": c.ts})
        df["iat"] = df.groupby("id")["t"].diff()
        df["prev"] = df["t"] - df["iat"]
        df = df.dropna(subset=["iat"])
        # flag ID intervals that span a bus-wide recording gap (no frame at all for > gap seconds)
        g = bus_gaps[-1]
        if len(g):
            k = np.searchsorted(g[:, 0], df["t"].to_numpy(), side="right") - 1
            kk = np.maximum(k, 0)
            t_now = df["t"].to_numpy()
            spans = (k >= 0) & (df["prev"].to_numpy() <= g[kk, 0]) & (t_now >= g[kk, 1])
        else:
            spans = np.zeros(len(df), bool)
        df["spans_gap"] = spans
        pieces.append(df)
    allf = pd.concat(pieces)
    raw = allf.groupby("id")["iat"].agg(cv_raw=lambda s: s.std() / s.mean())
    clean = allf[~allf["spans_gap"]]
    g = clean.groupby("id")["iat"]
    stats = g.agg(n="size", mean="mean", std="std", median="median",
                  p05=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95)).join(raw).reset_index()
    stats["n_gap_spanning_excluded"] = stats["id"].map(allf.groupby("id")["spans_gap"].sum()).astype(int)
    stats["cv"] = stats["std"] / stats["mean"]
    stats["robust_spread"] = (stats["p95"] - stats["p05"]) / stats["median"]
    stats["periodic"] = (stats["cv"] < cv_thr) & (stats["n"] >= min_n)
    stats.insert(0, "dataset", dataset)
    stats["can_id"] = stats.pop("id").map(lambda i: f"0x{i:03X}")
    cols = ["dataset", "can_id", "n", "n_gap_spanning_excluded", "mean", "std", "median", "p05", "p95",
            "cv_raw", "cv", "robust_spread", "periodic"]
    return stats[cols].to_dict("records")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")


def main() -> int:
    global RECORDING_GAP_S
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    RECORDING_GAP_S = cfg["periods"]["recording_gap_s"]
    setup_matplotlib()
    index = S.all_views()
    road_meta = {c.name: c.meta for c in L.road_captures()}
    rng = np.random.default_rng(cfg["grouping_rule"]["C2_id_distribution_jsd"]["seed"])

    profile, iat_all, signatures, period_summary = [], [], [], []
    for dataset in ("car_hacking", "road"):
        rows = [v for v in index if v.dataset == dataset]
        caps = [load_capture(r) for r in rows]
        profile += [profile_row(c) for c in caps]
        print(f"{dataset}: loaded {len(caps)} captures", flush=True)

        ref_names = cfg["reference_traffic"][dataset]
        refs = [c for c in caps if (c.row["kind"] == "ambient" if ref_names == "ambient" else c.name in ref_names)]
        ref_ids = set(np.unique(np.concatenate([c.can_id for c in refs])).tolist())
        iat = iat_rows(dataset, refs, cfg)
        iat_all += iat
        per = pd.DataFrame(iat)
        period_summary.append({
            "dataset": dataset, "n_ids": len(per), "n_periodic": int(per.periodic.sum()),
            "share_ids_periodic": float(per.periodic.mean()),
            "share_frames_periodic": float(per.loc[per.periodic, "n"].sum() / per.n.sum()),
            "median_cv": float(per.cv.median()),
            "n_periodic_raw_cv": int(((per.cv_raw < cfg["periodicity"]["cv_threshold"]) & (per.n >= cfg["periodicity"]["min_frames"])).sum()),
            "median_cv_raw": float(per.cv_raw.median()),
            "n_robust_spread_lt_threshold": int((per.robust_spread < cfg["periodicity"]["cv_threshold"]).sum()),
            "share_frames_robust_spread_lt_threshold": float(
                per.loc[per.robust_spread < cfg["periodicity"]["cv_threshold"], "n"].sum() / per.n.sum()),
        })

        attack_caps = [c for c in caps if c.label.any()]
        for c in attack_caps:
            assign_periods(c, cfg, road_meta)
        normal_frames = [int(((c.period < 0) & (c.period != EXCLUDED)).sum()) for c in attack_caps]
        c2cfg = cfg["grouping_rule"]["C2_id_distribution_jsd"]
        null = null_jsd(refs, c2cfg["window_frames"], int(np.median(normal_frames)), c2cfg["null_draws"], rng)
        null_q = float(np.quantile(null, c2cfg["null_quantile"]))
        print(f"{dataset}: null JSD q{c2cfg['null_quantile']} = {null_q:.4f} "
              f"(comparator {int(np.median(normal_frames))} frames)", flush=True)

        by_type: dict[str, list[Capture]] = {}
        for c in attack_caps:
            by_type.setdefault(c.attack_type, []).append(c)
        for attack, group in sorted(by_type.items()):
            m = attack_type_metrics(group, ref_ids, cfg, null_q)
            row = {"dataset": dataset, "attack_type": attack, **m, "plan_group": plan_group(attack)}
            row["agrees_with_plan"] = row["proposed_group"] == row["plan_group"]
            signatures.append(row)
            signature_figure(dataset, attack, group, ref_ids, m)
            print(f"  {attack}: {m['triggered'] or '-'} -> {m['proposed_group']}", flush=True)

    write_csv(profile, TABLES / "data_profile.csv")
    write_csv(iat_all, TABLES / "iat_by_id.csv")
    write_csv(period_summary, TABLES / "periodicity_summary.csv")
    write_csv(signatures, TABLES / "attack_signatures.csv")
    summary_figure(pd.DataFrame(signatures), cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
