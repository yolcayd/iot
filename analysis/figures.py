"""Figures and tables for the paper (P7, plan F6; presentation decisions: docs/DECISIONS.md K-065).

Reads only results/tables (the P6/P1 outputs) and configs/; nothing is trained, re-scored or bootstrapped, so
every number drawn here is a number already in a CSV.

* Figures (results/figures/fig<N>_<name>.{pdf,png}; vector PDF + 300 dpi PNG, IEEE widths 3.5" / 7.16", all
  text 8-9 pt, Okabe-Ito colours always paired with marker shape or fill):
    1 system overview; 2 attack signatures (C1-C4, drawn by analysis.eda.summary_figure);
    3 MAIN: dRecall of coarse vs stealthy types + paired fabrication-masquerade difference (H2 not supported);
    4 recall heat map (attack type x model x level, colour = dRecall); 5 Pareto (latency range of the two host
    sessions, both latency budgets); 6 size and latency ratios vs Q0; 7 supervised vs unsupervised (the three
    stages of the H4 verdict, K-062 -> K-063 -> K-064); 8 calibration shift (Q0 threshold applied to Qk).
* Tables (results/tables/table_<N>_<name>.{csv,tex}; LaTeX needs only booktabs):
    I dataset; II recall mean +- std (and IIb F1); III resources; IV statistics.
* Every figure passes ``layout_problems`` (overlapping or clipped text, text outside 8-9 pt, empty panels)
  before it is written. Provenance (rule 7) goes to results/tables/analysis_manifest.json ("figures") and the
  git commit into the PDF/PNG metadata.

Usage:
    python -m analysis.figures
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")  # the machine is shared (CLAUDE.md: 4 CPU threads)

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory

from analysis import aggregate as A
from analysis import eda
from src import train as T
from src.data import features as F
from src.data import sources as S

ROOT = S.PROJECT_ROOT
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
CONFIGS = ROOT / "configs"

SINGLE, DOUBLE = 3.5, 7.16              # IEEE column widths (inch)
FONT_MIN, FONT_MAX = 8.0, 9.0           # every text in a figure
DPI = 300

INK, INK_2, GRID, FILL = "#0b0b0b", "#52514e", "#d9d8d4", "#f3f2ef"
FLOOR = "#a3a29d"                       # types whose Q0 recall is below the K-041 floor
C_GROUP = {"coarse": "#D55E00", "stealthy": "#0072B2"}       # validated pair (dataviz validator: all PASS)
M_GROUP = {"coarse": "s", "stealthy": "o"}
FILLED_GROUP = {"coarse": True, "stealthy": False}
C_MODEL = {"cnn1d": "#009E73", "conv_ae": "#AA3377", "xgb": INK_2}   # validated pair + neutral reference
M_MODEL = {"cnn1d": "o", "conv_ae": "s", "xgb": "D"}

DS = {"car_hacking": "CH", "road": "ROAD"}
DS_LONG = {"car_hacking": "Car-Hacking", "road": "ROAD"}
MODEL = {"cnn1d": "1D-CNN", "conv_ae": "conv-AE", "xgb": "XGBoost"}
LEVEL = {"q0": "Q0 (fp32)", "q1": "Q1 (fp16)", "q2": "Q2 (INT8)", "native": "native"}
QUANT = ("q0", "q1", "q2")
DEEP = ("cnn1d", "conv_ae")
BASE_NAME = {"dos": "DoS", "fuzzy": "fuzzy", "gear_spoofing": "gear spoofing", "rpm_spoofing": "RPM spoofing",
             "fuzzing": "fuzzing", "correlated_signal": "correlated signal", "max_speedometer": "max speedometer",
             "reverse_light_off": "reverse light off", "reverse_light_on": "reverse light on",
             "max_engine_coolant_temp": "max coolant temp"}
MASQ = "_masquerade"


class LayoutError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


class Data:
    """Every table and config the figures read (sha256 recorded through aggregate.Inputs)."""

    CSVS = ("all_results", "delta_f1", "stats", "data_profile",       # the four sources of plan F6
            "pareto", "split_summary", "attack_signatures")            # P6/P1 derived tables (K-065 item 10)
    YAMLS = ("analysis", "data", "quantization", "models", "eda")

    def __init__(self, tables: Path = TABLES, configs: Path = CONFIGS):
        self.inp = A.Inputs()
        for name in self.CSVS:
            setattr(self, name, self.inp.csv(tables / f"{name}.csv"))
        self.cfg = {}
        for name in self.YAMLS:
            path = configs / f"{name}.yaml"
            self.inp._add(path)
            self.cfg[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.floor = float(self.cfg["analysis"]["as1"]["floor"])
        self.check_consistency()

    def check_consistency(self) -> None:
        """Per-type delta estimates in stats.csv must be the seed means of delta_f1.csv (they are two P6 outputs)."""
        st = self.stats[self.stats.analysis.eq("per_type_delta") & self.stats.statistic.eq("d_recall")]
        dd = self.type_deltas("recall").groupby(["model", "dataset", "quantization", "attack_type"]).delta.mean()
        for r in st.itertuples():
            ref = dd.loc[(r.model, r.dataset, r.quantization, r.unit)]
            if abs(ref - r.estimate) > 1e-6:
                raise ValueError(f"stats.csv vs delta_f1.csv: {r.model} {r.dataset} {r.quantization} {r.unit}")

    def type_deltas(self, metric: str, mode: str = "recalibrated") -> pd.DataFrame:
        d = self.delta_f1
        return d[d.level.eq("attack_type") & d.threshold_mode.eq(mode) & d.metric.eq(metric)]

    def type_meta(self) -> pd.DataFrame:
        """dataset, attack_type, group, role (from the delta table) in the display order of the paper."""
        m = (self.delta_f1[self.delta_f1.level.eq("attack_type")][["dataset", "attack_type", "attack_group", "role"]]
             .drop_duplicates().rename(columns={"attack_group": "group"}))
        m["base"] = m.attack_type.str.removesuffix(MASQ)
        m["key"] = list(zip(m.dataset.ne("car_hacking"), m.role.ne("primary"), m.group.ne("coarse"),
                            m.base.ne("fuzzing"), m.base))
        return m.sort_values("key").drop(columns=["key", "base"]).reset_index(drop=True)

    def metric_means(self, metric: str, split: str = "test") -> pd.DataFrame:
        """Seed mean / std of a test metric per (dataset, model, quantization, level, attack_type).
        Quantized models: own validation threshold (recalibrated, K-036 a); XGB: its calibrated threshold."""
        a = self.all_results
        own = (a.model.isin(DEEP) & a.threshold_mode.eq("recalibrated")) | (
            a.model.eq("xgb") & a.threshold_mode.eq(A.CALIBRATED))
        m = a[own & a.split.eq(split) & a.metric.eq(metric) & a.level.isin(["attack_type", "pooled"])]
        return (m.groupby(["dataset", "model", "quantization", "level", "attack_type"]).value
                .agg(["mean", "std", "count"]).reset_index())

    def q0_recall(self) -> pd.Series:
        m = self.metric_means("recall")
        m = m[m.quantization.eq("q0") & m.level.eq("attack_type")]
        return m.set_index(["model", "dataset", "attack_type"])["mean"]

    def model_metric(self, metric: str) -> pd.DataFrame:
        a = self.all_results
        return a[a.level.eq("model") & a.metric.eq(metric)]

    def stat(self, analysis: str, **kw) -> pd.DataFrame:
        s = self.stats[self.stats.analysis.eq(analysis)]
        for k, v in kw.items():
            s = s[s[k].eq(v)]
        return s


def type_label(attack_type: str) -> str:
    base = attack_type.removesuffix(MASQ)
    return BASE_NAME.get(base, base.replace("_", " ")) + (" (masq.)" if attack_type.endswith(MASQ) else "")


def seeds(s: str) -> np.ndarray:
    return np.array([float(v) for v in str(s).split(";")]) if isinstance(s, str) and s else np.array([])


def um(text: str) -> str:
    """Typographic minus for figure text."""
    return text.replace("-", "−")


def num(x: float, nd: int = 4) -> str:
    return um(f"{x:.{nd}f}")


# --------------------------------------------------------------------------------------------
# Style, layout check, saving
# --------------------------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8,
        "figure.titlesize": 9, "mathtext.default": "regular",
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.titlelocation": "left", "lines.linewidth": 1.0,
        "hatch.linewidth": 0.6, "legend.frameon": False, "legend.handlelength": 1.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": DPI, "figure.dpi": 100,
        "axes.unicode_minus": True,
    })


def _drawn_texts(fig: plt.Figure) -> list[Text]:
    """The text artists that are actually drawn (tick labels only for ticks inside the view limits)."""
    out: list[Text] = list(fig.texts)
    if getattr(fig, "_suptitle", None) is not None:
        out.append(fig._suptitle)
    legends = list(fig.legends)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        out += [ax.title, ax._left_title, ax._right_title, *ax.texts]
        if ax.get_legend() is not None:
            legends.append(ax.get_legend())
        if not ax.axison:
            continue
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            out += [axis.label, axis.get_offset_text()]
            lo, hi = sorted(lim)
            tol = 1e-9 * max(abs(hi - lo), 1e-30)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if lo - tol <= tick.get_loc() <= hi + tol:
                    out += [tick.label1, tick.label2]
    for leg in legends:
        out += [*leg.get_texts(), leg.get_title()]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def layout_problems(fig: plt.Figure) -> list[str]:
    """Overlapping text, text outside the figure, font size outside 8-9 pt, panels without data."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    probs, boxes = [], []
    for t in _drawn_texts(fig):
        label = t.get_text().replace("\n", " ")[:40]
        if not FONT_MIN - 1e-6 <= t.get_fontsize() <= FONT_MAX + 1e-6:
            probs.append(f"font {t.get_fontsize():.1f} pt: {label!r}")
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            probs.append(f"outside figure: {label!r}")
        boxes.append((label, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1.0 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.0:
                probs.append(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")
    for ax in fig.axes:
        if ax.get_visible() and not (ax.lines or ax.collections or ax.patches or ax.images or ax.texts
                                     or ax.containers):
            probs.append(f"empty panel: {ax.get_title() or ax.get_label()!r}")
    return probs


class Saver:
    def __init__(self, out: Path, commit: str, draft: bool = False):
        self.out, self.commit, self.written, self.draft = Path(out), commit, [], draft

    def __call__(self, fig: plt.Figure, stem: str) -> None:
        probs = layout_problems(fig)
        if probs and self.draft:                      # inspection only; never used for the final outputs
            print(f"LAYOUT {stem}: " + "; ".join(probs[:12]))
        elif probs:
            plt.close(fig)
            raise LayoutError(f"{stem}: " + "; ".join(probs))
        self.out.mkdir(parents=True, exist_ok=True)
        pdf, png = self.out / f"{stem}.pdf", self.out / f"{stem}.png"
        fig.savefig(pdf, metadata={"Title": stem, "Subject": f"git {self.commit}",
                                   "Creator": "analysis/figures.py", "CreationDate": None})
        fig.savefig(png, dpi=DPI, metadata={"Title": stem, "Description": f"git {self.commit}"})
        plt.close(fig)
        self.written += [pdf, png]


def errbar_v(ax, x, lo, hi, color, lw=0.9, z=2):
    ax.vlines(x, lo, hi, color=color, linewidth=lw, zorder=z)


def errbar_h(ax, y, lo, hi, color, lw=0.9, z=2):
    ax.hlines(y, lo, hi, color=color, linewidth=lw, zorder=z)


def group_marker(ax, x, y, group, floor=False, s=22, z=3, **kw):
    c = FLOOR if floor else C_GROUP[group]
    ax.scatter(x, y, s=s, marker=M_GROUP[group], facecolors=c if FILLED_GROUP[group] else "white",
               edgecolors=c, linewidths=1.0, zorder=z, **kw)


# --------------------------------------------------------------------------------------------
# Figure 1: system overview
# --------------------------------------------------------------------------------------------


def fig1_system(d: Data, save) -> None:
    w = d.cfg["data"]["windows"]
    q = d.cfg["quantization"]
    fpr = d.cfg["models"]["decision"]["target_fpr"]
    threads = q["benchmark"]["profiles"]["host"]["threads"]
    boxes = [
        ("CAN captures", "Car-Hacking: 4 attack files\n+ attack-free recordings\nROAD: ambient, targeted,\n"
                         "fuzzing, masquerade"),
        ("Leakage-safe split", "train / val / test by source\ncapture (ROAD) or by time\nblock (Car-Hacking); "
                               "no file\nfeeds two splits"),
        (f"Per-frame features ({F.N_FEATURES})", "ID code, 8 payload bytes,\nlog inter-arrival time;\n"
                                                 "scaling fitted on\ntraining frames only"),
        (f"Windows (N = {w['size']})", f"{w['size']} frames per window\ntrain stride {w['stride']['train']} (overlap)\n"
                                              f"val / test stride {w['stride']['test']}\n(non-overlapping)"),
        ("Models", "1D-CNN (supervised)\nconv-AE (unsupervised,\nnormal windows only)\nXGBoost (reference)"),
        ("Quantization (PTQ)","Q0 fp32, Q1 fp16, Q2 INT8\nINT8 calibration:\n"
                                       f"{q['q2']['calibration_windows']} training windows"),
        ("Threshold per version", f"val attack-free windows,\nFPR = {fpr:.0%}; same rule\nfor Q0, Q1 and Q2"),
        ("Evaluation", "test: recall, F1, ROC-AUC\nper attack type, Q0 − Qk\n"
                       f"latency: host CPU,\n{threads} threads, batch 1"),
    ]
    W, Hh = DOUBLE, 2.45
    fig = plt.figure(figsize=(W, Hh))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, Hh)
    ax.axis("off")
    bw, bh, gap, x0 = 1.62, 1.02, 0.2, 0.04
    rows_y = (Hh - 0.08 - bh, 0.08)
    pos = []
    for i in range(8):
        row, col = divmod(i, 4)
        col = col if row == 0 else 3 - col          # second row runs right to left (snake)
        pos.append((x0 + col * (bw + gap), rows_y[row]))
    for (title, body), (x, y) in zip(boxes, pos):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=FILL, edgecolor=INK_2, linewidth=0.7))
        ax.text(x + 0.07, y + bh - 0.07, title, fontsize=8, fontweight="bold", va="top", ha="left")
        ax.text(x + 0.07, y + bh - 0.25, body, fontsize=8, va="top", ha="left", color=INK, linespacing=1.15)
    arrow = dict(arrowstyle="-|>", mutation_scale=8, color=INK_2, linewidth=0.8, shrinkA=0, shrinkB=0)
    for i in range(7):
        (xa, ya), (xb, yb) = pos[i], pos[i + 1]
        if ya == yb:
            right = xb > xa
            start = (xa + (bw if right else 0), ya + bh / 2)
            end = (xb + (0 if right else bw), yb + bh / 2)
        else:
            start, end = (xa + bw / 2, ya), (xb + bw / 2, yb + bh)
        ax.add_patch(FancyArrowPatch(start, end, **arrow))
    save(fig, "fig1_system")


# --------------------------------------------------------------------------------------------
# Figure 2: attack signatures (drawn by analysis.eda)
# --------------------------------------------------------------------------------------------


def fig2_signatures(d: Data, save) -> None:
    styles = {g: dict(marker=M_GROUP[g], s=20, facecolors=C_GROUP[g] if FILLED_GROUP[g] else "white",
                      edgecolors=C_GROUP[g], linewidths=1.0) for g in C_GROUP}
    eda.summary_figure(d.attack_signatures, d.cfg["eda"], stem="fig2_attack_signatures",
                       figsize=(DOUBLE, 3.9), note_size=8, legend_size=8, save_fn=save,
                       label_fn=lambda ds, t: f"{DS[ds]}: {type_label(t)}", styles=styles,
                       xticks={0: [0, 1e-2, 1]})


# --------------------------------------------------------------------------------------------
# Figure 3 (main): dRecall of coarse vs stealthy types; paired difference D
# --------------------------------------------------------------------------------------------


def fig3_main(d: Data, save) -> None:
    meta = d.type_meta()
    meta = meta[meta.dataset.eq("road") & meta.role.eq("primary")]
    q0 = d.q0_recall()
    pt = d.stat("per_type_delta", statistic="d_recall", dataset="road")
    fig = plt.figure(figsize=(DOUBLE, 3.25), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.95, 1.3])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    titles = {"cnn1d": "(a) 1D-CNN (supervised)", "conv_ae": "(b) conv-AE (unsupervised)"}
    for ax, model in zip(axes[:2], DEEP):
        ticklabels = []
        for xi, k in enumerate(("q1", "q2")):
            for group, off in (("coarse", -0.22), ("stealthy", 0.22)):
                types = meta[meta.group.eq(group)].attack_type.tolist()
                n = len(types)
                vals = []
                for j, t in enumerate(types):
                    r = pt[pt.model.eq(model) & pt.quantization.eq(k) & pt.unit.eq(t)].iloc[0]
                    x = xi + off + (j - (n - 1) / 2) * 0.07
                    fl = q0.loc[(model, "road", t)] < d.floor
                    errbar_v(ax, x, r.ci_low, r.ci_high, FLOOR if fl else C_GROUP[group])
                    group_marker(ax, x, r.estimate, group, floor=fl)
                    vals.append(r.estimate)
                ax.hlines(np.mean(vals), xi + off - 0.19, xi + off + 0.19, color=C_GROUP[group], linewidth=1.8,
                          zorder=4)
            g = d.stat("group_mwu", model=model, quantization=k, holm_family="group_road_drecall").iloc[0]
            ticklabels.append(um(f"{LEVEL[k]}\nδ = {g.estimate:.2f}\n$p_{{Holm}}$ = {g.p_holm:.2f}"))
        ax.axhline(0, color=INK_2, linewidth=0.7, linestyle="--", zorder=1)
        ax.set_xticks([0, 1], ticklabels)
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(titles[model])
        ax.grid(axis="y")
    axes[0].set_ylabel("ΔRecall = Recall(Q0) − Recall(Qk)\n(recall scale 0–1; + = loss)")

    # (c) paired difference D = dR_masq - dR_fab, CNN (K-057)
    ax = axes[2]
    pairs = d.stat("as1_pair", model="cnn1d", statistic="D_R", block="block")
    primary = pairs[pairs.note.eq("primary unit")].unit.drop_duplicates().tolist()
    floor_units = pairs[pairs.note.str.contains("floor")].unit.drop_duplicates().tolist()
    rows = [(type_label(u), "pair", u, INK) for u in primary]
    rows.append((f"mean of {len(primary)} pairs", "mean", None, INK))
    rows += [(type_label(u) + "†", "pair", u, FLOOR) for u in floor_units]
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[len(primary) + 1:] -= 0.35                      # gap before the floor unit(s)
    for (label, kind, unit, color), y in zip(rows, ys):
        for k, dy, filled in (("q1", 0.17, False), ("q2", -0.17, True)):
            if kind == "mean":
                r = d.stat("as1_pooled", model="cnn1d", quantization=k, statistic="mean_D_R", block="block").iloc[0]
            else:
                r = pairs[pairs.unit.eq(unit) & pairs.quantization.eq(k)].iloc[0]
            errbar_h(ax, y + dy, r.ci_low, r.ci_high, color)
            ax.scatter(seeds(r.seed_values), np.full(3, y + dy), marker="|", s=20, color=INK_2 if color == INK
                       else FLOOR, linewidths=0.8, zorder=3)
            ax.scatter(r.estimate, y + dy, marker="D", s=20, facecolors=color if filled else "white",
                       edgecolors=color, linewidths=1.0, zorder=4)
    ax.axhline(ys[len(primary)] + 0.5, color=GRID, linewidth=0.7)
    ax.axvline(0, color=INK_2, linewidth=0.7, linestyle="--", zorder=1)
    ax.set_yticks(ys, [r[0] for r in rows])
    for lab, (_, kind, _, color) in zip(ax.get_yticklabels(), rows):
        lab.set_color(INK if color == INK else INK_2)
        if kind == "mean":
            lab.set_fontweight("bold")
    ax.set_ylim(ys.min() - 0.6, ys.max() + 1.9)
    m2 = d.stat("as1_pooled", model="cnn1d", quantization="q2", statistic="mean_D_R", block="block").iloc[0]
    ax.text(0.03, 0.99, um(f"H2(a) not supported:\nmean D(Q2) = {m2.estimate:.4f}\n"
                           f"[{m2.ci_low:.4f}, {m2.ci_high:.4f}]"),
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.set_xlabel("D = ΔR(masq.) − ΔR(fab.)\n(+ = more loss, stealthy side)")
    ax.set_title("(c) 1D-CNN, paired units")
    ax.grid(axis="x")

    handles = [
        Line2D([], [], marker="s", color=C_GROUP["coarse"], linestyle="none", markersize=5,
               label="coarse type (3-seed mean)"),
        Line2D([], [], marker="o", markerfacecolor="white", color=C_GROUP["stealthy"], linestyle="none",
               markersize=5, label="stealthy type (masquerade)"),
        Line2D([], [], marker="s", color=FLOOR, linestyle="none", markersize=5,
               label=f"Q0 recall < {d.floor:.2f} (floor)†"),
        Line2D([], [], color=INK_2, linewidth=1.8, label="group mean"),
        Line2D([], [], color=INK_2, linewidth=0.9, label="95% bootstrap interval"),
        Line2D([], [], marker="D", markerfacecolor="white", color=INK, linestyle="none", markersize=4.5,
               label="(c) Q1 (fp16)"),
        Line2D([], [], marker="D", color=INK, linestyle="none", markersize=4.5, label="(c) Q2 (INT8)"),
        Line2D([], [], marker="|", color=INK_2, linestyle="none", markersize=6, label="(c) seed value"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.2, handletextpad=0.4)
    save(fig, "fig3_delta_recall_coarse_vs_stealthy")


# --------------------------------------------------------------------------------------------
# Figure 4: recall heat map
# --------------------------------------------------------------------------------------------


def fig4_heatmap(d: Data, save) -> None:
    meta = d.type_meta()
    mm = d.metric_means("recall")
    mm = mm[mm.level.eq("attack_type")].set_index(["dataset", "model", "quantization", "attack_type"])["mean"]
    dl = d.type_deltas("recall").groupby(["dataset", "model", "quantization", "attack_type"]).delta.mean()
    q0 = d.q0_recall()
    cols = [(m, k) for m in DEEP for k in QUANT] + [("xgb", "native")]
    R = np.array([[mm.loc[(r.dataset, m, k, r.attack_type)] for m, k in cols] for r in meta.itertuples()])
    D = np.array([[dl.loc[(r.dataset, m, k, r.attack_type)] if k in ("q1", "q2") else np.nan for m, k in cols]
                  for r in meta.itertuples()])
    lim = np.nanmax(np.abs(D))
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = plt.get_cmap("PuOr_r")
    fig = plt.figure(figsize=(DOUBLE, 3.9), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.022])
    ax, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    rgba = cmap(norm(np.where(np.isnan(D), 0.0, D)))
    rgba[np.isnan(D)] = mcolors.to_rgba(FILL)          # Q0 and XGB: reference cells, no delta
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    nr, nc = R.shape
    for i in range(nr):
        r = meta.iloc[i]
        for j, (m, k) in enumerate(cols):
            ref = R[i, cols.index((m, "q0"))] if m in DEEP else R[i, j]
            fl = ref < d.floor
            dark = not np.isnan(D[i, j]) and abs(norm(D[i, j]) - 0.5) > 0.33
            ax.text(j, i, f"{R[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if dark else (INK_2 if fl else INK), fontstyle="italic" if fl else "normal")
    ax.set_xticks(np.arange(-0.5, nc), minor=True)
    ax.set_yticks(np.arange(-0.5, nr), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(nc), [LEVEL[k] if m in DEEP else "native" for m, k in cols])
    ax.xaxis.tick_top()
    ylabels = [type_label(t) + ("\u2020" if role != "primary" else "") for t, role in zip(meta.attack_type, meta.role)]
    ax.set_yticks(range(nr), ylabels)
    ax.tick_params(axis="y", pad=11, length=0)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for i, g in enumerate(meta.group):
        ax.scatter(-0.012, i, transform=tr, marker=M_GROUP[g], s=16, clip_on=False,
                   facecolors=C_GROUP[g] if FILLED_GROUP[g] else "white", edgecolors=C_GROUP[g], linewidths=1.0)
    for j0, m in ((1, "cnn1d"), (4, "conv_ae"), (6, "xgb")):
        ax.text(j0, -1.3, MODEL[m], ha="center", va="bottom", fontsize=8, fontweight="bold")
    for xv in (2.5, 5.5):
        ax.axvline(xv, color=INK_2, linewidth=0.8)
    bounds = meta.dataset.ne(meta.dataset.shift()).to_numpy().nonzero()[0]
    for b in bounds[1:]:
        ax.axhline(b - 0.5, color=INK_2, linewidth=0.8)
    edges = list(bounds) + [nr]
    for a, b in zip(edges[:-1], edges[1:]):
        ax.text(1.008, (a + b - 1) / 2, DS_LONG[meta.dataset.iloc[a]], transform=tr, rotation=270,
                ha="left", va="center", fontsize=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("\u0394Recall vs own Q0, Q0 \u2212 Qk (+ = loss)")
    cb.outline.set_linewidth(0.5)
    cb.set_ticks([-lim, -lim / 2, 0, lim / 2, lim], labels=[num(v, 3) for v in (-lim, -lim / 2, 0, lim / 2, lim)])
    save(fig, "fig4_recall_heatmap")


# --------------------------------------------------------------------------------------------
# Figure 5: Pareto (latency range of two sessions, budgets)
# --------------------------------------------------------------------------------------------


SIZE_AREA = 700.0     # marker area (pt^2) per MB


def fig5_pareto(d: Data, save) -> None:
    p = d.pareto[d.pareto.latency_measured.astype(bool)].copy()
    p["lat_lo"] = p[["latency_median_ms_s1", "latency_median_ms_s2"]].min(axis=1)
    p["lat_hi"] = p[["latency_median_ms_s1", "latency_median_ms_s2"]].max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.15), layout="constrained")
    spec = [("road", "stealthy", "(a) ROAD: recall on stealthy (masquerade) types"),
            ("car_hacking", "pooled", "(b) Car-Hacking: pooled recall")]
    for ax, (ds, kind, title) in zip(axes, spec):
        q = p[p.dataset.eq(ds)]
        for r in q.itertuples():
            c, mk = C_MODEL[r.model], M_MODEL[r.model]
            y = r.stealthy_recall_mean if kind == "stealthy" else r.pooled_recall_mean
            if kind == "stealthy":
                lo, hi = r.stealthy_recall_ci_low, r.stealthy_recall_ci_high
                if np.isnan(lo):          # XGB: deterministic over seeds (K-055), no interval in pareto.csv
                    lo = hi = y
            else:
                lo, hi = y - r.pooled_recall_std, y + r.pooled_recall_std
            ax.hlines(y, r.lat_lo, r.lat_hi, color=c, linewidth=2.6, alpha=0.55, zorder=2)
            errbar_v(ax, r.lat_hi, lo, hi, c, lw=0.8)
            filled = r.quantization != "q0"
            ax.scatter(r.lat_hi, y, s=SIZE_AREA * r.size_mb, marker=mk, facecolors=c if filled else "white",
                       edgecolors=c, linewidths=1.0, zorder=3)
        col = "robust_front_stealthy_recall_mean" if kind == "stealthy" else "robust_front_pooled_recall_mean"
        front = q[q[col].astype(str).str.lower().eq("true")].sort_values("lat_hi")
        fy = front.stealthy_recall_mean if kind == "stealthy" else front.pooled_recall_mean
        ax.step(front.lat_hi, fy, where="post", color=INK, linewidth=0.8, linestyle=":", zorder=1)
        ylim = (0.0, 0.72) if kind == "stealthy" else (0.90, 1.012)
        ax.set_ylim(*ylim)
        for budget, ls, name in ((q.budget_strict_ms.iloc[0], "--", "1-message budget"),
                                 (q.budget_window_ms.iloc[0], "-.", "window budget")):
            ax.axvline(budget, color=INK_2, linewidth=0.9, linestyle=ls, zorder=5)   # above the latency box
            top = kind == "stealthy"                    # empty region: top in ROAD, bottom in Car-Hacking
            if name == "window budget":                 # horizontal, right of the line (free corner)
                ax.text(budget * 1.15, ylim[1] - 0.03 * (ylim[1] - ylim[0]), f"window\nbudget\n{budget:.3g} ms",
                        ha="left", va="top", fontsize=8, color=INK_2)
            else:
                ax.text(budget / 1.12, ylim[1] - 0.03 * (ylim[1] - ylim[0]) if top else
                        ylim[0] + 0.03 * (ylim[1] - ylim[0]), f"{name}: {budget:.3g} ms", rotation=90,
                        ha="right", va="top" if top else "bottom", fontsize=8, color=INK_2)
        # K-050: the two-session range is narrower than the markers on a log axis, so it is also written out
        lines = ["Median, 2 sessions (ms):"]
        for r in q.itertuples():
            name = f"{MODEL[r.model]} {r.quantization.upper()}" if r.model != "xgb" else MODEL[r.model]
            lines.append(f"{name}: {r.lat_lo:.3f}–{r.lat_hi:.3f}")
        ax.text(0.385 if kind == "stealthy" else 0.405, 0.66 if kind == "stealthy" else 0.62, "\n".join(lines),
                transform=ax.transAxes, ha="left",
                va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        ax.set_xscale("log")
        ax.set_xlim(0.015, 150)
        ax.set_xlabel("Median latency per window (ms, log scale)")
        ax.set_title(title)
        ax.grid(True, which="major")
    axes[0].set_ylabel("Recall (0–1): mean, 95% bootstrap interval")
    axes[1].set_ylabel("Recall (0–1): mean ± 1 s.d. (3 seeds)")
    handles = [Line2D([], [], marker=M_MODEL[m], color=C_MODEL[m], linestyle="none", markersize=5.5,
                      label=MODEL[m]) for m in ("cnn1d", "conv_ae", "xgb")]
    handles += [
        Line2D([], [], marker="o", markerfacecolor="white", color=INK_2, linestyle="none", markersize=5.5,
               label="open: Q0 (fp32)"),
        Line2D([], [], marker="o", color=INK_2, linestyle="none", markersize=5.5,
               label="filled: Q2 (INT8) / XGBoost"),
        Line2D([], [], color=INK_2, linewidth=2.6, alpha=0.55, label="range of 2 sessions (marker: slower)"),
        Line2D([], [], color=INK, linewidth=0.8, linestyle=":", label="robust Pareto front"),
    ]
    for mb in (0.04, 0.1, 0.25):
        handles.append(Line2D([], [], marker="o", markerfacecolor="white", color=INK_2, linestyle="none",
                              markersize=np.sqrt(SIZE_AREA * mb), label=f"file {mb:g} MB"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, columnspacing=1.0, handletextpad=0.4)
    save(fig, "fig5_pareto")


# --------------------------------------------------------------------------------------------
# Figure 6: size and latency ratios vs Q0
# --------------------------------------------------------------------------------------------


def session_ratio(d: Data, ds: str, model: str, k: str) -> tuple[float, float] | None:
    lat = d.model_metric("latency_median_ms_session1"), d.model_metric("latency_median_ms_session2")
    out = []
    for t in lat:
        t = t[t.dataset.eq(ds) & t.model.eq(model)]
        a, b = t[t.quantization.eq("q0")].value, t[t.quantization.eq(k)].value
        if a.empty or b.empty:
            return None
        out.append(b.iloc[0] / a.iloc[0])
    return min(out), max(out)


def fig6_gains(d: Data, save) -> None:
    size = d.model_metric("size_mb").groupby(["dataset", "model", "quantization"]).value.mean()
    groups = [(m, ds) for m in DEEP for ds in ("car_hacking", "road")]
    x = np.arange(len(groups))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SINGLE, 3.9), layout="constrained", sharex=True)
    for k, off, style in (("q1", -0.19, dict(facecolor="white", edgecolor=INK_2, hatch="////")),
                          ("q2", 0.19, dict(facecolor=INK_2, edgecolor=INK_2))):
        vals = [size.loc[(ds, m, k)] / size.loc[(ds, m, "q0")] for m, ds in groups]
        ax1.bar(x + off, vals, 0.34, linewidth=0.7, label=LEVEL[k], **style)
        for xi, v in zip(x + off, vals):
            ax1.text(xi, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.axhline(1, color=INK_2, linewidth=0.7, linestyle=":")
    ax1.text(x[-1] + 0.5, 1.0, "Q0", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax1.set_ylim(0, 1.3)
    ax1.set_ylabel("File size ratio Qk / Q0 (–)")
    ax1.set_title("(a) Model file size")
    ax1.legend(loc="upper left", ncol=2)
    for xi, (m, ds) in zip(x, groups):
        lo, hi = session_ratio(d, ds, m, "q2")
        ax2.bar(xi + 0.19, max(hi - lo, 0.004), 0.34, bottom=lo, color=INK_2, linewidth=0)
        ax2.text(xi + 0.19, hi + 0.02, f"{lo:.2f}–{hi:.2f}", ha="center", va="bottom", fontsize=8)
        ax2.text(xi - 0.19, 0.53, "Q1\nn/a", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax2.axhline(1, color=INK_2, linewidth=0.7, linestyle=":")
    ax2.set_ylim(0.5, 1.42)
    ax2.set_ylabel("Median latency ratio\nQ2 / Q0 (–), 2 sessions")
    ax2.set_title("(b) Latency, host CPU")
    ax2.text(0.02, 0.97, "Q1 not measured: no fp16 kernel on CPU", transform=ax2.transAxes, ha="left",
             va="top", fontsize=8, color=INK_2)
    ax2.set_xticks(x, [f"{MODEL[m]}\n{DS[ds]}" for m, ds in groups])
    for ax in (ax1, ax2):
        ax.grid(axis="y")
    save(fig, "fig6_size_latency_gain")


# --------------------------------------------------------------------------------------------
# Figure 7: supervised vs unsupervised (H4: K-062 -> K-063 -> K-064)
# --------------------------------------------------------------------------------------------


def fig7_sup_vs_unsup(d: Data, save) -> None:
    meta = d.type_meta()
    meta = meta[meta.role.eq("primary")].reset_index(drop=True)
    pt = d.stat("per_type_delta", statistic="d_roc_auc", quantization="q2")
    fig = plt.figure(figsize=(DOUBLE, 3.9), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 0.9, 0.9], height_ratios=[1, 1.05])
    axa = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1:])
    axc, axd = fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

    ys = np.arange(len(meta))[::-1]
    for r, y in zip(meta.itertuples(), ys):
        for m, dy in (("cnn1d", 0.17), ("conv_ae", -0.17)):
            s = pt[pt.model.eq(m) & pt.dataset.eq(r.dataset) & pt.unit.eq(r.attack_type)].iloc[0]
            errbar_h(axa, y + dy, s.ci_low, s.ci_high, C_MODEL[m])
            axa.scatter(s.estimate, y + dy, marker=M_MODEL[m], s=18, color=C_MODEL[m], zorder=3)
    axa.axvline(0, color=INK_2, linewidth=0.7, linestyle="--", zorder=1)
    axa.set_yticks(ys, [f"{DS[r.dataset]}: {type_label(r.attack_type)}" for r in meta.itertuples()])
    nb = int(meta.dataset.eq("car_hacking").sum())
    axa.axhline(ys[nb - 1] - 0.5, color=GRID, linewidth=0.8)
    axa.set_ylim(-0.6, len(meta) - 0.4)
    axa.set_xlabel("ΔROC-AUC, Q0 − Q2 (+ = loss), 95% interval")
    axa.set_title("(a) Per attack type, INT8")
    axa.grid(axis="x")

    rows = [("as3", "mean_abs_diff_droc_auc", "|ΔAE| − |ΔCNN|, AUC"),
            ("as3", "mean_abs_diff_drecall", "|ΔAE| − |ΔCNN|, recall"),
            ("as3_signed_diff", "mean_signed_diff_droc_auc", "ΔAE − ΔCNN, AUC"),
            ("as3_signed_diff", "mean_signed_diff_drecall", "ΔAE − ΔCNN, recall")]
    yb = np.array([3.0, 2.0, 0.4, -0.6])
    for (an, stat, _), y in zip(rows, yb):
        for k, dy, filled in (("q1", 0.17, False), ("q2", -0.17, True)):
            s = d.stat(an, statistic=stat, quantization=k).iloc[0]
            errbar_h(axb, y + dy, s.ci_low, s.ci_high, INK)
            axb.scatter(s.estimate, y + dy, marker="D", s=18, facecolors=INK if filled else "white", edgecolors=INK,
                        linewidths=1.0, zorder=3)
    axb.axvline(0, color=INK_2, linewidth=0.7, linestyle="--", zorder=1)
    axb.axhline(1.45, color=GRID, linewidth=0.8)
    axb.set_yticks(yb, [r[2] for r in rows])
    trb = blended_transform_factory(axb.transAxes, axb.transData)
    for y, text in ((3.62, "pre-registered rule (K-062), biased upward"), (1.02, "signed check (K-063)")):
        axb.text(0.01, y, text, transform=trb, ha="left", va="center", fontsize=8, color=INK_2, fontstyle="italic",
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.5), zorder=4)
    axb.set_ylim(-1.05, 4.0)
    axb.set_xlabel("Mean over types (+ = conv-AE loses more)")
    axb.set_title("(b) conv-AE − 1D-CNN, 95% interval")
    axb.grid(axis="x")

    churn = d.stat("as3_churn", quantization="q2")
    for ax, stat, ylabel, title in (
            (axc, "churn_attack", "Attack windows with\nchanged decision (0–1)", "(c) Changed decisions"),
            (axd, "spearman_q0_qk", "1 − Spearman ρ,\nQ0 vs Q2 (log)", "(d) Ranks")):
        for xi, ds in enumerate(("car_hacking", "road")):
            for m, off in (("cnn1d", -0.17), ("conv_ae", 0.17)):
                s = churn[churn.statistic.eq(stat) & churn.dataset.eq(ds) & churn.model.eq(m)].iloc[0]
                v = seeds(s.seed_values)
                mean = s.estimate
                if stat == "spearman_q0_qk":
                    v, mean = 1 - v, 1 - mean
                ax.scatter(np.full(len(v), xi + off), v, marker=M_MODEL[m], s=14, facecolors="white",
                           edgecolors=C_MODEL[m], linewidths=0.9, zorder=3)
                ax.hlines(mean, xi + off - 0.12, xi + off + 0.12, color=C_MODEL[m], linewidth=1.8, zorder=4)
        ax.set_xticks([0, 1], [DS[ds] for ds in ("car_hacking", "road")])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y")
    axd.set_yscale("log")
    handles = [Line2D([], [], marker=M_MODEL[m], color=C_MODEL[m], linestyle="none", markersize=5,
                      label=f"{MODEL[m]} ({'supervised' if m == 'cnn1d' else 'unsupervised'})") for m in DEEP]
    handles += [Line2D([], [], marker="o", markerfacecolor="white", color=INK_2, linestyle="none", markersize=4.5,
                       label="(c, d) seed value"),
                Line2D([], [], color=INK_2, linewidth=1.8, label="(c, d) mean of 3 seeds"),
                Line2D([], [], marker="D", markerfacecolor="white", color=INK, linestyle="none", markersize=4.5,
                       label="(b) Q1 (fp16)"),
                Line2D([], [], marker="D", color=INK, linestyle="none", markersize=4.5, label="(b) Q2 (INT8)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, columnspacing=1.2, handletextpad=0.4)
    save(fig, "fig7_supervised_vs_unsupervised")


# --------------------------------------------------------------------------------------------
# Figure 8: calibration shift (Q0 threshold applied to Qk, K-036 b)
# --------------------------------------------------------------------------------------------


def fig8_calibration_shift(d: Data, save) -> None:
    cs = d.stats[d.stats.analysis.eq("calib_shift")]
    clusters = [(ds, m) for ds in ("car_hacking", "road") for m in DEEP]
    fpr = d.cfg["models"]["decision"]["target_fpr"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.75), layout="constrained")
    panels = [("fpr_val", "(a) Validation FPR, attack-free", "False-positive rate (0–1, log)"),
              ("fpr_test", "(b) Test FPR, attack-free", "False-positive rate (0–1, log)"),
              ("pooled_recall", "(c) Test pooled recall", "Recall (0–1)")]
    step = 3.6
    for ax, (stem, title, ylabel) in zip(axes, panels):
        allv = []
        for c, (ds, m) in enumerate(clusters):
            def vals(k: str, which: str) -> np.ndarray:
                row = cs[cs.dataset.eq(ds) & cs.model.eq(m) & cs.quantization.eq(k) & cs.statistic.eq(f"{stem}_{which}")]
                return seeds(row.iloc[0].seed_values)
            V = np.vstack([vals("q2", "q0"), vals("q1", "qk"), vals("q2", "qk")])   # rows: Q0, Q1, Q2
            xs = c * step + np.arange(3)
            for s in range(V.shape[1]):
                ax.plot(xs, V[:, s], color=C_MODEL[m], linewidth=0.9, alpha=0.85, marker=M_MODEL[m], markersize=3.5,
                        markerfacecolor="white", markeredgewidth=0.8, zorder=3)
            allv.append(V)
        if stem.startswith("fpr"):
            ax.axhline(fpr, color=INK_2, linewidth=0.8, linestyle="--", zorder=1)   # labelled in the legend
            ax.set_yscale("log")
        minor = np.concatenate([c * step + np.arange(3) for c in range(len(clusters))])
        # level numbers are written as text: level 1 sits on the cluster centre, where matplotlib would drop a
        # minor tick label in favour of the cluster name
        ax.set_xticks(minor, minor=True)
        ax.tick_params(axis="x", which="minor", length=2, labelbottom=False)
        for i, x in enumerate(minor):
            ax.annotate(str(i % 3), (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -3.5),
                        textcoords="offset points", ha="center", va="top", fontsize=8, color=INK_2)
        ax.set_xticks([c * step + 1 for c in range(len(clusters))],
                      [f"{DS[ds]}\n{MODEL[m]}" for ds, m in clusters])
        ax.tick_params(axis="x", which="major", length=0, pad=14)
        ax.set_xlim(-0.6, (len(clusters) - 1) * step + 2.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", which="major")
    handles = [Line2D([], [], marker=M_MODEL[m], markerfacecolor="white", color=C_MODEL[m], markersize=4,
                      label=f"{MODEL[m]}: one line per seed") for m in DEEP]
    handles.append(Line2D([], [], color=INK_2, linestyle="--", linewidth=0.8,
                          label=f"target FPR of the threshold ({fpr:.0%}, val)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, columnspacing=1.2,
               title="x: quantization level k (0 = fp32, 1 = fp16, 2 = INT8); Q0 threshold kept for every k")
    save(fig, "fig8_calibration_shift")


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


TEX_MAP = {"Δ": r"$\Delta$", "δ": r"$\delta$", "±": r"$\pm$", "–": "--", "−": r"$-$",
           "ρ": r"$\rho$", "†": r"$^\dagger$", "→": r"$\rightarrow$", "%": r"\%", "_": r"\_",
           "&": r"\&", "#": r"\#", "|": r"$|$"}


def tex(s) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return "".join(TEX_MAP.get(ch, ch) for ch in s).replace("$$", "")   # merge adjacent math pieces


def latex_table(header: list[str], rows: list, colspec: str, caption: str, label: str, star: bool = True) -> str:
    """booktabs table. ``rows``: lists of cells (escaped here) or ("section", text) / ("midrule",)."""
    env = "table*" if star else "table"
    ncol = len(colspec.replace("@{}", ""))
    out = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", f"\\caption{{{tex(caption)}}}",
           f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header, "\\midrule"]
    first = True
    for r in rows:
        if isinstance(r, tuple) and r[0] == "section":
            if not first:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{tex(r[1])}}}}} \\\\")
        elif isinstance(r, tuple) and r[0] == "midrule":
            out.append("\\midrule")
        else:
            out.append(" & ".join(tex(c) for c in r) + " \\\\")
        first = False
    out += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(out)


def write_table(df: pd.DataFrame, latex: str, stem: str, out: Path, written: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv", index=False, float_format="%.10g")
    (out / f"{stem}.tex").write_text(latex, encoding="utf-8")
    written += [out / f"{stem}.csv", out / f"{stem}.tex"]


SPLIT_RULE = {"car_hacking": "time blocks in each file (K-017)", "road": "by source capture (K-018)"}


def table1_dataset(d: Data, out: Path, written: list) -> pd.DataFrame:
    meta = d.type_meta()
    ss, prof = d.split_summary, d.data_profile[~d.data_profile.excluded.astype(bool)]
    recs = []
    for ds in ("car_hacking", "road"):
        types = [("normal", "attack-free", None)] + [(r.attack_type, r.group, r.role)
                                                     for r in meta[meta.dataset.eq(ds)].itertuples()]
        for t, group, role in types:
            s = ss[ss.dataset.eq(ds) & ss.attack_type.eq(t)].set_index("split")
            pr = prof[prof.dataset.eq(ds) & (prof.kind.eq("ambient") if t == "normal" else prof.attack_type.eq(t))]
            rule = SPLIT_RULE[ds]
            if role == "unseen":
                rule = "test only (K-019)"
            elif ds == "road" and t == "fuzzing":
                rule = "source capture; no val (K-018)"
            rec = {"dataset": ds, "attack_type": t, "group": group, "split_rule": rule,
                   "captures": len(pr), "frames": int(pr.n_messages.sum()), "injected_frames": int(pr.n_attack.sum())}
            for sp in ("train", "val", "test"):
                rec[f"captures_{sp}"] = int(s.n_captures.get(sp, 0))
                rec[f"windows_{sp}"] = int(s.n_windows.get(sp, 0))
            recs.append(rec)
    df = pd.DataFrame(recs)
    rows = []
    for ds in ("car_hacking", "road"):
        rows.append(("section", f"{DS_LONG[ds]} (split: {SPLIT_RULE[ds]})"))
        for r in df[df.dataset.eq(ds)].itertuples():
            name = "attack-free" if r.attack_type == "normal" else type_label(r.attack_type)
            rows.append([name, r.group if r.attack_type != "normal" else "--",
                         f"{r.captures_train}/{r.captures_val}/{r.captures_test}", f"{r.frames:,}",
                         f"{r.injected_frames:,}" if r.injected_frames else "--",
                         f"{r.windows_train:,}", f"{r.windows_val:,}", f"{r.windows_test:,}",
                         "frames: attack-free captures only" if r.attack_type == "normal" else
                         (r.split_rule if r.split_rule != SPLIT_RULE[ds] else "")])
    w = d.cfg["data"]["windows"]
    header = ["Attack type & Group & Captures & Frames & Injected & \\multicolumn{3}{c}{Windows} & Note \\\\",
              "\\cmidrule(lr){6-8}",
              "& & tr/va/te & & frames & train & val & test & \\\\"]
    cap = (f"Dataset summary. Captures: number of captures that contribute windows of the row to the train, val and "
           f"test split (attack-free windows also come from the attack-free parts of attack captures). Frames and "
           f"injected frames: whole captures of that attack type. Windows of N = {w['size']} frames, stride "
           f"{w['stride']['train']} in training and {w['stride']['test']} (non-overlapping) in validation and test. "
           f"ROAD is split by source capture, so no capture (and no masquerade log derived from it) feeds two "
           f"splits; each Car-Hacking attack file is cut into disjoint time blocks separated by recording gaps "
           f"(K-017), because the dataset has one file per attack.")
    write_table(df, latex_table(header, rows, "@{}llrrrrrrl@{}", cap, "tab:dataset"), "table_1_dataset", out, written)
    return df


def _wide_metric(d: Data, metric: str) -> pd.DataFrame:
    meta = d.type_meta()
    mm = d.metric_means(metric).set_index(["dataset", "model", "quantization", "level", "attack_type"])
    cols = [(m, k) for m in DEEP for k in QUANT] + [("xgb", "native")]
    recs = []
    for ds in ("car_hacking", "road"):
        items = [(r.attack_type, r.group, r.role, "attack_type") for r in meta[meta.dataset.eq(ds)].itertuples()]
        items.append(("primary_pool", "pooled", "secondary", "pooled"))
        for t, g, role, level in items:
            rec = {"dataset": ds, "attack_type": t, "group": g, "role": role}
            for m, k in cols:
                row = mm.loc[(ds, m, k, level, t)]
                rec[f"{m}_{k}_mean"], rec[f"{m}_{k}_std"], rec[f"{m}_{k}_n"] = row["mean"], row["std"], row["count"]
            recs.append(rec)
    return pd.DataFrame(recs)


def table2_main(d: Data, out: Path, written: list, metric: str = "recall") -> pd.DataFrame:
    df = _wide_metric(d, metric)
    cols = [(m, k) for m in DEEP for k in QUANT] + [("xgb", "native")]
    rows = []
    for ds in ("car_hacking", "road"):
        rows.append(("section", DS_LONG[ds]))
        for r in df[df.dataset.eq(ds)].to_dict("records"):
            if r["attack_type"] == "primary_pool":
                name = "pooled (primary pool)"
            else:
                name = type_label(r["attack_type"]) + ("†" if r["role"] != "primary" else "")
            cells = [f"{r[f'{m}_{k}_mean']:.3f} ± {r[f'{m}_{k}_std']:.3f}" for m, k in cols]
            rows.append([name, r["group"] if r["group"] != "pooled" else "--", *cells])
    header = ["& & \\multicolumn{3}{c}{1D-CNN} & \\multicolumn{3}{c}{conv-AE} & XGBoost \\\\",
              "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
              "Attack type & Group & Q0 & Q1 & Q2 & Q0 & Q1 & Q2 & (not quantized) \\\\"]
    what = "Recall" if metric == "recall" else "F1"
    cap = (f"Test {what} (mean ± s.d. over 3 seeds) per attack type, model and quantization level "
           f"(Q0 fp32, Q1 fp16, Q2 INT8); each version uses its own validation threshold (FPR = 1%). "
           f"XGBoost is deterministic over seeds (s.d. 0). † test-only pair, outside the primary pool.")
    stem = "table_2_recall" if metric == "recall" else "table_2b_f1"
    write_table(df, latex_table(header, rows, "@{}llrrrrrrr@{}", cap, f"tab:{metric}"), stem, out, written)
    return df


def table3_resources(d: Data, out: Path, written: list) -> pd.DataFrame:
    a = d.all_results[d.all_results.level.eq("model")]
    piv = a.groupby(["dataset", "model", "quantization", "metric"]).value.mean().unstack("metric")
    par = d.pareto.set_index(["dataset", "model", "quantization"])
    recs = []
    for ds in ("car_hacking", "road"):
        for m, k in [(m, k) for m in DEEP for k in QUANT] + [("xgb", "native")]:
            r = piv.loc[(ds, m, k)]
            q0 = piv.loc[(ds, m, "q0" if m != "xgb" else "native")]
            rec = {"dataset": ds, "model": m, "quantization": k, "size_mb": r["size_mb"],
                   "size_ratio_vs_q0": r["size_mb"] / q0["size_mb"] if m != "xgb" else np.nan}
            for met in A.SESSION_METRICS:
                for s in (1, 2):
                    rec[f"{met}_s{s}"] = r.get(f"{met}_session{s}", np.nan)
            pr = par.loc[(ds, m, k)] if (ds, m, k) in par.index else None
            for c in ("latency_ratio_min", "latency_ratio_max", "meets_strict_budget", "meets_window_budget",
                      "budget_strict_ms", "budget_window_ms"):
                rec[c] = pr[c] if pr is not None else np.nan
            recs.append(rec)
    df = pd.DataFrame(recs)

    def rng(lo, hi, nd=3):
        if np.isnan(lo):
            return "--"
        lo, hi = sorted((lo, hi))
        return f"{lo:.{nd}f}–{hi:.{nd}f}"

    rows = []
    for ds in ("car_hacking", "road"):
        b = df[df.dataset.eq(ds)].iloc[0]
        rows.append(("section", f"{DS_LONG[ds]} (budgets: 1 message {b.budget_strict_ms:.3f} ms, "
                                f"window {b.budget_window_ms:.1f} ms)"))
        for r in df[df.dataset.eq(ds)].itertuples():
            meas = not np.isnan(r.latency_median_ms_s1)
            yes = lambda v: "--" if np.isnan(v) else ("yes" if v else "no")
            rows.append([MODEL[r.model], LEVEL[r.quantization] if r.model != "xgb" else "native",
                         f"{r.size_mb:.3f}", "--" if np.isnan(r.size_ratio_vs_q0) else f"{r.size_ratio_vs_q0:.2f}",
                         rng(r.latency_median_ms_s1, r.latency_median_ms_s2),
                         rng(r.latency_p95_ms_s1, r.latency_p95_ms_s2),
                         rng(r.latency_ratio_min, r.latency_ratio_max, 2),
                         rng(r.peak_rss_mb_s1, r.peak_rss_mb_s2, 1),
                         yes(r.meets_strict_budget) if meas else "--", yes(r.meets_window_budget) if meas else "--"])
    header = ["& & \\multicolumn{2}{c}{File} & \\multicolumn{2}{c}{Latency (ms), 2 sessions} & Ratio vs & "
              "Peak RSS & \\multicolumn{2}{c}{Meets budget} \\\\",
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){9-10}",
              "Model & Level & MB & /Q0 & median & p95 & CNN Q0 & (MB) & 1 msg & window \\\\"]
    cap = ("Resources on the host CPU (4 threads, batch 1, seed-0 models). Latency is the range over two "
           "measurement sessions; p95 is reported but not interpreted, absolute latencies are orders of "
           "magnitude only. Q1 has no CPU latency (no fp16 kernel in the CPU provider). A model meets a budget "
           "if the slower session's median is below it.")
    write_table(df, latex_table(header, rows, "@{}llrrrrrrcc@{}", cap, "tab:resources"), "table_3_resources",
                out, written)
    return df


def table4_stats(d: Data, out: Path, written: list) -> pd.DataFrame:
    recs = []
    fams = {"group_road_drecall": ("A", "ROAD stealthy (4) vs coarse (5)", "Cliff's δ of ΔRecall"),
            "group_road_df1": ("A", "ROAD stealthy (4) vs coarse (5)", "Cliff's δ of ΔF1"),
            "group_road+car_hacking_coarse_drecall": ("A'", "sensitivity: + CH coarse (4 vs 9)",
                                                      "Cliff's δ of ΔRecall"),
            "group_road+car_hacking_coarse_df1": ("A'", "sensitivity: + CH coarse (4 vs 9)", "Cliff's δ of ΔF1")}
    g = d.stat("group_mwu")
    for fam, (blk, comp, stat) in fams.items():
        for r in g[g.holm_family.eq(fam)].itertuples():
            recs.append({"block": blk, "comparison": comp, "model": r.model, "quantization": r.quantization,
                         "statistic": stat, "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high,
                         "p": r.p_value, "p_holm": r.p_holm, "min_attainable_p": r.min_attainable_p,
                         "test": "exact Mann-Whitney U"})
    for stat, name in (("mean_D_R", "mean D (ΔRecall)"), ("mean_D_F1", "mean D (ΔF1)")):
        for r in d.stat("as1_pooled", statistic=stat, block="block").itertuples():
            recs.append({"block": "B", "comparison": "paired masquerade − fabrication (3 pairs)",
                         "model": r.model, "quantization": r.quantization, "statistic": name,
                         "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high, "p": np.nan,
                         "p_holm": np.nan, "min_attainable_p": r.min_attainable_p, "test": "none (K-057)"})
    for an, stat, comp, name, test in (
            ("as3", "mean_abs_diff_droc_auc", "conv-AE vs 1D-CNN, 13 types", "|ΔAE| − |ΔCNN|, ROC-AUC (K-062)", "Wilcoxon"),
            ("as3", "mean_abs_diff_drecall", "conv-AE vs 1D-CNN, 6 types", "|ΔAE| − |ΔCNN|, recall (K-062)", "Wilcoxon"),
            ("as3_signed_diff", "mean_signed_diff_droc_auc", "conv-AE vs 1D-CNN, 13 types", "ΔAE − ΔCNN, ROC-AUC (K-063)", "none"),
            ("as3_signed_diff", "mean_signed_diff_drecall", "conv-AE vs 1D-CNN, 6 types", "ΔAE − ΔCNN, recall (K-063)", "none")):
        for r in d.stat(an, statistic=stat).itertuples():
            recs.append({"block": "C", "comparison": comp, "model": "conv_ae-cnn1d", "quantization": r.quantization,
                         "statistic": name, "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high,
                         "p": r.p_value, "p_holm": r.p_holm, "min_attainable_p": r.min_attainable_p, "test": test})
    df = pd.DataFrame(recs)

    def p3(v):
        return "--" if np.isnan(v) else f"{v:.3g}"

    def est(r) -> str:
        vals = (r.estimate, r.ci_low, r.ci_high)
        if r.statistic.startswith("Cliff"):
            f = "{:.2f}"
        elif max(abs(v) for v in vals) < 1e-3:         # Q1 ROC-AUC differences are below 10^-3
            def sci(v: float) -> str:                  # LaTeX math, passed through tex() unchanged
                mant, exp = f"{v:.1e}".split("e")
                return f"${mant}\\times10^{{{int(exp)}}}$"
            return f"{sci(vals[0])} [{sci(vals[1])}, {sci(vals[2])}]"
        else:
            f = "{:.4f}"
        e, lo, hi = (um(f.format(v)) for v in vals)
        return f"{e} [{lo}, {hi}]"

    titles = {"A": "A: coarse vs stealthy, group level (K-059; Holm over 4 tests per family)",
              "A'": "A': sensitivity analysis, Car-Hacking coarse types added",
              "B": "B: coarse vs stealthy, paired fabrication-masquerade units, 1D-CNN (K-057; exploratory)",
              "C": "C: supervised vs unsupervised (K-061, K-062, K-063)"}
    rows = []
    for blk in ("A", "A'", "B", "C"):
        rows.append(("section", titles[blk]))
        for r in df[df.block.eq(blk)].itertuples():
            model = MODEL.get(r.model, "conv-AE vs 1D-CNN")
            rows.append([model, r.quantization.upper(), r.statistic,
                         est(r), p3(r.p), p3(r.p_holm),
                         p3(r.min_attainable_p)])
    header = ["Model & Level & Statistic & Estimate [95\\% CI] & $p$ & $p_\\mathrm{Holm}$ & min. $p$ \\\\"]
    cap = ("Statistical comparisons (Δ = Q0 − Qk; positive = loss). Intervals: hierarchical bootstrap "
           "over seeds and test windows (moving blocks, B = 10,000). min. p: smallest p the design can attain "
           "(group test: 2/126; after Holm 0.063 > 0.05). The K-062 |Δ| difference is biased upward for the "
           "noisier model (K-063); the final H4 verdict (partially supported, K-064) rests on both rows.")
    write_table(df, latex_table(header, rows, "@{}lllrrrr@{}", cap, "tab:stats"), "table_4_stats", out, written)
    return df


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


FIGURE_FUNCS = (fig1_system, fig2_signatures, fig3_main, fig4_heatmap, fig5_pareto, fig6_gains,
                fig7_sup_vs_unsup, fig8_calibration_shift)


def run(figures_dir: Path = FIGURES, tables_dir: Path = TABLES, inputs_dir: Path = TABLES,
        only: list[str] | None = None, draft: bool = False) -> tuple[Data, list[Path]]:
    setup_style()
    d = Data(inputs_dir)
    save = Saver(figures_dir, T.git_info()["commit"], draft)
    for f in FIGURE_FUNCS:
        if only and not any(f.__name__.startswith(o) for o in only):
            continue
        f(d, save)
    written = list(save.written)
    if not only:
        table1_dataset(d, tables_dir, written)
        table2_main(d, tables_dir, written, "recall")
        table2_main(d, tables_dir, written, "f1")
        table3_resources(d, tables_dir, written)
        table4_stats(d, tables_dir, written)
    return d, written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures-dir", type=Path, default=FIGURES)
    ap.add_argument("--tables-dir", type=Path, default=TABLES)
    ap.add_argument("--only", nargs="*", help="figure function prefixes, e.g. fig3 (no tables, no manifest)")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--draft", action="store_true", help="print layout problems instead of failing (inspection)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")   # Windows console code page (Greek letters in labels)
    d, written = run(args.figures_dir, args.tables_dir, only=args.only, draft=args.draft)
    if not args.no_manifest and not args.only and not args.draft:
        A.write_manifest("figures", {"figures": [f.__name__ for f in FIGURE_FUNCS], "dpi": DPI,
                                     "widths_in": [SINGLE, DOUBLE], "font_pt": [FONT_MIN, FONT_MAX]},
                         d.inp.files, written,
                         {"versions_plot": A.versions(["matplotlib", "numpy", "pandas", "PyYAML"])})
    for p in written:
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
