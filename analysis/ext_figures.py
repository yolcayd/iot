"""Figure of the P10 feature ablation (docs/DECISIONS.md K-074): ROAD test, per fabrication-masquerade pair.

Rows: TPR at test FPR 1 % (K-040, descriptive) and ROC-AUC; columns: fuzzing and the four primary pairs; x: feature
set s0 -> s4; colour = model (validated pair of analysis/figures.py), marker/line = group (filled square, solid =
coarse/fabrication; open circle, dashed = stealthy/masquerade); error bars = std over 3 seeds.

Output: results/figures/ext_ablation_road.{pdf,png}

Usage:
    python -m analysis.ext_figures
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from analysis import figures as FG  # noqa: E402

SETS = ["s0", "s1", "s2", "s3", "s4"]
SET_LABEL = ["S0", "S1", "S2", "S3", "S4"]   # S1 +bit flips, S2 +byte deltas, S3 +ID rate, S4 +window ID distribution
PAIRS = ["fuzzing", "correlated_signal", "max_speedometer", "reverse_light_off", "reverse_light_on"]
ROWS = [("roc_tpr_at_fpr_0.01", "TPR at test FPR 1 %"), ("roc_auc", "ROC-AUC")]


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    FG.setup_style()
    s = pd.read_csv(FG.TABLES / "ext_ablation_summary.csv")
    s = s[(s["dataset"] == "road") & (s["split"] == "test") & (s["level"] == "attack_type")]
    fig, axes = plt.subplots(len(ROWS), len(PAIRS), figsize=(FG.DOUBLE, 3.7), sharex=True, sharey="row")
    x = list(range(len(SETS)))
    for i, (metric, ylab) in enumerate(ROWS):
        for j, base in enumerate(PAIRS):
            ax = axes[i, j]
            for model in FG.DEEP:
                for name, grp in ((base, "coarse"), (base + FG.MASQ, "stealthy")):
                    d = s[(s["model"] == model) & (s["name"] == name) & (s["metric"] == metric)]
                    if d.empty:
                        continue
                    d = d.set_index("feature_set").reindex(SETS)
                    filled = FG.FILLED_GROUP[grp]
                    ax.errorbar(x, d["mean"], yerr=d["std"], color=FG.C_MODEL[model], marker=FG.M_GROUP[grp],
                                ms=4, mfc=FG.C_MODEL[model] if filled else "white", mew=0.9, lw=1.0,
                                ls="-" if grp == "coarse" else "--", elinewidth=0.6, capsize=1.5)
            ax.grid(axis="y")
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(FG.BASE_NAME[base], fontsize=8)
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks(x, SET_LABEL if i == len(ROWS) - 1 else [""] * len(x), fontsize=7)
    handles = [Line2D([], [], color=FG.C_MODEL[m], lw=1.2, label=FG.MODEL[m]) for m in FG.DEEP]
    handles += [Line2D([], [], color=FG.INK_2, marker=FG.M_GROUP["coarse"], ls="-", ms=4, label="fabrication (coarse)"),
                Line2D([], [], color=FG.INK_2, marker=FG.M_GROUP["stealthy"], mfc="white", ls="--", ms=4,
                       label="masquerade (stealthy)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FG.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FG.FIGURES / f"ext_ablation_road.{ext}", dpi=FG.DPI)
    plt.close(fig)
    print(f"wrote {FG.FIGURES / 'ext_ablation_road.pdf'} and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
