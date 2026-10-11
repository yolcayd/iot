"""Paper figures and tables (analysis/figures.py, P7). Layout check on synthetic figures; tables and figures on the
committed P6 tables (results/tables), written to a temporary directory."""

import copy
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import figures as FG


@pytest.fixture(autouse=True)
def _style():
    FG.setup_style()
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("p7")
    d, written = FG.run(out / "figures", out / "tables")
    return d, out, written


def _clean_figure():
    fig, ax = plt.subplots(figsize=(FG.SINGLE, 2.0), layout="constrained")
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x (ms)")
    ax.set_ylabel("y (-)")
    return fig


# --------------------------------------------------------------------------------------------
# Layout check
# --------------------------------------------------------------------------------------------


def test_layout_check_passes_a_clean_figure():
    assert FG.layout_problems(_clean_figure()) == []


def test_layout_check_finds_overlapping_text():
    fig = _clean_figure()
    ax = fig.axes[0]
    ax.text(0.5, 0.5, "first label", fontsize=8)
    ax.text(0.5, 0.5, "second label", fontsize=8)
    assert any(p.startswith("overlap") for p in FG.layout_problems(fig))


def test_layout_check_finds_small_font_text_outside_and_empty_panel():
    fig, (a, b) = plt.subplots(1, 2, figsize=(FG.SINGLE, 2.0))
    a.plot([0, 1], [0, 1])
    a.text(0.1, 0.1, "tiny", fontsize=6)
    a.text(5.0, 0.5, "far right", transform=a.transAxes, fontsize=8)
    probs = FG.layout_problems(fig)
    assert any(p.startswith("font 6.0") for p in probs)
    assert any(p.startswith("outside figure") for p in probs)
    assert any(p.startswith("empty panel") for p in probs)


def test_saver_refuses_a_figure_with_layout_problems(tmp_path):
    fig = _clean_figure()
    fig.axes[0].text(0.2, 0.2, "x", fontsize=5)
    with pytest.raises(FG.LayoutError):
        FG.Saver(tmp_path, "abc")(fig, "bad")
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------

FIG_STEMS = ["fig1_system", "fig2_attack_signatures", "fig3_delta_recall_coarse_vs_stealthy", "fig4_recall_heatmap",
             "fig5_pareto", "fig6_size_latency_gain", "fig7_supervised_vs_unsupervised", "fig8_calibration_shift"]
TABLE_STEMS = ["table_1_dataset", "table_2_recall", "table_2b_f1", "table_3_resources", "table_4_stats"]


def _png_size_and_dpi(path):
    b = path.read_bytes()
    w, h = struct.unpack(">II", b[16:24])
    i = b.find(b"pHYs")
    ppm = struct.unpack(">I", b[i + 4:i + 8])[0] if i > 0 else None
    return w, h, ppm


def test_every_figure_is_written_as_pdf_and_300_dpi_png_at_ieee_width(outputs):
    _, out, _ = outputs
    for stem in FIG_STEMS:
        pdf, png = out / "figures" / f"{stem}.pdf", out / "figures" / f"{stem}.png"
        assert pdf.read_bytes()[:4] == b"%PDF"
        w, _, ppm = _png_size_and_dpi(png)
        assert ppm is not None and abs(ppm * 0.0254 - FG.DPI) < 1
        assert min(abs(w - FG.SINGLE * FG.DPI), abs(w - FG.DOUBLE * FG.DPI)) <= 1, stem


def test_every_table_is_written_as_csv_and_booktabs_latex(outputs):
    _, out, _ = outputs
    for stem in TABLE_STEMS:
        assert len(pd.read_csv(out / "tables" / f"{stem}.csv")) > 0
        tex = (out / "tables" / f"{stem}.tex").read_text(encoding="utf-8")
        for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
            assert rule in tex
        body = tex.replace("\\_", "")
        assert "_" not in body.replace("p_\\mathrm", ""), stem          # every underscore escaped
        assert not any(ord(ch) > 127 for ch in tex), stem                 # Greek letters etc. as LaTeX macros


def test_table2_cells_are_seed_means_of_all_results(outputs):
    d, out, _ = outputs
    t2 = pd.read_csv(out / "tables" / "table_2_recall.csv")
    a = d.all_results
    for model, quant, mode in (("cnn1d", "q2", "recalibrated"), ("conv_ae", "q0", "recalibrated"),
                               ("xgb", "native", "calibrated")):
        sel = a[a.model.eq(model) & a.quantization.eq(quant) & a.threshold_mode.eq(mode) & a.split.eq("test")
                & a.level.eq("attack_type") & a.metric.eq("recall") & a.dataset.eq("road")
                & a.attack_type.eq("max_speedometer_masquerade")]
        assert len(sel) == 3
        row = t2[t2.dataset.eq("road") & t2.attack_type.eq("max_speedometer_masquerade")].iloc[0]
        assert row[f"{model}_{quant}_mean"] == pytest.approx(sel.value.mean(), abs=1e-9)
        assert row[f"{model}_{quant}_std"] == pytest.approx(sel.value.std(ddof=1), abs=1e-9)


def test_table3_latency_ranges_come_from_both_sessions(outputs):
    d, out, _ = outputs
    t3 = pd.read_csv(out / "tables" / "table_3_resources.csv")
    a = d.all_results
    for s in (1, 2):
        v = a[a.model.eq("cnn1d") & a.quantization.eq("q2") & a.dataset.eq("road")
              & a.metric.eq(f"latency_median_ms_session{s}")].value.iloc[0]
        row = t3[t3.dataset.eq("road") & t3.model.eq("cnn1d") & t3.quantization.eq("q2")].iloc[0]
        assert row[f"latency_median_ms_s{s}"] == pytest.approx(v)
    q1 = t3[t3.quantization.eq("q1")]
    assert q1.latency_median_ms_s1.isna().all() and q1.latency_median_ms_s2.isna().all()   # K-043


def test_table4_values_are_those_of_stats_csv(outputs):
    d, out, _ = outputs
    t4 = pd.read_csv(out / "tables" / "table_4_stats.csv")
    s = d.stats
    ref = s[s.analysis.eq("as1_pooled") & s.statistic.eq("mean_D_R") & s.block.eq("block")
            & s.quantization.eq("q2")].iloc[0]
    row = t4[t4.block.eq("B") & t4.quantization.eq("q2") & t4.statistic.str.contains("Recall")].iloc[0]
    assert (row.estimate, row.ci_low, row.ci_high) == pytest.approx((ref.estimate, ref.ci_low, ref.ci_high))
    g = s[s.analysis.eq("group_mwu") & s.holm_family.eq("group_road_drecall")]
    a = t4[t4.block.eq("A") & t4.statistic.str.contains("Recall")]
    assert sorted(a.p_holm) == pytest.approx(sorted(g.p_holm))
    assert len(t4[t4.block.eq("C")]) == 8                                   # K-062 rule and K-063 check, Q1 and Q2


def test_table4_split_into_data_and_latex_keeps_the_p7_output(outputs):
    d, out, _ = outputs
    df = FG.table4_data(d)
    pd.testing.assert_frame_equal(df, pd.read_csv(out / "tables" / "table_4_stats.csv"), check_dtype=False)
    renamed = FG.table4_latex(df, {k: k for k in ("A", "A'", "B", "C")}, "cap", lambda s: s.replace("(K-062)", "(x)"))
    assert "(K-062)" not in renamed and "(x)" in renamed


def test_stats_and_delta_tables_are_cross_checked(outputs):
    d = copy.copy(outputs[0])
    d.stats = d.stats.copy()
    d.stats.loc[d.stats.analysis.eq("per_type_delta") & d.stats.statistic.eq("d_recall"), "estimate"] += 0.01
    with pytest.raises(ValueError):
        d.check_consistency()


def test_tex_escapes_symbols_and_merges_math():
    assert FG.tex("a_b 5% Δ±δ") == r"a\_b 5\% $\Delta\pm\delta$"
    assert FG.tex("|ΔAE| − |ΔCNN|") == r"$|\Delta$AE$|$ $-$ $|\Delta$CNN$|$"
