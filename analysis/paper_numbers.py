"""Numbers and generated tables for the paper text (P8; docs/DECISIONS.md K-066).

Every number in the paper sections comes from a LaTeX macro defined here; the sections contain no numeric
literals (tests/test_paper.py). Sources: configs/*.yaml, results/tables/*.csv, normalization_params.json and
the run_config.json / quant_meta.json / benchmark JSON files of the full grid (configs/grid.yaml: full).
Fields that must not differ between seeds (or datasets) are checked; a mismatch stops the run.

Outputs (paper/generated/; deterministic, input sha256 in the header, no timestamp and no commit so that a
committed file stays up to date):
    numbers.tex            \\newcommand macros, each with its source
    table_signatures.tex   Section III: traffic criteria C1-C4 and group per attack type (single column)
    table_dataset.tex      Table I of the paper (analysis.figures.table1_dataset data, no internal decision ids)

Usage:
    python -m analysis.paper_numbers
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")  # the machine is shared (CLAUDE.md: 4 CPU threads)

import argparse
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis import aggregate as A
from analysis import figures as FG
from src import benchmark as B
from src import run_grid as G
from src.models import classical, cnn1d, conv_autoencoder

ROOT = FG.ROOT
OUT = ROOT / "paper" / "generated"
RAW = ROOT / "results" / "raw"
LEAKAGE_TEST = ROOT / "tests" / "test_no_leakage.py"
MACRO_NAME = re.compile(r"^[A-Za-z]+$")


# --------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------


def fint(n) -> str:
    """Integer with a LaTeX thousands separator: 10000 -> 10{,}000."""
    if float(n) != int(n):
        raise ValueError(f"not an integer: {n}")
    return f"{int(n):,}".replace(",", "{,}")


def ffix(x: float, nd: int) -> str:
    return f"{float(x):.{nd}f}"


def fpct(x: float, nd: int = 0) -> str:
    """Fraction -> percent with escaped sign: 0.01 -> 1\\%."""
    return f"{100 * float(x):.{nd}f}\\%"


def fpow10(x: float) -> str:
    """1e-3 -> $10^{-3}$, 3e-4 -> $3\\times10^{-4}$ (mantissa printed only if not 1)."""
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if abs(m - round(m)) > 1e-9:
        raise ValueError(f"not a simple power of ten: {x}")
    m = int(round(m))
    return f"${'' if m == 1 else f'{m}\\times'}10^{{{e}}}$"


def flist(xs) -> str:
    return ", ".join(str(x) for x in xs)


def frange(lo: float, hi: float, nd: int) -> str:
    a, b = ffix(lo, nd), ffix(hi, nd)
    return a if a == b else f"{a}--{b}"


def fnum(x: float, nd: int) -> str:
    """Fixed decimals with a typographic minus that works in text and math mode."""
    s = ffix(x, nd)
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "\\ensuremath{-}")


def fci(est: float, lo: float, hi: float, nd: int) -> str:
    """Estimate with its interval: 0.0006 [-0.0035, 0.0057]."""
    return f"{fnum(est, nd)} [{fnum(lo, nd)}, {fnum(hi, nd)}]"


def fsci(x: float) -> str:
    """Two significant digits as a power of ten: -7.8e-05 -> -7.8 x 10^-5 (math)."""
    mant, exp = f"{x:.1e}".split("e")
    return f"\\ensuremath{{{mant}\\times10^{{{int(exp)}}}}}"


def fsig(x: float, n: int = 1) -> str:
    """``n`` significant digits in positional notation: 0.000156 -> 0.0002."""
    nd = max(0, n - 1 - math.floor(math.log10(abs(x))))
    return fnum(x, nd)


def fp(p: float) -> str:
    """p-value with three significant digits; 1 stays 1."""
    return "1" if p >= 1 else f"{p:.3g}"


def check(cond: bool, claim: str) -> None:
    """A qualitative statement of the Results text that must hold in the data."""
    if not cond:
        raise ValueError(f"claim of the Results text does not hold: {claim}")


def same(values, what: str):
    """The single value of ``values``; stops if they differ (e.g. between seeds)."""
    vals = list(values)
    first = vals[0]
    for v in vals[1:]:
        if v != first:
            raise ValueError(f"{what} differs between runs: {vals}")
    return first


def null_q(sig: pd.DataFrame, ds: str) -> str:
    """C2 null 99th percentile of a dataset with two significant digits."""
    q = same(sig[sig.dataset.eq(ds)].C2_null_q.round(12), "C2 null quantile")
    return ffix(q, max(0, 1 - math.floor(math.log10(q))))


# --------------------------------------------------------------------------------------------
# Macro collection
# --------------------------------------------------------------------------------------------


class Macros:
    def __init__(self):
        self.items: dict[str, tuple[str, str]] = {}

    def add(self, name: str, value, source: str) -> None:
        if not MACRO_NAME.match(name):
            raise ValueError(f"macro names are letters only: {name}")
        if name in self.items:
            raise ValueError(f"duplicate macro: {name}")
        self.items[name] = (str(value), source)

    def render(self, inputs: dict[str, str]) -> str:
        out = ["% Generated by `python -m analysis.paper_numbers` (docs/DECISIONS.md K-066). Do not edit.",
               "% Every number in the paper sections comes from a macro below; the comment names its source.",
               "% Inputs (sha256):"]
        out += [f"%   {k} {v}" for k, v in sorted(inputs.items())]
        out.append("")
        for name, (value, source) in self.items.items():
            out.append(f"\\newcommand{{\\{name}}}{{{value}}}% {source}")
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------------


class Sources:
    """Everything the paper numbers read; each file goes through ``inp`` (sha256 in the header)."""

    def __init__(self, tables: Path = FG.TABLES, configs: Path = FG.CONFIGS, raw: Path = RAW):
        self.d = FG.Data(tables, configs)
        self.inp = self.d.inp
        self.cfg = dict(self.d.cfg)
        for name in ("grid", "attack_groups"):
            path = configs / f"{name}.yaml"
            self.inp._add(path)
            self.cfg[name] = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.norm = self.inp.json(tables / "normalization_params.json")
        self.periodicity = self.inp.csv(tables / "periodicity_summary.csv")
        self.lineage = self.inp.csv(tables / "capture_lineage.csv")
        self.feasibility = self.inp.csv(tables / "ae_feasibility.csv")
        self.calib = self.inp.csv(tables / "calibration_shift.csv")
        self.inp._add(LEAKAGE_TEST)
        self.runs = G.stage_runs(self.cfg["analysis"]["stage"])
        self.run_cfg, self.quant_meta = {}, {}
        for r in self.runs:
            rid = r["run_id"]
            self.run_cfg[rid] = self.inp.json(raw / rid / "run_config.json")
            if r["quantize"]:
                self.quant_meta[rid] = self.inp.json(raw / rid / "quant" / "quant_meta.json")
        self.bench = []
        bcfg = self.cfg["quantization"]["benchmark"]
        for r in self.runs:
            if not r["quantize"] or r["seed"] != 0:
                continue
            for session in self.cfg["analysis"]["pareto"]["sessions"]:
                for q in ("q0", "q2"):
                    res = B.load_result(raw / r["run_id"] / "quant", "host", session, q, bcfg["protocol"])
                    if res is None:
                        raise FileNotFoundError(f"benchmark result missing: {r['run_id']} s{session} {q}")
                    self.inp._add(B.result_path(raw / r["run_id"] / "quant", "host", session, q))
                    self.bench.append(res)

    def runs_of(self, model: str, dataset: str | None = None) -> list[dict]:
        return [r for r in self.runs if r["model"] == model and (dataset is None or r["dataset"] == dataset)]


# --------------------------------------------------------------------------------------------
# Macros by topic
# --------------------------------------------------------------------------------------------

DS_TAG = {"car_hacking": "CH", "road": "ROAD"}


def data_macros(m: Macros, s: Sources) -> None:
    dc = s.cfg["data"]
    w, sp = dc["windows"], dc["splits"]
    src = "configs/data.yaml"
    if w["stride"]["val"] != w["stride"]["test"]:
        raise ValueError("validation and test strides differ")
    m.add("WinSize", w["size"], f"{src} windows.size")
    m.add("StrideTrain", w["stride"]["train"], f"{src} windows.stride.train")
    m.add("StrideEval", w["stride"]["test"], f"{src} windows.stride.val/test")
    m.add("OverlapTrainPct", fpct(1 - w["stride"]["train"] / w["size"]), f"{src} 1 - stride.train / size")
    m.add("NormalMarginS", ffix(dc["normal_subset"]["margin_s"], 0), f"{src} normal_subset.margin_s")
    m.add("MinTestAttackWindows", sp["min_test_attack_windows"], f"{src} splits.min_test_attack_windows")
    m.add("MinBlockGapS", ffix(sp["min_block_gap_s"], 0), f"{src} splits.min_block_gap_s")
    tb = sp["temporal_block_rule"]
    m.add("BurstGapS", ffix(tb["burst_gap_s"], 0), f"{src} temporal_block_rule.burst_gap_s")
    m.add("TrainBursts", tb["train_bursts"], f"{src} temporal_block_rule.train_bursts")
    m.add("ValBursts", tb["val_bursts"], f"{src} temporal_block_rule.val_bursts")
    m.add("GapBursts", tb["gap_bursts"], f"{src} temporal_block_rule.gap_bursts")
    m.add("NormalFractions", "/".join(str(round(100 * f)) for f in tb["normal_fractions"]),
          f"{src} temporal_block_rule.normal_fractions (percent)")
    m.add("NormalGapS", ffix(tb["normal_gap_s"], 0), f"{src} temporal_block_rule.normal_gap_s")
    m.add("ExcludedROAD", len(dc["datasets"]["road"]["excluded_captures"]),
          f"{src} datasets.road.excluded_captures")
    gaps = s.cfg["data"]["splits"]["car_hacking"]["blocks"]
    m.add("CHAttackFiles", sum(1 for k in gaps if not k.endswith("normal_run_data")),
          f"{src} splits.car_hacking.blocks (attack files)")
    m.add("SharedTailFiles", len(dc["shared_segments"]["car_hacking/shared_tail"]["contained_in"]),
          f"{src} shared_segments.car_hacking/shared_tail.contained_in")
    ngram = re.search(r"^NGRAM\s*=\s*(\d+)", LEAKAGE_TEST.read_text(encoding="utf-8"), re.M)
    m.add("LeakNgram", int(ngram.group(1)), "tests/test_no_leakage.py NGRAM")


def feature_macros(m: Macros, s: Sources) -> None:
    n = s.norm
    names = n["feature_names"]
    src = "results/tables/normalization_params.json"
    m.add("NFeatures", len(names), f"{src} feature_names")
    m.add("NPayloadBytes", sum(1 for f in names if f.startswith("byte_")), f"{src} feature_names (byte_*)")
    iat = s.cfg["data"]["features"]["iat"]
    m.add("IATFloorQ", iat["floor_quantile"], "configs/data.yaml features.iat.floor_quantile")
    m.add("IATCapQ", iat["cap_quantile"], "configs/data.yaml features.iat.cap_quantile")
    for ds, tag in DS_TAG.items():
        p = n["datasets"][ds]
        m.add(f"Vocab{tag}", p["n_ids"], f"{src} datasets.{ds}.n_ids")
        m.add(f"IdLevels{tag}", p["n_ids"] + 1, f"{src} datasets.{ds}.n_ids + 1 (reserved code 0)")
        m.add(f"IATFloorUs{tag}", ffix(p["iat_floor_s"] * 1e6, 2), f"{src} datasets.{ds}.iat_floor_s (us)")
        m.add(f"IATCapS{tag}", ffix(p["iat_cap_s"], 3), f"{src} datasets.{ds}.iat_cap_s")
        m.add(f"IATSamples{tag}", fint(p["n_iat_samples"]), f"{src} datasets.{ds}.n_iat_samples")
    # input quantizer of every Q2 graph (asymmetric uint8): scale 1/255, zero point 0, 256 levels on [0, 1]
    iq = [q["levels"]["q2"]["input_quantization"] for q in s.quant_meta.values()]
    inv = same([round(1 / x["scale"]) for x in iq], "Q2 input scale")
    m.add("ByteScale", inv, "quant_meta.json levels.q2.input_quantization 1/scale (= payload byte divisor)")
    m.add("InputScale", f"$1/{inv}$", "quant_meta.json levels.q2.input_quantization.scale")
    m.add("InputLevels", same([x["levels_on_unit_interval"] for x in iq], "Q2 input levels"),
          "quant_meta.json levels.q2.input_quantization.levels_on_unit_interval")
    m.add("InputZeroPoint", same([x["zero_point"] for x in iq], "Q2 zero point"),
          "quant_meta.json levels.q2.input_quantization.zero_point")
    lossless = [c["lossless"] for q in s.quant_meta.values() for c in q["levels"]["q2"]["input_levels"]["columns"]
                if c["feature"] != "iat"]
    if not all(lossless):
        raise ValueError("an ID or byte column is not lossless on the INT8 input grid")


def model_macros(m: Macros, s: Sources) -> None:
    import torch

    mc = s.cfg["models"]
    src = "configs/models.yaml"
    com, cnn, ae, lstm, xgb = (mc["common"], mc["models"]["cnn1d"], mc["models"]["conv_ae"],
                               mc["models"]["lstm_ae"], mc["models"]["xgb"])
    m.add("BatchSize", com["batch_size"], f"{src} common.batch_size")
    m.add("AEBatchSize", same([ae["batch_size"], lstm["batch_size"]], "autoencoder batch size"),
          f"{src} models.conv_ae/lstm_ae.batch_size")
    m.add("LearningRate", fpow10(com["lr"]), f"{src} common.lr")
    m.add("WeightDecay", com["weight_decay"], f"{src} common.weight_decay")
    m.add("MaxEpochs", com["max_epochs"], f"{src} common.max_epochs")
    m.add("Patience", com["patience"], f"{src} common.patience")
    m.add("CNNBlocks", len(cnn["channels"]), f"{src} models.cnn1d.channels")
    m.add("CNNChannels", flist(cnn["channels"]), f"{src} models.cnn1d.channels")
    m.add("CNNKernels", flist(cnn["kernels"]), f"{src} models.cnn1d.kernels")
    m.add("CNNHidden", cnn["hidden"], f"{src} models.cnn1d.hidden")
    m.add("CNNDropout", cnn["dropout"], f"{src} models.cnn1d.dropout")
    m.add("AEChannels", flist(ae["channels"]), f"{src} models.conv_ae.channels")
    m.add("AELatentChannels", ae["latent_channels"], f"{src} models.conv_ae.latent_channels")
    nf = len(s.norm["feature_names"])
    w = s.cfg["data"]["windows"]["size"]
    net_c, net_a = cnn1d.build(cnn, nf), conv_autoencoder.build(ae, nf)
    with torch.no_grad():
        z = net_a.encoder(torch.zeros(1, w, nf).transpose(1, 2))
    m.add("AELatentLen", z.shape[2], "src/models/conv_autoencoder.py encoder output length (window = WinSize)")
    m.add("CNNParams", fint(sum(p.numel() for p in net_c.parameters())), "src/models/cnn1d.py parameter count")
    m.add("AEParams", fint(sum(p.numel() for p in net_a.parameters())),
          "src/models/conv_autoencoder.py parameter count")
    m.add("LSTMHidden", lstm["hidden"], f"{src} models.lstm_ae.hidden")
    m.add("LSTMLatent", lstm["latent"], f"{src} models.lstm_ae.latent")
    fe = mc["ae_feasibility"]
    m.add("FeasMinAUC", fe["min_val_roc_auc"], f"{src} ae_feasibility.min_val_roc_auc")
    m.add("RetryHidden", fe["retry"]["hidden"], f"{src} ae_feasibility.retry.hidden")
    m.add("RetryLR", fpow10(fe["retry"]["lr"]), f"{src} ae_feasibility.retry.lr")
    m.add("RetryMaxEpochs", fe["retry"]["max_epochs"], f"{src} ae_feasibility.retry.max_epochs")
    cand = {"baseline": "LSTMBase", "retry": "LSTMRetry", "conv_ae": "ConvAE"}
    for r in s.feasibility.itertuples():
        m.add(f"Feas{cand[r.candidate]}{DS_TAG[r.dataset]}", ffix(r.val_roc_auc, 3),
              f"results/tables/ae_feasibility.csv {r.candidate} {r.dataset}")
    m.add("XGBEstimators", fint(xgb["n_estimators"]), f"{src} models.xgb.n_estimators")
    m.add("XGBDepth", xgb["max_depth"], f"{src} models.xgb.max_depth")
    m.add("XGBLR", xgb["learning_rate"], f"{src} models.xgb.learning_rate")
    m.add("XGBSubsample", ffix(same([xgb["subsample"], xgb["colsample_bytree"]], "xgb sampling"), 1),
          f"{src} models.xgb.subsample = colsample_bytree")
    m.add("XGBEarlyStop", xgb["early_stopping_rounds"], f"{src} models.xgb.early_stopping_rounds")
    m.add("NSummaryFeatures", len(classical.SUMMARY_NAMES), "src/models/classical.py SUMMARY_NAMES")
    dec = mc["decision"]
    m.add("TargetFPR", fpct(dec["target_fpr"]), f"{src} decision.target_fpr")
    m.add("CalibQuantile", ffix(1 - dec["target_fpr"], 2), f"{src} 1 - decision.target_fpr")
    m.add("Threads", mc["runtime"]["threads"], f"{src} runtime.threads")


def run_macros(m: Macros, s: Sources) -> None:
    g = s.cfg["grid"]["full"]
    m.add("Seeds", flist(g["seeds"]), "configs/grid.yaml full.seeds")
    m.add("NSeeds", len(g["seeds"]), "configs/grid.yaml full.seeds")
    m.add("NRuns", len(s.runs), "configs/grid.yaml full (models x datasets x seeds)")
    rc = s.run_cfg
    src = "run_config.json"
    for ds, tag in DS_TAG.items():
        cnn = [rc[r["run_id"]] for r in s.runs_of("cnn1d", ds)]
        m.add(f"PosWeight{tag}", ffix(same([round(c["meta"]["pos_weight"], 6) for c in cnn], "pos_weight"), 1),
              f"{src} meta.pos_weight (cnn1d, {ds})")
        m.add(f"CNNTrainWindows{tag}", fint(same([c["train_windows"]["n_windows"] for c in cnn], "train windows")),
              f"{src} train_windows.n_windows (cnn1d, {ds})")
        ae = [rc[r["run_id"]] for r in s.runs_of("conv_ae", ds)]
        m.add(f"AETrainWindows{tag}", fint(same([c["train_windows"]["n_windows"] for c in ae], "AE windows")),
              f"{src} train_windows.n_windows (conv_ae, {ds})")
        if any(c["train_windows"]["n_attack"] for c in ae):
            raise ValueError("autoencoder training windows contain attacks")
        allr = [rc[r["run_id"]] for r in s.runs if r["dataset"] == ds]
        m.add(f"CalibNormals{tag}", fint(same([c["meta"]["decision"]["n_val_normal"] for c in allr],
                                              "calibration windows")),
              f"{src} meta.decision.n_val_normal (every model, {ds})")
    env = [c["environment"] for c in rc.values()]
    versions = same([e["versions"] for e in env], "library versions")
    m.add("PythonVersion", same([e["python"] for e in env], "python"), f"{src} environment.python")
    for key, name in (("torch", "TorchVersion"), ("onnx", "ONNXVersion"), ("onnxruntime", "ORTVersion"),
                      ("scikit-learn", "SklearnVersion"), ("xgboost", "XGBVersion")):
        m.add(name, versions[key].split("+")[0], f"{src} environment.versions.{key}")
    m.add("GPUName", same([e["gpu"] for e in env], "gpu").removeprefix("NVIDIA "), f"{src} environment.gpu")
    m.add("TrainThreads", same([e["threads"] for e in env], "threads"), f"{src} environment.threads")
    for ds in DS_TAG:   # XGBoost is seed-independent (K-055): every seed must stop at the same iteration
        same([rc[r["run_id"]]["meta"]["best_iteration"] for r in s.runs_of("xgb", ds)], f"xgb best iteration {ds}")


def quant_macros(m: Macros, s: Sources) -> None:
    qc = s.cfg["quantization"]
    src = "configs/quantization.yaml"
    m.add("Opset", qc["onnx"]["opset"], f"{src} onnx.opset")
    m.add("QZeroAtol", fpow10(qc["onnx"]["q0_atol"]), f"{src} onnx.q0_atol")
    m.add("QZeroRtol", fpow10(qc["onnx"]["q0_rtol"]), f"{src} onnx.q0_rtol")
    m.add("MinHeavyFPShare", fpct(qc["q1"]["min_heavy_fp16_share"]), f"{src} q1.min_heavy_fp16_share")
    q2 = qc["q2"]
    m.add("CalibWindowsQ", fint(q2["calibration_windows"]), f"{src} q2.calibration_windows")
    m.add("MinCalibWindowsQ", q2["min_calibration_windows"], f"{src} q2.min_calibration_windows")
    m.add("QOpTypes", ", ".join(q2["op_types"]), f"{src} q2.op_types")
    for model, tag in (("cnn1d", "CNN"), ("conv_ae", "AE")):
        metas = [s.quant_meta[r["run_id"]] for r in s.runs_of(model)]
        cov = [q["levels"]["q1"]["fp16_coverage"] for q in metas]
        ex = [q["levels"]["q2"]["execution"] for q in metas]
        m.add(f"Heavy{tag}", same([c["n_heavy"] for c in cov] + [e["n_heavy"] for e in ex], "heavy layers"),
              "quant_meta.json levels.q1.fp16_coverage.n_heavy / levels.q2.execution.n_heavy")
        m.add(f"HeavyFP{tag}", same([c["n_heavy_fp16"] for c in cov], "fp16 heavy kernels"),
              "quant_meta.json levels.q1.fp16_coverage.n_heavy_fp16")
        m.add(f"HeavyInt{tag}", same([e["n_heavy_int8_kernel"] for e in ex], "int8 kernels"),
              "quant_meta.json levels.q2.execution.n_heavy_int8_kernel")
        m.add(f"HeavySim{tag}", same([e["n_heavy_qdq_simulated"] for e in ex], "QDQ simulated"),
              "quant_meta.json levels.q2.execution.n_heavy_qdq_simulated")
        cal = [q["levels"]["q2"]["calibration"] for q in metas]
        if any(c["split"] != "train" for c in cal):
            raise ValueError("INT8 calibration windows outside the training split")
        same([c["n_windows"] for c in cal] + [q2["calibration_windows"]], "calibration windows")
        if model == "conv_ae" and any(c["n_attack"] for c in cal):
            raise ValueError("autoencoder INT8 calibration contains attack windows")


def bench_macros(m: Macros, s: Sources) -> None:
    bc = s.cfg["quantization"]["benchmark"]
    src = "configs/quantization.yaml benchmark"
    res = s.bench
    m.add("BenchRepeats", fint(same([r["repeats"] for r in res] + [bc["repeats"]], "repeats")), f"{src}.repeats")
    m.add("BenchMinRepeats", fint(bc["min_repeats"]), f"{src}.min_repeats")
    m.add("BenchWarmup", same([r["warmup"] for r in res] + [bc["warmup"]], "warm-up"), f"{src}.warmup")
    m.add("BenchWarmupS", ffix(same([r["warmup_min_seconds"] for r in res], "warm-up s"), 0),
          f"{src}.warmup_min_seconds")
    m.add("BenchInputs", fint(same([r["n_inputs"] for r in res] + [bc["n_inputs"]], "inputs")), f"{src}.n_inputs")
    m.add("BenchThreads", same([r["threads"] for r in res], "threads"), f"{src}.profiles.host.threads")
    m.add("BenchBatch", same([r["batch_size"] for r in res], "batch"), "benchmark JSON batch_size")
    ort = same([(r["onnxruntime"]["version"], r["onnxruntime"]["intra_op_threads"],
                 r["onnxruntime"]["inter_op_threads"], r["onnxruntime"]["execution_mode"]) for r in res], "ORT")
    m.add("BenchORTVersion", ort[0], "benchmark JSON onnxruntime.version")
    m.add("BenchIntraOp", ort[1], "benchmark JSON onnxruntime.intra_op_threads")
    m.add("BenchInterOp", ort[2], "benchmark JSON onnxruntime.inter_op_threads")
    env = [r["environment"] for r in res]
    m.add("CPUName", same([e["cpu"] for e in env], "cpu").replace("(R)", "").replace("(TM)", ""),
          "benchmark JSON environment.cpu")
    plat = same([e["platform"] for e in env], "platform").split("-")
    m.add("OSName", f"{plat[0]} {plat[1]}", "benchmark JSON environment.platform")
    m.add("PhysCores", same([e["physical_cores"] for e in env], "cores"), "benchmark JSON environment.physical_cores")
    m.add("LogicalCores", same([e["logical_cores"] for e in env], "cores"), "benchmark JSON environment.logical_cores")
    m.add("MemGiB", ffix(same([e["memory_total_mb"] for e in env], "memory") / 1024, 1),
          "benchmark JSON environment.memory_total_mb / 1024")
    pin = same([tuple(r["pinning"]["cpus"]) for r in res], "pinned cpus")
    if not all(r["pinning"]["applied"] for r in res):
        raise ValueError("core pinning not applied")
    m.add("PinnedCores", len(pin), "benchmark JSON pinning.cpus")
    sets = res[0]["pinning"]["cpu_sets"]
    top = max(c["efficiency_class"] for c in sets)
    m.add("PerfCores", len({c["core"] for c in sets if c["efficiency_class"] == top}),
          "benchmark JSON pinning.cpu_sets (cores of the highest efficiency class)")
    m.add("EffCores", len({c["core"] for c in sets if c["efficiency_class"] != top}),
          "benchmark JSON pinning.cpu_sets (other cores)")
    if not all(e["power"]["power_mode"] == "best_performance" for e in env):
        raise ValueError("benchmark outside the best-performance power mode")
    m.add("BenchSessions", len(s.cfg["analysis"]["pareto"]["sessions"]), "configs/analysis.yaml pareto.sessions")
    rep = bc["repeatability"]
    m.add("MedianTol", fpct(rep["median_rel_tol"]), f"{src}.repeatability.median_rel_tol")
    m.add("PNinetyFiveTol", fpct(rep["p95_rel_tol"]), f"{src}.repeatability.p95_rel_tol")
    tail = same([sorted(k for k in r["latency"] if re.fullmatch(r"p\d+_ms", k)) for r in res], "percentiles")
    m.add("TailPct", int(tail[0][1:-3]), "benchmark JSON latency.p<N>_ms (lowest tail percentile; tolerance key)")
    m.add("BenchSeed", same([r["seed"] for r in s.runs if r["quantize"] and r["seed"] == 0], "seed"),
          "configs/grid.yaml full.latency = reuse_seed0")
    # message rates and latency budgets (K-060): read from pareto.csv, which stats.py derived from data_profile.csv
    pc = s.cfg["analysis"]["pareto"]["budgets"]
    m.add("BudgetWindowMsgs", pc["window_messages"], "configs/analysis.yaml pareto.budgets.window_messages")
    m.add("BudgetStrictMsgs", pc["strict_messages"], "configs/analysis.yaml pareto.budgets.strict_messages")
    par = s.d.pareto
    for ds, tag in DS_TAG.items():
        p = par[par.dataset.eq(ds)]
        m.add(f"MsgRate{tag}", fint(round(same(p.msg_rate_per_s.round(6), "rate"))), f"pareto.csv msg_rate_per_s ({ds})")
        m.add(f"BudgetWindow{tag}", ffix(same(p.budget_window_ms.round(9), "budget"), 1),
              f"pareto.csv budget_window_ms ({ds})")
        m.add(f"BudgetStrict{tag}", ffix(same(p.budget_strict_ms.round(9), "budget"), 3),
              f"pareto.csv budget_strict_ms ({ds})")


def analysis_macros(m: Macros, s: Sources) -> None:
    an = s.cfg["analysis"]
    src = "configs/analysis.yaml"
    bs = an["bootstrap"]
    m.add("NBoot", fint(bs["n_boot"]), f"{src} bootstrap.n_boot")
    m.add("CIPct", fpct(bs["ci"]), f"{src} bootstrap.ci")
    m.add("FloorRecall", ffix(an["as1"]["floor"], 2), f"{src} as1.floor")
    m.add("RecallFloorAS", ffix(an["as3"]["recall_floor"], 2), f"{src} as3.recall_floor")
    h = an["hypotheses"]
    m.add("HOneMax", ffix(h["h1_pooled_df1_max"], 2), f"{src} hypotheses.h1_pooled_df1_max")
    m.add("HTwoRecallMin", ffix(h["h2_coarse_recall_min"], 2), f"{src} hypotheses.h2_coarse_recall_min")
    meta = s.d.type_meta()
    road = meta[meta.dataset.eq(an["group_test"]["dataset"]) & meta.role.eq(an["as1"]["primary_role"])]
    m.add("GroupStealthy", int(road.group.eq("stealthy").sum()), "delta_f1.csv ROAD primary types (stealthy)")
    m.add("GroupCoarse", int(road.group.eq("coarse").sum()), "delta_f1.csv ROAD primary types (coarse)")
    prim = meta[meta.role.eq(an["as1"]["primary_role"])]
    m.add("ASThreeTypes", len(prim), "delta_f1.csv primary types, both datasets")
    pairs = road[road.group.eq("stealthy")]
    m.add("PrimaryPairs", len(pairs), "delta_f1.csv ROAD primary masquerade types (one pair each)")


def signature_macros(m: Macros, s: Sources) -> None:
    eda, sig, groups = s.cfg["eda"]["grouping_rule"], s.d.attack_signatures, s.cfg["attack_groups"]
    src = "configs/eda.yaml grouping_rule"
    m.add("COneThr", eda["C1_unseen_id_share"]["threshold"], f"{src}.C1_unseen_id_share.threshold")
    m.add("CTwoFrames", fint(eda["C2_id_distribution_jsd"]["window_frames"]), f"{src}.C2.window_frames")
    m.add("CTwoDraws", eda["C2_id_distribution_jsd"]["null_draws"], f"{src}.C2.null_draws")
    m.add("CTwoPct", round(100 * eda["C2_id_distribution_jsd"]["null_quantile"]), f"{src}.C2.null_quantile (pct)")
    band = same([tuple(eda["C3_target_rate_ratio"]["band"]), tuple(eda["C4_target_median_iat_ratio"]["band"])],
                "C3/C4 band")
    if abs(band[0] * band[1] - 1) > 1e-6:
        raise ValueError("C3/C4 band is not symmetric on the log scale")
    m.add("CBand", ffix(band[1], 1), f"{src}.C3/C4 band upper (lower = 1/upper)")
    m.add("RecGapS", ffix(s.cfg["eda"]["periods"]["recording_gap_s"], 0), "configs/eda.yaml periods.recording_gap_s")
    sig = sig.copy()
    sig["key"] = sig.dataset + "/" + sig.attack_type
    g = {f"{ds}/{t}": v["group"] for ds, types in groups.items() for t, v in types.items()}
    if any(g[k] != v for k, v in zip(sig.key, sig.proposed_group)):
        raise ValueError("attack_signatures.csv and attack_groups.yaml disagree")
    m.add("NTypes", len(sig), "attack_signatures.csv rows")
    m.add("NCoarse", int(sig.proposed_group.eq("coarse").sum()), "attack_signatures.csv proposed_group")
    m.add("NStealthy", int(sig.proposed_group.eq("stealthy").sum()), "attack_signatures.csv proposed_group")
    m.add("NCoarseCaptures", int(sig[sig.proposed_group.eq("coarse")].n_captures.sum()), "attack_signatures.csv")
    m.add("NStealthyCaptures", int(sig[sig.proposed_group.eq("stealthy")].n_captures.sum()), "attack_signatures.csv")
    m.add("NPlanContradict", int((~sig.agrees_with_plan.astype(bool)).sum()), "attack_signatures.csv agrees_with_plan")
    for ds, tag in DS_TAG.items():
        m.add(f"NullQ{tag}", null_q(sig, ds), f"attack_signatures.csv C2_null_q ({ds})")

    def row(ds, t):
        r = sig[sig.dataset.eq(ds) & sig.attack_type.eq(t)]
        if len(r) != 1:
            raise KeyError(f"{ds}/{t}")
        return r.iloc[0]

    def rows(pred):
        return sig[[pred(r) for r in sig.itertuples()]]

    sa = "attack_signatures.csv"
    m.add("DoSUnseenPct", fpct(row("car_hacking", "dos").C1_unseen_id_share, 1), f"{sa} dos C1")
    m.add("DoSJSD", ffix(row("car_hacking", "dos").C2_jsd_median, 3), f"{sa} dos C2")
    m.add("DoSTarget", row("car_hacking", "dos").target_id, f"{sa} dos target_id")
    m.add("FuzzyUnseenPct", fpct(row("car_hacking", "fuzzy").C1_unseen_id_share, 1), f"{sa} fuzzy C1")
    m.add("FuzzyJSD", ffix(row("car_hacking", "fuzzy").C2_jsd_median, 3), f"{sa} fuzzy C2")
    spoof = rows(lambda r: r.dataset == "car_hacking" and r.attack_type.endswith("spoofing"))
    m.add("SpoofRate", frange(spoof.C3_target_rate_ratio.min(), spoof.C3_target_rate_ratio.max(), 2),
          f"{sa} CH spoofing C3")
    m.add("SpoofIAT", frange(spoof.C4_target_median_iat_ratio.min(), spoof.C4_target_median_iat_ratio.max(), 2),
          f"{sa} CH spoofing C4")
    m.add("SpoofJSD", frange(spoof.C2_jsd_median.min(), spoof.C2_jsd_median.max(), 3), f"{sa} CH spoofing C2")
    rf = row("road", "fuzzing")
    m.add("RoadFuzzUnseenPct", fpct(rf.C1_unseen_id_share, 1), f"{sa} road fuzzing C1")
    m.add("RoadFuzzJSD", ffix(rf.C2_jsd_median, 3), f"{sa} road fuzzing C2")
    fab = rows(lambda r: r.dataset == "road" and r.attack_type != "fuzzing" and not r.attack_type.endswith(FG.MASQ))
    masq = rows(lambda r: r.attack_type.endswith(FG.MASQ))
    m.add("NFabTypes", len(fab), f"{sa} ROAD fabrication types")
    m.add("FabRate", frange(fab.C3_target_rate_ratio.min(), fab.C3_target_rate_ratio.max(), 2), f"{sa} fabrication C3")
    m.add("FabIAT", frange(fab.C4_target_median_iat_ratio.min(), fab.C4_target_median_iat_ratio.max(), 2),
          f"{sa} fabrication C4")
    m.add("FabJSD", frange(fab.C2_jsd_median.min(), fab.C2_jsd_median.max(), 3), f"{sa} fabrication C2")
    m.add("MasqRate", frange(masq.C3_target_rate_ratio.min(), masq.C3_target_rate_ratio.max(), 2), f"{sa} masq. C3")
    m.add("MasqIAT", frange(masq.C4_target_median_iat_ratio.min(), masq.C4_target_median_iat_ratio.max(), 2),
          f"{sa} masq. C4")
    m.add("MasqJSDMax", ffix(masq.C2_jsd_median.max(), 3), f"{sa} masq. C2 (max)")
    if (masq.C1_unseen_id_share > 0).any():
        raise ValueError("a masquerade type has unseen IDs")
    shift = masq.payload_max_byte_median_shift
    m.add("MasqByteShift", f"{int(shift.min())}--{int(shift.max())}", f"{sa} masq. payload_max_byte_median_shift")
    # periodicity (P1)
    per = s.periodicity.set_index("dataset")
    m.add("CVThr", s.cfg["eda"]["periodicity"]["cv_threshold"], "configs/eda.yaml periodicity.cv_threshold")
    m.add("PeriodicCH", int(per.loc["car_hacking", "n_periodic"]), "periodicity_summary.csv car_hacking n_periodic")
    m.add("IdsCH", int(per.loc["car_hacking", "n_ids"]), "periodicity_summary.csv car_hacking n_ids")
    m.add("MedianCVCH", ffix(per.loc["car_hacking", "median_cv"], 3), "periodicity_summary.csv car_hacking median_cv")
    m.add("RobustPeriodicROAD", int(per.loc["road", "n_robust_spread_lt_threshold"]),
          "periodicity_summary.csv road n_robust_spread_lt_threshold")
    m.add("IdsROAD", int(per.loc["road", "n_ids"]), "periodicity_summary.csv road n_ids")
    # masquerade lineage (P1, K-007)
    lin = s.lineage[s.lineage.derived.astype(str).eq("True")]
    if not lin.verified.astype(str).eq("True").all():
        raise ValueError("unverified masquerade lineage")
    m.add("NMasqCaptures", len(lin), "capture_lineage.csv derived rows")
    m.add("MasqInSourcePct", fpct(lin.derived_frames_in_source.astype(float).min()),
          "capture_lineage.csv derived_frames_in_source (min)")
    if not (lin.injected_frames_kept.astype(float) == lin.injected_frames_in_source.astype(float)).all():
        raise ValueError("masquerade logs do not keep all injected frames")


# --------------------------------------------------------------------------------------------
# Results (Section VI; P9, docs/DECISIONS.md K-069)
# --------------------------------------------------------------------------------------------

ST = "stats.csv"


def one(df: pd.DataFrame, what: str):
    if len(df) != 1:
        raise KeyError(f"{what}: {len(df)} rows")
    return df.iloc[0]


def stat_row(s: Sources, analysis: str, **kw):
    return one(s.d.stat(analysis, **kw), f"{analysis} {kw}")


def note_value(note: str, key: str) -> float:
    """A number written into the note column of stats.csv, e.g. "Cliff's delta(...)=0.5503"."""
    hit = re.search(re.escape(key) + r"=([-+0-9.e]+)", note)
    if not hit:
        raise KeyError(f"{key} not in note: {note}")
    return float(hit.group(1))


def rq1_macros(m: Macros, s: Sources) -> None:
    """Per-type loss (H1) and the coarse-stealthy comparison (H2)."""
    d, an = s.d, s.cfg["analysis"]
    # H1 (a): pooled dF1 at INT8, seed mean per deep model x dataset
    df = d.delta_f1
    pooled = (df[df.level.eq("pooled") & df.threshold_mode.eq("recalibrated") & df.metric.eq("f1")
                 & df.quantization.eq("q2")].groupby(["model", "dataset"]).delta.mean())
    check(pooled.idxmax() == ("cnn1d", "road"), "the largest pooled INT8 F1 loss is the 1D-CNN on ROAD")
    check(pooled.max() <= an["hypotheses"]["h1_pooled_df1_max"], "H1 (a) holds")
    m.add("PooledDFOneMax", ffix(pooled.max(), 4), "delta_f1.csv pooled f1 q2 recalibrated (max seed mean)")
    # H1 (b): per-type dRecall(Q2) intervals of the deep models on the primary types
    pt = d.stat("per_type_delta", statistic="d_recall", quantization="q2")
    pt = pt[pt.note.str.contains("role=primary")]
    loss, zero, gain = pt[pt.ci_low > 0], pt[(pt.ci_low <= 0) & (pt.ci_high >= 0)], pt[pt.ci_high < 0]
    m.add("NTypeRows", len(pt), f"{ST} per_type_delta d_recall q2 primary (deep models)")
    m.add("NLossRows", len(loss), f"{ST} per_type_delta d_recall q2: ci_low > 0")
    m.add("NZeroRows", len(zero), f"{ST} per_type_delta d_recall q2: interval contains 0")
    m.add("NGainRows", len(gain), f"{ST} per_type_delta d_recall q2: ci_high < 0")
    check(loss.model.eq("cnn1d").all() and loss.dataset.eq("road").all(), "every loss row is the 1D-CNN on ROAD")
    m.add("LossRowsRange", frange(loss.estimate.min(), loss.estimate.max(), 4), f"{ST} per_type_delta loss rows")
    g = one(gain, "gain row")
    check((g.model, g.dataset, g.unit) == ("conv_ae", "car_hacking", "gear_spoofing"),
          "the only gain is the conv-AE on Car-Hacking gear spoofing")
    check((FG.seeds(g.seed_values) < 0).all(), "the gear-spoofing gain has the same sign in every seed")
    m.add("GearGainCI", fci(g.estimate, g.ci_low, g.ci_high, 3), f"{ST} per_type_delta conv_ae CH gear_spoofing q2")
    auc = stat_row(s, "per_type_delta", model="conv_ae", dataset="car_hacking", quantization="q2",
                   unit="gear_spoofing", statistic="d_roc_auc")
    m.add("GearGainAUC", fsci(auc.estimate), f"{ST} per_type_delta conv_ae CH gear_spoofing q2 d_roc_auc")
    ch = d.stat("per_type_delta", model="cnn1d", dataset="car_hacking", statistic="d_recall")
    check(ch.estimate.eq(0).all(), "1D-CNN dRecall is exactly 0 on Car-Hacking (ceiling)")
    par = d.pareto.set_index(["dataset", "model", "quantization"])
    m.add("CHCNNRecall", ffix(par.loc[("car_hacking", "cnn1d", "q0"), "pooled_recall_mean"], 4),
          "pareto.csv pooled_recall_mean car_hacking cnn1d q0")

    # H2 (a): paired units (block bootstrap; iid as sensitivity)
    def pooled_d(stat, q, block="block"):
        return stat_row(s, "as1_pooled", statistic=stat, quantization=q, block=block)

    r = pooled_d("mean_D_R", "q2")
    check(r.ci_low < 0 < r.ci_high, "the mean D interval at INT8 contains 0")
    m.add("DRQTwoCI", fci(r.estimate, r.ci_low, r.ci_high, 4), f"{ST} as1_pooled mean_D_R q2 block")
    m.add("DBoundPts", ffix(100 * r.ci_high, 1), f"{ST} as1_pooled mean_D_R q2 ci_high x 100 (recall points)")
    m.add("NPairUnits", int(r.n_units), f"{ST} as1_pooled n_units")
    m.add("PairMinP", ffix(r.min_attainable_p, 2), f"{ST} as1_pooled min_attainable_p")
    r1 = pooled_d("mean_D_R", "q1")
    m.add("DRQOneCI", fci(r1.estimate, r1.ci_low, r1.ci_high, 4), f"{ST} as1_pooled mean_D_R q1 block")
    rf = pooled_d("mean_D_F1", "q2")
    m.add("DFQTwoCI", fci(rf.estimate, rf.ci_low, rf.ci_high, 4), f"{ST} as1_pooled mean_D_F1 q2 block")
    ri = pooled_d("mean_D_R", "q2", "iid")
    m.add("DRQTwoIid", f"[{fnum(ri.ci_low, 4)}, {fnum(ri.ci_high, 4)}]", f"{ST} as1_pooled mean_D_R q2 iid")
    pair = d.stat("as1_pair", quantization="q2", block="block")
    pair = pair[pair.note.eq("primary unit")]
    check(pair.unit.nunique() == r.n_units, "primary units of as1_pair and as1_pooled agree")
    dr = pair[pair.statistic.eq("D_R")]
    check(((dr.ci_low < 0) & (dr.ci_high > 0)).all(), "every unit's D interval at INT8 contains 0")
    sides = pair[pair.statistic.isin(["dR_fab", "dR_masq"])]
    m.add("PairSideRange", frange(sides.estimate.min(), sides.estimate.max(), 4),
          f"{ST} as1_pair dR_fab / dR_masq q2 primary units")

    # H2 (a), secondary: group test on the ROAD primary types; sensitivity with the Car-Hacking coarse types
    gm = d.stat("group_mwu")
    tag = {"cnn1d": "CNN", "conv_ae": "AE"}
    qtag = {"q1": "QOne", "q2": "QTwo"}
    for fam, pre, suffix in (("group_road_drecall", "Group", ""), ("group_road_df1", "Group", "F"),
                             ("group_road+car_hacking_coarse_drecall", "Sens", "")):
        for row in gm[gm.holm_family.eq(fam)].itertuples():
            name = f"{pre}{tag[row.model]}{qtag[row.quantization]}{suffix}"
            src = f"{ST} group_mwu {fam} {row.model} {row.quantization}"
            m.add(f"{name}CI", fci(row.estimate, row.ci_low, row.ci_high, 2), f"{src} (Cliff's delta)")
            m.add(f"{name}P", fp(row.p_value), f"{src} p_value")
            m.add(f"{name}PHolm", fp(row.p_holm), f"{src} p_holm")
    for fam, pre in (("group_road_drecall", "Group"), ("group_road+car_hacking_coarse_drecall", "Sens")):
        f = gm[gm.holm_family.eq(fam)]
        n_s, n_c = (int(x) for x in re.fullmatch(r"stealthy n=(\d+) vs coarse n=(\d+)", same(f.unit, "unit")).groups())
        minp = same(f.min_attainable_p.round(12), "min p")
        check(abs(minp - 2 / math.comb(n_s + n_c, n_s)) < 1e-6, f"{pre}: min p = 2 / C(n, k)")
        check((minp * len(f) > 0.05) == (pre == "Group"), f"{pre}: adjusted significance unattainable only in the ROAD test")
        m.add(f"{pre}MinP", fsig(minp, 2), f"{ST} group_mwu {fam} min_attainable_p")
        m.add(f"{pre}MinPHolm", fsig(minp * len(f), 2), f"{ST} group_mwu {fam} min_attainable_p x tests (Holm)")
        if pre == "Group":
            m.add("GroupMinPFrac", f"2/{math.comb(n_s + n_c, n_s)}", f"{ST} group_mwu exact MWU 2 / C({n_s + n_c}, {n_s})")
        else:
            m.add("SensCoarse", n_c, f"{ST} group_mwu {fam} coarse n")
    road_fams = gm[gm.holm_family.isin(["group_road_drecall", "group_road_df1"])]
    m.add("GroupMinHolm", fp(road_fams.p_holm.min()), f"{ST} group_mwu ROAD families min p_holm")
    ptr = d.stat("per_type_delta", model="cnn1d", dataset="road", quantization="q2", statistic="d_recall")
    ptr = ptr[ptr.note.str.contains("role=primary")]
    st_mean = ptr[ptr.note.str.contains("group=stealthy")].estimate.mean()
    co_mean = ptr[ptr.note.str.contains("group=coarse")].estimate.mean()
    note = stat_row(s, "group_mwu", holm_family="group_road_drecall", model="cnn1d", quantization="q2").note
    check(abs(note_value(note, "stealthy") - st_mean) < 1e-6 and abs(note_value(note, "coarse") - co_mean) < 1e-6,
          "group means of per_type_delta equal those of the group test")
    m.add("GroupMeanStealthy", ffix(st_mean, 4), f"{ST} per_type_delta cnn1d road q2 stealthy mean")
    m.add("GroupMeanCoarse", ffix(co_mean, 4), f"{ST} per_type_delta cnn1d road q2 coarse mean")
    m.add("GroupDiffPts", ffix(100 * (st_mean - co_mean), 1), "GroupMeanStealthy - GroupMeanCoarse (recall points)")

    # H2 (b): coarse types still detected under INT8
    mm = d.metric_means("recall").set_index(["dataset", "model", "quantization", "level", "attack_type"])
    coarse = [("car_hacking", "dos"), ("car_hacking", "fuzzy"), ("road", "fuzzing")]
    q2 = {(ds, mod, t): mm.loc[(ds, mod, "q2", "attack_type", t), "mean"] for ds, t in coarse for mod in FG.DEEP}
    fz = ("road", "conv_ae", "fuzzing")
    others = [v for k, v in q2.items() if k != fz]
    check(q2[fz] < an["hypotheses"]["h2_coarse_recall_min"], "conv-AE ROAD fuzzing is below the H2 (b) bar")
    check(min(others) >= an["hypotheses"]["h2_coarse_recall_min"], "every other coarse recall meets the H2 (b) bar")
    m.add("AEFuzzQTwo", ffix(q2[fz], 3), "all_results.csv recall road conv_ae q2 fuzzing (seed mean)")
    q0 = mm.loc[("road", "conv_ae", "q0", "attack_type", "fuzzing")]
    m.add("AEFuzzQZero", ffix(q0["mean"], 3), "all_results.csv recall road conv_ae q0 fuzzing (seed mean)")
    m.add("AEFuzzQZeroSD", ffix(q0["std"], 3), "all_results.csv recall road conv_ae q0 fuzzing (seed s.d.)")
    m.add("CoarseOtherMin", ffix(min(others), 3), "all_results.csv recall q2 CH dos/fuzzy, ROAD fuzzing (min, others)")

    # further observation: masquerade vs fabrication at Q0
    for mod, mt in (("cnn1d", "CNN"), ("xgb", "XGB")):
        q = "q0" if mod != "xgb" else "native"
        for base, bt in (("correlated_signal", "Corr"), ("max_speedometer", "Speed"), ("reverse_light_off", "Rev")):
            for t, side in ((base, "Fab"), (base + FG.MASQ, "Masq")):
                m.add(f"Rec{mt}{bt}{side}", ffix(mm.loc[("road", mod, q, "attack_type", t), "mean"], 3),
                      f"all_results.csv recall road {mod} {q} {t} (seed mean)")


def rq2_macros(m: Macros, s: Sources) -> None:
    """Latency, size and recall (H3, Pareto front, budgets)."""
    d = s.d
    par = d.pareto.set_index(["dataset", "model", "quantization"])
    sz = d.model_metric("size_mb").groupby(["dataset", "model", "quantization"]).value.mean()
    for k, name in (("q1", "SizeRatioQOne"), ("q2", "SizeRatioQTwo")):
        ratios = [sz[(ds, mod, k)] / sz[(ds, mod, "q0")] for ds in DS_TAG for mod in FG.DEEP]
        m.add(name, frange(min(ratios), max(ratios), 2), f"all_results.csv size_mb {k}/q0 (seed means)")
    for mod, mt in (("cnn1d", "CNN"), ("conv_ae", "AE")):
        for k, kt in (("q0", "QZero"), ("q1", "QOne"), ("q2", "QTwo")):
            m.add(f"Size{mt}{kt}", ffix(same([round(par.loc[(ds, mod, k), "size_mb"], 3) for ds in DS_TAG], "size"), 3),
                  f"pareto.csv size_mb {mod} {k}")
    for ds, tag in DS_TAG.items():
        m.add(f"SizeXGB{tag}", ffix(par.loc[(ds, "xgb", "native"), "size_mb"], 3), f"pareto.csv size_mb xgb {ds}")
    # H3 (a): mean |dRecall| per type, Q1 vs Q2
    pt = d.stat("per_type_delta", statistic="d_recall")
    pt = pt[pt.note.str.contains("role=primary")]
    absd = pt.assign(a=pt.estimate.abs()).groupby(["model", "dataset", "quantization"]).a.mean()
    check(all(absd[(mod, ds, "q1")] <= absd[(mod, ds, "q2")] for mod in FG.DEEP for ds in DS_TAG), "H3 (a) holds")
    m.add("AbsDRQOne", ffix(absd[("cnn1d", "road", "q1")], 4), f"{ST} per_type_delta mean |d_recall| cnn1d road q1")
    m.add("AbsDRQTwo", ffix(absd[("cnn1d", "road", "q2")], 4), f"{ST} per_type_delta mean |d_recall| cnn1d road q2")
    # latency: orders of magnitude and within-session ratios (two sessions)
    lat = d.pareto[d.pareto.latency_measured.astype(str).eq("True")]
    deep, xgb = lat[lat.model.isin(FG.DEEP)], lat[lat.model.eq("xgb")]

    def both(t):
        return pd.concat([t.latency_median_ms_s1, t.latency_median_ms_s2])

    m.add("DeepLatRange", frange(both(deep).min(), both(deep).max(), 2), "pareto.csv latency_median_ms_s1/s2 deep")
    m.add("DeepLatMax", ffix(both(deep).max(), 2), "pareto.csv latency_median_ms_s1/s2 deep (max)")
    m.add("XGBLatRange", frange(both(xgb).min(), both(xgb).max(), 2), "pareto.csv latency_median_ms_s1/s2 xgb")
    cnn2 = lat[lat.model.eq("cnn1d") & lat.quantization.eq("q2")]
    check((cnn2[["latency_ratio_vs_ref_s1", "latency_ratio_vs_ref_s2"]] < 1).all().all(),
          "INT8 lowers the 1D-CNN median latency in both sessions and datasets")
    lo, hi = cnn2.latency_ratio_min.min(), cnn2.latency_ratio_max.max()
    m.add("CNNRatio", frange(lo, hi, 2), "pareto.csv latency_ratio_min/max cnn1d q2")
    m.add("CNNCutPct", f"{round(100 * (1 - hi))}--{round(100 * (1 - lo))}", "1 - CNNRatio (percent)")
    aer = [par.loc[(ds, "conv_ae", "q2"), f"latency_median_ms_s{k}"]
           / par.loc[(ds, "conv_ae", "q0"), f"latency_median_ms_s{k}"] for ds in DS_TAG for k in (1, 2)]
    check(min(aer) > 1, "INT8 does not lower the conv-AE median latency in any session")
    m.add("AERatio", frange(min(aer), max(aer), 2), "pareto.csv conv_ae q2/q0 latency_median_ms per session")
    ae = lat[lat.model.eq("conv_ae")]
    m.add("AEvsCNN", frange(ae.latency_ratio_min.min(), ae.latency_ratio_max.max(), 1),
          "pareto.csv latency_ratio_min/max conv_ae (vs CNN Q0)")
    m.add("XGBvsCNN", frange(xgb.latency_ratio_min.min(), xgb.latency_ratio_max.max(), 1),
          "pareto.csv latency_ratio_min/max xgb (vs CNN Q0)")
    # recall axes
    for mod, k, name in (("cnn1d", "q0", "StealthCNNQZero"), ("cnn1d", "q2", "StealthCNNQTwo")):
        r = par.loc[("road", mod, k)]
        m.add(name, fci(r.stealthy_recall_mean, r.stealthy_recall_ci_low, r.stealthy_recall_ci_high, 3),
              f"pareto.csv stealthy_recall_mean/ci road {mod} {k}")
    diff = (par.loc[("road", "cnn1d", "q0"), "stealthy_recall_mean"]
            - par.loc[("road", "cnn1d", "q2"), "stealthy_recall_mean"])
    m.add("StealthDiff", ffix(diff, 3), "pareto.csv stealthy_recall_mean road cnn1d q0 - q2")
    for mod, k, name in (("conv_ae", "q0", "StealthAEQZero"), ("conv_ae", "q2", "StealthAEQTwo"),
                         ("xgb", "native", "StealthXGB"), ("cnn1d", "q1", "StealthCNNQOne")):
        m.add(name, ffix(par.loc[("road", mod, k), "stealthy_recall_mean"], 3),
              f"pareto.csv stealthy_recall_mean road {mod} {k}")
    m.add("PooledCHCNNQTwo", ffix(par.loc[("car_hacking", "cnn1d", "q2"), "pooled_recall_mean"], 4),
          "pareto.csv pooled_recall_mean car_hacking cnn1d q2")
    m.add("PooledCHXGB", ffix(par.loc[("car_hacking", "xgb", "native"), "pooled_recall_mean"], 3),
          "pareto.csv pooled_recall_mean car_hacking xgb")

    # robust fronts and budgets as stated in the text
    def front(ds, col):
        return {(i[1], i[2]) for i, v in par[col].items() if i[0] == ds and str(v) == "True"}

    check(front("road", "robust_front_stealthy_recall_mean") == {("cnn1d", "q0"), ("cnn1d", "q2")},
          "ROAD robust front = 1D-CNN Q0 and Q2")
    check(front("car_hacking", "robust_front_pooled_recall_mean") == {("cnn1d", "q2"), ("xgb", "native")},
          "Car-Hacking robust front = 1D-CNN Q2 and XGBoost")
    check(("cnn1d", "q1") in front("road", "size_front_stealthy_recall_mean"), "Q1 is on the ROAD size-recall front")
    check(lat.meets_window_budget.astype(float).eq(1).all(), "every candidate meets the window budget")
    check(deep.meets_strict_budget.astype(float).eq(1).all(), "every deep model meets the strict budget")
    check(par.loc[("road", "xgb", "native"), "meets_strict_budget"] == 0, "XGBoost misses the strict budget on ROAD")
    check(par.loc[("car_hacking", "xgb", "native"), "meets_strict_budget"] == 1, "XGBoost meets it on Car-Hacking")
    for ds, tag in DS_TAG.items():
        r = par.loc[(ds, "xgb", "native")]
        m.add(f"XGBSlow{tag}", ffix(max(r.latency_median_ms_s1, r.latency_median_ms_s2), 3),
              f"pareto.csv xgb {ds} slower-session median latency")


def rq3_macros(m: Macros, s: Sources) -> None:
    """Supervised vs unsupervised (H4): pre-registered statistic, signed check, output-change measures."""
    for stat, tag in (("mean_abs_diff_droc_auc", "AUC"), ("mean_abs_diff_drecall", "Rec")):
        for q, qt in (("q1", "QOne"), ("q2", "QTwo")):
            r = stat_row(s, "as3", statistic=stat, quantization=q)
            src = f"{ST} as3 {stat} {q}"
            if max(abs(r.estimate), abs(r.ci_low), abs(r.ci_high)) < 1e-3:
                ci = f"{fsci(r.estimate)} [{fsci(r.ci_low)}, {fsci(r.ci_high)}]"
            else:
                ci = fci(r.estimate, r.ci_low, r.ci_high, 4)
            m.add(f"Pre{tag}{qt}CI", ci, src)
            m.add(f"Pre{tag}{qt}P", fp(r.p_value), f"{src} Wilcoxon p")
            m.add(f"Pre{tag}{qt}PHolm", fp(r.p_holm), f"{src} p_holm")
            m.add(f"Pre{tag}{qt}Delta", ffix(note_value(r.note, "Cliff's delta(|d_conv_ae| vs |d_cnn1d|)"), 2),
                  f"{src} note: Cliff's delta of |d|")
        r = stat_row(s, "as3", statistic=stat, quantization="q2")
        check(r.ci_low > 0, f"pre-registered {tag} interval at INT8 is above 0")
        n_family = len(s.d.stat("as3"))
        m.add(f"Pre{tag}Units", int(r.n_units), f"{ST} as3 {stat} n_units")
        m.add(f"Pre{tag}MinP", fsig(r.min_attainable_p, 3), f"{ST} as3 {stat} min_attainable_p")
        m.add(f"Pre{tag}MinPHolm", fsig(r.min_attainable_p * n_family, 3),
              f"{ST} as3 {stat} min_attainable_p x tests in the family (Holm)")
    for stat, tag in (("mean_signed_diff_droc_auc", "AUC"), ("mean_signed_diff_drecall", "Rec")):
        r = stat_row(s, "as3_signed_diff", statistic=stat, quantization="q2")
        check(r.ci_low < 0 < r.ci_high, f"signed {tag} interval at INT8 contains 0")
        m.add(f"Sgn{tag}QTwoCI", fci(r.estimate, r.ci_low, r.ci_high, 4), f"{ST} as3_signed_diff {stat} q2")
    r = stat_row(s, "as3_signed_diff", statistic="mean_signed_diff_droc_auc", quantization="q1")
    check(r.ci_high < 0 and abs(r.ci_low) < 1e-3, "signed Q1 AUC difference is below 0 and below 1e-3 in size")
    m.add("SgnAUCQOneCI", f"{fsci(r.estimate)} [{fsci(r.ci_low)}, {fsci(r.ci_high)}]", f"{ST} as3_signed_diff auc q1")
    for mod, mt in (("conv_ae", "AE"), ("cnn1d", "CNN")):
        r = stat_row(s, "as3_signed", model=mod, statistic="mean_droc_auc", quantization="q2")
        check(r.ci_low < 0 < r.ci_high, f"{mod} mean dAUC interval contains 0")
        m.add(f"Mean{mt}AUC", fci(r.estimate, r.ci_low, r.ci_high, 4), f"{ST} as3_signed {mod} mean_droc_auc q2")
    # seed variability example (ROAD correlated signal, dAUC at INT8)
    for mod, mt, n in (("conv_ae", "AE", 2), ("cnn1d", "CNN", 1)):
        r = stat_row(s, "per_type_delta", model=mod, dataset="road", quantization="q2", unit="correlated_signal",
                     statistic="d_roc_auc")
        m.add(f"SD{mt}Corr", fsig(FG.seeds(r.seed_values).std(ddof=1), n),
              f"{ST} per_type_delta {mod} road correlated_signal d_roc_auc q2 (seed s.d.)")
    # output change (descriptive, min-max over seeds)
    ch = s.d.stat("as3_churn", quantization="q2")

    def val(mod, ds, st):
        return one(ch[ch.model.eq(mod) & ch.dataset.eq(ds) & ch.statistic.eq(st)], f"as3_churn {mod} {ds} {st}")

    for ds, tag in DS_TAG.items():
        for mod, mt in (("conv_ae", "AE"), ("cnn1d", "CNN")):
            c, sp = val(mod, ds, "churn_attack"), val(mod, ds, "spearman_q0_qk")
            src = f"{ST} as3_churn {mod} {ds} q2"
            m.add(f"Churn{mt}{tag}", "0" if c.estimate == 0 else ffix(c.estimate, 4), f"{src} churn_attack")
            m.add(f"Churn{mt}{tag}Range", frange(c.ci_low, c.ci_high, 4), f"{src} churn_attack min-max")
            nd = 3 if (mt, ds) == ("AE", "road") else 4
            m.add(f"Spear{mt}{tag}", ffix(sp.estimate, nd), f"{src} spearman_q0_qk")
            m.add(f"Spear{mt}{tag}Range", frange(sp.ci_low, sp.ci_high, nd), f"{src} spearman_q0_qk min-max")
    ratio = val("conv_ae", "road", "churn_attack").estimate / val("cnn1d", "road", "churn_attack").estimate
    check(ratio > 1, "INT8 changes more attack-window decisions for the conv-AE on ROAD")
    m.add("ChurnRatioROAD", ffix(ratio, 1), f"{ST} as3_churn churn_attack road conv_ae / cnn1d")


def calib_macros(m: Macros, s: Sources) -> None:
    """Calibration shift: the float32 threshold kept for the quantized model."""
    c = s.calib
    src = "calibration_shift.csv"

    def sub(ds, mod, q):
        return c[c.dataset.eq(ds) & c.model.eq(mod) & c.quantization.eq(q)]

    r = sub("road", "conv_ae", "q2")
    st = s.d.stat("calib_shift", model="conv_ae", dataset="road", quantization="q2")
    for col, nd0, tag in (("fpr_val", 4, "Val"), ("fpr_test", 4, "Test"), ("pooled_recall", 3, "Rec")):
        q0, qk = r[f"{col}_q0"], r[f"{col}_qk"]
        check(abs(one(st[st.statistic.eq(f"{col}_qk")], col).estimate - qk.mean()) < 1e-6,
              f"{col} of calibration_shift.csv agrees with stats.csv")
        m.add(f"Cal{tag}QZero", ffix(q0.mean(), nd0), f"{src} road conv_ae q2 {col}_q0 (seed mean)")
        m.add(f"Cal{tag}QTwo", ffix(qk.mean(), 3), f"{src} road conv_ae q2 {col}_qk (seed mean)")
        m.add(f"Cal{tag}Range", frange(qk.min(), qk.max(), 3), f"{src} road conv_ae q2 {col}_qk (min-max over seeds)")
    fold = r.fpr_val_qk / r.fpr_val_q0
    m.add("CalFold", f"{round(fold.min())}--{round(fold.max())}",
          f"{src} road conv_ae q2 fpr_val_qk / fpr_val_q0 per seed (min-max)")
    m.add("CalTypeMin", fnum(r.type_d_recall_min.min(), 2), f"{src} road conv_ae q2 type_d_recall_min (min)")
    rc = sub("car_hacking", "conv_ae", "q2")
    m.add("CalCHValQZero", ffix(rc.fpr_val_q0.mean(), 4), f"{src} car_hacking conv_ae q2 fpr_val_q0 (seed mean)")
    m.add("CalCHValQTwo", ffix(rc.fpr_val_qk.mean(), 4), f"{src} car_hacking conv_ae q2 fpr_val_qk (seed mean)")
    g = c.groupby(["dataset", "model", "quantization"])[["fpr_val_q0", "fpr_val_qk", "fpr_test_q0", "fpr_test_qk"]].mean()
    shift = pd.concat([(g.fpr_val_qk - g.fpr_val_q0).abs(), (g.fpr_test_qk - g.fpr_test_q0).abs()], axis=1).max(axis=1)
    lv = shift.index.get_level_values
    m.add("ShiftQOneMax", ffix(shift[lv("quantization") == "q1"].max(), 4),
          f"{src} max |FPR_qk - FPR_q0| over val/test (seed means), q1")
    m.add("ShiftCNNMax", ffix(shift[lv("model") == "cnn1d"].max(), 4),
          f"{src} max |FPR_qk - FPR_q0| over val/test (seed means), cnn1d")


# --------------------------------------------------------------------------------------------
# Generated tables
# --------------------------------------------------------------------------------------------



def signature_table(s: Sources) -> str:
    sig = s.d.attack_signatures
    meta = s.d.type_meta()
    order = list(zip(meta.dataset, meta.attack_type))
    eda = s.cfg["eda"]["grouping_rule"]

    def cell(x, nd, hit):
        if isinstance(x, float) and np.isnan(x):
            v = "--"
        elif isinstance(x, float) and np.isinf(x):
            v = "$\\infty$"
        else:
            v = ffix(x, nd)
        return f"\\textbf{{{v}}}" if hit else v

    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\caption{Traffic criteria that define the attack groups (exploratory analysis, before any model was "
             "trained). C1: share of attack-period frames with an ID unseen in attack-free traffic (coarse if "
             f"$>{eda['C1_unseen_id_share']['threshold']}$). C2: median Jensen--Shannon divergence of the ID "
             f"distribution of {fint(eda['C2_id_distribution_jsd']['window_frames'])}-frame attack windows against "
             "the normal period of the same capture (coarse if above the "
             f"{round(100 * eda['C2_id_distribution_jsd']['null_quantile'])}th percentile of an attack-free null: "
             + ", ".join(f"{FG.DS[ds]} {null_q(sig, ds)}" for ds in DS_TAG)
             + "). C3, C4: target-ID rate and median inter-arrival-time ratio, attack/normal period (coarse if "
             f"outside $[1/{ffix(eda['C3_target_rate_ratio']['band'][1], 1)},\\,"
             f"{ffix(eda['C3_target_rate_ratio']['band'][1], 1)}]$; not defined without a single target ID). "
             "Bold: criterion fires. A type is coarse if any criterion fires.}",
             "\\label{tab:signatures}", "\\begin{tabular}{@{}lrrrrl@{}}", "\\toprule",
             "Attack type & C1 & C2 & C3 & C4 & Group \\\\", "\\midrule"]
    first = True
    for ds in DS_TAG:
        if not first:
            lines.append("\\midrule")
        first = False
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{FG.DS_LONG[ds]}}}}} \\\\")
        for d_, t in order:
            if d_ != ds:
                continue
            r = sig[sig.dataset.eq(ds) & sig.attack_type.eq(t)].iloc[0]
            fired = set(str(r.triggered).split(",")) if isinstance(r.triggered, str) else set()
            vals = [cell(r.C1_unseen_id_share, 3, "C1" in fired), cell(r.C2_jsd_median, 3, "C2" in fired),
                    cell(r.C3_target_rate_ratio, 2, "C3" in fired), cell(r.C4_target_median_iat_ratio, 2, "C4" in fired)]
            lines.append(" & ".join([FG.tex(FG.type_label(t)), *vals, r.proposed_group]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


SPLIT_TEXT = {"car_hacking": "disjoint time blocks of each file", "road": "by source capture"}


def dataset_table(s: Sources) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        df = FG.table1_dataset(s.d, Path(tmp), [])
    w = s.cfg["data"]["windows"]
    rows = []
    for ds in DS_TAG:
        rows.append(("section", f"{FG.DS_LONG[ds]} (split: {SPLIT_TEXT[ds]})"))
        for r in df[df.dataset.eq(ds)].itertuples():
            normal = r.attack_type == "normal"
            if normal:
                note = "frames: attack-free captures only"
            elif "test only" in r.split_rule:
                note = "test only (unseen in training)"
            elif "no val" in r.split_rule:
                note = "no validation capture"
            else:
                note = ""
            rows.append(["attack-free" if normal else FG.type_label(r.attack_type), "--" if normal else r.group,
                         f"{r.captures_train}/{r.captures_val}/{r.captures_test}", f"{r.frames:,}",
                         "--" if not r.injected_frames else f"{r.injected_frames:,}",
                         f"{r.windows_train:,}", f"{r.windows_val:,}", f"{r.windows_test:,}", note])
    header = ["Attack type & Group & Captures & Frames & Injected & \\multicolumn{3}{c}{Windows} & Note \\\\",
              "\\cmidrule(lr){6-8}", "& & tr/va/te & & frames & train & val & test & \\\\"]
    cap = (f"Datasets and splits. Captures: captures that contribute windows of the row to the training, validation "
           f"and test split (attack-free windows also come from the attack-free parts of attack captures). Frames "
           f"and injected frames: whole captures of that attack type. Windows of N = {w['size']} frames, stride "
           f"{w['stride']['train']} in training and {w['stride']['test']} (non-overlapping) in validation and "
           f"test. ROAD is split by source capture, so no capture, and no masquerade log derived from it, feeds two "
           f"splits; Car-Hacking has one file per attack, which is cut into disjoint time blocks separated by "
           f"discarded gaps.")
    return FG.latex_table(header, rows, "@{}llrrrrrrl@{}", cap, "tab:dataset")


def compact(latex: str, sep: str = "3pt") -> str:
    """Narrower column gaps so that a P7 table fits the IEEE text width."""
    if latex.count("\\footnotesize\n") != 1:
        raise ValueError("unexpected table preamble")
    return latex.replace("\\footnotesize\n", f"\\footnotesize\n\\setlength{{\\tabcolsep}}{{{sep}}}\n")


def result_tables(s: Sources) -> dict[str, str]:
    """Tables II (recall; F1 in the appendix) and III of the P7 generator; their captions carry no decision ids."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        FG.table2_main(s.d, out, [], "recall")
        FG.table2_main(s.d, out, [], "f1")
        FG.table3_resources(s.d, out, [])
        return {name: compact((out / f"{stem}.tex").read_text(encoding="utf-8"))
                for name, stem in (("table_recall.tex", "table_2_recall"), ("table_f1.tex", "table_2b_f1"),
                                   ("table_resources.tex", "table_3_resources"))}


PAPER_STAT_NAMES = {"(K-062)": "(pre-registered)", "(K-063)": "(corrected)"}


def stats_table(s: Sources) -> str:
    """Table IV for the paper: the rows of analysis.figures.table4_data with titles and caption without decision ids."""
    df = FG.table4_data(s.d)
    gm = s.d.stat("group_mwu")

    def sizes(fam):
        f = gm[gm.holm_family.eq(fam)]
        hit = re.fullmatch(r"stealthy n=(\d+) vs coarse n=(\d+)", same(f.unit, "unit"))
        return int(hit.group(1)), int(hit.group(2)), len(f), same(f.min_attainable_p.round(12), "min p")

    n_s, n_c, n_tests, minp = sizes("group_road_drecall")
    _, n_c2, _, _ = sizes("group_road+car_hacking_coarse_drecall")
    b = df[df.block.eq("B")]
    n_pairs = same([int(re.search(r"\((\d+) pairs\)", c).group(1)) for c in b.comparison], "pairs")
    pair_p = same(b.min_attainable_p.round(12), "pair min p")
    n_c_tests = len(df[df.block.eq("C") & df.p_holm.notna()])
    titles = {
        "A": f"A: coarse vs stealthy, group level, ROAD ({n_s} stealthy vs {n_c} coarse types; exact Mann–Whitney U, "
             f"Holm over {n_tests} tests)",
        "A'": f"A': sensitivity, Car-Hacking coarse types added ({n_s} vs {n_c2}; dataset and group confounded)",
        "B": f"B: paired fabrication–masquerade units, 1D-CNN, ROAD ({n_pairs} pairs; exploratory, not tested)",
        "C": f"C: supervised vs unsupervised, conv-AE vs 1D-CNN (Wilcoxon signed-rank, Holm over {n_c_tests} tests)"}
    nboot = s.cfg["analysis"]["bootstrap"]["n_boot"]
    cap = (f"Statistical comparisons (Δ = Q0 − Qk; positive = loss) with 95% hierarchical bootstrap intervals over "
           f"seeds and test windows (moving blocks, B = {nboot:,}). min. p: smallest p-value the design can attain. "
           f"In block A it is 2/{math.comb(n_s + n_c, n_s)} = {minp:.3f}, which the Holm correction raises to "
           f"{minp * n_tests:.3f} > 0.05, so adjusted significance is unattainable whatever the data; block B is not "
           f"tested (min. p {pair_p:.2f}). Block C: the pre-registered statistic, the mean over attack types of "
           f"|ΔAE| − |ΔCNN|, is biased upward for the model whose Δ varies more across seeds; the corrected statistic "
           f"is the signed difference ΔAE − ΔCNN. The final H4 verdict (partially supported) rests on both, read "
           f"together with the output-change measures.")

    def name(stat: str) -> str:
        for k, v in PAPER_STAT_NAMES.items():
            stat = stat.replace(k, v)
        return stat

    latex = compact(FG.table4_latex(df, titles, cap, name), "2.5pt")
    if re.search(r"K-\d{3}", latex):
        raise ValueError("decision id left in the paper version of Table IV")
    return latex


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def build(s: Sources | None = None) -> dict[str, str]:
    s = s or Sources()
    m = Macros()
    for fn in (data_macros, feature_macros, model_macros, run_macros, quant_macros, bench_macros,
               analysis_macros, signature_macros, rq1_macros, rq2_macros, rq3_macros, calib_macros):
        fn(m, s)
    files = {"table_signatures.tex": signature_table(s), "table_dataset.tex": dataset_table(s),
             **result_tables(s), "table_stats.tex": stats_table(s)}
    files = {"numbers.tex": m.render(s.inp.files), **files}
    return files


def write(files: dict[str, str], out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in files.items():
        p = out / name
        p.write_bytes(text.encode("utf-8"))
        written.append(p)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    for p in write(build(), args.out):
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
