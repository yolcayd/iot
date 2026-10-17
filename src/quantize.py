"""Post-training conversions of the deep models: Q0 (float32 ONNX), Q1 (float16), Q2 (INT8 PTQ).

Settings: configs/quantization.yaml; decisions: docs/DECISIONS.md K-042 ... K-045.

* Q0: the trained PyTorch model exported to ONNX (opset 17). The graph outputs the anomaly score that
  ``src.train.score_deep`` computes: the classifier logit, or the reconstruction MSE of the autoencoders (the
  score is part of the graph, so the benchmark times the whole inference). Checked against PyTorch on CPU.
* Q1: float16 conversion of the Q0 graph (inputs and outputs stay float32). Which kernels the ONNX Runtime CPU
  execution provider really runs in float16 is read from its profiler. If fewer than
  ``q1.min_heavy_fp16_share`` of the heavy layers run in float16, Q1 is ``cpu_unsupported`` and, by the user
  decision K-043, its accuracy comes from real float16 arithmetic in PyTorch (weights and activations in half
  precision); without that decision ``Q1CoverageError`` stops the run.
* Q2: ONNX Runtime static INT8 quantization (QDQ; asymmetric uint8 activations, symmetric per-channel int8
  weights, MinMax calibration). The calibration windows are drawn from the model's own TRAINING selection only
  (CLAUDE.md rule 4). Operators outside ``q2.op_types`` (e.g. LSTM, the score tail of the autoencoders) stay
  float (selective quantization); they are listed in quant_meta.json together with the heavy layers that the CPU
  provider runs with an INT8 kernel versus as DequantizeLinear -> float op -> QuantizeLinear (simulated).
  The input quantizer is checked to be asymmetric with every ID code and payload byte on its own level (P2).
* Classical models (RF, XGBoost) are not quantized; they enter the size and latency benchmark only.

Evaluation (K-036, K-045): every version scores the clean validation normals, the validation and the test
windows on the CPU (Q0/Q2 with ONNX Runtime). (a) primary: the threshold is recalibrated per version with the
K-035/K-038 rule; (b) the Q0 threshold is applied unchanged (calibration drift). Deltas are Q0 minus Qk.

Outputs: results/raw/<run_id>/quant/ (ONNX files, quant_meta.json, metrics.csv, scores_<q>_<split>.parquet),
data/processed/bench/<dataset>.npy (benchmark inputs), results/tables/quant_pilot.csv and
results/tables/benchmark_manifest.json.

Usage:
    python -m src.quantize --pilot [--threads 4]   # quantize + evaluate the pilot runs, manifest, pilot table
    python -m src.quantize --run <run_id>          # one trained deep run
    python -m src.quantize --pilot-table           # rebuild quant_pilot.csv (merges benchmark results if any)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.data import sources as S

QCONFIG = S.PROJECT_ROOT / "configs" / "quantization.yaml"
TABLES = S.PROJECT_ROOT / "results" / "tables"
PILOT_TABLE = TABLES / "quant_pilot.csv"
MANIFEST = TABLES / "benchmark_manifest.json"
BENCH_DIR = S.PROJECT_ROOT / "data" / "processed" / "bench"
QUANT_DIR = "quant"
LEVELS = ("q0", "q1", "q2")
FILES = {"q0": "q0_fp32.onnx", "q1": "q1_fp16.onnx", "q2": "q2_int8.onnx"}
PRECISION = {"q0": "float32", "q1": "float16", "q2": "int8"}
THRESHOLD_MODES = ("recalibrated", "q0_threshold")      # K-036 (a) primary, (b) calibration drift
INPUT, OUTPUT = "x", "score"
HEAVY = {"Conv", "ConvTranspose", "Gemm", "MatMul", "LSTM", "GRU"}
# ONNX Runtime CPU kernel -> layer type it implements (fused and quantized kernels)
KERNEL_LAYER = {"FusedConv": "Conv", "NhwcFusedConv": "Conv", "QLinearConv": "Conv", "NhwcQLinearConv": "Conv",
                "ConvInteger": "Conv", "FusedGemm": "Gemm", "QGemm": "Gemm", "FusedMatMul": "MatMul",
                "QLinearMatMul": "MatMul", "MatMulInteger": "MatMul", "DynamicQuantizeLSTM": "LSTM",
                "QLinearConvTranspose": "ConvTranspose"}
DELTA_METRICS = ("recall", "f1", "roc_auc", "average_precision", "roc_tpr_at_fpr_0.01")
PILOT_ROW_LEVELS = ("attack_type", "attack_free", "pooled")


class Q1CoverageError(RuntimeError):
    """Too few heavy layers run in float16 on the CPU provider and no user decision is configured (K-043)."""


class InputQuantizationError(RuntimeError):
    """The INT8 input quantizer is symmetric or merges distinct ID codes / payload bytes (P2 open issue)."""


def load_qconfig(path: Path = QCONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_info(path: Path) -> dict:
    return {"file": path.name, "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


class ScoreModel(nn.Module):
    """Window -> anomaly score, the quantity ``src.train.score_deep`` thresholds (logit or reconstruction MSE)."""

    def __init__(self, net: nn.Module, autoencoder: bool):
        super().__init__()
        self.net = net
        self.autoencoder = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.autoencoder:
            return ((self.net(x) - x) ** 2).mean(dim=(1, 2))
        return self.net(x)


def load_trained(run_dir: Path) -> tuple[dict, ScoreModel]:
    from src import train as T

    rc = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if rc["model"] not in T.DEEP:
        raise ValueError(f"{rc['run_id']}: {rc['model']} is not a deep model; classical models are not quantized")
    net = T.build_deep(rc["model"], rc["hparams"], T.run_n_features(rc))
    net.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    return rc, ScoreModel(net, rc["model"] in T.AUTOENCODERS).eval()


def torch_scores(module: nn.Module, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    module.eval()
    with torch.no_grad():
        return np.concatenate([module(torch.from_numpy(x[i:i + batch])).numpy() for i in range(0, len(x), batch)])


# --------------------------------------------------------------------------------------------
# ONNX export and ONNX Runtime helpers
# --------------------------------------------------------------------------------------------


def _inline_constants(model):
    """Constant nodes -> initializers; Identity of an initializer -> a copy of it.

    The TorchScript exporter shares equal tensors through Identity nodes (e.g. two folded BatchNorm biases); the
    INT8 quantizer only treats initializers as biases, so such a Conv would otherwise stay float."""
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    keep = []
    for n in model.graph.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            t = numpy_helper.from_array(numpy_helper.to_array(n.attribute[0].t), n.output[0])
        elif n.op_type == "Identity" and n.input[0] in inits:
            t = numpy_helper.from_array(numpy_helper.to_array(inits[n.input[0]]), n.output[0])
        else:
            keep.append(n)
            continue
        model.graph.initializer.append(t)
        inits[t.name] = t
    del model.graph.node[:]
    model.graph.node.extend(keep)
    return model


def export_onnx(module: nn.Module, path: Path, window: int, n_features: int, opset: int) -> None:
    import onnx

    module.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # legacy (TorchScript) exporter deprecation notice
        torch.onnx.export(module, (torch.zeros(1, window, n_features),), str(path), input_names=[INPUT],
                          output_names=[OUTPUT], dynamic_axes={INPUT: {0: "batch"}, OUTPUT: {0: "batch"}},
                          opset_version=opset, do_constant_folding=True, dynamo=False)
    m = _inline_constants(onnx.load(str(path)))
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def ort_session(path: Path, threads: int, profile_dir: str | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(profile_dir) / "ort")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_scores(sess, x: np.ndarray, batch: int) -> np.ndarray:
    return np.concatenate([sess.run(None, {INPUT: x[i:i + batch]})[0] for i in range(0, len(x), batch)])


def executed_kernels(path: Path, sample: np.ndarray, threads: int) -> list[dict]:
    """Kernels the CPU execution provider runs for one inference, with the element type of their first input
    (read from the ONNX Runtime profiler, i.e. after graph optimisation and inserted precision casts)."""
    with tempfile.TemporaryDirectory() as tmp:
        sess = ort_session(path, threads, profile_dir=tmp)
        sess.run(None, {INPUT: sample[:1]})
        prof = Path(sess.end_profiling())
        events = json.loads(prof.read_text(encoding="utf-8"))
        del sess
    out = {}
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            a = e["args"]
            types = [next(iter(d)) for d in a.get("input_type_shape", [])]
            out[e["name"][: -len("_kernel_time")]] = {"node": e["name"][: -len("_kernel_time")], "op": a["op_name"],
                                                       "input_type": types[0] if types else None,
                                                       "provider": a.get("provider")}
    return list(out.values())


def kernel_layer(op: str) -> str:
    return KERNEL_LAYER.get(op, op)


# --------------------------------------------------------------------------------------------
# Q1: float16
# --------------------------------------------------------------------------------------------


def to_fp16(q0: Path, q1: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = convert_float_to_float16(onnx.load(str(q0)), keep_io_types=True)
    onnx.save(m, str(q1))


def fp16_coverage(path: Path, sample: np.ndarray, threads: int) -> dict:
    """Share of kernels (and of heavy layers) that the CPU provider runs in float16 (Casts not counted)."""
    ks = executed_kernels(path, sample, threads)
    casts = sum(k["op"] == "Cast" for k in ks)
    ks = [k for k in ks if k["op"] != "Cast"]
    heavy = [k for k in ks if kernel_layer(k["op"]) in HEAVY]
    by_op = {}
    for k in ks:
        d = by_op.setdefault(k["op"], {"float16": 0, "other": 0})
        d["float16" if k["input_type"] == "float16" else "other"] += 1
    n16 = sum(k["input_type"] == "float16" for k in ks)
    h16 = sum(k["input_type"] == "float16" for k in heavy)
    return {"provider": "CPUExecutionProvider", "n_kernels": len(ks), "n_kernels_fp16": n16,
            "share_fp16": n16 / len(ks) if ks else float("nan"), "n_heavy": len(heavy), "n_heavy_fp16": h16,
            "heavy_share_fp16": h16 / len(heavy) if heavy else float("nan"), "n_casts": casts, "by_op": by_op,
            "kernels": ks}


def q1_status(cov: dict, qcfg: dict) -> str:
    """``cpu_fp16`` or ``cpu_unsupported`` (decision K-043); raises Q1CoverageError without a configured action."""
    bar = qcfg["q1"]["min_heavy_fp16_share"]
    if cov["n_heavy"] and cov["heavy_share_fp16"] >= bar:
        return "cpu_fp16"
    action = qcfg["q1"].get("low_cpu_coverage_action")
    if action != "simulate_fp16":
        raise Q1CoverageError(
            f"only {cov['n_heavy_fp16']}/{cov['n_heavy']} heavy layers run in float16 on the CPU provider "
            f"(bar {bar}); Q1 must not be reported as is. Decide: mark it cpu_unsupported and measure accuracy by "
            f"float16 simulation, or drop Q1 (configs/quantization.yaml q1.low_cpu_coverage_action).")
    return "cpu_unsupported"


def fp16_device(qcfg: dict) -> str:
    dev = qcfg["q1"].get("simulation_device", "cuda")
    return dev if dev != "cuda" or torch.cuda.is_available() else "cpu"


def fp16_scorer(module: nn.Module, device: str):
    """Real float16 arithmetic in PyTorch: weights, input and activations in half precision (K-043)."""
    m16 = copy.deepcopy(module).to(device).half().eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def run(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return m16(torch.from_numpy(x).to(device).half()).float().cpu().numpy()
    return run


# --------------------------------------------------------------------------------------------
# Q2: INT8
# --------------------------------------------------------------------------------------------


def check_calibration_selection(sel: pd.DataFrame, n_min: int) -> pd.DataFrame:
    """The INT8 calibration windows must be TRAINING windows (CLAUDE.md rule 4) and at least ``n_min``."""
    from src.data import dataset as D

    if len(sel) < n_min:
        raise ValueError(f"INT8 calibration needs at least {n_min} training windows, got {len(sel)}")
    return D._require_split(sel, "train")


def calibration_selection(model: str, dataset: str, win: pd.DataFrame, segs: dict, n: int, n_min: int,
                          seed: int) -> pd.DataFrame:
    """Random sample of the model's OWN training selection (supervised: train windows minus masquerade normals,
    K-028; autoencoders: the normal training subset, K-022). Validation and test windows are never used for the
    INT8 calibration (CLAUDE.md rule 4, K-042); ``check_calibration_selection`` enforces it."""
    from src import train as T
    from src.data import dataset as D

    if n < n_min:
        raise ValueError(f"calibration size {n} below the minimum {n_min}")
    pool = D.ae_train(win, dataset) if model in T.AUTOENCODERS else D.supervised_train(win, dataset, segs)
    pool = check_calibration_selection(pool, n_min)
    idx = np.sort(np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False))
    return check_calibration_selection(pool.iloc[idx].reset_index(drop=True), n_min)


def to_int8(q0: Path, q2: Path, calib: np.ndarray, qcfg: dict) -> None:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    c = qcfg["q2"]
    if len(calib) < c["min_calibration_windows"]:
        raise ValueError(f"INT8 calibration needs at least {c['min_calibration_windows']} windows")
    bs = c["calibration_batch"]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = iter([{INPUT: calib[i:i + bs]} for i in range(0, len(calib), bs)])

        def get_next(self):
            return next(self.batches, None)

    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(q0), str(pre), skip_symbolic_shape=True)
        quantize_static(str(pre), str(q2), Reader(), quant_format=getattr(QuantFormat, c["format"]),
                        activation_type=getattr(QuantType, c["activation_type"]),
                        weight_type=getattr(QuantType, c["weight_type"]), per_channel=c["per_channel"],
                        calibrate_method=getattr(CalibrationMethod, c["calibrate_method"]),
                        op_types_to_quantize=list(c["op_types"]),
                        extra_options={"ActivationSymmetric": c["activation_symmetric"],
                                       "WeightSymmetric": c["weight_symmetric"]})


def graph_nodes_by_quantization(q2: Path, op_types) -> tuple[list[dict], list[dict]]:
    """(quantized, float) compute nodes of a QDQ graph. A node is quantized when its type is in ``op_types`` and
    every non-initializer input comes from a DequantizeLinear; everything else runs in float (selective
    quantization)."""
    import onnx

    m = onnx.load(str(q2))
    inits = {i.name for i in m.graph.initializer}
    dq = {o for n in m.graph.node if n.op_type == "DequantizeLinear" for o in n.output}
    quant, flt = [], []
    for n in m.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        acts = [i for i in n.input if i and i not in inits]
        q = n.op_type in op_types and bool(acts) and all(i in dq for i in acts)
        (quant if q else flt).append({"node": n.name, "op": n.op_type})
    return quant, flt


def int8_execution(q2: Path, sample: np.ndarray, threads: int, op_types) -> dict:
    quant, flt = graph_nodes_by_quantization(q2, set(op_types))
    ks = [k for k in executed_kernels(q2, sample, threads) if k["op"] not in ("QuantizeLinear", "DequantizeLinear")]
    heavy = [{**k, "layer": kernel_layer(k["op"]), "int8_kernel": k["input_type"] in ("uint8", "int8")}
             for k in ks if kernel_layer(k["op"]) in HEAVY]
    hq = sum(n["op"] in HEAVY for n in quant)
    hf = sum(n["op"] in HEAVY for n in flt)
    n8 = sum(k["int8_kernel"] for k in heavy)
    return {"n_heavy": hq + hf, "n_heavy_quantized_in_graph": hq, "n_heavy_float_in_graph": hf,
            "n_heavy_int8_kernel": n8,
            # quantized in the graph, but the CPU provider has no INT8 kernel: DQ -> float op -> Q. Numerically the
            # int8 weights/activations are kept; only the arithmetic (and so the latency) is float.
            "n_heavy_qdq_simulated": hq - n8,
            "selective": bool(hf) or bool(flt), "float_nodes": flt, "quantized_nodes": quant,
            "heavy_kernels": heavy, "kernels": ks}


def input_quantization(q2: Path) -> dict:
    """Scale / zero point of the QuantizeLinear applied to the model input (directly or after layout ops)."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(q2))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    consumers: dict[str, list] = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    frontier, found = [INPUT], None
    while frontier and found is None:
        t = frontier.pop(0)
        for n in consumers.get(t, []):
            if n.op_type == "QuantizeLinear":
                found = n
                break
            if n.op_type in ("Transpose", "Reshape", "Squeeze", "Unsqueeze", "Identity"):
                frontier.append(n.output[0])
    if found is None:
        raise InputQuantizationError("no QuantizeLinear on the model input: the input is not quantized")
    scale = float(inits[found.input[1]])
    zp = inits[found.input[2]]
    info = np.iinfo(zp.dtype)
    z = int(zp)
    symmetric = (zp.dtype == np.uint8 and z == 128) or (zp.dtype == np.int8 and z == 0)
    q_top = min(info.max, z + int(np.floor(1.0 / scale * (1 + 1e-5))))    # scale is float32: 1/255 rounds up
    return {"tensor": found.input[0], "dtype": str(zp.dtype), "scale": scale, "zero_point": z,
            "representable_range": [(info.min - z) * scale, (info.max - z) * scale],
            "scheme": "symmetric" if symmetric else "asymmetric",
            "levels_on_unit_interval": max(0, q_top - max(info.min, z) + 1)}


def quantize_values(x: np.ndarray, iq: dict) -> np.ndarray:
    info = np.iinfo(np.dtype(iq["dtype"]))
    return np.clip(np.rint(np.asarray(x, np.float64) / iq["scale"]) + iq["zero_point"], info.min, info.max)


def input_level_report(iq: dict, calib: np.ndarray, n_ids: int | None, names: list[str] | None = None) -> dict:
    """Distinct values vs distinct INT8 levels per input column (calibration windows), plus the full value grids
    of the payload bytes (k/255), of the ID codes (k/K) and of the extended grid columns (k/n; P10, K-073)."""
    from src.data import features_ext as FX
    from src.data.features import FEATURE_NAMES

    names = list(names or FEATURE_NAMES)
    if len(names) != calib.shape[-1]:
        raise ValueError(f"{len(names)} feature names for {calib.shape[-1]} input columns")
    ecfg = FX.load_ext_config() if len(names) > len(FEATURE_NAMES) else None
    flat = calib.reshape(-1, calib.shape[-1])
    cols = []
    for j, name in enumerate(names):
        v = np.unique(flat[:, j])
        lv = np.unique(quantize_values(v, iq))
        cols.append({"feature": name, "kind": FX.column_kind(name, ecfg) if ecfg else
                     ("continuous" if name == "iat" else "grid"),
                     "min": float(v.min()), "max": float(v.max()), "n_values": int(len(v)),
                     "n_levels": int(len(lv)), "lossless": bool(len(v) == len(lv))})
    grids = {"bytes": np.arange(256, dtype=np.float32) / np.float32(255)}
    if n_ids:
        grids["id_codes"] = np.arange(n_ids + 1, dtype=np.float32) / np.float32(n_ids)
    if ecfg:
        for key, k in ecfg["column_kind"].items():
            if k["kind"] == "grid" and any(n == key or n.startswith(f"{key}_") for n in names):
                grids[key] = np.arange(k["n"] + 1, dtype=np.float32) / np.float32(k["n"])
    grid = {k: {"n_values": int(len(g)), "n_levels": int(len(np.unique(quantize_values(g, iq))))}
            for k, g in grids.items()}
    return {"columns": cols, "grids": grid,
            "calibration_input_range": [float(flat.min()), float(flat.max())]}


def verify_input_quantization(iq: dict, report: dict) -> None:
    if iq["scheme"] != "asymmetric":
        raise InputQuantizationError(f"input quantizer is {iq['scheme']} ({iq}); payload bytes would fall to 128 levels")
    bad = [c["feature"] for c in report["columns"] if c.get("kind", "continuous" if c["feature"] == "iat" else "grid")
           == "grid" and not c["lossless"]]
    bad += [k for k, g in report["grids"].items() if g["n_levels"] != g["n_values"]]
    if bad:
        raise InputQuantizationError(f"distinct input values merged by the INT8 input quantizer: {bad}")


# --------------------------------------------------------------------------------------------
# One run: Q0/Q1/Q2
# --------------------------------------------------------------------------------------------


def quantize_run(run_dir: Path, qcfg: dict, threads: int, log=print) -> dict:
    from src import train as T
    from src.data import dataset as D
    from src.data import features as F

    torch.set_num_threads(threads)
    rc, module = load_trained(run_dir)
    qdir = run_dir / QUANT_DIR
    qdir.mkdir(exist_ok=True)
    cfg = S.load_config()
    size, n_feat = rc["window_size"], T.run_n_features(rc)
    fset = rc.get("feature_set", "s0")
    paths = {q: qdir / f for q, f in FILES.items()}

    # calibration windows: training split only (rule 4); they also serve as the Q0 equivalence sample
    store = D.FeatureStore.build(rc["dataset"], ("train",), cfg, log=log, feature_set=fset)
    if store.params_hash != rc["data_hash"]:
        raise RuntimeError(f"{rc['run_id']}: features changed since training ({store.params_hash} != {rc['data_hash']})")
    win = D.load_window_index()
    c2 = qcfg["q2"]
    sel = calibration_selection(rc["model"], rc["dataset"], win, store.segments, c2["calibration_windows"],
                                c2["min_calibration_windows"], rc["seed"])
    calib = store.windows(store.global_starts(sel), size).astype(np.float32)
    del store

    # Q0
    export_onnx(module, paths["q0"], size, n_feat, qcfg["onnx"]["opset"])
    ref = torch_scores(module, calib)
    got = ort_scores(ort_session(paths["q0"], threads), calib, 512)
    diff = np.abs(got - ref)
    ok = bool(np.allclose(got, ref, atol=qcfg["onnx"]["q0_atol"], rtol=qcfg["onnx"]["q0_rtol"]))
    q0 = {**file_info(paths["q0"]), "precision": "float32", "backend": "onnxruntime-cpu",
          "equivalence_vs_pytorch_cpu": {"n_windows": int(len(calib)), "max_abs_diff": float(diff.max()),
                                         "max_rel_diff": float((diff / np.maximum(np.abs(ref), 1e-12)).max()),
                                         "atol": qcfg["onnx"]["q0_atol"], "rtol": qcfg["onnx"]["q0_rtol"],
                                         "allclose": ok}}
    if not ok:
        raise RuntimeError(f"{rc['run_id']}: Q0 ONNX differs from PyTorch beyond tolerance: {q0}")
    log(f"{rc['run_id']}: Q0 max |ONNX - PyTorch| = {diff.max():.3g}")

    # Q1
    to_fp16(paths["q0"], paths["q1"])
    cov = fp16_coverage(paths["q1"], calib, threads)
    log(f"{rc['run_id']}: Q1 heavy layers in fp16 on CPU EP: {cov['n_heavy_fp16']}/{cov['n_heavy']}, "
        f"all kernels {cov['n_kernels_fp16']}/{cov['n_kernels']}")
    status = q1_status(cov, qcfg)           # raises Q1CoverageError without a user decision (K-043)
    q1 = {**file_info(paths["q1"]), "precision": "float16", "cpu_status": status, "fp16_coverage": cov,
          "accuracy_backend": "onnxruntime-cpu" if status == "cpu_fp16" else f"pytorch-fp16-{fp16_device(qcfg)}",
          "latency_benchmark": "onnxruntime-cpu" if status == "cpu_fp16" else "not measured: cpu_unsupported (K-043)"}

    # Q2
    to_int8(paths["q0"], paths["q2"], calib, qcfg)
    exe = int8_execution(paths["q2"], calib, threads, c2["op_types"])
    iq = input_quantization(paths["q2"])
    n_ids = F.load_params()["datasets"][rc["dataset"]]["n_ids"]
    levels = input_level_report(iq, calib, n_ids, rc.get("feature_names"))
    verify_input_quantization(iq, levels)
    q2 = {**file_info(paths["q2"]), "precision": "int8", "backend": "onnxruntime-cpu",
          "scheme": {k: c2[k] for k in ("format", "activation_type", "activation_symmetric", "weight_type",
                                        "weight_symmetric", "per_channel", "calibrate_method", "op_types")},
          "calibration": {"split": "train", "selector": "ae_train" if rc["model"] in T.AUTOENCODERS
                          else "supervised_train", "n_windows": int(len(sel)), "seed": rc["seed"],
                          "digest": D.selection_digest(sel), **D.describe(sel)},
          "execution": exe, "input_quantization": iq, "input_levels": levels}
    log(f"{rc['run_id']}: Q2 heavy layers: {exe['n_heavy_int8_kernel']} INT8 kernel, "
        f"{exe['n_heavy_qdq_simulated']} QDQ-simulated, {exe['n_heavy_float_in_graph']} float; "
        f"input {iq['scheme']} scale {iq['scale']:.6g} zp {iq['zero_point']} ({iq['dtype']})")

    for q, d in (("q1", q1), ("q2", q2)):
        d["size_ratio_vs_q0"] = d["size_bytes"] / q0["size_bytes"]
    meta = {"run_id": rc["run_id"], "model": rc["model"], "dataset": rc["dataset"], "seed": rc["seed"],
            "source": file_info(run_dir / "model.pt"), "created_utc": _now(), "git": T.git_info(),
            "environment": T.environment("cpu", threads), "config": qcfg, "opset": qcfg["onnx"]["opset"],
            "levels": {"q0": q0, "q1": q1, "q2": q2}}
    (qdir / "quant_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return meta


# --------------------------------------------------------------------------------------------
# Evaluation of the versions (K-036)
# --------------------------------------------------------------------------------------------


def write_bench_inputs(dataset: str, store, val: pd.DataFrame, size: int, qcfg: dict, force: bool = False) -> Path:
    """Fixed sample of validation windows (labels unused) as the benchmark input, so the benchmark needs neither
    the data pipeline nor PyTorch (docker container)."""
    from src.data import dataset as D

    b = qcfg["benchmark"]
    fset = getattr(store, "feature_set", "s0")
    path = BENCH_DIR / (f"{dataset}.npy" if fset == "s0" else f"{dataset}_{fset}.npy")
    if path.exists() and not force:
        return path
    if b["input_split"] != "val":
        raise ValueError("benchmark inputs are validation windows (K-046)")
    D._require_split(val, "val")
    idx = np.sort(np.random.default_rng(b["input_seed"]).choice(len(val), b["n_inputs"], replace=False))
    sel = val.iloc[idx]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, store.windows(store.global_starts(sel), size).astype(np.float32))
    path.with_suffix(".json").write_text(json.dumps(
        {"dataset": dataset, "split": "val", "n": int(len(sel)), "seed": b["input_seed"],
         "digest": D.selection_digest(sel), "data_hash": store.params_hash, "created_utc": _now()}, indent=2) + "\n",
        encoding="utf-8")
    return path


def _score_split(fn, store, w: pd.DataFrame, size: int, batch: int) -> np.ndarray:
    g = store.global_starts(w)
    out = [fn(store.windows(g[i:i + batch], size).astype(np.float32)) for i in range(0, len(g), batch)]
    return np.concatenate(out).astype(np.float64) if out else np.empty(0)


def evaluate_run(run_dir: Path, qcfg: dict, threads: int, log=print) -> pd.DataFrame:
    from src import calibration as C
    from src import evaluate as E
    from src.data import dataset as D
    from src.data import splits as SP

    rc, module = load_trained(run_dir)
    qdir = run_dir / QUANT_DIR
    meta = json.loads((qdir / "quant_meta.json").read_text(encoding="utf-8"))
    p3 = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    cfg = S.load_config()
    groups = yaml.safe_load(SP.GROUPS_PATH.read_text(encoding="utf-8"))
    size, ds, batch = rc["window_size"], rc["dataset"], qcfg["evaluation"]["batch_size"]
    target_fpr = rc["decision_config"]["target_fpr"]

    store = D.FeatureStore.build(ds, ("val", "test"), cfg, log=log, feature_set=rc.get("feature_set", "s0"))
    if store.params_hash != rc["data_hash"]:
        raise RuntimeError(f"{rc['run_id']}: features changed since training ({store.params_hash} != {rc['data_hash']})")
    win = D.load_window_index()
    wins = {s: D.eval_windows(win, ds, s) for s in ("val", "test")}
    cal = D.calibration_windows(win, ds, store.segments, cfg)       # clean validation normals (K-038)
    key = wins["val"][["segment", "start"]].astype({"segment": str}).reset_index().merge(
        cal[["segment", "start"]].astype({"segment": str}), on=["segment", "start"], how="inner")
    if len(key) != len(cal):
        raise RuntimeError("calibration windows are not a subset of the validation windows")
    cal_pos = key["index"].to_numpy()
    write_bench_inputs(ds, store, wins["val"], size, qcfg)

    scorers = {"q0": None, "q1": None, "q2": None}
    for q in LEVELS:
        if q == "q1" and meta["levels"]["q1"]["cpu_status"] != "cpu_fp16":
            scorers[q] = fp16_scorer(module, fp16_device(qcfg))
        else:
            sess = ort_session(qdir / FILES[q], threads)
            scorers[q] = (lambda s: (lambda x: s.run(None, {INPUT: x})[0]))(sess)
    scores, decisions = {}, {}
    for q in LEVELS:
        scores[q] = {s: _score_split(scorers[q], store, wins[s], size, batch) for s in wins}
        for s, w in wins.items():
            pd.DataFrame({"segment": w["segment"].astype(str), "start": w["start"], "label": w["label"],
                          "attack_type": w["attack_type"].astype(str), "score": scores[q][s]}).to_parquet(
                qdir / f"scores_{q}_{s}.parquet", index=False)
        # K-036 (a): every version recalibrated with the K-035/K-038 rule on its own validation-normal scores
        decisions[q] = {**C.calibrate_threshold(scores[q]["val"][cal_pos], cal["label"].to_numpy(), "val", target_fpr),
                        "calibration_windows_digest": D.selection_digest(cal)}
        log(f"{rc['run_id']}: {q} threshold {decisions[q]['threshold']:.6g}")

    rows = []
    for q in LEVELS:
        backend = meta["levels"][q].get("accuracy_backend", meta["levels"][q].get("backend"))
        for mode in THRESHOLD_MODES:
            dec = decisions[q] if mode == "recalibrated" else decisions["q0"]   # (b): Q0 threshold unchanged
            for s, w in wins.items():
                m = E.evaluate_windows(w, scores[q][s], C.predict(scores[q][s], dec), ds, groups, cfg)
                m.insert(0, "split", s)
                m.insert(0, "threshold", dec["threshold"])
                m.insert(0, "threshold_mode", mode)
                m.insert(0, "backend", backend)
                m.insert(0, "precision_name", PRECISION[q])
                m.insert(0, "quant", q)
                rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    for k in ("seed", "dataset", "model", "run_id"):
        out.insert(0, k, rc[k])
    out.to_csv(qdir / "metrics.csv", index=False)

    meta["thresholds"] = decisions
    meta["q0_consistency"] = q0_consistency(run_dir, p3, decisions["q0"], out, scores["q0"]["test"], wins["test"])
    meta["evaluated_utc"] = _now()
    (qdir / "quant_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    c = meta["q0_consistency"]
    log(f"{rc['run_id']}: Q0-ONNX vs P3 PyTorch: threshold rel diff {c['threshold_rel_diff']:.2e}, "
        f"max |recall diff| {c.get('max_abs_recall_diff_attack_types', float('nan')):.4f}, "
        f"test decision flips {c.get('test_decision_flips')}")
    return out


def q0_consistency(run_dir: Path, p3_meta: dict, dec0: dict, metrics: pd.DataFrame, s0_test: np.ndarray,
                   w_test: pd.DataFrame) -> dict:
    """Q0 ONNX (CPU) vs the P3 PyTorch evaluation (GPU): threshold, per-type test recall, decision flips."""
    t_p3 = p3_meta["decision"]["threshold"]
    mine = metrics[(metrics.quant == "q0") & (metrics.threshold_mode == "recalibrated") & (metrics.split == "test")
                   & (metrics.level == "attack_type")].set_index("name")["recall"]
    out = {"p3_threshold": t_p3, "q0_onnx_threshold": dec0["threshold"],
           "threshold_rel_diff": abs(dec0["threshold"] - t_p3) / max(abs(t_p3), 1e-12)}
    f = run_dir / "metrics_test.csv"
    if f.exists():
        p3 = pd.read_csv(f)
        p3 = p3[p3.level == "attack_type"].set_index("name")["recall"]
        out["max_abs_recall_diff_attack_types"] = float((mine - p3.reindex(mine.index)).abs().max())
    s = run_dir / "scores_test.parquet"
    if s.exists():
        p = pd.read_parquet(s)
        k = w_test[["segment", "start"]].astype({"segment": str}).assign(onnx=s0_test).merge(
            p.astype({"segment": str}), on=["segment", "start"])
        out["test_decision_flips"] = int(((k["onnx"] > dec0["threshold"]) != (k["score"] > t_p3)).sum())
        out["test_max_abs_score_diff"] = float((k["onnx"] - k["score"]).abs().max())
        out["n_test_windows_compared"] = int(len(k))
    return out


# --------------------------------------------------------------------------------------------
# Pilot: run list, benchmark manifest, table
# --------------------------------------------------------------------------------------------


def pilot_runs(qcfg: dict) -> dict[str, list[Path]]:
    """{"deep": [...], "classical": [...]} run directories of the pilot (current-config seed-0 baseline runs; the
    autoencoder selected by K-033/K-039)."""
    from src import evaluate as E
    from src import run_grid as G
    from src import train as T

    p = qcfg["pilot"]
    planned = {(r["model"], r["dataset"], r["seed"]): r["run_id"] for r in G.stage_runs("baseline")}
    selected = E.selected_autoencoders()
    out = {"deep": [], "classical": []}
    for d in p["datasets"]:
        for m in qcfg["models"]:
            if m in T.AUTOENCODERS:
                rid = selected.get(d)
                if not rid or not rid.startswith(f"{m}_"):
                    raise RuntimeError(f"{d}: the selected autoencoder is {rid}, not {m}")
            else:
                rid = planned[(m, d, p["seed"])]
            out["deep"].append(T.RAW_DIR / rid)
        for m in qcfg["not_quantized"]:
            out["classical"].append(T.RAW_DIR / planned[(m, d, p["seed"])])
    for r in out["deep"] + out["classical"]:
        if not (r / "DONE").exists():
            raise RuntimeError(f"pilot run {r.name} is not finished")
    return out


def _rel(p: Path) -> str:
    return p.resolve().relative_to(S.PROJECT_ROOT.resolve()).as_posix()


def write_manifest(runs: dict[str, list[Path]]) -> list[dict]:
    """Benchmark targets (read by src.benchmark, which must not need PyTorch or the data pipeline)."""
    targets = []
    for r in runs["deep"]:
        meta = json.loads((r / QUANT_DIR / "quant_meta.json").read_text(encoding="utf-8"))
        for q in LEVELS:
            lv = meta["levels"][q]
            st = "measure" if q != "q1" or lv["cpu_status"] == "cpu_fp16" else "cpu_unsupported"
            targets.append({"id": f"{r.name}:{q}", "run_id": r.name, "model": meta["model"], "dataset": meta["dataset"],
                            "quant": q, "kind": "onnx", "model_path": _rel(r / QUANT_DIR / FILES[q]),
                            "input_path": _rel(BENCH_DIR / f"{meta['dataset']}.npy"), "out_dir": _rel(r / QUANT_DIR),
                            "status": st})
    for r in runs["classical"]:
        rc = json.loads((r / "run_config.json").read_text(encoding="utf-8"))
        f = r / ("model.joblib" if rc["model"] == "rf" else "model.ubj")
        t = {"id": f"{r.name}:native", "run_id": r.name, "model": rc["model"], "dataset": rc["dataset"],
             "quant": "native", "kind": rc["model"], "model_path": _rel(f),
             "input_path": _rel(BENCH_DIR / f"{rc['dataset']}.npy"), "out_dir": _rel(r), "status": "measure",
             "note": "classical model, not quantized (K-042)"}
        if rc["model"] == "xgb":
            t["best_iteration"] = int(rc["meta"]["best_iteration"])
        targets.append(t)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"created_utc": _now(), "targets": targets}, indent=2) + "\n", encoding="utf-8")
    return targets


def quant_rows(run_dir: Path) -> pd.DataFrame | None:
    """Attack-type, attack-free and pooled rows of one evaluated run's quant/metrics.csv (Q0/Q1/Q2, both threshold
    modes) with the deltas Q0 - Qk and the file size of every level; None if the run is not evaluated."""
    f = run_dir / QUANT_DIR / "metrics.csv"
    if not f.exists():
        return None
    meta = json.loads((run_dir / QUANT_DIR / "quant_meta.json").read_text(encoding="utf-8"))
    m = pd.read_csv(f)
    m = m[m["level"].isin(PILOT_ROW_LEVELS)].copy()
    keys = ["threshold_mode", "split", "level", "name"]
    ref = m[m.quant == "q0"].set_index(keys)[list(DELTA_METRICS)]
    j = m.join(ref, on=keys, rsuffix="_q0")
    for k in DELTA_METRICS:
        m[f"d_{k}"] = j[f"{k}_q0"] - j[k]           # Q0 - Qk: positive = loss (K-037)
    for q in LEVELS:
        lv = meta["levels"][q]
        sel = m.quant == q
        m.loc[sel, "size_mb"] = lv["size_bytes"] / 2**20
        m.loc[sel, "size_ratio_vs_q0"] = lv.get("size_ratio_vs_q0", 1.0)
    return m


def pilot_table(runs: dict[str, list[Path]] | None = None, profile: str = "host", session: int = 1) -> pd.DataFrame:
    """quant_pilot.csv: per attack type (and attack-free / pooled rows) metrics of Q0/Q1/Q2 under both threshold
    modes, deltas Q0 - Qk, file size, and latency/memory of the given benchmark profile and session when
    measured (otherwise empty with ``latency_status`` pending / cpu_unsupported)."""
    from src import benchmark as B

    runs = runs or pilot_runs(load_qconfig())
    frames = []
    for r in runs["deep"]:
        m = quant_rows(r)
        if m is None:
            continue
        meta = json.loads((r / QUANT_DIR / "quant_meta.json").read_text(encoding="utf-8"))
        for q in LEVELS:
            lv = meta["levels"][q]
            sel = m.quant == q
            res = B.load_result(r / QUANT_DIR, profile, session, q)
            unsupported = q == "q1" and lv["cpu_status"] != "cpu_fp16"
            m.loc[sel, "latency_profile"] = profile
            m.loc[sel, "latency_status"] = "cpu_unsupported" if unsupported else "measured" if res else "pending"
            if res and not unsupported:
                for c, v in B.summary_columns(res).items():
                    m.loc[sel, c] = v
            # K-049 rule 7: both sessions side by side, inconsistent targets flagged
            both = [B.load_result(r / QUANT_DIR, profile, s, q) for s in (1, 2)]
            for s, rs in zip((1, 2), both):
                if rs and not unsupported:
                    m.loc[sel, f"latency_median_ms_s{s}"] = rs["latency"]["median_ms"]
                    m.loc[sel, f"latency_p95_ms_s{s}"] = rs["latency"]["p95_ms"]
            if all(both) and not unsupported:
                tol = load_qconfig()["benchmark"]["repeatability"]
                m.loc[sel, "latency_consistent"] = B.repeatability(*both, tol)["consistent"]
        frames.append(m)
    if not frames:
        raise RuntimeError("no evaluated pilot run (results/raw/<run>/quant/metrics.csv)")
    out = pd.concat(frames, ignore_index=True)
    for c in B.SUMMARY_COLUMNS + ["latency_median_ms_s1", "latency_p95_ms_s1", "latency_median_ms_s2",
                                  "latency_p95_ms_s2", "latency_consistent"]:
        if c not in out:
            out[c] = np.nan
    out = out.drop(columns=["tp", "fp", "fn", "tn", "captures", "attack_types"], errors="ignore")
    out = out.sort_values(["dataset", "model", "threshold_mode", "split", "level", "name", "quant"]).reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(PILOT_TABLE, index=False, float_format="%.6g")
    return out


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from src import train as T

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pilot", action="store_true", help="quantize + evaluate the pilot runs, manifest, table")
    ap.add_argument("--run", help="one trained deep run id under results/raw")
    ap.add_argument("--pilot-table", action="store_true", help="rebuild results/tables/quant_pilot.csv")
    ap.add_argument("--profile", default="host", help="benchmark profile merged into the pilot table")
    ap.add_argument("--threads", type=int, help="CPU threads (default: configs/models.yaml runtime.threads)")
    ap.add_argument("--force", action="store_true", help="redo runs that are already quantized/evaluated")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    qcfg = load_qconfig()
    threads = a.threads or T.load_models_config()["runtime"]["threads"]
    T.set_threads(threads)
    if (a.pilot or a.run) and T.git_info()["dirty"] and not a.allow_dirty:
        raise SystemExit("working tree has uncommitted changes (CLAUDE.md rule 7); commit or pass --allow-dirty")
    runs = pilot_runs(qcfg) if a.pilot or a.pilot_table else None
    todo = runs["deep"] if a.pilot else [T.RAW_DIR / a.run] if a.run else []
    for r in todo:
        meta_f = r / QUANT_DIR / "quant_meta.json"
        if a.force or not meta_f.exists():
            quantize_run(r, qcfg, threads)
        if not a.skip_eval and (a.force or not (r / QUANT_DIR / "metrics.csv").exists()):
            evaluate_run(r, qcfg, threads)
    if a.pilot:
        write_manifest(runs)
    if a.pilot or a.pilot_table:
        t = pilot_table(runs, a.profile)
        print(f"{PILOT_TABLE}: {len(t)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
