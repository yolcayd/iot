"""Tests for quantised model outputs, size reduction and ONNX/PyTorch equivalence.

Synthetic models (random weights) and synthetic [0, 1] windows whose ID and payload columns cover their full
value grids (codes k/K, bytes k/255); no dataset needed. The last test checks the pilot outputs if they exist.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import quantize as Q
from src.models.cnn1d import CNN1D
from src.models.conv_autoencoder import ConvAutoencoder
from src.models.lstm_autoencoder import LSTMAutoencoder

WINDOW, N_FEAT, K_IDS = 100, 10, 38
QCFG = Q.load_qconfig()


def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty((n, WINDOW, N_FEAT), dtype=np.float32)
    x[:, :, 0] = rng.integers(0, K_IDS + 1, (n, WINDOW)) / np.float32(K_IDS)
    x[:, :, 1:9] = rng.integers(0, 256, (n, WINDOW, 8)) / np.float32(255)
    x[:, :, 9] = rng.random((n, WINDOW))
    x[0, : K_IDS + 1, 0] = np.arange(K_IDS + 1) / np.float32(K_IDS)     # every ID code and every byte value
    x[1, :, 1] = np.arange(100) / np.float32(255)
    x[2, :, 1] = np.arange(100, 200) / np.float32(255)
    x[3, :56, 1] = np.arange(200, 256) / np.float32(255)
    x[4, :, 9] = 0.0
    x[5, :, 9] = 1.0
    return x


def _model(name, seed=0):
    torch.manual_seed(seed)
    if name == "cnn1d":
        net = CNN1D()
        for m in net.modules():          # trained-looking BatchNorm statistics
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
        return Q.ScoreModel(net.eval(), autoencoder=False).eval()
    net = ConvAutoencoder() if name == "conv_ae" else LSTMAutoencoder(hidden=16, latent=8)
    return Q.ScoreModel(net.eval(), autoencoder=True).eval()


@pytest.fixture(scope="module")
def calib():
    return _windows(600)


@pytest.fixture(scope="module")
def built(tmp_path_factory, calib):
    """Q0/Q1/Q2 of the three model families."""
    out = {}
    for name in ("cnn1d", "conv_ae", "lstm_ae"):
        d = tmp_path_factory.mktemp(name)
        m = _model(name)
        p = {q: d / f for q, f in Q.FILES.items()}
        Q.export_onnx(m, p["q0"], WINDOW, N_FEAT, QCFG["onnx"]["opset"])
        Q.to_fp16(p["q0"], p["q1"])
        Q.to_int8(p["q0"], p["q2"], calib, QCFG)
        out[name] = (m, p)
    return out


@pytest.mark.parametrize("name", ["cnn1d", "conv_ae", "lstm_ae"])
def test_q0_onnx_matches_pytorch(built, name):
    m, p = built[name]
    x = _windows(64, seed=1)
    got = Q.ort_scores(Q.ort_session(p["q0"], 2), x, 32)
    ref = Q.torch_scores(m, x)
    np.testing.assert_allclose(got, ref, atol=QCFG["onnx"]["q0_atol"], rtol=QCFG["onnx"]["q0_rtol"])


@pytest.mark.parametrize("name,q", [("cnn1d", "q1"), ("conv_ae", "q1"), ("cnn1d", "q2"), ("conv_ae", "q2"),
                                    ("lstm_ae", "q2")])
def test_quantized_models_produce_outputs(built, name, q):
    # (lstm_ae, q1) is left out: ONNX Runtime 1.30 crashes (access violation) while optimising the float16 LSTM
    # graph at session creation (fine with optimisations off). The LSTM autoencoder is not quantized in this
    # project (K-039); see docs/phase-logs/P4.md.
    m, p = built[name]
    x = _windows(16, seed=2)
    y = Q.ort_scores(Q.ort_session(p[q], 2), x, 8)
    assert y.shape == (16,) and y.dtype == np.float32 and np.isfinite(y).all()
    assert not np.array_equal(y, Q.torch_scores(m, x))     # it is a different (quantized) model


@pytest.mark.parametrize("name", ["cnn1d", "conv_ae"])
def test_file_sizes_shrink(built, name):
    _, p = built[name]
    s = {q: p[q].stat().st_size for q in Q.LEVELS}
    assert s["q1"] / s["q0"] <= 0.6          # float16 ~ half
    assert s["q2"] / s["q0"] <= 0.4          # int8 ~ quarter (+ scales / zero points / int32 biases)


def test_identity_shared_biases_are_inlined_so_every_conv_is_quantized(tmp_path, calib):
    m = Q.ScoreModel(CNN1D().eval(), autoencoder=False).eval()   # default BatchNorm -> equal (zero) folded biases
    q0, q2 = tmp_path / "q0.onnx", tmp_path / "q2.onnx"
    Q.export_onnx(m, q0, WINDOW, N_FEAT, 17)
    import onnx

    assert not [n for n in onnx.load(str(q0)).graph.node if n.op_type in ("Identity", "Constant")]
    Q.to_int8(q0, q2, calib, QCFG)
    ex = Q.int8_execution(q2, calib, 2, QCFG["q2"]["op_types"])
    assert ex["n_heavy"] == 5 and ex["n_heavy_quantized_in_graph"] == 5


def test_int8_input_quantizer_is_asymmetric_with_256_levels(built, calib):
    _, p = built["cnn1d"]
    iq = Q.input_quantization(p["q2"])
    assert iq["scheme"] == "asymmetric" and iq["dtype"] == "uint8" and iq["zero_point"] == 0
    assert iq["scale"] == pytest.approx(1 / 255, rel=1e-6)
    assert iq["levels_on_unit_interval"] == 256
    rep = Q.input_level_report(iq, calib, K_IDS)
    assert rep["grids"]["bytes"] == {"n_values": 256, "n_levels": 256}
    assert rep["grids"]["id_codes"] == {"n_values": K_IDS + 1, "n_levels": K_IDS + 1}
    assert rep["calibration_input_range"] == [0.0, 1.0]
    cols = {c["feature"]: c for c in rep["columns"]}
    assert cols["byte_0"]["n_values"] == 256 and cols["byte_0"]["lossless"]
    Q.verify_input_quantization(iq, rep)


def test_symmetric_input_quantizer_is_rejected(tmp_path, built, calib):
    _, p = built["cnn1d"]
    cfg = {**QCFG, "q2": {**QCFG["q2"], "activation_symmetric": True}}
    q2 = tmp_path / "sym.onnx"
    Q.to_int8(p["q0"], q2, calib, cfg)
    iq = Q.input_quantization(q2)
    assert iq["scheme"] == "symmetric"
    with pytest.raises(Q.InputQuantizationError):
        Q.verify_input_quantization(iq, Q.input_level_report(iq, calib, K_IDS))


def test_lstm_layers_stay_float_and_are_listed(built, calib):
    _, p = built["lstm_ae"]
    ex = Q.int8_execution(p["q2"], calib, 2, QCFG["q2"]["op_types"])
    assert ex["selective"]
    assert [n["op"] for n in ex["float_nodes"]].count("LSTM") == 2
    assert ex["n_heavy_float_in_graph"] == 2


def test_conv_ae_reports_int8_and_simulated_layers(built, calib):
    _, p = built["conv_ae"]
    ex = Q.int8_execution(p["q2"], calib, 2, QCFG["q2"]["op_types"])
    assert ex["n_heavy"] == 6 and ex["n_heavy_quantized_in_graph"] == 6
    assert ex["n_heavy_int8_kernel"] + ex["n_heavy_qdq_simulated"] == 6
    assert {n["op"] for n in ex["float_nodes"]} >= {"Sigmoid", "Sub", "Pow", "ReduceMean"}


def test_int8_calibration_size_is_enforced(built, calib):
    _, p = built["cnn1d"]
    with pytest.raises(ValueError):
        Q.to_int8(p["q0"], p["q0"].with_name("x.onnx"), calib[:499], QCFG)


def _index(split, n):
    return pd.DataFrame({"dataset": "road", "split": split, "segment": "s", "start": np.arange(n) * 100,
                         "label": 0, "capture": "c"})


def test_calibration_windows_must_come_from_training():
    Q.check_calibration_selection(_index("train", 500), 500)
    for split in ("val", "test"):
        with pytest.raises(ValueError):
            Q.check_calibration_selection(_index(split, 1000), 500)
    with pytest.raises(ValueError):
        Q.check_calibration_selection(_index("train", 499), 500)
    mixed = pd.concat([_index("train", 600), _index("test", 1)])
    with pytest.raises(ValueError):
        Q.check_calibration_selection(mixed, 500)


def test_fp16_coverage_is_measured(built, calib):
    _, p = built["cnn1d"]
    cov = Q.fp16_coverage(p["q1"], calib, 2)
    assert cov["n_heavy"] == 5
    assert 0 <= cov["n_heavy_fp16"] <= cov["n_heavy"] and 0 <= cov["share_fp16"] <= 1
    assert sum(d["float16"] + d["other"] for d in cov["by_op"].values()) == cov["n_kernels"]


def test_q1_low_coverage_needs_a_decision():
    low = {"n_heavy": 5, "n_heavy_fp16": 0, "heavy_share_fp16": 0.0}
    high = {"n_heavy": 5, "n_heavy_fp16": 5, "heavy_share_fp16": 1.0}
    stop = {**QCFG, "q1": {**QCFG["q1"], "low_cpu_coverage_action": None}}
    with pytest.raises(Q.Q1CoverageError):
        Q.q1_status(low, stop)
    assert Q.q1_status(high, stop) == "cpu_fp16"
    sim = {**QCFG, "q1": {**QCFG["q1"], "low_cpu_coverage_action": "simulate_fp16"}}
    assert Q.q1_status(low, sim) == "cpu_unsupported"
    edge = {"n_heavy": 4, "n_heavy_fp16": 2, "heavy_share_fp16": 0.5}
    assert Q.q1_status(edge, stop) == "cpu_fp16"          # "more than half in float32" stops; exactly half passes


@pytest.mark.parametrize("name", ["cnn1d", "conv_ae"])
def test_fp16_simulation_is_close_but_not_float32(built, name):
    m, _ = built[name]
    x = _windows(32, seed=3)
    y16 = Q.fp16_scorer(m, "cpu")(x)
    ref = Q.torch_scores(m, x)
    assert np.isfinite(y16).all() and not np.array_equal(y16, ref)
    np.testing.assert_allclose(y16, ref, rtol=2e-2, atol=2e-3)


PILOT_META = sorted(Path(Q.S.PROJECT_ROOT / "results" / "raw").glob(f"*/{Q.QUANT_DIR}/quant_meta.json"))


@pytest.mark.skipif(not PILOT_META, reason="no quantized run under results/raw")
@pytest.mark.parametrize("path", PILOT_META, ids=lambda p: p.parent.parent.name)
def test_pilot_quant_meta(path):
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["git"]["commit"] and meta["environment"]["versions"]["onnxruntime"]
    lv = meta["levels"]
    assert lv["q0"]["equivalence_vs_pytorch_cpu"]["allclose"]
    cal = lv["q2"]["calibration"]
    assert cal["split"] == "train" and cal["n_windows"] >= QCFG["q2"]["min_calibration_windows"]
    iq = lv["q2"]["input_quantization"]
    assert iq["scheme"] == "asymmetric" and iq["levels_on_unit_interval"] == 256
    assert all(g["n_levels"] == g["n_values"] for g in lv["q2"]["input_levels"]["grids"].values())
    assert lv["q1"]["cpu_status"] in ("cpu_fp16", "cpu_unsupported")
    assert lv["q1"]["size_ratio_vs_q0"] <= 0.6 and lv["q2"]["size_ratio_vs_q0"] <= 0.4
