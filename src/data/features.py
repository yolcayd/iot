"""Per-message 10-dimensional feature vector (encoded ID, 8 payload bytes, log-scaled inter-arrival time).

The representation is designed so that single-scale (per-tensor) INT8 quantisation of the input is
lossless or close to it: a measured quantisation loss should come from the model, not from the
input scaling (docs/DECISIONS.md K-024, K-025). Every feature lies in [0, 1].

    x[0]    code / K                   ID vocabulary = sorted IDs of the benign (label 0) frames of
                                       the TRAINING segments, K entries; a known ID gets its rank
                                       1..K, an ID outside the vocabulary gets the reserved code 0
    x[1:9]  payload byte / 255         dlc < 8 payloads are zero-padded by the loader
    x[9]    log-scaled IAT             (log10(clip(iat, floor, cap)) - log10(floor))
                                       / (log10(cap) - log10(floor)); iat = time since the previous
                                       frame with the same ID inside the same segment; the first
                                       frame of an ID in a segment gets 1.0, so no feature looks
                                       across a segment boundary

Fitted parameters (ID vocabulary; IAT floor and cap = quantiles of the per-ID IATs of benign
training frames) come only from training segments (``fit_normalization`` refuses anything else)
and are stored with the list of segments they were fitted on in
results/tables/normalization_params.json. ``int8_levels`` maps features to the integer levels of a
per-tensor INT8 quantiser over [0, 1]; results/tables/feature_int8_levels.csv reports how many
distinct levels each column keeps.

Usage:
    python -m src.data.features      # fit; write normalization_params.json and feature_int8_levels.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import loaders as L
from src.data import sources as S

N_FEATURES = 10
FEATURE_NAMES = ["can_id"] + [f"byte_{i}" for i in range(8)] + ["iat"]
PARAMS_PATH = S.PROJECT_ROOT / "results" / "tables" / "normalization_params.json"
LEVELS_PATH = S.PROJECT_ROOT / "results" / "tables" / "feature_int8_levels.csv"
# Per-tensor INT8 quantisers over the calibration range [0, 1] (every feature spans it: payload
# bytes take both 0x00 and 0xFF). Asymmetric: scale 1/255 with a zero point (uint8, or int8 with
# zero point -128), 256 levels. Symmetric: scale 1/127, zero point 0, 128 levels on [0, 1].
INT8_SCHEMES = {"asymmetric": 255, "symmetric": 127}


def per_id_iat(ts: np.ndarray, can_id: np.ndarray) -> np.ndarray:
    """Seconds since the previous frame with the same ID (sequence order); NaN for an ID's first frame."""
    n = len(ts)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    order = np.argsort(can_id, kind="stable")
    sid, sts = can_id[order], ts[order]
    d = np.empty(n)
    d[0] = np.nan
    d[1:] = np.diff(sts)
    d[np.r_[True, sid[1:] != sid[:-1]]] = np.nan
    out[order] = d
    return out


def encode_ids(can_id: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """code / K: rank 1..K of the ID in the sorted vocabulary, 0 for an ID outside it."""
    vocab = np.asarray(vocab)
    k = len(vocab)
    if k == 0 or np.any(np.diff(vocab) <= 0):
        raise ValueError("vocabulary must be non-empty, sorted and unique")
    pos = np.searchsorted(vocab, can_id)
    known = (pos < k) & (vocab[np.minimum(pos, k - 1)] == can_id)
    return np.where(known, pos + 1, 0) / k


def scale_iat(iat: np.ndarray, floor: float, cap: float) -> np.ndarray:
    """Log-scaled IAT in [0, 1]; NaN (first frame of an ID in the segment) maps to 1.0."""
    if not 0 < floor < cap:
        raise ValueError("need 0 < floor < cap")
    lo, hi = np.log10(floor), np.log10(cap)
    with np.errstate(invalid="ignore"):
        v = (np.log10(np.clip(iat, floor, cap)) - lo) / (hi - lo)
    return np.where(np.isnan(iat), 1.0, v)


def message_features(ts: np.ndarray, can_id: np.ndarray, data: np.ndarray, id_vocab: np.ndarray,
                     iat_floor: float, iat_cap: float) -> np.ndarray:
    """(N, 10) float32 feature matrix of one contiguous segment."""
    x = np.empty((len(ts), N_FEATURES), dtype=np.float32)
    x[:, 0] = encode_ids(can_id, id_vocab)
    x[:, 1:9] = data / 255.0
    x[:, 9] = scale_iat(per_id_iat(ts, can_id), iat_floor, iat_cap)
    return x


def int8_levels(x: np.ndarray, scheme: str = "asymmetric") -> np.ndarray:
    """Integer level of each value under a per-tensor INT8 quantiser calibrated on [0, 1]."""
    q = INT8_SCHEMES[scheme]
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) * q), 0, q).astype(np.int64)


def benign_iats(ts: np.ndarray, can_id: np.ndarray, label: np.ndarray) -> np.ndarray:
    iat = per_id_iat(ts, can_id)
    return iat[(label == 0) & ~np.isnan(iat)]


def fit_normalization(segments: list, floor_quantile: float, cap_quantile: float) -> dict:
    """ID vocabulary and IAT floor/cap from the benign frames of training segments only."""
    bad = [s.id for s in segments if s.split != "train"]
    if bad:
        raise ValueError(f"normalization may only be fitted on training segments, got {bad[:3]}")
    if not segments:
        raise ValueError("no training segments")
    samples, ids = [], set()
    for seg in segments:
        t = seg.read(["timestamp", "can_id", "label"])
        cid, lab = t["can_id"].to_numpy(), t["label"].to_numpy()
        samples.append(benign_iats(t["timestamp"].to_numpy(), cid, lab))
        ids |= set(np.unique(cid[lab == 0]).tolist())
    iat = np.concatenate(samples)
    floor, cap = np.quantile(iat, [floor_quantile, cap_quantile])
    return {"id_vocab": sorted(int(i) for i in ids), "iat_floor_s": float(floor), "iat_cap_s": float(cap),
            "n_iat_samples": int(len(iat))}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fit_all(cfg: dict | None = None) -> dict:
    from src.data import splits as SP  # local import: splits imports this module

    cfg = cfg or S.load_config()
    icfg = cfg["features"]["iat"]
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256_16": config_hash({k: cfg[k] for k in ("features", "splits", "shared_segments")}),
        "feature_names": FEATURE_NAMES,
        "datasets": {},
    }
    segs = SP.segments(cfg)
    for dataset in ("car_hacking", "road"):
        train = [s for s in segs if s.dataset == dataset and s.split == "train"]
        fit = fit_normalization(train, icfg["floor_quantile"], icfg["cap_quantile"])
        out["datasets"][dataset] = {
            "id_encoding": cfg["features"]["id_encoding"],
            "n_ids": len(fit["id_vocab"]),
            "id_vocab": fit["id_vocab"],
            "iat_scale": icfg["scale"],
            "iat_floor_s": fit["iat_floor_s"],
            "iat_cap_s": fit["iat_cap_s"],
            "iat_floor_quantile": icfg["floor_quantile"],
            "iat_cap_quantile": icfg["cap_quantile"],
            "n_iat_samples": fit["n_iat_samples"],
            "fitted_on_splits": sorted({s.split for s in train}),
            "fitted_on_segments": [s.id for s in train],
        }
        print(f"{dataset}: {len(fit['id_vocab'])} IDs, IAT floor {fit['iat_floor_s'] * 1e3:.4f} ms, "
              f"cap {fit['iat_cap_s']:.6f} s from {fit['n_iat_samples']} benign IATs in {len(train)} "
              f"training segments", flush=True)
    return out


def level_report(params: dict) -> list[dict]:
    """Distinct INT8 levels per feature column, computed on the representation itself."""
    rows = []
    for dataset, p in params["datasets"].items():
        k = p["n_ids"]
        id_values = np.arange(k + 1) / k  # reserved unseen code 0 + vocabulary ranks 1..K
        byte_values = np.arange(256) / 255.0
        lo, hi = p["iat_floor_s"], p["iat_cap_s"]
        for scheme, q in INT8_SCHEMES.items():
            at = {ms: int(int8_levels(scale_iat(np.array([ms / 1e3]), lo, hi), scheme)[0]) for ms in (1, 10, 20, 100)}
            rows += [
                {"dataset": dataset, "scheme": scheme, "feature": "can_id", "n_values": k + 1,
                 "n_levels": len(np.unique(int8_levels(id_values, scheme))),
                 "iat_level_1ms": np.nan, "iat_level_10ms": np.nan, "iat_levels_per_decade": np.nan},
                {"dataset": dataset, "scheme": scheme, "feature": "byte (each of 8)", "n_values": 256,
                 "n_levels": len(np.unique(int8_levels(byte_values, scheme))),
                 "iat_level_1ms": np.nan, "iat_level_10ms": np.nan, "iat_levels_per_decade": np.nan},
                {"dataset": dataset, "scheme": scheme, "feature": "iat", "n_values": np.nan, "n_levels": q + 1,
                 "iat_level_1ms": at[1], "iat_level_10ms": at[10],
                 "iat_levels_per_decade": q / (np.log10(hi) - np.log10(lo))},
            ]
    return rows


def load_params(path: Path = PARAMS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_features(seg, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and frame labels of one split segment."""
    p = params["datasets"][seg.dataset]
    t = seg.read(["timestamp", "can_id", "data", "label"])
    x = message_features(t["timestamp"].to_numpy(), t["can_id"].to_numpy(), L.data_matrix(t),
                         np.asarray(p["id_vocab"]), p["iat_floor_s"], p["iat_cap_s"])
    return x, t["label"].to_numpy()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    params = fit_all()
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    report = pd.DataFrame(level_report(params))
    report.to_csv(LEVELS_PATH, index=False, float_format="%.4g")
    print(report.to_string(index=False))
    print(f"wrote {PARAMS_PATH.relative_to(S.PROJECT_ROOT)} and {LEVELS_PATH.relative_to(S.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
