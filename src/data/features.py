"""Per-message 10-dimensional feature vector (normalised ID, 8 payload bytes, normalised inter-arrival time).

    x[0]    can_id / id_space          id_space is fixed per dataset in configs/data.yaml
                                       (Car-Hacking 0x7FF, ROAD 0xFFF), not learned from data
    x[1:9]  payload byte / 255         dlc < 8 payloads are zero-padded by the loader
    x[9]    min(iat, cap) / cap        iat = time since the previous frame with the same ID inside
                                       the same segment; the first frame of an ID in a segment gets
                                       the cap (1.0), so no feature looks across a segment boundary

The IAT cap is the only fitted parameter: the ``iat_cap_quantile`` of the per-ID IATs of benign
(label 0) frames in the TRAINING segments of a dataset. It is fitted only from training segments
(``fit_iat_cap`` refuses anything else) and stored with the list of segments it was fitted on in
results/tables/normalization_params.json.

Usage:
    python -m src.data.features      # fit and write normalization_params.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data import loaders as L
from src.data import sources as S

N_FEATURES = 10
FEATURE_NAMES = ["can_id"] + [f"byte_{i}" for i in range(8)] + ["iat"]
PARAMS_PATH = S.PROJECT_ROOT / "results" / "tables" / "normalization_params.json"


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


def message_features(ts: np.ndarray, can_id: np.ndarray, data: np.ndarray, id_space: int, iat_cap: float) -> np.ndarray:
    """(N, 10) float32 feature matrix of one contiguous segment."""
    if iat_cap <= 0:
        raise ValueError("iat_cap must be positive")
    if np.any(can_id > id_space) or np.any(can_id < 0):
        raise ValueError(f"CAN ID outside [0, {id_space:#x}]")
    x = np.empty((len(ts), N_FEATURES), dtype=np.float32)
    x[:, 0] = can_id / id_space
    x[:, 1:9] = data / 255.0
    iat = per_id_iat(ts, can_id)
    iat = np.where(np.isnan(iat), iat_cap, np.clip(iat, 0.0, iat_cap))
    x[:, 9] = iat / iat_cap
    return x


def benign_iats(ts: np.ndarray, can_id: np.ndarray, label: np.ndarray) -> np.ndarray:
    iat = per_id_iat(ts, can_id)
    return iat[(label == 0) & ~np.isnan(iat)]


def fit_iat_cap(segments: list, quantile: float) -> tuple[float, int]:
    """IAT cap from training segments only; returns (cap, number of IAT samples)."""
    bad = [s.id for s in segments if s.split != "train"]
    if bad:
        raise ValueError(f"normalization may only be fitted on training segments, got {bad[:3]}")
    if not segments:
        raise ValueError("no training segments")
    samples = []
    for seg in segments:
        t = seg.read(["timestamp", "can_id", "label"])
        samples.append(benign_iats(t["timestamp"].to_numpy(), t["can_id"].to_numpy(), t["label"].to_numpy()))
    iat = np.concatenate(samples)
    return float(np.quantile(iat, quantile)), int(len(iat))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fit_all(cfg: dict | None = None) -> dict:
    from src.data import splits as SP  # local import: splits imports this module

    cfg = cfg or S.load_config()
    fcfg = cfg["features"]
    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256_16": config_hash({k: cfg[k] for k in ("features", "splits", "shared_segments")}),
        "feature_names": FEATURE_NAMES,
        "datasets": {},
    }
    segs = SP.segments(cfg)
    for dataset in ("car_hacking", "road"):
        train = [s for s in segs if s.dataset == dataset and s.split == "train"]
        cap, n = fit_iat_cap(train, fcfg["iat_cap_quantile"])
        out["datasets"][dataset] = {
            "id_space": int(fcfg["id_space"][dataset]),
            "iat_cap_s": cap,
            "iat_cap_quantile": fcfg["iat_cap_quantile"],
            "n_iat_samples": n,
            "fitted_on_splits": sorted({s.split for s in train}),
            "fitted_on_segments": [s.id for s in train],
        }
        print(f"{dataset}: iat cap {cap:.6f} s from {n} benign IATs in {len(train)} training segments", flush=True)
    return out


def load_params(path: Path = PARAMS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def segment_features(seg, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and frame labels of one split segment."""
    p = params["datasets"][seg.dataset]
    t = seg.read(["timestamp", "can_id", "data", "label"])
    x = message_features(t["timestamp"].to_numpy(), t["can_id"].to_numpy(), L.data_matrix(t), p["id_space"], p["iat_cap_s"])
    return x, t["label"].to_numpy()


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    params = fit_all()
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {PARAMS_PATH.relative_to(S.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
