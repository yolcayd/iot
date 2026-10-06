"""Feature cache and window selection for training and evaluation.

The frame-level feature matrix of every split segment (``src.data.features.segment_features``) is cached
under data/processed/features/ (keyed by the normalisation config hash, so a re-fit invalidates it). All
segments of a dataset are concatenated into one array; a window is addressed by the global row of its
first frame and gathered on the fly (on the GPU for the deep models), so overlapping training windows
are never copied.

Window selectors (each checks the split it may draw from):

    supervised_train  training windows, minus the label-0 windows of masquerade captures, which duplicate
                      the attack-free part of their source capture (docs/DECISIONS.md K-028)
    ae_train          ``normal_subset`` windows (train only, label 0, +/-5 s margin, no masquerade; K-022)
    ae_val            validation windows with label 0, no masquerade capture, same +/-5 s margin; early
                      stopping of the autoencoders (K-030) and, as ``calibration_windows``, the decision
                      threshold of every model (K-038)
    eval_windows      every window of the validation or test split
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import features as F
from src.data import loaders as L
from src.data import sources as S
from src.data import splits as SP
from src.data import windows as W

FEATURE_CACHE = L.PROCESSED_DIR / "features"


def load_window_index() -> pd.DataFrame:
    return pd.read_parquet(SP.WINDOW_INDEX)


def check_params_current(params: dict, cfg: dict) -> None:
    """Refuse to train on stale normalisation parameters (K-024, K-025)."""
    want = F.config_hash({k: cfg[k] for k in ("features", "splits", "shared_segments")})
    if params.get("config_sha256_16") != want:
        raise RuntimeError("normalization_params.json does not match configs/data.yaml; "
                           "re-run `python -m src.data.features`")


def _cache_path(seg: SP.Segment, params_hash: str):
    key = hashlib.sha256(f"{seg.id}|{params_hash}".encode()).hexdigest()[:16]
    return FEATURE_CACHE / seg.dataset / f"{key}.npy"


def segment_matrix(seg: SP.Segment, params: dict) -> np.ndarray:
    path = _cache_path(seg, params["config_sha256_16"])
    if path.exists():
        x = np.load(path, mmap_mode=None)
        if x.shape == (seg.n_frames, F.N_FEATURES):
            return x
    x, _ = F.segment_features(seg, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x.astype(np.float32))
    return x


@dataclass
class FeatureStore:
    """All frames of the requested splits of one dataset, concatenated."""
    dataset: str
    frames: np.ndarray                    # (n_frames, 10) float32
    offsets: dict[str, tuple[int, int]]   # segment id -> (offset in frames, segment start row)
    segments: dict[str, SP.Segment]
    params_hash: str

    @classmethod
    def build(cls, dataset: str, splits=SP.SPLITS, cfg: dict | None = None, params: dict | None = None,
              log=print) -> "FeatureStore":
        cfg = cfg or S.load_config()
        params = params or F.load_params()
        check_params_current(params, cfg)
        segs = [s for s in SP.segments(cfg) if s.dataset == dataset and s.split in splits]
        mats, offsets, off = [], {}, 0
        for s in segs:
            mats.append(segment_matrix(s, params))
            offsets[s.id] = (off, s.start)
            off += s.n_frames
        log(f"{dataset}: {off:,} frames in {len(segs)} segments ({', '.join(splits)})")
        return cls(dataset, np.concatenate(mats), offsets, {s.id: s for s in segs}, params["config_sha256_16"])

    def global_starts(self, win: pd.DataFrame) -> np.ndarray:
        off = win["segment"].astype(str).map(lambda s: self.offsets[s][0]).to_numpy(np.int64)
        base = win["segment"].astype(str).map(lambda s: self.offsets[s][1]).to_numpy(np.int64)
        return off + win["start"].to_numpy(np.int64) - base

    def windows(self, gstarts: np.ndarray, size: int) -> np.ndarray:
        """(B, size, 10) copy of the windows starting at the given global rows (CPU)."""
        return self.frames[np.asarray(gstarts)[:, None] + np.arange(size)]


def _require_split(win: pd.DataFrame, split: str) -> pd.DataFrame:
    bad = set(win["split"].astype(str)) - {split}
    if bad:
        raise ValueError(f"selector for split {split!r} got windows from {sorted(bad)}")
    return win


def _masquerade_capture(win: pd.DataFrame, store_segments: dict[str, SP.Segment]) -> np.ndarray:
    return win["segment"].astype(str).map(lambda s: store_segments[s].is_masquerade).to_numpy(bool)


def supervised_train(win: pd.DataFrame, dataset: str, segs: dict[str, SP.Segment]) -> pd.DataFrame:
    w = win[(win["dataset"] == dataset) & (win["split"] == "train")]
    drop = _masquerade_capture(w, segs) & (w["label"].to_numpy() == 0)
    return _require_split(w[~drop].reset_index(drop=True), "train")


def ae_train(win: pd.DataFrame, dataset: str) -> pd.DataFrame:
    w = win[(win["dataset"] == dataset) & win["normal_subset"]].reset_index(drop=True)
    if (w["label"] != 0).any():
        raise ValueError("normal subset contains attack windows")
    return _require_split(w, "train")


def ae_val(win: pd.DataFrame, dataset: str, segs: dict[str, SP.Segment], cfg: dict) -> pd.DataFrame:
    size, margin = cfg["windows"]["size"], cfg["normal_subset"]["margin_s"]
    w = win[(win["dataset"] == dataset) & (win["split"] == "val") & (win["label"] == 0)]
    w = w[~_masquerade_capture(w, segs)]
    keep = np.zeros(len(w), dtype=bool)
    pos = np.arange(len(w))
    for seg_id, idx in w.groupby(w["segment"].astype(str), observed=True).indices.items():
        seg = segs[seg_id]
        ts = seg.read(["timestamp"])["timestamp"].to_numpy()
        rel = w["start"].to_numpy()[idx] - seg.start
        keep[pos[idx]] = W.margin_clean(ts, rel, size, SP.injected_timestamps(seg.parquet), margin)
    return _require_split(w[keep].reset_index(drop=True), "val")


def calibration_windows(win: pd.DataFrame, dataset: str, segs: dict[str, SP.Segment], cfg: dict) -> pd.DataFrame:
    """Clean validation normals for the decision threshold of every model (docs/DECISIONS.md K-038).

    The same windows as ``ae_val``: label 0, not from a masquerade capture (they duplicate the attack-free part
    of their fabrication source) and no frame within +/-5 s of an injected frame (attack after-effects).
    """
    return ae_val(win, dataset, segs, cfg)


def eval_windows(win: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    if split not in ("val", "test"):
        raise ValueError("evaluation runs on val or test")
    return _require_split(win[(win["dataset"] == dataset) & (win["split"] == split)].reset_index(drop=True), split)


def describe(win: pd.DataFrame) -> dict:
    return {"n_windows": int(len(win)), "n_attack": int(win["label"].sum()),
            "n_normal": int((win["label"] == 0).sum()),
            "n_captures": int(win["capture"].astype(str).nunique())}


def selection_digest(win: pd.DataFrame) -> str:
    """Short hash of the selected windows (recorded in run_config.json)."""
    h = hashlib.sha256(json.dumps([win["segment"].astype(str).tolist(), win["start"].tolist()]).encode())
    return h.hexdigest()[:16]
