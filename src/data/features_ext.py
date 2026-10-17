"""Extended per-frame features for the advisor-requested ablation (P10; docs/DECISIONS.md K-073).

The 10 reference features of ``src.data.features`` (set ``s0``) are kept unchanged; the sets ``s1`` ... ``s4`` of
configs/features_ext.yaml append, in a fixed order, the columns of four groups:

    bit_flips      G1  Hamming distance between the 64-bit payload and the payload of the previous frame with
                       the same ID, / 64 (values k/64); 0 for the first frame of an ID in the segment
    dbyte_0..7     G2  |byte_i - byte_i of the previous frame with the same ID| / 255 (values k/255); 0 likewise
    id_rate        G3  r = frames of the same ID in the trailing window (t - window_s, t] / c_ref(ID);
                       (clip(log2 r, -c, c) + c) / (2c). ``neutral`` in the first window_s seconds of the segment,
                       ``unreferenced`` for an ID without reference count
    id_share_w     G4  frames of the same ID among the trailing ``frames`` frames / frames (values k/frames)
    id_entropy_w   G4  Shannon entropy of the raw-ID histogram of the trailing ``frames`` frames / log2(frames)

Every column lies in [0, 1], is computed inside one split segment and only from the current and earlier frames
(causal). Payloads of frames with dlc < 8 are zero-padded by the loader; the comparison with the previous frame
of the same ID uses the padded bytes.

The only fitted parameter, c_ref(ID) (median trailing count of the ID), comes from benign TRAINING frames that lie
more than ``margin_s`` from any injected frame of their capture and not in a masquerade capture (``fit_ext``
refuses any other segment). It is written with the list of segments it was fitted on to
results/tables/normalization_params_ext.json.

Usage:
    python -m src.data.features_ext --fit                    # fit c_ref; write normalization_params_ext.json
    python -m src.data.features_ext --build-cache [DATASET]  # compute the extended feature cache of every segment
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
import yaml

from src.data import features as F
from src.data import loaders as L
from src.data import sources as S

EXT_CONFIG = S.PROJECT_ROOT / "configs" / "features_ext.yaml"
EXT_PARAMS_PATH = S.PROJECT_ROOT / "results" / "tables" / "normalization_params_ext.json"
REFERENCE_SET = "s0"
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------


def load_ext_config(path: Path = EXT_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def all_ext_columns(ecfg: dict) -> list[str]:
    """Extended columns in the order of the largest set (every set is a prefix of it)."""
    largest = max(ecfg["sets"].values(), key=len)
    return [c for g in largest for c in ecfg["groups"][g]["columns"]]


def set_columns(feature_set: str, ecfg: dict | None = None) -> list[str]:
    ecfg = ecfg or load_ext_config()
    if feature_set not in ecfg["sets"]:
        raise ValueError(f"unknown feature set {feature_set!r}; known: {sorted(ecfg['sets'])}")
    cols = [c for g in ecfg["sets"][feature_set] for c in ecfg["groups"][g]["columns"]]
    if cols != all_ext_columns(ecfg)[:len(cols)]:
        raise ValueError(f"feature set {feature_set!r} is not a prefix of the largest set (sets must be cumulative)")
    return cols


def feature_names(feature_set: str, ecfg: dict | None = None) -> list[str]:
    return list(F.FEATURE_NAMES) + set_columns(feature_set, ecfg)


def n_features(feature_set: str, ecfg: dict | None = None) -> int:
    return len(feature_names(feature_set, ecfg))


def column_kind(name: str, ecfg: dict | None = None) -> str:
    """``grid`` (values k/n, must stay lossless under INT8 input quantization) or ``continuous``."""
    if name == "iat":
        return "continuous"
    if name in F.FEATURE_NAMES:
        return "grid"
    kinds = (ecfg or load_ext_config())["column_kind"]
    key = "dbyte" if name.startswith("dbyte_") else name
    return kinds[key]["kind"]


# --------------------------------------------------------------------------------------------
# Column computations (one contiguous segment, frames in recording order)
# --------------------------------------------------------------------------------------------


def _id_groups(can_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable order by ID (time order kept within an ID) and the group number of every sorted position."""
    order = np.argsort(can_id, kind="stable")
    sid = can_id[order]
    gid = np.cumsum(np.r_[True, sid[1:] != sid[:-1]]) - 1 if len(sid) else np.empty(0, dtype=np.int64)
    return order, gid.astype(np.int64)


def previous_same_id(can_id: np.ndarray) -> np.ndarray:
    """Row of the previous frame with the same ID; -1 for the first frame of an ID."""
    n = len(can_id)
    out = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return out
    order, gid = _id_groups(can_id)
    prev = np.r_[-1, order[:-1]]
    prev[np.r_[True, gid[1:] != gid[:-1]]] = -1
    out[order] = prev
    return out


def bit_flips(data: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Hamming distance of the 64-bit payload to the previous same-ID payload, / 64 (0 without predecessor)."""
    has = prev >= 0
    out = np.zeros(len(data), dtype=np.float64)
    x = np.bitwise_xor(data[has], data[prev[has]])
    out[has] = _POPCOUNT[x].sum(axis=1) / 64.0
    return out


def payload_delta(data: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """|byte_i - byte_i of the previous same-ID frame| / 255, (N, 8); 0 without predecessor."""
    has = prev >= 0
    out = np.zeros((len(data), 8), dtype=np.float64)
    out[has] = np.abs(data[has].astype(np.int16) - data[prev[has]].astype(np.int16)) / 255.0
    return out


def trailing_time_count(ts: np.ndarray, can_id: np.ndarray, window_s: float) -> np.ndarray:
    """Number of frames with the same ID in (t - window_s, t], the frame itself included."""
    n = len(ts)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    order, gid = _id_groups(can_id)
    rel = ts[order] - ts.min()
    # groups are laid out one after another with a gap larger than the window, so a search never crosses a group
    key = gid * (float(rel.max()) + 2.0 * window_s + 1.0) + rel
    lo = np.searchsorted(key, key - window_s, side="right")
    out = np.empty(n, dtype=np.int64)
    out[order] = np.arange(n) - lo + 1
    return out


def id_rate(ts: np.ndarray, can_id: np.ndarray, ref_ids: np.ndarray, ref_counts: np.ndarray, window_s: float,
            log2_clip: float, neutral: float, unreferenced: float) -> np.ndarray:
    """Trailing same-ID count relative to the ID's benign training reference, log2-scaled to [0, 1]."""
    count = trailing_time_count(ts, can_id, window_s)
    out = np.full(len(ts), float(unreferenced))
    ref_ids = np.asarray(ref_ids)
    if len(ref_ids):
        pos = np.searchsorted(ref_ids, can_id)
        known = (pos < len(ref_ids)) & (ref_ids[np.minimum(pos, len(ref_ids) - 1)] == can_id)
        r = count[known] / np.asarray(ref_counts, dtype=np.float64)[pos[known]]
        out[known] = (np.clip(np.log2(r), -log2_clip, log2_clip) + log2_clip) / (2 * log2_clip)
    if len(ts):
        out[ts - ts[0] < window_s] = neutral
    return out


def _xlog2x(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c > 0, c * np.log2(np.maximum(c, 1.0)), 0.0)


def id_window(can_id: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Share of the frame's ID and normalised ID entropy over the trailing ``frames`` frames of the segment.

    Window of frame t = rows max(0, t - frames + 1) .. t (shorter at the start of the segment). Share uses the
    fixed denominator ``frames``; entropy is that of the window's histogram, divided by log2(frames).
    The entropy is updated incrementally: with S = sum_k c_k log2 c_k over the window, H = log2 n - S / n.
    """
    n = len(can_id)
    if n == 0:
        return np.empty(0), np.empty(0)
    order, gid = _id_groups(can_id)
    key = gid * (n + frames + 1) + order               # row numbers laid out per ID, groups never overlap
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    k = key[pos]                                         # key of every row (recording order)
    # c_new[t]: frames of ID(t) among rows (t - frames, t]
    c_new = pos - np.searchsorted(key, k - frames, side="right") + 1
    ds = _xlog2x(c_new) - _xlog2x(c_new - 1)
    t = np.arange(n)
    j = t[frames:] - frames                              # row leaving the window when row t enters
    if len(j):
        # frames of ID(j) among rows j .. j + frames - 1 (= the window of t - 1)
        c_old = np.searchsorted(key, k[j] + frames - 1, side="right") - pos[j]
        same = can_id[j] == can_id[t[frames:]]
        leave = _xlog2x(c_old - 1) - _xlog2x(c_old)
        ds[frames:] = np.where(same, 0.0, ds[frames:] + leave)
    s = np.cumsum(ds)
    m = np.minimum(t + 1, frames).astype(np.float64)
    ent = (np.log2(m) - s / m) / np.log2(frames)
    return c_new / float(frames), np.clip(ent, 0.0, 1.0)


def ext_columns(ts: np.ndarray, can_id: np.ndarray, data: np.ndarray, ref: dict, ecfg: dict) -> np.ndarray:
    """(N, n_ext) float32 matrix of every extended column (order of ``all_ext_columns``) of one segment."""
    prev = previous_same_id(can_id)
    rc, wc = ecfg["id_rate"], ecfg["id_window"]
    cols = {"bit_flips": bit_flips(data, prev)}
    dd = payload_delta(data, prev)
    for i in range(8):
        cols[f"dbyte_{i}"] = dd[:, i]
    cols["id_rate"] = id_rate(ts, can_id, np.asarray(ref["ref_ids"]), np.asarray(ref["ref_counts"]),
                              rc["window_s"], rc["log2_clip"], rc["neutral"], rc["unreferenced"])
    cols["id_share_w"], cols["id_entropy_w"] = id_window(can_id, wc["frames"])
    names = all_ext_columns(ecfg)
    return np.stack([cols[c] for c in names], axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Fitting (benign training frames only) and parameters
# --------------------------------------------------------------------------------------------


def clean_frames(ts: np.ndarray, injected_ts: np.ndarray, margin_s: float) -> np.ndarray:
    """True where no injected frame of the capture lies within +/- margin_s."""
    if len(injected_ts) == 0:
        return np.ones(len(ts), dtype=bool)
    inj = np.sort(injected_ts)
    return np.searchsorted(inj, ts - margin_s, side="left") == np.searchsorted(inj, ts + margin_s, side="right")


def reference_counts(segments: list, ecfg: dict) -> dict:
    """c_ref(ID) = median trailing count over the reference frames of the given TRAINING segments."""
    from src.data import splits as SP  # local import: splits imports src.data.features

    rc = ecfg["id_rate"]
    ref = rc["reference"]
    bad = [s.id for s in segments if s.split != ref["split"]]
    if bad:
        raise ValueError(f"id_rate reference may only be fitted on training segments, got {bad[:3]}")
    if not segments:
        raise ValueError("no training segments")
    ids, counts, used = [], [], []
    for seg in segments:
        if ref["exclude_masquerade"] and seg.is_masquerade:
            continue
        t = seg.read(["timestamp", "can_id", "label"])
        ts, cid, lab = t["timestamp"].to_numpy(), t["can_id"].to_numpy(), t["label"].to_numpy()
        c = trailing_time_count(ts, cid, rc["window_s"])
        ok = (lab == ref["label"]) & (ts - ts[0] >= rc["window_s"])
        ok &= clean_frames(ts, SP.injected_timestamps(seg.parquet), ref["margin_s"])
        ids.append(cid[ok])
        counts.append(c[ok])
        used.append(seg.id)
    df = pd.DataFrame({"id": np.concatenate(ids), "count": np.concatenate(counts)})
    med = df.groupby("id")["count"].median().sort_index()
    return {"ref_ids": [int(i) for i in med.index], "ref_counts": [float(v) for v in med.to_numpy()],
            "n_reference_frames": int(len(df)), "fitted_on_segments": used}


def ext_hash(ecfg: dict, base_hash: str) -> str:
    return F.config_hash({"features_ext": ecfg, "base_config_sha256_16": base_hash})


def set_hash(base_hash: str, feature_set: str, ext_h: str) -> str:
    """Data hash of a feature set; the reference set keeps the hash of the 10-feature parameters (run ids of S0)."""
    if feature_set == REFERENCE_SET:
        return base_hash
    return hashlib.sha256(f"{base_hash}|{feature_set}|{ext_h}".encode()).hexdigest()[:16]


def fit_all(cfg: dict | None = None, ecfg: dict | None = None, base: dict | None = None) -> dict:
    from src.data import splits as SP

    cfg = cfg or S.load_config()
    ecfg = ecfg or load_ext_config()
    base = base or F.load_params()
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "base_config_sha256_16": base["config_sha256_16"],
           "ext_sha256_16": ext_hash(ecfg, base["config_sha256_16"]),
           "columns": all_ext_columns(ecfg), "datasets": {}}
    segs = SP.segments(cfg)
    for dataset in ("car_hacking", "road"):
        train = [s for s in segs if s.dataset == dataset and s.split == "train"]
        fit = reference_counts(train, ecfg)
        vocab = set(base["datasets"][dataset]["id_vocab"])
        out["datasets"][dataset] = {**fit, "n_ref_ids": len(fit["ref_ids"]),
                                    "vocab_ids_without_reference": sorted(vocab - set(fit["ref_ids"])),
                                    "fitted_on_splits": sorted({s.split for s in train})}
        print(f"{dataset}: c_ref for {len(fit['ref_ids'])} IDs from {fit['n_reference_frames']:,} reference frames in "
              f"{len(fit['fitted_on_segments'])} training segments; vocabulary IDs without reference: "
              f"{len(out['datasets'][dataset]['vocab_ids_without_reference'])}", flush=True)
    return out


def load_ext_params(path: Path = EXT_PARAMS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_ext_params_current(ext_params: dict, base: dict, ecfg: dict) -> None:
    """Refuse stale extended parameters (fitted on another base or extension config)."""
    if ext_params.get("base_config_sha256_16") != base["config_sha256_16"] or \
            ext_params.get("ext_sha256_16") != ext_hash(ecfg, base["config_sha256_16"]):
        raise RuntimeError("normalization_params_ext.json does not match configs/features_ext.yaml or the base "
                           "parameters; re-run `python -m src.data.features_ext --fit`")


def segment_ext_features(seg, ext_params: dict, ecfg: dict) -> np.ndarray:
    t = seg.read(["timestamp", "can_id", "data"])
    return ext_columns(t["timestamp"].to_numpy(), t["can_id"].to_numpy(), L.data_matrix(t),
                       ext_params["datasets"][seg.dataset], ecfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--build-cache", nargs="*", metavar="DATASET",
                    help="compute the extended feature cache (all segments of the given datasets; default both)")
    a = ap.parse_args(argv)
    if a.fit:
        params = fit_all()
        EXT_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXT_PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {EXT_PARAMS_PATH.relative_to(S.PROJECT_ROOT)}")
    if a.build_cache is not None:
        from src.data import dataset as D

        for dataset in a.build_cache or ["car_hacking", "road"]:
            D.FeatureStore.build(dataset, feature_set=max(load_ext_config()["sets"], key=lambda s: len(
                load_ext_config()["sets"][s])))
    if not a.fit and a.build_cache is None:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
