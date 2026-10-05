"""Classical reference models: Random Forest and XGBoost on timing/ID features (M0).

Both models see window-level summary features computed from the same 10-dimensional feature windows
the deep models use (so they share the training-only normalisation, K-024/K-025):

    id_entropy, n_unique_ids, unseen_id_share          ID distribution (code 0 = ID outside vocabulary)
    iat_mean, iat_std, iat_min, iat_q10                log-scaled IAT feature
    byte{i}_mean, byte{i}_var (i = 0..7)               payload bytes
    payload_change_rate                                share of consecutive same-ID frame pairs whose
                                                       payload differs
"""

from __future__ import annotations

import numpy as np

SUMMARY_NAMES = (["id_entropy", "n_unique_ids", "unseen_id_share", "iat_mean", "iat_std", "iat_min", "iat_q10"]
                 + [f"byte{i}_mean" for i in range(8)] + [f"byte{i}_var" for i in range(8)]
                 + ["payload_change_rate"])


def _id_stats(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shannon entropy (bits) and number of distinct values of every row."""
    b, n = ids.shape
    s = np.sort(ids, axis=1)
    new = np.ones_like(s, dtype=bool)
    new[:, 1:] = s[:, 1:] != s[:, :-1]
    grp = np.cumsum(new, axis=1) - 1
    counts = np.zeros((b, n), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(b), n), grp.ravel()), 1)
    p = counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return ent, new.sum(axis=1)


def _payload_change_rate(ids: np.ndarray, payload: np.ndarray) -> np.ndarray:
    order = np.argsort(ids, axis=1, kind="stable")              # time order kept within an ID
    sid = np.take_along_axis(ids, order, axis=1)
    sp = np.take_along_axis(payload, order[:, :, None], axis=1)
    same = sid[:, 1:] == sid[:, :-1]
    changed = (sp[:, 1:] != sp[:, :-1]).any(axis=2) & same
    pairs = same.sum(axis=1)
    return np.where(pairs > 0, changed.sum(axis=1) / np.maximum(pairs, 1), 0.0)


def summary_features(windows: np.ndarray) -> np.ndarray:
    """(B, T, 10) feature windows -> (B, len(SUMMARY_NAMES)) float32."""
    w = np.asarray(windows, dtype=np.float32)
    ids, payload, iat = w[:, :, 0], w[:, :, 1:9], w[:, :, 9]
    ent, n_unique = _id_stats(ids)
    cols = [ent, n_unique, (ids == 0).mean(axis=1), iat.mean(axis=1), iat.std(axis=1), iat.min(axis=1),
            np.quantile(iat, 0.1, axis=1)]
    cols += list(payload.mean(axis=1).T) + list(payload.var(axis=1).T)
    cols.append(_payload_change_rate(ids, payload))
    return np.stack(cols, axis=1).astype(np.float32)


def pos_ratio(y: np.ndarray) -> float:
    """N_neg / N_pos (K-027)."""
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("no positive training windows")
    return float((y == 0).sum() / n_pos)


def build_rf(cfg: dict, seed: int, threads: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                  min_samples_leaf=cfg["min_samples_leaf"], class_weight=cfg["class_weight"],
                                  random_state=seed, n_jobs=threads)


def build_xgb(cfg: dict, seed: int, threads: int, device: str, scale_pos_weight: float):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                         learning_rate=cfg["learning_rate"], subsample=cfg["subsample"],
                         colsample_bytree=cfg["colsample_bytree"], scale_pos_weight=scale_pos_weight,
                         early_stopping_rounds=cfg["early_stopping_rounds"], eval_metric="logloss",
                         tree_method="hist", device=device, random_state=seed, n_jobs=threads)
