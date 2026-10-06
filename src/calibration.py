"""Decision threshold calibrated on validation normals (docs/DECISIONS.md K-035, K-036, K-038).

threshold = the (1 - target_fpr) quantile ("higher" method) of the scores of the label-0 windows passed in;
a window is flagged as an attack when score > threshold, so the realised false-positive rate on those
normals is at most ``target_fpr`` (lower when scores tie at the threshold). Attack labels are never used,
and scores from any split other than validation are refused. The windows passed in by ``src.train`` are
the clean validation normals (``src.data.dataset.calibration_windows``, K-038).
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(scores: np.ndarray, labels: np.ndarray, split: str, target_fpr: float) -> dict:
    if split != "val":
        raise ValueError(f"the decision threshold may only be calibrated on validation scores, got split={split!r}")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must lie in (0, 1)")
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels differ in length")
    normal = scores[labels == 0]                 # attack windows are ignored
    if normal.size == 0 or not np.isfinite(normal).all():
        raise ValueError("need a non-empty, finite set of normal validation scores")
    thr = float(np.quantile(normal, 1.0 - target_fpr, method="higher"))
    return {"rule": "score > threshold", "method": "val_normal_fpr", "threshold": thr, "target_fpr": float(target_fpr),
            "realized_val_fpr": float((normal > thr).mean()), "n_val_normal": int(normal.size),
            "fitted_on_split": split}


def predict(scores: np.ndarray, decision: dict) -> np.ndarray:
    if decision.get("rule") != "score > threshold":
        raise ValueError(f"unsupported decision rule {decision.get('rule')!r}")
    return np.asarray(scores) > decision["threshold"]
