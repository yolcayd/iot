"""Capture views: the frame ranges that splits and analyses are built from.

A view is a contiguous row range ``[start, stop)`` of one converted capture (parquet file under
data/processed). Normally a view is the whole capture. ``shared_segments`` in configs/data.yaml
declares recordings that a dataset publishes more than once: every Car-Hacking attack file ends
with the same 482 s attack-free recording (identical timestamps, IDs and payloads, see
docs/DECISIONS.md). Such a segment is cut from every file that contains it and exposed once as a
view of its own, so it is counted once and can only ever land in one split.

Usage:
    views = all_views()
    table = read_view(views[0], columns=["timestamp", "label"])
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from src.data import loaders as L

PROJECT_ROOT = L.PROJECT_ROOT
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"


@dataclass(frozen=True)
class View:
    dataset: str
    capture: str          # view id; equals the capture id unless the view is a shared segment
    source_capture: str   # split unit (K-007): derived captures carry the id of their source
    parquet: str          # path relative to the project root
    kind: str             # ambient | attack | targeted | fuzzing | unlabeled
    attack_type: str      # attack type of the injected frames, "normal" if the view has none
    start: int
    stop: int
    excluded: bool

    @property
    def n_frames(self) -> int:
        return self.stop - self.start

    @property
    def is_masquerade(self) -> bool:
        return self.attack_type.endswith("_masquerade")


def load_config(path: Path = DATA_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_index(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or L.PROCESSED_DIR / "index.csv", dtype={"attack_types": str})
    df["attack_types"] = df["attack_types"].fillna("")
    df["excluded"] = df["excluded"].astype(str).str.lower() == "true"
    return df


@lru_cache(maxsize=None)
def _timestamps(parquet: str) -> np.ndarray:
    return L.read_processed(PROJECT_ROOT / parquet, columns=["timestamp"])["timestamp"].to_numpy()


def segment_start_row(parquet: str, start_timestamp: float) -> int:
    """Row index of the frame recorded at ``start_timestamp`` (must match a frame exactly to 1 us)."""
    ts = _timestamps(parquet)
    i = int(np.searchsorted(ts, start_timestamp - 5e-7))
    if i >= len(ts) or abs(ts[i] - start_timestamp) > 5e-7:
        raise ValueError(f"{parquet}: no frame at timestamp {start_timestamp}")
    return i


def all_views(cfg: dict | None = None, index: pd.DataFrame | None = None) -> list[View]:
    """Every capture as a view, with declared shared segments cut out and added once."""
    cfg = cfg or load_config()
    index = read_index() if index is None else index
    by_capture = {r.capture: r for r in index.itertuples(index=False)}
    stops: dict[str, int] = {}
    extra: list[View] = []
    for seg_id, seg in (cfg.get("shared_segments") or {}).items():
        for cap in seg["contained_in"]:
            stops[cap] = segment_start_row(by_capture[cap].parquet, seg["start_timestamp"])
        owner = by_capture[seg["read_from"]]
        start = stops[seg["read_from"]]
        extra.append(View(
            dataset=owner.dataset, capture=seg_id, source_capture=seg_id, parquet=owner.parquet,
            kind="ambient", attack_type=L.NORMAL, start=start, stop=int(owner.n_frames), excluded=False,
        ))
    out = []
    for r in index.itertuples(index=False):
        out.append(View(
            dataset=r.dataset, capture=r.capture, source_capture=r.source_capture, parquet=r.parquet,
            kind=r.kind, attack_type=r.attack_types or L.NORMAL, start=0,
            stop=stops.get(r.capture, int(r.n_frames)), excluded=bool(r.excluded),
        ))
    return out + extra


def read_view(view: View, columns: list[str] | None = None) -> pa.Table:
    t = L.read_processed(PROJECT_ROOT / view.parquet, columns=columns)
    return t.slice(view.start, view.n_frames)
