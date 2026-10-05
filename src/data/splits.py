"""Source-level train/validation/test split definition, window index and split summary.

The split is read from configs/data.yaml; nothing is drawn at random (docs/DECISIONS.md K-016...).

* ROAD: whole captures, assigned by ``source_capture`` (K-007, K-015c). A masquerade capture always
  follows the fabrication capture it was derived from.
* Car-Hacking: every attack type is a single file, so files are split into temporal blocks (train
  first, then validation, then test) cut inside attack-free stretches, with whole attack bursts
  discarded between blocks. The block row ranges are written out explicitly in the config;
  ``propose_car_hacking_blocks`` re-derives them from the rule so a test can check the two agree.
* The shared attack-free tail of the Car-Hacking attack files is one view (src/data/sources.py)
  and is assigned once.

Usage:
    python -m src.data.splits                   # window index + results/tables/split_summary.csv
    python -m src.data.splits --propose-blocks  # print the Car-Hacking block rule output
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

from src.data import loaders as L
from src.data import sources as S
from src.data import windows as W

SPLITS = ("train", "val", "test")
SUMMARY_PATH = S.PROJECT_ROOT / "results" / "tables" / "split_summary.csv"
WINDOW_INDEX = L.PROCESSED_DIR / "windows" / "windows.parquet"
GROUPS_PATH = S.PROJECT_ROOT / "configs" / "attack_groups.yaml"


@dataclass(frozen=True)
class Segment:
    dataset: str
    capture: str
    source_capture: str
    parquet: str
    attack_type: str      # attack type of the capture view ("normal" if attack-free)
    split: str
    start: int            # absolute row range in the parquet file
    stop: int

    @property
    def id(self) -> str:
        return f"{self.capture}[{self.start}:{self.stop}]"

    @property
    def n_frames(self) -> int:
        return self.stop - self.start

    @property
    def is_masquerade(self) -> bool:
        return self.attack_type.endswith("_masquerade")

    def read(self, columns: list[str] | None = None) -> pa.Table:
        return L.read_processed(S.PROJECT_ROOT / self.parquet, columns=columns).slice(self.start, self.n_frames)


def _segment(view: S.View, split: str, start: int, stop: int) -> Segment:
    if split not in SPLITS:
        raise ValueError(f"{view.capture}: unknown split {split!r}")
    if not (view.start <= start < stop <= view.stop):
        raise ValueError(f"{view.capture}: block [{start}, {stop}) outside view [{view.start}, {view.stop})")
    return Segment(view.dataset, view.capture, view.source_capture, view.parquet, view.attack_type, split, start, stop)


def segments(cfg: dict | None = None, views: list[S.View] | None = None) -> list[Segment]:
    """All split segments; raises if a usable view is unassigned or an excluded one is assigned."""
    cfg = cfg or S.load_config()
    views = views or S.all_views(cfg)
    sc = cfg["splits"]
    ch_whole = sc["car_hacking"].get("whole") or {}
    ch_blocks = sc["car_hacking"].get("blocks") or {}
    road_split: dict[str, str] = {}
    for split in SPLITS:
        for src in sc["road"][split]:
            if src in road_split:
                raise ValueError(f"{src} listed in more than one ROAD split")
            road_split[src] = split
    known = {v.capture for v in views} | {v.source_capture for v in views}
    for name in list(ch_whole) + list(ch_blocks) + list(road_split):
        if name not in known:
            raise ValueError(f"split config names unknown capture {name}")

    out: list[Segment] = []
    for v in views:
        listed = v.capture in ch_whole or v.capture in ch_blocks or v.source_capture in road_split
        if v.excluded:
            if listed:
                raise ValueError(f"excluded capture {v.capture} is assigned to a split")
            continue
        if v.dataset == "car_hacking":
            if v.capture in ch_whole:
                out.append(_segment(v, ch_whole[v.capture], v.start, v.stop))
            elif v.capture in ch_blocks:
                for split, ranges in ch_blocks[v.capture].items():
                    out += [_segment(v, split, int(a), int(b)) for a, b in ranges]
            else:
                raise ValueError(f"{v.capture} is not assigned to a split")
        else:
            if v.source_capture not in road_split:
                raise ValueError(f"{v.capture} (source {v.source_capture}) is not assigned to a split")
            out.append(_segment(v, road_split[v.source_capture], v.start, v.stop))
    return out


# --------------------------------------------------------------------------------------------
# Car-Hacking temporal block rule
# --------------------------------------------------------------------------------------------


def attack_bursts(ts: np.ndarray, label: np.ndarray, burst_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(first, last) row of each run of injected frames separated by less than ``burst_gap_s``."""
    ai = np.where(label == 1)[0]
    br = np.where(np.diff(ts[ai]) > burst_gap_s)[0]
    return np.r_[ai[0], ai[br + 1]], np.r_[ai[br], ai[-1]]


def propose_car_hacking_blocks(cfg: dict | None = None, views: list[S.View] | None = None) -> dict:
    """Re-derive the Car-Hacking block row ranges from the rule in ``splits.temporal_block_rule``.

    Attack file (bursts numbered 1..B): train = bursts 1..T, ``gap_bursts`` bursts discarded,
    validation = the next V bursts, ``gap_bursts`` discarded, test = the remaining bursts. Every cut
    lies at the time midpoint of the attack-free interval between two bursts. Attack-free file:
    time fractions ``normal_fractions`` with ``normal_gap_s`` seconds discarded at each cut.
    """
    cfg = cfg or S.load_config()
    rule = cfg["splits"]["temporal_block_rule"]
    views = views or S.all_views(cfg)
    out = {}
    for v in views:
        if v.dataset != "car_hacking" or v.capture in (cfg["splits"]["car_hacking"].get("whole") or {}):
            continue
        t = S.read_view(v, ["timestamp", "label"])
        ts, lab = t["timestamp"].to_numpy(), t["label"].to_numpy()
        if lab.any():
            bs, be = attack_bursts(ts, lab, rule["burst_gap_s"])

            def cut_after(k: int) -> int:  # k = 1-based burst number
                return int(np.searchsorted(ts, (ts[be[k - 1]] + ts[bs[k]]) / 2))

            tr, va, g = rule["train_bursts"], rule["val_bursts"], rule["gap_bursts"]
            if tr + va + 2 * g >= len(bs):
                raise ValueError(f"{v.capture}: only {len(bs)} bursts")
            a, b = cut_after(tr), cut_after(tr + g)
            c, d = cut_after(tr + g + va), cut_after(tr + 2 * g + va)
        else:
            f_tr, f_va, _ = rule["normal_fractions"]
            gap = rule["normal_gap_s"]
            t0, dur = ts[0], ts[-1] - ts[0]
            a = int(np.searchsorted(ts, t0 + f_tr * dur))
            b = int(np.searchsorted(ts, t0 + f_tr * dur + gap))
            c = int(np.searchsorted(ts, t0 + (f_tr + f_va) * dur))
            d = int(np.searchsorted(ts, t0 + (f_tr + f_va) * dur + gap))
        s = v.start
        out[v.capture] = {"train": [[s, s + a]], "val": [[s + b, s + c]], "test": [[s + d, v.stop]]}
    return out


# --------------------------------------------------------------------------------------------
# Windows and summary
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def injected_timestamps(parquet: str) -> np.ndarray:
    t = L.read_processed(S.PROJECT_ROOT / parquet, columns=["timestamp", "label"])
    return t["timestamp"].to_numpy()[t["label"].to_numpy() == 1]


def segment_windows(seg: Segment, cfg: dict) -> pd.DataFrame:
    size = cfg["windows"]["size"]
    stride = cfg["windows"]["stride"][seg.split]
    t = seg.read(["timestamp", "label", "attack_type"])
    ts, lab = t["timestamp"].to_numpy(), t["label"].to_numpy()
    cat = pd.Categorical(t["attack_type"].to_pandas())
    starts = W.window_starts(len(ts), size, stride)
    labels = W.window_labels(lab, starts, size)
    types = W.dominant_attack_type(np.asarray(cat.codes), lab, starts, size, list(cat.categories))
    # margin is measured against every injected frame of the capture, also those outside the segment
    clean = W.margin_clean(ts, starts, size, injected_timestamps(seg.parquet), cfg["normal_subset"]["margin_s"])
    normal_ok = (seg.split == "train") & (labels == 0) & clean
    if cfg["normal_subset"]["exclude_masquerade"] and seg.is_masquerade:
        normal_ok[:] = False
    return pd.DataFrame({
        "dataset": seg.dataset, "split": seg.split, "capture": seg.capture, "source_capture": seg.source_capture,
        "segment": seg.id, "start": seg.start + starts, "label": labels, "attack_type": types,
        "normal_subset": normal_ok,
    })


def build_windows(segs: list[Segment], cfg: dict) -> pd.DataFrame:
    df = pd.concat([segment_windows(s, cfg) for s in segs], ignore_index=True)
    for col in ("dataset", "split", "capture", "source_capture", "segment", "attack_type"):
        df[col] = df[col].astype("category")
    return df


def attack_group(dataset: str, attack_type: str, groups: dict) -> str:
    return "" if attack_type == W.NORMAL else groups[dataset][attack_type]["group"]


def summarize(windows: pd.DataFrame, segs: list[Segment], cfg: dict) -> pd.DataFrame:
    """Window counts per dataset x split x (window-level dominant) attack type."""
    groups = yaml.safe_load(GROUPS_PATH.read_text(encoding="utf-8"))
    g = windows.groupby(["dataset", "split", "attack_type"], observed=True)
    out = g.agg(n_windows=("start", "size"), n_captures=("capture", "nunique"),
                n_normal_subset_windows=("normal_subset", "sum")).reset_index()
    inj = {}
    for s in segs:
        if s.attack_type != W.NORMAL:
            n = int(s.read(["label"])["label"].to_numpy().sum())
            key = (s.dataset, s.split, s.attack_type)
            inj[key] = inj.get(key, 0) + n
    out["n_injected_frames"] = [inj.get((r.dataset, r.split, r.attack_type), 0) for r in out.itertuples()]
    out["group"] = [attack_group(r.dataset, r.attack_type, groups) for r in out.itertuples()]
    out["stride"] = out["split"].map(cfg["windows"]["stride"])
    order = {s: i for i, s in enumerate(SPLITS)}
    out = out.sort_values(["dataset", "split", "attack_type"], key=lambda c: c.map(order) if c.name == "split" else c)
    return out[["dataset", "split", "attack_type", "group", "n_windows", "n_captures", "n_injected_frames",
                "n_normal_subset_windows", "stride"]].reset_index(drop=True)


def check_test_coverage(summary: pd.DataFrame, cfg: dict) -> list[str]:
    """Attack types (from attack_groups.yaml) below the minimum number of test windows."""
    groups = yaml.safe_load(GROUPS_PATH.read_text(encoding="utf-8"))
    minimum = cfg["splits"]["min_test_attack_windows"]
    test = summary[summary.split == "test"].set_index(["dataset", "attack_type"])["n_windows"]
    return [f"{d}/{a}: {int(test.get((d, a), 0))} < {minimum}"
            for d, types in groups.items() for a in types if test.get((d, a), 0) < minimum]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--propose-blocks", action="store_true", help="print the Car-Hacking block rule output")
    args = ap.parse_args(argv)
    cfg = S.load_config()
    if args.propose_blocks:
        print(yaml.safe_dump(propose_car_hacking_blocks(cfg), default_flow_style=None, sort_keys=True))
        return 0
    segs = segments(cfg)
    win = build_windows(segs, cfg)
    WINDOW_INDEX.parent.mkdir(parents=True, exist_ok=True)
    win.to_parquet(WINDOW_INDEX, index=False)
    summary = summarize(win, segs, cfg)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {SUMMARY_PATH.relative_to(S.PROJECT_ROOT)} and {WINDOW_INDEX.relative_to(S.PROJECT_ROOT)} "
          f"({len(win)} windows, {len(segs)} segments)")
    missing = check_test_coverage(summary, cfg)
    if missing:
        print("attack types below min_test_attack_windows:\n  " + "\n  ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
