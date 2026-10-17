"""Pre-processing checks requested by the advisor (P10, docs/DECISIONS.md K-072). Read-only; no model output.

* DLC profile: share of frames with dlc < 8 per dataset and split, and per ID in the BENIGN TRAINING frames whether
  the DLC is constant. Payloads with dlc < 8 are zero-padded by the loader; if an ID always has the same DLC the
  padding carries no ambiguity (the ID implies the payload length), otherwise a padded zero is indistinguishable from
  a real zero byte and the row is flagged ``padding_ambiguous`` (information-loss candidate; no DLC feature is added
  in P10).
* Normalisation provenance: the segments the 10-feature parameters and the extended id_rate reference were fitted
  on, compared with the training segments of configs/data.yaml.

Outputs: results/tables/ext_dlc_profile.csv, ext_dlc_by_id.csv, ext_normalization_provenance.csv.

Usage:
    python -m analysis.ext_preproc_checks
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.data import features as F
from src.data import features_ext as FX
from src.data import sources as S
from src.data import splits as SP

TABLES = S.PROJECT_ROOT / "results" / "tables"


def dlc_tables(segs: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    prof, by_id = [], []
    for seg in segs:
        t = seg.read(["can_id", "dlc", "label"])
        cid, dlc, lab = t["can_id"].to_numpy(), t["dlc"].to_numpy(), t["label"].to_numpy()
        prof.append({"dataset": seg.dataset, "split": seg.split, "segment": seg.id, "n_frames": len(dlc),
                     "n_dlc_lt8": int((dlc < 8).sum()), "n_dlc_lt8_benign": int(((dlc < 8) & (lab == 0)).sum()),
                     "n_dlc_lt8_injected": int(((dlc < 8) & (lab == 1)).sum())})
        if seg.split == "train":
            b = lab == 0
            by_id.append(pd.DataFrame({"dataset": seg.dataset, "can_id": cid[b], "dlc": dlc[b]}))
    prof = pd.DataFrame(prof)
    summary = prof.groupby(["dataset", "split"], as_index=False)[
        ["n_frames", "n_dlc_lt8", "n_dlc_lt8_benign", "n_dlc_lt8_injected"]].sum()
    summary["share_dlc_lt8"] = summary["n_dlc_lt8"] / summary["n_frames"]
    ids = pd.concat(by_id, ignore_index=True)
    g = ids.groupby(["dataset", "can_id"])["dlc"]
    per_id = g.agg(n_frames="size", dlc_min="min", dlc_max="max", n_distinct_dlc="nunique").reset_index()
    per_id["padding_ambiguous"] = (per_id["n_distinct_dlc"] > 1) & (per_id["dlc_min"] < 8)
    per_id["can_id_hex"] = per_id["can_id"].map(lambda v: f"0x{v:03X}")
    return summary, per_id


def provenance(segs: list) -> pd.DataFrame:
    base, ext = F.load_params(), FX.load_ext_params()
    rows = []
    for ds in ("car_hacking", "road"):
        train = sorted(s.id for s in segs if s.dataset == ds and s.split == "train")
        train_no_masq = sorted(s.id for s in segs if s.dataset == ds and s.split == "train" and not s.is_masquerade)
        for name, fitted, expected in (
                ("id_vocab_and_iat (K-024, K-025)", base["datasets"][ds]["fitted_on_segments"], train),
                ("id_rate reference (K-073)", ext["datasets"][ds]["fitted_on_segments"], train_no_masq)):
            fitted = sorted(fitted)
            split_of = {s.id: s.split for s in segs}
            rows.append({"dataset": ds, "parameters": name, "n_fitted_segments": len(fitted),
                         "n_expected_segments": len(expected), "equal_to_expected": fitted == expected,
                         "splits_of_fitted_segments": ";".join(sorted({split_of[s] for s in fitted}))})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    segs = SP.segments()
    summary, per_id = dlc_tables(segs)
    prov = provenance(segs)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "ext_dlc_profile.csv", index=False, float_format="%.6g")
    per_id.to_csv(TABLES / "ext_dlc_by_id.csv", index=False)
    prov.to_csv(TABLES / "ext_normalization_provenance.csv", index=False)
    print(summary.to_string(index=False))
    amb = per_id[per_id["padding_ambiguous"]]
    print(f"\nIDs with dlc < 8 in benign training frames: "
          f"{per_id[per_id['dlc_min'] < 8].groupby('dataset').size().to_dict()}; "
          f"with varying DLC (padding ambiguous): {amb.groupby('dataset').size().to_dict()}")
    if len(amb):
        print(amb.to_string(index=False))
    print()
    print(prov.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
