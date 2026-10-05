"""Per-attack-type evaluation on a small hand-made example."""

import numpy as np
import pandas as pd
import pytest

from src import evaluate as E

GROUPS = {"road": {"max_speedometer": {"group": "coarse"}, "max_speedometer_masquerade": {"group": "stealthy"},
                   "max_engine_coolant_temp": {"group": "coarse"}}}
CFG = {"splits": {"analysis_roles": {"unseen_in_training": ["road/max_engine_coolant_temp"],
                                     "no_validation_windows": [], "excluded_from_primary_as1": []}}}


def _example():
    rows = [
        # capture, source, attack_type, label, pred, score
        ("fab", "src", "max_speedometer", 1, 1, 0.9),
        ("fab", "src", "max_speedometer", 1, 0, 0.4),
        ("fab", "src", "normal", 0, 1, 0.6),
        ("fab", "src", "normal", 0, 0, 0.1),
        ("masq", "src", "max_speedometer_masquerade", 1, 0, 0.3),
        ("masq", "src", "max_speedometer_masquerade", 1, 1, 0.8),
        ("masq", "src", "normal", 0, 0, 0.2),
        ("cool", "cool", "max_engine_coolant_temp", 1, 1, 0.9),
        ("cool", "cool", "normal", 0, 0, 0.1),
        ("amb", "amb", "normal", 0, 1, 0.7),
        ("amb", "amb", "normal", 0, 0, 0.1),
        ("amb", "amb", "normal", 0, 0, 0.2),
    ]
    df = pd.DataFrame(rows, columns=["capture", "source_capture", "attack_type", "label", "pred", "score"])
    return df, df["score"].to_numpy(), df["pred"].to_numpy().astype(bool)


def test_binary_metrics_by_hand():
    m = E.binary_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]), np.array([0.9, 0.4, 0.6, 0.1]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["fpr"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)


def test_per_type_uses_normals_of_same_captures_only():
    df, s, p = _example()
    m = E.evaluate_windows(df, s, p, "road", GROUPS, CFG).set_index(["level", "name"])
    fab = m.loc[("attack_type", "max_speedometer")]
    assert (fab["n_attack"], fab["n_normal"], fab["tp"], fab["fp"]) == (2, 2, 1, 1)
    assert fab["recall"] == 0.5 and fab["group"] == "coarse" and fab["role"] == "primary"
    masq = m.loc[("attack_type", "max_speedometer_masquerade")]
    assert (masq["n_normal"], masq["fp"], masq["recall"]) == (1, 0, 0.5)
    assert masq["precision"] == 1.0


def test_unseen_type_reported_separately_and_out_of_pool():
    df, s, p = _example()
    m = E.evaluate_windows(df, s, p, "road", GROUPS, CFG).set_index(["level", "name"])
    assert m.loc[("attack_type", "max_engine_coolant_temp"), "role"] == "unseen"
    pool = m.loc[("pooled", "primary_pool")]
    assert pool["n_attack"] == 4 and pool["n_normal"] == 6    # coolant capture excluded


def test_attack_free_fpr_and_pairs():
    df, s, p = _example()
    m = E.evaluate_windows(df, s, p, "road", GROUPS, CFG)
    amb = m.set_index(["level", "name"]).loc[("attack_free", "ambient")]
    assert amb["fpr"] == pytest.approx(1 / 3) and amb["n_attack"] == 0
    pairs = E.pairs_table(m, df)
    assert len(pairs) == 1
    r = pairs.iloc[0]
    assert (r["fabrication"], r["masquerade"]) == ("fab", "masq")
    assert r["recall_fabrication"] == 0.5 and r["recall_masquerade"] == 0.5
