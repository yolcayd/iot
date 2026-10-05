"""configs/attack_groups.yaml must cover every attack type and agree with the a-priori rule's output."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
GROUPS = yaml.safe_load((ROOT / "configs" / "attack_groups.yaml").read_text(encoding="utf-8"))
SIGNATURES = ROOT / "results" / "tables" / "attack_signatures.csv"


def test_groups_are_valid():
    for dataset, attacks in GROUPS.items():
        for attack, entry in attacks.items():
            assert entry["group"] in {"coarse", "stealthy"}, (dataset, attack)
            assert entry["plan_group"] in {"coarse", "stealthy"}, (dataset, attack)


@pytest.mark.skipif(not SIGNATURES.exists(), reason="attack_signatures.csv not generated")
def test_groups_match_rule_output():
    sig = pd.read_csv(SIGNATURES)
    expected = {(r.dataset, r.attack_type): (r.proposed_group, r.plan_group) for r in sig.itertuples()}
    actual = {(d, a): (e["group"], e["plan_group"]) for d, attacks in GROUPS.items() for a, e in attacks.items()}
    assert actual == expected
