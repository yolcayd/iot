"""Paper sources (P8, docs/DECISIONS.md K-066): generated numbers up to date, no hand-written numbers in the written
sections, every macro defined, bibliography limited to the proposal's references, figures in sync."""

import re
from pathlib import Path

import pytest
import yaml

from analysis import paper_numbers as PN
from scripts import sync_figures

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
WRITTEN = ("threat_model", "method", "setup", "results", "appendix")   # sections whose text is written (others: skeleton)
PROPOSAL_KEYS = {"hanselmann2020canet", "oualil2026cantempo", "rajapaksha2023odtl", "song2020dcnn", "verma2024road"}

# Name fragments that contain digits but are not numbers taken from the experiments.
ALLOWED = re.compile(r"\b(?:1D|F1|INT8|float16|float32|fp16|fp32|Q[0-2]|C[1-4]|H[1-4]|RQ[1-3])\b")


@pytest.fixture(scope="module")
def generated():
    return PN.build()


def _strip(tex: str) -> str:
    """Section text without comments, label/ref/cite/input/graphics arguments, environments and macro names."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    tex = re.sub(r"\\(?:label|ref|cite|input|includegraphics|begin|end)(?:\[[^\]]*\])?\{[^}]*\}", "", tex)
    tex = re.sub(r"\\[A-Za-z]+", "", tex)
    return ALLOWED.sub("", tex)


def _macros_used(tex: str) -> set[str]:
    return set(re.findall(r"\\([A-Za-z]+)", re.sub(r"(?<!\\)%.*", "", tex)))


def test_generated_files_are_up_to_date(generated):
    for name, text in generated.items():
        assert (PAPER / "generated" / name).read_text(encoding="utf-8") == text, (
            f"paper/generated/{name} is stale: run `python -m analysis.paper_numbers`")


@pytest.mark.parametrize("section", WRITTEN)
def test_written_sections_contain_no_numeric_literals(section):
    text = _strip((PAPER / "sections" / f"{section}.tex").read_text(encoding="utf-8"))
    hits = [m.group(0) for m in re.finditer(r"\S*\d\S*", text)]
    assert not hits, f"{section}.tex: numbers must come from generated/numbers.tex macros: {hits}"


def test_generated_tables_carry_no_decision_ids(generated):
    for name, text in generated.items():
        if name.startswith("table_"):
            assert not re.search(r"K-\d{3}", text), f"paper/generated/{name} contains an internal decision id"


def test_results_claims_stop_the_generator_when_they_fail():
    with pytest.raises(ValueError):
        PN.check(False, "a claim")


def test_strip_catches_a_typed_number():
    assert re.search(r"\d", _strip(r"a window of 100 frames \WinSize{} % 7 in a comment"))
    assert not re.search(r"\d", _strip(r"\WinSize{} INT8 Q2 C3 1D-CNN \ref{tab:x1} % 7"))


def test_every_macro_used_in_the_sections_is_defined(generated):
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", generated["numbers.tex"]))
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (PAPER / "main.tex").read_text(encoding="utf-8")))
    for section in WRITTEN:
        used = _macros_used((PAPER / "sections" / f"{section}.tex").read_text(encoding="utf-8"))
        # our macros are capitalised; the only capitalised LaTeX command in the sections is \Delta
        unknown = {u for u in used if u[0].isupper() and u not in defined | {"Delta"}}
        assert not unknown, f"{section}.tex uses undefined macros: {unknown}"


def _macro(numbers: str, name: str) -> str:
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*)\}%", numbers)
    assert m, name
    return m.group(1)


def test_macro_values_match_their_sources(generated):
    n = generated["numbers.tex"]
    data = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert _macro(n, "WinSize") == str(data["windows"]["size"])
    assert _macro(n, "StrideTrain") == str(data["windows"]["stride"]["train"])
    assert _macro(n, "StrideEval") == str(data["windows"]["stride"]["test"])
    assert _macro(n, "CNNChannels") == ", ".join(map(str, models["models"]["cnn1d"]["channels"]))
    assert _macro(n, "TargetFPR") == f"{100 * models['decision']['target_fpr']:.0f}\\%"
    assert _macro(n, "NStealthy") == str(sum(v["group"] == "stealthy" for ds in yaml.safe_load(
        (ROOT / "configs" / "attack_groups.yaml").read_text(encoding="utf-8")).values() for v in ds.values()))


def test_formatters():
    assert PN.fint(10000) == "10{,}000"
    assert PN.fpct(0.01) == "1\\%"
    assert PN.fpow10(1e-3) == "$10^{-3}$"
    assert PN.fpow10(3e-4) == "$3\\times10^{-4}$"
    assert PN.frange(2.0, 2.0, 2) == "2.00" and PN.frange(1.0, 1.14, 2) == "1.00--1.14"
    with pytest.raises(ValueError):
        PN.same([1, 2], "x")
    with pytest.raises(ValueError):
        PN.Macros().add("Bad1", 1, "src")


def test_bibliography_holds_exactly_the_proposal_references():
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    assert set(re.findall(r"@\w+\{([^,]+),", bib)) == PROPOSAL_KEYS


def test_every_citation_is_in_the_bibliography():
    cited = set()
    for p in [PAPER / "main.tex", *(PAPER / "sections").glob("*.tex")]:
        text = re.sub(r"(?<!\\)%.*", "", p.read_text(encoding="utf-8"))
        for group in re.findall(r"\\cite\{([^}]*)\}", text):
            cited |= {k.strip() for k in group.split(",")}
    assert cited <= PROPOSAL_KEYS


def test_paper_figures_are_in_sync_with_results():
    assert sync_figures.differences() == []


def test_main_includes_every_section_and_the_ai_statement():
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    for s in ("abstract", "introduction", "related_work", "threat_model", "method", "setup", "results",
              "discussion", "limitations", "conclusion"):
        assert f"\\input{{sections/{s}}}" in main
    assert ("An AI assistant (Claude, Anthropic) was used for developing the experiment code and drafting the "
            "manuscript; the experimental design, analyses, interpretation of results and final text are the "
            "authors' responsibility.") in " ".join(main.split())
