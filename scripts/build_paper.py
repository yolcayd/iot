"""Build the paper PDF (P8): sync figures, regenerate the numbers, run pdflatex/bibtex, check the log.

Steps:
    1. scripts/sync_figures.py (results/figures -> paper/figures)
    2. python -m analysis.paper_numbers (paper/generated/)
    3. pdflatex, bibtex, pdflatex, pdflatex in paper/ (output in paper/build/, ignored by git)
    4. log check: undefined references or citations, missing files, overfull boxes; page count

pdflatex is taken from $PDFLATEX, the PATH or a user-level TinyTeX (%APPDATA%/TinyTeX); bibtex from the same
directory. Exit status 1 if a check fails.

Usage:
    python scripts/build_paper.py            # full build
    python scripts/build_paper.py --no-gen   # skip steps 1-2 (only compile)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
sys.path.insert(0, str(ROOT))


def find_tex(tool: str) -> Path:
    env = os.environ.get("PDFLATEX")
    if env:
        cand = Path(env).with_name(Path(env).name.replace("pdflatex", tool))
        if cand.exists():
            return cand
    found = shutil.which(tool)
    if found:
        return Path(found)
    appdata = os.environ.get("APPDATA")
    if appdata:
        for sub in ("windows", "win32", "win64"):
            cand = Path(appdata) / "TinyTeX" / "bin" / sub / f"{tool}.exe"
            if cand.exists():
                return cand
    raise FileNotFoundError(f"{tool} not found: set PDFLATEX, put TeX on the PATH or install TinyTeX")


def run(cmd: list, cwd: Path) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:])
        raise RuntimeError(f"failed: {' '.join(str(c) for c in cmd)}")


CHECKS = {
    "undefined reference": re.compile(r"LaTeX Warning: Reference `[^']+' on page \d+ undefined"),
    "undefined citation": re.compile(r"LaTeX Warning: Citation `[^']+' on page \d+ undefined"),
    "missing file": re.compile(r"(File `[^']+' not found|! LaTeX Error: File)"),
    "multiply defined label": re.compile(r"LaTeX Warning: Label `[^']+' multiply defined"),
}
OVERFULL = re.compile(r"Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)[^\n]*")


def check_log(log: str, overfull_tol_pt: float) -> tuple[list[str], int | None]:
    problems = []
    for name, rx in CHECKS.items():
        problems += [f"{name}: {m.group(0)}" for m in rx.finditer(log)]
    problems += [f"overfull box: {m.group(0)}" for m in OVERFULL.finditer(log) if float(m.group(1)) > overfull_tol_pt]
    pages = re.search(r"Output written on .*?\((\d+) pages?", log)
    return problems, int(pages.group(1)) if pages else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-gen", action="store_true", help="skip figure sync and number generation")
    ap.add_argument("--overfull-tol", type=float, default=1.0, help="ignore overfull boxes up to this width (pt)")
    args = ap.parse_args(argv)
    if not args.no_gen:
        from analysis import paper_numbers
        from scripts import sync_figures

        for line in sync_figures.sync():
            print(f"sync: {line}")
        paper_numbers.write(paper_numbers.build())
    pdflatex, bibtex = find_tex("pdflatex"), find_tex("bibtex")
    BUILD.mkdir(exist_ok=True)
    latex = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"]
    run(latex, PAPER)
    run([bibtex, "build/main"], PAPER)
    run(latex, PAPER)
    run(latex, PAPER)
    log = (BUILD / "main.log").read_text(encoding="latin-1")
    problems, pages = check_log(log, args.overfull_tol)
    for p in problems:
        print(p)
    print(f"pdf: {(BUILD / 'main.pdf').relative_to(ROOT)}  pages: {pages}  problems: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
