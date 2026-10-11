"""Copy the paper figures from results/figures/ to paper/figures/ (P8).

Only the numbered paper figures (fig<N>_*.pdf, produced by `python -m analysis.figures`) are copied; the
exploratory signature figures of P1 stay in results/figures/. The copies are committed so that paper/ builds
on its own (e.g. on Overleaf).

Usage:
    python scripts/sync_figures.py           # copy new or changed figures (stale copies are only reported)
    python scripts/sync_figures.py --check   # exit 1 if paper/figures/ differs from results/figures/
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "figures"
DST = ROOT / "paper" / "figures"
PATTERN = "fig[0-9]*_*.pdf"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def differences(src: Path = SRC, dst: Path = DST) -> list[str]:
    """Human-readable list of figures that are missing, changed or stale in ``dst``."""
    want = {p.name: p for p in sorted(src.glob(PATTERN))}
    have = {p.name: p for p in sorted(dst.glob(PATTERN))} if dst.exists() else {}
    out = [f"missing: {n}" for n in want if n not in have]
    out += [f"changed: {n}" for n in want if n in have and _sha(want[n]) != _sha(have[n])]
    out += [f"stale: {n}" for n in have if n not in want]
    return out


def sync(src: Path = SRC, dst: Path = DST) -> list[str]:
    if not any(src.glob(PATTERN)):
        raise FileNotFoundError(f"no {PATTERN} in {src}; run `python -m analysis.figures` first")
    dst.mkdir(parents=True, exist_ok=True)
    done = []
    for line in differences(src, dst):
        kind, name = line.split(": ", 1)
        if kind == "stale":           # never deleted automatically (CLAUDE.md: deletions need approval)
            done.append(f"{line} (not in results/figures; remove it by hand if it is obsolete)")
            continue
        shutil.copy2(src / name, dst / name)
        done.append(line)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report differences and exit 1 if there are any")
    args = ap.parse_args(argv)
    if args.check:
        diff = differences()
        for line in diff:
            print(line)
        return 1 if diff else 0
    for line in sync():
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
