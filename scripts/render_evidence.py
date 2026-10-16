"""Render excerpts of real console logs as terminal-style PNG images for the report (lines are selected and
wrapped; no text is edited or dropped).

Usage: python scripts/render_evidence.py <log file> <out.png> <regex selecting lines> [--max N] [--title T]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("pattern")
    p.add_argument("--max", type=int, default=24)
    p.add_argument("--title", default="")
    p.add_argument("--width", type=int, default=118)
    p.add_argument("--step", type=float, default=0.24, help="line spacing in inches")
    p.add_argument("--fig-width", type=float, default=11)
    a = p.parse_args()
    raw = a.log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    rx = re.compile(a.pattern)
    lines = [ln.rstrip() for ln in raw.split("\n") if rx.search(ln)]
    lines = lines[-a.max:]
    import textwrap

    clipped = []
    for ln in lines:  # wrap long lines (no text is dropped)
        clipped += textwrap.wrap(ln, width=a.width, subsequent_indent="    ", drop_whitespace=False) or [""]
    st = a.step
    h = st * (len(clipped) + 2) + 0.2
    fig = plt.figure(figsize=(a.fig_width, h), dpi=220)
    fig.patch.set_facecolor("#1e1e1e")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    y = 1 - 0.12 / h
    if a.title:
        ax.text(0.01, y, a.title, family="monospace", fontsize=8, color="#9cdcfe", va="top", transform=ax.transAxes)
        y -= (st + 0.06) / h
    for ln in clipped:
        color = "#b5cea8" if "RESULT" in ln or "AP)" in ln or "passed" in ln else "#d4d4d4"
        ax.text(0.01, y, ln, family="monospace", fontsize=7, color=color, va="top", transform=ax.transAxes)
        y -= st / h
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=fig.get_facecolor())
    print(f"{a.out}: {len(clipped)} lines from {a.log}")


if __name__ == "__main__":
    main()
