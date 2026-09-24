#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract every ```mermaid block from the tutorial and render it with mermaid-cli.

Catches diagram syntax errors that would otherwise show up as a broken box on
GitHub. Skips gracefully (exit 0 with a warning) when mmdc is unavailable, so
it never blocks a plain `python check_diagrams.py` on a machine without node.

Setup (one time)::

    npm install -g @mermaid-js/mermaid-cli
    # or point at a local install:
    set MERMAID_BIN=C:\\path\\to\\node_modules\\.bin\\mmdc.cmd

Usage::

    python check_diagrams.py              # validate all mermaid blocks
    python check_diagrams.py --list       # just list them
    python check_diagrams.py --render out # also write SVGs into out/
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

FENCE = re.compile(r"^```mermaid\s*$")
END = re.compile(r"^```\s*$")


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8()


def find_mmdc() -> str | None:
    env = os.environ.get("MERMAID_BIN")
    if env and Path(env).exists():
        return env
    for name in ("mmdc", "mmdc.cmd"):
        found = shutil.which(name)
        if found:
            return found
    # common local-install locations
    for cand in (
        HERE / "node_modules/.bin/mmdc.cmd",
        HERE / "node_modules/.bin/mmdc",
        Path(tempfile.gettempdir()) / "mmd-validate/node_modules/.bin/mmdc.cmd",
        Path(tempfile.gettempdir()) / "mmd-validate/node_modules/.bin/mmdc",
        Path.home() / "AppData/Roaming/npm/mmdc.cmd",
    ):
        if cand.exists():
            return str(cand)
    return None


def blocks(md: Path) -> list[tuple[int, str]]:
    """Return (start_line, mermaid_source) for each mermaid block in a file."""
    lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if FENCE.match(lines[i]):
            start = i + 1  # 1-based line of the fence
            j = i + 1
            while j < len(lines) and not END.match(lines[j]):
                j += 1
            out.append((start, "\n".join(lines[i + 1 : j])))
            i = j + 1
        else:
            i += 1
    return out


def check_fences(md: Path) -> list[str]:
    """Report unmatched ``` fences.

    A broken fence is the most common way to silently wreck a markdown file:
    everything after it renders as code. Counting alone is not enough -- a stray
    ```` ```bash ```` inside a fence keeps the count even while breaking the
    document -- so we also flag two openers in a row.
    """
    problems: list[str] = []
    lines = md.read_text(encoding="utf-8", errors="replace").splitlines()
    fences = [(i, ln.strip()) for i, ln in enumerate(lines, 1) if ln.startswith("```")]

    for k in range(0, len(fences) - 1, 2):
        opener, closer = fences[k], fences[k + 1]
        if closer[1] != "```":
            problems.append(
                f"line {opener[0]} opens {opener[1]!r} but the next fence "
                f"(line {closer[0]}) is {closer[1]!r} -- looks like two openers in a row")
    if len(fences) % 2 == 1:
        problems.append(f"line {fences[-1][0]} opens {fences[-1][1]!r} and is never closed")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", type=Path, default=None,
                    help="also write rendered SVGs here")
    ap.add_argument("--list", action="store_true", help="only list blocks")
    args = ap.parse_args()

    # 文档现在都在 docs/；HERE 仍是 code/（mermaid-cli 的 node_modules 找这里）
    mds = sorted(p for p in (HERE.parent / "docs").glob("*.md"))
    mds += sorted(p for p in HERE.parent.glob("*.md"))

    # Fence balance first: a broken fence makes everything downstream render as
    # code, so it is the highest-severity markdown problem we can detect here.
    fence_problems = 0
    for md in mds:
        for prob in check_fences(md):
            print(f"FENCE {md.name}: {prob}")
            fence_problems += 1
    if fence_problems:
        print(f"\n{fence_problems} unbalanced fence(s) -- fix these first\n")
    else:
        print(f"fences: all {len(mds)} markdown files balanced")

    found: list[tuple[Path, int, str]] = []
    for md in mds:
        for start, src in blocks(md):
            found.append((md, start, src))
    total = len(found)

    print(f"{len(mds)} markdown files, {total} mermaid blocks\n")
    if args.list or total == 0:
        for md, start, src in found:
            kind = src.strip().split(maxsplit=1)[0] if src.strip() else "?"
            print(f"  {md.name}:{start}  {kind}")
        return 1 if fence_problems else 0

    mmdc = find_mmdc()
    if mmdc is None:
        print("WARNING: mermaid-cli (mmdc) not found -- cannot validate rendering.")
        print("         install:  npm install -g @mermaid-js/mermaid-cli")
        print("         or set:   MERMAID_BIN=<path to mmdc>")
        return 1 if fence_problems else 0

    print(f"using: {mmdc}\n")
    if args.render:
        args.render.mkdir(parents=True, exist_ok=True)

    bad = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for idx, (md, start, src) in enumerate(found, 1):
            infile = tmp / f"d{idx}.mmd"
            infile.write_text(src, encoding="utf-8")
            outfile = tmp / f"d{idx}.svg"
            proc = subprocess.run(
                [mmdc, "-i", str(infile), "-o", str(outfile), "--quiet"],
                capture_output=True, text=True, timeout=180,
            )
            if proc.returncode != 0 or not outfile.exists():
                bad += 1
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                detail = " | ".join(err[-3:])[:300] if err else "unknown error"
                print(f"  FAIL {md.name}:{start}")
                print(f"       {detail}")
            else:
                size = outfile.stat().st_size
                print(f"  OK   {md.name}:{start}  ({size // 1024} KiB)")
                if args.render:
                    dest = args.render / f"{md.stem}-{start}.svg"
                    shutil.copyfile(outfile, dest)

    print(f"\n{total - bad}/{total} diagrams rendered OK")
    return 1 if (bad or fence_problems) else 0


if __name__ == "__main__":
    sys.exit(main())
