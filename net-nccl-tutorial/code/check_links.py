#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that every relative Markdown link in this tutorial resolves.

Walks the repository root, ``docs/`` and ``code/`` (following the docs/ + code/
split), and reports any ``[label](target)`` whose target does not exist.

    python code/check_links.py            # report
    python code/check_links.py -v         # also list the links that passed

Skips absolute URLs, pure anchors, and targets inside fenced code blocks
(so C++ lambdas like ``[&](int a)`` and Markdown inside ``\u0060\u0060\u0060`` fences are ignored).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent          # .../code
ROOT = HERE.parent                              # 仓库根
SEARCH_DIRS = [ROOT, ROOT / "docs", HERE, HERE / "interview"]

LINK_RX = re.compile(r"\[([^\]]*)\]\(([^)\s]+?)(#[^)]*)?\)")
FENCE_RX = re.compile(r"^\s*(```|~~~)")


def strip_fences(text: str) -> str:
    """把围栏代码块整段替换成等长空白，避免把代码里的 [x](y) 当链接。"""
    out, in_fence = [], False
    for ln in text.splitlines():
        if FENCE_RX.match(ln):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else ln)
    return "\n".join(out)


def main() -> int:
    verbose = "-v" in sys.argv[1:]
    mds: list[Path] = []
    for d in SEARCH_DIRS:
        if d.is_dir():
            mds += sorted(d.glob("*.md"))
    mds = sorted(set(mds))

    total = 0
    bad: list[tuple[Path, int, str]] = []
    for md in mds:
        lines = strip_fences(md.read_text(encoding="utf-8")).splitlines()
        for i, ln in enumerate(lines, 1):
            for m in LINK_RX.finditer(ln):
                target = m.group(2).strip()
                if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
                    continue
                total += 1
                path = target.split("#")[0]
                if not path:
                    continue
                if (md.parent / path).exists():
                    if verbose:
                        print(f"  OK   {md.relative_to(ROOT)}:{i}  -> {target}")
                else:
                    bad.append((md, i, target))

    print(f"检查了 {len(mds)} 个 markdown 文件、{total} 条相对链接")
    if bad:
        print(f"\n{len(bad)} 条断链：")
        for md, ln, t in bad:
            print(f"  {md.relative_to(ROOT)}:{ln}  -> {t}")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
