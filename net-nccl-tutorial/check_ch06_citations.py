#!/usr/bin/env python3
"""Check the abbreviated ``file.py:LINE`` citations used inside ch06.

ch06 uses short forms (``all2all.py:171``) rather than full repo-relative paths.
This resolves each basename to candidate files under the vLLM checkout and
reports whether *any* candidate has plausible content at that line.

A line is considered plausible when it is non-empty and looks like real code or
a comment (this is a smoke test for "the citation points at something", not a
symbol-exact check -- use verify_citations.py for the exact ones).

Usage: python check_ch06_citations.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

CH06 = Path(__file__).resolve().parent / "ch06-interview-bank.md"


def find_vllm_repo() -> Path | None:
    env = os.environ.get("VLLM_REPO")
    if env and (Path(env) / "vllm").is_dir():
        return Path(env)
    for parent in CH06.parents:
        cand = parent / "vllm"
        if (cand / "vllm/distributed/parallel_state.py").exists():
            return cand
    fallback = Path.home() / "Documents/GitHub/vllm"
    return fallback if (fallback / "vllm").is_dir() else None


def main() -> int:
    repo = find_vllm_repo()
    if repo is None:
        print("ERROR: set VLLM_REPO to the vLLM checkout")
        return 2

    text = CH06.read_text(encoding="utf-8", errors="replace")

    # file.py:LINE  (allow paths and bare basenames)
    pattern = re.compile(r"([A-Za-z0-9_./\-]+\.(?:py|md|cu|cuh|h|sh)):(\d+)")
    hits = sorted({(m.group(1), int(m.group(2))) for m in pattern.finditer(text)})
    print(f"vLLM checkout: {repo}")
    print(f"distinct file:line citations found: {len(hits)}\n")

    # Index every relevant file once.
    index: dict[str, list[Path]] = {}
    for sub in ("vllm", "csrc", "docs", "tests", "benchmarks"):
        root = repo / sub
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in {".py", ".md", ".cu", ".cuh", ".h", ".sh"}:
                index.setdefault(p.name, []).append(p)

    ok = bad = unresolved = 0
    problems: list[str] = []
    for name, line in hits:
        candidates = index.get(Path(name).name)
        if not candidates:
            unresolved += 1
            problems.append(f"NO FILE   {name}:{line}")
            continue

        # Prefer the candidate whose path ends with the cited relative path.
        ranked = sorted(candidates, key=lambda p: (not str(p).endswith(name.replace("/", os.sep)), len(str(p))))
        found = False
        for cand in ranked[:6]:
            try:
                lines = cand.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            if line > len(lines):
                continue
            content = lines[line - 1].strip()
            # Plausible: non-empty, and not a bare closing bracket / blank.
            if content and content not in {")", "]", "}", "):", "],"}:
                found = True
                ok += 1
                break
        if not found:
            bad += 1
            problems.append(f"EMPTY/EOF {name}:{line}")

    print(f"plausible : {ok}")
    print(f"empty/EOF : {bad}")
    print(f"no file   : {unresolved}")
    if problems:
        print("\n--- items to review ---")
        for p in problems[:40]:
            print(" ", p)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
    # Only real failures (empty target) are errors; "no file" is usually a
    # doc/external reference.
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
