#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sanity-check the hand-generated SVG figures in figures/.

SVG is XML, so "well-formed" is a real correctness test: a malformed figure
renders as nothing on GitHub. On top of that we check that no element's
coordinates fall outside the declared canvas (which is how labels silently get
clipped) and that the expected structural pieces are present.

    python check_figures.py
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8()


def tag_of(el) -> str:
    return el.tag.split("}")[-1]


def check_one(path: Path) -> tuple[bool, list[str]]:
    notes: list[str] = []
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        return False, [f"not well-formed XML: {exc}"]

    if tag_of(root) != "svg":
        return False, [f"root element is <{tag_of(root)}>, expected <svg>"]

    try:
        w = float(root.get("width"))
        h = float(root.get("height"))
    except (TypeError, ValueError):
        return False, ["missing/invalid width/height"]

    notes.append(f"canvas {w:.0f}x{h:.0f}")

    oob: list[str] = []
    counts = {"polyline": 0, "text": 0, "line": 0, "circle": 0, "rect": 0}
    for el in root.iter():
        t = tag_of(el)
        if t in counts:
            counts[t] += 1
        try:
            if t == "polyline":
                for pair in el.get("points", "").split():
                    x, y = (float(v) for v in pair.split(","))
                    if not (0 <= x <= w and 0 <= y <= h):
                        oob.append(f"polyline point ({x:.0f},{y:.0f})")
            elif t == "circle":
                x, y = float(el.get("cx")), float(el.get("cy"))
                if not (0 <= x <= w and 0 <= y <= h):
                    oob.append(f"circle ({x:.0f},{y:.0f})")
            elif t in ("text", "rect"):
                x, y = float(el.get("x", 0)), float(el.get("y", 0))
                if not (0 <= x <= w and 0 <= y <= h):
                    label = (el.text or "").strip()[:28]
                    oob.append(f"{t} ({x:.0f},{y:.0f}) {label!r}")
            elif t == "line":
                for key in (("x1", "y1"), ("x2", "y2")):
                    x, y = float(el.get(key[0])), float(el.get(key[1]))
                    if not (0 <= x <= w and 0 <= y <= h):
                        oob.append(f"line ({x:.0f},{y:.0f})")
        except (TypeError, ValueError):
            oob.append(f"{t} has non-numeric coordinates")

    notes.append(" ".join(f"{k}={v}" for k, v in counts.items() if v))
    if oob:
        return False, notes + [f"{len(oob)} element(s) outside canvas"] + oob[:8]
    notes.append("all coordinates inside canvas")
    return True, notes


def main() -> int:
    if not FIGDIR.is_dir():
        print(f"no {FIGDIR.name}/ directory yet -- run: python make_figures.py")
        return 0

    svgs = sorted(FIGDIR.glob("*.svg"))
    if not svgs:
        print(f"{FIGDIR.name}/ has no SVG files -- run: python make_figures.py")
        return 0

    bad = 0
    for svg in svgs:
        ok, notes = check_one(svg)
        mark = "OK  " if ok else "FAIL"
        print(f"{mark} {svg.name}  ({svg.stat().st_size // 1024} KiB)")
        for n in notes:
            print(f"       {n}")
        if not ok:
            bad += 1

    print(f"\n{len(svgs) - bad}/{len(svgs)} figures OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
