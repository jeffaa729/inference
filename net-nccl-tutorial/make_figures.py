#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the quantitative SVG charts used by the tutorial.

Mermaid is great for structure but bad for curves, so the alpha-beta plot is
generated here as a standalone SVG (no matplotlib dependency, works offline and
diffs cleanly in git).

    python make_figures.py            # writes figures/*.svg
    python make_figures.py --list     # list what would be generated

The chart models an 8-rank ring all-reduce:

    T(S) = 2(N-1)*alpha + 2(N-1)/N * S / BW

and plots the *effective* per-rank bandwidth ``(2(N-1)/N * S) / T(S)``, which is
the number the tutorial's ``effective_gbps`` column reports. Three curves are
drawn so the reader can see that the knee moves with alpha -- which is exactly
why NVLink (sub-microsecond) and IB (a few microseconds) behave so differently
for decode-sized messages.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"


def _force_utf8() -> None:
    """Keep CJK output safe on consoles defaulting to a legacy codepage."""
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8()

# (label, alpha in seconds, link bandwidth in bytes/s, colour, dash)
CURVES = [
    ("NVLink 4 (alpha=0.5us, 450GB/s)", 0.5e-6, 450e9, "#2d7a3e", None),
    ("IB NDR (alpha=5us, 50GB/s)", 5.0e-6, 50e9, "#b8860b", None),
    ("Same NVLink, alpha=3us (toy)", 3.0e-6, 450e9, "#8a8a8a", "6,4"),
]

WORLD = 8
SIZES = [2 ** e for e in range(10, 31)]  # 1 KiB .. 1 GiB


def time_s(size: int, alpha: float, bw: float, n: int = WORLD) -> float:
    steps = 2 * (n - 1)
    moved = 2 * (n - 1) / n * size
    return steps * alpha + moved / bw


def eff_gbps(size: int, alpha: float, bw: float, n: int = WORLD) -> float:
    moved = 2 * (n - 1) / n * size
    return moved / time_s(size, alpha, bw, n) / 1e9


# ---------------------------------------------------------------- tiny SVG kit

W, H = 900, 520
ML, MR, MT, MB = 78, 190, 46, 62
PW, PH = W - ML - MR, H - MT - MB


def x_of(log2_size: float) -> float:
    lo, hi = math.log2(SIZES[0]), math.log2(SIZES[-1])
    return ML + (log2_size - lo) / (hi - lo) * PW


def y_of(gbps: float) -> float:
    lo, hi = math.log10(1.0), math.log10(600.0)  # 1 GB/s .. 600 GB/s
    v = max(lo, min(hi, math.log10(max(gbps, 1e-3))))
    return MT + PH - (v - lo) / (hi - lo) * PH


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_svg() -> str:
    p: list[str] = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">'
    )
    p.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # Title
    p.append(f'<text x="{ML}" y="24" font-size="16" font-weight="bold" fill="#111">'
             f'all-reduce 的有效带宽 vs 消息大小（ring 模型，N={WORLD}）</text>')
    p.append(f'<text x="{ML}" y="41" font-size="12" fill="#555">'
             f'有效带宽 = 每 rank 搬运量 / 耗时；搬运量按 ring 的 2(N-1)/N·S 计</text>')

    # Y grid + labels
    gbps_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    for g in gbps_ticks:
        y = y_of(g)
        p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" '
                 f'stroke="#e6e6e6" stroke-width="1"/>')
        p.append(f'<text x="{ML-10}" y="{y+4:.1f}" font-size="11" fill="#444" '
                 f'text-anchor="end">{g} GB/s</text>')

    # X grid + labels
    for e in range(10, 31, 2):
        x = x_of(e)
        p.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+PH}" '
                 f'stroke="#f0f0f0" stroke-width="1"/>')
        label = f"{2**(e-10)} MiB" if e >= 20 else f"{2**(e-10)*1024:.0f} KiB"
        if e >= 30:
            label = "1 GiB"
        p.append(f'<text x="{x:.1f}" y="{MT+PH+18}" font-size="11" fill="#444" '
                 f'text-anchor="middle">{label}</text>')
    p.append(f'<text x="{ML+PW/2:.0f}" y="{MT+PH+44}" font-size="12" fill="#333" '
             f'text-anchor="middle">消息大小 S（对数轴）</text>')

    # Axes
    p.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#333" stroke-width="1.5"/>')
    p.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#333" stroke-width="1.5"/>')

    # Reference lines: the two sizes the tutorial keeps quoting.
    for size, label in ((16 * 1024, "decode 小消息 16 KiB"),
                        (256 * 1024, "custom AR 上限 256 KiB"),
                        (64 * 1024 * 1024, "prefill 大消息 64 MiB")):
        x = x_of(math.log2(size))
        p.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+PH}" '
                 f'stroke="#c0392b" stroke-width="1" stroke-dasharray="3,4" opacity="0.55"/>')
        p.append(f'<text x="{x+4:.1f}" y="{MT+13}" font-size="10" fill="#c0392b" '
                 f'transform="rotate(0)">{esc(label)}</text>')

    # Curves
    for label, alpha, bw, colour, dash in CURVES:
        pts = []
        for s in SIZES:
            pts.append(f"{x_of(math.log2(s)):.1f},{y_of(eff_gbps(s, alpha, bw)):.1f}")
        dashattr = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colour}" '
                 f'stroke-width="2.5"{dashattr}/>')

    # Legend
    ly = MT + 14
    for label, alpha, bw, colour, dash in CURVES:
        dashattr = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<line x1="{ML+PW+16}" y1="{ly}" x2="{ML+PW+44}" y2="{ly}" '
                 f'stroke="{colour}" stroke-width="2.5"{dashattr}/>')
        p.append(f'<text x="{ML+PW+50}" y="{ly+4}" font-size="10.5" fill="#333">'
                 f'{esc(label)}</text>')
        ly += 22

    # Annotations pointing at the knee.
    knee = 16 * 1024
    kv = eff_gbps(knee, 0.5e-6, 450e9)
    p.append(f'<circle cx="{x_of(math.log2(knee)):.1f}" cy="{y_of(kv):.1f}" r="4" '
             f'fill="none" stroke="#2d7a3e" stroke-width="2"/>')
    p.append(f'<text x="{ML+18}" y="{MT+PH-46}" font-size="11.5" fill="#2d7a3e">'
             f'小消息区：NVLink 上 16 KiB 只剩约 {kv:.2f} GB/s</text>')
    p.append(f'<text x="{ML+18}" y="{MT+PH-30}" font-size="11.5" fill="#2d7a3e">'
             f'（理论线速 450 GB/s 的 {kv/450*100:.2f}%）—— 延迟项 alpha 完全主导</text>')
    p.append(f'<text x="{ML+18}" y="{MT+PH-12}" font-size="11.5" fill="#b8860b">'
             f'大消息区：曲线趋于平线，此时才轮到带宽说话</text>')

    p.append("</svg>")
    return "\n".join(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("figures/alpha-beta-bandwidth.svg")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    path = OUT / "alpha-beta-bandwidth.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size // 1024} KiB)")

    # Also print the key numbers so the tutorial text can quote them.
    print("\n关键数值（用于正文）：")
    for size in (16 * 1024, 256 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024):
        for label, alpha, bw, _, _ in CURVES[:2]:
            t = time_s(size, alpha, bw) * 1e6
            e = eff_gbps(size, alpha, bw)
            print(f"  {size/1024:>9.0f} KiB  {label:<34} "
                  f"T={t:>9.1f} us  有效带宽={e:>7.2f} GB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
