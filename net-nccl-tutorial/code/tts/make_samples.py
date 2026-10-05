#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a small voice-comparison sample set (no full run)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interview_parser import DEFAULT_MD, duration_estimate, parse   # noqa: E402
from build_audiobook import VOICES                                   # noqa: E402

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "audio" / "samples"

# 同一个问题换 4 个音色（方便比较），再用默认音色试 2 个「难念」的题
PLAN = [
    ("voice_yunjian_Q132.mp3", "yunjian", 132, "男声 · 解说腔（默认）"),
    ("voice_xiaoxiao_Q132.mp3", "xiaoxiao", 132, "女声 · 自然"),
    ("voice_yunxi_Q132.mp3", "yunxi", 132, "男声 · 年轻"),
    ("voice_yunyang_Q132.mp3", "yunyang", 132, "男声 · 新闻播报"),
    ("content_Q48_formula.mp3", "yunjian", 48, "符号/公式最多的一题"),
    ("content_Q262_longest.mp3", "yunjian", 262, "最长的一题（约 5 分钟）"),
]


async def main() -> int:
    import edge_tts

    qas = {q.num: q for q in parse(DEFAULT_MD)}
    SAMPLES.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{SAMPLES}\n")
    total = 0
    for fname, vkey, qnum, note in PLAN:
        q = qas[qnum]
        text = q.speech()
        voice = VOICES[vkey]
        dest = SAMPLES / fname
        secs = duration_estimate(text)
        total += secs
        print(f"  {fname:<28} {voice:<34} 约 {secs:>5.0f}s  {note}")
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, voice, rate="-4%").save(str(dest))
                break
            except Exception as exc:
                print(f"      retry {attempt+1}: {type(exc).__name__}: {exc}")
                await asyncio.sleep(3 + 3 * attempt)
        else:
            print("      FAILED")
            continue
        if dest.exists():
            print(f"      -> {dest.stat().st_size//1024} KB")
        await asyncio.sleep(1.0)
    print(f"\n共 {len(PLAN)} 个样音，约 {total/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
