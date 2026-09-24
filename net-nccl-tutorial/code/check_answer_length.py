"""Annotate each interview answer with its spoken time and report outliers.

Speaking rate matters, and counting raw characters is wrong for mixed
Chinese/English text. A Chinese character is one syllable (~3.5 chars/second at
a technical pace); an English token like `PagedAttention` is 15 characters but
is spoken in about 1.5 seconds. Counting every non-space character at the
Chinese rate therefore *overestimates* English-heavy answers by 30-40%.

So we split each answer into CJK characters and everything else, and bill them
at different rates:

    CJK      240 chars/min  (220-280 band; slower when thinking)
    non-CJK  600 chars/min  (~1.5s per 15-char identifier, read not spelled)

The two rates are the same underlying pace expressed per script, not a fudge to
make answers look short — the reported time is the sum of the two bills.

Advisory, not a gate: answers slightly over 2 minutes are fine (you can drop the
💬 extension note or the last bullet). This script reports the distribution so
the reader can prioritise which answers to rehearse shortest.

**This script targets question-bank files formatted with `**Q<n> \\`[难度]\\` …**`
headings and `⏱ ~Ns` tags** (the original 附录 G/H style). The current
``docs/interview.md`` uses plain ``### §n`` sections instead, so running it on
that file reports "no question blocks found" (exit 2) — that is expected, not a
failure. Point it at a file that still uses the per-question heading + ⏱ format.

    python code/check_answer_length.py <file.md>              # report
    python code/check_answer_length.py <file.md> --annotate   # rewrite headings
"""

import re
import sys
import unicodedata
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

RATE_FAST, RATE_SLOW = 280, 200
NOMINAL = 240
ASCII_RATE = 600
TARGET_SEC = 120
SHORT_SEC = 60

positional = [a for a in sys.argv[1:] if not a.startswith("--")]
annotate = "--annotate" in sys.argv[1:]
# 文档在 <root>/docs/，本脚本在 <root>/code/ —— 只给文件名时去 docs/ 找
DOCS = Path(__file__).resolve().parent.parent / "docs"
if positional:
    p = Path(positional[0])
    if not p.exists() and (DOCS / p).exists():
        p = DOCS / p
    path = p
else:
    path = DOCS / "interview.md"

lines = path.read_text(encoding="utf-8").splitlines()

blocks: list[tuple[int, str, int, int]] = []   # (line index, title, cjk, other)
cur_idx: int | None = None
cur_title = ""
cur_cjk = 0
cur_other = 0


def flush() -> None:
    if cur_idx is not None:
        blocks.append((cur_idx, cur_title, cur_cjk, cur_other))


def is_cjk(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def split_count(text: str) -> tuple[int, int]:
    """Return (cjk chars, other non-space chars) excluding markdown noise."""
    text = re.sub(r"[*`>#]|\s+", "", text)
    cjk = sum(1 for ch in text if is_cjk(ch))
    return cjk, len(text) - cjk


for i, ln in enumerate(lines):
    if ln.startswith("### "):
        flush()
        cur_idx, cur_title, cur_cjk, cur_other = i, ln[4:].strip(), 0, 0
        continue
    if cur_idx is None:
        continue
    if not ln.startswith(">"):
        continue
    # The 💬 extension note is explicitly NOT part of the spoken answer, and
    # the heading already carries a hand-written ⏱ estimate in some files.
    if "💬" in ln or "可延伸" in ln:
        continue
    c, o = split_count(ln.lstrip("> "))
    cur_cjk += c
    cur_other += o
flush()

blocks = [b for b in blocks if b[1].startswith("Q") or "[" in b[1][:12]]
if not blocks:
    print("no question blocks found")
    sys.exit(2)


def spoken(cjk: int, other: int = 0, rate: int = NOMINAL) -> int:
    """Seconds to speak, billing CJK and non-CJK at their own rates."""
    return round(cjk / rate * 60 + other / ASCII_RATE * 60)


print(f"{len(blocks)} answers; CJK {NOMINAL} chars/min (band {RATE_SLOW}-{RATE_FAST}), "
      f"non-CJK {ASCII_RATE} chars/min\n")
print(f"{'nominal':>7} {'fast':>6} {'slow':>6} {'cjk':>6} {'en':>5}  question")
print("-" * 100)

over, short = [], []
for _, title, cjk, other in blocks:
    nom = spoken(cjk, other)
    fast = spoken(cjk, other, RATE_FAST)
    slow = spoken(cjk, other, RATE_SLOW)
    flag = ""
    if nom > TARGET_SEC:
        flag = "  <-- 可缩短"
        over.append((title, nom))
    elif fast < SHORT_SEC:
        flag = "  (偏短)"
        short.append((title, fast))
    label = title if len(title) <= 42 else title[:39] + "..."
    print(f"{nom:6d}s {fast:5d}s {slow:5d}s {cjk:6d} {other:5d}  {label}{flag}")

print()
tot = sum(spoken(c, o) for _, _, c, o in blocks)
print(f"中位 {sorted(spoken(c, o) for _, _, c, o in blocks)[len(blocks)//2]}s ｜ "
      f"范围 {min(spoken(c, o) for _, _, c, o in blocks)}-"
      f"{max(spoken(c, o) for _, _, c, o in blocks)}s ｜ "
      f"全部一次讲完约 {tot // 60} 分 {tot % 60} 秒")
print(f"≤{TARGET_SEC}s 的有 {len(blocks) - len(over)}/{len(blocks)}；"
      f"其余超过 2 分钟（可删掉 💬 延伸或最后一条）")

if short:
    print(f"\n偏短（{len(short)}）—— 可能缺深度：")
    for t, s in short:
        print(f"  ~{s}s  {t[:58]}")

if annotate:
    # Rewrite each question heading with a time tag, replacing any previous tag.
    # Handles both `⏱ ~120s` and the coding-problem form `⏱ 讲思路 ~90s`.
    tag = re.compile(r"\s*⏱\s*(?:讲思路\s*)?~?\d+s\s*$")
    for idx, title, cjk, other in blocks:
        clean = tag.sub("", lines[idx][4:].strip())
        lines[idx] = f"### {clean}  ⏱ ~{spoken(cjk, other)}s"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写入时间标注 → {path}")
