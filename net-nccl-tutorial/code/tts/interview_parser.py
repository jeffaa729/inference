#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn ``docs/interview.md`` into speech-ready plain text.

This module is the part that decides whether the audiobook is listenable. TTS
engines are all good enough now; what ruins a technical audiobook is raw Markdown
— code fences, tables, emphasis markers, and symbols like ``→ ≈ × ·`` read out
as "right arrow", "almost equal to", "multiplication sign", "middle dot".

    python interview_parser.py --stats          # 有多少字、多少题、多长时间
    python interview_parser.py --dump Q132      # 看第 132 题会怎么念
    python interview_parser.py --chapter 5      # 看第 5 章开头
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                      # net-nccl-tutorial/
DEFAULT_MD = REPO / "docs" / "interview.md"

# --------------------------------------------------------------------------
# 符号 / 缩写 → 口语。顺序有意义：长的先替。
# --------------------------------------------------------------------------
SPOKEN = [
    ("≈", "约等于"), ("≤", "小于等于"), ("≥", "大于等于"), ("≠", "不等于"),
    ("→", "，"), ("⇒", "，所以"), ("←", "，"), ("↔", "双向"),
    ("×", "乘"), ("·", "，"), ("÷", "除以"),
    ("µ", "微"), ("μ", "微"), ("Ω", "欧"),
    ("%", "百分之"), ("°", "度"),
    ("&", "和"), ("@", "at"),
    ("「", ""), ("」", ""), ("『", ""), ("』", ""),
    ("——", "，"), ("--", "，"),
    ("§", "第"), ("★", ""), ("⚠️", "注意"), ("⚠", "注意"),
    ("✅", "可以"), ("❌", "不行"), ("⭐", ""), ("📌", ""),
]

# 常见技术词的口语化（中文 TTS 读英文缩写常出错）
ABBREV = [
    (r"\bTP\b", "T P"), (r"\bPP\b", "P P"), (r"\bDP\b", "D P"), (r"\bEP\b", "E P"),
    (r"\bCP\b", "C P"), (r"\bSP\b", "S P"), (r"\bDCP\b", "D C P"), (r"\bPCP\b", "P C P"),
    (r"\bKV\b", "K V"), (r"\bFA(\d)\b", r"F A \1"), (r"\bGEMM\b", "G E M M"),
    (r"\bGEMV\b", "G E M V"), (r"\bSM\b", "S M"), (r"\bHBM\b", "H B M"),
    (r"\bGPU\b", "G P U"), (r"\bCPU\b", "C P U"), (r"\bNCCL\b", "N C C L"),
    (r"\bRDMA\b", "R D M A"), (r"\bIB\b", "I B"), (r"\bPCIe\b", "P C I e"),
    (r"\bNVLink\b", "N V Link"), (r"\bMoE\b", "Mo E"), (r"\bMLA\b", "M L A"),
    (r"\bGQA\b", "G Q A"), (r"\bMQA\b", "M Q A"), (r"\bMHA\b", "M H A"),
    (r"\bTTFT\b", "T T F T"), (r"\bTPOT\b", "T P O T"), (r"\bITL\b", "I T L"),
    (r"\bQPS\b", "Q P S"), (r"\bOOM\b", "O O M"), (r"\bFP(\d+)\b", r"F P \1"),
    (r"\bBF(\d+)\b", r"B F \1"), (r"\bINT(\d+)\b", r"int \1"),
    (r"\bTMA\b", "T M A"), (r"\bWGMMA\b", "W G M M A"), (r"\bDBO\b", "D B O"),
    (r"\bEPLB\b", "E P L B"), (r"\bLLM\b", "L L M"), (r"\bQKV\b", "Q K V"),
]

# 数学/代码符号 → 口语（放在 SPOKEN 之后单独跑，因为要带正则）
MATH = [
    (r"!=", "不等于"), (r"==", "等于"), (r"<=", "小于等于"), (r">=", "大于等于"),
    (r"\^2\b", "的平方"), (r"\^3\b", "的立方"),
    (r"\^(-?\d+)", r"的 \1 次方"),
    (r"\^(\w)", r" 的 \1 次方"),
    (r"\bmax\b", "max"), (r"→", "，"),
]

# 这些行是引文/元数据，念出来只会打断思路
SKIP_LINE = re.compile(r"^\*?\s*(图源|来源|参考文献|参考仓库|源码|作者|许可)[：:]")

MARKERS = {
    "答·": "",              # 已由 speech() 统一加「答案：」，这一层去掉避免念两遍
    "数字·": "关键数字：",
    "坑·": "常见错答：",
}
CJK_RX = re.compile(r"[\u4e00-\u9fff]")

FENCE = re.compile(r"```.*?```", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
QUESTION = re.compile(r"^\*\*Q(\d+)\s*`\[([^\]]+)\]`\s*(★?)\s*(.*?)\*\*\s*$")
Q_REF = re.compile(r"\bQ(\d+)\b")
SEC_REF = re.compile(r"§(\d+(?:\.\d+)*)")


def strip_inline(s: str) -> str:
    """去掉行内 markdown，但保留文字。"""
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)          # 图片
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)      # 链接 -> 文字
    s = re.sub(r"`([^`]+)`", r"\1", s)                  # 行内代码 -> 原文
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", s)
    s = re.sub(r"~~(.+?)~~", r"\1", s)
    s = re.sub(r"^\s*[-*+]\s+", "", s)                  # 列表符号
    s = re.sub(r"^\s*\d+\.\s+", "", s)
    return s


def speakable(s: str) -> str:
    """把一行 markdown 文本变成适合朗读的一句话。"""
    s = strip_inline(s)
    for k, v in MARKERS.items():                 # 标记要在去掉反引号后再认
        if s.lstrip().startswith(k):
            rest = s.lstrip()[len(k):].lstrip()
            # 如果正文自带小标题（如「关键恒等式：」），就别再加一层「关键数字：」
            if v and re.match(r"^[^\s：]{1,10}：", rest):
                s = rest
            else:
                s = v + rest
            break
    for a, b in SPOKEN:
        s = s.replace(a, b)
    for pat, rep in MATH:
        s = re.sub(pat, rep, s)
    s = Q_REF.sub(lambda m: f"问题 {m.group(1)}", s)
    s = SEC_REF.sub(lambda m: f"第 {m.group(1)} 节", s)
    for pat, rep in ABBREV:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[，、]{2,}", "，", s)
    s = re.sub(r"：{2,}", "：", s)
    s = re.sub(r"[。？]{2,}", "。", s)
    s = re.sub(r"？。", "？", s)
    s = s.replace("**", "")              # 跨行加粗会留下孤立的 **
    s = re.sub(r"(?<![\w*])\*(?![\w*])", "", s)
    s = re.sub(r"（\s*，", "（", s)
    s = re.sub(r"（\s*$", "", s)
    return s.strip(" ，。;；")


@dataclass
class QA:
    num: int
    tag: str
    star: bool
    title: str
    chapter: str          # "§5"
    chapter_title: str    # 朗读用（缩写已展开，如 "T P"）
    section: str = ""     # "5.2"
    chapter_raw: str = ""    # 显示用（原文，如 "TP"）
    title_raw: str = ""      # 显示用（原文，不改缩写）
    body: list[str] = field(default_factory=list)
    code_ratio: float = 0.0   # 题目正文里来自代码/表格的比例（用于判断能不能听）

    # ---- 能不能听 ----
    @property
    def text(self) -> str:
        return "\n".join(self.body)

    @property
    def cjk(self) -> int:
        return len(CJK_RX.findall(self.text))

    @property
    def sentence_ratio(self) -> float:
        """正文里「成句的中文」占多少行 —— 念不出来的题这一项会很低。"""
        lines = [l for l in self.body if l.strip()]
        if not lines:
            return 0.0
        return sum(1 for l in lines if len(CJK_RX.findall(l)) >= 10) / len(lines)

    def unreadable_reason(self, *, min_cjk=25, min_sentence_ratio=0.15,
                          skip_code=True) -> str | None:
        """返回「不适合听」的原因；None 表示可以朗读。

        阈值故意做得可调：``§7`` 是纯代码（中文只剩 7–27 字）必须跳过，
        但像「集合通信有哪些操作」这种中文少、却全是可朗读的英文术语的题，
        砍掉反而是损失 —— 所以中文阈值定得低，只挡真正的公式/命令墙。
        """
        if skip_code and self.section.lstrip("§").startswith("7"):
            return "手撕代码题（§7）"
        if not self.body:
            return "没有可朗读正文"
        if self.cjk < min_cjk:
            return f"中文只有 {self.cjk} 字"
        if self.sentence_ratio < min_sentence_ratio:
            return f"成句中文仅占 {self.sentence_ratio:.0%}"
        return None

    def speech(self, headers: bool = True) -> str:
        """口语格式：``第 N 题。<题目>`` 换行 ``答案：<正文>``。"""
        parts = []
        if headers:
            t = self.title.rstrip("。")
            if not t.endswith(("？", "！", "?")):
                t += "。"
            parts.append(f"第 {self.num} 题。{t}")
        body = "\n".join(x for x in self.body if x).strip()
        if body:
            parts.append("答案：" + body)
        return "\n".join(parts)


def table_to_prose(rows: list[list[str]]) -> list[str]:
    """把 markdown 表格读成「表头：单元格；表头：单元格」。"""
    if not rows:
        return []
    head = rows[0]
    out = []
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        parts = []
        for i, cell in enumerate(r):
            cell = cell.strip()
            if not cell:
                continue
            key = head[i].strip() if i < len(head) else ""
            parts.append(f"{key}：{cell}" if key else cell)
        if parts:
            out.append("；".join(parts) + "。")
    return out


def parse(md_path: Path, *, tables: bool = True, skip_code_sections=True) -> list[QA]:
    raw = md_path.read_text(encoding="utf-8")
    raw = HTML_COMMENT.sub("", raw)
    lines = raw.split("\n")

    chapter = chapter_title = chapter_raw = ""
    section = ""
    qas: list[QA] = []
    cur: QA | None = None
    in_code = False
    tbl: list[list[str]] = []

    def flush_table():
        nonlocal tbl
        if tbl and cur is not None:
            cur.body.extend(table_to_prose(tbl) if tables else [])
        tbl = []

    for ln in lines:
        if ln.strip().startswith("```"):
            flush_table()
            in_code = not in_code
            continue
        if in_code:
            continue

        m = HEADING.match(ln)
        if m:
            flush_table()
            lvl, txt = len(m.group(1)), speakable(m.group(2))
            # 任何章节标题都意味着「上一题结束了」—— 否则最后一题会把下一章的
            # 导语与速查表整段吞掉（§11 的 Q262 就曾吞掉整个 §12 的 L.1–L.5）。
            cur = None
            if lvl == 2 and txt.startswith("第"):        # ## §5 xxx
                chapter = txt.split("（")[0].strip()
                chapter_title = txt
                chapter_raw = m.group(2).strip()
            elif lvl == 3:
                section = txt
            continue

        if TABLE_ROW.match(ln):
            if TABLE_SEP.match(ln):
                continue
            tbl.append([speakable(c) for c in ln.strip().strip("|").split("|")])
            continue
        flush_table()

        mq = QUESTION.match(ln)
        if mq:
            num = int(mq.group(1))
            cur = QA(num=num, tag=mq.group(2), star=bool(mq.group(3).strip()),
                     title=speakable(mq.group(4)), title_raw=mq.group(4).strip(),
                     chapter=chapter, chapter_title=chapter_title,
                     chapter_raw=chapter_raw, section=section)
            qas.append(cur)
            continue

        if cur is None:
            continue

        line = ln.rstrip()
        if not line.strip() or line.strip() == "---":
            continue
        if SKIP_LINE.match(line.strip()):        # 图源/来源/源码 这类引文不念
            continue
        # 题内元信息（> **考点**：… / > **源码**：… ）-> 只读考点
        if line.lstrip().startswith(">"):
            body = line.lstrip()[1:].strip()
            if body.startswith(("**源码**", "**参考实现**", "**怎么用**")):
                continue
            body = body.replace("**考点**", "考点：")
            if not body.strip():
                continue
            line = body
        s = speakable(line)
        if s and not SKIP_LINE.match(s):
            cur.body.append(s)

    flush_table()
    return [q for q in qas if q.body or q.title]


def duration_estimate(text: str) -> float:
    """中文按 4.6 字/秒、英文按 2.6 词/秒 粗估秒数。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    words = len(re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*", text))
    return cjk / 4.6 + words / 2.6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--dump", metavar="Q", help="打印某题会怎么念（Q132 或 132）")
    ap.add_argument("--chapter", metavar="N", help="打印某章开头")
    ap.add_argument("--json", action="store_true", help="按 JSON 输出全部题目")
    a = ap.parse_args()

    qas = parse(a.md)
    if a.json:
        print(json.dumps([{"num": q.num, "tag": q.tag, "star": q.star,
                           "chapter": q.chapter, "section": q.section,
                           "title": q.title, "speech": q.speech()}
                          for q in qas], ensure_ascii=False, indent=1))
        return 0
    if a.dump:
        n = int(re.sub(r"\D", "", a.dump))
        q = next((q for q in qas if q.num == n), None)
        if not q:
            print(f"没有 Q{n}")
            return 1
        print(q.speech())
        return 0
    if a.chapter:
        want = "§" + a.chapter.lstrip("§")
        sel = [q for q in qas if q.chapter.startswith(want)]
        for q in sel[:3]:
            print(q.speech())
            print("-" * 60)
        return 0

    total = 0.0
    per_ch: dict[str, list[int]] = {}
    for q in qas:
        total += duration_estimate(q.speech())
        per_ch.setdefault(q.chapter, []).append(q.num)
    print(f"题目数：{len(qas)}")
    print(f"预估总时长：{total/3600:.1f} 小时（{total/60:.0f} 分钟）")
    print()
    print(f"{'章':<34}{'题数':>6}{'时长':>10}   题号")
    for ch, nums in sorted(per_ch.items(), key=lambda kv: kv[1][0]):
        secs = sum(duration_estimate(q.speech()) for q in qas if q.chapter == ch)
        print(f"{ch[:32]:<34}{len(nums):>6}{secs/60:>9.1f}m   Q{nums[0]}–Q{nums[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
