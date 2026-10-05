#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""把题目+答案做成「边看边听」的 MP4：整段文字在屏幕上，配音在音轨上。

版式（v2，按反馈改过）：

    ┌──────────────────────────────────────────────────────┐
    │ Q132 · §5 分布式与通信                    3 / 8      │  ← 静态页眉
    │                                                      │
    │  第 132 题。TP、PP、DP、EP 分别解决什么问题？         │
    │                                                      │
    │  答案：策略：TP（张量并行）；切什么：层内的权重矩阵   │  ← 当前页：整段
    │  （按列/行切）；解决什么：单层放不下一张卡；代价：     │     自动换行
    │  每层两次 all-reduce，通信最频繁。                    │
    │                                                      │
    │  策略：PP（流水并行）；切什么：按层切；…               │
    │                                                      │
    │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  ← 进度条
    └──────────────────────────────────────────────────────┘

和 v1 的区别（v1 是逐句高亮 + 上一句保留）：

* **不逐句高亮** —— 一屏给一整段，方便回看、也方便停下来读。
* **去掉「上一句」那一行** —— v1 里当前句与上一句会同时出现同一段文字，
  看起来像重复；题目更是会在页眉、当前句、上一句里出现三次。
* 页眉只留 ``Q编号 · 章名`` 与页码，**题目本身作为正文的第一页**，不再固定置顶。

分页规则：优先在空行（段落）处断，其次句末标点，最后才硬切；
每页预算受 ``--page-chars`` 与 ``--page-lines`` 控制。

用法::

    python build_video.py --probe Q132          # 只做一题，先看效果
    python build_video.py --chapter 5           # 第 5 章
    python build_video.py --all                 # 全部可朗读的题
    python build_video.py --all --chapter-merge # 每章再合成一个长视频
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interview_parser import DEFAULT_MD, QA, parse          # noqa: E402
from build_audiobook import VOICES                          # noqa: E402

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "video"
WORK = OUT / ".work"

W, H = 1280, 720
FPS = 12
FONT = "Microsoft YaHei"
CJK = re.compile(r"[\u4e00-\u9fff]")
BG = "0x14161a"

# 版式常量（调字号/边距就改这里）
PAGE_SIZE = 37          # 正文字号
MARGIN_LR = 76          # 左右留白
MARGIN_TOP = 128        # 正文距顶
LINE_H = 1.42           # 行高倍数（用于估每页行数）


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not exe:
        raise SystemExit("需要 ffmpeg（PATH 里找不到）")
    return exe


def ffprobe() -> str | None:
    return shutil.which("ffprobe") or shutil.which("ffprobe.exe")


def probe_duration(path: Path) -> float:
    fp = ffprobe()
    if not fp:
        return 0.0
    r = subprocess.run([fp, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------
# 1) 语音合成 + 逐词时间戳
# --------------------------------------------------------------------------
async def synth(text: str, voice: str, rate: str) -> tuple[bytes, list[tuple[float, float, str]]]:
    import edge_tts
    audio = bytearray()
    words: list[tuple[float, float, str]] = []
    # 关键：edge-tts 的 boundary 默认是 "SentenceBoundary"，不显式指定就
    # 一个 WordBoundary 都收不到（字幕会退化成按字数估算）。
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    async for chunk in comm.stream():
        kind = chunk["type"]
        if kind == "audio":
            audio.extend(chunk["data"])
        elif kind in ("WordBoundary", "SentenceBoundary"):
            s = chunk["offset"] / 1e7            # 100ns -> s
            d = chunk["duration"] / 1e7
            words.append((s, s + d, chunk["text"]))
    return bytes(audio), words


def align(text: str, words: list[tuple[float, float, str]]
          ) -> list[tuple[float, float, int, int]]:
    """把 word 对齐回原文的字符区间。

    edge-tts **不为标点发边界事件**，所以不能把 word 直接拼起来（会得到
    「第132题TPPPDPEP分别解决什么问题」）。行文本一律从原文切片。
    """
    spans = []
    pos = 0
    for s, e, w in words:
        w = w.strip()
        if not w:
            continue
        i = text.find(w, pos)
        if i < 0:
            i = text.find(w)
        if i < 0:
            continue
        spans.append((s, e, i, i + len(w)))
        pos = i + len(w)
    return spans


# 朗读时把 TP 展开成 "T P" 是为了让 TTS 念对；但**屏幕上的字幕应该显示 TP**。
# 这里把展开反向折叠回去（只针对我们自己插入的空格，所以是安全的）。
_DISPLAY_FIX = [
    (re.compile(r"\b([A-Z]{2,}) (\d+)\b"), r"\1\2"),   # "FP 32"   -> "FP32"
    (re.compile(r"\bMo E\b"), "MoE"),
    (re.compile(r"\bN V Link\b"), "NVLink"),
    (re.compile(r"\bP C I e\b"), "PCIe"),
    (re.compile(r"\bint (\d+)\b"), r"INT\1"),
    # 折叠成单字母串之后才出现的形态（顺序在 _SINGLE_CAP 之后）
    (re.compile(r"\bNV Link\b"), "NVLink"),
    (re.compile(r"\bPCI e\b"), "PCIe"),
]
_SINGLE_CAP = re.compile(r"(?<![A-Za-z0-9])([A-Z]) (?=[A-Z](?![A-Za-z0-9]))")


def display_text(s: str) -> str:
    """把「朗读形式」折回「显示形式」：``T P`` -> ``TP``、``Mo E`` -> ``MoE``。"""
    prev = None
    while prev != s:
        prev = s
        s = _SINGLE_CAP.sub(r"\1", s)
    for rx, rep in _DISPLAY_FIX:
        s = rx.sub(rep, s)
    # 「第 132 题。」在屏幕上单独占一行没意义（它是朗读时的停顿，不是断句）
    s = re.sub(r"^(第 \d+ 题)[。：]\s*", r"\1 ", s)
    return s


def _w(s: str) -> float:
    """显示宽度：CJK 记 1，拉丁/数字/半角标点记 0.5。

    早先的版本「有中文就只数中文」，结果 ``这是常见的口误（推理没有反向）推理 DP …``
    这种混排行算出来是 31，实际约 35，会超出右边界。
    """
    return sum(1.0 if CJK.match(c) else 0.5 for c in s)


NLBREAK = "\\N"


L1_BREAK = "；。！？!?;"          # 一级：分号/句号 —— 语义边界，必须换行
L2_BREAK = "，、：,:"             # 二级：逗号/冒号 —— 一行放不下时才在这里断


def semantic_lines(s: str, per_line: int) -> list[str]:
    r"""按标点分层断行，返回不带 ``\N`` 的行列表。

    表格转出来的答案形如
    ``策略：TP（张量并行）；切什么：层内的权重矩阵；解决什么：…`` ——
    **分号是语义边界**（每个「字段：值」是一组）。按宽度硬折会把一组劈成两行，
    读起来很别扭；所以断行优先级是：

    1. ``；。！？`` —— 每个子句独占一行；
    2. ``，、：`` —— 子句本身超宽时才在这里断；
    3. 还是超宽才硬切（英文长标识符会出现）。
    """
    # --- 一级切分（保留标点）---
    chunks = [c for c in re.split(rf"(?<=[{L1_BREAK}])", s) if c.strip()]
    if not chunks:
        chunks = [s]

    lines: list[str] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        if _w(ch) <= per_line:
            lines.append(ch)
            continue
        # --- 二级：逗号/冒号处断 ---
        cur = ""
        for piece in re.split(rf"(?<=[{L2_BREAK}])", ch):
            piece = piece.strip()
            if not piece:
                continue
            if _w(cur + piece) <= per_line:
                cur += piece
            else:
                if cur:
                    lines.append(cur)
                cur = piece
        if cur:
            lines.append(cur)

    # --- 三级：仍然超宽的行硬切 ---
    out: list[str] = []
    for ln in lines:
        while _w(ln) > per_line:
            cut, w = 0, 0.0
            for i, c in enumerate(ln):
                w += 1.0 if CJK.match(c) else 0.5
                if w > per_line:
                    break
                cut = i + 1
            if cut <= 0:
                cut = 1
            out.append(ln[:cut].strip())
            ln = ln[cut:].lstrip()
        if ln.strip():
            out.append(ln.strip())
    return [x for x in out if x]


def wrap_ass(s: str, per_line: int) -> tuple[str, int]:
    r"""按标点分层断行，返回 (带 ``\N`` 的文本, 行数)。

    **libass 不会给中文自动断行** —— 开了自动换行后整段文字会同时溢出左右
    边界（实测左留白里出现 YMAX=230 的白字）。所以断点用 ``\N`` 写死，
    ASS 头配 ``WrapStyle: 2``。
    """
    lines = semantic_lines(s.replace("\n", " "), per_line)
    # 题目后面空一行：题干和答案挤在一起不好读
    if lines and re.match(r"^第 \d+ 题", lines[0]):
        lines.insert(1, "")
    return NLBREAK.join(lines), len(lines)


# --------------------------------------------------------------------------
# 2) 分页：一屏一整段
# --------------------------------------------------------------------------
@dataclass
class Page:
    a: int          # 起始字符
    b: int          # 结束字符
    t0: float
    t1: float
    wrapped: str = ""      # 已按宽度插好 \N 的正文
    n_lines: int = 0

    def text(self, src: str) -> str:
        return src[self.a:self.b].strip()


def paginate(text: str, spans: list[tuple[float, float, int, int]],
             *, max_chars: int, max_lines: int, dur: float) -> list[Page]:
    """按段落切页，**并且保证每页真的放得下**。

    先算出每行能放多少字、每页最多几行，然后在候选断点（段落 > 句末 > 逗号）
    里挑「最远的、且换行后不超过 max_lines 的那个」；一个都放不下时二分找边界。
    """
    per_line = max(8, int((W - 2 * MARGIN_LR) / PAGE_SIZE))
    body_lines = max(3, int((H - MARGIN_TOP - 46) / (PAGE_SIZE * LINE_H)))
    max_lines = max(1, min(max_lines, body_lines))
    budget = min(max_chars, per_line * max_lines)

    hard = [m.start() for m in re.finditer(r"\n\s*\n", text)]
    soft = [m.end() for m in re.finditer(r"[。！？；]", text)]   # 分号也是语义边界
    comma = [m.end() for m in re.finditer(r"[，、；：,;]", text)]

    def page_txt(a: int, b: int) -> str:
        return display_text(text[a:b].strip())

    def nlines(a: int, b: int) -> int:
        return wrap_ass(page_txt(a, b), per_line)[1]

    def best_end(a: int) -> int:
        limit = min(len(text), a + budget)
        if limit >= len(text):
            return len(text)
        lo = a + max(8, int(budget * 0.4))
        cands = sorted({c for c in (*hard, *soft, *comma, limit) if lo <= c <= limit})
        for c in reversed(cands):
            if nlines(a, c) <= max_lines:
                return c
        # 连最短的候选都放不下 -> 二分找放得下的最大边界
        lo2, hi2 = a + 1, limit
        while lo2 < hi2:
            mid = (lo2 + hi2 + 1) // 2
            if nlines(a, mid) <= max_lines:
                lo2 = mid
            else:
                hi2 = mid - 1
        return max(lo2, a + 1)

    cuts, i = [0], 0
    guard = 0
    while i < len(text) and guard < 10000:
        j = best_end(i)
        cuts.append(j)
        i = j
        guard += 1
    if cuts[-1] < len(text):
        cuts.append(len(text))

    def t_of(char_pos: int, forward: bool) -> float:
        if not spans:
            return dur * char_pos / max(1, len(text))
        if forward:
            for s, e, a, b in spans:
                if b > char_pos:
                    return s
            return spans[-1][1]
        for s, e, a, b in spans:
            if a >= char_pos:
                return s
        return spans[-1][1]

    pages: list[Page] = []
    for k in range(len(cuts) - 1):
        a, b = cuts[k], cuts[k + 1]
        if not text[a:b].strip():
            continue
        wrapped, nl = wrap_ass(page_txt(a, b), per_line)
        if not wrapped:
            continue
        pages.append(Page(a, b, t_of(a, True), t_of(b, False), wrapped, nl))
    for k in range(len(pages) - 1):                # 不留空档，避免闪黑
        pages[k].t1 = max(pages[k].t1, pages[k + 1].t0)
    if pages:
        pages[-1].t1 = max(pages[-1].t1, dur)
    return pages


# --------------------------------------------------------------------------
# 3) ASS 字幕（libass 自动换行）
# --------------------------------------------------------------------------
ASS_HEAD = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Head,{FONT},25,&H008A8A8A,&H008A8A8A,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,{MARGIN_LR},40,34,1
Style: PgNo,{FONT},25,&H008A8A8A,&H008A8A8A,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,9,{MARGIN_LR},40,34,1
Style: Page,{FONT},{PAGE_SIZE},&H00F2F2F2,&H00F2F2F2,&H00101010,&H00000000,0,0,0,0,100,100,0.4,0,1,0,0,7,{MARGIN_LR},{MARGIN_LR},{MARGIN_TOP},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def esc_ass(s: str) -> str:
    return (s.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
             .replace("\n", " "))


def build_ass(q: QA, pages: list[Page], src: str, dur: float) -> str:
    head = f"Q{q.num} · {(q.chapter_raw or q.chapter_title).split('（')[0].strip()}"
    ev = [f"Dialogue: 0,{ass_time(0)},{ass_time(dur)},Head,,0,0,0,,{esc_ass(head)}"]
    for k, p in enumerate(pages, 1):
        ev.append(f"Dialogue: 0,{ass_time(p.t0)},{ass_time(p.t1)},PgNo,,0,0,0,,"
                  f"{k} / {len(pages)}")
        body = p.wrapped.replace("{", "｛").replace("}", "｝")
        ev.append(f"Dialogue: 1,{ass_time(p.t0)},{ass_time(p.t1)},Page,,0,0,0,,{body}")
    return ASS_HEAD + "\n".join(ev) + "\n"


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(pages: list[Page], src: str) -> str:
    out = []
    for i, p in enumerate(pages, 1):
        # 用屏幕上真正显示的断行（\N -> 换行），这样字幕和画面一致
        body = (p.wrapped or display_text(p.text(src))).replace("\\N", "\n")
        out.append(f"{i}\n{srt_time(p.t0)} --> {srt_time(p.t1)}\n{body}\n")
    return "\n".join(out)


# --------------------------------------------------------------------------
# 4) 合成
# --------------------------------------------------------------------------
DUR = {"v": 0.0}


def render(ass: Path, audio: Path, dest: Path) -> bool:
    # ffmpeg 滤镜里的 Windows 盘符冒号必须转义，否则被当成选项分隔符
    ass_arg = ass.as_posix().replace(":", r"\:").replace("'", r"\'")
    vf = (f"ass='{ass_arg}'"
          f",drawbox=x=0:y={H-8}:w='iw*t/{max(0.1, DUR['v']):.3f}':h=8:"
          f"color=0x4a90d9@0.9:t=fill")
    cmd = [ffmpeg(), "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c={BG}:s={W}x{H}:r={FPS}:d={DUR['v']:.3f}",
           "-i", str(audio),
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
           "-shortest", "-movflags", "+faststart", str(dest)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("      ffmpeg:", r.stderr.decode("utf-8", "replace")[-400:])
    return r.returncode == 0


async def one(q: QA, voice: str, rate: str, *, max_chars: int, max_lines: int,
              keep_work: bool) -> Path | None:
    text = q.speech()
    t_start = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    (OUT / "srt").mkdir(parents=True, exist_ok=True)

    audio, words = await synth(text, voice, rate)
    if not audio:
        print(f"  Q{q.num}: 没有音频")
        return None

    mp3 = WORK / f"q{q.num}.mp3"
    mp3.write_bytes(audio)
    dur = 0.0
    fp = ffprobe()
    if fp:
        r = subprocess.run([fp, "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(mp3)], capture_output=True, text=True)
        try:
            dur = float(r.stdout.strip())
        except ValueError:
            dur = 0.0
    if not dur:
        dur = len(audio) / 6000.0            # 48kbps ≈ 6 KB/s 兜底

    spans = align(text, words)
    pages = paginate(text, spans, max_chars=max_chars, max_lines=max_lines, dur=dur)
    if not pages:
        print(f"  Q{q.num}: 分页为空")
        return None

    DUR["v"] = dur
    ass = WORK / f"q{q.num}.ass"
    ass.write_text(build_ass(q, pages, text, dur), encoding="utf-8")
    (OUT / "srt" / f"q{q.num:03d}.srt").write_text(build_srt(pages, text),
                                                   encoding="utf-8-sig")

    dest = OUT / f"q{q.num:03d}.mp4"
    ok = render(ass, mp3, dest)
    if ok and not keep_work:
        ass.unlink(missing_ok=True)
        mp3.unlink(missing_ok=True)
    print(f"  Q{q.num:<4} {dur:>6.1f}s  {len(pages):>2} 页  "
          f"{'OK ' if ok else 'FAIL'}  {dest.stat().st_size//1024 if ok else 0:>5} KB  "
          f"({time.time()-t_start:.1f}s)")
    return dest if ok else None


def merge_all(done: list[tuple[QA, Path]], dest: Path) -> None:
    """把每题一个的 MP4 拼成**一个**长视频，并写入章节标记。

    3 小时以上的视频没有章节没法用 —— 所以每个题目是一个 chapter，
    标题形如 ``Q132 · TP、PP、DP、EP 分别解决什么问题？``。
    全部片段参数一致（同分辨率/编码/采样率），所以 ``-c copy`` 直接拼接，不重编码。
    """
    ff = ffmpeg()
    lst = WORK / "all.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for _, p in done),
                   encoding="utf-8")

    meta = [";FFMETADATA1", "title=AI Infra 面试题库 · 语音版"]
    t = 0.0
    for q, p in done:
        d = probe_duration(p)
        meta += ["[CHAPTER]", "TIMEBASE=1/1000",
                 f"START={int(t * 1000)}", f"END={int((t + d) * 1000)}",
                 f"title=Q{q.num} · {display_text(q.title_raw or q.title)}"]
        t += d
    metaf = WORK / "all.ffmeta"
    metaf.write_text("\n".join(meta) + "\n", encoding="utf-8")

    r = subprocess.run(
        [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-i", str(metaf), "-map_metadata", "1", "-map", "0",
         "-c", "copy", "-movflags", "+faststart", str(dest)],
        capture_output=True)
    if r.returncode != 0:
        print("  合并失败：", r.stderr.decode("utf-8", "replace")[-500:])
        return
    got = probe_duration(dest)
    mb = dest.stat().st_size / 1024 / 1024
    print(f"  合并全部 -> {dest.name}　{len(done)} 段　"
          f"{got/3600:.2f} 小时（期望 {t/3600:.2f}）　{mb:.0f} MB")

    # 章节时刻表另存一份，方便别的工具用（B 站分 P、剪辑软件）
    # 注意：累计时长要边走边加 —— 早先写成 sum(...done[:i]) 是 O(n²)，
    # 254 题会触发 3 万多次 ffprobe，把收尾拖成几十分钟。
    rows, acc = [], 0.0
    for q, p in done:
        rows.append(f"{int(acc):>7}  {int(acc)//3600:d}:{int(acc)%3600//60:02d}:"
                    f"{int(acc)%60:02d}  Q{q.num} · {display_text(q.title_raw or q.title)}")
        acc += probe_duration(p)
    (OUT / "chapters.txt").write_text(
        "# 起始秒    时间戳    标题\n" + "\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--voice", default="yunjian")
    ap.add_argument("--rate", default="-4%")
    ap.add_argument("--probe", metavar="Q", help="只做一题看效果，如 132")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--chapter")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--page-chars", type=int, default=270, help="每页最多多少字")
    ap.add_argument("--page-lines", type=int, default=10, help="每页最多多少行")
    ap.add_argument("--min-cjk", type=int, default=25)
    ap.add_argument("--min-sentence", type=float, default=0.15)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--chapter-merge", action="store_true")
    ap.add_argument("--merge-all", action="store_true",
                    help="把所有题拼成一个长视频（写入章节标记）")
    ap.add_argument("--out-name", help="--merge-all 的输出文件名")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "srt").mkdir(exist_ok=True)
    voice = VOICES.get(a.voice, a.voice)

    qas = parse(a.md)
    if a.probe:
        n = int(re.sub(r"\D", "", a.probe))
        qas = [q for q in qas if q.num == n]
    else:
        qas = [q for q in qas if not q.unreadable_reason(
            min_cjk=a.min_cjk, min_sentence_ratio=a.min_sentence)]
        if a.chapter:
            want = "§" + a.chapter.lstrip("§")
            qas = [q for q in qas if q.chapter.startswith(want)]
        if a.limit:
            qas = qas[:a.limit]
    if not qas:
        print("没有可生成的题")
        return 1

    print(f"音色 {voice}　{len(qas)} 题　{W}x{H}@{FPS}fps　"
          f"每页 ≤{a.page_chars} 字 / ≤{a.page_lines} 行")
    print(f"输出 {OUT}\n")

    done = []
    for q in qas:
        d = asyncio.run(one(q, voice, a.rate, max_chars=a.page_chars,
                            max_lines=a.page_lines, keep_work=a.keep_work))
        if d:
            done.append((q, d))

    if a.chapter_merge and done:
        by: dict[str, list[Path]] = {}
        for q, d in done:
            by.setdefault(q.chapter, []).append(d)
        for ch, parts in sorted(by.items()):
            lst = WORK / f"{ch}.txt"
            lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
                           encoding="utf-8")
            dest = OUT / f"{ch.replace('§','ch').split()[0]}.mp4"
            r = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-f", "concat",
                                "-safe", "0", "-i", str(lst), "-c", "copy", str(dest)],
                               capture_output=True)
            if r.returncode == 0:
                print(f"  合并 {ch} -> {dest.name}（{len(parts)} 段，"
                      f"{dest.stat().st_size//1024//1024} MB）")

    if a.merge_all and done:
        merge_all(done, OUT / (a.out_name or "interview-all.mp4"))
    print(f"\n完成 {len(done)}/{len(qas)} 题；字幕在 {OUT/'srt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
