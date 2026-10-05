#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a listenable audiobook (MP3 + podcast feed) from ``docs/interview.md``.

Design notes — why MP3 and not MP4:

* You want to *listen*, so the container should be audio-first. MP4 forces a
  video track (or a static still), doubling the size and breaking variable-speed
  playback / resume in most phone players.
* **One MP3 per question** is what makes it usable: 270 tracks you can jump
  around in, not one 4-hour blob.
* The best "listen any time" experience is not a file at all — it is a
  **podcast feed**. Any podcast app then gives you resume, speed control,
  download-all and per-episode progress for free. ``--feed`` writes one.

Usage::

    python build_audiobook.py --check                 # 只算时长，不联网
    python build_audiobook.py --preview 3             # 生成 3 题试听（多音色）
    python build_audiobook.py --chapter 5             # 只生成第 5 章
    python build_audiobook.py --all                   # 全部 270 题
    python build_audiobook.py --chapter 5 --m4b       # 合并成有声书（带章节）
    python build_audiobook.py --feed                  # 生成播客 RSS + 本地服务

Engine: ``--engine edge`` (free, no API key) | ``openai`` | ``azure`` | ``sapi``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interview_parser import DEFAULT_MD, QA, duration_estimate, parse  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent
OUT = HERE / "audio"
CACHE = OUT / ".cache"

VOICES = {
    "yunjian":  "zh-CN-YunjianNeural",    # 男声，解说腔（富感染力，适合长听）
    "xiaoyi":   "zh-CN-XiaoyiNeural",     # ★ 女声，最年轻、活泼（Cartoon/Novel 场景）
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",   # 女声，温暖自然（News/Novel 场景，最稳）
    "yunxi":    "zh-CN-YunxiNeural",      # 男声，年轻阳光
    "yunyang":  "zh-CN-YunyangNeural",    # 男声，新闻播报（专业、可靠）
    "yunxia":   "zh-CN-YunxiaNeural",     # 男声，可爱（Cartoon 场景）
    "liaoning": "zh-CN-liaoning-XiaobeiNeural",   # 女声，东北口音，幽默
    "shaanxi":  "zh-CN-shaanxi-XiaoniNeural",     # 女声，陕西口音，明亮
}
DEFAULT_VOICE = "yunjian"
RATE = "-4%"          # 稍慢，技术内容听不清会很难受


def ffmpeg() -> str | None:
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


# --------------------------------------------------------------------------
# 引擎
# --------------------------------------------------------------------------
async def _edge_one(text: str, voice: str, rate: str, dest: Path) -> None:
    import edge_tts
    for attempt in range(4):
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate)
            await comm.save(str(dest))
            if dest.exists() and dest.stat().st_size > 512:
                return
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"      retry {attempt+1}: {type(exc).__name__}")
            await asyncio.sleep(2 + 3 * attempt)
    raise RuntimeError("empty audio")


def tts_edge(items: list[tuple[str, Path]], voice: str, rate: str, jobs: int) -> None:
    async def run():
        sem = asyncio.Semaphore(jobs)

        async def one(text, dest):
            async with sem:
                await _edge_one(text, voice, rate, dest)
        await asyncio.gather(*(one(t, d) for t, d in items))
    asyncio.run(run())


def tts_openai(items, voice, _rate, jobs) -> None:
    from openai import OpenAI
    client = OpenAI()                       # 读 OPENAI_API_KEY
    for text, dest in items:
        with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts", voice=voice or "alloy",
                input=text, response_format="mp3") as r:
            r.stream_to_file(dest)


def tts_azure(items, voice, rate, _jobs) -> None:
    import azure.cognitiveservices.speech as sp
    cfg = sp.SpeechConfig(subscription=os.environ["AZURE_SPEECH_KEY"],
                          region=os.environ.get("AZURE_SPEECH_REGION", "eastasia"))
    cfg.speech_synthesis_voice_name = voice or "zh-CN-YunxiNeural"
    cfg.set_speech_synthesis_output_format(
        sp.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3)
    rate_txt = f"{'+' if not rate.startswith('-') else ''}{rate.replace('%','')}%"
    cfg.speech_synthesis_rate = rate_txt
    for text, dest in items:
        sp.SpeechSynthesizer(speech_config=cfg, audio_config=None) \
            .speak_text_async(text).get().audio_data
        out = sp.SpeechSynthesizer(speech_config=cfg, audio_config=None)
        res = out.speak_text_async(text).get()
        dest.write_bytes(res.audio_data)


def tts_sapi(items, voice, _rate, _jobs) -> None:
    """离线兜底：Windows SAPI。中文需要系统装了中文语音，否则会读成英文。"""
    import win32com.client                                   # noqa
    sp = win32com.client.Dispatch("SAPI.SpVoice")
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    for text, dest in items:
        stream.Open(str(dest), 3, False)                     # 3 = SSFMCreateForWrite
        sp.AudioOutputStream = stream
        sp.Speak(text)
        stream.Close()


ENGINES = {"edge": tts_edge, "openai": tts_openai,
           "azure": tts_azure, "sapi": tts_sapi}


# --------------------------------------------------------------------------
# 后处理：MP3 合并 / M4B / 播客
# --------------------------------------------------------------------------
def concat_mp3(parts: list[Path], dest: Path) -> bool:
    ff = ffmpeg()
    if not ff or not parts:
        return False
    lst = dest.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
                   encoding="utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
           "-i", str(lst), "-c", "copy", str(dest)]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0
    lst.unlink(missing_ok=True)
    return ok


def to_m4b(chapters: list[tuple[str, Path]], dest: Path) -> bool:
    """把每章一个 mp3 合成带章节标记的 M4B（有声书格式）。"""
    ff = ffmpeg()
    if not ff or not chapters:
        return False
    parts = []
    meta = [";FFMETADATA1"]
    t = 0.0
    for title, mp3 in chapters:
        dur = probe_duration(mp3)
        parts.append(mp3)
        meta += ["[CHAPTER]", "TIMEBASE=1/1000",
                 f"START={int(t*1000)}", f"END={int((t+dur)*1000)}",
                 f"title={title}"]
        t += dur
    lst = dest.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
                   encoding="utf-8")
    meta_f = dest.with_suffix(".ffmeta")
    meta_f.write_text("\n".join(meta) + "\n", encoding="utf-8")
    cmd = [ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
           "-i", str(lst), "-i", str(meta_f),
           "-map_metadata", "1", "-c:a", "aac", "-b:a", "64k",
           "-f", "mp4", str(dest)]
    ok = subprocess.run(cmd, capture_output=True).returncode == 0
    lst.unlink(missing_ok=True)
    meta_f.unlink(missing_ok=True)
    return ok


def probe_duration(path: Path) -> float:
    ff = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not ff:
        return 0.0
    r = subprocess.run([ff, "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def write_feed(tracks: list[tuple[QA, Path, float]], dest: Path,
               base_url: str, title: str) -> None:
    """极简 podcast RSS：小宇宙 / Apple Podcasts / Pocket Casts 都能订。"""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    items = []
    for q, mp3, dur in tracks:
        secs = int(dur) or int(duration_estimate(q.speech()))
        items.append(f"""    <item>
      <title>Q{q.num} · {esc(q.title)}</title>
      <description>{esc(q.chapter)} {esc(q.section)}　[{esc(q.tag)}]</description>
      <enclosure url="{base_url}/{mp3.name}" length="{mp3.stat().st_size}" type="audio/mpeg"/>
      <guid isPermaLink="false">q{q.num}</guid>
      <pubDate>{now}</pubDate>
      <itunes:duration>{secs}</itunes:duration>
      <itunes:episode>{q.num}</itunes:episode>
    </item>""")
    dest.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>{esc(title)}</title>
    <description>AI Infra / 推理优化 / 分布式系统 校招面试题库（语音版）</description>
    <language>zh-CN</language>
    <itunes:author>net-nccl-tutorial</itunes:author>
    <itunes:category text="Technology"/>
{chr(10).join(items)}
  </channel>
</rss>
""", encoding="utf-8")


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


# --------------------------------------------------------------------------
def target_path(q: QA, voice: str) -> Path:
    return OUT / f"q{q.num:03d}.mp3"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--engine", default="edge", choices=sorted(ENGINES))
    ap.add_argument("--voice", default=DEFAULT_VOICE,
                    help=f"{DEFAULT_VOICE} 或 {', '.join(VOICES)} 或完整音色名")
    ap.add_argument("--rate", default=RATE)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--chapter", help="只做某一章，如 5 或 §5")
    ap.add_argument("--preview", type=int, metavar="N", help="每章抽 1 题，共 N 题")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--m4b", action="store_true", help="按章合并成 M4B 有声书")
    ap.add_argument("--merge", action="store_true", help="按章合并成单个 MP3")
    ap.add_argument("--feed", action="store_true", help="生成播客 RSS")
    ap.add_argument("--serve", action="store_true", help="起本地 HTTP 供手机订阅")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--check", action="store_true", help="只统计，不生成")
    ap.add_argument("--skip-code", action=argparse.BooleanOptionalAction, default=True,
                    help="跳过手撕代码题（§7），默认开")
    ap.add_argument("--min-cjk", type=int, default=25,
                    help="正文中文少于该字数就跳过（默认 25）")
    ap.add_argument("--min-sentence", type=float, default=0.15,
                    help="成句中文占比低于该值就跳过（默认 0.15）")
    ap.add_argument("--include", help="强制包含的题号，逗号分隔，如 70,136")
    ap.add_argument("--exclude", help="强制排除的题号，逗号分隔")
    a = ap.parse_args()

    qas = parse(a.md)

    # ---- 过滤掉念不出来的题 ----
    force_in = {int(x) for x in re.findall(r"\d+", a.include or "")}
    force_out = {int(x) for x in re.findall(r"\d+", a.exclude or "")}
    skipped: list[tuple[QA, str]] = []
    kept: list[QA] = []
    for q in qas:
        if q.num in force_in:
            kept.append(q)
            continue
        reason = (q.unreadable_reason(min_cjk=a.min_cjk,
                                      min_sentence_ratio=a.min_sentence,
                                      skip_code=a.skip_code)
                  if q.num not in force_out else "手动排除")
        if reason:
            skipped.append((q, reason))
        else:
            kept.append(q)
    qas = kept

    if a.chapter:
        want = "§" + a.chapter.lstrip("§")
        qas = [q for q in qas if q.chapter.startswith(want)]
    if a.preview:
        seen, pick = set(), []
        for q in qas:
            if q.chapter not in seen:
                seen.add(q.chapter)
                pick.append(q)
        qas = pick[:a.preview]

    secs = sum(duration_estimate(q.speech()) for q in qas)
    print(f"可朗读 {len(qas)} 题，预估 {secs/60:.1f} 分钟音频")
    if skipped:
        print(f"\n跳过 {len(skipped)} 题（代码 / 念不出来）：")
        for q, r in skipped[:24]:
            print(f"  Q{q.num:<4} {r:<22} {q.title[:46]}")
        if len(skipped) > 24:
            print(f"  … 共 {len(skipped)} 题（用 --include 70,136 可强制加回）")
        print("\n  调整策略：--min-cjk 25 / --min-sentence 0.25 / --no-skip-code")
        print()
    if a.check:
        for q in qas[:10]:
            print(f"  Q{q.num:<4} {duration_estimate(q.speech()):>6.0f}s  {q.title[:56]}")
        if len(qas) > 10:
            print(f"  … 其余 {len(qas)-10} 题")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    voice = VOICES.get(a.voice, a.voice)
    todo = [(q, target_path(q, voice)) for q in qas]
    fresh = [(q, p) for q, p in todo if not (p.exists() and p.stat().st_size > 512)]
    print(f"音色 {voice}　已有 {len(todo)-len(fresh)} 题（跳过）　待生成 {len(fresh)} 题")
    if fresh:
        t0 = time.time()
        ENGINES[a.engine]([(q.speech(), p) for q, p in fresh], voice, a.rate, a.jobs)
        dt = time.time() - t0
        print(f"完成，用时 {dt:.0f}s（{len(fresh)/max(dt,1e-9):.1f} 题/秒）")

    made = [(q, p) for q, p in todo if p.exists()]
    if a.merge or a.m4b:
        by_ch: dict[str, list[tuple[QA, Path]]] = {}
        for q, p in made:
            by_ch.setdefault(q.chapter, []).append((q, p))
        chapters = []
        for ch, lst in sorted(by_ch.items(), key=lambda kv: kv[1][0][0].num):
            dest = OUT / f"{ch.replace('§','ch').split()[0]}.mp3"
            if concat_mp3([p for _, p in lst], dest):
                chapters.append((ch, dest))
                print(f"  合并 {ch} -> {dest.name}（{len(lst)} 题）")
        if a.m4b and chapters:
            m4b = OUT / "interview.m4b"
            if to_m4b(chapters, m4b):
                print(f"  M4B -> {m4b.name}（{m4b.stat().st_size//1024//1024} MB，带章节）")

    if a.feed or a.serve:
        tracks = [(q, p, probe_duration(p)) for q, p in made]
        base = f"http://127.0.0.1:{a.port}" if a.serve else "."
        write_feed(tracks, OUT / "feed.xml", base, "AI Infra 面试题库 · 语音版")
        print(f"  播客 feed -> {OUT/'feed.xml'}（{len(tracks)} 集）")
    if a.serve:
        import functools
        import http.server
        import socketserver
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
        with socketserver.TCPServer(("0.0.0.0", a.port), handler) as httpd:
            print(f"\n手机同一 Wi-Fi 下订阅： http://<这台电脑的 IP>:{a.port}/feed.xml")
            print("按 Ctrl-C 停止")
            httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
