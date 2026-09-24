#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export ``interview.md`` (and optionally other chapters) to a print-ready PDF.

Pipeline — chosen because this machine has **no LaTeX, no pandoc, no wkhtmltopdf**:

    markdown --(python-markdown + pymdownx + pygments)--> HTML
             --(mermaid-cli renders fences to PNG, inlined as data URIs)--> one .html
             --(headless Chrome --print-to-pdf)--> .pdf

Everything is derived from the Markdown source, so the PDF never drifts from the
text. The intermediate HTML is written to ``$TEMP`` and deleted unless ``--keep``.

    python export_pdf.py                       # interview.md -> interview.pdf
    python export_pdf.py ch06-interview-bank.md
    python export_pdf.py --all                 # every chapter + appendix
    python export_pdf.py --keep                # keep the intermediate .html

``interview.md`` embeds its §7 kernel code from ``interview-code/kernels/*.cu``
(see ``interview-code/build_kernels.py``). This script calls that builder first,
so the PDF always reflects the code that ``build.sh`` actually compiles.

Requirements (all already present in this environment):
    python -m pip install markdown pymdown-extensions pygments
    node + npx + @mermaid-js/mermaid-cli   (only needed when the file has mermaid)
    Google Chrome / Microsoft Edge          (used as the HTML->PDF engine)
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Locate the HTML renderer (Chrome first, Edge as fallback)
# --------------------------------------------------------------------------
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

CHROME_ARGS = [
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--no-first-run",
    "--disable-extensions",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=20000",
]


def find_chrome() -> Path | None:
    for p in CHROME_CANDIDATES:
        if p and p.exists():
            return p
    env = os.environ.get("CHROME_PATH")
    if env and Path(env).exists():
        return Path(env)
    return None


def find_mmdc() -> Path | None:
    """Locate mermaid-cli, reusing the copy the diagram checker installed."""
    env = os.environ.get("MMDC")
    if env and Path(env).exists():
        return Path(env)
    candidates = [
        Path(tempfile.gettempdir()) / "mmd-validate/node_modules/.bin/mmdc.cmd",
    ]
    for c in candidates:
        if c.exists():
            return c
    for name in ("mmdc.cmd", "mmdc"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


# --------------------------------------------------------------------------
# CSS: print-first. Light theme (dark backgrounds waste toner and break links).
# --------------------------------------------------------------------------
CSS = r"""
@page {
  size: A4;
  margin: 16mm 14mm 18mm 14mm;
}
@page :first { margin-top: 14mm; }

:root {
  --fg: #1a1a1a;
  --muted: #5a5a5a;
  --rule: #d8d8d8;
  --code-bg: #f6f7f9;
  --accent: #b03a2e;
  --accent-soft: #fdecea;
  --note-bg: #fdf6e3;
  --note-rule: #b8860b;
}

html { font-size: 10.5pt; }

body {
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC",
               "Hiragino Sans GB", "Source Han Sans SC", sans-serif;
  color: var(--fg);
  line-height: 1.62;
  margin: 0;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

h1, h2, h3, h4 {
  font-weight: 650;
  line-height: 1.3;
  break-after: avoid-page;
  page-break-after: avoid;
}
h1 { font-size: 1.9rem; margin: 0 0 .6em; }
h2 {
  font-size: 1.35rem; margin: 1.6em 0 .5em;
  padding-bottom: .22em; border-bottom: 2px solid var(--accent);
}
h3 {
  font-size: 1.12rem; margin: 1.3em 0 .4em;
  padding-left: .5em; border-left: 4px solid var(--accent);
}
h4 { font-size: 1rem; margin: 1.1em 0 .35em; color: #333; }

p { margin: .5em 0; orphans: 3; widows: 3; }

/* ---- questions: keep a question's first paragraph with its heading ---- */
h3 + p, h3 + blockquote, h3 + table, h3 + pre { break-before: avoid-page; }

strong { font-weight: 650; }
code {
  font-family: "Cascadia Mono", Consolas, "Courier New", monospace;
  font-size: .875em;
  background: var(--code-bg);
  border: 1px solid #e3e6ea;
  border-radius: 3px;
  padding: .05em .32em;
  word-break: break-word;
}
pre {
  background: var(--code-bg);
  border: 1px solid #e3e6ea;
  border-left: 3px solid #a9b2bd;
  border-radius: 4px;
  padding: .6em .8em;
  overflow-x: auto;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
pre code { background: none; border: none; padding: 0; font-size: .84em; }

blockquote {
  margin: .7em 0;
  padding: .55em .9em;
  background: var(--note-bg);
  border-left: 4px solid var(--note-rule);
  border-radius: 0 4px 4px 0;
  break-inside: avoid-page;
}
blockquote p { margin: .3em 0; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: .7em 0;
  font-size: .93em;
  break-inside: auto;
}
thead { display: table-header-group; }
tr { break-inside: avoid-page; page-break-inside: avoid; }
th, td {
  border: 1px solid var(--rule);
  padding: .3em .5em;
  text-align: left;
  vertical-align: top;
}
th { background: #eef1f5; font-weight: 650; }

ul, ol { margin: .45em 0 .45em 1.3em; padding: 0; }
li { margin: .18em 0; }

hr { border: none; border-top: 1px solid var(--rule); margin: 1.4em 0; }

a { color: #1a4f8a; text-decoration: none; }

img.mermaid {
  display: block;
  width: auto;
  max-width: 100%;
  /* A4 is 297mm tall; page box is 297-16-18 = 263mm. Cap diagrams just under
     that so a tall one scales down instead of overflowing onto a blank page. */
  max-height: 250mm;
  margin: .8em auto;
  break-inside: avoid-page;
  page-break-inside: avoid;
}

/* ---- title block (injected) ---- */
.title-block {
  border-bottom: 3px solid var(--accent);
  padding-bottom: .8em;
  margin-bottom: 1.2em;
}
.title-block .sub { color: var(--muted); font-size: .95em; margin-top: .35em; }
.title-block .meta { color: var(--muted); font-size: .82em; margin-top: .5em; }

/* ---- table of contents (injected, generated from h2/h3) ---- */
nav.toc {
  background: #fbfcfd;
  border: 1px solid var(--rule);
  border-radius: 5px;
  padding: .9em 1.1em;
  margin: 1.2em 0 1.8em;
  break-inside: avoid-page;
  break-after: page;
}
nav.toc h2 {
  font-size: 1.05rem; margin: 0 0 .5em; border: none;
  padding: 0; color: var(--muted); letter-spacing: .05em;
}
nav.toc ol { list-style: none; margin: 0; padding: 0; }
nav.toc li { margin: .12em 0; font-size: .9em; }
nav.toc li.lvl3 { padding-left: 1.4em; color: #444; font-size: .85em; }
nav.toc .pg { color: var(--muted); }
"""

# --------------------------------------------------------------------------
# Markdown -> HTML
# --------------------------------------------------------------------------
MD_EXTENSIONS = [
    "extra",            # tables, fenced_code, footnotes, attr_list, def_list
    "toc",
    "sane_lists",
    "smarty",
    "pymdownx.tilde",   # ~~strike~~
    "pymdownx.caret",
    "pymdownx.mark",
    "pymdownx.tasklist",
    "pymdownx.superfences",
]
MD_EXT_CONFIGS = {
    "toc": {"permalink": False, "toc_depth": "2-3"},
    "pymdownx.tasklist": {"custom_checkbox": False},
}

MERMAID_RX = re.compile(r"```mermaid\n(.*?)```", re.S)
FENCE_RX = re.compile(r"^(`{3,}|~{3,})", re.M)


def check_fences(text: str) -> None:
    n = len(FENCE_RX.findall(text))
    if n % 2:
        raise SystemExit(f"ERROR: unbalanced code fences ({n} markers) — fix the "
                         f"Markdown before exporting, or the PDF will be wrong.")


def render_mermaid_blocks(text: str, workdir: Path) -> tuple[str, int]:
    """Replace every ```mermaid fence with an <img> holding a base64 PNG."""
    mmdc = find_mmdc()
    blocks = list(MERMAID_RX.finditer(text))
    if not blocks:
        return text, 0
    if mmdc is None:
        print(f"  ! {len(blocks)} mermaid block(s) found but mermaid-cli is missing; "
              f"they will appear as plain code blocks")
        print("    install:  npm install -g @mermaid-js/mermaid-cli")
        return text, 0

    cfg = workdir / "mermaid.json"
    # fontSize / -s interact: the PNG comes out at (natural px * scale). A wide,
    # flat diagram (perf curve, decision tree) is then printed at full page width,
    # so its printed size is (page width / natural width) * fontSize. 16px keeps
    # the smallest labels around 5pt in print, which is legible.
    cfg.write_text(
        '{"theme":"neutral","themeVariables":{"fontFamily":"Segoe UI, Microsoft YaHei, sans-serif",'
        '"fontSize":"16px","lineColor":"#4a4a4a"},"flowchart":{"useMaxWidth":false,'
        '"htmlLabels":true,"curve":"basis"},"sequenceDiagram":{"useMaxWidth":false}}',
        encoding="utf-8",
    )
    puppeteer = workdir / "puppeteer.json"
    puppeteer.write_text('{"args":["--no-sandbox","--disable-gpu"]}', encoding="utf-8")

    rendered = 0
    for i, m in enumerate(reversed(blocks)):  # reverse so offsets stay valid
        idx = len(blocks) - i
        src = workdir / f"diagram_{idx}.mmd"
        out = workdir / f"diagram_{idx}.png"
        src.write_text(m.group(1), encoding="utf-8")
        proc = subprocess.run(
            ["cmd", "/c", str(mmdc), "-i", str(src), "-o", str(out),
             "-c", str(cfg), "-p", str(puppeteer), "-b", "white", "-s", "3"],
            capture_output=True,
        )
        if proc.returncode != 0 or not out.exists():
            err = proc.stderr.decode("utf-8", "replace")
            err = re.sub(r"\x1b\[[0-9;]*m", "", err).strip().splitlines()
            head = err[1] if len(err) > 1 else (err[0] if err else "unknown error")
            print(f"  ! diagram #{idx} failed to render: {head[:120]}")
            continue
        warn = check_diagram_fit(out, idx, m.group(1))
        b64 = base64.b64encode(out.read_bytes()).decode("ascii")
        img = f'\n<img class="mermaid" src="data:image/png;base64,{b64}" alt="diagram {idx}">\n'
        text = text[: m.start()] + img + text[m.end():]
        rendered += 1
        if warn:
            print(warn)
    return text, rendered


def check_diagram_fit(png: Path, idx: int, source: str) -> str | None:
    """Warn when a diagram cannot be read at page width.

    A4 portrait minus our margins gives ~180mm x ~250mm of usable area. A
    top-down flowchart with many ranks comes out far taller than wide; CSS then
    shrinks it to fit the height, leaving it ~20mm wide and illegible. The fix
    is a content fix (``flowchart TD`` -> ``flowchart LR``), so we tell the
    author instead of silently producing an unreadable page.
    """
    try:
        import pymupdf
    except ImportError:
        return None
    try:
        rect = pymupdf.open(png)[0].rect
    except Exception:
        return None
    if rect.width <= 0:
        return None
    ratio = rect.height / rect.width
    if ratio <= 1.6:
        return None
    mm_wide_at_page_height = 250 / ratio  # width if scaled to the 250mm height cap
    hint = ""
    if "flowchart TD" in source:
        hint = " — consider  flowchart TD -> flowchart LR"
    return (f"  ! diagram #{idx} is {ratio:.1f}x taller than wide: only "
            f"{mm_wide_at_page_height:.0f}mm wide once fitted to the page height, "
            f"so it will be hard to read in print{hint}")


def build_toc(html_body: str, html_mod) -> str:
    """Build a two-level TOC from the h2/h3 ids that python-markdown generated."""
    items = re.findall(
        r'<h([23])[^>]*id="([^"]+)"[^>]*>(.*?)</h\1>', html_body, re.S
    )
    if not items:
        return ""
    rows = []
    for lvl, anchor, raw in items:
        title = re.sub(r"<[^>]+>", "", raw)
        title = re.sub(r"\s+", " ", title).strip()
        # the heading text already carries its own numbering; keep it short
        if len(title) > 78:
            title = title[:75] + "…"
        cls = "lvl3" if lvl == "3" else "lvl2"
        rows.append(f'<li class="{cls}"><a href="#{anchor}">{title}</a></li>')
    return (
        '<nav class="toc"><h2>目录</h2><ol>' + "".join(rows) + "</ol></nav>"
    )


HTML_SHELL = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head><body>
{title_block}
{toc}
{body}
</body></html>
"""


def md_to_html(md_path: Path, workdir: Path) -> tuple[str, dict]:
    try:
        import markdown
    except ImportError:
        raise SystemExit("ERROR: python-markdown missing. Install with:\n"
                         "  python -m pip install markdown pymdown-extensions pygments")

    raw = md_path.read_text(encoding="utf-8")
    check_fences(raw)

    stats = {"mermaid_total": len(MERMAID_RX.findall(raw)), "mermaid_ok": 0}
    raw, stats["mermaid_ok"] = render_mermaid_blocks(raw, workdir)

    html_mod = markdown.Markdown(extensions=MD_EXTENSIONS, extension_configs=MD_EXT_CONFIGS)
    body = html_mod.convert(raw)

    toc = build_toc(body, html_mod)
    first_h1 = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = re.sub(r"<[^>]+>", "", first_h1.group(1)).strip() if first_h1 else md_path.stem
    # the h1 becomes the title block, so drop it from the body
    body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S)

    title_block = (
        f'<div class="title-block"><h1>{title}</h1>'
        f'<div class="sub">AI Infra / 推理优化 / 分布式系统 · 校招与实习面试题库</div>'
        f'<div class="meta">源文件 {md_path.name}　·　'
        f'{stats["mermaid_total"]} 张 mermaid 图已渲染为图片　·　'
        f'导出工具 export_pdf.py</div></div>'
    )
    return HTML_SHELL.format(title=title, css=CSS, title_block=title_block,
                             toc=toc, body=body), stats


def html_to_pdf(html_path: Path, pdf_path: Path, chrome: Path) -> None:
    args = list(CHROME_ARGS) + [
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        html_path.as_uri(),
    ]
    proc = subprocess.run([str(chrome), *args], capture_output=True)
    # Chrome logs unrelated USB/device noise on Windows; only the file matters
    if not pdf_path.exists():
        err = proc.stderr.decode("utf-8", "replace")
        raise SystemExit(f"ERROR: Chrome did not write {pdf_path}\n{err[-1500:]}")


def export(md_path: Path, keep: bool) -> Path | None:
    if not md_path.exists():
        print(f"  skip: {md_path.name} not found")
        return None
    chrome = find_chrome()
    if chrome is None:
        raise SystemExit("ERROR: no Chrome/Edge found. Set CHROME_PATH=<chrome.exe>")

    # §7 of interview.md is generated from the .cu sources; refresh it first so the
    # PDF can never show code that build.sh does not compile.
    builder = HERE / "interview" / "build_kernels.py"
    if md_path.name == "interview.md" and builder.exists():
        proc = subprocess.run([sys.executable, str(builder), "--check"], capture_output=True)
        out = proc.stdout.decode("utf-8", "replace").strip().splitlines()
        print(f"  kernels: {out[-1] if out else 'builder produced no output'}")
        if proc.returncode != 0:
            print(proc.stderr.decode("utf-8", "replace")[-800:])
            raise SystemExit("ERROR: interview-code/build_kernels.py failed; "
                             "not exporting a PDF with stale code.")

    pdf_path = md_path.with_suffix(".pdf")
    workdir = Path(tempfile.mkdtemp(prefix="md2pdf_"))
    try:
        html, stats = md_to_html(md_path, workdir)
        html_path = workdir / (md_path.stem + ".html")
        html_path.write_text(html, encoding="utf-8")
        if keep:
            kept = md_path.with_suffix(".html")
            kept.write_text(html, encoding="utf-8")
            print(f"  kept intermediate HTML -> {kept.name}")
        html_to_pdf(html_path, pdf_path, chrome)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    kb = pdf_path.stat().st_size / 1024
    note = ""
    if stats["mermaid_total"]:
        note = f", mermaid {stats['mermaid_ok']}/{stats['mermaid_total']}"
    print(f"  OK  {md_path.name} -> {pdf_path.name}  ({kb:,.0f} KB{note})")
    return pdf_path


def main() -> int:
    argv = sys.argv[1:]
    keep = "--keep" in argv
    argv = [a for a in argv if not a.startswith("--")]

    docs = HERE.parent / "docs"          # 文档在 docs/，本脚本在 code/
    if "--all" in sys.argv:
        targets = sorted(docs.glob("ch0*.md")) + sorted(docs.glob("appendix-*.md")) + [
            HERE.parent / "README.md", docs / "cheatsheet.md", docs / "labs.md",
            docs / "labs-gpu.md", docs / "interview.md",
        ]
    elif argv:
        # 接受三种写法：绝对路径、相对当前目录、以及只给文件名（去 docs/ 找）
        targets = []
        for a in argv:
            p = Path(a)
            if p.is_absolute():
                targets.append(p)
            elif p.exists():
                targets.append(p)
            elif (docs / a).exists():
                targets.append(docs / a)
            else:
                targets.append(HERE / a)
    else:
        targets = [docs / "interview.md"]

    print(f"exporting {len(targets)} file(s) with {find_chrome().name}")
    made = [p for p in (export(t, keep) for t in targets) if p]
    print(f"\n{len(made)} PDF(s) written next to their Markdown source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
