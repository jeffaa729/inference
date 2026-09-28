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

/* ---- figures pulled in from papers / Wikimedia / textbooks ---- */
img:not(.mermaid) {
  display: block;
  margin: .9em auto .3em;
  max-width: 100%;
  max-height: 165mm;
  border: 1px solid var(--rule);
  border-radius: 4px;
  background: #fff;
  break-inside: avoid-page;
  page-break-inside: avoid;
}
p.figcap {
  text-align: center;
  color: var(--muted);
  font-size: .8em;
  line-height: 1.45;
  margin: 0 0 1em;
  break-before: avoid-page;
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
nav.toc li {
  display: flex; align-items: baseline; gap: .4em;
  margin: .1em 0; font-size: .88em; line-height: 1.45;
}
nav.toc li .t { flex: 1 1 auto; }
nav.toc li .pg {
  flex: 0 0 auto; color: var(--muted);
  font-variant-numeric: tabular-nums;
}
nav.toc li.lvl2 { font-weight: 600; }
nav.toc li.lvl3 { padding-left: 1.5em; color: #444; font-size: .82em; }
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


MD_TOC_RX = re.compile(r"<!-- TOC:BEGIN -->.*?<!-- TOC:END -->\s*", re.S)

# h2 = 1.35rem, h3 = 1.12rem at a 10.5pt root; Chrome renders these verbatim.
H2_PT, H3_PT, TITLE_PT = 1.35 * 10.5, 1.12 * 10.5, 1.9 * 10.5
SIZE_TOL = 0.35


def heading_list(html_body: str) -> list[tuple[int, str, str]]:
    """[(level, anchor, plain_title)] for every h2/h3, in document order."""
    import html as _html

    out = []
    for lvl, anchor, raw in re.findall(
        r'<h([23])[^>]*id="([^"]+)"[^>]*>(.*?)</h\1>', html_body, re.S
    ):
        # unescape so the title matches the glyphs the PDF actually contains
        title = _html.unescape(re.sub(r"<[^>]+>", "", raw))
        title = re.sub(r"\s+", " ", title).strip()
        out.append((int(lvl), anchor, title))
    return out


def build_toc(headings: list[tuple[int, str, str]], pages: dict[str, int] | None) -> str:
    """Render the menu: chapters only, each with its page number.

    Sub-sections stay out on purpose — the point is a one-glance map of the
    bank, and the PDF's own outline still carries the full heading tree.
    """
    chapters = [h for h in headings if h[0] == 2]
    if not chapters:
        return ""
    rows = []
    for lvl, anchor, title in chapters:
        short = title if len(title) <= 66 else title[:64] + "…"
        pg = pages.get(anchor) if pages else None
        num = f'<span class="pg">{pg}</span>' if pg else ""
        rows.append(f'<li class="lvl2"><a class="t" href="#{anchor}">{short}</a>{num}</li>')
    return '<nav class="toc"><h2>目录</h2><ol>' + "".join(rows) + "</ol></nav>"


def pdf_heading_pages(pdf_path: Path, headings: list[tuple[int, str, str]]) -> dict[str, int]:
    """Read the printed PDF back and map each heading anchor to its page number.

    Both lists are in document order, so this is a two-pointer walk: consume PDF
    lines whose text is a *prefix* of the heading we are looking for, which also
    stitches back together headings that wrapped onto a second line. Keying off
    the declared font sizes keeps the TOC's own small print and the title block
    out of the candidate set.
    """
    try:
        import pymupdf
    except ImportError:
        print("  ! pymupdf missing — TOC will have no page numbers")
        return {}

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

    doc = pymupdf.open(pdf_path)
    hits: list[tuple[int, int, str]] = []
    for pno in range(doc.page_count):
        for blk in doc[pno].get_text("dict").get("blocks", []):
            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                size = max(s["size"] for s in spans)
                lvl = 2 if abs(size - H2_PT) <= SIZE_TOL else (
                    3 if abs(size - H3_PT) <= SIZE_TOL else None)
                if lvl is None:
                    continue
                txt = "".join(s["text"] for s in spans).strip()
                if txt:
                    hits.append((pno + 1, lvl, txt))

    pages: dict[str, int] = {}
    hi = 0
    for lvl, anchor, title in headings:
        target = norm(title)
        acc, page, skipped = "", None, 0
        # Embedded SVG figures put their own labels into the PDF text layer, and
        # some land at h3 size — so skip anything that is not a prefix of the
        # heading we are looking for, and accept only an exact full match.
        while hi < len(hits) and skipped < 60:
            hp, hl, ht = hits[hi]
            if hl != lvl:
                hi += 1
                skipped += 1
                continue
            combined = acc + norm(ht)
            if acc and page is not None and hp != page:
                break                      # a wrapped heading never spans pages
            if target.startswith(combined):
                if page is None:
                    page = hp
                acc, hi, skipped = combined, hi + 1, 0
                if acc == target:
                    break
            elif acc:
                break                      # matched, then diverged: give up
            else:
                hi += 1                    # pure noise before the heading
                skipped += 1
        if acc == target and page is not None:
            pages[anchor] = page

    if len(pages) != len(headings):
        missing = [t for _, a, t in headings if a not in pages][:3]
        print(f"  ! heading/page match incomplete ({len(pages)}/{len(headings)})"
              f" — page numbers omitted; first misses: {missing}")
        return {}
    return pages



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


IMG_SRC_RX = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}


def inline_local_images(html_body: str, base: Path) -> tuple[str, int, list[str]]:
    """Turn ``<img src="relative/path">`` into a data URI.

    The intermediate HTML lives in a temp dir, so a relative path would never
    resolve at print time. Inlining also means the PDF is self-contained.
    """
    n, missing = 0, []

    def repl(m: re.Match) -> str:
        nonlocal n
        src = m.group(2)
        if src.startswith(("data:", "http://", "https://", "file:")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.exists():
            missing.append(src)
            return m.group(0)
        mime = MIME.get(path.suffix.lower())
        if mime is None:
            missing.append(src)
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        n += 1
        return f'{m.group(1)}data:{mime};base64,{b64}{m.group(3)}'

    return IMG_SRC_RX.sub(repl, html_body), n, missing


def md_to_html(
    md_path: Path, workdir: Path, pages: dict[str, int] | None = None
) -> tuple[str, dict, list[tuple[int, str, str]]]:
    try:
        import markdown
    except ImportError:
        raise SystemExit("ERROR: python-markdown missing. Install with:\n"
                         "  python -m pip install markdown pymdown-extensions pygments")

    raw = md_path.read_text(encoding="utf-8")
    check_fences(raw)
    # The Markdown carries its own menu for readers on GitHub; the PDF gets the
    # generated one instead (same content, plus real page numbers).
    raw = MD_TOC_RX.sub("", raw)

    stats = {"mermaid_total": len(MERMAID_RX.findall(raw)), "mermaid_ok": 0}
    raw, stats["mermaid_ok"] = render_mermaid_blocks(raw, workdir)

    html_mod = markdown.Markdown(extensions=MD_EXTENSIONS, extension_configs=MD_EXT_CONFIGS)
    body = html_mod.convert(raw)
    body, n_img, missing = inline_local_images(body, md_path.parent)
    if n_img:
        stats["images"] = n_img
    for miss in missing:
        print(f"  ! image not found (skipped): {miss}")

    headings = heading_list(body)
    toc = build_toc(headings, pages)
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
                             toc=toc, body=body), stats, headings


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
        # ---- pass 1: render without page numbers to learn the pagination ----
        html, stats, headings = md_to_html(md_path, workdir)
        html_path = workdir / (md_path.stem + ".html")
        html_path.write_text(html, encoding="utf-8")
        html_to_pdf(html_path, pdf_path, chrome)

        # ---- pass 2..n: inject page numbers, re-render until pagination settles ----
        pages = pdf_heading_pages(pdf_path, headings)
        for _ in range(3):
            if not pages:
                break
            if not any(pages.values()):
                break
            html, stats, _ = md_to_html(md_path, workdir, pages)
            html_path.write_text(html, encoding="utf-8")
            html_to_pdf(html_path, pdf_path, chrome)
            again = pdf_heading_pages(pdf_path, headings)
            if again == pages:
                print(f"  TOC page numbers settled ({len(pages)} entries)")
                break
            pages = again
        else:
            print("  ! TOC page numbers did not settle after 3 passes")

        if keep:
            kept = md_path.with_suffix(".html")
            kept.write_text(html, encoding="utf-8")
            print(f"  kept intermediate HTML -> {kept.name}")
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
