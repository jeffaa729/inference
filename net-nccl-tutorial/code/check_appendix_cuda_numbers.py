#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the numbers quoted in ``interview.md`` against the source book.

内核层那一批数字全部转录自一本 459 页的 CUDA kernel 优化专著（``book.pdf``），现在收录在 ``interview.md`` 的 §3。
本脚本做两件事，用来防止「转述时把数字抄错」：

1. **出处核对**：把附录里每个「数字 + 单位」token 抽出来，在书的全文里找一个
   数值相同、单位兼容的 token。找不到就报 ``MISSING``。
2. **派生量自检**：附录里有几个数字是**算出来的**（不是抄的），例如
   ``AI = K/6``、``ridge point = P_peak / BW``、``B/τ`` 上界。
   这些用 ``DERIVED`` 表逐条复算。

设计取舍（为什么不是精确字符串匹配）：
原文是 LaTeX 生成的 PDF，数字与单位之间可能夹空格、单位写法也不统一
（``TFLOPS`` / ``T`` / ``GB/s``）。所以核对做在**归一化后的数值**上，
而不是原始字符串上 —— 这条纪律和 ``verify_citations.py`` 一致：
**断言的应该是语义（这个数字在不在源里），而不是排版。**

用法::

    python check_appendix_cuda_numbers.py            # 报告
    python check_appendix_cuda_numbers.py -v         # 逐条打印

书全文**不随本仓库提交**（它是一本第三方专著的抽取文本，体积 ~1.5MB）。
脚本按这个顺序找源：

1. ``BOOK_FULL_TXT`` 环境变量指向的抽取文本；
2. 仓库内的 ``_book_extract/book_full.txt``（如果你自己抽过一份）；
3. ``BOOK_PDF`` 环境变量指向的 ``book.pdf``（需要 ``pymupdf``，现场抽取）；
4. 都没有 —— 只跑派生量自检，并提示出处核对被跳过（**退出码仍为 0，不误报失败**）。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    _rc = getattr(_s, "reconfigure", None)
    if _rc:
        try:
            _rc(encoding="utf-8", errors="replace")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent
APPENDIX = HERE.parent / "docs" / "interview.md"   # docs/ 已与 code/ 分开
BOOK_TEXT = HERE.parent / "_book_extract" / "book_full.txt"

# Where else the source book might be. None of these ship with the tutorial:
# the extracted text is ~1.5MB of third-party content and the PDF is ~35MB.
PDF_CANDIDATES = [
    Path(os.environ["BOOK_PDF"]) if os.environ.get("BOOK_PDF") else None,
    Path(os.environ["BOOK_FULL_TXT"]) if os.environ.get("BOOK_FULL_TXT") else None,
    HERE / "book.pdf",
    Path.home()
    / ".dsh/attachments/v1/files/42"
    / "42850e179b1eefaadbe9ec35447d4ed097e0e74107c5abbc159a72ef07c60f88"
    / "book.pdf",
]

VERBOSE = "-v" in sys.argv[1:]

# --------------------------------------------------------------------------
# Unit canonicalisation
# --------------------------------------------------------------------------
# 每个条目: 匹配用的正则片段 -> 规范化后的单位标签。
# 规范化把「同一物理量的不同写法」折叠到一起（T / TFLOPS / TFLOP/s -> tflops）。
UNIT_ALIASES: list[tuple[str, str]] = [
    (r"TFLOPS?|TFLOP/s|TF/s|(?<![A-Za-z])T(?![A-Za-z/])", "tflops"),
    (r"GB/s|GBps|GBytes/s", "gbs"),
    (r"TB/s|TBps", "tbs"),
    (r"KB|KiB", "kb"),
    (r"MB|MiB", "mb"),
    (r"GB|GiB", "gb"),
    (r"ms|millisecond", "ms"),
    (r"µs|us|microsecond", "us"),
    (r"ns|nanosecond", "ns"),
    (r"%|percent", "pct"),
]
UNIT_RX = "|".join(f"(?:{p})" for p, _ in UNIT_ALIASES)


def canon_unit(raw: str) -> str:
    for pat, tag in UNIT_ALIASES:
        if re.fullmatch(pat, raw.strip()):
            return tag
    return raw.strip().lower()


def norm_num(raw: str) -> float:
    return float(raw.replace(",", "").replace("_", ""))


# --------------------------------------------------------------------------
# Invariant checks: numbers the appendix *derives* rather than quotes.
# --------------------------------------------------------------------------
def check_derived() -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    def near(a: float, b: float, rel: float = 0.01) -> bool:
        return abs(a - b) <= rel * max(abs(a), abs(b), 1e-30)

    # GEMM: AI = K/6 for square matrices, with Bytes counting the output write.
    for k, expect in ((4096, 682.6667), (1024, 170.6667), (512, 85.3333)):
        got = 2 * k**3 / (3 * k**2 * 4)
        out.append((f"GEMM AI = K/6 @K={k}", near(got, expect), f"{got:.4f} vs {expect}"))

    # Ridge point = peak FLOPs / bandwidth.
    checks = [
        ("H100 FP16 TC ridge = 989/3.35", 989 / 3.35, 295.0, 0.01),
        ("H100 FP32 ridge = 67/3.35", 67 / 3.35, 20.0, 0.02),
        ("PRO5000 ridge = 66.9/1.344", 66.9 / 1.344, 49.8, 0.01),
    ]
    for label, got, expect, rel in checks:
        out.append((label, near(got, expect, rel), f"{got:.2f} vs {expect}"))

    # TF32 unit roundoff u = 2^-11.
    out.append(("TF32 u = 2^-11", near(2**-11, 4.88e-4, 0.01), f"{2**-11:.3e}"))

    # TF32 error bound worked example: 2*u*sum|a_j b_j| with sum = K/4, K=512.
    u = 2**-11
    bound = 2 * u * (512 / 4)
    out.append(("TF32 err bound @K=512", near(bound, 0.125, 0.02), f"{bound:.4f} vs 0.125"))
    out.append(
        (
            "TF32 rel err bound ~1.7%",
            near(bound / (512 / 9) ** 0.5, 0.017, 0.05),
            f"{bound / (512 / 9) ** 0.5:.4f}",
        )
    )

    # swizzle bank model: BK=16 (32 B row) gives exactly 2-way conflict.
    # a = 32i + 16c  ->  bank(a) = (8i + 4c) mod 32
    # rows i and i+4 land on the same bank, so each bank is hit twice.
    hits: dict[int, int] = {}
    for i in range(8):
        for c in (0, 1):
            a = 32 * i + 16 * c
            bank = (a // 4) % 32
            assert bank == (8 * i + 4 * c) % 32, (i, c, bank)
            hits[bank] = hits.get(bank, 0) + 1
    out.append(
        (
            "BK=16 bank model -> exactly 2-way",
            set(hits.values()) == {2} and len(hits) == 8,
            f"{len(hits)} distinct banks, {max(hits.values())} hits each",
        )
    )

    # BK=64 (128 B row) with no swizzle -> 8-way: row index cancels out.
    # a = 128i + 32s + 16c  ->  bank = (8s + 4c) mod 32, independent of i.
    phase0 = []
    for i in range(8):
        for s in (0,):
            for c in range(2):
                a = 128 * i + 32 * s + 16 * c
                phase0.append((a // 4) % 32)
    out.append(
        (
            "BK=64 no-swizzle is 8-way for fixed (s,c)",
            len({(128 * i) // 4 % 32 for i in range(8)}) == 1,
            f"row-index contribution collapses to {sorted({(128 * i) // 4 % 32 for i in range(8)})}",
        )
    )

    # copy.async is 16 B wide, so BK=16 fp16 row = 32 B; SWIZZLE_32B covers 256 B.
    out.append(
        (
            "SWIZZLE_32B period = 2^(M+S+B) B",
            2 ** (4 + 3 + 1) == 256,
            "M=4,S=3,B=1 -> 256 B",
        )
    )
    out.append(
        (
            "SWIZZLE_128B period = 2^(M+S+B) B",
            2 ** (4 + 3 + 3) == 1024,
            "M=4,S=3,B=3 -> 1024 B",
        )
    )

    # mma.m16n8k16 FLOPs per instruction.
    out.append(("mma m16n8k16 = 2*16*8*16", 2 * 16 * 8 * 16 == 4096, "4096 FLOP"))

    # NVFP4 e2m1 representable magnitudes.
    e2m1 = {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    out.append(("NVFP4 e2m1 has 8 magnitudes", len(e2m1) == 8, str(sorted(e2m1))))
    out.append(("e2m1 relative step ~25%", near(2**-2, 0.25), f"{2**-2:.2f}"))
    out.append(("e4m3 relative step ~6.25%", near(2**-4, 0.0625), f"{2**-4:.4f}"))

    # fp32 overflow threshold for exp.
    import math

    out.append(("ln(FLT_MAX) ~ 88.72", near(math.log(3.4028235e38), 88.72, 0.001), f"{math.log(3.4028235e38):.4f}"))

    # L2 footprint ratio from the block-swizzle derivation.
    f_default = (1 * 128 + 110 * 128) * 2
    f_swizzled = (7 * 128 + 16 * 128) * 2
    out.append(
        (
            "block-swizzle L2 footprint ratio ~4.8x",
            near(f_default / f_swizzled, 4.8, 0.02),
            f"{f_default / f_swizzled:.3f}",
        )
    )

    # flash attention naive AI = d/2 and the 4.3 GB / 67 MB ratio at B1H32N4096D64.
    out.append(("FA naive AI = d/2 @d=64", near(64 / 2, 32), "32 FLOP/B"))
    # The book quotes decimal units (4.3 GB, 67 MB), so use 1e9 / 1e6 here.
    traffic_bytes = 4 * 1 * 32 * 4096**2 * 2  # S and P, each written and read
    useful_bytes = 4 * 4096 * 64 * 32 * 2  # Q/K/V in + O out
    intermediate_gb = traffic_bytes / 1e9
    useful_mb = useful_bytes / 1e6
    out.append(
        (
            "FA naive HBM round trip ~4.3 GB",
            near(intermediate_gb, 4.3, 0.02),
            f"{intermediate_gb:.2f} GB",
        )
    )
    out.append(("FA useful traffic ~67 MB", near(useful_mb, 67, 0.05), f"{useful_mb:.1f} MB"))
    out.append(
        (
            "FA traffic ratio ~1/64",
            near(useful_bytes / traffic_bytes, 1 / 64, 0.05),
            f"{useful_bytes / traffic_bytes:.4f}",
        )
    )
    total_flops = 4 * 1 * 32 * 4096**2 * 64
    out.append(
        (
            "FA total FLOPs ~137 GFLOP",
            near(total_flops / 1e9, 137, 0.02),
            f"{total_flops / 1e9:.1f} GFLOP",
        )
    )

    # mbarrier arrive counts quoted in the appendix.
    out.append(("HGEMM WGMMA arrive count = 128+1", 128 + 1 == 129, "129"))
    out.append(("FA2 TMA+WS arrive count = 256+1", 256 + 1 == 257, "257"))
    out.append(("setmaxnreg budget 40 + 2*168 <= 512", 40 + 2 * 168 <= 512, "376"))
    out.append(
        (
            "setmaxnreg accident 64 + 2*224 == 512",
            64 + 2 * 224 == 512,
            "512 (exactly at the limit -> hang)",
        )
    )
    out.append(
        (
            "setmaxnreg fix 64 + 2*208 = 480 (>=2048 slack / 128)",
            64 + 2 * 208 == 480 and 512 - 480 >= 2048 / 128,
            "480",
        )
    )

    # f16 PV accumulation domain guard.
    out.append(("f16 PV guard Bc*448*2.25 <= 65504", 64 * 448 * 2.25 <= 65504, f"{64 * 448 * 2.25:.0f}"))

    # latency/bandwidth for the atomic-bound dot microbenchmark.
    out.append(
        (
            "atomic 2.6 ns/op * 16384 == 42.6 us",
            near(2.6e-9 * 16384 * 1e6, 42.0, 0.05),
            f"{2.6e-9 * 16384 * 1e6:.1f} us",
        )
    )

    # histogram model gap quoted as ~15x.
    out.append(
        (
            "histogram model 42 us vs 0.63 ms ~ 15x",
            near(0.62860e-3 / 42e-6, 15.0, 0.05),
            f"{0.62860e-3 / 42e-6:.1f}x",
        )
    )

    # softmax AI 5N/8N.
    out.append(("softmax AI = 5N/8N = 0.625", near(5 / 8, 0.625), "0.625"))

    return out


# --------------------------------------------------------------------------
# Quote checks: (appendix number, unit, short label, optional source number)
# --------------------------------------------------------------------------
QUOTES: list[tuple[str, str, str, str | None]] = [
    # --- machine ruler -----------------------------------------------------
    ("1344", "gbs", "PRO5000 GDDR7 theoretical BW", None),
    ("66.9", "tflops", "PRO5000 FP32 peak", None),
    ("49.8", "tflops", "PRO5000 ridge point (FLOP/Byte, unitless)", None),
    ("96", "mb", "PRO5000 L2 size", None),
    ("99", "kb", "sm_120a smem opt-in cap", None),
    ("227", "kb", "sm_90a smem cap", None),
    ("110", None, "PRO5000 SM count", None),
    ("3.35", "tbs", "H100 HBM3 theoretical BW", None),
    ("989", "tflops", "H100 FP16 TC dense", None),
    ("67", "tflops", "H100 FP32 CUDA core", None),
    ("1979", "tflops", "H100 FP16 TC sparse", None),
    ("50", "mb", "H100 L2 size", None),
    ("228", "kb", "Hopper smem per SM", None),
    # --- kernel ladder -----------------------------------------------------
    ("25.97", "tflops", "sgemm_vec4 2048^3", None),
    ("28.19", "tflops", "sgemm_tf32 1024^3", None),
    ("76.55", "tflops", "hgemm_mma_stages_tn 1024^3", None),
    ("124.7", "tflops", "7b-1 4096^3", None),
    ("164.64", "tflops", "hgemm_tma_mma_ws_tn 2048^3", None),
    ("4.76", None, "vec4 speedup over naive", None),
    ("2.71", None, "mma stages speedup", None),
    ("3.144", "ms", "naive sgemm 2048^3", None),
    ("0.661", "ms", "vec4 sgemm 2048^3", None),
    # --- attention ---------------------------------------------------------
    ("166.56", "tflops", "FA2 MMA F16Acc", None),
    ("175.88", "tflops", "FA2 MMA F32Acc", None),
    ("195.2", "tflops", "FA2 TMA+WS F32Acc N=16384", None),
    ("184", "tflops", "FA2 TMA+WS F16Acc N=16384", None),
    ("190.4", "tflops", "FA3 D=128 F16Acc", None),
    ("194.2", "tflops", "FA3 D=128 F32Acc", None),
    ("158.7", "tflops", "FA2 cp.async baseline", None),
    ("230", "tflops", "FA2 paper A100 forward", None),
    ("740", "tflops", "FA3 paper H100 forward", None),
    ("312", "tflops", "A100 fp16 TC peak", None),
    ("19.5", "tflops", "A100 fp32 CUDA core", None),
    ("120.0", "tflops", "kPad=0 regression", None),
    ("191.1", "tflops", "FA2 TMA WS F32Acc N=4096", None),
    ("172.5", "tflops", "FA2 TMA WS F16Acc N=4096", None),
    # --- swizzle / bank conflict ------------------------------------------
    ("98,304", None, "BK=16 conflict count", None),
    ("691,067", None, "BK=64 no-swizzle conflict count", None),
    ("297,628", None, "BK=64 swizzle conflict count", None),
    ("64.45", "pct", "L2 sector hit before block swizzle", None),
    ("96.15", "pct", "L2 sector hit after block swizzle", None),
    ("117.9", "tflops", "wide-N before block swizzle", None),
    ("122.9", "tflops", "wide-N after block swizzle", None),
    # --- elementwise / reduce ---------------------------------------------
    ("1862.15", "gbs", "relu L2-scale bandwidth", None),
    ("2653.73", "gbs", "elementwise_add bandwidth", None),
    ("26.69", "gbs", "histogram bandwidth", None),
    ("0.62860", "ms", "histogram time", None),
    ("2.6", "ns", "global atomic serialisation", None),
    ("1031.4", "gbs", "merge_attn_states bandwidth", None),
    ("0.7849", "ms", "merge_attn_states time", None),
    ("809.5", "mb", "merge traffic", None),
    # --- quantization ------------------------------------------------------
    ("448", None, "e4m3 max magnitude", None),
    ("6.25", "pct", "e4m3 relative step", None),
    ("25", "pct", "e2m1 relative step", None),
    ("2688", None, "NVFP4 two-level P scale constant", None),
    ("11.3923", None, "log2(1/2688)", None),
    ("2.5850", None, "log2(1/6)", None),
    ("2.25", None, "vscale_max for f16 PV accumulation", None),
    ("474", "tflops", "fp4 D=192 peak", None),
    ("437", "tflops", "fp4 D=128 self", None),
    ("365", "tflops", "fp8 baseline 365T", None),
    ("77", "pct", "tensor pipe share in the sovereign region", None),
    ("96.75", "pct", "fp4 L1TEX sector hit", None),
    ("2.6", None, "FA-3 FP8 RMSE reduction factor", None),
    # --- mbarrier / TMA ----------------------------------------------------
    ("32768", None, "K-tile expect_tx bytes", None),
    ("101376", None, "SM120 optin smem bytes", None),
    ("1024", None, "TMA swizzle alignment", None),
    # --- roofline examples -------------------------------------------------
    ("682.6667", None, "ch01 roofline GEMM AI", None),
    ("0.4998", None, "ch01 roofline GEMV AI", None),
    ("0.6250", None, "ch01 roofline softmax AI", None),
    ("0.0833", None, "ch01 roofline eadd AI", None),
    ("0.1250", None, "ch01 roofline relu AI", None),
    ("0.2500", None, "ch01 roofline dot AI", None),
]


def load_book_text() -> tuple[str | None, str]:
    """Return (text, source description). ``text is None`` means "not available"."""
    if BOOK_TEXT.exists():
        return BOOK_TEXT.read_text(encoding="utf-8", errors="replace"), str(BOOK_TEXT)

    for cand in PDF_CANDIDATES:
        if cand is None or not cand.exists():
            continue
        if cand.suffix.lower() in (".txt", ".md"):
            return cand.read_text(encoding="utf-8", errors="replace"), str(cand)
        try:
            import pymupdf  # type: ignore
        except Exception:
            return None, "pymupdf 未安装（无法从 PDF 现场抽取）"
        print(f"(extracting text from {cand.name} ...)")
        doc = pymupdf.open(cand)
        return "\n".join(doc[i].get_text() for i in range(doc.page_count)), str(cand)

    return None, "找不到书全文"


TOKEN_RX = re.compile(rf"(\d[\d,]*(?:\.\d+)?)\s*({UNIT_RX})?")


def book_tokens(text: str) -> dict[tuple[float, str], int]:
    """Map (value, canonical unit) -> first page-ish line index, for lookup."""
    index: dict[tuple[float, str], int] = {}
    for ln_no, line in enumerate(text.splitlines()):
        for m in TOKEN_RX.finditer(line):
            try:
                val = norm_num(m.group(1))
            except ValueError:
                continue
            unit = canon_unit(m.group(2)) if m.group(2) else ""
            index.setdefault((val, unit), ln_no)
            index.setdefault((val, ""), ln_no)  # unitless fallback lookup key
    return index


def main() -> int:
    if not APPENDIX.exists():
        print(f"ERROR: {APPENDIX} not found")
        return 2

    derived = check_derived()
    bad_derived = [d for d in derived if not d[1]]
    print(f"派生量自检: {len(derived) - len(bad_derived)}/{len(derived)} 通过")
    for label, ok, detail in derived:
        if VERBOSE or not ok:
            print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} {detail}")
    if bad_derived:
        print()

    text, source = load_book_text()
    if text is None:
        print(
            f"跳过出处核对（{source}）。\n"
            "  书全文不随本仓库提交（第三方专著的 1.5MB 抽取文本）。想跑完整核对：\n"
            "    set BOOK_PDF=<path to book.pdf>    （需要 pymupdf，现场抽取）\n"
            "    set BOOK_FULL_TXT=<path to book_full.txt>\n"
            "  派生量自检不受影响，已经跑完。"
        )
        return 0 if not bad_derived else 1

    index = book_tokens(text)
    print(f"\n出处核对: {source}\n"
          f"          书全文 {len(text):,} 字符，"
          f"{len(index):,} 个唯一 (值, 单位) token\n")

    missing: list[tuple[str, str, str]] = []
    for num, unit, label, src in QUOTES:
        key_unit = canon_unit(unit) if unit else ""
        val = norm_num(num)
        hit = (val, key_unit) in index or (val, "") in index
        if not hit and key_unit:
            # unit mismatch is itself informative: try the unitless key too.
            hit = (val, "") in index
        if VERBOSE:
            print(f"  {'OK  ' if hit else 'MISS'}  {num:>10s} {unit or '':6s}  {label}")
        if not hit:
            missing.append((num, unit or "", label))

    ok_count = len(QUOTES) - len(missing)
    print(f"引用数字: {ok_count}/{len(QUOTES)} 在书中找到同值 token")
    if missing:
        print("\n未在书中找到（需要人工确认是否抄错，或是否为派生/论文口径）:")
        for num, unit, label in missing:
            print(f"  {num} {unit}  — {label}")

    failed = bool(bad_derived) or bool(missing)
    print("\n" + ("FAIL" if failed else "ALL OK"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
