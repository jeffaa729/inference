#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Embed ``../interview/kernels/*.cu`` into §7 of ``interview.md``.

The interview bank's §7 must show the **complete** code, and that code must be
the same code ``build.sh`` compiles — otherwise the PDF silently drifts from
the sources. So the Markdown is *generated* from the ``.cu`` files:

    ../interview/kernels/q176_vecadd.cu  ──┐
    ../interview/kernels/q177_reduce.cu  ──┤ build_kernels.py
                       ...                   ├─> interview.md §7 fenced blocks
    ../interview/kernels/q190_merge_attn.cu┘

Each kernel file is self-documenting: it starts with a ``// qNNN_*.cu — <title>``
comment, then the题面, then its 解析要点, then the code. The embedder keeps the
whole file verbatim (```cuda fence) so what you read is what compiles.

    python build_kernels.py            # regenerate §7 in interview.md
    python build_kernels.py --check    # verify §7 is up to date (exit 1 if not)
    python build_kernels.py --list     # list the kernel files and their Q numbers
"""

from __future__ import annotations

import hashlib
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

HERE = Path(__file__).resolve().parent          # .../code/interview
REPO = HERE.parent.parent                        # 仓库根（code/ 的上一层）
KERNEL_DIR = HERE / "kernels"
INTERVIEW = REPO / "docs" / "interview.md"        # 文档在 docs/

BEGIN = "<!-- BEGIN GENERATED SECTION-7 KERNELS (build_kernels.py) -->"
END = "<!-- END GENERATED SECTION-7 KERNELS -->"

# 每个 kernel 文件对应的 Q 号由文件名前缀决定（q176_vecadd.cu -> 176）
Q_FROM_NAME = re.compile(r"^q(\d+)_(.+)\.cu$")


def kernel_files() -> list[tuple[int, str, Path]]:
    out = []
    for p in sorted(KERNEL_DIR.glob("q*.cu")):
        m = Q_FROM_NAME.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), p))
    return out


def read_title(cu: Path) -> str:
    """First line looks like: ``// q176_vecadd.cu — Q176：向量加法 c = a + b``."""
    first = cu.read_text(encoding="utf-8").splitlines()[0]
    m = re.search(r"—\s*(?:Q\d+：)?(.*)$", first)
    return m.group(1).strip() if m else cu.stem


# 每题的难度标签与「一句话考点」，写在这里而不是 .cu 里 —— 这样 .cu 保持纯粹的
# 可编译单元，题面元数据集中在生成器里。
META: dict[int, tuple[str, str]] = {
    176: ("[手撕]", "一线程一元素 / float4 二段式守卫 / grid-stride —— 三个版本对照"),
    177: ("[手撕]", "XOR 蝶形归约 + block 两级归约：步数账、单位元、广播那一步为什么不能省"),
    178: ("[手撕]", "原子版 vs 两级归约 vs vec4 —— 用数字证明「瓶颈不是访存指令数」"),
    179: ("[手撕]", "fmaxf 而不是 if/else；融合省的是流量，不是 launch"),
    180: ("[手撕]", "原子代价模型与 smem 私有化的「次数账」（T > B 才有收益）"),
    181: ("[手撕]", "transpose + padding：bank conflict 的算术与「口径决定加速比」"),
    182: ("[手撕]", "两趟 vs 一趟（单 pass）与 Welford；减少流量 vs 减少指令的区别"),
    183: ("[手撕]", "tiled SGEMM：三个 AI 公式、两道 __syncthreads、`/` 与 `%` 的索引法则"),
    184: ("[手撕]", "寄存器双缓冲 + XOR swizzle（含对合性自检与 padding 对照）"),
    185: ("[手撕]", "cp.async 多级流水：commit/wait_group 语义、wait 后为什么还要同步"),
    186: ("[手撕]", "mma.sync m16n8k16 + ldmatrix 手写 HGEMM（对照 WMMA 版）"),
    187: ("[hard]", "TMA + mbarrier + WS 骨架：arrive count 怎么算、账本不闭合的两种后果"),
    188: ("[手撕]", "naive / safe / online 三版 softmax，含「溢出用例」与带宽对照"),
    189: ("[手撕]", "FlashAttention 内层循环：online rescale 五步 + 与朴素版的显存账对照"),
    190: ("[hard]", "merge_attn_states：LSE 合并、两套口径不可混搭、空段退化"),
}

# ---------------------------------------------------------------------------
# LeetCUDA 溯源：每题都能在参考仓库里找到对应的实现。
#
# 参考仓库：https://github.com/xlite-dev/LeetCUDA
# 本地 checkout：`C:\Users\Jeff\Documents\GitHub\LeetCUDA`
# **引用锚点：HEAD `6c86259`（2026-09-22）** —— 行号会漂，符号名稳定。
#   （写这份材料期间上游从 e831d97 前进到 6c86259；对本节引用的文件，唯一的功能改动是
#    `hgemm.cuh` 的 `hgemm_tma_mma_ws_tn` 把 `__launch_bounds__(kNumThreads)` 改成了
#    `__launch_bounds__(kNumThreads, 1)` —— 正是 setmaxnreg 要求的那条约束。）
#
# 这张表有两个用途：
#   1) 读本节的代码时，知道去参考仓库里对哪一份实现；
#   2) 本节的宏命名与工具函数刻意与参考仓库的 `kernels/interview/common.cuh`
#      对齐（FLOAT4 / CP_ASYNC_* / LDMATRIX_* / HMMA16816 / permuted / SwizzleBMS /
#      warpgroup_reg_* / make_smem_desc / tma_*），所以在两边看到的写法是同一套。
# ---------------------------------------------------------------------------
LEETCUDA_REPO = "https://github.com/xlite-dev/LeetCUDA"
LEETCUDA_HEAD = "6c86259"

# Q -> [(参考文件, 该文件里的对应符号, 一句话关系)]
PROVENANCE: dict[int, list[tuple[str, str, str]]] = {
    176: [
        ("kernels/relu/relu.cu", "relu / relu_vec4", "向量化的边界守卫"),
        ("kernels/elementwise/elementwise.cu", "elementwise_add / elementwise_add_vec4",
         "二段式守卫 + 尾部标量处理的标准写法"),
    ],
    177: [
        ("kernels/interview/base.cuh", "warp_reduce_sum / warp_reduce_max / block_reduce_sum / block_reduce_all",
         "Phase 1a/1b：本节的归约结构与之一致"),
        ("kernels/reduce/block_all_reduce.cu", "block_all_reduce 各版本",
         "独立文件版，含更多归约变体"),
    ],
    178: [
        ("kernels/interview/base.cuh", "dot / dot_vec4",
         "Phase 2：同址 atomicAdd 的写法与「vec4 不提速」的结论"),
        ("kernels/dot-product/dot_product.cu", "dot_product 各版本", "独立文件版"),
    ],
    179: [
        ("kernels/gelu/gelu.cu", "gelu 变体", "激活函数的多种近似"),
        ("kernels/interview/base.cuh", "relu / elementwise_add", "Phase 2 的融合写法"),
    ],
    180: [
        ("kernels/histogram/histogram.cu", "histogram", "全局原子版（对照 smem 私有化）"),
        ("kernels/interview/base.cuh", "histogram", "Phase 2 里的同名 kernel"),
    ],
    181: [
        ("kernels/mat-transpose/mat_transpose.cu", "mat_transpose / mat_transpose_padded",
         "padding 打散 bank conflict 的原始实现"),
        ("kernels/interview/base.cuh", "mat_transpose / mat_transpose_padded",
         "Phase 5：Bank Conflict 专题"),
        ("kernels/swizzle/mat_trans_swizzle.cu", "swizzle 版转置",
         "用 XOR swizzle 替代 padding"),
    ],
    182: [
        ("kernels/rms-norm/rms_norm.cu", "rms_norm / rms_norm_vec4", "独立文件版"),
        ("kernels/layer-norm/layer_norm.cu", "layer_norm / layer_norm_vec4", "独立文件版"),
        ("kernels/interview/base.cuh", "rms_norm / rms_norm_vec4 / layer_norm / layer_norm_vec4",
         "Phase 3b/3c：1-pass RMSNorm vs 2-pass LayerNorm"),
    ],
    183: [
        ("kernels/sgemm/sgemm.cu", "sgemm 阶梯（naive -> tile -> vec4）", "本节三级阶梯的参考实现"),
        ("kernels/interview/sgemm.cuh", "Phase 7a 的 SGEMM", "面试精简版"),
    ],
    184: [
        ("kernels/interview/common.cuh", "permuted<kColStride,kStep> / SwizzleBMS<B,M,S>",
         "★ 本节的 swizzle 就取自这里（三种档位的位运算公式逐字对齐）"),
        ("kernels/swizzle/hgemm_mma_swizzle.cu", "swizzle 版 HGEMM",
         "swizzle 在真实 GEMM 里的用法（含 register 双缓冲）"),
        ("kernels/swizzle/mma_simple_swizzle.cu", "最小 swizzle 示例", "先看这个再看 GEMM"),
        ("kernels/swizzle/print_swizzle_layout.py", "swizzle 布局打印",
         "把 swizzle 后的 bank 分布打出来，验证「到底降了几路」"),
    ],
    185: [
        ("kernels/sgemm/sgemm_async.cu", "cp.async 多 stage 版 SGEMM", "★ 本节流水的参考实现"),
        ("kernels/sgemm/sgemm_wmma_tf32_stage.cu", "TF32 WMMA + stage", "同一套流水的 Tensor Core 版"),
        ("kernels/interview/common.cuh", "CP_ASYNC_CG / CP_ASYNC_COMMIT_GROUP / CP_ASYNC_WAIT_GROUP",
         "语义注释（wait_group N = 允许保留 N 组未完成）取自这里"),
    ],
    186: [
        ("kernels/interview/hgemm.cuh", "Phase 7b-d 的 HGEMM 全部阶梯",
         "★ 本节的手写 mma 版与之一致（TN 布局 + ldmatrix + multistage）"),
        ("kernels/interview/common.cuh", "HMMA16816 / HMMA16816F32 / LDMATRIX_X4 / LDMATRIX_X2 / LDMATRIX_X2_T",
         "★ 本节的 PTX 宏与这里的定义逐字一致"),
    ],
    187: [
        ("kernels/interview/common.cuh", "make_smem_desc / tma_load_2d / tma_arrive_expect_tx / warpgroup_reg_*",
         "★ descriptor 位域布局与 TMA/mbarrier helper 取自这里"),
        ("kernels/interview/hgemm.cuh", "WGMMA m64n128k16 + TMA MMA WS",
         "本骨架对应的完整实现"),
        ("kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu", "warp specialization 的教学版",
         "在 sm_80 上就能理解的 WS 结构（不含 TMA）"),
    ],
    188: [
        ("kernels/interview/base.cuh",
         "softmax_per_token / safe_softmax_per_token / online_safe_softmax_per_token / warp_reduce_md / MD",
         "★ 三级递进的原始实现与 `MD` 结构体（本节的 md_merge 与之同构）"),
        ("kernels/softmax/softmax.cu", "softmax 各版本", "独立文件版"),
    ],
    189: [
        ("kernels/interview/flash_attn.cuh",
         "FA2 MMA cp.async split-Q / FA2 TMA MMA WS / FA3 dual-consumer / FA Split-D",
         "★ 本节内层循环对应的完整实现（含 split-Q 与 WS 版本）"),
        ("kernels/interview/ffpa_attn.cuh", "FFPA：fp8/fp4 量化注意力",
         "再往前一步：scale 折叠与 split-D 的工程实现"),
    ],
    190: [
        ("kernels/interview/base.cuh", "merge_attn_states（kernel）",
         "★ LSE 合并的原始实现（本节的公式与之一致）"),
        ("kernels/interview/notes-v2.cu", "merge_attn_states 的 test/bench",
         "完整测试：跑 --help 可以看到 CLI 入口"),
    ],
}


def sha(cu: Path) -> str:
    return hashlib.sha256(cu.read_bytes()).hexdigest()[:12]


def rel_from_docs(p: Path) -> str:
    """仓库内路径 -> 相对 docs/ 的链接（interview.md 在 docs/ 下）。"""
    import os
    try:
        return str(p.resolve().relative_to(INTERVIEW.parent.resolve())).replace("\\", "/")
    except ValueError:
        return os.path.relpath(p.resolve(), INTERVIEW.parent.resolve()).replace("\\", "/")


def build_section() -> str:
    files = kernel_files()
    if not files:
        raise SystemExit(f"no kernel files under {KERNEL_DIR}")

    parts: list[str] = [BEGIN, ""]
    parts += [
        "### §7.0 LeetCUDA 溯源（每题对应参考仓库的哪份实现）",
        "",
        "| 题 | 本节的实现 | LeetCUDA 参考实现（文件 · 符号） |",
        "|---|---|---|",
    ]
    for q, name, cu in files:
        refs = PROVENANCE.get(q, [])
        cells = []
        for path, sym, rel in refs:
            cells.append(f"`{path}` · `{sym}`　— *{rel}*")
        parts.append(
            f"| **Q{q}** | [`{cu.name}`]({rel_from_docs(cu)}) | "
            + "<br>".join(cells) + " |"
        )
    parts.append("")

    for q, name, cu in files:
        src = cu.read_text(encoding="utf-8").rstrip("\n")
        tag, gist = META.get(q, ("[手撕]", ""))
        n_lines = len(src.splitlines())
        refs = PROVENANCE.get(q, [])
        ref_line = ""
        if refs:
            links = "、".join(f"`{p}`（`{s}`）" for p, s, _ in refs[:2])
            more = f" 等 {len(refs)} 处" if len(refs) > 2 else ""
            ref_line = f"> **参考实现**：{links}{more}\n"
        parts += [
            f"**Q{q} `{tag}` ★ {read_title(cu)}**",
            "",
            f"> **考点**：{gist}",
            f"> **源码**：[`{rel_from_docs(cu)}`]({rel_from_docs(cu)})"
            f"（{n_lines} 行，sha256 `{sha(cu)}`）",
            "",
            "```cuda",
            src,
            "```",
            "",
        ]
    parts.append(END)
    return "\n".join(parts)


def patch(text: str, section: str) -> str:
    # 从生成的 section 里读出题号区间，保证 header 与实际内容一致
    qs = [int(m) for m in re.findall(r"^\*\*Q(\d+) ", section, re.M)]
    rng = f"（Q{qs[0]}–Q{qs[-1]}，{len(qs)} 题）" if qs else ""

    if BEGIN in text and END in text:
        i = text.index(BEGIN)
        j = text.index(END) + len(END)
        return text[:i] + section + text[j:]
    # first run: replace everything after the §7 header line
    m = re.search(r"^## §7 .*$", text, re.M)
    if not m:
        raise SystemExit("interview.md has no '## §7 ' header to anchor the section")
    nxt = re.search(r"^## §8 .*$", text[m.end():], re.M)
    end = m.end() + (nxt.start() if nxt else len(text) - m.end())
    return text[: m.start()] + f"## §7 CUDA 与算子手撕题{rng}\n\n" + section + "\n\n---\n\n" + text[end:]


def main() -> int:
    args = sys.argv[1:]
    files = kernel_files()

    if "--list" in args:
        for q, name, cu in files:
            print(f"Q{q:<4} {cu.name:<28} {len(cu.read_text(encoding='utf-8').splitlines()):>4} lines")
        return 0

    section = build_section()
    original = INTERVIEW.read_text(encoding="utf-8")
    updated = patch(original, section)

    if "--check" in args:
        if original == updated:
            print(f"§7 up to date ({len(files)} kernels embedded)")
            return 0
        print("§7 is STALE — run: python build_kernels.py")
        return 1

    INTERVIEW.write_text(updated, encoding="utf-8")
    total = sum(len(cu.read_text(encoding='utf-8').splitlines()) for _, _, cu in files)
    print(f"embedded {len(files)} kernels ({total} lines of CUDA) into {INTERVIEW.name} §7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
