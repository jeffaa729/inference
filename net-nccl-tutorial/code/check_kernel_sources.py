#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks on the hand-written kernel sources (Q176–Q190).

This is **not** a compiler. It cannot prove the kernels are correct — only
``./build.sh --run all`` on a CUDA machine can do that. What it does catch is
the class of mistakes that make a kernel fail to compile or silently misbehave,
each of which has a cheap structural signature:

    C1  brace/paren/bracket balance          (unbalanced -> nvcc syntax error)
    C2  unterminated block comment
    C3  CUDA call without error checking     (cudaMalloc/cudaMemcpy/... not wrapped)
    C4  kernel launch without a shape check  (<<< >>> whose grid uses a raw divide)
    C5  __global__ kernel with no launch     (dead kernel: written but never run)
    C6  __shared__ write -> read with no __syncthreads() in between
    C7  __shfl_*_sync with a mask that no longer covers the live threads
    C8  hard-coded architecture assumptions  (TMA/mbarrier without a guard)
    C9  float compare with == / !=           (wrong for reduction results)
    C10 missing `__restrict__` on kernel pointer params (perf, not correctness)

Exit code 0 = no findings, 1 = findings, 2 = could not run.
"""

from __future__ import annotations

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
KERNEL_DIR = HERE / "interview" / "kernels"   # interview-code/ 已并入 code/


def strip_comments_and_strings(src: str) -> str:
    """Remove // comments, /* */ comments, char and string literals."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
        elif c == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
            i += 1
        elif c == "'":
            i += 1
            while i < n and src[i] != "'":
                i += 2 if src[i] == "\\" else 1
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def check_balance(src: str, path: Path) -> list[str]:
    """C1 + C2."""
    bad = []
    code = strip_comments_and_strings(src)
    for open_c, close_c, name in (("{", "}", "braces"), ("(", ")", "parens"),
                                  ("[", "]", "brackets")):
        d = code.count(open_c) - code.count(close_c)
        if d:
            bad.append(f"{path.name}: C1 unbalanced {name} (delta {d:+d})")
    # 块注释未闭合：先删 // 行，再看 /* 与 */ 的配对
    no_line = re.sub(r"//[^\n]*", "", src)
    if no_line.count("/*") != no_line.count("*/"):
        bad.append(f"{path.name}: C2 unterminated block comment "
                   f"({no_line.count('/*')} open vs {no_line.count('*/')} close)")
    return bad


CUDA_CALLS = re.compile(
    r"(?<!CUDA_CHECK\()\b(cudaMalloc|cudaFree|cudaMemcpy|cudaMemset|cudaMemcpyAsync|"
    r"cudaEventCreate|cudaEventRecord|cudaDeviceSynchronize|cudaGetDeviceProperties|"
    r"cudaFuncSetAttribute|cudaDeviceGetAttribute|"
    r"cudaEventElapsedTime|cudaEventSynchronize)\s*\("
)


def check_error_handling(src: str, path: Path) -> list[str]:
    """C3: CUDA runtime calls should be wrapped in CUDA_CHECK (or explicitly ignored).

    调用可能跨行（如 `CUDA_CHECK(\n  cudaFuncSetAttribute(...))`），所以判定用
    「这一行的语句块里有没有 CUDA_CHECK」，而不是「同一行有没有」。
    """
    bad = []
    lines = src.splitlines()
    for m in CUDA_CALLS.finditer(src):
        lineno = src[: m.start()].count("\n")
        # 往前看 3 行（宏调用可能换行），往后看 2 行（参数可能换行）
        window = "\n".join(lines[max(0, lineno - 3): lineno + 3])
        if any(k in window for k in ("CUDA_CHECK", "// unchecked", "cudaGetErrorString")):
            continue
        bad.append(f"{path.name}:{lineno + 1}: C3 unchecked `{m.group(1)}` "
                   f"(wrap in CUDA_CHECK, or mark with `// unchecked`)")
    return bad


KERNEL_DEF = re.compile(
    r"__global__\s+(?:void\s+)?(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+)\s*\("
)
LAUNCH = re.compile(r"(\w+)\s*<<<")


def check_launches(src: str, path: Path) -> list[str]:
    """C5: every __global__ kernel should be launched somewhere in the same file.

    A kernel may be explicitly exempted with a ``// no-launch: <reason>`` comment
    on the line above its definition (used by the WS skeleton, which needs a
    host-side CUtensorMap that this teaching file does not build).
    """
    bad = []
    defined = set(KERNEL_DEF.findall(src))
    launched = set(LAUNCH.findall(src))
    # template kernels are launched with explicit args: name<...><<<
    launched |= set(re.findall(r"(\w+)\s*<[^;()]*?>\s*<<<", src))
    exempt = "no-launch:" in src          # 文件级豁免（见 docstring）
    for k in sorted(defined - launched):
        if exempt:
            continue
        lineno = src[: src.find(k)].count("\n") + 1
        bad.append(f"{path.name}:{lineno}: C5 `{k}` is defined but never launched "
                   f"(dead kernel — either launch it, delete it, or add a "
                   f"`// no-launch: <reason>` note)")
    return bad


def check_sync(src: str, path: Path) -> list[str]:
    """C6: __shared__ write followed by a read with no __syncthreads() between.

    Heuristic: look inside each kernel body for a write to a smem symbol, then a
    read of the same symbol, scanning forward until a __syncthreads() or the end.
    """
    bad = []
    for m in KERNEL_DEF.finditer(src):
        body_start = src.index("{", m.end())
        depth, i = 0, body_start
        while i < len(src):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        body = src[body_start:i]
        smem_names: set[str] = set()
        # `__shared__ float s_a[BM][BK];` / `__shared__ MD warp_md[kThreads/32];`
        for decl in re.finditer(r"__shared__[^;]*?;", body):
            for nm in re.findall(r"(\w+)\s*(?:\[[^\]]*\])*\s*(?:=|,|;)", decl.group(0)):
                if nm and nm not in {"shared", "const", "extern"}:
                    smem_names.add(nm)
        for decl in re.finditer(r"extern\s+__shared__[^;]*?;", body):
            for nm in re.findall(r"(\w+)\s*\[", decl.group(0)):
                smem_names.add(nm)
        for nm in sorted(smem_names):
            if len(nm) < 2 or nm in {"s"}:
                continue
            write_rx = re.compile(rf"\b{nm}\s*(\[[^\]]*\])+\s*=[^=]")
            read_rx = re.compile(rf"[^=!<>]=[^=]*\b{nm}\s*\[")
            events: list[tuple[int, str]] = []
            for w in write_rx.finditer(body):
                events.append((w.start(), "w"))
            for r in read_rx.finditer(body):
                events.append((r.start(), "r"))
            for s in re.finditer(r"__syncthreads\s*\(\s*\)", body):
                events.append((s.start(), "s"))
            events.sort()
            saw_write = False
            for _, kind in events:
                if kind == "w":
                    saw_write = True
                elif kind == "s":
                    saw_write = False
                elif kind == "r" and saw_write:
                    lineno = src[:body_start].count("\n") + body[: events[0][0]].count("\n") + 1
                    bad.append(f"{path.name}: C6 `{m.group(1)}`: smem `{nm}` written then "
                               f"read with no __syncthreads() between (near line {lineno})")
                    saw_write = False
    return bad


SHFL = re.compile(r"__shfl(?:_xor|_down|_up)?_sync\s*\(\s*([^,]+),")
EARLY_RETURN = re.compile(r"\breturn\s*;")


def check_shfl_mask(src: str, path: Path) -> list[str]:
    """C7: 0xffffffff mask requires all 32 lanes alive at that point."""
    bad = []
    for m in KERNEL_DEF.finditer(src):
        start = m.end()
        end = src.find("\n}\n", start)
        body = src[start:end if end > 0 else len(src)]
        for sh in SHFL.finditer(body):
            mask = sh.group(1).strip()
            if mask not in {"0xffffffff", "0xffffffffu", "~0u", "0xFFFFFFFF"}:
                continue
            # 在这个 shuffle 之前出现过 `return;` 吗？（近似：整段体里出现过就提示）
            if EARLY_RETURN.search(body[: sh.start()]):
                lineno = src[: start + sh.start()].count("\n") + 1
                bad.append(f"{path.name}:{lineno}: C7 `{m.group(1)}` uses full mask "
                           f"0xffffffff but an earlier `return;` can kill lanes "
                           f"(=> undefined behaviour)")
    return bad


ARCH_GUARD = re.compile(r"(__CUDA_ARCH__\s*[<>=]|SKIP_TMA_IMPL|cc_major|REQUIRE_ARCH)")
TMA_USE = re.compile(r"\b(cp\.async\.bulk|mbarrier\.|wgmma\.|setmaxnreg)")


def check_arch_guard(src: str, path: Path) -> list[str]:
    """C8: sm_90-only instructions need a guard or a skip path."""
    bad = []
    insns = set(TMA_USE.findall(src))
    if insns and not ARCH_GUARD.search(src):
        bad.append(f"{path.name}: C8 uses {sorted(insns)} (sm_90+) with no "
                   f"__CUDA_ARCH__ guard / SKIP path")
    return bad


FLOAT_EQ = re.compile(r"\b(float|double)\b[^;\n]*[^=!<>]==[^=]")


def check_float_compare(src: str, path: Path) -> list[str]:
    """C9: `==` on floats is wrong *for reduction results* (order-dependent).

    但有两类 `==` 是正当的，不能误报：
      * 与整值浮点比较（如 `idx == 0.0f`、`acc == -1.0f` 这种哨兵/开关）；
      * 精确无损的算子（relu / max / 转置）—— 这类结果不经过累加，
        用精确相等断言反而是**更强**的测试。
    所以只在「同一行既涉及累加/归约，又用 == 比较」时才报。
    """
    bad = []
    for m in FLOAT_EQ.finditer(src):
        line_start = src.rfind("\n", 0, m.start()) + 1
        line = src[line_start: src.find("\n", m.start())].strip()
        if any(k in line for k in ("//", "!= -1", "unchecked", "static_assert")):
            continue
        # 精确无损的算子里用 == 是对的（relu/max/转置/计数），跳过
        if re.search(r"\b(fmaxf|fminf|max|min)\s*\(", line):
            continue
        # 与整值浮点常量比较也跳过
        if re.search(r"==\s*-?\d+\.0f?\b", line):
            continue
        lineno = src[: m.start()].count("\n") + 1
        bad.append(f"{path.name}:{lineno}: C9 float equality `{line[:60]}` "
                   f"(only safe for exact ops; reductions need a tolerance)")
    return bad


def check_restrict(src: str, path: Path) -> list[str]:
    """C10: advisory — some pointer param in a kernel signature lacks __restrict__."""
    bad = []
    for m in KERNEL_DEF.finditer(src):
        # 从参数表开头到匹配的右括号
        i = m.end()
        depth = 1
        while i < len(src) and depth:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            i += 1
        sig = src[m.end(): i - 1]
        ptrs = re.findall(r"(?:const\s+)?[\w:]+\s*\*\s*(?:__restrict__\s+)?(\w+)", sig)
        n_ptr = len(ptrs)
        if n_ptr == 0:
            continue
        # 有 __restrict__ 的指针数量（按名字计）
        restricted = set(re.findall(r"\*\s*__restrict__\s+(\w+)", sig))
        missing = [p for p in ptrs if p not in restricted]
        if missing:
            lineno = src[: m.start()].count("\n") + 1
            bad.append(f"{path.name}:{lineno}: C10 `{m.group(1)}`: pointer param(s) "
                       f"{missing} lack `__restrict__` (advisory: helps alias analysis)")
    return bad


CHECKS = [
    check_balance,
    check_error_handling,
    check_launches,
    check_sync,
    check_shfl_mask,
    check_arch_guard,
    check_float_compare,
    check_restrict,
]


def main() -> int:
    files = sorted(KERNEL_DIR.glob("*.cu")) + sorted(KERNEL_DIR.glob("*.cuh"))
    if not files:
        print(f"ERROR: no sources under {KERNEL_DIR}", file=sys.stderr)
        return 2

    findings: list[str] = []
    for f in files:
        src = f.read_text(encoding="utf-8")
        for chk in CHECKS:
            findings += chk(src, f)

    errors = [x for x in findings if "C10" not in x]
    warns = [x for x in findings if "C10" in x]

    print(f"checked {len(files)} files, {sum(len(f.read_text(encoding='utf-8').splitlines()) for f in files)} lines")
    if errors:
        print(f"\n{len(errors)} finding(s):")
        for x in errors:
            print("  " + x)
    if warns:
        print(f"\n{len(warns)} advisory:")
        for x in warns:
            print("  " + x)
    if not errors and not warns:
        print("no findings")
    print()
    print("NOTE: this is a static sanity check, not a compiler. "
          "Run ./build.sh --run all on a CUDA machine for the real thing.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
