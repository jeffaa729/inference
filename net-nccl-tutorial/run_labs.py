#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-command lab suite runner for a rented GPU instance.

Collects the GPU-dependent labs from labs-gpu.md into a timestamped results
directory so nothing is lost when the instance is released.

    python run_labs.py list                 # show the plan for this machine
    python run_labs.py all                  # run everything that fits
    python run_labs.py core                 # A2 + A5 (the highest-value pair)
    python run_labs.py run a2 a3 a5         # pick specific labs
    python run_labs.py collect              # tar up results/ for download
    python run_labs.py markdown             # render results/SUMMARY.md from CSVs

Design notes
------------
* Every command is also printed and appended to ``results/commands.sh`` so the
  session is reproducible after the instance is gone.
* Failures do not abort the suite by default (``--fail-fast`` changes that):
  on a metered instance it is better to collect the labs that DO work.
* Nothing here needs network access except the model downloads for the vLLM
  labs, which are skipped when the model is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc is None:
            continue
        try:
            rc(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()


# ---------------------------------------------------------------- machine probe


def gpu_count() -> int:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return 0
    try:
        p = subprocess.run([exe, "--query-gpu=index", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return len([ln for ln in p.stdout.splitlines() if ln.strip()])
    except Exception:
        return 0


def has_rdma() -> bool:
    return Path("/sys/class/infiniband").is_dir() and any(
        Path("/sys/class/infiniband").iterdir())


def torchrun_available() -> bool:
    return shutil.which("torchrun") is not None


# ---------------------------------------------------------------- lab definitions

# id -> (title, min_gpus, needs, shell command builder)
def _tr(nproc: int, script: str, extra: str = "") -> str:
    return f"torchrun --nproc_per_node={nproc} {script} {extra}".strip()


LABS: dict[str, dict] = {
    "a1": dict(title="A1 拓扑与冒烟", min_gpus=1, needs="vLLM + 模型",
               note="打印拓扑；用 2 卡起一次 0.5B 模型确认能跑",
               shell="vllm serve Qwen/Qwen2.5-0.5B --tensor-parallel-size {tp} "
                     "--max-model-len 2048"),
    "a2": dict(title="A2 all-reduce 实测 vs 手算 ★", min_gpus=2, needs="torch",
               note="大小扫描，产出 CSV + 屏幕表格",
               shell=_tr("{gpus}", "allreduce_bench.py",
                         "--sizes 16K,64K,256K,1M,4M,16M,64M --out results/ar_scan.csv "
                         "--tag scan --report-backend")),
    "a3": dict(title="A3 custom AR vs NCCL ★", min_gpus=2, needs="vLLM + 模型",
               note="对比 TTFT/TPOT；预期 TPOT 差距 > TTFT 差距",
               shell="vllm bench latency --model Qwen/Qwen2.5-0.5B "
                     "--tensor-parallel-size {gpus} --input-len 128 --output-len 128"),
    "a4": dict(title="A4 ring vs tree crossover", min_gpus=2, needs="torch + NCCL",
               note="每个算法各扫一遍大小",
               shell=_tr("{gpus}", "allreduce_bench.py",
                         "--sizes 16K,256K,4M,64M --algo Ring --tag algo=Ring "
                         "--out results/ar_algo.csv")),
    "a5": dict(title="A5 通信正确性 + 拓扑", min_gpus=2, needs="torch",
               note="all_reduce/all_gather/reduce_scatter/broadcast/send-recv",
               shell=_tr("{gpus}", "verify_collectives.py", "--json results/verify.json")),
    "b1": dict(title="B1 custom AR 的 size 上限 ★", min_gpus=8, needs="8 卡 + NVLink",
               note="扫到 1 MiB，找 256 KiB 附近的后端切换",
               shell=_tr("{gpus}", "allreduce_bench.py",
                         "--sizes 16K,64K,128K,192K,256K,320K,512K,1M "
                         "--tag size-limit --report-backend --out results/ar_sizelimit.csv")),
    "b2": dict(title="B2 TP 缩放曲线", min_gpus=4, needs="torch",
               note="TP=2/4/8 各跑一次，看小消息延迟如何随 N 增长",
               shell=_tr("{gpus}", "allreduce_bench.py",
                         "--sizes 16K,1M,64M --tag tp{gpus} --out results/ar_scale.csv")),
    "b3": dict(title="B3 真实负载 TP 缩放", min_gpus=4, needs="vLLM + 7B 模型",
               note="bench throughput，找最优 TP 拐点",
               shell="vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct "
                     "--tensor-parallel-size {gpus} --num-prompts 200 "
                     "--input-len 512 --output-len 128"),
    "b5": dict(title="B5 nsys 通信占比 ★", min_gpus=4, needs="nsys + vLLM",
               note="产出「通信占比 X%」这个数字",
               shell="nsys profile -t cuda,nvtx --cuda-graph-trace=node "
                     "-o results/prof -f true vllm bench latency "
                     "--model Qwen/Qwen2.5-0.5B --tensor-parallel-size {gpus} "
                     "--input-len 128 --output-len 256"),
}


def plan(gpus: int) -> list[tuple[str, bool, str]]:
    out = []
    for lid, spec in LABS.items():
        ok = gpus >= spec["min_gpus"]
        reason = "" if ok else f"需要 ≥{spec['min_gpus']} 卡"
        out.append((lid, ok, reason))
    return out


def cmd_for(lid: str, gpus: int) -> str:
    return LABS[lid]["shell"].format(gpus=gpus, tp=min(gpus, 8))


# ---------------------------------------------------------------- execution


def run_cmd(cmd: str, logfile: Path, timeout: int) -> tuple[int, float]:
    """Run a shell command, tee output to logfile. Returns (rc, seconds)."""
    print(f"\n$ {cmd}")
    start = time.time()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        with logfile.open("w", encoding="utf-8", errors="replace") as fh:
            p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               errors="replace", timeout=timeout)
            fh.write(p.stdout or "")
            # Echo a trimmed version to the console.
            lines = (p.stdout or "").splitlines()
            for ln in (lines if len(lines) <= 60 else lines[-60:]):
                print("   " + ln)
            rc = p.returncode
    except subprocess.TimeoutExpired:
        print(f"   [TIMEOUT after {timeout}s]")
        rc = 124
    except Exception as exc:
        print(f"   [ERROR {exc}]")
        rc = 1
    return rc, time.time() - start


def append_command_log(results: Path, lid: str, cmd: str, rc: int, secs: float) -> None:
    with (results / "commands.sh").open("a", encoding="utf-8") as fh:
        fh.write(f"# {lid} rc={rc} {secs:.1f}s\n{cmd}\n\n")


def write_status(results: Path, rows: list[dict]) -> None:
    (results / "status.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["| Lab | 状态 | 耗时 s | 日志 |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} {r['title']} | "
                     f"{'OK' if r['rc'] == 0 else 'FAIL(' + str(r['rc']) + ')'} | "
                     f"{r['seconds']:.0f} | `{r['log']}` |")
    md = ["# 实验运行状态", "",
          f"- 时间: {datetime.now().isoformat(timespec='seconds')}",
          f"- GPU 数: {rows[0]['gpus'] if rows else '?'}", ""] + lines + [""]
    (results / "STATUS.md").write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------- summary


def build_summary(results: Path) -> Path:
    """Render SUMMARY.md from whatever CSVs/logs exist in results/."""
    out = ["# 实测结果汇总", "",
           f"生成时间: {datetime.now().isoformat(timespec='seconds')}", ""]

    for csvfile in sorted(results.glob("ar_*.csv")):
        try:
            # utf-8-sig: tolerate a BOM, which some editors (and PowerShell's
            # Set-Content -Encoding UTF8) prepend and csv.DictReader would
            # otherwise turn into a mangled first column name.
            with csvfile.open(encoding="utf-8-sig") as fh:
                rows = list(csv.DictReader(fh))
        except Exception as exc:
            out += [f"## {csvfile.name}", "",
                    f"（解析失败：{exc}）", ""]
            continue
        if not rows:
            out += [f"## {csvfile.name}", "", "（空文件）", ""]
            continue
        missing = {"size_bytes", "median_ms", "world_size"} - set(rows[0])
        if missing:
            out += [f"## {csvfile.name}", "",
                    f"（列不完整，缺少 {sorted(missing)}；实际列 {list(rows[0])}）", ""]
            continue
        out += [f"## {csvfile.name}", "",
                "| tag | world | 消息 | 中位耗时 μs | 有效带宽 GB/s | 后端 |",
                "|---|---|---|---|---|---|"]
        for r in rows:
            size = int(r["size_bytes"])
            out.append(
                f"| {r.get('tag','')} | {r.get('world_size','')} "
                f"| {size/1024:.0f} KiB "
                f"| {float(r['median_ms'])*1e3:.1f} "
                f"| {r.get('effective_gbps','')} "
                f"| {r.get('backend_hint','')} |")
        out.append("")

        # Derive alpha from the smallest message (bandwidth term is negligible).
        try:
            small = min(rows, key=lambda r: int(r["size_bytes"]))
            n = int(small["world_size"])
            t_us = float(small["median_ms"]) * 1e3
            bw = 450e9  # NVLink 4 unidirectional; adjust for your machine
            s = int(small["size_bytes"])
            steps = 2 * (n - 1)
            alpha = (t_us * 1e-6 - 2 * (n - 1) / n * s / bw) / steps * 1e6
            out += [f"反推 α（用 {s/1024:.0f} KiB、按单向 450 GB/s 估算，"
                    f"ring {steps} 步）≈ **{alpha:.2f} μs**", "",
                    "> 注意：这个 α 是「从上界反推」的粗估，受实现细节影响；"
                    "把它和教程 ch01 的量级表对照，而不是当精确值。", ""]
        except Exception:
            pass

    for jf in sorted(results.glob("verify.json")):
        try:
            # utf-8-sig: json.loads rejects a leading BOM.
            data = json.loads(jf.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            out += [f"## {jf.name}", "", f"（解析失败：{exc}）", ""]
            continue
        out += [f"## {jf.name} — 通信正确性", "",
                f"- {data.get('checks_passed')}/{data.get('checks_total')} 通过",
                f"- world_size = {data.get('world_size')}", ""]
        for c in data.get("checks", []):
            out.append(f"  - [{'OK' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
        out.append("")

    for lf in sorted(results.glob("prof*.nsys-rep")):
        out += [f"## nsys profile", "",
                f"发现 `{lf.name}`。转换成 kernel 汇总：", "",
                "```bash",
                f"nsys stats --report cuda_gpu_kern_sum {lf} > results/kern_sum.txt",
                "grep -iE 'allreduce|reduce_scatter|all_gather|nccl' results/kern_sum.txt",
                "```", ""]

    path = results / "SUMMARY.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="list",
                    choices=["list", "all", "core", "run", "collect", "markdown"])
    ap.add_argument("labs", nargs="*", help="lab ids for `run`, e.g. a2 a5 b1")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="results dir (default results/<timestamp>)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-lab timeout seconds")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    gpus = gpu_count()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = args.out_dir or Path("results") / stamp
    results.mkdir(parents=True, exist_ok=True)

    if args.action == "list":
        print(f"检测到 {gpus} 张 GPU  (torchrun: {torchrun_available()}, "
              f"RDMA: {has_rdma()})\n")
        print(f"{'id':<4} {'可跑':<4} {'需要':<16} 说明")
        print("-" * 78)
        for lid, ok, reason in plan(gpus):
            spec = LABS[lid]
            flag = "✅" if ok else "❌"
            print(f"{lid:<4} {flag:<4} {spec['needs']:<16} {spec['title']}")
            if not ok:
                print(f"{'':<4} {'':<4} {'':<16} └─ {reason}")
        print("\n命令示例：")
        runnable = [lid for lid, ok, _ in plan(gpus) if ok]
        if runnable:
            print(f"  python {Path(__file__).name} core")
            print(f"  python {Path(__file__).name} run {' '.join(runnable[:3])}")
        return 0

    if args.action == "collect":
        base = args.out_dir or Path("results")
        tarpath = Path(f"lab-results-{stamp}.tar.gz")
        with tarfile.open(tarpath, "w:gz") as tar:
            tar.add(base, arcname=base.name)
        print(f"wrote {tarpath}  ({tarpath.stat().st_size/1024:.0f} KiB)")
        print("把这个文件拉回本地 —— 实例释放后就没了。")
        return 0

    if args.action == "markdown":
        base = args.out_dir or (sorted(Path("results").glob("*"))[-1]
                                if Path("results").is_dir() else Path("results"))
        path = build_summary(base)
        print(f"wrote {path}")
        print(path.read_text(encoding="utf-8")[:2000])
        return 0

    # ---- which labs -------------------------------------------------------
    if args.action == "core":
        wanted = ["a2", "a5"]
    elif args.action == "all":
        wanted = [lid for lid, ok, _ in plan(gpus) if ok]
    else:
        wanted = args.labs or []

    if not wanted:
        print("没有选中任何 lab。用 `list` 看可用的，或 `run a2 a5`。")
        return 1

    avail = {lid for lid, ok, _ in plan(gpus) if ok}
    skipped = [lid for lid in wanted if lid not in avail]
    wanted = [lid for lid in wanted if lid in avail]
    for lid in skipped:
        print(f"跳过 {lid}（{LABS.get(lid, {}).get('needs', '未知')} 不满足）")
    if not wanted:
        return 1

    print(f"\n结果目录: {results}")
    print(f"计划: {', '.join(wanted)}   （GPU={gpus}）")
    print("=" * 76)

    rows: list[dict] = []
    for lid in wanted:
        spec = LABS[lid]
        cmd = cmd_for(lid, gpus)
        log = results / f"{lid}.log"
        rc, secs = run_cmd(cmd, log, args.timeout)
        append_command_log(results, lid, cmd, rc, secs)
        rows.append(dict(id=lid, title=spec["title"], rc=rc,
                         seconds=round(secs, 1), log=log.name, gpus=gpus,
                         note=spec["note"]))
        write_status(results, rows)
        if rc != 0 and args.fail_fast:
            print("\n--fail-fast: 中止")
            break

    print("\n" + "=" * 76)
    ok_n = sum(1 for r in rows if r["rc"] == 0)
    print(f"完成 {ok_n}/{len(rows)}；状态见 {results/'STATUS.md'}")
    summary = build_summary(results)
    print(f"汇总见 {summary}")
    print(f"\n下一步：  python {Path(__file__).name} collect --out-dir {results}")
    return 0 if ok_n == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
