#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rented-GPU provisioning self-check. Run this FIRST after the instance is up.

Answers, in about 30 seconds and without needing multi-GPU:

  1. What hardware did I actually get? (GPU, memory, topology, RDMA, driver)
  2. Are the versions new enough for the features the tutorial talks about?
  3. Which labs can I run right now, and which need something I don't have?
  4. Does collective communication actually work? (2-rank smoke test if >= 2 GPUs)

Writes ``env_report.txt`` next to the results directory so you can pull it off
the instance before releasing it.

Usage::

    python provision_rented_gpu.py                 # human report
    python provision_rented_gpu.py --json          # also env_report.json
    python provision_rented_gpu.py --out-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _force_utf8_stdio() -> None:
    """Make CJK/emoji output safe on consoles that default to a legacy codepage.

    Linux rented boxes are UTF-8 already; this matters on Windows (cp1252/cp936)
    and inside odd container locales, where printing Chinese would otherwise
    raise UnicodeEncodeError. Never fatal.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()

# ---------------------------------------------------------------- helpers


def run(cmd: list[str], timeout: int = 25) -> tuple[int, str]:
    """Run a command, return (returncode, combined output)."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return 127, f"<{cmd[0]} not found>"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"<timeout after {timeout}s>"
    except Exception as exc:  # pragma: no cover
        return 1, f"<error: {exc}>"


def section(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


# ---------------------------------------------------------------- probes


def probe_gpus() -> dict:
    info: dict = {"nvidia_smi": None, "gpus": [], "topo": None, "count": 0}

    rc, out = run(["nvidia-smi",
                   "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                   "--format=csv,noheader"])
    if rc == 0:
        info["nvidia_smi"] = out.strip()
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                info["gpus"].append(dict(index=parts[0], name=parts[1],
                                         memory=parts[2], driver=parts[3],
                                         compute_cap=parts[4]))
        info["count"] = len(info["gpus"])

    rc, out = run(["nvidia-smi", "topo", "-m"])
    if rc == 0:
        info["topo"] = out.rstrip()

    return info


def probe_rdma() -> dict:
    info: dict = {"devices": [], "ibv_devinfo": None}
    ib = Path("/sys/class/infiniband")
    if ib.is_dir():
        info["devices"] = sorted(p.name for p in ib.iterdir())
    rc, out = run(["ibv_devinfo", "-l"])
    if rc == 0:
        info["ibv_devinfo"] = out.strip()
    return info


def probe_interconnect(topo_text: str | None) -> dict:
    """Classify GPU-GPU interconnect from `nvidia-smi topo -m`."""
    verdict = {"nvlink": False, "pcie_only": False, "single_gpu": False, "note": ""}
    if not topo_text:
        verdict["note"] = "no topology output available"
        return verdict

    lines = [ln for ln in topo_text.splitlines() if ln.strip()]
    gpu_rows = [ln for ln in lines if ln.strip().upper().startswith("GPU")]
    if len(gpu_rows) < 2:
        verdict["single_gpu"] = True
        verdict["note"] = "only one GPU visible"
        return verdict

    # Cells beyond the row label describe each pair.
    has_nv = False
    has_far = False
    for ln in gpu_rows:
        cells = ln.split()[1:]
        for cell in cells:
            u = cell.upper()
            if u.startswith("NV"):
                has_nv = True
            elif u in {"SYS", "PHB", "PXB", "PIX", "NODE"}:
                has_far = True
    verdict["nvlink"] = has_nv
    verdict["pcie_only"] = (not has_nv) and has_far
    if has_nv and has_far:
        verdict["note"] = "mixed: some pairs NVLink, some PCIe/NUMA"
    elif has_nv:
        verdict["note"] = "all pairs appear to be NVLink"
    elif has_far:
        verdict["note"] = "no NVLink detected; PCIe/NUMA only"
    return verdict


def probe_python() -> dict:
    info: dict = {"python": sys.version.split()[0], "executable": sys.executable,
                  "torch": None, "nccl": None, "cuda": None, "device_count": 0,
                  "vllm": None, "deep_ep": False}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            try:
                info["nccl"] = list(torch.cuda.nccl.version())  # type: ignore[arg-type]
            except Exception:
                info["nccl"] = None
            info["device_name"] = torch.cuda.get_device_name(0)
            try:
                cap = torch.cuda.get_device_capability(0)
                info["compute_capability"] = f"{cap[0]}.{cap[1]}"
            except Exception:
                pass
    except ModuleNotFoundError:
        pass

    try:
        import vllm

        info["vllm"] = getattr(vllm, "__version__", "unknown")
    except Exception:
        pass

    try:
        import deep_ep  # noqa: F401

        info["deep_ep"] = True
    except Exception:
        info["deep_ep"] = False

    return info


# ---------------------------------------------------------------- capability matrix

def nccl_version_tuple(raw) -> tuple[int, int, int] | None:
    if not raw:
        return None
    try:
        return (int(raw[0]), int(raw[1]), int(raw[2]))
    except Exception:
        return None


def capability_matrix(py: dict) -> list[dict]:
    """Compare the runtime versions against the thresholds the tutorial cites."""
    caps: list[dict] = []
    nccl = nccl_version_tuple(py.get("nccl"))

    def add(name, ok, need, got, note=""):
        caps.append(dict(feature=name, ok=ok, needs=need, got=got, note=note))

    add("pynccl / custom all-reduce", py.get("torch") is not None,
        "PyTorch + CUDA", py.get("torch") or "missing")
    add("NCCL symmetric memory (window API)",
        nccl is not None and nccl >= (2, 27, 3),
        "NCCL >= 2.27.3", f"{nccl}" if nccl else "unknown")
    add("ncclCommSuspend / Resume",
        nccl is not None and nccl >= (2, 29, 7),
        "NCCL >= 2.29.7", f"{nccl}" if nccl else "unknown")
    add("DeepEP v2 (NCCL GIN)",
        nccl is not None and nccl >= (2, 30, 4) and py.get("deep_ep", False),
        "NCCL >= 2.30.4 AND deep_ep installed",
        f"nccl={nccl}, deep_ep={py.get('deep_ep')}")
    add("DeepEP (v1) / DBO labs", py.get("deep_ep", False),
        "deep_ep installed", str(py.get("deep_ep")))
    add("vLLM importable", py.get("vllm") is not None,
        "pip install vllm", str(py.get("vllm") or "missing"))
    return caps


def recommend_labs(gpus: int, ic: dict, rdma: dict, caps: list[dict]) -> list[str]:
    """Return concrete lab recommendations for this machine."""
    out: list[str] = []
    nvlink = ic.get("nvlink")

    out.append("A1 拓扑与冒烟 — 任何机器")
    if gpus >= 2:
        out.append("A2 all-reduce 实测 vs 手算 ★ 核心")
        out.append("A3 custom AR vs NCCL 对比 ★")
        out.append("A4 ring vs tree crossover")
        out.append("A5 通信正确性验证")
    else:
        out.append("(只有 1 卡 → A2–A5 需要 ≥2 卡)")

    if gpus >= 4:
        out.append("B2 TP 缩放曲线")
        out.append("B3 8 卡真实负载（若 gpus>=8）" if gpus >= 8 else "B3 TP 缩放（4 卡版）")
    if gpus >= 8:
        out.append("B1 custom AR 的 size 上限实测 ★")
        out.append("B5 nsys 通信占比")
    if not nvlink and gpus >= 2:
        out.append("⚠️ 无 NVLink：TP 会走 PCIe，实测数据与教程的 NVLink 数字不可直接对比")
    if not rdma.get("devices"):
        out.append("⚠️ 无 RDMA：跨机实验（套餐 C）不可做，或只能测以太网")
    if next((c["ok"] for c in caps if c["feature"].startswith("DeepEP (v1)")), False):
        out.append("B4 DBO / MoE 实验")
    else:
        out.append("(未装 deep_ep → B4 的 DBO 实验跳过)")

    return out


# ---------------------------------------------------------------- smoke test


def smoke_test(gpus: int, out_dir: Path) -> dict:
    """Run a tiny 2-rank all-reduce; returns dict with ok/detail."""
    if gpus < 2:
        return dict(ok=None, detail="need >= 2 GPUs")
    here = Path(__file__).resolve().parent
    script = here / "allreduce_bench.py"
    if not script.exists():
        return dict(ok=None, detail="allreduce_bench.py not found next to this script")

    rc, out = run([
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=2", str(script),
        "--sizes", "1M", "--iters", "10", "--warmup", "3",
        "--out", str(out_dir / "smoke.csv"),
    ], timeout=300)

    if rc == 0:
        tail = [ln for ln in out.splitlines() if "correctness" in ln or "size" in ln.lower()]
        return dict(ok=True, detail=" | ".join(tail[-2:])[:200] or "completed")

    last = out.strip().splitlines()[-1][:200] if out.strip() else "(no output)"
    return dict(ok=False, detail=f"exit {rc}: {last}")


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--json", action="store_true", help="also write env_report.json")
    ap.add_argument("--no-smoke", action="store_true", help="skip the 2-rank smoke test")
    ap.add_argument("--smoke-timeout", type=int, default=300)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gpus_info = probe_gpus()
    rdma = probe_rdma()
    ic = probe_interconnect(gpus_info.get("topo"))
    py = probe_python()
    caps = capability_matrix(py)
    gpus = max(gpus_info["count"], py.get("device_count") or 0)

    smoke = dict(ok=None, detail="skipped")
    if not args.no_smoke and gpus >= 2:
        smoke = smoke_test(gpus, args.out_dir)

    recs = recommend_labs(gpus, ic, rdma, caps)

    # ---------------- human report ----------------
    section("1) 硬件")
    print(f"  主机名      : {platform.node()}")
    print(f"  GPU 数量    : {gpus}")
    for g in gpus_info["gpus"]:
        print(f"    [{g['index']}] {g['name']}  {g['memory']}  driver {g['driver']}  "
              f"cc {g['compute_cap']}")
    if not gpus_info["gpus"]:
        print("    (nvidia-smi 没有返回 GPU —— 检查驱动/容器权限)")
    print(f"  卡间互联    : NVLink={ic['nvlink']}  PCIe-only={ic['pcie_only']}  "
          f"单卡={ic['single_gpu']}  ({ic['note']})")
    print(f"  RDMA 设备   : {rdma['devices'] or '无'}")
    print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    section("2) 版本")
    print(f"  python {py['python']}")
    print(f"  torch  {py['torch']}  cuda {py['cuda']}  devices {py['device_count']}")
    print(f"  nccl   {py['nccl']}")
    print(f"  vllm   {py['vllm']}")
    print(f"  deep_ep {py['deep_ep']}")

    section("3) 能跑哪些实验（对照教程的版本要求）")
    for c in caps:
        mark = "✅" if c["ok"] else "❌"
        print(f"  {mark} {c['feature']:<38} needs: {c['needs']:<38} got: {c['got']}")

    section("4) 通信冒烟测试")
    if smoke["ok"] is True:
        print(f"  ✅ 2 卡 all-reduce 通过: {smoke['detail']}")
    elif smoke["ok"] is False:
        print(f"  ❌ 失败: {smoke['detail']}")
        print("     → 先修通信再往下做（见教程 ch05 §5.3）")
    else:
        print(f"  ⏭️  {smoke['detail']}")

    section("5) 建议的实验顺序")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r}")

    section("6) 提醒")
    print("  * 跑完实验立刻把 results/ 拉走 —— 实例一释放数据就没了")
    print("  * 每组实验至少重复 3 次，报中位数")
    print("  * 数据用 labs-gpu.md 的模板记录，否则等于没做")

    # ---------------- dump ----------------
    report = dict(
        hostname=platform.node(),
        gpus=gpus_info,
        rdma=rdma,
        interconnect=ic,
        python=py,
        capabilities=caps,
        smoke_test=smoke,
        recommendations=recs,
        env={k: v for k, v in os.environ.items()
             if k.startswith(("NCCL_", "VLLM_", "CUDA_", "UCX_"))},
    )
    txt = args.out_dir / "env_report.txt"
    with txt.open("w", encoding="utf-8") as fh:
        fh.write(f"host={platform.node()} gpus={gpus} nvlink={ic['nvlink']} "
                 f"rdma={rdma['devices']}\n")
        fh.write(f"torch={py['torch']} cuda={py['cuda']} nccl={py['nccl']} "
                 f"vllm={py['vllm']} deep_ep={py['deep_ep']}\n\n")
        fh.write((gpus_info.get("topo") or "") + "\n\n")
        for c in caps:
            fh.write(f"{'OK ' if c['ok'] else 'NO '} {c['feature']} "
                     f"(needs {c['needs']}, got {c['got']})\n")
    print(f"\nwrote {txt}")
    if args.json:
        (args.out_dir / "env_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out_dir / 'env_report.json'}")

    # Exit non-zero only when we positively know communication is broken.
    return 1 if smoke["ok"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
