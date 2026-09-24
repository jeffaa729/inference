#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU all-reduce benchmark for the net/NCCL tutorial (see labs-gpu.md).

Sweeps message sizes, repeats each measurement, and reports the median together
with the *effective* per-rank bandwidth. The point is to let you compare a real
machine against the alpha-beta model in ch01/ch02 of the tutorial.

Run with torchrun::

    torchrun --nproc_per_node=2 allreduce_bench.py --sizes 16K,256K,4M,64M
    torchrun --nproc_per_node=8 allreduce_bench.py --sizes 16K,64M \\
        --out results/ar_tp8.csv --tag tp8

Optional extras:
  --algo Ring|Tree      force NCCL_ALGO before init (must be set before init)
  --compare-backends    also measure with vLLM's custom all-reduce disabled
  --report-backend      print which vLLM all-reduce backend is active (best effort)

Notes
-----
* ``moved_bytes_per_rank`` is the ring model's figure: ``2*(N-1)/N * S``
  (reduce-scatter + all-gather). It is a *reference*, not a claim about the
  implementation actually chosen.
* We time with CUDA events around a loop of ``--iters`` all-reduces, which
  amortizes launch overhead the same way a real decode step does.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# torch is imported lazily inside main() so that --help and the size-parsing
# helpers work on a machine without CUDA/torch (e.g. when preparing a rented
# instance). The script itself obviously needs a CUDA box to do anything useful.
torch = None  # type: ignore[assignment]
dist = None  # type: ignore[assignment]


def _force_utf8_stdio() -> None:
    """Keep output safe on consoles defaulting to a legacy codepage. Never fatal."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()


def _require_torch():
    global torch, dist
    try:
        import torch as _torch
        import torch.distributed as _dist
    except ModuleNotFoundError:
        print(
            "ERROR: PyTorch is not installed in this interpreter.\n"
            "       Install vLLM/torch first, or run this on the GPU machine.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    torch, dist = _torch, _dist
    return torch, dist


# ---------------------------------------------------------------- utilities


def parse_size(text: str) -> int:
    """Parse '16K' / '4M' / '1048576' into bytes."""
    t = text.strip().upper()
    mult = 1
    for suffix, m in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if t.endswith(suffix):
            mult = m
            t = t[: -len(suffix)]
            break
    try:
        value = float(t)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad size: {text!r}") from exc
    return int(value * mult)


def parse_sizes(text: str) -> list[int]:
    sizes = sorted({parse_size(p) for p in text.split(",") if p.strip()})
    if not sizes:
        raise argparse.ArgumentTypeError("no sizes given")
    return sizes


def human(n: int) -> str:
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:.4g} {unit}"
    return f"{n} B"


def human_bytes_per_s(bps: float) -> str:
    return f"{bps / 1e9:.2f} GB/s"


# ---------------------------------------------------------------- timing core


def bench_one(tensor: torch.Tensor, iters: int, warmup: int) -> tuple[float, float, float]:
    """Return (median_ms, min_ms, max_ms) for ``iters`` all-reduces."""
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_reduce(tensor)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    return statistics.median(samples), min(samples), max(samples)


def backend_hint(device_communicator) -> str:
    """Best-effort label of which all-reduce path vLLM would take."""
    if device_communicator is None:
        return "torch.distributed"
    ca = getattr(device_communicator, "ca_comm", None)
    if ca is not None and not getattr(ca, "disabled", True):
        return "vllm-custom-ar"
    if getattr(device_communicator, "pynccl_comm", None) is not None:
        p = device_communicator.pynccl_comm
        if not getattr(p, "disabled", True):
            return "pynccl"
    return "torch.distributed"


def try_vllm_tp_group():
    """Return the vLLM TP group coordinator, or None if unavailable."""
    try:
        from vllm.distributed.parallel_state import (
            get_tp_group,
            model_parallel_is_initialized,
        )

        if not model_parallel_is_initialized():
            return None
        return get_tp_group()
    except Exception:
        return None


def try_disable_custom_ar():
    """Disable vLLM's custom all-reduce (used by --compare-backends)."""
    try:
        from vllm.distributed.parallel_state import set_custom_all_reduce

        set_custom_all_reduce(False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=parse_sizes, default=parse_sizes("16K,256K,4M,64M"),
                    help="comma list, e.g. 16K,1M,64M (default: 16K,256K,4M,64M)")
    ap.add_argument("--iters", type=int, default=50, help="timed iterations per size (default 50)")
    ap.add_argument("--warmup", type=int, default=10, help="warmup iterations per size (default 10)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--algo", default=None, help="force NCCL_ALGO (Ring/Tree), set before init")
    ap.add_argument("--out", type=Path, default=None, help="write CSV here")
    ap.add_argument("--tag", default="", help="label stored in the CSV")
    ap.add_argument("--compare-backends", action="store_true",
                    help="also run with vLLM custom all-reduce disabled")
    ap.add_argument("--report-backend", action="store_true",
                    help="print the active vLLM all-reduce backend")
    args = ap.parse_args()

    torch, dist = _require_torch()

    # NCCL_ALGO must be in the environment before the process group is created.
    if args.algo:
        os.environ["NCCL_ALGO"] = args.algo

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device visible", file=sys.stderr)
        return 2
    if "RANK" not in os.environ:
        print("ERROR: launch with torchrun (RANK/WORLD_SIZE not set)", file=sys.stderr)
        return 2

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    dtype = getattr(torch, args.dtype)

    if rank == 0:
        props = torch.cuda.get_device_properties(device)
        print("=" * 78)
        print(f"all-reduce benchmark   world_size={world}  dtype={args.dtype}")
        print(f"GPU: {props.name}  ({props.total_memory / (1 << 30):.1f} GiB)")
        print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
              f"nccl {torch.cuda.nccl.version()}")
        if args.algo:
            print(f"NCCL_ALGO forced to {args.algo}")
        print("=" * 78)

    # --- optional: report / toggle the vLLM backend -------------------------
    tp_group = try_vllm_tp_group()
    disabled = False
    if args.compare_backends:
        disabled = try_disable_custom_ar()
        if rank == 0:
            print(f"[compare-backends] set_custom_all_reduce(False) -> {disabled}")

    if args.report_backend and rank == 0:
        dc = getattr(tp_group, "device_communicator", None) if tp_group else None
        print(f"[backend] {backend_hint(dc)}")
        if dc is not None:
            ca = getattr(dc, "ca_comm", None)
            if ca is not None:
                print(f"[backend] custom AR disabled={getattr(ca, 'disabled', '?')} "
                      f"world_size={getattr(ca, 'world_size', '?')} "
                      f"max_size={getattr(ca, 'max_size', '?')} "
                      f"fully_connected={getattr(ca, 'fully_connected', '?')}")

    rows: list[dict] = []
    if rank == 0:
        print(f"{'size':>12} {'median':>10} {'min':>10} {'max':>10} "
              f"{'moved/rank':>12} {'eff BW':>12}")
        print("-" * 78)

    for size in args.sizes:
        # Keep the element count a multiple of 16 bytes worth of elements so
        # custom all-reduce's 16-byte alignment gate can pass when applicable.
        elem = torch.tensor([], dtype=dtype).element_size()
        count = max(16, size // elem)
        count -= count % 16
        size_bytes = count * elem

        tensor = torch.ones(count, dtype=dtype, device=device) * (rank + 1)

        median_ms, min_ms, max_ms = bench_one(tensor, args.iters, args.warmup)

        # Ring model: each rank sends/recvs 2*(N-1)/N * S in total.
        moved = 2 * (world - 1) / world * size_bytes
        eff_gbps = moved / (median_ms * 1e-3) / 1e9

        dc = getattr(tp_group, "device_communicator", None) if tp_group else None
        rows.append(dict(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tag=args.tag,
            world_size=world,
            size_bytes=size_bytes,
            dtype=args.dtype,
            iters=args.iters,
            median_ms=round(median_ms, 6),
            min_ms=round(min_ms, 6),
            max_ms=round(max_ms, 6),
            moved_bytes_per_rank=int(moved),
            effective_gbps=round(eff_gbps, 3),
            backend_hint=("disabled-custom-ar" if disabled else backend_hint(dc)),
            nccl_algo=args.algo or "",
        ))

        if rank == 0:
            print(f"{human(size_bytes):>12} {median_ms * 1e3:>9.1f}us "
                  f"{min_ms * 1e3:>9.1f}us {max_ms * 1e3:>9.1f}us "
                  f"{human(moved):>12} {human_bytes_per_s(eff_gbps * 1e9):>12}")

    # correctness spot check (cheap, catches a broken world)
    if rank == 0:
        probe = torch.full((1024,), float(rank), dtype=torch.float32, device=device)
    else:
        probe = torch.full((1024,), float(rank), dtype=torch.float32, device=device)
    dist.all_reduce(probe)
    expected = float(sum(range(world)))
    ok = bool(torch.allclose(probe, torch.full_like(probe, expected)))
    if rank == 0:
        print("-" * 78)
        print(f"correctness: all_reduce(sum of ranks) == {expected:.0f} -> "
              f"{'OK' if ok else 'FAIL'}")

    # --- write CSV (rank 0 only) -------------------------------------------
    if args.out is not None and rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.out.exists()
        with args.out.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")

    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
