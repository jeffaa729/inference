#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collective-communication correctness checks + parallelism topology dump.

Companion to labs-gpu.md (A5). Two jobs:

1. **Correctness.** Fill each tensor with this rank's index, all-reduce, and
   assert every rank ends up with ``sum(range(world_size))``. This catches a
   broken world without reading a single kernel. Repeated across dtypes so you
   also see how much bf16 tolerance you need.

2. **Topology.** Print each rank's global rank / device / node, and the
   rank membership of the vLLM parallelism groups when vLLM is importable.
   This is the cheapest way to *verify* the tutorial's claim that TP groups are
   built from adjacent ranks.

Run::

    torchrun --nproc_per_node=2 verify_collectives.py
    torchrun --nproc_per_node=8 verify_collectives.py --json results/verify.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

torch = None
dist = None


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
        print("ERROR: PyTorch not installed here; run on the GPU machine.", file=sys.stderr)
        raise SystemExit(2)
    torch, dist = _torch, _dist
    return torch, dist


# ---------------------------------------------------------------- checks

DTYPE_TOLERANCE = {
    # bf16 has 8 mantissa bits: a sum of N ones needs ~log2(N) bits of headroom.
    "float32": dict(rtol=1e-5, atol=1e-5),
    "float16": dict(rtol=1e-2, atol=1e-2),
    "bfloat16": dict(rtol=2e-2, atol=2e-2),
}


def check_all_reduce(device, world: int) -> list[dict]:
    results = []
    expected = float(sum(range(world)))

    for name, tol in DTYPE_TOLERANCE.items():
        dtype = getattr(torch, name)
        t = torch.full((4096,), float(dist.get_rank()), dtype=dtype, device=device)
        dist.all_reduce(t)
        got = t.float()
        want = torch.full_like(got, expected)
        ok = bool(torch.allclose(got, want, **tol))
        results.append(dict(
            check=f"all_reduce[{name}]",
            ok=ok,
            detail=f"got {got[0].item():.4f}, want {expected:.4f}",
            max_abs_err=float((got - want).abs().max().item()),
        ))
    return results


def check_all_gather(device, world: int) -> dict:
    """all_gather_into_tensor: rank i's block i must equal i."""
    t = torch.full((16,), float(dist.get_rank()), device=device)
    out = torch.empty(16 * world, device=device)
    dist.all_gather_into_tensor(out, t)
    blocks = out.view(world, 16)
    ok = all(bool(torch.allclose(blocks[i], torch.full_like(blocks[i], float(i))))
             for i in range(world))
    return dict(check="all_gather_into_tensor", ok=ok,
                detail=f"{world} blocks checked")


def check_reduce_scatter(device, world: int) -> dict:
    """reduce_scatter_tensor: every rank's scalar slice must equal the full sum."""
    t = torch.full((16 * world,), float(dist.get_rank()), device=device)
    out = torch.empty(16, device=device)
    dist.reduce_scatter_tensor(out, t)
    expected = float(sum(range(world)))
    ok = bool(torch.allclose(out, torch.full_like(out, expected)))
    return dict(check="reduce_scatter_tensor", ok=ok,
                detail=f"chunk0={out[0].item():.1f}, want {expected:.1f}")


def check_broadcast(device) -> dict:
    src = 0
    t = torch.full((64,), -1.0, device=device)
    if dist.get_rank() == src:
        t.fill_(7.0)
    dist.broadcast(t, src=src)
    ok = bool(torch.allclose(t, torch.full_like(t, 7.0)))
    return dict(check="broadcast", ok=ok, detail=f"src={src}")


def check_send_recv(device, world: int) -> dict:
    """Ring of point-to-point sends; each rank should receive rank-1's value."""
    rank = dist.get_rank()
    nxt = (rank + 1) % world
    t = torch.full((32,), float(rank), device=device)
    recv = torch.full((32,), -1.0, device=device)

    ops = [dist.P2POp(dist.isend, t, nxt), dist.P2POp(dist.irecv, recv, (rank - 1) % world)]
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()

    expected = float((rank - 1) % world)
    ok = bool(torch.allclose(recv, torch.full_like(recv, expected)))
    return dict(check="send/recv ring", ok=ok,
                detail=f"got {recv[0].item():.0f}, want {expected:.0f}")


# ---------------------------------------------------------------- topology

def topology_record(device, world: int) -> dict:
    """Per-rank hardware/process identity; rank 0 aggregates via all_gather_object."""
    rank = dist.get_rank()
    props = torch.cuda.get_device_properties(device)
    return dict(
        rank=rank,
        world_size=world,
        hostname=socket.gethostname(),
        device_index=device.index,
        device_name=props.name,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        local_rank_env=os.environ.get("LOCAL_RANK", "<unset>"),
    )


def vllm_groups() -> dict | None:
    """Rank membership of the vLLM parallelism groups, if vLLM is importable."""
    try:
        from vllm.distributed import parallel_state as ps
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"available": False, "error": str(exc)}

    out: dict = {"available": True}
    for name in ("tp", "pp", "dp", "ep", "pcp", "dcp", "world"):
        getter = getattr(ps, f"get_{name}_group", None)
        if getter is None:
            continue
        try:
            g = getter()
            if g is None:
                continue
            out[name] = dict(ranks=list(g.ranks), world_size=g.world_size,
                             my_rank_in_group=g.rank_in_group)
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


def vllm_allreduce_backends() -> dict | None:
    """Which all-reduce backends exist/enabled for the TP group."""
    try:
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        g = get_tp_group()
        dc = g.device_communicator
        if not isinstance(dc, CudaCommunicator):
            return {"available": False, "reason": "not a CudaCommunicator"}
        info: dict = {}
        for attr in ("ca_comm", "qr_comm", "symm_mem_comm", "fi_ar_comm",
                     "fi_pcie_ipc_ar_comm", "aiter_ar_comm", "pynccl_comm"):
            obj = getattr(dc, attr, None)
            if obj is None:
                info[attr] = None
            else:
                info[attr] = dict(
                    disabled=getattr(obj, "disabled", None),
                    world_size=getattr(obj, "world_size", None),
                    max_size=getattr(obj, "max_size", None),
                    fully_connected=getattr(obj, "fully_connected", None),
                )
        return info
    except Exception as exc:
        return {"available": False, "error": str(exc)}


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None, help="write full report as JSON")
    ap.add_argument("--skip-vllm", action="store_true",
                    help="skip vLLM group/backend introspection")
    args = ap.parse_args()

    torch, dist = _require_torch()

    if "RANK" not in os.environ:
        print("ERROR: launch with torchrun", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: no CUDA device visible", file=sys.stderr)
        return 2

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    # ---- correctness (all ranks participate; rank 0 prints) --------------
    checks: list[dict] = []
    checks += check_all_reduce(device, world)
    checks.append(check_all_gather(device, world))
    checks.append(check_reduce_scatter(device, world))
    checks.append(check_broadcast(device))
    checks.append(check_send_recv(device, world))

    failed = [c for c in checks if not c["ok"]]
    if rank == 0:
        print("=" * 74)
        print(f"collective correctness   world_size={world}")
        print("=" * 74)
        for c in checks:
            print(f"  [{'OK ' if c['ok'] else 'FAIL'}] {c['check']:<28} {c['detail']}")
        print("-" * 74)
        print(f"  {len(checks) - len(failed)}/{len(checks)} passed")
        if failed:
            print("  FAILURES: " + ", ".join(c["check"] for c in failed))
        print("=" * 74)

    # ---- topology --------------------------------------------------------
    rec = topology_record(device, world)
    gathered: list[dict] = [None] * world  # type: ignore[list-item]
    dist.all_gather_object(gathered, rec)

    if rank == 0:
        print("\nranks by host (verify: TP groups should be on one host):")
        by_host: dict[str, list[int]] = {}
        for r in gathered:
            by_host.setdefault(r["hostname"], []).append(r["rank"])
        for host, ranks in by_host.items():
            print(f"  {host}: {sorted(ranks)}")
        print(f"  distinct hosts = {len(by_host)}"
              f"{'  (single node)' if len(by_host) == 1 else '  (multi-node)'}")

    groups = None if args.skip_vllm else vllm_groups()
    backends = None if args.skip_vllm else vllm_allreduce_backends()

    if rank == 0 and groups is not None:
        print("\nvLLM parallelism groups:")
        if not groups.get("available"):
            print(f"  (not available: {groups.get('error', groups.get('reason'))})")
        else:
            for name, g in groups.items():
                if name == "available":
                    continue
                if "error" in g:
                    print(f"  {name:<6} error: {g['error']}")
                else:
                    print(f"  {name:<6} ranks={g['ranks']} world_size={g['world_size']}")

    if rank == 0 and backends is not None:
        print("\nall-reduce backends on the TP group:")
        if not backends.get("available"):
            print(f"  (not available: {backends.get('error', backends.get('reason'))})")
        else:
            for name, info in backends.items():
                if info is None:
                    print(f"  {name:<20} (not created)")
                else:
                    print(f"  {name:<20} disabled={info['disabled']} "
                          f"max_size={info['max_size']} "
                          f"fully_connected={info['fully_connected']}")

    # ---- json dump -------------------------------------------------------
    if args.json is not None and rank == 0:
        payload = dict(
            world_size=world,
            ranks=gathered,
            checks=checks,
            checks_passed=len(checks) - len(failed),
            checks_total=len(checks),
            vllm_groups=groups,
            vllm_allreduce_backends=backends,
            env=dict(
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                nccl_debug=os.environ.get("NCCL_DEBUG"),
                nccl_algo=os.environ.get("NCCL_ALGO"),
                torch=torch.__version__,
                nccl=str(torch.cuda.nccl.version()),
            ),
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
