"""Moebius expert-weight resharding benchmark: EP <-> TP.

Compares two data-movement strategies for converting expert weights between the
EP layout (each rank owns whole experts) and the TP layout (each rank owns a
shard of every expert), per Moebius (arXiv 2606.26607) Sec 3.1 / 4.3:

  1. NCCL all-to-all (baseline; stages through HBM).
  2. CUDA IPC direct write (no staging, no all-to-all; 1 HBM read + 1 NVLink
     write, mirroring vLLM's ``torch.multiprocessing.reductions`` IPC pattern in
     ``vllm/distributed/weight_transfer/packed_tensor.py``).

Both weight tensors are benchmarked:

  * W13 (gate+up, shard the 2I dim = dim 1 of (E, 2I, H))
  * W2  (down,      shard the  I dim = dim 2 of (E, H, I))

Run on N GPUs of one node (e.g. 8xA100):

    torchrun --nproc_per_node=8 swiftep/benchmarks/resharding_benchmark.py \
        --experts 128 --two_i 8192 --hidden 2048 --iters 100
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

# Optional fused CUDA kernel (JIT-compiled at first use; CUDA-only).
_FUSED = None
try:
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernels")
    )
    from fused_reshard_cuda import ep_to_tp_fused, tp_to_ep_fused

    _FUSED = (ep_to_tp_fused, tp_to_ep_fused)
except Exception:  # noqa: BLE001 - optional CUDA dependency
    _FUSED = None


# --------------------------------------------------------------------------- #
# NCCL all-to-all resharding (baseline)
# --------------------------------------------------------------------------- #
def ep_to_tp_nccl(local_ep: torch.Tensor, p: int, shard_dim: int) -> torch.Tensor:
    e_per = local_ep.shape[0]
    d = local_ep.shape[shard_dim]
    if shard_dim == 1:
        x = local_ep.reshape(e_per, p, d // p, -1).transpose(0, 1).contiguous()
    else:
        x = local_ep.reshape(e_per, -1, p, d // p).permute(2, 0, 1, 3).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    return out.reshape(e_per * p, *out.shape[2:])


def tp_to_ep_nccl(local_tp: torch.Tensor, p: int, shard_dim: int) -> torch.Tensor:
    e = local_tp.shape[0]
    d = local_tp.shape[shard_dim]
    if shard_dim == 1:
        x = local_tp.reshape(p, e // p, d, -1)
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x)
        return out.transpose(0, 1).reshape(e // p, d * p, -1)
    x = local_tp.reshape(p, e // p, -1, d)
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x)
    return out.permute(1, 2, 0, 3).reshape(e // p, out.shape[2], d * p)


# --------------------------------------------------------------------------- #
# CUDA IPC direct write
# --------------------------------------------------------------------------- #
def share_buffers(buf: torch.Tensor) -> list[torch.Tensor]:
    """Export ``buf`` via CUDA IPC; return a tensor view of every rank's buf."""
    _, args = reduce_tensor(buf)
    world = dist.get_world_size()
    all_args = [None] * world
    dist.all_gather_object(all_args, args)

    peers: list[torch.Tensor] = []
    for r in range(world):
        if r == dist.get_rank():
            peers.append(buf)
            continue
        a = list(all_args[r])
        a[6] = torch.cuda.current_device()
        peers.append(rebuild_cuda_tensor(*a))
    return peers


def ep_to_tp_ipc(local_ep: torch.Tensor, dst: list[torch.Tensor],
                 p: int, rank: int, shard_dim: int) -> torch.Tensor:
    e_per = local_ep.shape[0]
    d = local_ep.shape[shard_dim]
    for dest in range(p):
        src = local_ep.narrow(shard_dim, dest * (d // p), d // p)
        dst[dest][rank * e_per : (rank + 1) * e_per].copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return dst[rank]


def tp_to_ep_ipc(local_tp: torch.Tensor, dst: list[torch.Tensor],
                 p: int, rank: int, shard_dim: int) -> torch.Tensor:
    e = local_tp.shape[0]
    e_per = e // p
    shard = local_tp.shape[shard_dim]  # D/P: my TP slice size
    for dest in range(p):
        dst[dest].narrow(shard_dim, rank * shard, shard).copy_(
            local_tp[dest * e_per : (dest + 1) * e_per], non_blocking=True
        )
    torch.cuda.synchronize()
    return dst[rank]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def timeit(fn, iters: int) -> float:
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--two_i", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--fused", action="store_true",
                    help="also benchmark the fused CUDA kernel (W13 only)")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    e, two_i, h, p = args.experts, args.two_i, args.hidden, world
    assert e % p == 0 and two_i % p == 0, "require E and 2I divisible by ranks"
    i = two_i // 2  # intermediate size of a single SwiGLU projection
    assert i % p == 0, "require I divisible by ranks"
    dtype = torch.bfloat16
    dev = torch.device(f"cuda:{rank}")

    # (name, EP-local source, TP-local source, shard_dim, nbytes_moved)
    cases = [
        ("W13", torch.randn(e // p, two_i, h, dtype=dtype, device=dev),
         torch.randn(e, two_i // p, h, dtype=dtype, device=dev), 1,
         e * two_i * h),
        ("W2", torch.randn(e // p, h, i, dtype=dtype, device=dev),
         torch.randn(e, h, i // p, dtype=dtype, device=dev), 2,
         e * h * i),
    ]

    for name, ep_src, tp_src, shard_dim, nbytes in cases:
        # --- correctness ---
        ref = ep_to_tp_nccl(ep_src, p, shard_dim)
        tp_buf = torch.empty_like(ref)
        tp_peers = share_buffers(tp_buf)
        dist.barrier()
        got = ep_to_tp_ipc(ep_src, tp_peers, p, rank, shard_dim)
        dist.barrier()
        assert torch.equal(ref, got), f"{name} EP->TP mismatch on rank {rank}"

        ref2 = tp_to_ep_nccl(tp_src, p, shard_dim)
        ep_buf = torch.empty_like(ref2)
        ep_peers = share_buffers(ep_buf)
        dist.barrier()
        got2 = tp_to_ep_ipc(tp_src, ep_peers, p, rank, shard_dim)
        dist.barrier()
        assert torch.equal(ref2, got2), f"{name} TP->EP mismatch on rank {rank}"

        # --- benchmark ---
        for _ in range(args.warmup):
            ep_to_tp_nccl(ep_src, p, shard_dim)
        nccl_ms = timeit(lambda: ep_to_tp_nccl(ep_src, p, shard_dim), args.iters)
        for _ in range(args.warmup):
            ep_to_tp_ipc(ep_src, tp_peers, p, rank, shard_dim)
        ipc_ms = timeit(lambda: ep_to_tp_ipc(ep_src, tp_peers, p, rank, shard_dim), args.iters)

        moved = nbytes * dtype.itemsize * (p - 1) / p
        if rank == 0:
            print(f"{name} EP->TP  E={e} P={p}: NCCL {nccl_ms:.3f} ms | "
                  f"IPC {ipc_ms:.3f} ms | speedup {nccl_ms/ipc_ms:.2f}x | "
                  f"IPC BW {moved/ipc_ms/1e6:.1f} GB/s")

    if args.fused:
        if _FUSED is None:
            if rank == 0:
                print("fused CUDA kernel unavailable (JIT compile failed)")
        else:
            ep_to_tp_fused, tp_to_ep_fused = _FUSED
            name, ep_src, tp_src, shard_dim, nbytes = cases[0]  # W13 only
            moved = nbytes * dtype.itemsize * (p - 1) / p

            # EP -> TP
            ref = ep_to_tp_nccl(ep_src, p, shard_dim)
            tp_buf = torch.empty_like(ref)
            tp_peers = share_buffers(tp_buf)
            dist.barrier()
            got = ep_to_tp_fused(ep_src, tp_peers, p, rank)
            dist.barrier()
            assert torch.equal(ref, got), f"fused {name} EP->TP mismatch on rank {rank}"
            for _ in range(args.warmup):
                ep_to_tp_fused(ep_src, tp_peers, p, rank)
            fused_ms = timeit(lambda: ep_to_tp_fused(ep_src, tp_peers, p, rank), args.iters)
            if rank == 0:
                print(f"{name} EP->TP  E={e} P={p}: fused CUDA {fused_ms:.3f} ms | "
                      f"BW {moved/fused_ms/1e6:.1f} GB/s")

            # TP -> EP
            ref2 = tp_to_ep_nccl(tp_src, p, shard_dim)
            ep_buf = torch.empty_like(ref2)
            ep_peers = share_buffers(ep_buf)
            dist.barrier()
            got2 = tp_to_ep_fused(tp_src, ep_peers, p, rank)
            dist.barrier()
            assert torch.equal(ref2, got2), f"fused {name} TP->EP mismatch on rank {rank}"
            for _ in range(args.warmup):
                tp_to_ep_fused(tp_src, ep_peers, p, rank)
            fused_ms2 = timeit(lambda: tp_to_ep_fused(tp_src, ep_peers, p, rank), args.iters)
            if rank == 0:
                print(f"{name} TP->EP  E={e} P={p}: fused CUDA {fused_ms2:.3f} ms | "
                      f"BW {moved/fused_ms2/1e6:.1f} GB/s")

    # Release CUDA IPC peer mappings while every rank is still alive, so no
    # rank exits while another still holds its buffer open. This avoids the
    # "Producer process has been terminated before all shared CUDA tensors
    # released" warning at shutdown.
    tp_peers = ep_peers = None  # noqa: F841
    import gc

    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
