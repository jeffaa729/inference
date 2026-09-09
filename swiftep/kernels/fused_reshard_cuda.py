"""Fused CUDA direct-transfer kernel for Moebius EP <-> TP weight resharding.

Implements the "fused direct transfer" from Moebius (arXiv 2606.26607) Sec 4.3:
a single kernel launch copies every owner-changed slice directly into the
peer's target buffer (no staging, no all-to-all, one HBM read + one NVLink
write). This is the optimized counterpart to the torch ``copy_`` loop in
``benchmarks/resharding_benchmark.py`` (which launches P kernels and issues
strided copies).

Scope: the W13 (gate+up) projection, sharded along the 2I dim. The block being
copied is a contiguous ``(2I/P, H)`` slice, moved as raw bytes so it works for
any dtype. W2 (down) is analogous and left to the torch path for now.

Correctness is checked in the benchmark (NCCL vs torch-copy IPC vs fused):
    torchrun --nproc_per_node=8 swiftep/benchmarks/resharding_benchmark.py --fused
"""

import torch
from torch.utils.cpp_extension import load_inline

_CSRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// E copies of contiguous `block_elems`-element blocks (each `itemsize` bytes),
// one per thread block. Source pointer and destination peer/offset per block
// are precomputed on the host and passed as device arrays.
__global__ void fused_reshard_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* const* __restrict__ dst_ptrs,     // P device pointers
    const int64_t* __restrict__ src_offsets,   // E (elements)
    const int64_t* __restrict__ dst_offsets,   // E (elements)
    const int32_t* __restrict__ dst_peer,      // E (peer index)
    int64_t block_elems, int itemsize) {
  const int64_t e = blockIdx.x;
  const uint8_t* s = src + src_offsets[e] * itemsize;
  uint8_t* t = dst_ptrs[dst_peer[e]] + dst_offsets[e] * itemsize;
  const int64_t nbytes = block_elems * itemsize;

  const uint64_t* s8 = reinterpret_cast<const uint64_t*>(s);
  uint64_t* t8 = reinterpret_cast<uint64_t*>(t);
  const int64_t nvec = nbytes / 8;
  const int64_t tail = nbytes % 8;
  for (int64_t i = threadIdx.x; i < nvec; i += blockDim.x) {
    t8[i] = s8[i];
  }
  if (tail && threadIdx.x == 0) {
    for (int64_t i = nvec * 8; i < nbytes; ++i) {
      t[i] = s[i];
    }
  }
}

void launch(torch::Tensor src, torch::Tensor dst_ptrs, torch::Tensor src_offsets,
            torch::Tensor dst_offsets, torch::Tensor dst_peer,
            int64_t block_elems, int itemsize, int64_t num_blocks) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_reshard_kernel<<<(unsigned)num_blocks, 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(src.data_ptr()),
      reinterpret_cast<uint8_t* const*>(dst_ptrs.data_ptr()),
      reinterpret_cast<const int64_t*>(src_offsets.data_ptr()),
      reinterpret_cast<const int64_t*>(dst_offsets.data_ptr()),
      reinterpret_cast<const int32_t*>(dst_peer.data_ptr()),
      block_elems, itemsize);
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = load_inline(
            name="fused_reshard_cuda",
            cpp_sources=r"""
#include <torch/extension.h>
#include <cstdint>
void launch(torch::Tensor src, torch::Tensor dst_ptrs,
            torch::Tensor src_offsets, torch::Tensor dst_offsets,
            torch::Tensor dst_peer, int64_t block_elems, int itemsize,
            int64_t num_blocks);
""",
            cuda_sources=_CSRC,
            functions=["launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _kernel


def _plan_ep_to_tp(local_ep, p, rank, itemsize):
    e_per, two_i, h = local_ep.shape
    two_i_per = two_i // p
    block_elems = two_i_per * h
    assert block_elems * itemsize % 8 == 0, "require 8-byte-aligned blocks"
    dev = local_ep.device
    e_local = torch.arange(e_per, dtype=torch.int64, device=dev)
    d = torch.arange(p, dtype=torch.int64, device=dev)
    src_off = (e_local[:, None] * (two_i * h) + d[None, :] * two_i_per * h).reshape(-1)
    dst_off = ((rank * e_per + e_local[:, None]) * two_i_per * h).expand(e_per, p).reshape(-1)
    dst_peer = d[None, :].expand(e_per, p).reshape(-1).to(torch.int32)
    return src_off, dst_off, dst_peer, block_elems, e_per * p


def _plan_tp_to_ep(local_tp, p, rank, itemsize):
    e, two_i_per, h = local_tp.shape
    e_per = e // p
    two_i = two_i_per * p
    block_elems = two_i_per * h
    assert block_elems * itemsize % 8 == 0, "require 8-byte-aligned blocks"
    dev = local_tp.device
    d = torch.arange(p, dtype=torch.int64, device=dev)
    e_local = torch.arange(e_per, dtype=torch.int64, device=dev)
    src_off = (d[:, None] * e_per * two_i_per * h + e_local[None, :] * two_i_per * h).reshape(-1)
    dst_off = (e_local[None, :] * (two_i * h) + rank * two_i_per * h).expand(p, e_per).reshape(-1)
    dst_peer = d[:, None].expand(p, e_per).reshape(-1).to(torch.int32)
    return src_off, dst_off, dst_peer, block_elems, e


_EP_PLAN_CACHE: dict[tuple, tuple] = {}
_TP_PLAN_CACHE: dict[tuple, tuple] = {}


def _launch(local, dst_ptrs_list, plan, itemsize):
    src_off, dst_off, dst_peer, block_elems, num_blocks = plan
    dst_ptrs = torch.tensor([t.data_ptr() for t in dst_ptrs_list],
                            dtype=torch.int64, device=local.device)
    _get_kernel().launch(local, dst_ptrs, src_off, dst_off, dst_peer,
                         block_elems, itemsize, num_blocks)
    torch.cuda.synchronize()


def ep_to_tp_fused(local_ep, dst: list[torch.Tensor], p: int, rank: int):
    key = (local_ep.shape, p, rank)
    plan = _EP_PLAN_CACHE.get(key)
    if plan is None:
        plan = _plan_ep_to_tp(local_ep, p, rank, local_ep.element_size())
        _EP_PLAN_CACHE[key] = plan
    _launch(local_ep, dst, plan, local_ep.element_size())
    return dst[rank]


def tp_to_ep_fused(local_tp, dst: list[torch.Tensor], p: int, rank: int):
    key = (local_tp.shape, p, rank)
    plan = _TP_PLAN_CACHE.get(key)
    if plan is None:
        plan = _plan_tp_to_ep(local_tp, p, rank, local_tp.element_size())
        _TP_PLAN_CACHE[key] = plan
    _launch(local_tp, dst, plan, local_tp.element_size())
    return dst[rank]


if __name__ == "__main__":
    raise SystemExit(
        "Run the benchmark to exercise this kernel: "
        "torchrun --nproc_per_node=8 swiftep/benchmarks/resharding_benchmark.py --fused"
    )
