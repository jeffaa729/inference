#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the ``path:line`` citations used by the net/NCCL tutorial.

The tutorial cites real vLLM entities as ``path:LINE``. Line numbers drift as
the codebase moves; symbol names are stable. This script checks each citation
by *searching for the expected symbol* in a window around the cited line, so a
moved symbol reports ``MOVED`` (with the new line) instead of failing blindly.

Usage::

    python verify_citations.py
    # or point at a specific vLLM checkout:
    VLLM_REPO=/path/to/vllm python verify_citations.py

The tutorial lives in a different repository from the vLLM source it cites, so
the checkout is located by the ``VLLM_REPO`` env var first, then by probing the
usual sibling locations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_vllm_repo() -> Path | None:
    """Locate the vLLM checkout that the citations refer to.

    Order: ``VLLM_REPO`` env var, then candidate directories that contain the
    marker files we cite (``vllm/distributed/parallel_state.py``).
    """
    env = os.environ.get("VLLM_REPO")
    if env:
        p = Path(env)
        return p if (p / "vllm/distributed/parallel_state.py").exists() else None

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        # siblings of this repo, and this repo itself
        candidates += [parent / "vllm", parent]
    candidates += [Path.home() / "Documents/GitHub/vllm", Path.cwd()]

    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / "vllm/distributed/parallel_state.py").exists():
            return cand
    return None


REPO_ROOT = find_vllm_repo()
if REPO_ROOT is None:
    print(
        "ERROR: could not locate the vLLM checkout.\n"
        "Set VLLM_REPO to the repo root, e.g.\n"
        "  VLLM_REPO=C:\\Users\\Jeff\\Documents\\GitHub\\vllm python verify_citations.py"
    )
    raise SystemExit(2)
print(f"vLLM checkout: {REPO_ROOT}\n")

# (relative path, cited line, expected substring, human label)
CITATIONS: list[tuple[str, int, str, str]] = [
    # --- ch01 / ch02: TP collective entry points -------------------------
    ("vllm/distributed/communication_op.py", 14, "get_tp_group().all_reduce",
     "TP all-reduce semantic alias"),
    ("vllm/model_executor/layers/linear.py", 1769, "tensor_model_parallel_all_reduce",
     "RowParallelLinear all-reduce call"),
    ("vllm/model_executor/layers/linear.py", 606, "tensor_model_parallel_all_gather",
     "ColumnParallelLinear optional all-gather"),
    # --- ch04: group topology --------------------------------------------
    ("vllm/distributed/parallel_state.py", 1977, "ExternalDP x DP x PP x PCP x TP",
     "parallel layout order comment"),
    ("vllm/distributed/parallel_state.py", 2117, "EPLB group",
     "EPLB isolated process group"),
    ("vllm/distributed/parallel_state.py", 561, "1 << 22",
     "TP message-queue broadcaster size"),
    # --- ch04: all-reduce backend selection ------------------------------
    ("vllm/distributed/device_communicators/cuda_communicator.py", 305, "def all_reduce",
     "8-way all-reduce dispatch entry"),
    ("vllm/distributed/device_communicators/custom_all_reduce.py", 108,
     "_SUPPORTED_WORLD_SIZES", "custom AR supported world sizes"),
    ("vllm/distributed/device_communicators/custom_all_reduce.py", 406, "world_size > 8",
     "custom AR all-reduce world-size cap"),
    ("vllm/distributed/device_communicators/custom_all_reduce.py", 418, "fully_connected",
     "custom AR fully-connected requirement"),
    ("vllm/distributed/device_communicators/all_reduce_utils.py", 31,
     "CUSTOM_ALL_REDUCE_MAX_SIZES", "per-arch custom AR size table"),
    ("vllm/distributed/device_communicators/all_reduce_utils.py", 109,
     "NCCL_SYMM_MEM_ALL_REDUCE_CONFIG", "symm-mem vs custom AR ranges"),
    # --- ch04: all-to-all backends ---------------------------------------
    ("vllm/distributed/device_communicators/all2all.py", 44, "class AgRsAll2AllManager",
     "default AG-RS all-to-all manager"),
    ("vllm/distributed/device_communicators/all2all.py", 345, "RDMA so no SMs",
     "DeepEP LL uses no SMs"),
    ("vllm/distributed/device_communicators/all2all.py", 1005, "class DeepEPV2All2AllManager",
     "DeepEP v2 manager"),
    ("vllm/distributed/device_communicators/all2all.py", 1043, "_check_gin_support",
     "NCCL GIN admission check"),
    ("vllm/distributed/device_communicators/base_device_communicator.py", 47,
     "merged from dp and tp group", "EP group = DP x TP"),
    # --- ch03 / ch04: pynccl + symmetric memory --------------------------
    ("vllm/distributed/device_communicators/pynccl.py", 99, "world_size == 1",
     "pynccl disable gate"),
    ("vllm/distributed/device_communicators/pynccl.py", 158, "torch.zeros",
     "pynccl warm-up all_reduce"),
    ("vllm/distributed/device_communicators/pynccl_wrapper.py", 54,
     "NCCL_UNIQUE_ID_BYTES", "ncclUniqueId size"),
    ("vllm/distributed/device_communicators/pynccl_wrapper.py", 63, "23102",
     "ncclCommProperties layout version"),
    ("vllm/distributed/device_communicators/pynccl_allocator.py", 50,
     "VLLM_USE_NCCL_SYMM_MEM", "symmetric-memory master switch"),
    ("vllm/distributed/device_communicators/pynccl_allocator.py", 166, "22703",
     "minimum NCCL for symm memory"),
    ("vllm/utils/nccl.py", 30, "libnccl.so.2", "NCCL soname default"),
    # --- ch04 / ch05: config, startup, control plane ---------------------
    ("vllm/config/parallel.py", 143, "data_parallel_rpc_port", "DP rpc port default"),
    ("vllm/config/parallel.py", 147, "data_parallel_master_port", "DP master port default"),
    ("vllm/config/parallel.py", 197, "allgather_reducescatter",
     "default all2all backend"),
    ("vllm/config/parallel.py", 722, "def use_all2all", "all2all enablement predicate"),
    ("vllm/v1/executor/multiproc_executor.py", 143, "get_file_store_init_method",
     "single-node rendezvous via FileStore"),
    ("vllm/v1/worker/gpu_worker.py", 360, "NCCL_ASYNC_ERROR_HANDLING",
     "vLLM pops the Ray-set NCCL var"),
    ("vllm/model_executor/determinism/batch_invariant.py", 1154, "allreduce:tree",
     "determinism forces NCCL_ALGO"),
    ("vllm/utils/network_utils.py", 34, "def get_ip", "IP selection logic"),
    ("vllm/ray/ray_env.py", 40, "NCCL_", "NCCL env propagated to Ray workers"),
    ("vllm/distributed/eplb/eplb_utils.py", 103, "NCCL_MAX_CTAS",
     "EPLB sets NCCL_MAX_CTAS"),
    # --- ch07: comm/compute fusion (DBO) ---------------------------------
    ("vllm/v1/worker/ubatching.py", 95, "only one thread is running",
     "DBO single-thread correctness invariant"),
    ("vllm/v1/worker/ubatching.py", 150, "def dbo_enabled",
     "DBO zero-overhead gate"),
    ("vllm/v1/worker/ubatching.py", 193, "def dbo_get_previous_event",
     "DBO previous-event capture"),
    ("vllm/v1/worker/ubatch_utils.py", 39, "class SMControlContextManager",
     "DBO SM arbitration context manager"),
    ("vllm/v1/worker/ubatch_utils.py", 88, "VLLM_DBO_COMM_SMS",
     "SMs reserved for communication"),
    ("vllm/v1/worker/ubatch_utils.py", 96, "corrupts DP+EP",
     "ROCm: no CU reservation under DBO"),
    ("vllm/v1/worker/ubatch_utils.py", 118, "besides DeepGEMM",
     "only DeepGEMM accepts compute-SM limits"),
    ("vllm/config/parallel.py", 220, "enable_dbo", "DBO enable flag"),
    ("vllm/config/parallel.py", 225, "dbo_decode_token_threshold",
     "DBO decode threshold"),
    ("vllm/config/parallel.py", 589, "def num_ubatches", "ubatch count"),
    ("vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py", 119,
     "Capture a DeepEP event", "event captured before yielding"),
    ("vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py", 43,
     "xfer_atom_size", "DeepEP transfer atom (512 bytes)"),
    ("vllm/model_executor/layers/fused_moe/modular_kernel.py", 187,
     "class FusedMoEPrepareAndFinalize", "prepare/finalize interface"),
    ("vllm/model_executor/layers/fused_moe/modular_kernel.py", 248,
     "def supports_async", "async capability flag"),
    ("vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py", 61,
     "SUPPORTED_HIDDEN_SIZES", "DeepEP LL closed set of hidden sizes"),
    ("vllm/compilation/passes/fusion/sequence_parallelism.py", 40,
     "SP_MIN_HIDDEN_SIZE", "SP device gate"),
    ("vllm/compilation/passes/fusion/sequence_parallelism.py", 511,
     "does not directly yield", "SP is a prerequisite, not a win"),
    # --- ch08: advanced networking ---------------------------------------
    ("vllm/distributed/device_communicators/all2all.py", 171, "num_sms = 20",
     "DeepEP HT communication SM budget"),
    ("vllm/models/common/ops/sequence_parallel.py", 23, "def sp_all_gather",
     "SP all-gather + custom-collective fast path"),
    ("vllm/models/common/ops/sequence_parallel.py", 33, "sp_pad",
     "SP reduce-scatter alignment padding"),
    ("vllm/model_executor/layers/linear.py", 624,
     "class DCPGroupColumnParallelLinear", "DCP-group weight sharding"),
    ("vllm/config/parallel.py", 126, "prefill_context_parallel_size",
     "PCP config field"),
    ("vllm/config/parallel.py", 351, "decode_context_parallel_size",
     "DCP config field"),
    ("vllm/config/parallel.py", 363, "dcp_comm_backend",
     "DCP ag_rs vs a2a backend"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/base.py", 124,
     "class KVConnectorRole", "scheduler vs worker connector roles"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/base.py", 307,
     "wait_for_layer_load", "layer-granular KV load sync"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py",
     433, "future.set_result(False)", "HF3FS failure swallowed at completion check"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py",
     499, "hf3fs_metadata_server_url", "HF3FS metadata URL read but never used"),
    ("vllm/v1/kv_cache_layout.py", 15, "class KVCacheLayout",
     "KV cache physical layout descriptor"),
]

# How far a symbol may move before we call it "gone" rather than "moved".
SEARCH_WINDOW = 40


def main() -> int:
    ok = drifted = missing = 0
    for rel, cited, needle, label in CITATIONS:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"MISSING FILE  {rel}")
            missing += 1
            continue

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if cited <= len(lines) and needle in lines[cited - 1]:
            print(f"OK      {rel}:{cited}  {label}")
            ok += 1
            continue

        # Search a window around the cited line for a moved symbol.
        lo = max(0, cited - 1 - SEARCH_WINDOW)
        hi = min(len(lines), cited - 1 + SEARCH_WINDOW)
        found = next((i + 1 for i, ln in enumerate(lines[lo:hi], start=lo)
                      if needle in ln), None)
        if found is not None:
            print(f"MOVED   {rel}:{cited} -> {found}  ({label}; symbol {needle!r})")
            drifted += 1
        else:
            actual = lines[cited - 1].strip()[:70] if cited <= len(lines) else "<out of range>"
            print(f"GONE    {rel}:{cited}  ({label}; expected {needle!r}, got {actual!r})")
            missing += 1

    total = len(CITATIONS)
    print(f"\n{ok}/{total} exact, {drifted} moved, {missing} unresolved")
    if drifted or missing:
        print("Re-locate drifted symbols with:  grep -n '<symbol>' <path>")
    return 1 if (drifted or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
