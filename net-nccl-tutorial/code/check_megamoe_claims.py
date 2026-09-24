#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the MegaMoE appendix's factual claims against a vLLM checkout.

The appendix (``appendix-megamoe.md``) makes a number of *checkable* claims:
symbol names, magic constants, comment text, and one environment override.
This script asserts each one, so if upstream refactors MegaMoE the appendix's
staleness is detected mechanically instead of being discovered by a reader.

Symbol-based (not line-based) on purpose: line numbers drift, symbols do not.

    python check_megamoe_claims.py
    VLLM_REPO=/path/to/vllm python check_megamoe_claims.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _force_utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8()

HERE = Path(__file__).resolve().parent


def find_repo() -> Path | None:
    env = os.environ.get("VLLM_REPO")
    if env and (Path(env) / "vllm").is_dir():
        return Path(env)
    for parent in HERE.parents:
        for cand in (parent / "vllm", parent):
            if (cand / "vllm/config/kernel.py").exists():
                return cand
    fallback = Path.home() / "Documents/GitHub/vllm"
    return fallback if (fallback / "vllm").is_dir() else None


# (relative file, needle, human description)
CLAIMS: list[tuple[str, str, str]] = [
    # --- backend registry -------------------------------------------------
    ("vllm/config/kernel.py", '"deep_gemm_mega_moe"',
     "native MegaMoE backend is a valid MoEBackend"),
    ("vllm/config/kernel.py", '"flashinfer_moe_ep_mega_deep_gemm"',
     "FlashInfer + DeepGEMM megakernel backend"),
    ("vllm/config/kernel.py", '"flashinfer_moe_ep_mega_cutedsl"',
     "FlashInfer + CuteDSL megakernel backend"),
    ("vllm/config/kernel.py", "MEGA_MOE_BACKENDS = frozenset",
     "MEGA_MOE_BACKENDS set exists"),
    ("vllm/config/kernel.py", "DeepSeek-V4, Kimi K3",
     "comment names the models that use the mega path"),

    # --- Blackwell-only ---------------------------------------------------
    ("vllm/utils/flashinfer_moe_ep.py", "Every mega kernel is Blackwell-only",
     "mega kernels are arch-locked to Blackwell"),

    # --- the deadlock-avoidance override (the appendix's centrepiece) -----
    ("vllm/distributed/eplb/eplb_utils.py", "def override_envs_for_eplb",
     "EPLB env override entry point"),
    ("vllm/distributed/eplb/eplb_utils.py", "cooperative launch",
     "mentions cooperative launch"),
    ("vllm/distributed/eplb/eplb_utils.py", "causing a deadlock",
     "comment explains the deadlock risk"),
    ("vllm/distributed/eplb/eplb_utils.py", "NCCL_MAX_CTAS",
     "limits NCCL occupancy"),
    ("vllm/distributed/eplb/eplb_utils.py", "override_value = 8",
     "the value is 8"),

    # --- input staging ----------------------------------------------------
    ("vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py",
     "def prepare_megamoe_inputs", "staging kernel entry point"),
    ("vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py",
     "E8M0 group scales", "docstring mentions E8M0 group scales"),
    ("vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py",
     "hidden_size to be", "hidden_size divisibility error"),
    ("vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py",
     "tl.float8e4nv", "quantizes to fp8 e4m3"),
    ("vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py",
     "GROUP_K=32", "scale group size is 32"),

    # --- DeepGEMM mega kernels -------------------------------------------
    ("vllm/third_party/deep_gemm/mega/__init__.py", "class SymmBuffer",
     "SymmBuffer class"),
    ("vllm/third_party/deep_gemm/mega/__init__.py", "symm_mem.rendezvous",
     "uses torch symmetric memory rendezvous"),
    ("vllm/third_party/deep_gemm/mega/__init__.py", "group.size() == 1",
     "degenerates to a local buffer for single rank"),
    ("vllm/third_party/deep_gemm/mega/__init__.py", "def fp8_fp4_mega_moe",
     "fp8xfp4 megakernel entry"),
    ("vllm/third_party/deep_gemm/mega/__init__.py", "def bf16_mega_moe",
     "bf16 megakernel entry"),
    ("vllm/third_party/deep_gemm/mega/__init__.py",
     "def transform_weights_for_mega_moe", "weight transform entry"),
    ("vllm/third_party/deep_gemm/mega/__init__.py", "def _interleave_weights",
     "gate/up interleave helper"),
    ("vllm/third_party/deep_gemm/mega/__init__.py",
     "def _transpose_sf_for_utccp", "UTCCP scale transpose helper"),

    # --- workspace layout constants --------------------------------------
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "kNumCandidateBlockMs = 7", "7 candidate BLOCK_M values"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "{8, 16, 32, 64, 96, 128, 192}", "the candidate BLOCK_M list"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "kLCMCandidateBlockM = 384", "LCM used for worst-case padding"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "kNumBarrierSignalBytes = 128", "barrier block padded to an L2 line"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "hot atomics", "comment names the false-sharing hazard"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "struct TokenSrcMetadata", "12-byte combine write-back metadata"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "uint32_t rank_idx", "metadata field 1"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "uint32_t topk_idx", "metadata field 3"),
    ("vllm/third_party/deep_gemm/include/deep_gemm/layout/mega_moe.cuh",
     "Grid sync counters", "in-kernel grid sync counters"),

    # --- tests that encode the constraints -------------------------------
    ("tests/models/test_deepseek_v4_mega_moe.py",
     "def test_deep_gemm_mega_moe_capture_precedes_eplb",
     "capture-before-EPLB ordering constraint"),
    ("tests/models/test_deepseek_v4_mega_moe.py",
     "does_not_double_add_fused_shared_expert",
     "shared-expert double-add guard"),
]


def normalize(text: str) -> str:
    """Make a needle match across source hard-wraps and comment markers.

    Three normalizations, all cosmetic:
    * unify the Unicode dashes that appear in ``--option`` and comments;
    * drop comment markers, so a phrase split as ``causing a\\n# deadlock``
      still matches a single-line quote;
    * collapse all whitespace runs to one space.
    """
    for ch in ("\u2014", "\u2013", "\u2212"):
        text = text.replace(ch, "-")
    # Remove comment markers at line starts *before* collapsing whitespace.
    cleaned = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("#", "//", "*")):
            stripped = stripped.lstrip("#/*").lstrip()
        cleaned.append(stripped)
    return " ".join(" ".join(cleaned).split())


def main() -> int:
    repo = find_repo()
    if repo is None:
        print("ERROR: could not locate a vLLM checkout; set VLLM_REPO")
        return 2

    print(f"vLLM checkout: {repo}\n")
    ok = bad = 0
    missing_files: set[str] = set()

    cache: dict[str, str | None] = {}
    for rel, needle, desc in CLAIMS:
        if rel not in cache:
            p = repo / rel
            cache[rel] = (normalize(p.read_text(encoding="utf-8", errors="replace"))
                          if p.exists() else None)
        text = cache[rel]
        if text is None:
            missing_files.add(rel)
            print(f"MISSING FILE  {rel}  ({desc})")
            bad += 1
            continue
        if normalize(needle) in text:
            print(f"OK   {desc}")
            ok += 1
        else:
            print(f"FAIL {desc}")
            print(f"       {rel} does not contain {needle!r}")
            bad += 1

    print(f"\n{ok}/{len(CLAIMS)} claims verified")
    if missing_files:
        print("\nfiles not found in this checkout:")
        for f in sorted(missing_files):
            print(f"  {f}")
        print("(MegaMoE is Blackwell-only and evolving; absence here may be expected)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
