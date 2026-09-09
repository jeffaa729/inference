# Moebius 风格 TP↔EP 运行时切换：vLLM MoE 专家并行优化

在 vLLM 主线（`v0.28.1rc0-560-g60ad959b6f`）上落地 Moebius（arXiv 2606.26607，USC）的核心思想：

- **运行时 TP↔EP 切换**（不重启、不丢请求）。
- 核心交付：**融合直传内核**（CUDA IPC 直写专家权重，无 staging/all-to-all，vs NCCL）。
- 扩展：统一内存管理 + 切换策略 + KV 重分布（完整切换）。

> 注：目录名 `swiftep/` 是早期探索 SwiftEP 时命名的，现保留作为项目目录；
> `design/swiftep_backend_design.md` 是早期 SwiftEP 探索的历史产物，已弃用。

## 决策记录

| 项 | 决定 |
|---|---|
| 方向 | Moebius 风格 TP↔EP 运行时切换 |
| 核心 | 专家权重重切分（CUDA IPC 直写 vs NCCL） |
| 扩展 | 统一内存管理 + 切换策略 + KV 重分布 |
| 引擎 | vLLM main |
| 模型 | Qwen3 系列 MoE（30B-A3B 开发 / 235B-A22B） |
| GPU | 8×A100（NVLink + CUDA IPC + CUDA graph，无需 TMA/FP8） |
| 基线 | 静态 TP / 静态 EP；重切分内核 vs NCCL all-to-all |

## 目录结构

```text
swiftep/
  PROJECT_PLAN.md                          # 项目计划
  README.md                                # 本文件
  design/
    moebius_resharding_kernel.md           # Phase 1 重切分内核设计
    moebius_vllm_switch_design.md          # Phase 2 完整切换设计
  benchmarks/
    resharding_benchmark.py                # 重切分微基准（NCCL vs CUDA IPC vs fused）
  kernels/
    fused_reshard_cuda.py                  # fused CUDA 直传内核（load_inline）
  tests/
    test_resharding.py                     # torch 单测（CPU 纯逻辑）
    verify_resharding_numpy.py             # numpy 本地验证（无需 GPU）
  switch_policy.py                         # TP↔EP 切换策略 + 负载仿真
  2606.26607v1.pdf                         # Moebius 论文
```

> vLLM 源码改动单独留在源码树内（`vllm/model_executor/layers/fused_moe/...`），
> 本目录只放脚本、文档等非源码产物。

## 本地验证（无需 GPU）

```bash
# 验证重切分数学（NCCL all-to-all 与 CUDA IPC 直写，W13 与 W2 双向）
python swiftep/tests/verify_resharding_numpy.py

# torch 版单测（需 torch）
python -m pytest swiftep/tests/test_resharding.py -v

# 切换策略 + 负载仿真（numpy，无需 GPU）
python swiftep/switch_policy.py
```

## 8×A100 基准

```bash
torchrun --nproc_per_node=8 swiftep/benchmarks/resharding_benchmark.py \
    --experts 128 --two_i 8192 --hidden 2048 --iters 100
```

## 状态

- [x] Phase 0：读 Moebius + 定位 vLLM TP/EP 注入点
- [x] Phase 1：重切分内核（NCCL 基线 + CUDA IPC 直写，torch 版）；数学已 numpy 验证
- [ ] Phase 1：fused CUDA 内核（单 kernel 直写）+ 8×A100 实测
- [ ] Phase 2：统一内存管理 + 切换策略 + KV 重分布（设计已完成）
- [ ] Phase 3：动态负载 benchmark + 性能报告

## 参考

- Moebius: https://arxiv.org/abs/2606.26607 （USC）
- 同组 StreamEP: SOSP'26（barrier-free decode，未来扩展）
- vLLM 并行: `vllm/distributed/parallel_state.py`, `vllm/model_executor/layers/fused_moe/`
