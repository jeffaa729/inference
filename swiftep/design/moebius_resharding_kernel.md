# Phase 1 设计：专家权重重切分融合直传内核（Moebius 核心）

## 1. 目标

在 vLLM 里实现 Moebius 的「融合直传内核」：把专家权重在 **EP 布局 ↔ TP 布局** 之间重切分，
用 CUDA IPC 直写对端固定地址，做到 **1 读 1 写、无 staging buffer、无 all-to-all**，
benchmark 对比 NCCL all-to-all（论文目标 1.49×）。

## 2. 权重布局（Moebius §3.1）

记 `E`=专家总数、`P`=切换组 rank 数、`H`=hidden、`I`=专家中间维（SwiGLU 的 2I）：

| 布局 | W13（gate+up 合并） | W2（down） |
|---|---|---|
| EP（每 rank 整专家） | `(E/P, 2I, H)` | `(E/P, H, I)` |
| TP（每 rank 每专家 shard） | `(E, 2I/P, H)` | `(E, H, I/P)` |

- **EP→TP**：permute（本地整专家打包成 per-peer 块）→ exchange（每 rank 拿到每个专家的 shard）。
- **TP→EP**：exchange → permute（收到的 shard 交错拼回整专家）。
- W13 沿中间维 `2I` 切；W2 沿中间维 `I` 切；两方向对称。

## 3. 内核设计（融合直传）

1. **统一地址槽位**：每个 rank 预分配一块连续 GPU buffer，专家权重按层放入固定槽位；
   槽位预留 `N+1`（多 1 个作 in-place 安全区），TP/EP 两套视图 alias 同一 buffer。
2. **CUDA IPC**：每 rank 把 buffer export 成 `cudaIpcMemHandle`，映射对端地址到本地
   （`torch.cuda.ipc_collect` / 手动 CUDA IPC），直写对端槽位，不经 NCCL staging。
3. **内核**：一个 CUDA kernel，读源布局槽位、按 per-peer 描述符直写目标布局槽位；
   专家权重重切分 + KV 页重分布共用"读源→写对端"这一模式（Phase 1 只做权重）。
4. **正确性**：重切分后与基准（NCCL all-to-all 实现同样重切分）逐位一致。

## 4. 与 vLLM 的关系（Phase 1 范围）

- Phase 1 交付**独立的重切分内核 + 微基准**，不依赖 vLLM 执行器改动。
- 输入：EP 布局的专家权重张量（或 TP 布局），输出：切分后的张量（另一布局）。
- 用 `torch.distributed`（NCCL all-to-all）做正确性基准；用 CUDA IPC 直写做目标实现。
- 后续 Phase 2 再把该内核接入 vLLM 的 model_runner/executor 做完整切换。

## 5. 基准与验收

- 微基准：8×A100，Qwen3-30B-A3B（或 235B）的专家权重张量，测 EP→TP 与 TP→EP 重切分耗时：
  - NCCL all-to-all 基线
  - CUDA IPC 直写实现
- 验收：逐位一致 + 直写比 NCCL 快（目标 ≥1.3×，论文 1.49×）。

## 6. 文件规划（都在 swiftep/ 下）

```text
swiftep/
  benchmarks/res hard_kernel.py      # 重切分微基准（NCCL vs CUDA IPC 直写）
  kernels/                           # 内核源码（.cu / triton）
  tests/test_resharding.py           # 布局正确性单测
```
