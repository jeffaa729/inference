# Phase 2 设计：vLLM 上的 TP↔EP 运行时切换（Moebius 完整切换）

## 1. 目标与范围

在 vLLM 上实现"不重启、不丢请求"的 TP↔EP 运行时切换。Phase 1 交付的重切分内核
（`swiftep/design/moebius_resharding_kernel.md`）是数据面核心；本设计把 Moebius 的四个机制
映射到 vLLM 架构，并给出切合 vLLM 的落地方案与分阶段范围。

## 2. Moebius → vLLM 概念映射

| Moebius（自研 serving 系统） | vLLM 对应物 | 难度 |
|---|---|---|
| per-rank scheduler | `vllm/v1/engine` + `gpu_model_runner` | — |
| 统一内存管理器（UMM） | 模型加载（`model_loader`）+ 固定地址 buffer | 中 |
| 融合直传内核 | Phase 1 内核（`prepare_finalize` 侧或独立 op） | 中（已做） |
| 双运行时驻留 | 双 CUDA graph 捕获（TP/EP 各一套） | 高 |
| 切换策略 | 调度器/协调器：按并发选 TP 或 EP | 低 |
| 请求/KV 重分布 | executor + KV cache 所有权改写 | **高（最难）** |

## 3. 四个机制在 vLLM 的落点

### 3.1 统一内存管理（UMM）
- vLLM 现状：专家权重由 `model_loader` 按启动期的 `enable_expert_parallel` 一次性分片，
  分片后各 rank 的 `nn.Module` 参数是固定形状（EP 整专家 vs TP row/col shard）。
- 改动：为专家权重预留一块固定地址的连续 buffer，建立 TP/EP 两套视图（alias 同一 buffer），
  参考 Moebius 的 `N+1` 槽位做 in-place 安全重切分。vLLM 已有 `model_runner` 的
  "fixed-address 权重 + CUDA graph 捕获" 基础（见 `vllm/v1/worker/gpu_model_runner.py`），
  可复用其 buffer 管理。
- 注意：vLLM 的 TP 专家 shard 是 `ColumnParallel/RowParallel`（row/col 切分，且带
  all-reduce），与 EP 的整专家 + all-to-all 是两套 forward 代码路径；切换 = 换 forward 路径
  + 换权重视图，而非简单换张量。

### 3.2 融合直传内核（Phase 1 已交付）
- 权重重切分：EP↔TP 的 `W13/W2` 重排，CUDA IPC 直写（无 staging/all-to-all）。
- KV 重分布：paged KV 的 gather(读 page table)→直写对端页 →scatter，需要 page-table 索引
  描述符（Moebius §4.3 的 KV kernel）。这是 Phase 2 的 KV 部分。

### 3.3 双运行时驻留（runtime preserving）
- vLLM 已用 CUDA graph 捕获 decode 路径；改动：捕获 **TP 与 EP 两套 graph**，均常驻，
  切换 = 选择复用哪套（不重建）。`gpu_model_runner` 的 graph 捕获/回放需要支持双模式。
- 难点：两套 graph 的输入/输出 buffer、attention 元数据（block table、KV 布局）不同，
  切换时要一致地切元数据。

### 3.4 切换策略（switch policy）
- 复现 Moebius §2.1 的 TP↔EP 交叉点：测出"并发 B 低于某阈值用 TP、高于用 EP"。
- vLLM 落点：`vllm/v1/engine` 的调度循环或 coordinator，按当前 in-flight 数（或 token 数）
  决定目标模式，广播给所有 rank；加滞回（hysteresis）避免抖动。
- 策略本身简单（阈值+滞回），成本低，是"锦上添花"。

## 4. 最难部分：请求/KV 重分布

EP（DP-attention）与 TP（TP-attention）的 KV 所有权不同：
- EP：每 rank 拥有一个请求子集，存该子集的**全部 KV head**。
- TP：每 rank 服务**全部请求**，但只存 **KV head shard**（Qwen3 4 KV head @ TP8 = 每 head 2 rank）。

切换 = 重写请求元数据（host 侧，便宜）+ KV 页所有权（GPU 侧，贵，用 §3.2 的 KV 内核）。
vLLM 落点：`gpu_model_runner` 的 KV cache 管理 + `block_table`/`input_batch` 的所有权切换。
**这是全项目最深的 executor 改动。**

## 5. 现实的分阶段范围（建议）

| 阶段 | 交付 | 可测性 | 状态 |
|---|---|---|---|
| P1 | 权重重切分内核 + 微基准（vs NCCL） | ✅ 独立可测 | 已写 torch 版 |
| P2a | 权重重切分 fused CUDA 内核（单 kernel 直写） | ✅ 微基准 | 待做 |
| P2b | UMM + 双 CUDA graph 驻留 + 切换策略（权重级切换，不含 KV） | ⚠️ 需真机 | 设计完成 |
| P2c | 请求/KV 重分布（完整切换） | ⚠️ 最难 | 设计完成 |
| P3 | 8×A100 动态负载 benchmark + 报告 | ✅ | 待做 |

**建议主线：P1 → P2a → P2b（权重级切换，含双 graph + 策略）→ P3。P2c（KV）作为 stretch，**
因为 KV 重分布是 vLLM executor 最深、最易出错的部分，可留到最后或只出设计。

## 6. 验收（Phase 2b 权重级切换）

1. 切换后 decode 输出与"固定 TP / 固定 EP"一致（< 1e-3）；切换不丢请求。
2. 权重重切分内核比 NCCL 快（P1 目标 ≥1.3×）。
3. 动态负载（突发 + RL rollout 长尾）下，切换系统优于固定 TP 或固定 EP。

## 8. vLLM 集成精确落点（真机实现直接上手）

| 机制 | 精确文件/函数 | 改动 |
|---|---|---|
| 并行态运行时化 | `vllm/config/parallel.py` `enable_expert_parallel` | 启动期 bool → 运行期可切换的 `mode`（tp/ep） |
| 双视图权重加载 | `vllm/model_executor/model_loader/default_loader.py:365`（EP 过滤分支）、`weight_utils.py:1283` `sharded_weight_loader` | 同时建 TP( row/col shard ) 与 EP( 整专家 ) 两套视图，alias 固定地址 buffer |
| 双 CUDA graph 捕获 | `gpu_model_runner.py:6905` `capture_model()`、`:7059` `_capture_cudagraphs()`、`:919` `CudagraphDispatcher` | 捕获 TP 与 EP 两套 graph 常驻，按模式 dispatch |
| 每步模式选择 | `gpu_model_runner.py:4274` `execute_model()` | 每步读切换策略，选 TP/EP graph + 对应 runtime |
| KV 所有权切换 | `gpu_model_runner.py:7491` `initialize_kv_cache()` + `:7419` `initialize_kv_cache_tensors()` | EP(DP-attn 整 head) ↔ TP(TP-attn head shard) 的 KV 重分布 |
| MoE forward 路径 | `vllm/model_executor/layers/fused_moe/layer.py` `moe_parallel_config` | 按当前模式选 EP(all-to-all) 或 TP(all-reduce) forward |
| 重切分内核调用 | 上述 execute_model 的切换临界区 | 调 `swiftep/kernels/fused_reshard_cuda.py`（权重） + 未来 KV 内核 |

## 9. 参考
- Moebius §3–§4（arXiv 2606.26607）：权重重切分 §3.1、请求/KV 重分布 §3.2、UMM §4.2、
  融合内核 §4.3、运行时驻留 §4.4、切换策略 §4.5。
- vLLM：`vllm/v1/worker/gpu_model_runner.py`（CUDA graph + KV 管理）、
  `vllm/model_executor/layers/fused_moe/`（EP 路径）、`vllm/model_loader/`（权重分片）。
