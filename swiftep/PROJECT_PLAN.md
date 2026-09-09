# 项目计划：vLLM MoE 专家并行运行时切换（Moebius 风格 TP↔EP）

> 简历项目定位：在 vLLM 主线（`v0.28.1rc0-560-g60ad959b6f`）上落地 Moebius 的核心思想——
> **运行时在 TP 与 EP 之间切换**（不重启、不丢请求），核心交付是 **CUDA IPC 融合直传内核**
> 做专家权重重切分（比 NCCL all-to-all 快），扩展为完整切换。硬件 **8×A100**。

## 0. 决策记录

| 项 | 决定 |
|---|---|
| 方向 | Moebius 风格 **TP↔EP 运行时切换**（arXiv 2606.26607，USC） |
| 核心交付 | 融合直传内核（CUDA IPC 直写专家权重，无 staging/all-to-all） |
| 扩展 | 统一内存管理 + 负载切换策略 + KV 重分布（完整切换） |
| 引擎 | vLLM 主线 |
| 模型 | Qwen3 系列 MoE（30B-A3B 开发 / 235B-A22B 若显存够） |
| 硬件 | **8×A100**（NVLink + CUDA IPC + CUDA graph 均可用；无需 TMA/FP8） |
| 基线 | 静态 TP / 静态 EP；重切分内核 vs NCCL all-to-all |

## 1. 关键事实

- Moebius 核心洞察：**EP 和 TP 是"同一模型的两套布局"**——权重/KV 字节完全相同，只是分片
  归属不同；切换的不可约代价只有"搬运 owner 变化的切片"。
- 三个机制：①统一内存管理器（固定地址连续 buffer + 双模式视图 alias）；②融合直传内核
  （CUDA IPC 直写对端，1 读 1 写，比 NCCL 少 HBM 往返）；③双运行时驻留（CUDA graph 常驻，
  切换=选状态而非重建）。
- 论文结果（8×H200, Qwen3-235B-A22B）：每个工作点追平更优静态并行；RL rollout +1.16–1.25×；
  单次切换 215–434ms；权重重切分 152ms（比 NCCL 快 1.49×）；内存开销 2.4%。
- **A100 兼容**：机制用 NVLink/CUDA IPC/CUDA graph，均 A100 可用（非 TMA/FP8 依赖）。

## 2. 阶段划分

### Phase 0 — 环境与基线
- 代码勘察：定位 vLLM 的 TP/EP 实现——专家权重如何分片（TP 的 row/col 切分 vs EP 的
  整专家）、`parallel_state` 的 TP/EP group、`fused_moe` 的 dispatch/combine、模型 runner
  的并行执行路径。
- 基线（8×A100）：静态 TP 与静态 EP 的 decode 并发扫描（复现"TP↔EP 交叉点"）；NCCL
  all-to-all 权重重切分耗时。

### Phase 1 — 核心：融合直传内核（专家权重重切分）
- 实现 CUDA IPC 直写内核：把 EP 布局的整专家（W13/W2）重切分为 TP 布局的 shard（及反向），
  直接写对端固定地址槽位，无 staging、无 all-to-all。
- benchmark vs NCCL all-to-all（目标对齐论文 1.49×）。
- 单测（张量布局正确性：重切分后逐位一致）。

### Phase 2 — 完整切换（扩展）
- 统一内存管理：专家权重/KV 放固定地址槽位，TP/EP 双视图 alias。
- 切换策略：基于并发/负载的 TP↔EP 决策点（复现交叉点判定）。
- 请求/KV 重分布（EP↔TP 的 attention sharding 切换）。
- 端到端：不重启、不丢请求地在 TP/EP 间切换，decode 正确性 < 1e-3。

### Phase 3 — 基准与报告
- 8×A100：静态 TP vs 静态 EP vs 切换系统，动态负载（突发 + RL rollout 长尾）。
- 记录吞吐/TTFT/TPOT/切换耗时/内存；瀑布式贡献报告；ruff/mypy；PR 描述。

## 3. 执行分工
- 本地（AI agent）：读源码、写代码、单测脚本、benchmark 脚本、文档。
- 8×A100（用户）：跑基线 + benchmark，按 README 回传日志。

## 4. 验收标准
1. 重切分内核：输出与基准逐位一致；比 NCCL all-to-all 快（目标 ≥1.3×）。
2. 端到端切换：TP↔EP 切换后 decode 输出与不切换一致（< 1e-3）；切换不丢请求。
3. 动态负载下切换系统优于固定 TP 或固定 EP（以实测为准）。
4. ruff/mypy 通过。

## 5. 风险与备选
- vLLM 执行器对 TP/EP 是启动期固定的，运行时切换需要深度改动 model_runner/executor。
  → 若改动过大，退化为"只交付融合直传内核 + 切换设计文档"（内核本身仍是完整可测贡献）。
- 8×A100 无 NVSwitch 全互联时，CUDA IPC 直写带宽受限 → 用 NVLink 拓扑感知的直写调度。
- Qwen3-235B 在 8×A100 放不下 → 用 Qwen3-30B-A3B 做完整切换，235B 仅做重切分内核微基准。

## 6. 参考
- Moebius: arXiv 2606.26607（USC）
- 同组 StreamEP: SOSP'26（barrier-free decode，未来扩展）
- vLLM 并行: vllm/distributed/parallel_state.py, vllm/model_executor/layers/fused_moe/
