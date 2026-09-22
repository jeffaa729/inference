# 附录 C　速查卡（打印版）

> 面试前 10 分钟看这个。所有 ✅ 都可以在 `verify_citations.py` 里核对。

---

## C.1 单位与换算

```
400 Gb/s = 50 GB/s          （÷8）
NVLink 4 (H100) 双向 900 GB/s → 单向 450 GB/s   ← 说清口径！
NVLink 5 (B200) 双向 1800 GB/s
PCIe 3/4/5 x16 ≈ 16 / 32 / 64 GB/s（单向）
IB: HDR 200Gb/s=25GB/s · NDR 400Gb/s=50GB/s · XDR 800Gb/s=100GB/s
1 MiB = 2^20 B · 1 GiB = 2^30 B（vLLM 代码用 MiB）
```

## C.2 延迟量级

```
NVLink 亚 μs · PCIe P2P ~1-2μs · IB 端到端 2-5μs · RoCE 3-8μs
TCP 内核栈 10-30μs · ZMQ/UDS 10-50μs
Kernel launch 3-10μs（decode 的隐形敌人）
```

## C.3 集合通信

```
α-β 模型：T = 步数 × α + 搬运量 / BW

all-reduce:
  ring  步数 2(N-1)   每 rank 搬运 2(N-1)/N·S → 2S    大消息最优
  tree  步数 2log₂N   每 rank 搬运 ≈2S               小消息最优
  halving-doubling  log₂N 步，搬运 ≈S  但要求 N 是 2 的幂 + 全交换模式

所有操作的搬运量都是 (N-1)/N × 总字节数 的量级
all-reduce = reduce-scatter + all-gather      ← 最有用的分解
all-gather 输出 ×N ｜ all-to-all 总量不变只换分布  ← 最易混的一对
```

## C.4 NCCL

```
三层选择：算法（Ring/Tree/NVLS/CollNet）× 协议（LL/LL128/Simple）× channel 数
ncclUniqueId = 128 字节，rank0 生成，走 gloo/socket 分发（鸡生蛋）
ncclGroupStart/End = 批量并发下发（vLLM 用它拼变长 all-gather）
能力字段：multimemSupport(SHARP) · ginType(GIN) · deviceApiSupport · nLsaTeams

版本要求：
  window register (对称内存) ≥ 2.27.03
  ncclCommSuspend/Resume     ≥ 2.29.7
  对称内存完整支持            ≥ 2.27.3 (22703)
  DeepEP v2 (GIN)            ≥ 2.30.4 (23004)
  vLLM 声明的结构体布局       = 2.31.2 (23102)
```

## C.5 vLLM 并行组（✅ `parallel_state.py:1977`）

```
布局顺序：ExternalDP × DP × PP × PCP × TP     ← TP 是最后一维，所以 TP 组相邻
EP_SIZE = TP × DP × PCP                        ← ✅ :2087-2096
world_size = pp × tp × pcp                     （DCP 不增加 world size！）
PP 组 = transpose(2,4) ｜ DP 组 = transpose(1,4) ｜ EP 组 = transpose(1,2)
```

## C.6 vLLM 8 路 all-reduce（✅ `cuda_communicator.py:305-377`）

```
① NCCL 对称内存 → ② QuickReduce(ROCm) → ③ FlashInfer PCIe IPC
→ ④ FlashInfer → ⑤ AITER → ⑥ vLLM custom AR → ⑦ torch symm_mem
→ ⑧ PyNCCL → ⑧b torch.distributed（兜底）

⚠️ 启动日志打印的顺序 ≠ 实际 dispatch 顺序
custom AR: world_size≤8 · dtype∈{fp32,fp16,bf16} · size%16==0 · 弱连续
           world_size>2 时必须 fully_connected（NVLink 全互联）
           max_size 查表：H100 TP2/4/6/8 = 64M/32M/512K/256K
对称内存 vs custom AR 区间：TP4 (16K,512K)｜TP8 (16K,128K) 用 custom AR
```

## C.7 MoE all-to-all（ch08）

```
EP_SIZE = TP × DP × PCP
dispatch: token → 持有目标专家的 rank ｜ combine: 反向 + 加权求和
三套坐标系：全局(router) / 物理(LL&NIXL 寻址) / 本地(kernel)
expert_map[global] = local 或 -1（不在本 rank）
轮转放置：owner = gid % ep_size; local = gid // ep_size
         ← 只有 DeepEP-LL 和 NIXL 支持，其它退化为线性
-1 sentinel：HT 把它换成"肯定不属于本 rank"的合法 id（rank0→num_experts-1，其它→0）

后端：allgather_reducescatter(默认) / deepep_ht(prefill,20SM) /
      deepep_ll(decode,0SM,支持graph) / deepep_v2(需 GIN) / nixl_ep / flashinfer_nvlink_*
约束：LL/NIXL hidden ∈ [2048,2560,3072,4096,5120,6144,7168,8192]
      fp8 需 hidden%128==0 ｜ LL/NIXL 不支持 per-token scale
      combine 只接受 bf16（LL/HT/v2）
```

## C.8 通算融合 / DBO（ch07）

```
依赖链：GEMM → all-reduce → 下一层 GEMM   ← 严格串行，必须"切 batch"才能重叠
五种思路：① 切 batch 交替推进(DBO) ② 融进 kernel ③ 换通信模式(SP)
         ④ 通信不占 SM(DeepEP LL) ⑤ 阶段分离(PD)

DBO = 2 个 ubatch + 2 个 CPU 线程 + 1 对共享 stream
     CPU 侧严格交替（保提交顺序），GPU 侧两个 stream 并行（重叠来源）
     三个 assert 保证"同一时刻只有一个线程在跑"
切换原语：switch_to_comm/compute = 只换 stream
         switch_to_comm_sync/compute_sync = 换 stream + 事件等待
准入：DP>1 · EP · backend∈{deepep_ll,deepep_ht,nixl_ep} · token≥32/512
     所有 DP rank 投票一致（否则死锁）· 非 CPU · 非 elastic EP
SM 仲裁：VLLM_DBO_COMM_SMS 默认 20(CUDA)/64(ROCm)
       comm_sms = min(env, max_sms_used())；LL/NIXL 返回 0 → 不做限制
       ROCm + DeepEP HT → 强制 0（保精度）
       只有 DeepGEMM 支持 set_num_sms（compute 侧）
```

## C.9 CP / SP / KV（ch08）

```
SP：all_reduce → reduce_scatter + 局部norm + all_gather
    通信量【不变】！省的是冗余计算和显存（中间张量小 N 倍）
    ★ SP 本身不提性能，它是 AsyncTP（GEMM+RS 融合）的前置条件
    gate: hidden ≥ 8192，sm90 需 8MB/GPU、sm100 需 32MB/GPU；要求全图编译

DCP：切 decode 的 KV cache，【不增加 GPU 数】，tp % dcp == 0
     通信：query AG + LSE AG + output RS（a2a 后端可 3→2 次）
     LSE all-gather 是必须的：softmax 分母依赖所有 KV
PCP：切 prefill 序列，【增加 world size】，world_size = pp×tp×pcp

KV 传输：KVConnectorRole = SCHEDULER（决策）/ WORKER（搬数据）
        wait_for_layer_load → 层粒度同步 → 与计算流水线重叠
        PD 分离【不提升吞吐】，只解耦 TTFT 与 ITL、压尾延迟
        文档里的 KV pipe / KV lookup buffer 层已被删除（以代码为准）
```

## C.10 排障第一原则

```
集合通信三个必须一致：① 参与者集合 ② 操作顺序与类型 ③ 张量形状/dtype
→ 任何"条件分支里做通信"的代码都是危险的

hang 的三步分流：引导(端口/IP) → 建连(NCCL bootstrap/IB) → 首次前向(集合操作)
常用开关：NCCL_DEBUG=INFO · NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH · NCCL_DEBUG_FILE
         NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX
         VLLM_HOST_IP（多网卡必设）· VLLM_SKIP_P2P_CHECK（默认 1）
性能排查：先 nsys profile 确认时间分布，再怀疑通信
        如果通信占比 <15%，别从通信下手
```

## C.11 高频「反直觉」事实清单

| 事实 | 为什么反直觉 |
|---|---|
| 单机 TP>1 的引导走 **FileStore（file://）**，不是 TCP | 大家默认「分布式=开端口」 |
| 启动日志的后端顺序 **不是** dispatch 顺序 | 容易据日志下错结论 |
| **SP 不减少通信量** | 名字里有「序列并行」，容易以为省带宽 |
| **PD 分离不提升吞吐**（官方文档明说） | 容易以为分离=更快 |
| **vLLM 没有** `VLLM_DISABLE_CUSTOM_ALL_REDUCE` 环境变量 | 名字太像存在了 |
| DeepEP **HT** 的 `max_sms_used()` 返回 `None`（不是 20） | 容易以为它显式声明了 20 |
| H100 上 TP=8 的 custom AR 只覆盖 **256 KiB** 以内 | 容易以为"小消息"是 MB 级 |
| KV 传输 README 描述的两层抽象**已被删除** | 文档滞后于代码 |
| `expert_map` 被所有 prepare 签名接收，但**没人读它** | 容易以为通信层自己用了它 |
| `batch_invariant` 会把 `NCCL_MAX_NCHANNELS` 压到 **1** | 名字看不出和 NCCL 有关 |
| **TP 同时切分权重和 KV cache** | 容易以为 TP 只影响权重 |
| **Chunked prefill 和 PD 分离解决同一个问题** | 容易以为是两件不相干的事 |
| **TPOT ≠ ITL**（平均 vs 逐次抖动） | 名字太像，常被当同一个指标 |
| **FlashAttention 是精确算法** | 名字带"Flash"容易以为是近似 |
| **投机解码可能比不用更慢** | 容易以为它无条件加速 |
| **n-gram 投机解码不需要训练** | 容易以为投机解码都要训 draft 模型 |
| `gpu_memory_utilization` 默认 **0.92**（不是 0.9） | 记忆偏差 |

---

## C.12 单卡侧速查（附录 E）

```
PagedAttention = 把 OS 虚拟内存搬到 KV cache：
  block(默认 16 token) ↔ 页 ｜ block table ↔ 页表 ｜ 共享前缀 ↔ COW
  解决：内部碎片 / 外部碎片 / 无法共享
  代价：attention kernel 要按 block table gather + block 管理器开销
  收益：显存利用率 ~30% → ~90%+（直接决定并发数与吞吐）

FlashAttention：精确算法（非近似），快的原因是【减少 HBM IO】
  核心：tiling + online softmax（运行中维护 max/sum 增量修正）
  ★ 追问必答：attention 是 memory-bound，不是 compute-bound

Continuous batching：每个 decode step 重新组 batch（≠ static batching 等最慢的）
  抢占两策略：Recompute（费算力）/ Swap（费 PCIe 带宽）
Chunked prefill：把长 prefill 切块与 decode 混跑
  ★ 和 PD 分离解决同一个问题；【先试 chunked prefill】（代价近零）

量化命名 = 权重位宽 + 激活位宽：W4A16 / W8A8 / NVFP4
  ★ decode 是 memory-bound → 低比特量化对 decode 加速更明显
  ★ 量化受三方约束：模型精度 / kernel 支持 / 【通信库支持】（ch08）

投机解码：小模型猜 k 个 → 大模型一次验证
  ★ 成立的物理基础：大模型前向 memory-bound，多算几个 token 几乎不额外花时间
  ★ 接受率低会比不用更慢 ｜ n-gram 方案不需要训练
  ★ 与 DBO 冲突：投机解码产生【动态 token 数】，DBO/CUDA Graph 要求形状可预测

指标体系：
  TTFT ← prefill + 排队 ｜ TPOT ← decode ｜ ITL ← 逐次抖动（SLA 约束它的 p99）
  ★ 吞吐必须绑定 SLO；Goodput = 满足 SLO 的有效吞吐

OOM 三类分诊：启动就 OOM（权重放不下）→ 加 TP/PP 或量化
             跑起来才 OOM（KV 不够）→ 降 max_model_len 或加 TP
             跑久了 OOM（碎片/泄漏）→ 查 prefix cache 上限

★ Maximum concurrency 那行日志 = 并发上限（算法：KV 容量 ÷ max_model_len）
```

---

## C.13 最后 4 句可以背的话

1. **「TP 的瓶颈是延迟×步数，不是带宽 —— 所以 TP 只能在 NVLink 域内。」**
2. **「MoE 用 all-to-all 是因为通信量由路由的稀疏性决定，而不是由参与者数量决定。」**
3. **「通算融合的本质是打破『计算→通信→计算』的串行链；DBO 的做法是 CPU 侧严格交替、
   GPU 侧真并行。」**
4. **「PagedAttention 不是新的 attention 算法，而是把 KV cache 的管理方式
   从『连续预分配』改成『分页按需分配』—— 和操作系统虚拟内存一一对应。」**

> 前 3 句覆盖「多卡视角」，第 4 句覆盖「单卡视角」。
> **面试官通常先问第 4 句的方向，再问前 3 句** —— 这也是为什么
> 附录 E 被放在「必读」而不是「选读」。
