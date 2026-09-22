# 第 8 章　AI Infra 的通信全景：MoE all-to-all、CP/SP、KV 传输

> **本章目标**：把「通信」这个概念从「TP 的 all-reduce」扩展到 AI Infra 的**全部通信形态**。
> 面试里只懂 TP all-reduce 是不够的 —— MoE 的 all-to-all、PD 分离的 KV 传输、
> 序列并行的 reduce-scatter，都是高频考点。

---

## 8.0 先给一张「谁在通信、传什么」的总表

| 并行/机制 | 集合操作 | 传的数据 | 频率 | 跨机可行 | vLLM 实体 |
|---|---|---|---|---|---|
| **TP** | all-reduce | activation（部分和） | 每层 2 次 | ❌ 延迟不可接受 | `linear.py:1769` |
| **TP + SP** | reduce-scatter + all-gather | activation（分片） | 每层 2 次 | ❌ | `models/common/ops/sequence_parallel.py` |
| **PP** | send/recv（P2P） | activation | 每阶段 1 次 | ✅ | `parallel_state.py:1366-1380` |
| **DP attention** | all-gather（变长） | token hidden + 元数据 | 每步 | ✅ | `all2all.py:70-99` |
| **EP（MoE）** | **all-to-all** | token hidden + topk | 每层 1 次 | ✅ | `all2all.py:101-151` |
| **CP / DCP** | all-gather / reduce-scatter | KV cache 分片 | 每层 | ✅ | `get_dcp_group()` 调用点 |
| **PD 分离** | RDMA 传输（非集合） | **KV cache** | 每请求 1 次 | ✅ | `kv_transfer/` |
| **权重同步（RL）** | broadcast | 模型权重（GB 级） | 每轮训练 | ✅ | `weight_transfer/nccl_engine.py` |
| **控制面** | 字节 broadcast | 元数据/配置 | 不定 | ✅ | `shm_broadcast.py` |
| **EPLB** | 自定义 pynccl | 专家权重 | 每重平衡 | ✅ | `eplb/eplb_communicator.py` |

**读这张表的三个要点**：

1. **只有 MoE 用 all-to-all** —— 因为只有 MoE 的「数据去哪」是**数据依赖的**（路由决定）。
2. **跨机性差异巨大**：TP 不能跨机（延迟×步数），但 EP/PP/DP/KV 传输都能。
3. **每层最多 2 次通信**（TP）—— 这是所有优化的预算上限。

---

## 8.1 MoE 的 all-to-all：完整拆解

### 8.1.1 为什么 MoE 必须用 all-to-all（而不是 all-reduce / all-gather）

| 方案 | 通信量 | 为什么不行 / 行 |
|---|---|---|
| **all-reduce** | 所有 rank 的专家输出都要汇总 → 与 EP size 相关 | 专家数有几百个，但 EP size 只有 8/16/32，**all-reduce 会把「稀疏路由」的收益抹掉** |
| **all-gather** | 每 rank 拿到**所有** token → 数据量 ×EP | 每个 rank 只需要**发给自己专家的** token，all-gather 让每个人都看全部 |
| **all-to-all** | 只发给「持有目标专家」的 rank | ✅ 通信量与**路由分布**相关，不随 EP size 线性涨 |

**一句话**：
> **all-reduce / all-gather 的通信量由「参与者的数量」决定；
> all-to-all 的通信量由「数据的去向」决定。**
> MoE 的稀疏性让后者远小于前者。

### 8.1.2 一个 MoE 层的完整数据流

```
输入 hidden_states [T_local, H]
        │
        ▼
① Router (gate)
   router_logits = hidden @ W_gate          → [T_local, num_experts]
   topk_ids, topk_weights = topk(logits, k) → [T_local, topk]
        │
        ▼
② dispatch（all-to-all）
   把每个 token 的 hidden 发给「持有它选中的某个专家」的 rank
   变长！每个 rank 发出去的量不同
        │
        ▼
③ 本地专家计算
   expert_map: 全局专家 id → 本地专家 id
   只算落在本 rank 上的专家（grouped GEMM）
        │
        ▼
④ combine（all-to-all，反向）
   把专家输出按 token 送回原 rank
   乘 topk_weights 后求和
        │
        ▼
输出 hidden_states [T_local, H]
```

**两次 all-to-all 的分工**：
- **dispatch**：token → 专家（按专家所在 rank 分组）
- **combine**：专家输出 → token（按原始 rank 分组，并加权求和）

### 8.1.3 token permute / unpermute：MoE 通信的核心数据结构

**问题**：token 被送到别的 rank，计算完要能**还原到原来的位置**。

**机制**：
- dispatch 时记录**每个 token 去了哪里**（或每个收到的 token 原本属于谁）；
- combine 时按这个映射把结果放回原位置，并做加权求和。

**vLLM 里的体现**：`topk_ids` 在多轮变换中被反复改写：

| 阶段 | 坐标含义 | 说明 |
|---|---|---|
| router 输出 | **全局专家 id** | 0..num_experts-1 |
| dispatch 前 | 映射到「按 rank 分组」的维度 | 见 `deepep_ll.py` 里的 `global_to_physical` 映射 |
| dispatch 后 | **本地专家 id** | 0..num_experts_per_rank-1 |
| 计算前 | 再加回 `rank_expert_offset` | 让 kernel 看到一致的全局偏移 |
| 无效位置 | **-1 sentinel** | 表示「这个位置没有真实 token」（padding） |

**面试点**：**「-1 sentinel」是 MoE 实现的通用约定**。
因为 dispatch 后的形状是定长 buffer（`[num_local_experts, max_tokens, H]`），
但实际 token 数不足，**空位必须用哨兵值标记**，让 kernel 跳过。

**DeepEP HT 的哨兵处理很讲究**（`prepare_finalize/deepep_ht.py`）：

```python
# 代码注释原文：
# "The existing MOE kernels assume that all entries of topk_ids are valid.
#  To that effect, set the -1s in expert_topk_ids to some expert outside this rank
#  so the expert_map can remap it to -1 when safe.
#  With Expert Parallel, the experts are divided amongst the rank sequentially.
#  For rank 0, set it to num_experts - 1 and for all other ranks set it to 0 as we
#  know that expert_map will have a -1 in those regions for those ranks."
torch.where(expert_topk_ids == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            expert_topk_ids + self.rank_expert_offset)
```

**这段逻辑的精妙之处**：
- 下游 kernel 假设「没有 -1」（它不理解哨兵）；
- 所以要把 -1 **换成一个「肯定不属于本 rank」的合法专家 id**；
- 因为专家是**按 rank 顺序连续分配**的，所以「rank 0 用最后一个专家 id、
  其它 rank 用 0 号专家」这两个 id 一定落在别人的地盘 → `expert_map` 会把它们映射回 -1。
- 同时还要**加上 `rank_expert_offset`**，把 DeepEP 输出的**本地**专家 id
  搬回**全局**专家空间，与 vLLM 其它接口对齐。

**面试价值**：这是「**跨模块接口约定不匹配时，用数据变换做适配**」的典型案例。
能讲清「为什么是 num_experts-1 和 0 这两个数」的人，一定真读过代码。

### 8.1.3b 专家 id 的三套坐标系与 routing tables

这是 MoE 里最容易绕晕的部分。**必须分清三套坐标系**：

| 坐标系 | 范围 | 谁用 |
|---|---|---|
| **全局/逻辑专家 id** | `0 .. num_experts-1` | router 的输出；模型语义 |
| **物理专家 id** | `0 .. ep_size*local_experts-1`，但按「轮转放置」排布 | DeepEP-LL / NIXL 的 A2A 寻址 |
| **本地专家 id** | `0 .. num_local_experts-1` | 本地专家 kernel |

`expert_map` 的定义（`expert_map_manager.py`）：

> *"`expert_map[global_id] = local_id` if expert is on this rank;
> `expert_map[global_id] = -1` if expert is not on this rank"*

**⚠️ 注意一个命名陷阱**（读代码时容易找错）：
vLLM 里**没有** `physical_to_logical` 这个标识符。
实际名字是 `physical_to_global` + `local_expert_global_ids`。

**三张 routing table** 由 `ExpertMapManager.routing_tables` 提供
（`(global_to_physical, physical_to_global, local_expert_global_ids)`），
**但只有 `global_to_physical` 被真正使用**（在 LL/NIXL 里做 A2A 寻址查表），
另外两张存下来但当前没有再引用 —— **这也是一个可以指出的「代码里留有未用接口」的观察**。

**轮转放置（round-robin）的算法**（`expert_map_manager.py`）：

```python
owner = global_indices % ep_size            # 哪个 rank 拥有它
local_index = global_indices // ep_size     # 在该 rank 内的第几号
physical_offset = owner * base + min(owner, remainder)   # 处理不整除
global_to_physical = physical_offset + local_index
```

**为什么需要轮转**：默认的「线性分配」（rank 0 拿前 N 个专家）会让
**不同 rank 的专家在语义上聚集**（相邻专家往往相关），导致负载不均。
轮转打散了这个相关性。

**但轮转有准入限制**（代码原文）：

> *"Round-robin placement requires DeepEP-ll or NIXL backend. Falling back to linear."*

→ **只有 LL 和 NIXL 实现了物理 id 查表**；HT 和 v2 用
`rank_expert_offset = rank * num_local_experts`（即线性分配）。
**这是一个「优化与后端能力绑定」的例子**：
你想要更好的负载均衡，就必须用支持它的通信后端。

**面试题**：「EPLB / 专家放置策略会不会改变模型输出？」
→ **不会**。路由的 top-k 结果和权重完全不变，只是「同一个逻辑专家由哪个物理副本执行」
变了。**这是「优化不得改变语义」的教科书级实践。**

### 8.1.4 变长消息：sizes 元数据怎么传

**这是 all-to-all 和 all-reduce 最大的工程差异**。

all-reduce 的每个 rank 发同样大小的数据 → NCCL 直接支持。
all-to-all 每个 rank 发给不同对端的数据量不同 → **需要额外交换「尺寸元数据」**。

**vLLM 的两种做法**：

**做法 1（AgRs 路径）：走 DP metadata，不在通信内交换**
（第 4 章讲过）各 rank 的 token 数通过 **CPU all-reduce** 汇总到
`DPMetadata.num_tokens_across_dp_cpu`，forward 时读出（`all2all.py:60-68`）。
→ **零额外通信开销**（因为 DP 同步本来就要做）。

**做法 2（DeepEP 路径）：由通信库自己算**
DeepEP 的 `get_dispatch_layout` 从 `topk_idx` 直接算出
`num_tokens_per_rank` / `num_tokens_per_rdma_rank` / `is_token_in_rank`，
`dispatch` 时一并传给 kernel。

**面试题**：「变长 all-to-all 的尺寸信息怎么同步？会不会成为瓶颈？」
→ 两种方案：① 搭便车在已有的 DP 同步里（vLLM 的 AgRs）；
② 由 kernel 从路由信息算出（DeepEP）。**核心是避免「为了传尺寸而单独做一次通信」。**

### 8.1.5 负载不均（straggler）：EP 的头号敌人

**问题**：如果某个专家特别热门，持有它的 rank 要算的 token 是别人的好几倍。
**all-to-all 的完成时间由最慢的那个 rank 决定** —— 其它 rank 都在等它。

**为什么比 TP 更严重**：
- TP 的 all-reduce 是对称的（每卡工作量一样）；
- EP 的负载**由输入数据决定**，天然倾斜（真实数据的路由分布是长尾的）。

**vLLM 的对策：EPLB（Expert Parallelism Load Balancer）**
（`vllm/distributed/eplb/`）

核心思想：**把热门专家复制多份，分散到不同 rank**，让负载均衡。

需要维护三张映射表：

| 表 | 含义 |
|---|---|
| `physical_to_logical_map` | 每个物理副本对应哪个逻辑专家 |
| `logical_to_physical_map` | 每个逻辑专家有哪些物理副本 |
| `logical_replica_count` | 每个逻辑专家有几个副本 |

**注意 vLLM 的架构决策**：
- 路由（router）产出的**逻辑专家 id 不变**；
- EPLB 只改变「逻辑专家 → 物理副本」的映射；
- **所以模型的语义完全不变**，只是执行位置变了。

**面试题**：「EPLB 会不会改变模型输出？」
→ **不会**。路由的 top-k 和权重完全不变，只是同一个逻辑专家由不同的物理副本执行。
**这是「优化不能改变语义」这条原则的教科书级实践。**

### 8.1.6 六种 all-to-all 后端的取舍表

（`cuda_communicator.py:163-225` 的 if/elif 链）

| 后端 | 库 | 传输 | 最适合 | 关键约束 |
|---|---|---|---|---|
| `allgather_reducescatter`（默认） | NCCL | NVLink + IB | 通用、任何配置 | 通信量最大（AG 会把所有 token 拉过来） |
| `deepep_high_throughput` | DeepEP HT | NVLink + RDMA | **prefill**（大 batch） | 占 **20 SM**；不支持 CUDA Graph |
| `deepep_low_latency` | DeepEP LL | **RDMA** | **decode**（小 batch） | 占 **0 SM**；支持 CUDA Graph；hidden size 限于固定集合 |
| `deepep_v2` | DeepEP v2 | **NCCL GIN** | 新一代统一 API | 需 **NCCL ≥ 2.30.4 + IBGDA 网卡**；SM 数解析计算 |
| `nixl_ep` | NIXL | RDMA | 弹性 EP、跨机 | `max_num_ep_ranks` 默认 32 |
| `flashinfer_nvlink_*` | FlashInfer/TRT-LLM | **MNNVL** | 多节点 NVLink（GB200 NVL72） | 需要 MNNVL 硬件域 |
| `mori_*` | MoRI | ROCm 专用 | AMD 平台 | 仅 gfx942/gfx950 |

### 8.1.6b wire format 逐后端对比（本节是「读过代码」的分水岭）

`FusedMoEPrepareAndFinalize` 的接口契约（`fused_moe/modular_kernel.py:187`）：

```python
class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps described above.

    There are two variants of this class:
    * FusedMoEPrepareAndFinalizeModular - this operates on topk ids and weights
    * FusedMoEPrepareAndFinalizeMonolithic - the operates on router_logits
    """
```

**Modular vs Monolithic 的区别（面试常问）**：

| | Modular | Monolithic |
|---|---|---|
| 输入 | **`topk_ids` + `topk_weights`**（router 已在 vLLM 里算完） | **`router_logits`**（router 在专家 kernel 内部跑） |
| 传输内容 | token hidden + 每 token 的 topk id/weight | token hidden + router logits |
| 代表实现 | DeepEP HT/LL/v2、NIXL | `naive_dp_ep`、`no_dp_ep` |

**为什么会有 Monolithic**：某些融合 kernel 把 router（gate）也吃进 kernel 里，
避免一次额外的 kernel launch 和显存往返。**代价是通信要传 router_logits（比 topk id 大）。**

**逐后端 wire format 表**（★ = 实现细节，面试提出来很加分）：

| 后端 | dispatch 传什么 | combine 传什么 | sizes 元数据 | 异步机制 | CUDA Graph |
|---|---|---|---|---|---|
| **DeepEP HT** | `(tokens[, token_scales])` + `topk_idx`(int64 全局) + `topk_weights`；布局元数据 `num_tokens_per_rank`/`num_tokens_per_rdma_rank`/`is_token_in_rank` 由 **`get_dispatch_layout` 在设备上算出** | `fused_expert_output` **只能 bf16** + handle + `topk_weights=None` | `expert_num_tokens_per_expert_list`（**Python list → 有 GPU-CPU 拷贝**） | `prepare_async` 返回 receiver；`async_finish = ... and not dbo_enabled()` | ❌ **不兼容**（`modular_kernel.py` 注释：*"CUDAGraph incompatible all2all kernels like the DeepEP high-throughput kernels"*） |
| **DeepEP LL** | 一个融合调用 `low_latency_dispatch`，**同时量化 + 搬运**；输出是 BatchedExperts 布局 `(E, max_tokens, H)` | `low_latency_combine(..., out=output)`，**权重加权和规约在 combine kernel 里做**★ | `expert_num_tokens`（**GPU 张量**，无 CPU 拷贝） | `(hook, receiver)` 对；`do_recv_hook = dbo_enabled() or do_async` | ✅ 兼容（*"always batched and can never run into the tensor.numel() == 0 case"*） |
| **DeepEP v2** | `dispatch(..., do_expand, do_cpu_sync)`；输出 `recv_x`, `recv_topk_idx`, `recv_topk_weights` | bf16（否则 `ValueError`） | decode 走 `handle.psum_num_recv_tokens_per_scaleup_rank`（GPU 前缀和）；prefill 走 `handle.num_recv_tokens_per_expert_list`（CPU list） | **`supports_async()=True` 但 `finalize_async` 其实是同步的**★ | ✅ decode 完全可捕获；prefill 不可（CPU polling） |
| **NIXL EP** | 同 LL 形状（`dispatch(a1, physical_topk_ids, max_tokens_per_rank, ...)`） | 权重在 combine kernel 里做 | `expert_num_tokens`（GPU 张量） | `(hook, receiver)` | 未说明 |
| **Naive DP/EP（AgRs）** | `get_ep_group().dispatch(a1q, topk_weights, topk_ids, extra_tensors=[scales?, lora_mapping?])` | 本地加权规约后 `combine(...)` | 无（`expert_tokens_meta = None`）★ | ❌ **不支持异步**（没实现 `prepare_async`） | 未说明 |
| **No DP/EP（纯本地）** | **不通信**（只量化） | 直接写 output | 无 | ❌ | — |

**★★ 三个「必须读过代码才知道」的细节**：

1. **HT 会做 GPU→CPU 拷贝拿 per-expert 计数**
   （`ExpertTokensMetadata.make_from_list`），代码里还留着 TODO：
   *"Makes a GPU-CPU copy. TODO (varun): Maybe it is better to re-compute the
   expert_num_tokens on GPU."*
   → **这是一个已知的性能损失点**：为了给 batched expert kernel 提供 CPU 端的计数，
   必须同步一次。**LL 和 v2 都避免了这一点**（用 GPU 张量）。

2. **LL 的「权重加权和规约在 combine kernel 里做」是一个必须遵守的契约**：
   ```python
   assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), \
       "Weight application and reduction happens in the combine kernel."
   ```
   所以 vLLM 不能自己去 apply 权重 —— 否则会算两次。

3. **`recv_hook` 的存在理由**（`modular_kernel.py` 原文）：
   > *"if a hook is returned this is more lightweight check that the recv is complete
   > without doing extra work (used by DBO, will be refactored in the very near future)"*
   → hook 是「**轻量地问一句到了吗**」，而 receiver 是「**真的去等并拿结果**」。
   **DBO 需要前者** —— 因为它只想在合适的时机检查，不想阻塞。

### 8.1.6c 为什么异步 prepare/finalize 是 DBO 的硬前提

`modular_kernel.py` 里有两条 assert：

```python
assert not dbo_enabled()      # 当 not supports_async() 时
```

**原因**：DBO 的重叠依赖「**通信可以先发起、稍后再取结果**」这个能力。
如果 P/F 只提供同步接口（发起就必须等完），那「ping-pong」就无从谈起 ——
线程 A 发起通信后只能干等，无法让出 CPU 给线程 B。

**所以：`supports_async()` 是 DBO 的必要条件。**
这也解释了为什么 DBO 只支持 DeepEP / NIXL —— 只有它们实现了异步接口
（`naive_dp_ep` 和 `no_dp_ep` 都没有）。

**面试题**：「为什么 DBO 只能配 DeepEP / NIXL？」
→ 三层原因：① **异步接口**（`prepare_async`/`finalize_async`）只有它们有；
② **SM 可控**（`set_num_sms`）只有 DeepEP HT 有；
③ 代码层面直接 **assert 限制**了后端集合（`config/vllm.py:1744-1757`）。

**再加一个细节**：`modular_kernel.py` 的编排逻辑是 ——
如果 P/F 支持异步，就把 hook 注册到 ubatch context，交给
`dbo_maybe_run_recv_hook` 而不是直接传给 receiver（代码注释：

> *"If DBO is being used, register the hook with the ubatch context and call it in
> dbo_maybe_run_recv_hook instead of passing it to the receiver."*）

**这就是第 7 章讲的「控制反转」**：通信库说「你必须调用 hook」，
调度器说「调用时机我来定」→ 用回调注册实现。

### 8.1.7 量化 × all-to-all：三条必须知道的规则

**规则 1：DeepEP 只支持 fp8 block scale 的直接传输，其它量化要「先传 bf16，传完再量化」**

代码原文（`deepep_ht.py`，`:288-292`）：

> *"DeepEP only supports fp8 block scales so quantize before the dispatch for these models.
> For all other quantization, dispatch after.
> For expert kernels that require unquantized inputs, defer quantization to
> FusedMoEExpertsPermuteUnpermute."*

**为什么**：量化后的数据必须带 scale 才能反量化，而**通信库只认识 block scale 这一种格式**。
其它格式（per-token、nvfp4）如果要走 A2A，只能传原始 bf16，落地后再量化。

**代价**：传 bf16 是 fp8 的**两倍带宽**。**这是一个真实的「量化省下来的带宽被通信吃掉」的例子。**

**规则 2：scale 的 swizzle 必须推迟到 A2A 之后**

代码原文（`naive_dp_ep.py`）：

> *"NOTE: swizzling pads the scales to multiple of 128 which makes the scales tensor
> different shape than the hidden states, breaking the A2A kernel.
> So, we delay the swizzling until after the A2A."*

**为什么**：A2A kernel 假设「scale 和 hidden 的形状对应」。但某些量化格式要求把 scale
**重排（swizzle）** 成 CUTLASS 期望的 `F8_128x4` 布局，这会改变形状 → 破坏 A2A 的假设。
**所以必须「按 row-major 传过去，落地后再 swizzle」。**

**规则 3：per-token scale 在 LL / NIXL 上直接不允许**

```python
assert not has_per_token_scales, \
    "low_latency kernels doesn't support dispatching per-token scales"
```

**面试题**：「做 MoE 量化时，通信会带来哪些约束？」
→ 完整答案就是这三条：① 只有 fp8 block scale 能直接传，其它格式要传 bf16；
② scale 的 swizzle 必须推迟到 A2A 之后（因为会改形状）；
③ low-latency kernel 不支持 per-token scale。
**结论：量化方案的选择不只受 kernel 支持影响，还受通信库支持影响。**

### 8.1.7b MoE 后端的尺寸与对齐约束汇总

| 后端 | hidden size 约束 | 其它 |
|---|---|---|
| DeepEP HT | 按 **512 字节**原子块向上取整（`xfer_atom_size = 512  # 32 * 16 (size(int4))`；例：2880 bf16 → 3072） | `num_tokens_per_rank() = None`（非 batched）；可用 rank 配置 `[2,4,8,16,24,32,64,128,144,160]`★ |
| DeepEP LL | **必须是固定集合之一**：`[2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]`；fp8 时还要 `% 128 == 0` | `max_num_tokens_per_rank = self.max_tokens_per_rank`（batched） |
| DeepEP v2 | 512 字节原子对齐 | decode 时 `num_max_tokens_per_rank` **向上取到 2 的幂**★ |
| NIXL EP | 同 LL 的 8 个 hidden size；fp8 时 `% 128 == 0` | `max_num_tokens_per_rank = max_tokens_per_rank` |
| naive / no-dp | 无 | `max_num_tokens_per_rank = None` |

**★ v2 的「2 的幂向上取整」有一个很精彩的工程理由**（代码注释原文）：

> *"DeepEP JIT-compiles a separate dispatch kernel per distinct `num_max_tokens_per_rank`,
> so feeding it the raw per-step size would make it recompile for every batch size
> (**a cicc storm that starves the GPU at high concurrency**).
> Round up to a power of 2 instead: this bounds the set to ~log2(max_num_batched_tokens)
> values (compiled once, then cached) while staying small for decode (e.g. 1 token -> 1)
> and capped at the buffer's init capacity for prefill."*

**这是本章最值得记住的一个工程案例**：
一个看似无聊的「向上取整」背后，是**JIT 编译缓存键的设计**。
如果不取整，每来一个不同的 batch size 就触发一次 JIT 编译，
编译器（cicc）抢占 GPU → **高并发下 GPU 被编译器饿死**。

**面试话术**：
> 「MoE 通信里很多看似随意的常数（512 字节对齐、2 的幂取整、8 个固定 hidden size），
> 背后都是**下游 kernel 的 JIT 特化策略**决定的。
> 系统工程师的价值就在于理解这些约束并把它们正确地传导到上层配置。」

**「可用 rank 配置」这一条也很实用**（DeepEP HT）：
`[2, 4, 8, 16, 24, 32, 64, 128, 144, 160]` ——
**如果你的 EP size 不在这里面，HT 的 dispatch config 查询会返回 None**，退到 DeepEP 默认配置。
**这类「魔法数字列表」是部署调优时的隐藏坑。**

```
你的场景是？
├─ 没有特殊库 / 想先跑起来 → allgather_reducescatter（默认，永远能用）
├─ NVIDIA + 有 DeepEP
│   ├─ prefill 为主（大 batch） → deepep_high_throughput
│   ├─ decode 为主（小 batch） → deepep_low_latency
│   └─ 有 IBGDA + NCCL≥2.30.4 → deepep_v2（统一 API，最省心）
├─ 需要弹性扩缩容 → nixl_ep
├─ GB200 NVL72 / 多节点 NVLink → flashinfer_nvlink_one_sided
└─ AMD MI300 系列 → mori_*
```

**补充：各量化格式的 scale 具体约束**

| 量化格式 | scale 约束 | 原因 |
|---|---|---|
| **fp8 block-wise** | 要求 `hidden % 128 == 0` | kernel 按 128 分块量化（`DEEPEP_QUANT_BLOCK_SIZE = 128`） |
| **mxfp8** | scale 要 **4 个打包成一个 int32**；`scale.size(1) % 4 == 0` | 通信库把 scale 当 4 字节不透明包传（`sf_pack_t` 是 float/UE8M0x4 union） |
| **nvfp4** | scale 元素数 = `hidden // 16` | 每 16 个元素一个 scale |
| **per-token scale** | DeepEP LL / NIXL **不支持** | 低延迟 kernel 不支持逐 token scale（代码有明确 assert） |

**面试话术**：
> 「做 MoE 量化的时候，**通信约束会反过来限制你的量化方案**。
> 比如想做 per-token scale 的 fp8，DeepEP 的 low-latency kernel 不接受；
> 想做 mxfp8，必须按通信库期望的方式（4 个 scale 打包进一个 int32）打包；
> swizzle 还必须推迟到 A2A 之后。
> **这不是模型问题，是系统问题** —— 这类约束在论文里往往看不到。」

---

## 8.2 Context Parallel 与 Sequence Parallel

### 8.2.1 Sequence Parallel（SP）：省的不是通信，是计算和显存

**回顾第 7 章**：SP 把 `all_reduce` 变成 `reduce_scatter + [局部计算] + all_gather`。

**为什么成立**：`all_reduce` 之后**每张卡拿到的结果完全相同**，
所以紧接着的逐 token 计算（RMSNorm）是**冗余的**（N 张卡算同样的事）。
SP 让每张卡只算自己那份。

**收益的精确表述（面试重点）**：

| 维度 | 变化 |
|---|---|
| **通信量** | **不变**（RS 的 `(N-1)/N·S` + AG 的 `(N-1)/N·S` = AR 的 `2(N-1)/N·S`） |
| 中间张量大小 | **减少 N 倍**（`S` → `S/N`）→ **省显存** |
| norm 计算量 | **减少 N 倍** → **省算力** |

**所以 SP 的正确说法是「用同样的通信量，换更少的冗余计算和显存」。**
**不要说「SP 减少了通信」** —— 这是面试常见的错误。

### 8.2.2 vLLM 的 SP 实现：`sequence_parallel.py`（68 行，值得全读）

```python
def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)      # :15-20

def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)        # 先试自定义实现
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)              # 回退到通用路径

def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size                           # 对齐 padding
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)
```

**三个可讲的点**：

1. **「先试自定义，失败回退」的模式**（`:23-27`）：和 all-reduce 的 8 路选择同一个哲学。
   注意 `custom_all_gather` / `custom_reduce_scatter` 正是第 4 章那个
   `CustomAllreduce` 类的另外两个方法（它还实现了 AG/RS，不只是 AR！）。
   **所以 vLLM 的 custom collective 是一套，不只是 all-reduce。**

2. **必须做对齐 padding**（`:33-35`）：token 数不一定能被 `tp_size` 整除，
   而 reduce-scatter 要求能均分。`(-x.shape[0]) % tp_size` 是「补齐到整除」的惯用写法。

3. **padding 需要 mask 配合**（`:53-68` 的 `sp_padding_mask`）：
   补进来的 token 是假的，必须在后续计算中被忽略，否则会污染结果（比如影响 norm 的统计量）。
   **这是「为了通信对齐而引入 padding，padding 又必须被下游正确处理」的典型连锁问题。**

### 8.2.2b SP 是「编译期 pattern 匹配」实现的（这是重点）

**关键认知**：vLLM 不是手写 SP，而是用 **Inductor 的 pattern matcher** 自动改图。

`vllm/compilation/passes/fusion/sequence_parallelism.py` 的类 docstring 原文：

> *"It identifies patterns where an AllReduce operation is followed by an RMSNorm
> (or RMSNorm and then Quantization) operation. These patterns are replaced with a
> ReduceScatter operation, followed by a local RMSNorm/Quantization, and then an AllGather
> operation."*
>
> *"The general transformation is: `Input -> AllReduce -> RMSNorm -> Output` becomes
> `Input -> ReduceScatter -> RMSNorm -> AllGather -> Output`"*
>
> *"**While this pass itself does not directly yield performance improvements, it lays the
> groundwork for subsequent fusion passes, such as GEMM + ReduceScatter and
> AllGather + GEMM fusions.**"*

**最后这句话极其重要，是本题的满分答案**：

> **SP 本身不提性能！它只是「把图改成可融合的形状」，真正的收益来自后续的
> AsyncTP pass（GEMM + ReduceScatter / AllGather + GEMM 融合）。**

**为什么必须这样分两步**：
`all_reduce` 是一个**不可拆的黑盒**（第 2 章讲过：它的语义是「先规约再广播」，中间状态不可见）。
而 `reduce_scatter` + `all_gather` 是**两个可见的操作**，中间夹着局部计算 ——
于是编译器就有机会把 `GEMM` 和 `reduce_scatter` 融成一个 kernel。

**顺带解释了一个现象**：vLLM 里 `enable_sp` 是 `fuse_gemm_comms` 的**前置条件**
（`docs/design/fusions.md`：AsyncTP *"Requires `enable_sp=True` (enabled automatically).
This pass is a no-op if Sequence Parallelism has not been applied."*）。
**没有 SP 就不可能做 GEMM-通信融合** —— 因为图上没有可融合的接缝。

**注册的 pattern 类**（同一文件）：

| Pattern | 匹配的结构 |
|---|---|
| `FirstAllReduceRMSNormPattern` | `rms_norm(all_reduce(x))` |
| `MiddleAllReduceRMSNormPattern` | `fused_add_rms_norm(all_reduce(mm_1), residual, w)` ← **带 residual 和前置 matmul** |
| `FirstAllReduceRMSNormStaticFP8Pattern` | 上述 + FP8 量化 |
| `MiddleAllReduceRMSNormStaticFP8Pattern` | 上述 + FP8 量化 |
| NVFP4 变体 | 依赖 `SCALED_FP4_QUANT_OUT_OVERLOAD` 是否可用 |

**device gate（很硬的限制）**：

```python
SP_MIN_HIDDEN_SIZE = {90: 8192, 100: 8192}      # sm90 / sm100 都要求 hidden >= 8192
SP_MIN_PER_GPU_SIZE_MB = {90: 8, 100: 32}
```

注释解释了为什么 Blackwell 的门槛更高（32 MB vs 8 MB）：*"Use a more conservative
threshold on Blackwell so TP8 starts later."*
**其它平台直接返回 `None` → SP 被禁用。**

**SP 需要 full-graph 编译**，docstring 原文：

> *"This pass is only supported when compiling the whole graph (fullgraph mode, i.e. using
> Inductor graph partition or empty splitting_ops). **Piecewise compilation is not supported
> because the residual tensor gets split across TP ranks, causing size mismatches at subgraph
> boundaries.**"*

**这是一个「编译器能力约束反向限制优化」的例子**：
SP 让 residual 张量在 TP rank 之间被切开 → 子图边界上的形状不一致 →
piecewise 编译无法处理 → **必须全图编译**。

**还有一个非常好玩的细节**：pattern 匹配是**从图的末尾往前**替换的，
所以匹配到中间层时，**前一层的 residual 还是全尺寸的**，
代码里那段 slice 在替换完成后会变成语义错误 —— 但注释解释了为什么无害：

> *"Once the preceding layer IS replaced, its residual output shrinks to `[local_len, H]`,
> and this slice becomes semantically incorrect (e.g. for rank > 0, the indices would be
> out of bounds). However, since the symbolic output shape equals the input shape,
> **NoOpEliminationPass ... removes these slices before the graph is ever executed or compiled**."*

**面试价值**：这说明**编译期 pass 的顺序和临时不一致状态**是真实存在的工程问题，
而且解法是「靠后续 pass 清理」。能讲这个层次的人极少。

### 8.2.3 Context Parallel（CP）：DCP 和 PCP 是两件不同的事

**先纠正一个常见误解**：vLLM 的 CP 不是一个东西，而是**两个独立的机制**，
`ParallelConfig` 的 docstring 说得很清楚：

| | **PCP**（prefill context parallel） | **DCP**（decode context parallel） |
|---|---|---|
| 字段 | `prefill_context_parallel_size` | `decode_context_parallel_size` |
| 做什么 | **切分 prefill 的序列计算** | **切分 decode 的 KV cache** |
| 是否增加 GPU 数 | ✅ **会增加 world size** | ❌ **不增加**（复用现有 TP rank） |
| KV cache 分片数 | 不增加 | 增加（这正是目的） |
| docstring 原文 | *"PCP expands the process world size but does not increase the KV-cache shard count."* | *"DCP does not expand the process world size. Without PCP, DCP reuses TP ranks."* |

**所以最简单的理解**：
- **PCP = 加机器来切 prefill**（`world_size = pp × tp × pcp`）；
- **DCP = 不加机器，把 KV cache 摊到已有的 TP rank 上**。

官方文档（`docs/serving/context_parallel_deployment.md`）对 DCP 的定位说得很直白：

> *"This is as simple as adding `-dcp <size>` ... Note that `size` does not increase the
> number of GPUs we need to launch, but just **reduces the KV cache duplication**. ...
> With larger dcp size, the KV cache duplication is reduced, **but the communication overhead
> increases**."*

→ **DCP 的本质是「用通信换显存」**。这句话是 DCP 最精炼的定义。

**DCP 的通信模式（面试可以画出来）**：

```
decode 时，每张卡只持有部分 KV，但 attention 需要算「query 对所有 KV」：
① query all-gather      —— 把各卡的 query 收集起来（因为每卡只算自己那部分 KV）
② 本地算 attention + LSE（log-sum-exp，softmax 的中间统计量）
③ LSE all-gather        —— 交换各卡的 LSE，才能正确合并
④ 输出按 head 维 reduce-scatter（或 all-reduce）
```

**关键洞察：为什么需要 all-gather LSE**？
因为 **softmax 的归一化因子（分母）是所有 KV 的函数**。
每张卡只看到部分 KV，所以只能算出**部分分母** →
必须把所有卡的 LSE 收集起来才能得到正确的全局 softmax。
**这是「分片 attention」的核心数学困难**，也是 flash-decoding / ring-attention 的共同课题。

**vLLM 的优化：`--dcp-comm-backend`**

| 值 | NCCL 调用数/层 | 做法 |
|---|---|---|
| `ag_rs`（默认） | **3** | query AG + LSE AG + output RS |
| `a2a` | **2** | 用 all-to-all 一次交换「部分输出 + LSE」，再用 Triton kernel 合并 |

docstring 原文：*"All-to-All exchange of partial outputs + LSE, then combine with Triton kernel.
Reduces NCCL calls from 3 to 2 per layer for MLA models."*

**注意「减少 NCCL 调用数」本身就是优化目标** —— 因为 decode 阶段每次调用都有 μs 级固定开销
（第 1 章讲过）。**把 3 次调用变 2 次，比提高带宽更有价值。**

**PCP 的通信模式**：

| 操作 | 为什么 |
|---|---|
| prefill KV 写入的 all-gather（含 slot mapping） | prefill 时每卡算部分 token 的 KV，但都要写进同一份 KV cache |
| indexer K 的 all-gather | 稀疏 attention 的索引需要全局视野 |
| decode attention 输出按 head all-gather | 合并各卡的 head 分片 |
| hidden states all-gather（采样前还原） | 采样需要完整序列 |
| **MoE dispatch 的 all-gather / combine 的 reduce-scatter** | ⚠️ **PCP > 1 且不用 all2all kernel 时**，MoE 也要通信！ |

**最后一条容易被忽略**：PCP 会把序列切开，所以进入 MoE 前要先把 token 收集齐。
这在 `fused_moe/runner/moe_runner.py` 里，条件正是
`pcp_size > 1 and not use_all2all_kernels`。
**→ 用了 PCP 又不用 all-to-all kernel，MoE 会额外付出 all-gather/reduce-scatter 的代价。**
这是一个真实的「并行维度之间互相影响」的例子。

**DCP 的约束（部署时容易踩）**：

| 约束 | 说明 |
|---|---|
| `tp % dcp == 0`（PCP=1 时） | DCP 复用 TP rank，必须整除 |
| PCP>1 时 `dcp ∈ {1, pcp, tp*pcp}` | "DCP must be disabled, span the PCP axis, or span the full TP x PCP axis" |
| GQA/MQA：`dcp ≤ tp // num_kv_heads` | 且 `num_q_per_kv % dcp == 0` |
| attention 后端必须支持返回 LSE | 否则报错并提示换后端 |
| `block_size >= cp_kv_cache_interleave_size` 且能整除 | KV cache 交错存储的对齐要求 |
| **DCP 与 `--enable-return-routed-experts` 不兼容** | |

**部署建议（官方文档）**：
> *"try to increase `-tp` size until you get satisfactory performance, and then add `-dcp`"*

**案例**：DeepSeek-R1 用 `-tp 8 -dcp 8`；Kimi-K2 用 `-tp 16 -dcp 16`（或 `-dcp 8`，
因为 *"the communication overhead is smaller since the DCP communication only happens
inside one node"*）。

**→ 这条注释点出了 DCP 的关键权衡：DCP 组不要跨机。**

### 8.2.4 `DCPGroupColumnParallelLinear`：用「冗余计算」换掉一次通信

（`vllm/model_executor/layers/linear.py:624`）

**问题**：DCP 下每张卡只持有一部分 KV，所以 MLA decode 必须看到**整个 DCP 组的 head**。
如果权重按普通 TP 切分，每张卡只有一部分 head → 必须 all-gather query。

**vLLM 的解法**：让这个线性层的权重**按 DCP 组（而不是每个 rank）切分**，
于是组内每个 rank 都持有**整个组的全部 head** → **decode 时可以跳过 query all-gather**。

docstring 原文：

> *"With Decode Context Parallelism (DCP) the KV cache is sharded across a DCP group,
> so MLA decode must attend the group's full head set. This layer shards its output across
> DCP **groups** (effective tp size `tp_size // dcp_world_size`) rather than across every rank,
> so each rank in a group holds the whole group's heads, **letting decode skip the query
> all-gather**."*

**实现**（很简洁）：

```python
group_size = max(dcp_world_size, 1)
qrep_active = group_size > 1
rank_in_group = rank % group_size
super().__init__(..., tp_rank=rank // group_size,
                      tp_size=world_size // group_size)
```

**代价**（`config/parallel.py` 的注释说得很清楚）：

> *"Replicating the (small) query projection at load time lets each rank materialize the full
> group-local head set and skip that collective, **at the cost of computing the projection
> redundantly on every rank in the group**."*

**面试价值 —— 这是一个极好的「工程权衡」教学案例**：

| | 不复制（要 all-gather） | 复制（q-replicate） |
|---|---|---|
| 通信 | 每层一次 query all-gather | **零** |
| 计算 | 每个 head 算一次 | **组内每 rank 都算一遍**（冗余 `dcp_size` 倍） |
| 显存 | query 权重分片 | query 权重复制 |
| 为什么划算 | — | 因为 **query projection 相对小**，而 decode 阶段**通信延迟是主要成本** |

**通用的判断法则**：
> 当「被复制的计算」很小、而「省掉的通信」在关键路径上延迟敏感时，
> **用冗余计算换通信是赚的。**
>
> 这正是 TP 里「column-parallel 不通信」的同一个思路，只是方向相反 ——
> 这里是主动**增加冗余**来消除通信。

**vLLM 用同一个类名把这个设计显式化了**：`DCPGroupColumnParallelLinear` 的存在本身
就是在说「这里有一个非标准的切分维度」。**读代码时看到「为了某个并行维度专门开一个 Layer 子类」，
就意味着那个维度改变了张量布局。**

---

## 8.3 KV cache 传输与 PD 分离

### 8.3.1 为什么需要 PD 分离

**问题**：prefill 和 decode 的资源特征**完全相反**：

| | Prefill | Decode |
|---|---|---|
| 计算特征 | **计算密集**（大矩阵乘） | **访存密集**（每步只算 1 个 token，但要读全部 KV） |
| 瓶颈 | GPU 算力（FLOPS） | HBM 带宽 + KV cache 容量 |
| batch 特征 | 少量长请求 | 大量短请求 |
| 最优 batch size | 大 | 极大 |

**放在一起的代价**：两者的最优 batch 策略冲突 ——
长 prefill 会阻塞 decode（**队头阻塞 / head-of-line blocking**），
导致 decode 的**尾延迟（p99 TPOT）爆炸**。

**PD 分离**：prefill 用一组实例，decode 用另一组，中间传 KV cache。

**收益边界（面试重点）**：
- ✅ 收益大：**p99 延迟敏感** + prefill/decode 比例悬殊 + 规模大
- ❌ 收益小甚至负：小规模部署（KV 传输的开销 > 收益）、
  **KV 传输量大而网络差**（比如 KV 传输要跨机但只有 100 Gb/s 以太网）、
  请求很短（KV 本来就小）
- **判断标准**：`KV 传输时间` vs `队头阻塞造成的延迟损失`。

**⚠️ 官方文档有一句极其重要的话**（`docs/features/disagg_prefill.md:15-16`）：

> *"**Disaggregated prefill DOES NOT improve throughput.**"*

**这句话必须记住，但也要理解它的准确边界**（否则容易答错）：

| 说法 | 对不对 |
|---|---|
| 「PD 分离能提升吞吐」 | ❌ **作为直接因果是错的**（官方明确否认） |
| 「PD 分离不提升吞吐，所以没用」 | ❌ **也错** —— 它买的是**延迟可控性** |
| 「PD 分离不提升**单实例**吞吐；但在**集群层面**，因为你终于可以给 prefill 和 decode 配不同规模/不同并行策略的实例池，端到端的**有效吞吐往往还是会涨**」 | ✅ **这是最准确的表述** |

**为什么**：混部时你必须用**一套**配置同时伺候两种负载特征相反的工作负载 ——
必然有一种是次优的。拆开之后，prefill 池可以堆算力/用 PP，decode 池可以堆显存/用大 batch。
**所以「不减吞吐」+「更好的延迟」+「可独立扩缩容」三者叠加，才是 PD 分离的真实价值。**

官方对它的定位原文：

> *"This gives you the flexibility to assign different parallel strategies (e.g. `tp` and `pp`)
> to tune TTFT without affecting ITL, or to tune ITL without affecting TTFT."*
> *"Disaggregated prefilling helps you solve this issue and control tail ITL."*
> *"Chunked prefill with a proper chunk size also can achieve the same goal, but in practice
> it's hard to figure out the correct chunk size value. So disaggregated prefilling is a much
> more reliable way to control tail ITL."*

**最后一句很关键**：PD 分离是 **chunked prefill 的「更可靠的替代方案」**，
而不是一个全新的能力。**面试里可以先说「两者解决同一个问题，chunked prefill 更简单」**，
再讲 PD 分离什么时候值得上。

**面试话术**：
> 「PD 分离**不提升单实例吞吐**，官方文档明确说了 —— 它多了一次 KV 传输的开销。
> 它解决的是 **TTFT 和 ITL 的耦合**：混部时长 prefill 会顶住 decode，造成尾延迟尖刺。
> 拆开之后两边可以各自用最优的并行策略和 batch 策略，所以**集群层面**往往还能顺带提升
> 有效吞吐。判断要不要上，看你的 SLO 是**尾延迟**还是纯吞吐 ——
> 纯吞吐的话，先试 chunked prefill，它更简单。」

### 8.3.2 KV 传输的抽象层次（⚠️ 文档已过时，这是一个重要发现）

`vllm/distributed/kv_transfer/README.md` 定义了**三层抽象**：

> *"The KV transfer contains three layer of abstractions:*
> *- **KV pipe**: a FIFO pipe for torch.tensor transmission. Key APIs: `send_tensor` and `recv_tensor`.*
> *- **KV lookup buffer**: a lookup buffer for KV caches. Key: the tokens, value: the KV caches
>   (and/or hidden states). Key APIs: `insert` and `drop_select` (similar to SQL semantics).*
> *- **KV connector**: a connector that connects the KV pipe and KV lookup buffer to vLLM.
>   Key APIs: `send_kv_caches_and_hidden_states` and `recv_kv_caches_and_hidden_states`."*

**⚠️ 但实际代码里，前两层已经不存在了。**

你可以自己验证：

```bash
ls vllm/distributed/kv_transfer/
# 只有：kv_connector/  kv_transfer_state.py  README.md  __init__.py  disagg_prefill_workflow.jpg

grep -rn "KVLookupBuffer\|send_tensor\|drop_select" vllm/distributed/kv_transfer/   # 无输出
```

→ **`KV pipe` 和 `KV lookup buffer` 目录已被删除**，现在只剩 connector 层。
`kv_connector/base.py` 只是一个 10 行的向后兼容别名
（`KVConnectorBase = KVConnectorBase_V1`）。

**为什么这个发现值得记住（面试价值很高）**：

1. **vLLM 的文档明显滞后于代码。** `README.md:11-12` 和
   `docs/features/disagg_prefill.md:86-93` 都还在描述已删除的层。
2. **面试时以代码为准**。如果面试官问「KV 传输有几层抽象」，
   你答「文档说三层，但代码里只剩 connector 层」——
   **这比背文档强得多**，说明你会主动核对。
3. **它同时告诉我们一个真实的架构演进方向**：
   connector 层（和它下面的 `NixlAgentMetadata` / `MooncakeXferMetadata`）**吞掉了**
   原来 pipe + lookup buffer 的职责。
   **原本「FIFO → 查表」的两段式设计，被「RDMA 直接按 key 传」取代了。**

**为什么当初需要 lookup buffer**（README 原文，这个**问题**依然存在且值得理解）：

> *"FIFO pipe itself is not enough as prefill vLLM worker may process requests in a different
> order compared to decode vLLM worker. Say the QPS is really high, prefill worker may handle
> requests in order A -> B -> C, but the decode worker may process request C first.
> This is not the case that can be naturally handled by FIFO pipe, so we provide KV lookup
> buffer to help translate a FIFO pipe to a lookup buffer."*

**这段是极好的系统设计素材**：
> **「两个独立进程的处理顺序天然不一致」是分布式系统的普遍问题。
> 解法是把 FIFO（隐含顺序假设）升级成 key-value lookup（解耦顺序）。**
>
> 同一个思路在别处也见过：NCCL 用 tag 匹配乱序到达的数据；
> HTTP/2 用 stream id 让响应乱序返回。
> **核心都是「用标识符解耦传输顺序与使用顺序」。**
>
> 而在现代实现里，这个「lookup」被**下沉到了传输层**（NIXL/Mooncake 本身就是按
> key/descriptor 寻址的），所以 vLLM 不再需要自己实现一层。

### 8.3.3 Connector 的接口契约

（`vllm/distributed/kv_transfer/kv_connector/v1/base.py`）

**角色分离**：`KVConnectorRole`（`:124`）区分 **Scheduler 侧**和 **Worker 侧**：

| 侧 | 职责 |
|---|---|
| Scheduler | 决定**哪些请求**的 KV 需要传输、什么时候传（元数据层） |
| Worker | 实际**搬数据**（GPU 侧） |

**Worker 侧的核心接口**（`:288-344`）：

| 方法 | 语义 |
|---|---|
| `register_kv_caches(kv_caches)` | 把本实例的 KV cache 张量注册给 connector（要传地址给对端） |
| `start_load_kv(forward_context)` | 开始加载（异步的，不阻塞） |
| `wait_for_layer_load(layer_name)` | 等**某一层**的 KV 加载完 → **按层同步，实现流水线重叠** |
| `save_kv_layer(...)` | 保存某层的 KV |
| `wait_for_save()` | 等保存完成 |

**`wait_for_layer_load(layer_name)` 是关键设计**：
不要求「全部 KV 都到齐才能开始计算」，而是**逐层等待** ——
第 0 层的 KV 到了就可以算第 0 层，同时第 1 层的 KV 还在传。
**这就是 KV 传输与计算的流水线重叠。**

**`KVConnectorRole` 的角色分离**（`base.py:124`）：

| 角色 | 职责 | 为什么分开 |
|---|---|---|
| **Scheduler 侧** | 决定**哪些请求**的 KV 要传、什么时候传（只处理元数据） | 调度器**没有 GPU**，不能碰数据 |
| **Worker 侧** | 真正搬数据 | 只有 worker 有 KV cache 的指针 |

**这是「谁决策」和「谁执行」分离的经典设计** ——
和第 4 章讲的「控制面/数据面分离」是同一个原则的不同切面。

**Worker 侧接口清单**（`base.py`）：

| 方法 | 语义 | 关键点 |
|---|---|---|
| `register_kv_caches(kv_caches)`（`:264`） | 把 KV cache 张量注册给 connector | 要交换地址给对端，所以必须先注册再通信 |
| `start_load_kv(forward_context)`（`:288`） | **开始**加载（异步，立即返回） | 不阻塞 |
| `wait_for_layer_load(layer_name)`（`:307`） | 等**某一层**加载完 | **层粒度同步 → 流水线重叠** |
| `save_kv_layer(...)`（`:321`） | 保存某层 KV | 配对使用 |
| `wait_for_save()`（`:343`） | 等保存完成 | 请求结束前必须调 |

**面试题**：「KV 传输怎么和计算重叠？」
→ Connector 接口提供的是**层粒度**的同步点（`wait_for_layer_load`），而不是请求粒度。
计算可以「边算边等后面几层的 KV」。**这是一个把同步粒度做细来换并行的标准手法。**

**面试题**：「KV 传输怎么和计算重叠？」
→ Connector 接口提供**层粒度**的同步点（`wait_for_layer_load`），
而不是请求粒度 —— 计算可以边算边等后面的层。

### 8.3.4 传输实现对比

| 实现 | 库 | 传输 | 特点 |
|---|---|---|---|
| **NIXL** | NVIDIA NIXL | **RDMA** | 生产级，支持 push/pull，有 side channel 握手 |
| **Mooncake** | Mooncake | RDMA | 带 store 的层次化 KV 缓存 |
| **LMCache** | LMCache | 多种 | 面向 KV 复用的缓存层 |
| **MoRI-IO** | MoRI | RDMA | AMD 平台 |
| **HF3FS** | 3FS | 文件系统 | 走存储而非网络 |
| **Offloading** | 本地 | CPU/磁盘 | 单机场景 |

**握手与 side channel**：KV 传输需要交换「对端地址 + KV block 映射」
→ 用一个独立的 **side channel**（ZMQ，默认端口见附录环境变量表），
**而不是把元数据塞进数据通道**。这和「控制面/数据面分离」是同一个原则。

### 8.3.5 一个真实的代码质量案例（面试可以当「批评性阅读」的素材）

> 下面这些是**当前 checkout 里实际存在的缺陷**，全部可核对。
> 面试里主动指出「我读代码时发现文档和实现不一致 / 这里有 bug」，
> 是**极强的信号** —— 它证明你不是在背材料，而是真的在读。

**案例 A：文档描述的三层抽象，代码里只剩一层**（见 §8.3.2）
`README.md` 还在讲 `KV pipe` + `KV lookup buffer`，但这两个目录已被删除。
→ **教训：以代码为准，文档会滞后。**

**案例 B：配置项「读了但没用」—— 静默失效**（HF3FS）

```python
# hf3fs_connector.py:498-500  读出来存着
self._metadata_server_url = kv_config.get_from_extra_config(
    "hf3fs_metadata_server_url", "http://localhost:18000"
)
# hf3fs_connector.py:514  构造客户端时【没有传进去】
self._metadata_client = Hf3fsMetadataClient()
# hf3fs_connector.py:601  另一处也一样
self._metadata_client = Hf3fsMetadataClient()
```
`Hf3fsMetadataClient()` **不带参数**，所以内部永远用硬编码的默认值
（`hf3fs_metadata_server.py:434` 的 `http://localhost:18000`）。

**后果**：多节点 HF3FS 部署即使正确配置了远端 metadata server，
**也会静默地去连 localhost** —— 不报错，只是不工作。

**这个 bug 的类别值得记住**：「**读了配置但没把它传到使用点**」（dead/misrouted config）。
它在静态检查里很难被发现（变量确实被赋值了），只有沿着数据流追到底才看得出来。
**面试里被问「你怎么 review 别人写的分布式代码」，这就是一个具体的答案。**

**案例 C：失败被当成成功上报**（HF3FS）

```python
# :422-433
def _fail_task(self, operation, error_msg, request_id, future):
    logger.error(...)
    self.hf3fs_stats.record_failed_task_count(operation)
    future.set_result(False)        # ← 用 False 表示失败
```

但上层的「完成检查」只看 `future.done()`：

```python
# :207-229  _check_completed_saves
# 只要所有 future 都 done() 就认为这个请求完成了
```

→ **失败的任务在调度器看来和成功无法区分**（统计里记了 `record_failed_task_count`，
但控制流不知道）。**这是一个典型的「错误传播被吞掉」问题。**

**教训**：在分布式系统里，**「失败」必须是一个能被上层区分的一等状态**，
不能只记日志/指标然后返回一个看起来正常的值。
（对比 vLLM 的 NIXL connector：它有 `get_block_ids_with_load_errors()`
和 `kv_load_failure_policy`（`"recompute"` / `"fail"`），把失败**显式暴露**给调度器。
**同一个代码库里，两种做法的对比本身就很有教学价值。**）

**案例 D：MoRIIO 的自我声明限制**

```python
# moriio_connector.py 注释原文：
# "MoRIIO is not guaranteed to be thread-safe, limit 1 worker."
# "NOTE(rob): we need each rank to have a unique port. This is a hack to keep us moving."
```

→ **代码里留着「临时方案」的标记是正常的**，重要的是它们被**显式标注**了。
读代码时看到 `# TODO` / `# NOTE` / `# HACK`，
**要当作「这里有已知风险」的路标**。这些注释往往比文档更接近真相。

---

## 8.4 权重同步：RL 训练场景的通信

（`vllm/distributed/weight_transfer/nccl_engine.py`，文档 `docs/training/weight_transfer/nccl.md`）

**场景**：RLHF/RLVR 里，trainer 训练完一轮 → 把新权重复制给 inference 引擎（rollout）。

**为什么用 broadcast 而不是循环 send**：

| 方案 | trainer 出口流量 | 说明 |
|---|---|---|
| 循环 send 给 N 个 worker | **N × 权重大小** | trainer 的网卡成为瓶颈 |
| **broadcast** | **1 × 权重大小**（逻辑上） | 由 NCCL 内部负责分发，可用树形/环形 |

**关键设计**（文档 `:11-19`）：
1. trainer 和所有 worker 加入**同一个** NCCL 组（用 `StatelessProcessGroup`）；
2. trainer 是 rank 0，worker 从 `rank_offset = 1` 开始；
3. **可选 packed broadcasting**：把许多小张量打包成大 buffer，配 double/triple buffering
   + CUDA stream 重叠。

**为什么需要打包**：模型有几千个参数张量，其中大量是小张量（bias、norm 的 weight）。
**每次 NCCL 调用都有固定开销** —— 几千次小调用会被延迟吃掉。
打包成 1 GiB 的 buffer（默认）后，调用次数降低几个数量级。

**内存代价**（文档 `:104-107`）：`packed_buffer_size_bytes × packed_num_buffers`
= 1 GiB × 2 = **每侧 2 GiB**。

**pynccl 在这里的独特价值**：`PyNcclCommunicator.from_unique_id_bytes`
（`pynccl.py:163-217`）允许**不加入任何进程组**的 peer 参与通信 ——
比如 trainer 是 JAX 写的，根本不用 torch。代价是**没有 barrier**，
所以「所有 rank 必须同时进入初始化」，且**外部 peer 必须匹配 vLLM 内部的 warm-up all-reduce**
（否则死锁）。docstring 明确警告了这两点。

**面试题**：「两个不同框架怎么直接用 NCCL 通信？」
→ 交换 128 字节 uniqueId（走 socket，不走 NCCL）→ 双方各自 `ncclCommInitRank`
→ **双方必须对操作序列达成一致**（包括对方框架内部的 warm-up 操作）。
**NCCL 是「隐式合约」：操作顺序不匹配就 hang。**

---

## 8.5 训练 vs 推理：通信模式的根本差异

面试里常被问「你了解训练和推理的通信差异吗」。这张表是答案：

| 维度 | 训练 | 推理 |
|---|---|---|
| **主导通信** | **all-reduce**（梯度规约 + TP） | **all-to-all**（MoE）+ all-reduce（TP） |
| 通信量级 | **极大**（梯度 = 参数量，70B 模型每步 140 GB） | 相对小（激活值） |
| 频率 | 每 step 一次大 all-reduce | 每层 2 次小 all-reduce |
| 优化重点 | **带宽**（梯度规约）+ 与反向计算重叠 | **延迟**（decode 小消息） |
| 关键技巧 | ZeRO 分片、梯度压缩、通信与 backward 重叠 | DBO、custom AR、CUDA Graph、EPLB |
| 典型并行 | DP + TP + PP + ZeRO（**DP 为主**） | TP + EP + DP（**EP/TP 为主**） |
| 容错 | checkpoint + 重启 | 弹性 EP、故障转移 |
| 消息大小 | MB–GB 级 | KB 级 |
| 算法选择 | ring（带宽最优） | tree/NVLS/custom（延迟最优） |

**一句话总结**：
> **训练优化「吞吐」（大消息、带宽），推理优化「延迟」（小消息、步数）。
> 这决定了两者选了 N C C L 里完全不同的算法和协议。**

**面试加分**：能指出 **vLLM 同时在做这两件事** ——
它的 prefill 阶段像训练（大消息、带宽敏感），
decode 阶段像推理（小消息、延迟敏感）。
**这就是为什么 vLLM 的 all-reduce 要分档（第 4 章），为什么需要 chunked prefill。**

---

## 8.6 通信趋势：未来两年会考什么

（这部分是「视野题」，面试里能主动提是强信号。以下为方向性判断，非代码事实。）

| 趋势 | 含义 | vLLM 里的迹象 |
|---|---|---|
| **网内计算** | 规约/多播由交换机或 NVSwitch 做，不走 GPU 往返 | SHARP / NVLS；`multimemSupport` 字段 |
| **GPU 自主通信** | GPU kernel 直接发网络包，不等 CPU | **NCCL GIN**；DeepEP v2 强制要求 |
| **对称内存** | 对端内存可直接 load/store，不用消息传递 | `ncclCommWindowRegister`；`VLLM_USE_NCCL_SYMM_MEM` |
| **通信库与 kernel 融合** | 不再是「先算再传」，而是一个 kernel 里做两件事 | `scaled_matmul_reduce_scatter`、all-reduce+RMSNorm |
| **机柜级 NVLink 域** | NVL72 让 72 卡像一个大 GPU | MNNVL 支持；`flashinfer_nvlink_*` 后端 |
| **弹性与容错** | rank 动态加入/退出，故障不重启 | `elastic_ep/`；`enable_fault_tolerance` |
| **算力互联开放标准** | UALink / UEC 挑战 NVLink 与 IB 的封闭生态 | （vLLM 侧暂无直接证据） |

**面试话术**：
> 「通信的演进方向可以概括为一句话：**让通信越来越不像通信**。
> 从『CPU 下发的消息传递』→『GPU 自主发起』→『网内计算』→『对端内存直接访存』。
> 每一步都在减少通信的**语义开销**（协议、拷贝、同步），而不只是提高带宽。
> **带宽的提升有物理上限，但语义开销的消除空间还很大。**」

---

## 8.7 本章自检题

1. 为什么 MoE 用 all-to-all 而不是 all-reduce 或 all-gather？
2. 画出 MoE 一层的完整数据流，标出两次 all-to-all 的位置。
3. `topk_ids` 在 dispatch 前后分别是什么坐标？-1 sentinel 是干什么的？
4. 变长 all-to-all 的 sizes 元数据有哪两种同步方式？
5. 为什么 EP 的负载不均比 TP 更严重？EPLB 怎么解决？它会不会改变模型输出？
6. 六种 all-to-all 后端各自最适合什么场景？决策树怎么画？
7. MoE 量化方案会被什么通信约束限制？举两个例子。
8. Sequence Parallel 省的是什么？通信量减少了吗？（这题是陷阱）
9. SP 里为什么必须做 padding？padding 带来了什么新问题？
10. CP 和 SP 的区别是什么？CP 的收益边界在哪？
11. vLLM 为什么需要 PCP 和 DCP 两组 context parallel 配置？
12. KV 传输为什么要 lookup buffer 而不是纯 FIFO？这个设计模式还在哪些地方出现过？
13. `wait_for_layer_load(layer_name)` 为什么是「按层」而不是「按请求」？
14. 权重同步为什么用 broadcast 而不是循环 send？为什么还要打包？
15. 训练和推理的通信模式有哪些根本差异？

**下一章**：[`ch05-debugging-runbook.md`](ch05-debugging-runbook.md)（排障）或
[`ch06-interview-bank.md`](ch06-interview-bank.md)（面试题库，已扩充大量 MoE/DBO/CP 题目）。
