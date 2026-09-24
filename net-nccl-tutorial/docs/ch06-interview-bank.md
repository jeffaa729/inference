# 第 6 章　校招面试题库：从基础到系统设计

> **使用方式**：先自己答，再看答案。每题标注了难度：
> `[基础]` 必须会、`[进阶]` 能区分候选人、`[系统]` 决定是否给 offer。
>
> 带 ✅ 的答案有 vLLM 代码证据（文件:行号），可以主动说出来 —— **主动引用源码是校招面试里的强信号**。
> 带 📖 的是通用知识（NCCL/网络常识），不需要代码依据。

---

## 第一部分：网络与硬件基础

### Q1 `[基础]` 400 Gb/s 的 IB 网卡，单向带宽是多少 GB/s？

**答**：`400 / 8 = 50 GB/s`。

**要点**：Gb/s 是比特，GB/s 是字节，差 8 倍。这是最常见的口误。
进一步：双口 = 100 GB/s；`GB/s` 用 `10^9`，`GiB/s` 用 `2^30`，论文里常混用。

📖 **追问**：NVLink「900 GB/s」是单向还是双向？→ H100 NVLink 4 是**双向 900 GB/s**，
单向约 450 GB/s。**答题时主动说明口径**是加分项。

---

### Q2 `[基础]` RDMA 为什么比 TCP 快一个数量级？

**答**：四个原因，按重要性排：

| # | 机制 | 效果 |
|---|---|---|
| 1 | **内核旁路（kernel bypass）** | 应用直接操作网卡，不经过内核协议栈，省掉系统调用和上下文切换 |
| 2 | **零拷贝（zero-copy）** | 网卡直接 DMA 读写用户态内存，省掉内核缓冲区的两次拷贝 |
| 3 | **CPU 不参与数据搬运** | CPU 只下发一次 doorbell，之后网卡自己干活，CPU 可以去做别的事 |
| 4 | **硬件卸载 + 无损网络** | 校验、重传、流控都在硬件；IB 用 credit-based 流控从根本上不丢包 |

📖 **追问**：「为什么 AI Infra 必须用 RDMA？」
→ decode 阶段每步可能只有几百 μs，TCP 的 10–30 μs 延迟占比太大，
而且 CPU 参与会打满 CPU（vLLM 的主进程还要做调度、采样）。

**追问 2**：「RoCE 和 IB 怎么选？」
→ IB：无损原生、低抖动、贵、需要 subnet manager；
RoCE：跑在以太网上、便宜、但必须手工配 PFC + ECN 造出无损，PFC 配错会引发广播风暴/死锁。

---

### Q3 `[基础]` `nvidia-smi topo -m` 里的 `NV4`、`PIX`、`SYS` 是什么？

**答**：GPU/NIC 之间的**连接距离矩阵**：

| 值 | 含义 | 相对速度 |
|---|---|---|
| `NV#` | 经 # 条 NVLink | 最快 |
| `PIX` | 单个 PCIe Switch | 快 |
| `PXB` | 多个 PCIe Switch | 中 |
| `PHB` | PCIe Host Bridge | 较慢 |
| `SYS` | 跨 NUMA/CPU 互联 | 最慢 |

📖 **追问**：「为什么这个矩阵重要？」
→ 因为 NCCL 靠它决定 ring 顺序和选哪张网卡；手工写通信代码如果忽略它就会慢。
**NIC 列**（`nvidia-smi topo -m` 里网卡那一行）决定 GPU↔网卡的亲和性。

✅ **可以补一句 vLLM 的做法**：
vLLM 用 `VLLM_GPU_NIC_PCIE_MAPPING` + `VLLM_NIC_SELECTION_VARS`
（`vllm/v1/executor/vllm_net_devices.py:180`）**自动按 PCIe 距离选网卡**并写进 `NCCL_IB_HCA`。

---

### Q4 `[进阶]` 为什么 TP 必须待在单机 NVLink 域内？手算说明。

**答**：因为 TP 每层都要 all-reduce，而**延迟项随 world size 线性增长**。

✅ **手算**（第 1、2 章的方法）：TP=8，hidden=8192，bf16，80 层，每层 2 次 all-reduce。

```
每次 all-reduce 数据量 S = 8192 × 2 B = 16 KiB
ring 步数 = 2(N-1) = 14

机内（NVLink 单向 450 GB/s，α=3 μs）：
  T ≈ 14×3 μs + 2×(7/8)×16 KiB / 450 GB/s ≈ 42 + 0.06 ≈ 42 μs
  每层 2 次 → 84 μs；80 层 → 6.7 ms？
```

⚠️ 注意：**上面这个算法是错的**，因为 NCCL 的 ring 用多 channel 并发，实际不是
`14 × α` 串行。**面试时正确的说法是**：

> 「用 α-β 模型做**量级估算**：机内 NVLink 的 all-reduce 每 token 每层大约几 μs 量级；
> 换成跨机 IB（α≈5 μs，且 ring 步数不变但每步都要过网卡），延迟会涨一个数量级。
> 而 decode 一步的计算时间只有几十 μs 量级，**所以跨机 TP 会被通信吞掉**。
> 精确值必须实测，但结论（跨机 TP 不可行）是确定的。」

**这个「承认模型不精确、但坚持结论方向」的回答方式，比背一个错误数字好得多。**

✅ **vLLM 的代码证据**（`parallel_state.py:1936-1939`）：

> *"Note that for efficiency, the caller should make sure adjacent ranks are on the same DGX box."*

以及布局顺序 `ExternalDP x DP x PP x PCP x TP`（`parallel_state.py:1977`）——
**TP 是最后一维，所以 TP 组永远是相邻 rank**。

📖 **追问**：「跨机怎么扩展？」
→ 用 **PP**（点对点传 activation，量小）或 **DP/EP**（EP 的 all-to-all 可跨机且能和计算重叠）。

---

## 第二部分：集合通信

### Q5 `[基础]` 说出 8 个集合操作，并指出哪两对最容易混。

**答**：

| 操作 | 一句话 |
|---|---|
| broadcast | root 发给所有人 |
| reduce | 全部规约，只给 dst |
| all-reduce | 全部规约，人人有份 |
| gather | 各人一份给 dst |
| scatter | root 把 N 份分下去 |
| all-gather | 收集后广播给所有人 |
| reduce-scatter | 规约后切成 N 份分发 |
| all-to-all | 各人切 N 份，第 i 份给 rank i |

**最容易混的两对**：

1. **`reduce` vs `all-reduce`** → 结果落在 dst 还是所有人。
2. **`all-gather` vs `all-to-all`** → 前者**每个人拿到所有人的全部数据**（数据量 ×N），
   后者**每个人只拿到属于自己的那份**（数据量不变，只是重新分布）。

📖 **追加金句**：「MoE 用 all-to-all 而不是 all-gather，因为 token 只需要送到
**持有它选中专家**的 rank，不需要所有人都看到所有 token。」

---

### Q6 `[进阶]` all-reduce 有哪几种算法？各自的复杂度？

**答**：

| 算法 | 顺序步数 | 每 rank 搬运量 | 适用 |
|---|---|---|---|
| **Ring** | `2(N-1)` | `2(N-1)/N × S → 2S` | 大消息（带宽最优） |
| **Tree** | `2 log₂ N` | `≈ 2S` | 小消息（延迟最优） |
| **Recursive halving-doubling** | `log₂ N` | `≈ S` | 理论最优但有 2 的幂约束 |

📖 **关键补充（这道题的区分点）**：

> Recursive halving-doubling 带宽比 ring 还好一倍，但**有两个约束**：
> ① 要求 N 是 2 的幂；② 通信模式是**全交换**（任意两 rank 直接通信），
> 在 ring 拓扑或 rank 跨节点时假设不成立。
> **所以 NCCL 主要用 ring 和 tree。**

**N=8 的步数对比**：ring 14 步 vs tree 6 步。
若 α=3 μs，仅延迟项就差 `(14-6)×3 = 24 μs` —— 对一个总传输时间只有 0.04 μs 的
16 KiB 消息，**ring 完全不可接受**。

✅ **vLLM 里的同一个 tradeoff**：custom all-reduce 的 **one-shot vs two-shot**
（`csrc/custom_all_reduce.cuh`）：one-shot 步数少但每 block 要读 N 份；
two-shot 两次 kernel 但每次只搬 1/N。切换点是
`world_size<=4 && bytes<512KiB` 或 `world_size<=8 && bytes<256KiB` → one-shot。

**面试技巧**：把 GPU 内部的 one-shot/two-shot 和网络上的 ring/tree 联系起来讲，
**说明你理解的是 tradeoff 的本质，而不是两个孤立的知识点**。

---

### Q7 `[进阶]` 推导 ring all-reduce 的搬运量，并解释那个「2」。

**答**：

```
阶段 1 Reduce-Scatter：N-1 步，每步每 rank 发一个 S/N 的块
  → 每 rank 搬运 (N-1)/N × S
阶段 2 All-Gather：N-1 步，每步每 rank 发一个 S/N 的块
  → 每 rank 搬运 (N-1)/N × S
总搬运量 = 2(N-1)/N × S → 2S（N 大时）
```

**那个「2」的含义**：**每个字节都要在网络上走两遍** ——
第一遍为了规约（每个 rank 收到的部分和要往前传），
第二遍为了扩散（规约好的结果要传回所有人）。

📖 **追问**：「那 `reduce-scatter + all-gather` 和 all-reduce 是一样的吗？」
→ 带宽项**相同**（各 `(N-1)/N × S`），但 `reduce-scatter + all-gather` 的优势是
**中间结果可以立即被消费且更小（S/N）**，还能和计算重叠。
ZeRO 的参数分片、序列并行都基于这个分解。

---

### Q8 `[进阶]` NCCL 怎么决定用 ring 还是 tree？

**答**：基于**消息大小 + world size + 拓扑 + 是否支持 SHARP/NVLS** 自动选择。
可用 `NCCL_ALGO` / `NCCL_PROTO` 强制覆盖（调试用）。

**核心逻辑**：
- 小消息 → 延迟主导 → 选**步数少**的（tree / NVLS）；
- 大消息 → 带宽主导 → 选**搬运量少**的（ring）。

**协议层还有第二层权衡**：

| 协议 | 策略 | 何时用 |
|---|---|---|
| LL | 小包立刻发 | 极小消息 |
| LL128 | 128 B 粒度 | 中等（NVLink 常用） |
| Simple | 攒大块再发 | 大消息 |

📖 **面试禁区**：不要只说「小消息 tree、大消息 ring」就停。
**准确的表述要包含「算法层 + 协议层两层权衡」**。

✅ **vLLM 的同类做法**（这道题的最佳加分点）：vLLM 的 all-reduce 也是按
**消息大小分档**选实现的：
- 极小（≤16K）：NCCL 对称内存（`NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`，`all_reduce_utils.py:109-118`）
- 中段（16K–128K/512K）：custom AR
- 大：对称内存
- 超过 custom AR 上限（H100 TP=8 是 **256 KiB**，`CUSTOM_ALL_REDUCE_MAX_SIZES`，
  `all_reduce_utils.py:31-56`）→ 回退 NCCL

> **「同一个操作按消息大小分档用不同实现」是工业级通信库的通用模式** —— 能主动指出
> 「NCCL 内部这么做，vLLM 在上层也这么做」，说明你理解了模式的普适性。

---

## 第三部分：vLLM 实战（✅ 为主）

### Q9 `[进阶]` vLLM 为什么不直接用 NCCL 的 all-reduce，要自己写一个？✅

**答**：三个具体理由，都有代码依据：

**理由 1：CUDA Graph 兼容性**（这是 pynccl 存在的第一原因）
`vllm/distributed/device_communicators/pynccl_wrapper.py:4-12` 的文件头注释说明：
cupy 在 comm init 时会卡住，`torch.distributed` 会调用 graph capture 期间被禁止的 CUDA API。
→ vLLM 用 ctypes 直接 `dlopen` NCCL，**自己控制通信器创建时机**。

**理由 2：NCCL 的固定开销对 decode 太大**
decode 每步只处理 1 个 token，all-reduce 只有 16 KiB，
但 NCCL 每次调用有 μs 级的协议栈/调度开销。
custom AR 用 **CUDA IPC + 对端指针直接读写**，一次 kernel launch 完成。

**理由 3：NCCL 不是全区间最优**
`all_reduce_utils.py` 的调优表明确显示：**不同大小区间的最优实现不同**
（小消息看延迟、中段看拷贝开销、大消息看带宽）。vLLM 用一组实现覆盖全区间。

**追问**：「那 custom AR 什么时候不用？」✅
→ 完整 gate 链（`custom_all_reduce.py`）：
① C++ 扩展未编译 → 禁用；
② `world_size == 1` → 不需要；
③ `world_size not in [2,4,6,8,16]` → 禁用；
④ **`world_size > 2` 时要求 NVLink 全互联**（`:236-242`）—— 非全互联的 4+ 卡
   PCIe 机器上「custom AR 收益不如 NCCL」（`:416-417` 的注释）；
⑤ 无 P2P → 禁用；
⑥ 消息超过 `max_size` → 回退 NCCL。

---

### Q10 `[系统]` vLLM 一次 all-reduce 的完整链路是什么？✅

**这是本章最重要的一题。** 答案就是第 4 章 §4.4 的 8 层链路：

```
模型代码 (linear.py:1769)
 → communication_op.py:14          语义层
 → GroupCoordinator.all_reduce     世界组/TP 组，world_size==1 短路 (parallel_state.py:722-749)
 → CudaCommunicator.all_reduce     8 路后端选择 (cuda_communicator.py:305-377)
 → CustomAllreduce.custom_all_reduce (custom_all_reduce.py:441)
 → C++ kernel (csrc/custom_all_reduce.cuh)
 → NVLink P2P 读写对端 IPC buffer
```

**8 路后端的实际顺序**（`cuda_communicator.py:305-377`）：

1. NCCL 对称内存（NVLS）
2. QuickReduce（ROCm）
3. FlashInfer PCIe IPC
4. FlashInfer（mnnvl/trtllm）
5. AITER custom（ROCm）
6. **vLLM custom all-reduce**
7. torch 对称内存
8. PyNCCL → 最后兜底 `torch.distributed`

⚠️ **加分点**：主动指出**启动日志打印的顺序和实际 dispatch 顺序不同**
（日志函数 `_log_all_reduce_backend_selection`，`:233-252`），
它的 docstring 自己说明只是「可能的子集」（`:234-241`）。
**能指出这种「文档/日志与实际行为不一致」的细节，是读过代码的强证明。**

---

### Q11 `[进阶]` vLLM 的 custom all-reduce 为什么能在 CUDA Graph 里工作？代价是什么？✅

**答**：
**机制**：每个 rank 分配一块显存 → 用 `cudaIpcGetMemHandle` 交换句柄 →
`cudaIpcOpenMemHandle` 打开所有对端 → 把对端指针**写进设备内存**（`rank_data`）→
kernel 里直接读对端地址。

**CUDA Graph 的关键难点**：**graph 捕获时 kernel 参数必须固定**，但输入张量地址是变的。
vLLM 的解法（C++ 注释说明）：
1. graph 捕获时，输入先 **copy 进预注册的 IPC buffer**（地址固定）；
2. 捕获期间收集所有用到的 buffer 地址，事后 `register_graph_buffers` 让各 rank 交换。

**代价**（`custom_all_reduce.py:454-456` 的注释）：

> *"outside of cuda graph context, custom allreduce incurs a cost of cudaMemcpy, which should be
> small (<=1% of overall latency)"*

→ **eager 模式多一次显存拷贝**，但收益远大于成本。

📖 **追问**：「CUDA Graph 和通信还有哪些冲突？」
→ ① 捕获期间不能做同步/建连（NCCL 懒初始化会炸）；
② 通信 kernel 的 SM 占用要固定，否则 replay 行为不一致。

---

### Q12 `[系统]` MoE 的 all-to-all 和 TP 的 all-reduce 有什么区别？✅

**答**：

| 维度 | TP all-reduce | EP all-to-all |
|---|---|---|
| 通信对象 | **所有** TP rank | **只发给持有目标专家**的 rank |
| 每层次数 | 2 次（o_proj + down_proj） | 1 次 dispatch + 1 次 combine |
| 通信量与什么相关 | hidden size，**与专家数无关** | topk 和专家分布 |
| 扩展性瓶颈 | 延迟随 N 线性增长 | **负载不均** + 跨机带宽 |
| 典型范围 | 机内 NVLink | 可跨机（IB） |
| 库 | NCCL | DeepEP/NIXL/FlashInfer/MoRI/AG-RS |

✅ **vLLM 的证据**：
- 两个阶段：`all2all.py:101`（dispatch）/ `:138`（combine）
- 后端选择是 if/elif 链（`cuda_communicator.py:163-225`），**没有自动选择**
- 默认后端 `allgather_reducescatter`（`vllm/config/parallel.py:197`）
- **EP_SIZE = DP × PCP × TP**（`parallel_state.py:2087-2096`）

**追问**：「为什么 DeepSeek-V3 能用 EP 扩展到几百卡，dense 模型用 TP 只能到 8 卡？」
→ 因为 EP 的通信量由**路由稀疏性**决定（每 token 只去 topk 个专家），
不随 EP size 线性增长；而 TP 的 all-reduce 是全局的、每层两次、延迟随 N 线性涨。
**EP 的代价转移到「负载均衡」和「all-to-all 重叠」上，这两个都能用工程手段解决。**

---

### Q13 `[进阶]` 变长的 all-to-all 怎么实现？✅

**答**（`AgRsAll2AllManager`）：

**问题**：MoE 里每个 rank 发给不同对端的数据量不同（取决于 token 选了多少该 rank 的专家），
而 NCCL **没有原生的变长 all-gather/all-to-all**。

**vLLM 的解法**：
1. **sizes 走旁路**：各 rank 的 token 数通过 **DP 组的 CPU all-reduce** 汇总，
   存进 `DPMetadata.num_tokens_across_dp_cpu`，forward 时读出
   （`AgRsAll2AllManager._get_sizes`，`all2all.py:60-68`）——
   **不在通信内部交换元数据**，省一次往返。
2. **用 N 次 broadcast 拼 all-gatherv**（`pynccl.py:293-326`）：
   ```python
   ncclGroupStart()
   for i, size in enumerate(sizes):
       ncclBroadcast(recvbuff=dst_slice, sendbuff=input_tensor, count=..., root=i, ...)
   ncclGroupEnd()
   ```
   每次 broadcast 的 root 不同 → 等价于「每个 rank 播自己的那份」。

**为什么用 `ncclGroupStart/End`**：把 N 个操作**并发下发**，避免逐个同步
（这是 `ncclGroup` 语义的核心用途）。

---

### Q14 `[进阶]` DeepEP 的 HT 和 LL 模式怎么选？为什么 LL「不占 SM」很重要？✅

**答**：

| | High Throughput | Low Latency |
|---|---|---|
| 阶段 | **prefill** | **decode** |
| `low_latency_mode` | `False`（`all2all.py:239`） | `True`（`:321`） |
| CUDA Graph | 不支持 | **支持** |
| SM 占用 | **20**（`:171`） | **0**（`:345-347`） |
| RDMA buffer | 固定 1 GiB | 按需计算（`get_low_latency_rdma_size_hint`） |
| qps/rank | `num_sms // 2 = 10` | `num_local_experts` |

**为什么 LL 不占 SM 是关键**（代码注释原文）：
> *"DeepEP LL uses RDMA so no SMs are used for communication"*

→ **通信 kernel 不占 SM，意味着计算和通信可以物理并行**（不是时间片轮转）。
这是 decode 阶段「通信藏在计算后面」的硬件基础。

**为什么 HT 反而要用 SM**：prefill 大 batch 通信量大，用 SM 做数据搬运能打满带宽；
此时计算本来就是瓶颈，牺牲 SM 可接受。

**追问**：「vLLM 为什么不做自动切换？」
→ 因为一个部署常同时服务 prefill 和 decode（chunked prefill），
**最优后端取决于负载比例**，让用户按场景显式选（`--all2all-backend`）比自动选错更安全。
**这也是「配置显式化」优于「智能默认」的一个案例。**

---

### Q15 `[进阶]` 什么是 NCCL GIN？为什么 DeepEP v2 依赖它？✅

**答**：
**GIN = GPU-Initiated Networking**：让 **GPU kernel 内部直接发起网络操作**，
不需要 CPU 参与下发。

**为什么重要**：传统模式下 GPU 要通信必须先通知 CPU，CPU 再下发命令给网卡 ——
这个往返是 μs 级的，且无法和计算细粒度重叠。GIN 让通信像访存一样在 kernel 里发起。

**vLLM 怎么检查**（`all2all.py:1043-1064`）：
```python
probe = torch.zeros(1, device="cuda")
torch.distributed.all_reduce(probe, group=group)   # 强制创建 NCCL communicator
gin_type = query_nccl_gin_type(group)              # 读 ncclCommProperties.ginType
if gin_type == 0:
    raise RuntimeError("DeepEPv2 requires NCCL GIN ... IBGDA-capable InfiniBand NICs ...")
```

**两个工程细节**（都是加分点）：
1. **为什么要先做 dummy all_reduce**：`ProcessGroupNCCL` **懒创建** communicator，
   不先初始化就会拿到空指针，误判为「不支持 GIN」（代码注释原话：
   *"Initialize this exact group before querying so a null comm pointer is not mistaken for
   missing GIN support."*）。
2. **为什么快速失败而不是降级**：v2 的性能完全依赖 GIN，降级后不如直接用 v1 LL。
   **「宁可报错也不要静默降级到更慢的路径」是一个成熟的工程判断。**

📖 **延伸（展示视野）**：`ncclCommProperties` 结构体（vLLM 用 ctypes 声明了它，
`pynccl_wrapper.py:66-87`）里有 `multimemSupport`（NVLink SHARP 多播）、`nLsaTeams`
（可用 load/store 直接访问的对端）、`hostRmaSupport` 等字段。
**这些字段代表的方向是一致的：通信正在从「CPU 下发的消息传递」转向「GPU 自主的网内计算」。**

---

### Q16 `[系统]` 什么是「确定性推理」？为什么它和性能互斥？✅

**答**：
**确定性（batch invariance）** = 同样的请求，不管和哪些请求一起 batch，输出**逐 bit 相同**。

**为什么难**：浮点加法**不满足结合律**，`(a+b)+c ≠ a+(b+c)`。
所有提高通信性能的手段都会改变求和顺序：
- tree vs ring（不同归约树）
- 多 channel（把数据切开并行，改变累加分组）
- NVLS/SHARP（网内规约，硬件决定顺序）
- 自定义 kernel（自己的累加顺序）

**vLLM 的具体措施**（`vllm/model_executor/determinism/batch_invariant.py:1141-1156`）：
强制设置一组 NCCL 环境变量：
```
NCCL_ALGO=allreduce:tree     NCCL_PROTO=Simple
NCCL_NVLS_ENABLE=0           NCCL_COLLNET_ENABLE=0
NCCL_MIN_NCHANNELS=1         NCCL_MAX_NCHANNELS=1     ← 最致命
NCCL_P2P_NET_DISABLE=1       NCCL_NTHREADS=1
NCCL_LAUNCH_MODE=GROUP       NCCL_SOCKET_NTHREADS=1
```
**外加**：禁用 custom all-reduce、FlashInfer AR、PCIe IPC AR、对称内存
（`cuda_communicator.py:64,68`；`symm_mem.py:114`；`all_reduce_utils.py:136,165`）。

**`NCCL_MAX_NCHANNELS=1` 把 channel 并发压到 1，带宽直接掉** —— 这就是性能代价的来源。

**面试加分表述**：
> 「确定性和性能在**物理上**互斥，不是工程没做好 —— 因为最快的归约顺序取决于
> 消息大小和拓扑，而这两者随 batch 变化。要确定性就必须固定顺序，
> 固定顺序就必须放弃自适应优化。」

---

## 第四部分：系统设计

### Q17 `[系统]` 设计：把 LLaMA-70B（bf16，140 GB 权重）部署到 8×A100-80GB 上做在线服务。

**答题框架**（不要直接跳答案）：

**第 1 步：算显存**
```
权重 140 GB ÷ 8 卡 = 17.5 GB/卡
A100-80GB 剩 62.5 GB 给 KV cache + activation
```
→ 单副本 TP=8 可行。

**第 2 步：选并行策略**
- **TP=8 单机**：优先。因为 A100 有 NVLink 全互联（600 GB/s），
  `nvidia-smi topo -m` 会显示 `NV`。
- **PP=8**（TP=1）：如果卡间只有 PCIe（比如 L40S），
  官方文档明确建议用 PP 而非 TP（`docs/serving/parallelism_scaling.md` 的 edge case note）。

**第 3 步：算通信预算**
每层 2 次 all-reduce（`linear.py:1769` 的 o_proj 和 down_proj），
hidden=8192 → 每次 16 KiB，80 层。**必须确认通信占比**（Nsight 实测）。

**第 4 步：关键优化**
- 确认 custom all-reduce 生效（A100 是 9.0，TP=8 上限 **256 KiB**，16 KiB 远小于上限 ✓）
- 开 CUDA Graph（decode 阶段省 launch 开销）
- 检查有没有 `VLLM_BATCH_INVARIANT`（**开了性能会掉**）

**第 5 步：容量与并发**
看 vLLM 启动日志的 `GPU KV cache size: N tokens` 和 `Maximum concurrency: Mx`
（`docs/serving/parallelism_scaling.md` 明确指导看这两行）。

📖 **追问**：「如果要 4 副本？」→ 每副本 TP=2（2×A100），
4 个副本 × 2 卡 = 8 卡。**TP 越小通信越少，副本越多吞吐越高** ——
这正是「TP vs 副本数」的经典权衡，也是 vLLM 里 DP 的动机。

---

### Q18 `[系统]` 设计：DeepSeek-V3（671B MoE）部署到 2 节点 × 8×H100。怎么配？

**答**（这是 EP + DP 的典型场景）：

**推荐配置**（参考 `docs/serving/expert_parallel_deployment.md`）：
```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address <节点1 IP> \
    --data-parallel-rpc-port 13345 \
    --api-server-count=8
```

**关键设计决策及理由**：

| 决策 | 理由 |
|---|---|
| **TP=1** | 671B 模型的 attention 部分很小，不需要 TP；省掉每层的 all-reduce |
| **DP=16（跨 2 节点）** | attention 权重复制 16 份，每份只需容纳 attention + 自己那份专家 |
| **EP=TP×DP=16** | 专家分散到 16 张卡上（✅ `parallel_state.py:2087-2096`） |
| **`deepep_low_latency`** | 在线服务以 decode 为主；LL 模式**不占 SM** 且支持 CUDA Graph |
| **`--api-server-count=8`** | 官方建议「缩放到 local ranks 数」 |
| **`--data-parallel-address` + rpc port** | 多节点 DP 的协调（默认 `data_parallel_rpc_port=29550`，✅ `vllm/config/parallel.py:143`） |

**⚠️ 关键前提检查**：
- **`deepep_low_latency` 需要 RDMA（IB）**；
- 如果用 `deepep_v2` → **需要 NCCL ≥ 2.30.4 + IBGDA 网卡**
  （✅ `all2all.py:1058-1064`），而 *"PyTorch ships an older NCCL"*
  （官方文档原文），需要手动升级；
- **跨节点 NVLink（MNNVL/GB200）** 才考虑 `flashinfer_nvlink_*` 后端。

**追问**：「如果节点间只有 100 Gb/s 以太网呢？」
→ 那么 EP 的 all-to-all 会成为瓶颈。此时应该：
① 缩小跨机 EP（`--data-parallel-size-local` 更大，减少跨机通信）；
② 或者改用 **PP 跨机** + TP/EP 机内；
③ 或者考虑 disaggregated prefill/decode。
**核心原则：把延迟敏感的通信（TP）留在机内，把可容忍延迟的（EP/PP）放到机间。**

---

### Q19 `[系统]` 线上 TPOT 突然从 30 ms 涨到 90 ms，你怎么排查？

**答题框架（体现方法论，比背参数重要）**：

**第 0 步：先分类，不要直接怀疑通信**
```
① 是全局变慢还是尾延迟变差？（p50 vs p99）
② 是突然变化还是渐变？（对应事件：上线变更 / 流量增长 / 邻居干扰）
③ 是所有请求还是特定请求？（特定长度/特定 prompt → 不是通信问题）
```

**第 1 步：看 vLLM 自己的指标**
- `GPU KV cache size` / `Maximum concurrency`（启动日志）→ 有没有因为显存碎片掉下来
- **preemption / 抢占数** → KV cache 不够会导致反复重算
- running/waiting 队列长度 → 是不是排队而不是算得慢

**第 2 步：区分计算 vs 通信**（关键分水岭）
```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node ...
```
- 如果通信 kernel 占比没变 → **不是通信问题**（去查 attention/MoE kernel、checkpoint）
- 如果通信 kernel 时间涨了 → 继续第 3 步

**第 3 步：通信侧排查**（对照第 5 章）
| 现象 | 可能原因 | 验证 |
|---|---|---|
| NCCL 走了慢路径 | 网卡选错 / IB 降级到 socket | `NCCL_DEBUG=INFO` 看选了哪个 HCA 和传输 |
| 退出了 custom AR | 消息大小变了 / P2P 失效 | 看后端选择日志；对比 `VLLM_DISABLE_CUSTOM_ALL_REDUCE` |
| 被 batch invariant 影响 | 有人设了 `VLLM_BATCH_INVARIANT` | 查环境变量（会禁用全部快速路径）|
| EP 负载不均（straggler） | 路由分布变了（新流量模式） | 看 EPLB 是否开启；每个 rank 的 token 数 |
| 邻居干扰 | 同机其它进程抢 NVLink/PCIe | `nvidia-smi` 看有没有别的进程 |
| 跨机带宽被抢 | 别的任务在跑 collective | 看 IB 计数器 |

**第 4 步：如果最近有变更** → `VLLM_BATCH_INVARIANT` 是头号嫌疑（它会禁用所有快速路径）。

**面试加分点（主动说出）**：
> 「如果 profiler 显示通信占比只有 15%，那 3 倍变慢不可能来自通信 ——
> **不要假设慢就是通信问题**。我会先确认时间分布，再针对性排查。」

---

### Q20 `[系统]` 「设计一个推理框架的通信层」—— 你会怎么分层？

**答**（这题考架构能力，用 vLLM 的实际分层作为参考）：

```
第 4 层  模型代码        linear.py 里一次 all_reduce 调用（不关心怎么实现）
第 3 层  语义/拓扑层     通信操作 → 进程组的映射（GroupCoordinator）
                        管理 TP/PP/DP/EP 组的创建、CPU/GPU 双通道
第 2 层  后端选择层      按消息大小/dtype/硬件选具体实现（CudaCommunicator）
                        一组实现 + 实测调优的阈值表
第 1 层  实现层          custom AR / pynccl / 对称内存 / DeepEP / NIXL ...
第 0 层  库/硬件         NCCL / NVLink / IB
```

**分层的四个设计原则**（每条都能从 vLLM 代码里找到依据）：

1. **控制面与数据面分离** —— 控制面走共享内存 + ZMQ（`shm_broadcast.py`），
   数据面走 NCCL/自定义 kernel。**混在一起会引入 GPU 同步和死锁风险。**

2. **每个组双通道（CPU gloo + GPU NCCL）** —— ✅ `parallel_state.py:496-513`。
   原因：有些事 NCCL 做不了（传 Python 对象、交换 uniqueId 的鸡生蛋问题）。

3. **单卡零开销短路** —— ✅ 每个通信方法开头都 `if world_size == 1: return input_`
   （`parallel_state.py:737-739`）。

4. **不同用途的通信流隔离到不同 process group** —— ✅ vLLM 给 EPLB 单独建组
   （`parallel_state.py:2117-2120`），防止重平衡和前向互相阻塞。

**追问**：「如果要支持一个新的加速器（比如某国产 NPU），要改哪里？」
→ vLLM 的抽象点：`current_platform.get_device_communicator_cls()`
（`parallel_state.py:544-547`）+ 平台类里的 `dist_backend` / `use_custom_allreduce`。
**这就是 `GroupCoordinator` 委派给 `DeviceCommunicatorBase` 的价值。**

---

## 第五部分：追问链（面试官的真实打法）

面试官通常不会问孤立问题，而是**沿着一个点往下挖**。以下是四条真实感很强的追问链。

### 链 A：从「all-reduce 快不快」挖到「你怎么验证」

```
Q: TP 的 all-reduce 会占多少时间？
 └→ Q: 你怎么算的？（考 α-β 模型和 ring 搬运量）
     └→ Q: 那为什么 vLLM 还要自己写 custom all-reduce，NCCL 不够快吗？
         └→ Q: custom all-reduce 什么时候不生效？
             └→ Q: 如果它不生效了，你怎么发现？（考日志 + 对比实验）
                 └→ Q: 你怎么验证一次 all-reduce 的结果是对的？
                     答：「每个 rank 填自己的 rank 号，all-reduce 后应该都等于 0+1+...+N-1」
```

**最后一问的答案**（第 5 章 §5.5 的实用技巧）：
```python
t = torch.full((1024,), float(g.rank), device="cuda")
r = g.all_reduce(t)
assert torch.allclose(r, torch.full_like(r, sum(range(g.world_size))))
```

---

### 链 B：从「MoE 怎么扩展」挖到「负载不均怎么办」

```
Q: MoE 为什么能用 EP 扩展到几百卡？
 └→ Q: all-to-all 的通信量怎么算？
     └→ Q: 如果某些专家特别热门会怎样？（考 straggler）
         └→ Q: vLLM 怎么解决？（考 EPLB）
             └→ Q: EPLB 的通信和前向的通信会不会互相阻塞？
                 答：会 → 所以 vLLM 给 EPLB 单独建 process group
                     （✅ parallel_state.py:2117-2120 的注释就是答案）
```

---

### 链 C：从「CUDA Graph」挖到「NCCL 的懒初始化」

```
Q: decode 阶段为什么要用 CUDA Graph？
 └→ Q: CUDA Graph 和通信有什么冲突？
     答：① kernel 参数必须固定 → 通信 buffer 要预注册；
         ② 捕获期间不能建连/同步 → NCCL 懒初始化会炸
     └→ Q: 那 vLLM 怎么解决？（考 pynccl 的存在理由）
         └→ Q: 除了 pynccl 还有哪些后端要特殊处理 graph？
             答：custom AR 的 capture()、FlashInfer PCIe IPC 的 capture()
                 （✅ parallel_state.py:688-706 把它们都包在 graph_capture 里）
```

---

### 链 D：从「多网卡」挖到「框架集成」

```
Q: 一台机器有 8 张网卡，NCCL 会选哪张？
 └→ Q: 选错了会怎样？怎么改？（NCCL_IB_HCA / NCCL_SOCKET_IFNAME）
     └→ Q: 用 Ray 起多机，在 driver 上 export 有用吗？
         答：有用 —— vLLM 会把 NCCL_ 前缀的变量复制给 Ray worker
             （✅ vllm/ray/ray_env.py:37-44）
         └→ Q: 那为什么官方文档还强调「在集群创建时设置」？
             答：因为只在启动 worker 时复制；而且多节点要保证每个节点都设对
         └→ Q: Ray 设置的哪个环境变量会破坏 vLLM？
             答：NCCL_ASYNC_ERROR_HANDLING —— 会导致 graph building 抛异常，
                 vLLM 在 CUDA 初始化时主动 pop 掉它（✅ gpu_worker.py:360）
```

**最后这个答案是「框架集成的真实痛点」**，能说出来说明你有工程 sense。

---

## 第六部分：易错点清单（考前 10 分钟看）

| # | 常见错误说法 | 正确说法 |
|---|---|---|
| 1 | 「TP=8 的 all-reduce 每层 1 次」 | **2 次**（o_proj + down_proj，都是 row-parallel） |
| 2 | 「all-to-all 就是 all-gather」 | all-gather 输出 ×N；all-to-all 总量不变，只换分布 |
| 3 | 「NCCL 小消息用 tree」 | 准确说是「按大小和拓扑选算法，另有协议层（LL/LL128/Simple）继续调权衡」 |
| 4 | 「TP 跨机也行，带宽够就行」 | 瓶颈是**延迟 × 步数**，不是带宽 |
| 5 | 「单机多卡要开放端口」 | 默认走 **FileStore**（`file://`），不需要（✅ `multiproc_executor.py:143`） |
| 6 | 「NVLink 900 GB/s」 | 那是**双向**；单向约 450（H100）。**说清口径** |
| 7 | 「确定性只是软件没优化」 | 浮点加法不满足结合律 → **物理上互斥** |
| 8 | 「vLLM 的 all-reduce 就是调 NCCL」 | 有 **8 路后端**，按大小/硬件分档；NCCL 只是其中一路（兜底） |
| 9 | 「`VLLM_DISABLE_CUSTOM_ALL_REDUCE` 环境变量」 | **不存在**；开关是引擎参数 `disable_custom_all_reduce` |
| 10 | 「DeepEP 只有一种模式」 | HT（prefill，占 20 SM）/ LL（decode，**0 SM**）/ v2（GIN）|
| 11 | 「慢就是通信问题」 | 先用 profiler 确认时间分布 |
| 12 | 「EPLB 可以用 EP 的组做通信」 | 必须独立组，否则和前向死锁（✅ 代码注释明确说明） |

---

## 第七部分：如何展示「我真的读过源码」

面试里想体现深度，不要背结论，而是**给出「符号名 + 判据 + 数字」**。示例话术：

> ❌ 「vLLM 有一个自定义的 all-reduce，比 NCCL 快。」
>
> ✅ 「vLLM 的 `CustomAllreduce` 用 CUDA IPC 交换对端显存句柄，kernel 里直接读对端 buffer。
> 它的准入条件里有一条很有意思：`world_size > 2` 时要求 `fully_connected`（NVLink 全互联），
> 代码注释解释了原因 —— **4 卡以上如果不是全互联，custom all-reduce 相比 NCCL 收益很小**。
> 另外它的适用范围是查表的：H100 上 TP=8 只有 **256 KiB**，超过就回退 NCCL。
> 所以它不是一个『更快的 all-reduce』，而是**针对 decode 小消息的专用优化**。」

**这个回答包含：符号名（`CustomAllreduce` / `fully_connected`）、判据、具体数字（256 KiB）、
以及「为什么」**。这就是「读过源码」和「看过博客」的区别。

---

## 第八部分：进阶网络与硬件

> 这一部分**大部分标 📖**（通用网络/硬件知识），只有少数几处能在 vLLM 里找到证据。
> 但恰恰是这些 ✅ 最能体现「你不只读过博客，还读过框架怎么用硬件」。

### Q21 `[进阶]` NVLink SHARP（NVLS）和 CollNet 是什么？网内规约为什么快？

**答**：把「规约」这一步从 GPU 搬到**交换机内部**做 —— 数据在网络上「边传边算，算完广播」。

| 名称 | 归属 | 谁做规约 | 覆盖范围 |
|---|---|---|---|
| **NVLink SHARP / NVLS** | NVIDIA | **NVSwitch** ASIC | 机内 NVLink 域（H100 单机 8 卡 / GB200 NVL72） |
| **CollNet** | NCCL 的抽象层 | 由后端插件决定 | 跨机 |
| **IB SHARP** | NVIDIA Quantum 交换机 | 交换机里的 **SHARP aggregation node** | IB 网络树 |

📖 **原理（比背数字重要）**：ring all-reduce 要 `2(N-1)` 步，且中间结果必须在每个 rank 上
**落地 → 读出 → 再发出**。网内规约让中间结果**根本不落到 GPU**：
每个 rank 只把本地分片送进交换机，交换机内部做加法 + 多播回来，**步数从 `2(N-1)` 降到约 2**。

**收益结构因此是**：
- **延迟项收益巨大**：`∝ N` → `O(1)`；
- **带宽项收益有限**：每个字节仍然要进出网络一趟（搬运量还是 `≈ 2S`）。

**推论（这才是面试官想听的）**：NVLS/SHARP **对小消息、大 world size 最香**；
对大消息（本身带宽受限）提升有限。这也解释了为什么 vLLM 只在**小消息区间**用
NCCL 对称内存（在 NVSwitch 平台上底层就是 NVLS 多播）（`all_reduce_utils.py:109-118`）：

```
2K  - 16K : PyNCCL-symm 胜（1.35x - 1.48x）   ← 小消息，延迟主导 → NVLS 赢
32K - 64K : custom_AR 胜                       ← 中段，NVLS 的固定开销不划算
128K - 1G : PyNCCL-symm 胜（1.12x - 6.14x）   ← 大消息，NVLS 的带宽优势
```
（注释原文就在 `all_reduce_utils.py:94-108`，**这是「网内规约的收益结构」在 vLLM 里的实测落地**。）

✅ **vLLM 怎么探测这些能力**：它用 ctypes 声明了 NCCL 的 `ncclCommProperties`
（`pynccl_wrapper.py:66-79`），四个字段直接对应「网内计算」：

| 字段 | 含义 |
|---|---|
| `multimemSupport` | NVLink SHARP 多播（NVLS）是否可用 |
| `ginType` | GPU-Initiated Networking 类型（`0` = 不支持） |
| `nLsaTeams` | 可用 load/store 直接访问的对端组数 |
| `hostRmaSupport` | 主机侧 RMA |

**面试加分**：
> 「vLLM 不自己在用户态实现 SHARP，它只**查询**能力再据此选路径 ——
> 比如 `ginType == 0` 就直接拒绝 DeepEP v2（`all2all.py:1058-1064`）。
> 成熟框架会把『硬件能力探测』和『算法选择』分开：
> **探测是启动时一次性的，选择是每次调用按消息大小做的**。」

📖 **追问**：「那为什么确定性模式要把它们关掉？」
→ ✅ `batch_invariant.py:1148-1149` 明确设置 `NCCL_COLLNET_ENABLE=0` + `NCCL_NVLS_ENABLE=0`。
**因为网内规约的加法顺序由交换机硬件决定，软件无法控制**，而确定性要求逐 bit 可复现。
（和 Q16 是同一个知识点的两个侧面 —— 主动说出这个联系是加分项。）

---

### Q22 `[进阶]` IB SHARP 什么时候真的有效？什么时候反而是负优化？

**答**：SHARP 的收益上限由**「规约树能不能稳定建立」**决定，不是由带宽决定。

📖 **有效的场景**：

| 条件 | 为什么 |
|---|---|
| 拓扑是**规整的树 / fat-tree** | SHARP 的 aggregation node 需要一棵确定的树 |
| **world size 大**（≥ 32，跨多个 leaf） | 树深，软件 ring 的步数惩罚最大 |
| **all-reduce 是主要模式**（训练为主） | SHARP 只加速 reduce 类；all-to-all 用不上 |
| 消息**中等偏大**（MB 级） | 小消息延迟本来就小；太大受交换机 buffer 限制 |

📖 **失效 / 负优化**：

| 情况 | 原因 |
|---|---|
| **自适应路由（adaptive routing）开着** | 包走不同路径 → 树结构被打破，SHARP 退化为软件 |
| 非 2 的幂 / 非对称 rank 布局 | aggregation 分组不均衡 |
| **all-to-all / all-gather 主导**（MoE 推理） | SHARP 只做 reduce，对这些模式毫无帮助 |
| 多租户共享 fabric | SHARP 的资源（aggregation node、buffer）是独占的 |
| 需要**确定性** | 顺序由硬件决定（见 Q21 追问）|

📖 **关键工程细节**：
- 需要 `NCCL_COLLNET_ENABLE=1`，并加载 SHARP 插件（`libnccl-net.so` 走 `sharp`）；
- 需要 subnet manager 配置 SHARP 资源（`sharp_cmd` / `sharpd`），**不是插上就能用**；
- 报错通常不是「不支持」，而是「拿不到 SHARP 资源」→ 静默退化到 ring，**性能悄悄掉**。

**面试加分**：
> 「SHARP 的坑在于**静默降级**：拓扑/路由一改，它就不生效了，但日志里不报错。
> 所以验证 SHARP 生效不能看『有没有报错』，要看 **IB 交换机的 SHARP 计数器**
> 或者对比 `NCCL_COLLNET_ENABLE=0/1` 的实测带宽。
> 这和 vLLM 里『不要静默降级，宁可快速失败』（`all2all.py:1053-1064` 直接 raise）
> 是同一种工程审美的两面。」

---

### Q23 `[进阶]` PCIe P2P、ACS、IOMMU、BAR1 怎么影响多卡通信？

**答**：这是**「为什么 custom all-reduce 在有些机器上不生效」**的硬件根源。

📖 **四个概念**：

| 机制 | 作用 | 对通信的影响 |
|---|---|---|
| **PCIe P2P** | 一张卡直接读写另一张卡的显存，**不经过 CPU/内存** | 关闭时 GPU↔GPU 必须走 host bounce buffer，带宽掉一个数量级 |
| **ACS**（Access Control Services） | PCIe 交换机上的访问控制；**默认会把 P2P 流量强制上送到 root complex** | ACS 开着 → P2P 被降级成「经 CPU 绕一圈」 |
| **IOMMU** | DMA 地址翻译与隔离 | 开着时 P2P 需要正确的 DMA 映射；配错则 P2P 直接失败或被禁用 |
| **BAR1** | GPU 通过 PCIe 暴露给 Host 的显存窗口 | 窗口太小 → 大 buffer 无法映射给对端，IPC/P2P 受限 |

📖 **BAR1 的量级（记住这两个数就够）**：

| GPU 类型 | BAR1 |
|---|---|
| 数据中心卡 A100/H100（ReBAR 生效） | **64 GiB 级** |
| 消费级卡（默认 BAR 限制） | **256 MiB** |

→ 所以「同样写 NVLink P2P 代码，A100 上能跑、4090 上 4 卡就挂」往往是 BAR1 或 ACS 的问题，
不是代码问题。查法：`nvidia-smi -q | grep -A4 BAR1`；Linux 上看 `/sys/bus/pci/devices/*/resource`。

✅ **vLLM 怎么应对**：它**不假设 P2P 可用**，而是**实测**。
`all_reduce_utils.py:245` 有 `can_actually_p2p(from_gpu, to_gpu)` ——
**真的做一次 P2P 拷贝并校验结果**，而不是相信 `nvidia-smi topo -m` 的 `NV`/`PIX` 标记。

**面试加分**：
> 「`nvidia-smi topo -m` 报的是**硬件能力**，不是**当前可用性**：
> ACS/IOMMU/驱动/容器权限都可能让 P2P 实际不可用。
> vLLM 的做法是 `can_actually_p2p()` —— 写一个魔数、通过 P2P 读回来、校验。
> **能力探测用实测而不是读表**，这个模式值得在别的地方复用。」

📖 **追问**：「容器里为什么 P2P 经常坏？」
→ 需要 `--gpus all` + `NVIDIA_DRIVER_CAPABILITIES=compute,utility`，
且宿主机 IOMMU 组划分要允许同组内 P2P；某些云厂商直接禁掉 ACS 的修改权限。
**这是「同一份代码在 A 集群快、B 集群慢」的常见原因。**

---

### Q24 `[进阶]` rail-optimized 拓扑是什么？GPUDirect RDMA 和 GDRCopy 有什么区别？

**答**：两者一个是**网络拓扑设计**，一个是**数据路径优化**，经常被混着问。

📖 **rail-optimized（轨道对齐）拓扑**：

```
传统：每台机器 8 卡 → 2 张网卡（所有卡共享），跨机流量在网卡上排队
Rail：每台机器 8 卡 → 8 张网卡，GPU i 固定接 NIC i
      所有机器的「第 i 号 GPU + 第 i 号 NIC」构成一条 rail，接同一组 leaf 交换机
```

| 好处 | 说明 |
|---|---|
| **无阻塞** | 每个 GPU 有独立上行，不会互相排队 |
| **路径唯一** | 同 rail 内一跳到达 → 延迟低且可预测 |
| **可预测的 ring** | NCCL 能构造「同号卡通信」的 ring，避免跨 rail 绕行 |

**代价**：需要 `N` 倍的 leaf 交换机端口与光模块，成本高；跨 rail 通信要过 spine，容易拥塞。

✅ **vLLM 的对应物**：它让用户把「GPU PCIe 地址 ↔ NIC PCIe 地址」显式映射出来，
再据此给每个 worker 设置 `NCCL_IB_HCA` / `UCX_NET_DEVICES`
（`vllm_net_devices.py:171-181`：`os.environ[var_name] = value`；
变量名单由 `VLLM_NIC_SELECTION_VARS` 指定，`vllm_net_devices.py:10-12`）。

📖 **GPUDirect RDMA vs GDRCopy**：

| | GPUDirect RDMA (GDR) | GDRCopy |
|---|---|---|
| 是什么 | **网卡直接 DMA 读写 GPU 显存** | 一张 **GPU 显存 → CPU 可访问** 的低延迟映射 |
| 解决什么 | 省掉「显存→主机内存→网卡」的两次拷贝 | 省掉 `cudaMemcpy` 的中转，让 CPU 能直接读 GPU 的一小块内存 |
| 典型用途 | NCCL/DeepEP 的大块数据传输 | **小控制消息**（flag、计数器、doorbell） |
| 延迟量级 | μs（网络主导） | **亚 μs**（PCIe MMIO 读） |

📖 **为什么两者都需要**：大块数据用 GDR 省带宽，小控制消息用 GDRCopy 省延迟 ——
**用一个 4 KB 的 `cudaMemcpy` 去读一个 flag，开销比 flag 本身大三个数量级**。

📖 **追问**：「GPU 怎么知道『网卡已经收到了』？」
→ 传统做法是 CPU 轮询/中断 → 再 `cudaMemcpy` 到 GPU → 再同步。
现代做法（**IBGDA / GPU-Initiated Networking**）是 **GPU kernel 内直接读写网卡的
doorbell 和 completion queue**，完全不要 CPU 参与 ——
这正是 vLLM 检查 `ginType`（`pynccl_wrapper.py:77`、`utils/nccl.py:86-129`）
并拒绝 `ginType == 0` 的 DeepEP v2 的原因。

---

### Q25 `[系统]` 为什么 RoCE 需要 PFC/ECN/DCQCN，而 IB 不需要？配错会怎样？

**答**：因为**丢包对 RDMA 是灾难**，而 IB 从协议层就不丢包，RoCE 跑在以太网上必须先「造出无损」。

📖 **演进链条（按顺序讲，逻辑最清楚）**：

```
RDMA 的假设：网络不丢包
   ├─ IB：credit-based 链路层流控 → 从根上不丢包 → 不需要额外配置
   └─ RoCEv2：跑在以太网上，以太网会丢包 → 必须人工造出无损
        ├─ 第一层：PFC（Priority Flow Control，L2 逐跳反压）
        │     给 RDMA 流量打一个无损优先级（通常 3），队列快满时发 PAUSE
        ├─ 第二层：ECN（L3 端到端标记）—— 交换机在拥塞时给包打 CE 标记
        └─ 第三层：DCQCN（端侧反应）—— 收到 CE 标记后降速（类似 TCP 的 AIMD，但是硬件实现）
```

📖 **为什么单靠 PFC 不行**：
- PFC 是**逐跳反压**，会引发**队头阻塞**和**PFC 风暴**（pause 帧在全网传播）；
- PFC 是**死锁温床**：环形依赖 + 全队列 pause → 整个 fabric 卡死；
- 所以必须配 ECN/DCQCN，让**端侧先主动降速**，PFC 只作为最后一道保险。

📖 **参数形状（记住「有哪些旋钮」比记住数值重要）**：

| 参数 | 形态 | 位置 |
|---|---|---|
| PFC 优先级 | 给 RDMA 流量指定一个**无损优先级**（常见约定是用 3 号） | 交换机 + 网卡 |
| ECN 标记阈值 | 三个水位 `Kmin / Kmax / Pmax`：低于 Kmin 不标记；Kmin–Kmax 之间按概率标记；高于 Kmax 全标记 | 交换机 |
| DCQCN 速率下降/恢复 | 由网卡固件控制（收到 CE 标记就降速，定时器到期再试探性恢复） | 网卡 |

⚠️ **具体数值强依赖交换机和网卡型号，不能背 —— 面试时说「要按厂商推荐值配 + 用 ECN 标记计数验证」更稳。**

📖 **配错的典型症状**（这部分最有面试价值）：

| 症状 | 原因 |
|---|---|
| 吞吐忽高忽低、P99 抖动大 | ECN 阈值太激进 → DCQCN 反复降速恢复 |
| 「PFC deadlock」，全网卡死 | PFC 配了但 ECN 没配，pause 帧循环 |
| 部分节点快、部分节点慢 | 个别交换机没打开无损优先级 |
| 带宽只有理论值一半 | PFC/ECN 没生效，RDMA 在丢包重传 |

**面试加分**：
> 「一句话记：**IB 的无损是『协议自带的』，RoCE 的无损是『运维配出来的』**。
> 所以 RoCE 集群的性能问题，一半在网卡/交换机配置，不在模型代码。
> 排查时先看 **PFC pause 帧计数**和 **ECN 标记计数**，再看 NCCL。」

---

### Q26 `[基础]` DAC、AOC 和光模块怎么选？线缆会不会成为瓶颈？

**答**：按**距离 + 成本 + 功耗**三选二。

📖 **对比表**：

| 类型 | 距离 | 成本 | 功耗/端口 | 延迟 | 场景 |
|---|---|---|---|---|---|
| **DAC**（无源铜缆） | ≤ 2–3 m | 最低 | ~0 W | 最低 | 机内、同机柜 ToR |
| **ACC/AEC**（有源铜缆） | ≤ 5–7 m | 低 | 低 | 低 | 相邻机柜 |
| **AOC**（有源光缆） | ≤ 30 m | 中 | 中 | 略高 | 同排机柜 |
| **光模块 + 光纤**（SR/DR/FR） | 100 m – 2 km | 最高 | 10–15 W/端 | 略高 | 跨排、跨机房 |

📖 **关键取舍**：
- **DAC 的隐藏成本是「走线」**：粗、硬、不能弯折，800G DAC 的线径和弯曲半径很夸张，
  机柜内布线很快就放不下 —— 此时被迫上 AOC/光模块；
- **光模块的隐藏成本是「功耗和故障率」**：一个 800G 光模块 10–15 W，
  一台 64 端口的交换机光模块功耗就接近 1 kW；光模块也是整网故障率最高的部件；
- **延迟差异在 AI 网络里通常可忽略**（都是 ns 级），**别用延迟当理由选线**，
  要谈就谈**布线半径、功耗、故障率、成本**。

📖 **面试里怎么用**：这题不是考点，是**「你有没有见过真机柜」**的探针。
一句话答到点上即可：
> 「距离决定物理层，功耗和布线半径决定真实成本；延迟在这里不是选型依据。」

---

### Q27 `[进阶]` 什么是 GPU-Initiated Networking（GIN）？它改变了什么？

**答**：让 **GPU kernel 内部直接发起网络操作**，把「通信」从「CPU 下发的命令」变成「GPU 的访存」。

📖 **传统路径 vs GIN**：

```
传统（CPU-initiated）
  GPU 算完 → 写 flag → CPU 轮询到 → CPU 下发 WQE 到网卡 → 网卡 DMA 发送
  代价：一次 CPU 往返（μs 级）+ 必须把控制流从 GPU 交回 CPU

GIN / IBGDA（GPU-initiated）
  GPU kernel 内直接写网卡 doorbell + 轮询 completion queue
  代价：0 次 CPU 往返；通信像访存一样可以被 kernel 交错调度
```

✅ **vLLM 的三处证据**（可以串起来讲，很能体现「读过源码」）：

1. **探测**：`utils/nccl.py:86-129` 的 `query_nccl_gin_type()` ——
   通过 `ncclCommQueryProperties` 拿到 `props.ginType`，
   并且**要求传入已初始化的 comm 指针**（注释：*"GIN is a property of this
   initialized communicator, not just the NCCL version."*）。
2. **准入**：`all2all.py:1058-1064` —— `gin_type == 0` 直接 raise，
   而不是降级（理由：v2 的性能完全依赖 GIN，降级后不如用 v1 LL）。
3. **强制初始化**：`all2all.py:1046-1050` —— 先做一次 dummy all-reduce，
   因为 `ProcessGroupNCCL` **懒创建** communicator，
   不先初始化就会拿到空指针、误判为「不支持 GIN」。

📖 **为什么这件事和校招有关**：它是过去五年 AI 网络最重要的范式转变 ——
从「**通信是 CPU 的事**」变成「**通信是 GPU kernel 的一部分**」。
一旦通信在 kernel 里发起，就可以：
- 和计算**细粒度交错**（不是粗粒度 overlap）；
- 用 **programmatic dependent launch / warp specialization** 做更激进的流水；
- 让 compiler/DSL（如 DeepGEMM、CUTLASS）**把通信当成一个 tile op** 来调度。

**面试加分**：
> 「vLLM 对 GIN 的处理体现了两个判断：
> ① **能力探测必须绑定到具体 communicator**，不能只看版本号；
> ② **不支持时快速失败，不静默降级** —— 因为降级路径比次优更糟（它更慢且更难 debug）。」

---

### Q28 `[系统]` 设计：给你一台 8×H100 + 8×400G NIC 的机器，怎么规划通信？

**答题框架（先问清楚场景，再给方案，最后给验证方法）**：

**第 0 步：问清楚 3 件事**
```
① 模型是 dense 还是 MoE？（决定 TP vs EP）
② 部署规模是单机还是多机？（决定要不要跨 IB）
③ 是训练、prefill 还是 decode 为主？（决定延迟 vs 带宽优先）
```

**第 1 步：机内**
```
8×H100 NVLink/NVSwitch 全互联 → TP ≤ 8 无争议
→ 开 custom all-reduce（条件：world_size ∈ {2,4,6,8,16} 且 NVLink 全互联）
→ 消息 16 KiB（decode）落在 custom AR 的最优区间
```

**第 2 步：机间（rail-optimized）**
```
GPU i ↔ NIC i，同 rail 内一跳到达
部署时用 VLLM_GPU_NIC_PCIE_MAPPING 把 (GPU PCI, NIC PCI) 显式绑定，
再让 VLLM_NIC_SELECTION_VARS 写 NCCL_IB_HCA / UCX_NET_DEVICES
（✅ vllm_net_devices.py:10-12, :171-181）
```

**第 3 步：把「什么通信走哪里」写成表**（这就是这题的核心答案）

| 通信 | 走哪里 | 为什么 |
|---|---|---|
| TP all-reduce | **NVLink（custom AR / NVLS）** | 每层 2 次、延迟敏感，绝不能跨机 |
| EP dispatch/combine | **IB（DeepEP LL，decode）** | 只发给持有专家的 rank，可跨机，且 LL **0 SM** |
| EP dispatch/combine | **NVLink（DeepEP HT 机内）** | prefill 通信量大，用 SM 搬运换带宽 |
| PP activation | **IB（点对点，量小）** | 只有 stage 边界，频率低 |
| KV 传输（PD 分离） | **IB（NIXL/Mooncake）** | 一次性大块；和计算不重叠也没关系 |
| 控制面 | **shm + ZMQ / TCP** | 不走 NCCL（避免 GPU 同步和死锁）|

**第 4 步：必须实测的 4 件事**（说出验证方法比说出方案更值钱）
```
① 启动日志里 custom AR / pynccl 是否真的启用
② NCCL_DEBUG=INFO 看选了哪个 HCA、哪个算法/协议
③ nsys 看通信 kernel 占总时间比例（decode 应 < 30%）
④ 单点带宽：nccl-tests all_reduce / alltoall 打满看是否接近线速
```

**追问**：「如果只有 2 张 IB 网卡而不是 8 张？」
→ 那就没有 rail 的优势：跨机流量要在 2 张卡上排队。
此时应该**减少跨机通信**：把 EP 组尽量放在机内（`--data-parallel-size-local` 调大），
或者用 PP 跨机（activation 量远小于 all-to-all）。

---

## 第九部分：MoE 通信与 EP 进阶（✅ 为主）

> 这一部分是**新的技术深水区**。vLLM 最近把 MoE 的通信拆成了
> `prepare_finalize/*` 一组实现，代码非常新，**面试时能说出文件名就是强信号**。

### Q29 `[基础]` 为什么 MoE 需要 all-to-all，而不是 all-reduce / all-gather？

**答**：因为 MoE 的通信本质是**「把 token 路由到持有它专家的卡上」**，
是**重新分布**而不是**聚合**。

| 操作 | 数据量 | MoE 里为什么不行 |
|---|---|---|
| all-gather | 每卡 ×N | 每卡都要看到所有 token，但 token 只需要去 topk 个专家 |
| all-reduce | 每卡 ≈2S | reduce 语义是「求和」，但 token 是**互不相同**的，不能相加 |
| **all-to-all** | **总量不变** | 每个 rank 只把自己那份发给需要它的 rank，语义精确匹配 |

📖 **一句话金句**：
> 「MoE 用 all-to-all 而不是 all-gather，因为 token 只需要送到**持有它选中专家**的 rank，
> 不需要所有人都看到所有 token。all-gather 是 `O(N)` 的放大，all-to-all 是**守恒**的。」

✅ **vLLM 里的两种实现路径**（这题的最佳加分点）：

| 后端 | dispatch 实现 | combine 实现 | 代码 |
|---|---|---|---|
| `allgather_reducescatter`（默认） | all-gather | reduce-scatter | `all2all.py:101-136`（dispatch）/ `:138-150`（combine）|
| DeepEP / NIXL EP / MoRI | 真正的 all-to-all kernel | 真正的 all-to-all kernel | `all2all.py:210-377` 等 |

**注意默认后端的名字**：`allgather_reducescatter` 说明
**vLLM 在「没有专用 a2a kernel」时的原始做法就是「AG + RS 拼一个 all-to-all」**：
- dispatch = all-gatherv（每个 rank 拿到**所有人**的 token）
- combine = reduce-scatter（把结果切片分回各自 rank）

这在 `ep_size` 小的时候是对的（NVLink 带宽便宜），但 **`ep_size` 变大后会 `O(N)` 爆炸** ——
这就是"为什么需要 DeepEP"的定量理由。

**追问**：「那为什么 `allgather_reducescatter` 还能当默认？」
→ 因为它**零依赖**（只用 NCCL），而且在小规模下 NVLink 带宽足够。
**「默认选项要保证零依赖可用，最优选项交给用户显式开启」** 是框架设计的通用原则
（和 Q14 的「不做自动切换」是同一个道理）。

---

### Q30 `[进阶]` 画出 MoE dispatch / combine 的完整数据流（含 permute/unpermute）。

**答**：一句话 —— **「dispatch 前 permute（按专家重排），combine 后 unpermute（还原 token 顺序）」**。

✅ **完整链路**（以 DeepEP HT 为例，`deepep_ht.py`）：

```
① 输入 a1: [T, H]（本 rank 的 token）+ topk_ids/topk_weights: [T, K]
   ↓ moe_kernel_quantize_input（若 block-quant）           deepep_ht.py:293-311
② get_dispatch_layout(topk_idx, num_experts)
   → num_tokens_per_rank / num_tokens_per_rdma_rank /
     dispatch_expert_num_tokens / is_token_in_rank         deepep_ht.py:136-142
   ★ 这一步是「metadata 阶段」：算出每个对端要收多少 —— 见 Q31
③ buffer.dispatch(...)                                      deepep_ht.py:155-171
   → token_data / expert_topk_ids / expert_topk_weights /
     expert_num_tokens_per_expert_list / handle
   ★ 网络里传的是「按目标专家拼好的 token 块」（已经 permute 过）
④ _receiver: event.current_stream_wait() 等通信完成         deepep_ht.py:207-208
   → 把 local expert id 加偏移转回 global id                  deepep_ht.py:226-231
   → ExpertTokensMetadata.make_from_list（每专家 token 数，GPU-CPU 拷贝）:236-238
   → 量化（如果不是 block-quant，dispatch 后再量化）           :244-256
⑤ 专家计算（GEMM + activation，输入已是 [num_local_experts, ...] 布局）
⑥ _finalize:
   → TopKWeightAndReduceContiguous 做加权求和               deepep_ht.py:364-372
   → buffer.combine(...)（unpermute + 回传）                 :378-387
   → output.copy_(combined_x)                               :399
```

✅ **两个关键事实**：
1. **HT 的 combine 只支持 bf16**：`deepep_ht.py:375-377` 有 assert
   `fused_expert_output.dtype == torch.bfloat16`；
   代码注释直接写 *"HT combine only supports BF16"*（`:379`）。
   → **所以「专家计算输出能不能保持低精度」不是你能选的，是后端决定的。**
2. **可能有专家收到 0 个 token**：`deepep_ht.py:361-363` 的注释
   *"fused_expert_output can have 0 tokens - This happens when none of the tokens
   from the all2all reach this EP rank."* → 代码用 `numel() != 0` 守卫。
   **面试时说出「空专家是正常情况而不是 bug」很加分。**

📖 **permute/unpermute 的代价**：dispatch/combine 本质都是 **gather/scatter + 重排**，
所以真正的通信量 = `tokens × H × dtype_size`，但 **GPU 侧的 index 计算和 copy 也不便宜**，
这就是为什么有 `expert_alignment`（`deepep_ht.py:164-166` 设为 1）和
`FusedMoEExpertsPermuteUnpermute` 这类「把 permute 融进 GEMM」的 kernel。

**追问**：「为什么 dispatch 之前要 quantize，而不是之后？」（深入题）
→ `deepep_ht.py:288-292` 的注释给了答案：
*"DeepEP only supports fp8 block scales so quantize before the dispatch for these models."*
**因为网络带宽是瓶颈：先量化，网络上传的就是 fp8 而不是 bf16，通信量直接减半。**
反之如果是 per-tensor 量化，就 dispatch 之后再量化（省得传 scale）。

---

### Q31 `[进阶]` 变长的 all-to-all 怎么处理？sizes / metadata 怎么传？

**答**：MoE 的 all-to-all 里，**每个 rank 发给每个对端的数据量都不一样**，
而 NCCL 没有原生变长 all-to-all。vLLM 有两套解法。

**解法 A：旁路传 sizes + N 次 broadcast 拼 all-gatherv**（默认后端）

✅ `all2all.py:60-68` 的 `_get_sizes`：
```python
def _get_sizes(self, num_local_tokens, comm_group):
    if self.dp_world_size == 1:
        return [num_local_tokens] * comm_group.world_size
    dp_metadata = get_forward_context().dp_metadata
    sizes = dp_metadata.get_chunk_sizes_across_dp_rank()   # ← 从 DP metadata 里读！
    return sizes
```
→ **sizes 不在通信内部交换**，而是**之前已经在 DP 组的 CPU all-reduce 里汇总过**
（`DPMetadata.num_tokens_across_dp_cpu`）。**省掉一次额外的往返**。

✅ 然后 `GroupCoordinator.all_gatherv` → `PyNcclCommunicator.all_gatherv`（`pynccl.py:293-326`）：
```python
self.nccl.ncclGroupStart()
for root, split_size in enumerate(sizes):
    dst_slice = output_tensor[split_offset : split_offset + split_size]
    self.nccl.ncclBroadcast(..., root, self.comm, ...)   # 每次 root 不同
    split_offset += split_size
self.nccl.ncclGroupEnd()
```
**「N 次 root 不同的 broadcast = 一次 all-gatherv」** —— 这是个很漂亮的 trick，
而 `ncclGroupStart/End` 的作用是把 N 个操作**并发下发**，避免逐个同步。

✅ 对称地，`reduce_scatterv`（`pynccl.py:356-391`）用 **N 次 root 不同的 `ncclReduce`** 拼出来。

**解法 B：把 metadata 塞进 kernel**（DeepEP 系）

✅ DeepEP 用 `get_dispatch_layout` 一次性算出所有需要的信息，随 dispatch 一起走：
```python
(num_tokens_per_rank, num_tokens_per_rdma_rank,
 dispatch_expert_num_tokens, is_token_in_rank, event) = \
    self.buffer.get_dispatch_layout(topk_idx=..., num_experts=..., ...)
```
（`deepep_ht.py:136-142`）

| | 解法 A（AG/RS） | 解法 B（DeepEP） |
|---|---|---|
| sizes 从哪来 | **外部（DP metadata）** | **kernel 内部算** |
| 元数据往返 | 0（复用已有） | 0（融合进 kernel） |
| 灵活性 | 只有 all-gather/reduce 语义 | 真 all-to-all，可跨机 |
| 依赖 | 纯 NCCL | 需要 DeepEP 内核 |

**面试加分**：
> 「变长集合通信的通用套路只有两种：**① 先传 metadata 再传数据**（多一次往返，但简单）；
> **② 把 metadata 融合进 kernel**（零额外往返，但需要专用内核）。
> vLLM 在默认后端起见了第一种的变体 —— **metadata 从已有的 DP 通信里『蹭』出来**，
> 所以连额外往返都没有。这是**「复用已有通信通道」**的典型优化。」

---

### Q32 `[系统]` expert id 有哪几种坐标？`-1` 是什么意思？

**答**：这是 MoE 里**最容易出 bug 的地方**，vLLM 里有**三套坐标系**。

✅ **三种坐标系**（`expert_map_manager.py:325-334` 的 docstring 直接给出名字）：

| 坐标 | 范围 | 谁在用 |
|---|---|---|
| **global**（全局专家 id） | `0 .. global_num_experts-1` | 路由输出、checkpoint、用户可见的一切 |
| **local**（本 rank 第几个专家） | `0 .. local_num_experts-1` | 本 rank 的权重张量下标 |
| **physical**（物理槽位） | `0 .. num_local_experts-1` | EPLB 复制后，一个 logical 专家可能有多个 physical 副本 |

✅ `expert_map_manager.py:503-516` 给出了 global ↔ physical ↔ local 的构造：
```python
owner = torch.remainder(global_indices, self.ep_size)      # 哪个 rank 拥有
local_index = torch.div(global_indices, self.ep_size, rounding_mode="floor")
global_to_physical = physical_offset + local_index
physical_to_global[global_to_physical] = global_indices
local_global = torch.arange(self.ep_rank, self.global_num_experts, self.ep_size)
```
→ **注意 `owner = global_id % ep_size` 时是 round-robin；`expert_placement_strategy`
可以选 `"linear"`（连续切分）或 `"round_robin"`（`config/parallel.py:187-196`）。**

✅ **`-1` 的两种含义**（**这是本题的核心区分点**）：

| 出现位置 | `-1` 的含义 | 证据 |
|---|---|---|
| `expert_map[global_id] = -1` | **「这个专家不在本 rank 上」** | `expert_map_manager.py:301-303`|
| `recv_topk_idx == -1` | **「这个槽位不是本地专家 / 是 padding 行」** | `deepep_v2.py:294-310` |

✅ **两个必须知道的处理细节**：

1. **HT 路径要把 `-1` 换成「一个肯定不会命中本 rank 的专家」**，而不是留给 MoE kernel 报错：
   `deepep_ht.py:216-231` 的注释解释了为什么及其取法：
   > *"The existing MOE kernels assume that all entries of topk_ids are valid. To that effect,
   > set the -1s in expert_topk_ids to some expert outside this rank so the expert_map can
   > remap it to -1 when safe. ... For rank 0, set it to num_experts - 1 and for all other
   > ranks set it to 0."*

   **答题时说出「为什么 rank 0 特殊」是强信号**：
   因为 rank 0 持有 `[0, k)` 的专家，所以要用**最后一个**专家 `num_experts-1` 把它挤出本地范围；
   其他 rank 都持有 0 号专家，所以用 `0`。

2. **decode/CUDA Graph 路径下，越界行是「未初始化」的脏数据**，必须显式置 `-1`：
   `deepep_v2.py:294-310` 的注释说明 —— dispatch 只写了 `[0, num_recv_tokens)` 行，
   其余是 worst-case 分配的未初始化内存，**里面的陈旧数据可能恰好是合法的专家 id**，
   会被 triton MoE 的 `make_routing_data` 当成真 token 处理，**污染真实 token**。
   所以有一个专门的 Triton kernel `_globalize_recv_topk_idx_kernel`（`deepep_v2.py:518-540`）
   在 device 上完成转换（**不 sync host**，保证 CUDA Graph 可捕获）。

**追问**：「为什么 LL 路径要做 `global_to_physical` 映射？」
→ `deepep_ll.py:291`：`dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)`，
combine 时也要映射一次（`:396`）。**因为 DeepEP LL 的 kernel 只认「本地物理槽位」**，
它不知道 global 专家编号，也不知道 EPLB 的复制关系。**框架层负责翻译坐标。**

---

### Q33 `[系统]` MoE 负载不均（straggler）怎么解决？EPLB 的原理和工程约束。

**答**：EPLB = **Expert Parallel Load Balancer**，用「复制热门专家 + 重排物理槽位」把负载摊平。

✅ **算法来源**：`distributed/eplb/policy/default.py:8-13` 的 docstring 明确写
*"The rearrangement algorithm is adapted from [DeepSeek EPLB]"*。

✅ **两个核心步骤**（都在 `policy/default.py`）：

| 步骤 | 函数 | 做什么 |
|---|---|---|
| **1. 复制** | `replicate_experts(weight, num_phy)`（`:75-90`）| 把 `num_log` 个逻辑专家复制成 `num_phy` 个物理副本，**最小化最大副本负载** |
| **2. 装箱** | `balanced_packing(weight, num_packs)`（`:22-73`）| 把专家按权重「贪心装进最轻的包」，每个包内数量相等 |

`balanced_packing` 的关键思路（`:47-71`）：**按权重降序排序，每个专家放进当前最轻的包**，
包满（`items == groups_per_pack`）后把权重设为 `inf` 屏蔽掉 ——
**经典的 LPT（Longest Processing Time）贪心**，近似比 `4/3 - 1/(3m)`。

✅ **执行阶段会真的搬权重**：`distributed/eplb/rebalance_execute.py:1-7` 的头注释：
> *"The actual execution of the rearrangement. This involves the exchange of expert weights between GPUs."*

→ **EPLB 不是零成本的**：每次重平衡都要在 EP 组里**搬几十 GB 的专家权重**，
所以是**周期性**触发而不是每步做。几个关键旋钮（`EPLBConfig`）：

| 字段 | 默认 | 含义 |
|---|---|---|
| `window_size` | **1000** | 专家负载记录的窗口（`config/parallel.py:62-63`）|
| `step_interval` | **3000** | **多少步重排一次专家**（`:64-70`）|
| `num_redundant_experts` | **0** | 冗余专家数 —— ⚠️ **为 0 时没有副本可换，复制机制形同虚设**（`:72-73`）|
| `use_async` | **True** | 非阻塞 EPLB（`:84-87`）|
| `communicator` | `None` | 权重搬运后端：`nixl` / `torch_gloo` / `torch_nccl` / `pynccl`；`None` 时优先 nixl、退回 gloo（`:92-99`）|

⚠️ **两个校验**（`:102-107`）：`use_async=True` 时**不能用非 default 的 policy**，
也**不能用 `torch_nccl` / `pynccl` 做 communicator** ——
因为异步搬运必须走一条**不会和前向 collective 抢同一个 communicator** 的通道
（和下面「EPLB 要独立 process group」是同一个约束的两个层次）。

✅ **最关键的一条工程约束（和 Q20 呼应）**：
`parallel_state.py:2117-2120` 的注释原文：
> *"Create EPLB group with the same ranks as EP if EPLB is enabled.
> This is a separate process group to isolate EPLB communications from MoE forward
> pass collectives and **prevent deadlocks when using torch.distributed in execution
> with torch.distributed in EPLB**."*

**为什么必须独立 group**：EPLB 在一个**后台线程**里跑（`eplb/async_worker.py`），
如果和前向共用同一个 NCCL communicator，两个线程的执行顺序不确定
→ **collective 顺序错乱 → 死锁**。**「不同用途的通信流隔离到不同 process group」**
是分布式编程的铁律。

**追问**：「EPLB 会带来什么副作用？」
→ ① **权重搬运本身占带宽**（和 KV 传输抢 IB）；
② **物理槽位变化** → 所有 `global_to_physical` 映射都要更新，
这就是为什么 `deepep_ll.py`/`nixl_ep.py` 要把映射表当参数传进来（`deepep_ll.py:91-93`）；
③ **和前向的 kernel 抢占** → 所以要靠在 `rebalance_execute.py` 里用 `CpuGpuEvent`
做**中间 buffer 的握手**（`:56-63` 的注释：async worker 完成后 `wait()`，
主线程搬出后 `record()`）—— 双缓冲，不阻塞前向。

---

### Q34 `[系统]` DeepEP HT、LL、v2(GIN) 到底怎么选？给一个决策表。

**答**：按「**阶段 × 是否需要 CUDA Graph × 网络能力**」三问决定。

✅ **完整对比（全部有代码依据）**：

| | **HT**（High Throughput） | **LL**（Low Latency） | **v2**（ElasticBuffer/GIN） |
|---|---|---|---|
| 构造参数 | `low_latency_mode=False`（`all2all.py:239`）| `low_latency_mode=True`（`:321`）| 无此参数 |
| 目标阶段 | **prefill** | **decode** | 两者统一 |
| **SM 占用** | **20**（`all2all.py:171`）| **0**（`all2all.py:345-347`）| 动态：`get_theoretical_num_sms`（`:1078-1082`）|
| qps/rank | `num_sms // 2 = 10`（`all2all.py:226`）| `num_local_experts`（`all2all.py:306`）| 自动 |
| CUDA Graph | 不支持 | **支持**（`return_recv_hook`）| **decode 模式支持** |
| RDMA buffer | 固定 1 GiB（`VLLM_DEEPEP_BUFFER_SIZE_MB`）| 按需计算（`get_low_latency_rdma_size_hint`，`all2all.py:307-312`）| 自动 |
| hidden size | **round up 到 512 字节对齐** | **必须在上表 8 个值里** | 同 HT |
| 依赖 | DeepEP | DeepEP + RDMA | DeepEP ≥ 2.0 + **NCCL ≥ 2.30.4 + IBGDA 网卡** |
| 激活布局 | `Standard`（`deepep_ht.py:87-88`）| `BatchedExperts`（`deepep_ll.py:147`）| `Standard`（`deepep_v2.py:152`）|

**决策流程**：
```
① 有 IBGDA 网卡 + NCCL ≥ 2.30.4 ？
   ├─ 是 → 想要统一 prefill/decode 代码路径 → deepep_v2
   └─ 否 → 继续
② 主要跑 decode（在线服务）？
   ├─ 是 → deepep_low_latency（0 SM + 支持 CUDA Graph + 要开 DBO 时必需）
   └─ 否 → deepep_high_throughput（prefill 通信量大，用 SM 换带宽）
③ 要开 DBO 做通算重叠？
   → 只能 deepep_low_latency 或 deepep_high_throughput
     （✅ docs/design/dbo.md:37 "Currently, DBO is only supported with DeepEP"）
```

✅ **vLLM 不做自动选择的理由**：后端选择是**启动参数**（`--all2all-backend`），
if/elif 链在 `cuda_communicator.py:163-225`，**没有 fallback 逻辑**。
因为真实部署常常同时服务 prefill 和 decode（chunked prefill），
**最优后端取决于负载比例**，让用户按场景显式选比自动选错更安全。

**追问**：「为什么 HT 反而要占 20 个 SM，不是越少越好吗？」
→ 因为 **prefill 的通信量大**：HT 用 SM 做数据搬运（copy engine 式），能打满带宽；
此时**计算本来就是瓶颈，牺牲 SM 可接受**。
而 decode 每步只有几个 token，通信是纯延迟开销，**必须用 RDMA 硬件（0 SM）**，
让通信和计算在 SM 层面**物理并行**而不是时间片轮转。

---

### Q35 `[进阶]` NIXL EP、MoRI、FlashInfer 三个后端各自的取舍是什么？

**答**：它们解决的是**DeepEP 覆盖不到的场景**：弹性扩缩、AMD 平台、MNNVL 域。

✅ **三个后端的定位**（读 `all2all.py` 的类注释就能拿到）：

| 后端 | 类 | 定位（代码注释原文提炼）| 平台 |
|---|---|---|---|
| **NIXL EP** | `NixlEPAll2AllManager`（`all2all.py:384-388`）| *"supports elastic EP with dynamic rank connection/disconnection"* | CUDA |
| **MoRI** | `MoriAll2AllManager`（`all2all.py:903-912`）| ROCm 上的 EP dispatch/combine | **仅 gfx942 / gfx950** |
| **FlashInfer NVLink** | `FlashInferNVLinkTwoSidedManager`（`:594`）/ `OneSidedManager`（`:701`）| MNNVL（跨节点 NVLink）域的 a2a | GB200 类 |

✅ **NIXL EP 的独特能力 —— 真正的弹性**（这是它唯一不可替代的点）：
```python
_connect_to_ep_size(ep_size, make_active)      # all2all.py:442-453
_disconnect_to_ep_size(ep_size)                # :455-465
_stage_ep_size()                               # :479-487  先连上但保持 masked
commit_staged_state()                          # :489-501  一次性切换
```
- **`stage` / `commit` 两阶段**：扩容时先把新 rank 连上但**保持 masked**（不影响在跑的请求），
  到安全点再 commit —— **这是「在线扩缩容」的标准做法**；
- `max_num_ep_ranks` 默认 **32**（`all2all.py:405` + `envs.py:319`）→ 预留了扩容空间；
- 两个阶段在 `NixlEPPrepareAndFinalize.on_commit()`（`nixl_ep.py:144-153`）里触发。

✅ **三者共同的「0 SM」承诺**：NIXL EP 也是 `max_sms_used() -> 0`
（`all2all.py:558-560` 注释 *"NIXL EP uses RDMA so no SMs are used for communication"*）
→ **和 DeepEP LL 一样可以做 DBO**。

✅ **MoRI 的平台约束是硬编码的**：`all2all.py:939-941` 直接 `assert on_gfx942() or on_gfx950()`；
并且**机内/机外走不同 kernel**：单机用 `IntraNode`，多机按后端名分
`mori_low_latency → InterNodeV1LL` / `mori_high_throughput → InterNodeV1`（`:951-955`），
block/warp 数还按 gfx942 vs gfx950 分别调（`:956-963`）。
→ **「同一份框架代码，不同硬件的调优参数完全不同」**，这就是平台抽象存在的意义。

**面试加分**：
> 「这三个后端的共同点是**都在解决『DeepEP 假设不成立』的场景**：
> NIXL EP 解决『集群大小会变』，MoRI 解决『不是 NVIDIA 卡』，
> FlashInfer 解决『NVLink 跨了节点（MNNVL）』。
> **框架的后端数量，等于它承认的现实场景数量。**」

---

### Q36 `[进阶]` 量化跟 all-to-all 怎么交互？fp8 / mxfp8 / nvfp4 各有什么坑？

**答**：核心矛盾是 —— **量化能减半通信量，但 scale 的形状会被 a2a 打乱**。

📖 **收益**：`[T, H]` bf16 → 每 token `2H` 字节；fp8 → `H` 字节；nvfp4 → `0.5H` 字节。
**通信量直接按位宽减小**，这在 all-to-all 上是线性收益。

✅ **坑 1：scale 必须「跟着 token 一起走」，但 swizzle 会改变形状**
`naive_dp_ep.py:32-35` 的注释是标准答案：
> *"NOTE: swizzling pads the scales to multiple of 128 which makes the scales tensor
> different shape than the hidden states, **breaking the A2A kernel**. So, we delay the
> swizzling until after the A2A."*

→ 所以流程是：**先做「不 swizzle」的量化 → 把 scale 当 extra tensor 一起 dispatch
→ 落地后再 swizzle**（`naive_dp_ep.py:62-66` 的 `nvfp4_block_scale_interleave`）。
**「先传后整容」是处理「传输要求规整、计算要求 swizzle」冲突的通用套路。**

✅ **坑 2：静态量化不需要传 scale**
`naive_dp_ep.py:46-50`：
```python
# Skip gathering scales if we have static quantization
# (the scale is a scalar, replicated on all ranks)
skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
```
→ **per-tensor 量化的 scale 是个标量，所有 rank 上一样**，传它纯属浪费。
**这是「按量化粒度决定通信内容」的典型例子。**

✅ **坑 3：不同后端支持的最低精度不一样**

| 后端 | 支持的 dispatch 精度 | 证据 |
|---|---|---|
| DeepEP HT | **只支持 fp8 block scale**（其他精度在 dispatch 后量化）| `deepep_ht.py:288-292` |
| DeepEP LL | fp8 block / **nvfp4（需 hybrid-ep 分支）** | `deepep_ll.py:184-198` |
| DeepEP v2 | blockfp8 + **mxfp8**（scale 要 pack 成 int32）| `deepep_v2.py:27-52` |
| NIXL EP | fp8 block | `nixl_ep.py:169-182` |

✅ **mxfp8 的额外约束（很好的细节题）**：
`deepep_v2.py:39-52` 的 `_pack_mxfp8_scale` 注释：
> *"DeepEP moves scale factors as opaque 4-byte packs (`sf_pack_t` is a
> float/UE8M0x4 union), so 1-byte UE8M0 scales must be packed 4-per-int32."*

因为 `MXFP8_BLOCK_SIZE = 32`（`mxfp8_utils.py:11`），每 32 个元素 1 个 scale，
pack 成 4 个一组 → **要求 `hidden_size % 128 == 0`**（`deepep_v2.py:48-51` 的 assert）。
→ **「量化格式的 block 大小，最终变成了对 hidden_size 的整除约束」** ——
这是「数值格式 → 通信接口 → 模型架构」三层耦合的例子，能讲清楚很加分。

✅ **坑 4：per-token scale 在 LL 路径上不被支持**
`deepep_ll.py:277-280`：
```python
if not use_nvfp4:
    assert not has_per_token_scales, (
        "low_latency kernels doesn't support dispatching per-token scales"
    )
```
→ **「低延迟内核的功能子集更小」是普遍规律**：为了极致延迟，它砍掉了灵活性。
**这解释了为什么 LL 和 HT 不能只当成「同一个东西的快慢两档」。**

---

### Q37 `[进阶]` 为什么 DeepEP LL 只支持 8 个 hidden size？这个约束怎么传导到上层？

**答**：因为 LL 的 kernel 是为**特定 hidden size 编译**的，不是模板化的。

✅ **硬约束**：`deepep_ll.py:57-61`
```python
# DeepEP low-latency kernels are compiled only for certain
# specific hidden sizes.
# NOTE: Keep this list sorted, maybe_roundup_layer_hidden_size depends on it.
SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]
```
`prepare_async` 里直接断言（`deepep_ll.py:247-250`）；
不满足时 `maybe_roundup_layer_hidden_size` **向上取到最近的档位**，
超出最大值（> 8192）直接 `ValueError`（`:80-83`）。

✅ **对比 HT / v2 的策略完全不同** —— 这是本题的核心：
```python
# deepep_ht.py:35-48  /  deepep_v2.py:96-104
hidden_size_bytes = hidden_size * dtype.itemsize
xfer_atom_size = 512  # 32 * 16 (size(int4))
if hidden_size_bytes % xfer_atom_size == 0:
    return hidden_size
hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
return hidden_size_bytes // dtype.itemsize
```
**HT 只要求「512 字节对齐」**（理由是 *"DeepEP intranode kernels make copies in units of
32(warp-size) int4 elements"*，`:38-39`），**不要求枚举值**。

| | HT / v2 | LL |
|---|---|---|
| 约束 | `hidden_size × itemsize % 512 == 0` | **必须精确等于 8 个值之一** |
| 不满足时 | 向上对齐（无损，只是多算一点） | 向上取档（**会改变 hidden size！**）|
| 举例 | 2880×2=5760 → 6144 → **3072** | 2880 → **3072** |

✅ **这个约束怎么传导到上层**：
`all2all_utils.py:123-162` 的 `maybe_roundup_layer_hidden_size()` 会被 **MoE 层的构造**调用，
**改的是模型的 hidden size**，而不是只改通信 buffer：
```python
if moe_parallel_config.use_deepep_ht_kernels:   # all2all_utils.py:142
    hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(hidden_size, act_dtype)
if moe_parallel_config.use_deepep_ll_kernels:   # :147
    hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(hidden_size)
```

**面试加分（这题的真正考点）**：
> 「`2880 → 3072` 意味着**权重矩阵变大了**（多了 6.7% 的列），
> 而这些多出来的列在模型里根本不存在、永远是 0。
> 这是**「通信库的编译期约束反向渗透到模型定义」**的真实案例。
> 判断一个框架成熟不成熟，就看它有没有**把这种约束封在一处**
> （vLLM 封在 `all2all_utils.maybe_roundup_layer_hidden_size`），
> 而不是让每个模型自己处理。」

📖 **额外思考**：为什么 LL 要用「枚举编译」而不是「模板」？
因为 LL 的 kernel 里**每个 rank 的 buffer 大小、RDMA QP 数量都要提前精确算出来**
（`get_low_latency_rdma_size_hint`），编译期确定才能做到**零动态分配 + CUDA Graph 可捕获**。
**「用灵活性换确定性」是延迟敏感路径的通用取舍。**

---

### Q38 `[系统]` 设计：DeepSeek 类 MoE 要跨 2 个节点，a2a 后端怎么选？

**答题框架**：

**第 1 步：先算通信量，别先选后端**
```
设 T = 每 rank token 数，H = 7168，K = topk = 8，EP = 16
dispatch 发送量/rank  ≈ T × K × H × dtype   （每 token 发给 K 个专家）
bf16: T × 8 × 7168 × 2 = T × 114 KiB
decode T=128 → 14.6 MiB / rank / 层
```
→ **听到「每层十几 MB、每步都要做」就该知道：必须用 RDMA 硬件卸载 + 和计算重叠。**

**第 2 步：按阶段选后端**

| 场景 | 后端 | 理由 |
|---|---|---|
| 在线服务（decode 为主） | `deepep_low_latency` | **0 SM**、支持 CUDA Graph、可开 DBO |
| 离线批处理（prefill 为主） | `deepep_high_throughput` | 通信量大，用 20 SM 换带宽 |
| 集群会弹性伸缩 | `nixl_ep` | 唯一支持动态 rank 连接/断开的 a2a |
| 有 NCCL ≥ 2.30.4 + IBGDA | `deepep_v2` | 统一 API + 动态 SM |

**第 3 步：必须同时打开的开关**（只选后端是不够的）
```bash
--enable-expert-parallel
--enable-dbo                          # decode 场景做通算重叠
--all2all-backend deepep_low_latency
--enable-eplb                         # 负载均衡（否则热门专家成瓶颈）
--data-parallel-size 16 --data-parallel-size-local 8
```
✅ 注意 `--enable-eplb` 要求 `enable_expert_parallel`（`config/parallel.py:522-523` 会报错），
并且 EPLB 会用**独立的 process group**（`parallel_state.py:2117-2120`）。

**第 4 步：逐项检查前提**
```
① 网卡：IBGDA 能力？（deepEP v2 的硬前提）
② hidden_size：7168 在 LL 的 SUPPORTED_HIDDEN_SIZES 里 ✅（deepep_ll.py:61）
③ 量化：如果是 fp8 block，DeepEP 会在 dispatch 时就量化（省一半带宽）
④ 显存：LL 的 RDMA buffer 是按需算的（get_low_latency_rdma_size_hint）
⑤ DBO：阈值够不够？（dbo_decode_token_threshold 默认 32，要低于实际 batch）
```

**第 5 步：验证顺序**（说出来就赢一半）
```
1) 启动日志：all2all manager 类型 + EPLB 是否启用
2) nsys：确认 dispatch/combine 和计算真的 overlap（不是串行）
3) 每 rank 的 token 数分布：有长尾说明 EPLB 没配好或 window 太大
4) 关掉 DBO 对比：如果性能没差别，说明 overlap 没生效
```

**追问**：「如果只有 100 Gb/s 以太网（没有 IB）？」
→ **DeepEP LL 用不了**（需要 RDMA）。此时：
① 退回 `allgather_reducescatter`，但要把 EP 尽量收进机内；
② 或者干脆**不用 EP，用 TP**（TP 的 all-reduce 在以太网上也很惨，但量更小）；
③ 或者降低 topk / 换更小的模型。
**核心原则还是 Q18 那句：把延迟敏感的通信留在机内，把可容忍延迟的放到机间。**

---

## 第十部分：通算融合 / 通信计算重叠（✅ 为主）

> DBO 是 vLLM 最近最有"系统味"的设计，`docs/design/dbo.md` 是官方设计文档。
> **能讲清楚 DBO 的候选人，通常能直接进系统方向。**

### Q39 `[基础]` 什么是 DBO？它解决什么问题？

**答**：**DBO = Dual Batch Overlap（双批次重叠）**：把一个 batch 切成两个
microbatch（ubatch），用两个 CPU 线程 + 两组 CUDA stream 让
**「A 的通信」和「B 的计算」在时间上重叠**。

✅ **官方动机**（`docs/design/dbo.md:5` 原文）：
> *"The core motivation of the DBO system in vLLM is to overlap the sparse all-to-all
> communication in the MoE layer with the surrounding computation.
> This system currently only targets DP+EP deployments."*

✅ **机制**（`docs/design/dbo.md:9`）：
> *"splitting the batch in the model runner, creating two worker threads, and then running
> the model on each of these worker threads. When DBO is enabled, yield points within the
> `FusedMoEModularKernel` allow the two CPU worker threads (also called UBatch threads) to
> **ping-pong** between each other so that when one is running compute, the other is waiting
> on communication."*

✅ **实施排期**（`docs/design/dbo.md:23-27`，背下来很唬人）：
```
# Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
# （S=共享专家, A0=qkv proj, A1=core attn+out proj+MoE gate, D=dispatch, C=combine）
```
**注意 D₁ 在前、A0₀ 在后** —— 这就是重叠：**ubatch 1 的 dispatch 在跑的时候，
ubatch 0 在做 attention 的 QKV 投影。**

📖 **为什么 MoE 特别需要它**：MoE 的 all-to-all **既不是纯延迟敏感（像 TP），
也不是纯带宽敏感（像 PP）**，而是「**中等消息 + 高频 + 可被切分**」。
可切分是重叠的前提：**只有能切成独立的两半，才有东西可以交错。**

**追问**：「切一半不会让每个 kernel 变小、效率变低吗？」
→ 会。所以有阈值（`config/parallel.py:225-234`）：
`dbo_decode_token_threshold` 默认 **32**，`dbo_prefill_token_threshold` 默认 **512**。
**batch 太小就不切**（切了以后每个 ubatch 太小，kernel 打不满）。
另外还会检查「切完第二个 ubatch 会不会是空的」（见 Q40）。

---

### Q40 `[进阶]` 双线程 + 双 stream 的 ping-pong 到底怎么实现的？

**答**：**CPU 侧严格互斥（一次只有一个线程在跑），GPU 侧用跨 stream 的 event 依赖构成流水。**

✅ **先纠正一个常见误解**（**这是本部分最好的加分点**）：
不是「每个 ubatch 各有一对 stream」，而是**两个 ubatch 共用同一个 compute stream
和同一个 comm stream**。证据是 `make_ubatch_contexts`（`ubatching.py:202-241`）：
```python
for i in range(num_micro_batches):
    ctx = UBatchContext(
        id=i,
        compute_stream=compute_stream,   # ← 同一个对象
        comm_stream=comm_stream,         # ← 同一个对象
        ...
    )
```
→ **真正的「双份」是「双 CPU 线程 + 两组 ForwardContext + 两组 event」**，
不是双 stream。**能指出这一点，说明你是真的读了代码而不是看了示意图。**

✅ **CPU 侧的 ping-pong**（`ubatching.py:94-105` 的 `_cpu_yield`）：
```python
def _cpu_yield(self):
    # It is critical for correctness that only one thread is running
    # at a time. These asserts just make sure that this is the only
    # thread running before waking the other one up and going to sleep
    assert forward_context._forward_context == self.forward_context
    assert current_stream() == self.current_stream
    assert not self.cpu_wait_event.is_set()

    self.cpu_signal_event.set()      # 叫醒对方
    self.cpu_wait_event.wait()       # 自己睡
    self.cpu_wait_event.clear()
    self._restore_context()          # 醒来后恢复「我的」ForwardContext
```
→ **`cpu_signal_event` 是对方的 `cpu_wait_event`**（`ubatching.py:233-234` 的交叉赋值），
这就是「接力棒」。

✅ **GPU 侧的跨 stream 依赖**（`ubatching.py:133-147`）：
```python
def yield_and_switch_from_compute_to_comm(self):
    assert current_stream() == self.compute_stream
    self._signal_compute_done()          # comm_stream 上 record 一个 event
    self._cpu_yield()                    # ← 叫醒对方，对方在 compute stream 上干活
    self.update_stream(self.comm_stream) # 自己切到 comm stream
    self._wait_compute_done()            # comm_stream 等这个 event
```
**注意 `_cpu_yield()` 在中间** —— 这是设计的关键：
**在「挂起自己、切到通信」之前先把对方叫醒，让对方去填满 compute stream**，
这样通信和计算才真的并行。

✅ **四个同步原语的分工**：

| 函数 | 语义 | 用在哪 |
|---|---|---|
| `switch_to_comm` / `switch_to_compute` | 只切 stream，**不等待** | 不构成依赖的场合 |
| `switch_to_compute_sync` | record comm done → 切 → **等 compute done** | DeepEP HT dispatch 之后（`deepep_ht.py:179`）|
| `yield_and_switch_from_compute_to_comm` | signal + **让出 CPU** + 切 + 等 | dispatch 之前（`deepep_ht.py:128`）|
| `yield_and_switch_from_comm_to_compute` | 对称 | combine 收尾（`deepep_ht.py:403`）|

✅ **yield 点在哪**：`docs/design/dbo.md:78` 说 *"The current implementation has all
`dbo_yield` and `dbo_maybe_run_recv_hook` calls in the `FusedMoEModularKernel.forward` method."*
代码里共 3 处：
- `modular_kernel.py:1244` — **dispatch 前**（注释：*"Overlap shared expert compute with all2all dispatch."*）
- `modular_kernel.py:1263-1271` — prepare 的 hook：**注册到对方 context 再 `dbo_yield()`**
- `modular_kernel.py:1427-1435` — finalize 的 hook，同样注册 + yield

**「共享专家（shared expert）的计算藏在 all-to-all 后面」** 就是这里实现的：
`modular_kernel.py:1416` 的 `_maybe_apply_shared_experts(...)` 在 `finalize_async` 之后立即执行，
而 combine 的完成由对方线程的 hook 来等。

✅ **CUDA Graph 的处理**（`gpu_ubatch_wrapper.py:121-212`）：
捕获时两个线程先各自**初始化 CUDA context**（`torch.cuda.current_blas_handle()`，`:148-151`），
然后主线程开图捕获、唤醒第一个线程、等两个都 join。
捕获后 **replay 不需要任何多线程或 CPU 同步**（`docs/design/dbo.md:62`）——
**这是 DBO 能做进生产的前提：稳态下零 Python 开销。**

**追问**：「如果 batch 切不均匀怎么办？」
→ 两层处理：
① `_post_process_ubatch`（`dp_utils.py:62-79`）：如果按 padding 后的 token 数切，
**最后一个 ubatch 会是空的** → 直接放弃 microbatch（`is_last_ubatch_empty`，`ubatch_utils.py:130-133`）；
② DP 维度上**必须所有 rank 一致**（`dp_utils.py:67` 的 `torch.all(tensor[2] == 1)`），
因为 collective 的**成员必须一致** —— 一头切一头不切就直接死锁。

---

### Q41 `[系统]` DBO 的 SM 仲裁是怎么做的？为什么只有 DeepGEMM 和 DeepEP HT 支持？

**答**：因为**通信 kernel 和计算 kernel 会抢 SM**，重叠不一定更快 ——
**必须显式划分 SM 配额，才能保证两边都不被饿死。**

✅ **机制**（`ubatch_utils.py:39-127`）：

```python
class SMControlContextManager:
    def __init__(self, comm_sms, set_comm_sms, set_compute_sms):
        total_sms = num_compute_units(device)
        assert comm_sms < total_sms
        self.compute_sms = total_sms - comm_sms      # ← 余下的给计算

    def __enter__(self):
        self.set_comm_sms(self.comm_sms)
        self.set_compute_sms(self.compute_sms)

    def __exit__(self, ...):
        self.set_comm_sms(self.total_sms)            # ← 退出时全部还回去
        self.set_compute_sms(self.total_sms)
```
**关键设计**：这是一个**上下文管理器**，只在 microbatch 运行期间改变配额，
退出后立刻恢复 —— **不影响 DBO 之外的任何 kernel**。

✅ **`comm_sms` 的确定过程**（`ubatch_utils.py:84-127`）：
```python
comm_sms = envs.VLLM_DBO_COMM_SMS          # 默认 20（envs.py:287）
...
if parallel_config.enable_expert_parallel:
    all2all_manager = ep_group.device_communicator.all2all_manager
    if all2all_manager is not None:
        max_sms_used = all2all_manager.max_sms_used()
        if max_sms_used is not None:
            comm_sms = min(comm_sms, max_sms_used)     # ← 取小
    if comm_sms > 0 and all2all_manager is not None:
        set_comm_sms = lambda sms: all2all_manager.set_num_sms(sms)
```
→ **`min(20, max_sms_used())`**：DeepEP LL 返回 **0**（`all2all.py:345-347`），
所以 **LL + DBO 时 `comm_sms = 0`，`compute_sms = 全部`** ——
这正好对应「LL 不占 SM」的事实，**配额逻辑和硬件事实是自洽的**。

✅ **`set_num_sms` 在 DeepEP HT 上只能减不能加**（`all2all.py:260-268`）：
```python
def set_num_sms(self, num_sms: int):
    # Right now the buffers are sized for only what the kernels were
    # created with. So we can only reduce the number of SMS used
    # but not increase it.
    if num_sms > self.num_sms:
        num_sms = self.num_sms
    deep_ep.Buffer.set_num_sms(num_sms)
```

✅ **计算侧只支持 DeepGEMM**（`ubatch_utils.py:118-121`）：
```python
# TODO(lucas): support other kernels besides DeepGEMM
set_compute_sms = lambda sms: None
if has_deep_gemm() and comm_sms > 0:
    set_compute_sms = lambda sms: deep_gemm_set_num_sms(sms)
```
通信侧只支持 DeepEP HT（`ubatch_utils.py:101-103` 的注释：
*"Currently only DeepEP highthroughput supports SM control so this only affects that case."*）。

**为什么只有它们**：**要能被「限流」，kernel 必须自己支持传入 SM 数量**
（DeepGEMM 的 `set_num_sms`、DeepEP 的 `Buffer.set_num_sms`）。
**普通 PyTorch/cuBLAS kernel 根本接受不了这个参数** ——
所以这不是 vLLM 偷懒，而是**上游 kernel 的能力边界**。

**面试加分**：
> 「SM 仲裁的本质是**把『时间片轮转』变成『空间划分』**：
> 不加控制时，通信 kernel 和计算 kernel 会争抢所有 SM，互相打断、互相拖慢；
> 显式划分后，通信固定用 20 个 SM、计算用剩下的，**两端同时满负荷**。
> 代价是**两边都用不到全部资源** —— 所以只在大 batch、计算和通信时间接近时才划算。
> 这就是 `VLLM_DBO_COMM_SMS` 需要调的原因。」

---

### Q42 `[进阶]` 为什么 ROCm + DeepEP HT 必须把 `comm_sms` 设成 0？

**答**：因为**在 ROCm 上，给 DeepEP HT 预留 CU 会破坏 DP+EP 的生成正确性**。

✅ **代码与注释原文**（`ubatch_utils.py:89-98`）：
```python
rocm_deepep_ht_dbo = (
    current_platform.is_rocm()
    and parallel_config.enable_dbo
    and parallel_config.all2all_backend == "deepep_high_throughput"
)
if rocm_deepep_ht_dbo:
    # On ROCm, reserving CUs for DeepEP HT communication under DBO
    # corrupts DP+EP generation accuracy. Keep the backend active, but
    # leave all CUs visible to the compute and communication kernels.
    comm_sms = 0
```
→ **不是「关掉 DBO」，也不是「换后端」，而是「保持后端但取消 SM 划分」。**
**这个取舍很值得学：功能保留，优化退让。**

✅ **配套的第二处修改**（`deepep_ht.py:62-78`）：
```python
self.sync_dbo_comm = current_platform.is_rocm()      # :63

def _sync_dbo_comm_if_needed(self) -> None:          # :73-78
    if self.sync_dbo_comm and dbo_enabled():
        # ROCm DeepEP HT dispatch/combine reuse Buffer-owned communication
        # workspace. Do not let the next DBO ubatch reuse that workspace
        # before this ubatch's HT kernel has completed.
        torch.cuda.current_stream().synchronize()
```
→ **ROCm 上 HT 的 dispatch/combine 复用同一块 Buffer-owned 通信 workspace**，
两个 ubatch 会**争用同一块 buffer** → 必须显式同步（在 comm stream 上）。
**这是用一点点性能换正确性的典型做法。**

📖 **为什么 NVIDIA 上不需要**：CUDA 侧的 DeepEP HT 在 workspace 管理上更严格
（每次 dispatch 返回独立的 `handle` 并绑定到 ubatch，见 `deepep_ht.py:65-68`：
*"Under DBO microbatching we must track one handle per micro-batch to avoid races between threads."*
→ `self.handles = [None, None]`）。**ROCm 版 DeepEP 还没做到这个隔离度。**

**面试加分**：
> 「这两处修改连起来看，是一个很成熟的工程判断：
> **发现『SM 预留会算错』时，第一反应不是骂硬件，而是
> ① 保留功能、去掉优化（`comm_sms = 0`），② 加显式同步补上正确性（stream sync）。
> 并且两处都写了注释说明『为什么』和『不做会怎样』。**
> 我在自己的项目里也会这样处理平台差异：**不是 `if rocm: do_something_different()`，
> 而是 `if rocm: 退回正确但慢一点的路径 + 注释说明原因`。**」

---

### Q43 `[进阶]` 512 字节对齐（`xfer_atom_size`）和 DBO 有什么关系？

**答**：**没有直接关系 —— 它是 DeepEP 内核的对齐要求**，
但因为 DBO 只能用 DeepEP，所以它变成了「想开 DBO 就要接受」的前置条件。

✅ **事实**（`deepep_ht.py:35-48`，`deepep_v2.py:96-104` 完全相同）：
```python
hidden_size_bytes = hidden_size * dtype.itemsize
xfer_atom_size = 512  # 32 * 16 (size(int4))
if hidden_size_bytes % xfer_atom_size == 0:
    return hidden_size
hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
return hidden_size_bytes // dtype.itemsize
```
注释解释原因：*"DeepEP intranode kernels make copies in units of 32(warp-size) int4 elements.
Round up hidden size to respect this. For example, an input hidden size of 2880 with dtype
torch.bfloat16 will be rounded up to 3072."*

📖 **换算**：`512 字节 = 32 × 16 字节`，即「**一个 warp（32 线程）× 一个 int4（16 字节）**」
—— 这是**一次 warp 级 vectorized copy 的原子大小**。
`hidden_size × itemsize` 不是 512 的整数倍时，warp 的最后一次拷贝就会越界/不满，
所以要么补齐 buffer，要么让编译器生成 tail 处理（更慢）。
**DeepEP 选择在框架层补齐，换取内核里没有 tail 分支。**

| hidden_size (bf16) | 字节数 | % 512 | 结果 |
|---|---|---|---|
| 7168（DeepSeek-V3） | 14336 | 0 ✅ | 7168 |
| 2880（GPT-OSS 类） | 5760 | ≠0 | **3072** |
| 4096 | 8192 | 0 ✅ | 4096 |
| 5120 | 10240 | 0 ✅ | 5120 |

📖 **顺带提一个 LL 的约束对比**（见 Q37）：
LL 要求 `hidden_size % 128 == 0` 才能用 fp8 dispatch
（`deepep_ll.py:254-257`：*"DeepEP kernels quantize the inputs in blocks of shape 128"*）
—— 这是**量化 block 大小**（128）带来的约束，和 512 字节对齐是**两个独立的约束**。
**面试时能区分「512 是拷贝原子大小」「128 是量化 block 大小」是很好的细节分。**

**追问**：「补齐 hidden size 有什么副作用？」
→ ① **显存和算力浪费**：多出来的列永远是 0，但 GEMM 还是要算；
② **权重加载要处理形状不匹配**（checkpoint 是 2880，模型是 3072）；
③ **和量化 scale 的 block 大小可能再次冲突**（比如 mxfp8 要求 %128）。
**所以 `maybe_roundup_layer_hidden_size` 必须是一个统一的入口**
（`all2all_utils.py:123-162`），不能每个后端各改各的。

---

### Q44 `[进阶]` 除了 DBO，还有哪些通算重叠手段？各自的重叠粒度是多少？

**答**：从细到粗一共 5 个层次 —— **能按粒度排序讲清楚，这题就满分了。**

| 层次 | 手段 | 重叠粒度 | vLLM 里的证据 |
|---|---|---|---|
| **① kernel 内融合** | GEMM + ReduceScatter 融合成一个 kernel | **单次操作内** | `collective_fusion.py:454` 的 `patched_fused_scaled_matmul_reduce_scatter` |
| **② 逐层流水** | 一边算第 i 层，一边传第 i-1 层 | **一层** | KV connector 的 `save_kv_layer` / `wait_for_layer_load`（`kv_connector/v1/base.py:321-343`）|
| **③ 双 microbatch** | DBO：A 算 / B 传 | **半个 batch** | `docs/design/dbo.md` |
| **④ batch 内混排** | chunked prefill：prefill chunk 和 decode 同批 | **chunk** | `config/scheduler.py:116` `enable_chunked_prefill: bool = True` |
| **⑤ 实例级** | PD 分离：prefill 实例和 decode 实例分开 | **整个实例** | `docs/features/disagg_prefill.md` |

✅ **① 融合 kernel 的两种做法**：
- **AllReduce + RMSNorm 融合**：`compilation/passes/fusion/allreduce_rms_fusion.py`，
  ROCm 上落到 AITER 的 `rocm_aiter_fused_allreduce_rmsnorm`（`_aiter_ops.py:979-1040`）；
- **GEMM + ReduceScatter 融合**：`collective_fusion.py:107-161` 的
  `fused_flashinfer_scaled_matmul_reduce_scatter`，把量化 GEMM 和 RS 合成一个 op。

✅ **注意 SP pass 和融合 pass 的关系**（**很好的细节题**）：
`sequence_parallelism.py:511-515` 的 docstring 明说：
> *"While this pass itself does not directly yield performance improvements,
> it lays the groundwork for subsequent fusion passes, such as GEMM + ReduceScatter and
> AllGather + GEMM fusions. These fusions can significantly reduce communication overhead."*

→ **SP pass 只做「改写形式」（AR → RS+AG），真正的收益来自后续的融合 pass。**
**「先改写成可融合的形式，再融合」** 是编译器优化的经典分阶段思路。

✅ **④ chunked prefill 的取舍**（`docs/features/disagg_prefill.md:12-13`）：
> *"Without disaggregated prefilling, vLLM may insert some prefill jobs during the decoding
> of one request. This results in higher tail latency. ... **Chunked prefill with a proper
> chunk size also can achieve the same goal, but in practice it's hard to figure out the
> correct chunk size value.**"*

**面试加分**：
> 「这五种手段不是互斥的，而是**按粒度互补**的：
> 融合 kernel 拿掉固定开销，逐层流水藏住 per-layer 通信，
> DBO 藏住 MoE 的 all-to-all，chunked prefill 藏住 prefill 的抖动，
> PD 分离干脆把两种负载放到不同实例上。
> **选哪一层，取决于『通信在时间轴上的形状』**：
> 是每次几百微秒的脉冲（→ DBO），还是持续的大块传输（→ 逐层流水）。」

---

### Q45 `[系统]` 为什么通信计算重叠这么难？三大障碍是什么？

**答**：**依赖关系、SM 争抢、CUDA Graph 限制** —— 三者都会让「看起来能重叠」变成「实际串行」。

**障碍 1：真依赖 vs 假依赖**

| 类型 | 例子 | 能不能重叠 |
|---|---|---|
| **真数据依赖** | combine 的结果是下一层的输入 | ❌ 必须等 |
| **假依赖（可消除）** | 为了复用 buffer 而强行同步 | ✅ 用双缓冲消除 |
| **同步引入的依赖** | `stream.synchronize()`、`.item()`（D2H） | ✅ 用 device-side 逻辑消除 |

✅ **vLLM 的实例**：
- **假依赖**：DBO 给每个 ubatch 一个独立 handle（`deepep_ht.py:65-68`）避免 buffer race；
- **同步引入的依赖**：`deepep_v2.py:532` 注释 *"num_recv_tokens read on-device (no host sync)
  -> cudagraph-safe"* —— 一个专门的 Triton kernel 来避免 `.item()`。

**障碍 2：SM 争抢**（见 Q41）
不划分配额时，通信 kernel 和计算 kernel 会互相抢占。
**「重叠」可能比「串行」更慢** —— 因为两个 kernel 互相打断对方的 cache 和 occupancy。

**障碍 3：CUDA Graph 限制**

| 限制 | 后果 | vLLM 的解法 |
|---|---|---|
| 捕获期间不能做同步/建连 | NCCL 懒初始化会炸 | pynccl（自己控制建连时机）|
| kernel 参数地址必须固定 | 输入 tensor 地址是变的 | capture 时先 copy 进预注册 buffer |
| **所有 host 同步都不可捕获** | `.item()` / `.cpu()` 直接让 graph 失败 | 全部改成 device-side（如 Q32 的 Triton kernel）|
| 多线程捕获需要固定 stream | 两个 ubatch 混在一个图里 | 同一 `compute_stream`（`gpu_ubatch_wrapper.py:196-200`）|
| 控制流分叉不能捕获 | Python `if` 依赖 device 值 | 用 `torch.cond` 或预先算好 |

✅ **vLLM 在 DBO 上的具体处理**（`docs/design/dbo.md:62`）：
> *"Because of this, DBO only supports running with Full CUDA graphs.
> However, once a DBO CUDA graph has been captured, it can be replayed **without any
> multithreading or CPU synchronization**."*
→ **用「捕获期一次性的复杂性」换「稳态期的零开销」** —— 和 Q11 的 custom AR 是同一个哲学。

**障碍 4（加分项）：正确性**（重叠最容易出的 bug）
- `gpu_model_runner.py:3995-4007` 的 `_allow_microbatching`：
  **prefix cache 的「读者」和「写者」被切到不同 ubatch → 读者读到还没写好的块** →
  直接 veto 这次 microbatch。注释原文解释了机制：
  > *"Run whole, the step issues every KV cache write before any attention read and the hit
  > holds; split, a reader in the first half would attend over blocks the writer in the second
  > half has not filled in yet."*
- **两个 ubatch 的 collective 顺序必须一致** —— 否则死锁。

**面试加分（收尾金句）**：
> 「重叠的难点从来不是『怎么让两个 kernel 并行』，而是**『怎么证明并行之后结果还对』**。
> vLLM 里到处都是这种守卫：`_allow_microbatching` 防 KV race、
> `_post_process_ubatch` 防空 microbatch、
> ROCm 上 `_sync_dbo_comm_if_needed` 防 workspace 复用。
> **一个重叠优化能不能上线，取决于它的正确性守卫写得够不够细。**」

---

### Q46 `[系统]` 设计：什么情况下该开 DBO？给一个决策流程。

**答**：DBO **不是免费的午餐** —— 它用「一半的 batch」换「重叠」，
所以在小 batch 或通信占比低时是负优化。

**第 1 步：硬前提检查**（不满足直接放弃）
```
① DP > 1 且 enable_expert_parallel        （docs/design/dbo.md:32）
② 后端是 deepep_low_latency / deepep_high_throughput（docs/design/dbo.md:37）
③ 至少 2 张 GPU 可见                      （docs/design/dbo.md:42）
④ CUDA Graph 模式是 FULL（DBO 只支持 full graph）
```

**第 2 步：估算「通信占比 × 可切分性」**
```
设 decode 一步总时间 T，其中 MoE all-to-all 占 C
  理论收益上限 = C / T          （完美重叠）
  实际收益     ≈ 0.5 ~ 0.7 × (C/T)   （阈值、SM 划分、尾部损失）
→ C/T < 20% 时基本不值得
```
**怎么测 `C/T`**：`nsys` 里看 dispatch/combine kernel 占一步时间的比例（第 5 章的方法）。

**第 3 步：调阈值**（`config/parallel.py:225-234`）
```bash
--enable-dbo
--dbo-decode-token-threshold 32     # 默认 32；batch 小于它就完全不切
--dbo-prefill-token-threshold 512   # 默认 512
```
⚠️ **约束**（`config/parallel.py:1070-1077`）：两个阈值都必须 `>= num_ubatches`（=2），
否则启动直接 `ValueError`。**理由在注释里**：
*"A batch below one token per microbatch cannot be split, so the thresholds have to keep it
out rather than the split having to cope."*

**第 4 步：调 `VLLM_DBO_COMM_SMS`**
```
默认 20。规则：min(20, all2all_manager.max_sms_used())
  - deepep_low_latency  → max_sms_used() = 0 → comm_sms = 0（不划分）
  - deepep_high_throughput → 20 → comm_sms = min(20, 20) = 20
  - ROCm + HT → 强制 0（正确性问题，见 Q42）
经验：通信偏重就把该值调大，计算偏重就调小。永远记得这是「零和」的。
```

**第 5 步：验证（必须做对比实验）**
```
① 开/关 DBO 各跑一遍，看 p50 TPOT 和 p99 TPOT
   ⚠️ 注意 DBO 的收益常常体现在 p99（重叠摊平了抖动）而不是 p50
② nsys 确认 dispatch/combine 和计算 kernel 在时间轴上真的交叠
③ 检查有没有正确性问题：输出对不对（KV race、collective 顺序）
④ 极端 case：batch 恰好等于阈值时行为是否符合预期
```

**追问**：「什么时候应该关掉 DBO？」
→ ① 小 batch（decode 并发 < 阈值）；② 通信占比低（比如纯 dense 模型）；
③ 用的是 LL 且 `max_sms_used() == 0`（这时重叠的收益只是「通信不占 SM」，
   可能已经被 LL 的硬件卸载实现了，DBO 的额外复杂度未必划算）；
④ **调试期**：DBO 让栈回溯和多线程调试变得非常痛苦。

---

## 第十一部分：新维度与传输层（✅ 为主）

### Q47 `[进阶]` Sequence Parallelism 为什么能把 all-reduce 变成 reduce-scatter + all-gather？

**答**：因为 all-reduce 后面常常跟着一个**逐 token 的操作**（RMSNorm / 量化），
而这个操作**不需要完整的 hidden 维度**，所以可以先规约再算、算完再扩散。

✅ **vLLM 的 pass 定义**（`sequence_parallelism.py:500-515` docstring 原文）：
```
The general transformation is:
Input -> AllReduce -> RMSNorm -> Output
becomes
Input -> ReduceScatter -> RMSNorm -> AllGather -> Output
```

✅ **真实替换代码**（`sequence_parallelism.py:166-179`）：
```python
# pattern（原图）
all_reduce = self._all_reduce(input)
rmsnorm = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)
return rmsnorm, all_reduce

# replacement（替换后）
reduce_scatter = self._reduce_scatter(input)
rmsnorm = vllm.ir.ops.rms_norm(reduce_scatter, weight, self.epsilon)
all_gather = self._all_gather(rmsnorm)
return all_gather, reduce_scatter
```

📖 **通信量对比（这是本题的核心，也是最容易答错的地方）**：

| | 搬运量 | 说明 |
|---|---|---|
| AllReduce（ring） | `≈ 2S` | RS + AG 两阶段 |
| ReduceScatter + AllGather | `≈ 2S` | **和 all-reduce 一样！** |

→ **所以「通信量减半」不是来自 RS/AG 本身，而是来自中间的量化！**

✅ **看 fp8 版本的替换就明白了**（`sequence_parallelism.py:282-297`）：
```python
# pattern
all_reduce = self._all_reduce(input)
rms = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)
quant, _ = self.quant_matcher(rms, scale)     # ← 量化在 all-reduce 之后

# replacement
reduce_scatter = self._reduce_scatter(input)
rms = vllm.ir.ops.rms_norm(reduce_scatter, weight, self.epsilon)
quant, _ = self.quant_matcher(rms, scale)     # ← 量化移到 all-gather 之前
all_gather = self._all_gather(quant)          # ★ AllGather 传的是 fp8，不是 bf16！
```
**AllGather 传的是量化后的张量**：bf16（2 字节）→ fp8（1 字节）**通信量减半**，
→ nvfp4（0.5 字节 + scale）**约 1/4**（`sequence_parallelism.py:411-422` 的 NVFP4 版本）。

📖 **所以 SP 的收益来自两个独立的点**：
1. **量化前移**：AG 传低精度 → 通信量按位宽比例下降（**这才是"减半"**）；
2. **RMSNorm 本地化**：norm 在 `S/N` 的数据上做，**计算量除以 N**；
   而且 residual 也被切开了（`sequence_parallelism.py:523-528` 说明 residual 会被 split）。

✅ **收益顺序很重要（面试官会追问）**：
`sequence_parallelism.py:511-515` 的 docstring 明确说：
> *"**While this pass itself does not directly yield performance improvements**,
> it lays the groundwork for subsequent fusion passes, such as GEMM + ReduceScatter and
> AllGather + GEMM fusions. These fusions can significantly reduce communication overhead."*

→ **SP pass 单独用几乎没有收益**（`2S` 对 `2S`），
真正赚钱的是**后续把 RS 融进上一个 GEMM、把 AG 融进下一个 GEMM**（去掉一次显存往返）。

**追问**：「那为什么还要单独做这个 pass？」
→ 因为**融合 pass 需要「可融合的图结构」**。
`AR → RMSNorm` 这种形式没法融合（AR 是全局同步点，前后都是完整张量）；
`RS → local op → AG` 才能让 `GEMM → RS` 和 `AG → GEMM` 各自成为一个融合单元。
**「先改写成可优化的形式，再做优化」是编译器分阶段设计的标准做法。**

---

### Q48 `[进阶]` SP 什么时候开、什么时候不能开？阈值是怎么算的？

**答**：三个条件 —— **模型够大、硬件支持、token 数够多**，缺一不可。

✅ **条件 1：hidden size 下限**（`sequence_parallelism.py:38-43`）
```python
SP_MIN_HIDDEN_SIZE: dict[int, int] = {
    90:  8192,   # H100
    100: 8192,   # Blackwell
}
```
→ **H100 上只有 `hidden_size >= 8192` 的模型才考虑 SP**。
（DeepSeek-V3 的 7168 都不到 → 默认不开；Qwen 系列的小模型更不用想。）
小模型上 SP **必然亏**：通信固定成本摊不掉，norm 省下的计算可以忽略。

✅ **条件 2：token 数下限**（`sequence_parallelism.py:45-52, 102-104`）
```python
SP_MIN_PER_GPU_SIZE_MB: dict[int, float] = {
    90:  8,    # H100: 每 GPU 至少 8 MB
    100: 32,   # Blackwell: 更保守
}
...
min_size = min_per_gpu_size_mb * MiB * tp_size
return int(min_size // (hidden_size * element_size))
```
**手算 H100 / TP=8 / hidden=8192 / bf16**：
```
min_token_num = (8 × 1024 × 1024 × 8) / (8192 × 2) = 4096
```
**Blackwell / TP=8 / hidden=8192 / bf16**：
```
min_token_num = (32 × 1024 × 1024 × 8) / (8192 × 2) = 16384
```
→ **「每 GPU 至少要处理 8 MB（H100）/ 32 MB（Blackwell）的 hidden states 才值得切」。**
这个数字的物理含义就是「通信的固定开销 vs 省下的计算量」的平衡点。

✅ **条件 3：编译模式必须支持**（`sequence_parallelism.py:517-521`）
> *"This pass is only supported when compiling the whole graph (fullgraph mode...
> **Piecewise compilation is not supported because the residual tensor gets split across
> TP ranks, causing size mismatches at subgraph boundaries.**"*

→ **SP 会改变 residual 的形状**，而 piecewise CUDA graph 的每个子图边界都要求形状一致
→ 只能用 fullgraph（`is_applicable_for_range`，`:592-616`）。
**「图变换改变了张量形状 → 和分图编译冲突」** 是很典型的编译器约束。

✅ **条件 4：token 数必须能被 TP 整除**（`gpu_model_runner.py:4099-4107`）
```python
if self.compilation_config.pass_config.enable_sp:
    assert batch_descriptor.num_tokens % self.config.parallel_config.tensor_parallel_size == 0, (
        "Sequence parallelism requires num_tokens to be a multiple of tensor parallel size"
    )
```
→ **启动时就 assert**：所以做 SP 必然要开 CUDA Graph 的 pad（把 token 补到 TP 的倍数）。

✅ **运行时用的工具函数**（`models/common/ops/sequence_parallel.py`）：
- `sp_all_gather`（`:23-27`）/ `sp_reduce_scatter`（`:30-39`）：
  **优先用 device communicator 的 `custom_all_gather` / `custom_reduce_scatter`**，
  没有才退回 TP 组的通用实现 —— **「有专用 kernel 就用，没有就通用兜底」**；
- `sp_reduce_scatter` 会自动 pad 到 `tp_size` 的倍数（`:32-35`）；
- `sp_shard`（`:42-50`）/ `sp_padding_mask`（`:53-68`）：手工做 TP-aware 切片，
  **并且保证 padding 位置被标记出来**（否则 pad 出来的 token 会污染 norm 统计量）。

📖 **SP 和 MoE 的关系**（✅ 有代码依据）：
`config/parallel.py:703-719` 的 `use_sequence_parallel_moe`：
```python
return (
    self.all2all_backend in ("allgather_reducescatter", "deepep_high_throughput",
                             "deepep_low_latency", "flashinfer_nvlink_one_sided",
                             "mori_high_throughput", "mori_low_latency", "nixl_ep")
    and self.enable_expert_parallel
    and self.tensor_parallel_size > 1
    and self.data_parallel_size > 1
)
```
→ **「TP > 1 且 DP > 1 的 EP 部署」下，vLLM 会让 MoE 的输入保持 sequence-parallel**，
避免先 all-gather 再 dispatch 的重复工作（注释 `:700-701`：
*"ensure the input to the experts is sequence parallel to avoid the excess work"*）。
**这是 SP 思想渗透到 MoE 路径的例子 —— SP 不只是 attention 的优化，而是一个通用的「别过早 all-gather」原则。**

---

### Q49 `[系统]` Context Parallel / DCP / PCP 在 vLLM 里各是什么角色？

**答**：三个不同的东西，**共同点是「沿序列维切分」，区别在切谁、什么时候切**。

✅ **三者的定义（直接引配置的 docstring）**：

| | 全称 | 切什么 | 是否扩大 world size | 证据 |
|---|---|---|---|---|
| **PCP** | Prefill Context Parallel | **prefill 的序列** | ✅ **扩大** | `config/parallel.py:126-128` |
| **DCP** | Decode Context Parallel | **decode 的 KV cache** | ❌ **不扩大** | `config/parallel.py:351-354` |
| **CP**（泛称） | Context Parallel | 序列维（PCP/DCP 的统称） | 视实现 | `v1/worker/cp_utils.py:21-51` |

✅ **PCP 的原文**（`config/parallel.py:127-128`）：
> *"Number of ranks that split prefill sequence computation.
> **PCP expands the process world size but does not increase the KV-cache shard count.**"*

✅ **DCP 的原文**（`config/parallel.py:352-354`）：
> *"Number of ranks that shard the decode KV cache.
> **DCP does not expand the process world size. Without PCP, DCP reuses TP ranks.**
> With PCP, DCP either spans the PCP axis or the full TP x PCP block."*

→ **DCP 是「白嫖」TP 的 rank**：不新增进程，只是在 TP 组内再按序列切一次 KV cache。
**这是"在已有并行维度上做二次切分"的省钱做法**（省通信域、省显存副本）。

✅ **DCP 的通信代价与优化**（`config/parallel.py:363-372`）：
```python
dcp_comm_backend: DCPCommBackend | None = None
"""Communication backend for Decode Context Parallel (DCP).
- "ag_rs": AllGather + ReduceScatter (existing behavior)
- "a2a": All-to-All exchange of partial outputs + LSE, then combine with Triton kernel.
  Reduces NCCL calls from 3 to 2 per layer for MLA models.
"""
```
→ **每层 3 次 NCCL 调用 → 2 次**。为什么能省一次？
因为 attention 是 `softmax(QK^T)V`，**分片后每个 rank 只算了一部分 attention**，
合并时需要「**部分输出 + LSE（log-sum-exp）**」来正确重归一化 ——
`a2a` 把「部分输出」和「LSE」打包在一次 all-to-all 里，
省掉了 `ag_rs` 里额外的 LSE 交换。

✅ **DCP 的硬性实现要求**（`v1/worker/cp_utils.py:44-51`）：
```python
if dcp_size > 1:
    assert layer_impl.need_to_return_lse_for_decode, (
        "Decode Context Parallelism (DCP) requires attention implementations to return
        the softmax LSE during decode, but ... does not. Try a different backend ...")
```
→ **注意力后端必须支持返回 LSE**，否则 DCP 直接不可用。
**这解释了为什么 DCP 主要在 MLA（DeepSeek 系）上成熟 —— 它的 backend 本来就返回 LSE。**

✅ **PCP 和 all2all 的联动**（`config/parallel.py:722-727`）：
```python
def use_all2all(self) -> bool:
    return (self.data_parallel_size > 1
            or self.use_sequence_parallel_moe
            or (self.enable_expert_parallel and self.prefill_context_parallel_size > 1))
```
→ **只要开了 PCP + EP，就自动启用 all-to-all**，因为专家已经在 PCP 维度上分片了。

📖 **三个概念的关系图**（记住这个就够了）：
```
                    ┌── PCP：切 prefill 的序列（新增 rank，长 prompt 的首 token 延迟优化）
序列维切分（CP）────┤
                    └── DCP：切 decode 的 KV cache（复用 TP rank，长上下文 decode 的显存优化）

维度：切权重叫 TP；切层叫 PP；切 batch 叫 DP；切专家叫 EP；切 KV cache 叫 DCP
```

**追问**：「PCP 和 PD 分离能一起用吗？」
→ 能，但有约束：见 `nixl/base_worker.py:767` 和 `config/vllm.py:1162-1177` 的校验，
**NixlConnector 的 PCP 要求 `decode_context_parallel_size == 1`**。
**「并行维度的组合爆炸」是真实工程里的主要复杂度来源。**

---

### Q50 `[进阶]` PD 分离为什么需要 KV 传输？NIXL / Mooncake / LMCache 怎么选？

**答**：PD 分离把 prefill 和 decode 放到**不同实例**上，那么 prefill 算出来的 KV cache
必须**通过网络送到 decode 实例** —— 这就是 KV 传输的存在理由。

✅ **为什么值得做**（`docs/features/disagg_prefill.md:10-13` 原文）：
> *"**Tuning time-to-first-token (TTFT) and inter-token-latency (ITL) separately.**
> ... This gives you the flexibility to assign different parallel strategies (e.g. `tp` and `pp`)
> to tune TTFT without affecting ITL...
> **Controlling tail ITL.** ... Chunked prefill with a proper chunk size also can achieve the
> same goal, but in practice it's hard to figure out the correct chunk size value.
> So disaggregated prefilling is a much more reliable way to control tail ITL."*

⚠️ ✅ **但官方也明确说了**（`docs/features/disagg_prefill.md:16`）：
> **"Disaggregated prefill DOES NOT improve throughput."**

→ **这句话是这题的核心得分点**：PD 分离买的是**延迟可控性**，不是吞吐。
如果不加这句，答案就是"背了优点没背代价"。

✅ **三层抽象**（`kv_transfer/README.md:9-13`）：

| 层 | 做什么 | 关键 API |
|---|---|---|
| **KV pipe** | FIFO 的 tensor 通道 | `send_tensor` / `recv_tensor` |
| **KV lookup buffer** | 按 token 查 KV 的 buffer | `insert` / `drop_select`（类 SQL 语义）|
| **KV connector** | 接进 vLLM | `send_kv_caches_and_hidden_states` / `recv_...` |

✅ **为什么不能只用 FIFO**（`README.md:15`，**这段原文非常精辟**）：
> *"FIFO pipe itself is not enough as prefill vLLM worker may process requests in a
> different order compared to decode vLLM worker. Say the QPS is really high, prefill worker
> may handle requests in order A -> B -> C, but the decode worker may process request C first.
> This is not the case that can be naturally handled by FIFO pipe, so we provide KV lookup
> buffer to help translate a FIFO pipe to a lookup buffer."*
→ **「FIFO → Lookup」这个转换是 KV 传输设计的核心洞察**：两个实例的调度顺序不同步。

✅ **传输后端的选择（2025 年的现实）**：

| Connector | 传输底座 | 定位 | 关键特征 |
|---|---|---|---|
| **NixlConnector** | **NIXL**（UCX/GDS 等可插拔） | 通用 PD 分离 | **pull（READ）为主**，也有 push 模式；异步 send/recv |
| **MooncakeConnector** | **Mooncake Transfer Engine** | 高性能 PD 分离 | **P-push**（P 主动 WRITE）；有 bootstrap 服务做发现 |
| **LMCacheConnectorV1** | 外部 LMCache | **前缀缓存复用** | 不只是 PD 分离，还能跨请求复用 KV；有 MP 模式（独立 server）|
| **MoRIIOConnector** | MoRI-IO | **ROCm 平台** | AMD 对应方案 |
| **OffloadingConnector** | 本地 CPU/文件 | **单实例 KV 卸载** | 不是 PD 分离，是「显存不够时把 KV 换出去」|
| **MultiConnector** | 组合 | 级联多种 | 比如「先查 LMCache，miss 就走 NIXL」 |

**怎么选（决策表）**：
```
① 只想做 PD 分离，要通用 → NixlConnector（生态最广，NIXL 可换 backend）
② 要极致传输性能 + 愿意运维 → MooncakeConnector（专用传输引擎）
③ 要跨请求复用前缀（不只是 PD） → LMCacheConnectorV1
④ 是 AMD 平台 → MoRIIOConnector
⑤ 只是显存不够，不拆实例 → OffloadingConnector
⑥ 要多级存储（GPU→CPU→文件） → MultiConnector 组合
```

**面试加分**：
> 「KV 传输的本质是**「一个非阻塞的、带 key 语义的 RDMA 服务」**。
> 所以判断一个 KV 传输方案好不好，就看三件事：
> **① 它怎么处理 P/D 顺序不一致（lookup vs FIFO）；
> ② 它怎么和计算重叠（逐层 save/load，还是整块传完再 decode）；
> ③ 它怎么保证不泄漏/不提前释放（lease/引用计数）。**
> vLLM 的 `KVConnectorBase_V1` 把这三件事都抽象成了接口
> （`base.py:289-343` 的 `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save`）。」

---

### Q51 `[进阶]` KV 传输的 push 和 pull 有什么区别？side channel 握手是干什么的？

**答**：**pull = 消费者去读（NIXL READ）；push = 生产者主动写（NIXL WRITE）。**
区别不在传输方向，而在**谁掌握块的生命周期**。

✅ **官方定义**（`docs/design/nixl_kv_push_connector.md:3-7`）：
> *"The default NIXL connector is **pull-based**: the decode (D) instance reads KV blocks
> from the prefill (P) instance via `NIXL READ` after prefill completes.
> `NixlPushConnector` adds a **push-based** alternative in which P writes the KV blocks
> directly into D's pre-allocated memory via `NIXL WRITE`."*

✅ **对比表**：

| | **Pull**（NixlConnector） | **Push**（NixlPushConnector） |
|---|---|---|
| 谁发起 | **D 发起 READ** | **P 发起 WRITE** |
| 内存注册 | P 的 buffer 要暴露给 D | **D 的 buffer 要暴露给 P** |
| 谁先知道对方 | D 先知道 P 的块 | D 先发 `PUSH_REG` 注册自己的块 |
| 调度耦合 | D 决定何时读（灵活） | P 决定何时写（P 需要 lease）|
| 线程模型 | 引擎主线程 | **专用 `nixl-push-writer` 后台线程**（`:72-74`）|
| 一致性风险 | D 读时 P 不能释放 | **P 写时 D 的块不能被回收** |
| 代码 | `nixl/pull_worker.py` | `nixl/push_worker.py` |

✅ **为什么 push 需要一个后台线程**（`push_worker.py:3-5`）：
> *"A dedicated `nixl-push-writer` thread owns all push-related NIXL ops"*

因为 push 是**事件驱动**的：P 的块算完（一个事件）和 D 的注册到达（另一个事件）
**任意顺序到达，任意时间发生**。放在引擎主线程里会阻塞出 token。
**「异步、事件驱动、不可预测到达 → 专用线程 + 单消费者队列」** 是标准解法。

✅ **side channel 握手做什么**（`kv_connector/v1/base.py:132-138`）：
```python
class KVConnectorHandshakeMetadata(ABC):
    """
    Metadata used for out of band connector handshake between
    P/D workers. This needs to serializable.
    """
```
→ **「out of band（带外）」** 是关键：**KV 数据走 RDMA，但「我要传哪些块、你的地址是什么」
这些控制信息走另一条通道（ZMQ/TCP）**。

为什么必须带外？
1. RDMA 是**单边操作**，它只搬数据，**不传语义**；
2. 控制信息量小但要求可靠（可靠用 TCP 更简单）；
3. 控制通道可以**独立重试/超时**，不影响数据面。

✅ **vLLM 里的具体实现**（以 Mooncake 为例）：
```python
self.side_channel_port: int = 0   # mooncake_connector.py:938  "we will bind it in register_kv_caches()"
...
sock = self.async_zmq_ctx.socket(zmq.ROUTER)
self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")   # :1121-1122
```
→ **ROUTER socket + 随机端口**：每个 worker 自己选端口，
再通过 **bootstrap server**（`:983-987`）告诉对端 ——
**「先随机绑定，再通过中心服务做服务发现」** 是分布式系统的标准套路。

✅ **push 模式额外的生命周期管理**（`push_scheduler.py`）：
- **D 侧 watchdog**：注册后等不到 push 完成就超时丢弃（`:190-191`、
  `_push_registration_deadlines`）；
- **P 侧 lease**：块在 `_kv_lease_duration` 内没被取走就回收（`docs/.../nixl_kv_push_connector.md:219-224`）；
- **`has_pending_push_work`**（`base.py:586-596`）：让引擎主循环在还有 push 在飞时
  **继续 step**（否则调度器以为没事干就停下来了，push 永远发不出去）。
  ⚠️ 代码里有 `TODO`：*"replace with a more general connector hook for keeping the scheduler
  alive"* —— **能指出这个 TODO 是强信号**。

**追问**：「什么时候必须用 push？」
→ 当 **P 比 D 更清楚「数据什么时候准备好」**，且不希望 D 轮询时。
典型场景：P 的 prefill 是 chunked 的，每算完一层就能推一层，
push 让 P 主动驱动，**D 不需要为「数据到没到」做任何判断** ——
这在 decode 的延迟敏感路径上很有价值。

---

### Q52 `[系统]` RL 场景的权重同步为什么用 broadcast，而不是循环 send？

**答**：因为**broadcast 是「一次通信、N 个接收者」的 O(1) 下发**，
而循环 send 是 **O(N) 次点对点**，在 N 大时线性变慢；而且 broadcast 天然保证**所有 rank 拿到同一份**。

✅ **vLLM 的实现**（`distributed/weight_transfer/nccl_engine.py:116-123` 的类 docstring）：
> *"This implementation uses **NCCL broadcast operations** to transfer dense checkpoint-format
> weights from the trainer (rank 0) to all inference workers in a process group.
> Received weights are loaded via the model's `load_weights` using the layerwise reload lifecycle."*

✅ **拓扑设计**（`nccl_engine.py:54-59, 296`）：
```python
class NCCLTrainerInitInfo(TrainerInitInfo):
    """...
    The sender opens its endpoint as NCCL rank 0, so it needs no `rank_offset`.
    ..."""
...
# Workers sit at rank_offset 1, after the single trainer sender rank 0.
```
→ **一个 NCCL 组里：rank 0 = trainer，rank 1..N = inference workers。**
broadcast 的 root 就是 0。**这个设计让「训练进程」和「推理进程」共用一个通信组**，
避免了「trainer 循环发 N 次」的代码。

✅ **为什么 broadcast 更优（讲清三条）**：

| 维度 | 循环 send（N 次 P2P） | **broadcast（1 次集合）** |
|---|---|---|
| 通信步数 | `N` 次（线性） | **树形 ≈ `log N` 步**（NCCL 内部用 tree/ring）|
| 发送端 CPU 开销 | N 次下发 | 1 次下发 |
| 一致性 | 需要自己保证每个 rank 都收到 | **集合语义天然保证** |
| 与计算重叠 | 难（逐个等待） | 容易（一次 enqueue，异步完成）|
| 带宽利用 | 发送端网卡出口是瓶颈 | **树上并行**，能打满 |

✅ **两个很硬的工程细节（加分点）**：

1. **packed broadcast**：多个小 tensor 可以打包成一个大 buffer 再播
   （`nccl_engine.py:210-218` 的 `packed_nccl_broadcast_consumer`；
   `:378-386` 的 `packed_nccl_broadcast_producer`）。
   **因为小消息的固定开销占比极高** —— 一个 16 KiB 的权重和一次 broadcast 的
   μs 级开销相比，协议开销可能比数据还大。**「攒大块再发」是通信优化的第一性原则。**

2. **非连续 tensor 必须先 `contiguous()`**（`nccl_engine.py:390-394`）：
   ```python
   # NCCL sends `numel` elements straight from `data_ptr()`, so a
   # non-contiguous view would ship whatever follows its base pointer.
   send = tensor if tensor.is_contiguous() else tensor.contiguous()
   ```
   **⚠️ 这是一个「如果不注释就一定会有人踩」的坑**：NCCL 只认 `data_ptr()` + `numel`，
   不认 stride。传一个 transpose 后的 view 会**静默发出错误的数据**（不报错！）。
   **面试时主动说出这个坑，说明你真的写过通信代码。**

3. **必须并发发起**（`nccl_engine.py:341-345`）：
   ```python
   # update_weights (workers receive) must run concurrently with the
   # trainer-side broadcast — both rendezvous inside the same NCCL calls.
   future = exe.submit(self.client.update_weights, asdict(update_info))
   ...
   self._broadcast(source, meta)
   future.result()
   ```
   **trainer 的 broadcast 和 worker 的 receive 必须在同一时刻 rendezvous**，
   所以要用线程池让 RPC 下发和 broadcast **并发**进行。

✅ **vLLM 的其他权重同步后端**（说明这题的视野）：

| backend | 传输方式 | 场景 |
|---|---|---|
| `nccl` | NCCL broadcast（本文）| 全量同步，跨机 |
| `ipc` | **CUDA IPC**（同机共享显存）| trainer 和 inference 同机 → 零拷贝 |
| `sparse_nccl` | 只传变化的参数 | 稀疏更新（部分层/部分专家变了）|
| `sharded_rdt` | **NIXL/Ray 的按需 pull** | 每卡只拉自己需要的那一片 |

→ **「全量 broadcast 是最简单但不是最省的」**：`sharded_rdt` 的定位就是
*"transports only the slice each worker needs"*（`sharded_rdt_engine.py:272-275`），
**pull 模式 + 分片，避免每张卡都收全量权重。**

**追问**：「为什么 RL 的权重同步和训练的反向传播不一样？」
→ 因为**方向和数据量都不同**：
训练反向是 reduce-scatter/all-reduce（**聚合梯度，每 layer 一次**）；
RL 的权重同步是 **broadcast/点播（分发权重，每 N 步一次）**。
**一个是高频小消息、一个是低频大消息** —— 优化手段完全不同（前者重延迟、后者重带宽）。

---

### Q53 `[系统]` 训练和推理的通信模式有什么本质差异？

**答**：**训练是「梯度聚合 + 激活重算」，推理是「KV cache 搬运 + 专家路由」。**
一句话：**训练通信是 all-reduce 主导，推理通信是 all-to-all 主导。**

| 维度 | 训练 | 推理（vLLM） |
|---|---|---|
| **主要集合操作** | all-reduce / reduce-scatter / all-gather | **all-to-all（MoE）** + all-reduce（TP）|
| **通信对象** | **梯度**（和参数量同阶）| **激活 / KV cache**（和 token 数同阶）|
| **每步通信量** | 与模型参数量成正比（与 batch 无关）| **与 batch/token 数成正比** |
| **瓶颈** | **带宽**（梯度是全量同步）| **延迟**（decode 每步只有几十 μs）|
| **并行主轴** | DP + TP + PP + **ZeRO/FSDP 分片** | TP + **EP** + DP；PP 较少 |
| **典型消息大小** | MB – GB | **KB – 百 KB** |
| **能否重叠** | 靠 PP 的流水（bubble）| 靠 DBO / chunked prefill / PD 分离 |
| **优化目标** | MFU（吞吐）| **TPOT / TTFT / p99** |
| **谁主导优化** | 框架（Megatron/DeepSpeed）+ 编译器 | **通信库选型 + 调度策略** |

📖 **差异的三个根因**：

1. **状态不同**：训练要保存 optimizer state（显存 = 参数的 12–16 倍），
   所以必须用 ZeRO/FSDP 把参数和梯度**切碎**，通信变成「每层 RS + AG」；
   推理只需要权重（2 字节/参数）+ KV cache，**参数不用切得那么碎**。

2. **批处理方式不同**：训练是**固定大 batch、全同步**（每步一次全局 barrier）；
   推理是**连续批处理（continuous batching）**，每个请求的长度和到达时间都不同
   → **通信量是动态的、不对称的**（这正是变长 all-to-all 的来源，见 Q31）。

3. **延迟预算不同**：训练的每一步是 ms 级（可以等通信）；
   decode 的每一步是**几十 μs**（通信来不及就吃掉整个 budget）。
   → **这是为什么推理必须用「0 SM 的 RDMA」和「通算重叠」，而训练更关心总带宽。**

✅ **vLLM 里能看到的「推理特有」的通信形态**：

| 形态 | 训练有吗 | vLLM 证据 |
|---|---|---|
| 变长 all-to-all | ❌（训练用固定 shape 的 EP 更常见）| `all2all.py:60-68`（sizes 元数据）|
| **KV cache 传输** | ❌ 完全不存在 | `distributed/kv_transfer/` 整个模块 |
| **0 SM 的 RDMA 通信** | 少见 | `all2all.py:345-347` |
| **CUDA Graph 内的通信** | ❌ 训练不用 graph | `pynccl_wrapper.py:4-12` |
| **权重广播（RL）** | — | `distributed/weight_transfer/` |
| batched sampling 的通信 | — | `enable_batch_sharded_sampling`（`config/parallel.py:169-175`）|

**面试加分**：
> 「如果我要猜一个优化方向，会先问『是训练还是推理』：
> **训练看 MFU，主导项是梯度同步的带宽；推理看 TPOT，主导项是每步通信的延迟。**
> 所以同一个问题（比如『要不要用 SHARP』）在两个场景下答案可能相反 ——
> SHARP 对训练的大 all-reduce 有用，对推理 decode 的小消息 + all-to-all 就没那么重要。
> **先分清场景再谈优化，比背结论重要。**」

---

## 第十二部分：更多追问链（面试官的真实打法 · 续）

> 第五部分给了 4 条链，这里再给 4 条**新**的。注意每条链的最后一问都是
> **「你怎么验证」或「为什么这么设计」** —— 这才是真正的评分点。

### 链 E：从「MoE 负载不均」挖到「EPLB 与独立 process group」

```
Q: MoE 为什么能用 EP 扩展到几百卡？
 └→ Q: 那如果某些专家特别热门会怎样？（考 straggler：一个 rank 拖慢整个 all-to-all）
     └→ Q: 怎么量化这个不均匀？（考「每 rank token 数的分布」+ p99 而不是均值）
         └→ Q: vLLM 怎么解决？（考 EPLB）
             ├→ Q: EPLB 的算法是什么？
             │    答：复制热门专家（replicate_experts）+ 贪心装箱（balanced_packing），
             │        算法改编自 DeepSeek EPLB
             │        （✅ eplb/policy/default.py:8-13, :22-73, :75-90）
             └→ Q: 重平衡要搬权重吗？代价多大？
                  答：要。rebalance_execute.py 的头注释就是
                      "This involves the exchange of expert weights between GPUs"
                      （✅ rebalance_execute.py:1-7）
                  └→ Q: 那 EPLB 和前向的通信会不会互相阻塞？
                      答：会 → 所以 vLLM 给 EPLB 单独建 process group
                          （✅ parallel_state.py:2117-2120 的注释就是答案：
                            "prevent deadlocks when using torch.distributed in execution
                             with torch.distributed in EPLB"）
                      └→ Q: 那 EPLB 在哪个线程跑？会不会卡住前向？
                          答：后台线程（eplb/async_worker.py），
                              并且用双缓冲 + CpuGpuEvent 做中间 buffer 握手
                              （✅ rebalance_execute.py:58-64 的 consumed_event 注释）：
                              async worker 写完 wait()，主线程搬完 record()
```

**这条链的最后一问是「并发正确性」** —— 能答出来说明你不只会调 API。

---

### 链 F：从「DBO」挖到「SM 仲裁与为什么 ROCm 要关掉」

```
Q: 想优化 decode 的 MoE，有什么手段？
 └→ Q: 什么是 DBO？（考 dual batch overlap + 两个 ubatch ping-pong）
     └→ Q: 两个 ubatch 是不是各有一对 stream？
        答：❌ 不是！两个 ubatch **共用同一个 compute stream 和同一个 comm stream**
            （✅ ubatching.py:226-239 的 make_ubatch_contexts 传的是同一个对象）。
            真正的「双份」是双线程 + 双 ForwardContext + 双 event。
        └→ Q: 那 CPU 侧怎么保证不打架？
            答：_cpu_yield 里有三个 assert，保证「一次只有一个线程在跑」
                （✅ ubatching.py:94-105）
            └→ Q: GPU 侧通信和计算真的并行了吗？靠什么？
                答：跨 stream 的 event：_signal_compute_done / _wait_compute_done
                    （✅ ubatching.py:82-92, :133-147）
                └→ Q: 通信和计算抢 SM 怎么办？
                    答：SMControlContextManager 显式划分：
                        comm 用 VLLM_DBO_COMM_SMS（默认 20），compute 用剩下的
                        （✅ ubatch_utils.py:39-81；envs.py:287）
                    └→ Q: 为什么只有 DeepGEMM 和 DeepEP HT 支持？
                        答：因为只有它们的 kernel 支持 set_num_sms
                            （✅ ubatch_utils.py:101-103 的注释 + :118-121 的 TODO）
                        └→ Q: 那 ROCm 上呢？
                            答：DeepEP HT + DBO 时 comm_sms 被强制设成 0，
                                因为「reserving CUs ... corrupts DP+EP generation accuracy」
                                （✅ ubatch_utils.py:89-98），
                                并且额外加 stream sync 保护 Buffer-owned workspace
                                （✅ deepep_ht.py:73-78）
                            └→ Q: 你怎么验证 DBO 真的生效了？
                                答：① nsys 看 dispatch/combine 和计算 kernel 时间轴是否交叠；
                                    ② 开关 DBO 对比 p50/p99 TPOT；
                                    ③ 极端 case：batch 刚好等于阈值时的行为
```

**这条链的最后一问又回到「验证」** —— 而倒数第二问（ROCm）是**平台工程**的加分点。

---

### 链 G：从「KV 传输」挖到「PD 分离的收益边界」

```
Q: PD 分离是什么？为什么要做？
 └→ Q: 那 prefill 算出来的 KV cache 怎么给 decode？（考 KV 传输）
     └→ Q: KV 传输为什么不能只用一个 FIFO pipe？
        答：因为 P 和 D 处理请求的顺序可能不同（P: A→B→C，D 可能先要 C），
            所以需要 lookup buffer 把 FIFO 翻译成按 key 查询
            （✅ kv_transfer/README.md:15）
        └→ Q: 三个抽象层分别是什么？（KV pipe / lookup buffer / connector）
            （✅ README.md:9-13）
            └→ Q: NIXL 的 push 和 pull 有什么区别？
                答：pull = D 发 NIXL READ；push = P 发 NIXL WRITE
                    （✅ docs/design/nixl_kv_push_connector.md:3-7）
                └→ Q: push 模式为什么要一个后台线程？
                    答：事件驱动，P 的块完成和 D 的注册任意顺序到达
                        （✅ nixl/push_worker.py:3-5）
                    └→ Q: 控制信息走哪？（考 side channel / out-of-band handshake）
                        答：KV 数据走 RDMA，元数据走带外的 ZMQ/TCP
                            （✅ kv_connector/v1/base.py:132-138；
                              mooncake_connector.py:1121-1122 的 ROUTER socket）
                        └→ Q: push 模式下块的生命周期怎么管？
                            答：D 侧 watchdog + P 侧 lease
                                （✅ push_scheduler.py:190-191；
                                  docs/design/nixl_kv_push_connector.md:206-224）
                            └→ Q: ★ PD 分离能提升吞吐吗？
                                答：**不能**。官方文档原文：
                                    "Disaggregated prefill DOES NOT improve throughput."
                                    （✅ docs/features/disagg_prefill.md:16）
                                    它买的是「TTFT 和 ITL 可以分别调」+
                                    「尾延迟可控」，代价是 KV 传输。
                                └→ Q: 那什么情况下 PD 分离不划算？
                                    答：① prompt 很短（KV 传输的固定开销摊不掉）；
                                        ② 网络带宽小（KV 传输成为新瓶颈）；
                                        ③ 单实例本来就能满足 SLO 时纯属增加复杂度
```

**最后一个答案就是「收益边界」**：**PD 分离用吞吐换延迟可控性**，
所以在「延迟已经达标」或「KV 传输比 prefill 还慢」时是负优化。

---

### 链 H：从「sequence parallel」挖到「通信量减半的原理」

```
Q: TP 的 all-reduce 能不能优化？
 └→ Q: all-reduce 后面通常跟着什么操作？（RMSNorm / 量化）
     └→ Q: 那能不能先规约再算，然后扩散？（考 RS + local op + AG）
         └→ Q: 这样通信量变了吗？
            答：没变！ring all-reduce ≈ 2S，RS+AG 也 ≈ 2S
            └→ Q: ★ 那「通信量减半」是怎么来的？
                答：**来自量化前移**。看 fp8 的 pattern：
                    pattern 是 AR → RMSNorm → Quant
                    replacement 是 RS → RMSNorm → Quant → AG
                    所以 AllGather 传的是 **fp8（1 字节）而不是 bf16（2 字节）**
                    （✅ sequence_parallelism.py:282-297）
                    nvfp4 版本能到 ~1/4（✅ :411-422）
                └→ Q: 那这个 pass 单独开有收益吗？
                    答：**几乎没有**。docstring 原文：
                        "While this pass itself does not directly yield performance
                         improvements, it lays the groundwork for subsequent fusion
                         passes, such as GEMM + ReduceScatter and AllGather + GEMM
                         fusions."（✅ sequence_parallelism.py:511-515）
                    └→ Q: 那为什么不直接做融合？
                        答：因为融合需要「可融合的图结构」。
                            AR → RMSNorm 里 AR 是全局同步点、前后都是完整张量，
                            没法融；RS → local → AG 才能让 GEMM→RS 和 AG→GEMM
                            各自成为一个融合单元。
                        └→ Q: 什么时候不该开 SP？
                            答：① hidden_size < 8192（H100）—— 模型太小
                                （✅ sequence_parallelism.py:40-43）
                                ② token 数 < min_token_num
                                   （H100/TP8/hidden8192 算出来是 4096，
                                    Blackwell 是 16384；✅ :102-104）
                                ③ piecewise 编译模式（SP 要 fullgraph）
                                   （✅ :517-521）
                                ④ num_tokens 不能被 TP 整除
                                   （✅ gpu_model_runner.py:4099-4107 的 assert）
```

**这条链的关键是倒数第三问**：
面试官想看你**能不能承认「这个 pass 本身不赚钱」** ——
敢说「这个优化单独用没有收益，价值在于为后续融合铺路」，
比硬吹「SP 让通信减半」要可信得多。

---

## 附：面试前 30 分钟速查

```
网络：  400Gb/s = 50 GB/s | NVLink4 双向900/单向450 | PCIe5 x16 ≈ 64 GB/s
        延迟：NVLink 亚μs | IB 2-5μs | TCP 10-30μs
        网内规约：NVLS(NVSwitch) / CollNet / IB SHARP —— 步数 2(N-1)→≈2，
                  **延迟大赚、带宽不赚**；对确定性有害（NCCL_COLLNET/NVLS_ENABLE=0）
        RoCE 无损 = PFC + ECN + DCQCN（运维配出来的）；IB 无损是协议自带的
        BAR1：数据中心卡 64GiB / 消费卡 256MiB；P2P 要过 ACS/IOMMU 关
集合：  ring 2(N-1)步 / 2(N-1)/N×S | tree 2logN步 | all-reduce = RS + AG
        变长集合：N 次 root 不同的 broadcast = all-gatherv（pynccl.py:293-326）
                  / N 次 ncclReduce = reduce_scatterv（pynccl.py:356-391）
vLLM：  布局 [DP,PP,PCP,TP] | EP = DP×PCP×TP | TP 组相邻
        8 路 AR：symm→QR→FI-PCIe→FI→AITER→custom→torch-symm→pynccl→torch
        custom AR：≤8 卡，H100/TP8 上限 256KiB，需 NVLink 全互联
        控制面：shm + ZMQ（不是 NCCL）
MoE：   dispatch=AG / combine=RS（默认 allgather_reducescatter，all2all.py:101/138）
        DeepEP：HT(prefill,20SM,bf16 only) / LL(decode,0SM,hidden∈8个枚举)
                / v2(需 NCCL≥2.30.4+GIN, decode 可 graph / prefill do_cpu_sync)
        expert 坐标：global / local / physical；expert_map 的 -1 = 不在本 rank
                     HT 把 -1 换成「rank0→N-1，其他→0」（deepep_ht.py:227-231）
        对齐：HT/v2 要 hidden×itemsize % 512 == 0（2880→3072，deepep_ht.py:43）
              LL 要 hidden ∈ {2048,2560,3072,4096,5120,6144,7168,8192}（deepep_ll.py:61）
        量化：block-quant 先量化再 dispatch（省一半带宽）；
              swizzle 必须在 a2a 之后（naive_dp_ep.py:32）
        EPLB：复制+贪心装箱；必须独立 process group（parallel_state.py:2117-2120）
DBO：   dual batch overlap，只支持 DP+EP+DeepEP+FULL graph（docs/design/dbo.md）
        两个 ubatch 共用同一 compute stream 和同一 comm stream（ubatching.py:226-239）
        SM 仲裁：comm=VLLM_DBO_COMM_SMS(默认20)，compute=total-comm（ubatch_utils.py:39-127）
                 只有 DeepEP HT（set_num_sms）+ DeepGEMM 支持；LL 的 max_sms_used()=0
                 ROCm+HT 强制 comm_sms=0（正确性）（ubatch_utils.py:89-98）
        阈值：decode 32 / prefill 512，都必须 ≥ num_ubatches（parallel.py:225-234, 1070）
CP/SP： SP = AR→RMSNorm 改写成 RS→RMSNorm→AG；通信量减半来自量化前移（AG 传 fp8）
        阈值：H100 hidden≥8192 且 token≥4096（TP8/H8192 算出来的）；要 fullgraph
        DCP 不扩 world size（复用 TP rank）；PCP 扩 world size 但不增加 KV shard
        DCP 要求 attention 返回 LSE；ag_rs→a2a 可把每层 NCCL 从 3 次降到 2 次
KV：    PD 分离 不提升吞吐（docs/features/disagg_prefill.md:16），只换延迟可控性
        三层抽象：KV pipe / lookup buffer / connector（FIFO→按 key 查）
        NIXL 默认 pull(READ)，NixlPushConnector 是 push(WRITE)，用专用 writer 线程
        数据走 RDMA，元数据走带外 ZMQ side channel
RL：    权重同步用 broadcast（rank0=trainer，workers 从 1 开始）；
        非连续 tensor 必须先 contiguous()（NCCL 只认 data_ptr+numel）
排障：  NCCL_DEBUG=INFO | 三一致（成员/顺序/形状）| 先 profile 再下结论
        新排障点：SM 争抢（DBO）、PFC pause 计数（RoCE）、EPLB 是否真的在搬
```

**下一章**：[`appendix-code-tour.md`](appendix-code-tour.md) —— 代码地图与读数路径。
