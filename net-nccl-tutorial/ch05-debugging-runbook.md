# 第 5 章　排障手册：hang、报错、带宽不达标

> 本章是**操作手册**，不是理论。目标：给你一套可以在面试里复述的**诊断流程**，
> 以及一组可以直接粘贴的命令。
>
> 所有 vLLM 侧的环境变量/代码引用都可核对；纯 NCCL 的变量标注为「NCCL 官方」。

---

## 5.0 第一原则：集合通信的三个「必须一致」

**90% 的分布式 hang 都源于这三个不一致之一。** 背下来，面试直接能用：

| # | 必须一致的东西 | 不一致的后果 |
|---|---|---|
| 1 | **谁参与了这次通信**（进程组的成员集合） | 有的 rank 在等一个永远不来的对端 → **hang** |
| 2 | **操作的顺序和类型**（rank A 做 all-reduce，rank B 做 all-gather） | NCCL 内部状态错位 → **hang 或数据错乱** |
| 3 | **张量的形状和 dtype** | 死锁（长度对不上）或**静默的垃圾数据** |

**推论**：任何「条件分支里做通信」的代码都是危险的。比如：

```python
# 危险！如果 rank 0 走 if 而 rank 1 走 else，就 hang
if rank == 0:
    tensor = all_reduce(x)
else:
    tensor = all_gather(y)
```

**vLLM 里能看到这个原则被严格遵守**：所有 gate 判断（比如
`should_custom_ar`、`should_nccl_symm_mem_allreduce`）**都基于输入张量的大小/形状**，
而这些在所有 rank 上**相同**。所以「有的 rank 用 custom AR、有的用 NCCL」不会发生。

反例（真实存在的陷阱）：**基于 per-rank 不同状态做通信决策**。
vLLM 在 `cuda_communicator.py:477-479` 有一段注释专门讲这个：

> *"ncclCommWindowRegister is collective: **asymmetric pool allocations from variable per-rank
> sizes cause deadlocks**."*

→ 中文：对称内存的 window 注册是**集合操作**，如果各 rank 分配的显存池大小不同，
注册调用就会死锁。**所以文档说要「对称」不只是为了性能，是为了不死锁**。
这是一个极好的面试素材：**「为什么叫对称内存」有两层含义 —— 性能上的地址对称，和正确性上的操作对称。**

---

## 5.1 三步分流：先定位是哪一层

```
① 进程起来了吗？
   ├─ 没起来 → 引导/端口/DNS 问题        → §5.2
   └─ 起来了
      ↓
② 卡在哪一步？
   ├─ 卡在 NCCL 初始化（建连）            → §5.3
   ├─ 卡在第一次前向（第一个集合操作）     → §5.4
   └─ 跑着跑着卡住 / 报错                 → §5.5
      ↓
③ 能跑但慢？
   └─ 带宽/延迟不达标                     → §5.6
```

**第一步永远是打开日志**：

```bash
# 最有用的一条命令
export NCCL_DEBUG=WARN          # 生产：只看警告
export NCCL_DEBUG=INFO          # 排障：看拓扑、算法、协议、传输选择
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING   # 只看子系统，避免刷屏
export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log       # 多机必备：%h=主机名 %p=pid
```

**vLLM 自己也会打印关键信息**：

- `vLLM is using nccl==X.Y.Z`（`vllm/distributed/device_communicators/pynccl.py:117-120`，
  仅 rank 0，只打一次）→ **确认实际加载的 NCCL 版本**。
- `Using <X> all2all manager.`（`cuda_communicator.py:227-231`）→ 确认 all-to-all 后端。
- `rank X in world size Y is assigned as DP rank ..., PP rank ..., TP rank ..., EP rank ...`
  （`parallel_state.py:2142-2154`）→ **确认并行组划分符合预期**。
- all-reduce 后端选择日志（`_log_all_reduce_backend_selection`）→ **注意它只是「可能的子集」，
  不是优先级顺序**（见第 4 章 §4.4.1）。

---

## 5.2 进程起不来 / 引导失败

### 症状 A：`Address already in use`

**根因**：端口冲突。vLLM 的端口来源：
- `data_parallel_master_port` 默认 **29500**（`vllm/config/parallel.py:147`）
- `data_parallel_rpc_port` 默认 **29550**（`vllm/config/parallel.py:143`）
- `master_port` 默认 **29501**（`vllm/config/parallel.py:282`）
- `VLLM_PORT`（`vllm/envs.py:720`，未设则随机）

**vLLM 已经做了递增分配**（`vllm/config/parallel.py:600-613` 的 `get_next_dp_init_port`），
但**引擎进程和 worker 进程会分别建组**，所以仍可能冲突。

**诊断**：

```bash
ss -ltnp | grep 295       # 看谁占了 295xx
python -c "
from vllm.utils.network_utils import find_process_using_port
print(find_process_using_port(29500))
"
```

**修复**：显式指定不冲突的端口：

```bash
export VLLM_PORT=39600
vllm serve ... --data-parallel-rpc-port 39650 --master-port 39601
```

### 症状 B：`VLLM_PORT ... appears to be a URI`

`vllm/envs.py:540-544` 专门为这个情况写了错误信息，并且提到 **Kubernetes service discovery**：

> 在 K8s 里如果有个叫 `VLLM_PORT` 的 **Service**，环境变量可能被注入成
> `tcp://10.96.x.x:8000` 这种形式。vLLM 期望的是整数。

**修复**：`VLLM_PORT` 必须是纯数字端口。

### 症状 C：`No available node types can fulfill resource request`（Ray）

**这不是 GPU 不够**。vLLM 官方排障文档（`docs/serving/distributed_troubleshooting.md`）指出：

> *"The issue often occurs when nodes have multiple IP addresses and vLLM can't select the
> correct one. Ensure that vLLM and Ray use the same IP address by setting `VLLM_HOST_IP`
> (with a different value on each node). Use `ray status` and `ray list nodes` to verify."*

**根因**：多网卡机器上，Ray 选了一个 IP，vLLM 选了另一个，Ray 的调度器认为资源不可达。

**修复**：

```bash
# 每个节点设置成自己的、可被其它节点访问的 IP
export VLLM_HOST_IP=192.168.1.10     # 节点 1
export VLLM_HOST_IP=192.168.1.11     # 节点 2
```

**机制**：`vllm/utils/network_utils.py:34-73` 的 `get_ip()` 优先用 `VLLM_HOST_IP`；
否则用 UDP connect 到 `8.8.8.8:80` 取本机出口 IP；再不行试 IPv6；**最后兜底 `"0.0.0.0"`**
（这个兜底在多机场景基本等于坏掉）。

**一个容易被忽略的坑**：Ray 的 **V2 executor（当前默认）** 用的是
`ray.util.get_node_ip_address()` 而不是 `get_ip()`，**会绕过 `VLLM_HOST_IP`**
（`vllm/v1/executor/ray_executor_v2.py:127-139`，注释说明 `get_ip()` 可能返回不可路由地址）。

**面试题**：「vLLM 怎么确定用哪个 IP？」
→ 优先 `VLLM_HOST_IP` → UDP 探测出口 IP → IPv6 → `0.0.0.0`；Ray 路径另有机制。
**多网卡环境必须显式设置。**

### 症状 D：单机多卡的引导到底走什么协议？

**一个反直觉的事实**：普通的**单机 TP>1** 路径用的是 **FileStore（`file://`）**，
不是 TCP！

```python
# vllm/v1/executor/multiproc_executor.py:143
distributed_init_method = get_file_store_init_method()
```

`get_file_store_init_method()` 返回 `f"file://{tempfile.gettempdir()}/vllm_dist_{uuid4().hex}"`
（`vllm/utils/network_utils.py:135-136`）。

**只有以下情况才走 TCP**：
- ROCm + AITER custom all-reduce（`aiter_requires_tcp_store()`）
- 多机（`nnodes > 1`）→ `tcp://{master_addr}:{master_port}`
- DP > 1 → `tcp://{data_parallel_master_ip}:{get_next_dp_init_port()}`

**为什么这样设计**：单机场景用文件系统 rendezvous **最省事**（不需要开端口、不需要网络、
不会有端口冲突），**只在必须跨机时才引入网络**。这是一个很实用的工程选择。

**面试题**：「单机 8 卡启 TP，需要开放端口吗？」
→ 默认**不需要**（走 FileStore）。多机/DP 才需要。

---

## 5.3 卡在 NCCL 初始化（建连阶段）

### 症状：日志停在 `NCCL INFO Bootstrap` / `Init COMPLETE` 之前

NCCL 的初始化分两阶段：
1. **bootstrap**：通过 socket（TCP）交换连接信息、uniqueId、拓扑数据。
2. **建连**：建立 NVLink/IB 连接，分配 channel。

卡在 1 → **网络/端口/主机名问题**。卡在 2 → **IB/RDMA/GID 配置问题**。

### 诊断清单

```bash
# 1. 主机名解析
ping -c1 $(hostname)
getent hosts $(hostname)

# 2. 节点间连通性（NCCL 的 bootstrap 端口也要通）
#    NCCL 用的端口范围由 NCCL_SOCKET_IFNAME / 内核临时端口决定
nc -zv <对端IP> <端口>

# 3. 选对了网卡吗？（多网卡环境头号问题）
export NCCL_SOCKET_IFNAME=eth0            # 或 =^docker0,lo 排除法
export NCCL_IB_HCA=mlx5_0,mlx5_1          # 或 =^mlx5_bond 排除法

# 4. RoCE 的 GID index 对不对
show_gids                                  # Mellanox 工具，看 RoCEv2 的 GID index
export NCCL_IB_GID_INDEX=3
```

**vLLM 官方文档也指向这个做法**（`docs/serving/distributed_troubleshooting.md`）：

> *"If you need additional environment variables for communication configuration, append them
> to `examples/ray_serving/run_cluster.sh`, for example `-e NCCL_SOCKET_IFNAME=eth0`.
> **Setting environment variables during cluster creation is recommended because the variables
> propagate to all nodes. In contrast, setting environment variables in the shell affects only
> the local node.**"*

**为什么强调「在集群创建时设置」**：因为 Ray 只在**启动 worker 时**把环境变量复制过去
（`vllm/ray/ray_env.py:37-44` 的 `NCCL_` 前缀复制机制）。
如果你在某个节点上 `export`，那个节点的 driver 进程有，但别的节点没有。

**vLLM 有更进一步的自动化**：`VLLM_GPU_NIC_PCIE_MAPPING` + `VLLM_NIC_SELECTION_VARS`
（`vllm/v1/executor/vllm_net_devices.py:180` 附近）可以根据 GPU 到网卡的 PCIe 距离
**自动为每个 rank 选择最近的网卡**，并写进 `NCCL_IB_HCA` / `UCX_NET_DEVICES`。

> 这是「NIC–GPU 亲和性」的自动化实现（第 1 章讲的 `nvidia-smi topo -m` 的 NIC 列）。
> **多网卡多 GPU 的机器上，手工配 `NCCL_IB_HCA` 往往会配错，让 vLLM 自动选更可靠。**
> 注意：这两个变量**必须同时设置**（代码里是成对判断的）。

### 症状：`NCCL error: unhandled system error` / `ibv_create_qp failed`

通常是 IB 资源不足或 GID 配错：

```bash
# 看 IB 设备状态
ibstat                    # 端口 state 应该是 Active
ibv_devinfo               # 看 port state / link layer
# 减少 QP 数（资源不足时）
export NCCL_IB_QPS_PER_CONNECTION=1
# 加大超时（大集群/拥塞时）
export NCCL_IB_TIMEOUT=22
```

---

## 5.4 卡在第一次前向（第一个集合操作）

**这是最常见的一类 hang**，也是最能考察「是否理解集合通信语义」的场景。

### 排查思路：确认「三一致」（§5.0）

```bash
# 1. 确认所有 rank 的 world_size / 组划分一致
#    看 vLLM 的日志：
#    "rank X in world size Y is assigned as DP rank .., PP rank .., TP rank .., EP rank .."
#    （parallel_state.py:2142-2154）
```

**如果各 rank 的 world_size 不一样** —— 这是致命的，因为它们会等不同的对端数量。

### 常见根因 1：CUDA Graph 捕获期间的通信

**症状**：第一次跑 graph 时 hang，或者报 CUDA API 相关的错。

**根因**（第 4 章讲过）：NCCL 的**懒初始化**会在 CUDA Graph 捕获期间触发 CUDA API 调用，
而捕获期间不允许这些调用。**这正是 vLLM 写 pynccl 的第一原因**
（`pynccl_wrapper.py:4-12` 的文件头注释）：

> cupy 在 comm init 时会卡住；`torch.distributed` 会调用 graph capture 期间被禁止的 CUDA API。

**vLLM 的解法**：pynccl 用 ctypes 直接 `dlopen` NCCL，**通信器在捕获前就建好**。

**验证**：

```bash
# 临时关掉 cudagraph，看是否恢复（不 hang 就说明是 graph 相关问题）
vllm serve <model> --enforce-eager
```

### 常见根因 2：DeepEP v2 的 GIN 检查失败

**症状**（明确的报错，不会 hang）：

```
RuntimeError: DeepEPv2 requires NCCL GIN (GPU-Initiated Networking).
This usually means IBGDA-capable InfiniBand NICs or drivers are not available.
```

（`vllm/distributed/device_communicators/all2all.py:1058-1064`）

**根因**：硬件/驱动不支持 IBGDA（IB GPU Direct Async），或 NCCL < 2.30.4。

**修复**：换 `--all2all-backend deepep_low_latency`（v1 LL，只要求 RDMA 不要求 GIN），
或者升级 NCCL + 换支持 IBGDA 的网卡。**注意 vLLM 在这里选择「快速失败」而不是偷偷降级** ——
因为 v2 的性能完全依赖 GIN，降级后不如直接用 v1。

### 常见根因 3：P2P 检测与驱动 bug

**背景**（第 4 章 §4.4.4）：vLLM 会实测 P2P（起子进程用 CUDA IPC 真传数据），
因为 **535 系列驱动会让 `torch.cuda.can_device_access_peer` 撒谎**。

**vLLM 的逃生舱**（`vllm/envs.py:1205-1210` 的注释）：

> *"We assume drivers can report p2p status correctly. If the program hangs when using custom
> allreduce, potentially caused by a bug in the driver (535 series), it might be helpful to set
> `VLLM_SKIP_P2P_CHECK=0`"*

**注意默认值的方向**（**面试可以提这个，显示你读过源码**）：
`VLLM_SKIP_P2P_CHECK` 的**默认是 `1`（跳过真测，信任驱动）**，
而上面那段注释说的是「如果 hang，设成 `0` 开启真检测」。

**这是一个值得讨论的设计**：真检测更可靠但要起子进程（慢），
所以默认信任驱动（快），只在出问题时才启用真检测。

### 常见根因 4：混用了不同的 backend 或消息大小

**症状**：不 hang 但结果错，或者偶发 hang。

**根因**：某些后端有**严格的对齐要求**，如果不满足会静默走别的路径。
vLLM 的对齐规则（第 4 章）：

| 后端 | 对齐要求 | 证据 |
|---|---|---|
| custom AR | `inp_size % 16 == 0` | `custom_all_reduce.py:410-413` |
| symm_mem | `inp_size % 4 == 0`（只支持 bf16） | `symm_mem.py:120-124` |
| QuickReduce | `inp_size % 16 == 0` | `quick_all_reduce.py:325-326` |

**关键**：这些 check **在所有 rank 上结果相同**（因为张量形状一致），
所以不会导致「部分 rank 走 A、部分走 B」—— **这正是 §5.0 原则 2 的体现**。

---

## 5.5 跑着跑着卡住 / 报错

### 症状：`NCCL timeout` / `Watchdog caught collective operation timeout`

**先看是不是真的慢**（而不是死锁）：调大超时看它是否最终完成。

```bash
export NCCL_ASYNC_ERROR_HANDLING=1     # NCCL 官方变量，让错误快速暴露而非静默 hang
export TORCH_NCCL_TIMEOUT=...          # torch 侧
```

**vLLM 有一个相关的重要行为**（`vllm/v1/worker/gpu_worker.py:360`）：

```python
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
```

注释说明：**这个变量由 Ray 设置，会导致 graph building 时抛异常**，所以 vLLM 在 CUDA 初始化时
主动删掉它。

**这是一个真实的框架间冲突案例**：Ray 为了故障检测设了一个变量，
但它和 CUDA Graph 的构建冲突，vLLM 必须显式清除。
**面试题**：「不同框架的环境变量会互相干扰吗？」
→ 会。这个例子说明：**框架集成时要检查对方设置的全局副作用**。

### 症状：某个 rank 输出垃圾数据（不是 hang）

**根因**：形状/dtype 不匹配但长度恰好兼容 → NCCL 不报错，数据错位。

**诊断**：

```python
# 在通信前后加校验（临时调试）
import torch
from vllm.distributed.parallel_state import get_tp_group
g = get_tp_group()
print(f"rank={g.rank} tp_rank={g.rank_in_group} shape={x.shape} dtype={x.dtype}")
# 用一个已知值做 all-reduce 验证
t = torch.full((1024,), float(g.rank), device="cuda")
r = g.all_reduce(t)
expected = sum(range(g.world_size))
assert torch.allclose(r, torch.full_like(r, expected)), f"AR broken: {r[:4]}"
```

**这是最实用的分布式调试技巧**：用「每个 rank 填自己的 rank 号，all-reduce 后应该都等于
`0+1+...+N-1`」来验证通信正确性。**面试里被问「怎么验证集合通信是对的」，这就是答案。**

### 症状：EPLB 相关的死锁

**根因**（第 4 章 §4.1.2）：EPLB 的通信和 MoE 前向的通信如果共用 process group，
会互相阻塞。vLLM 的对策是**为 EPLB 单独建一个组**（`parallel_state.py:2117-2120`）：

> *"This is a separate process group to isolate EPLB communications from MoE forward pass
> collectives and prevent deadlocks when using torch.distributed in execution with
> torch.distributed in EPLB."*

**面试题**：「多个通信流怎么避免互相阻塞？」
→ ① 用不同的 process group（vLLM 对 EPLB 的做法）；
② 用不同的 CUDA stream；
③ 用 NCCL 的 `ncclGroupStart/End` 控制并发批次。

---

## 5.6 能跑但慢：性能诊断

### 第一步：确认通信到底占了多少时间

**方法 1：Nsight Systems profile**（最权威）

```bash
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node \
  -o /tmp/vllm_prof --force-overwrite true \
  vllm serve <model> --tensor-parallel-size 8
```

看 timeline 里 `ncclAllReduce` / `ncclDevKernel_AllReduce` / custom AR kernel 的占比。

**方法 2：算理论值对比实测**（第 1 章的手算）

```
单次 all-reduce 每 rank 搬运量 ≈ 2(N-1)/N × S
理论时间 = 步数 × α + 搬运量 / 有效带宽
```

如果实测比理论慢 3 倍以上 → 有问题。

### 检查清单

| 检查项 | 命令 / 变量 | 说明 |
|---|---|---|
| **NVLink 有没有用上** | `nvidia-smi nvlink -s`、`nvidia-smi topo -m` | 如果 TP 组走了 PCIe，会很慢 |
| **P2P 是否可用** | `nvidia-smi topo -m` 看有没有 `NV#`；日志里看 custom AR 是否被禁用 | 无 P2P → custom AR 不可用 |
| **custom AR 是否生效** | vLLM 日志的后端选择；或手动 `VLLM_DISABLE_CUSTOM_ALL_REDUCE` 对比 | 如果没生效，会退到 NCCL |
| **是不是消息太大超出了 custom AR 范围** | 对照 `CUSTOM_ALL_REDUCE_MAX_SIZES` | H100 TP=8 只有 256 KiB！ |
| **网卡选择是否正确** | `NCCL_DEBUG=INFO` 看它选了哪个 HCA | 选错网卡掉一半带宽 |
| **NCCL 选了哪个算法/协议** | `NCCL_DEBUG=INFO` 里的 `NCCL INFO ... algorithm ... protocol` | 小消息应该是 tree/NVLS |
| **SHARP/NVLS 有没有开** | `NCCL_NVLS_ENABLE`（默认 1）；看日志有没有 NVLS | 关了会少一个优化 |
| **是否 batch-invariant 模式** | `VLLM_BATCH_INVARIANT` | **开了会禁用所有快速 all-reduce 和 NCCL 优化** |

**最后一行是常见的性能陷阱**：如果你为了「结果可复现」开了 `VLLM_BATCH_INVARIANT`，
vLLM 会（见第 4 章）：
- 禁用 custom all-reduce（`config/parallel.py` 里强制 `disable_custom_all_reduce`）
- 禁用 FlashInfer AR 和 PCIe IPC AR（`cuda_communicator.py:64,68`）
- 禁用对称内存（`symm_mem.py:114`、`all_reduce_utils.py:136,165`）
- **强制设置一堆 NCCL 变量**（`batch_invariant.py:1147-1156`）：
  `NCCL_ALGO=allreduce:tree`、`NCCL_PROTO=Simple`、`NCCL_NVLS_ENABLE=0`、
  `NCCL_MIN_NCHANNELS=1`、`NCCL_MAX_NCHANNELS=1`、`NCCL_NTHREADS=1`、
  `NCCL_P2P_NET_DISABLE=1`

**`NCCL_MAX_NCHANNELS=1` 这一条最致命** —— 把 channel 并发度压到 1，带宽直接掉。
**面试题**：「为什么开了确定性模式性能掉这么多？」
→ 因为确定性要求**固定的规约顺序**，而所有提高性能的手段（多 channel、
NVLS 多播、tree 之外的算法、自定义 kernel）都会改变求和顺序。
**浮点加法不满足结合律，所以「确定性」和「最快」在物理上互斥。**

### 一个具体的诊断案例（面试可以当故事讲）

**现象**：8×H100 单机，TP=8，decode TPOT 比预期慢 40%。
**排查**：
1. `nvidia-smi topo -m` → 发现 GPU 间是 `NV18`（NVLink 正常）。
2. `NCCL_DEBUG=INFO` → 看到 all-reduce 选了 **Ring** 而不是 NVLS。
3. 检查 custom AR 是否生效 → 日志显示生效。
4. Nsight 看时间分布 → 通信占比只有 15%，**不是通信瓶颈**。
5. 继续查 → 发现是 attention kernel 的问题，与网络无关。

**这个案例的教训**：**不要假设慢就是通信问题**。
先用 profiler 确认时间分布，再针对性排查。**面试里这样说比直接背 NCCL 调优参数更显成熟。**

---

## 5.7 最小复现清单（提 issue / 面试描述问题时用）

```markdown
## 环境
- GPU: 8×H100 SXM，NVLink 全互联
- vLLM 版本 / commit:
- PyTorch + NCCL 版本（`vLLM is using nccl==X.Y.Z` 那行日志）:
- 启动命令:
- 相关环境变量: `VLLM_*` / `NCCL_*`

## 现象
- 卡在哪一步（初始化 / 第一次前向 / 第 N 步）
- 完整报错栈
- `NCCL_DEBUG=INFO` 日志（`NCCL_DEBUG_FILE` 收集）

## 已排除
- [ ] 单卡能跑（`--tensor-parallel-size 1`）
- [ ] `--enforce-eager` 能跑（排除 CUDA Graph 问题）
- [ ] `VLLM_DISABLE_CUSTOM_ALL_REDUCE` 后行为变化
- [ ] `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` 后行为变化
- [ ] `nvidia-smi topo -m` 输出
```

**面试技巧**：被问「你怎么报一个分布式 bug」时，这个清单就是答案。
**关键是要展示「二分定位」的思路**：通过逐个关掉优化（eager / 禁 custom AR / 禁 P2P / 禁 IB）
来缩小范围。

---

## 5.8 本章自检题

1. 集合通信 hang 的三个「必须一致」是什么？
2. `NCCL_DEBUG=INFO` 能看到哪些关键信息？
3. 多网卡机器上 vLLM 怎么选 IP？Ray V2 路径有什么不同？
4. 单机 8 卡 TP 的引导走 FileStore 还是 TCP？为什么？
5. 怎么验证一个 all-reduce 是正确的（不去读 kernel 代码）？
6. `VLLM_SKIP_P2P_CHECK` 的默认值是什么？什么时候要改？
7. 为什么 DeepEP v2 在 GIN 不可用时**直接报错**而不是降级？
8. Ray 设置的哪个环境变量会导致 vLLM 的 CUDA Graph 构建失败？vLLM 怎么处理？
9. `VLLM_BATCH_INVARIANT` 具体禁用了哪些东西？为什么每个都必要？
10. 如果 TPOT 慢了 40%，你的第一步是什么？（不是「调 NCCL 参数」）

**下一章**：[`ch06-interview-bank.md`](ch06-interview-bank.md) —— 把这些变成面试答案。
