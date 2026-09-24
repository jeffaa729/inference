# 附录 B　动手实验（Labs）

> **设计原则**：不是每个人都有 8 卡机器。
> 所以实验分成两类：
> - **本地可做**（1 张卡，甚至 CPU）：验证概念、读代码、看日志；
> - **多卡可做**（2–8 卡）：真正测量通信。
>
> 每个实验都有：**目的 / 前提 / 步骤 / 你应该看到什么 / 思考题**。
>
> **📌 如果你要按小时租 GPU，请改用 [`labs-gpu.md`](labs-gpu.md)** ——
> 那份文件按预算分档、给出可直接粘贴的命令和脚本（`provision_rented_gpu.py` /
> `allreduce_bench.py` / `run_labs.py`），并强调「跑完立刻存数据」。
> 本文件偏「概念实验」，租机场景下性价比最高的实验在 `labs-gpu.md` 的套餐 A。


---

## Lab 0：验证教程里的代码引用（无需 GPU，2 分钟）

**目的**：确认你手上的 vLLM 版本和本教程的引用是否一致（行号会漂移）。

**步骤**：

```bash
# 会自动定位 vLLM checkout；也可以显式指定
python code/verify_citations.py
VLLM_REPO=/path/to/vllm python code/verify_citations.py

# ch06 里 200+ 条简写引用的存在性烟测
python check_ch06_citations.py
```

**预期输出**：`OK` 行 + 末尾 `63/63 exact, 0 moved, 0 unresolved`。

**如果出现 `MOVED` / `GONE`**：说明行号已经变了 —— `MOVED` 会告诉你**新行号**，
`GONE` 需要用符号名重新定位：

```bash
grep -n "CUSTOM_ALL_REDUCE_MAX_SIZES" vllm/distributed/device_communicators/all_reduce_utils.py
```

**思考题**：为什么本教程坚持「符号名比行号重要」？（提示：想想 `git blame` 和重构。）

---

## Lab 1：观测你的机器拓扑（无需多卡）

**目的**：建立「我手上这台机器能做什么并行」的判断力。

**步骤**：

```bash
# 1. GPU 与互连
nvidia-smi -L
nvidia-smi topo -m          # ★ 最重要：看 GPU 之间是 NV# 还是 PIX/PHB/SYS
nvidia-smi topo -m -p       # 加 PCIe 总线 ID 和 NUMA 亲和

# 2. NVLink 状态（如果有）
nvidia-smi nvlink -s
nvidia-smi nvlink -c        # 能力

# 3. 网卡与 RDMA 设备
ip -br addr
ibv_devinfo                 # 没有 IB 会报 command not found，也是有用信息
ls /sys/class/infiniband/   # 有 RoCE 也会出现在这里
```

**你应该看到什么 / 怎么解读**：

| 观察 | 结论 |
|---|---|
| `topo -m` 里 GPU 间是 `NV#` | 有 NVLink → **TP 可行** |
| 只有 `PIX`/`PHB`/`SYS` | 无 NVLink → **优先 PP 而非 TP**（官方文档的建议） |
| `SYS` 出现在多卡之间 | 跨 NUMA，通信会慢 |
| `ibv_devinfo` 无输出 | 没有 RDMA → 跨机只能走 TCP 或 RoCE |
| 有多张 HCA | 记录它们和 GPU 的相对位置（`topo -m` 的 NIC 列） |

**思考题**：
1. 你这台机器上，TP=4 的 all-reduce 会走 NVLink 还是 PCIe？
   （用 `topo -m` 矩阵判断：TP 组是相邻 rank，检查相邻 GPU 之间的连接类型。）
2. 如果只有 PCIe，为什么自定义 all-reduce 在 `world_size > 2` 时会被禁用？
   （回到 `custom_all_reduce.py:236-242` 和 `:416-417` 的注释。）

---

## Lab 2：读一次真实的 vLLM 启动日志（单卡或多卡）

**目的**：把教程里的「日志里有什么」变成肌肉记忆。

**步骤**：

```bash
# 单卡也能看到大部分日志；有 2 卡以上更好
vllm serve Qwen/Qwen2.5-0.5B \
    --tensor-parallel-size 1 \
    --max-model-len 2048 2>&1 | tee /tmp/vllm_boot.log
```

**在日志里找这几行**（用 `grep`）：

```bash
grep -n "using nccl"        /tmp/vllm_boot.log   # NCCL 版本（pynccl.py:117-120）
grep -n "assigned as"       /tmp/vllm_boot.log   # 各并行组 rank（parallel_state.py:2142-2154）
grep -n "all2all manager"   /tmp/vllm_boot.log   # all-to-all 后端（cuda_communicator.py:227-231）
grep -n "KV cache size"     /tmp/vllm_boot.log   # KV cache 容量
grep -n "Maximum concurrency" /tmp/vllm_boot.log # 并发估计
grep -niE "disabl|fallback|warn" /tmp/vllm_boot.log  # ★ 最有用：看哪些优化被禁用了
```

**你应该看到什么**：

- `vLLM is using nccl==X.Y.Z` → 记录这个版本，和 `pynccl_allocator.py:166` 的
  `22703`（对称内存）、`import_utils.py:456` 的 `23004`（DeepEP v2）对比。
- 「assigned as DP rank .., PP rank .., TP rank ..」→ 验证你理解的拓扑。
- **被禁用的优化** —— 这是最容易被忽略的：日志里会有
  「Custom allreduce is disabled because ...」之类的 warning，
  每一条都对应教程里讲过的一个 gate。

**思考题**：日志里有没有出现「Custom allreduce is disabled」？
如果有，把它对应的 gate 从 `custom_all_reduce.py` 的 `__init__` 里找出来。

---

## Lab 3：`NCCL_DEBUG=INFO` 读拓扑与算法选择（需要 ≥2 卡）

**目的**：亲眼看 NCCL 怎么选传输/算法/协议。

**步骤**：

```bash
# 用一个最小的 all-reduce 测试，不要用 vLLM（避免噪音）
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TUNING
export NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log

python - <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

# 测三种大小，看 NCCL 的选择是否不同
for nbytes in (16 << 10, 1 << 20, 128 << 20):
    n = nbytes // 2
    t = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):                     # 预热
        dist.all_reduce(t)
    torch.cuda.synchronize()
    import time
    s = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(t)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - s) / 20 * 1e3
    # ring 模型：每 rank 搬运 2(N-1)/N × S
    moved = 2 * (ws - 1) / ws * nbytes
    print(f"rank{rank} size={nbytes/1024:.0f}KiB  {ms:.3f} ms  "
          f"每rank搬运={moved/1024:.0f}KiB  有效带宽={moved/ (ms*1e-3) / 1e9:.1f} GB/s")
dist.destroy_process_group()
PY
```

多卡运行（2 卡）：
```bash
torchrun --nproc_per_node=2 /tmp/ar_test.py
```

**你应该看到什么**：

1. 日志里的 `NCCL INFO NET/IB : Using ...` 或 `NET/Socket : Using ...` → 实际传输。
2. `NCCL INFO Channel ...` 行 → channel 数（并发度）。
3. **不同大小下算法/协议可能不同** → 这就是第 3 章讲的决策。
4. 你的实测带宽 vs 「有效带宽」列 —— 小消息的有效带宽会很低，
   因为**延迟项主导**（这正是第 2 章的 α-β 模型）。

**思考题**：
1. 16 KiB 时的「有效带宽」是多少？和 128 MiB 时差几倍？
   把这个差距换算成「延迟占比」，验证 `T = 步数×α + S/BW` 里 α 项的主导地位。
2. 把 `NCCL_MAX_NCHANNELS` 设成 `1` 再跑一遍，看大消息带宽掉多少。
   （这就是 `VLLM_BATCH_INVARIANT` 的性能代价来源之一。）

---

## Lab 4：手算 vs 实测（需要 ≥2 卡）★ 最有价值的实验

**目的**：验证第 1、2 章的定量模型，建立「量级估算」的直觉。

**步骤**：用 Lab 3 的脚本，把实测值和理论值列表对比：

| 消息大小 | 理论（ring, `2(N-1)/N × S / BW`） | 实测 | 差异原因 |
|---|---|---|---|
| 16 KiB | | | α 主导 |
| 1 MiB | | | 过渡区 |
| 128 MiB | | | BW 主导 |

**你应该看到什么**：
- **小消息**：实测远大于纯带宽预测 → 延迟项的贡献。
- **大消息**：实测接近纯带宽预测 → 带宽项主导。
- **存在一个「最差效率点」**：既不够小（延迟不占绝对优势）也不够大（带宽没打满）。

**思考题**：
1. 用实测数据反推 α：`α ≈ (T_measured - S/BW) / 步数`。
   算出的 α 和你在第 1 章看到的「IB 2–5 μs」量级一致吗？
2. **把 TP 换成跨机（如果有 2 台机器）**：把 `torchrun` 换成跨机启动，
   对比同样的消息大小的延迟差异。这个倍数就是「为什么 TP 不能跨机」的定量证据。

---

## Lab 5：验证 all-reduce 的正确性（需要 ≥2 卡）

**目的**：掌握「不读 kernel 也能验证通信正确」的技巧。

```python
import torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

# 技巧：每个 rank 填自己的 rank 号，all-reduce 后应全部等于 0+1+...+(N-1)
expected = float(sum(range(ws)))
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    t = torch.full((4096,), float(rank), dtype=dtype, device="cuda")
    dist.all_reduce(t)
    ok = torch.allclose(t.float(), torch.full_like(t, expected, dtype=torch.float32),
                        rtol=1e-2, atol=1e-2)
    print(f"rank{rank} dtype={dtype} -> {'OK' if ok else 'FAIL'} (got {t[0].item()}, want {expected})")
```

**你应该看到什么**：三种 dtype 全部 `OK`。

**思考题**：
1. 为什么 bf16 需要放宽 `rtol`？（提示：bf16 只有 8 位尾数。）
2. 如果把两个 rank 的消息大小设成不一样（比如 rank 0 发 4096，rank 1 发 2048），
   会发生什么？**先预测再实验** —— 这就是第 5 章说的「形状必须一致」。

---

## Lab 6：在 vLLM 里对比通信后端（需要 ≥2 卡）★

**目的**：亲手验证「同样的模型，换个通信后端，性能不一样」。

**步骤**：

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct
TP=2

# A: 默认（custom AR 开启）
vllm bench latency --model $MODEL --tensor-parallel-size $TP \
    --input-len 128 --output-len 128 2>&1 | tail -20

# B: 禁用 custom all-reduce（走 NCCL）
vllm bench latency --model $MODEL --tensor-parallel-size $TP \
    --disable-custom-all-reduce \
    --input-len 128 --output-len 128 2>&1 | tail -20

# C: 确定性模式（禁用几乎所有快速路径）
VLLM_BATCH_INVARIANT=1 vllm bench latency --model $MODEL \
    --tensor-parallel-size $TP --input-len 128 --output-len 128 2>&1 | tail -20
```

（`--disable-custom-all-reduce` 对应 `vllm/config/parallel.py:214`；
如果 CLI 参数名不同，用 `vllm serve --help | grep -i custom` 确认。）

**你应该看到什么**：

- A 最快，B 稍慢，C 明显最慢。
- **关键**：注意 TPOT 的差异，**TTFT 的差异会小得多** ——
  因为 TTFT 主要是 prefill（大消息，本来就该走 NCCL），
  而 TPOT 是 decode（小消息，custom AR 优势最大）。
  **这直接验证了「custom AR 是为 decode 小消息设计的」这个结论。**

**思考题**：
1. A 和 B 的差距有多大？和你的理论预期（`CUSTOM_ALL_REDUCE_MAX_SIZES` 表）一致吗？
2. C 慢多少？把它的原因拆到具体变量上（对照 `batch_invariant.py:1147-1156` 的 10 个变量）。
3. **如果 A 和 B 差不多**，可能是什么原因？（提示：检查消息大小是否超过了 custom AR 的上限，
   或者 gate 链里哪一条没过 —— 去日志里找「Custom allreduce is disabled」。）

---

## Lab 7：MoE 的 all-to-all（需要 ≥2 卡 + MoE 模型）

**目的**：观察 EP 的通信形态。

**步骤**：

```bash
# 需要 MoE 模型，比如 Qwen3-30B-A3B（小一点的 MoE）
MODEL=Qwen/Qwen3-30B-A3B

# 默认 AG-RS 后端
vllm serve $MODEL --tensor-parallel-size 1 --data-parallel-size 2 \
    --enable-expert-parallel 2>&1 | grep -iE "all2all|ep rank|assigned"

# 如果有 DeepEP，换成 deepep_low_latency
vllm serve $MODEL --tensor-parallel-size 1 --data-parallel-size 2 \
    --enable-expert-parallel --all2all-backend deepep_low_latency
```

**你应该看到什么**：

- `Using AgRsAll2AllManager all2all manager.`（或对应的类名）
  → 对应 `cuda_communicator.py:227-231` 的日志。
- `assigned as ... EP rank ...` → 验证 `EP_SIZE = TP × DP`。
- **没有 DeepEP 时会报什么错** → 看 `all2all.py:162-165` 的 assert 信息
  （它会指向 `tools/ep_kernels/README.md`）。

**思考题**：
1. 你的 EP rank 是多少？和 `TP=1, DP=2` 的预期（EP=2）一致吗？
2. 如果把 `--all2all-backend` 设成一个不存在的值，报错信息是什么？
   （对应 `cuda_communicator.py:224-225` 的 `ValueError` —— 注意 vLLM 在这里**不做自动回退**。）
3. 观察开启 EP 前后，MoE 层的显存占用变化 —— 专家权重被分散了。

---

## Lab 8：控制面 vs 数据面（观察实验，无需多卡）

**目的**：亲眼确认「控制面不走 NCCL」。

**步骤**：

```bash
# 1. 找一个跑起来的 vLLM 进程，看它开的 socket
ss -xnp | grep -i vllm        # Unix domain socket（ZMQ ipc://）
ss -ltnp | grep -i vllm       # TCP 监听端口

# 2. 看共享内存段
ls -la /dev/shm/ | head -30
ipcs -m | head -20

# 3. 如果装了 py-spy，看进程卡在哪
py-spy dump --pid <vllm_pid> | head -50
```

**你应该看到什么**：

- **ZMQ 的 `ipc://` socket 文件**（在 `VLLM_RPC_BASE_PATH`，默认系统临时目录下）
  → 对应 `network_utils.py:157-159` 的 `get_open_zmq_ipc_path`。
- **共享内存段** → `MessageQueue` 的 `ShmRingBuffer`（`shm_broadcast.py:250-370`）。
- **TCP 端口** → DP 协调、API server。

**思考题**：
1. 为什么 API server ↔ engine core 用 ZMQ 而不是 NCCL？
   （提示：engine core 可能没有 GPU；而且这是字节级的控制消息。）
2. `MessageQueue.create_from_process_group(self.cpu_group, 1 << 22, 6)`
   （`parallel_state.py:560-562`）里的 `1 << 22` 和 `6` 是什么？
   （答：4 MiB 每块，6 块环形缓冲。）
3. **为什么用「共享内存 + ZMQ 通知」而不是纯共享内存自旋**？
   （提示：读 `shm_broadcast.py` 里 `SpinCondition` 的 docstring —— 纯自旋会浪费 CPU。）

---

## Lab 9（挑战）：写一个最小的「自定义 all-reduce」

**目的**：真正理解 custom AR 的机制 —— 用 CUDA IPC 让两个进程互相读写显存。

**骨架**（2 卡，只用 CUDA IPC，不用 NCCL）：

```python
"""最小 P2P all-reduce：用 cudaIpcMemHandle 交换显存指针。
仅用于教学：真实实现还需要同步、多 block、one-shot/two-shot 等。"""
import torch, torch.distributed as dist
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

dist.init_process_group("nccl")
rank, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
lib = CudaRTLibrary()

N = 1024
# 1. 每 rank 分配一块「共享」显存
ptr = lib.cudaMalloc(N * 2)
lib.cudaMemset(ptr, rank, N * 2)                 # 填入自己的 rank 号

# 2. 交换 IPC handle
handle = lib.cudaIpcGetMemHandle(ptr)
handles = [None] * ws
dist.all_gather_object(handles, handle)          # 走 gloo/NCCL 传 Python 对象

# 3. 打开所有对端的显存
peer_ptrs = [lib.cudaIpcOpenMemHandle(h) for h in handles]

# 4. 从每个对端读回来并求和（用 torch 的 from_blob 包装指针）
tensors = []
for r in range(ws):
    t = torch.empty(N, dtype=torch.int32, device="cuda")
    lib.cudaMemcpy(t.data_ptr(), peer_ptrs[r], N * 2, 4)   # 4 = cudaMemcpyDefault
    tensors.append(t)
total = sum(tensors)
expected = sum(range(ws))
print(f"rank{rank}: got {total[0].item()}, want {expected} "
      f"-> {'OK' if total[0].item() == expected else 'FAILA'}")

for p in peer_ptrs:
    lib.cudaIpcCloseMemHandle(p)
lib.cudaFree(ptr)
dist.destroy_process_group()
```

**运行**：`torchrun --nproc_per_node=2 min_p2p_ar.py`

**你应该看到什么**：两个 rank 都打印 `OK`。

**思考题（这才是本 Lab 的重点）**：
1. 上面的实现**缺了什么**才能变成 vLLM 的 custom AR？
   （至少说出 3 点：① 没有 rank 间同步/信号量；② 用 `cudaMemcpy` 而不是 kernel 内直接读；
   ③ 没有 CUDA Graph 支持；④ 没有 one-shot/two-shot 分档；
   ⑤ 没有 P2P 能力检测和 gate 链。）
2. 为什么 vLLM 要把对端指针**存进设备内存**（`rank_data`）而不是每次当参数传？
   （提示：CUDA Graph 要求 kernel 参数固定。）
3. `cudaIpcMemLazyEnablePeerAccess` 这个 flag 是什么含义？
   （看 `cuda_wrapper.py` 里 `cudaIpcOpenMemHandle` 的绑定。）

---

## Lab 10（挑战）：用 profile 定位瓶颈

**目的**：把「不要假设慢就是通信问题」变成实操能力。

**步骤**：

```bash
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node \
    -o /tmp/vllm_prof --force-overwrite true \
    vllm bench latency --model Qwen/Qwen2.5-7B-Instruct \
        --tensor-parallel-size 2 --input-len 128 --output-len 256
```

把 `.nsys-rep` 拖进 Nsight Systems GUI（或 `nsys stats /tmp/vllm_prof.nsys-rep`）。

**你应该看到什么 / 要算什么**：

```bash
# 命令行版：看 kernel 时间分布
nsys stats --report cuda_gpu_kern_sum /tmp/vllm_prof.nsys-rep | head -30
nsys stats --report nvtx_sum /tmp/vllm_prof.nsys-rep | head -20
```

**关键指标**：
1. **通信 kernel 占比**（找 `AllReduce` / `ncclDevKernel` / `cross_device_reduce`）。
   - decode 阶段如果 >30%，说明通信是瓶颈；
   - 如果 <15%，**别再从通信下手**。
2. **有没有 GPU 空闲**（timeline 上的空隙）→ 说明是 CPU/调度瓶颈（Python 开销、sampling）。
3. **CUDA Graph 是否生效**（graph 会显示为一个大的 `cudaGraphLaunch`）。

**思考题**：
1. prefill 阶段和 decode 阶段的通信占比分别是多少？为什么不同？
2. 如果你的通信占比很低但 TPOT 仍然慢，下一步查什么？
   （答：attention kernel、MoE kernel、采样、CPU 调度 —— 按时间占比排序逐个排除。）

---

## Lab 11：观察 DBO 是否真的在重叠（需要 ≥2 GPU + MoE 模型）★

**目的**：验证第 7 章讲的「ping-pong 重叠」在真实运行中长什么样。

**前提**：`--enable-dbo` 要求 `data_parallel_size > 1` + `--enable-expert-parallel`
+ `all2all_backend ∈ {deepep_low_latency, deepep_high_throughput, nixl_ep}`。
**没有 DeepEP 时这个实验做不了** —— 此时改为做「思考题」部分（仍然有价值）。

**步骤**：

```bash
MODEL=deepseek-ai/DeepSeek-V2-Lite      # 或任意 MoE 模型

# A: 不开 DBO
vllm serve $MODEL --trust-remote-code \
    --data-parallel-size 2 --enable-expert-parallel \
    --all2all-backend deepep_low_latency 2>&1 | tee /tmp/no_dbo.log

# B: 开 DBO
vllm serve $MODEL --trust-remote-code \
    --data-parallel-size 2 --enable-expert-parallel \
    --all2all-backend deepep_low_latency \
    --enable-dbo 2>&1 | tee /tmp/dbo.log
```

**在日志里找这些**：

```bash
grep -iE "dbatch|ubatch|DBO|all2all manager|dual" /tmp/dbo.log
grep -iE "disabl|warn" /tmp/dbo.log        # 看有没有被静默关掉
```

**用 profiler 看重叠**（关键）：

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o /tmp/dbo_prof \
    vllm bench latency --model $MODEL ... 
# 在 Nsight Systems 的 timeline 上观察：
#  - 是否有两个 stream 的 kernel 在时间上重叠
#  - 通信 kernel（AllToAll / dispatch / combine）是否落在计算 kernel 的阴影里
```

**你应该看到什么**：

| 观察 | 含义 |
|---|---|
| B 的 TPOT 低于 A（decode 场景） | 重叠生效 |
| timeline 上两个 stream 有交叠区间 | 真正的并行 |
| 通信 kernel 只占 20 个 SM 左右 | `VLLM_DBO_COMM_SMS` 默认值在起作用 |
| 日志里出现 "Disabling cascade attention when DBO is enabled" | 预期行为（DBO 与 cascade attention 冲突） |
| 日志里出现 "DBO is not supported..." / 静默不切 | 准入条件没过，见第 7 章 §7.3.3 的表 |

**思考题（不需要 DeepEP 也能答）**：
1. 阈值是 `dbo_decode_token_threshold=32` / `dbo_prefill_token_threshold=512`。
   **为什么 decode 的阈值（32）远小于 prefill 的（512）？**
   （提示：想想每个 ubatch 的计算量、kernel launch 开销、以及「切一刀」本身的成本。）
2. 为什么 `enable_dbo` 硬编码 2 个 ubatch，而 `--ubatch-size` 允许更多？
   3 个 ubatch 的 ping-pong 会有什么新问题？
   （提示：SM 要分给几个流？谁等谁？）
3. **如果某个 DP rank 的 batch 只有一个 token**，会发生什么？
   （提示：`is_last_ubatch_empty` + DP 全体投票。）
4. 为什么 DBO 与 cascade attention 冲突？（自动关闭的那条 warning）
5. 为什么 DBO 需要 `shared_experts` 的异步支持（`supports_async` assert）？

---

## Lab 12：MoE all-to-all 的流量与耗时拆解（需要 ≥2 GPU + MoE）★

**目的**：把第 8 章的 wire format 和真实 profile 对上。

**步骤**：

```bash
# 用 nsys 抓一次 MoE 模型的服务，然后看 kernel 名字
nsys stats --report cuda_gpu_kern_sum /tmp/moe_prof.nsys-rep > /tmp/kern.txt

# 找 all-to-all 相关的 kernel
grep -iE "dispatch|combine|alltoall|a2a|nvlink|rdma|deepep" /tmp/kern.txt | head -30

# 看时间分布：dispatch / 专家 GEMM / combine 各占多少
```

**你应该看到什么 / 要算什么**：

对每一层，理想的时间结构是：

```
dispatch ──► 专家 GEMM ──► combine
   ↑            ↑             ↑
 变长消息      本地计算      反向变长消息
```

**关键指标**：

| 指标 | 怎么算 | 说明 |
|---|---|---|
| **通信:计算 比** | `(dispatch + combine) / expert_gemm` | > 1 说明 EP 的通信是瓶颈 |
| **负载不均** | 各 rank 的 expert GEMM 时间差 | 差得多 → straggler，考虑开 EPLB |
| **rank 间气泡** | timeline 上某 rank 空等的时间 | all-to-all 是集合操作，**等最慢的** |

**思考题**：
1. 如果你观察到「一半 rank 忙、一半 rank 闲」的锯齿状 timeline，
   **这说明了什么**？（答：负载不均 / straggler。）
2. 对比 `--all2all-backend allgather_reducescatter` 和 `deepep_low_latency` 的
   kernel 时间分布 —— 前者应该能看到明显的 `all_gather` / `reduce_scatter` kernel。
3. 为什么 `combine` 阶段也要通信？（提示：专家输出要回到原 rank 才能和别的专家结果相加。）
4. 打开 `--enable-eplb` 后，各 rank 的 expert GEMM 时间是否更均匀了？

---

## Lab 13（纸面实验，无需 GPU）：PD 分离值不值得做

**目的**：练「先算账再动手」的工程判断力。

**场景**：一个在线服务，SLO 是 **p99 ITL ≤ 50 ms**，当前观测到：
- p50 ITL = 20 ms，p99 ITL = 180 ms（**尾延迟超标 3.6 倍**）
- prefill 请求平均 4000 token，decode 请求平均输出 300 token
- 单机 8×H100，KV cache 每 token 约 0.25 MB（按模型算）
- 节点间网络：IB NDR 400 Gb/s（50 GB/s 单向）
- 观测到：每隔约 2 s 出现一次 ITL 尖刺，对应一个长 prefill 被调度

**问题**：
1. 这个场景该不该上 PD 分离？**先算 KV 传输时间**：
   ```
   KV 大小 = 4000 token × 0.25 MB/token = 1000 MB = 1 GB
   传输时间 ≈ 1 GB / 50 GB/s = 20 ms（理想）
             + 实际效率折损（按 50% 算）→ 40 ms
   ```
2. **40 ms 的传输 vs 180 ms 的尾延迟** —— 谁大？
   （答：传输的 40 ms 是**每请求一次**，而尖刺是**尾延迟**；
   如果 PD 分离能把 p99 从 180 ms 降到 60 ms，**净赚**。
   但如果 SLO 只要 p50，PD 分离就是纯增开销。）
3. **如果网络只有 100 Gb/s（12.5 GB/s）呢？**
   （答：`1 GB / 12.5 GB/s = 80 ms`，加折损就是 160 ms —— 已经接近尾延迟本身，
   **PD 分离不划算**。此时应该先优化 prefill 的调度（chunked prefill）。）
4. **官方文档说 "Disaggregated prefill DOES NOT improve throughput"**，
   那它到底改善了哪个指标？（答：TTFT 和 ITL 的**解耦**，以及尾延迟。）

**思考题**：
5. 如果要**同时**要低尾延迟和高吞吐，除了 PD 分离还有什么手段？
   （答：chunked prefill 限制单步 prefill 的 token 数、请求优先级调度、
   把长 prefill 和 decode 分开排队但共用实例、增大 batch 摊薄 prefill 影响。）
6. **chunked prefill 和 PD 分离的关系是什么**？
   （答：chunked prefill 是**在单实例内**把长 prefill 切碎，避免长时间占据 GPU；
   PD 分离是**把两类负载物理分开**。前者更简单，后者更彻底 ——
   **先试 chunked prefill，不够再上 PD 分离**。）

---

## 附录：实验速查表

| Lab | 需要 | 时间 | 学到什么 |
|---|---|---|---|
| 0 | 无 | 2 min | 引用可验证性 |
| 1 | 1 GPU | 5 min | 拓扑判断力 |
| 2 | 1+ GPU | 10 min | 日志解读、发现被禁用的优化 |
| 3 | 2+ GPU | 20 min | NCCL 的传输/算法/协议选择 |
| 4 | 2+ GPU | 30 min | **定量模型验证（最有价值）** |
| 5 | 2+ GPU | 10 min | 通信正确性验证技巧 |
| 6 | 2+ GPU | 30 min | **后端对比：custom AR 的价值** |
| 7 | 2+ GPU + MoE | 30 min | EP 的 all-to-all 形态 |
| 8 | 无 | 10 min | 控制面/数据面分离 |
| 9 | 2 GPU | 60 min | **亲手实现 P2P all-reduce** |
| 10 | 2+ GPU | 40 min | profile 定位瓶颈 |
| 11 | 2+ GPU + DeepEP | 40 min | **DBO 重叠的实测观察** |
| 12 | 2+ GPU + MoE | 40 min | **MoE all-to-all 的流量拆解** |
| 13 | **无** | 20 min | **PD 分离的收益判断（纸面算账）** |

**如果只有 2 小时**：做 Lab 0 → 1 → 3 → 5。
**如果有 1 天**：加上 Lab 4 → 6 → 9。
**没有多卡**：Lab 0、1、2、8、13 都能做 —— **Lab 13 尤其值得做，它训练的是工程判断力，
而不是动手能力**。

---

## 写在最后：实验的「正确姿势」

1. **先预测，再实验**。每次跑之前写下你预期看到什么。
   如果和预期不符 —— **那才是真正学到东西的时刻**，去搞清为什么。
2. **记录配置**。任何性能数字离开配置就没有意义（模型、并行度、batch、dtype、驱动版本）。
3. **多次重复，报中位数**。分布式测量噪声大，单次结果不可信。
4. **不要只测一个点**。通信的性能是**随消息大小变化的曲线**，一个点得不出结论。

> 这四条，也是面试官问「你怎么做性能评测」时想听到的答案。
