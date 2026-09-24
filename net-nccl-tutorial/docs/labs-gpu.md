# 附录 D　租机多卡实验手册（GPU 版）

> **为什么单独一个文件**：`labs.md` 假设的是「手边有什么卡就用什么」。
> 但如果你要**按小时租 GPU**，实验设计原则完全不同 ——
> 每一分钟都在花钱，所以必须**先想清楚要测什么、跑完立刻存数据**。
>
> 本文件给的是：**按预算分档的实验套餐 + 可直接粘贴的脚本 + 预期结果 + 避坑清单**。

---

## D.0 租机前的 5 分钟决策（先看这个，别急着开机器）

### D.0.1 你到底需要几张卡？

| 你想验证的结论 | 最少需要 | 理想配置 | 说明 |
|---|---|---|---|
| 冒烟：vLLM 多卡能起、日志正确 | **2 卡** | 2 卡 | 最便宜的验证，1 小时够 |
| TP 的 all-reduce 实测 vs 手算 | **2 卡** | 4 卡 | 2 卡就能验证 α-β 模型 |
| custom AR vs NCCL 的差距 | **2 卡** | 4 卡（NVLink） | 4 卡时 custom AR 的 gate 链才完整 |
| NVLink vs PCIe 的差距 | 需要**两种机器** | — | 或者一台机器上用 `NCCL_P2P_DISABLE` 模拟 |
| ring vs tree 的 crossover | 4 卡 | 8 卡 | 消息大小扫描是重点 |
| 跨机 TP 为什么不可行 | **2 机 × 2 卡** | 2 机 × 8 卡 | 需要**多节点**（租两台的费用翻倍，谨慎） |
| MoE all-to-all / EP | 2 卡 + MoE 模型 | 4–8 卡 | 小 MoE（Qwen3-30B-A3B 量化版）2 卡可跑 |
| DBO | ≥2 卡 + MoE + **DeepEP** | 8 卡 + IB | DeepEP 安装门槛高，**先确认能装上再租** |
| EPLB | 4 卡 + MoE | 8 卡 | 专家数要够多才能看出重平衡效果 |

**结论**：
- **预算 < $30**：租 **1 台 4 卡机 2–3 小时**，做 D.2 套餐。
- **预算 $100 左右**：租 **1 台 8 卡 NVLink 机 4–6 小时**，做 D.2 + D.3。
- **要做跨机实验**：至少 2 台 × 2 卡，且**必须有 IB 或高速以太网**，否则测的是网络而不是你的代码。

### D.0.2 选机器时必须确认的 4 件事

```bash
# 1. 卡间互联是 NVLink 还是 PCIe？（决定 TP 实验有没有意义）
nvidia-smi topo -m
#   看到 NV# → 好；只有 PIX/PHB/SYS → TP 会慢，但仍可做实验（结论会不同）

# 2. 有没有 IB / RDMA？（决定跨机实验能不能做）
ibv_devinfo || echo "NO RDMA"
ls /sys/class/infiniband/

# 3. 驱动 / CUDA / NCCL 版本（决定有些特性能不能用）
nvidia-smi --query-gpu=driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"

# 4. 显存够不够跑你要的模型
nvidia-smi --query-gpu=name,memory.total --format=csv
```

**⚠️ 三个最常见的租机踩坑**：
1. **租到 PCIe-only 的机器做 TP 实验** → 结果会和所有教程/NVLink 数据对不上。
   **先跑 `nvidia-smi topo -m` 再开跑。**
2. **NCCL 版本比 PyTorch 里的旧** → 对称内存（需 ≥2.27.3）、DeepEP v2（需 ≥2.30.4）不可用。
   用 `VLLM_NCCL_SO_PATH` 指向自己装的 NCCL 可以绕过。
3. **容器里 `CUDA_VISIBLE_DEVICES` 被限制成 1 卡** → 明明租了 8 卡却只能看到 1 张。
   `echo $CUDA_VISIBLE_DEVICES` 先确认。

### D.0.3 省钱的三条铁律

1. **本地准备好一切再开机器**。脚本、命令、模型下载（或提前缓存到网络盘）都先就绪。
   **租机后的第一件事不是写代码，是跑 `python code/provision_rented_gpu.py`**（见 §D.1）。
2. **模型用小号的**。验证通信行为**不需要大模型** ——
   `Qwen2.5-0.5B`（~1 GB）足够测 TP 通信，`Qwen2.5-7B` 足够测真实负载。
   **不要为了「真实感」去下 70B，那会把时间全花在下载上。**
3. **跑完立刻 `python code/run_labs.py collect`**，把数据落到文件并从机器上拉走。
   **实例一释放，数据就没了。**

---

## D.1 租机后第一件事：一键环境自检

在机器上放一个 `provision_rented_gpu.py`（本目录已提供），它做 5 件事：

1. 打印 GPU / 拓扑 / RDMA / 驱动信息（存成 `env_report.txt`）
2. 检查 vLLM / PyTorch / NCCL 版本，并对照教程里的版本要求表给出「哪些特性可用」
3. 检查模型缓存，列出「哪些实验现在就能跑」
4. 跑一次 2 卡的 all-reduce 冒烟测试（最快确认通信真的能用）
5. 打印建议的实验套餐（按你实际有几张卡）

```bash
python code/provision_rented_gpu.py
```

**预期输出末尾**会长这样（示意）：

```
================ 你的机器能做什么 ================
GPU: 8×NVIDIA H100 80GB SXM | NVLink: YES (NV18) | RDMA: mlx5_0..7 | 节点数: 1
NCCL 2.27.5   → 对称内存 ✅  DeepEP v2 ❌(需 2.30.4)  suspend/resume ✅
vLLM 0.x.y    → pynccl ✅  custom AR ✅
建议套餐: D.3（8 卡 NVLink，预估 4–6 小时）
=================================================
```

---

## D.2 套餐 A：2–4 卡（最便宜的「真通信」实验）

**目标**：把手算模型和真实测量对上。**这是性价比最高的一组实验。**

**预估**：4 卡机 2–3 小时（约 $10–25，视卡型）

### A1. 拓扑与冒烟（10 分钟）

```bash
nvidia-smi topo -m | tee results/topo.txt
nvidia-smi -L | tee -a results/topo.txt

# 最小冒烟：2 卡起一个 0.5B 模型
vllm serve Qwen/Qwen2.5-0.5B --tensor-parallel-size 2 \
    --max-model-len 2048 2>&1 | tee results/smoke_tp2.log &
sleep 60
curl -s localhost:8000/v1/models | head -20
kill %1
```

**看什么**：启动日志里有没有
`Custom allreduce is disabled because ...` —— 每一条都对应一个 gate 条件（教程 ch04 §4.4.4）。

### A2. ★ all-reduce 实测 vs 手算（30 分钟，**本套餐的核心**）

```bash
torchrun --nproc_per_node=2 allreduce_bench.py --sizes 16K,64K,256K,1M,4M,16M,64M,256M
```

（脚本本目录提供，见 §D.5）

**要记录的三个数**（每个消息大小）：`耗时 ms` / `有效带宽 GB/s` / `自定义 AR 是否命中`

**预期形状**（示意，绝对值取决于机器）：

| 消息 | 实测耗时 | 有效带宽 | 谁主导 |
|---|---|---|---|
| 16 KiB | ~30–60 μs | 很低（<1 GB/s） | **延迟** |
| 256 KiB | ~40–80 μs | 几 GB/s | 过渡 |
| 4 MiB | ~150–300 μs | 十几 GB/s | 过渡 |
| 64 MiB | ~1.5–3 ms | 接近线速 | **带宽** |

**怎么用它验证教程**：
1. 小消息的有效带宽**极低** → 证明 `T = 步数×α + S/BW` 里 α 项主导；
2. 用 `T_measured - S/BW` 反推 α，看是否落在教程说的「NVLink 亚 μs、IB 2–5 μs」量级；
3. 找到「效率最低的点」（既不小也不大），那正是教程 §2.7 题 3 讲的 crossover 区。

### A3. 后端对比：custom AR 值多少（30 分钟）

```bash
MODEL=Qwen/Qwen2.5-0.5B
for flag in "" "--disable-custom-all-reduce"; do
  echo "=== flag: ${flag:-default} ==="
  vllm bench latency --model $MODEL --tensor-parallel-size 2 \
      --input-len 128 --output-len 128 $flag 2>&1 | tail -25
done
```

**关键**：**对比 TTFT 和 TPOT 的差异幅度**。

- 预期：**TPOT 的差距明显大于 TTFT 的差距**。
- 原因：TTFT 主要是 prefill（大消息，本来就该走 NCCL）；
  TPOT 是 decode（小消息，custom AR 的主场）。
- **这就直接验证了「custom AR 是为 decode 小消息设计的」。**

⚠️ 如果两者差不多，去日志里找 `Custom allreduce is disabled because ...`，
说明 gate 没过（2 卡时要求 `world_size==2` 即可，通常能过）。

### A4. ring vs tree 的 crossover（20 分钟）

```bash
for algo in Ring Tree; do
  for size in 16K 256K 4M 64M; do
    NCCL_ALGO=$algo NCCL_DEBUG=INFO \
      torchrun --nproc_per_node=2 allreduce_bench.py --sizes $size --tag "algo=$algo" \
      2>&1 | grep -E "size=|NCCL INFO.*(Ring|Tree)"
  done
done
```

**预期**：小消息 `Tree` 赢，大消息 `Ring` 赢，中间有个 crossover。

**产出**：一张 `消息大小 × 算法` 的对比表 —— **这是面试里可以直接说的实测数据。**

### A5. 通信正确性（5 分钟）

```bash
torchrun --nproc_per_node=2 verify_collectives.py
```

**预期**：三种 dtype 全部 `OK`，且额外打印出每个 rank 的 `rank_in_group` 与设备号 ——
**顺手验证了「TP 组是相邻 rank」这条教程结论。**

---

## D.3 套餐 B：8 卡 NVLink（完整版）

**目标**：覆盖 custom AR 的全部门槛、SM 仲裁、EP 的行为。

**预估**：8 卡 H100 机 4–6 小时（约 $100–200）

**前置**：先跑完套餐 A（那些实验在 8 卡机上一样做，只是 world size 更大）。

### B1. custom AR 的 size 上限是真实存在的（20 分钟）★

这是**最能体现「读过教程」的实验**：教程说 H100 上 TP=8 的 custom AR 只覆盖 256 KiB。

```bash
# 8 卡，扫到 1 MiB，看什么时候从 custom AR 掉回 NCCL
torchrun --nproc_per_node=8 allreduce_bench.py \
    --sizes 16K,64K,128K,192K,256K,320K,512K,1M --report-backend
```

**预期**：在 **256 KiB 附近**出现后端切换（`custom` → `nccl`）。
对照 `all_reduce_utils.py` 的 `CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"][8] = MiB // 4`。

**如果没看到切换**：可能 `world_size` 或 `fully_connected` 不满足，
去日志找 gate 原因 —— **排查过程本身就是学习。**

### B2. TP 规模的 all-reduce 缩放曲线（30 分钟）

```bash
for tp in 2 4 8; do
  torchrun --nproc_per_node=$tp allreduce_bench.py --sizes 16K,1M,64M --tag "tp=$tp"
done
```

**预期**：
- **小消息（16 KiB）**：耗时应**随 TP 增长而变差**（步数随 N 线性增长）；
- **大消息（64 MiB）**：耗时增长较慢（带宽项趋近 `2S`，与 N 弱相关）。
- **这正是教程 §2.3 的核心结论的实测版。**

### B3. 8 卡上的 vLLM 真实负载（60 分钟）

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct
for tp in 1 2 4 8; do
  echo "=== TP=$tp ==="
  vllm bench throughput --model $MODEL --tensor-parallel-size $tp \
      --num-prompts 200 --input-len 512 --output-len 128 2>&1 | tee results/tp_$tp.log
done
```

**要提取**：`output token throughput` / `TPOT p50` / `TTFT p50`。

**预期**：吞吐**不一定单调上升**。存在一个最优 TP ——
超过它之后，每卡的计算量太小，通信占比上升，吞吐反而下降。
**找到这个拐点，就是你这台机器上「TP 该开多大」的答案。**

### B4. SM 仲裁与 DBO（如果 MoE + DeepEP 可用，90 分钟）

**先确认 DeepEP 能装**（装不上就跳过，别浪费时间）：

```bash
python -c "import deep_ep; print('deep_ep OK')" || echo "DeepEP 不可用 → 跳过 DBO 实验"
```

能装的话：

```bash
MODEL=Qwen/Qwen3-30B-A3B        # 或更小的 MoE

# A: 不开 DBO
vllm serve $MODEL --data-parallel-size 8 --enable-expert-parallel \
    --all2all-backend deepep_low_latency 2>&1 | tee results/moe_no_dbo.log &
sleep 300
# 压测
vllm bench latency --model $MODEL --input-len 1024 --output-len 256 2>&1 | tee results/moe_no_dbo_bench.log
kill %1

# B: 开 DBO
VLLM_DBO_COMM_SMS=20 vllm serve $MODEL --data-parallel-size 8 --enable-expert-parallel \
    --all2all-backend deepep_low_latency --enable-dbo 2>&1 | tee results/moe_dbo.log &
sleep 300
vllm bench latency --model $MODEL --input-len 1024 --output-len 256 2>&1 | tee results/moe_dbo_bench.log
kill %1
```

**要对比**：TPOT p50/p99。**预期 DBO 改善 decode 的 TPOT**（因为它重叠的是 MoE 的 all-to-all）。

**同时扫 `VLLM_DBO_COMM_SMS`**：`0 / 10 / 20 / 40` ——
看 SM 配额对性能的影响，**这直接对应教程 ch07 §7.4 的讨论**。
（注意：`0` 表示不做 SM 划分，不是「不给通信 SM」。）

### B5. nsys profile（40 分钟）★

```bash
nsys profile -t cuda,nvtx,osrt --cuda-graph-trace=node -o results/prof_tp8 \
    vllm bench latency --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 8 \
    --input-len 128 --output-len 256 2>&1 | tail -5

nsys stats --report cuda_gpu_kern_sum results/prof_tp8.nsys-rep > results/kern_sum.txt
grep -iE "allreduce|reduce_scatter|all_gather|nccl" results/kern_sum.txt
```

**要算的**：`通信 kernel 总时间 / 总 kernel 时间`。
**这是你写进简历/面试的那个「通信占比 X%」的来源** —— 必须是实测的。

---

## D.4 套餐 C：跨机（贵，只在必要时做）

**目标**：证明「跨机 TP 不可行」和「EP/PP 可以跨机」。

**⚠️ 前置**：**必须有 IB 或 ≥100 Gb/s 以太网**。用 10 GbE 测出来的结论没有意义。

**预估**：2 台 × 8 卡，4 小时（约 $200–400）

### C1. 跨机点对点带宽基线（20 分钟）

```bash
# 两台机器分别跑
# 机器 1
NCCL_DEBUG=INFO torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --master_addr=<机器1IP> --master_port=29500 allreduce_bench.py --sizes 1M,64M

# 机器 2
NCCL_DEBUG=INFO torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --master_addr=<机器1IP> --master_port=29500 allreduce_bench.py --sizes 1M,64M
```

**先测这个**：如果机间 all-reduce 的带宽远低于网卡标称值，
**说明是网络配置问题，先修网络再谈应用**（教程 ch05 §5.3）。

### C2. 跨机 TP vs 机内 TP（40 分钟）★

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct

# 机内 TP=2
vllm bench latency --model $MODEL --tensor-parallel-size 2 ... | tee results/intra_tp2.log

# 跨机 TP=2（1 台 1 卡）
vllm serve $MODEL --tensor-parallel-size 2 --nnodes 2 --node-rank 0 \
    --master-addr <机器1IP> --master-port 29500 ...
```

**预期**：**跨机 TP 的 TPOT 显著恶化**（尤其小 batch）。
**把这个倍数记录下来** —— 它就是教程 §1.8.2 那句「跨机 TP 代价是机内的 ~10 倍」的实测证据。

### C3. 跨机 PP / DP / EP（60 分钟）

```bash
# 跨机 PP（应该是可用的）
vllm serve $MODEL --tensor-parallel-size 1 --pipeline-parallel-size 2 \
    --nnodes 2 --node-rank 0 --master-addr <IP> --master-port 29500 ...

# 跨机 DP + EP（MoE 模型）
vllm serve $MOE_MODEL --data-parallel-size 16 --data-parallel-size-local 8 \
    --data-parallel-address <IP> --data-parallel-rpc-port 13345 \
    --enable-expert-parallel --all2all-backend deepep_low_latency --headless
```

**结论要形成一句话**：
> 「在这台机器的网络上，跨机 TP 的 TPOT 是机内的 __ 倍，不可用；
> 跨机 PP/EP 的代价是 __%，可接受。」
>
> **这就是面试里「跨机怎么扩展」的实测答案。**

---

## D.5 配套脚本

本目录提供 5 个脚本，**全部可以直接跑，不需要额外依赖**（除脚本自身要求的 torch/vllm）：

| 脚本 | 用途 | 需要 |
|---|---|---|
| `provision_rented_gpu.py` | **上机第一件事**：环境自检 + 能力对照 + 建议套餐 + 2 卡冒烟 | 无（torch 可选） |
| `allreduce_bench.py` | ★ 核心：多卡 all-reduce 扫描（大小 × 算法 × 后端），输出 CSV | torch + ≥2 卡 |
| `verify_collectives.py` | 集合通信正确性验证 + rank/设备/并行组拓扑打印 | torch + ≥2 卡 |
| `run_labs.py` | ★ **一键跑套餐 + 汇总结果 + 打包带走** | 无（按需调用上面两个） |
| `check_ch06_citations.py` | 校验 ch06 的简写引用（本地就能跑） | 无 |

### D.5.1 推荐工作流（三步）

```bash
# 第 1 步：上机自检（30 秒）——先知道这台机器能做什么
python code/provision_rented_gpu.py

# 第 2 步：看计划，然后跑
python code/run_labs.py list            # 按实际卡数显示哪些能跑
python code/run_labs.py all             # 跑全部能跑的（或 core / run a2 a5）
python code/run_labs.py core            # 只有 2 小时时：a2 + a5

# 第 3 步：汇总 + 打包（一定要做！）
python code/run_labs.py markdown        # 把 CSV/JSON 渲染成 SUMMARY.md
python code/run_labs.py collect         # 打成 tar.gz，拉回本地
```

`run_labs.py` 会：
- 自动按 GPU 数**跳过跑不了的实验**（不会浪费你的租机时间）
- 每个实验的完整输出存进 `results/<时间戳>/<lab>.log`
- 把实际执行的命令追加到 `results/<时间戳>/commands.sh`（**事后可复现**）
- 生成 `STATUS.md`（哪个成功/失败/耗时）
- `markdown` 会把 CSV 渲染成表格，并**反推 α** 供你和教程的量级表对照

### D.5.2 单独使用（想细调参数时）

```bash
# 环境自检
python code/provision_rented_gpu.py --json           # 额外写 env_report.json

# all-reduce 扫描（核心）
torchrun --nproc_per_node=2 allreduce_bench.py \
    --sizes 16K,256K,4M,64M --out results/ar_tp2.csv --report-backend

# 强制算法对比（NCCL_ALGO 必须在 init 前设置，脚本会处理）
torchrun --nproc_per_node=2 allreduce_bench.py --sizes 16K,4M --algo Ring --tag Ring
torchrun --nproc_per_node=2 allreduce_bench.py --sizes 16K,4M --algo Tree --tag Tree

# 正确性 + 拓扑
torchrun --nproc_per_node=2 verify_collectives.py --json results/verify.json
```

**`allreduce_bench.py` 的 CSV 列**：
`timestamp, tag, world_size, size_bytes, dtype, iters, median_ms, min_ms, max_ms,`
`moved_bytes_per_rank, effective_gbps, backend_hint, nccl_algo`

其中 `moved_bytes_per_rank` 用的是**ring 模型的参考值** `2(N-1)/N·S`，
`effective_gbps` 是由它推出来的 —— **它是「相对 ring 模型的有效带宽」，不是物理链路带宽**。
这个区分很重要：小消息时它会显得很低，那正是**延迟主导**的证据，不是链路不行。

### D.5.3 故障时的行为

- `run_labs.py` **默认不因为一个实验失败就中止**（租机场景下应该把能跑的先跑完拿数据）。
  需要严格模式用 `--fail-fast`。
- 每个实验有超时（默认 3600 s，用 `--timeout` 改），超时记为 `rc=124`。
- 脚本读到带 BOM 的 CSV/JSON 也不会崩（已做兼容），但**建议统一用不带 BOM 的 UTF-8**。

---

## D.6 数据必须留下：结果模板

租机实验最大的浪费是「跑完了但没记录」。**用这个模板**（每跑完一组就填）：

```markdown
# 实测记录 — <日期> — <机器型号>

## 环境
- GPU:                        （nvidia-smi -L 的第一行）
- 卡间互联:                    （nvidia-smi topo -m 结论：NV# / PIX / PHB）
- RDMA:                       （ibv_devinfo 摘要，或 NO RDMA）
- 驱动 / CUDA:                
- vLLM / PyTorch / NCCL:       （贴 `vLLM is using nccl==x.y.z` 那行）

## 实验 1：all-reduce 大小扫描（TP=<N>）
| 消息大小 | 中位耗时 μs | 有效带宽 GB/s | 预期（手算） | 差异说明 |
|---|---|---|---|---|
| 16 KiB |  |  |  |  |
| 256 KiB |  |  |  |  |
| 4 MiB |  |  |  |  |
| 64 MiB |  |  |  |  |

**反推的 α = ______ μs**（用 `α ≈ (T - S/BW)/步数`）

## 实验 2：custom AR vs NCCL
| 配置 | TTFT p50 | TPOT p50 | 吞吐 |
|---|---|---|---|
| 默认（custom AR） |  |  |  |
| `--disable-custom-all-reduce` |  |  |  |

**结论**：TPOT 差 ___%，TTFT 差 ___% → 是否符合「custom AR 为 decode 设计」？

## 实验 3：TP 缩放
| TP | 吞吐 | TPOT p50 | TTFT p50 |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 4 |  |  |  |
| 8 |  |  |  |

**最优 TP = ____**，拐点原因：____________

## 实验 4：通信占比（nsys）
- 总 kernel 时间: ______
- 通信 kernel 时间: ______
- **通信占比 = ____%**

## 最意外的观察（最重要的一栏）
（写下和你预期不符的地方 —— 那才是真正学到的）
```

---

## D.7 租机避坑清单（照着核对）

**上机前**
- [ ] 脚本、命令、模型名都已确定，写在文件里（别指望现场想）
- [ ] 确认机器有 **NVLink**（要做 TP 实验）和 **IB**（要做跨机）
- [ ] 确认 `CUDA_VISIBLE_DEVICES` 没被容器限制
- [ ] 准备好 `results/` 目录和拉取方式（`scp` / 对象存储）

**上机后立刻**
- [ ] `python code/provision_rented_gpu.py` → 存 `env_report.txt`
- [ ] `nvidia-smi topo -m` → 存 `topo.txt`
- [ ] 跑一次 2 卡冒烟，确认通信真的能用（**别等到半小时后才发现网络不通**）
- [ ] 检查模型是否能下载（或已在缓存里）

**实验过程中**
- [ ] 每跑完一组**立刻**写进结果模板，别攒着
- [ ] 每组至少重复 3 次，报中位数（分布式测量噪声大）
- [ ] 出现异常先 `NCCL_DEBUG=INFO`，别猜
- [ ] 注意别把「机器被别人共享」当成自己的结论（`nvidia-smi` 看有没有别的进程）

**下机前**
- [ ] `python code/run_labs.py collect`（或手动打包 `results/`）
- [ ] **确认数据已经拉走并在本地能打开**
- [ ] 记下花了多少钱、哪个实验最值得

---

## D.8 常见失败模式（租机版）

| 现象 | 最可能的原因 | 处理 |
|---|---|---|
| 2 卡起不来 / hang | `NCCL_SOCKET_IFNAME` 选错网卡（多网卡机器） | `export NCCL_SOCKET_IFNAME=<正确网卡>`，见 ch05 §5.3 |
| 卡在 NCCL 初始化很久 | 缺 IB 却配了 `NCCL_IB_HCA` | `unset NCCL_IB_HCA` 或 `NCCL_IB_DISABLE=1` 试 |
| 日志说 `Custom allreduce is disabled` | gate 没过（不是 NVLink 全互联 / 无 P2P / 超 size） | 按日志原文对照 ch04 §4.4.4 的表 |
| 实测带宽远低于标称 | 走了 PCIe 而不是 NVLink；或跨机走了 socket | `NCCL_DEBUG=INFO` 看实际传输 |
| TP=8 比 TP=4 还慢 | 超过最优 TP 拐点 | 正常现象，记录拐点 |
| 跨机实验完全不可用 | 网络是 10 GbE 而非 IB | 换机器，别硬调 |
| `VLLM_BATCH_INVARIANT` 下性能腰斩 | 确定性模式禁用了所有快速路径 | 确认是否真的需要确定性 |
| DeepEP 装不上 | NCCL 版本 / 驱动 / IBGDA 不满足 | **放弃 DBO 实验，别耗时间** |
| nsys 打不开 / 报权限 | 容器缺 `--cap-add=SYS_ADMIN` 或 ptrace 限制 | 改用 `--cuda-graph-trace=node` 仍失败就跳过 |

---

## D.9 一页速查：2 小时最小套餐

如果只有 1 台 4 卡机和 2 小时：

```bash
# 0:00 上机，5 分钟环境
python code/provision_rented_gpu.py | tee env_report.txt
nvidia-smi topo -m | tee topo.txt

# 0:05 all-reduce 扫描（15 分钟）★ 最重要
torchrun --nproc_per_node=4 allreduce_bench.py \
    --sizes 16K,256K,4M,64M --out results/ar_tp4.csv

# 0:20 正确性（5 分钟）
torchrun --nproc_per_node=4 verify_collectives.py | tee results/verify.txt

# 0:25 后端对比（25 分钟）★ 第二重要
vllm bench latency --model Qwen/Qwen2.5-0.5B --tensor-parallel-size 4 \
    --input-len 128 --output-len 128 2>&1 | tee results/custom_ar.log
vllm bench latency --model Qwen/Qwen2.5-0.5B --tensor-parallel-size 4 \
    --input-len 128 --output-len 128 --disable-custom-all-reduce 2>&1 | tee results/no_custom_ar.log

# 0:50 TP 缩放（30 分钟）
for tp in 1 2 4; do
  vllm bench throughput --model Qwen/Qwen2.5-0.5B --tensor-parallel-size $tp \
      --num-prompts 200 --input-len 512 --output-len 128 2>&1 | tee results/tp_$tp.log
done

# 1:20 nsys 抓一次通信占比（20 分钟）
nsys profile -t cuda --cuda-graph-trace=node -o results/prof \
    vllm bench latency --model Qwen/Qwen2.5-0.5B --tensor-parallel-size 4 \
    --input-len 128 --output-len 256
nsys stats --report cuda_gpu_kern_sum results/prof.nsys-rep | tee results/kern.txt

# 1:40 填结果模板 + 打包数据
# 1:50 拉走数据，释放实例
```

**这 2 小时能产出**：
1. 一张「消息大小 → 实测带宽」的表（对照手算）
2. custom AR 的实测收益（TTFT/TPOT 分别差多少）
3. 你机器上的最优 TP
4. 一个实测的「通信占比 X%」

**这四条就是面试里可以直接讲的实测数据 —— 比任何背诵都值钱。**

---

## D.10 把实测数据回填进教程

跑完之后，回到这几处用**你的真实数字**替换估算值：

| 教程位置 | 现在是什么 | 替换成 |
|---|---|---|
| ch01 §1.8.1 的 ~10 μs | 量级估算 | 你的 TP=8 实测 |
| ch01 §1.8.2 的「~10 倍」 | 量级估算 | 你的跨机/机内比值 |
| ch01 延迟量级表 | 通用量级 | 你的机器实测 |
| ch04 §4.4.3 的 custom AR 上限 | 代码里的表 | 你实测到的切换点 |
| ch07 §7.0 的「通信和计算一样多」 | 极端假设示例 | 你的 nsys 实测占比 |

**这一步做完，这份材料就从「教程」变成了「你自己的实测报告」** ——
面试时可以说「我在 8×H100 上实测了 …」，而不是「我看资料说 …」。

---

## D.11 本文件的自检题

1. 你租的机器上，TP 的 all-reduce 走的是 NVLink 还是 PCIe？怎么确认？
2. 你实测的 α 是多少？和教程的量级表对得上吗？对不上可能是什么原因？
3. custom AR 在你的机器上覆盖到多大的消息？和 `CUSTOM_ALL_REDUCE_MAX_SIZES` 一致吗？
4. 你这台机器的最优 TP 是几？为什么再大就变慢？
5. nsys 测出来的通信占比是多少？在 prefill 和 decode 阶段分别是多少？
6. 如果只能保留一个实验结果，你留哪个？为什么？
