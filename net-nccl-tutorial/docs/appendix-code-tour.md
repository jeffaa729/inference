# 附录 A　vLLM 网络/通信代码地图 + 读数路径

> 用途：① 当速查索引；② 当「从零读 vLLM 通信代码」的路线图。
> 行号基于本仓库当前 checkout；**符号名稳定，行号会漂移** —— 用符号名搜索。

---

## A.1 代码地图（按职责）

### 核心：进程组与通信抽象

| 文件 | 关键符号 | 看什么 |
|---|---|---|
| `vllm/distributed/parallel_state.py` | `GroupCoordinator`, `initialize_model_parallel`, `init_distributed_environment` | **一切的中心**：组拓扑 + 通信算子分发 |
| `vllm/distributed/communication_op.py` | `tensor_model_parallel_all_reduce` 等 | 只是一层语义别名（43 行） |
| `vllm/distributed/utils.py` | `StatelessProcessGroup`, `init_gloo_process_group`, `create_tcp_store` | 不依赖 torch 的轻量进程组 + gloo 组 |
| `vllm/utils/network_utils.py` | `get_ip`, `get_open_port`, `get_distributed_init_method`, `get_file_store_init_method`, `get_tcp_uri` | **所有网络/端口/IP 逻辑在这里**（不在 `distributed/utils.py`） |
| `vllm/v1/executor/multiproc_executor.py` | `_init_executor`, `WorkerProc` | 单机多进程启动 + `file://` 引导 |

### 设备通信器（后端选择）

| 文件 | 关键符号 | 看什么 |
|---|---|---|
| `vllm/distributed/device_communicators/cuda_communicator.py` | `CudaCommunicator.all_reduce`（8 路选择链）、`__init__`（构造期 gate）、all2all 后端 if/elif | **最重要的一站** |
| `vllm/distributed/device_communicators/base_device_communicator.py` | `All2AllManagerBase`, `DeviceCommunicatorBase` | 基类契约 + `internode` 探测 |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | `CustomAllreduce`, `should_custom_ar`, `capture` | CUDA IPC + CUDA Graph 适配 |
| `vllm/distributed/device_communicators/pynccl.py` | `PyNcclCommunicator`, `from_unique_id_bytes`, `all_gatherv` | ctypes 直连 NCCL + uniqueId 交换 |
| `vllm/distributed/device_communicators/pynccl_wrapper.py` | `NCCLLibrary`, `ncclCommProperties`, 全部 NCCL 符号绑定 | NCCL C API 的完整 ctypes 声明 |
| `vllm/distributed/device_communicators/pynccl_allocator.py` | `nccl_symm_mem_context`, `is_symmetric_memory_enabled` | 对称内存的四层实现 |
| `vllm/distributed/device_communicators/all_reduce_utils.py` | `CUSTOM_ALL_REDUCE_MAX_SIZES`, `SYMM_MEM_ALL_REDUCE_MAX_SIZES`, `should_nccl_symm_mem_allreduce`, `gpu_p2p_access_check` | **调优表 + P2P 真检测** |
| `vllm/distributed/device_communicators/all2all.py` | 9 个 `*All2AllManager` 类 | 所有 all-to-all 后端 |
| `vllm/distributed/device_communicators/symm_mem.py` | `SymmMemCommunicator` | torch 对称内存（multimem / two-shot） |
| `vllm/distributed/device_communicators/quick_all_reduce.py` | `QuickAllReduce`, `QuickReduceRegime` | ROCm 量化 all-reduce |
| `vllm/distributed/device_communicators/flashinfer_all_reduce.py` | `FlashInferAllReduce` | FlashInfer mnnvl/trtllm 后端 |
| `vllm/distributed/device_communicators/shm_broadcast.py` | `MessageQueue`, `ShmRingBuffer`, `SpinCondition` | **控制面**：共享内存 + ZMQ |
| `vllm/distributed/device_communicators/cuda_wrapper.py` | `CudaRTLibrary` | P2P 探测用的 cudart ctypes 绑定 |

### C++ / CUDA 侧

| 文件 | 看什么 |
|---|---|
| `csrc/custom_all_reduce.cuh` | `cross_device_reduce_1stage/2stage`、one-shot/two-shot 切换、`CustomAllreduce` C++ 类 |
| `csrc/custom_collective_common.cuh` | `Signal`、`RankData.ptrs[16]`、`kMaxBlocks=36`、同步原语 |
| `csrc/libtorch_stable/custom_all_reduce.cu` | torch op 绑定、`cudaIpcGetMemHandle`/`OpenMemHandle` |
| `csrc/custom_quickreduce.cu`, `csrc/quickreduce/quick_reduce.h` | ROCm QuickReduce |

### MoE 通信消费方

| 文件 | 看什么 |
|---|---|
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | `get_ep_all2all_manager`, `maybe_make_prepare_finalize`, `maybe_roundup_layer_hidden_size` |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/*.py` | 每种后端的 prepare/finalize 实现、wire format、stream 同步 |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py` | `SUPPORTED_HIDDEN_SIZES`, `DEEPEP_QUANT_BLOCK_SHAPE` |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py` | `xfer_atom_size = 512`、dispatch config 表 |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` | AgRs 路径的 prepare/finalize |

### 其他通信用途

| 文件 | 看什么 |
|---|---|
| `vllm/distributed/weight_transfer/nccl_engine.py` | 权重传输（RL 场景）用 NCCL broadcast |
| `vllm/distributed/weight_transfer/nccl_common.py` | `uid_init_process_group`, `stateless_init_process_group` |
| `vllm/distributed/eplb/eplb_communicator.py` | EPLB 的通信（独立 process group） |
| `vllm/distributed/elastic_ep/` | 弹性 EP（rank 动态加入/退出） |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` | KV cache 的 NIXL/RDMA 传输 |
| `vllm/ray/ray_env.py` | Ray worker 的环境变量传播（含 `NCCL_` 前缀） |
| `vllm/v1/executor/vllm_net_devices.py` | GPU↔NIC PCIe 亲和性自动选卡 |

### 通算融合 / 重叠（ch07）

| 文件 | 关键符号 | 看什么 |
|---|---|---|
| `vllm/v1/worker/ubatching.py` | `UBatchContext`, `_cpu_yield`, `dbo_*` 包装器, `make_ubatch_contexts` | **DBO 的心脏**：双线程 + 双 stream 的 ping-pong（241 行，建议全读） |
| `vllm/v1/worker/ubatch_utils.py` | `SMControlContextManager`, `create_sm_control_context`, `maybe_create_ubatch_slices`, `split_attn_metadata` | SM 仲裁 + batch 切分 + 元数据切分 |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | `UBatchWrapper`, `_capture_ubatches`, `CUDAGraphMetaData` | DBO × CUDA Graph（每 ubatch 一张 graph） |
| `vllm/v1/worker/gpu/ubatch_utils.py` | `UBatchRunner` | V2 runner 的 DBO（eager-only） |
| `vllm/compilation/passes/fusion/sequence_parallelism.py` | `FirstAllReduceRMSNormPattern`, `MiddleAllReduceRMSNormPattern`, `SP_MIN_HIDDEN_SIZE` | **SP 的编译期 pattern 匹配**（623 行） |
| `vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py` | eager 版 AR+RMSNorm 融合 | 手动融合（不走 torch.compile） |
| `docs/design/dbo.md` | DBO 设计文档 + **精确重叠时间线** | 面试引用它很加分 |
| `docs/design/fusions.md` | 所有融合 pass 的总览表 | 看清 SP / AsyncTP / 各种 fusion 的关系 |

### 进阶通信形态（ch08）

| 文件 | 关键符号 | 看什么 |
|---|---|---|
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | `FusedMoEPrepareAndFinalize`（:187）, Modular vs Monolithic | **MoE 通信的接口契约** |
| `.../prepare_finalize/deepep_ht.py` | DBO 编排、`xfer_atom_size`、sentinel 约定 | 唯一实现了 DBO stream 编排的后端 |
| `.../prepare_finalize/deepep_ll.py` | `SUPPORTED_HIDDEN_SIZES`、BatchedExperts 布局 | decode 首选后端 |
| `.../prepare_finalize/deepep_v2.py` | 两种模式契约、2 的幂取整理由 | 最详尽的设计注释 |
| `.../prepare_finalize/naive_dp_ep.py` | AgRs 路径 + scale swizzle 延迟 | 最易读的 wire format |
| `vllm/models/common/ops/sequence_parallel.py` | `sp_all_gather`, `sp_reduce_scatter`, `sp_shard`, `sp_padding_mask` | SP 的算子实现（68 行） |
| `vllm/v1/attention/ops/dcp.py` | `cp_lse_ag_out_rs`, MLA DCP manager | DCP 的集合通信（LSE all-gather 是关键） |
| `vllm/v1/attention/ops/pcp.py` | prefill KV gather | PCP 的通信模式 |
| `vllm/v1/kv_cache_layout.py` | `KVCacheLayout` 枚举 | KV 传输的物理布局契约 |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | `KVConnectorRole`, 7 个抽象方法 | **KV 传输的接口** |

### 文档（补充）

| 文件 | 内容 |
|---|---|
| `docs/design/dbo.md` | **DBO 设计文档**：动机、重叠时间线、启动命令、限制 |
| `docs/design/fusions.md` | 所有 fusion pass 总览（SP 是 AsyncTP 的前置条件） |
| `docs/features/disagg_prefill.md` | PD 分离：连接器列表、配置、**「不提升吞吐」的结论** |
| `docs/features/nixl_connector_usage.md` | NIXL 连接器：side channel 端口、lease、布局要求 |
| `docs/serving/context_parallel_deployment.md` | DCP 配置建议与约束（`-dcp` 与 `-tp` 的关系） |

### 文档

| 文件 | 内容 |
|---|---|
| `docs/serving/parallelism_scaling.md` | 并行策略选择指南（含「无 NVLink 时用 PP」的建议） |
| `docs/serving/expert_parallel_deployment.md` | EP 部署 + backend 选择表 + 多节点示例 |
| `docs/serving/data_parallel_deployment.md` | DP 部署 |
| `docs/serving/distributed_troubleshooting.md` | Ray/网络排障（`VLLM_HOST_IP`、`NCCL_SOCKET_IFNAME`） |
| `docs/training/weight_transfer/nccl.md` | NCCL 权重传输原理 |
| `vllm/distributed/kv_transfer/README.md` | KV 传输概览 ⚠️ **注：文中描述的两层抽象已被删除，见 ch08 §8.3.2** |

---

## A.2 四条读数路径（按目标选一条）

### 路径 1：理解「一次 TP all-reduce」——约 40 分钟

```
1. vllm/model_executor/layers/linear.py
   搜 RowParallelLinear.forward → 找到 tensor_model_parallel_all_reduce 调用
2. vllm/distributed/communication_op.py            （43 行，读完）
3. vllm/distributed/parallel_state.py
   - 搜 class GroupCoordinator      → 读 docstring + __init__ + all_reduce
   - 搜 def initialize_model_parallel → 读布局注释（:1977 附近）
4. vllm/distributed/device_communicators/cuda_communicator.py
   - 搜 def all_reduce               → 读完整 8 路选择链
5. vllm/distributed/device_communicators/custom_all_reduce.py
   - 搜 class CustomAllreduce        → 读 __init__ 的 gate 链
   - 搜 def should_custom_ar         → 读准入条件
6. vllm/distributed/device_communicators/all_reduce_utils.py
   - 搜 CUSTOM_ALL_REDUCE_MAX_SIZES  → 读调优表
```

**读完你应该能默画出第 4 章 §4.9 的完整链路图。**

### 路径 2：理解「pynccl 为什么存在」——约 30 分钟

```
1. vllm/distributed/device_communicators/pynccl_wrapper.py
   - 读文件头注释（:1-30）  ← 存在理由在这里
   - 搜 class NCCLLibrary     → 看 __init__ 的符号绑定与失败处理
   - 搜 class ncclCommProperties → 看 NCCL 的能力清单
2. vllm/distributed/device_communicators/pynccl.py
   - 搜 def __init__          → 读两条 gate + uniqueId 交换
   - 搜 def _init_comm        → 读 warm-up all_reduce
   - 搜 def all_gatherv       → 读变长 all-gather 的实现技巧
3. vllm/utils/nccl.py         （129 行，读完）
4. vllm/distributed/device_communicators/pynccl_allocator.py
   - 读内联 C++ 源码（:25-34）+ nccl_symm_mem_context
```

### 路径 3：理解「MoE all-to-all」——约 60 分钟

```
1. docs/serving/expert_parallel_deployment.md
   - 读 Backend Selection Guide 表
2. vllm/distributed/parallel_state.py
   - 搜 expert parallel group → 读 EP 组的构造（:2087-2096）
3. vllm/distributed/device_communicators/all2all.py
   - 读 AgRsAll2AllManager（:44-153）—— 最基础，先读这个
   - 读 DeepEPLLAll2AllManager（:271-374）—— 对比 HT
   - 读 DeepEPV2All2AllManager（:1005-1092）—— 看 GIN 检查
4. vllm/distributed/device_communicators/base_device_communicator.py
   - 读 All2AllManagerBase.__init__（internode 探测 + 构造顺序注释）
5. vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py
   - 读 SUPPORTED_HIDDEN_SIZES 及其断言位置
6. vllm/model_executor/layers/fused_moe/all2all_utils.py
   - 搜 maybe_make_prepare_finalize → 看通信后端与专家 kernel 的解耦
```

### 路径 4：排障速查（不需要通读）

```
1. vllm/distributed/device_communicators/cuda_communicator.py
   搜 _log_all_reduce_backend_selection → 知道日志里会打什么
2. vllm/utils/network_utils.py
   搜 def get_ip             → 确认 IP 选择逻辑
   搜 def get_open_port      → 确认端口逻辑
3. vllm/v1/executor/multiproc_executor.py
   搜 distributed_init_method → 确认是 file:// 还是 tcp://
4. vllm/ray/ray_env.py         → 环境变量怎么传给 worker
5. docs/serving/distributed_troubleshooting.md
```

### 路径 5：理解「通算融合 / DBO」——约 50 分钟（★ 高价值）

```
1. docs/design/dbo.md
   - 读 Motivation + Introduction + 那张【重叠时间线】注释
   - 记住：DBO 的目标是重叠 MoE 的 all-to-all，不是 TP 的 all-reduce
2. vllm/v1/worker/ubatching.py（241 行，全读）
   - UBatchContext 的字段与两套事件（CPU threading.Event / GPU torch.Event）
   - _cpu_yield 的三个 assert（正确性证明）
   - switch_to_comm vs switch_to_comm_sync
   - dbo_get_previous_event 的「事件覆盖范围」问题
   - _register_ubatch_function 的「零开销」包装
3. vllm/v1/worker/ubatch_utils.py
   - SMControlContextManager：SM 怎么切、退出时怎么还
   - create_sm_control_context：为什么只有 DeepEP HT + DeepGEMM 支持
   - maybe_create_ubatch_slices：按 token 均分（不是按请求）
4. vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
   - 搜 dbo_ → 看唯一实现了 DBO 编排的后端
   - 读 :119-128 的注释（为什么事件必须在 yield 前捕获）
5. vllm/v1/worker/gpu_ubatch_wrapper.py
   - _capture_ubatches 的 docstring（4 步初始化顺序）
6. vllm/compilation/passes/fusion/sequence_parallelism.py
   - 读类 docstring（SP 改图的规则）
   - 记住那句 "does not directly yield performance improvements"
```

### 路径 6：理解「MoE 通信的 wire format」——约 70 分钟

```
1. vllm/model_executor/layers/fused_moe/modular_kernel.py
   - 读 FusedMoEPrepareAndFinalize 基类（:187-260）的每个 abstract 方法
   - 分清 Modular（topk ids）vs Monolithic（router_logits）
2. prepare_finalize/naive_dp_ep.py（最易读，先看它）
   - dispatch/combine 传了哪些张量
   - scale swizzle 为什么推迟到 A2A 之后
3. prepare_finalize/deepep_ll.py
   - BatchedExperts 布局
   - SUPPORTED_HIDDEN_SIZES 与 per-token scale 的 assert
4. prepare_finalize/deepep_ht.py
   - get_dispatch_layout 怎么算 sizes
   - sentinel 的 num_experts-1 / 0 约定（读注释）
5. prepare_finalize/deepep_v2.py
   - 两种模式的契约（docstring）
   - 2 的幂取整 + cicc storm 的理由
6. vllm/distributed/device_communicators/all2all.py
   - 对比各 manager 的 buffer 大小与 max_sms_used
```

---

## A.3 环境变量索引（vLLM 自有的，与 `NCCL_*` 区分）

### 网络 / 引导

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| `VLLM_HOST_IP` | `""` | **指定本机 IP**（多网卡必设） | `vllm/envs.py:715`，用于 `network_utils.py:34` |
| `VLLM_PORT` | `None` | 端口基址；**必须是纯数字** | `vllm/envs.py:720`（`get_vllm_port` `:519-545`） |
| `VLLM_LOOPBACK_IP` | `""` | 强制 loopback IP | `vllm/envs.py:1844` |
| `VLLM_RPC_BASE_PATH` | 系统临时目录 | ZMQ `ipc://` 路径根 | `vllm/envs.py:723-725` |
| `VLLM_DP_MASTER_IP` | `"127.0.0.1"` | DP master IP | `vllm/envs.py:1469` |
| `VLLM_DP_MASTER_PORT` | `0` | DP master 端口（并预留 [p, p+10)） | `vllm/envs.py:1471` |
| `VLLM_DP_RANK` / `VLLM_DP_RANK_LOCAL` / `VLLM_DP_SIZE` | `0` / `=DP_RANK` / `1` | SPMD DP 身份 | `vllm/envs.py:1460/1463/1467` |
| `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` | `False` | 用新 `split_group` 路径建组 | `vllm/envs.py:935-937` |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` / `_PORT` | `"localhost"` / `5600` | NIXL 握手 | `vllm/envs.py:1677/1681` |

**vLLM 内部默认端口**（冲突时改这些）：

| 端口 | 值 | 位置 |
|---|---|---|
| `data_parallel_master_port` | **29500** | `vllm/config/parallel.py:147` |
| `data_parallel_rpc_port` | **29550** | `vllm/config/parallel.py:143` |
| `master_port` | **29501** | `vllm/config/parallel.py:282` |
| API server `port` | **8000** | `vllm/entrypoints/launchers/cli_args.py:255` |
| NIXL side channel | **5600** | `vllm/envs.py:1681` |
| P2P KV offload | **5710** | `vllm/envs.py:1691` |
| EC transfer | **5601** | `vllm/envs.py:1701` |

### NCCL / all-reduce 相关

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| `VLLM_NCCL_SO_PATH` | `None` | **指定 libnccl 路径**（不重装 PyTorch 换 NCCL 版本） | `vllm/envs.py:745` |
| `VLLM_NCCL_INCLUDE_PATH` | `None` | 编译对称内存垫片时找 `nccl.h` | `vllm/envs.py:2021` |
| `VLLM_USE_NCCL_SYMM_MEM` | **`False`** | 开关 NCCL 对称内存 | `vllm/envs.py:2017-2019` |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | **`True`** | 开关 torch 对称内存 all-reduce | `vllm/envs.py:1888-1890` |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | **`True`** | 开关 FlashInfer all-reduce | `vllm/envs.py:1892-1894` |
| `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` | `False` | 开关 FlashInfer PCIe IPC（实验性） | `vllm/envs.py:1898-1900` |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `"auto"` | `auto`/`trtllm`/`mnnvl` | `vllm/envs.py:1747-1751` |
| `VLLM_DISABLE_PYNCCL` | `False` | 禁用 pynccl，强制走 torch.distributed | `vllm/envs.py:1238-1240` |
| `VLLM_BATCH_INVARIANT` | `False` | **确定性模式**：禁用几乎所有快速路径 | `vllm/envs.py:631` |
| `VLLM_SKIP_P2P_CHECK` | **`True`** | 跳过真实 P2P 检测（`0`=强制真测） | `vllm/envs.py:1211`，用于 `custom_all_reduce.py:91` |
| `VLLM_CUDART_SO_PATH` | `None` | P2P 探测用的 libcudart 路径 | `vllm/envs.py:1458`，用于 `cuda_wrapper.py:111` |

⚠️ **注意 `VLLM_DISABLE_CUSTOM_ALL_REDUCE` 不存在**。禁用 custom AR 用引擎参数
`disable_custom_all_reduce`（`vllm/config/parallel.py:214`）→ `set_custom_all_reduce()`
（`vllm/distributed/parallel_state.py:1643-1645`，模块变量 `:1640`）。

### All-to-all / EP

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_DEEPEP_BUFFER_SIZE_MB` | **1024** | DeepEP 的 NVLink/RDMA buffer（MB） |
| `VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE` | `False` | 跨机也强制用 intra-node HT kernel |
| `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` | `False` | LL 模式允许 MNNVL |
| `VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE` | `False` | v2 两级 NVLink+RDMA 混合 |
| `VLLM_DEEPEP_V2_PREFER_OVERLAP` | `False` | v2 用更少 SM 换重叠 |
| `VLLM_DEEPEP_V2_ALLOW_MULTIPLE_REDUCTION` | `False` | 用精度换传输量 |
| `VLLM_NIXL_EP_MAX_NUM_RANKS` | **32** | NIXL EP 最大 rank 数 |

### 通算融合 / DBO（ch07）

| 变量 / 参数 | 默认 | 作用 |
|---|---|---|
| `--enable-dbo` (`enable_dbo`) | `False` | 开启 DBO；**硬编码 2 个 ubatch** |
| `--ubatch-size` (`ubatch_size`) | `0` | 手动指定 ubatch 数（>1 生效） |
| `--dbo-decode-token-threshold` | **32** | 纯 decode batch 超过此值才切 |
| `--dbo-prefill-token-threshold` | **512** | 含 prefill 的 batch 超过此值才切（代码带 `# TODO(lucas): tune`） |
| `VLLM_DBO_COMM_SMS` | **20**（CUDA SM）/ **64**（ROCm CU） | DBO 下留给通信的 SM 数；ROCm + DeepEP HT 时被强制为 `0` |
| `--all2all-backend` | `allgather_reducescatter` | **DBO 只允许** `deepep_low_latency` / `deepep_high_throughput` / `nixl_ep` |
| `disable_custom_all_reduce` | `False` | 关掉 custom AR（**不是**环境变量） |
| `VLLM_BATCH_INVARIANT` | `False` | 确定性模式：**同时禁用 SP / FP8/FP4 融合 / 快速 all-reduce** |

### CP / SP / KV 传输（ch08）

| 参数 | 默认 | 作用 |
|---|---|---|
| `--decode-context-parallel-size` / `-dcp` | `1` | 切分 decode 的 KV cache；**不增加 GPU 数**（复用 TP rank） |
| `--prefill-context-parallel-size` / `-pcp` | `1` | 切分 prefill 序列；**会增加 world size** |
| `--dcp-comm-backend` | `auto`（→`ag_rs`） | `ag_rs`（3 次 NCCL/层）vs `a2a`（2 次，用 Triton 合并） |
| `--dcp-q-replicate` | `None` | 复制 query 投影以跳过 decode 的 query all-gather（用冗余计算换通信） |
| `--cp-kv-cache-interleave-size` | `1` | KV cache 的交错粒度；须 ≤ `block_size` 且能整除 |
| `PassConfig.enable_sp` | `None`（自动） | 开启 Sequence Parallel（**是 AsyncTP 的前置条件**） |
| `PassConfig.sp_min_token_num` | `None`（自动） | SP 生效的最小 token 数（低于此值通信开销摊不平） |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` / `_PORT` | `"localhost"` / **5600** | NIXL 握手；端口 = base + `data_parallel_index` |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | **8998** | Mooncake 引导 HTTP 服务端口 |
| `VLLM_DBO_COMM_SMS` | 20 (CUDA) / 64 (ROCm) | DBO 下给通信的 SM 数 |

### 网卡自动选择

| 变量 | 作用 |
|---|---|
| `VLLM_GPU_NIC_PCIE_MAPPING` + `VLLM_NIC_SELECTION_VARS` | **必须成对设置**，自动按 PCIe 距离选网卡并写进 `NCCL_IB_HCA`/`UCX_NET_DEVICES`（`vllm/v1/executor/vllm_net_devices.py:180`） |

### NCCL 自己写的变量（vLLM 只在特定模式设置）

| 变量 | 由谁设置 | 值 |
|---|---|---|
| 一组确定性变量 | `batch_invariant.py:1147-1156`（仅 `VLLM_BATCH_INVARIANT`） | `NCCL_ALGO=allreduce:tree`、`NCCL_PROTO=Simple`、`NCCL_NVLS_ENABLE=0`、`NCCL_MAX_NCHANNELS=1` 等 10 个 |
| `NCCL_MAX_CTAS` | `vllm/distributed/eplb/eplb_utils.py:103` | `"8"` |
| `NCCL_ASYNC_ERROR_HANDLING` | `vllm/v1/worker/gpu_worker.py:360` | **被 vLLM `pop` 掉**（Ray 设的值会破坏 graph building） |

---

## A.4 手写一个「行号验证脚本」

因为 vLLM 迭代快，你自己核对时可以用这个脚本：

```python
#!/usr/bin/env python3
"""验证教程里的 path:line 引用是否仍然指向预期的符号。

用法： python code/verify_citations.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # 按需调整

# (文件, 行号, 该行应包含的子串)
CHECKS = [
    ("vllm/distributed/communication_op.py", 14, "get_tp_group().all_reduce"),
    ("vllm/model_executor/layers/linear.py", 1769, "tensor_model_parallel_all_reduce"),
    ("vllm/model_executor/layers/linear.py", 606, "tensor_model_parallel_all_gather"),
    ("vllm/distributed/parallel_state.py", 1977, "ExternalDP x DP x PP x PCP x TP"),
    ("vllm/distributed/parallel_state.py", 2117, "EPLB group"),
    ("vllm/distributed/device_communicators/cuda_communicator.py", 305, "def all_reduce"),
    ("vllm/distributed/device_communicators/custom_all_reduce.py", 108, "_SUPPORTED_WORLD_SIZES"),
    ("vllm/distributed/device_communicators/custom_all_reduce.py", 406, "world_size > 8"),
    ("vllm/distributed/device_communicators/all_reduce_utils.py", 31, "CUSTOM_ALL_REDUCE_MAX_SIZES"),
    ("vllm/distributed/device_communicators/all_reduce_utils.py", 109, "NCCL_SYMM_MEM_ALL_REDUCE_CONFIG"),
    ("vllm/distributed/device_communicators/all2all.py", 44, "class AgRsAll2AllManager"),
    ("vllm/distributed/device_communicators/all2all.py", 345, "RDMA so no SMs"),
    ("vllm/distributed/device_communicators/all2all.py", 1043, "_check_gin_support"),
    ("vllm/distributed/device_communicators/pynccl_wrapper.py", 63, "23102"),
    ("vllm/distributed/device_communicators/pynccl_allocator.py", 166, "22703"),
    ("vllm/v1/executor/multiproc_executor.py", 143, "get_file_store_init_method"),
    ("vllm/v1/worker/gpu_worker.py", 360, "NCCL_ASYNC_ERROR_HANDLING"),
    ("vllm/ray/ray_env.py", 40, "NCCL_"),
    ("vllm/utils/network_utils.py", 34, "def get_ip"),
]

def main() -> int:
    bad = 0
    for rel, lineno, needle in CHECKS:
        path = REPO / rel
        if not path.exists():
            print(f"MISSING FILE {rel}")
            bad += 1
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lineno > len(lines):
            print(f"OUT OF RANGE {rel}:{lineno}")
            bad += 1
            continue
        line = lines[lineno - 1]
        if needle in line:
            print(f"OK   {rel}:{lineno}")
        else:
            print(f"DRIFT {rel}:{lineno} 期望含 {needle!r}，实际: {line.strip()[:80]!r}")
            bad += 1
    print(f"\n{len(CHECKS) - bad}/{len(CHECKS)} 通过")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
```

**在写这份教程时，上面的检查全部通过。** 如果你在自己的 checkout 上跑出 `DRIFT`，
说明版本变了 —— 用符号名搜索（`grep -n "符号名" 文件`）重新定位即可。

---

## A.5 三个「读懂就能举一反三」的设计模式

读 vLLM 通信代码，真正可迁移的是这三个模式：

### 模式 1：分档 + 阈值表（而不是一个「最优」实现）

同一个语义操作（all-reduce / all-to-all），按**消息大小 × world size × 硬件**选不同实现，
阈值来自实测。

**证据**：`CUSTOM_ALL_REDUCE_MAX_SIZES`、`SYMM_MEM_ALL_REDUCE_MAX_SIZES`、
`NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`、`_QR_MIN_SIZE`、`FI_MNNVL_ALLREDUCE_MAX_SIZE_MB`。

**可迁移到**：任何性能敏感的路径选择（kernel 选择、量化策略、batch 策略）。

### 模式 2：gate 链 + 快速失败 + 可操作错误信息

每个实现都有一串准入检查，不满足就**明确禁用并说明原因**，而不是静默降速。

**证据**：`CustomAllreduce.__init__` 的 7 道 gate、`_check_gin_support` 的 RuntimeError、
`pynccl_wrapper.py` 加载失败时的日志（含 `VLLM_NCCL_SO_PATH` 提示）。

**可迁移到**：任何「多后端 + 硬件依赖」的系统。

### 模式 3：控制面 / 数据面分离 + 功能隔离

- 数据面走 GPU/NCCL/自定义 kernel；
- 控制面走共享内存 + ZMQ（`shm_broadcast.py`）；
- **不同用途的通信流用不同的 process group**（EPLB 独立建组）。

**可迁移到**：任何分布式系统的架构设计。

---

## A.6 术语中英对照（读英文代码/文档用）

| 中文 | 英文 | 备注 |
|---|---|---|
| 集合通信 | collective communication | |
| 规约 | reduce | |
| 全规约 | all-reduce | |
| 全收集 | all-gather | |
| 规约分发 | reduce-scatter | |
| 全交换 | all-to-all | MoE 的核心 |
| 点对点 | point-to-point (P2P) | |
| 进程组 | process group / communicator | |
| 通信器 | communicator | NCCL 的会话 |
| 通道 | channel | NCCL 的并发流 |
| 协议 | protocol | LL / LL128 / Simple |
| 拓扑 | topology | |
| 无损网络 | lossless network | |
| 拥塞控制 | congestion control | |
| 对称内存 | symmetric memory | |
| 全互联 | fully connected | NVLink 全互联 |
| 多节点 NVLink | MNNVL | GB200 NVL72 |
| 负载不均 / 拖后腿 | straggler | EP 的头号敌人 |
| 通信计算重叠 | comm-compute overlap | |
| 确定性 / 批次无关 | batch invariance | |
| 抢占 | preemption | KV cache 不足时 |
| 首 token 延迟 | TTFT (time to first token) | |
| 每 token 输出延迟 | TPOT (time per output token) | |
