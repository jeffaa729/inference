# SwiftEP 后端设计文档（Phase 0 勘察结论）

## 1. vLLM all-to-all 后端架构

vLLM 的 EP all-to-all 用「策略 + 工厂」模式组织，新增后端只需改 5 处 + 新增 1 个实现文件：

```text
vllm/config/parallel.py                     # All2AllBackend 字面量 + all2all_backend 字段
vllm/model_executor/layers/fused_moe/config.py          # use_<backend>_kernels 判定属性
vllm/model_executor/layers/fused_moe/all2all_utils.py   # maybe_make_prepare_finalize() 工厂
vllm/model_executor/layers/fused_moe/prepare_finalize/<backend>.py  # 后端实现（新增）
vllm/model_executor/layers/fused_moe/modular_kernel.py  # 抽象基类接口
```

## 2. 注入点清单

1. **`vllm/config/parallel.py`**（`All2AllBackend = Literal[...]`，line 42）：加入 `"swiftep"`。
2. **`vllm/model_executor/layers/fused_moe/config.py`**：
   - `FusedMoEParallelConfig` 加属性 `use_swiftep_kernels`（`all2all_backend == "swiftep"`）。
   - `FusedMoEConfig` 加同名前向属性（仿 `use_deepep_ht_kernels`，line 1473）。
3. **`vllm/model_executor/layers/fused_moe/all2all_utils.py`**：
   - `maybe_roundup_layer_hidden_size()` 加 `use_swiftep_kernels` 分支（若需对齐 hidden size）。
   - `maybe_make_prepare_finalize()` 加 `elif moe.use_swiftep_kernels:` 分支，构造
     `SwiftEPPrepareAndFinalize`。
4. **新增 `vllm/model_executor/layers/fused_moe/prepare_finalize/swiftep.py`**：
   实现 `SwiftEPPrepareAndFinalize(FusedMoEPrepareAndFinalizeModular)`。
5. （可选）`all2all_utils.py` 顶部按 `has_swiftep()` 条件导入（仿 deepep/flashinfer）。

## 3. 后端需实现的接口

`FusedMoEPrepareAndFinalizeModular`（`modular_kernel.py`）抽象方法：

| 方法 | 说明 |
|---|---|
| `activation_format` (property) | 返回 `FusedMoEActivationFormat.Standard`（与 DeepEP HT 一致） |
| `topk_indices_dtype()` | DeepEP HT 返回 `torch.int64` |
| `max_num_tokens_per_rank()` | 无限制返回 `None` |
| `num_dispatchers()` | 返回 `all2all_manager.world_size` |
| `output_is_reduced()` | combine 是否已跨 rank 归约；DeepEP HT 返回 `True` |
| `prepare(a1, topk_weights, topk_ids, num_experts, expert_map, apply_router_weight_on_input, quant_config, defer_input_quant) -> PrepareResultType` | 量化+dispatch；返回 `(dispatched_a, a_scales, expert_tokens_meta, expert_topk_ids, expert_topk_weights)` |
| `prepare_async(...)` / `finalize_async(...)` | 可选；`supports_async()` 置 True 时实现 |
| `finalize(output, fused_expert_output, topk_weights, topk_ids, apply_router_weight_on_input, weight_and_reduce_impl) -> None` | combine（含 topk 加权归约） |
| `maybe_roundup_layer_hidden_size()`（static，可选） | 对齐 hidden size |

参考实现：`prepare_finalize/deepep_ht.py`（DeepEP 高吞吐 = prefill 场景，SwiftEP 的直接对标）。
其 `prepare` 流程：quantize → `buffer.dispatch()`（返回句柄）→ `_receiver()` 组装
`expert_tokens_meta`（`mk.ExpertTokensMetadata.make_from_list`）+ 把 local expert id 偏移回
global 空间。`finalize`：topk 加权归约 → `buffer.combine()` → 拷贝回 `output`。

## 4. SwiftEP 实现思路

对照 DeepEP 的两个痛点，SwiftEP 用两项技术：

1. **Buffer fusion（零拷贝）**：消除 dispatch/combine 前的 staging 拷贝，token 直接从
   源 buffer 走 NVLink/RDMA，避免 DeepEP 的多段中间 buffer。
2. **TMA offload**：用 Hopper 的 TMA（Tensor Memory Accelerator）做跨卡 multicast/reduce，
   把 SM 从搬运中解放出来（DeepEP 的 SM 占用高是痛点），最大化 NVLink 利用率。
3. 辅助：RDMA scatter-gather list、QP 并行化、CUDA IPC（跨卡显存直接访问）。

落地要点：

- `prepare` 的 dispatch：输入 `a1`（[num_tokens, hidden]）+ `topk_ids/weights`；按路由
  计算 send/recv 布局；TMA 搬运；输出 per-expert 排列的 `expert_x` + `expert_tokens_meta`。
- `finalize` 的 combine：`fused_expert_output` 先 topk 加权归约，再 TMA reduce/搬运回
  `output`（[num_tokens, hidden]）。
- 若没有独立 SwiftEP 库，初期可复用 FlashInfer 的 one-sided NVLink 原语 + 自写 buffer
  融合逻辑；TMA 部分用 `torch`/自定义 Triton/CUDA 内核。

## 5. 风险

- **TMA 仅 Hopper+**（H100/H200/H20）；A100 无 TMA，只能做 buffer fusion + RDMA 部分。
- DeepEP 已很强，收益口径需与论文对齐（带宽/SM 占用/服务容量），并补 FlashInfer 基线。
- 需要多卡真机验证；单卡只能做逻辑单测。

## 6. 下一步

1. 读 SwiftEP 全文补全 buffer fusion / TMA 的实现细节。
2. 本地搭环境（uv/.venv/vLLM）。
3. 写 `swiftep.py` 骨架 + 单测；8 卡真机 benchmark。
