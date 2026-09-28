# 外部插图来源与许可

本目录下的图片**全部来自 Wikimedia Commons**，许可已逐张通过 Commons API 的
`extmetadata.LicenseShortName` 核实（`SOURCES.json` 是脚本写入的机器可读版本）。

> ⚠️ **ShareAlike 提醒**：`CC BY-SA` 系列的图片带**传染性** —— 如果把本 PDF
> **对外分发**，严格解释下整份 PDF 需要以相同许可释出。**自己学习用不受影响。**
> 若将来要公开发布，优先替换成 `CC0` / `Public domain` / `CC BY` 的那几张：
> `memory_hierarchy.svg`、`roofline_model.png`、`amdahl.svg`(SA)、`fmt_fp32.svg`。

| 文件 | 许可 | 作者 / 标题 | 用在 | 来源 |
|---|---|---|---|---|
| `roofline_model.png` | **CC0** | Mewtow · Roofline model | docs/interview.md | [Commons](https://commons.wikimedia.org/wiki/File:Roofline_model.png) |
| `memory_hierarchy.svg` | **Public domain** | — · ComputerMemoryHierarchy | docs/ch01-networking-basics.md | [Commons](https://commons.wikimedia.org/wiki/File:ComputerMemoryHierarchy.svg) |
| `deepep_low_latency.png` | **MIT** | DeepSeek-AI, DeepEP（低延迟 all-to-all kernel） · deepep_low_latency | docs/ch08-advanced-networking.md | [Commons](https://github.com/deepseek-ai/DeepEP) |
| `deepep_normal.png` | **MIT** | DeepSeek-AI, DeepEP（高吞吐 all-to-all kernel） · deepep_normal | docs/ch08-advanced-networking.md | [Commons](https://github.com/deepseek-ai/DeepEP) |
| `dsv2_architecture.png` | **MIT** | DeepSeek-AI, DeepSeek-V2（仓库 LICENSE-CODE 为 MIT；模型权重另许） · dsv2_architecture | docs/interview.md | [Commons](https://github.com/deepseek-ai/DeepSeek-V2) |
| `fmt_fp32.svg` | **CC BY 3.0** | DnetSvg · IEEE 754 Single Floating Point Format | docs/interview.md | [Commons](https://commons.wikimedia.org/wiki/File:IEEE_754_Single_Floating_Point_Format.svg) |
| `fa3_pipelining.png` | **CC BY 4.0** | Shah et al., FlashAttention-3（2-stage 软件流水：GEMM 与 softmax 重叠） · fa3_pipelining | docs/appendix-single-gpu.md | [Commons](https://arxiv.org/abs/2407.08608) |
| `nvfp4_tensor.svg` | **CC BY 4.0** | NVFP4：16 个连续 FP4 元素共享一个 FP8(E4M3) scale · nvfp4_tensor | docs/interview.md | [Commons](https://arxiv.org/abs/2509.25149) |
| `vllm_block_sharing.svg` | **CC BY 4.0** | 同上（多序列共享物理块 —— prefix caching 的原型） · vllm_block_sharing | docs/appendix-single-gpu.md | [Commons](https://arxiv.org/abs/2309.06180) |
| `vllm_block_table.svg` | **CC BY 4.0** | Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention (SOSP'23) · vllm_block_table | docs/appendix-single-gpu.md | [Commons](https://arxiv.org/abs/2309.06180) |
| `vllm_pagedattention.svg` | **CC BY 4.0** | 同上（PagedAttention 的 KV 分块与注意力计算） · vllm_pagedattention | docs/appendix-single-gpu.md | [Commons](https://arxiv.org/abs/2309.06180) |
| `amdahl.svg` | **CC BY-SA 3.0** | Daniels220 · AmdahlsLaw | docs/interview.md | [Commons](https://commons.wikimedia.org/wiki/File:AmdahlsLaw.svg) |
| `fmt_fp16.svg` | **CC BY-SA 3.0** | — · IEEE 754r Half Floating Point Format | docs/interview.md | [Commons](https://commons.wikimedia.org/wiki/File:IEEE_754r_Half_Floating_Point_Format.svg) |
| `coll_all_gather.png` | **CC BY-SA 4.0** | RenderFlamingo · All-Gather | docs/ch02-collectives.md | [Commons](https://commons.wikimedia.org/wiki/File:All-Gather.png) |
| `coll_all_reduce.png` | **CC BY-SA 4.0** | RenderFlamingo · All-Reduce | docs/ch02-collectives.md | [Commons](https://commons.wikimedia.org/wiki/File:All-Reduce.png) |
| `coll_all_to_all.png` | **CC BY-SA 4.0** | RenderFlamingo · All-to-All | docs/ch02-collectives.md | [Commons](https://commons.wikimedia.org/wiki/File:All-to-All.png) |
| `coll_broadcast_collective_operation.png` | **CC BY-SA 4.0** | RenderFlamingo · Broadcast (collective operation) | docs/ch02-collectives.md | [Commons](https://commons.wikimedia.org/wiki/File:Broadcast_(collective_operation).png) |
| `coll_reduce.png` | **CC BY-SA 4.0** | RenderFlamingo · Reduce | docs/ch02-collectives.md | [Commons](https://commons.wikimedia.org/wiki/File:Reduce.png) |
| `coll_scatter.png` | **CC BY-SA 4.0** | RenderFlamingo · Scatter | docs/ch02-collectives.md | [Commons](https://commons.wikimedia.org/wiki/File:Scatter.png) |
| `fmt_bfloat16.svg` | **CC BY-SA 4.0** | — · Bfloat16 format | docs/interview.md | [Commons](https://commons.wikimedia.org/wiki/File:Bfloat16_format.svg) |
| `hbm_section.png` | **CC BY-SA 4.0** | — · Simplyfied diagram of HBM section view | docs/ch01-networking-basics.md | [Commons](https://commons.wikimedia.org/wiki/File:Simplyfied_diagram_of_HBM_section_view.png) |

## 为什么没有更多

以下几类图**故意没有采用网络图源**，而是保留了自绘的 mermaid：

| 想要的图 | 结论 |
|---|---|
| **Ring all-reduce 经典四步环图** | 核对过 6 篇 arXiv 集合通信论文（2402.13499 / 1804.06826 / 2510.03491 / 2402.06787 / 2206.03382 / 2211.15841），**全部是 arXiv 非独占许可，不可复用**；Baidu / Horovod / NCCL 的图同样不可用。⇒ 正文自绘的逐块环图保留 |
| **Nsight Compute SOL 四象限** | 已核实 `docs.nvidia.com/nsight-compute/CopyrightAndLicenses/` 适用 NVIDIA 软件许可协议（§9.1 保留所有权利、§2.4 禁止复制与衍生）⇒ **不可复用**，保留自绘 |
| **PCIe vs NVLink 带宽对比 / IB vs RoCE 协议栈 / mma fragment 布局** | 查找后**无可用授权图源** ⇒ 保留自绘 |
| **vLLM / FlashAttention / MLA / DeepEP 的论文原图** | 见 `_figs_ref/` 下的考证底稿（逐篇核对了 arXiv `<link rel="license">`）⇒ 可用的会加入，不可用的只留链接 |

（其余 vLLM / vLLM 内部的图——如 8 路 all-reduce 选择链、DBO 的 gantt、排障决策树——**只可能自绘**，因为它们是本材料从源码里读出来的结论。）
