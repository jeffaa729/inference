# Kimi / MiniMax / GLM / Llama 4 架构事实速查（2024–2026 世代）

> 用途：中文技术面试出题（要求精确数字）。所有数字均来自下列 URL；`(推断)` 开头的是本文件的推导，非原文；抓不到的写「未找到」。
> 标注约定：**[官方]** = 技术报告 / 官方 model card / 官方 config.json / 官方文档；**[实现]** = vLLM / transformers 源码；**[三方]** = 社区实测或媒体。

---

## 1. 规格表（每个模型一行）

| 模型 | 总参数 | 激活参数 | 层数 | 专家数 / 激活数 | 注意力类型 | 上下文 | 备注 |
|---|---|---|---|---|---|---|---|
| **Kimi K2** (Instruct) | 1T（论文 Table 2 写 **1.04T**） | 32B（Table 2 写 **32.6B**） | 61（1 层 dense） | 384 routed / 8 + 1 shared | **MLA** | 128K（131,072） | sparsity 48；FP8 e4m3 原生权重 |
| **Kimi K2 Thinking** | 1T | 32B | 61（1 dense） | 384 / 8 + 1 shared | **MLA** | **256K**（262,144） | 原生 INT4 QAT（仅 MoE 部分） |
| **Kimi Linear** (48B-A3B) | 48B | 3B | 27 | 256 / 8 + 1 shared | **KDA + MLA 混合 3:1** | **1M**（1,048,576） | 20 KDA + 7 MLA；MLA 用 NoPE |
| **MiniMax-Text-01** | 456B | 45.9B | 80 | 32 / 2 | **Lightning(线性) + softmax 全注意力 7:1** | 训练 1M，外推 4M | 70 线性层 + 10 全注意力层 |
| **MiniMax-M1** (80k) | 456B | 45.9B | 80 | 32 / 2 | 同上（7:1 hybrid） | 1M（config max 10,240,000） | 首个开源大规模 hybrid-attention 推理模型；40K/80K 两档 thinking budget |
| **MiniMax-M2** | 230B（技术报告精写 **229.9B**） | 10B（报告 **9.8B**） | 62 | 256 / 8 | **全注意力（放弃线性混合）** | 192K（config 196,608） | `attn_type_list` 全为 1；3 个 MTP 模块；预训练 29.2T tokens |
| **GLM-4.5** | 355B | 32B | 92（3 dense + 89 MoE） | **160** routed / 8 + 1 shared | GQA + **partial RoPE** | 128K（131,072） | 1 层 MTP；QK-Norm 开 |
| **GLM-4.5-Air** | 106B | 12B | 46（1 dense + 45 MoE） | 128 / 8 + 1 shared | GQA + partial RoPE | 128K | 1 层 MTP；QK-Norm **关** |
| **GLM-4.6** | 355B | 32B | 92 | 160 / 8 + 1 shared | GQA + partial RoPE | **200K**（202,752） | config 与 4.5 逐字节相同，只改了 max_position_embeddings |
| **Llama 4 Scout** | 109B | 17B | 48（全 MoE） | 16 / **top-1** | **iRoPE**（36 RoPE local + 12 NoPE global） | 宣称 10M（config 10,485,760） | 训练/后训练上下文仅 256K |
| **Llama 4 Maverick** | 400B | 17B | 48（24 层 MoE） | 128 / **top-1** | **iRoPE**（同上） | 宣称 1M（config 1,048,576） | MoE 隔层插（step=2） |
| **DeepSeek-V3**（对照基线） | 671B | 37B | 61（3 dense） | 256 / 8 + 1 shared | MLA | 163,840 | 本表所有"与 DeepSeekMoE 异同"的参照 |

数字出处逐条：
- Kimi K2 1T/32B、61 层、384 专家、sparsity 48、64 头：**官方**论文 Table 2 与 §2.3 <https://ar5iv.labs.arxiv.org/html/2507.20534>；config <https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json>
- Kimi K2 Thinking 表格（1T / 32B / 61 / 1 dense / 7168 / 2048 / 64 heads / 384 experts / top-8 / 1 shared / 160K vocab / 256K ctx / MLA / SwiGLU）：**官方** model card <https://huggingface.co/moonshotai/Kimi-K2-Thinking/raw/main/README.md>
- Kimi Linear 48B/3B、layerwise KDA+MLA hybrid：**官方**摘要 <https://arxiv.org/abs/2510.26692>；层号清单见 config <https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json>
- MiniMax-01 "32 experts and 456 billion total parameters, of which 45.9 billion are activated for each token"：**官方**摘要 <https://arxiv.org/abs/2501.08313>
- MiniMax-M1 与 Text-01 同构：**官方** config <https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/raw/main/config.json>
- MiniMax-M2 230B/10B：**官方** model card <https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/README.md>
- GLM-4.5 "355B total parameters and 32B activated parameters"：**官方**论文 <https://arxiv.org/html/2508.06471v1>；Table 1 的 160 experts / 8 active / 1 shared / 96 heads / 8 KV heads / 3 dense + 89 MoE 同源
- GLM-4.5-Air 106B/12B：**官方** model card <https://huggingface.co/zai-org/GLM-4.5/raw/main/README.md>
- GLM-4.6 "355B / 32B"、"上下文窗口由 128K→200K"：**官方**文档 <https://docs.bigmodel.cn/cn/guide/models/text/glm-4.6>
- Llama 4 Scout "17 billion active parameters, 16 experts, and 109 billion total parameters"、"Maverick 17B active and 400B total"：**官方**博客 <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>；上下文 10M / 1M 见 **官方** MODEL_CARD <https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/MODEL_CARD.md>
- Llama 4 各 config 字段：**实现**（meta-llama 仓库 gated 401，使用字节一致的镜像）<https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json>、<https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct/raw/main/config.json>
- DeepSeek-V3 对照行：**官方** config <https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json>

来源: https://arxiv.org/abs/2507.20534 · https://arxiv.org/abs/2510.26692 · https://arxiv.org/abs/2501.08313 · https://arxiv.org/html/2508.06471v1 · https://ai.meta.com/blog/llama-4-multimodal-intelligence/

---

## 2. 每个模型的注意力方案

### 2.1 Kimi：K2 是 MLA；长上下文靠 YaRN；Kimi Linear 才是混合线性注意力

- **K2 是 MLA，不是别的。** 官方论文原文："employing Multi-head Latent Attention (MLA) as the attention mechanism"。model card 的 "Attention Mechanism" 一行直接写 **MLA**。且 K2 的 `config.json` 里 `"architectures": ["DeepseekV3ForCausalLM"]`、`"model_type": "kimi_k2"`，官方部署指南原话："Kimi-K2 reuses the `DeepSeekV3CausalLM` architecture and convert it's weight into proper shape to save redevelopment effort."
- MLA 结构数字（**仅存在于 config，论文正文未给**）：`q_lora_rank 1536`、`kv_lora_rank 512`、`qk_nope_head_dim 128`、`qk_rope_head_dim 64`、`v_head_dim 128`、`num_attention_heads 64`、`num_key_value_heads 64`。
- **为什么 K2 把头数从 128 砍到 64**：官方原话 "with a sequence length of 128k, increasing the number of attention heads from 64 to 128, while keeping the total expert count fixed at 384, leads to an **83% increase in inference FLOPs**"。
- **长上下文靠位置编码外推，不是稀疏也不是线性**：训练 4,096 → 400B tokens @4k + 60B tokens @32k 退火 → "To extend the context window to 128k, we employed the **YaRN** method"。config：`rope_theta 50000.0`，`rope_scaling {type:"yarn", factor:32.0, original_max_position_embeddings:4096}`。K2 Thinking 把 factor 提到 **64.0**、`max_position_embeddings 262144`。
- **K2 没有 MTP**：`num_nextn_predict_layers: 0`；GLM-4.5 论文 Table 1 也把 Kimi K2 的 "MTP Layers" 记为 **0**（对照 DeepSeek-V3 = 1、GLM-4.5 = 1）。
- **Kimi Linear 才是新注意力**：**KDA (Kimi Delta Attention)**，官方定义 "extends Gated DeltaNet with a finer-grained gating mechanism"；"While GDN, similar to Mamba2, employs a coarse head-wise forget gate, KDA introduces a **channel-wise** variant in which each feature dimension maintains an independent forgetting rate, akin to Gated Linear Attention (GLA)"。递推式（Eq.1）：`S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ`，`o_t = S_tᵀ q_t`。
- KDA 的高效性来自 **DPLR（Diagonal-Plus-Low-Rank）转移矩阵的特化变体** + **chunkwise 并行算法**（WY representation、UT transform、Appendix C 伪代码）。论文说明 "substantially reduces computation relative to general DPLR formulations while remaining consistent with the classical delta rule"。
- **混合比例 = 均匀 3:1**：官方原话 "Kimi Linear interleaves KDA with periodic full attention layers in a **uniform 3:1 ratio**"。落到层号（config，1-indexed）：`full_attn_layers = [4,8,12,16,20,24,27]`，其余 20 层是 KDA。KDA 侧：`head_dim 128`、`num_heads 32`、`short_conv_kernel_size 4`。
- Kimi Linear 的 MLA 层用 **NoPE**：`mla_use_nope: true`，`rope_scaling: null`，`rope_theta: 10000.0`；论文目录有专节 "No Position Encoding (NoPE) for MLA Layers"。
- 另有一条**稀疏**长上下文路线：**MoBA**（arXiv:2502.13189，Moonshot），摘要原文 "MoBA has already been deployed to support Kimi's long-context requests"；块大小 512、top-k 3，Llama-8B-1M-MoBA 用块 4096 / top-K 12（稀疏度 95.31%），并保留最后 3 层为 full attention。

来源: https://ar5iv.labs.arxiv.org/html/2507.20534 · https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json · https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/docs/deploy_guidance.md · https://ar5iv.labs.arxiv.org/html/2510.26692 · https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json · https://huggingface.co/moonshotai/Kimi-K2-Thinking/raw/main/README.md · https://ar5iv.labs.arxiv.org/html/2502.13189

### 2.2 MiniMax：线性 + 全注意力 7:1（Text-01 / M1），但 M2 又退回全注意力

- **Lightning Attention 是什么（论文原文）**："Lightning attention (Qin et al., 2024b,c) represents an **I/O-aware, optimized implementation of TransNormer** (Qin et al., 2022a). This approach identifies the primary bottleneck in the computational efficiency of existing linear attention mechanisms: the **slow cumsum operation** inherent in causal language modeling. To alleviate this problem, Lightning Attention proposes a novel **tiling** technique that effectively circumvents the cumsum operation. The key innovation lies in the strategic division of the attention calculation into two distinct components: **intra-block and inter-block** computations. The **left product** attention calculation is employed for intra-block operations, while the **right product** is utilized for inter-block operations." 补充线索：论文自述 lightning attention 由该团队自己提出（"originally proposed by our team members in Qin et al. (2024c)"）。
- **混合比例 = 7:1，官方原文两处**：§1 "one transformer block with **softmax attention follows every seven transnormer blocks** with lightning attention"；§2 "a transformer block with softmax attention is positioned **after every 7 transnormer blocks** of linear attention, leading to a total of **80 layers**"。M1 论文重复同一句。
- **可由官方 config 逐层复算**：`MiniMax-Text-01` 的 `attn_type_list` 是"7 个 `0` + 1 个 `1`"的循环，共 80 项 → **70 层线性（lightning）: 10 层全注意力（softmax）= 7:1**。`MiniMax-M1-80k` 的 `attn_type_list` 完全一致。
- 块结构（官方）："each comprises a **channel mixer** (an attention block) and a **feature mixer** (an MLP block)"；"We employ two types of channel mixers: lightning attention and softmax attention. The feature mixer is an **MoE**"。
- MiniMax-Text-01/M1 的具体配置：`num_hidden_layers 80`、`hidden_size 6144`、`num_attention_heads 64`（论文："Each attention module is composed of **64 heads**, each with a head dimension of **128**"）、`num_key_value_heads 8`、`head_dim 128`、`rotary_dim 64`、`postnorm: true`、三套 LayerNorm 缩放常数 `layernorm_full_attention_alpha = layernorm_linear_attention_alpha = layernorm_mlp_alpha = 3.5565588200778455`（beta 均为 1.0）。
- **位置编码（⚠️ 官方论文与官方 config 不一致）**：论文写 "RoPE is applied to **half of the attention head dimension**, with a **base frequency set to 10,000**"；但实际发布的 config 是 `rope_theta: 10000000`（1e7）、`rotary_dim 64`（恰好是 head_dim 128 的一半）、`rope_scaling: null`。**(推断)** 三方解读称长上下文分三阶段把 rope base 从 10k → 5M → 10M，但官方文本未见此说法。
- **上下文**：官方摘要 "The context window of MiniMax-Text-01 can reach up to **1 million tokens during training and extrapolate to 4 million tokens during inference** at an affordable cost"，以及 "offering a **20-32 times** longer context window"（对手是 GPT-4o / Claude-3.5-Sonnet）。长上下文靠**三阶段训练**（"a three-stage training procedure, successfully extending the context window to one million tokens"）+ 训练侧并行（"**varlen ring attention**" 和 "improved version of **Linear Attention Sequence Parallelism (LASP)**"）。**1M→4M 的具体外推方法名：未找到。**
- **M2 反转（有官方技术报告背书）**：MiniMax-M2 Series 技术报告（arXiv:2605.26494）原文——"The flagship M2 is a **62-layer** decoder-only Transformer with **229.9B total parameters** and only **9.8B activated per token**, organized as **256 fine-grained experts** with **sigmoid gating**, **full multi-head attention with GQA**, a **192K-token** native context window, and a **Multi-Token Prediction (MTP) module**… Pre-training on **29.2T tokens**"；以及直接点名的转向——"**M2 adopts full multi-head attention across all layers, departing from the hybrid design used in MiniMax-Text-01, which interleaves Lightning Attention with full attention.** Despite the theoretical appeal of efficient attention mechanisms, we found no variant that reliably matches full attention quality in production settings spanning reasoning, coding, and agent tasks."
- 官方另发一篇博客专讲这个转向："Why Did M2 End Up as a Full Attention Model?"，关键句："the price paid became obvious at a larger scale: the model had **clear deficits in complex, multi-hop reasoning** tasks"（指 Text-01 的 hybrid）；"in a real-world, industrial-grade system, the truth is that efficient attention still has some way to go before it can definitively beat full attention"；并自曝 "We accidentally left the SWA inference code in the open-source release… the performance wasn't good enough."（试过 hybrid SWA，inter/intra-layer mixing 都失败）。
- M2 的注意力细节（官方报告 + config）：48 个 query head / 8 个 KV head（GQA），"Rotary Position Embeddings (RoPE) are applied **throughout** the model"；`hidden_size 3072`、`head_dim 128`、`rotary_dim 64`、`rope_theta 5000000`、`vocab_size 200064`、`use_qk_norm: true`、`qk_norm_type: "per_layer"`、`use_mtp: true`、`num_mtp_modules: 3`。
- M2 的 MoE 消融（报告原文）："The baseline has **32 larger experts and selects two**. The fine-grained version splits the expert capacity into **128 smaller experts and selects eight**."（表 1：Activated 2B / Total 17.8B / 500B tokens；MATH 19.6→24.1，HumanEval 29.7→32.5）。
- M2 的 MoE 路由（报告原文）："Routing is implemented using **sigmoid gating with learnable expert-specific bias terms**, which improves load balancing while greatly reducing reliance on auxiliary losses."

来源: https://arxiv.org/abs/2501.08313 · https://arxiv.org/html/2501.08313v1 · https://arxiv.org/html/2506.13585v1 · https://huggingface.co/MiniMaxAI/MiniMax-Text-01/raw/main/config.json · https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/raw/main/config.json · https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/config.json · https://ar5iv.labs.arxiv.org/html/2605.26494 · https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model · https://platform.minimax.io/docs/guides/text-m2-full-attention · https://huggingface.co/MiniMaxAI/MiniMax-Text-01/raw/main/config.json · https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/raw/main/config.json · https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/config.json · https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model · https://platform.minimax.io/docs/guides/text-m2-full-attention

### 2.3 GLM：GQA + partial RoPE，无 MLA、无混合注意力、无 NoPE

- 官方论文原文："In the self-attention component, we employ **Grouped-Query Attention with partial RoPE**. Furthermore, we utilize **2.5 times more attention heads (96 heads for a 5120 hidden dimension)**… We also incorporate **QK-Norm** to stabilize the range of attention logits."
- 数字：`num_attention_heads 96`、`num_key_value_heads 8`、`head_dim 128`、`partial_rotary_factor 0.5`（→ 64 维做旋转）、`attention_bias: true`、`rope_theta 1000000`、`rope_scaling: null`。
- **位置编码换代**：官方论文 "When extending the sequence length to 32K, we also adjusted **RoPE's base frequency from 10,000 to 1,000,000** for better long-context modeling ability."
- **QK-Norm 是 4.5 与 Air 的差异点**：GLM-4.5 `use_qk_norm: true`，GLM-4.5-Air `use_qk_norm: false`（论文 Table 1 "QK-Norm | Yes | No"）。
- 头数为什么这么夸张：官方解释 "this increased head count does not improve training loss compared to models with fewer heads, it consistently improves performance on reasoning benchmarks such as MMLU and BBH"。
- 在论文与 config 中**均未找到**任何 hybrid / sliding-window / sparse attention 的表述，`rope_scaling` 为 null → 视为纯 full attention。
- **GLM-4（上一代 dense）**：官方论文（arXiv:2406.12793）写 "No Bias Except QKV"、"RMSNorm and SwiGLU"、"**Group Query Attention (GQA)**: We replaced Multi-Head Attention (MHA) with Group Query Attention (GQA)"、"Rotary positional embeddings (RoPE): We extended the RoPE to a **two-dimensional form**"，词表 150,000，GLM-4-9B config：40 层 / hidden 4096 / `multi_query_attention: true` / `multi_query_group_num: 2`。**GLM-4 与 GLM-4-Plus 的参数总量：未找到**（官方未公布）。

来源: https://arxiv.org/html/2508.06471v1 · https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json · https://huggingface.co/zai-org/GLM-4.5-Air/raw/main/config.json · https://huggingface.co/zai-org/GLM-4.6/raw/main/config.json · https://arxiv.org/abs/2406.12793 · https://huggingface.co/zai-org/GLM-4-9B/raw/main/config.json

### 2.4 Llama 4 的 iRoPE：**36 层 RoPE + 局部 chunked 注意力，12 层 NoPE + 全局注意力**

- 官方定义（Meta 博客原文）："We call this the **iRoPE** architecture, where "i" stands for "**interleaved**" attention layers, highlighting the long-term goal of supporting "**infinite**" context length, and "RoPE" refers to the rotary position embeddings **employed in most layers**."
- **逐层开关**：config 字段 `no_rope_layers`（两个模型完全一致）= `[1,1,1,0, 1,1,1,0, …]`；HF docstring 明确定义："A `1` at an index position indicates that the corresponding layer **will use RoPE**, while a `0` indicates that it's a **NoPE** layer." → **36 层用 RoPE，12 层是 NoPE（层号 3,7,11,…,47）**。
- **⚠️ 高频陷阱（面试可直接拿来考）**：字段名叫 `no_rope_layers`，但 `1 = 用 RoPE`、`0 = NoPE`，**与字段名的直觉相反**。很多三方解读因此把"3/4 的层是 NoPE"讲反了。可三方交叉验证的官方口径是 Meta 自己那句 "RoPE … employed in **most** layers"。
- **哪几层是 local、哪几层是 global**：HF config 里 `layer_types = ["chunked_attention" if no_rope else "full_attention" for no_rope in no_rope_layers]`；vLLM 侧 `self.global_layer = config.no_rope_layers[layer_idx] == 0`。合起来：**NoPE（12 层）→ 全局 full attention；RoPE（36 层）→ chunked local attention**。vLLM 博客把它写死为 "Llama 4 interleaves global attention (without RoPE) with chunked local attention (with RoPE) in a **1:3 ratio**"。
- **局部窗口大小**：`attention_chunk_size = 8192`（HF docstring："Chunk size for the attention computation. Smaller value enforces more local attention and lowers memory."）。mask 语义是非重叠分块：`(kv_idx - left_padding) // chunk_size == (q_idx - left_padding) // chunk_size`。
- **为什么**：iRoPE 的目标是长度泛化 + "infinite" context；NoPE 层负责全局、没有位置先验，因此需要在推理时对**全局(NoPE)层**做温度缩放。
- **推理期温度缩放（精确公式，vLLM 实现）**：`floor = floor((positions + 1.0) / floor_scale)`；`attn_scale = log(floor + 1.0) * attn_scale + 1.0`；然后 `q = q * attn_scale`。源码注释："We are applying temperature tuning (arXiv:2501.19399) to **NoPE layers**, where the inference-time temperature tuning function is customized to not affect short context while working at very long context." config：`attn_temperature_tuning: true`、`attn_scale: 0.1`、`floor_scale: 8192`；vLLM 在 `max_model_len > 32768` 时自动打开。
- RoPE 参数差异：Scout `rope_theta 500000` + `rope_scaling {factor 16.0, original_max_position_embeddings 8192, rope_type "llama3"}`；Maverick `rope_theta 500000` + `rope_scaling: null`。两者 KV 头都是 `num_attention_heads 40 / num_key_value_heads 8 / head_dim 128`，`hidden_size 5120`。
- **注意：Meta 从未发布 Llama 4 技术报告。** 网上流传的 "The Llama 4 Herd: Architecture, Training, Evaluation, and Deployment Notes"（arXiv:2601.11659）已被 arXiv **撤稿**（"This version has been removed by arXiv administrators due to incorrect authorship"），且非 Meta 出品、无 PDF。权威口径只有 Meta 博客 + `meta-llama/llama-models` 里的 MODEL_CARD.md。

来源: https://ai.meta.com/blog/llama-4-multimodal-intelligence/ · https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/MODEL_CARD.md · https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama4/configuration_llama4.py · https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/llama4.py · https://vllm.ai/blog/2025-04-05-llama4 · https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/raw/main/config.json

---

## 3. MoE 设计差异

| 模型 | routed 专家 | 激活 | shared 专家 | gate 打分 | 归一化 | 负载均衡手段 | 分组路由 |
|---|---|---|---|---|---|---|---|
| Kimi K2 | 384 | 8 | 1 | sigmoid | `norm_topk_prob: true` | `topk_method: "noaux_tc"`（无辅助损失的 bias 纠正）+ `aux_loss_alpha 0.001` + `seq_aux: true` | `n_group 1`、`topk_group 1` → **未启用**节点受限路由 |
| Kimi Linear | 256 | 8 | 1 | sigmoid（`moe_router_activation_func`） | `moe_renormalize: true` | `use_grouped_topk: true`，`routed_scaling_factor 2.446` | `num_expert_group 1`、`topk_group 1` |
| MiniMax-Text-01 / M1 | 32 | 2 | config `shared_intermediate_size: 0`（(推断) 即无共享专家） | softmax（`shared_moe_mode: "sigmoid"`；论文 Eq.1 用 `Softmax_i(TopK(x_t·W_g))`） | — | 论文：**global router**（受 GShard 启发的辅助损失 `L_aux = α_aux · (1/E) Σ f_i·m_i`，外加一次 allgather 同步各 EP group 待处理 token 数）；config `router_aux_loss_coef 0.001`、`router_jitter_noise 0.0`；**token-drop** 策略 | 无 |
| MiniMax-M2 | 256 | 8 | config `shared_intermediate_size: 0`（(推断) 即无共享专家） | sigmoid | `use_routing_bias: true` | **sigmoid gating + learnable expert-specific bias terms**（"greatly reducing reliance on auxiliary losses"）+ `router_aux_loss_coef 0.001` | 无 |
| GLM-4.5 / 4.6 | **160** | 8 | 1 | sigmoid（"sigmoid gates for MoE layers"） | `norm_topk_prob: true` | **"loss-free balance routing"**：bias 更新率 0.001（前 15T tokens）→ 0；外加 seq-level 均衡损失权重 **0.0001** | `n_group 1`、`topk_group 1` → 未启用 |
| GLM-4.5-Air | 128 | 8 | 1 | sigmoid | `norm_topk_prob: true` | 同上；`routed_scaling_factor 1.0` | 未启用 |
| Llama 4 Scout | 16 | **1** | 1 | sigmoid（`renormalize=False`） | 不重归一化 | `router_aux_loss_coef 0.001`、`router_jitter_noise 0.0` | 无 |
| Llama 4 Maverick | 128 | **1** | 1 | sigmoid | 不重归一化 | `router_aux_loss_coef 0.001` | 无 |

### 与 DeepSeekMoE 的异同（一句话一条）

- **Kimi K2 vs DeepSeekMoE**：同源（细粒度专家 + 共享专家 + sigmoid + `noaux_tc` 无辅助损失偏置 + `aux_loss_alpha 0.001`），但 routed 专家数 384 vs 256、`routed_scaling_factor` **2.827 vs 2.5**，且 K2 把节点受限路由关掉（`n_group/topk_group = 1`，V3 是 8/4）。
- **Kimi Linear vs DeepSeekMoE**：同为"细粒度 + 1 共享专家"，但规模减到 256 专家 / 27 层，并新增 `moe_renormalize` 与 `use_grouped_topk`，`routed_scaling_factor 2.446`。
- **MiniMax-Text-01/M1 vs DeepSeekMoE**：粗粒度得多（32 专家 top-2）、**没有共享专家**（`shared_intermediate_size: 0`），均衡靠自研的 **global router**（GShard 风格辅助损失 + 跨 EP group 的 allgather 同步）与 **token-drop**，**没有** DeepSeek 式的无辅助损失 bias 纠正。
- **MiniMax-M2 vs DeepSeekMoE**：专家粒度追平（256 top-8），并补上了 **learnable expert-specific bias**（`use_routing_bias: true`），思路与 DeepSeek 的 aux-loss-free bias 同源但自述为"greatly reducing reliance on auxiliary losses"；依旧**没有共享专家**，活跃参数只有 9.8B（远低于 V3 的 37B）。
- **GLM-4.5 vs DeepSeekMoE**：结构最接近（160 routed + 1 shared + top-8 + sigmoid + 无辅助损失均衡 + 1 层 MTP），差异在"更浅更宽 vs 更深更窄"的取向——官方原话 "we reduce the width … and increase its height … deeper models exhibited better reasoning capacity"，以及 GLM 额外保留了权重 **0.0001** 的 sequence-level 均衡损失。
- **Llama 4 vs DeepSeekMoE**：**反向设计**——粗粒度（16 或 128 专家）、**top-1** 路由、sigmoid 但不重归一化、有共享专家、只用传统 aux loss 0.001，官方未声明任何无辅助损失均衡（**未找到**）。

补充两处易错点：
- **GLM-4.5 的 routed 专家是 160，不是 128**（128 是 GLM-4.5-Air）。
- **GLM-4.5 论文 Table 1 的 "MTP Layers" 一行是判定各代是否带 MTP 的最快出处**：GLM-4.5 = 1、GLM-4.5-Air = 1、DeepSeek-V3 = 1、**Kimi K2 = 0**。

来源: https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json · https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json · https://huggingface.co/MiniMaxAI/MiniMax-Text-01/raw/main/config.json · https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/config.json · https://arxiv.org/html/2508.06471v1 · https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json · https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct/raw/main/config.json · https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json

---

## 4. 长上下文与推理成本

### 4.1 KV cache 公式

- **MLA 模型（Kimi K2 / K2 Thinking / Kimi Linear 的 MLA 层）**：每 token 每层只缓存压缩潜向量 + 共享 RoPE 键：
  `KV/token/layer = kv_lora_rank + qk_rope_head_dim`
  K2：`512 + 64 = 576` 元素/层；61 层 → **35,136 元素/token**。
- **GQA 模型（MiniMax / GLM / Llama 4 的注意力层）**：
  `KV/token/layer = 2 × num_key_value_heads × head_dim = 2 × 8 × 128 = 2048` 元素/层。
- **线性注意力层（MiniMax Text-01/M1、Kimi Linear 的 KDA 层）**：状态大小与序列长度**无关**（常数），这是它们长上下文成本优势的来源。

### 4.2 各模型 KV cache 大小（BF16 每元素 2 字节；FP8 减半）

| 模型 | 元素/token | BF16 每 token | BF16 @128K(131,072) | BF16 @1M(1,048,576) | FP8 @1M |
|---|---|---|---|---|---|
| Kimi K2 / K2 Thinking | 35,136 | 68.6 KiB | **8.58 GiB**（9.21 GB） | **68.6 GiB**（73.7 GB） | 34.3 GiB |
| Kimi Linear（仅 7 个 MLA 层） | 4,032 | 7.875 KiB | 0.98 GiB | **7.875 GiB** | 3.94 GiB |
| Kimi Linear（若 27 层全 MLA，反事实） | 15,552 | 30.4 KiB | 3.80 GiB | 30.4 GiB | 15.2 GiB |
| MiniMax-Text-01 / M1（仅 10 个全注意力层） | 20,480 | 40 KiB | **5 GiB** | **40 GiB** | 20 GiB |
| 　└ 70 个线性层 | 常数 | 常数（与长度无关） | 常数 | 常数 | 常数 |
| MiniMax-M2（62 层全注意力） | 126,976 | 248 KiB | **31 GiB**（33.3 GB） | 248 GiB（266 GB） | 124 GiB |
| GLM-4.5 / 4.6（92 层） | 188,416 | 368 KiB | **46 GiB**（49.4 GB） | 184 GiB（197 GB） | 92 GiB |
| GLM-4.6 @200K(202,752) | 188,416 | 368 KiB | — | 71.2 GiB @202,752 | 35.6 GiB |
| Llama 4 Scout/Maverick（12 个全局层） | 24,576 | 48 KiB | 6 GiB | **48 GiB** | 24 GiB |
| 　└ 36 个 chunked 局部层（窗口 8192） | 封顶 | — | 1.125 GiB（常数，不随总长增长） | 1.125 GiB | 0.56 GiB |
| Llama 4（若 48 层全注意力，反事实） | 98,304 | 192 KiB | 24 GiB | 192 GiB | 96 GiB |

> 上表为本文件按 config 字段推导 **(推断)**，公式与各项原始数字见 4.1 与各 config URL。交叉验证：
> - **Kimi Linear 的官方 "up to 75%"**：若 27 层全为 MLA 是 30.4 GiB，实际 7.875 GiB → 降幅 **74.1%**，与官方"up to 75%"吻合。
> - **Llama 4 的 iRoPE 收益**：12 全局 + 36 局部封顶 ≈ 49 GiB vs 全注意力 192 GiB → 约 **1/4**。
> - **MiniMax-M2 官方口径**："Memory requirements: **220 GB for weights, 240 GB per 1M context tokens**"（官方部署指南原话）；本表按 62 层算出 248 GiB ≈ 266 GB，两者差约 5%，差异应来自层数/进位口径。
> - **vLLM 实测**：Kimi-K2-Thinking 在 8×H200 上 "TP8 KV cache: **715,072** tokens"、"TP8+DCP8 KV cache: **5,721,088** tokens（8x）"。按 68.6 KiB/token 反推约 47 GiB/GPU 量级，与"1T INT4 权重约 500 GB / 8 卡"的剩余显存相容。

### 4.3 混合线性注意力的官方推理收益

| 模型 | 官方收益数字 | 口径 |
|---|---|---|
| Kimi Linear | **KV cache 降幅 up to 75%**；**1M 上下文解码吞吐 up to 6×**；Fig.1 给出 **TPOT 6.3× 更快（1.84 ms vs 11.48 ms）@1M**；RULER 128k 上 **3.98× 加速**；图 1(b) 标注 1M/512k/256k 分别 **6.3× / 5.7× / 4.8×** | 论文摘要 + Fig.1 caption；**测试硬件未找到** |
| MiniMax-01 | 摘要称可在百万级 token 上高效训练/推理，1M 训练上下文、**4M 推理外推**，"20-32 times longer context window"；算力侧官方数字：lightning attention 的定制 CUDA kernel "achieving over **75% Model Flops Utilization (MFU)** end-to-end on the **Nvidia H20**"；对照实验口径为 "**H800 GPUs with tensor parallelism set to 8**，W8A16"。**"比 FlashAttention-2 快 X 倍"这类倍数：未找到** | 论文 §1/§7 + 图注 |
| MiniMax-M1 | **对比 DeepSeek R1："M1 consumes less than 50% of the FLOPs at a generation length of 64K tokens, and approximately 25% of the FLOPs at a length of 100K tokens"** | 论文原文 |
| MiniMax-M2 | 反方向：技术报告明确 "M2 adopts full multi-head attention across all layers, departing from the hybrid design used in MiniMax-Text-01… we found no variant that reliably matches full attention quality in production settings spanning reasoning, coding, and agent tasks" | 官方技术报告 |

### 4.4 质量侧的对照数字（可用于"效率换质量"的讨论）

- Kimi Linear Fig.1：MMLU-Pro 4k **51.0（Kimi Linear）vs 47.2（MLA）**；RULER 128k **84.3 vs 81.3**。
- MoBA（Kimi 的稀疏路线）：Llama-8B-1M 上 RULER **0.7818（MoBA）vs 0.7849（full attention）**。
- MiniMax-01 宣称在标准与内部基准上"match the performance of state-of-the-art models like GPT-4o and Claude-3.5-Sonnet"。
- MiniMax-M2 的细粒度专家消融（官方报告，2B activated / 17.8B total / 500B tokens）：**MATH 19.6 → 24.1**、**HumanEval 29.7 → 32.5**（32 大专家 top-2 → 128 小专家 top-8）。

来源: https://arxiv.org/abs/2510.26692 · https://ar5iv.labs.arxiv.org/html/2510.26692 · https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json · https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json · https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/docs/vllm_deploy_guide.md · https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/config.json · https://arxiv.org/abs/2501.08313 · https://arxiv.org/html/2501.08313v1 · https://arxiv.org/html/2506.13585v1 · https://ar5iv.labs.arxiv.org/html/2605.26494 · https://huggingface.co/zai-org/GLM-4.6/raw/main/config.json · https://recipes.vllm.ai/moonshotai/Kimi-K2-Thinking · https://ar5iv.labs.arxiv.org/html/2502.13189

---

## 5. 训练稳定性 / 优化器

| 模型 | 优化器 | 稳定性技巧 | 关键数字 |
|---|---|---|---|
| **Kimi K2** | **MuonClip**（Muon + QK-clip） | **QK-clip**：更新后按 head 重新缩放 Q/K 投影权重，抑制注意力 logits 爆炸 | 阈值 **τ = 100**；"pre-trained on **15.5 trillion tokens with zero loss spike**"；对比：小规模（9B activated / 53B total）用 vanilla Muon 时 logits "quickly exceed a magnitude of 1000" |
| **GLM-4.5** | **Muon**（除 word embedding / bias / RMSNorm 权重外全用） | loss-free balance routing + sigmoid gate + QK-Norm（仅 4.5）+ 更深更窄 | Newton–Schulz N=5；momentum μ=0.95；scaled Muon update RMS 0.2；LR warmup→**2.5e-4**，cosine decay 到 **2.5e-5**；batch 16M→64M tokens（前 500B tokens）；weight decay 0.1；**23T tokens**；MTP 损失权重 λ=**0.3**（前 15T）→ **0.1** |
| **MiniMax** | 未找到预训练优化器 | RL 算法 **CISPO**（Clipped IS-weight Policy Optimization）："clips **importance sampling weights** rather than token updates"，比 DAPO **快 2×**；M1 论文自述 RL 训练中遇到"serious precision issues"；M2 博客称"Linear attention is currently **far more sensitive to numerical precision** than full attention" | RL 成本：512× H800 / 3 周 / **$534,700** |
| **Llama 4** | 未找到 | **MetaP**（自动定 per-layer learning rate 与初始化尺度）；FP8 预训练 | Behemoth "using FP8 and 32K GPUs, we achieved **390 TFLOPs/GPU**"；总 GPU 时长 **7.38M H100-80GB hours**（Scout 5.0M、Maverick 2.38M）；数据量 "more than **30 trillion** tokens"（MODEL_CARD：Scout ~40T、Maverick ~22T） |

**MuonClip 的 QK-clip 细节（Kimi K2 论文 §2.1，出题可深挖）**：
- 定义每个 head 的最大 logits `S_max^h = (1/√d) · max_{i,j} (Q_i^h · K_j^hᵀ)`，当 `S_max^h > τ` 时裁剪。
- 朴素做法 `W_q^h ← γ^α W_q^h`、`W_k^h ← γ^(1−α) W_k^h`，其中 `γ = min(1, τ/S_max)`，`α` 是平衡参数"typically set to **0.5**"；K2 实际用 **per-head** `γ_h = min(1, τ/S_max^h)`。
- **MLA 的特殊处理**（高频考点）：`q^C` 与 `k^C`（head-specific 部分）各乘 `√γ_h`；`q^R`（head-specific rotary）乘 `γ_h`；**`k^R`（shared rotary）保持不动**，以避免影响其他 head。
- Muon 更新式（Algorithm 1）：`M_t = μM_{t−1} + G_t`；`O_t = NewtonSchulz(M_t) · √max(n,m) · 0.2`（"Match Adam RMS"）；`W_t = W_{t−1} − η(O_t + λW_{t−1})`。
- K2 训练超参：15.5T tokens、WSD 调度；"The first **10T** tokens were trained with a constant learning rate of **2e-4** after a 500-step warm-up, followed by **5.5T** tokens with a cosine decay from 2e-4 to **2e-5**. Weight decay was set to **0.1** throughout, and the global batch size was held at **67M** tokens."
- K2 训练集群：NVIDIA **H800**，节点内 NVLink/NVSwitch，节点间 **8×400 Gbps RoCE**；并行策略 **16-way PP + 16-way EP + ZeRO-1 DP**。

来源: https://ar5iv.labs.arxiv.org/html/2507.20534 · https://arxiv.org/html/2508.06471v1 · https://ai.meta.com/blog/llama-4-multimodal-intelligence/ · https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/MODEL_CARD.md · https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model

---

## 6. 部署侧

### 6.1 官方/半官方部署配置（EP/TP）

| 模型 | 精度 | 推荐硬件 | 并行配置 | 来源 |
|---|---|---|---|---|
| Kimi K2 | FP8 | **16× H800 或 16× H200**（128k seqlen 的最小部署单元） | TP16，或 **DP16 + EP** | 官方 deploy_guidance |
| Kimi K2 | FP8 | 8× MI300X/MI325X/MI355X | TP8 | 同上 |
| Kimi K2 | — | 超过 16 卡时 TP 需叠加 PP | `--tensor-parallel-size 8 --pipeline-parallel-size 2` | vLLM recipe |
| Kimi K2 Thinking | INT4 | **8× H200 或 8× H20** | TP8；高吞吐加 `--decode-context-parallel-size 8`（DCP8） | vLLM recipe |
| Kimi Linear | BF16 | **单节点 4 或 8 卡** | TP4 / TP8 + `--max-model-len 1048576` | vLLM recipe |
| MiniMax-M2 | FP8 | 4× H200/H20/H100 或 4× A100/A800 | **TP4**；"Pure TP8 is not supported"，>4 卡用 **DP+EP** 或 **TP4+EP**；ROCm TP2 | vLLM recipe |
| MiniMax-M1 | INT8 专家 | 8× H800 **或** 8× H20 | `--tensor-parallel-size 8 --quantization experts_int8`；需 vLLM ≥ 0.9.2 | 官方 vLLM 部署指南 |
| GLM-4.5 | BF16 / FP8 | BF16：H100×16 / H200×8；FP8：H100×8 / H200×4 | sglang，TP | 官方 model card |
| GLM-4.5（跑满 128K） | FP8 | **H100×16 / H200×8** | sglang | 官方 model card |
| GLM-4.5-Air | FP8 | H100×2 / H200×1（128K 需 H100×4 / H200×2） | `--tensor-parallel-size 8`（示例） | 官方 model card |
| GLM-4.6 | FP8 | 4–8× H200（BF16 需 8× H200） | TP8；MTP 投机解码时 TP4 | vLLM recipe |
| Llama 4 Scout | INT4 | **单张 H100**（on-the-fly int4）；BF16 需多卡 | vLLM recipe `--tensor-parallel-size 1` | 官方博客 / MODEL_CARD |

### 6.2 实测吞吐/延迟（务必带硬件口径）

- **Kimi K2 Thinking，8×H200**（`vllm bench serve`，random 8000-in / 4000-out，request-rate 100，1000 prompts）：
  - TP8：请求吞吐 **1.25 req/s**，输出吞吐 **485.78 tok/s**，mean TTFT **271.2 s**，KV cache 715,072 tokens
  - **TP8+DCP8**：1.57 req/s（**+25.6%**），**695.13 tok/s（+43.1%）**，mean TTFT **227.8 s（TTFT 改善 16.0%）**，KV cache **5,721,088 tokens（8×）**
  - 质量不退化：GSM8K exact_match（flexible）TP8 **0.9416** vs TP8+DCP8 **0.9386**
- **Kimi K2 在 H200 上加 `-dcp 8`**：vLLM recipe 称"observed ~**33% lower mean TTFT** and higher tok/s in internal benchmarks"。
- **Kimi K2 Thinking INT4 QAT**：官方称 "**roughly 2x generation speed improvement**"（低延迟模式，lossless）。
- **Llama 4（vLLM 博客，三方实测）**：8×H100 可服务 Scout 到 **1M** 上下文、Maverick 到约 **430K**；8×H200：Scout 最高 **3.6M**、Maverick 最高 **1M**。
- **MiniMax-M2 显存口径（官方）**：权重 **220 GB**，KV **240 GB / 1M context tokens**。官方换算："4× 96GB GPUs → 支持约 400K 上下文；8× 144GB GPUs → 支持约 3M 上下文"；另注 "The maximum context length per individual sequence remains **196K** tokens."
- **MiniMax-M1 官方长上下文服务口径（vLLM 部署指南）**："a server with **8 H800 GPUs can process context inputs up to 2 million tokens**, while a server equipped with **8 H20 GPUs can support ultra-long context processing capabilities of up to 5 million tokens**"；"requires vLLM version **0.9.2** or later for full support"。
- **MiniMax-M1 的 RL 成本（官方）**：hybrid attention + CISPO 让全量 RL 在 **512× H800** 上 **3 周**完成，"rental cost of just **$534,700**"；CISPO 相比 DAPO 有 **2× 加速**（在 Qwen2.5-32B 对照实验上）。
- **Kimi Linear 的效率数字**：TPOT **1.84 ms vs 11.48 ms**（vs MLA @1M，**6.3×**）；论文称 "drop-in compatible with existing full-attention pipelines, requiring no modification to caching or scheduling interfaces"。
- **Mooncake（Kimi 的 KV-cache 中心化 PD 分离服务系统）**：官方论文 "Compared to the baseline method, Mooncake can achieve up to a **525% increase in throughput** in certain simulated scenarios while adhering to SLOs. Under real workloads, Mooncake's innovative architecture enables Kimi to handle **75% more requests**." 生产 trace：23,608 条请求，平均输入 **7,590** tokens、平均输出 **182** tokens；SLO 设为 `TTFT_P90 = 10×`、`TBT_P90 = 5×`。
- **GLM-4.5 MTP 投机解码（三方 vLLM recipe）**：`--speculative-config.num_speculative_tokens 1`；"With **1** speculative token, acceptance rates typically exceed **90%**"；更大的值提高 mean acceptance length 但显著降低接受率。官方 model card 给的却是 `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（**两套口径不一致，值得出成"你怎么选"的题**）。
- **GLM-4.6**：官方称 "finishes tasks with about **15% fewer tokens** than GLM-4.5"。

来源: https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/docs/deploy_guidance.md · https://recipes.vllm.ai/moonshotai/Kimi-K2-Thinking · https://recipes.vllm.ai/moonshotai/Kimi-K2-Instruct · https://recipes.vllm.ai/moonshotai/Kimi-Linear-48B-A3B-Instruct · https://recipes.vllm.ai/MiniMaxAI/MiniMax-M2 · https://huggingface.co/MiniMaxAI/MiniMax-M2/raw/main/docs/vllm_deploy_guide.md · https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/raw/main/docs/vllm_deployment_guide.md · https://huggingface.co/zai-org/GLM-4.5/raw/main/README.md · https://recipes.vllm.ai/zai-org/GLM-4.6 · https://docs.vllm.ai/projects/recipes/en/stable/GLM/GLM.html · https://vllm.ai/blog/2025-04-05-llama4 · https://ar5iv.labs.arxiv.org/html/2407.00079 · https://ar5iv.labs.arxiv.org/html/2510.26692 · https://arxiv.org/html/2506.13585v1

---

## 7. 附：可作为"延伸题"的其他 2025 世代模型

### 7.1 Step-3（StepFun，arXiv:2507.19427）——"注意力算术强度"路线的代表

- 规格（官方摘要）：**321B 参数 VLM**，每 token 激活 **38B**（"more than DeepSeek-V3 and Qwen3 MoE 235B"）。
- 注意力：**MFA（Multi-Matrix Factorization Attention）**——官方原话 "significantly reduces both **KV cache size and computation** while maintaining high attention expressiveness"。
- **MFA 的 KV cache 压缩数字（来自 MFA 原始论文 arXiv:2412.19255，作者含 Step-3 团队成员）**："MFA enhances model capacity by efficiently scaling up both the number and dimension of attention heads through **low-rank matrix factorization in the Query-Key (QK) circuit**"；扩展版 **MFA-KR** "further reduces memory requirements by **repurposing the key cache as value** through **value projection re-parameterization**"。硬数字："the proposed architecture **outperforms MLA** and performs **comparably to MHA**, while reducing KV cache usage by up to **56%**（相对 MLA）and **93.7%**（相对 MHA）, respectively."
- 系统：**AFD（Attention-FFN Disaggregation）**——"decouples attention and Feed-Forward Network (FFN) layers into specialized subsystems"。
- **可引用的硬数字（Hopper GPU，4K 上下文，FP8，无 MTP，50 ms TPOT SLA）**：Step-3 解码吞吐 **最高 4,039 tokens/s/GPU**，对比 DeepSeek-V3 在同一设定下的 **2,324 tokens/s/GPU**。
- 论文标题里的定位："Model-system Co-design"，核心论点是 **hardware-aligned attention arithmetic intensity + MoE sparsity + AFD**。

来源: https://arxiv.org/abs/2507.19427 · https://ar5iv.labs.arxiv.org/html/2507.19427v1 · https://www.stepfun.com/research/zh/step3 · https://arxiv.org/abs/2412.19255

### 7.2 Hunyuan-Large（腾讯，arXiv:2411.02265）——"KV cache 压缩 + 专家专属学习率"

- 规格（官方摘要）：**389B 总参数 / 52B 激活**，"currently the largest open-source Transformer-based mixture of experts model"，支持 **256K tokens**。
- 四项关键实践（官方原话列举）："large-scale synthetic data that is orders larger than in previous literature, a **mixed expert routing strategy**, a **key-value cache compression technique**, and an **expert-specific learning rate strategy**"。
- 论文另外研究了 MoE 的 **scaling laws 与 learning rate schedule**。
- 对比口径：官方称 "outperforms LLama3.1-70B and exhibits comparable performance when compared to the significantly larger LLama3.1-405B"。
- 专家数 / 激活数 / KV cache 压缩倍数的具体数值：**未找到**（摘要未给，需读正文）。

来源: https://arxiv.org/abs/2411.02265 · https://huggingface.co/papers/2411.02265

### 7.3 Seed-OSS-36B（字节 Seed）——稠密模型、512K 上下文、thinking budget

- 官方 config（**无任何 MoE 字段 → 稠密模型**）：`num_hidden_layers 64`、`hidden_size 5120`、`intermediate_size 27648`、`num_attention_heads 80`、`num_key_value_heads 8`、`head_dim 128`、`rope_theta 10000000.0`、`max_position_embeddings 524288`（**512K**）、`vocab_size 155136`、`attention_bias: true`、`tie_word_embeddings: false`。
- 特色是 **thinking budget**（可控思考预算），而非新注意力；注意力即标准 **GQA**（80 Q head / 8 KV head，10:1）。
- 参数量口径 "36B"、训练 token 数等：**未找到**（未在抓取到的页面中出现）。

来源: https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct/raw/main/config.json

### 7.4 ERNIE 4.5（百度）——"异构多模态 MoE"

- 家族规格（官方博客）：**10 个变体**；MoE 有 **47B 和 3B 两档激活参数**，"with the largest model having **424B total parameters**"，外加一个 **0.3B 稠密**模型。命名即规格：`ERNIE-4.5-300B-A47B`（文本旗舰）、`ERNIE-4.5-21B-A3B`、`ERNIE-4.5-VL-424B-A47B`（最大）、`ERNIE-4.5-VL-28B-A3B`、`ERNIE-4.5-0.3B`。
- 核心创新（官方原文）："we propose a novel **heterogeneous modality structure**, which supports **parameter sharing across modalities** while also allowing **dedicated parameters for each individual modality**"；配套 **modality-isolated routing**、**router orthogonal loss**、**multimodal token-balanced loss**。
- 训练基础设施：**异构混合并行 + 分层负载均衡**、intra-node expert parallelism、FP8 混合精度；官方口径 "We achieve **47% Model FLOPs Utilization (MFU)** in our largest ERNIE 4.5 language model pre-training."
- 推理侧：**multi-expert parallel collaboration** + **convolutional code quantization**，官方称可实现 **4-bit/2-bit lossless quantization**；PD 分离 + dynamic role switching。
- 后训练：SFT + DPO + 自研 **UPO（Unified Preference Optimization）**；VL 模型同时支持 thinking / non-thinking。
- 对比口径：官方称 `ERNIE-4.5-300B-A47B-Base` "surpasses DeepSeek-V3-671B-A37B-Base on **22 out of 28** benchmarks"。
- 层数 / 专家数 / 注意力类型 / 位置编码：**未找到**（官方博客未给；需读 ERNIE 4.5 Technical Report PDF 正文）。

来源: https://ernie.baidu.com/blog/posts/ernie4.5 · https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf

---

## 8. 明确「未找到」清单（避免面试时编造）

- Kimi K2 论文正文**没有**写出 MoE 路由函数名、归一化方式、bias 纠正方案与均衡损失数值（只有 config 字段）。
- Kimi K2 论文正文**没有**给出 `kv_lora_rank / qk_nope_head_dim / qk_rope_head_dim / v_head_dim / q_lora_rank`（仅 config 有）。
- Kimi Linear 效率实验（1.84 ms vs 11.48 ms、3.98×）**所用硬件未找到**；chunk size C 的具体取值、DPLR 相对通用 DPLR 的 FLOP 节省比例**未找到**。
- MiniMax-01 论文 §2.2.2.4 "Speed" 正文与 **"比 FlashAttention-2 快 X 倍" 之类的倍数未找到**（目录里有该节，正文抓取被截断）；§2.2.3 "Hybrid Architecture" 消融细节、§2.4 "Model Spec" 表同样未找到。
- MiniMax 的**预训练优化器未找到**；1M→4M 推理外推所用的**具名外推方法未找到**（论文只写了三阶段训练 + varlen ring attention + LASP）。
- **⚠️ MiniMax-01 的 RoPE base 官方自相矛盾**：论文写 "base frequency set to 10,000"，发布 config 写 `rope_theta: 10000000`；只有三方解读给出 10k→5M→10M 的分阶段解释，**官方文本未见**。
- GLM-4.5 论文中 **MTP 接受长度 / 投机解码加速比未找到**；GLM-4 / GLM-4-Plus 的**参数总量未找到**；GLM 系列的**流水线并行（PP）配置未找到**；GLM-4.6 **没有独立架构论文**（沿用 arXiv:2508.06471）。
- GLM 系列的 **NoPE 组件未找到**；GLM 论文/config 中**没有任何 hybrid / sliding-window / sparse attention 表述**。
- Llama 4 **不存在 Meta 官方技术报告**；"The Llama 4 Herd"（arXiv:2601.11659）已撤稿且非 Meta 出品。
- Llama 4 **无辅助损失均衡（auxiliary-loss-free）的任何来源**；**256K→…→10M 的上下文训练阶梯未找到**。
- Llama 4 Scout 的 **10M 上下文存在官方与实测的巨大落差**：Meta 博客称 10M、config `max_position_embeddings` 确实是 10,485,760，但 Meta 自家 cookbook 写 "On 8xH100, in bf16 you can get upto **1.4M** tokens"；三方服务商实际限制在 **128,000**（Groq、Fireworks）或 **328,000**（Together AI）。
- Llama 4 官方内部数据不一致：博客称"post-training 最多 8 张图"，MODEL_CARD 称"tested for image understanding up to **5** input images"。

来源: https://arxiv.org/abs/2507.20534 · https://arxiv.org/abs/2510.26692 · https://arxiv.org/abs/2501.08313 · https://arxiv.org/html/2508.06471v1 · https://arxiv.org/abs/2601.11659 · https://simonwillison.net/2025/Apr/5/llama-4-notes/ · https://arstechnica.com/ai/2025/04/metas-surprise-llama-4-drop-exposes-the-gap-between-ai-ambition-and-reality/ · https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama4/MODEL_CARD.md
