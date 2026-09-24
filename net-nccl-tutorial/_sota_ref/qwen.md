# Qwen 家族架构事实速查（Qwen2.5 / Qwen3 / Qwen3-MoE / Qwen2.5-Omni / Qwen3-Omni）

> 口径说明：`[官方报告]` = 技术报告/论文；`[官方配置]` = HuggingFace `config.json`；`[官方 README]` = 模型卡；
> `[vLLM 源码]` = 本地 vLLM 仓库实现；`[第三方]` = 非 Qwen 团队的实测或论文。
> 所有推断均以 `(推断)` 开头；未抓到的写「未找到」。范围限定 2024–2026 代的 Qwen2.5 / Qwen3 / Qwen3-MoE / Omni 系列。

---

## 1. 模型规格表

### 1.1 Qwen2.5 dense 系列

层数/heads/上下文/Tie 来自 `[官方报告]` 表 1；hidden/intermediate/vocab/config 上限/rope_theta 来自 `[官方配置]`。
`head_dim` 在任何 Qwen2.5 config 中都**未找到**该键 → 用 hidden/heads 推断。

| 版本 | 层数 | hidden | Q / KV heads | head_dim | intermediate | 上下文 / 生成 | 词表 | Tie | rope_theta | 注意力 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 896 | 14 / 2 | 64 (推断) | 4864 | 32K / 8K | 151936 | Yes | 1e6 | GQA + QKV bias |
| Qwen2.5-1.5B | 28 | 1536 | 12 / 2 | 128 (推断) | 8960 | 32K / 8K | 151936 | Yes | 1e6 | 同上 |
| Qwen2.5-3B | 36 | 2048 | 16 / 2 | 128 (推断) | 11008 | 32K / 8K | 151936 | Yes | 1e6 | 同上 |
| Qwen2.5-7B | 28 | 3584 | 28 / 4 | 128 (推断) | 18944 | 128K / 8K | 152064 | No | 1e6 | 同上 |
| Qwen2.5-14B | 48 | 5120 | 40 / 8 | 128 (推断) | 13824 | 128K / 8K | 152064 | No | 1e6 | 同上 |
| Qwen2.5-32B | 64 | 5120 | 40 / 8 | 128 (推断) | 27648 | 128K / 8K | 152064 | No | 1e6 | 同上 |
| Qwen2.5-72B | 80 | 8192 | 64 / 8 | 128 (推断) | 29568 | 128K / 8K | 152064 | No | 1e6 | 同上 |

- 预训练数据：7T → **18T** tokens；后训练 SFT 样本 > **100 万**；SFT 序列长度 **32,768**，学习率 7e-6 → 7e-7，weight decay 0.1，grad clip 1.0；DPO 约 **150,000** pair，lr 7e-7；GRPO 全局 batch **2048**，每 query 采样 **8** 条 `[官方报告]`。
- 词表：报告写「**151,643** regular tokens」，控制 token 从 3 扩到 **22**；config 的 `vocab_size` 是 151936 / 152064 `[官方配置]`（两处口径不同，面试可当陷阱题）。
- 口径冲突（值得注意）：报告表 1 说 0.5B/1.5B/3B 都是 32K，但 config 里 **1.5B 的 `max_position_embeddings` = 131072**（0.5B/3B = 32768）`[官方配置]`。
- MoE 专有字段（`num_experts` / `moe_intermediate_size` / `norm_topk_prob` / `shared_expert_intermediate_size` / `use_qk_norm`）在全部 Qwen2.5 dense config 中：**未找到**。
- `use_sliding_window` 全为 false；`sliding_window` = 32768（0.5B/3B）或 131072（1.5B/7B/14B/32B/72B）`[官方配置]`。

API 侧 MoE 型号（Qwen3 报告的对照表里给出的口径）：Qwen2.5-Turbo **42B total / 6B activated**；Qwen2.5-Plus **271B total / 37B activated** `[官方报告]`。

来源: https://ar5iv.labs.arxiv.org/html/2412.15115 | https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json | https://huggingface.co/Qwen/Qwen2.5-72B/raw/main/config.json | https://ar5iv.labs.arxiv.org/html/2505.09388

### 1.2 Qwen3 dense 系列

| 版本 | 层数 | hidden | Q / KV heads | head_dim | intermediate | 上下文 | 词表 | Tie | config max_pos | attention_bias |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 1024 | 16 / 8 | 128（显式键） | 3072 | 32K | 151936 | Yes | 40960 | false |
| Qwen3-1.7B | 28 | 2048 | 16 / 8 | 128 | 6144 | 32K | 151936 | Yes | 40960 | false |
| Qwen3-4B | 36 | 2560 | 32 / 8 | 128 | 9728 | 128K | 151936 | Yes | 40960 | false |
| Qwen3-8B | 36 | 4096 | 32 / 8 | 128 | 12288 | 128K | 151936 | No | 40960 | false |
| Qwen3-14B | 40 | 5120 | 40 / 8 | 128 | 17408 | 128K | 151936 | No | 40960 | false |
| Qwen3-32B | 64 | 5120 | 64 / 8 | 128 | 25600 | 128K | 151936 | No | 40960 | false |

- ⚠️ Qwen3 dense 的 `head_dim` 是**显式键 = 128**，而 4B（2560/32）与 32B（5120/64）的 hidden/heads = **80 ≠ 128**。即 Qwen3 **不能**用 hidden/heads 推 head_dim（与 Qwen2.5 相反）`[官方配置]`。
- 官方 README 口径：Qwen3-32B「Number of Parameters: **32.8B**；Non-Embedding: **31.2B**；Layers **64**；GQA **64 Q / 8 KV**；Context **32,768 natively and 131,072 tokens with YaRN**」`[官方 README]`。
- README 明示：config 的 `max_position_embeddings` = **40,960** 是「**32,768 输出 + 8,192 典型 prompt**」的预留，不是可用的上下文长度 `[官方 README]`。
- 预训练：**36T** tokens、**119** 种语言（Qwen2.5 为 29 种）；三阶段 S1 通用 >30T@4096、S2 推理约 5T@4096、S3 长上下文数百 B@32768（长文语料 75% 在 16,384–32,768，25% 在 4,096–16,384）`[官方报告]`。
- 报告写 tokenizer vocab = **151,669**，config 写 `vocab_size` = **151936**（口径不同）`[官方报告]` `[官方配置]`。
- 思考模式的推荐采样：thinking `Temperature=0.6, TopP=0.95, TopK=20, MinP=0`；non-thinking `Temperature=0.7, TopP=0.8, TopK=20, MinP=0`；输出长度建议 **32,768**，竞赛题可到 **38,912** `[官方 README]`。

来源: https://ar5iv.labs.arxiv.org/html/2505.09388 | https://huggingface.co/Qwen/Qwen3-32B/raw/main/README.md | https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json

### 1.3 Qwen3-MoE

| 版本 | 总参数 | 激活参数 | 层数 | hidden | Q / KV | head_dim | moe_intermediate | 专家（总/激活） | 上下文 | 词表 | tie |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | **30.5B**（Non-Emb 29.9B） | **3.3B** | 48 | 2048 | 32 / 4 | 128 | 768 | **128 / 8** | 32,768 原生 / 131,072 YaRN | 151936 | No |
| Qwen3-235B-A22B | **235B**（Non-Emb 234B） | **22B** | 94 | 4096 | 64 / 4 | 128 | 1536 | **128 / 8** | 同上 | 151936 | No |

- 两档都：`decoder_sparse_step` = 1、`mlp_only_layers` = []、`norm_topk_prob` = **true**、`router_aux_loss_coef` = **0.001**、`rope_theta` = 1e6、`attention_bias` = false、`sliding_window` = null `[官方配置]`。
- `intermediate_size`（dense FFN 分支）：30B-A3B = 6144；235B-A22B = 12288；`max_position_embeddings` = 40960（同为「32768 输出 + 8192 prompt」口径）。
- 两个 MoE config 中 `shared_expert_intermediate_size`：**未找到**（= 不启用共享专家，见 §4）。

来源: https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/README.md | https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json

### 1.4 Omni 系列

**Qwen2.5-Omni**（Thinker 文本塔 + 视觉塔 + 音频塔 + Talker + token2wav）

| 子模块 | 关键超参 | 数值 |
|---|---|---|
| Thinker text（7B） | hidden / 层数 / Q-KV / intermediate / vocab / max_pos | 3584 / 28 / 28-4 / 18944 / 152064 / 32768 |
| Thinker text（3B） | 同上 | 2048 / 36 / 16-2 / 11008 / 151936 / 32768 |
| Thinker 位置编码 | `rope_scaling.mrope_section`（两档相同） | **[16, 24, 24]** |
| 视觉编码器 | 出处 / 参数量 / 深度 / hidden / intermediate / patch / merge / 窗口 / 全注意力层 | Qwen2.5-VL ViT / 约 **675M** / 32 / 1280 / 3420 / **14** / 2×2 / 112 / [7,15,23,31] |
| 视觉编码器 | `tokens_per_second` | 25 |
| 音频编码器 | 出处 / 层数 / d_model / heads / ffn / mel bins / n_window | Whisper-large-v3 初始化 / 32 / 1280 / 20 / 5120 / **128** / 100 |
| Talker（7B 版） | hidden / 层数 / Q-KV / head_dim / intermediate / vocab | 896 / 24 / 12-4 / 128 / 18944 / 8448 |
| Talker（3B 版） | 同上 | 896 / 24 / 14-2 / 64 / 4864 / 8448 |
| token2wav | DiT depth/dim/heads/head_dim / num_embeds / mel_dim | 22 / 1024 / 16 / 64 / 8193 / 80 |
| token2wav | BigVGAN `upsample_initial_channel` / `upsample_rates` | 1536 / [5,3,2,2,2,2] |

**Qwen3-Omni-30B-A3B**（Thinker 与 Talker 都是 MoE）

| 子模块 | 架构名 | 参数量 | 关键超参 |
|---|---|---|---|
| Audio Encoder | **AuT**（Audio Transformer，从零训练） | **650M**（报告正文写约 0.6B） | 32 层 / d_model 1280 / 20 heads / ffn 5120 / mel 128 / `n_window` 50 / `n_window_infer` 800 / Conv2D 下采样 **8×** → **12.5 Hz** |
| Vision Encoder | **SigLIP2-So400M**（Qwen3-VL 初始化） | **540M**（正文写约 543M） | 27 层 / hidden 1152 / 16 heads / patch **16** / image_size 768 / deepstack [8,16,24] / merge 2×2 / `tokens_per_second` 2 |
| Thinker | MoE Transformer | **30B-A3B** | hidden 2048 / 48 层 / 32Q-4KV / head_dim 128 / moe_intermediate 768 / **128 专家，8 激活** / `use_qk_norm: true` / `shared_expert_intermediate_size: 0` / vocab 152064 / max_pos 65536 |
| Talker | MoE Transformer | **3B-A0.3B** | hidden 1024 / 20 层 / 16Q-2KV / head_dim 128 / moe_intermediate 384 / **128 专家，6 激活** / `shared_expert_intermediate_size: 768` / vocab 3072 |
| MTP | Dense Transformer | **80M** | 5 层 / hidden 1024 / 16Q-8KV / head_dim 128 / vocab 2048 / `num_code_groups` **16** |
| Code2Wav | ConvNet（因果卷积） | **200M** | codebook 2048 / semantic codebook 4096 / `num_quantizers` 16 / decoder_dim 1536 / 8 层 / `sliding_window` 72 / upsample_rates [8,5,4,3] |

- Qwen3-Omni 的 `thinker_config.text_config.rope_scaling` = `{"interleaved": true, "mrope_interleaved": true, "mrope_section": [24,20,20]}`，`position_id_per_seconds` = **13**；Qwen2.5-Omni 是 `mrope_section: [16,24,24]`，`position_id_per_seconds` = **25**，`seconds_per_chunk` = **2**（两者都是 2）`[官方配置]`。

来源: https://huggingface.co/Qwen/Qwen2.5-Omni-7B/raw/main/config.json | https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json | https://ar5iv.labs.arxiv.org/html/2509.17765

---

## 2. 注意力与归一化

### 2.1 GQA 配置（每档的 Q/KV head 数）

| 家族 | 0.5/0.6B | 1.5/1.7B | 3B / 4B | 7B / 8B | 14B | 32B | 72B | 30B-A3B | 235B-A22B |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5 | 14/2 | 12/2 | 3B: 16/2 | 28/4 | 40/8 | 40/8 | 64/8 | — | — |
| Qwen3 dense | 16/8 | 16/8 | 4B: 32/8 | 32/8 | 40/8 | 64/8 | — | — | — |
| Qwen3-MoE | — | — | — | — | — | — | — | 32/4 | 64/4 |
| Qwen2.5-Omni（Thinker） | — | — | 3B: 16/2 | 7B: 28/4 | — | — | — | — | — |
| Qwen3-Omni（Thinker） | — | — | — | — | — | — | — | 32/4 | — |

注意 Qwen3 dense 小模型反而更「不省 KV」：0.6B/1.7B 是 **16 Q / 8 KV**（GQA 比例 2:1），而 Qwen2.5 同档是 14/2、12/2（比例 7:1、6:1）；MoE 两档都是激进的 **4 个 KV head**，Qwen3-Omni 的 Talker 更极端（16Q / 2KV）。

来源: https://ar5iv.labs.arxiv.org/html/2412.15115 | https://ar5iv.labs.arxiv.org/html/2505.09388 | https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json

### 2.2 QK-Norm（Qwen3 的关键改动）

- 原文表述 `[官方报告]`：「Besides, we **remove QKV-bias used in Qwen2** and **introduce QK-Norm** (Dehghani et al. 2023) to the attention mechanism **to ensure stable training for Qwen3**.」
- 对比：Qwen2.5 的注意力里保留 **QKV bias**（报告原文：「**QKV bias** (Su 2023) in the attention mechanism and RMSNorm with pre-normalization」）`[官方报告]`。
- 加在哪里（vLLM 实现口径）`[vLLM 源码]`：在 `qkv_proj` 之后、**按 head 切分**施加 —— `q_by_head = q.view(..., head_dim)` → `self.q_norm(q_by_head)`；`k` 同理（`vllm/model_executor/models/qwen3.py` L150–165）。
- 对哪一维：`RMSNorm(self.head_dim)`，即**每个 head 的 head_dim 维**做归一化（不是 hidden 维、也不是跨 head）。
- 用什么算子：**RMSNorm**（不是 LayerNorm），`eps = rms_norm_eps`（Qwen3 系列 = **1e-06**）。
- 解决什么问题：官方只说「**ensure stable training**」`[官方报告]`；原始出处是 ViT-22B 的 QK-Norm（Dehghani et al. 2023）。
- 在 Qwen2.5 代码路径里 QK-Norm 是**可选**的：`qwen2.py` 里写着 `# QK Normalization support (used in BAGEL and some other models)`，只有 `if self.qk_norm:` 才建 `q_norm/k_norm`；而 `qwen3.py` 里 **无条件**创建 `q_norm` / `k_norm` `[vLLM 源码]`。

### 2.3 attention bias / mask 的特殊处理

- Qwen2.5：`qkv_proj` 带 bias，`o_proj` 不带；`attention_bias` 键在 Qwen2.5 config 中**未找到**（老版 config 不写）`[官方配置]` `[vLLM 源码]`。
- Qwen3 全系（dense + MoE + Omni Thinker）：`attention_bias` = **false** `[官方配置]`。但 vLLM 的 `qwen3.py` 仍从 config 读 `qkv_bias` 传给 `QKVParallelLinear(bias=qkv_bias)`，属于兼容位 `[vLLM 源码]`。
- 不使用 sliding window attention：Qwen3 全系 `sliding_window: null` / `use_sliding_window: false`；Qwen2.5 的 `sliding_window` 键存在但 `use_sliding_window` = false `[官方配置]`。
- Omni 侧有两处**真的改了 mask/注意力范围**：① Qwen2.5-Omni 音频编码器「from full attention over the entire audio to performing attention **in blocks of 2 seconds** each」；② codec→wave 的 DiT 用滑动窗口块注意力，感受野 **4 个 block = lookback 2 + lookahead 1** `[官方报告]`。Qwen3-Omni 的 AuT「uses flash attention with **dynamic attention window sizes**, covering attention query patterns ranging from **1 to 8 seconds**」`[官方报告]`。

### 2.4 长上下文扩展

| 项 | Qwen2.5 | Qwen3 |
|---|---|---|
| 预训练阶段 1 / 阶段 2 | 4,096 → 32,768 | 4,096 → 32,768 |
| RoPE base 提升 | **10,000 → 1,000,000**（ABF 技术） | **10,000 → 1,000,000**（ABF，Following Qwen2.5） |
| 推理期扩展 | **YARN + Dual Chunk Attention (DCA)**，序列长度容量 **4×** | **YARN + DCA**，容量 **4×** |
| 效果 | 其他模型到 **131,072**；Qwen2.5-Turbo 到 **1,000,000**（Turbo 训练用 rope base **10,000,000**，四阶段 32,768 → 65,536 → 131,072 → 262,144，每阶段 40% 最长 + 60% 短样本） | 全系 32,768 原生 → **131,072** |
| YaRN 具体参数（README） | 未找到（报告只写 YARN） | `{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`；`factor` 可按需调（如常跑 65,536 就设 2.0） |
| 官方警告 | — | 框架实现的是**静态 YaRN**，`factor` 不随输入长度变化，「potentially impacting performance on shorter texts」，不需要长上下文时**不建议**开 |

来源: https://ar5iv.labs.arxiv.org/html/2412.15115 | https://ar5iv.labs.arxiv.org/html/2505.09388 | https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/README.md

---

## 3. Qwen3 的「思考模式」切换

- 官方定位原文：把 **thinking mode**（复杂多步推理）与 **non-thinking mode**（快速上下文响应）**统一进同一个模型**，「eliminates the need to switch between different models — such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., **QwQ-32B**)」`[官方报告]`。
- **是同一份权重**：没有任何独立权重，只靠 prompt/chat template 切换；官方 README 明说「Uniquely support of **seamless switching between thinking mode and non-thinking mode within single model**」`[官方 README]`。
- **开关名就是 `enable_thinking`**：`tokenizer.apply_chat_template(..., enable_thinking=True/False)`，**默认 True**；「Setting `enable_thinking=False` disables thinking mode … will not include a `<think>...</think>` block」`[官方 README]`。
- 服务侧同名的 API 开关：vLLM 的 `--default-chat-template-kwargs '{"chat_template_kwargs":{"enable_thinking":false}}'` / 请求体 `chat_template_kwargs.enable_thinking`；vLLM 的 Qwen3 parser 默认 `self.thinking_enabled = chat_kwargs.get("enable_thinking", True)` `[vLLM 源码]` `[vLLM 文档]`。
- **软开关**：`enable_thinking=True` 时可在 user/system 里写 `/think` 与 `/no_think`，逐轮生效，「The model will follow the most recent instruction in multi-turn conversations」。注意：`enable_thinking=True` 时**无论有没有 `/no_think`，输出都会带 `<think>...</think>` 块（内容可能为空）**；`enable_thinking=False` 时软开关**完全无效** `[官方 README]`。
- 解析标记：`</think>` 的 token id 是 **151668**（官方示例用 `output_ids[::-1].index(151668)` 切分 thinking 与正文）`[官方 README]`；vLLM 侧对话边界 token 是 `<|im_start|>`/`<|im_end|>`（151644 / 151645）`[vLLM 源码]` `[官方配置]`。
- **thinking budget**：官方文档实现的是**客户端两步法**而不是模型内建参数 —— ①第一次请求设 `max_tokens = thinking_budget`（示例 **512**）拿 `reasoning_content`；②把 reasoning 以 `"<think>\n{reasoning}\n</think>\n\n"` 追加进 messages，用 `continue_final_message=True` 再请求一次拿最终答案（示例 `max_tokens` **1024**）。硬约束：`max_tokens > thinking_budget`，且 remainder 必须为正；若第一次被截断，官方示例会注入一句「Considering the limited time by the user, I have to give the solution based on the thinking directly now.」`[官方文档]`。报告层面把它描述为「Qwen3 introduces a **thinking budget mechanism**, allowing users to allocate computational resources adaptively during inference」，并称「increasing the thinking budget for thinking tokens leads to a consistent improvement」`[官方报告]`。
- 多轮里历史**不带** thinking 内容（由官方 Jinja2 chat template 实现）`[官方 README]`。
- 部署：官方 README 给的是 `sglang>=0.4.6.post1` 配 `--reasoning-parser qwen3`，以及 `vllm>=0.8.5` 配 `--enable-reasoning --reasoning-parser deepseek_r1`；当前 vLLM 已有独立的 `qwen3` reasoning parser（`vllm/parser/qwen3.py`，`CONFIG_NAME = "qwen3"`）`[官方 README]` `[vLLM 源码]`。
- 与预算相关的一条官方口径：Qwen3-235B-A22B 在 thinking 下 **AIME'24 85.7 / AIME'25 81.5 / LiveCodeBench v5 70.7 / CodeForces 2,056 / BFCL v3 70.8** `[官方报告]`。

来源: https://ar5iv.labs.arxiv.org/html/2505.09388 | https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/README.md | https://raw.githubusercontent.com/QwenLM/Qwen3/main/docs/source/getting_started/thinking_budget.md

---

## 4. MoE 部分（Qwen3-MoE）与 DeepSeekMoE 的逐条对比

### 4.1 Qwen3-MoE 的路由与专家设计

- 专家数 / 每 token 激活数：**128 total experts，8 activated experts per token**（30B-A3B 与 235B-A22B 相同）`[官方报告]` `[官方配置]`。
- **没有共享专家**：原文「**Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts.**」而在 §2 又说「We follow Qwen2.5-MoE and implement **fine-grained expert segmentation** (Dai et al. 2024)」`[官方报告]`。config 里两个 MoE 都没有 `shared_expert_intermediate_size` 键，佐证这一点 `[官方配置]`。
- 路由归一化：`norm_topk_prob` = **true** → vLLM 实现把它直接映射为 router 的 `renormalize=config.norm_topk_prob`（`vllm/model_executor/models/qwen3_moe.py` L206）`[官方配置]` `[vLLM 源码]`。config 无 `scoring_func` 键 → (推断) 用默认的 softmax top-k 打分（DeepSeek 是显式 `sigmoid`）。
- 负载均衡：**global-batch load balancing loss**（Qiu et al. 2025），`router_aux_loss_coef` = **0.001** `[官方报告]` `[官方配置]`。
- 专家粒度（`moe_intermediate_size`）：30B-A3B = **768**（dense 分支 intermediate 6144 的 1/8）；235B-A22B = **1536**（dense 分支 12288 的 1/8）`[官方配置]`。
- MoE 效率官方口径：Qwen3 MoE base「can achieve similar performance to Qwen3 dense base models with only **1/5 activated parameters**」；「outperform the Qwen2.5 MoE base models with **less than 1/2 activated parameters**」`[官方报告]`。
- Omni 的例外：Qwen3-Omni 的 **Talker** 虽然也是 128 专家，但每 token 激活 **6** 个，且 **`shared_expert_intermediate_size` = 768**（即 Talker 有共享专家，Thinker 没有）`[官方配置]`。

### 4.2 与 DeepSeekMoE / DeepSeek-V3 的差异（逐条）

| 维度 | Qwen3-MoE（30B-A3B / 235B-A22B） | DeepSeek-V3 | 差异要点 |
|---|---|---|---|
| 路由专家数 | **128** | **256**（`n_routed_experts`） | Qwen3 专家更少、粒度更细的「细粒度切分」用更少专家实现 |
| 每 token 激活 | **8** | **8** | 相同 |
| 共享专家 | **无**（`shared_expert_intermediate_size` 不设） | **1 个共享专家**（`n_shared_experts = 1`） | 最大的设计分歧：Qwen3 显式去掉共享专家 |
| 专家中间维 | 768（30B-A3B）/ 1536（235B-A22B） | `moe_intermediate_size = 2048` | — |
| 打分函数 | config 无 `scoring_func` → (推断) softmax | **`scoring_func: "sigmoid"`** | DeepSeek 用 sigmoid 打分 |
| Top-k 选择法 | config 无 `topk_method` → (推断) 普通 top-k | **`topk_method: "noaux_tc"`**、`n_group = 8`、`topk_group = 4`、`routed_scaling_factor = 2.5` | DeepSeek 是分组 + 无辅助损失偏置的 route |
| top-k 概率归一化 | `norm_topk_prob: true` | `norm_topk_prob: true` | 相同 |
| 负载均衡手段 | **global-batch load balancing loss** + `router_aux_loss_coef 0.001` | **aux-loss-free**（偏置项）+ 组路由 | 均衡机制不同：显式 aux loss vs 偏置补偿 |
| 稠密层处理 | `decoder_sparse_step = 1`、`mlp_only_layers = []` → **每层都是 MoE** | `first_k_dense_replace = 3`、`moe_layer_freq = 1` → **前 3 层是 dense FFN** | Qwen3-MoE 无 dense 前缀层 |
| 注意力 | **GQA**（30B: 32Q/4KV；235B: 64Q/4KV），head_dim 128 | **MLA**（`kv_lora_rank 512`、`q_lora_rank 1536`、`qk_nope_head_dim 128`、`qk_rope_head_dim 64`、`v_head_dim 128`，128 Q heads / 128 KV heads） | Qwen3 用 GQA，DeepSeek 用低秩 KV 压缩的 MLA |
| 模型规模 | 48 层 / hidden 2048（30B-A3B）；94 层 / hidden 4096（235B-A22B） | 61 层 / hidden 7168；总 671B / 激活 37B | Qwen3-235B-A22B 总参数约 DeepSeek-V3 的 **1/3**、激活参数约 **2/3** `[官方报告]` |
| 词表 | 151936 | 129280 | — |
| 长上下文 RoPE | `rope_theta` 1e6，YaRN factor 4.0 | `rope_theta` 10000 + YaRN（factor 40，`original_max_position_embeddings` 4096，`beta_fast 32`/`beta_slow 1`，`mscale 1.0`） | Qwen3 靠高 base + YaRN×4，DeepSeek 靠 YaRN factor 40 |

来源: https://ar5iv.labs.arxiv.org/html/2505.09388 | https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json | https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json | https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json

---

## 5. Omni 多模态架构（Qwen2.5-Omni / Qwen3-Omni）★

### 5.1 组件命名

| 代次 | 视觉编码器 | 音频编码器 | 推理塔 | 生成塔 | 声码/波形 |
|---|---|---|---|---|---|
| Qwen2.5-Omni | Qwen2.5-VL 的 ViT（约 675M） | Qwen2-Audio 的音频编码器（**Whisper-large-v3** 初始化） | **Thinker**（生成文本） | **Talker**（dual-track AR，直接吃 Thinker 的 hidden 表征） | **qwen-tts-tokenizer** → Flow-Matching **DiT** → 改造版 **BigVGAN** |
| Qwen3-Omni | Qwen3-VL 的 ViT，由 **SigLIP2-So400m** 初始化（约 543M / config 540M） | **AuT（Audio Transformer）**，20M 小时监督音频从零训练，约 0.6B / 650M | **Thinker**（MoE） | **Talker**（MoE，多码本 AR） | **MTP 模块**（补全残差码本）+ **Code2Wav**（轻量因果 ConvNet） |

- Qwen3-Omni 相对 Qwen2.5-Omni 的五项升级（原文）：① Thinker 与 Talker 都改为 MoE；② Whisper 音频编码器换成自研 **AuT**；③ 语音生成改用**多码本（multi-codebook）**表示；④ Talker 从单轨改**多轨**，用 **MTP** 自回归预测多个 codebook 层，波形阶段 **Code2Wav** 用轻量卷积网络取代 block-wise DiT；⑤ 输入/输出音频码率都降到 **12.5 Hz**，输出 codec 支持**单帧即时合成** `[官方报告]`。
- 音频前处理（两代一致）：重采样到 **16 kHz**，转成 **128 通道 mel-spectrogram**，**窗长 25 ms、hop 10 ms** `[官方报告]`。
  - Qwen2.5-Omni：每帧音频表征 ≈ **40 ms** 原始音频（Whisper-large-v3 初始化的编码器）。
  - Qwen3-Omni：AuT 先用 Conv2D 把 filter bank 特征**下采样 8 倍**，token rate 降到 **12.5 Hz**，每帧表征 ≈ **80 ms** 音频。
- AuT 训练数据配比：**80%** 中英伪标 ASR、**10%** 其他语种 ASR、**10%** 音频理解；token rate **12.5 Hz**；注意力窗口动态覆盖 **1–8 秒** `[官方报告]`。
- 发布形态：Qwen3-Omni 公开 **Qwen3-Omni-30B-A3B / -Thinking / -Captioner** 三个权重（Apache 2.0）；Captioner 由 30B-A3B 在音频描述数据上微调而来。Qwen2.5-Omni 公开 3B 与 7B 两档 `[官方报告]` `[官方 README]`。
- 语言覆盖（Qwen3-Omni）：文本 **119** 种；语音理解 **19** 种（ar, de, en, es, fr, id, it, ja, ko, ms, nl, pt, ru, th, tr, ur, vi, yue, zh）；语音生成 **10** 种（de, en, es, fr, it, ja, ko, pt, ru, zh）；单实例可处理**最长 40 分钟**音频做 ASR/口语理解 `[官方报告]`。

### 5.2 TMRoPE（Time-aligned Multimodal RoPE）到底做什么

- 名字：Qwen2.5-Omni 报告写作 **TMRoPE (Time-aligned Multimodal RoPE)**；Qwen3-Omni 报告写作 **TM-RoPE**，并明确「extends the Multimodal Rotary Position Embedding (**M-RoPE**) by incorporating **absolute temporal information**」`[官方报告]`。
- 结构：把旋转位置编码**拆成三个分量：temporal / height / width**（Qwen2.5-Omni 原文「deconstructing the original rotary embedding into three components: temporal, height, and width」）`[官方报告]`。
- 各模态怎么编：
  - **文本**：三个分量用**相同的 position ID** → TMRoPE 退化成 **1D-RoPE**。
  - **音频**：也用相同 ID，但额外引入**绝对时间**位置编码；Qwen2.5-Omni「**one temporal ID corresponds to 40 ms**」，Qwen3-Omni「each temporal ID corresponds to a duration of **80 ms**」。
  - **图像**：所有视觉 token 的 temporal ID **恒定**，height/width ID 按 token 在图像中的行列分配；每张图按**两帧相同帧**处理。
  - **带音轨的视频**：音频每 40 ms（Qwen3 为 80 ms）一个 temporal ID；视频按帧递增 temporal ID，且因为帧率不固定，**按每帧真实时间动态调整**，保证一个 temporal ID 恒等于 40 ms / 80 ms。
  - **多模态拼接**：每个模态的编号从**前一个模态最大 position ID + 1** 开始，避免位置冲突。
- 为什么需要它：为了**同步视频与音频的时间戳**（「to synchronize the timestamps of video inputs with audio」），并让模型能同时理解多模态的**绝对时间对齐**信息 `[官方报告]`。
- 交织（interleaving）方式：Qwen2.5-Omni 把带音轨视频**按真实时间每 2 秒切块**，每块内**视觉在前、音频在后**排列；Qwen3-Omni **抛弃固定 2 秒分块**，「directly aligns these representations using their **temporal IDs**, which are explicitly anchored to absolute time」，从而支持**任意时长**的流式输入 `[官方报告]`。
- 角度分配（Qwen3-Omni 的关键改动）：M-RoPE 原本用**最前面的 16 个 rotary angles** 建模时间（高频、震荡强，利于局部但不利于长序列外推）；Qwen3-Omni 改成 **temporal / height / width = 24 / 20 / 20 且交织分配** `[官方报告]`。
- 与 config 的交叉验证：Qwen2.5-Omni `mrope_section = [16, 24, 24]`（temporal 16 = 报告所说的「first 16 rotary angles」），Qwen3-Omni `mrope_section = [24, 20, 20]` + `interleaved: true`，与报告文字完全对应；两者之和都是 **64**，而 head_dim 是 **128**，即 (推断) 64 = head_dim/2 的旋转维度被三个分量瓜分 `[官方配置]` `[官方报告]`。
- 时间分辨率换算：Qwen2.5-Omni `position_id_per_seconds = 25` ↔ 报告「1 temporal ID = 40 ms」；Qwen3-Omni `position_id_per_seconds = 13` ↔ 报告「1 temporal ID = 80 ms」（(推断) 13 ≈ 1000/80 = 12.5 向上取整）`[官方配置]` `[官方报告]`。

### 5.3 流式（streaming）与低延迟

- **输入侧 chunked prefill**：两类编码器都改成 **block-wise / chunked** 处理以支持分块预填充。Qwen2.5-Omni：音频编码器「changed from full attention over the entire audio to performing attention in **blocks of 2 seconds** each」；视觉编码器用 flash attention + 一个把相邻 **2×2 token 合并成 1 个 token** 的 MLP，`patch size = 14`，从而把不同分辨率图像打包成一个序列。Qwen3-Omni 保留 chunked prefill，并且 Thinker 与 Talker **异步预填充**（Thinker 填完当前 chunk 立刻用其高层表征去预填 Talker 的当前 chunk）`[官方报告]`。
- **输出侧流式 codec**：
  - Qwen2.5-Omni：Flow-Matching **DiT** 用滑动窗口块注意力，**感受野 4 个 block = lookback 2 + lookahead 1**；mel-spectrogram 分块生成；BigVGAN 也按块解码以支持流式波形。
  - Qwen3-Omni：**left context only** 的多码本生成 —— Talker 出一个 token，MTP 立刻补齐当前帧的其余码本，Code2Wav 立即出波形；「Qwen3-Omni can output the waveform **immediately after the Talker generates each token**, significantly reducing first-packet latency」（Qwen2.5-Omni 必须等攒够 block-context）。
- **首包延迟实测/理论数字**：见 §5.5 表格。Qwen3-Omni 的另一处口径：「In **cold-start** settings (no prior context), Qwen3-Omni achieves a **theoretical end-to-end first-packet latency of 234 ms**」`[官方报告]`。
- 音频码率：Qwen3-Omni 输入/输出都是 **12.5 Hz** → **1 个 token = 80 ms 音频**；RTF 定义 = (Thinker 出 1 token + Talker 出 1 token + MTP 每 token 耗时 + Codec 每码耗时) ÷ 80 ms `[官方报告]`。

### 5.4 Talker 与 Thinker 的分工，为什么要分开

- Qwen2.5-Omni（**双轨 / dual-track**）：Talker 是 dual-track 自回归 Transformer Decoder（原文注明 motivated by **Mini-Omni**），**同时**接收 ①Thinker 的高维表征 ②Thinker 采样出的**文本 token 的 embedding**。官方给的两条理由：高维表征隐含**语气/态度**（流式合成必须在整句文本生成完之前就知道）；而 Thinker 的表征表达的是**语义相似**而非**音素相似**（读音完全不同的词表征可能很接近），所以必须再喂**离散采样 token** 来消除歧义。二者共享 Thinker 的全部历史上下文，端到端联合训练 `[官方报告]`。
- Qwen3-Omni（**解耦**）：Talker **不再消费 Thinker 的高层文本表征**，只条件于**音频与视觉多模态特征**。理由：① 对文本内容而言离散 token 与 embedding 信息等价；② 多模态条件对**音画协同**的语音生成是必要的（如语音翻译中保持韵律/音色）；③ 解耦后外部模块（**RAG、function calling、安全过滤**）可以介入 Thinker 的文本输出，再按需把文本喂给 Talker 做流式合成；④ Thinker 与 Talker 因此可以使用**各自独立的 system prompt**，分别控制回答风格与音频风格 `[官方报告]`。
- 为什么不能用单轨：Qwen2.5-Omni 原文的动机是「avoiding **interference between the two modalities**」（文本与语音两种输出的训练互相干扰），并且「Talker having to anticipate the content's tone and attitude before the entire text is fully generated」。Qwen2.5-Omni 还要求语音生成**不需要**与文本做 word-level / timestamp-level 对齐，从而简化训练数据与推理 `[官方报告]`。
- Training recipe 差异：Qwen2.5-Omni Talker 三阶段（上下文续写 → DPO 稳定 → 多说话人 SFT）；Qwen3-Omni Talker 四阶段（亿级多模态语音数据建立映射 → CPT + 长上下文 → DPO → 说话人微调）`[官方报告]`。

### 5.5 延迟 / RTF 数字（官方 + 第三方）

**Qwen3-Omni-30B-A3B（Table 1 + Table 2，音频/视频两值，单位 ms）**`[官方报告]`

| 指标 | 1 并发 | 4 并发 | 6 并发 |
|---|---|---|---|
| Thinker-Talker Tail Packet Preprocessing | 72 / 160 | 94 / 180 | 100 / 200 |
| Thinker Time-to-First-Token (TTPT) | 88 / 160 | 468 / 866 | 673 / 1330 |
| Talker Time-to-First-Token (TTPT) | 57 / 210 | 145 / 450 | 376 / 734 |
| MTP 每 token 耗时 | 14 | 16 | 18 |
| Codec Decoder 每 code 耗时 | 3 | 5 | 5 |
| **端到端首包延迟（Audio / Video）** | **234 / 547** | 728 / 1517 | 1172 / 2284 |
| Thinker 生成速率 | 75 tokens/s | 63 tokens/s | 53 tokens/s |
| Talker 生成速率 | 140 tokens/s | 125 tokens/s | 110 tokens/s |
| **Generation RTF** | **0.47** | 0.56 | 0.66 |

- 并发每翻一档 RTF 仍 < 1；报告明确这些数字是「**theoretical** first-packet latency」，跑在 **vLLM** 上，并对 MTP 模块与 codec decoder 施加了 `torch.compile` + CUDA Graph 优化 `[官方报告]`。
- 组件参数量口径：Audio Encoder **650M**、Vision Encoder **540M**、Thinker **30B-A3B**、Talker **3B-A0.3B**、MTP **80M**、Code2wav **200M** `[官方报告]`。
- **Qwen2.5-Omni 官方从来没有公布过实测首包延迟 / RTF / TTFP 数字**：报告（arXiv 2503.20215 只有 v1）、GitHub README、官方博客、HF 模型卡、qwen.readthedocs.io 全部查过，`RTF` 出现 **0** 次、没有任何 ms 级延迟表。报告 §2.4 只给**四类延迟来源**的定性描述：①多模态输入处理的延迟 ②收到首个文本输入到输出首个语音 token 的延迟 ③首段语音转成音频的延迟 ④架构本身的延迟（与模型大小、FLOPs 相关）；另有「we introduce a **sliding-window DiT** that restricts the receptive field, **aiming to reduce the initial package delay**」`[官方报告]` `[官方 README]`。
- **第三方 RTF（VocalBench, arXiv 2505.15727v3）**：Qwen2.5-Omni **RTF(EN) = 1.7243 / RTF(ZH) = 1.7970**，**First Chunk Latency（FCL）报为「-」即未测出**。口径：`RTF = 1/|K| Σ t_s_k / t_g_k`（t_s = 响应时长，t_g = 生成时长）；FCL = 从用户 query 结束到首个响应 chunk 开始；单卡 **NVIDIA L20**；英文单轮对话集，5 条 warmup + 200 条计时；对没有官方流式实现的模型统一按 **~0.5 秒**切语音块。该表里没有 Qwen2.5-Omni-3B 的行，也没有区分 7B/3B`[第三方]`。
- **可引用的对照**：VocalBench 结论是「the majority of interaction frameworks exhibit a first-packet latency **exceeding 0.5 seconds** at the model level」，而 Qwen2.5-Omni 的 RTF > 1（即生成比播放慢），这正是 Qwen3-Omni 把 RTF 压到 0.47 要解决的问题 `[第三方]` `[官方报告]`。
- Qwen2.5-Omni 的语音生成质量数字（seed-tts-eval）：test-zh **1.42%** / test-en **2.33%** / test-hard **6.54%** WER `[官方报告]`；README 的对照表里 7B_RL 是 **1.42 / 2.32 / 6.54**（test-en 差 0.01，属两处口径差异）`[官方 README]`。说话人相似度 7B_RL = **0.754 / 0.641 / 0.752** `[官方 README]`。
- Qwen2.5-Omni 的其他可引用分：VoiceBench 平均 **74.12**、OmniBench 平均 **56.13%**、MMAU 平均 **65.60** `[官方 README]`。

来源: https://ar5iv.labs.arxiv.org/html/2503.20215 | https://arxiv.org/html/2503.20215v1 | https://ar5iv.labs.arxiv.org/html/2509.17765 | https://arxiv.org/html/2505.15727v3 | https://huggingface.co/Qwen/Qwen2.5-Omni-7B/raw/main/README.md | https://raw.githubusercontent.com/QwenLM/Qwen2.5-Omni/main/README.md | https://huggingface.co/Qwen/Qwen2.5-Omni-7B/raw/main/config.json

---

## 6. 部署侧的数字

### 6.1 vLLM / vLLM-Omni 是怎么接的

- vLLM 注册名与实现文件 `[vLLM 源码]`（`vllm/model_executor/models/registry.py`）：
  - `Qwen3ForCausalLM` → `qwen3.py`；`Qwen3MoeForCausalLM` → `qwen3_moe.py`（第 202–203 行）。
  - `Qwen2AudioForConditionalGeneration` → `qwen2_audio.py`；`Qwen2_5OmniModel` / `Qwen2_5OmniForConditionalGeneration` → `Qwen2_5OmniThinkerForConditionalGeneration`（`qwen2_5_omni_thinker.py`）；`Qwen3OmniMoeForConditionalGeneration` → `Qwen3OmniMoeThinkerForConditionalGeneration`（`qwen3_omni_moe_thinker.py`）。
  - 即：**主仓 vLLM 里 Omni 注册的是 Thinker（多模态理解 + 文本生成）**；Talker/Code2Wav 这套语音生成流水线在 **vLLM-Omni**。
- `docs/models/supported_models.md` 的支持矩阵 `[vLLM 文档]`：`Qwen3ForCausalLM`（Qwen3，如 `Qwen/Qwen3-8B`）、`Qwen3MoeForCausalLM`（Qwen3MoE，如 `Qwen/Qwen3-30B-A3B`）均为 ✅（text generation + LoRA）；`Qwen2AudioForConditionalGeneration` = **T + A**（Qwen2-Audio-7B-Instruct）；`Qwen2_5OmniThinkerForConditionalGeneration` = **T + I + V + A**（`Qwen/Qwen2.5-Omni-3B`、`Qwen/Qwen2.5-Omni-7B`）；`Qwen3OmniMoeThinkerForConditionalGeneration` = **T + I + V + A**（`Qwen/Qwen3-Omni-30B-A3B-Instruct`、`-Thinking`）。
- vLLM 里 Qwen 家族的两个实现细节：MoE 的 `renormalize = config.norm_topk_prob`（Qwen3 全系 true）；共享专家是**可选**分支（`if shared_expert_intermediate_size > 0` 才创建 `shared_expert` 与 `shared_expert_gate`），与 Qwen3-MoE 无共享专家的设计一致 `[vLLM 源码]`。
- Omni 的 M-RoPE 在 vLLM 里由 `get_mrope_input_positions` 计算，video+audio 走 `_compute_interleaved_positions`，按 `seconds_per_chunk × tokens_per_second` 切块、每块「视觉在前音频在后」（源码注释的示意图 `|vision chunk 1|audio chunk 1|vision chunk 2|audio chunk 2|...`）`[vLLM 源码]`。

### 6.2 官方实测/理论显存与延迟

**Qwen3-Omni 服务化实测（vLLM-Omni 官方博客，`Qwen3-Omni-30B-A3B-Instruct`，Seed-TTS `en`，并发 64 时 640 条 prompt，5 次预热，3 张可见 GPU：Thinker/Talker/Code2Wav 分别放 GPU 0/1/2）**`[第三方/vLLM 团队]`

| 优化步骤 | Talker/Code2Wav 副本 | Req/s | 平均音频 TTFP | 平均音频 RTF |
|---|---|---|---|---|
| Batch（基线） | 1 / 1 | 2.2 | **5884 ms** | **1.15** |
| + CUDA Graph（三阶段都上） | 1 / 1 | 8.6（+299%，约 4×） | 2790 ms（−53%） | 0.59（−49%） |
| + Async chunk（异步分块交接） | 1 / 1 | 9.3（+8%） | **655 ms**（−77%，单项降幅最大） | 0.63 |
| + Async output（非阻塞 payload 构建） | 1 / 1 | 11.3（+22%） | **631 ms**（−4%） | **0.47**（−25%） |
| + Stage replicas（2× Talker + 2× Code2Wav） | 2 / 2 | **11.7**（+4%） | 632 ms | 0.47 |

- 同文的长上下文单请求热路径清理效果：E2EL **21.28 s → 7.37 s**，音频 TTFP **3197 ms → 1796 ms**，音频 RTF **0.71 → 0.28**；异步输出把解码步间隙从约 **2.8 ms** 压到约 **41 µs** `[第三方/vLLM 团队]`。
- 硬件口径注意：该博客只写「3 张可见 GPU（0/1/2）」与并发 `1/16/32/64`，**未写 GPU 型号**（未找到）；数据里 c=64 是 640 条 prompt 的那一档 `[第三方/vLLM 团队]`。
- vLLM-Omni 的分阶段拓扑：`Thinker -> Talker -> Code2Wav`，请求走 `/v1/chat/completions`，用 `modalities` 声明输出（`["text"]` 或 `["text","audio"]`）；启动命令 `vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091` `[第三方/vLLM 团队]`。

**官方 HF 理论最小显存（transformers + BF16 + flash_attention_2，注：实际通常还要 ×1.2）**`[官方 README]`

| 模型 | 精度 | 15s 视频 | 30s 视频 | 60s 视频 | 120s 视频 |
|---|---|---|---|---|---|
| Qwen2.5-Omni-3B | BF16 | 18.38 GB | 22.43 GB | 28.22 GB | 未找到 |
| Qwen2.5-Omni-7B | BF16 | 31.11 GB | 41.85 GB | 60.19 GB | 未找到 |
| Qwen3-Omni-30B-A3B-Instruct | BF16 | **78.85 GB** | 88.52 GB | 107.74 GB | 144.81 GB |
| Qwen3-Omni-30B-A3B-Thinking | BF16 | 68.74 GB | 77.79 GB | 95.76 GB | 131.65 GB |

- 关掉 Talker 省显存：Qwen2.5-Omni `model.disable_talker()` 约省 **~2 GB**；Qwen3-Omni 同接口约省 **10 GB** `[官方 README]`。
- 量化后的显存（Qwen2.5-Omni-7B，官方 README）：**GPTQ-Int4** 15s/30s/60s = **11.64 / 17.43 / 29.51 GB**；**AWQ** = **11.77 / 17.84 / 30.31 GB**；官方称显存下降 **50%+**，代价是推理略慢（CPU offload + 量化）。配套三项优化：权重按模块 on-demand 加载并在用完后 offload 到 CPU；**code2wav 改成流式推理**避免预分配过多显存；ODE solver 从二阶 **RK4** 改成**一阶 Euler** 以降算力 `[官方 README]`。
- **端侧（MNN，官方 README 基准表）**：7B 内存峰值 **5.8G**（Snapdragon 8 Gen 1 与 8 Elite 相同），3B 为 **3.6G**；

| 平台 / 模型 | SD 8 Gen 1 + 7B | SD 8 Elite + 7B | SD 8 Gen 1 + 3B | SD 8 Elite + 3B |
|---|---|---|---|---|
| Thinker Prefill | 25.58 tok/s | 46.32 tok/s | 54.31 tok/s | 55.16 tok/s |
| Thinker Decode | **8.35 tok/s** | 11.52 tok/s | 15.84 tok/s | 23.31 tok/s |
| Talker Prefill | 17.21 tok/s | 97.77 tok/s | 34.58 tok/s | 217.82 tok/s |
| Talker Decode | 18.75 tok/s | 38.65 tok/s | 51.90 tok/s | 62.34 tok/s |
| Code2Wav | 20.83 tok/s | 27.36 tok/s | 28.45 tok/s | 27.36 tok/s |

### 6.3 服务端的两个「官方限制」事实

- **vLLM 主仓只服务 Thinker**：官方 Qwen2.5-Omni README 原文「vLLM serve for Qwen2.5-Omni **only supports thinker now**, meaning **only text output** is supported」；要出音频必须走 Qwen 自己 fork 的 vLLM（`fyabc/vllm` 分支 `qwen2_omni_public`），并且能把三个阶段拆到不同 GPU：`--thinker-devices [0,1] --talker-devices [2] --code2wav-devices [3]`，或单卡 `--thinker-only`。这与主仓 registry 里 Omni 映射到 `...ThinkerForConditionalGeneration` 一致 `[官方 README]` `[vLLM 源码]`。
- **音频输出必须用官方 system prompt**：不设「You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.」这句时，音频输出可能不工作；且 Qwen2.5-Omni 在音频输出模式下**不支持自定义 prompt**（官方建议用「user 设定 + assistant 确认」的模板绕过）；7B 支持 **Chelsie（女）/ Ethan（男）** 两种音色，默认 Chelsie `[官方 README]`。

### 6.4 显存/带宽瓶颈在哪（基于上面数字的定位）

- **视觉编码器不是主瓶颈，长视频的 KV/激活才是**：Qwen3-Omni 显存从 15s 的 78.85 GB 涨到 120s 的 144.81 GB（+65.9 GB），是官方表格里增长最快的一维；而视觉塔本身只有 540M 参数 `[官方 README]` `[官方报告]`。
- **Talker + Code2Wav 是吞吐瓶颈，不是显存瓶颈**：vLLM-Omni 实测在并发 64 时「Talker 和 Code2Wav 饱和并排队，在 Thinker 仍有余量时成为尾部瓶颈」，并且**只**复制这两个阶段（Thinker 保持单副本）就把吞吐从 11.3 推到 11.7 req/s；原因是「每个请求 Thinker 只出一次文本，而 Talker/Code2Wav 要跑数百个短解码步和声码器前向」`[第三方/vLLM 团队]`。
- **首包延迟的瓶颈是流水线屏障，不是算力**：CUDA Graph 把 RTF 从 1.15 压到 0.59，但 TTFP 仍高达 2790 ms；换成异步分块交接后 TTFP 直接掉到 655 ms（−77%）——说明「Code2Wav 等完整 Talker payload」这一屏障才是首包延迟主因 `[第三方/vLLM 团队]`。
- **MoE 的收益点在 KV cache IO**：Qwen3-Omni 报告原文「the MoE architecture significantly decreases **IO consumption arising from KV cache** during processing of long sequences, thereby increasing tokens per second (TPS) during generation and enhancing concurrency」；Talker/Thinker 都是 MoE 就是为了高并发下 prefill 与 TTPT 不被拖垮 `[官方报告]`。
- **单请求微算子开销**：Talker 每步只是一个很短的 code predictor 前向、Code2Wav 每块只是一个小 vocoder 前向，并发 64 时单请求小算子让 SM 在两次 launch 之间空转 → 必须做阶段级 batching + CUDA Graph（MTP 每 token 14→18 ms、codec 每 code 3→5 ms 的官方数字就是这类固定开销的量级）`[第三方/vLLM 团队]` `[官方报告]`。

来源: https://huggingface.co/Qwen/Qwen2.5-Omni-7B/raw/main/README.md | https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/README.md | https://raw.githubusercontent.com/QwenLM/Qwen2.5-Omni/main/README.md | https://arxiv.org/html/2505.15727v3 | https://blog.vllm.com.cn/2026/07/01/qwen3-omni-optimization.html | https://ar5iv.labs.arxiv.org/html/2509.17765 | 本地 vLLM 源码 `vllm/model_executor/models/registry.py`、`qwen3.py`、`qwen3_moe.py`、`qwen2_5_omni_thinker.py`、`docs/models/supported_models.md`

---

## 附：可出题的「数字陷阱」清单

1. Qwen2.5 报告表 1 说 1.5B 是 32K，但 config 写 `max_position_embeddings = 131072`（3B/0.5B 才是 32768）。
2. Qwen3 dense 的 `head_dim` 是显式 **128**，**不等于** hidden/heads（4B = 80、32B = 80）；Qwen2.5 则是 hidden/heads。
3. Qwen3 config 的 `max_position_embeddings = 40960` ≠ 可用上下文 32768（官方说是 32768 输出 + 8192 prompt 的预留）。
4. 词表三处口径：报告 151,643（regular tokens，Qwen2.5）/ 151,669（Qwen3）、config 151936（小模型与 MoE）/ 152064（≥7B 与 Omni Thinker）。
5. QK-Norm 是 **Qwen3 才引入**，Qwen2.5 用的是 **QKV bias**；Qwen3 是「remove QKV-bias + introduce QK-Norm」。
6. Qwen3-MoE **没有共享专家**（Qwen2.5-MoE 有），但 **Qwen3-Omni 的 Talker 有共享专家**（`shared_expert_intermediate_size = 768`）。
7. TMRoPE 的时间粒度：Qwen2.5-Omni **40 ms / ID（25 IDs per second）**，Qwen3-Omni **80 ms / ID（13 IDs per second，音频 12.5 Hz）**。
8. Qwen2.5-Omni 用固定 **2 秒**块交织音视频，Qwen3-Omni 改为按**绝对 temporal ID** 对齐，因此支持任意时长流式输入。
9. 首包延迟 234 ms 是 **Qwen3-Omni 单并发冷启动的理论值**（视频口径是 547 ms），不是 Qwen2.5-Omni 的数字。
10. RTF 的分母是 **80 ms**（12.5 Hz 一个 token 对应 80 ms 音频），官方 RTF 1/4/6 并发 = **0.47 / 0.56 / 0.66**。
11. **Qwen2.5-Omni 没有任何官方延迟数字**；唯一可引用的 Qwen2.5-Omni RTF 是第三方的 **1.7243 (EN) / 1.7970 (ZH)**（NVIDIA L20），且 FCL 未测出。别把 234 ms 安到 Qwen2.5-Omni 头上。
12. Qwen2.5-Omni 在 vLLM 主仓**只能出文本**（只支持 thinker）；端侧 MNN 的 7B 内存峰值只要 **5.8G**，但 Thinker Decode 只有 **8.35 tok/s**（SD 8 Gen 1）。
