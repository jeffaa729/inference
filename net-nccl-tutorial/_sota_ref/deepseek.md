# DeepSeek 架构事实速查（V2 / V3 / V3.1 / V3.2 / R1 / V4）

> 标注约定：**【官方】**= 论文 / 官方 repo / 官方 API 文档；**【第三方】**= vLLM、SGLang、NVIDIA、社区配置；**（推断）**= 本文件基于官方数字的算术或常识推理，非源中原文；**未找到**= 本轮抓取中确实未见到该数字。
> 所有数字均为「源里真实出现」的原文数值，未做四舍五入美化（KB 沿用源中的 1000 进制）。

---

## 1. 模型规格表（每个版本一行）

| 版本 | 总参数 | 激活参数 | 层数 | hidden | heads | 专家数 / 每 token 激活 | 共享专家 | 上下文 | 词表 |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2【官方】 | 236B | 21B | 未找到 | 未找到 | 未找到 | 未找到 | 未找到 | 128K | 未找到 |
| DeepSeek-V2-Lite【官方】 | 15.7B | 2.4B | 未找到 | 未找到 | 未找到 | 未找到 | 未找到 | 未找到 | 未找到 |
| DeepSeek-V3 / V3-Base【官方】 | 671B（HF 上合计 685B = 671B 主模型 + 14B MTP） | 37B（README_WEIGHTS：36.7B，含 0.9B Embedding + 0.9B Head） | 61 | 7168 | 128（KV heads 128） | 256 routed / 8 activated | 1 | 128K | 129280 |
| DeepSeek-V3-0324【官方】 | 架构同 V3（仅 post-training 更新） | — | — | — | — | — | — | — | — |
| DeepSeek-R1 / R1-0528【官方】 | 671B | 37B | 61 | 7168 | 128 | 256 / 8 | 1 | 128K（config `max_position_embeddings`=163840） | 129280 |
| DeepSeek-V3.1 / V3.1-Terminus【官方】 | 671B（架构与 V3 相同；V3.1-Base = V3 + 840B tokens 长上下文继续预训练） | 37B | 61 | 7168 | 128 | 256 / 8 | 1 | 128K | 129280 |
| DeepSeek-V3.2-Exp / V3.2 / V3.2-Speciale【官方】 | 671B | 37B | 61 | 7168 | 128 | 256 / 8 | 1 | 128K | 129280 |
| DeepSeek-V4-Pro【官方】 | 1.6T | 49B | 未找到（权重 gated） | 未找到 | 未找到 | 未找到 | 1（HF 文档：single shared expert） | 1M | 未找到 |
| DeepSeek-V4-Flash【官方】 | 284B | 13B | 未找到 | 未找到 | 未找到 | 未找到 | 1 | 1M | 未找到 |
| DeepSeek-V4.1-Flash【官方】 | 未找到（2026/09/10 发布，取代 V4-Flash 承接 `deepseek-flash`） | — | — | — | — | — | — | — | — |

**V3 系 config 精确值（`inference/configs/config_671B.json`，R1 的 HF `config.json` 完全一致）【官方】**

| 字段 | 值 | 字段 | 值 |
|---|---|---|---|
| `vocab_size` | 129280 | `dim` | 7168 |
| `inter_dim` | 18432 | `moe_inter_dim` | 2048 |
| `n_layers` | 61 | `n_dense_layers` / `first_k_dense_replace` | 3 |
| `n_heads` | 128 | `n_routed_experts` | 256 |
| `n_shared_experts` | 1 | `n_activated_experts` / `num_experts_per_tok` | 8 |
| `n_expert_groups` / `n_group` | 8 | `n_limited_groups` / `topk_group` | 4 |
| `route_scale` / `routed_scaling_factor` | 2.5 | `score_func` / `scoring_func` | `sigmoid` |
| `topk_method` | `noaux_tc` | `norm_topk_prob` | true |
| `q_lora_rank` | 1536 | `kv_lora_rank` | 512 |
| `qk_nope_head_dim` | 128 | `qk_rope_head_dim` | 64 |
| `v_head_dim` | 128 | `dtype` | `fp8` |
| `num_nextn_predict_layers` | 1 | `rope_theta` | 10000 |
| `rope_scaling` | `yarn`, factor 40, `original_max_position_embeddings` 4096, beta_fast 32, beta_slow 1, mscale 1.0 | `rms_norm_eps` | 1e-06 |

**V3 / V3.1 / V3.2 训练与成本【官方】**

| 项 | 数值 |
|---|---|
| 预训练 tokens | 14.8T |
| 上下文扩展 | 两阶段：32K → 128K |
| 训练算力（H800 GPU hours） | 预训练 2664K / 上下文扩展 119K / 后训练 5K / 合计 2788K |
| 折合美元（$2/GPU·hour） | $5.328M + $0.238M + $0.01M = $5.576M |
| 训练集群 | 2048 × H800；每 1T tokens 需 180K GPU hours（≈ 3.7 天） |
| V3.1-Base 增量 | 840B tokens 长上下文继续预训练 |
| V3.2 DSA 增量 | dense warm-up 1000 step × 16 seq × 128K = 2.1B tokens（LR 1e-3）；sparse 训练 15000 step × 480 seq × 128K = 943.7B tokens（LR 7.3e-6，top-k 2048） |

来源: https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/configs/config_671B.json ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/README_WEIGHTS.md ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/README.md ; https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json ; https://arxiv.org/html/2412.19437v2 ; https://arxiv.org/html/2512.02556v1 ; https://api-docs.deepseek.com/news/news250821/ ; https://api-docs.deepseek.com/news/news260424 ; https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/deepseek_v4.md

---

## 2. MLA（Multi-head Latent Attention）

### 2.1 精确维度（V3 系）【官方】

| 符号 | 含义 | V3/V3.1/V3.2/R1 取值 |
|---|---|---|
| `kv_lora_rank` = d_c | KV 压缩维度（joint KV compression） | **512** |
| `q_lora_rank` = d_c′ | Q 压缩维度（只为省训练激活，不减 KV cache） | **1536** |
| `qk_nope_head_dim` = d_h | 每头非位置维度 | **128** |
| `qk_rope_head_dim` = d_h^R | 每头 decoupled RoPE 维度 | **64** |
| `qk_head_dim` | = 128 + 64 | **192** |
| `v_head_dim` | 每头 value 维度 | **128** |
| n_h | 头数 | **128** |
| `softmax_scale` | `qk_head_dim ** -0.5`，长上下文再乘 mscale²，mscale = 0.1·mscale·ln(rope_factor)+1 | 192^-0.5 · (1.3689)² ≈ 0.1353（推断：按源码公式代入） |

**只有两个向量需要缓存**（报告原文用蓝色框标出）：`c_t^KV ∈ R^{d_c}` 与 `k_t^R`。即每 token 每层 **512 + 64 = 576 个元素**。

### 2.2 「吸收」（absorption / W_absorb）到底是什么

- V2 报告原文：「since W^{UK} can be absorbed into W^Q, and W^{UV} can be absorbed into W^O, we even do not need to compute keys and values out for attention.」【官方】
  - **被合并的两对矩阵**：`W^{UK}` → 并入 `W^Q`；`W^{UV}` → 并入 `W^O`。
- V3 参考实现里的落地方式（`inference/model.py`, `attn_impl = "absorb"`）【官方】：
  - 加载时把 `wkv_b.weight` view 成 `(n_local_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)`；
  - 打分：`q_nope = einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :qk_nope_head_dim])` —— 把 K 的上投影吸收进 Q，直接与 512 维 latent 做内积；
  - 输出：`x = einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])` —— 把 V 的上投影吸收到 attention 之后；
  - RoPE 部分单独走 `pe_cache`（64 维），因为 RoPE 不能吸收。
- 为什么可以不物化 K/V：attention 里的三处矩阵乘可以重结合 —— `(W^UK c)^T q = c^T (W^{UK,T} q)`；V 侧同理 `softmax(...)·(W^{UV} c) = (softmax(...)·c)·W^{UV}`。因此显存里永远只存 576 维 latent，**每层每 token 只读 576 个元素**。
- **为什么必须有 decoupled RoPE**（V2 原文）：RoPE 对 key 是位置敏感的；若把 RoPE 施加在 `k_t^C` 上，`W^{UK}` 就会和一个位置相关的 RoPE 矩阵耦合，当前生成 token 的 RoPE 矩阵会夹在 `W^Q` 与 `W^{UK}` 之间，而矩阵乘法不可交换，于是 `W^{UK}` 无法再吸收进 `W^Q`，推理时必须为所有前缀 token 重算 key。【官方】

### 2.3 与 MHA / GQA / MQA 的 KV cache 对比

每 token 每层的字节数公式（L_bytes = 每个元素的字节数）：

| 机制 | 缓存元素数 / token / layer | V3 系代入（n_h=128, d_h=128, d_c=512, d_h^R=64） | BF16 字节 | FP8 字节 |
|---|---|---|---|---|
| MHA | `2 · n_h · d_h` | 2·128·128 = **32768** | 65536 B | 32768 B |
| GQA（n_g 组） | `2 · n_g · d_h` | 取决于 n_g（V3 系未用） | — | — |
| MQA | `2 · d_h` | 256 | 512 B | 256 B |
| **MLA** | `d_c + d_h^R` | **576** | **1152 B** | **576 B** |

- **MLA vs MHA（V3 配置）= 32768 / 576 ≈ 56.9×，即省 98.2%**（推断：基于上表官方维度）。
- 官方实测表（BF16，整模型每 token）【官方，ISCA'25】：DeepSeek-V3（MLA）**70.272 KB**，倍率 1×；Qwen-2.5 72B（GQA）327.680 KB，4.66×；LLaMA-3.1 405B（GQA）516.096 KB，7.28×。
  - 校验：576 × 61 层 × 2 B = 70,272 B ✔（推断：与官方 70.272 KB 完全吻合，说明官方 KB 用 1000 进制）
- **128K 上下文单序列 KV cache**：70.272 KB × 131072 ≈ **9.21 GB（BF16）/ 4.61 GB（FP8）**（推断）。
- 对比口径注意：V2 报告宣称相对 DeepSeek 67B「reduces the KV cache by 93.3%」、吞吐提升至 **5.76×**、训练成本省 **42.5%**；这是与 DeepSeek 67B（GQA）比，**不是**与 V3 的 MHA 等价配置比。【官方】

### 2.4 MHA 模式 vs 非 MHA（MQA）模式

| | MHA 模式 | MQA / absorb 模式 |
|---|---|---|
| 何时用 | **prefill**（`mask is not None`，seqlen > 1） | **decode**（单 token 增量） |
| 做法 | 物化 K/V：`k = [k_nope; k_pe]`、`v`，逐头 attention | 只读 latent `kv_cache`(512) + `pe_cache`(64)，把 `wkv_b` 吸收进 Q / 吸收到输出 |
| 依据 | V3.2-Exp 参考实现的分支注释 `# MHA prefill` / `# MQA decode`【官方】；V3.2 论文 Appendix A 专门讲 MLA 的 MHA 与 MQA 两种模式【官方】；TensorRT-LLM 描述 V3/R1/V3.1 的 MLA「alternates between MHA mode (prefill) and MQA mode (decoding)」【第三方】 | 同左 |
| V3.2 特殊点 | DSA 基于 **MQA 模式** 实例化（每个 latent KV entry 被所有 query head 共享，kernel 级效率）；短序列 prefill 官方专门实现 **masked MHA mode 来模拟 DSA**【官方】 | TRT-LLM 的 DSA 实现 prefill/decode **都用 MQA mode**【第三方】 |

来源: https://arxiv.org/html/2412.19437v2 ; https://arxiv.org/html/2405.04434v5 ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py ; https://arxiv.org/html/2512.02556v1 ; https://arxiv.org/html/2505.09343v1 ; https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/refs/heads/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md

---

## 3. DeepSeekMoE

### 3.1 细粒度专家 + 共享专家的动机（原文表述）【官方】

- V3 §2.1.2：「Compared with traditional MoE architectures like GShard, DeepSeekMoE uses **finer-grained experts** and **isolates some experts as shared ones**.」
- V2 §1：「DeepSeekMoE architecture, which adopts **fine-grained expert segmentation** and **shared expert isolation** for higher potential in expert specialization.」
- 结构式（V3 公式 12）：`h'_t = u_t + Σ_{i=1}^{N_s} FFN^{(s)}_i(u_t) + Σ_{i=1}^{N_r} g_{i,t} FFN^{(r)}_i(u_t)`，N_s=1、N_r=256、K_r=8。

### 3.2 路由：打分、top-k、归一化【官方】

| 步骤 | 公式 / 实现 | V3 值 |
|---|---|---|
| 亲和度打分 | `s_{i,t} = Sigmoid(u_t^T e_i)`（V2 用 softmax，**V3 改成 sigmoid**） | `score_func=sigmoid` |
| 分组受限路由 | 8 组选 4 组，组内再 top-k；组打分实现为 `scores.topk(2).sum(-1)`（有 bias 时） | `n_group=8`, `topk_group=4` |
| top-k | `Topk({s_{j,t}}, K_r)` | K_r = 8 |
| 归一化 | `g_{i,t} = g'_{i,t} / Σ_{j=1}^{N_r} g'_{j,t}`，**只在被选中的专家之间归一化** | `norm_topk_prob=true` |
| 缩放 | `weights *= route_scale` | `route_scale = 2.5` |
| 稠密层 | 前 3 层是普通 MLP（`n_dense_layers=3`），其余 58 层是 MoE | 61 = 3 + 58 |

### 3.3 auxiliary-loss-free 负载均衡

- 路由判定（V3 公式 16）【官方】：
  `g'_{i,t} = s_{i,t}` if `s_{i,t} + b_i ∈ Topk({s_{j,t} + b_j | 1≤j≤N_r}, K_r)` else 0。
- **bias 不参与梯度**：原文「the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score s_{i,t}」；Loss-Free 论文亦强调 `b_i` 「is not added to the g_{i,t} that weights the output」⇒ 不产生 interference gradients。【官方】
- **更新规则（原始论文 Algorithm 1）**【官方】：统计上一 batch 每个专家的 token 数 `c_i` 与均值 `c̄_i`；**violation error** `e_i = c̄_i − c_i`；更新
  **`b_i = b_i + u · sign(e_i)`**
  即过载（c_i 大、e_i<0）则减、欠载则加。用「上一 batch / 历史负载」而非当前序列，是为了不破坏语言建模的因果约束（否则会泄漏 future token 信息）。
- **更新速率 γ 的精确值**【官方，Loss-Free 论文 §4.3】：论文在 1B 模型上扫 u ∈ {0.0001, 0.001, 0.01}：`u=0.0001` 收敛过慢；`u=0.01` 后期 bias 抖动、反而恶化均衡；**最佳 `u = 0.001`**（表中 `b_i = b_i + u·sign(e_i), u=0.001` → PPL 9.50 / MaxVio_global 0.044）。
  - V3 报告中该超参被称为 **bias update speed γ**，正文只给出定义与「decrease/increase by γ」的更新方向；**V3 报告自己写的 γ 数值在本轮抓到的正文里未出现（未找到）**，常用引用值 0.001 来自上述原始论文。
  - 对照：辅助损失系数基线 α=0.001 是 Loss-Free 论文里 auxiliary-loss-controlled 基线的取值；V3 报告只说序列级 α「will be assigned an extremely small value」，**精确数值未找到**。
  - 变体 `b_i = b_i + u·e_i` 略改善均衡但性能不升（PPL 9.53 / MaxVio 0.028），故保留 sign 版本。【官方】
- 复杂度/收益数据（1B/3B 验证）【官方】：Loss-Controlled vs Loss-Free —— 1B：PPL 9.56 vs **9.50**，MaxVio_global 0.72 vs **0.04**；3B：PPL 7.97 vs **7.92**，MaxVio 0.52 vs **0.04**。
- 与 Expert Choice 的对比表：Loss-Free = balanced + no interference gradients + no future token leakage。【官方】

### 3.4 序列级辅助损失（complementary sequence-wise auxiliary loss）

- 为什么需要（V3 原文）：「Although DeepSeek-V3 mainly relies on the auxiliary-loss-free strategy for load balance, **to prevent extreme imbalance within any single sequence**, we also employ a complementary sequence-wise balance loss」；§4.5.3 专门做「Batch-Wise Load Balance VS. Sequence-Wise Load Balance」的消融。【官方】
- 公式（V3 公式 17–20）【官方】：
  - `L_Bal = α Σ_{i=1}^{N_r} f_i P_i`
  - `f_i = (N_r / (K_r T)) Σ_{t=1}^{T} 1(s_{i,t} ∈ Topk({s_{j,t}}, K_r))`
  - `s'_{i,t} = s_{i,t} / Σ_{j=1}^{N_r} s_{j,t}`
  - `P_i = (1/T) Σ_{t=1}^{T} s'_{i,t}`
  - 注意与 batch 级版本的区别：`s'` 的分母是**全专家**求和（不是 top-k 内归一化），`f_i` 带 `N_r/K_r` 放大系数。

### 3.5 通信：all-to-all 规模与 EP 取值

- **payload 计算（ISCA'25 官方示例）**：dispatch 用 FP8（1 Byte）、combine 用 BF16（2 Bytes），hidden ≈ 7K，每 token 送到 8 个 routed expert + 1 个 shared expert（因子 **9**），每卡批 32 token，CX7 400Gbps IB 有效 50 GB/s：
  `Comm. Time = (1+2) × 32 × 9 × 7K / 50 GB/s = 120.96 μs`；双 micro-batch overlap 下每层 `2 × 120.96 = 241.92 μs`，61 层 = **14.76 ms** ⇒ 理论 TPOT 上限 ≈ 67 tokens/s。若换 GB200 NVL72（900 GB/s 单向）⇒ 每步 6.72 μs，理论 TPOT 上限 > 0.82 ms ≈ 1200 tokens/s（作者注明纯理论、未经实测）。【官方】
- 生产环境：prefill/decode 分离，给大 batch prefill 与低延迟 decode **分配不同的 EP group size**；MLA 与 MoE 计算分两阶段与 dispatch/combine 重叠。【官方】
- **常见 EP 取值（第三方/官方 recipe）**：
  - vLLM V3/R1：`--tensor-parallel-size 8 --enable-expert-parallel`（TP8+EP）或 `--data-parallel-size 8 --enable-expert-parallel`（DP8+EP）；8×H200 / 8×MI300X FP8。
  - vLLM V3.2-Exp：推荐 `-dp 8 --enable-expert-parallel`（DP=8, EP=8, TP=1，因为 kernel 主要按 TP=1 优化），备选 `-tp 8`。
  - TRT-LLM：Attention DP8、MoE sparse experts EP8、shared experts DP8、Router GEMM DP8；GB200 NVL72 上 Wide-EP 扩到 **EP16 / EP32**。
  - 小模型参考：V3 自带 `convert.py` 示例用 `--n-experts 256 --model-parallel 16`（2 节点 × 8 卡）。
- 节点受限路由（node-limited routing）让每个 token 最多只路由到 M 个节点；V3 的 `n_group=8 / topk_group=4` 即该机制的配置。【官方】

来源: https://arxiv.org/html/2412.19437v2 ; https://arxiv.org/html/2408.15664v1 ; https://arxiv.org/html/2505.09343v1 ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py ; https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html ; https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html ; https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/refs/heads/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md

---

## 4. MTP（Multi-Token Prediction）

| 项 | 数值 / 结构 | 来源级别 |
|---|---|---|
| MTP 模块个数 | **1**（`num_nextn_predict_layers = 1`）；权重文件里 MTP 层 id = **61**（紧接 0..60 主模型层） | 【官方】README_WEIGHTS / config |
| 模块参数 | **11.5B unique params**，激活 **2.4B**（含共享的 0.9B Embedding + 0.9B Head） | 【官方】 |
| 模块层结构 | `embed_tokens`（与主模型共享）→ `enorm` / `hnorm`（RMSNorm）→ `eh_proj`（降维投影）→ 1 个 transformer 层（`model.layers.61.self_attn & mlp`，与主模型层同构）→ `shared_head`（与主模型输出头共享） | 【官方】 |
| 训练 loss 权重 λ | **未找到**（本轮抓到的 V3 正文中未出现 λ 的具体数值） | — |
| 训练期定位 | 「Our MTP strategy mainly aims to improve the performance of the main model, so during inference, we can directly discard the MTP modules」 | 【官方】引文见第三方笔记 |
| 推理用法 | self-speculative decoding：低成本产出候选 token 并并行验证 | 【官方】 |
| **接受率** | V3 报告：第二个 token 的接受率「ranges between **85% and 90%** across various generation topics」；ISCA'25 表述为「**80% to 90%** for predicting the second subsequent token」 | 【官方】(两处口径略有差异) |
| **加速比** | **1.8× TPS**（"delivering 1.8 times TPS"）；ISCA 同样写 1.8×，并注明会「slightly hurting the throughput」 | 【官方】 |
| 工程取值 | TRT-LLM：低延迟场景推荐 **MTP-3**（num_nextn_predict_layers=3），其他场景 MTP-1；其 DSA indexer 的 MQA kernel 原本只支持 seqlen 1 或 2（即仅 MTP-0/MTP-1），后加 MTP-3 支持 | 【第三方】 |
| 第三方实测 | 社区反馈对 Qwen3.5 微调后 MTP-1 接受率可达 98%+（与 DeepSeek 本体无关，仅作参考） | 【第三方】 |

来源: https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/README_WEIGHTS.md ; https://jbarrow.ai/field_notes/deepseek-v3/ （转引 V3 报告 §5.4.3）; https://arxiv.org/html/2505.09343v1 ; https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/refs/heads/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md

---

## 5. FP8 训练与推理

### 5.1 训练侧（V3 官方 + ISCA'25 官方）

| 项 | 精确值 | 来源级别 |
|---|---|---|
| 格式 | `e4m3`（对应 `torch.float8_e4m3fn`） | 【官方】HF `quantization_config.fmt` |
| **激活量化块** | **tile-wise 1×128** | 【官方】ISCA'25 §3.1 |
| **权重（模型权重）量化块** | **block-wise 128×128** | 【官方】ISCA'25 §3.1 + `weight_block_size: [128,128]` |
| 激活量化方案 | `dynamic`（动态在线量化）；运行时按 **per-token-per-128-channel** 在线量化 | 【官方】README_WEIGHTS |
| 权重 scale 存储 | `weight_scale_inv`，**float32 Tensor**，与权重一同存储；反量化 = `(128×128 block) × weight_scale_inv`；不足 128 的块先 zero-pad 到 128 再算 scale，量化后去掉 pad | 【官方】README_WEIGHTS |
| 反量化顺序 | 源码里权重反量化 shape 变换为 `view(M/128,128,N/128,128).transpose(1,2)`，即 128×128 tile 主序 | 【官方】V3.2-Exp `inference/model.py: weight_dequant` |
| 累加精度（Hopper Tensor Core 限制） | 「After aligning 32 mantissa products by right-shifting based on the maximum exponent, the Tensor Core only maintains their **highest 13 fraction bits** for addition, and truncates bits exceeding this range. Addition results are accumulated to **FP22 registers (1 sign bit, 8 exponent bits, and 13 mantissa bits)**」 | 【官方】ISCA'25 §3.1.1 |
| 「每 128 个元素提升到 CUDA core 用 FP32 累加」 | V3 报告 §3.3.2 的 `promote every 128` 说法在本轮只见到二手片段，**未在原文正文中直接读到（未找到/未核实）** | — |
| 细粒度量化的代价 | 从 Tensor Core 把部分和搬到 CUDA Core 乘 scale 的 dequant 开销大、数据搬移频繁，降低效率 | 【官方】ISCA'25 |
| 精度损失验证 | 先在 V2 的 16B 与 230B 上做 FP8 消融，「the relative accuracy loss compared to BF16 remains **below 0.25%**」 | 【官方】ISCA'25 §2.4 |
| 哪些 GEMM 走 FP8 | 官方 Figure 1 用图示标注了前向/反向哪些组件用 FP8，**本轮未能读到该图与其配套文字清单（未找到精确清单）**；可确认的是：MoE expert 的 GEMM 与注意力的线性投影属于 FP8 计算范围（推断，基于 DeepGEMM 提供的 grouped/masked FP8 GEMM 与其 MoE 用途） | 部分推断 |
| 训练稳定性 | 全程「did not experience any irrecoverable loss spikes or perform any rollbacks」 | 【官方】 |

### 5.2 scale 的格式：e8m0 / fp32，以及为什么可以是 2 的幂

- **权重 scale = fp32**（V3 runtime 明确：`weight_scale_inv` 是 float32；DeepGEMM 亦说明 **SM90 要求 FP32 scale**）。【官方】
- **推理激活 quant 支持 `ue8m0`**：`inference/kernel.py` 的 `act_quant_kernel` 里
  `s = amax / 448.`；若 `scale_fmt == "ue8m0"`：`exp = ceil(log2(s)); s = exp2(exp)` —— 即**把 scale 向上取整到 2 的幂**；`amax` 会 clamp 到最小 `1e-4`。【官方】
- **为什么可以是 2 的幂**：UE8M0 是「8 位指数、0 位尾数」的格式（DeepGEMM README 引用 NVIDIA PTX 的 alternate floating-point data formats；SM100 要求把 4 个 UE8M0 打包进一个 `torch.int`），指数位能表达的实数**恰好只有 2 的幂**，且乘以 2 的幂在硬件上等价于指数加减、可无损完成（推断：格式定义 + 实现动机）。
- **448 的来历**：e4m3 的最大可表示值 448，`s = amax/448` 是把动态范围顶到 e4m3 上限，再向上取到 2 的幂以保证不溢出（推断：基于代码常量 448 与 e4m3 的定义）。
- DeepGEMM 侧格式差异：SM90 用 FP32 scale；SM100 用 packed UE8M0；DeepGEMM 在 H800 上实测最高 **1550 TFLOPS**。【官方】

### 5.3 推理时 KV cache 用 FP8

- **V3.2-Exp 官方参考实现**直接模拟 FP8 KV cache：`kv_fp8, kv_scale = act_quant(kv, 128, scale_fmt)` 再反量化回 bf16 存入 cache，注释写「we use fp8 kv cache in actual deployment, so here we simulate the precision by casting kv to fp8 and then back to bf16」；block_size = **128**。【官方】
- **vLLM V3.2-Exp recipe**：默认使用自定义 **fp8 kvcache**；可用 `kv_cache_dtype=bfloat16` 切回 BF16；建议「短请求用 bfloat16，长请求用 fp8」（默认能缓存更多 token，但有额外量化/反量化开销）。【第三方】
- **精度影响（TRT-LLM，GPQA-Diamond）**【第三方】：

  | KV cache / 稀疏 MLA 精度 | FP8 checkpoint | NVFP4 checkpoint |
  |---|---|---|
  | BF16 Sparse MLA + BF16 KV | 80.30 | 79.29 |
  | FP8 Sparse MLA + FP8 KV | **78.28** | **80.30** |

  即 FP8 化在 FP8 权重下掉 2.02 分、在 NVFP4 权重下反而不掉；FP8 稀疏 MLA + FP8 KV 带来 **最高 +47.03% TPS/GPU** 吞吐提升。【第三方】
- Indexer K cache 单独一套：V3.2 的 indexer 用 **blockwise FP8**（Q/K 都量化，K cache 按 block 存量化值与 scale），Top-K 用 FP32；Sparse MLA 的 KV cache 用 per-tensor FP8。【第三方】

来源: https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/README_WEIGHTS.md ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/kernel.py ; https://arxiv.org/html/2505.09343v1 ; https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py ; https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html ; https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/refs/heads/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md

---

## 6. 部署侧的数字（务必带硬件口径）

### 6.1 官方/第三方实测吞吐与延迟

| 平台与配置 | 负载 | 结果 | 来源级别 |
|---|---|---|---|
| **B200**，`trtllm-bench` + NVFP4 权重，**TP4**，MTP-3，batch 1 | ISL 8K / OSL 1K，10 req | TTFT **425.99 ms**；TPOT **3.2344 ms**；Per-GPU output **68.54 tok/s**；Per-user **312.07 tok/s**；总输出 274.18 tok/s | 【第三方】TRT-LLM |
| **B200**，TP8+EP8，MTP-1，max batch 256，concurrency 256 | ISL 8K / OSL 1K，768 req | 总输出 **8618.22 tok/s**；总 token **77563.94 tok/s**；Per-GPU output **1077.28 tok/s**；TTFT **19537.78 ms**；TPOT **98.52 ms** | 【第三方】TRT-LLM |
| **GB200 NVL72**，Wide-EP | ISL 8K / OSL 1K，rate matching | EP16/EP32 相对 EP4/EP8 **最高 2.28× per-GPU output throughput** | 【第三方】TRT-LLM |
| **8×H200 / 8×MI300X**，TP8+EP 或 DP8+EP，FP8 | `vllm bench serve`，8K/1K，batch 1（recipe 的 Expected Output） | TTFT **560.00 ms**；TPOT **15.85 ms**；输出吞吐 **61.00 tok/s**；总 token 吞吐 **543.06 tok/s** | 【第三方】vLLM |
| **8×H200/H20（141GB×8）**，DP8+EP8，V3.2-Exp | GSM8K，`num_concurrent=100` | 5-shot **0.9591**；20-shot **0.9538**（精度校验，非吞吐） | 【第三方】vLLM |
| **H800 集群**，DeepSeek 线上服务实测（$2/GPU·hour 折算） | prefill / decode 曲线随 token 位置变化 | 只有曲线图，**具体数值未找到**（论文 Figure 3） | 【官方】 |
| **CX7 400Gbps IB，每卡 1 expert、批 32 token** | 理论分析 | 每层 241.92 μs，61 层 **14.76 ms** ⇒ 上限 ≈ **67 tok/s**；GB200 NVL72（900GB/s）⇒ 理论 > **0.82 ms**、≈ **1200 tok/s**（未实测） | 【官方】ISCA'25 |

### 6.2 单机 8 卡能不能放下、需要多少显存

| 项 | 数字 | 来源级别 |
|---|---|---|
| FP8 权重体积 | 671B 参数 × 1 B ≈ **671 GB**（+ 128×128 块的 fp32 scale：671e9/16384×4 ≈ **164 MB**，可忽略；MTP 另 11.5B ≈ 11.5 GB） | （推断：官方参数量 + 官方 128×128 scale 规格） |
| BF16 权重体积 | ≈ **1342 GB**（仅权重，不含 KV cache 与激活） | （推断） |
| 8×H200 / H20（141 GB × 8 = 1128 GB） | FP8 权重可放下（约 671/1128 ≈ 59% 显存），余量给 KV cache；这正是 vLLM 官方 recipe 的 8×H200(H20) 配置 | （推断）+【第三方】 |
| 8×H100/H800（80 GB × 8 = 640 GB） | **640 GB < 671 GB，纯 FP8 权重已放不下**，需 2 节点（V3 官方 demo 即 `--nnodes 2 --nproc-per-node 8`，MP=16） | （推断）+【官方】README |
| FP4 路径 | vLLM 官方支持 `nvidia/DeepSeek-R1-FP4` 在 **4×B200** 上跑（TP4+EP 或 DP4+EP），需 `VLLM_USE_FLASHINFER_MOE_FP4=1` | 【第三方】 |
| KV cache 占用（整模型每 token） | BF16 **70.272 KB**；FP8 ≈ **35.1 KB**（推断：576 B/token/layer × 61） | 【官方】+（推断） |
| 单序列 128K KV cache | BF16 ≈ **9.21 GB**，FP8 ≈ **4.61 GB**（推断） | （推断） |
| 官方 demo 启动 | `torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR generate.py --ckpt-path ... --config configs/config_671B.json` | 【官方】README |
| 消费级部署 | KTransformers 可在约 **$10,000** 的消费级 GPU 服务器上跑完整 V3，约 **20 TPS**；AI SoC PC 上 MoE「nearly 20 tokens per second (TPS)，甚至翻倍」 | 【官方】ISCA'25 |
| 硬件支持面 | vLLM v0.6.6 起支持 V3 的 FP8/BF16（NVIDIA 与 AMD）；SGLang v0.4.1 起 NVIDIA + AMD；TRT-LLM 支持 BF16 与 INT4/INT8 weight-only；昇腾 MindIE 适配 BF16；V3.2-Exp 仅支持 Hopper 与 Blackwell 数据中心 GPU | 【官方】+【第三方】 |
| 计算量口径（对比用） | V3 **250 GFLOPS/token**；V2 236B **155**；Qwen-72B dense **394**；LLaMa-405B dense **2448**（序列长 4096） | 【官方】ISCA'25 |
| 推荐的常见 TP/EP 组合 | vLLM V3/R1：TP8+EP / DP8+EP；vLLM V3.2-Exp：DP8+EP8（TP=1，kernel 优化口径）；TRT-LLM：Attention DP8 + Expert EP8，Wide-EP 用 EP16/EP32；若报 `flashmla ... invalid configuration argument`，把 `--max-num-seqs` 降到 256 或更小（默认 1024） | 【第三方】 |

来源: https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/refs/heads/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md ; https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html ; https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html ; https://arxiv.org/html/2505.09343v1 ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/README.md

---

## 7. 附加：DeepSeek Sparse Attention（V3.2-Exp / V3.2，2025-09/12）

| 项 | 数值 / 公式 | 来源级别 |
|---|---|---|
| 定位 | V3.2-Exp 相对 V3.1-Terminus 的**唯一**架构改动；V3.2 与 V3.2-Exp 架构完全相同，仅继续训练 + 后训练 | 【官方】 |
| indexer 打分 | `I_{t,s} = Σ_{j=1}^{H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)`；选 ReLU 是为了吞吐 | 【官方】 |
| 选择机制 | 只保留 top-k index 分数对应的 KV entry，然后在这些 entry 上做注意力；**k = 2048** | 【官方】+【第三方】 |
| 参考实现的精确超参 | `index_n_heads = 64`，`index_head_dim = 128`，`index_topk = 2048`；indexer 的 `weights_proj` 用 fp32（checkpoint 存 bf16）；indexer K cache 存 `float8_e4m3fn` + 每 128 维一个 fp32 scale | 【官方】V3.2-Exp `inference/model.py` |
| indexer 的两个坑 | (1) q/k 在 FP8 量化前先做 **Hadamard 旋转**（`rotate_activation`，`scale = hidden_size**-0.5`）；(2) indexer 里 RoPE 必须是**非交错（non-interleaved）**布局，而 MLA 里是交错布局 —— 官方 2025.11.17 专门发更正说明 | 【官方】 |
| 复杂度 | 主注意力 O(L²) → **O(Lk)**；indexer 仍是 O(L²) 但计算量远小于 MLA | 【官方】 |
| 训练两阶段 | dense warm-up：冻结除 indexer 外所有参数，目标 `L^I = Σ_t D_KL(p_{t,:} ‖ Softmax(I_{t,:}))`，其中 `p` 是主注意力分数按头求和后 L1 归一化；1000 step × 16 seq × 128K = 2.1B tokens，LR 1e-3。sparse 训练：全参数 + indexer 只在选中集合 S_t 上对齐，15000 step × 480 seq × 128K = 943.7B tokens，LR 7.3e-6，indexer 输入 detach | 【官方】 |
| 短序列优化 | 官方的短序列 prefill 用 masked MHA mode 模拟 DSA；TRT-LLM 在 N ≤ 2048 时走 dense fast path（约 1.03× @1K/1K） | 【官方】+【第三方】 |
| Top-K kernel | FP32 top-k；非确定性 vs 确定性在 GPQA-Diamond 上 79.9 vs 79.8（FP8 权重）/ 79.4 vs 80.3（NVFP4 权重），故选非确定性；自研 radix-select 版平均 **7.41×** 于 `torch.topk`（输入 [64, 9295]），整体 e2e 提速 25%~40%（低延迟）/14%~24%（吞吐） | 【第三方】TRT-LLM |

来源: https://arxiv.org/html/2512.02556v1 ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/README.md ; https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py ; https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/ ; https://raw.githubusercontent.com/longcheng-nv/TensorRT-LLM/refs/heads/main/docs/source/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md

---

## 8. 附加：DeepSeek-V4（2026-04-24 Preview，1M 上下文）

**官方规格**【官方，api-docs 2026/04/24】：V4-Pro **1.6T 总 / 49B 激活**；V4-Flash **284B 总 / 13B 激活**；1M 上下文为所有官方服务默认；注意力 = **token-wise compression + DSA**；首发即开源权重（tech report `DeepSeek_V4.pdf`）+ OpenAI 与 Anthropic 双 API。后续版本：V4-Pro GA 2026/08/13、V4-Flash-Vision-Exp 2026/08/21、**V4.1-Flash 2026/09/10**（`deepseek-flash`）。

**架构要点（HF Transformers 文档转述论文 §2）**【第三方转述官方论文，模型具体尺寸 gated 未拿到 config】：

| 组件 | 内容 |
|---|---|
| 注意力替换 | **MLA 被 hybrid local + long-range 设计取代**；每层按 `config.layer_types[i]` 三选一：`sliding_attention`（纯滑窗）、**CSA**（Compressed Sparse Attention，压缩率 `m=4` 重叠窗口 + Lightning Indexer 每 query 取 top `index_topk` 个 block）、**HCA**（Heavily Compressed Attention，压缩率 `m'=128` 非重叠窗口，无 indexer） |
| 共享 K=V MQA | `num_key_value_heads = 1`，`kv_proj` 产出单个 KV head，同一张量既当 K 又当 V |
| Partial RoPE | `qk_rope_head_dim = head_dim * partial_rotary_factor`，对每个 head 尾部通道做 interleaved-pair RoPE；输出侧 rope 片段用位置 `-i` 旋转（eq. 26）保持相对距离 |
| 其他 | per-head learnable attention sink（eq. 27）；grouped low-rank output projection（`o_groups` 组 → `o_lora_rank`）；一条共享 sliding-window K=V 分支 |
| 残差 | 残差连接替换为 **mHC（Manifold-Constrained Hyper-Connections）**：`hc_mult` 条并行残差流，`[B,S,hc_mult,D]`；`attn_hc` / `ffn_hc` 用 `(pre, post, comb)` 三元组混合，`comb` 经 `hc_sinkhorn_iters` 次 Sinkhorn–Knopp 迭代投影为双随机矩阵 |
| MoE 调度 | 按层由 `mlp_layer_types` 决定：`hash_moe`（前若干层，默认 3 层，专家 index 来自冻结的 `tid2eid[input_ids]` 查表，只有「选哪些专家」是静态的，权重仍由 gate 学）+ `moe`（常规 top-k） |
| 路由打分 | 亲和度从 V3 的 `Sigmoid(·)` 改为 **`Sqrt(Softplus(·))`**；**去掉 V3 的 `n_group`/`topk_group` 约束**；仍保留 aux-loss-free（`noaux_tc`）的 `e_score_correction_bias`（只改 top-k argmax、不流梯度） |
| 专家 FFN | **clamped SwiGLU**：`gate.clamp(max=swiglu_limit)`、`up.clamp(min=-swiglu_limit, max=swiglu_limit)`；权重布局 `[num_experts, 2*moe_intermediate_size, hidden_size]`；单个 shared expert 是普通 SwiGLU MLP，宽度 `moe_intermediate_size` |
| Cache | `DeepseekV4HCACache`（滑窗 K=V + HCA compressor buffer/pool/count）、`DeepseekV4CSACache`（再加 CSA overlap 状态与 indexer buffer/pool/count，维度 `index_head_dim`） |

**社区 mini 复现配置中的架构默认值**【第三方，非官方 Pro/Flash 数值，仅供理解机制】：`vocab_size=129280`、`num_key_value_heads=1`、`q_lora_rank=256`、`o_lora_rank=256`、`o_groups=2`、`n_shared_experts=1`、`norm_topk_prob=True`、`scoring_func="sqrtsoftplus"`、`topk_method="noaux_tc"`、`routed_scaling_factor=1.5`、`index_n_heads=4`、`index_head_dim=32`、`index_topk=64`、`sliding_window=32`、`hc_mult=4`、`hc_eps=1e-6`、`hc_sinkhorn_iters=20`、`swiglu_limit=10.0`、`num_nextn_predict_layers=1`、`rope_theta=10000.0`、`compress_rope_theta=160000.0`、`rope_scaling={yarn, factor 16, original_max_position_embeddings 65536, beta_fast 32, beta_slow 1}`、`max_position_embeddings=1048576`；`compress_ratios` 默认模式 = 前 2 层纯滑窗、中间交替 CSA(4)/HCA(大)、末层滑窗。

**V4 待补（未找到）**：Pro/Flash 的层数、hidden、专家数/激活数、词表、上下文扩展方式、DSA 在 V4 里的 top-k、MTP 模块数与接受率、部署吞吐。

来源: https://api-docs.deepseek.com/news/news260424 ; https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/deepseek_v4.md ; https://huggingface.co/kshitijthakkar/deepseek-v4-mini-300M-from-flash/raw/main/code/deepseek_v4/configuration_deepseek_v4.py

---

## 9. 面试可直接用的「精确数字」速查（Top 20）

1. 671B / 37B（README_WEIGHTS 口径 36.7B，含 0.9B embedding + 0.9B head）、61 层、7168 hidden、128 heads、256 routed + 1 shared、top-8、词表 129280。
2. MLA：`kv_lora_rank=512`、`q_lora_rank=1536`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128` ⇒ 192 的头维；每 token 每层只缓存 **576** 个元素。
3. KV cache：MLA **70.272 KB/token**（BF16）vs LLaMA-3.1 405B 516.096 KB（7.28×）vs Qwen-2.5 72B 327.680 KB（4.66×）。
4. 吸收：`W^UK → W^Q`、`W^UV → W^O`；RoPE 必须解耦，否则 RoPE 矩阵夹在中间导致不可交换、无法吸收。
5. 路由：`Sigmoid(u_t^T e_i)` + 组受限（8 组选 4）+ top-8 + 选中内归一化 + `route_scale=2.5`。
6. aux-loss-free：`b_i = b_i + u·sign(c̄_i − c_i)`，最优 **u = 0.001**；bias 只参与 top-k 选择、不进 gating 值 ⇒ 无干扰梯度；须用上一 batch 的负载以免泄漏 future token。
7. 序列级辅助损失：`L_Bal = α Σ f_i P_i`，`f_i` 带 `N_r/(K_r T)` 系数；V3 只说 α 极小（精确值未找到）。
8. MTP：**1** 个模块、11.5B unique 参数、2.4B 激活、接受率 **85%–90%**（ISCA 写 80%–90%）、**1.8× TPS**。
9. FP8：激活 **1×128** tile-wise、权重 **128×128** block-wise；scale 为 fp32；Hopper 累加只有 **13 位尾数**、落在 **FP22** 寄存器。
10. ue8m0：`s = amax/448` → `exp2(ceil(log2 s))`，因此 scale 必为 2 的幂；UE8M0 = 8 位指数、无尾数。
11. 训练成本：2788K H800 GPU hours / $5.576M（$2/GPU·hour），2048×H800，14.8T tokens，每 1T tokens = 180K GPU hours ≈ 3.7 天。
12. 部署：B200 TP4 + MTP-3 单并发 **TPOT 3.2344 ms / TTFT 425.99 ms**；B200 TP8+EP8 批 256 达 **8618 tok/s** 输出（per-GPU 1077.28 tok/s）。
13. Wide-EP：GB200 NVL72 上 EP16/EP32 相对 EP4/EP8 **最高 2.28×** per-GPU 吞吐。
14. 每层 all-to-all：`(1B+2B)×32×9×7K/50GB/s = 120.96 μs`，×2×61 ⇒ **14.76 ms**（≈67 tok/s 理论上限）。
15. V3.2 DSA：top-k **2048**，indexer **64 heads × 128 dim**，ReLU + FP8 + Hadamard 旋转，indexer K cache = blockwise FP8；复杂度 O(L²)→O(Lk)。
16. DSA 继续训练：warm-up 2.1B tokens（LR 1e-3）/ sparse 943.7B tokens（LR 7.3e-6，top-k 2048）。
17. 单序列 128K 的 KV cache ≈ **9.21 GB**（BF16）/ 4.61 GB（FP8）（推断）。
18. 单机 8×H100/H800（640 GB）**放不下** 671 GB 的 FP8 权重；8×H200/H20（1128 GB）可放（推断）。
19. V3.1 = V3 架构 + **840B tokens** 长上下文继续预训练；V3.2 与 V3.2-Exp 架构完全一致。
20. V4（2026）：Pro **1.6T/49B**、Flash **284B/13B**、1M 上下文；MLA 被「滑窗 + CSA(m=4) + HCA(m'=128)」混合注意力取代，残差换成 mHC，前几层用 hash MoE，亲和度改 `Sqrt(Softplus(·))`。
