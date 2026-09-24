# E — LeetCUDA 参考实现精确事实提取（FlashAttention + TMA/WS + 公共基础设施）

> 本文件是 `interview.md` §7（FlashAttention + TMA/WS 章节）的溯源底稿：所有模板参数、宏展开、
> PTX 字符串、tol 数值、注释原文均从源码**逐字摘出**，不含推测。只读，未修改 LeetCUDA 任何文件。
>
> | 项 | 值 |
> |---|---|
> | 任务给定锚点 | HEAD `e831d970a099f5ce8fd0495ddd1df09206d56918` |
> | 提取时仓库实际 HEAD | **`6c86259d7eca5e6b34fcd4bd21e4ebe1c890abc2`**（`Fix link text for LeetCUDA PDF in README`，2026-09-22，工作区 clean） |
> | 同目录同名章节 | 本目录 `README.md` 的「# E · FlashAttention / WGMMA+WS / common.cuh」一节即本节内容（被合并进总底稿）；本文件是任务要求的独立分册 |
>
> **版本前移说明（重要）**：抓取期间上游从 `e831d97` 前进到 `6c86259`，`kernels/interview/` 被改动过
> （`git diff --stat e831d97 6c86259`：`flash_attn.cuh` +689、`notes-v2.cu` +356、`build.sh` +89、
> `common.cuh` +66、`ffpa_attn.cuh` +4）。**本文件全部事实已在 `6c86259` 上逐条复核**：
> `flash_attn.cuh` 被要求精读的中段（L1–3445）逐字未变，L29/L65/L68/L808/L838/L949/L962/L1081/L1501/L1503
> 等引用行均命中原文；新增内容全部落在 L3445 之后（persist-D kernel）与 `notes-v2.cu` 的 test/bench 段。
>
> **实测物理行数 @ `6c86259`**（空行计入）：`flash_attn.cuh` **4179**、`notes-v2.cu` **5219**、
> `base.cuh` **909**、`common.cuh` **803**、`ffpa_attn.cuh` **641**、`build.sh` **245**、
> `README.md` **55**、`ws-hgemm/naive_ws_hgemm_sm8x.cu` **486**。
> 任务描述给的 `3245 / 4465 / 815 / 719 / 584 / 164 / 61 / 385` 里只有 `base.cuh` 与
> `naive_ws_hgemm_sm8x.cu` 对得上，其余是更早快照。**写正文一律以实测为准。**

---

## kernels/interview/common.cuh

- **文件定位**：整个 `interview/` 章节的最底层公共模块——CUDA 头、基础类型宏、MMA/WGMMA PTX 宏、
  XOR swizzle 函数、TMA/mbarrier helper、host 端 TensorMap helper。被 `base.cuh` 及所有上层 `.cuh` 依赖。
- **导出的符号 = 模板参数全列表**（逐条原样；本文件**不含任何 `__global__` kernel**，是纯工具层）
  - 常量：`static constexpr int kWarpSize = 32;`
  - 宏：`INT4(value)` / `FLOAT4(value)` / `HALF2(value)` / `HOST_DEVICE_INLINE`
  - 宏：`CP_ASYNC_COMMIT_GROUP()` / `CP_ASYNC_WAIT_ALL()` / `CP_ASYNC_WAIT_GROUP(n)` / `CP_ASYNC_CG(dst, src, bytes)`
  - 宏：`LDMATRIX_X4(R0,R1,R2,R3,addr)` / `LDMATRIX_X2(R0,R1,addr)` / `LDMATRIX_X2_T(R0,R1,addr)`
  - 宏：`HMMA16816(RD0,RD1,RA0..RA3,RB0,RB1,RC0,RC1)` / `HMMA16816F32(RD0..RD3,RA0..RA3,RB0,RB1,RC0..RC3)`
  - 宏（`NOTES_V2_ENABLE_WGMMA` 门控）：`WGMMA_FENCE()` / `WGMMA_COMMIT_GROUP()` / `WGMMA_WAIT_GROUP(n)` /
    `SMEM_DESC_ENCODE(x)` / `WGMMA_M64N128K16_F16F16F16(d, sA, sB, ScaleD, ScaleA, ScaleB, TransA, ...)`
  - 宏（`NOTES_V2_ENABLE_SETMAXNREGS` 门控）：`NOTES_V2_REG_DEALLOC(N)` → `warpgroup_reg_dealloc<N>()`；
    `NOTES_V2_REG_ALLOC(N)` → `warpgroup_reg_alloc<N>()`；**未定义该宏时两者都展开成 `((void)0)`**
  - `template <const int kColStride = 16, const int kStep = 8> static __host__ __device__ __forceinline__ int permuted(int i, int j)`
  - `template <const int kColStride = 16> static __host__ __device__ __forceinline__ int swizzle_v1_impl(int i, int j)`
  - `template <int B, int M, int S> struct SwizzleBMS`
  - `template <const int kColStride = 16> static __host__ __device__ __forceinline__ int swizzle_v2_impl(int i, int j)`
  - 派发器 `swizzle<kColStride>(i,j)`：未定义 `NOTES_V2_ENABLE_SWIZZLE_V2` → v1；定义 → v2
  - `static __device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(void const *ptr)`；`static __device__ __forceinline__ void tma_fence_proxy_async_shared_cta()`
  - `static __device__ __forceinline__ void tma_load_2d(void *dst, const CUtensorMap *tensor_map, int minor_coord, int major_coord, cuda::barrier<cuda::thread_scope_block> &barrier)`
  - `static __device__ __forceinline__ void tma_arrive_expect_tx(cuda::barrier<cuda::thread_scope_block> &barrier, uint32_t bytes)`
  - `template <uint32_t kNumRegs> __device__ __forceinline__ void warpgroup_reg_dealloc()` / `warpgroup_reg_alloc()`
  - `template <int BM, int BN, int BK, int QSIZE> struct WgmmaSMem`（`alignas(128) half A[BM*BK*QSIZE];` + `alignas(128) half B[BN*BK*QSIZE];`）
  - `template <int BM, int BN, int BK, int QSIZE> struct TmaMmaWSSMem`（`static_assert(BK == 64, "The 128B swizzle helper below is specialized for BK=64");`；成员 `half A[BM*BK*QSIZE]; half B[BN*BK*QSIZE];`——**A/B 无 alignas**）
  - `template <int BlockMajorSize, int BlockMinorSize> __host__ static inline void create_tensor_map(CUtensorMap *tma_map, half *gmem_ptr, int blocks_height, int blocks_width)`
  - `template <int BlockMajorSize = 128, int BlockMinorSize = 64> __host__ static inline CUtensorMap *allocate_and_create_tensor_map(half *src, int blocks_height, int blocks_width)`
- **PTX 宏与工具宏**（宏名 + 完整展开）
  - `INT4` → `(reinterpret_cast<int4 *>(&(value))[0])`；`FLOAT4` / `HALF2` 同构（float4 / half2）
  - `CP_ASYNC_COMMIT_GROUP()` → `asm volatile("cp.async.commit_group;\n" ::)`
  - `CP_ASYNC_WAIT_ALL()` → `asm volatile("cp.async.wait_all;\n" ::)`
  - `CP_ASYNC_WAIT_GROUP(n)` → `asm volatile("cp.async.wait_group %0;\n" ::"n"(n))`
  - `CP_ASYNC_CG(dst, src, bytes)` → `asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))`
    （注释原文：`注意：cg 只支持 16 bytes，ca 支持 4/8/16 bytes`）
  - `LDMATRIX_X4(R0,R1,R2,R3,addr)` → `ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];`
  - `LDMATRIX_X2(R0,R1,addr)` → `ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];`
  - `LDMATRIX_X2_T(R0,R1,addr)` → `ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];`
  - `HMMA16816(...)` → `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};`
  - `HMMA16816F32(...)` → `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};`
  - `WGMMA_FENCE()` → `wgmma.fence.sync.aligned;`；`WGMMA_COMMIT_GROUP()` → `wgmma.commit_group.sync.aligned;`；
    `WGMMA_WAIT_GROUP(n)` → `wgmma.wait_group.sync.aligned %0;`
  - `SMEM_DESC_ENCODE(x)` → `((((uint64_t)(x)) & 0x3FFFF) >> 0x4)`
  - `tma_load_2d` 内 PTX → `"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"`
  - `tma_arrive_expect_tx` 内 PTX → `"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"`
  - `tma_fence_proxy_async_shared_cta()` 内 PTX → `"fence.proxy.async.shared::cta;\n"`
  - **注意：不存在名为 `SWIZZLE_*` 的宏**。`SWIZZLE_32B/64B/128B` 只出现在**注释**里，是
    `permuted<kColStride>` 的别名说明：`kColStride=16 fp16=32B/row → Swizzle<1,4,3> = SWIZZLE_32B`；
    `=32 → Swizzle<2,4,3> = SWIZZLE_64B`；`=64 → Swizzle<3,4,3> = SWIZZLE_128B`。
  - swizzle 位级公式（`permuted` 原文）：`kStep==4` → `(((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2`；
    `kColStride==16` → `(((j >> 3) ^ (i >> 2)) & 1) << 3;  // SWIZZLE_32B`；
    `kColStride==32` → `chunk=(j>>3)&3; xor_mask=((i>>1)&1)|(((i>>2)&1)<<1); return (chunk ^ xor_mask) << 3;  // SWIZZLE_64B`；
    `kColStride==64` → `chunk=(j>>3)&7; xor_mask=(i&1)|(((i>>1)&1)<<1)|(((i>>2)&1)<<2); return (chunk ^ xor_mask) << 3;  // SWIZZLE_128B`。
    `swizzle_v2_impl` 走 cute 位公式：`off=(i*kColStride+j)*sizeof(half)`；`sw = SwizzleBMS<B,M,S>::apply(off)`，
    其中 `B=(kColStride==16)?1:(kColStride==32)?2:3`, `M=4`, `S=3`；`return ((sw >> M) & ((1 << B) - 1)) * kStep;`（kStep=8）。
  - TMA helper 有**两套实现**，由 `NOTES_V2_FORCE_INLINE_ASYNC_PROXY` 门控：定义 → 裸 `asm volatile` PTX；
    未定义 → `cuda::ptx::` / `cuda::device::` C++ 包装（后者在 sm_120a 上让 ptxas 以 C7506 丢弃 `setmaxnreg`）。
- **kernel 的完整签名**：无（本文件无 kernel）。
- **块/线程/warp 组织**：无（仅 `kWarpSize = 32` 常量）。
- **smem 布局与字节数**：只有 `WgmmaSMem` / `TmaMmaWSSMem` 两个布局结构体，本文件无具体字节数常量
  （字节数计算 `kSmemAllocateAB/Acc` 在 `ws-hgemm` 里）。
- **同步原语**：`fence.proxy.async.shared::cta`、`mbarrier.arrive.expect_tx.shared::cta.b64`、
  `cp.async.bulk.tensor.2d ... mbarrier::complete_tx::bytes`；`cuda::barrier<cuda::thread_scope_block>`
  的 init/arrive/wait 直接复用（注释：它内联为裸 mbarrier PTX，不触发 C7506，因此无需门控）。
- **测试与容差**：本文件无测试；只有 host 端 `create_tensor_map` 失败时
  `printf("cuTensorMapEncodeTiled failed: %d\n", (int)result);`
- **注释里的关键结论**（原样摘录）
  1. 「注意：cg 只支持 16 bytes，ca 支持 4/8/16 bytes」
  2. 「v2 不是新算法, 而是把 v1 的手写展开重新表达为 cute 的统一位置换公式, 便于脱离 cute 框架理解 swizzle 本质. v1/v2 bit-exact 等价.」（等价性由 host 端 `test_swizzle_equiv` / `--swizzle-eq-check` 验证）
  3. 「★ 关键易错点：TMA shape 参数写的是 (W,H) 而不是 (H,W)! 对 row-major [M,K] 矩阵，TMA shape = (K,M)，minor=K，major=M」
  4. 「On sm_120a (Blackwell, CUDA 13.2) ptxas drops setmaxnreg with C7506 even when the PTX is fully inlined (no call.uni), because ptxas treats cp.async.bulk.tensor (TMA) usage as an implicit extern-call boundary. sm_90a (Hopper) is unaffected.」
  5. 「SM120 不支持 WGMMA，但支持相同的 TMA 生产者协议和 warp 级 mma.sync。」
  6. TensorMap 细节：`gmem_prob_stride + 1` 跳过隐式最内维 stride；`smem_box_stride[5] = {1,1,1,1,1}` 不需要 +1；
     固定 `CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`；
     `gmem_prob_stride[5] = {sizeof(half), sizeof(half) * BlockMinorSize * blocks_width, 0, 0, 0}`。
- **编译门槛**：`#if defined(NOTES_V2_ENABLE_TMA_MMA_WS) && CUDART_VERSION < 13000` → `#error "NOTES_V2_ENABLE_TMA_MMA_WS requires CUDA Toolkit 13.0 or newer"`

---

## kernels/interview/base.cuh

- **文件定位**：Phase 0–5 的“面试速查表 + 基础算子”层——架构/带宽/Roofline 速查（纯注释）、
  Warp/Block Reduce、Dot Product、Elementwise、Softmax 三级、RMS/Layer Norm、RoPE、Mat Transpose。
- **导出的符号 = 模板参数全列表 = kernel 签名**（逐条原样）
  - `struct __align__(8) MD { float m; float d; };`（running max + running denominator）
  - `template <const int kWarpWidth = kWarpSize, typename T = float> __device__ __forceinline__ T warp_reduce_sum(T val)`
  - `template <const int kWarpWidth = kWarpSize, typename T = float> __device__ __forceinline__ T warp_reduce_max(T val)`
  - `template <const int kNumThreads = 256> __device__ float block_reduce_sum(float val)` / `block_reduce_max(float val)`
  - `template <const int kWarpWidth = kWarpSize> __device__ __forceinline__ MD warp_reduce_md(MD md1)`
  - `template <const int kNumThreads = 256> __global__ void block_reduce_all(float *a, float *y, int N)`
  - `template <const int kNumThreads = 256> __global__ void dot(float *a, float *b, float *y, int N)`
  - `template <const int kNumThreads = 256 / 4> __global__ void dot_vec4(float *a, float *b, float *y, int N)`
  - `__global__ void relu(float *x, float *y, int N)` / `__global__ void relu_vec4(float *x, float *y, int N)`
  - `__global__ void elementwise_add(float *a, float *b, float *c, int N)` / `__global__ void elementwise_add_vec4(float *a, float *b, float *c, int N)`
  - `__global__ void histogram(int *a, int *y, int N)`
  - `__global__ void merge_attn_states(float *output, const float *prefix_output, const float *prefix_lse, const float *suffix_output, const float *suffix_lse, int num_tokens, int num_heads, int head_size)`（无模板）
  - `template <const int kNumThreads = 256> __global__ void softmax_per_token(float *x, float *y, int N)` / `safe_softmax_per_token(...)` / `online_safe_softmax_per_token(const float *x, float *y, int N)`
  - `template <const int kNumThreads = 128> __global__ void rms_norm(float *x, float *y, float g, int N, int K)` / `layer_norm(float *x, float *y, float g, float b, int N, int K)`
  - `template <const int kNumThreads = 128 / 4> __global__ void rms_norm_vec4(...)` / `layer_norm_vec4(...)`
  - `__global__ void rope(float *x, float *out, int seq_len, int N)`
  - `__global__ void mat_transpose(float *x, float *y, const int row, const int col)` / `mat_transpose_padded(float *x, float *y, const int row, const int col)`
- **PTX 宏与工具宏**：无自有宏，全部依赖 `common.cuh` 的 `INT4/FLOAT4/HALF2/kWarpSize`。
- **块/线程/warp 组织**
  - Reduce：`constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;` +
    `__shared__ float shared[kNumWarps]`；两级 warp→smem→warp，最后 `__shfl_sync(0xffffffff, value, 0, 32)` broadcast。
  - `merge_attn_states`：`constexpr int kNumThreads = 128;`、
    `constexpr int kPackSize = 16 / sizeof(float); // 4 floats = 128-bit`、`using pack_t = uint4;`、
    `threads_per_head = head_size / kPackSize`；**Grid `((total_threads + 127) / 128, 1, 1)`，Block `(128, 1, 1)`**。
  - `online_safe_softmax_per_token`：Grid `(S, 1, 1)`，Block `(H, 1, 1)`，由外层 dispatch 选 H=32/64/128/256/512/1024。
  - `mat_transpose`：Grid `((col + 15) / 16, (row + 15) / 16, 1)`，Block `(16, 16, 1)`，每线程 1 元素。
  - `mat_transpose_padded`：Grid `((col + 15) / 16, (row + 63) / 64, 1)`，Block `(16, 16, 1)`，每线程 4 元素(float4)。
- **smem 布局与字节数**（原样，含 padding 值）
  - `block_reduce_*`：`__shared__ float shared[kNumWarps]`（无 padding）；
    `online_safe_softmax_per_token`：`__shared__ MD shared[kNumWarps]`（`MD` 为 `__align__(8)` 的 2×float）。
  - `mat_transpose_padded`：`constexpr int TILE = 16; constexpr int PAD = 1;`
    `__shared__ float tile[TILE * 4][TILE + PAD]; // 64x16`；
    注释原文：「BCF: smem 布局 `[kWarpSize_S*4][kWarpSize_S+PAD] = [64][17]`，PAD=1 加在第二维消除 bank conflict」
- **同步原语**：`__syncthreads()`（block reduce、transpose；merge 无同步）；
  `__shfl_xor_sync(0xffffffff, val, mask, kWarpWidth)`（蝶形归约，`kWarpWidth` 作为第 4 实参 segment width）；
  `__shfl_sync(0xffffffff, value, 0, 32)`（broadcast）。**没有 mbarrier / cp.async**。
- **测试与容差**：本文件无测试（测试在 `notes-v2.cu`）。相关调用 shape：
  `test_merge_attn_states(512, 16, 128)`、`test_softmax(256)`、`test_rms_norm(8, 128)`、`test_layer_norm(8, 128)`、
  `test_rope(8, 128)`、`test_mat_transpose(256, 256)`、`test_mat_transpose_padded(256, 256)`、
  `test_block_reduce(N)`、`test_dot(N)`、`test_relu(1024)`、`test_elementwise(1024)`、`test_histogram(1024)`
  （默认 M=N=K=1024，可用 argv[1..3] 覆盖）。
- **注释里的关键结论**（原样摘录）
  1. 「为什么不用 `__shfl_down_sync`？xor 模式所有线程做相同工作量，更均衡」
  2. 「block_reduce: 两级归约（warp → shared memory → warp0 broadcast），注意最后必须 broadcast 回所有线程（`__shfl_sync`），否则只有 warp0 知道结果」
  3. 「Tensor Cores：Hopper 每 SM 4 个；Blackwell 数量随型号/定义不同，建议以官方 ISV guide 为准」
  4. 「注：本文件仅实现 Level 1(naive) 与 Level 4(BCF+merge_write)，Level 2/3 省略」
  5. merge_attn_states 数学：`L_max = max(LSE_1, LSE_2)`；`w_i = exp(LSE_i - L_max)`；
     `alpha = w_1/(w_1+w_2), beta = w_2/(w_1+w_2)`；`O = alpha*O_1 + beta*O_2`；
     「inf LSE → -inf：空 attention 段（causal mask 等导致全部 score 为 -inf）的 LSE 可能为 +inf，替换后 exp(-inf - L_max) = 0，该段权重退化为 0」；
     布局：`LSE [num_heads, num_tokens]`（`lse[head_idx][token_idx]`），`Output [num_tokens, num_heads, head_size]`
     （token 维在最外，展平为 `[T0H0, T0H1, ..., T1H0, ...]`）。
  6. RoPE：`θ_i = 1 / (10000^(2i/d))`，`exp_v = 1.0f / powf(10000.0f, 2 * token_idx / (N * 2.0f))`。
  7. Roofline（原文）：GEMM 4096³ AI≈685 FLOPS/Byte → compute-bound（H100 ridge point：FP16 TC ≈ 295:1，FP32 ≈ 20:1）；
     GEMV AI≈0.5 → severely memory-bound；Softmax(N=4096) AI = 5/8 ≈ 0.625 → memory-bound。

---

## kernels/interview/flash_attn.cuh

- **文件定位**：Phase 8 全部 FlashAttention 实现。头部注释原文：
  `// flash_attn.cuh: Phase 8 FlashAttention 2/3 (MMA/TMA_WS/FA3/CuTe)`
- **导出的符号 + 模板参数全列表 + 完整签名**（三合一，逐条原样；**除注明者外均无默认值**）
  1. `template <typename T, int M, const int N, const int K = 2> __device__ inline void fill_3D_regs(T (&R)[M][N][K], T val)`
  2. `template <typename T, int M, const int N = 2> __device__ inline void fill_2D_regs(T (&R)[M][N], T val)`
  3. `template <int kHeadDim, int Br> __device__ __forceinline__ int swizzle_fa(int row, int col)`
  4. **FA2 手写 MMA Split-Q**（L93–110，**17 个模板参数，全无默认值**）：
     `template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK, const int kMmaAccF32, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK, const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ, const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV, const int kStagesK, const int kPadQ, const int kPadK, const int kPadV>`
     `__global__ void __launch_bounds__(kWarpSize * kMmaTileSeqLenQ * kMmaTileSeqLenK) flash_attn_mma_stages_split_q(half *Q, half *K, half *V, half *O, int N, int H)`
     （参数注释原文：`kStagesK, // pipeline stages for K: >= 1; NO stages required for Q/V`；
     `kPadQ, // Q row padding; 0 selects compact XOR swizzle`；kPadK/kPadV 同）
  5. **FA2 TMA+WS Split-Q**（L871–887，**16 个模板参数，全无默认值**）：
     `template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK, const int kMmaAccF32, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK, const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ, const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV, const int kStagesK, const int kStagesV, const int kNumThreads>`
     `__global__ void __launch_bounds__(kNumThreads, 1) flash_attn_tma_mma_ws_stages_split_q(half *Q, half *K, half *V, half *O, int N, int H, const CUtensorMap *__restrict__ tensorMapQ, const CUtensorMap *__restrict__ tensorMapK, const CUtensorMap *__restrict__ tensorMapV)`
     （注释：`kStagesV, // V pipeline depth (>=1; 1=single buffer, >=2=pipelined)`；`kNumThreads> // 384, 128 producer + 256 consumer`）
  6. **FA3 双 consumer WG**（L1505–1514，**15 个模板参数**）：与 #5 顺序完全相同但**去掉 kPadQ/kPadK/kPadV**，
     参数表逐字相同，名字换成 `flash_attn_3_tma_ws_stages_split_q`，`__launch_bounds__(kNumThreads, 1)`，入参同 #5。
  7. `namespace fa_cute`：`convert_layout_acc_rowcol<Layout>`、`convert_layout_acc_Aregs<TiledMma>(Layout)`、
     `convert_type<To>(Tensor)`、`gemm_ss<TensorC,TensorA,TensorB,...>(...)`、`gemm_rs<...>(...)`；
     `template <int kHeadDim> struct FlashAttn2CuTeTraits`、`template <int kHeadDim> struct FlashAttn3CuTeTraits`
  8. `template <int kHeadDim, int kStagesK = 2> __global__ void __launch_bounds__(256) flash_attn_mma_stages_split_q_cute(cutlass::half_t *Q, cutlass::half_t *K, cutlass::half_t *V, cutlass::half_t *output, int rows, int seqlen)`
  9. `template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV, int kStagesK = 1, int kStagesV = 1> __global__ void __launch_bounds__(384, 1) flash_attn_tma_mma_ws_split_q_cute(CUTLASS_GRID_CONSTANT TmaQ const tma_q, CUTLASS_GRID_CONSTANT TmaK const tma_k, CUTLASS_GRID_CONSTANT TmaV const tma_v, cutlass::half_t *output, int rows, int seqlen)`
  10. `template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV, int kStagesK = 1> __global__ void __launch_bounds__(384, 1) flash_attn_3_tma_mma_ws_split_q_cute(CUTLASS_GRID_CONSTANT TmaQ const tma_q, CUTLASS_GRID_CONSTANT TmaK const tma_k, CUTLASS_GRID_CONSTANT TmaV const tma_v, cutlass::half_t *output, int rows, int seqlen)`
  11. `template <int kHeadDim, typename TmaQ> __global__ void flash_attn_3_cute_tma_copy_smoke(...)`（L3449，**无 `__launch_bounds__`**）
  12. **（`6c86259` 新增，超出任务原始范围）** `template <typename Traits, typename TmaQ, typename TmaK, typename TmaV, typename TmaO> __global__ void __launch_bounds__(384, 1) flash_attn_cute_persist_d_sm120(CUTLASS_GRID_CONSTANT TmaQ const tma_q, CUTLASS_GRID_CONSTANT TmaK const tma_k, CUTLASS_GRID_CONSTANT TmaV const tma_v, CUTLASS_GRID_CONSTANT TmaO const tma_o, typename Traits::Element* __restrict__ O, int Nq, int Nkv, int Nh, int Nh_kv, float scale, int Tc, int causal, int q_tiles, int total_q_tiles, int total_q_rows, int total_kv_rows)`；
     注释原文：`// WS persist-D persistent kernel: 128T producer (TMA-only) + 256T consumer (MMA-only)。Q/K/V/O 均为 BHND packed -> flat (B*H*N, D) 行的 2D TMA。barrier 相位: K/V 用跨 q-tile 的全局 kv 计数 (stage=g%S, phase=(g/S)&1), 与非 persistent 版的相对序完全一致; q_full/epi_done 用 q-tile 迭代号。`
  13. 条件编译门：`NOTES_V2_ENABLE_TMA_MMA_WS`、`NOTES_V2_ENABLE_CUTE`
- **PTX 宏与工具宏**：本文件不自造 PTX 宏，全部调用 `common.cuh` 的
  `CP_ASYNC_CG` / `CP_ASYNC_COMMIT_GROUP` / `CP_ASYNC_WAIT_GROUP(n)` / `LDMATRIX_X4` / `LDMATRIX_X2` / `LDMATRIX_X2_T` /
  `HMMA16816` / `HMMA16816F32` / `tma_load_2d` / `tma_arrive_expect_tx` / `tma_fence_proxy_async_shared_cta` /
  `NOTES_V2_REG_DEALLOC(40)` / `NOTES_V2_REG_ALLOC(80|168)` / `swizzle<16>`（经 `swizzle_fa` 间接）。
- **块/线程/warp 组织**
  - **FA2 手写版**：`constexpr int kNumThreads = kWarpSize * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*8*1=256`
    （头注释另写 `kNumThreads=...=128`，且 `Block: (128, 1, 1)`、`Grid: ((N + 63) / 64, B * H, 1)，Br=64`
    ——**注释与代码不一致**，128 只对应 Br=64 那组实例化）。
    `constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ; // 16*8*1=128`、
    `constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK; // 8*1*8=64`。
    `warp_QP = warp_id`（各 warp 处理不同 Q 行片段），`warp_KV = 0`（所有 warp 共享 K）——**这就是 Split-Q**。
    测试端实际用 `dim3 grid((seqlen + Br - 1) / Br, B * H);`
  - **FA2 TMA+WS**：`static_assert(kNumThreads == 384, "128 producer + 256 consumer");`、
    `kConsumerThreads = 256; // 8 warps`、`kProducerThreads = 128; // 4 warps, only thread 0 issues TMA`；
    consumer 里 `warp_QP = warp_id (0~7)`、`warp_KV = 0`，每 warp 16 行 → Br = 16*8*1 = 128。
  - **FA3 双 consumer WG**：384 threads；`WG0 [0,127]: TMA producer (仅 thread 0 发 TMA)`、
    `WG1 [128,255]: consumer_id=0, 处理偶数 KV tile (0,2,4,...)`、`WG2 [256,383]: consumer_id=1, 处理奇数 KV tile (1,3,5,...)`；
    `kConsumerThreadsPerWG = 128; kProducerThreads = 128; kNumConsumerWGs = 2;`，Br=Bc=64。
  - **CuTe FA2**：`kBr = 128; kBc = 64;`，`__launch_bounds__(256)`，
    `TiledMma = TiledMMA<MmaAtom, Layout<Shape<_8,_1,_1>>, Tile<_128,_16,_16>>`（8 warps，单 WG）。
  - **CuTe FA3**：`kTile = 64; kNumConsumers = 2; kConsumerThreads = 128; kProducerThreads = 128;`，
    `TiledMma = TiledMMA<MmaAtom, Layout<Shape<_4,_1,_1>>, Tile<_64,_16,_16>>`。
  - **CuTe FA2 TMA WS**：`kBr = 128; kBc = 64; kConsumerThreads = 256; kProducerThreads = 128;`。
- **smem 布局与字节数**
  - FA2 手写版（**padding 与 XOR 二选一，绝不叠加**）：`Q_tile_size = Br * (kHeadDim + kPadQ);`
    `K_tile_size = Bc * (kHeadDim + kPadK);`、`kSmemStrideQ/K/V = kHeadDim + kPadQ/K/V;`；
    指针：`Q_tile_smem = smem; K_tile_smem = Q_tile_smem + Q_tile_size; V_tile_smem = K_tile_smem + kStagesK * K_tile_size;`
    注释原文：「Q/K/V independently use padded row-major when `kPad* > 0` and compact XOR swizzle when `kPad* == 0`.
    Padding and XOR are never combined per operand. The swizzled physical layout is `[col / 16][row][16]`;
    `swizzle<16>()` selects the 0/8 phase inside the final 16-half tile.」
    **默认 kPad = 8**（`notes-v2.cu` bench dispatch 用 `<64, 2, 8, 8, 8, acc>`；`--bench-fa` 注释亦写 `kPadQ=kPadK=kPadV=8 only`）。
  - FA2 TMA+WS：`kQTileBytes = Br * kHeadDim * sizeof(half); // 16 KB`、
    `kKTileBytes = Bc * kHeadDim * sizeof(half); // 8 KB`、`kVTileBytes = Bc * kHeadDim * sizeof(half); // 8 KB`、
    `kTmaBoxMinor = 64;`、`kTmaChunks = kHeadDim / kTmaBoxMinor;`、
    `extern __shared__ __align__(1024) uint8_t smem_fa_tma_ws[];`（Q 在前、K 中、V 后）。
    头注释 smem 总量（原文）：`smem = (Q[Br,D] + K[kStagesK,Bc,D] + V[kStagesV,Bc,D]) * sizeof(half)
    = D * (Br + kStagesK*Bc + kStagesV*Bc) * 2 = D * (128 + (kStagesK+kStagesV)*64) * 2`（Br=128, Bc=64）；
    表格 `| Sk | Sv | D=64 | D=128 |` → `1,1: 32KB/64KB`；`2,1: 40KB/80KB`；`2,2: 48KB/96KB`；
    `3,1: 48KB/96KB`；`3,2: 56KB/112KB`；`4,1: 56KB/112KB`。
  - FA3：`smem_fa3_tma_ws`，布局注释原文：`[Q_shared: Br*D]` /
    `[K[cid=0][s=0..Sk-1]: Sk*Bc*D] [K[cid=1][s=0..Sk-1]]` / `[V[cid=0]: Bc*D] [V[cid=1]: Bc*D]`；
    地址 `K_smem_base + cid * (kStagesK * Bc * kHeadDim) + stg * (Bc * kHeadDim)`、`V_smem_base + cid * (Bc * kHeadDim)`。
  - CuTe 版：`extern __shared__ __align__(1024) Element shm[];`（CuTe FA2：Q[128,D] + K[kStagesK,64,D] + V[1,64,D]）。
- **同步原语**
  - FA2 手写版：`CP_ASYNC_COMMIT_GROUP()` / `CP_ASYNC_WAIT_GROUP(0|1)` / `CP_ASYNC_WAIT_GROUP(kStagesK - 2)` / `__syncthreads()`。
    同步链注释（CuTe 版复述手写版）：`1. Q load: copy + fence + wait<0> + sync`；
    `2. PREFETCH K[0..Sk-2]: copy + fence，然后 wait<Sk-2> + sync`；
    每轮 `3b QK 前 wait`、`3d PV 前 wait`、`循环末尾: kStagesK>1 且非最后 -> wait<0> + sync`。
  - FA2/FA3 TMA+WS：`cuda::barrier<cuda::thread_scope_block>` 的 `init/arrive/wait` + `tma_load_2d`（`cp.async.bulk.tensor.2d`）+ `tma_arrive_expect_tx` + `tma_fence_proxy_async_shared_cta()` + `__syncthreads()`（只用于 barrier 初始化后）。两处都用 `#pragma nv_diag_suppress static_var_with_dynamic_init` 包住 `__shared__ cuda::barrier`。**arrive_count 协议（原文）**：`arrive_count = kConsumerThreads + 1 = 257 (256 consumer arrives + 1 producer arrive_tx)`；FA3：`init(&full_Q, 256 + 1);  // 2 consumer WGs * 128 + 1 producer`，per-WG 的 K/V barrier 为 `kConsumerThreadsPerWG + 1 = 129`。barrier 集合：FA2 = `full_Q`, `full_K[kStagesK]`, `empty_K[kStagesK]`, `full_V[kStagesV]`, `empty_V[kStagesV]`；FA3 = `full_Q`, `full_K[2][kStagesK]`, `empty_K[2][kStagesK]`, `full_V[2]`, `empty_V[2]`。
  - 寄存器再平衡：producer 侧 `NOTES_V2_REG_DEALLOC(40);`；FA2 consumer 侧（原文注释：`Consumer register budget per Triton flash_attn_v2 maxnreg strategy on Blackwell warp_specialize: D=128 -> 168, otherwise -> 80.`）`if constexpr (kHeadDim == 128) { NOTES_V2_REG_ALLOC(168); } else { NOTES_V2_REG_ALLOC(80); }`
  - **K/V 早释放**（原文）：「Release K[stage] for producer reuse as early as possible: QK^T GEMM has consumed all K smem data; the subsequent softmax (3c) and PV GEMM (3d) only touch registers (R_S, R_O) and V smem, so K[stage] is free to be overwritten by the producer's next TMA prefetch.」V 同理：`empty_V[stage_v].arrive()` 放在 PV 之后、rescale 之前。
  - **FA3 split-KV 合并**（原文公式）：`alpha = exp(m_0 - m) (<=1)`、`beta = exp(m_1 - m) (<=1)`、`l = alpha*l_0 + beta*l_1`、`Oacc = alpha*Oacc_0 + beta*Oacc_1`、`O = Oacc / l`；`Tc==1 退化: WG2 无 tile, m_1=-inf, l_1=0, Oacc_1=0; beta = exp(-inf) = 0, 退化为 WG1 结果。`
- **测试与容差**
  - **tol 的真实情况（务必按此写，不要编第三档）**：本仓库 FA 路径**只有一个统一的失败阈值** `bool is_fail = max_err >= 5e-1f;`（`notes-v2.cu` @`6c86259`：L3446、L3561、L3710、L3867、L4029、L4177、L4267；其中 L3561/L4029/L4177/L4267 带 `checked &&` 前缀）。**不存在 TF32 容差档**；全仓库 `.cu` 中没有 `1e-2f` / `1e-3f` 之类的 tol 常量（grep 无命中）。
  - **“三档容差”的真实形态是 README 报告的经验 Max Err**（报告值 ≠ 判据，两者相差 3–4 个数量级）：`1.831e-04`（全部 F16Acc 变体）、`1.526e-05`（全部 F32Acc 变体 + Split-D）、`9.155e-05`（仅 `FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F16Acc)`）、`0.000e+00`（HGEMM CuTe Swizzle）。**TF32 不存在于 `interview/`**。
  - **参考实现**：CPU FP32 参考（`srand(42)`，输入 `__float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f)`，`float scale = 1.0f / sqrtf((float)head_dim);`，`double sum_exp += (double)expf(S[kj] - smax);`，输出亦用 `double o_acc`）+ cuDNN SDPA（cudnn-frontend：`graph->sdpa(Q,K,V, SDPA_attributes().set_name("sdpa_ref").set_attn_scale(1.0f / sqrtf((float)head_dim)))`、`set_io_data_type(HALF).set_intermediate_data_type(FLOAT).set_compute_data_type(compute_type)`、`graph->build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})`）。
  - **shape 列表**（`notes-v2.cu` 主流程）：`test_flash_attn(1024, 64)`；`test_flash_attn_tma_mma_ws(1024, 64)` 与 `(1024, 128)`；`test_flash_attn_3_tma_ws(1024, 64)` 与 `(1024, 128)`（内部对 `<64>`/`<128>` 都跑 S=1..4）。bench 默认 `g_bench_B = 1, g_bench_H = 32, g_bench_Nfa = 8192, g_bench_D = 128;`（`--bhnd` 覆盖）；README 基线 shape `1,32,16384,128` 与 `1,32,16384,320`（Split-D）。前置条件（原文）：`if (seqlen < Br || seqlen % Br != 0 || seqlen % Bc != 0)` → 打印 `SKIP`。
  - **smem 可行性兜底**：`check_smem_feasible()` 比较 `cudaDevAttrMaxSharedMemoryPerBlockOptin` 与 `dyn_smem_bytes + attrs.sharedSizeBytes`，失败时打印 `SMEM too large` 行并跳过（不报错）。计时 `g_warmup = 2, g_repeat = 3`。
- **注释里的关键结论**（原样摘录——5 条最重要）
  1. **作者明确说不要泛化（P 写回 R_S）**：「为什么 R_S 可以直接用作 P@V 的 A 矩阵？… 当前实现依赖 m16n8k16
     这一路径下约定好的 fragment 布局，使 softmax 后的 P 可以继续留在 R_S 中供后面的 P@V 直接消费。
     这是此实现的寄存器布局复用技巧，**不要背成“所有 MMA A/C fragment 都天然同构”的通用结论**」；
     3d 处再次强调：「…复习时不要把它背成对所有 MMA fragment 都无条件成立的通用结论。」
  2. **XOR swizzle 的实测反例（SM120, B=1,H=32,N=4096,D=64）**：「compact Q/K/V XOR 改变了 cp.async 的 shared
     destination pattern：LDGSTS wavefronts 从 pad 的 30.15M 增至 68.16M（2.26x），long-scoreboard、LG-throttle、
     MIO-throttle 分别约为 pad 的 3.25x、2.74x、1.60x。compact XOR 虽节省约 5 KiB smem，但没有提高此 kernel 的
     occupancy；最终约 120.0 TFLOPS，显著低于 Q/K/V kPad=8 的 166.6 TFLOPS。所以当前 kernel 默认对 Q/K/V 都用 kPad=8。
     保留 swizzle 路径是为了学习和消融：评价 shared layout 必须同时观察 ldmatrix reads 与 cp.async/LDGSTS writes，
     不能只看 bank-conflict counter，也不能只优化 XOR 地址算术。」
  3. **注释与代码不一致（尾 tile）**：「原始实现默认 seqlen 与 Bc 对齐；最后一个不完整 tile 需要额外 pad/边界处理。
     这里保留 ceil 写法是为了说明 tile 划分方式，**不等于当前实现已经完整处理了尾 tile**。」
     同一处 v1 限制原文：`v1 限制：kHeadDim=64 or 128；seqlen % Br == 0 且 seqlen % Bc == 0（不处理尾 tile）`；
     FA3 限制原文：`kStagesV == 1`、`D=128 时 Sk=1 (Sk=2 需 112KB > 101KB optin 上限)`、
     `Br=64, Bc=64, D=64/128, aligned seqlen`、**`仅 self-attention, 无 causal/varlen/GQA`**。
     另有线程划分警告：「WARN: Must use kProducerThreads (not kConsumerThreads) as the divisor, otherwise the split
     is wrong: 384/256=1.5 would give Producer 256 threads and Consumer only 128, but barriers expect 256 consumer
     arrives → deadlock.」
  4. **TMA 128B 的硬约束与 chunk-major 方案**：「CU_TENSOR_MAP_SWIZZLE_128B 硬件要求 box innermost dim ≤ 128B = 64 half。
     因此 D=128 不能用单个 TMA box=(128, Br) 覆盖整行。方案：D=128 时 box 固定为 (64, Br)，沿 head_dim 方向连续发
     kTmaChunks 次 TMA（minor_coord = c*64），写入 chunk-major smem 布局 [kTmaChunks, Br, 64]」；
     以及「TMA CU_TENSOR_MAP_SWIZZLE_128B requires 1024B-aligned smem base so the hardware swizzle phase starts at zero.
     Consumer swizzle<64>() assumes zero phase (no base_offset compensation like WGMMA descriptor).」
  5. **V ldmatrix.x2.trans 正确性（原文）**：「TMA 128B SWIZZLE 是 1-1 映射，ldmatrix 用 swizzle<64>(row,col) 计算的物理地址
     = TMA 写入的物理地址（smem 1024B 对齐保证 phase=0）。ldmatrix.x2.trans 的转置语义在寄存器层面工作，与 smem
     物理布局无关 → 必然正确。」
     （另：`swizzle_fa` 有 `static_assert(kHeadDim == 64 || kHeadDim == 128, "D=64 or 128 only");`；
     `D=64 退化：kTmaChunks=1, chunk=0, offset = row*64 + swizzle<64>(row, col) 与原公式完全一致 → 无回归。`；
     FA2 TMA WS 与 HGEMM 对比：「HGEMM 只做一次 GEMM，K 维迭代；FA 做 QK^T 和 PV 两次 GEMM，KV seqlen 迭代」、
     「HGEMM 的 A/B 都 staged；FA 中 Q 只 load 一次（split-Q），K staged，V 单 buffer」。）

---

## kernels/interview/ffpa_attn.cuh

- **文件定位**：large head-dim（D>128）的 Split-D attention（按 64-wide D chunk 切分 head_dim）。
  `#include "flash_attn.cuh"` 复用 `fa_cute` 命名空间。
- **导出的符号 = 模板参数全列表 = 完整签名**（逐条原样）
  - `template <int kHeadDim, int TILE_M = 64, int TILE_N = 64> struct FFPAAttnSplitDCuTeTraits`（`namespace fa_cute` 内；`static_assert(kHeadDim % 64 == 0, "Split-D requires head-dim multiple of 64");`、`static_assert(TILE_M == 64 && TILE_N == 64, "Current impl supports 64x64 only");`）
  - `template <int kHeadDim, int kStagesQK = 2, int kStagesV = 2> __global__ void __launch_bounds__(128) ffpa_split_d_cute(cutlass::half_t *Q, cutlass::half_t *K, cutlass::half_t *V, cutlass::half_t *output, int rows, int seqlen)`（cp.async 版，无 TMA/WS，**128 threads**）
  - `template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV, int kStagesQK = 2, int kStagesV = 2> __global__ void __launch_bounds__(256, 1) ffpa_attn_tma_mma_ws_split_d_cute(CUTLASS_GRID_CONSTANT TmaQ const tma_q, CUTLASS_GRID_CONSTANT TmaK const tma_k, CUTLASS_GRID_CONSTANT TmaV const tma_v, cutlass::half_t *output, int rows, int seqlen)`
  - 门控：cp.async 版 `#if defined(NOTES_V2_ENABLE_CUTE)`；TMA WS 版 `#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)`
- **PTX 宏与工具宏**：不自造 PTX 宏；用 CuTe `SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>` / `SM75_U32x4_LDSM_N` / `SM75_U16x8_LDSM_T` / `MMA_Atom<SM80_16x8x16_F32F16F16F16F32_TN>`，以及 `NOTES_V2_REG_DEALLOC(40)` / `NOTES_V2_REG_ALLOC(232)`。TMA 侧用 `cutlass::arch::ClusterTransactionBarrier` / `cutlass::arch::ClusterBarrier` 的 `init / wait(&bar, phase) / arrive_and_expect_tx(&bar, bytes)`。
- **块/线程/warp 组织**：Traits 注释原文 `QK: Tile<64,64,16> + Layout<4,1,1> → EURepeat<1,8,1>`（「一次 TiledMMA 覆盖完整 S[64,64]，省掉 N-tile 循环」）、`PV: Tile<64,16,16> + Layout<4,1,1> → EURepeat<1,2,1>`（「保持小 tile 控制 acc_O 寄存器」）；TMA WS 版 `kProducerThreads = 128; kConsumerThreads = 128;` = **256 total threads (vs 原始 384)**；`kBr = 64; kBc = 64; kDChunk = 64; kDChunks = kHeadDim / kDChunk;`；`q_tile = blockIdx.y * (seqlen / kBr) + blockIdx.x; kv_tiles = seqlen / kBc;`
- **smem 布局与字节数**：`extern __shared__ __align__(1024) Element shm[];`；`q_base = shm; k_base = q_base + kStagesQK * kQChunkElements; v_base = k_base + kStagesQK * kKVChunkElements;`；布局注释原文 `SMEM: sQ[kStagesQK,64,64] + sK[kStagesQK,64,64] + sV[kStagesV,64,64]`、`stage 偏移通过基地址指针算术管理，不使用 stride-0 的 stage-mode layout。`；无显式字节数常量（用 `cosize(SmemLayoutQ{})` / `cosize(SmemLayoutKV{})`；`SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>`，Q/KV 均为 `[64, 64]` tile）。
- **同步原语**：cp.async 版 `cp_async_fence()` / `cp_async_wait<kStagesQK - 2>()` / `cp_async_wait<kStagesV - 2>()` / `cp_async_wait<0>()` / `__syncthreads()`；TMA WS 版 `__shared__ uint64_t qk_full[kStagesQK]; qk_empty[kStagesQK]; v_full[kStagesV]; v_empty[kStagesV];`，`TmaBarrier::init(&qk_full[stage], 1); CtaBarrier::init(&qk_empty[stage], kConsumerThreads);`（**TmaBarrier 的 arrive_count 是 1，不是 N+1**），`const int phase = (chunk_index / kStagesQK) & 1;`、`CtaBarrier::wait(&qk_empty[stage], phase);`、`TmaBarrier::arrive_and_expect_tx(&qk_full[stage], sizeof(Element) * (size(sQ) + size(sK)));`
- **测试与容差**：本文件无测试。bench 在 `notes-v2.cu`（`bench_fa_split_d_launch/dispatch` 用 `<D,1,1>` 与 `<D,2,2>`）与 `bench/bench_ffpa.cu`。头注释性能原文 `(B=1,H=32,N=8192,D=512, SM120a RTX PRO 5000): cuDNN SDPA: 57.2 TFLOPS | FFPA TMA WS: 110.7 TFLOPS (1.94x)`。
- **注释里的关键结论**（原样摘录）：①「通过 include flash_attn.cuh 复用 fa_cute namespace 中的 FA traits 和 helpers。仅定义 FFPA 特有的 FFPAAttnSplitDCuTeTraits 和 ffpa_attn_tma_mma_ws_split_d_cute kernel。支持 head_dim > 128 的 large head-dim attention，通过 64-wide Split-D chunks 处理。」②「消费者逻辑与 ffpa_attn_tma_mma_ws_split_d_cute 完全一致（双 TiledMma，相同的 fragment 流转：QK->convert_layout_acc_Aregs<TiledMmaPV>->PV）。唯一区别：生产者从 TMA 换成 cp.async，用 cp_async_fence/wait 替代 TMA barrier。」③「128 线程：与 TiledMmaQK/PV 的 Layout<4,1,1> 一致，每个线程在 G2S 和 S2R/MMA 中有唯一分区，消除 256-thread G2S 与 128-thread MMA 之间的映射不匹配。」④「128 producer + 128 consumer = 256 total threads (vs 原始 384)」

---

## kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu

- **文件定位**：独立的最朴素 CuTe warp-specialization HGEMM 教学样例（PyTorch 扩展），
  用 `cuda::pipeline` 而不是 mbarrier 做 producer/consumer 同步。
- **导出的符号 = 模板参数全列表 = 完整签名**（逐条原样）
  - `template <class CTATile, int ProducerThread, int Stage> struct WSHGEMMTraits`（含 `struct Arguments`；**无默认值**）
  - `template <typename WSHGEMMTraits> __global__ void ws_hgemm_naive_cute_kernel(typename WSHGEMMTraits::Arguments args)`
    （**无默认值**，前有 `#pragma nv_diag_suppress static_var_with_dynamic_init`）
  - host：`void ws_hgemm_naive_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c)`；
    `inline int get_max_smem_size()`；`template <typename Kernel> void config_smem(Kernel kernel, int smem_size)`
  - 宏：`DEVICE` → `__device__ __forceinline__`；`STRINGFY(str)` → `#str`；
    `TORCH_BINDING_COMMON_EXTENSION(func)` → `m.def(STRINGFY(func), &func, STRINGFY(func));`；
    `CHECK_TORCH_TENSOR_DTYPE(T, th_type)`、`CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)`
  - Traits 内静态函数模板：`producer<Pipeline,AEngine,ALayout,BEngine,BLayout>(void *smem_ptr, Pipeline&, Tensor<AEngine,ALayout> const&, Tensor<BEngine,BLayout> const&)`、
    `main_loop<Pipeline,CEngine,CLayout>(Arguments const&, void*, Pipeline&, Tensor<CEngine,CLayout> const&)`、
    `epilog<AccEngine,AccLayout,CEngine,CLayout>(Arguments const&, void*, Tensor<AccEngine,AccLayout> const&, Tensor<CEngine,CLayout>&)`、
    `consumer<Pipeline,CEngine,CLayout>(Arguments const&, void*, Pipeline&, Tensor<CEngine,CLayout>&)`
  - Traits 常量：`kMmaThrLayoutM = 2; kMmaThrLayoutN = 2; kMmaThrLayoutK = 1;`、`kSwizzleB = 3; kSwizzleM = 3; kSwizzleS = 3;`、
    `kSmemStageAcc = 2;`、`kConsumerThread = size(TiledMMA{}); kProducerThread = ProducerThread; kAllThread = kProducerThread + kConsumerThread;`
  - 实例化点（原文）：`using GEMM_Traits = WSHGEMMTraits<decltype(make_shape(_128{}, _256{}, _32{})), 32, 3>;`
    → CTATile = 128×256×32，ProducerThread = 32，Stage = 3
- **PTX 宏与工具宏**：无自有 PTX 宏。CuTe copy op（原文）：`using mma_op = SM80_16x8x16_F16F16F16F16_TN;`
  （**F16 累加**，`AccType = half`）、`g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>`、
  `s2r_copy_op = SM75_U32x4_LDSM_N`（`S2RCopyA = S2RCopyB = s2r_copy_atom`）、
  `R2SCopyC = Copy_Atom<UniversalCopy<int>, AccType>`、`S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, AccType>`；
  Smem atom：`SmemLayoutAtom = composition(Swizzle<3,3,3>{}, make_layout(make_shape(Int<8>{}, Int<kCTAK>{}), make_stride(Int<kCTAK>{}, Int<1>{})))`
- **块/线程/warp 组织**
  - `constexpr static int kConsumerThread = size(TiledMMA{});`；
    `static_assert(ProducerThread % 32 == 0, "The number of ProducerThreads must be a multiple of 32");`（注释 `// To avoid warp divergence`）；
    `constexpr static int kProducerThread = ProducerThread; kAllThread = kProducerThread + kConsumerThread;`
    对 `<..., 32, 3>`：Producer 32 + Consumer（`size(TiledMMA{})` = 2×2×32 = 128）= **160 线程**。
  - `TiledMMA = make_tiled_mma(mma_atom{}, MmaThrLayout{2,2,1}, MmaPermutation{})`，
    `kMmaPermuteM = 2*16 = 32`、`kMmaPermuteN = 2*2*8 = 32`、`kMmaPermuteK = 1*16 = 16`；注释
    `// The expanded TiledMMA can process matrices of size 32x32x16 in a single operation.`
  - 线程角色：`const auto thread_role = tidx < WSHGEMMTraits::kProducerThread ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;`
  - 启动：Grid = `args.get_grid()` → `dim3(ceil_div(M, kCTAM), ceil_div(N, kCTAN))`；Block = `dim3 block(block_size)`，
    `block_size = GEMM_Traits::kAllThread`。
- **smem 布局与字节数**：`constexpr static int kSmemSizeA = cosize(SmemLayoutA{}); kSmemSizeB = cosize(SmemLayoutB{});`
  `kSmemAllocateAB = (kSmemSizeA + kSmemSizeB) * sizeof(MatrixTypeAB);`
  `kSmemSizeAcc = cosize(SmemLayoutAcc{}); kSmemAllocateAcc = kSmemSizeAcc * sizeof(AccType);`
  `kAllSmemAllocate = cute::max(kSmemAllocateAB, kSmemAllocateAcc);`（**AB 与 Acc 复用同一块 smem**）；
  `SmemLayoutA = tile_to_shape(atom, (kCTAM, kCTAK, kStage))`、`SmemLayoutB = tile_to_shape(atom, (kCTAN, kCTAK, kStage))`；
  `extern __shared__ MatrixTypeAB smem_ptr[];`。**无 padding 值**（用 `Swizzle<3,3,3>` 而非 PAD）；
  `config_smem()` 仅在 `smem_size >= 32 * 1024` 时调 `cudaFuncSetAttribute`。
- **同步原语**：`cooperative_groups::this_thread_block()`；
  `__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, kStage> shared_state;` +
  `auto pipeline = cuda::make_pipeline(block, &shared_state, thread_role);`；
  `pipeline.producer_acquire()` / `producer_commit()`（producer 侧）、`pipeline.consumer_wait()` / `consumer_release()`（consumer 侧）；
  `__syncthreads()` ×3（epilog 前、r2s 后、s2g 后）。**主循环里没有 `__syncthreads`；没有 mbarrier / TMA**。
- **测试与容差**：本文件**无测试、无 tol、无参考实现**——只提供 PyTorch 绑定 `ws_hgemm_naive_cute(a, b, c)`，
  输入校验为 `CHECK_TORCH_TENSOR_DTYPE(..., torch::kHalf)` 与 shape `a:(M,K) b:(K,N) c:(M,N)`。
- **注释里的关键结论**（原样摘录）：`// The expanded TiledMMA can process matrices of size 32x32x16 in a single operation.`；
  `// To avoid warp divergence`（配 `static_assert(ProducerThread % 32 == 0, ...)`）；`// Different thread_roles execute different branches.`；
  `__syncthreads(); // wait all consumer thread finish main_loop` / `... finish r2s` / `... finish s2g`（共 3 处）

---

## kernels/interview/build.sh

- **文件定位**：`notes-v2.cu` 的编译脚本——**两步编译+链接**（compile 走 ccache，link 不走），每个 arch 一套 gencode/宏/库/输出名。
- **导出的符号**：无（bash）。关键变量：`SCRIPT_DIR`、`USE_CCACHE`、`NVCC="/usr/local/cuda/bin/nvcc"`、
  `COMMON_FLAGS`（数组）、`ARCH_GENCODE/ARCH_DEFINES/ARCH_LIB_PATH/ARCH_LIBS/ARCH_OUTPUT`（5 个 `declare -A`）、
  `VALID_ARCHS="sm_86 sm_89 sm_90a sm_120a sm_120f"`（**当前 HEAD 是 5 个 arch，多了 `sm_120f`**）、
  `ARCH`、`CLEAN_ONLY`、函数 `usage()`、`build_one()`。
- **模板参数全列表**：不适用。
- **PTX 宏与工具宏**：无 PTX。**arch 是怎么给的（原文逐行 @ `6c86259`；GENCODE 与 DEFINES 一一对应）**：
  | arch | `ARCH_GENCODE` | `ARCH_DEFINES` | `ARCH_OUTPUT` |
  |---|---|---|---|
  | sm_86 | `-gencode arch=compute_86,code=sm_86` | `-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_CUDNN` | `notes_v2_sm86.bin` |
  | sm_89 | `-gencode arch=compute_89,code=sm_89` | `-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_CUDNN` | `notes_v2_cute_sm89.bin` |
  | sm_90a | `-gencode arch=compute_90a,code=sm_90a` | `-DNOTES_V2_ENABLE_WGMMA -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN` | `notes_v2_sm90a.bin` |
  | sm_120a | `-gencode arch=compute_120a,code=sm_120a` | `-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN` | `notes_v2_sm120a.bin` |
  | sm_120f | `-gencode arch=compute_120f,code=sm_120f` | 同 sm_120a **再加** `-DNOTES_V2_ENABLE_SETMAXNREGS -DNOTES_V2_FORCE_INLINE_ASYNC_PROXY` | `notes_v2_sm120f.bin` |
  **重要**：只有 `sm_90a`/`sm_120a`/`sm_120f` 带 `NOTES_V2_ENABLE_TMA_MMA_WS`；
  `common.cuh` 里 TMA/mbarrier helper 的门控是 `#if defined(NOTES_V2_ENABLE_WGMMA) || defined(NOTES_V2_ENABLE_TMA_MMA_WS)`。
  **`sm_120f` 是唯一打开 `NOTES_V2_ENABLE_SETMAXNREGS` + `NOTES_V2_FORCE_INLINE_ASYNC_PROXY` 的 arch**
  （原因见 `common.cuh` L469–488 与 L353–364 注释：ptxas 在 sm_120a 上会因 TMA 用法触发 C7506 丢掉 setmaxnreg）。
  **COMMON_FLAGS（原文，跨 arch 共用）**：`-std=c++20`、`-O3`、`--expt-relaxed-constexpr`、`--use_fast_math`、
  `-I ../../third-party/cutlass/include`、`-I ../../third-party/cudnn-frontend/include`（注意 **`--use_fast_math` 是开的**）。
  库与 stub 路径（5 个 arch 完全相同）：`-L/usr/local/cuda/targets/x86_64-linux/lib/stubs`、`-lcublas -lcudnn -lnvrtc -lcuda`。
  ccache 环境（原文）：`CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"`、
  `CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-include_file_mtime,time_macros,locale,pch_defines}"`、`CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"`。
- **kernel 签名 / 块线程组织 / smem / 同步原语 / 测试容差**：均不适用（脚本层）。脚本本身 `set -euo pipefail`。
- **关键结论**（原样摘录）：两步流程 `1. ccache nvcc ... -c notes-v2.cu -o notes-v2.o   (cached)` /
  `2. nvcc notes-v2.o -o notes_v2_<arch>.bin ...  (uncached link)`；
  `--arch all` 按 `VALID_ARCHS` 顺序构建，`--clean` 删 `notes-v2.o` 与各 `.bin`；
  头注释 arch 说明原文 `./build.sh --arch sm_90a      # Hopper (H100/H200)`、
  `./build.sh --arch sm_120a     # Blackwell (RTX 5090 / PRO 5000/6000)`。
  4. ccache 参考来源：`(ref: ffpa-attn/tools/build_fast.sh)`。

---

## kernels/interview/README.md

- **文件定位**：`interview/` 章节门面文档（@ `6c86259` **只剩 55 行**：快速开始 + 两张基准表；
  旧版的「文件结构」表与 arch 列表已被删除）。
- **导出的符号 / 模板参数 / 宏 / kernel 签名 / 块线程组织 / smem / 同步原语**：均不适用（Markdown）。
- **测试与容差 / 基线（原文照抄 @ `6c86259`）**
  - 依赖关系（来自 `notes-v2.cu` 的 include 顺序，README 已不再写）：
    `common.cuh ← base.cuh ← sgemv.cuh / sgemm.cuh / hgemm.cuh / flash_attn.cuh ← notes-v2.cu`（`ffpa_attn.cuh` 在 `flash_attn.cuh` 之后）
  - 编译（原文仅 2 条）：`./build.sh --arch sm_120a   # Blackwell (RTX 5090 / PRO 5000/6000, CUDA Toolkit >= 13.2)`；
    `./build.sh --help           # Show help for build options`
  - 环境准备原文：`apt remove -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13`；
    `apt install -y cudnn9-cuda-13 ccache # Also install ccache for faster rebuilds`
  - 复现命令：`./notes_v2_sm120a.bin --bench --mnk 4096,4096,4096 --bhnd 1,32,16384,128 # MMA ACC F16/F32 Acc`
    （运行环境描述 `e.g., NVIDIA PRO 5000, Blackwell SM_120a`）
  - **Max Err 列只有 4 个不同值（这就是“容差档”的真实形态）**：
    `0.000e+00` = 全部 `HGEMM CuTe Swizzle (S=2/3, BLK_SW=0/1, F16Acc 与 F32Acc)`；
    `9.155e-05` = 仅 `FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F16Acc)`；
    `1.831e-04` = F16Acc 组（`FA2 MMA Stages (Sk=1/2, Pad, F16Acc)`、`FA2 TMA MMA WS (1 Consumer WG)` 的 (Sk=1,Sv=1)/(Sk=2,Sv=1)/(Sk=2,Sv=2)）；
    `1.526e-05` = F32Acc 组 + Split-D（`FA2 MMA Stages (Sk=1/2, Pad, F32Acc)`、`FA2 CuTe MMA Stages (Sk=1/2)`、
    `FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc)`、`FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F32Acc)`、
    `FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=2/3, Sv=1, F32Acc)`、`FA2 CuTe TMA MMA Persistent-CTA WS (D=128)`、
    `FA Split-D CuTe TMA MMA WS (D=320, Sk=1,Sv=1)` 与 `(D=320, Sk=2,Sv=2)`）。
    → **规律：F16Acc → 1.831e-04（FA3 的 F16Acc 因 split-KV 合并反而更好，9.155e-05）；F32Acc → 1.526e-05；HGEMM → 0。**
  - TFLOPS 基线**已更新**（旧表数值不可再用）：例 `FA3 TMA MMA WS (2 Consumer WG) (Sk=1,Sv=1,F16Acc)`
    `305.2/222.9 (1.37x)` → **`210.1/232.4 (0.90x)`**；新增行
    `FA2 CuTe TMA MMA Persistent-CTA WS (D=128) | 1.526e-05 | 242.5/232.4 (1.04x)`；
    Split-D 段 `# Speedup: Split-D for large headdim (e.g, D=320) ~2.06x faster than cuDNN SDPA (with F32 Acc)`
    （旧值 ~2.20x），`(D=320, Sk=1,Sv=1) | 1.526e-05 | 96.5/70.3 (1.37x)`、`(D=320, Sk=2,Sv=2) | 1.526e-05 | 145.1/70.3 (2.06x)`。
    **引用绝对 TFLOPS 请注明版本；Max Err 列才是稳定事实。**
- **注释里的关键结论**：快速开始段唯一强调点 `# Install the latest CUDNN library for benchmarks (remove the old version first)`；
  运行环境描述从 “NVIDIA RTX 5090” 改成了 “NVIDIA PRO 5000”。

---

## kernels/interview/notes-v2.cu

- **文件定位**：整个章节的**唯一入口 TU**——`#include "base.cuh" / "sgemv.cuh" / "sgemm.cuh" / "hgemm.cuh" /
  "flash_attn.cuh" / "ffpa_attn.cuh"`，提供所有 test/bench 函数与 CLI 解析；**本身不定义任何 kernel**。
- **导出的符号（用 grep 精确统计，非通读）**
  - **`__global__` 命中数 = 0**（所有 kernel 都在被 include 的 `.cuh` 里）。
  - test 函数（`static`，逐条）：`test_flash_attn_3_cute_tma_copy_smoke`、`test_flash_attn_3_tma_mma_ws_split_q_cute`、
    `test_flash_attn_mma_stages_split_q_cute`、`test_flash_attn_tma_mma_ws_split_q_cute`、
    **（新增）** `test_flash_attn_cute_persist_d_sm120`（L519）、`test_block_reduce`、`test_dot`、`test_relu`、
    `test_elementwise`、`test_histogram`、`test_merge_attn_states`、`test_softmax`、`test_rms_norm`、`test_layer_norm`、
    `test_rope`、`test_mat_transpose`、`test_mat_transpose_padded`、`test_sgemv`、`test_sgemm`、`test_hgemm_mma`、
    `test_hgemm_swizzle`、`test_hgemm_cute`、`test_hgemm_wgmma`、`test_hgemm_tma_mma_ws`、`test_flash_attn`、
    `test_flash_attn_tma_mma_ws_impl`、`test_flash_attn_tma_mma_ws`、`test_flash_attn_3_tma_ws_impl`、
    `test_flash_attn_3_tma_ws`、`test_swizzle_equiv`
  - bench 函数（`static`，`launch/dispatch` 成对者合并书写）：`bench_hgemm_mma`、`bench_hgemm_swizzle`、
    `bench_hgemm_cute`、`bench_hgemm_wgmma`、`bench_hgemm_tma_mma_ws`、`bench_launch_tma_mma_ws`、`bench_fa_launch`、
    `bench_fa_2_mma_stages_cute_launch/dispatch`、`bench_fa_tma_mma_ws_launch/dispatch`、`bench_fa_3_tma_ws_launch/dispatch`、
    `bench_fa_3_tma_mma_ws_cute_launch/dispatch`、`bench_fa_2_tma_mma_ws_cute_launch/dispatch`、
    `bench_fa_persist_d_cute_launch`（L4227）、`bench_fa_split_d_launch/dispatch`、`bench_flash_attn`、
    `bench_cudnn_sdpa_tflops`（**L2508 与 L4290 定义两次**，各自被 `#if defined(NOTES_V2_ENABLE_CUDNN)` 门控，签名不同）、
    `bench_hgemm_tflops`、`bench_fa_tflops`、`bench_cublas_hgemm_tflops`；辅助 `check_smem_feasible`、`check`、
    `should_print_fa_tflops`、`should_print_hgemm_tflops`（`bench_ffpa_split_d_*` 在 `bench/bench_ffpa.cu`，不在本文件）
  - 全局开关（`static`）：`g_debug`、`g_bench_hgemm`、`g_bench_hgemm_all`、`g_bench_fa`、`g_bench_fa3_cute_only`、
    `g_bench_all`、`g_fa_skip_check`、`g_swizzle_eq_check`、`g_fa_layout`（`enum class FALayout { All, Pad, SwizzleQ,
    SwizzleK, SwizzleV, SwizzleQK, SwizzleQV, SwizzleKV, Swizzle }`，默认 `FALayout::Pad`）、
    `g_bench_M/N/K = 8192`、`g_bench_B=1, g_bench_H=32, g_bench_Nfa=8192, g_bench_D=128`、`g_warmup=2, g_repeat=3`、
    `g_fa_f16_max_tflops`、`g_fa_f32_max_tflops`、`g_hgemm_f16_max_tflops`、`g_hgemm_f32_max_tflops`、`g_verbose`
  - **所有 `#define NOTES_V2_*` 开关名 = 0 条**（grep `^#define NOTES_V2_` 无命中）：所有 `NOTES_V2_*` 都是
    **外部 `-D` 编译宏**，本文件只做 `#if defined(...)`。**开关名单（去重，共 9 个）**：`NOTES_V2_ENABLE_CUTE`、
    `NOTES_V2_ENABLE_CUDNN`、`NOTES_V2_ENABLE_WGMMA`、`NOTES_V2_ENABLE_TMA_MMA_WS`、`NOTES_V2_ENABLE_SWIZZLE_V2`、
    `NOTES_V2_FORCE_INLINE_ASYNC_PROXY`、`NOTES_V2_ENABLE_SETMAXNREGS`、`NOTES_V2_REG_ALLOC`、`NOTES_V2_REG_DEALLOC`
    （后两者是 `common.cuh` 定义的调用宏）。**`NOTES_V2_ENABLE_TF32` 不存在**。头部注释的 10 个 Phase 见本节末。 
  - 头部注释 10 个 Phase 原文（逐条）：`Phase 0 — 面试框架速查`；`Phase 1 — 基础原语：Warp Reduce / Block Reduce / Dot Product（含 broadcast 增强版）`；
    `Phase 2 — Elementwise：ReLU / Elementwise Add / Histogram（基础 + float4 向量化 + atomic）`；
    `Phase 3 — Softmax：naive → safe → online + RMS/Layer Norm`；`Phase 4 — RoPE：旋转位置编码（Llama 风格 theta=10000）`；
    `Phase 5 — Mat Transpose：基础版 + BCF merge_write 最佳版（Bank Conflict专题）`；`Phase 6 — GEMV：SGEMV K32/K128/K16（warp-per-row）`；
    `Phase 7 — GEMM ★：SGEMM → HGEMM → MMA m16n8k16(TN布局) → WGMMA m64n128k16`；`Phase 8 — FlashAttention-2split_q（FA-2, 含 online softmax + P@V 寄存器复用）`
- **模板参数全列表（本文件内的模板函数，原样，行号 @ `6c86259`）**：全是 test/bench 分发模板：
  `template <int kHeadDim>`（L101/156/276/386/2235）、`template <int kHeadDim, int kNq, int kNkv, int kHq = 2, int kHkv = 2, ...>`（L517，新增 persist-D/QKV test）、
  `template <int kStages, int kBlockSwizzle = 0>`（L1951）、`template <int kHeadDim, int kStagesK>`（L2512/3478）、
  `template <int kStages, int kBlockSwizzle>`（L2716/2837/3067/3203）、
  `template <int kHeadDim, int kStagesK = 2, int kPadQ = 8, int kPadK = 8, ...>`（L3342 `bench_fa_launch`；**默认 kPad 都是 8**）、
  `template <int kHeadDim, int kStagesK, int kStagesV = 1, int kMmaAccF32 = 0>`（L3604 `bench_fa_tma_mma_ws_launch`）、
  `template <int kStagesK, int kStagesV = 1, int kMmaAccF32 = 0>`（L3745）、
  `template <int kHeadDim, int kStagesK, int kMmaAccF32 = 0>`（L3765）、`template <int kMmaAccF32 = 0>`（L3902）、
  `template <int kHeadDim, int kStagesK = 1>`（L3936）、`template <int kHeadDim, int kStagesK, int kStagesV = 1>`（L4071）、
  `template <int kHeadDim, int kStagesQK, int kStagesV>`（L4355）、`template <int kColStride>`（L4911）
- **PTX 宏与工具宏 / kernel 签名 / 同步原语**：无自有 PTX 宏、无 kernel；只用
  `common.cuh` 的宏 + `check()` / `check_smem_feasible()`，同步只用 `cudaStreamSynchronize` / `cudaEventSynchronize` / `cudaDeviceSynchronize`。
- **块/线程/warp 组织（只体现在 launch 配置）**：TMA MMA WS FA：`dim3 block(kNumThreads); // 384`、`dim3 grid((seqlen + Br - 1) / Br, B * H);`；`smem_bytes = (Br * kHeadDim + kStagesK * Bc * kHeadDim + kStagesV * Bc * kHeadDim) * sizeof(half);`；CuTe FA2/FA3：`dim3 grid(seqlen / kBr, B * H);`；手写 FA2：`dim3 grid((seqlen + Br - 1) / Br, B * H);`。
- **smem 布局与字节数**：用 `check_smem_feasible((const void *)fa_k, smem_bytes)` 判定后 `cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)`。
- **测试与容差**：**判据唯一值 `max_err >= 5e-1f`**（L3446/3561/3710/3867/4029/4177/4267）。参考实现 = CPU FP32（`srand(42)`）+ cuDNN SDPA。打印门槛注释原文：`only print when the current TFLOPS exceeds the running max for its accumulator category (f16/f32). Correctness failures always print so they are never silently dropped`。CLI：`--bench-hgemm`、`--bench`、`--bench-fa`、`--bench-fa3-cute`、`--bench-hgemm-all`、`--bench-fa-all`、`--bench-all`、`--mnk M,N,K`、`--bhnd B,H,N,D`、`--fa-layout <...>`、`--fa-skip-check`、`--swizzle-eq-check`、`--debug`、`--verbose`、`--warmup`、`--repeat`、`--tma-mma-ws M N K`（默认 128 128 64）。
- **注释里的关键结论**（原样摘录，5 条）：`// 整理自 LeetCUDA 项目（https://github.com/xlite-dev/LeetCUDA），涵盖：- 面试高频 CUDA kernel 的完整实现（~30 个 kernel）`；`// BLAS 语义：N=col-major(Normal), T=row-major(Transposed)`；`// 以下是测试代码，验证 Phase 1 - Phase 8 的kernel的正确性，不评估性能。`；`// ★ 多 head K/V offset: local_tile coord 需加 blockIdx.y * kv_tiles 偏移，否则所有 head 都读 head 0 的 K/V`；`// Default FA bench (kPadQ=kPadK=kPadV=8 only)`（印证 kPad 默认值是 8）

---

## 交叉核对：同一事实的不同表述（写教程时别混）

1. **tol**：仓库里 FA/GEMM 的**唯一硬判据是 `5e-1f`**；README 的 `1.831e-04 / 1.526e-05 / 9.155e-05 / 0.000e+00`
   是**实测 Max Err 报告值**，不是阈值。“F32Acc / F16Acc 两档经验误差 + 一个统一判据”才准确；**TF32 档不存在**。
2. **producer/consumer 划分有四套**（全部 @ `6c86259`）：
   - FA2 TMA WS（`flash_attn_tma_mma_ws_stages_split_q`）：`128 producer + 256 consumer = 384`，1 个 consumer WG（8 warps），全 KV 遍历，Br=128。
   - FA3 TMA WS（`flash_attn_3_tma_ws_stages_split_q`）：`128 producer + 2×128 consumer = 384`，2 个 consumer WG 按 **tile 奇偶** 分 KV，Br=Bc=64，需 split-KV 合并。
   - FFPA TMA WS（`ffpa_attn_tma_mma_ws_split_d_cute`）：`128 producer + 128 consumer = 256`（Large-D Split-D）。
   - persist-D WS（`flash_attn_cute_persist_d_sm120`，新增）：`128T producer + 256T consumer = 384`。
   - 对照：`ws-hgemm/naive_ws_hgemm_sm8x.cu` = `32 producer + 128 consumer = 160`，用 `cuda::pipeline` 而非 mbarrier。
3. **barrier arrive_count 有三种数**：FA2 = `256 + 1 = 257`；FA3 = `full_Q` 256+1、per-WG 的 K/V = `128 + 1 = 129`；
   FFPA = `TmaBarrier::init(&qk_full[stage], 1)`（**1**）+ `CtaBarrier::init(&qk_empty[stage], kConsumerThreads=128)`。
4. **smem 对齐**：FA TMA 路径必须 `__align__(1024)`（消费者用软件 `swizzle<64>`，假设 phase=0）；
   WGMMA 路径只需 `alignas(128)`（descriptor 的 `base_offset` 位域自动补偿 phase）——论证见 `hgemm.cuh` L1898–1919。

---

## 提取中发现的、要在教程里点名的“注释/代码不一致”

1. `flash_attn.cuh` 头注释写 `Block: (128, 1, 1)，kNumThreads=kWarpSize×kMmaTileSeqLenQ×kMmaTileSeqLenK=128`，
   而同一行公式在代码里的注释是 `// 32*8*1=256`；`Br = kMmaAtomM*kMmaTileSeqLenQ*kValTileSeqLenQ` 在不同实例化下为 64 或 128。
   头注释的 `Grid: ((N + 63) / 64, B * H, 1)，Br=64` 也只对应 Br=64 的那组实例化。
2. `flash_attn.cuh` L126–127 明确指出 `Tc = (N + Bc - 1) / Bc` 的 ceil 写法「是为了说明 tile 划分方式，不等于当前实现已经完整处理了尾 tile」。
3. `notes-v2.cu` **L2508 与 L4290 两处定义同名 `bench_cudnn_sdpa_tflops`**（不同 `#if` 门控、不同签名），引用时必须区分。
4. 仓库在本次提取期间被外部更新（HEAD `e831d97` → `6c86259`）：`flash_attn.cuh` 新增
   `flash_attn_cute_persist_d_sm120` 与 `flash_attn_cute_persist_d_sm120_launch<>`；
   `notes-v2.cu` 新增 `test_flash_attn_cute_persist_d_sm120`、`bench_fa_persist_d_cute_launch`；
   `build.sh` 新增 `sm_120f` arch。**任务要求的 8 个文件的“原内容”未变**，但涉及 persistent-CTA / sm_120f 时需补这一层。
5. 引用任何行号前请先重新 grep：该仓库 `kernels/interview/` 正在被活跃修改
   （本次会话内 `notes-v2.cu` 从 4865 行涨到 5219 行，`flash_attn.cuh` 从 3490 行涨到 4179 行）。
