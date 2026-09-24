// q189_flash_attn.cu — Q189：FlashAttention 的内层循环（online rescale）
//
//   1) attn_reference   —— 朴素三 kernel 版（物化 S/P 矩阵，演示 IO 灾难）
//   2) attn_flash       —— tiling + online softmax：S/P 只在寄存器/smem 里，
//                          O 全程未归一化、循环外只除一次 ℓ
//   3) attn_flash_split —— split-Q（每个 warp 独占若干行 Q，零跨 warp 归约）
//
// 编译：./build.sh q189
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ online rescale 的五步（必须能默写，第 2 步是"减【新】max"）
//      1) m_new = max(m_old, rowmax(S))
//      2) P     = exp(S - m_new)                     ← 减新 max
//      3) l_new = exp(m_old - m_new) * l_old + rowsum(P)
//      4) O    <- O * diag(exp(m_old - m_new)) + P @ V   ← l 与 O 用【同一个】因子
//      5) （全部块扫完后【只做一次】） O <- diag(1/l) O
//    ★ 顺序不能换：先算 m_new，再同时修正 l 与 O，最后累加新块。
//    ★ 首块 m_old = -inf 时用三元判断让 m_new 顶替旧值，避免 e^{-inf} 与 0*inf 污染。
//
// ② ★ 为什么 1/l 只做一次（FA2 相对 FA1 的关键简化）
//      O 全程保持【未归一化】⇒ 每块省掉一次对角修正。
//      副产品：退化情形天然安全（e^{-inf} = 0 精确给出，不产生 NaN）。
//
// ③ ★ 行归约宽度必须是 4，不是 32
//      由 m16n8 C fragment 布局：行 r ∈ [0,8) 的 8 个列元素分布在 lane 4r..4r+3
//      ⇒ 先在 lane 内跨 j tile 局部归约，再 warp_reduce_max<4>（掩码 {2,1}）。
//      **误用整宽 32 会把不同行的 max/sum 混在一起 —— 静默错误。**
//      本文件的简化实现按"一行一个 warp"来组织（教学清晰），完整 fragment 版见 q186。
//
// ④ ★ 为什么 K 双缓冲、V 单级（一个很漂亮的论证）
//      迭代 t 内提交两组搬运：V_t（当前）+ K_{t+1}（预取）。
//      **V_t 的在途窗口恰好覆盖 QK mma 与 softmax**（softmax 全在 CUDA Core/SFU 上跑）
//      ⇒ 一个 stage 就够；K_{t+1} 有整整一个迭代余量 ⇒ 需要双缓冲。
//
// ⑤ 复杂度与算术强度
//      FLOPs 不变（4*N^2*d，分块只重排顺序）；显存 Theta(N^2) -> Theta(N*d)
//      IO_FA = Theta(Nd + N^2 d^2 / M)，  AI_FA ≈ Br = 64~128
//      AI_naive ≈ d/2 = 32（d=64, fp16）
//    ★ 数字直觉：B=1,H=32,N=4096,D=64 时朴素实现的中间矩阵 HBM 往返 ≈ 4.3 GB，
//      而有效数据（Q/K/V 读入 + O 写出）只有 67 MB —— **1/64**。
//      总 FLOPs ≈ 137 GFLOP ⇒ 计算下界 0.83 ms；按 2 TB/s 乐观带宽仅搬运就要 2.2 ms。
//      ⇒ **memory-bound 无疑，且 O(N^2) 的显存账在长序列下先于时间把方案毙掉。**
//
// ⑥ 精度分层（两条 acc 路径共同的精度地板）
//      S/O 累加器 f32、exp/max/rowsum/rescale 全走 fp32、P 写回寄存器是 fp16。
//      实测（书里）：F16Acc ~2e-4、F32Acc ~5e-5；
//    ★ 而且 **F32Acc 反而更快**（FA2 MMA 版 175.88 T vs 166.56 T）——
//      因为 f32 累加省掉了 softmax/rescale 阶段 half<->float 的整簇转换。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① 朴素参考：物化 S 与 P（只在小 shape 上跑，用来对拍）
// ---------------------------------------------------------------------------
__global__ void attn_reference_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ O, float* __restrict__ S,
                                      int N, int D, float scale) {
  // 一个 block 处理一行 query
  const int qi = blockIdx.x;
  const int tid = threadIdx.x;
  // S[qi][j] = sum_d Q[qi][d] * K[j][d] * scale
  for (int j = tid; j < N; j += blockDim.x) {
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += Q[qi * D + d] * K[j * D + d];
    S[qi * N + j] = s * scale;
  }
  __syncthreads();
  // softmax 行
  float m = -FLT_MAX;
  for (int j = tid; j < N; j += blockDim.x) m = fmaxf(m, S[qi * N + j]);
  m = block_reduce_max<256>(m);
  float sum = 0.0f;
  for (int j = tid; j < N; j += blockDim.x) {
    float p = __expf(S[qi * N + j] - m);
    S[qi * N + j] = p;
    sum += p;
  }
  sum = block_reduce_sum<256>(sum);
  // O[qi][d] = sum_j P[qi][j] * V[j][d] / sum
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) acc += S[qi * N + j] * V[j * D + d];
    O[qi * D + d] = acc / sum;
  }
}

// ---------------------------------------------------------------------------
// ② FlashAttention：tiling + online softmax
//    组织方式：一个 block 处理一行 query；KV 按 kBc 分块；每线程负责一个输出维度。
//    这是"结构完全正确"的教学版（生产版要把 P 留在 fragment 寄存器里，
//    并且用 split-Q 让每个 warp 独占 16 行 Q —— 见下面的说明）。
// ---------------------------------------------------------------------------
template <int kThreads = 128, int kD = 64, int kBc = 64>
__global__ void attn_flash_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V, float* __restrict__ O,
                                  int N, float scale) {
  __shared__ float s_k[kBc][kD];
  __shared__ float s_v[kBc][kD];
  __shared__ float s_p[kBc];      // 本块的 P 行（一行 query ⇒ 只有一行）

  const int qi = blockIdx.x;
  const int tid = threadIdx.x;
  const float* q = Q + static_cast<size_t>(qi) * kD;

  float m = -FLT_MAX, l = 0.0f;
  float o_acc = 0.0f;
  const int dcol = tid;                          // 每线程负责一列输出

  for (int j0 = 0; j0 < N; j0 += kBc) {
    // 装载 K/V
    for (int i = tid; i < kBc * kD; i += kThreads) {
      const int r = i / kD, c = i % kD;
      const int j = j0 + r;
      s_k[r][c] = (j < N) ? K[static_cast<size_t>(j) * kD + c] : 0.0f;
      s_v[r][c] = (j < N) ? V[static_cast<size_t>(j) * kD + c] : 0.0f;
    }
    __syncthreads();

    // S 的一列 + 行 max 的候选
    float s_ij = -FLT_MAX;
    if (tid < kBc) {
      const int j = j0 + tid;
      if (j < N) {
        float acc = 0.0f;
#pragma unroll 4
        for (int d = 0; d < kD; ++d) acc += q[d] * s_k[tid][d];
        s_ij = acc * scale;
      }
    }
    const float block_max = block_reduce_max<kThreads>(s_ij);
    const float m_new = fmaxf(m, block_max);       // ★ 第 1 步
    const float p_ij = (s_ij > -FLT_MAX) ? __expf(s_ij - m_new) : 0.0f;  // ★ 第 2 步
    if (tid < kBc) s_p[tid] = p_ij;
    const float rowsum = block_reduce_sum<kThreads>(p_ij);
    const float rescale = __expf(m - m_new);       // ★ 同一个因子
    l = l * rescale + rowsum;                      // ★ 第 3 步
    __syncthreads();

    // O <- O*rescale + P @ V
    if (dcol < kD) {
      float acc = 0.0f;
#pragma unroll 4
      for (int r = 0; r < kBc; ++r) {
        const int j = j0 + r;
        if (j < N) acc += s_p[r] * s_v[r][dcol];
      }
      o_acc = o_acc * rescale + acc;               // ★ 第 4 步
    }
    m = m_new;
    __syncthreads();
  }
  if (dcol < kD) O[static_cast<size_t>(qi) * kD + dcol] = o_acc / l;   // ★ 第 5 步
}

// ---------------------------------------------------------------------------
// host 侧参考（attention 的 CPU 直算，double 累加）
// ---------------------------------------------------------------------------
static void ref_attn(const std::vector<float>& Q, const std::vector<float>& K,
                     const std::vector<float>& V, std::vector<float>& O, int N, int D) {
  const double scale = 1.0 / std::sqrt(double(D));
  for (int i = 0; i < N; ++i) {
    std::vector<double> s(N);
    double m = -1e308;
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int d = 0; d < D; ++d) acc += double(Q[i * D + d]) * K[j * D + d];
      s[j] = acc * scale;
      m = std::max(m, s[j]);
    }
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
      s[j] = std::exp(s[j] - m);
      sum += s[j];
    }
    for (int d = 0; d < D; ++d) {
      double acc = 0.0;
      for (int j = 0; j < N; ++j) acc += s[j] * V[j * D + d];
      O[i * D + d] = float(acc / sum);
    }
  }
}

int main() {
  TEST_BEGIN("Q189 FlashAttention inner loop (online rescale)");

  const int N = 256, D = 64;
  const float scale = 1.0f / std::sqrt(float(D));

  std::vector<float> hQ(size_t(N) * D), hK(size_t(N) * D), hV(size_t(N) * D),
      hO(size_t(N) * D), href(size_t(N) * D), hgot(size_t(N) * D);
  fill_random(hQ, 111u, -1.f, 1.f);
  fill_random(hK, 112u, -1.f, 1.f);
  fill_random(hV, 113u, -1.f, 1.f);
  ref_attn(hQ, hK, hV, href, N, D);

  float *dQ, *dK, *dV, *dO, *dS;
  CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dO, hO.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dS, size_t(N) * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(float), cudaMemcpyHostToDevice));

  // 注意力输出是加权平均 ⇒ 尺度小、舍入多；1e-3 的相对容差是合理口径
  const double rtol = 1e-3, atol = 1e-5;
  GpuTimer timer;

  {
    CUDA_CHECK(cudaMemset(dO, 0, hO.size() * sizeof(float)));
    double ms = timer.bench([&] {
      attn_reference_kernel<<<N, 256>>>(dQ, dK, dV, dO, dS, N, D, scale);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dO, hO.size());
    report("attn_reference (materialize S/P)", compare(hgot, href, rtol, atol),
           rtol, atol, ms);
    printf("        ★ 显存账：它额外物化了 S = %d x %d x 4 B = %.1f MB\n", N, N,
           double(N) * N * 4 / 1e6);
  }
  {
    CUDA_CHECK(cudaMemset(dO, 0, hO.size() * sizeof(float)));
    double ms = timer.bench([&] {
      attn_flash_kernel<128, D, 64><<<N, 128>>>(dQ, dK, dV, dO, N, scale);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dO, hO.size());
    report("attn_flash (tiling + online)", compare(hgot, href, rtol, atol), rtol, atol, ms);
    printf("        ★ 它没有物化 S/P：KV 分块后中间量只在 smem/寄存器里\n");
  }

  // ---- 长序列的显存对比（不解 O(N^2) 的话会直接 OOM）----
  {
    const int Nl = 4096;
    const double s_bytes = double(Nl) * Nl * 4;
    printf("\n  长序列（N=%d, D=%d）的显存账：\n", Nl, D);
    printf("    朴素版物化 S 需要 %.0f MB，再加上 P 又是 %.0f MB ⇒ 单头就 %.1f GB\n",
           s_bytes / 1e6, s_bytes / 1e6, 2 * s_bytes / 1e9);
    printf("    32 个头的 S/P 共约 %.1f GB —— **这就是 O(N^2) 显存账**\n",
           32 * 2 * s_bytes / 1e9);
    printf("    FlashAttention 只需 O(N*d) = %.1f MB（+ smem 里的分块）\n",
           double(Nl) * D * 4 * 3 / 1e6);
    printf("    书里的 kPad=0 消融则显示：不物化 S/P 只是入场券，**smem 读写两端的平衡\n");
    printf("    才是把 73%% 变成 90%% 的功课**（120.0 T vs 166.6 T 的差距来自 LDGSTS wavefronts）。\n");
  }

  CUDA_CHECK(cudaFree(dQ));
  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dO));
  CUDA_CHECK(cudaFree(dS));
  printf("\nQ189 done.\n");
  return 0;
}
