// q190_merge_attn.cu — Q190：merge_attn_states（LSE 合并，用于 split-KV / FlashDecoding）
//
//   1) split_attn_partial —— 把 KV 切成 2 段，各自算【未归一化】的 O 与 (m, l)
//   2) merge_attn_states  —— 用 LSE 把两段合并（本文件的核心）
//   3) 对照：一次性算完整 attention（证明合并结果与一次性结果逐元素相等）
//
// 编译：./build.sh q190
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 两套口径不可混搭（这是本题最重要的一句话）
//      **FA 主循环的 O 是【未归一化】的**（配合递推里的 e^{m_old - m_new}）；
//      **merge_attn_states 的 LSE 形式作用在【已归一化】的 O 上**。
//      混搭是最常见的隐蔽错误 —— 结果不会报错，只是数值系统性偏大/偏小。
//
// ② LSE 的定义与合并（要能默写）
//      LSE = m + log(l)                  ← 把 (m, l) 两个数【无损】打包成一个标量
//      L   = max(LSE_A, LSE_B)           ← 以最大者为 pivot
//      w_A = e^{LSE_A - L};  w_B = e^{LSE_B - L}
//      alpha = w_A / (w_A + w_B);  beta = w_B / (w_A + w_B)     ← alpha + beta = 1（凸组合）
//      O = alpha * O_A + beta * O_B
//    ★ 以 max 为 pivot 让两个权重都 <= 1 ⇒ **数值上绝不上溢**。
//
// ③ 与 FA 递推的关系（面试官很可能追问）
//      递推里的 e^{m_old - m_new} 是「**在线合并**」（寄存器里一份 O 滚动）；
//      LSE 式是「**事后合并**」（两块各落盘 O 与 LSE 后用 elementwise kernel 拼回）。
//      后者服务 split-KV / FlashDecoding：把长序列切成多段并行算，再拼回来。
//    ★ 块数 > 2 怎么办？→ 多用几次这个 kernel 折叠，或把 LSE 也做成可归约的
//      (m, l) 状态：m = max(m1,m2)；l = l1*e^{m1-m} + l2*e^{m2-m}（注意这次是【未归一化】口径）。
//
// ④ 书里的实测：merge kernel 0.7849 ms / 1031.4 GB/s（≈ 峰值 77%，搬 809.5 MB），
//    **结果与一次性 softmax 逐元素相等** —— 这是"合并是无损的"的实测证据。
//
// ⑤ 一个容易忽略的边界：如果某一段的 l == 0（该段全被 mask 掉），
//    LSE = m + log(0) = -inf ⇒ w = e^{-inf - L} = 0 ⇒ 该段权重为 0，自动退化。正确。
//    但如果**两段都**是 0，则 w_A + w_B = 0 ⇒ 除零 ⇒ NaN。**全 mask 的行要由调用方保证非空。**
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① 分段的 partial：对第 [j_begin, j_end) 段 KV 算【未归一化】的 O 与 (m, l)
//    一个 block 处理一行 query；输出 o_partial[qi][d] 与 lse[qi]
// ---------------------------------------------------------------------------
template <int kThreads = 128, int kD = 64, int kBc = 64>
__global__ void split_attn_partial(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   const float* __restrict__ V,
                                   float* __restrict__ o_partial,
                                   float* __restrict__ lse, int N, int j_begin,
                                   int j_end, float scale, bool normalize_o) {
  __shared__ float s_k[kBc][kD];
  __shared__ float s_v[kBc][kD];
  __shared__ float s_p[kBc];

  const int qi = blockIdx.x;
  const int tid = threadIdx.x;
  const float* q = Q + static_cast<size_t>(qi) * kD;

  float m = -FLT_MAX, l = 0.0f, o_acc = 0.0f;
  const int dcol = tid;

  for (int j0 = j_begin; j0 < j_end; j0 += kBc) {
    for (int i = tid; i < kBc * kD; i += kThreads) {
      const int r = i / kD, c = i % kD;
      const int j = j0 + r;
      const bool ok = (j < j_end);
      s_k[r][c] = ok ? K[static_cast<size_t>(j) * kD + c] : 0.0f;
      s_v[r][c] = ok ? V[static_cast<size_t>(j) * kD + c] : 0.0f;
    }
    __syncthreads();

    float s_ij = -FLT_MAX;
    if (tid < kBc && (j0 + tid) < j_end) {
      float acc = 0.0f;
#pragma unroll 4
      for (int d = 0; d < kD; ++d) acc += q[d] * s_k[tid][d];
      s_ij = acc * scale;
    }
    const float m_new = fmaxf(m, block_reduce_max<kThreads>(s_ij));
    const float p_ij = (s_ij > -FLT_MAX) ? __expf(s_ij - m_new) : 0.0f;
    if (tid < kBc) s_p[tid] = p_ij;
    const float rowsum = block_reduce_sum<kThreads>(p_ij);
    const float rescale = __expf(m - m_new);      // 首块 m = -inf ⇒ rescale = 0
    l = l * rescale + rowsum;
    __syncthreads();

    if (dcol < kD) {
      float acc = 0.0f;
#pragma unroll 4
      for (int r = 0; r < kBc; ++r) {
        const int j = j0 + r;
        if (j < j_end) acc += s_p[r] * s_v[r][dcol];
      }
      o_acc = o_acc * rescale + acc;
    }
    m = m_new;
    __syncthreads();
  }

  // ★ 输出两种口径，由 normalize_o 决定（这正是"混搭会错"的那个开关）
  if (dcol < kD) {
    o_partial[static_cast<size_t>(qi) * kD + dcol] =
        normalize_o ? (o_acc / l) : o_acc;
  }
  if (tid == 0) lse[qi] = m + __logf(l);          // ★ LSE = m + log(l)
}

// ---------------------------------------------------------------------------
// ② merge：把两段的 (O, LSE) 合成最终输出 —— 本文件的核心 kernel
// ---------------------------------------------------------------------------
__global__ void merge_attn_states_kernel(const float* __restrict__ o_a,
                                         const float* __restrict__ lse_a,
                                         const float* __restrict__ o_b,
                                         const float* __restrict__ lse_b,
                                         float* __restrict__ out, int rows, int D) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * D) return;
  const int row = i / D;

  const float la = lse_a[row], lb = lse_b[row];
  const float L = fmaxf(la, lb);                  // ★ 以 max 为 pivot
  const float wa = __expf(la - L), wb = __expf(lb - L);   // 两者都 <= 1 ⇒ 不上溢
  const float inv = 1.0f / (wa + wb);
  const float alpha = wa * inv, beta = wb * inv;  // alpha + beta = 1（凸组合）

  // ★ 这里要求 o_a / o_b 已经是【已归一化】的分段输出
  out[i] = alpha * o_a[i] + beta * o_b[i];
}

// ---------------------------------------------------------------------------
// host 侧参考
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
  TEST_BEGIN("Q190 merge_attn_states (LSE merge for split-KV)");

  const int N = 512, D = 64;
  const float scale = 1.0f / std::sqrt(float(D));
  const int mid = N / 2;

  std::vector<float> hQ(size_t(N) * D), hK(size_t(N) * D), hV(size_t(N) * D),
      href(size_t(N) * D);
  fill_random(hQ, 121u, -1.f, 1.f);
  fill_random(hK, 122u, -1.f, 1.f);
  fill_random(hV, 123u, -1.f, 1.f);
  ref_attn(hQ, hK, hV, href, N, D);

  float *dQ, *dK, *dV;
  CUDA_CHECK(cudaMalloc(&dQ, hQ.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dK, hK.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dV, hV.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK.data(), hK.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV.data(), hV.size() * sizeof(float), cudaMemcpyHostToDevice));

  const size_t o_elems = size_t(N) * D;
  float *dOa, *dOb, *dOmerged, *dLseA, *dLseB;
  CUDA_CHECK(cudaMalloc(&dOa, o_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOb, o_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dOmerged, o_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dLseA, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dLseB, N * sizeof(float)));

  const double rtol = 1e-3, atol = 1e-5;
  GpuTimer timer;

  // ---- 分两段各算 partial（★ 输出【已归一化】的 O 与 LSE，才符合 LSE 口径）----
  timer.bench([&] {
    split_attn_partial<128, D, 64><<<N, 128>>>(dQ, dK, dV, dOa, dLseA, N, 0, mid, scale, true);
    split_attn_partial<128, D, 64><<<N, 128>>>(dQ, dK, dV, dOb, dLseB, N, mid, N, scale, true);
  });
  CUDA_CHECK_KERNEL();

  // ---- 合并 ----
  const int threads = 256;
  const int blocks = (int)((o_elems + threads - 1) / threads);
  double ms_merge = timer.bench([&] {
    merge_attn_states_kernel<<<blocks, threads>>>(dOa, dLseA, dOb, dLseB, dOmerged, N, D);
  });
  CUDA_CHECK_KERNEL();

  std::vector<float> hgot = to_host(dOmerged, o_elems);
  CompareResult r = compare(hgot, href, rtol, atol);
  report("merge 2 partials -> full attn", r, rtol, atol, ms_merge);
  printf("        ★ 合并结果应当与【一次性算完整 attention】逐元素相等 —— 合并是无损的\n");
  printf("        merge kernel 峰值: 读 2*O(+LSE) + 写 O = %.2f MB, 带宽 %.0f GB/s\n",
           (2.0 * o_elems + 2.0 * N + o_elems) * 4 / 1e6,
           to_gbps((2 * o_elems + 2 * N + o_elems) * 4, ms_merge));

  // ---- 口径错误的演示：用【未归一化】的 O 去合并会怎样 ----
  {
    timer.bench([&] {
      split_attn_partial<128, D, 64><<<N, 128>>>(dQ, dK, dV, dOa, dLseA, N, 0, mid, scale, false);
      split_attn_partial<128, D, 64><<<N, 128>>>(dQ, dK, dV, dOb, dLseB, N, mid, N, scale, false);
    });
    CUDA_CHECK_KERNEL();
    merge_attn_states_kernel<<<blocks, threads>>>(dOa, dLseA, dOb, dLseB, dOmerged, N, D);
    CUDA_CHECK_KERNEL();
    hgot = to_host(dOmerged, o_elems);
    CompareResult bad = compare(hgot, href, rtol, atol);
    report("BAD: merge unnormalized O", bad, rtol, atol);
    printf("        ↑ 这是**故意做错**的版本：把未归一化的 O 喂给 LSE 合并。\n");
    printf("          它不会报错、形状也合法，只是数值系统性错 —— 这就是「口径混搭」的代价。\n");
  }

  // ---- 极端的一段为空（全 mask）----
  {
    const int D2 = D;
    std::vector<float> hz(size_t(N) * D2, 0.0f);
    std::vector<float> hlse_a(N, -1e30f), hlse_b(N, 0.0f);  // A 段全 mask ⇒ LSE = -inf 近似
    float *dz, *dla, *dlb, *dout;
    CUDA_CHECK(cudaMalloc(&dz, hz.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dla, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dlb, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, hz.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dz, hz.data(), hz.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dla, hlse_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dlb, hlse_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> ob(hz.size(), 3.0f);
    CUDA_CHECK(cudaMemcpy(dz, ob.data(), ob.size() * sizeof(float), cudaMemcpyHostToDevice));
    merge_attn_states_kernel<<<blocks, threads>>>(dz, dla, dz, dlb, dout, N, D2);
    CUDA_CHECK_KERNEL();
    std::vector<float> hz2 = to_host(dout, hz.size());
    double mx = 0.0;
    for (float v : hz2) mx = std::max(mx, std::fabs(double(v) - 3.0));
    printf("\n  %-34s %s  max_abs_err=%.3e  （A 段 LSE=-inf ⇒ w_A=0 ⇒ 结果应等于 B 段）\n",
           "merge with one empty segment", mx < 1e-6 ? "PASS" : "FAIL", mx);
    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(dla));
    CUDA_CHECK(cudaFree(dlb));
    CUDA_CHECK(cudaFree(dout));
  }

  CUDA_CHECK(cudaFree(dQ));
  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dOa));
  CUDA_CHECK(cudaFree(dOb));
  CUDA_CHECK(cudaFree(dOmerged));
  CUDA_CHECK(cudaFree(dLseA));
  CUDA_CHECK(cudaFree(dLseB));
  printf("\nQ190 done.\n");
  return 0;
}
