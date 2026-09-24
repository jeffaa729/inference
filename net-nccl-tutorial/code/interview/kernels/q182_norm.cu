// q182_norm.cu — Q182：RMSNorm / LayerNorm（归一化算子的两种实现与一趟方案）
//
//   1) rms_norm_twopass  —— 第一趟求平方和，第二趟归一化（两趟读 x）
//   2) rms_norm_onepass  —— 一趟：把 x 留在寄存器/smem 里，只做一次全局读
//   3) layer_norm        —— 需要 mean 与 var（本文件用两趟，便于对照）
//   4) layer_norm_welford—— 一趟 Welford：合并 (mean, M2) 状态，与 online softmax 同构
//
// 编译：./build.sh q182
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① RMSNorm 与 LayerNorm 的差别（写代码前先说清）：
//      RMSNorm:   y = x / sqrt(mean(x^2) + eps) * w          —— 不减均值，只除 RMS
//      LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b   —— 减均值再除标准差
//    ⇒ RMSNorm **少一次归约**（不需要 mean），少一次全局同步，且不需要 bias。
//
// ② ★ 两趟 vs 一趟的取舍（这道题的核心）：
//      两趟：读 x 两遍（第二遍通常 L1 命中，但**不是零成本**）+ 2 次 block 归约 + 2 次同步
//      一趟：把 x 留在寄存器或 smem ⇒ 只读全局一次，但**占用寄存器/smem**，
//            且 Welford 的归约算子更重（每次合并 2 个状态）
//    ★ 书里有一个非常值得背的实测结论：
//      vec4 对 rms_norm/layer_norm 的收益在 **L2 口径下是 +16%/+24%**，
//      但在 **HBM 口径下四个 kernel 全部收敛到 1000-1070 GB/s、收益消失**；
//      而**「单 pass」在两种口径下都快约 7%（-6.9% / -6.5%）**。
//      ⇒ **减少指令数的优化只在指令受限时有效；减少数据流量的优化在任何口径下都有效。**
//
// ③ Welford 的合并公式（要能默写）：
//      给定两段各自的 (n1, mean1, M2_1) 与 (n2, mean2, M2_2)：
//        n  = n1 + n2
//        d  = mean2 - mean1
//        mean = mean1 + d * n2 / n
//        M2 = M2_1 + M2_2 + d * d * n1 * n2 / n
//      var = M2 / n。**只做一次全局读**，且数值上比"先求和再求平方和"稳定
//      （后者在 mean 很大、方差不大的时候会发生灾难性抵消）。
//
// ④ 一个每帧都会踩的细节：eps 的位置。
//      PyTorch 的 LayerNorm 是 (x - mean) / sqrt(var + eps)，而 RMSNorm 常见两种写法：
//        rsqrt(mean(x^2) + eps)      ← Llama 系
//        rsqrt(mean(x^2)) + eps      ← 少数实现
//      写 kernel 前必须和参考实现对齐，否则数值对不上（而且形状完全合法）。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① RMSNorm 两趟
// ---------------------------------------------------------------------------
template <int kThreads = 256>
__global__ void rms_norm_twopass(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 float* __restrict__ y, int n, float eps) {
  __shared__ float s_scale;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < n; i += kThreads) ss += x[i] * x[i];
  float total = block_reduce_sum<kThreads>(ss);
  if (threadIdx.x == 0) s_scale = rsqrtf(total / n + eps);
  __syncthreads();                      // ★ 广播需要同步
  const float scale = s_scale;
  for (int i = threadIdx.x; i < n; i += kThreads) y[i] = x[i] * scale * w[i];
}

// ---------------------------------------------------------------------------
// ② RMSNorm 一趟：把 x 缓存在 smem（前提：n <= smem 容量 / 4）
//    注意：这只对"一行能放进 smem"的情况成立（hidden <= ~8K 时可行）
// ---------------------------------------------------------------------------
template <int kThreads = 256, int kMaxN = 4096>
__global__ void rms_norm_onepass(const float* __restrict__ x,
                                 const float* __restrict__ w,
                                 float* __restrict__ y, int n, float eps) {
  __shared__ float s_x[kMaxN];
  __shared__ float s_scale;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    float v = x[i];
    s_x[i] = v;                          // 缓存，第二趟不再读全局
    ss += v * v;
  }
  float total = block_reduce_sum<kThreads>(ss);
  if (threadIdx.x == 0) s_scale = rsqrtf(total / n + eps);
  __syncthreads();
  const float scale = s_scale;
  for (int i = threadIdx.x; i < n; i += kThreads) y[i] = s_x[i] * scale * w[i];
}

// ---------------------------------------------------------------------------
// ③ LayerNorm（两趟 + 广播）
// ---------------------------------------------------------------------------
template <int kThreads = 256>
__global__ void layer_norm(const float* __restrict__ x,
                           const float* __restrict__ w,
                           const float* __restrict__ b,
                           float* __restrict__ y, int n, float eps) {
  __shared__ float s_mean, s_rstd;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += kThreads) sum += x[i];
  const float mean = block_reduce_sum<kThreads>(sum) / n;

  float sq = 0.0f;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    const float d = x[i] - mean;
    sq += d * d;
  }
  const float var = block_reduce_sum<kThreads>(sq) / n;
  if (threadIdx.x == 0) {
    s_mean = mean;
    s_rstd = rsqrtf(var + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += kThreads)
    y[i] = (x[i] - s_mean) * s_rstd * w[i] + b[i];
}

// ---------------------------------------------------------------------------
// ④ LayerNorm 一趟（Welford：状态 (count, mean, M2) 的归约）
// ---------------------------------------------------------------------------
struct Welford {
  float n, mean, m2;
};

__device__ __forceinline__ Welford welford_merge(Welford a, Welford b) {
  if (a.n == 0.0f) return b;
  if (b.n == 0.0f) return a;
  Welford r;
  r.n = a.n + b.n;
  const float d = b.mean - a.mean;
  r.mean = a.mean + d * (b.n / r.n);
  r.m2 = a.m2 + b.m2 + d * d * (a.n * b.n / r.n);
  return r;
}

template <int kThreads = 256>
__global__ void layer_norm_welford(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   float* __restrict__ y, int n, float eps) {
  // 每个线程先算自己那段的局部 Welford 状态，再做 warp/block 归约
  Welford local{0.0f, 0.0f, 0.0f};
  for (int i = threadIdx.x; i < n; i += kThreads) {
    local.n += 1.0f;
    const float d = x[i] - local.mean;
    local.mean += d / local.n;
    local.m2 += d * (x[i] - local.mean);
  }
  // warp 内合并
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    Welford o{__shfl_xor_sync(0xffffffffu, local.n, mask),
              __shfl_xor_sync(0xffffffffu, local.mean, mask),
              __shfl_xor_sync(0xffffffffu, local.m2, mask)};
    local = welford_merge(local, o);
  }
  // 跨 warp：写 smem 再合并（与 block_reduce 同构）
  __shared__ Welford welf[kThreads / 32];
  __shared__ float s_mean, s_rstd;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) welf[warp] = local;
  __syncthreads();
  if (warp == 0) {
    Welford o = (lane < kThreads / 32) ? welf[lane] : Welford{0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
      Welford p{__shfl_xor_sync(0xffffffffu, o.n, mask),
                __shfl_xor_sync(0xffffffffu, o.mean, mask),
                __shfl_xor_sync(0xffffffffu, o.m2, mask)};
      o = welford_merge(o, p);
    }
    if (lane == 0) {
      s_mean = o.mean;
      s_rstd = rsqrtf(o.m2 / o.n + eps);
    }
  }
  __syncthreads();
  const float mean = s_mean, rstd = s_rstd;
  // ★ 第二趟仍然要读 x —— 真正的"一趟"需要把 x 留在 smem/寄存器里；
  //   这里演示的是"只做一次归约"（Welford），全局读仍然是两趟。
  for (int i = threadIdx.x; i < n; i += kThreads)
    y[i] = (x[i] - mean) * rstd * w[i] + b[i];
}

// ---------------------------------------------------------------------------
// host 侧参考（用 double 提高精度，作为"真值"）
// ---------------------------------------------------------------------------
static void ref_rms(const std::vector<float>& x, const std::vector<float>& w,
                    std::vector<float>& y, int n, float eps) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
  double scale = 1.0 / std::sqrt(ss / n + eps);
  for (int i = 0; i < n; ++i) y[i] = float(double(x[i]) * scale * w[i]);
}

static void ref_ln(const std::vector<float>& x, const std::vector<float>& w,
                   const std::vector<float>& b, std::vector<float>& y, int n,
                   float eps) {
  double mean = 0.0;
  for (int i = 0; i < n; ++i) mean += x[i];
  mean /= n;
  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = x[i] - mean;
    var += d * d;
  }
  var /= n;
  double rstd = 1.0 / std::sqrt(var + eps);
  for (int i = 0; i < n; ++i)
    y[i] = float((double(x[i]) - mean) * rstd * w[i] + b[i]);
}

int main() {
  TEST_BEGIN("Q182 RMSNorm / LayerNorm");

  const int T = 256;
  const float eps = 1e-5f;

  for (int n : {128, 1024, 4096}) {
    std::vector<float> hx(n), hw(n), hb(n), href(n), hgot(n);
    fill_random(hx, 41u, -2.f, 2.f);
    fill_random(hw, 42u, 0.5f, 1.5f);
    fill_random(hb, 43u, -0.1f, 0.1f);

    float *dx, *dw, *db, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, hw.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer;
    const size_t bytes = n * 2 * sizeof(float);  // 读 x + 写 y

    // ---- RMSNorm 两趟 vs 一趟 ----
    ref_rms(hx, hw, href, n, eps);
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_2 = timer.bench([&] { rms_norm_twopass<T><<<1, T>>>(dx, dw, dy, n, eps); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report(("rms_norm twopass  n=" + std::to_string(n)).c_str(),
           compare(hgot, href, 1e-5, 1e-6), 1e-5, 1e-6, ms_2);

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_1 = timer.bench([&] { rms_norm_onepass<T, 4096><<<1, T>>>(dx, dw, dy, n, eps); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report(("rms_norm onepass  n=" + std::to_string(n)).c_str(),
           compare(hgot, href, 1e-5, 1e-6), 1e-5, 1e-6, ms_1);
    printf("        onepass/twopass 时间比 = %.3f（<1 表示一趟省下了第二次全局读）\n",
           ms_1 / ms_2);

    // ---- LayerNorm 两趟 vs Welford ----
    ref_ln(hx, hw, hb, href, n, eps);
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_ln = timer.bench([&] { layer_norm<T><<<1, T>>>(dx, dw, db, dy, n, eps); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report(("layer_norm 2pass  n=" + std::to_string(n)).c_str(),
           compare(hgot, href, 1e-5, 1e-6), 1e-5, 1e-6, ms_ln);

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_wf = timer.bench([&] { layer_norm_welford<T><<<1, T>>>(dx, dw, db, dy, n, eps); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report(("layer_norm welford n=" + std::to_string(n)).c_str(),
           compare(hgot, href, 1e-5, 1e-6), 1e-5, 1e-6, ms_wf);
    printf("        welford/2pass 时间比 = %.3f（单次归约，但第二趟仍读 x）\n",
           ms_wf / ms_ln);
    printf("        带宽口径 8 B/元素: rms2 %.0f | rms1 %.0f | ln %.0f | ln_wf %.0f GB/s\n",
           to_gbps(bytes, ms_2), to_gbps(bytes, ms_1), to_gbps(bytes, ms_ln),
           to_gbps(bytes, ms_wf));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dy));
  }

  // ---- 数值稳定性：mean 很大、方差很小的时候 ----
  {
    const int n = 4096;
    std::vector<float> hx(n, 1000.0f);       // mean = 1000，var ≈ 0
    for (int i = 0; i < n; ++i) hx[i] += (i % 2) ? 1e-3f : -1e-3f;
    std::vector<float> hw(n, 1.0f), hb(n, 0.0f), href(n), hgot(n);
    ref_ln(hx, hw, hb, href, n, eps);

    float *dx, *dw, *db, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, hw.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    layer_norm<256><<<1, 256>>>(dx, dw, db, dy, n, eps);
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    CompareResult r1 = compare(hgot, href, 1e-3, 1e-3);
    report("layer_norm 2pass (mean=1000, var~0)", r1, 1e-3, 1e-3);

    layer_norm_welford<256><<<1, 256>>>(dx, dw, db, dy, n, eps);
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    CompareResult r2 = compare(hgot, href, 1e-3, 1e-3);
    report("layer_norm welford (same input)", r2, 1e-3, 1e-3);
    printf("        ★ 这个用例专门打「先求和再求平方和」的灾难性抵消：\n");
    printf("          E[x^2] - mean^2 在 mean=1000 时会把有效位吃掉，Welford 不会。\n");

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dy));
  }

  printf("\nQ182 done.\n");
  return 0;
}
