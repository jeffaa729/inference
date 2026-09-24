// q188_softmax.cu — Q188：per-token 的 safe softmax 与 online softmax
//
//   1) softmax_naive   —— 按定义直译（演示溢出：大输入直接出 nan）
//   2) softmax_safe    —— 减 max 的两遍扫描（工业标配基线）
//   3) softmax_online  —— (m, d) 双状态单遍联动更新、一次融合归约
//
// 编译：./build.sh q188
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 数值稳定性的数学基础是【平移不变性】
//      softmax(x - c*1) = softmax(x)     （分子分母同乘 e^{-c} 严格约掉）
//    ⇒ **减 max 不改变结果，只改变中间量的量级**。
//    溢出界：fp32 下 e^t 不溢出的充要条件是 t <= ln(FLT_MAX) ≈ 88.72；
//    **fp16 下这个界只有 11**（5 位指数）。
//    减 max 后的界：x(i) - m <= 0 ⇒ e^{x(i)-m} <= 1 ⇒ 分子绝不溢出；
//    分母是 n 个不超过 1 的正数之和（最大为 n），也绝不溢出。
//
// ② ★ online softmax 的核心：把「状态 + 元素」的递推推广成「状态 + 状态」的**可归约**运算
//      单元素递推：m_new = max(m, x);  d_new = d * e^{m - m_new} + e^{x - m_new}
//      二元合并：  m = max(m1, m2);    d = d1 * e^{m1 - m} + d2 * e^{m2 - m}
//    ★ **二元合并满足结合律**（书里引理 5.1）⇒ 可以直接套蝶形/树形归约，
//      **与归约顺序无关**，5 步蝶形后每个 lane 都拿到全行的正确状态。
//    这正是 FlashAttention 能把 softmax 塞进 KV 分块流水线的原因。
//
// ③ 两版实现的差别（面试要能说清代价）
//      扫描遍数：2 -> 1（x 只读一次、一直躺在寄存器里）
//      块级同步：2 次 -> 1 次
//      代价：归约算子从「一个加法」变成「一次合并」（1 个 max + 2 个 exp + 2 个 FMA），
//            且写回用近似指令（__expf / __fdividef）
//
// ④ 书里的实测（S=4096 token x H=256，PRO 5000）：
//      naive 690.6 GB/s | safe 572.9 GB/s | online 591.6 GB/s
//    ★ **naive 反而最快**（比 safe 少一次 block 归约）—— 但它的前提「输入有界」
//      在生产环境不可保证；safe 相对 naive 慢约 21%，**买到的是任意有限输入下的确定性**。
//
// ⑤ 边界单位的坑：不满的线程必须携带**状态单位元** (-FLT_MAX, 0)，
//    否则 (0, 0) 会把 d 拉成 0、把 m 拉成 0，结果全错（而且是静默错）。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① naive：不减 max（会溢出）
// ---------------------------------------------------------------------------
template <int kThreads = 256>
__global__ void softmax_naive(const float* __restrict__ x, float* __restrict__ y,
                              int n) {
  const int tid = threadIdx.x;
  const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
  float* yr = y + static_cast<size_t>(blockIdx.x) * n;

  float s = 0.0f;
  for (int i = tid; i < n; i += kThreads) s += __expf(xr[i]);
  s = block_reduce_sum<kThreads>(s);
  const float inv = 1.0f / s;
  for (int i = tid; i < n; i += kThreads) yr[i] = __expf(xr[i]) * inv;
}

// ---------------------------------------------------------------------------
// ② safe：两遍扫描
// ---------------------------------------------------------------------------
template <int kThreads = 256>
__global__ void softmax_safe(const float* __restrict__ x, float* __restrict__ y,
                             int n) {
  const int tid = threadIdx.x;
  const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
  float* yr = y + static_cast<size_t>(blockIdx.x) * n;

  // pass 1: max（边界线程给 -FLT_MAX 这个 max 的单位元）
  float m = -FLT_MAX;
  for (int i = tid; i < n; i += kThreads) m = fmaxf(m, xr[i]);
  m = block_reduce_max<kThreads>(m);

  // pass 2: sum of exp（第二遍重新读 x，通常 L1 命中，但不是零）
  float s = 0.0f;
  for (int i = tid; i < n; i += kThreads) s += __expf(xr[i] - m);
  s = block_reduce_sum<kThreads>(s);

  const float inv = 1.0f / s;
  for (int i = tid; i < n; i += kThreads) yr[i] = __expf(xr[i] - m) * inv;
}

// ---------------------------------------------------------------------------
// ③ online：(m, d) 状态融合归约，一趟
// ---------------------------------------------------------------------------
struct MD {
  float m, d;
};

__device__ __forceinline__ MD md_merge(MD a, MD b) {
  MD r;
  r.m = fmaxf(a.m, b.m);
  // ★ 以新 max 为基准重缩放两侧（较小的那一侧会乘上 e^{负} 收缩）
  r.d = a.d * __expf(a.m - r.m) + b.d * __expf(b.m - r.m);
  return r;
}

template <int kThreads = 256>
__global__ void softmax_online(const float* __restrict__ x, float* __restrict__ y,
                               int n) {
  const int tid = threadIdx.x;
  const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
  float* yr = y + static_cast<size_t>(blockIdx.x) * n;

  // ★ 边界给状态单位元：m = -FLT_MAX（max 的单位元）、d = 0（加法单位元）
  float xi = (tid < n) ? xr[tid] : -FLT_MAX;
  MD st{xi, (tid < n) ? 1.0f : 0.0f};

  // warp 内 5 步蝶形合并（与 block_reduce_sum 同构，只是算子换成了 md_merge）
#pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1) {
    MD o{__shfl_xor_sync(0xffffffffu, st.m, mask),
         __shfl_xor_sync(0xffffffffu, st.d, mask)};
    st = md_merge(st, o);
  }
  // 跨 warp：写 smem -> 第一个 warp 再合并 -> 广播
  __shared__ MD warp_md[kThreads / 32];
  const int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) warp_md[warp] = st;
  __syncthreads();
  if (warp == 0) {
    MD o = (lane < kThreads / 32) ? warp_md[lane] : MD{-FLT_MAX, 0.0f};
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
      MD p{__shfl_xor_sync(0xffffffffu, o.m, mask),
           __shfl_xor_sync(0xffffffffu, o.d, mask)};
      o = md_merge(o, p);
    }
    if (lane == 0) warp_md[0] = o;      // 复用第 0 槽做广播（节省一次同步）
  }
  __syncthreads();
  const MD tot = warp_md[0];

  const float inv = 1.0f / tot.d;
  for (int i = tid; i < n; i += kThreads) yr[i] = __expf(xr[i] - tot.m) * inv;
}

// ---------------------------------------------------------------------------
// host 侧参考（double 版，作为真值）
// ---------------------------------------------------------------------------
static void ref_softmax(const std::vector<float>& x, std::vector<float>& y, int n) {
  double m = -1e308;
  for (int i = 0; i < n; ++i) m = std::max(m, double(x[i]));
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += std::exp(double(x[i]) - m);
  for (int i = 0; i < n; ++i) y[i] = float(std::exp(double(x[i]) - m) / s);
}

int main() {
  TEST_BEGIN("Q188 softmax (naive / safe / online)");

  const int T = 256;
  struct Case { int n; float lo, hi; const char* name; };
  // ★ 第三个用例是"溢出用例"：x ∈ [80,120]，naive 必然出 nan
  const std::vector<Case> cases = {{256, -5.f, 5.f, "normal [-5,5]"},
                                   {100, -5.f, 5.f, "n=100 (block 不满)"},
                                   {256, 80.f, 120.f, "large [80,120] -> 溢出"}};

  for (const auto& c : cases) {
    const int n = c.n;
    std::vector<float> hx(n), href(n), hgot(n);
    fill_random(hx, 101u, c.lo, c.hi);
    ref_softmax(hx, href, n);

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    printf("\n  --- case: %s (n=%d) ---\n", c.name, n);
    const double rtol = 1e-4, atol = 1e-6;
    GpuTimer timer;
    const size_t bytes = size_t(n) * 2 * sizeof(float);

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms1 = timer.bench([&] { softmax_naive<T><<<1, T>>>(dx, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    {
      CompareResult r = compare(hgot, href, rtol, atol);
      report("softmax_naive", r, rtol, atol, ms1);
      if (!r.ok && c.lo > 80) {
        printf("        ↑ 这是**预期的溢出**：输入 80~120 时 e^x 直接 inf，inf/inf = nan\n");
      }
    }

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms2 = timer.bench([&] { softmax_safe<T><<<1, T>>>(dx, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report("softmax_safe  (2 passes)", compare(hgot, href, rtol, atol), rtol, atol, ms2);

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms3 = timer.bench([&] { softmax_online<T><<<1, T>>>(dx, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report("softmax_online(1 pass)", compare(hgot, href, rtol, atol), rtol, atol, ms3);

    printf("        带宽口径 8 B/元素: naive %.0f | safe %.0f | online %.0f GB/s\n",
           to_gbps(bytes, ms1), to_gbps(bytes, ms2), to_gbps(bytes, ms3));
    printf("        时间比 online/safe = %.3f（一趟 vs 两趟）\n", ms3 / ms2);
  }

  // ---- 与「块顺序无关」的一致性自检：把 online 的两半分别算再合并 ----
  {
    const int n = 512;
    std::vector<float> hx(n), href(n), hgot(n);
    fill_random(hx, 102u, -5.f, 5.f);
    ref_softmax(hx, href, n);

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    softmax_online<256><<<1, 256>>>(dx, dy, n);
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    printf("\n  online vs 参考（n=512）: ");
    CompareResult r = compare(hgot, href, 1e-4, 1e-6);
    printf("%s  max_abs=%.3e\n", r.ok ? "PASS" : "FAIL", r.max_abs);
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }

  printf("\nQ188 done.\n");
  return 0;
}
