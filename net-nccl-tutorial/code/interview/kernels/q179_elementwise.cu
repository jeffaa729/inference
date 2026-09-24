// q179_elementwise.cu — Q179：ReLU / elementwise 融合，以及"为什么用 fmaxf 不用 if"
//
//   1) relu_branchy   —— 写成 if/else（错误示范：warp 内会发散）
//   2) relu           —— fmaxf(0, x)（编译成单条 max.f32，无分支）
//   3) relu_vec4      —— float4 版
//   4) fused_bias_gelu—— 广播 bias + GELU 融合（演示"融合省的是流量"）
//   5) unfused         —— 分成两步做（bias 一个 kernel、gelu 一个 kernel），做对照
//
// 编译：./build.sh q179
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 分支要写进【指令选择】而不是【控制流】：
//    `if (x > 0) y = x; else y = 0;` 会让 warp 因 lane 间条件不同而发散 ——
//    硬件两条路径都执行、用掩码屏蔽，等价于串行化（最坏 32 路）。
//    `fmaxf(0.0f, x)` 编译成一条 `max.f32`，没有跳转、没有掩码、没有分支预测。
//    ★ 同一个道理在 tanh/gelu 上就是：用精确的数学函数或多项式近似，而不是分段 if。
//
// ② ★ 融合的收益来自【少一次全局往返】，不是"少一次 launch"：
//    unfused：读 x、写 t；读 t、读 bias、写 y   ⇒ 4 读 + 2 写
//    fused  ：读 x、读 bias、写 y              ⇒ 2 读 + 1 写
//    在 memory-bound 的逐元素算子上，**流量减半 ⇒ 时间减半**。
//    但注意★：如果 kernel 已经贴满带宽，融合"省 launch"是没有杠杆的
//    （书的原话：此时只有减流量才有用）。
//
// ③ 广播下标 `idx % hidden` 是最廉价的写法（bias 会被 L1/L2 命中）。
//    如果 hidden 是编译期常量，做成模板参数能让编译器把它变成位运算/常量折叠。
//
// ④ 测试要点：
//    - ReLU 是精确运算（没有舍入），可以用 err == 0 的严格断言；
//    - bias 广播下标写错是**静默 bug**（形状合法、结果错），所以要专门用
//      hidden 不整除 n 的用例来测；
//    - GELU 用 erff 精确版本，容差取 1e-6（fp32 单精度）。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① 错误示范：分支版 ReLU
// ---------------------------------------------------------------------------
__global__ void relu_branchy(const float* __restrict__ x, float* __restrict__ y,
                             int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // ★ 同一 warp 内 idx 相邻，x[idx] 的符号随机 ⇒ 50% 概率发散
    if (x[idx] > 0.0f)
      y[idx] = x[idx];
    else
      y[idx] = 0.0f;
  }
}

// ---------------------------------------------------------------------------
// ② 正确写法：fmaxf
// ---------------------------------------------------------------------------
__global__ void relu(const float* __restrict__ x, float* __restrict__ y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = fmaxf(0.0f, x[idx]);   // 一条 max.f32，零分支
}

// ---------------------------------------------------------------------------
// ③ vec4 版（二段式守卫）
// ---------------------------------------------------------------------------
__global__ void relu_vec4(const float* __restrict__ x, float* __restrict__ y,
                          int n) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 3 < n) {
    float4 v = FLOAT4(x + idx);
    v.x = fmaxf(0.0f, v.x);
    v.y = fmaxf(0.0f, v.y);
    v.z = fmaxf(0.0f, v.z);
    v.w = fmaxf(0.0f, v.w);
    FLOAT4(y + idx) = v;
  } else if (idx < n) {
    for (int i = 0; idx + i < n; ++i) y[idx + i] = fmaxf(0.0f, x[idx + i]);
  }
}

// ---------------------------------------------------------------------------
// ④ 融合：y = gelu(x + bias[col])，一个 kernel 走完
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gelu_exact(float t) {
  // 精确版（erf），与 PyTorch 的 gelu(approximate='none') 对齐
  return 0.5f * t * (1.0f + erff(t * 0.70710678118654752440f));
}

__global__ void fused_bias_gelu(const float* __restrict__ x,
                                const float* __restrict__ bias,
                                float* __restrict__ y, int n, int hidden) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float t = x[idx] + __ldg(bias + (idx % hidden));   // 广播 + 融合
    y[idx] = gelu_exact(t);
  }
}

// ---------------------------------------------------------------------------
// ⑤ 对照：分成两步（两个 kernel）—— 观察流量翻倍带来的差异
// ---------------------------------------------------------------------------
__global__ void add_bias(const float* __restrict__ x,
                         const float* __restrict__ bias,
                         float* __restrict__ t, int n, int hidden) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) t[idx] = x[idx] + __ldg(bias + (idx % hidden));
}

__global__ void gelu_only(const float* __restrict__ t, float* __restrict__ y,
                          int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = gelu_exact(t[idx]);
}

// ---------------------------------------------------------------------------
// host 侧参考
// ---------------------------------------------------------------------------
int main() {
  TEST_BEGIN("Q179 ReLU / elementwise fusion");

  const int T = 256;

  // ---- ReLU：三个版本 + 严格断言（ReLU 无舍入）----
  {
    const int n = 1 << 20;
    std::vector<float> hx(n), href(n), hgot(n);
    fill_random(hx, 5u, -2.f, 2.f);
    for (int i = 0; i < n; ++i) href[i] = std::max(0.0f, hx[i]);

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer;
    const size_t bytes = static_cast<size_t>(n) * 2 * sizeof(float);  // 读 + 写

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_b = timer.bench([&] { relu_branchy<<<(n + T - 1) / T, T>>>(dx, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report("relu_branchy (if/else)", compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_b);

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_f = timer.bench([&] { relu<<<(n + T - 1) / T, T>>>(dx, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report("relu (fmaxf)", compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_f);

    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_v = timer.bench([&] { relu_vec4<<<(n + 4 * 64 - 1) / (4 * 64), 64>>>(dx, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    report("relu_vec4", compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_v);

    printf("        带宽口径 8 B/元素: branchy %.0f | fmaxf %.0f | vec4 %.0f GB/s "
           "(峰值 %.0f)\n",
           to_gbps(bytes, ms_b), to_gbps(bytes, ms_f), to_gbps(bytes, ms_v),
           dev.theoretical_gbps);
    printf("        ★ 注意：在本机（L2 口径、数据 4MB 落在 L2 内）三者可能都在噪声内 ——\n");
    printf("          这正说明【瓶颈不在指令而是带宽】；把 n 放大到远超 L2 再看会有区别。\n");

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }

  // ---- 融合 vs 不融合 ----
  {
    const int seq = 4096, hidden = 1024;   // ★ hidden 不整除 seq，专门测广播下标
    const int n = seq * hidden;
    std::vector<float> hx(n), hb(hidden), href(n), hgot(n);
    fill_random(hx, 6u, -3.f, 3.f);
    fill_random(hb, 7u, -0.5f, 0.5f);
    for (int i = 0; i < n; ++i) {
      float t = hx[i] + hb[i % hidden];
      href[i] = 0.5f * t * (1.0f + std::erf(t * 0.70710678118654752440));
    }

    float *dx, *db, *dy, *dt;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, hidden * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dt, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), hidden * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer;
    const int blocks = (n + T - 1) / T;

    // fused：读 x + 读 bias + 写 y
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_fused = timer.bench([&] { fused_bias_gelu<<<blocks, T>>>(dx, db, dy, n, hidden); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    CompareResult rf = compare(hgot, href, 1e-6, 1e-6);
    report("fused bias+gelu", rf, 1e-6, 1e-6, ms_fused);

    // unfused：读 x/bias、写 t；读 t、写 y
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    double ms_unfused = timer.bench([&] {
      add_bias<<<blocks, T>>>(dx, db, dt, n, hidden);
      gelu_only<<<blocks, T>>>(dt, dy, n);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, n);
    CompareResult ru = compare(hgot, href, 1e-6, 1e-6);
    report("unfused (2 kernels)", ru, 1e-6, 1e-6, ms_unfused);

    size_t b_fused = static_cast<size_t>(n) * 2 * sizeof(float) + hidden * sizeof(float);
    size_t b_unfused = static_cast<size_t>(n) * 4 * sizeof(float) + hidden * sizeof(float);
    printf("        流量口径: fused %.1f MB / unfused %.1f MB（差 %.2fx）\n",
           b_fused / 1e6, b_unfused / 1e6, double(b_unfused) / b_fused);
    printf("        带宽: fused %.0f GB/s | unfused %.0f GB/s（按各自流量算）\n",
           to_gbps(b_fused, ms_fused), to_gbps(b_unfused, ms_unfused));
    printf("        ⇒ 融合的收益来自【少一次全局往返】，这里流量比 %.2fx，"
           "实测时间比 %.2fx\n",
           double(b_unfused) / b_fused, ms_unfused / (ms_fused > 0 ? ms_fused : 1));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dt));
  }

  printf("\nQ179 done.\n");
  return 0;
}
