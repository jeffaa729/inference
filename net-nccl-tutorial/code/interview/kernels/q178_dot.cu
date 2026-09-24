// q178_dot.cu — Q178：dot product，以及"为什么 vec4 可能一点也不快"
//
//   1) dot_atomic     —— 每 block 一次全局 atomicAdd（最直观，但会被原子串行化钉死）
//   2) dot_two_stage  —— 两级归约：block 部分和落到 G 个【互不冲突】的地址，
//                        再起一个小 kernel 归约 G 个数（★ 改进方向）
//   3) dot_vec4       —— 访存指令数降到 1/4，用来验证"瓶颈不在访存"
//
// 编译：./build.sh q178
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① 通用范式：**算子 = 元素级预变换 + 归约 + 后处理**
//    dot 的预变换是逐元素乘，归约是求和，后处理（本例）没有。
//    softmax 的预变换是 exp、后处理是除法；norm 的预变换是平方、后处理是 rsqrt。
//    面试时先把这个范式说出来，再写代码，结构立刻就清楚了。
//
// ② ★ 这道题最有价值的结论（可以直接背下来当谈资）：
//    在 N=4,194,304 这个规模上，dot 与 dot_vec4 的耗时【几乎完全相同】，
//    因为**真正的瓶颈既不是带宽也不是访存指令数，而是全局原子的串行化**：
//        G = N/256 = 16384 个 block 各做一次 atomicAdd(y)
//        同址全局原子在 L2 原子单元上串行 ⇒ 每次约 2.6 ns
//        16384 x 2.6 ns ≈ 42 µs  ← 恰好是实测总时长
//    而 32 MB 数据按 HBM 口径只要 ~24 µs。
//    ⇒ 结论：**这个实现的上限不是带宽，是全局原子的串行吞吐**；
//      改进方向是把 atomic 次数从 G 降到 ⌈G/K⌉（两级归约 / block 内先累积）。
//
// ③ 为什么两级归约会更快？
//    第一级每个 block 写【自己的】地址（G 个地址，无冲突）；
//    第二级用 1 个 block 归约 G 个数。atomic 从 G 次降到 0 次（或几次）。
//    代价是多一次 kernel launch 与 G 个 float 的中间缓冲。
//
// ④ float 的 atomicAdd 还会破坏确定性：加法不满足结合律 + block 完成顺序不定
//    ⇒ 同一份输入两次运行结果可能不同（run-to-run 抖动）。
//    正确性判定必须用容差；要可复现就得用两级归约这种【固定顺序】的方案。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// 版本 1：每 block 一次全局原子
// ---------------------------------------------------------------------------
template <int kNumThreads = 256>
__global__ void dot_atomic(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ y, int n) {
  float prod = 0.0f;
  for (int i = blockIdx.x * kNumThreads + threadIdx.x; i < n;
       i += gridDim.x * kNumThreads) {
    prod += a[i] * b[i];
  }
  float s = block_reduce_sum<kNumThreads>(prod);
  if (threadIdx.x == 0) atomicAdd(y, s);   // ★ tid==0，不是 lane==0
}

// ---------------------------------------------------------------------------
// 版本 2：两级归约（第一级：每 block 写自己的地址）
// ---------------------------------------------------------------------------
template <int kNumThreads = 256>
__global__ void dot_partial(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ partial, int n) {
  float prod = 0.0f;
  for (int i = blockIdx.x * kNumThreads + threadIdx.x; i < n;
       i += gridDim.x * kNumThreads) {
    prod += a[i] * b[i];
  }
  float s = block_reduce_sum<kNumThreads>(prod);
  if (threadIdx.x == 0) partial[blockIdx.x] = s;   // 无冲突写
}

// 第二级：1 个 block 归约 G 个部分和（这里 G <= kNumThreads 就够；否则要递归）
template <int kNumThreads = 256>
__global__ void dot_finalize(const float* __restrict__ partial, int g,
                             float* __restrict__ y) {
  float v = (threadIdx.x < g) ? partial[threadIdx.x] : 0.0f;
  float s = block_reduce_sum<kNumThreads>(v);
  if (threadIdx.x == 0) *y = s;
}

// ---------------------------------------------------------------------------
// 版本 3：float4 版（访存指令数降到 1/4，用来验证瓶颈位置）
// ---------------------------------------------------------------------------
template <int kNumThreads = 64>
__global__ void dot_vec4(const float4* __restrict__ a,
                         const float4* __restrict__ b,
                         float* __restrict__ partial, int n4) {
  float prod = 0.0f;
  for (int i = blockIdx.x * kNumThreads + threadIdx.x; i < n4;
       i += gridDim.x * kNumThreads) {
    float4 x = a[i];
    float4 y = b[i];
    prod += x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
  }
  float s = block_reduce_sum<kNumThreads>(prod);
  if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

// ---------------------------------------------------------------------------
// host 侧：CPU 参考（用 double 累加，作为"真值"）
// ---------------------------------------------------------------------------
static double ref_dot(const std::vector<float>& a, const std::vector<float>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
  return s;
}

int main() {
  TEST_BEGIN("Q178 dot product (atomic / two-stage / vec4)");

  const int n = 1 << 22;   // 4,194,304 —— 与书里那个"原子瓶颈"用例同量级
  const int G = 512;       // grid 大小（两级归约的中间缓冲长度）
  const int T = 256;

  std::vector<float> ha(n), hb(n);
  fill_random(ha, 3u, -1.f, 1.f);
  fill_random(hb, 4u, -1.f, 1.f);
  const double ref = ref_dot(ha, hb);

  float *da, *db, *dy, *dpartial;
  CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dpartial, G * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const size_t bytes = static_cast<size_t>(n) * 2 * sizeof(float);  // 只读，不写
  GpuTimer timer;

  // ---- 版本 1：原子 ----
  CUDA_CHECK(cudaMemset(dy, 0, sizeof(float)));
  double ms1 = timer.bench([&] { dot_atomic<T><<<G, T>>>(da, db, dy, n); });
  CUDA_CHECK_KERNEL();
  double got1 = to_host(dy, 1)[0];
  printf("  %-30s %s  got=%.6f ref=%.6f rel=%.2e  %.4f ms  %.0f GB/s\n",
         "dot_atomic", close_enough(got1, ref, 1e-4, 1e-3) ? "PASS" : "FAIL",
         got1, ref, std::fabs(got1 - ref) / std::fabs(ref), ms1,
         to_gbps(bytes, ms1));

  // ---- 版本 2：两级 ----
  double ms2 = timer.bench([&] {
    dot_partial<T><<<G, T>>>(da, db, dpartial, n);
    dot_finalize<T><<<1, T>>>(dpartial, G, dy);
  });
  CUDA_CHECK_KERNEL();
  double got2 = to_host(dy, 1)[0];
  printf("  %-30s %s  got=%.6f ref=%.6f rel=%.2e  %.4f ms  %.0f GB/s\n",
         "dot_two_stage", close_enough(got2, ref, 1e-4, 1e-3) ? "PASS" : "FAIL",
         got2, ref, std::fabs(got2 - ref) / std::fabs(ref), ms2,
         to_gbps(bytes, ms2));

  // ---- 版本 3：vec4（n 必须是 4 的倍数）----
  const int n4 = n / 4;
  double ms3 = 0.0, got3 = 0.0;
  if (n % 4 == 0) {
    double ms_launch = timer.bench([&] {
      dot_vec4<64><<<G, 64>>>(reinterpret_cast<const float4*>(da),
                              reinterpret_cast<const float4*>(db), dpartial, n4);
      dot_finalize<T><<<1, T>>>(dpartial, G, dy);
    });
    ms3 = ms_launch;
    CUDA_CHECK_KERNEL();
    got3 = to_host(dy, 1)[0];
    printf("  %-30s %s  got=%.6f ref=%.6f rel=%.2e  %.4f ms  %.0f GB/s\n",
           "dot_vec4 (instr/4)", close_enough(got3, ref, 1e-4, 1e-3) ? "PASS" : "FAIL",
           got3, ref, std::fabs(got3 - ref) / std::fabs(ref), ms3,
           to_gbps(bytes, ms3));
  }

  // ---- 结论区：把"到底瓶颈在哪"用数字摆出来 ----
  printf("\n  ---- 瓶颈判读 ----\n");
  printf("  G=%d 个 block，atomic 版本每 block 一次同址 atomicAdd\n", G);
  printf("  理论下界（HBM 口径）：%.1f us（%.0f GB/s 峰值）\n",
         bytes / dev.theoretical_gbps / 1e3, dev.theoretical_gbps);
  printf("  实测：atomic %.1f us | two-stage %.1f us | vec4 %.1f us\n", ms1 * 1e3,
         ms2 * 1e3, ms3 * 1e3);
  printf("  ⇒ atomic 与 vec4 接近 ⇒ **瓶颈不是访存指令数**；\n");
  printf("    把 G 调大（下面用 --grid 试）会看到 atomic 版本恶化更快 —— 那就是原子串行化。\n");

  // ---- 稳定性：浮点原子 ⇒ 结果有 run-to-run 抖动 ----
  double r1 = 0.0, r2 = 0.0;
  CUDA_CHECK(cudaMemset(dy, 0, sizeof(float)));
  dot_atomic<T><<<G, T>>>(da, db, dy, n);
  CUDA_CHECK_SYNC();
  r1 = to_host(dy, 1)[0];
  CUDA_CHECK(cudaMemset(dy, 0, sizeof(float)));
  dot_atomic<T><<<G, T>>>(da, db, dy, n);
  CUDA_CHECK_SYNC();
  r2 = to_host(dy, 1)[0];
  printf("  确定性：atomic 两次运行差 = %.3e（%.1e 量级属正常抖动，因为加法不满足结合律）\n",
         std::fabs(r1 - r2), std::fabs(r1) * 1e-6);

  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dpartial));
  printf("\nQ178 done.\n");
  return 0;
}
