// q177_reduce.cu — Q177：warp 内蝶形归约 + block 两级归约
//
// 这是 §7 里**最常被要求白板写出来**的一题，因为它是 softmax / norm / dot 的公共底座。
//
//   1) warp_reduce_sum<W>   —— XOR 蝶形，R 轮后【每个 lane 都持有全和】
//   2) warp_reduce_max<W>   —— 同构，哨兵换成 -FLT_MAX
//   3) block_reduce_sum<T>  —— warp 内蝶形 -> smem -> 各 warp 冗余二次归约 -> 广播
//   4) block_reduce_sum_v2  —— 对照实现：只让 warp 0 做二次归约（少算但也少广播）
//
// 编译：./build.sh q177
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 为什么用 __shfl_xor_sync 而不是 __shfl_down_sync？
//    down 模式 lane i 每轮读 lane i+m，**越界 lane 读自己（复制）**；
//    R 轮后只有 lane 0 凑齐全和，其余 lane 拿着被污染的部分和；
//    而且每轮真正有效的 lane 数逐轮减半，**硬件却在为全 warp 买单**。
//    xor 模式三个性质：配对对称 (i^m)^m = i；每轮一对一（m≠0 时 i^m≠i）；
//    距离逐轮减半（16→8→4→2→1）而信息量翻倍。
//    ⇒ R 轮后每个 lane 都拿到完整结果（all-reduce 语义，不是 reduce 到 lane 0）。
//
// ② ★ 步数账（T=256，W=8 个 warp）
//    log2(32)（warp 内 5 轮 shuffle）
//  + 1（smem 写 + 一次 __syncthreads）
//  + log2(W)=3（二次归约）
//  = 8 轮 + 1 次同步
//    smem 流量只有 W 次写 + W 次读（各 4 B），相比"所有中间量走 smem"的朴素树形
//    归约（log2(T)=8 轮 smem 往返）**降了一个量级** —— 这是"能在寄存器层完成的
//    绝不下放到 smem 层"的直接应用。
//
// ③ ★ 本实现的精髓：二次归约不做在 warp 0，而是【每个 warp 的前 W 个 lane 都
//    读入全部 W 个部分和、各自再归约一遍】。多余的 W-1 份计算白扔（反正是寄存器
//    加法），换来**不需要第二次 smem 往返与第二次同步**。
//
// ④ ★ 广播那一步不能省：归约函数的契约是"block 内全员可见"。
//    后续 softmax 除以 ℓ、norm 除以 σ 都依赖它。省掉的话只有每个 warp 的前 W 个
//    lane 持有正确结果 —— 而且在小例子里可能"碰巧"是对的（warp 0 的 lane 恰好对）。
//
// ⑤ 网格形状的坑：`__shfl_xor_sync(0xffffffff, ...)` 要求**参与线程都在**。
//    如果 kernel 里有 `if (idx < n) return;` 这种提前退出，mask 就得跟着改，
//    否则是未定义行为。本文件的 kernel 里**没有提前 return**，边界线程携带单位元
//    参与归约，所以 mask 用 0xffffffff 是安全的。
//
// ⑥ 但要注意：**一个 block 内的线程数必须是 32 的倍数**（否则最后一个不满的 warp
//    要单独处理）。本文件用 T=256 固定，规避了这个问题；生产代码要么保证整除，
//    要么让部分 warp 携带单位元。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① warp 内蝶形归约
// ---------------------------------------------------------------------------
template <int kWarpWidth = 32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int mask = kWarpWidth >> 1; mask >= 1; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, mask, kWarpWidth);
  }
  return v;
}

template <int kWarpWidth = 32>
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int mask = kWarpWidth >> 1; mask >= 1; mask >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, mask, kWarpWidth));
  }
  return v;
}

// ---------------------------------------------------------------------------
// ② block 两级归约（推荐写法：各 warp 冗余二次归约 + 广播）
// ---------------------------------------------------------------------------
template <int kNumThreads = 256>
__device__ __forceinline__ float block_reduce_sum(float v) {
  constexpr int kNumWarps = kNumThreads / 32;
  __shared__ float shared[kNumWarps];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) shared[warp] = v;   // 每个 warp 只需 lane 0 写
  __syncthreads();                   // ★ 不能省：smem 写与读之间

  // 每个 warp 的前 kNumWarps 个 lane 读全部部分和，各自再归约一遍
  v = (lane < kNumWarps) ? shared[lane] : 0.0f;   // 越界 lane 给【加法单位元】
  v = warp_reduce_sum<kNumWarps>(v);
  return __shfl_sync(0xffffffffu, v, 0, 32);      // ★ 广播给全员
}

template <int kNumThreads = 256>
__device__ __forceinline__ float block_reduce_max(float v) {
  constexpr int kNumWarps = kNumThreads / 32;
  __shared__ float shared[kNumWarps];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  v = warp_reduce_max(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();

  v = (lane < kNumWarps) ? shared[lane] : -FLT_MAX;  // ★ max 的单位元
  v = warp_reduce_max<kNumWarps>(v);
  return __shfl_sync(0xffffffffu, v, 0, 32);
}

// ---------------------------------------------------------------------------
// ③ 对照组：二次归约只留给 warp 0（结果只用于写出，不广播）
//    与推荐版的差别：少算了 W-1 份，但需要第二次 __syncthreads 才能让
//    lane 0 拿到结果并写出 —— 这就是"用计算换同步"的取舍。
// ---------------------------------------------------------------------------
template <int kNumThreads = 256>
__device__ __forceinline__ float block_reduce_sum_warp0(float v, float* out_smem) {
  constexpr int kNumWarps = kNumThreads / 32;
  __shared__ float shared[kNumWarps];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) shared[warp] = v;
  __syncthreads();

  if (warp == 0) {
    v = (lane < kNumWarps) ? shared[lane] : 0.0f;
    v = warp_reduce_sum<kNumWarps>(v);
    if (lane == 0) *out_smem = v;     // 只写给 tid==0
  }
  __syncthreads();                    // ★ 第二次同步：让写出可见
  return *out_smem;
}

// ---------------------------------------------------------------------------
// 用 reduce 实现两个真实算子：block 求和、block 求 max
// ---------------------------------------------------------------------------
template <int kNumThreads = 256>
__global__ void kernel_block_sum(const float* __restrict__ x,
                                 float* __restrict__ y, int n) {
  float v = 0.0f;
  for (int i = blockIdx.x * kNumThreads + threadIdx.x; i < n;
       i += gridDim.x * kNumThreads) {
    v += x[i];
  }
  float s = block_reduce_sum<kNumThreads>(v);
  if (threadIdx.x == 0) y[blockIdx.x] = s;
}

// ★ 这个 kernel 演示一个经典错误：用 warp0 版做"全员可见"的归约
//   —— 它只保证 tid==0 拿到结果，其它线程拿到的是 smem 里的旧值。
//   为了对比，我们仍然把它写完整，但只让 tid==0 使用结果。
template <int kNumThreads = 256>
__global__ void kernel_block_sum_warp0(const float* __restrict__ x,
                                       float* __restrict__ y, int n) {
  __shared__ float s_out;
  float v = 0.0f;
  for (int i = blockIdx.x * kNumThreads + threadIdx.x; i < n;
       i += gridDim.x * kNumThreads) {
    v += x[i];
  }
  float s = block_reduce_sum_warp0<kNumThreads>(v, &s_out);
  if (threadIdx.x == 0) y[blockIdx.x] = s;
}

// 演示 max 归约的用法：求一行的最大值（softmax 的第一步）
template <int kNumThreads = 256>
__global__ void kernel_block_max(const float* __restrict__ x,
                                 float* __restrict__ y, int n) {
  // 边界线程必须携带 -FLT_MAX，否则会把"没参与"当成 0 混进 max
  float v = (threadIdx.x < n) ? x[threadIdx.x] : -FLT_MAX;
  float m = block_reduce_max<kNumThreads>(v);
  if (threadIdx.x == 0) y[blockIdx.x] = m;   // ★ tid==0，不是 lane==0
}

// ---------------------------------------------------------------------------
// host 侧
// ---------------------------------------------------------------------------
int main() {
  TEST_BEGIN("Q177 warp/block reduction");

  // ---- 测试 1：一个 warp 的蝶形（32 个数）----
  {
    const int n = 32;
    std::vector<float> h(n);
    fill_random(h, 7u, -1.f, 1.f);
    float ref = std::accumulate(h.begin(), h.end(), 0.0f);

    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    // 复用 kernel_block_sum：grid=1 时它正好做一次 warp 内归约
    CUDA_CHECK(cudaMemset(d, 0, 0));
    float* out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, sizeof(float)));
    kernel_block_sum<32><<<1, 32>>>(d, out, n);
    CUDA_CHECK_KERNEL();
    std::vector<float> got = to_host(out, 1);

    // 32 项求和的 fp32 舍入误差在 1e-6 量级；用相对容差 1e-5
    CompareResult r = compare(got, std::vector<float>{ref}, 1e-5, 1e-6);
    report("warp butterfly (n=32)", r, 1e-5, 1e-6);
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(out));
  }

  // ---- 测试 2：block 归约 vs CPU（多个 block、非整除的 n）----
  for (int n : {256, 1000, 4096, 100000, 1000003}) {
    const int T = 256;
    const int blocks = 64;
    std::vector<float> h(n);
    fill_random(h, 11u, -1.f, 1.f);

    float *dx, *dy, *dy2;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy2, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    kernel_block_sum<T><<<blocks, T>>>(dx, dy, n);
    CUDA_CHECK_KERNEL();
    kernel_block_sum_warp0<T><<<blocks, T>>>(dx, dy2, n);
    CUDA_CHECK_KERNEL();

    std::vector<float> got = to_host(dy, blocks);
    std::vector<float> got2 = to_host(dy2, blocks);
    double sum = 0;
    for (float v : got) sum += v;
    double sum2 = 0;
    for (float v : got2) sum2 += v;
    double ref = std::accumulate(h.begin(), h.end(), 0.0);

    // 归约是并行的 ⇒ 累加顺序与 CPU 不同 ⇒ 只能用容差，不能逐位比
    double rel = std::fabs(sum - ref) / (std::fabs(ref) + 1e-12);
    printf("  %-34s %s  gpu=%.6f cpu=%.6f rel=%.2e\n",
           ("block sum  n=" + std::to_string(n)).c_str(),
           rel < 1e-4 ? "PASS" : "FAIL", sum, ref, rel);
    printf("  %-34s %s  gpu=%.6f cpu=%.6f rel=%.2e  (warp0 变体，数值应当一致)\n",
           "  └ warp0 variant", std::fabs(sum2 - ref) / (std::fabs(ref) + 1e-12) < 1e-4
                                    ? "PASS" : "FAIL",
           sum2, ref, std::fabs(sum2 - ref) / (std::fabs(ref) + 1e-12));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dy2));
  }

  // ---- 测试 3：max 归约（含单位元正确性）----
  for (int n : {1, 100, 256, 1000}) {
    std::vector<float> h(256, 0.0f);          // 先全 0；只有前 n 个填负值
    std::mt19937 gen(n);
    std::uniform_real_distribution<float> d(-100.f, -1.f);   // ★ 全负，专门验哨兵
    for (int i = 0; i < n; ++i) h[i] = d(gen);
    float ref = -FLT_MAX;
    for (int i = 0; i < n; ++i) ref = std::max(ref, h[i]);

    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, h.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
    kernel_block_max<256><<<1, 256>>>(dx, dy, n);
    CUDA_CHECK_KERNEL();
    std::vector<float> got = to_host(dy, 1);
    // 全负的输入是"哨兵写错就立刻暴露"的用例：如果哨兵用 0，结果会变成 0
    CompareResult r = compare(got, std::vector<float>{ref}, 0.0, 0.0);
    report(("block max  n=" + std::to_string(n) + " (all negative)").c_str(),
           r, 0.0, 0.0);
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
  }

  printf("\nQ177 done.\n");
  return 0;
}
