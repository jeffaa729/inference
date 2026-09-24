// q176_vecadd.cu — Q176：向量加法 c = a + b
//
// 三个版本，对应面试时的三步递进：
//   1) vecadd_naive      —— 一线程一元素 + 边界守卫（先写对的）
//   2) vecadd_vec4       —— float4 主路径 + 标量尾巴（二段式守卫）
//   3) vecadd_gridstride —— grid-stride loop（尾块不再需要单独处理，且 grid 可调）
//
// 编译：./build.sh q176
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点（面试官真正在看的东西）
// ─────────────────────────────────────────────────────────────────────────────
// ① 为什么 vecadd_naive 的守卫写成 `if (idx < n)`，而 vec4 版要写成两段？
//    vec4 版一次读 4 个 float。如果只写 `if (idx < n)`，当 n 不是 4 的倍数时
//    最后一个线程会读到数组外 —— **越界读不报错**，只是结果里混进垃圾。
//    二段式守卫 `if (idx + 3 < n) {...} else if (idx < n) {...}` 才是安全的。
//    （这是本书第 4/9 章反复强调的坑：k32/k128 的 GEMV 就因为没有守卫而静默越界。）
//
// ② 为什么 4 个 float 一定对齐？（float4 要求 16 B 对齐）
//    cudaMalloc 返回的基址满足 256 B 对齐；只要元素偏移是 4 的倍数就天然满足。
//    本文件里 idx = 4 * 全局线程号，所以成立。**传子数组（如 a+1）就会炸**。
//
// ③ vec4 到底省了什么？
//    省的是【load/store 指令数】（4 条 32-bit -> 1 条 128-bit），不是字节数。
//    Volta 起访存以 32 B sector 为粒度，coalescing 场景下 sector 数由总数据量决定
//    —— 所以"向量化把事务数减为 1/4"这个老说法已经过时（口径要纠正）。
//    什么时候真的快？当瓶颈在指令发射（小 shape、grid 受限）或 HBM 直读时。
//
// ④ grid-stride 的价值：固定 grid 大小（比如 = SM 数 × 每 SM 的 block 数），
//    每个线程处理多个元素。好处是 grid 与 n 解耦、尾块不需特判、且能控制占用率。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// 版本 1：一线程一元素
// ---------------------------------------------------------------------------
__global__ void vecadd_naive(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}

// ---------------------------------------------------------------------------
// 版本 2：float4，主路径 + 标量尾巴
// ---------------------------------------------------------------------------
__global__ void vecadd_vec4(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c, int n) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx + 3 < n) {  // 主路径：一次 16 B（= 4 个 sector 中的 4 个 float）
    float4 x = FLOAT4(a + idx);
    float4 y = FLOAT4(b + idx);
    float4 z = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    FLOAT4(c + idx) = z;
  } else if (idx < n) {  // 尾巴：最多 3 个元素，标量收
    for (int i = 0; idx + i < n; ++i) c[idx + i] = a[idx + i] + b[idx + i];
  }
}

// ---------------------------------------------------------------------------
// 版本 3：grid-stride loop
// ---------------------------------------------------------------------------
__global__ void vecadd_gridstride(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
  int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
    c[idx] = a[idx] + b[idx];
  }
}

// ---------------------------------------------------------------------------
// host 侧参考实现
// ---------------------------------------------------------------------------
static void ref_vecadd(const std::vector<float>& a, const std::vector<float>& b,
                       std::vector<float>& c) {
  for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
}

int main() {
  TEST_BEGIN("Q176 vector add (naive / vec4 / grid-stride)");

  // 关键：既要测 4 的倍数，也要测非 4 的倍数 —— 后者才是二段式守卫的考点
  const std::vector<int> sizes = {1, 3, 4, 5, 1021, 1024, 1 << 20, (1 << 20) + 3};

  for (int n : sizes) {
    std::vector<float> ha(n), hb(n), href(n), hgot(n);
    fill_random(ha, 1u, -1.f, 1.f);
    fill_random(hb, 2u, -1.f, 1.f);
    ref_vecadd(ha, hb, href);

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int T = 256;
    GpuTimer timer;

    // --- naive ---
    CUDA_CHECK(cudaMemset(dc, 0, n * sizeof(float)));
    double ms_naive = timer.bench([&] {
      vecadd_naive<<<(n + T - 1) / T, T>>>(da, db, dc, n);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dc, n);
    report(("naive        n=" + std::to_string(n)).c_str(),
           compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_naive);

    // --- vec4（block 用 64：4*64 = 256 元素/block，与 naive 的 256 元素/block 对齐）---
    CUDA_CHECK(cudaMemset(dc, 0, n * sizeof(float)));
    const int T4 = 64;
    double ms_vec4 = timer.bench([&] {
      vecadd_vec4<<<(n + 4 * T4 - 1) / (4 * T4), T4>>>(da, db, dc, n);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dc, n);
    report(("vec4         n=" + std::to_string(n)).c_str(),
           compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_vec4);

    // --- grid-stride（grid 固定成 4 倍 SM 数）---
    CUDA_CHECK(cudaMemset(dc, 0, n * sizeof(float)));
    int gs_blocks = dev.sm_count * 4;
    double ms_gs = timer.bench([&] {
      vecadd_gridstride<<<gs_blocks, T>>>(da, db, dc, n);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dc, n);
    report(("grid-stride  n=" + std::to_string(n)).c_str(),
           compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_gs);

    // 大 shape 才报带宽（小 shape 的耗时被 launch 开销主导，带宽没有意义）
    if (n >= (1 << 20)) {
      // 元素级加法：读 a + 读 b + 写 c = 12 B/元素
      size_t bytes = static_cast<size_t>(n) * 3 * sizeof(float);
      printf("        带宽口径 12 B/元素: naive %.0f GB/s | vec4 %.0f GB/s | "
             "grid-stride %.0f GB/s (理论峰值 %.0f)\n",
             to_gbps(bytes, ms_naive), to_gbps(bytes, ms_vec4),
             to_gbps(bytes, ms_gs), dev.theoretical_gbps);
    }

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
  }

  printf("\nQ176 done.\n");
  return 0;
}
