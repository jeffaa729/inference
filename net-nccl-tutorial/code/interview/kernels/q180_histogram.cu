// q180_histogram.cu — Q180：histogram（朴素全局原子 vs smem 私有化）
//
//   1) hist_global —— 每次命中直接 atomicAdd 到全局（最朴素）
//   2) hist_smem   —— block 内先在 smem 上计数，最后每 block 对全局做 B 次合并写
//
// 编译：./build.sh q180
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① 原子操作的代价模型：
//      atomicAdd 是 RMW（读-改-写），在 **L2 的原子处理单元**上排队执行，
//      **同地址串行、不同地址并行**。
//      设一次全局原子延迟约 τ（书里实测约 2.6 ns）：
//        - 全部 N 次砸向同一个 bin  → 吞吐被钉死在 1/τ
//        - B 个 bin 均匀承压        → 理论上限约 B/τ op/s
//
// ② ★ smem 私有化的"次数账"（最容易答漏的一点）：
//      全局原子总次数 = G x B（G = block 数，B = bin 数）
//      **仅当 T > B 时才低于 N**（T = 每 block 线程数）。
//      T <= B 时次数不变！收益来自：
//        (a) **smem 原子远快于全局原子**（smem 在片上，且冲突域只在 block 内）
//        (b) 写模式去热点（B 个 bin 分散到 B 个地址）
//    所以正确说法不是"原子次数变少了"，而是"原子变便宜了 + 冲突域缩小了"。
//
// ③ 书里的实测（RTX PRO 5000，N=4,194,304，256 bins）：
//      elementwise_add 0.01897 ms / 2653.73 GB/s
//      histogram       0.62860 ms /   26.69 GB/s   ← 慢两个数量级
//    按"256 bin 理想并行 + 同址 2.6 ns"模型，理想耗时 ≈ 4.2M/256 x 2.6ns ≈ 42 µs，
//    **实测 0.63 ms 慢了约 15 倍** —— 缺口来自 4M 个原子请求在 **L1 侧 atomic
//    路径**的排队与串行化。这正是"模型能算出的下界"与"实际硬件路径"的差距。
//
// ④ 两个必须注意的正确性问题：
//      (a) **复用缓冲区必须每轮清零**（atomicAdd 是加在旧值上）—— 用 cudaMemset；
//      (b) 输入值的范围必须严格 < B，否则 atomicAdd 写到数组外（越界写，比越界读更危险）。
//
// ⑤ 为什么不用寄存器/局部数组先数再合并？
//    每个线程先数自己那段、再用 __syncthreads 合并，本质是 smem 私有化的一种变体；
//    直接对 smem 做原子在 GPU 上已经足够快（smem 原子由硬件在片上完成）。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① 朴素：全局原子
// ---------------------------------------------------------------------------
__global__ void hist_global(const unsigned char* __restrict__ a,
                            int* __restrict__ y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) atomicAdd(&y[a[idx]], 1);
}

// ---------------------------------------------------------------------------
// ② smem 私有化
// ---------------------------------------------------------------------------
template <int kBins = 256, int kThreads = 256>
__global__ void hist_smem(const unsigned char* __restrict__ a,
                          int* __restrict__ y, int n) {
  __shared__ int cnt[kBins];

  // 初始化（避免用 memset 之外的隐式假设）
  for (int b = threadIdx.x; b < kBins; b += kThreads) cnt[b] = 0;
  __syncthreads();                      // ★ 必须在任何原子之前

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) atomicAdd(&cnt[a[idx]], 1);   // smem 原子：冲突域只在 block 内

  __syncthreads();                      // ★ 必须在合并写之前
  for (int b = threadIdx.x; b < kBins; b += kThreads) {
    if (cnt[b]) atomicAdd(&y[b], cnt[b]);    // 全局原子次数 = G x B（G 为 block 数）
  }
}

// ---------------------------------------------------------------------------
// ③ 变体：每个线程先在自己的一段里数（寄存器累加），再合并到 smem
//    适合"每线程处理的元素很多"的场景，能进一步减少 smem 原子次数
// ---------------------------------------------------------------------------
template <int kBins = 256, int kThreads = 256, int kItemsPerThread = 16>
__global__ void hist_smem_batched(const unsigned char* __restrict__ a,
                                  int* __restrict__ y, int n) {
  __shared__ int cnt[kBins];
  for (int b = threadIdx.x; b < kBins; b += kThreads) cnt[b] = 0;
  __syncthreads();

  // 每个线程负责连续的 kItemsPerThread 个元素（用 smem 暂存而不是寄存器数组，
  // 因为 B 可能很大；这里为了讲清结构，仍然是逐个原子提交）
  int base = (blockIdx.x * kThreads + threadIdx.x) * kItemsPerThread;
#pragma unroll
  for (int i = 0; i < kItemsPerThread; ++i) {
    int idx = base + i;
    if (idx < n) atomicAdd(&cnt[a[idx]], 1);
  }
  __syncthreads();
  for (int b = threadIdx.x; b < kBins; b += kThreads) {
    if (cnt[b]) atomicAdd(&y[b], cnt[b]);
  }
}

// ---------------------------------------------------------------------------
// host 侧参考
// ---------------------------------------------------------------------------
static void ref_hist(const std::vector<unsigned char>& a, std::vector<int>& y,
                     int bins) {
  std::fill(y.begin(), y.end(), 0);
  for (unsigned char v : a) y[v]++;   // 前提：v < bins
}

int main() {
  TEST_BEGIN("Q180 histogram");

  const int kBins = 256;
  const int n = 1 << 22;          // 4,194,304（与书里同量级）
  const int T = 256;
  const int G = 512;              // 与 dot 的 grid 对齐，便于比较

  // 均匀分布 + 倾斜分布：后者更接近真实场景（少数 bin 扛大部分流量）
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<unsigned char> ha(n);
    if (mode == 0) {
      std::mt19937 gen(21u);
      std::uniform_int_distribution<int> d(0, kBins - 1);
      for (auto& v : ha) v = static_cast<unsigned char>(d(gen));
    } else {
      // 倾斜：80% 的元素落在前 8 个 bin —— 冲突热点更明显
      std::mt19937 gen(22u);
      std::uniform_int_distribution<int> hot(0, 7), cold(8, kBins - 1);
      std::uniform_int_distribution<int> pick(0, 99);
      for (auto& v : ha)
        v = static_cast<unsigned char>(pick(gen) < 80 ? hot(gen) : cold(gen));
    }

    std::vector<int> href(kBins), hgot(kBins);
    ref_hist(ha, href, kBins);

    unsigned char* da = nullptr;
    int* dy = nullptr;
    CUDA_CHECK(cudaMalloc(&da, n));
    CUDA_CHECK(cudaMalloc(&dy, kBins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), n, cudaMemcpyHostToDevice));

    GpuTimer timer;
    const int blocks = (n + T - 1) / T;

    // --- 朴素：每轮必须清零 ---
    CUDA_CHECK(cudaMemset(dy, 0, kBins * sizeof(int)));
    double ms_g = timer.bench([&] { hist_global<<<blocks, T>>>(da, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, kBins);
    report(mode == 0 ? "hist_global (uniform)" : "hist_global (skewed)",
           compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_g);

    // --- smem 私有化 ---
    CUDA_CHECK(cudaMemset(dy, 0, kBins * sizeof(int)));
    double ms_s = timer.bench([&] { hist_smem<kBins, T><<<G, T>>>(da, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, kBins);
    report(mode == 0 ? "hist_smem   (uniform)" : "hist_smem   (skewed)",
           compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_s);

    // --- 每线程多元素 ---
    CUDA_CHECK(cudaMemset(dy, 0, kBins * sizeof(int)));
    const int items = 16;
    const int blocks_b = (n + T * items - 1) / (T * items);
    double ms_b = timer.bench([&] { hist_smem_batched<kBins, T, items><<<blocks_b, T>>>(da, dy, n); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dy, kBins);
    report(mode == 0 ? "hist_batched(uniform)" : "hist_batched(skewed)",
           compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_b);

    // 带宽口径：只读 1 B/元素（原子不计入流量）
    size_t bytes = n;
    printf("        %.0f MB 输入，带宽口径 1 B/元素: global %.1f GB/s | smem %.1f | "
           "batched %.1f\n",
           n / 1e6, to_gbps(bytes, ms_g), to_gbps(bytes, ms_s), to_gbps(bytes, ms_b));
    printf("        耗时比 smem/global = %.2fx，batched/global = %.2fx\n",
           ms_s / ms_g, ms_b / ms_g);
    if (mode == 0) {
      printf("        ★ 对照：教科书说 smem 私有化能快一个数量级，但**倾斜分布下"
             "收益会缩小**\n           （热点 bin 的 smem 原子同样会串行化）。\n");
    }

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(dy));
  }

  printf("\nQ180 done.\n");
  return 0;
}
