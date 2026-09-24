// q183_sgemm.cu — Q183：tiled SGEMM（shared memory + thread tile + vec4）
//
// 阶梯（对应书里第十章的三级）：
//   0) sgemm_naive  —— 直译：AI = 0.25 FLOP/B 且与 K 无关（常数！）
//   1) sgemm_tile   —— Level 1：Block Tile 进 smem，AI 抬到 BM*BN/(2(BM+BN))
//   2) sgemm_vec4   —— Level 1+：加载用 float4，搬运指令数砍到 1/4；
//                      每线程算 TM x TN = 4x4 的 thread tile
//
// 编译：./build.sh q183
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 三个 AI 公式（必须能手推，面试必问）：
//      naive:       AI = 2K / 8K = 0.25 FLOP/B      ← 分子分母同乘 K 相消，是【常数】
//      block tile:  AI = 2*BM*BN*BK / (4*BK*(BM+BN)) = BM*BN / (2(BM+BN))
//                   ★ BK 在分子分母相消 ⇒ AI 由 BM x BN 唯一决定（32^2 -> 8，128^2 -> 32）
//      thread tile: AI_smem = 2*TM*TN / (4*(TM+TN)) = TM*TN / (2(TM+TN))
//                   1x1 -> 0.25，4x4 -> 1.0  ⇒ **每 FMA 的 smem 读从 2 次降到 0.5 次**
//    ⇒ block tile 优化的是 gmem->smem 的复用，thread tile 优化的是 smem->寄存器 的复用，
//      **二者正交、可叠加**。
//
// ② ★ 两道 __syncthreads() 各防什么（必答）：
//      第一道（加载完、计算前）：防"读到没写完的 smem"；
//      第二道（计算完、下一轮加载前）：防"还没读完就被下一轮覆写"。
//    **省掉第二道是最常见的"优化"笔误** —— 症状是偶发错数、随形状与调度变化、难复现。
//
// ③ ★ 索引法则（书里的原话）：
//      `/` 表示线程在该维【不连续】排布，`%` 表示【连续】排布。
//      **需要合并访问的维度必须用 `%`。**
//    本文件里 A 的加载映射是 `a_m = tid / 8, a_k = (tid % 8) * 4`：
//    warp 内 8 个线程一行、每线程搬 4 个 float ⇒ 一行 128 B 恰好合并。
//    如果把 `/` 与 `%` 用反，warp 内地址立刻变成跨步散访，性能掉一个数量级。
//
// ④ 为什么 4x4 的 thread tile：AI_smem 从 0.25 抬到 1.0，
//    而且每线程有 16 条独立的 FMA 链（ILP），更容易填满流水线。
//
// ⑤ ★ 边界约定必须主动说出来：本文件要求
//      M % BM == 0, N % BN == 0, K % BK == 0（naive 版要求 % 32）
//    不满足时**会越界读且越界写**（比 GEMV 的只读越界更危险）。
//    生产代码要么加守卫，要么把形状 pad 到对齐。
//
// ⑥ 实测（书的表 10.1，RTX PRO 5000，方阵）：
//      naive@2048  3.144 ms /  5.46 TFLOPS
//      vec4 @2048  0.661 ms / 25.97 TFLOPS   ← 4.76x
//    naive 被带宽钉死在 ~5.5 T（AI=8 的隐含上限 8 x 934 GB/s ≈ 7.5 T，拿到 73%）；
//    vec4 到 25.97 T = AI=32 上限的 87%，距峰值 66.9 T 还有 61% 缺口
//    ⇒ 这就是"该往 Tensor Core 走"的定量依据。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// Level 0：naive（AI = 0.25）
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// Level 1：Block Tile
// ---------------------------------------------------------------------------
template <int BM = 32, int BN = 32, int BK = 32>
__global__ void sgemm_tile(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * BM + ty;
  const int col = blockIdx.x * BN + tx;

  float sum = 0.0f;
  for (int k0 = 0; k0 < K; k0 += BK) {
    // 协作加载（BM=BN=BK=32 时正好一个线程一个元素）
    s_a[ty][tx] = A[row * K + k0 + tx];
    s_b[ty][tx] = B[(k0 + ty) * N + col];
    __syncthreads();                    // ★ 第一道

    for (int k = 0; k < BK; ++k) sum += s_a[ty][k] * s_b[k][tx];

    __syncthreads();                    // ★ 第二道：防下一轮覆写
  }
  C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// Level 1+：128x128x32 tile + 4x4 thread tile + vec4 加载
//   配置：BM=BN=128, BK=32, block(32,32)=1024 线程，每线程 4x4=16 个 C 元素
//         1024*16 = 16384 = 128*128 ✓
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 32, int TM = 4, int TN = 4>
__global__ void sgemm_vec4(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  __shared__ float s_a[BM][BK];        // 128*32*4 = 16 KB
  __shared__ float s_b[BK][BN];        // 32*128*4 = 16 KB   （合计 32 KB）
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  // ---- 加载映射（与计算映射【解耦】）----
  const int a_m = tid / 8;             // 8 线程一行
  const int a_k = (tid % 8) * 4;       // 每线程搬 4 个 float ⇒ warp 内拼出 128 B
  const int b_k = tid / 32;            // 32 线程一行
  const int b_n = (tid % 32) * 4;      // warp 内拼出 512 B

  // ---- 计算映射：一个 warp 实际承包 4x128 的 C 横带 ----
  const int c_m = (tid / 32) * TM;
  const int c_n = (tid % 32) * TN;
  float sum[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) sum[i][j] = 0.0f;

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // ① 协作加载（vec4：一条 128-bit 指令搬 4 个 float）
    *reinterpret_cast<float4*>(&s_a[a_m][a_k]) =
        *reinterpret_cast<const float4*>(&A[(block_row + a_m) * K + k0 + a_k]);
    *reinterpret_cast<float4*>(&s_b[b_k][b_n]) =
        *reinterpret_cast<const float4*>(&B[(k0 + b_k) * N + block_col + b_n]);
    __syncthreads();

    // ② 计算：每个 k 取 4 个 A + 4 个 B，做 16 次 FMA（只花 8 次 smem 读）
    for (int k = 0; k < BK; ++k) {
      float ra[TM], rb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = s_a[c_m + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = s_b[k][c_n + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) sum[i][j] = fmaf(ra[i], rb[j], sum[i][j]);
    }
    __syncthreads();                   // ★ 第二道
  }

  // ③ 写回：每行一条 float4（要求 N % 4 == 0）
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    float* c_row = &C[(block_row + c_m + i) * N + block_col + c_n];
    FLOAT4(c_row) =
        make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
  }
}

// ---------------------------------------------------------------------------
// host 侧参考：CPU 三重循环（小 shape 用；大 shape 只用来对拍少量元素）
// ---------------------------------------------------------------------------
static void ref_sgemm(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += double(A[i * K + k]) * B[k * N + j];
      C[i * N + j] = float(s);
    }
}

int main() {
  TEST_BEGIN("Q183 tiled SGEMM (naive / tile / tile+vec4+threadtile)");

  // --- 正确性：小 shape（都要能被 128 整除，因为 vec4 版没有边界守卫）---
  {
    const int M = 256, N = 256, K = 128;
    std::vector<float> hA(M * K), hB(K * N), hC(M * N), href(M * N);
    fill_random(hA, 51u, -1.f, 1.f);
    fill_random(hB, 52u, -1.f, 1.f);
    ref_sgemm(hA, hB, href, M, N, K);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 256 个数累加，fp32 的误差在 1e-5 相对量级
    const double rtol = 1e-4, atol = 1e-4;

    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 blk(16, 16);
    dim3 grd((N + 15) / 16, (M + 15) / 16);
    sgemm_naive<<<grd, blk>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_naive 256x256x128", compare(hC, href, rtol, atol), rtol, atol);

    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 blk1(32, 32), grd1(N / 32, M / 32);
    sgemm_tile<32, 32, 32><<<grd1, blk1>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_tile 256x256x128", compare(hC, href, rtol, atol), rtol, atol);

    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 blk2(32, 32), grd2(N / 128, M / 128);
    sgemm_vec4<128, 128, 32, 4, 4><<<grd2, blk2>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_vec4 256x256x128", compare(hC, href, rtol, atol), rtol, atol);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  // --- 性能：方阵阶梯（这是书里表 10.1 的复现口径）---
  printf("\n  ---- 性能阶梯（方阵，warmup 3 + 10 次均值）----\n");
  for (int n : {512, 1024, 2048}) {
    const int M = n, N = n, K = n;
    std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N);
    fill_random(hA, 61u, -1.f, 1.f);
    fill_random(hB, 62u, -1.f, 1.f);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, size_t(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    const double flops = 2.0 * M * N * K;
    GpuTimer timer;
    printf("  n=%4d:\n", n);

    {
      dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
      double ms = timer.bench([&] { sgemm_naive<<<grd, blk>>>(dA, dB, dC, M, N, K); });
      printf("    naive   %8.4f ms  %7.2f TFLOPS\n", ms, to_tflops(flops, ms));
    }
    {
      dim3 blk(32, 32), grd(N / 32, M / 32);
      double ms = timer.bench([&] { sgemm_tile<32, 32, 32><<<grd, blk>>>(dA, dB, dC, M, N, K); });
      printf("    tile32  %8.4f ms  %7.2f TFLOPS  (AI = 8)\n", ms, to_tflops(flops, ms));
    }
    {
      dim3 blk(32, 32), grd(N / 128, M / 128);
      double ms = timer.bench([&] {
        sgemm_vec4<128, 128, 32, 4, 4><<<grd, blk>>>(dA, dB, dC, M, N, K);
      });
      printf("    vec4    %8.4f ms  %7.2f TFLOPS  (AI = 32, thread tile 4x4)\n",
             ms, to_tflops(flops, ms));
      if (n == 512) {
        printf("    ★ 注意 n=512 时 vec4 的 grid 只有 4x4=16 个 block，"
               "而 SM 有 %d 个 —— 大半在围观。\n", dev.sm_count);
        printf("      这就是「tile 尺寸是 AI 与并行度的折中」：大 tile 让 AI 上去，"
               "但也让 block 数变少。\n");
      }
    }
    printf("    → 理论上界: AI=8 时 %.2f TFLOPS, AI=32 时 %.2f TFLOPS "
           "(用 %.0f GB/s 估算)\n",
           8 * dev.theoretical_gbps / 1e3, 32 * dev.theoretical_gbps / 1e3,
           dev.theoretical_gbps);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  printf("\nQ183 done.\n");
  return 0;
}
