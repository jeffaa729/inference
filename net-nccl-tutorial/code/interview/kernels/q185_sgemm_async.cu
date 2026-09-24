// q185_sgemm_async.cu — Q185：软件流水（cp.async 多级 stage）
//
//   1) sgemm_sync      —— 同步搬运：加载 -> __syncthreads -> 计算 -> __syncthreads
//   2) sgemm_async_s2  —— cp.async + 2 级 stage
//   3) sgemm_async_s3  —— cp.async + 3 级 stage（稳态允许 1 组在途）
//
// 编译：./build.sh q185
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① cp.async 三件套的语义（★ 必背）
//      cp.async.cg.shared.global [smem], [gmem], 16;
//        - 从 global 直接拷到 shared，**绕过寄存器**
//        - `.cg` 只走 L2、不分配 L1，**16 B 是 `.cg` 唯一合法宽度**
//          （`.ca` 支持 4/8/16 B 且会分配 L1 —— 用 `.cg` 是为了不污染 L1）
//      cp.async.commit_group;
//        - 把**本线程**未提交的拷贝打包成一个组
//      cp.async.wait_group N;
//        - 阻塞到**最多剩 N 组未完成**
//        - ★ N 是"允许保留的未完成组数"，**不是"第 N 组完成"**
//        - 建立期/排空期用 0；稳态满载用 kStages - 2
//
// ② ★ 为什么 wait 之后必须再跟一次 __syncthreads()（这是本题最常考的一问）
//      cp.async 的组是 **per-thread** 的：`wait_group` 只保证「本线程发出的搬运」完成，
//      而计算要读的 smem **可能由别的线程搬入**。
//      ⇒ 漏掉这一次同步就是 warp 间数据竞争，**症状是偶发错数**（不是稳定错）。
//
// ③ 双缓冲的时序数学
//      单缓冲：T_step = T_ld + T_cp
//      双缓冲：T_step = max(T_ld, T_cp)          ← 稳态
//      S 级流水：允许 in-flight 的搬运组数为 S - 2
//      ⇒ "加深 stage" 的收益在 S 超过 (搬运延迟 / 迭代时间) 之后归零。
//      书里的实测佐证：FA2 TMA+WS 版 **Sk=2 即饱和**（Sk 2->3 收益 < 0.5%），
//      而 Sk=3 的 smem 增长反而挤掉 2 blocks/SM 的驻留可能。
//
// ④ 本文件用"统一循环"（k 从 0 起跳、带边界条件）而不是"主循环 + 尾部排空"：
//      好处是消除尾部重复代码块，代价是两个分支判断；
//      对 K 不是 kStages 倍数的形状更稳健。
//
// ⑤b ★ 与 LeetCUDA `kernels/sgemm/sgemm_async.cu` 的对照（读源码时别困惑）：
//      - 那份实现也是 cp.async 双缓冲，但 **wait_group 的参数恒为 0**，stage 数固定为 2，
//        换 stage 的写法是 `smem_sel = (bk - 1) & 1; smem_sel_next = bk & 1;`；
//        它**没有**做到"稳态只等 kStages-2 组"，所以流水深度其实没有真正拉开。
//      - 它还**只把 B 走 cp.async**，A 仍然走 LDG -> 寄存器 -> smem（顺带做 online 转置）。
//        本文件两个都走 cp.async（更接近书里 hgemm 的做法），A 不做转置。
//      - 它**没有动态 smem**（全部静态分配）；本文件用 `extern __shared__` +
//        `cudaFuncSetAttribute`，因为 S=3 的 96 KB 已经超过 48 KB 的默认上限。
//      ⇒ 所以"cp.async 双缓冲"这四个字下面至少有三种不同实现，面试时说清是哪一种。
//
// ⑤ smem 账：A 是 [BM][BK]、B 是 [BK][BN]，两个都是 4 B/元素。
//      BM=BN=128, BK=32 时每 stage = (128*32 + 32*128) * 4 = 32 KB；
//      S=2 要 64 KB、S=3 要 96 KB —— **必须 cudaFuncSetAttribute opt-in**
//      （默认每 block 动态 smem 上限只有 48 KB，超了直接 launch 报 invalid argument）。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① 同步搬运（基准）
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 32, int TM = 4, int TN = 4>
__global__ void sgemm_sync(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int a_m = tid / 8, a_k = (tid % 8) * 4;
  const int b_k = tid / 32, b_n = (tid % 32) * 4;
  const int c_m = (tid / 32) * TM, c_n = (tid % 32) * TN;
  const int br = blockIdx.y * BM, bc = blockIdx.x * BN;

  float sum[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) sum[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    *reinterpret_cast<float4*>(&s_a[a_m][a_k]) =
        *reinterpret_cast<const float4*>(&A[(br + a_m) * K + k0 + a_k]);
    *reinterpret_cast<float4*>(&s_b[b_k][b_n]) =
        *reinterpret_cast<const float4*>(&B[(k0 + b_k) * N + bc + b_n]);
    __syncthreads();                       // 加载完 -> 才能算
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
    __syncthreads();                       // 算完 -> 才能覆写
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
    *reinterpret_cast<float4*>(&C[(br + c_m + i) * N + bc + c_n]) =
        make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
}

// ---------------------------------------------------------------------------
// ② / ③ cp.async 多级流水（kStages 用模板参数）
// ---------------------------------------------------------------------------
template <int kStages, int BM = 128, int BN = 128, int BK = 32, int TM = 4, int TN = 4>
__global__ void sgemm_async(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
  extern __shared__ float smem[];
  // 每 stage 的 A 与 B 布局：A[BM][BK]、B[BK][BN]
  constexpr int kAElems = BM * BK;
  constexpr int kBElems = BK * BN;
  constexpr int kStageElems = kAElems + kBElems;
  float* s = smem;   // s[stage * kStageElems + ...]

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y;
  const int c_m = (tid / 32) * TM, c_n = (tid % 32) * TN;
  const int br = blockIdx.y * BM, bc = blockIdx.x * BN;
  constexpr int nk = 0;   // (占位，避免误用)

  float sum[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) sum[i][j] = 0.0f;

  // 一个 stage 的搬运：每个线程负责若干个 16 B（4 个 float）
  // A：BM*BK/4 个 16B 块；B：BK*BN/4 个 16B 块
  auto issue_stage = [] (int stage, int k0) {
    float* sa = s + stage * kStageElems;
    float* sb = sa + kAElems;
    // A 的行主序：第 m 行、第 k 列 -> 线性 m*BK + k
    for (int idx = tid * 4; idx < kAElems; idx += nthreads * 4) {
      int m = idx / BK, k = idx % BK;
      CP_ASYNC_CG(&sa[idx], &A[(br + m) * K + k0 + k]);
    }
    // B 的行主序：[k][n]，线性 k*BN + n
    for (int idx = tid * 4; idx < kBElems; idx += nthreads * 4) {
      int k = idx / BN, n = idx % BN;
      CP_ASYNC_CG(&sb[idx], &B[(k0 + k) * N + bc + n]);
    }
    CP_ASYNC_COMMIT_GROUP();
  };

  const int nk_tiles = K / BK;
  // 预载 kStages-1 个 stage
  for (int t = 0; t < kStages - 1 && t < nk_tiles; ++t) {
    issue_stage(t, t * BK);
  }

  for (int t = 0; t < nk_tiles; ++t) {
    const int cur = t % kStages;
    // 统一循环：先补发下一块（若还在范围内），再等 + 算
    const int next = t + kStages - 1;
    if (next < nk_tiles) issue_stage(next % kStages, next * BK);

    // 满载期等 (kStages-2) 组；排空期等 0 组
    if (t + kStages - 1 < nk_tiles) {
      CP_ASYNC_WAIT_GROUP(kStages - 2);
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();          // ★ per-thread 的组 + 跨线程可见性

    const float* sa = s + cur * kStageElems;
    const float* sb = sa + kAElems;
    for (int k = 0; k < BK; ++k) {
      float ra[TM], rb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = sa[(c_m + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = sb[k * BN + c_n + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) sum[i][j] = fmaf(ra[i], rb[j], sum[i][j]);
    }
    __syncthreads();          // ★ 算完才能让下一轮覆写这个 stage
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
    *reinterpret_cast<float4*>(&C[(br + c_m + i) * N + bc + c_n]) =
        make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
}

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
  TEST_BEGIN("Q185 software pipeline with cp.async");

  const int M = 512, N = 512, K = 512;
  std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N), hC(size_t(M) * N), href(size_t(M) * N);
  fill_random(hA, 81u, -1.f, 1.f);
  fill_random(hB, 82u, -1.f, 1.f);
  ref_sgemm(hA, hB, href, M, N, K);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  const double flops = 2.0 * M * N * K;
  const double rtol = 1e-4, atol = 1e-4;
  GpuTimer timer;
  dim3 blk(32, 32), grd(N / 128, M / 128);

  // 每个 stage 的字节数（动态 smem）
  constexpr int kAElems = 128 * 32, kBElems = 32 * 128;
  const int stage_bytes = (kAElems + kBElems) * 4;   // = 32 KB

  {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    double ms = timer.bench([&] { sgemm_sync<128, 128, 32, 4, 4><<<grd, blk>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_sync (no pipeline)", compare(hC, href, rtol, atol), rtol, atol, ms);
    printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
  }

  // S=2：动态 smem = 64 KB > 48 KB 默认上限 ⇒ 必须 opt-in
  {
    const int smem_bytes = stage_bytes * 2;
    CUDA_CHECK(cudaFuncSetAttribute(sgemm_async<2, 128, 128, 32, 4, 4>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    printf("  cp.async S=2 需要动态 smem %d B (opt-in 上限 %d B)\n", smem_bytes,
           dev.smem_per_block_optin);
    if (smem_bytes > dev.smem_per_block_optin) {
      printf("  SKIP: 超过本机 opt-in 上限\n");
    } else {
      CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
      double ms = timer.bench([&] {
        sgemm_async<2, 128, 128, 32, 4, 4><<<grd, blk, smem_bytes>>>(dA, dB, dC, M, N, K);
      });
      CUDA_CHECK_KERNEL();
      hC = to_host(dC, hC.size());
      report("sgemm_async S=2 (wait<0>)", compare(hC, href, rtol, atol), rtol, atol, ms);
      printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
    }
  }

  // S=3：96 KB，进一步考验 opt-in 上限
  {
    const int smem_bytes = stage_bytes * 3;
    printf("  cp.async S=3 需要动态 smem %d B\n", smem_bytes);
    if (smem_bytes > dev.smem_per_block_optin) {
      printf("  SKIP: 超过本机 opt-in 上限（%d B）—— 这正是「stage 不是越深越好」的物理上限\n",
             dev.smem_per_block_optin);
    } else {
      CUDA_CHECK(cudaFuncSetAttribute(sgemm_async<3, 128, 128, 32, 4, 4>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
      CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
      double ms = timer.bench([&] {
        sgemm_async<3, 128, 128, 32, 4, 4><<<grd, blk, smem_bytes>>>(dA, dB, dC, M, N, K);
      });
      CUDA_CHECK_KERNEL();
      hC = to_host(dC, hC.size());
      report("sgemm_async S=3 (wait<1>)", compare(hC, href, rtol, atol), rtol, atol, ms);
      printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
      printf("        ★ 注意 stage 加深会挤占 occupancy：S=3 的 96 KB 让每 SM 只能驻留 1 个\n");
      printf("          block（本机 smem/SM 约 %d B），stage 的收益可能被 occupancy 抵消。\n",
             dev.smem_per_block_optin);
    }
  }

  printf("\n  用 ncu 验证流水是否真的生效：\n");
  printf("    ncu --metrics smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio \\\n");
  printf("        --kernel-name regex:sgemm_ --launch-count 2 ./q185\n");
  printf("    cp.async 版应当看到 long_scoreboard 明显下降（装载不再裸露）。\n");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  printf("\nQ185 done.\n");
  return 0;
}
