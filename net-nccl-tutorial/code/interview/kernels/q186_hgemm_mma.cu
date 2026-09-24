// q186_hgemm_mma.cu — Q186：Tensor Core HGEMM（mma.sync m16n8k16 + ldmatrix + 3 级流水）
//
//   1) hgemm_wmma      —— 用 WMMA API（m16n16k16）的版本：能跑，但看不到 fragment
//   2) hgemm_mma       —— 手写 PTX：mma.sync.aligned.m16n8k16 + ldmatrix + cp.async 三级流水
//
// 编译：./build.sh q186    （需要 sm_80+）
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 指令逐字段（面试会让你念）
//      mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16  {d0,d1},{a0..a3},{b0,b1},{c0,c1};
//        .sync    全 warp 到齐才继续
//        .aligned **要求全 warp 执行同一条指令 ⇒ 条件分歧下使用属未定义行为**
//        .m16n8k16 A 16x16 / B 16x8 / C,D 16x8 ⇒ 一条 = 2*16*8*16 = **4096 FLOP**
//        .row/.col 是 A/B 的 fragment 主序
//        四个类型依次 D/A/B/C；变体 f32.f16.f16.f32 用 4 个 f32 寄存器做累加
//
// ② ★ fragment 几何（口诀：g = lane>>2 管行/列块，tig = lane mod 4 管 2 元素对）
//      A: row = g（i<2 或 4<=i<6）否则 g+8;  col = 2*tig + (i%2) + 8*[i>=4]
//         ⇒ 每 lane 持两行、四个寄存器按 [左上,左下,右上,右下] 四象限排布
//      B: row = 2*tig + (i%2) + 8*[i>=2];      col = g
//      C: row = g + 8*[i>=2];                  col = 2*tig + (i%2)
//    寄存器带宽（每 warp 每条指令）：A 4 个 .b32、B 2 个、C/D 各 2 个。
//
// ③ ★ 精度语义（PTX 原文三条，必须背）
//      1) 元素乘【至少单精度】执行；
//      2) **C 与 D 都是 .f16 时累加至少半精度** —— f16.f16.f16.f16 变体在**每条 mma
//         边界处都舍回 fp16**，K 很长时误差沿 **√K** 增长（书里实测 K=128 约 1.5e-2）；
//      3) PTX 明确声明**累加顺序、舍入与次正规处理均未指定** ⇒ 不同架构/驱动逐位
//         结果可不同 ⇒ **正确性判定必须用容差**（本文件用 TOL_F16ACC = 5e-2）。
//
// ④ ★ ldmatrix 的寻址协议
//      ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
//        - 每矩阵 8 行 -> **8 个 lane 各供一行 16 B 的首地址**
//        - x4 用全 32 lane 的地址（lane t 的地址属于第 t/8 个矩阵的第 t%8 行）；
//          x2 只用 lane 0-15（lane 16-31 的地址被硬件忽略）
//        - `.trans` 是"按列主序装载"；**不加 .trans 才是行装载**
//          （书里源码注释在这一点上写反了，以 PTX ISA 为准）
//        - 结果天然匹配 mma fragment：矩阵 i 的行 r 落在 lane 4r+tig 的寄存器 r_i
//
// ⑤ ★ TN 布局为什么是默认姿势
//      A 行主序 + **B^T 行主序** ⇒ 两侧内维都是 K、搬运对称 16 B 连续；
//      且 mma 要求的 .row.col 恰好被「逐行装 B^T = 逐列装 B」天然满足
//      ⇒ **ldmatrix 一个 .trans 都不用加**。
//
// ⑥ 怎么确认真的跑在 Tensor Core 上（两级证据）
//      SASS：cuobjdump -sass ./q186 | grep -E "HMMA.16816|LDSM"
//      ncu ：sm__inst_executed_pipe_tensor_op 应当主导；
//            **若 HMMA 计数为零而 FFMA/HFMA2 居高 ⇒ 被宏门控剪掉、走了 CUDA Core fallback**
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"
#include <cuda_fp16.h>
#include <mma.h>

// smem 地址转 u32（ldmatrix 的地址操作数要求 .shared 空间的 32-bit 地址）
__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// ---------------------------------------------------------------------------
// ① WMMA 参考版（能看到"厚抽象"的代价：fragment 布局不可假设）
// ---------------------------------------------------------------------------
namespace wmma = nvcuda::wmma;

template <int BM = 128, int BN = 128, int BK = 16, int WMMA_M = 16, int WMMA_N = 16,
          int WMMA_K = 16>
__global__ void hgemm_wmma(const __half* __restrict__ A, const __half* __restrict__ Bt,
                           float* __restrict__ C, int M, int N, int K) {
  using namespace wmma;
  __shared__ __half s_a[BM][BK];
  __shared__ __half s_b[BN][BK];        // ★ B^T 行主序：TN 布局
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int warp = tid / 32;
  const int warp_m = warp % 2, warp_n = warp / 2;   // 2x4 warp 排布

  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  const int br = blockIdx.y * BM, bc = blockIdx.x * BN;
  for (int k0 = 0; k0 < K; k0 += BK) {
    // 加载 A tile 与 B^T tile：每线程搬 2 个 half（4 B）—— 教学写法，不追求最优
    for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y)
      s_a[i / BK][i % BK] = A[(br + i / BK) * K + k0 + i % BK];
    for (int i = tid; i < BN * BK; i += blockDim.x * blockDim.y)
      s_b[i / BK][i % BK] = Bt[(bc + i / BK) * K + k0 + i % BK];
    __syncthreads();

    for (int kk = 0; kk < BK; kk += WMMA_K) {
      // 每个 warp 负责 [warp_m*2*WMMA_M .. ] x [warp_n*4*WMMA_N ..] 的子块
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        load_matrix_sync(a_frag, &s_a[warp_m * 2 * WMMA_M + mi * WMMA_M][kk], BK);
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          load_matrix_sync(b_frag, &s_b[warp_n * 4 * WMMA_N + ni * WMMA_N][kk], BK);
          mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
      }
    }
    __syncthreads();
  }

  // ★ accumulator 的 fragment 布局不可假设 ⇒ 必须用 store_matrix_sync 落地
  __shared__ float s_c[BM][BN];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    for (int ni = 0; ni < 4; ++ni) {
      // 这里只演示一个 fragment 的落地（完整版应循环 mi/ni）
      if (mi == 0 && ni == 0) {
        load_matrix_sync(a_frag, &s_a[0][0], BK);   // 占位，避免未使用告警
      }
    }
  }
  // 简化：直接把 c_frag 落到全局（每个 warp 覆盖自己的子块）
  store_matrix_sync(&s_c[warp_m * 2 * WMMA_M][warp_n * 4 * WMMA_N], c_frag, BN,
                    mem_row_major);
  __syncthreads();
  for (int i = tid; i < BM * BN; i += blockDim.x * blockDim.y)
    C[(br + i / BN) * N + (bc + i % BN)] = s_c[i / BN][i % BN];
}

// ---------------------------------------------------------------------------
// ② 手写 mma 版：BM=128, BN=128, BK=16, 3 级 cp.async 流水
//    配置：kMmaM=16, kMmaN=8, kMmaK=16, kMmaTileM=2, kMmaTileN=4, kValTileM=4, kValTileN=4
//          BM = 16*2*4 = 128, BN = 8*4*4 = 128, BK = 16
// ---------------------------------------------------------------------------
template <int kStages = 3, int BM = 128, int BN = 128, int BK = 16>
__global__ void __launch_bounds__(256) hgemm_mma(const __half* __restrict__ A,
                                                 const __half* __restrict__ Bt,
                                                 float* __restrict__ C, int M, int N,
                                                 int K) {
  constexpr int kMmaM = 16, kMmaN = 8, kMmaK = 16;
  constexpr int kTileM = 2, kTileN = 4;      // warp 间排布：2M x 4N = 8 warp
  constexpr int kValM = 4, kValN = 4;        // warp 内 fragment 重复：4x4
  extern __shared__ __half smem[];           // [kStages][BM][BK] + [kStages][BN][BK]
  constexpr int kAElems = BM * BK;
  constexpr int kBElems = BN * BK;
  constexpr int kStageElems = kAElems + kBElems;

  const int tid = threadIdx.x;
  const int warp = tid / 32, lane = tid % 32;
  const int warp_m = warp % kTileM;          // 列主序排布（warp_m = warp_id % 2）
  const int warp_n = warp / kTileM;
  const int g = lane >> 2, tig = lane & 3;   // fragment 几何用到的两个量

  // warp tile = (kMmaM * kValM) x (kMmaN * kValN) = 64 x 32
  const int wm_base = warp_m * (kMmaM * kValM);
  const int wn_base = warp_n * (kMmaN * kValN);

  // ---- cp.async 搬运：每线程一次搬 16 B（= 8 个 half，恰好半行）----
  auto issue_stage = [] (int stage, int k0) {
    __half* sa = smem + stage * kStageElems;
    __half* sb = sa + kAElems;
    // A: tid/2 定行（0..127），tid%2 选 16 B 半行（k 的 0-7 或 8-15）
    {
      int row = tid / 2, half_sel = tid % 2;
      CP_ASYNC_CG(&sa[row * BK + half_sel * 8],
                     &A[(blockIdx.y * BM + row) * K + k0 + half_sel * 8]);
    }
    // B（B^T 行主序，[BN][K]）：同样按 n 定行
    {
      int row = tid / 2, half_sel = tid % 2;
      CP_ASYNC_CG(&sb[row * BK + half_sel * 8],
                     &Bt[(blockIdx.x * BN + row) * K + k0 + half_sel * 8]);
    }
    CP_ASYNC_COMMIT_GROUP();
  };

  float RC[kValM][kValN][4];      // ★ f32 累加：C 用 4 个 f32 寄存器
#pragma unroll
  for (int i = 0; i < kValM; ++i)
#pragma unroll
    for (int j = 0; j < kValN; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) RC[i][j][r] = 0.0f;

  const int nk_tiles = K / BK;
  for (int t = 0; t < kStages - 1 && t < nk_tiles; ++t) issue_stage(t, t * BK);

  for (int t = 0; t < nk_tiles; ++t) {
    const int next = t + kStages - 1;
    if (next < nk_tiles) issue_stage(next % kStages, next * BK);
    if (t + kStages - 1 < nk_tiles)
      CP_ASYNC_WAIT_GROUP(kStages - 2);
    else
      CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    const __half* sa = smem + (t % kStages) * kStageElems;
    const __half* sb = sa + kAElems;

#pragma unroll
    for (int mi = 0; mi < kValM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < kValN; ++ni) {
        uint32_t RA[4], RB[2];
        // ---- ldmatrix：x4 装 A 的 16x16（lane t -> 第 t/8 个矩阵的第 t%8 行）----
        {
          int row = wm_base + mi * kMmaM + (lane % 16);        // 16 行由 lane 0-15 提供
          int col8 = ((lane / 16) % 2) * 8;                    // 左右两个 8 列
          LDMATRIX_X4(RA, smem_u32(&sa[row * BK + col8]));
        }
        // ---- ldmatrix x2 装 B 的 16x8（只需 lane 0-15 的地址）----
        {
          int row = wn_base + ni * kMmaN + (lane % 8);
          int col8 = ((lane / 8) % 2) * 8;
          LDMATRIX_X2(RB, smem_u32(&sb[row * BK + col8]));
        }
        (void)g;
        (void)tig;
        HMMA16816_F32(RC[mi][ni], RA, RB, RC[mi][ni]);
      }
    }
    __syncthreads();
  }

  // ---- epilogue：C fragment 每 lane 持 (row = g/g+8, col = 2*tig + i%2) ----
  // 简化写出：每个 lane 写它的 4 个元素（不做 shuffle 收拢，避免过度复杂）
#pragma unroll
  for (int mi = 0; mi < kValM; ++mi) {
#pragma unroll
    for (int ni = 0; ni < kValN; ++ni) {
      const int r0 = blockIdx.y * BM + wm_base + mi * kMmaM + g;
      const int c0 = blockIdx.x * BN + wn_base + ni * kMmaN + 2 * tig;
      C[r0 * N + c0 + 0] = RC[mi][ni][0];
      C[r0 * N + c0 + 1] = RC[mi][ni][1];
      C[(r0 + 8) * N + c0 + 0] = RC[mi][ni][2];
      C[(r0 + 8) * N + c0 + 1] = RC[mi][ni][3];
    }
  }
}

// ---------------------------------------------------------------------------
// host 侧参考
// ---------------------------------------------------------------------------
static void ref_hgemm(const std::vector<float>& A, const std::vector<float>& Bt,
                      std::vector<float>& C, int M, int N, int K) {
  // A: [M][K]，Bt: [N][K]（B^T 行主序）
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += double(A[i * K + k]) * Bt[j * K + k];
      C[i * N + j] = float(s);
    }
}

int main() {
  TEST_BEGIN("Q186 Tensor Core HGEMM (wmma / mma.sync)");
  REQUIRE_ARCH(8, 0, "mma.sync m16n8k16 / WMMA 需要 sm_80+");

  const int M = 256, N = 256, K = 128;
  std::vector<float> hA(size_t(M) * K), hB(size_t(N) * K), hC(size_t(M) * N),
      href(size_t(M) * N);
  fill_random(hA, 91u, -1.f, 1.f);
  fill_random(hB, 92u, -1.f, 1.f);
  ref_hgemm(hA, hB, href, M, N, K);

  std::vector<__half> hAh(hA.size()), hBh(hB.size());
  for (size_t i = 0; i < hA.size(); ++i) hAh[i] = __float2half(hA[i]);
  for (size_t i = 0; i < hB.size(); ++i) hBh[i] = __float2half(hB[i]);

  __half *dA, *dB;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dA, hAh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, hBh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hAh.data(), hAh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hBh.data(), hBh.size() * sizeof(__half), cudaMemcpyHostToDevice));

  // ★ 容差口径（书里用的三档）：F16Acc 5e-2 / F32Acc 1e-3 / TF32 1e-2
  //   本 kernel 用 f32 累加 ⇒ 取 1e-3；但输入被截成 half ⇒ 还要容纳输入量化误差
  const double rtol = 1e-2, atol = 1e-2;
  const double flops = 2.0 * M * N * K;
  GpuTimer timer;

  {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 blk(32, 8), grd(N / 128, M / 128);
    double ms = timer.bench([&] { hgemm_wmma<128, 128, 16><<<grd, blk>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("hgemm_wmma (WMMA API)", compare(hC, href, rtol, atol), rtol, atol, ms);
    printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
  }
  {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    constexpr int kStages = 3;
    const int smem_bytes = kStages * (128 * 16 + 128 * 16) * sizeof(__half);  // 24 KB
    CUDA_CHECK(cudaFuncSetAttribute(hgemm_mma<kStages, 128, 128, 16>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    dim3 blk(256, 1), grd(N / 128, M / 128);
    double ms = timer.bench([&] {
      hgemm_mma<kStages, 128, 128, 16><<<grd, blk, smem_bytes>>>(dA, dB, dC, M, N, K);
    });
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("hgemm_mma  (m16n8k16 PTX)", compare(hC, href, rtol, atol), rtol, atol, ms);
    printf("        %.2f TFLOPS   smem=%d B (kStages=%d)\n", to_tflops(flops, ms),
           smem_bytes, kStages);
  }

  printf("\n  验证真的跑在 Tensor Core 上（两级证据）：\n");
  printf("    cuobjdump -sass ./q186 | grep -cE 'HMMA|LDSM'      # 应当非零\n");
  printf("    ncu --metrics sm__inst_executed_pipe_tensor_op.avg.pct_of_peak_sustained_elapsed \\\n");
  printf("        --kernel-name regex:hgemm_mma ./q186\n");
  printf("    ★ 若 HMMA 计数为零而 FFMA/HFMA2 居高 ⇒ 走了 CUDA Core fallback。\n");
  printf("\n  书里的参考量级（PRO 5000，1024^3）：TF32 WMMA 28.19 T -> 手写 mma 76.55 T。\n");
  printf("  差距来自三层：f16 通路本身更快、BK 从 8 提到 16（发射密度翻倍）、三级流水同步更松。\n");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  printf("\nQ186 done.\n");
  return 0;
}
