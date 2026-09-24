// q184_sgemm_dbuf.cu — Q184：寄存器双缓冲 + XOR swizzle 打散 bank conflict
//
//   1) sgemm_smem_nopad —— 基准：smem tile 不 padding，B 的行 stride = BK
//   2) sgemm_smem_pad   —— padding 版（BK+1），把列方向的 bank 冲突打散
//   3) sgemm_swizzle    —— XOR swizzle 版（零 padding 达到同样的效果）
//
// 编译：./build.sh q184
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 两级双缓冲的分工（这是本题最容易答混的地方）：
//      **smem 多级流水（kStages + cp.async）** 盖的是 gmem -> smem 的【跨 k 迭代】延迟；
//      **寄存器双缓冲（RA/RB 乒乓）**          盖的是 smem -> 寄存器 的【k_step 级】延迟。
//    两者不能互相替代：前者解决"数据还没到"，后者解决"数据到了但取数指令串行"。
//    本文件把两者都写出来（kStages 见 q185，寄存器乒乓见下面的 sgemm_swizzle）。
//
// ② ★ bank conflict 的算术（smem 32 bank x 4 B，bank(a) = (a/4) mod 32）：
//      B 的 smem 布局是 [BK][BN]，读 `s_b[k][c_n + j]`：
//        warp 内 c_n 连续（c_n = (tid % 32) * TN，步长 TN=4）
//        ⇒ 地址 = (k * BN + c_n) * 4
//        ⇒ bank = (k * BN + c_n) mod 32
//      BN = 128 时：bank = (k * 128 + c_n) mod 32 = c_n mod 32
//        ⇒ **k 的贡献消失**！同一个 warp 里 32 个线程的 c_n = 0,4,8,...,124
//        ⇒ bank = 0,4,8,...,28,0,4,... ⇒ **8 路冲突**（步长 4 撞 8 个 bank）。
//      BN + 1 = 129 时：bank = (k * 129 + c_n) mod 32 = (k + c_n) mod 32
//        ⇒ 每个 k 都换一组 bank ⇒ **无冲突**。
//    ★ 这与书里 BK=64（128 B 行宽）8-way 的结论同源：**行 stride 是 bank 环长的
//      整数倍时，行号的贡献就消失了**。
//
// ③ ★ XOR swizzle 的原理与可逆性：
//      Swizzle<B,M,S>: apply(a) = a ^ ((a & yyy) >> S),  yyy = (2^B - 1) << (M + S)
//      即把【chunk 号位】（bit [M, M+B)）与【行号低位】（bit [M+S, M+S+B)）异或。
//      **可逆性来自对合**：设 σ(a) = a ^ m(a)，只要 XOR 的源位域与目标位域不相交
//      （S >= B），就有 σ(σ(a)) = a。
//      ⇒ **编码与解码用同一个函数**，不需要维护逆置换表。
//      （这是书里"引理 13.1"，也是 CuTe Swizzle 的实现基础。）
//
// ④ ★ 别把"swizzle 后零冲突"当结论背：冲突度由**行宽**决定。
//      32 B 行宽 + swizzle<16> = 1-way；128 B 行宽 + swizzle<16> = 4-way；
//      要 1-way 得用 swizzle<64>（c' = c ^ (i & 7)）。
//      本文件里 B 的 smem 行宽是 BN * 4 = 512 B（很大），所以用 padding 更直接；
//      swizzle 版本演示的是"用位运算换 padding"的手法本身。
//
// ⑤ 寄存器双缓冲的骨架（伪码，见 kernel 里的 ld/st 索引）：
//      float frag[2][TM][TN];
//      load(frag[st], k=0);
//      for (ks = 1; ks < K_SLICES; ++ks) { ld ^= 1; st ^= 1; mma(frag[ld]); load(frag[st], ks); }
//      mma(frag[ld]);
//    —— **计算第 ks-1 片的同时发起第 ks 片的装载**，把 ldmatrix/lds 的延迟藏进 MMA。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① 基准：不 padding
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 32, int TM = 4, int TN = 4>
__global__ void sgemm_smem_nopad(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];          // ★ 行 stride = BN = 128 个字
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
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
      float ra[TM], rb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = s_a[c_m + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = s_b[k][c_n + j];   // ★ 冲突在这里
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) sum[i][j] = fmaf(ra[i], rb[j], sum[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
    *reinterpret_cast<float4*>(&C[(br + c_m + i) * N + bc + c_n]) =
        make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
}

// ---------------------------------------------------------------------------
// ② padding 版：行 stride 变成 BN + 1，bank 随 k 轮转
// ---------------------------------------------------------------------------
template <int BM = 128, int BN = 128, int BK = 32, int TM = 4, int TN = 4>
__global__ void sgemm_smem_pad(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
  __shared__ float s_a[BM][BK + 1];      // ★ A 也 pad：A 是 [m][k]，读 s_a[c_m+i][k] 时
  __shared__ float s_b[BK][BN + 1];      //   同一列 k、不同行 m ⇒ 行 stride 决定 bank
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
    // ★ 注意：padding 之后行不再连续，所以 vec4 写必须落在行内（这里是 a_k 到 a_k+3）
    *reinterpret_cast<float4*>(&s_a[a_m][a_k]) =
        *reinterpret_cast<const float4*>(&A[(br + a_m) * K + k0 + a_k]);
    *reinterpret_cast<float4*>(&s_b[b_k][b_n]) =
        *reinterpret_cast<const float4*>(&B[(k0 + b_k) * N + bc + b_n]);
    __syncthreads();
    for (int k = 0; k < BK; ++k) {
      float ra[TM], rb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = s_a[c_m + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = s_b[k][c_n + j];   // ★ 现在 bank = (k + c_n)
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) sum[i][j] = fmaf(ra[i], rb[j], sum[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
    *reinterpret_cast<float4*>(&C[(br + c_m + i) * N + bc + c_n]) =
        make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
}

// ---------------------------------------------------------------------------
// ③ XOR swizzle 版：不 padding，靠位异或把 chunk 打散
//    Swizzle<B=1, M=4, S=3>：16 B chunk（M=4 位）、每行 8 个 chunk（S=3 位）
// ---------------------------------------------------------------------------
__device__ __forceinline__ int swizzle_chunk(int row, int chunk) {
  // 把 chunk 号与行号低位异或：(row >> 3) & 1 决定是否翻转 chunk 的最低位
  return chunk ^ ((row >> 3) & 1);
}

// 对合性自检（host 侧做全遍历，保证编码/解码同函数）
static bool swizzle_is_involution(int rows, int chunks) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < chunks; ++c)
      if (swizzle_chunk(r, swizzle_chunk(r, c)) != c) return false;
  return true;
}

template <int BM = 128, int BN = 128, int BK = 32, int TM = 4, int TN = 4>
__global__ void sgemm_swizzle(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
  // 以 8 个 float（32 B）为一个 chunk 单位做 swizzle
  constexpr int kChunk = 8;
  constexpr int kChunksPerRow = BN / kChunk;
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
    // 写侧：把 chunk 号做一次 swizzle（★ 写入与读取必须是同一个函数）
    {
      int row = a_m, col = a_k;
      int chunk = col / kChunk, in_chunk = col % kChunk;
      int scol = swizzle_chunk(row, chunk) * kChunk + in_chunk;
      *reinterpret_cast<float4*>(&s_a[row][scol]) =
          *reinterpret_cast<const float4*>(&A[(br + row) * K + k0 + col]);
    }
    {
      int row = b_k, col = b_n;
      int chunk = col / kChunk, in_chunk = col % kChunk;
      int scol = swizzle_chunk(row, chunk) * kChunk + in_chunk;
      *reinterpret_cast<float4*>(&s_b[row][scol]) =
          *reinterpret_cast<const float4*>(&B[(k0 + row) * N + bc + col]);
    }
    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      float ra[TM], rb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = s_a[c_m + i][k];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int col = c_n + j;
        int chunk = col / kChunk, in_chunk = col % kChunk;
        int scol = swizzle_chunk(k, chunk) * kChunk + in_chunk;   // 读侧同一函数
        rb[j] = s_b[k][scol];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) sum[i][j] = fmaf(ra[i], rb[j], sum[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
    *reinterpret_cast<float4*>(&C[(br + c_m + i) * N + bc + c_n]) =
        make_float4(sum[i][0], sum[i][1], sum[i][2], sum[i][3]);
}

// ---------------------------------------------------------------------------
// ④ 寄存器双缓冲（用一个更小的 tile 把结构讲清楚）
// ---------------------------------------------------------------------------
template <int BM = 64, int BN = 64, int BK = 32, int TM = 4, int TN = 4>
__global__ void sgemm_regdbuf(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
  __shared__ float s_a[2][BM][BK];
  __shared__ float s_b[2][BK][BN];        // 两级 smem stage（真正的多级流水见 q185）
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int c_m = (tid / 16) * TM, c_n = (tid % 16) * TN;
  const int br = blockIdx.y * BM, bc = blockIdx.x * BN;

  float sum[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) sum[i][j] = 0.0f;

  auto load_stage = [] (int st, int k0) {
    for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y)
      (&s_a[st][0][0])[i] = A[(br + i / BK) * K + k0 + i % BK];
    for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y)
      (&s_b[st][0][0])[i] = B[(k0 + i / BN) * N + bc + i % BN];
  };

  load_stage(0, 0);
  __syncthreads();

  int nk = K / BK;
  for (int kt = 0; kt < nk; ++kt) {
    int cur = kt & 1, nxt = (kt + 1) & 1;
    // ★ 先发起下一 stage 的装载（隐藏 gmem->smem 延迟），再算当前 stage
    if (kt + 1 < nk) load_stage(nxt, (kt + 1) * BK);

    // ★ 寄存器双缓冲：每个 k 片先把下一片的 A/B 片段预取进寄存器，再算当前片
    float ra_next[TM], rb_next[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) ra_next[i] = s_a[cur][c_m + i][0];
#pragma unroll
    for (int j = 0; j < TN; ++j) rb_next[j] = s_b[cur][0][c_n + j];

    for (int k = 0; k < BK; ++k) {
      float ra[TM], rb[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = ra_next[i];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = rb_next[j];
      // 预取下一片（k+1），与上面的 FMA 在流水线上重叠
      if (k + 1 < BK) {
#pragma unroll
        for (int i = 0; i < TM; ++i) ra_next[i] = s_a[cur][c_m + i][k + 1];
#pragma unroll
        for (int j = 0; j < TN; ++j) rb_next[j] = s_b[cur][k + 1][c_n + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) sum[i][j] = fmaf(ra[i], rb[j], sum[i][j]);
    }
    __syncthreads();     // ★ 保证所有线程读完 cur，再允许覆写
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
  TEST_BEGIN("Q184 register double-buffer + XOR swizzle");

  // ---- swizzle 的可逆性自检（对合）----
  {
    bool ok = swizzle_is_involution(64, 16);
    printf("  %-34s %s  （S>=B 保证源/目标位域不相交 ⇒ σ(σ(a))=a）\n",
           "swizzle involution self-check", ok ? "PASS" : "FAIL");
  }

  // ---- 正确性 + 性能 ----
  const int M = 512, N = 512, K = 256;
  std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N), hC(size_t(M) * N), href(size_t(M) * N);
  fill_random(hA, 71u, -1.f, 1.f);
  fill_random(hB, 72u, -1.f, 1.f);
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
  dim3 blk(32, 32);

  {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 grd(N / 128, M / 128);
    double ms = timer.bench([&] { sgemm_smem_nopad<128, 128, 32, 4, 4><<<grd, blk>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_smem_nopad (8-way)", compare(hC, href, rtol, atol), rtol, atol, ms);
    printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
  }
  {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 grd(N / 128, M / 128);
    double ms = timer.bench([&] { sgemm_smem_pad<128, 128, 32, 4, 4><<<grd, blk>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_smem_pad   (+1)", compare(hC, href, rtol, atol), rtol, atol, ms);
    printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
  }
  {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 grd(N / 128, M / 128);
    double ms = timer.bench([&] { sgemm_swizzle<128, 128, 32, 4, 4><<<grd, blk>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_swizzle (no pad)", compare(hC, href, rtol, atol), rtol, atol, ms);
    printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
  }
  {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    dim3 grd(N / 64, M / 64);
    double ms = timer.bench([&] { sgemm_regdbuf<64, 64, 32, 4, 4><<<grd, blk>>>(dA, dB, dC, M, N, K); });
    CUDA_CHECK_KERNEL();
    hC = to_host(dC, hC.size());
    report("sgemm_regdbuf (64x64 tile)", compare(hC, href, rtol, atol), rtol, atol, ms);
    printf("        %.2f TFLOPS\n", to_tflops(flops, ms));
  }

  printf("\n  ★ 观察：padding / swizzle 两个版本的数值结果应当【完全一致】（都是精确重排，\n");
  printf("    没有改变任何浮点运算的顺序）；差异只体现在时间上。\n");
  printf("    如果 swizzle 版数值不对，99%% 是「写入与读取用了不同的 swizzle 函数」——\n");
  printf("    这不会越界、不会报错，只是数据在 chunk 粒度错位。\n");

  // ---- 用 ncu 可观察的 bank conflict 探针（单独跑一次即可）----
  printf("\n  想看 bank conflict 的实测计数，用 ncu 抓这两个 kernel 对比：\n");
  printf("    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \\\n");
  printf("        --kernel-name regex:sgemm_smem_nopad ./q184\n");
  printf("    书里的参考量级（M=N=K=512）：BK=16 无 swizzle 98,304 次 / BK=64 无 swizzle\n");
  printf("    691,067 次 / BK=64 + swizzle 297,628 次（8-way -> 4-way，耗时 30.2 -> 20.7 µs）。\n");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  printf("\nQ184 done.\n");
  return 0;
}
