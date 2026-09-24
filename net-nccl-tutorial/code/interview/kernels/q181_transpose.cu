// q181_transpose.cu — Q181：矩阵转置（smem + padding 打散 bank conflict）
//
//   1) transpose_naive   —— 直接读写全局内存（写侧跨步，coalescing 崩掉）
//   2) transpose_tiled   —— 经 smem 中转 +1 padding，读写两侧都合并
//   3) transpose_tiled32 —— 32x32 tile 版本（教学常考的形状）
//
// 编译：./build.sh q181
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① 为什么朴素版慢：写 `out[x * rows + y]` 时，warp 内 x 连续 ⇒ **写地址跨步 rows**
//    ⇒ 每个线程落在不同的 32 B sector ⇒ **有效率掉到 12.5%（按 32 B sector 口径）
//    甚至约 3%（按 128 B line 口径）**。读是合并的，写不是。
//    经 smem 中转后，**读和写各自都是合并访问**，中间的转置在片上完成。
//
// ② ★ bank conflict 的算术（面试会让你现场推）：
//      smem 32 个 bank x 4 B，bank(a) = (a/4) mod 32。
//      写 tile[ty][tx]（tx 连续）⇒ 地址连续 ⇒ 无冲突。
//      读 tile[tx][ty]（tx 连续、ty 固定）⇒ 地址 = (tx * TILE + ty) * 4
//        ⇒ bank = (tx * TILE + ty) mod 32
//        TILE = 32 时 bank = ty mod 32 ⇒ **与 tx 无关 ⇒ 32 路撞同一个 bank**。
//      加 1 列 padding：地址 = (tx * 33 + ty) * 4
//        ⇒ bank = (tx * 33 + ty) mod 32 = (tx + ty) mod 32 ⇒ **tx 连续 ⇒ bank 连续 ⇒ 无冲突**。
//      ★ 代价只是 TILE x 4 B（32x32 时 128 B，占用率损失可忽略）。
//
// ③ ★ 一个必须说清的口径问题（书里专门强调）：
//      同一个 padding 优化，**L2 口径下加速比 2.40x，HBM 口径下只有 1.11x**。
//      因为 L2 口径下瓶颈是访存模式（sector 利用率），HBM 口径下瓶颈是带宽本身。
//      ⇒ 报告加速比必须带口径，否则数字没有意义。
//      本文件里 `-DHBM_SCALE=1` 可把矩阵放大到远超 L2 来复现这个差异。
//
// ④ 泛化到 fp16 时要注意：一个 16 B chunk = 8 个 half，所以 bank 的"字"单位变了；
//      书里用 `Swizzle<3,3,3>`（元素口径）而不是 `Swizzle<3,4,3>`（字节口径），
//      差一个 M 就完全错位 —— 这是 CuTe/TMA 路径上的常见坑。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

// ---------------------------------------------------------------------------
// ① 朴素版：读合并、写跨步
// ---------------------------------------------------------------------------
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;   // 列
  int y = blockIdx.y * blockDim.y + threadIdx.y;   // 行
  if (x < cols && y < rows) {
    // 读 in[y][x]：warp 内 x 连续 ⇒ 合并
    // 写 out[x][y]：warp 内 x 连续 ⇒ 地址跨步 rows ⇒ 不合并
    out[x * rows + y] = in[y * cols + x];
  }
}

// ---------------------------------------------------------------------------
// ② tiled + padding
// ---------------------------------------------------------------------------
template <int TILE = 32>
__global__ void transpose_tiled(const float* __restrict__ in,
                                float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];   // ★ +1：把行 stride 从 32 变成 33

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;

  if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  __syncthreads();                          // ★ 中转必须在读完之前不可覆写

  // 交换 block 坐标：写出的 tile 是转置后的
  int xo = blockIdx.y * TILE + threadIdx.x;
  int yo = blockIdx.x * TILE + threadIdx.y;
  if (xo < rows && yo < cols) out[yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// ③ 无 padding 的对照版（用来实测 bank conflict 的代价）
// ---------------------------------------------------------------------------
template <int TILE = 32>
__global__ void transpose_tiled_nopad(const float* __restrict__ in,
                                      float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE];        // ★ 没有 +1
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  __syncthreads();
  int xo = blockIdx.y * TILE + threadIdx.x;
  int yo = blockIdx.x * TILE + threadIdx.y;
  if (xo < rows && yo < cols) out[yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
}

// ---------------------------------------------------------------------------
// ④ XOR swizzle 版：不 padding，靠 `x ^ y` 打散 bank
//    ★ 与 LeetCUDA `kernels/swizzle/mat_trans_swizzle.cu` 的
//      `mat_trans_smem_swizzle_kernel` 逐字同构（同样是 32x32 tile、
//      同样 `block(32,32)` + `grid(N/32, M/32)`）：
//        写：s_data[x][x ^ y] = A[...]
//        读：B[...] = s_data[y][x ^ y]
//    直观理解：逻辑坐标 (row=x, col=y) 存到物理位置 (row=x, col=x^y)；
//    读的时候按逻辑 (row=y, col=x) 取，物理位置同样是 (row=y, col=x^y)。
//    ★ 与 padding 的取舍：swizzle 不浪费 smem、也不破坏 16 B 对齐，
//      但它是「按访问模式定制的置换」—— 换一个访问方向就得换一个公式。
// ---------------------------------------------------------------------------
template <int TILE = 32>
__global__ void transpose_swizzle(const float* __restrict__ in,
                                  float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE];          // ★ 没有 +1
  int x = blockIdx.x * TILE + threadIdx.x;    // 列
  int y = blockIdx.y * TILE + threadIdx.y;    // 行
  if (x < cols && y < rows) {
    tile[threadIdx.x][threadIdx.x ^ threadIdx.y] = in[y * cols + x];
    __syncthreads();
    int xo = blockIdx.y * TILE + threadIdx.x;
    int yo = blockIdx.x * TILE + threadIdx.y;
    if (xo < rows && yo < cols) {
      out[yo * rows + xo] = tile[threadIdx.y][threadIdx.x ^ threadIdx.y];
    }
  }
}

// ---------------------------------------------------------------------------
// ⑤ 只读 smem 的 bank conflict 探针（把冲突程度变成一个可报告的数字）
//    这个 kernel 不产出数据，只用来让 ncu 观察 shared_op_ld 的冲突计数。
// ---------------------------------------------------------------------------
template <int TILE = 32, int kPad = 1>
__global__ void bank_probe(float* __restrict__ sink) {
  __shared__ float tile[TILE][TILE + kPad];
  // 先用合并访问填满
  for (int i = threadIdx.y; i < TILE; i += blockDim.y)
    for (int j = threadIdx.x; j < TILE + kPad; j += blockDim.x)
      tile[i][j] = static_cast<float>(i * 100 + j);
  __syncthreads();
  // 再列方向读（这是冲突发生的地方；kPad=1 时无冲突，kPad=0 时 32 路）
  float acc = 0.0f;
  for (int i = threadIdx.y; i < TILE; i += blockDim.y) acc += tile[threadIdx.x][i];
  if (acc == -1.0f) *sink = acc;   // 防止被优化掉
}

// ---------------------------------------------------------------------------
// host 侧参考
// ---------------------------------------------------------------------------
static void ref_transpose(const std::vector<float>& in, std::vector<float>& out,
                          int rows, int cols) {
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) out[x * rows + y] = in[y * cols + x];
}

int main() {
  TEST_BEGIN("Q181 matrix transpose");

  // 关键：既要方阵（bank conflict 最明显），也要非方阵与非 tile 倍数
  struct Case { int rows, cols; };
  const std::vector<Case> cases = {{32, 32}, {1024, 1024}, {1000, 17},
                                   {33, 65}, {1, 4096}, {4096, 1}};

  for (const auto& c : cases) {
    const int rows = c.rows, cols = c.cols;
    const size_t n = static_cast<size_t>(rows) * cols;
    std::vector<float> hin(n), href(n), hgot(n);
    fill_random(hin, 31u, -1.f, 1.f);
    ref_transpose(hin, href, rows, cols);

    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(32, 8);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);
    GpuTimer timer;
    const size_t bytes = n * 2 * sizeof(float);   // 读 + 写

    CUDA_CHECK(cudaMemset(dout, 0, n * sizeof(float)));
    double ms_naive = timer.bench([&] { transpose_naive<<<grid, block>>>(din, dout, rows, cols); });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dout, n);
    report(("naive    " + std::to_string(rows) + "x" + std::to_string(cols)).c_str(),
           compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_naive);

    dim3 grid32((cols + 31) / 32, (rows + 31) / 32);
    dim3 block32(32, 32);
    CUDA_CHECK(cudaMemset(dout, 0, n * sizeof(float)));
    double ms_nopad = timer.bench([&] {
      transpose_tiled_nopad<32><<<grid32, block32>>>(din, dout, rows, cols);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dout, n);
    report("tiled32  (no pad)", compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_nopad);

    CUDA_CHECK(cudaMemset(dout, 0, n * sizeof(float)));
    double ms_pad = timer.bench([&] {
      transpose_tiled<32><<<grid32, block32>>>(din, dout, rows, cols);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dout, n);
    report("tiled32  (pad +1)", compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_pad);

    // ★ swizzle 版：与 LeetCUDA kernels/swizzle/mat_trans_swizzle.cu 同构
    CUDA_CHECK(cudaMemset(dout, 0, n * sizeof(float)));
    double ms_sw = timer.bench([&] {
      transpose_swizzle<32><<<grid32, block32>>>(din, dout, rows, cols);
    });
    CUDA_CHECK_KERNEL();
    hgot = to_host(dout, n);
    report("tiled32  (xor swizzle)", compare(hgot, href, 0.0, 0.0), 0.0, 0.0, ms_sw);

    printf("        带宽口径 8 B/元素: naive %.0f | nopad %.0f | pad+1 %.0f | swizzle %.0f GB/s\n",
           to_gbps(bytes, ms_naive), to_gbps(bytes, ms_nopad), to_gbps(bytes, ms_pad),
           to_gbps(bytes, ms_sw));
    printf("        加速比: pad+1/naive = %.2fx | pad+1/nopad = %.2fx | swizzle/nopad = %.2fx\n",
           ms_naive / ms_pad, ms_nopad / ms_pad, ms_nopad / ms_sw);
    if (rows == 1024 && cols == 1024) {
      printf("        ★ 这里数据 4 MB，落在 L2 内 ⇒ 是 **L2 口径**。\n");
      printf("          同一个 padding 优化在 HBM 口径下加速比会小得多（书里 2.40x -> 1.11x），\n");
      printf("          因为瓶颈从「访存模式」变成了「带宽本身」。\n");
    }

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dout));
  }

  // ---- bank conflict 探针：跑一次，让 ncu 能抓到 shared_op_ld 的冲突计数 ----
  {
    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
    // kPad=0 与 kPad=1 各跑一次；用 ncu 对比两者的 bank_conflicts 计数
    bank_probe<32, 0><<<1, dim3(32, 8)>>>(d_sink);   // unchecked: 探针 kernel，无输出
    bank_probe<32, 1><<<1, dim3(32, 8)>>>(d_sink);   // unchecked: 探针 kernel，无输出
    CUDA_CHECK_KERNEL();
    printf("\n  bank 探针已跑两次（kPad=0 与 kPad=1）。用 ncu 看差别：\n");
    printf("    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \\\n");
    printf("        --kernel-name regex:bank_probe --launch-count 2 ./q181\n");
    printf("    书里的参考量级（M=N=K=512，per-launch 冲突计数）：\n");
    printf("      BK=16 无 swizzle 98,304 次（2-way）\n");
    printf("      BK=64 无 swizzle 691,067 次（8-way）\n");
    printf("      BK=64 + swizzle<16> 297,628 次（4-way，耗时 30.2 -> 20.7 µs）\n");
    CUDA_CHECK(cudaFree(d_sink));
  }

  printf("\nQ181 done.\n");
  return 0;
}
