// q187_hgemm_ws_tma.cu — Q187：Warp Specialization（TMA + mbarrier）骨架
//
//   1) hgemm_ws_tma —— 1 producer warpgroup(128T) + 1 consumer warpgroup(256T) 的骨架
//
// 编译：./build.sh q187    （需要 sm_90a / sm_90+，且 CUDA >= 12.0；TMA 的 driver API 见下）
//
// ─────────────────────────────────────────────────────────────────────────────
// 解析要点
// ─────────────────────────────────────────────────────────────────────────────
// ① ★ 三件新原语各解决什么（这是本题的骨架，先讲清再写码）
//      TMA（cp.async.bulk.tensor.2d）
//        - 一条指令**由单线程提交**，硬件 DMA 搬整块 2D tile
//        - **地址计算 / OOB 填充 / smem swizzle 全在 descriptor 里声明**
//        - **搬运过程零寄存器开销**（这是相对 cp.async 最大的差别：
//          cp.async 每线程仍要算地址、占发射槽）
//      mbarrier
//        - 自带 **arrive count** 与 **tx count** 两个独立计数器
//        - TMA 搬完指定字节数后**硬件自动做 complete-tx 记账**
//        - **相位翻转条件：P = 0 ∧ T = 0**
//        - ★ 核心洞察：**「多少个信号」与「多少字节数据」是两把独立的锁**
//          ⇒ producer / consumer 之间**不需要任何 __syncthreads**
//      WGMMA
//        - 矩阵乘从 warp 级（32 线程）升到 **warpgroup 级（128 线程）**
//        - A/B **直接从 smem 读**，不需要 ldmatrix 中转
//        - **异步**：发射后 warpgroup 继续推进，用 wgmma.wait_group 收割
//    ⚠️ 本文件面向 **sm_90a**；如果在 sm_120 上编译 WGMMA 部分会失败，
//       用 `-DSKIP_WGMMA=1` 编译可只保留 TMA + mbarrier 的骨架（消费侧用 mma.sync）。
//
// ② ★ arrive count 怎么算（年年考、最容易错）
//      129 = 128 + 1：consumer 是**整个 128 线程 warpgroup**（warpgroup 对齐指令）
//      257 = 256 + 1：consumer 用 **warp 级 mma.sync** 时，**8 个 warp 独立推进，
//                     每个线程都必须 arrive 一次**
//      129（FA3 的 per-WG）：屏障按 [cid][stage] 切开，每张只放行一个 WG
//    ⇒ **「协议只认线程数，不认角色」**。
//    本文件是 WGMMA 路径 ⇒ **129**。
//
// ③ ★ 账本不闭合的两种后果
//      **少算 ⇒ 永不翻转 ⇒ 挂死**（没有报错、没有超时，最难查）
//      **多算 ⇒ 提前翻转 ⇒ 读到半新半旧数据 ⇒ 静默错误**（更危险）
//    TMA 的 expect_tx 同理：**少报挂死、多报静默错**。
//
// ④ ★ producer / consumer 的分工与「点火」
//      producer 四拍：P0 装 A 一次（arrive_expect_tx）；P1 预取前 Sk-1 个 stage；
//                     P2 稳态循环（等 empty -> 发 TMA -> arrive_expect_tx）
//      consumer：**C0 点火** = 把每张 empty[s] 预存 kConsumerThreads 次 arrive，
//                然后 full_Q.wait(...) 一次 —— **漏掉 C0 就是全体挂死**。
//
// ⑤ ★ 两个 ptxas 静默降级陷阱
//      TMA dst 写成 `shared::cluster` ⇒ ptxas 以 **C7506 静默丢弃 setmaxnreg**
//        （必须写 `shared::cta`）
//      `__launch_bounds__(N)` 缺第二个参数 ⇒ **C7508**
//      排查：cuobjdump -sass 检索 `USETMAXREG`（注意 SASS 助记符没有 "n"）
//
// ⑥ ★ 一个「挂死但没有任何报错」的经典事故（setmaxnreg 池数学）
//      每 SM 寄存器文件 65536 个，triple-WG 配置提出公因子 128 ⇒ r_p + 2*r_c <= 512
//      正常 40 + 2*168 = 376；**事故 64 + 2*224 = 512（恰好等于上界）**
//      —— 驱动对寄存器文件另有保留份额、有效池小于 65536 ⇒ alloc 永远凑不齐 ⇒
//      **consumer 卡在 setmaxnreg、producer 卡在等 barrier arrive，双向互等纯挂死**
//      修复：留 >= 2048 slack ⇒ 64/208/208 = 480。
// ─────────────────────────────────────────────────────────────────────────────

#include "lc_common.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#define SKIP_TMA_IMPL 1
#else
#define SKIP_TMA_IMPL 0
#endif

// driver API：cuTensorMapEncodeTiled。build.sh 会尝试 -lcuda；
// 拿不到就 dlopen，避免"驱动 API 头找不到"导致整个文件编译不过。
#if !SKIP_TMA_IMPL
#include <cuda.h>

// ---------------------------------------------------------------------------
// TMA 装载：一条指令搬整块 2D tile
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map, void* smem_dst,
                                            int c0, int c1) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(s),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(0));
}

// mbarrier：init / arrive_expect_tx / arrive / wait
__device__ __forceinline__ void mbar_init(uint64_t* bar, unsigned count) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(s), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, unsigned bytes) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(s), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(s));
}
// 等到相位翻转（阻塞版，教学用；生产应用 try_wait 循环 + nanosleep 退避）
__device__ __forceinline__ void mbar_wait(uint64_t* bar, unsigned phase) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n\t.reg .pred P;\n\t"
      "WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra WAIT;\n\t}\n" ::"r"(s),
      "r"(phase));
}
// TMA 是"异步代理"，写进 smem 的数据要 fence 之后普通 load 才可见
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::);
}
// setmaxnreg：producer 释放寄存器、consumer 领走
__device__ __forceinline__ void reg_dealloc(unsigned n) {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(n));
}
__device__ __forceinline__ void reg_alloc(unsigned n) {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(n));
}
#endif  // !SKIP_TMA_IMPL

// ---------------------------------------------------------------------------
// WS 骨架：128 producer + 256 consumer = 384 线程
//   BM = BN = 128, BK = 64（A/B 各 128*64 half = 16 KB / stage）
// ---------------------------------------------------------------------------
#define SK_STAGES 2
#define BM 128
#define BN 128
#define BK 64
#define CONSUMER_THREADS 256
#define ARRIVE_COUNT (CONSUMER_THREADS + 1)   // ★ 257：warp 级 mma.sync 的算法
// 若消费侧换成 WGMMA（warpgroup 对齐），应改成 129

// no-launch: 本文件是【协议骨架】—— 真正的 launch 需要 host 侧先用
// cuTensorMapEncodeTiled 构造 descriptor 并用 __grid_constant__ 传入，
// 那段 host 代码约 40 行、与要讲的协议无关，故此处只保留 kernel 结构。
__global__ void __launch_bounds__(384, 1)
hgemm_ws_tma(const __grid_constant__ CUtensorMap tma_a,
             const __grid_constant__ CUtensorMap tma_b, float* __restrict__ C, int M,
             int N, int K) {
#if SKIP_TMA_IMPL
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("  (arch < sm_90: TMA/mbarrier 路径已跳过)\n");
#else
  extern __shared__ __align__(1024) char smem_raw[];   // ★ 1024B 对齐是硬约束
  __half* s_a = reinterpret_cast<__half*>(smem_raw);                    // [Sk][BM][BK]
  __half* s_b = s_a + SK_STAGES * BM * BK;                              // [Sk][BN][BK]
  __shared__ __align__(8) uint64_t full[SK_STAGES], empty[SK_STAGES];

  constexpr int kTileBytesA = BM * BK * sizeof(__half);   // 16 KB
  constexpr int kTileBytesB = BN * BK * sizeof(__half);   // 16 KB
  const int tid = threadIdx.x;
  const bool is_producer = (tid < 128);
  const int prod_tid = tid;                    // producer 用 0..127
  const int cons_tid = tid - 128;              // consumer 用 0..255

  // ---- 屏障初始化（一个线程做即可）----
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < SK_STAGES; ++s) {
      mbar_init(&full[s], ARRIVE_COUNT);
      mbar_init(&empty[s], ARRIVE_COUNT);
    }
  }
  __syncthreads();   // 只用于保证 init 可见（init 之后就没用了）

  const int nk_tiles = K / BK;
  const int m_tiles = M / BM;
  const int bm = blockIdx.y % m_tiles;
  const int bn = blockIdx.y / m_tiles;

  if (is_producer) {
    // ------------------- producer：只有 thread 0 真正发 TMA -------------------
    if (prod_tid == 0) {
      for (int t = 0; t < nk_tiles; ++t) {
        const int st = t % SK_STAGES;
        const unsigned phase = (t / SK_STAGES) & 1;
        mbar_wait(&empty[st], phase ^ 1);      // 等 consumer 释放该 stage
        tma_load_2d(&tma_a, &s_a[st * BM * BK], /*k0=*/t * BK, /*m0=*/bm * BM);
        tma_load_2d(&tma_b, &s_b[st * BN * BK], /*k0=*/t * BK, /*n0=*/bn * BN);
        // ★ 一个 K-tile 的 A+B 共 32768 B，一次声明总事务量
        mbar_arrive_expect_tx(&full[st], kTileBytesA + kTileBytesB);
      }
    }
  } else {
    // ------------------- consumer：mbarrier 协调，消费侧用 mma.sync -------------------
    // ★ C0 点火：每张 empty[s] 预存 [本 WG 线程数] 次 arrive。
    //   漏掉这一步 ⇒ producer 首轮等 empty 永远等不到 ⇒ 全体挂死（而且没有任何报错）。
#pragma unroll
    for (int s = 0; s < SK_STAGES; ++s) mbar_arrive(&empty[s]);

    const int lane = cons_tid % 32, warp = cons_tid / 32;
    (void)lane;
    (void)warp;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int t = 0; t < nk_tiles; ++t) {
      const int st = t % SK_STAGES;
      const unsigned phase = (t / SK_STAGES) & 1;

      // 拍一：等数据落盘（P = 0 ∧ T = 0 才翻相位），然后 fence 才能读
      mbar_wait(&full[st], phase);
      fence_proxy_async_shared();

      // 拍二/拍三：这里本该是 ldmatrix + HMMA + softmax 之类；
      //             完整实现见 q186（本文件只保留协议骨架，不重复贴 100 行 mma 代码）。
      // 拍四：★ early release —— 数据已进寄存器就放行该 stage，
      //       producer 因此最早可以在下一拍就开始覆盖它。
      mbar_arrive(&empty[st]);
    }

    // 收尾：epilogue 归 consumer（producer 早就流到头退出了）
    if (cons_tid == 0) C[0] = acc[0];
  }
#endif  // SKIP_TMA_IMPL
}

int main() {
  TEST_BEGIN("Q187 warp specialization skeleton (TMA + mbarrier + WGMMA)");

  printf("\n  这一题是【骨架 + 协议】，不是完整可用的 HGEMM：\n");
  printf("    完整 TMA HGEMM 需要 host 侧先用 cuTensorMapEncodeTiled 建 descriptor\n");
  printf("    （row-major 时 globalDim 写作 (K, M) —— minor 在前，这是最容易背反的约定），\n");
  printf("    再用 __grid_constant__ 传进 kernel。那段 host 代码 40 行、与算法无关，\n");
  printf("    本文件省略，只保留面试要讲的三件事：TMA 装载、mbarrier 账本、WS 分工。\n\n");

  printf("  ★ 账本（本题要背的数字）：\n");
  printf("      arrive count = %d  （%d consumer + 1 producer）\n", ARRIVE_COUNT,
         CONSUMER_THREADS);
  printf("      每个 stage 的 expect_tx = %d B（A %d KB + B %d KB）\n",
         (BM * BK + BN * BK) * 2, BM * BK * 2 / 1024, BN * BK * 2 / 1024);
  printf("      相位翻转条件: P = 0 ∧ T = 0\n");
  printf("      smem = %d stages x %d KB = %d KB\n", SK_STAGES,
         (BM * BK + BN * BK) * 2 / 1024, SK_STAGES * (BM * BK + BN * BK) * 2 / 1024);

  printf("\n  ★ 与 Hopper 的数字对照（书里）：\n");
  printf("      HGEMM WGMMA 版 arrive count = 129（128 warpgroup + 1）\n");
  printf("      FA2 TMA+WS 版   arrive count = 257（256 线程 + 1，warp 级 mma.sync）\n");
  printf("      FA3 per-WG 版   arrive count = 129（屏障按 [cid][stage] 切开）\n");
  printf("      ⇒ 「协议只认线程数，不认角色」\n");

  printf("\n  ★ 实测收益（书里）：TMA+WS 相对 cp.async 版在 attention 上稳定 +20%%；\n");
  printf("      HGEMM 上 2048^3 从 104.66 T 提到 164.64 T（1.57x），**计算指令一条没改**。\n");

  // 真正跑需要 descriptor；这里只跑一个"协议自检"：mbarrier 的语义在 host 上是
  // 无法模拟的，所以我们只验证 kernel 能被 launch（架构不支持时 kernel 内部会打印跳过）
  DeviceInfo dev = query_device();
  if (dev.cc_major >= 9) {
    printf("\n  尝试 launch 骨架 kernel（无 descriptor，只验证能起来）...\n");
    // 注意：真要用 CUtensorMap 参数就必须走 driver API；这里故意不 launch，
    // 避免在没有 descriptor 的情况下产生无意义的非法访问。
    printf("  跳过（需要 host 侧 cuTensorMapEncodeTiled 构造 descriptor）。\n");
  } else {
    printf("\n  本机 sm_%d%d < 9.0，TMA/mbarrier 路径不适用，仅阅读协议。\n",
           dev.cc_major, dev.cc_minor);
  }

  printf("\nQ187 done.\n");
  return 0;
}
