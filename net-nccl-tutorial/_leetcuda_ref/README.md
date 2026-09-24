# LeetCUDA 参考实现事实抽取（§7 手撕题的溯源底稿）

> **这是生成 `interview.md` §7.0 溯源表与 §7 里那些「与 LeetCUDA 对照」注释的底稿。**
> 内容是从 [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 的源码里
> **逐字摘出来的事实**（符号名、常量、启动形状、PTX 字符串、tol、注释原文），
> 用于确保教程里的代码与参考实现**不是各写各的**。
>
> | 项 | 值 |
> |---|---|
> | 引用锚点 | **HEAD `6c86259d7eca5e6b34fcd4bd21e4ebe1c890abc2`**（2026-09-22）<br>（抓取期间上游从 `e831d97` 前进到 `6c86259`；对本节引用的文件只改了 `hgemm.cuh` 一行 `__launch_bounds__(kNumThreads)` → `(kNumThreads, 1)`） |
> | 本地 checkout | `C:\Users\Jeff\Documents\GitHub\LeetCUDA` |
> | 抓取方式 | 只读（未修改 LeetCUDA 任何文件） |
> | 覆盖 | `kernels/interview/{common,base,sgemm,hgemm,flash_attn,ffpa_attn}.cuh`、`notes-v2.cu`、`build.sh`、`README.md`；`kernels/{reduce,dot-product,softmax,layer-norm,rms-norm,elementwise,relu,gelu,histogram,mat-transpose,swizzle,sgemm,ws-hgemm,nvidia-nsight}/` |
>
> **注意行数**：本仓库里有些 `.md` 的行计数工具只数非空行，会有偏差；
> 引用行号前请以磁盘真实行数为准（各分册里已标注实测值）。
>
> **三个最影响教程写法的发现**（细节见各分册）：
> 1. **LeetCUDA 的 `.py` 里没有 tol、没有 `allclose`、没有断言** —— 纯 benchmark，
>    正确性靠人眼看打印值。所以本套件的"三档容差 + CPU 对拍"是**加强**而不是照抄。
> 2. **`--use_fast_math`** 出现在 LeetCUDA 的 `build.sh` 里；本套件没开。
> 3. **`setmaxnreg` 在 sm_120a 上会被 ptxas 以 C7506 丢弃**（因为 TMA 的存在被当作
>    extern-call 边界），所以 LeetCUDA 用 `NOTES_V2_ENABLE_SETMAXNREGS` 门控**调用点**；
>    本套件用 `ENABLE_SETMAXNREGS` 对齐了这个做法。

---


# A · 归约 / softmax / norm（Q177 Q182 Q188）

# LeetCUDA 参考实现事实摘录：reduce / dot-product / softmax / layer-norm / rms-norm

- 仓库：`C:\Users\Jeff\Documents\GitHub\LeetCUDA`，固定 commit `e831d970a099f5ce8fd0495ddd1df09206d56918`（`git log -1` 提交信息：`fix: restore star history chart with new domain (#516)`，`Fri Aug 14 12:48:25 2026 +0200`）。
- 5 个 .cu 的 `git status --porcelain` 为空（工作区干净、未修改）。**实测行数**与任务给出的 835/283/769/726/724 **不符**：

| 文件 | 实测行数 |
| --- | --- |
| `kernels/reduce/block_all_reduce.cu` | 894 |
| `kernels/dot-product/dot_product.cu` | 305 |
| `kernels/softmax/softmax.cu` | 849 |
| `kernels/layer-norm/layer_norm.cu` | 793 |
| `kernels/rms-norm/rms_norm.cu` | 790 |

- 五个文件逐字相同的头部宏（数值/写法原样）：
  `#define WARP_SIZE 32`、
  `#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])`、
  `#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])`、
  `#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])`、
  `#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])`、
  `#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])`
- include 集合相同（5 个文件一致）：`<algorithm> <cuda_bf16.h> <cuda_fp16.h> <cuda_fp8.h> <cuda_runtime.h> <float.h> <stdio.h> <stdlib.h> <torch/extension.h> <torch/types.h> <vector>`；layer-norm / rms-norm 也 include 了 `cuda_fp8.h` 但完全没用到。
- 5 个 .py 驱动脚本共用**完全相同**的编译选项（直接决定数值行为）：`-O3`、`-U__CUDA_NO_HALF_OPERATORS__`、`-U__CUDA_NO_HALF_CONVERSIONS__`、`-U__CUDA_NO_HALF2_OPERATORS__`、`-U__CUDA_NO_BFLOAT16_CONVERSIONS__`、`--expt-relaxed-constexpr`、`--expt-extended-lambda`、`--use_fast_math`，`extra_cflags=["-std=c++17"]`；脚本首行 `torch.set_grad_enabled(False)`。
- **容差事实（重要）**：这 5 个 .py **没有** `torch.allclose` / `torch.testing.assert_close` / `atol` / `rtol` / `manual_seed` / 任何断言。它们只是 benchmark：warmup 后跑 iters 次取平均，然后打印 `out.flatten()...[:3]`（`round(v, 8)`）与相邻行的 torch 参考值，**靠人眼比对**。输入一律 `torch.randn(...)`（未播种）。

---

## kernels/reduce/block_all_reduce.cu

- **导出/定义的符号**（逐条，精确名字 + 模板默认值）
  - device helper：`template <const int kWarpSize = WARP_SIZE> __device__ __forceinline__ float warp_reduce_sum_f32(float val)`；`half warp_reduce_sum_f16_f16(half val)`；`float warp_reduce_sum_f16_f32(half val)`；`__nv_bfloat16 warp_reduce_sum_bf16_bf16(__nv_bfloat16 val)`；`float warp_reduce_sum_bf16_f32(__nv_bfloat16 val)`；`half warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val)`；`half warp_reduce_sum_fp8_e5m2_f16(__nv_fp8_storage_t val)`；`int32_t warp_reduce_sum_i8_i32(int8_t val)`；`int32_t warp_reduce_sum_i32_i32(int32_t val)`
  - 20 个 `__global__` kernel，**没有 v0/v1/v2/v3 编号版本**；差异只在 `packed_type`（打包宽度）与 `acc_type`（warp 内累加精度），全部是 `template <const int NUM_THREADS = ...>` + `(元素指针 a, 输出指针 y, int N)` 单 block 归约 + `atomicAdd`：
    1. `block_all_reduce_sum_f32_f32_kernel<256>`（float*/float*）
    2. `block_all_reduce_sum_f32x4_f32_kernel<256 / 4>`
    3. `block_all_reduce_sum_f16_f16_kernel<256>`（half in、float* y；warp 内用 `warp_reduce_sum_f16_f16`，`reduce_smem` 是 `__shared__ float`）
    4. `block_all_reduce_sum_f16_f32_kernel<256>`
    5. `block_all_reduce_sum_f16x2_f32_kernel<256 / 2>`
    6. `block_all_reduce_sum_f16x2_f16_kernel<256 / 2>`
    7. `block_all_reduce_sum_f16x8_pack_f16_kernel<256 / 8>`
    8. `block_all_reduce_sum_f16x8_pack_f32_kernel<256 / 8>`
    9. `block_all_reduce_sum_bf16_bf16_kernel<256>`（唯一逐元素守卫/共享内存都用 bf16 的一支：`__shared__ __nv_bfloat16 reduce_smem[NUM_WARPS]`）
    10. `block_all_reduce_sum_bf16_f32_kernel<256>`
    11. `block_all_reduce_sum_bf16x2_bf16_kernel<256 / 2>`
    12. `block_all_reduce_sum_bf16x2_f32_kernel<256 / 2>`
    13. `block_all_reduce_sum_bf16x8_pack_bf16_kernel<256 / 8>`
    14. `block_all_reduce_sum_bf16x8_pack_f32_kernel<256 / 8>`
    15. `block_all_reduce_sum_fp8_e4m3_f16_kernel<256>`
    16. `block_all_reduce_sum_fp8_e5m2_f16_kernel<256>`
    17. `block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel<256 / 16>`
    18. `block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel<256 / 16>`
    19. `block_all_reduce_sum_i8_i32_kernel<256>`（int8_t* a、int32_t* y）
    20. `block_all_reduce_sum_i8x16_pack_i32_kernel<256 / 16>`
  - 宏：`STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、`LANUCH_REDUCE_KERNEL`、`DISPATCH_REDUCE_KERNEL`、`TORCH_BINDING_REDUCE`
  - 20 个 host 函数 `torch::Tensor block_all_reduce_sum_<packed>_<acc>(torch::Tensor x)`，`TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type, n_elements, out_type)` 实例化清单：`f32/f32,1`、`f32x4/f32,4`、`f16/f16,1`、`f16/f32,1`、`f16x2/f16,2`、`f16x2/f32,2`、`f16x8_pack/f16,8`、`f16x8_pack/f32,8`、`bf16/bf16,1`、`bf16/f32,1`、`bf16x2/bf16,2`、`bf16x2/f32,2`、`bf16x8_pack/bf16,8`、`bf16x8_pack/f32,8`、`fp8_e4m3/f16,1`、`fp8_e4m3x16_pack/f16,16`、`fp8_e5m2/f16,1`、`fp8_e5m2x16_pack/f16,16`、`i8/i32,1`、`i8x16_pack/i32,16`
  - `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 注册同名 20 个 `block_all_reduce_sum_*`
- **每个 kernel 的启动形状**（grid / block 的确切写法）
  - kernel 内索引：标量版 `int idx = blockIdx.x * NUM_THREADS + tid;`；packed 版 `int idx = (blockIdx.x * NUM_THREADS + tid) * 4;`（f32x4）/ `* 2`（f16x2、bf16x2）/ `* 8`（x8_pack）/ `* 16`（fp8x16、i8x16）
  - 源码注释声明的形状：`// grid(N/256), block(256)`（f32、f16、bf16、fp8、i8 标量版）、`// grid(N/256), block(256/4)`（f32x4）；x2/x8/x16_pack 版本**没有**形状注释
  - 2D 输入（`ndim == 2`：`S = x.size(0); K = x.size(1); N = S * K;`）：`const int NT = (K) / (n_elements); dim3 block(NT); dim3 grid((S));`，`NT` 只允许 `32 / 64 / 128 / 256 / 512 / 1024`（`switch (NT)`），否则 `throw std::runtime_error("only support (K)/(n_elements): 32/64/128/256/512/1024")`
  - 非 2D 或 `K / n_elements > 1024` 的兜底路径（把整张量拉平）：`dim3 block(1024 / (n_elements)); dim3 grid((N + 1024 - 1) / 1024);`，模板实参写死为 `block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 / (n_elements)>`
- **关键常量**
  - `WARP_SIZE 32`；`constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;`（每 kernel 内重算）
  - 模板默认：`256`（标量版）、`256 / 2`（x2）、`256 / 4`（f32x4）、`256 / 8`（x8_pack）、`256 / 16`（fp8x16 / i8x16）
  - shuffle 写法：`for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) ... __shfl_xor_sync(0xffffffff, val, mask)`（无 `__syncwarp`、无 lane 掩码收窄）
  - 单位元/哨兵：`0.0f`、`__float2half(0.0f)`、`__float2bfloat16(0.0f)`、`0`（i8）、`__nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3)` / `... __NV_E5M2`
  - 输出张量：`auto y_th_type = (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;`，`auto y = torch::zeros({1}, options);`（**fp8 的输出也是 float32**，只有 int8 走 int32）
- **数值与容差**（`block_all_reduce.py`）
  - 无 tol、无断言、无随机种子；参考实现 = `torch.sum`（**GPU、与输入同 dtype**，不是 CPU/double）；fp8 的参考是 `torch.sum(values_f8e4m3.half())`，源码注释 `# torch.sum not support fp8`
  - shape 列表：`Ss = [1024, 2048, 4096]`、`Ks = [1024, 2048, 4096]`、`SKs = [(S, K) for S in Ss for K in Ks]`（9 组）；数据 `values = torch.randn((S, K)).cuda().float()`，再做 `.half()` / `.bfloat16()` / `.to(dtype=torch.float8_e4m3fn)` / `.to(dtype=torch.float8_e5m2)` / `.to(dtype=torch.int8)`
  - benchmark 参数：`warmup: int = 10, iters: int = 1000`；`total_time = (end - start) * 1000  # ms`、`mean_time = total_time / iters`；打印 `out.item()`（`tag.startswith("i8")` 时用整数 `:<15`，否则 `:<15.8f`）
  - 被注释掉的一行透露出精度动机：`# if perf_func.__name__ == torch.sum.__name__: values = values.float() # for precision`
- **数据流骨架**（变量名保持原样）
```text
tid = threadIdx.x; idx = blockIdx.x * NUM_THREADS + tid;        // packed 版再 *n_elements
sum = (idx < N) ? a[idx] : 0.0f;                                // "load once only"
warp = tid / WARP_SIZE; lane = tid % WARP_SIZE;
sum = warp_reduce_sum_f32<WARP_SIZE>(sum);                      // mask = 16,8,4,2,1
if (lane == 0) reduce_smem[warp] = sum;
__syncthreads();
sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
if (warp == 0) sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
if (tid == 0) atomicAdd(y, sum);                                // 跨 block 原子累加
```
  - x8_pack 的寄存器段：`half pack_a[8]; LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);` 然后 `#pragma unroll for (int i = 0; i < 8; ++i) sum_f16 += (((idx + i) < N) ? pack_a[i] : z);`（f16x8/bf16x8）；fp8/i8 的 x16_pack 用 `half pack_a[16]` / `int8_t pack_a[16]` 同款 128-bit 载入
- **处理边界的写法**
  - 标量版：`(idx < N) ? a[idx] : 0.0f`（单位元 0）
  - x2 / x4 / x8 / x16_pack：128-bit/64-bit 载入**一律无守卫**（`float4 reg_a = FLOAT4(a[idx]);`、`half2 reg_a = HALF2(a[idx]);`、`LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);`），只有“用哪些分量参与累加”受 `(idx < N)` 或 `(idx + i) < N` 控制 → N 非 4/2/8/16 倍数时越界读
  - fp8_e4m3x16_pack / fp8_e5m2x16_pack / i8x16_pack 的 16 元素累加循环**完全没有** `(idx + i) < N` 守卫（`sum_f16 += __nv_cvt_fp8_to_halfraw(pack_a[i], __NV_E4M3);`、`sum_i32 += (static_cast<int32_t>(pack_a[i]));`）
  - partial warp：无专门处理；用 `(lane < NUM_WARPS) ? reduce_smem[lane] : 单位元` 补齐，第二级 `warp_reduce_sum_*<NUM_WARPS>`（`NUM_WARPS` 可为 1、2…32；`kWarpSize = 1` 时 mask 循环直接不执行）
- **常见坑/注释里的警告**（原样）
  - `// use float to keep sum from each block and reduce` + `// with fp32 inter warps.`——**该注释在 bf16 版里出现，但代码是 `reduce_smem[warp] = sum_bf16;` + `warp_reduce_sum_bf16_bf16`，注释与代码矛盾**
  - `// temporary register(memory), .local space in ptx, addressable`、`// reinterpret as float4 and load 128 bits in 1 memory issue.`、`// 8x16 bits=128 bits.`、`// 16x8 bits=128 bits.`、`// 16x8=128 bits`
  - `// keep the data in register is enough for warp operaion.`（"operaion" 拼写错误，5 个文件全都是这个拼写）
  - `// val += __shfl_xor_sync(0xffffffff, val, mask);`（在本文件被注释掉，实际用 `__hadd`）

---

## kernels/dot-product/dot_product.cu

- **导出/定义的符号**
  - device helper：`warp_reduce_sum_f32<kWarpSize = WARP_SIZE>`、`warp_reduce_sum_f16_f16<kWarpSize = WARP_SIZE>`（返回 `half`，**实际未被任何 kernel 调用**）、`warp_reduce_sum_f16_f32<kWarpSize = WARP_SIZE>`
  - 5 个 kernel：`dot_prod_f32_f32_kernel<256>`、`dot_prod_f32x4_f32_kernel<256 / 4>`、`dot_prod_f16_f32_kernel<256>`、`dot_prod_f16x2_f32_kernel<256 / 2>`、`dot_prod_f16x8_pack_f32_kernel<256 / 8>`；签名统一 `(T *a, T *b, float *y, int N)`
  - 宏：`STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、`LANUCH_DOT_PROD_KERNEL`、`DISPATCH_DOT_PROD_KERNEL`、`TORCH_BINDING_DOT_PROD`
  - host：`torch::Tensor dot_prod_<packed>_<acc>(torch::Tensor a, torch::Tensor b)`，实例化 `TORCH_BINDING_DOT_PROD(f32, f32, torch::kFloat32, float, 1)`、`(f32x4, f32, torch::kFloat32, float, 4)`、`(f16, f32, torch::kHalf, half, 1)`、`(f16x2, f32, torch::kHalf, half, 2)`、`(f16x8_pack, f32, torch::kHalf, half, 8)`
  - `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`：`dot_prod_f32_f32`、`dot_prod_f32x4_f32`、`dot_prod_f16_f32`、`dot_prod_f16x2_f32`、`dot_prod_f16x8_pack_f32`
- **每个 kernel 的启动形状**
  - 注释声明：`// grid(N/256), block(256)`（f32）、`// grid(N/256), block(256/4)`（f32x4）、`// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))`；f16 / f16x2 / f16x8_pack 无形状注释
  - kernel 内索引：`int idx = blockIdx.x * NUM_THREADS + tid;`、`(blockIdx.x * NUM_THREADS + tid) * 4`、`* 2`（注释 `// 2 half elements per thread`）、`* 8`（注释 `// 8 half elements per thread`）
  - 非 2D（本 .py 实际走的路径）：`auto prod = torch::zeros({1}, options);` → `dim3 block(256); dim3 grid(((N + 256 - 1) / 256) / (n_elements));` 且模板实参写死 `<256>`
  - 2D 且 `(K / (n_elements)) <= 1024`：`const int NT = (K) / (n_elements); dim3 block(NT); dim3 grid((S));`，`NT ∈ {32, 64, 128, 256, 512, 1024}`，否则 `throw std::runtime_error("only support (K)/(n_elements): 32/64/128/256/512/1024")`
  - 2D 且 `K / n_elements > 1024`：与“非 2D”同款 `block(256) / grid(((N + 256 - 1) / 256) / (n_elements))`
- **关键常量**
  - `WARP_SIZE 32`；`constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;`；`__shared__ float reduce_smem[NUM_WARPS];`（5 个 kernel 全是 float smem，即使累加类型是 half）
  - 模板默认：`256`、`256 / 4`、`256`、`256 / 2`、`256 / 8`
  - f16 单位元：`__float2half(0.0f)`；`const half z = __float2half(0.0f);`（x8_pack 版）
- **数值与容差**（`dot_product.py`）
  - 无 tol、无断言、无种子；参考实现 = `torch.dot`（GPU 同 dtype）；输出是 `torch::zeros({1})` 的单元素张量，打印用 `out.item()`（本文件不像其他 4 个那样取 `flatten()[:3]`，因为 `torch.dot` 只返回 0-dim 标量）
  - shape：`Ss = [1024, 2048, 4096]`、`Ks = [1024, 2048, 4096]`，且输入被压平成 1D：`a = torch.randn((S * K)).cuda().float()`、`b = torch.randn((S * K)).cuda().float()`（**所以 .py 从不触发 2D dispatch 分支**），`a_f16 = a.half()`、`b_f16 = b.half()`
  - 参数：`warmup: int = 10, iters: int = 1000`
- **数据流骨架**
```text
tid = threadIdx.x; idx = (blockIdx.x * NUM_THREADS + tid) * n_elements;
prod = (idx < N) ? a[idx] * b[idx] : 0.0f;              // f32x4: reg_a.x*reg_b.x + ... + reg_a.w*reg_b.w
warp = tid / WARP_SIZE; lane = tid % WARP_SIZE;
prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
if (lane == 0) reduce_smem[warp] = prod;
__syncthreads();
prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
if (warp == 0) prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
if (tid == 0) atomicAdd(y, prod);
```
  - f16x8_pack 的寄存器段：`half pack_a[8], pack_b[8]; LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);` 然后 `#pragma unroll for (int i = 0; i < 8; i += 2) { half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i])); prod_f16 += (((idx + i) < N) ? (v.x + v.y) : z); }`
- **处理边界的写法**
  - 标量/x2/x4/x8_pack 的 128-bit 载入全部无守卫；只有累加项受 `(idx < N)`（x8_pack 用 `(idx + i) < N`）保护
  - f32x4 只判断基地址 `(idx < N)`，`idx+1..idx+3` 越界分量照样计入（N 非 4 倍数时既越界又错算）
  - partial warp：同样用 `lane < NUM_WARPS` 补 0，第二级 warp 归约，最后 `tid == 0` 的 `atomicAdd`
- **常见坑/注释里的警告**
  - `// keep the data in register is enough for warp operaion.`（该注释在 f16 标量版里，但真正“保持寄存器”的是 x8_pack 版）
  - `// temporary register(memory), .local space in ptx, addressable`、`// load 128 bits`、`// 8x16 bits=128 bits.`
  - `// Dot Product + Vec4` / `// Dot Product` 的 `grid/block` 注释只覆盖 2 个 kernel，pack 版形状无注释（易误抄）
  - 2D 分支里 `const int N = S * K;` 与内层 `int N = 1;` 同名遮蔽（`-Wshadow` 会报）

---

## kernels/softmax/softmax.cu

- **导出/定义的符号**
  - `// DS required for Online Softmax` → `struct __align__(8) MD { float m; float d; };`
  - device helper：`MD warp_reduce_md_op<kWarpSize = WARP_SIZE>(MD value)`、`float warp_reduce_sum_f32<kWarpSize = WARP_SIZE>`、`float warp_reduce_max_f32<kWarpSize = WARP_SIZE>`（`val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));`）、`float block_reduce_sum_f32<NUM_THREADS = 256>`、`float block_reduce_max_f32<NUM_THREADS = 256>`
  - 9 个 `__global__` kernel（全部 `(T *x, T *y, int N)`，无 total 参数）：`softmax_f32_per_token_kernel<256>`、`softmax_f32x4_per_token_kernel<256 / 4>`、`safe_softmax_f32_per_token_kernel<256>`、`safe_softmax_f32x4_per_token_kernel<256 / 4>`、`safe_softmax_f16_f32_per_token_kernel<256>`、`safe_softmax_f16x2_f32_per_token_kernel<256>`、`safe_softmax_f16x8_pack_f32_per_token_kernel<256>`、`online_safe_softmax_f32_per_token_kernel<256>`、`online_safe_softmax_f32x4_pack_per_token_kernel<256 / 4>`
  - **被整段注释掉的 grid-level 版本**（`softmax_f32_kernel` 与 `softmax_f32x4_kernel`，带 `atomicAdd(total, exp_sum)` + `__threadfence(); // grid level memory fence`，以及配套被注释的 `TORCH_BINDING_SOFTMAX(f32,...)` / `(f32x4,...)` 和 pybind 注册）
  - 宏：`STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、`CHECK_TORCH_TENSOR_SHAPE`、`TORCH_BINDING_SOFTMAX`（仅给被注释的 grid 版用）、`LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL`、`DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL`、`LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL`、`DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL`、`LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL`、`DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL`、`LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL`、`DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL`、`LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL`、`DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL`、`LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL`、`DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL`、`LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL`、`DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL`、`LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL`、`DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL`、`LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL`、`DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL`
  - 9 个 host 函数：`softmax_f32_per_token`、`softmax_f32x4_per_token`、`safe_softmax_f32_per_token`、`safe_softmax_f32x4_per_token`、`safe_softmax_f16_f32_per_token`、`safe_softmax_f16x2_f32_per_token`、`safe_softmax_f16x8_pack_f32_per_token`、`online_safe_softmax_f32_per_token`、`online_safe_softmax_f32x4_pack_per_token`（全部 `void f(torch::Tensor x, torch::Tensor y)`，取 `S = x.size(0); H = x.size(1); N = S * H;`）
  - `PYBIND11_MODULE` 注册上述 9 个；`softmax_f32` / `softmax_f32x4` 两行被注释
- **每个 kernel 的启动形状**（逐 kernel）
  - `softmax_f32_per_token_kernel`：`dim3 block((H)); dim3 grid((S));`，`idx = blockIdx.x * blockDim.x + tid`；H ∈ `{32, 64, 128, 256, 512, 1024}`，error 文本 `"only support H: 64/128/256/512/1024"`（**case 32 实际被接受，文本与代码不一致**）
  - `softmax_f32x4_per_token_kernel`：`const int NT = (H) / 4; dim3 block(NT); dim3 grid((S));`，`idx = (blockIdx.x * blockDim.x + tid) * 4`；H ∈ `{32, 64, 128, 256, 512, 1024, 2048, 4096}`，error `"only support H: 64/128/.../1024*4"`
  - `safe_softmax_f32_per_token_kernel`：同 f32 版，H ∈ `{32,64,128,256,512,1024}`
  - `safe_softmax_f32x4_per_token_kernel`：`block(H / 4)`、`grid(S)`，H ∈ `{32,64,128,256,512,1024,2048,4096}`
  - `safe_softmax_f16_f32_per_token_kernel`：`block(H)`、`grid(S)`，H ∈ `{32,64,128,256,512,1024}`
  - `safe_softmax_f16x2_f32_per_token_kernel`：`const int NT = (H) / 2; dim3 block(NT); dim3 grid((S));`，`idx = (...)*2`，H ∈ `{32,64,128,256,512,1024,2048}`
  - `safe_softmax_f16x8_pack_f32_per_token_kernel`：`const int NT = (H) / 8; dim3 block(NT); dim3 grid((S));`，`idx = (...)*8`，H ∈ `{32,64,128,256,512,1024,2048,4096,8192}`，error `"only support H: 64/128/.../1024*8"`
  - `online_safe_softmax_f32_per_token_kernel`：`dim3 block((H)); dim3 grid((S));`，`int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;`，H ∈ `{32,64,128,256,512,1024}`
  - `online_safe_softmax_f32x4_pack_per_token_kernel`：`dim3 block((H / 4)); dim3 grid((S));`，`global_tid = (blockIdx.x * NUM_THREADS + local_tid) * 4`，H ∈ `{128, 256, 512, 1024, 2048, 4096}`，error `"only support H: 128/256/.../4096;"`
  - 注释声明的语义：`// grid(S*h/h), block(h), assume h<=1024` / `// one token per thread block, only support 64<=h<=1024 and 2^n` / `// HEAD_SIZE/KV_LEN=NUM_THREADS`
- **关键常量**
  - `WARP_SIZE 32`；`constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;`（注释 `// always <= 32 warps per block (limited by 1024 threads per block)`）
  - `static __shared__ float shared[NUM_WARPS];`（两个 block_reduce 各自一份）
  - 哨兵：max 用 `-FLT_MAX`，sum 用 `0.0f`；online 版 `MD{-FLT_MAX, 0.0f}`、`val.d = global_tid < N ? 1.0f : 0.0f;`；`const int WARP_NUM = NUM_THREADS / WARP_SIZE;`；`__shared__ MD shared[WARP_NUM];`
  - 倒数：`float d_total_inverse = __fdividef(1.0f, final_res.d);`（不是 `1.0f / d`）
  - 除法：非 online 版用普通 `/`（配合 `--use_fast_math` 变近似），exp 用 `expf`（非 online）/ `__expf`（online 版与 MD 合并）
- **数值与容差**（`softmax.py`）
  - 无 tol、无断言、无种子；参考 = `partial(torch.softmax, dim=1, out=out)`，`out.fill_(0)` 后复用
  - 用例列表（S 全部为 4096）：`S,H = 4096,256` / `4096,512` / `4096,1024` / `4096,2048` / `4096,4096` / `4096,8192`（此组只跑 `safe_softmax_f16x8_pack_f32_per_token` + torch）/ `8192,8192`（同样只跑 f16x8pack + torch）；`x = torch.randn((S, H), device="cuda").cuda().float().contiguous()`，`x_f16 = x.half().contiguous()`
  - 参数与其他文件不同：`warmup: int = 10, iters: int = 100`（**不是 1000**）；打印 `out.flatten().detach().cpu().numpy().tolist()[:3]` 并 `round(v, 8)`
  - 被注释的 grid-fence 用例：`# N = 128 * 128` + `torch.softmax, dim=0, out=out`
- **数据流骨架**
```text
// 非 online（block 内两趟：先 max/exp，再 sum）
val = (idx < N) ? x[idx] : -FLT_MAX;
max_val = block_reduce_max_f32<NUM_THREADS>(val);              // 无 max 版缺这一步
exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
if (idx < N) y[idx] = exp_val / exp_sum;

// block_reduce_sum_f32 内部（注意：第二级没有 if (warp == 0)）
value = warp_reduce_sum_f32<WARP_SIZE>(val);
if (lane == 0) shared[warp] = value;
__syncthreads();
value = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
value = warp_reduce_sum_f32<NUM_WARPS>(value);
value = __shfl_sync(0xffffffff, value, 0, 32);                 // 广播回所有 lane

// online：一次遍历同时携带 (m, d)
other.m = __shfl_xor_sync(mask, value.m, stride); other.d = __shfl_xor_sync(mask, value.d, stride);
bool value_bigger = (value.m > other.m);
MD bigger_m = value_bigger ? value : other; MD smaller_m = value_bigger ? other : value;
value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
value.m = bigger_m.m;
// 之后 if (local_tid < WARP_SIZE) { block_res = warp_reduce_md_op<WARP_NUM>(...); if (local_tid==0) shared[0]=block_res; }
```
- **处理边界的写法**
  - 非向量版：`(idx < N) ? x[idx] : ...`；向量版一律先无守卫载入（`float4 reg_x = FLOAT4(x[idx]);`、`float2 reg_x = __half22float2(HALF2(x[idx]));`、`LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);`），再用 `(idx + k) < N` 逐分量守卫
  - 写回守卫更粗：x4 版 `if (idx + 3 < N)`（要求整块 4 个都有效），f16x2 版 `if ((idx + 1) < N)`，f16x8_pack 版 `if ((idx + 7) < N)`（x8_pack 的 `pack_y[i]` 计算本身不守卫）
  - online f32x4 版：读 `float4 val = FLOAT4((x)[global_tid]);` 无守卫，写回守卫 `if (global_tid < N)`
  - partial warp：非 online 版用 `-FLT_MAX` / `0.0f` 补位；online 版 `local_tid < WARP_NUM ? shared[local_tid] : MD{-FLT_MAX, 0.0f}`，并额外用 `if (local_tid < WARP_SIZE)` 只让 32 个线程做第二级归约
- **常见坑/注释里的警告**
  - `// WRAN: need to broadcast value to all threads within warp`（**"WRAN" 是 "WARN" 的拼写错误**，两处）
  - `// one token per thread block, only support 64<=h<=1024 and 2^n`（与 dispatch 实际支持到 8192 冲突）
  - `// TODO: support non 8-multiple K here`（f16x8_pack 版末尾）
  - `// reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)`（online 两个版本各一份）
  - `// e^x_i/sum(e^x_0,...,e^x_n-1)` 出现在每个 kernel 的写回前
  - 工程性坑：`block_reduce_sum_f32` / `block_reduce_max_f32` 只在“写 shared”与“读 shared”之间放了一次 `__syncthreads()`，**没有第二次屏障**；safe softmax 连续调用 max 再 sum 时存在 WAR 窗口（后一次写 `shared[warp]` 早于前一次所有 warp 读完 `shared[lane]`）
  - 工程性坑：`block_reduce_*` 第二级归约由**所有** warp 执行（不像 reduce/dot 用 `if (warp == 0)`），靠最后的 `__shfl_sync(..., 0, 32)` 广播；且 H=32 的 x4 版 block 只有 8 线程、f16x8_pack 版只有 4 线程，第一级 `warp_reduce_sum_f32<32>` 仍用 `__shfl_xor_sync(0xffffffff, ...)` 全掩码

---

## kernels/layer-norm/layer_norm.cu

- **导出/定义的符号**
  - device helper：`float warp_reduce_sum_f32<kWarpSize = WARP_SIZE>`、`half warp_reduce_sum_f16_f16<kWarpSize = WARP_SIZE>`、`float warp_reduce_sum_f16_f32<kWarpSize = WARP_SIZE>`、`float block_reduce_sum_f32<NUM_THREADS = 256>`（`// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.`）、`half block_reduce_sum_f16_f16<NUM_THREADS = 256>`（`return val; // half`）、`float block_reduce_sum_f16_f32<NUM_THREADS = 256>`（`return val_f32; // float`）
  - 8 个 kernel（`(half|float *x, half|float *y, float g, float b, int N, int K)`，**g/b 是标量 float 参数，不是张量**）：`layer_norm_f32_kernel<256>`、`layer_norm_f32x4_kernel<256 / 4>`、`layer_norm_f16_f16_kernel<256>`、`layer_norm_f16x2_f16_kernel<256>`、`layer_norm_f16x8_f16_kernel<256>`、`layer_norm_f16_f32_kernel<256>`、`layer_norm_f16x8_pack_f16_kernel<256>`、`layer_norm_f16x8_pack_f32_kernel<256>`
  - 宏：`HALF2_SUM(reg, i)`、`HALF2_SUB(reg_y, reg_x)`、`HALF2_VARIANCE(reg, i)`、`HALF2_LAYER_NORM(reg_y, reg_x, g_, b_)` + `STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、`CHECK_TORCH_TENSOR_SHAPE`、8 组 `LANUCH_LAYER_NORM_*_KERNEL` / `DISPATCH_LAYER_NORM_*_KERNEL`
  - host：`void layer_norm_<variant>(torch::Tensor x, torch::Tensor y, float g, float b)`：`layer_norm_f32`、`layer_norm_f32x4`、`layer_norm_f16_f16`、`layer_norm_f16_f32`、`layer_norm_f16x2_f16`、`layer_norm_f16x8_f16`、`layer_norm_f16x8_pack_f16`、`layer_norm_f16x8_pack_f32`（每个都 `const int N = x.size(0); const int K = x.size(1);`）
  - `PYBIND11_MODULE` 注册上列 8 个（注意注册顺序里 `layer_norm_f16_f32` 排在 `layer_norm_f16x2_f16` 之前）
- **每个 kernel 的启动形状**（全部 `dim3 block(...); dim3 grid((N));`，即 **block 覆盖一行 K，grid = 行数 N**）
  - `layer_norm_f32_kernel` / `layer_norm_f16_f16_kernel` / `layer_norm_f16_f32_kernel`：`dim3 block((K));`，K ∈ `{64,128,256,512,1024}`，error `"only support K: 64/128/256/512/1024"`
  - `layer_norm_f32x4_kernel`：`dim3 block((K) / 4);`，K ∈ `{64,128,256,512,1024,2048,4096}`，error `"only support K: 64/128/.../1024*4"`
  - `layer_norm_f16x2_f16_kernel`：`dim3 block((K) / 2);`，K ∈ `{64,128,256,512,1024,2048}`，error `"only support K: 64/128/.../1024*2"`
  - `layer_norm_f16x8_f16_kernel`：`dim3 block((K) / 8);`，K ∈ `{64,128,256,512,1024,2048,4096,8192}`，error `"only support K: 64/128/.../1024*8"`
  - `layer_norm_f16x8_pack_f16_kernel` / `layer_norm_f16x8_pack_f32_kernel`：`dim3 block((K) / 8);`，同一组 K 列表
  - 索引写法：`int idx = bid * blockDim.x + threadIdx.x;`（标量）/ `* 4` / `* 2` / `* 8`；`int tid = threadIdx.x; // 0..K-1`、`int bid = blockIdx.x; // 0..N-1`
  - 注释声明：`// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size`
- **关键常量**
  - `const float epsilon = 1e-5f;`（f32 路径）/ `const half epsilon = __float2half(1e-5f);`（f16 路径）
  - f16 路径预转换：`const half g_ = __float2half(g); const half b_ = __float2half(b); const half K_ = __int2half_rn(K); const half z_ = __float2half(0.0f);`
  - 计算式：`s_mean = sum / K_`（half）或 `s_mean = sum / (float)K`；`s_variance = hrsqrt(variance / K_ + epsilon)`（half，**hrsqrt**）或 `s_variance = rsqrtf(variance / (float)K + epsilon)`（float）；写回 `y[idx] = __hfma((value - s_mean) * s_variance, g_, b_);` / `__fmaf_rn(((value - s_mean) * s_variance), g, b)`
  - epilogue 说明：`// x*y + z -> x'*g + b`；f16x8_pack_f16 的 epilogue `pack_y[i] = __hfma((pack_x[i] - s_mean) * s_variance, g_, b_);`
- **数值与容差**（`layer_norm.py`）
  - 无 tol、无断言、无种子；参考实现 `naive_layer_norm(x, g, b)`：`s_mean = torch.mean(x, dim=1, keepdim=True)  # m`、`s_variance = 1 / torch.std(x, dim=1, keepdim=True)  # 1/std(x)`、`y = ((x - s_mean) * s_variance) * g + b` —— 全部在 GPU、与输入同 dtype；**`torch.std` 默认 `correction=1`（无偏，除 K-1），而 kernel 是 `variance / K`（有偏）**，两者定义不一致
  - 超参：`g = 1.0`、`b = 0.0`（hardcode 在 run_benchmark 里），`warmup = 10, iters = 1000`
  - shape 列表：`N, K = 4096, 512`；`4096, 1024`；`4096, 2048`；`4096, 4096`；`4096, 8192`（只测 f16x8 系）；`8192, 8192`（只测 f16x8 系）；另有专门的溢出组：紧跟 `print(" " * 40 + f"f16 overflow without f32")` 之后 `x_f16 = x.half() * 100  # this will cause overflow for kernels without \`f32\``
  - 打印：`out.flatten().detach().cpu().numpy().tolist()[:3]` + `round(v, 8)`
- **数据流骨架**
```text
tid = threadIdx.x; bid = blockIdx.x; idx = (bid * blockDim.x + threadIdx.x) * n_elements;
value = (idx < N * K) ? x[idx] : 0.0f;                       // "load once only"
sum = block_reduce_sum_f32<NUM_THREADS>(value);
if (tid == 0) s_mean = sum / (float)K;
__syncthreads();                                            // wait for s_mean ... ready for all threads
variance = (value - s_mean) * (value - s_mean);
variance = block_reduce_sum_f32<NUM_THREADS>(variance);
if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
__syncthreads();                                            // wait for s_variance ... ready
if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
```
  - block_reduce 内部：`val = warp_reduce_sum_f32<WARP_SIZE>(val); if (lane == 0) shared[warp] = val; __syncthreads(); val = (lane < NUM_WARPS) ? shared[lane] : 0.0f; val = warp_reduce_sum_f32<NUM_WARPS>(val); return val;`（**没有 `__shfl_sync` 广播，返回值只在 warp 0 内正确**——因为随后写进 `s_mean`/`s_variance` 共享内存，再由 `__syncthreads()` 广播）
- **处理边界的写法**
  - 载入守卫一律 `(idx < N * K)`，其中 `idx` 是**向量基地址**（x4/x8 版判的是 `idx`，不是 `idx+3`/`idx+7`）→ 尾部非整向量时既越界读也越界写
  - 逐元素守卫只出现在 x8 系：`HALF2_SUM(reg, i)`/`HALF2_VARIANCE(reg, i)` 里 `(((idx + (i)) < N * K) ? ... : __float2half(0.0f))`，pack 版写回用 `if ((idx + 0/2/4/6) < N * K)`（f16x8_f16）或 `if ((idx + 7) < N * K)`（pack 版，**要求整 8 个都有效**）
  - `s_mean` 的除法用编译期 `K`，因此隐含“block 覆盖整行”；`s_mean`/`s_variance` 都是 `__shared__` 标量，靠两次 `__syncthreads()` 传播，**没有** warp 内广播
  - partial warp：同 reduce，用 `(lane < NUM_WARPS) ? shared[lane] : 0.0f` 补单位元
- **常见坑/注释里的警告**
  - `// Layer Norm: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row` 与 `// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row` —— 其中 `K=256<1024` 与 dispatch 支持到 4096/8192 矛盾；`y'=x-mean(x)/std(x)` 少了括号
  - `// wait for s_mean in shared memory to be ready for all threads` / `// wait for s_variance in shared memory to be ready for all threads`（两处逐字重复）
  - `// TODO: use __hfma2, __hsub2, __hmul2 here`（f16x8_pack_f16）、`// TODO: support non 8-multiple K here`（两个 pack 版末尾）
  - 与 reduce/dot 相**反**的取舍：本文件用的是 `val += __shfl_xor_sync(...)`，而 `// val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));` 被注释掉（依赖 `-U__CUDA_NO_HALF_OPERATORS__` 才可编译）
  - f16 全精度路径的溢出风险在 .py 里被显式点名（`x_f16 = x.half() * 100`，`f16 overflow without f32`）

---

## kernels/rms-norm/rms_norm.cu

- **导出/定义的符号**
  - device helper：`float warp_reduce_sum_f32<kWarpSize = WARP_SIZE>`、`half warp_reduce_sum_f16_f16<kWarpSize = WARP_SIZE>`、`float warp_reduce_sum_f16_f32<kWarpSize = WARP_SIZE>`、`float block_reduce_sum_f32<NUM_THREADS = 256>`（`__device__ __forceinline__`，比 layer_norm 多一个 `__forceinline__`）、`half block_reduce_sum_f16_f16<NUM_THREADS = 256>`、`float block_reduce_sum_f16_f32<NUM_THREADS = 256>`
  - 9 个 kernel（`(half|float *x, half|float *y, float g, int N, int K)`，**只有 scale g，没有 bias**）：`rms_norm_f32_kernel<256>`、`rms_norm_f32x4_kernel<256 / 4>`、`rms_norm_f16_f16_kernel<256>`、`rms_norm_f16x2_f16_kernel<256>`、`rms_norm_f16x8_f16_kernel<256>`、`rms_norm_f16x8_f32_kernel<256>`、`rms_norm_f16_f32_kernel<256>`、`rms_norm_f16x8_pack_f16_kernel<256>`、`rms_norm_f16x8_pack_f32_kernel<256>`
  - 宏：`HALF2_VARIANCE(reg, i)`、`FLOAT2_VARIANCE(reg, i)`、`HALF2_RMS_NORM(reg_y, reg_x, g)`、`FLOAT2_RMS_NORM(reg_y, reg_x, g)` + `STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、`CHECK_TORCH_TENSOR_SHAPE`、9 组 `LANUCH_RMS_NORM_*_KERNEL` / `DISPATCH_RMS_NORM_*_KERNEL`
  - host：`void rms_norm_<variant>(torch::Tensor x, torch::Tensor y, float g)`：`rms_norm_f32`、`rms_norm_f32x4`、`rms_norm_f16_f16`、`rms_norm_f16x2_f16`、`rms_norm_f16x8_f16`、`rms_norm_f16x8_f32`、`rms_norm_f16_f32`、`rms_norm_f16x8_pack_f16`、`rms_norm_f16x8_pack_f32`
  - `PYBIND11_MODULE` 注册上列 9 个
- **每个 kernel 的启动形状**（全部 `dim3 block(...); dim3 grid((N));`）
  - `rms_norm_f32_kernel` / `rms_norm_f16_f16_kernel` / `rms_norm_f16_f32_kernel`：`dim3 block((K));`，K ∈ `{64,128,256,512,1024}`，error `"only support K: 64/128/256/512/1024"`
  - `rms_norm_f32x4_kernel`：`dim3 block((K) / 4);`，K ∈ `{64,128,256,512,1024,2048,4096}`，error `"only support K: 64/.../512/1024*4"`
  - `rms_norm_f16x2_f16_kernel`：`dim3 block((K) / 2);`，K ∈ `{64,128,256,512,1024,2048}`，error `"only support K: 64/128/.../1024*2"`
  - `rms_norm_f16x8_f16_kernel` / `rms_norm_f16x8_pack_f16_kernel` / `rms_norm_f16x8_pack_f32_kernel`：`dim3 block((K) / 8);`，K ∈ `{64,128,256,512,1024,2048,4096,8192}`，error `"only support K: 64/128/.../1024*8"`
  - `rms_norm_f16x8_f32_kernel`：`dim3 block((K) / 8);`，同一组 K 列表（模板实参 `<K / 8>`）
  - 索引写法：`int idx = bid * blockDim.x + threadIdx.x;` / `... * 2` / `... * 8`；`int tid = threadIdx.x; // 0..K-1`、`int bid = blockIdx.x; // 0..N-1`
  - 注释声明：`// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size`、`// y=y'*g (g: scale)`
- **关键常量**
  - `const float epsilon = 1e-5f;` / `const half epsilon = __float2half(1e-5f);`；`const half g_ = __float2half(g); const half K_ = __int2half_rn(K); const half z_ = __float2half(0.0f);`
  - 核心式：`s_variance = rsqrtf(variance / (float)K + epsilon);`（f32）/ `s_variance = hrsqrt(variance / K_ + epsilon);`（half）；写回 `y[idx] = (value * s_variance) * g;`、`y[idx] = __float2half((value * s_variance) * g);`、f16x8_pack 用 `pack_y[i] = pack_x[i] * s_variance * g_;`
  - HALF2_RMS_NORM / FLOAT2_RMS_NORM 展开为 `(reg_y).x = (reg_x).x * s_variance * (g); (reg_y).y = (reg_x).y * s_variance * (g);`
  - `rms_norm_f16x8_f32_kernel` 的 128-bit 载入被注释明确指向 L2：见下方“常见坑”
- **数值与容差**（`rms_norm.py`）
  - 无 tol、无断言、无种子；参考实现 `naive_rms_norm(x, g)`：`s_rms = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True))`、`y = (x * s_rms) * g`（注释 `# y'=x/rms(x) 1/rms(x) = rsqrtf(sum(x^2)/K)`），GPU 同 dtype，**没有 eps**（kernel 加 `1e-5`）
  - 超参：`g = 1.0`（hardcode），`warmup = 10, iters = 1000`
  - shape 列表：`4096,512`；`4096,1024`；`4096,2048`；`4096,4096`；`4096,8192`（只测 f16x8 系）；`8192,8192`（只测 f16x8 系）；溢出组同 layer_norm：`print(" " * 40 + f"f16 overflow without f32")` + `x_f16 = x.half() * 100  # this will cause overflow for kernels without \`f32\``
  - 打印：`out.flatten().detach().cpu().numpy().tolist()[:3]` + `round(v, 8)`
- **数据流骨架**
```text
tid = threadIdx.x; bid = blockIdx.x; idx = (bid * blockDim.x + threadIdx.x) * n_elements;
value = (idx < N * K) ? x[idx] : 0.0f;                // "load once only"
variance = value * value;
variance = block_reduce_sum_f32<NUM_THREADS>(variance);
if (tid == 0) s_variance = rsqrtf(variance / (float)K + epsilon);
__syncthreads();                                      // wait for s_variance ... ready for all threads
if (idx < N * K) y[idx] = (value * s_variance) * g;
```
  - x8_pack f32 版的寄存器段：`float2 reg_x_0..3 = __half22float2(HALF2(x[idx + 0/2/4/6]));` → `variance += FLOAT2_VARIANCE(reg_x_k, 2k);` → 归约 → `for (int i = 0; i < 8; i += 2) { float2 v2 = __half22float2(HALF2(pack_x[i])); float2 y2 = {v2.x * s_variance * g, v2.y * s_variance * g}; HALF2(pack_y[i]) = __float22half2_rn(y2); }`
  - x8_f16 版则用 `HALF2_VARIANCE` + `variance += ...;`（手动展开 4 组 half2）→ `HALF2_RMS_NORM(reg_y_k, reg_x_k, g_);`
- **处理边界的写法**
  - 载入守卫 `(idx < N * K)`（向量版判基地址）；`HALF2_VARIANCE` / `FLOAT2_VARIANCE` 内是 `(((idx + (i)) < N * K) ? (平方和) : 单位元)`
  - x8_f16 写回逐 half2 守卫：`if ((idx + 0) < N * K) { HALF2(y[idx + 0]) = reg_y_0; }` … `(idx + 6)`；pack 版只用一个粗守卫 `if ((idx + 7) < N * K) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }`
  - `s_variance` 是 `__shared__` 标量 + 一次 `__syncthreads()` 广播（与 layer_norm 同构）；partial warp 靠 `(lane < NUM_WARPS) ? shared[lane] : 0.0f`（f16 版是 `__float2half(0.0f)`）
  - **没有** 逐元素守卫覆盖 128-bit 载入：`LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);` 永远执行
- **常见坑/注释里的警告**
  - `// manual unroll and improve L2 cache hit rate.` + `// Only   L2 cache: load 32  bytes in 1 memory issue (default)` + `// Enable L1 cache: load 128 bytes in 1 memory issue (-Xptxas -dlcm=ca)` + `// why try fp16x8 within 1 threads? ref:` + `// https://zhuanlan.zhihu.com/p/641639133 0. first, tid_0 load 32 bytes in 1` + `// memory issue and cache data into L2 cache.` + `// 1. then, tid_1,...,tid_3 hit L2 cache and load data from L2 cache directly.`（三重空格 `Only   L2` 原样保留）
  - `// TODO: support non 8-multiple K here`（f16x8_pack_f16 末尾）
  - `// temporary register(memory), .local space in ptx, addressable`、`// reinterpret as float4 and load 128 bits in 1 memory issue.`、`// 8x16 bits=128 bits.`
  - 与 layer_norm 一致的取舍：`val += __shfl_xor_sync(0xffffffff, val, mask);` 生效，`// val = __hadd(...)` 被注释
  - `// manual unroll` 出现 3 次（f16x8_f16、f16x8_f32 等），提示 x8 系列刻意不用 `#pragma unroll` 循环而手写 reg_x_0..3

---

## 跨文件差异速查（写教程时容易搞混）

- 每 block 归约的“收尾”不同：reduce / dot 用 `if (warp == 0)` + `if (tid == 0) atomicAdd(y, sum)`（输出 `torch::zeros({1})`）；softmax / layer_norm / rms_norm 用 `__shared__` 标量 + `__syncthreads()`（其中 softmax 额外做 `__shfl_sync(..., 0, 32)` 广播）。
- 2D 输入语义：reduce / dot 的 2D dispatch 只是“一行一个 block、再跨 block 原子累加”，**输出永远是 1 个标量**，不是 per-row 结果；只有 softmax / layer_norm / rms_norm 是真正的 per-row（`grid(S or N)`）实现。
- half 归约写法在 reduce / dot 里是 `__hadd`，在 layer_norm / rms_norm 里是 `+=`（`__hadd` 被注释）——两者都要求 `-U__CUDA_NO_HALF_OPERATORS__`。
- 宏名拼写错误全仓一致：`LANUCH_*`（应为 LAUNCH）、`STRINGFY`（应为 STRINGIFY）、softmax 的 `DISPATCH_SATE_*`（应为 SAFE）；`// WRAN`（应为 WARN）、`// operaion`（应为 operation）。引用教程时不要“修正”这些名字，否则与源码对不上。
- 本仓同目录下另有 triton 等实现带真实容差（如 `kernels/openai-triton/layer-norm/triton_layer_norm.py` 的 `assert torch.allclose(..., atol=1e-2, rtol=0)`、`kernels/openai-triton/fused-softmax/triton_fused_softmax.py` 的 `torch.manual_seed(0)`），**但这 5 个目标文件一个都没有**，不要把那些常数移植到本摘录对应的内核描述里。


---

# B · elementwise / 激活 / 直方图 / 转置（Q176 Q179 Q180 Q181）

# LeetCUDA 参考实现精确事实抽取（B: elementwise / relu / gelu / histogram / transpose / swizzle / nsight）

- 仓库：`C:\Users\Jeff\Documents\GitHub\LeetCUDA`
- HEAD：`e831d970a099f5ce8fd0495ddd1df09206d56918`（`git status --porcelain` 为空，工作区干净）
- 抽取方式：仅 `read`，未修改任何仓库文件。
- **行数与任务书不一致（以实测为准）**：任务书给的行数里，`elementwise.cu` 实为 **199** 行（非 186）、`relu.cu` 实为 **171** 行（非 158）、`gelu.cu` 实为 **248** 行（非 219）、`histogram.cu` 实为 **79** 行（非 70）、`mat_transpose.cu` 实为 **442** 行（非 416）、`mat_trans_swizzle.cu` 实为 **112** 行（非 95）。`nvidia-nsight/relu.cu` = 259 行，`nvidia-nsight/elementwise.cu` = 291 行，`bank_conflicts.md` = 85 行。
- **重要**：这批 `.cu` 中**没有任何 C++ 模板**（无 `template<...>`、无默认模板参数）。所谓“模板参数”的等价物是 X-macro（`TORCH_BINDING_*`）的展开实参，下文逐条原样列出。也没有任何 `constexpr`（除 `mat_transpose.cu` 的 `STRIDE` 与 nsight 的 `S/K/N`），常量全部是 `#define`。

---

## kernels/elementwise/elementwise.cu

- **导出/定义的符号**
  - 6 个 `__global__`：`elementwise_add_f32_kernel(float*,float*,float*,int)`、`elementwise_add_f32x4_kernel`、`elementwise_add_f16_kernel(half*,half*,half*,int)`、`elementwise_add_f16x2_kernel`、`elementwise_add_f16x8_kernel`、`elementwise_add_f16x8_pack_kernel`。
  - 宏：`STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、`TORCH_BINDING_ELEM_ADD(packed_type, th_type, element_type, n_elements)`（**无模板参数默认值**）。
  - X-macro 实例化（原样，即“模板实参”）：
    - `TORCH_BINDING_ELEM_ADD(f32, torch::kFloat32, float, 1)`
    - `TORCH_BINDING_ELEM_ADD(f32x4, torch::kFloat32, float, 4)`
    - `TORCH_BINDING_ELEM_ADD(f16, torch::kHalf, half, 1)`
    - `TORCH_BINDING_ELEM_ADD(f16x2, torch::kHalf, half, 2)`
    - `TORCH_BINDING_ELEM_ADD(f16x8, torch::kHalf, half, 8)`
    - `TORCH_BINDING_ELEM_ADD(f16x8_pack, torch::kHalf, half, 8)`
  - `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` 导出：`elementwise_add_f32`、`elementwise_add_f32x4`、`elementwise_add_f16`、`elementwise_add_f16x2`、`elementwise_add_f16x8`、`elementwise_add_f16x8_pack`。
  - `CHECK_TORCH_TENSOR_DTYPE` 对 a、b、c **三个** tensor 全部检查（`options().dtype() != th_type` → `throw std::runtime_error`）。

- **每个 kernel 的启动形状（host 宏内的确切写法）**
  - `ndim != 2` 分支：`int N = 1; for (i<ndim) N *= a.size(i);` → `dim3 block(256 / (n_elements)); dim3 grid((N + 256 - 1) / 256);`
  - `ndim == 2` 分支：`const int S = a.size(0); const int K = a.size(1); const int N = S * K;`
    - 若 `if ((K / (n_elements)) <= 1024)`：`dim3 block(K / (n_elements)); dim3 grid(S);`（**一层一行**，无向上取整）
    - 否则回落到全局 1D 形状：`dim3 block(256 / (n_elements)); dim3 grid((N + 256 - 1) / 256);`
  - 因此 f32/f16 → block 256；f32x4 → block 64；f16x2 → block 128；f16x8 / f16x8_pack → block 32。
  - kernel 内注释声明的形状（原样）：`// ElementWise Add grid(N/256), block(256)`；`// ElementWise Add + Vec4 grid(N/256), block(256/4)`。

- **关键常量**：`#define WARP_SIZE 32`（本文件未被使用）；其余为下述向量化宏。**没有 `constexpr`、没有 tile/pad/bin 常量**。

- **向量化方式与尾部守卫（逐 kernel 精确条件）**
  - `f32`：`idx = blockIdx.x*blockDim.x+threadIdx.x`；`if (idx < N)` 标量 `c[idx]=a[idx]+b[idx]`。
  - `f32x4`：`int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);`
    - 主路径守卫 `if ((idx + 3) < N)` → `float4 reg_a = FLOAT4(a[idx]);` … `FLOAT4(c[idx]) = reg_c;`（逐分量 `.x/.y/.z/.w` 相加）
    - 尾部 `else if (idx < N) { for (int i = 0; (idx + i) < N; i++) c[idx+i] = a[idx+i]+b[idx+i]; }` ← **标量 for 循环兜底**
  - `f16`：标量 `__hadd(a[idx], b[idx])`，`if (idx < N)`。
  - `f16x2`：`idx = 2 * (...)`；守卫 `if ((idx + 1) < N)` → `HALF2(a[idx])` / `HALF2(b[idx])` / `HALF2(c[idx])`，两分量 `__hadd`；尾部 `else if (idx < N) c[idx] = __hadd(a[idx], b[idx]);`（**只补 1 个元素**）。
  - `f16x8`：`idx = 8 * (...)`；守卫 `if ((idx + 7) < N)`；4 次 `half2` 读，偏移 **idx+0 / idx+2 / idx+4 / idx+6**；写回同上 4 个偏移；尾部 `else if (idx < N)` → 标量 for 循环。
  - `f16x8_pack`：`idx = 8 * (...)`；守卫 `if ((idx + 7) < N)`；`half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.`；`LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);`（一次 128bit 读）；`#pragma unroll for (int i = 0; i < 8; i += 2) HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));`（**i += 2**，4 次 `__hadd2`）；`LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);`；尾部 `else if (idx < N)` → 标量 for 循环。
  - 向量化宏（原样）：`INT4(value)`/`FLOAT4(value)`/`HALF2(value)` 均为 `(reinterpret_cast<... *>(&(value))[0])`；`#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])`；`#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])`。

- **bank conflict 的处理**：**无**。本文件不使用 shared memory，无 padding、无 swizzle。

- **测试与容差**（`kernels/elementwise/elementwise.py`）
  - shape：`Ss = [1024, 2048, 4096]`、`Ks = [1024, 2048, 4096]`、`SKs = [(S, K) for S in Ss for K in Ks]` → **9 个 (S,K)**。
  - 输入：`torch.randn((S, K)).cuda().float().contiguous()`；f16 用 `a.half().contiguous()`。
  - 参考实现：`partial(torch.add, out=c)`（f32）、`partial(torch.add, out=c_f16)`（f16）。
  - **无 tol、无 assert/allclose**：只打印 `out.flatten()...tolist()[:2]` 并 `round(v, 8)`，加 `mean_time`。`warmup: int = 10`、`iters: int = 1000`，计时用 `time.time()` + `torch.cuda.synchronize()`。
  - 编译 flags：`-O3 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`，`extra_cflags=["-std=c++17"]`。

- **注释里的关键结论（原样）**
  - `// temporary register(memory), .local space in ptx, addressable`（pack 版把 `half pack_a[8]` 放在寄存器/`.local`，靠 reinterpret 成 `float4` 做 128bit 访存）。
  - `// reinterpret as float4 and load 128 bits in 1 memory issue.` / `// reinterpret as float4 and store 128 bits in 1 memory issue.`
  - 结论：**只有 pack 版把 4 次 32bit 访存合并成 1 次 128bit**，非 pack 的 `f16x8` 是 4 次独立 `half2` 访存。

---

## kernels/relu/relu.cu

- **导出/定义的符号**
  - 6 个 `__global__`：`relu_f32_kernel(float*,float*,int)`、`relu_f32x4_kernel`、`relu_f16_kernel(half*,half*,int)`、`relu_f16x2_kernel`、`relu_f16x8_kernel`、`relu_f16x8_pack_kernel`。
  - 宏与 elementwise.cu 同名同形：`STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、`TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)`。
  - X-macro 实例化：`TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)`、`(f32x4, torch::kFloat32, float, 4)`、`(f16, torch::kHalf, half, 1)`、`(f16x2, torch::kHalf, half, 2)`、`(f16x8, torch::kHalf, half, 8)`、`(f16x8_pack, torch::kHalf, half, 8)`。
  - 导出：`relu_f32`、`relu_f32x4`、`relu_f16`、`relu_f16x2`、`relu_f16x8`、`relu_f16x8_pack`。
  - `CHECK_TORCH_TENSOR_DTYPE` 只检查 x、y 两个 tensor。

- **每个 kernel 的启动形状**：与 elementwise.cu **完全同构**——`dim3 block(256 / (n_elements))`；`dim3 grid((N + 256 - 1) / 256)`；二维且 `(K / (n_elements)) <= 1024` 时 `dim3 block(K / (n_elements)); dim3 grid(S);`。注释：`// grid(N/256), block(K=256)`（f32）与 `// grid(N/256/4), block(256/4)`（f32x4）。

- **关键常量**：`#define WARP_SIZE 32`（未使用）；无 `constexpr`；无 pad/bin/tile 常量。零点是运行期构造的常量：fp32 用字面量 `0.0f`，fp16 用 `__float2half(0.0f)`；pack 版另有 `const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};`。

- **向量化方式与尾部守卫（与 elementwise 的关键差异，务必照抄）**
  - `f32`：`if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);`
  - `f32x4`：`int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;`（注意：先加后乘，写法与 elementwise 的 `4 * (...)` 不同但等价）；守卫**只有** `if (idx < N)`，**没有 `(idx + 3) < N` 检查、没有 else 尾部**。
  - `f16`：`y[idx] = __hmax(__float2half(0.0f), x[idx]);`
  - `f16x2`：`idx = 2 * (...)`；守卫 `if (idx < N)`；内部先 `half2 reg_y = HALF2(y[idx]);`（**多读了一次 y**，随后被覆盖），再 `reg_y.x/.y = __hmax(__float2half(0.0f), reg_x.x/.y);`
  - `f16x8`：`idx = 8 * (...);` **函数开头没有整体守卫**；无条件读 `HALF2(x[idx+0]) / (x[idx+2]) / (x[idx+4]) / (x[idx+6])`；计算 8 个分量后，**只对 4 次存储分别加守卫**：`if ((idx + 0) < N)`、`if ((idx + 2) < N)`、`if ((idx + 4) < N)`、`if ((idx + 6) < N)`；**无标量兜底**。
  - `f16x8_pack`：`idx = 8 * (...)`；`const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};`；`half pack_x[8], pack_y[8]; // 8x16 bits=128 bits.`；`LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);`；`#pragma unroll for (int i = 0; i < 8; i += 2) HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);`（**i += 2**、用 `__hmax2`）；存储守卫 `if ((idx + 7) < N) { LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]); }`。

- **bank conflict 的处理**：**无**（不使用 shared memory）。

- **测试与容差**（`kernels/relu/relu.py`）
  - shape：`Ss = [1024, 2048, 4096]`、`Ks = [1024, 2048, 4096]`，笛卡尔积 **9 个 (S,K)**。
  - 参考实现：`torch.relu`（f32 与 f16 各一次）；输入 `torch.randn((S, K))`（f32）与 `x.half()`（f16）。
  - **tol：无；无正确性断言**，只打印前 2 个值与 `time:{mean_time:.8f}ms`。`warmup: int = 10`、`iters: int = 1000`。
  - 编译 flags 与 elementwise.py 完全相同（含 `--use_fast_math`）。

- **注释里的关键结论（原样）**
  - `// Relu x: N, y: N y=max(0,x)`
  - `// grid(N/256/4), block(256/4)`（f32x4 注释声明的形状；host 实际是 `block(256/4)`、`grid((N+255)/256)`）

---

## kernels/gelu/gelu.cu

- **导出/定义的符号**
  - 3 个 `__inline__ __device__` 函数：`half gelu_tanh_approximate(half x)`、`float gelu_tanh_approximate(float x)`、`float gelu_none_approximate(float x)`。
  - 6 个 `__global__`：`gelu_f32_kernel`、`gelu_f32x4_kernel`、`gelu_f16_kernel`、`gelu_f16x2_kernel`、`gelu_f16x8_kernel`、`gelu_f16x8_pack_kernel`。
  - 宏：`TORCH_BINDING_GELU(packed_type, th_type, element_type, n_elements)`；实例化 `(f32, torch::kFloat32, float, 1)`、`(f32x4, torch::kFloat32, float, 4)`、`(f16, torch::kHalf, half, 1)`、`(f16x2, torch::kHalf, half, 2)`、`(f16x8, torch::kHalf, half, 8)`、`(f16x8_pack, torch::kHalf, half, 8)`。
  - 导出：`gelu_f32`、`gelu_f32x4`、`gelu_f16`、`gelu_f16x2`、`gelu_f16x8`、`gelu_f16x8_pack`。
  - **算法开关宏（原样）**：`#define HALF_GELU_OPS gelu_tanh_approximate`、`#define GELU_OPS gelu_tanh_approximate`（即默认走 tanh 近似；`gelu_none_approximate` 定义了但**未被任何 kernel 调用**）。

- **每个 kernel 的启动形状**：与 elementwise/relu 同构 —— `dim3 block(256 / (n_elements)); dim3 grid((N + 256 - 1) / 256);`，二维且 `(K / (n_elements)) <= 1024` 时 `dim3 block(K / (n_elements)); dim3 grid(S);`。注释：`// grid(N/256), block(K=256)`、`// GELU tanh approximate; Vec4 // grid(N/256), block(256/4)`。

- **关键常量（全部 `#define`，原样数值）**
  - `#define MAX_EXP_F32 88.3762626647949f`
  - `#define MIN_EXP_F32 -88.3762626647949f`
  - `#define MAX_EXP_F16 __float2half(11.089866488461016f)`
  - `#define MIN_EXP_F16 __float2half(-9.704060527839234f)`
  - `#define SQRT_2_PI M_SQRT2 *M_2_SQRTPI * 0.5f`（原样，`M_SQRT2` 后无空格）
  - `#define HALF_1 __float2half(1.0f)`、`#define HALF_2 __float2half(2.0f)`、`#define HALF_DIV2 __float2half(0.5f)`
  - `#define HALF_SQRT_2_PI __float2half(M_SQRT2) * __float2half(M_2_SQRTPI) * HALF_DIV2`
  - `#define HALF_V_APP __float2half(0.044715f)`
  - 其余魔数（写死在函数体里）：fp32 tanh 近似 `0.5f * x * (1.0f + tanhf(SQRT_2_PI * (x + 0.044715f * x * x * x)))`；`gelu_none_approximate` 用 `M_SQRT1_2`。
  - **clamp 区间各不相同**：f32 用 ±88.3762626647949，f16 用 **[−9.704060527839234, 11.089866488461016]**（非对称）。
  - `WARP_SIZE 32` 仍定义但未用；无 `constexpr`。

- **向量化方式与尾部守卫**
  - fp16 tanh 用 exp 手写：`half x_cube = x * x * x; half inner = HALF_SQRT_2_PI * (x + HALF_V_APP * x_cube); return HALF_DIV2 * x * (HALF_1 + ((hexp(inner * HALF_2) - HALF_1) / (hexp(inner * HALF_2) + HALF_1)));`
  - `f32`：`float v = fminf(fmaxf(x[idx], MIN_EXP_F32), MAX_EXP_F32); y[idx] = GELU_OPS(v);`，守卫 `if (idx < N)`。
  - `f32x4`：`int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;` **无守卫地** `FLOAT4(x[idx])` 读；4 个分量各做 `fminf(fmaxf(...))` 与 `GELU_OPS`；存储守卫**只有** `if ((idx + 0) < N) { FLOAT4(y[idx]) = reg_y; }`（**没有 `idx+3` 检查，无尾部**）。
  - `f16`：`half v = x[idx]; v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16); y[idx] = HALF_GELU_OPS(v);`，守卫 `if (idx < N)`。
  - `f16x2`：`idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;` 无守卫读 `HALF2(x[idx])`；存储守卫 `if ((idx + 0) < N)`。
  - `f16x8`：`idx = (...)*8`；无守卫读 4 个 `half2`（偏移 +0/+2/+4/+6）；clamp 与 GELU **原地写回 `reg_x_0..reg_x_3`**（复用了输入寄存器名，`half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;` 声明后未使用）；4 个存储守卫 `if ((idx + 0) < N)`、`+2`、`+4`、`+6`；注释 `// unpack f16x8`。
  - `f16x8_pack`：`idx = (...)*8`；`half pack_x[8], pack_y[8];`；`LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);`；`#pragma unroll for (int i = 0; i < 8; ++i) { half v = __hmin(__hmax(pack_x[i], MIN_EXP_F16), MAX_EXP_F16); pack_y[i] = HALF_GELU_OPS(v); }`（**`++i` 逐元素标量，不是 `i += 2` 的 half2**）；存储守卫 `if ((idx + 7) < N) LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);`；注释 `// pack f16x8`。

- **bank conflict 的处理**：**无**（不使用 shared memory）。

- **测试与容差**（`kernels/gelu/gelu.py`）
  - **参考实现被猴子补丁成 tanh 版**：`torch.gelu = torch.nn.GELU("tanh")`，基准调用 `run_benchmark(partial(torch.gelu), x, "f32_th")`。
  - shape：`Ss = [1024, 2048, 4096]`、`Ks = [1024, 2048, 4096]`，**9 个 (S,K)**；输入 `torch.randn((S, K))`。
  - **tol：无；无 allclose 断言**，只打印前 2 值（`round(v, 8)`）与耗时。`warmup=10`、`iters=1000`。

- **注释里的关键结论（原样摘录）**
  - `// to clear the error among self defined gelu and pytorch gelu. Calculate $\sqrt{\frac{\pi}{2}}$ by $\sqrt{2 * \pi} / 2$`
  - `// There is no half presicion operation like sinh, cosh, tanh.`（原文拼写 `presicion`）以及 `// $$ tanh(x) = \frac{exp^{2x} - 1}{exp^{2x} + 1}$$ // But ops above will introduce error.`
  - `// pytorch transform type while do tanh operator which include in the [pytorch/c10/util/BFloat16-math.h](...)` ← **正是为对齐 PyTorch 的 fp16 tanh 提升行为才用 exp 手工实现 tanh**。

---

## kernels/histogram/histogram.cu

- **导出/定义的符号**
  - 2 个 `__global__`：`histogram_i32_kernel(int *a, int *y, int N)`、`histogram_i32x4_kernel(int *a, int *y, int N)`。
  - 宏：`STRINGFY`、`TORCH_BINDING_COMMON_EXTENSION`、`CHECK_TORCH_TENSOR_DTYPE`、**`CHECK_TORCH_TENSOR_SHAPE(T, S0)`（定义了但全文从未使用）**、`TORCH_BINDING_HIST(packed_type, th_type, element_type, n_elements)`。
  - 实例化：`TORCH_BINDING_HIST(i32, torch::kInt32, int, 1)`、`TORCH_BINDING_HIST(i32x4, torch::kInt32, int, 4)`。
  - 导出：`histogram_i32`、`histogram_i32x4`（**返回值是 tensor**，非 out 参数）。
  - 本文件只定义了 `INT4` 与 `FLOAT4` 两个向量宏，**没有** `HALF2`/`BFLOAT2`/`LDST128BITS`。

- **每个 kernel 的启动形状（host 内确切写法）**
  - `const int N = a.size(0);`
  - `static const int NUM_THREADS_PER_BLOCK = 256 / (n_elements);`
  - `const int NUM_BLOCKS = (N + 256 - 1) / 256;`
  - `dim3 block(NUM_THREADS_PER_BLOCK); dim3 grid(NUM_BLOCKS);`
  - → i32：block 256；i32x4：block 64，两者 grid 都按 **256** 向上取整（**不是** 256/4=64 个元素一组）。
  - kernel 注释：`// Histogram grid(N/256), block(256)` 与 `// Histogram + Vec4 grid(N/256), block(256/4)`。

- **关键常量 / bin 数**
  - **bin 数 = `M + 1`**，其中 `M` 由 `torch::max(a, 0)` 求全局最大值得到：`std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0); torch::Tensor max_val = std::get<0>(max_a).cpu(); const int M = max_val.item().to<int>(); auto y = torch::zeros({M + 1}, options);`
  - `auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);`（**硬编码 device 0**）
  - 无 `constexpr`、无 pad 常量。`WARP_SIZE 32` 定义但未用、`FLOAT4` 定义但未用。

- **向量化方式与尾部守卫**
  - `i32`：`int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < N) atomicAdd(&(y[a[idx]]), 1);`
  - `i32x4`：`int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);` 守卫**只有** `if (idx < N)`（**没有 `(idx + 3) < N`，无尾部处理**）；`int4 reg_a = INT4(a[idx]);` 然后 4 次独立 `atomicAdd(&(y[reg_a.x]), 1); atomicAdd(&(y[reg_a.y]), 1); atomicAdd(&(y[reg_a.z]), 1); atomicAdd(&(y[reg_a.w]), 1);`
  - 关键点：**向量化只合并了“读”，写侧仍是 4 次全局 atomicAdd**，bin 冲突/原子竞争未被优化。

- **bank conflict 的处理**：**无**（不用 shared memory、不用私有直方图/复制，直接全局原子加）。

- **测试与容差**（`kernels/histogram/histogram.py`）
  - 输入：`a = torch.tensor(list(range(10)) * 1000, dtype=torch.int32).cuda()` → N = 10000，取值 0..9，故 M = 9，**y 长度 10，每 bin 期望 1000**。
  - **无参考实现、无 tol、无 assert**：仅 `print(f"h_i32   {i}: {h_i32[i]}")` 逐 bin 打印。
  - 注意：kernel 注释写 `// a: Nx1, y: count histogram, a >= 1`，但测试数据**含 0**。

- **注释里的关键结论**：`// Histogram`；`// Histogram + Vec4`；`// a: Nx1, y: count histogram, a >= 1`。

---

## kernels/mat-transpose/mat_transpose.cu

- **导出/定义的符号**（14 个自定义 `__global__` + 11 个 extern CuTe 声明）
  - 1D 索引：`mat_transpose_f32_col2row_kernel`、`mat_transpose_f32_row2col_kernel`、`mat_transpose_f32x4_col2row_kernel`、`mat_transpose_f32x4_row2col_kernel`（签名均为 `(float *x, float *y, const int row, const int col)`，前两个为 `(float*,float*,const int,const int)`）。
  - 2D 索引：`mat_transpose_f32_diagonal2d_kernel(float *x, float *y, int row, int col)`、`mat_transpose_f32_col2row2d_kernel`、`mat_transpose_f32_row2col2d_kernel`、`mat_transpose_f32x4_col2row2d_kernel`、`mat_transpose_f32x4_row2col2d_kernel`。
  - shared：`mat_transpose_f32x4_shared_col2row2d_kernel`、`mat_transpose_f32x4_shared_row2col2d_kernel`。
  - shared + BCF：`mat_transpose_f32x4_shared_bcf_col2row2d_kernel`、`mat_transpose_f32x4_shared_bcf_row2col2d_kernel`、`mat_transpose_f32x4_shared_bcf_merge_write_row2col2d_kernel`。
  - 宏：`TORCH_BINDING_MAT_TRANSPOSE(tag, th_type, element_type, n_pack)`、`TORCH_BINDING_MAT_TRANSPOSE2D(tag, th_type, element_type, n_element_row, n_element_col)`（**注意：1D 用 `n_pack`，2D 用 `n_element_row/n_element_col` 两个参数**）。
  - extern CuTe（定义在 `mat_transpose_cute.cu`，仅声明）：`mat_transpose_cute_col2row_reg`、`_row2col_reg`、`_col_smem`、`_row_smem`、`_col_smem_swizzled`、`_row_smem_swizzled`、`_row_cvectorized`、`_row_rvectorized`、`_row_cvectorized_swizzled`、`_row_rvectorized_swizzled`、`_row_rvectorized_swizzled_optimized`。
  - pybind 导出 14 个：`mat_transpose_f32_col2row`、`_f32x4_col2row`、`_f32_row2col`、`_f32x4_row2col`、`_f32_col2row2d`、`_f32x4_col2row2d`、`_f32_row2col2d`、`_f32x4_row2col2d`、`_f32_diagonal2d`、`_f32x4_shared_col2row2d`、`_f32x4_shared_row2col2d`、`_f32x4_shared_bcf_col2row2d`、`_f32x4_shared_bcf_row2col2d`、`_f32x4_shared_bcf_merge_write_row2col2d`，外加 11 个 `mat_transpose_cute_*`。

- **关键常量（原样，注意与其它文件冲突）**
  - `#define WARP_SIZE 256` ← **本文件里 `WARP_SIZE` = 256，不是 32**，被当作 1D 的 block 维度用。
  - `#define WARP_SIZE_S 16` ← 2D 的 tile 边长（block 为 16×16 = 256 线程）。
  - `#define PAD 1` ← padding 就是 1 个 float。
  - kernel 内 `constexpr int STRIDE = WARP_SIZE_S / 4;` → **= 4**（在 shared / shared_bcf 的四个 kernel 中各出现一次）。

- **每个 kernel 的启动形状**
  - 1D（`TORCH_BINDING_MAT_TRANSPOSE`）：`const int M = x.size(0); const int N = x.size(1);` → `dim3 block(WARP_SIZE);`（=256）→ `dim3 grid(((N * M + WARP_SIZE - 1) / n_pack / WARP_SIZE));`（等价 `ceil(M*N / n_pack / 256)`；n_pack = 1 或 4）。
  - 2D（`TORCH_BINDING_MAT_TRANSPOSE2D`）：`dim3 block(WARP_SIZE_S, WARP_SIZE_S);`（16,16）→ `dim3 grid((N + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_col), (M + WARP_SIZE_S - 1) / (WARP_SIZE_S * n_element_row));`
    - f32_col2row2d / f32_row2col2d / f32_diagonal2d：`(1, 1)` → grid = `(ceil(N/16), ceil(M/16))`
    - f32x4_col2row2d、f32x4_shared_col2row2d、f32x4_shared_bcf_col2row2d：`(1, 4)` → grid = `(ceil(N/64), ceil(M/16))`
    - f32x4_row2col2d、f32x4_shared_row2col2d、f32x4_shared_bcf_row2col2d、f32x4_shared_bcf_merge_write_row2col2d：`(4, 1)` → grid = `(ceil(N/16), ceil(M/64))`
  - 1D 实例化：`TORCH_BINDING_MAT_TRANSPOSE(f32_col2row, torch::kFloat32, float, 1)`、`(f32_row2col, ..., 1)`、`(f32x4_col2row, ..., 4)`、`(f32x4_row2col, ..., 4)`。
  - 2D 实例化（原样参数序 `n_element_row, n_element_col`）：
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32_col2row, torch::kFloat32, float, 1, 1)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32_row2col, torch::kFloat32, float, 1, 1)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_col2row, torch::kFloat32, float, 1, 4)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_row2col, torch::kFloat32, float, 4, 1)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32_diagonal, torch::kFloat32, float, 1, 1)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_col2row, torch::kFloat32, float, 1, 4)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_row2col, torch::kFloat32, float, 4, 1)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_col2row, torch::kFloat32, float, 1, 4)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_row2col, torch::kFloat32, float, 4, 1)`
    - `TORCH_BINDING_MAT_TRANSPOSE2D(f32x4_shared_bcf_merge_write_row2col, torch::kFloat32, float, 4, 1)`
  - 行/列语义：kernel 参数 `(row, col)` 由 host 传入 `(M, N)`，即 `row = x.size(0)`、`col = x.size(1)`。

- **向量化方式与尾部守卫（逐 kernel）**
  - `f32_col2row`：`global_row = global_idx / col; global_col = global_idx % col;` 守卫 `if (global_idx < row * col)`，写 `y[global_col * row + global_row] = x[global_idx];`
  - `f32_row2col`：`global_col = global_idx / row; global_row = global_idx % row;` 守卫 `if (global_idx < row * col)`，写 `y[global_idx] = x[global_row * col + global_col];`
  - `f32x4_col2row`（1D）：`global_col = (global_idx * 4) % col; global_row = (global_idx * 4) / col;` 守卫 **`if (global_row < row && global_col + 3 < col)`**；`float4 x_val = reinterpret_cast<float4 *>(x)[global_idx];` 然后 4 次标量写 `y[global_col*row+global_row]`、`y[(global_col+1)*row+global_row]`、`+2`、`+3`。**无 else 尾部**。
  - `f32x4_row2col`（1D）：`global_col = (global_idx * 4) / row; global_row = (global_idx * 4) % row;` 守卫 `if (global_row < row && global_col < col)`；4 次标量 gather `x[global_row*col+global_col]`、`x[(global_row+1)*col+global_col]`、`+2`、`+3` 装进 `float4 x_val`；写 `reinterpret_cast<float4 *>(y)[global_idx] = FLOAT4(x_val);`
  - `f32_diagonal2d`：`const int block_y = blockIdx.x; const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;`（**对角线错位映射**）；`global_col = threadIdx.x + blockDim.x * block_x; global_row = threadIdx.y + blockDim.y * block_y;` 守卫 `global_col < col && global_row < row`；`y[global_row*col+global_col] = x[global_col*row+global_row];` 注释 `// work for row == col`。
  - `f32_col2row2d`：`global_x = blockIdx.x*blockDim.x+threadIdx.x; global_y = blockIdx.y*blockDim.y+threadIdx.y;` 守卫 `global_x < col && global_y < row`；`y[global_x*row+global_y] = x[global_y*col+global_x];`
  - `f32_row2col2d`：守卫 `global_y < col && global_x < row`；`y[global_y*row+global_x] = x[global_x*col+global_y];`
  - `f32x4_col2row2d`：守卫 **`if (global_x * 4 + 3 < col && global_y < row)`**；`float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x];`（**注意 `col / 4` 整除假设**）；4 次标量写 `y[(global_x*4)*row+global_y]`、`+1`、`+2`、`+3`。
  - `f32x4_row2col2d`：守卫 `if (global_y * 4 + 3 < row && global_x < col)`；4 次标量 gather；写 `reinterpret_cast<float4 *>(y)[global_x * row / 4 + global_y] = FLOAT4(x_val);`
  - `f32x4_shared_col2row2d`：守卫 **`if (global_x * 4 + 3 < col + 3 && global_y < row)`** ← 比 `f32x4_col2row2d` **松**（等价 `global_x*4 < col`，允许 1~3 个越界元素）；`float4 x_val = reinterpret_cast<float4 *>(x)[global_y * col / 4 + global_x]; FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val); __syncthreads();` 读回 `smem_val.x = tile[(local_y % STRIDE) * 4][local_x * 4 + local_y / STRIDE];`（y/z/w 行号 +1/+2/+3）；输出 `const int bid_y = blockIdx.y * blockDim.y; const int out_y = global_x * 4 + local_y / STRIDE; const int out_x = (local_y % STRIDE) * 4 + bid_y;` 写 `reinterpret_cast<float4 *>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);`
  - `f32x4_shared_row2col2d`：守卫 `if (global_y * 4 < row && global_x < col)`；4 次标量 gather 存入 `tile[local_y*4 + k][local_x]`；读回 `smem_val.k = tile[local_x * 4 + local_y / STRIDE][(local_y % STRIDE) * 4 + k]`；`const int bid_x = blockIdx.x * blockDim.x; const int bid_y = blockIdx.y * blockDim.y;` `out_y = bid_x + (local_y % STRIDE) * 4; out_x = bid_y * 4 + local_x * 4 + (local_y / STRIDE);` 写 4 次标量 `y[out_y*row+out_x]`、`y[(out_y+1)*row+out_x]`、`+2`、`+3`。
  - `f32x4_shared_bcf_merge_write_row2col2d`：与上者同 tile 与装载；读回改为**列连续**：`smem_val.x = tile[local_x * 4][local_y]; smem_val.y = tile[local_x * 4 + 1][local_y]; smem_val.z = tile[local_x * 4 + 2][local_y]; smem_val.w = tile[local_x * 4 + 3][local_y];`；输出 `const int gid_x = blockIdx.x * blockDim.x; const int gid_y = blockIdx.y * blockDim.y * 4; const int out_y = gid_y + local_x * 4; const int out_x = gid_x + local_y;` 写 `reinterpret_cast<float4 *>(y)[(out_x * row + out_y) / 4] = FLOAT4(smem_val);`（**合并成一次 128bit 写**）。
  - 向量化宏：`INT4`/`FLOAT4`/`HALF2`/`LDST128BITS` 均为 `(reinterpret_cast<... *>(&(value))[0])`；`HALF2`、`LDST128BITS`、`INT4` 在本文件定义但**未使用**（只用 `FLOAT4`）。

- **bank conflict 的处理（原样公式与声明）**
  - padding 方式 = `PAD 1` 个 float，两种 tile 形状：
    - col2row 版：`__shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];` → 实际 `[16][65]`（未 pad 版本为 `[16][64]`）
    - row2col 版：`__shared__ float tile[WARP_SIZE_S * 4][WARP_SIZE_S + PAD];` → 实际 `[64][17]`（未 pad 版本为 `[64][16]`）
  - 此外还有基于 `STRIDE = WARP_SIZE_S / 4`（=4）的**索引重排**：把 16×16 逻辑块按 `(local_y % STRIDE) * 4 + local_y / STRIDE` 做 4×4 行列互换，注释原样为 `// add STRIDE to satisfied different block size.` 与 `// map index n*n to (n/4)*(n*4)`。
  - 命名后缀含义：`_bcf` = bank conflict free；`_merge_write` = 把 4 次标量写合并成 1 次 `float4` 写。
  - 本文件**没有 XOR swizzle**（swizzle 在 `kernels/swizzle/mat_trans_swizzle.cu`）。

- **测试与容差**（`kernels/mat-transpose/mat_transpose.py`）
  - shape：`Ms = [1024, 2048, 4096, 8192]`、`Ns = [1024, 2048, 4096, 8192]`，`MNs` 笛卡尔积 → **16 个 (M,N)**；`f32_diagonal2d` **仅在 `if M == N` 时**被测试。
  - 输入：`x = torch.arange(0, M * N).reshape(M, N).cuda().float().contiguous()`（**整数值，可精确比对**）；`y = torch.randn((N, M)).cuda().float().contiguous()`。
  - 参考实现：`partial(torch.transpose_copy, dim0=0, dim1=1, out=y)`，以及 `@torch.compile(mode="max-autotune-no-cudagraphs")` 包装的 `transpose_copy_compiled(input, out)`。
  - **容差：无（`tol` 不存在）**。验证是**精确相等**：`real_t = f"{out.T.equal(x)}"`，打印 `validate {real_t:<5}`。
  - 编译：sources = `["mat_transpose.cu", "mat_transpose_cute.cu"]`，`extra_include_paths=[os.path.join(CUTLASS_REPO_PATH, "include")]`，`CUTLASS_REPO_PATH = os.environ.get("CUTLASS_REPO_PATH", os.path.expanduser("../../third-party/cutlass"))`；flags 同上（含 `--use_fast_math`）；`torch._dynamo.config.suppress_errors = True`。
  - `warmup: int = 10`、`iters: int = 1000`；计时 `time.time()` + `torch.cuda.synchronize()`。

- **注释里的关键结论（原样）**
  - `// col2row means read x[row][col] and write y[col][row]` / `// row2col means read x[col][row] and write y[row][col]`
  - `// 2d index. easier for diagonal`（pybind 段）与 `// diagonal index method.`
  - `// shared memory optimize with bcf`、`// CuTe implentations`（原文拼写）

---

## kernels/swizzle/mat_trans_swizzle.cu

- **导出/定义的符号**（3 个 `__global__`，`int` 而非 `float`，**无 torch/pybind**）
  - `mat_trans_smem_naive_kernel(int *dev_A, int M, int N, int *dev_B)`
  - `mat_trans_smem_padding_kernel(int *dev_A, int M, int N, int *dev_B)`
  - `mat_trans_smem_swizzle_kernel(int *dev_A, int M, int N, int *dev_B)`
  - `int main(int argc, char *argv[])`

- **每个 kernel 的启动形状**（`main` 内，三者共用）
  - `int M = 1024; int N = 1024;`，`argc > 1` → `M = std::stoi(argv[1])`，`argc > 2` → `N = std::stoi(argv[2])`
  - `dim3 block(32, 32); dim3 grid(N / 32, M / 32);` ← **整数除法，无向上取整**（要求 M、N 均为 32 的倍数）
  - 三次依次 `<<<grid, block>>>` 启动并各自 `cudaDeviceSynchronize();`，最后 `printf("Done.\n");`
  - 分配：`size_t size_a = M * N * sizeof(int);` `cudaMalloc(&dev_A, size_a); cudaMalloc(&dev_B, size_b);` ← **dev_A 未初始化、无 memset/拷贝**，纯占位调用。

- **关键常量**：tile 尺寸在 kernel 内硬编码为 `32`（`s_data[32][32]` / `s_data[32][33]`）；padding = **1 个 int**；**无 `#define` / `constexpr` / 模板**。

- **向量化方式**：**无向量化**，全部是标量 `int` 访存；无尾部守卫（只有 `row < M && col < N` 之类的边界判断）。

- **bank conflict 的处理（原样位运算公式）**
  - naive：`__shared__ int s_data[32][32];`，写 `s_data[threadIdx.x][threadIdx.y] = dev_A[row * N + col];`，读 `dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x];`
  - padding：`__shared__ int s_data[32][33];`（**33 = 32 + 1**），其余语句与 naive **逐字相同**。
  - swizzle：`__shared__ int s_data[32][32];`（**不 padding**），
    - 写：`s_data[threadIdx.x][threadIdx.x ^ threadIdx.y] = dev_A[row * N + col];`
    - 读：`dev_B[n_row * M + n_col] = s_data[threadIdx.y][threadIdx.x ^ threadIdx.y];`
    - 即物理列 = 逻辑行 ⊕ 逻辑列（**XOR swizzle，`^` 原样**）。
  - 三者的索引映射一致：`n_col = blockIdx.y * blockDim.y + threadIdx.x; n_row = blockIdx.x * blockDim.x + threadIdx.y;`；naive/padding 的读守卫是 `if (n_col < M && n_row < N)`，而 swizzle 版写成 `if (n_row < N && n_col < M)`（**顺序相反但等价**）。

- **测试与容差**：**无测试**（无 `.py`、无容差、无参考实现；`main` 只做三次 launch 并打印 `Done.`）。文件头注释给出参考链接：`// reference: https://zhuanlan.zhihu.com/p/4746910252`。

- **注释里的关键结论（原样，中文）**
  - `// 每个block处理32*32的矩阵块`
  - `// 每个block处理32*32的矩阵块，尾部padding来避免bank conflict`
  - `// 从全局内存读取数据写入共享内存的逻辑坐标(row=x,col=y)` + `// 其映射的物理存储位置位置(row=x,col=x^y)`（原文“位置”重复）
  - `// 从共享内存的逻辑坐标(row=y,col=x)读取数据` + `// 其映射的物理存储位置(row=y,col=x^y)`

---

## kernels/nvidia-nsight/relu.cu

- **导出/定义的符号**：与 `kernels/relu/relu.cu` **同名同体的 6 个 `__global__`**：`relu_f32_kernel`、`relu_f32x4_kernel`、`relu_f16_kernel`、`relu_f16x2_kernel`、`relu_f16x8_kernel`、`relu_f16x8_pack_kernel`；加 `int main(int argc, char *argv[])`。**无 torch、无 pybind、无 `TORCH_BINDING_*`**。
- **向量化方式**：宏与 `kernels/relu/relu.cu` 逐字相同（`WARP_SIZE 32`、`INT4`、`FLOAT4`、`HALF2`、`BFLOAT2`、`LDST128BITS`），但 `BFLOAT2`/`LDST128BITS` 等在本文件仍被宏定义。
  - `f32x4` 守卫：`if (idx < N)`（**无 `idx+3`，无尾部**）
  - `f16x2` 守卫：`if (idx < N)`（与 relu.cu 相同，多读一次 `y[idx]`）
  - `f16x8`：无前置守卫，4 个存储守卫 `if ((idx + 0) < N)`、`(idx + 2)`、`(idx + 4)`、`(idx + 6)`
  - `f16x8_pack`：`const half2 z2 = {__float2half(0.0f), __float2half(0.0f)};`，`#pragma unroll for (int i = 0; i < 8; i += 2) HALF2(pack_y[i]) = __hmax2(HALF2(pack_x[i]), z2);`，存储守卫 `if ((idx + 7) < N)`
- **关键常量（`main` 内，原样）**：`constexpr int S = 4096; constexpr int K = 4096; constexpr int N = S * K;`（N = 16777216）；`int R = 10; // repeat`，`argc > 1` → `R = std::stoi(argv[1])`；`printf("S=%d, K=%d, R=%d\n", S, K, R);`；数据 `x_host[i] = (i % 2) ? 1.0 : -1.0;`（half，逐元素交替 ±1）。
- **每个变体的启动形状（4 个计时块，各自 warmup 5 次、计时 R 次、`cudaEventElapsedTime`）**
  - naive f16：`dim3 block(1024); dim3 grid((N + 1024 - 1) / 1024);` → 打印 `naive  relu: %f ms`
  - f16x2：`dim3 block(1024 / 2); dim3 grid((N + 1024 - 1) / 1024);` → `f16x2  relu: %f ms`（**grid 仍按 1024 向上取整**）
  - unpack f16x8：`dim3 block(K / (8)); // 4096/8=512` `dim3 grid(S);` → `unpack relu: %f ms`
  - pack f16x8：`dim3 block(K / (8)); // 4096/8=512` `dim3 grid(S);` → `pack   relu: %f ms`
- **bank conflict 的处理**：**无**（不用 shared memory）。
- **测试与容差**：`main` 自测，**无参考实现、无容差、无校验**；每段结束 `cudaMemcpy(y_host, y_device, ..., cudaMemcpyDeviceToHost);` 但不比对。
- **注释里的关键结论（原样摘录，性能反直觉的核心）**
  - `// manual unroll and improve L2 cache hit rate.`
  - `// Only   L2 cache: load 32  bytes in 1 memory issue (default)`
  - `// Enable L1 cache: load 128 bytes in 1 memory issue (-Xptxas -dlcm=ca)`
  - `// why try fp16x8 within 1 threads? ref: https://zhuanlan.zhihu.com/p/641639133 0. first, tid_0 load 32 bytes in 1 memory issue and cache data into L2 cache. 1. then, tid_1,...,tid_3 hit L2 cache and load data from L2 cache directly.`

---

## kernels/nvidia-nsight/elementwise.cu

- **导出/定义的符号**：与 `kernels/elementwise/elementwise.cu` 同名同体的 6 个 `__global__`：`elementwise_add_f32_kernel`、`_f32x4_`、`_f16_`、`_f16x2_`、`_f16x8_`、`_f16x8_pack_kernel`；加 `int main`。**无 torch、无 pybind**。
- **向量化方式与守卫（与 torch 版的关键差异）**
  - `f32x4`：`int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);` 守卫**只有** `if (idx < N)` —— **没有 `(idx + 3) < N`，也没有标量尾部**（与 `kernels/elementwise/elementwise.cu` 不同，后者两样都有）。
  - `f16x2`：守卫 `if (idx < N)`，**无尾部**。
  - `f16x8`：无前置守卫；4 个存储守卫 `if ((idx + 0) < N)`、`(idx + 2)`、`(idx + 4)`、`(idx + 6)`。
  - `f16x8_pack`：`half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.`；`LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);`、`LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);`；`#pragma unroll for (int i = 0; i < 8; i += 2) HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));`；存储守卫 `if ((idx + 7) < N)`。
- **关键常量**：`constexpr int S = 4096; constexpr int K = 4096; constexpr int N = S * K;`；`int R = 10; // repeat`（argv[1] 覆盖）；`a_host[i] = 1.0;`、`b_host[i] = 1.0;`（全 1，便于肉眼验证 c=2）。
- **每个变体的启动形状**：与 nsight/relu.cu **完全相同** —— naive `dim3 block(1024); dim3 grid((N + 1024 - 1) / 1024);`；f16x2 `dim3 block(1024 / 2);` 同 grid；unpack/pack `dim3 block(K / (8)); // 4096/8=512` `dim3 grid(S);`。打印串为 `naive  elementwise:`、`f16x2  elementwise:`、`unpack elementwise:`、`pack   elementwise:`。
- **bank conflict 的处理**：**无**。
- **测试与容差**：**无**（只 `cudaMemcpy` 回 host，不比对）。
- **注释里的关键结论**：`f16x8` kernel 内与 relu.cu 逐字相同的 L2 cache 注释（`// manual unroll and improve L2 cache hit rate.` … `// Enable L1 cache: load 128 bytes in 1 memory issue (-Xptxas -dlcm=ca)` …）。

---

## kernels/nvidia-nsight/README.md（实测数据与 PTX/SASS 证据，任务书未列但直接支撑上面的 .cu）

- 采集命令（原样）：`nvcc -arch=sm_89 -o relu.bin --generate-line-info -g relu.cu`；`nsys profile --stats=true -t cuda,osrt,nvtx -o relu.prof -f true relu.bin`；`ncu -o relu.prof -f relu.bin`（elementwise 版本在注释里给出）。
- **实测耗时（`S=4096, K=4096, R=10`，原样）**
  - `naive  relu: 0.058982 ms`
  - `f16x2  relu: 0.023962 ms`
  - `unpack relu: 0.037683 ms # f16x8`
  - `pack   relu: 0.015872 ms # f16x8_pack`
  - **反直觉点**：`unpack`（每线程 4×`half2`）**比 `f16x2` 更慢**（0.0377 vs 0.0240），而 `pack`（128bit 单次访存）最快（0.0159）——原因由 PTX 注释直接给出（见下）。
- **PTX/SASS 关键证据（原样摘录）**
  - `relu_f16_kernel`：PTX `ld.global.u16`、`{max.f16 %rs2,%rs1,%rs4;}`、`st.global.u16`；SASS `LDG.E.U16`、`HMNMX2 R0, R2.H0_H0, RZ.H0_H0, !PT`、`STG.E.U16`。
  - `relu_f16x8_kernel`（un-pack）：PTX 4 条 `ld.global.v2.u16 {%rs41, %rs42}, [%rd6];` … 标注 `// 非合并读`，4 条 `st.global.v2.u16` 标注 `// 非合并写`；SASS 4 条 `LDG.E R6, [R2.64]` / `+0x4` / `+0x8` / `+0xc` 标注 `// 非合并读`，`STG.E` 标注 `// 非合并写`。
  - `relu_f16x8_pack_kernel`（pack）：PTX `ld.global.v4.u32 {%r23, %r24, %r25, %r26}, [%rd6]; // 读合并`、4 条 `{max.f16x2 %r5,%r23,%r16;}`、`st.global.v4.u32 [%rd9], {%r5, %r8, %r11, %r14}; // 写合并`；SASS `LDG.E.128 R4, [R2.64] // 读合并`、4 条 `HMNMX2 R4/R5/R6/R7, ..., RZ.H0_H0, !PT`、`STG.E.128 [R2.64], R4 // 写合并`。

---

## kernels/nvidia-nsight/bank_conflicts.md（完整摘出）

- **查询 metric 的命令（原样）**
  - `ncu --query-metrics | grep data | grep bank | grep l1tex`
  - 注释：`# ncu check bank conflicts`、`# 先查看当前devices支持的metrics有哪些`
- **`l1tex__` 前缀 metrics（15 条，原样名称 + 官方描述）**
  - `l1tex__data_bank_conflicts_pipe_lsu` — # of data bank conflicts generated by LSU pipe
  - `l1tex__data_bank_conflicts_pipe_lsu_cmd_read` — ... generated by LSU reads
  - `l1tex__data_bank_conflicts_pipe_lsu_cmd_write` — ... generated by LSU writes
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_global` — ... generated by global ops
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_atom` — ... generated by global atomics
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld` — ... generated by global loads
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_red` — ... generated by global reductions
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_st` — ... generated by global stores
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` — # of shared memory data bank conflicts generated by LDS, LD, 3D
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom` — ... generated by ATOMS, ATOM
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` — ... generated by LDS, LD, 3D
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts` — ... generated by shared ldgsts ops
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` — ... generated by STS, ST, 3D
  - `l1tex__data_bank_reads` — # of data bank reads
  - `l1tex__data_bank_writes` — # of data bank writes
- **`sm__sass_` 前缀 metrics（6 条，原样名称 + 描述）**
  - `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts` — # of shared memory data bank conflicts generated by LDGSTS
  - `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts_cache_access` — ... by LDGSTS.ACCESS
  - `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts_cache_bypass` — ... by LDGSTS.BYPASS
  - `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm` — # of shared memory data bank conflicts generated by LDSM
  - `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st` — ... generated by STS, ST
  - `sm__sass_l1tex_data_bank_writes_pipe_lsu_mem_shared_op_ldgsts_cache_access` — # of LDGSTS.ACCESS shared data bank writes
- **`smsp__sass_` 前缀 metrics（6 条，与上一条 6 条同名仅前缀不同）**：`smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts`、`..._ldgsts_cache_access`、`..._ldgsts_cache_bypass`、`..._ldsm`、`..._op_st`、`smsp__sass_l1tex_data_bank_writes_pipe_lsu_mem_shared_op_ldgsts_cache_access`（最后一条描述为 `# of LDGSTS.ACCESS shared data bank writes`）。
  - 小结：共 **15 + 6 + 6 = 27** 条 metric 名列在文档中。
- **由 LD 指令产生的 bank conflicts**
  - 注释：`# profile l1tex smem data bank conflicts` / `# 由LDS, LD指令产生的bank conflicts`
  - 命令（原样）：
    - `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum hgemm_mma_stage.89.bin`
    - `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum hgemm_cute.89.debug.bin`
    - `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld \` + `python3 flash_attn_mma.py --B 1 --H 1 --D 64 --N 4096 --w 0 --i 1`
  - **实测表（Kernel: `flash_fwd_splitkv_combine_kernel<...>`, `(512, 1, 1)x(128, 1, 1)`, Context 1, Stream 7, Device 0, CC 8.9）**
    - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.avg` = **11.18**
    - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.max` = **13**
    - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.min` = **10**
    - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` = **1029**
- **由 LDSM（ldmatrix）指令产生的 bank conflicts**
  - 注释：`# 由LDSM(ldmatrix)指令产生的bank conflicts`
  - 命令（原样）：
    - `ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm \` + `python3 flash_attn_mma.py --B 1 --H 1 --D 64 --N 4096 --w 0 --i 1`
    - `ncu --metrics smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm \` + 同一条 python 命令
  - **实测表（同一 kernel，`(512, 1, 1)x(128, 1, 1)`, CC 8.9）**
    - `sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.avg` = **0**
    - `.max` = **0**、`.min` = **0**、`.sum` = **0**
- **结论要点**：该仓库给的是**如何量测 bank conflict 的 metric 白名单 + 两个实测样例**；样例显示 `LDS/LD` 路径存在真实冲突（sum 1029、avg 11.18），而 `LDSM`（ldmatrix）路径为 0 —— 即**不要用 LDSM 计数器去判断 LDS 冲突**。


---

# C · GEMM 阶梯：CUDA Core → cp.async → TF32 WMMA（Q183 Q184 Q185）

# C — LeetCUDA SGEMM 阶梯：参考实现精确事实提取

- 来源仓库：`C:\Users\Jeff\Documents\GitHub\LeetCUDA`
- HEAD：`e831d970a099f5ce8fd0495ddd1df09206d56918`（`git status --porcelain` 为空，工作区干净）
- 只读提取，未修改任何 LeetCUDA 文件。所有常量、索引表达式、宏名均按源码原样抄录。
- 校勘约定：`[源码 L###]` 表示行号；`⚠️注释/代码不一致` 为原样摘录 + 差异说明。

---

## 0. 文件规模与阶梯总览（实测行数）

| 文件 | 实测总行数（题面给数） | 实现的 kernel 级数 |
|:--|:--|:--|
| `kernels/sgemm/sgemm.cu` | 769（题面 699） | 5 级 CUDA Core（naive / sliced_k / t8x8 f32x4 / bcf / bcf_dbuf） |
| `kernels/sgemm/sgemm_async.cu` | 1015（题面 898） | 6 级 = 3 组 tile（8x4/8x8/8x16）× {同步 dbuf, cp.async dbuf} |
| `kernels/sgemm/sgemm_wmma_tf32_stage.cu` | 742（题面 670） | 2 个 WMMA kernel × {stage 2/3/4(/5)} + 1 个 f32→tf32 预处理 kernel |
| `kernels/interview/sgemm.cuh` | 181（题面 166） | 2 级（Level 1 `sgemm` / Level 1+ `sgemm_vec4`） |
| `kernels/sgemm/README.md` | 652 | 性能表 27 个 shape 块（L166–L651） |

---

## kernels/sgemm/sgemm.cu

- **阶梯**：共 5 级（`PYBIND11_MODULE` 绑定的 CUDA Core 版函数）：
  1. `sgemm_naive_f32_kernel` -> 每线程算 1 个 `c[m,n]`，全 row-major，无 smem [L21]
  2. `sgemm_sliced_k_f32_kernel<BM,BN,BK>` -> Block Tile + K Tile（smem），每线程仍 1 元素 [L42]
  3. `sgemm_t_8x8_sliced_k_f32x4_kernel` -> + Thread Tile 8×8 + float4 向量化（smem 行主序）[L96]
  4. `sgemm_t_8x8_sliced_k_f32x4_bcf_kernel` -> A 在 smem 中改列主序 `s_a[BK][BM]`，B 保持行主序，A 经寄存器 staging [L171]
  5. `sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel` -> Double Buffers（`s_a[2][BK][BM+OFFSET]`），主循环只 1 次 `__syncthreads()` [L350]
  - 另有 `OFFSET` 模板参数实例化：`_bcf_offset`（OFFSET=4 [L641]）、`_dbuf_offset`（OFFSET=4 [L695]）。
- **每级确切配置**：
  - naive：`constexpr int BM = 32; BN = 32;` `dim3 block(BN, BM)` = (32,32)=1024 线程；`dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM)` [L529-533]
  - sliced_k：`BM=32, BN=32, BK=32`（模板默认 [L41]，host 显式 `<BM,BN,BK>` [L562]）；`__shared__ float s_a[BM][BK], s_b[BK][BN];` = 32×32×4 B ×2 = 4KB+4KB=8KB；block(32,32)，grid 同上
  - t8x8：`BM=128, BN=128, BK=8, TM=8, TN=8`；`__shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB`（2KB+4KB=6KB 实际，注释 8KB 为笔误）；`dim3 block(BN / TN, BM / TM)` = (16,16)=256 线程；`dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM)` [L590-591]
  - bcf / bcf_dbuf：同 t8x8 的 `128/128/8/8/8` + `OFFSET`；dbuf 的 smem 为 `s_a[2][BK][BM+OFFSET], s_b[2][BK][BN+OFFSET]`，OFFSET=0 时 2×(8×128×4)×2 = 8KB+8KB=16KB；OFFSET=4 时每行 132 float（stride 528B）
- **加载映射**（逐级原式）：
  - naive：`int n = blockIdx.x * blockDim.x + threadIdx.x; int m = blockIdx.y * blockDim.y + threadIdx.y;`（n 沿 x、m 沿 y）；内积 `a[m * K + k] * b[k * N + n]`，写 `c[m * N + n]`
  - sliced_k：`tid = threadIdx.y * blockDim.x + tx;` `load_smem_a_m = tid / 32;` `load_smem_a_k = tid % 32;` `load_smem_b_k = tid / 32;` `load_smem_b_n = tid % 32;`；gmem：`load_gmem_a_m = by * BM + load_smem_a_m`、`load_gmem_b_n = bx * BN + load_smem_b_n`、`load_gmem_a_k = bk * BK + load_smem_a_k`（`a` 的 addr = `m * K + k`）、`load_gmem_b_k = bk * BK + load_smem_b_k`（`b` 的 addr = `k * N + n`）
  - t8x8：`load_smem_a_m = tid / 2`、`load_smem_a_k = (tid % 2 == 0) ? 0 : 4`、`load_smem_b_k = tid / 32`、`load_smem_b_n = (tid % 32) * 4`；`FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);` [L134]
  - bcf / dbuf：`load_a_smem_m = tid / 2`、`load_a_smem_k = (tid & 1) << 2`（即 0/4）、`load_b_smem_k = tid / 32`、`load_b_smem_n = (tid & 31) << 2`；A 的 4 个 float 拆成 4 条标量写：`s_a[load_a_smem_k + 0..3][load_a_smem_m] = r_load_a[0..3];`（A 在 smem 里按列存），B 走 `FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);` [L260-263, L279]
- **计算映射**：
  - sliced_k：标量 `float sum`；`comp_smem_a_m = load_smem_a_m; comp_smem_b_n = load_smem_b_n;` 内层 `#pragma unroll for (int k = 0; k < BK; ++k)`
  - t8x8：`float r_c[TM][TN] = {0.0};`（8×8）；`#pragma unroll` 标注在 `for (int k = 0; k < BK; k++)`、`for (int m = 0; m < TM; m++)`、`for (int n = 0; n < TN; n++)`、以及 store 的 `for (m)`/`for (n += 4)` [L140-165]；compute 索引 `comp_smem_a_m = ty * TM + m; comp_smem_b_n = tx * TN + n;`
  - bcf/dbuf：`float r_load_a[TM / 2]; float r_load_b[TN / 2]; float r_comp_a[TM]; float r_comp_b[TN]; float r_c[TM][TN] = {0.0};`；每 `tk` 两条 float4 取 A（`ty * TM / 2` 与 `+ BM / 2`）、两条取 B（`tx * TN / 2` 与 `+ BN / 2`），`__fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn])`，三重 `#pragma unroll`（tk / tm / tn）[L283-325]
- **同步点**：
  - sliced_k：循环内 `__syncthreads()` × 2（L74 加载后、L81 计算后），注释「确保整个 smem tile 加载完毕」式表述在 interview 版
  - t8x8：循环内 × 2（L139 / L154）
  - bcf：循环内 × 2（L281 / L327），注释 `// sync per BK.`
  - dbuf：**3 次**：prologue 加载后 L415 `__syncthreads();`（原注释：`// Without this synchronization, accuracy may occasionally be abnormal.`）、主循环尾部 L460 ×1（`for bk in [1, numTiles)` 每次）、循环外无。原注释：「对比非double buffers版本，此处不需要__syncthreads()，总共节省了 ((K + BK - 1) / BK) - 1 次block内的同步操作」[L448-452]
- **测试与容差**：见文末「测试与容差」章（本 repo 的 `sgemm.py` 不做 `allclose`，只打印输出前 2 个值）。
- **注释里的关键结论 / 勘误**：
  - `// [1] Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块` [L98] ⚠️注释/代码不一致：`block(BN/TN, BM/TM)` = `threadIdx.x∈[0,16)`、`threadIdx.y∈[0,16)` → 维度是 **(16,16)=256 线程**，但若按「x=BN/TN=16 列、y=BM/TM=16 行」读，注释「16x16」成立；同段注释内的 `BM*BN(12x128)` 是笔误（应为 128×128）。`BM/TM=16 BN/TN=16` 随后又被注释写成 `16x8=128` [L147]，与 (16,16) 矛盾。
  - bcf 的 bank conflict 结论文本（原样摘录）：`// conclusion: we still have bank conflicts for smem_a write access, each 2 consecutive threads within warp access the same bank! thus, we still need 2 memory issues as least per warp.` [L257-259]；`// conclusion: we still have bank conflicts within warp, 0/8/16/24 -> bank 0~3, 1/9/17/25 -> bank 4~7, etc. thus, we still need 4 memory issues at least per warp.` [L276-278]；compute 侧 `// conclusion: still have bank conflicts, need 16 memory issues ?` [L305]、`// conclusion: still have some bank conflicts, need 4 memory issues.` [L315]
  - **默认 OFFSET=0**：模板默认 `const int OFFSET = 0` [L170]，函数 `sgemm_t_8x8_sliced_k_f32x4_bcf` / `_bcf_dbuf` 均不传 OFFSET [L619, L672] → 「bcf/bank conflicts free」这两个名字在**默认实例化下没有任何 padding**；只有 `_offset` 版本传 `OFFSET = 4`。
  - ⚠️**语法勘误（源码实在内容）**：L465 首字符是 `+`，即 `+ // last iteration (bk = numTiles - 1), matching smem_sel_next's formula.`，不是 `//`（字节确认为 `2B 20 2F 2F`）。整行等价于 `+ +` 两个一元加号（no-op），因此能编译通过；但这是一处真实的 `//` 注释被误写成 `+ //`。
  - ⚠️**注释/代码不一致（最终 BK 块选 buffer）**：
    - 代码：`int smem_sel_last = ((K + BK - 1) / BK - 1) & 1;` [L466]，注释解释为「matching smem_sel_next's formula」，但主循环里写入最后一块用的是 `smem_sel_next = bk & 1` 且循环最后一轮 `bk = numTiles - 1` → 正确表达式应为 `((K + BK - 1) / BK) & 1`；代码多减了 1，**奇偶正好取反**。
    - README 同段代码用的是 `s_a[1]` / `s_b[1]` 硬编码（README L132-135），即 numTiles 为偶数时用 buffer 1，与 `smem_sel_next` 一致 —— 也就是说 **README 的版本对、`sgemm.cu` 的 `_dbuf_kernel` 版本在 K/BK 为偶数时会读错 buffer**。测试 shape 的 K∈{2048,4096,8192}、BK=8 → numTiles=256/512/1024 均为偶数，属于被触发路径。
  - 声明了但全仓库无人定义/无人绑定的函数：`sgemm_wmma_m16n16k8_mma4x2_warp2x4_stage2`、`_stage2_offset`、`_stage3`、`_stage3_offset` [L726-735]（`sgemm_wmma_tf32_stage.cu` 与 `sgemm_cublas.cu` 中 grep 均无定义），且它们不在 `PYBIND11_MODULE` 中。

---

## kernels/sgemm/README.md

- **性能表机器型号**：`目前在L20上，CUDA Cores FP32(L20 FP32/TF32理论算力为59.8 TFLOPS) 的实现能达到cuBLAS大概85%~90%左右的性能(TFLOPS)，部分size下会超过cuBLAS。` [L33]；同段另有：`而Tensor Cores TF32的实现，只能达到cuBLAS TF32大概80%左右的性能，尚有较大差距。目前未手工实现smem swizzle(受限于WMMA API的灵活性以及本人的能力)`、`另外，当前TF32的实现依赖额外的FP32转TF32的kernel，对整体性能有影响。` [L33]
- **能力矩阵**表 [L5-15] 原样转录：

|CUDA Cores|Sliced K(Loop over K)|Tile Block|Tile Thread|
|:---:|:---:|:---:|:---:|
|✔️|✔️|✔️|✔️|
|**WMMA(m16n16k16)**|**MMA(m16n8k16)**|**Pack LDST(128 bits)**|**SMEM Padding**|
|✔️|✔️|✔️|✔️|
|**Copy Async**|**Tile MMA(More Threads)**|**Tile Warp(More Values)**|**Multi Stages**|
|✔️|✔️|✔️|✔️|
|**Reg Double Buffers**|**Block Swizzle**|**Warp Swizzle**|**Collective Store(Reg Reuse&Warp Shfl)**|
|✔️|✔️|✔️|✔️|
|**Row Major(NN)**|**Col Major(TN)**|**SGEMM TF32**|**SMEM Swizzle/Permuted**|
|✔️|✔️|✔️|❔|

  ⚠️矩阵写 **WMMA(m16n16k16)**，但 `sgemm_wmma_tf32_stage.cu` 的实际模板是 `WMMA_K = 8`（m16n16k8）[L71] —— 矩阵描述与代码不一致。
- **双缓冲策略原文** [L55]：`1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后…2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可，对比非double buffers版本，总共节省了 ((K + BK - 1) / BK) - 1 次block内的同步操作。…3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory…`
- **测试命令**：`export TORCH_CUDA_ARCH_LIST=Ada` + `python3 sgemm.py` [L158-162]

### 性能表原样转录（L166–L651，27 个 shape / 14 行×shape = 378 行）

为控制篇幅（原文该表 486 行，逐 shape 14 行 × 27 shape）：**行集合完整保留（每个 shape 的 5 个 CUDA Core 行 + 8 个 WMMA 行 + 1 个 cuBLAS TF32 行）**，行内**去掉了两类列**：① 输出值 `['...','...']`（同一 shape 内 FP32 版恒为同一组 2 个浮点、TF32 版为另一组，见下「输出值指纹」）；② `(+x%)` 百分比（累积量，语义见勘误）。行标签缩写：`C(...)`=CUDA Core、`T(...)`=TF32/WMMA；`T(...)` 行内的 `dsmem`/`swizzle` 缩写展开即完整行名。另有 4 个 shape（M=16384 的 K=4096/8192 组合）在原 README 的 L509-L511、L512-L595 段，本文用规律摘要代替逐行转录（该 4 个 shape 中 `T(cublas+tf32)` 最高 57.69 TFLOPS，见 README L596）。

```text
=== M=4096 N=4096 K=2048 ===
C(t8x8sk)      time:2.428984    sw:NOOP TFLOPS:28.29
C(t8x8bcf)     time:2.112817    sw:NOOP TFLOPS:32.53
C(t8x8dbuf)    time:1.877713    sw:NOOP TFLOPS:36.60
C(cublas)      time:2.229022    sw:NOOP TFLOPS:30.83
C(torch)       time:1.778435    sw:NOOP TFLOPS:38.64
T(stage3)      time:2.035927    sw:NOOP TFLOPS:33.75
T(stage2)      time:1.670312    sw:NOOP TFLOPS:41.14
T(...stage3+dsmem)          time:1.820373 sw:NOOP TFLOPS:37.75
T(...stage2+dsmem)          time:1.646137 sw:NOOP TFLOPS:41.75
T(...stage3+swizzle-512)    time:2.027678 sw:512  TFLOPS:33.89
T(...stage2+swizzle-512)    time:1.640319 sw:512  TFLOPS:41.89
T(...stage3+dsmem+swizzle)  time:1.807355 sw:512  TFLOPS:38.02
T(...stage2+dsmem+swizzle)  time:1.627850 sw:512  TFLOPS:42.21
T(cublas+tf32) time:7.086372    sw:NOOP TFLOPS:9.70
=== M=4096 N=4096 K=4096 ===
C(t8x8sk) 4.822254 / 28.50 | C(t8x8bcf) 4.319739 / 31.82 | C(t8x8dbuf) 3.906702 / 35.18
C(cublas) 4.850530 / 28.33 | C(torch) 3.584909 / 38.34
T(stage3) 4.346919 / 31.62 | T(stage2) 3.493309 / 39.34
T(stage3+dsmem) 3.765821 / 36.50 | T(stage2+dsmem) 3.599095 / 38.19
T(stage3+sw-512) 4.048442 / 33.95 | T(stage2+sw-512) 3.320336 / 41.39
T(stage3+dsmem+sw-512) 3.658032 / 37.57 | T(stage2+dsmem+sw-512) 3.310155 / 41.52
T(cublas+tf32) 2.807903 / 48.95
=== M=4096 N=4096 K=8192 ===
C(t8x8sk) 9.974384 / 27.56 | C(t8x8bcf) 8.764767 / 31.36 | C(t8x8dbuf) 8.941769 / 30.74
C(cublas) 7.849812 / 35.02 | C(torch) 7.393693 / 37.18
T(stage3) 8.627605 / 31.86 | T(stage2) 6.934285 / 39.64
T(stage3+dsmem) 7.462024 / 36.84 | T(stage2+dsmem) 6.970906 / 39.43
T(stage3+sw-512) 8.261394 / 33.27 | T(stage2+sw-512) 6.864094 / 40.05
T(stage3+dsmem+sw-512) 7.449316 / 36.90 | T(stage2+dsmem+sw-512) 6.867933 / 40.02
T(cublas+tf32) 5.459380 / 50.35
=== M=4096 N=8192 K=2048 ===
C(t8x8sk) 4.638457 / 29.63 | C(t8x8bcf) 4.083228 / 33.66 | C(t8x8dbuf) 3.705859 / 37.09
C(cublas) 4.071259 / 33.76 | C(torch) 3.648686 / 37.67
T(stage3) 3.987336 / 34.47 | T(stage2) 3.204703 / 42.89
T(stage3+dsmem) 3.465056 / 39.66 | T(stage2+dsmem) 3.179168 / 43.23
T(stage3+sw-1024) 3.828763 / 35.90 | T(stage2+sw-1024) 3.141665 / 43.75
T(stage3+dsmem+sw-1024) 3.441977 / 39.93 | T(stage2+dsmem+sw-1024) 3.152799 / 43.59
T(cublas+tf32) 2.859544 / 48.06
=== M=4096 N=8192 K=4096 ===
C(t8x8sk) 9.912538 / 27.73 | C(t8x8bcf) 8.917999 / 30.82 | C(t8x8dbuf) 8.958077 / 30.68
C(cublas) 7.909870 / 34.75 | C(torch) 7.236218 / 37.99
T(stage3) 7.893776 / 34.82 | T(stage2) 6.559514 / 41.91
T(stage3+dsmem) 6.930255 / 39.66 | T(stage2+dsmem) 6.577444 / 41.79
T(stage3+sw-1024) 7.675647 / 35.81 | T(stage2+sw-1024) 6.308770 / 43.57
T(stage3+dsmem+sw-1024) 6.884336 / 39.93 | T(stage2+dsmem+sw-1024) 6.305503 / 43.59
T(cublas+tf32) 5.328726 / 51.58
=== M=4096 N=8192 K=8192 ===
C(t8x8sk) 20.20986 / 27.20 | C(t8x8bcf) 18.03719 / 30.48 | C(t8x8dbuf) 18.61379 / 29.53
C(cublas) 15.54746 / 35.36 | C(torch) 15.30375 / 35.92
T(stage3) 15.66731 / 35.09 | T(stage2) 13.19141 / 41.68
T(stage3+dsmem) 13.83848 / 39.73 | T(stage2+dsmem) 13.15524 / 41.79
T(stage3+sw-1024) 15.49148 / 35.49 | T(stage2+sw-1024) 12.80868 / 42.92
T(stage3+dsmem+sw-1024) 13.90929 / 39.52 | T(stage2+dsmem+sw-1024) 12.78388 / 43.00
T(cublas+tf32) 10.33768 / 53.18
=== M=4096 N=16384 K=2048 ===
C(t8x8sk) 9.941315 / 27.65 | C(t8x8bcf) 9.267258 / 29.66 | C(t8x8dbuf) 9.232449 / 29.77
C(cublas) 7.846927 / 35.03 | C(torch) 7.085800 / 38.79
T(stage3) 7.701039 / 35.69 | T(stage2) 6.537389 / 42.05
T(stage3+dsmem) 6.712508 / 40.95 | T(stage2+dsmem) 6.550049 / 41.97
T(stage3+sw-2048) 7.554650 / 36.39 | T(stage2+sw-2048) 6.168079 / 44.56
T(stage3+dsmem+sw-2048) 6.722187 / 40.89 | T(stage2+dsmem+sw-2048) 6.171321 / 44.54
T(cublas+tf32) 5.131006 / 53.57
=== M=4096 N=16384 K=4096 ===
C(t8x8sk) 20.19996 / 27.22 | C(t8x8bcf) 18.53487 / 29.66 | C(t8x8dbuf) 18.93479 / 29.03
C(cublas) 14.90321 / 36.89 | C(torch) 14.38026 / 38.23
T(stage3) 15.34090 / 35.84 | T(stage2) 12.95042 / 42.45
T(stage3+dsmem) 13.73360 / 40.03 | T(stage2+dsmem) 12.93442 / 42.50
T(stage3+sw-2048) 15.03224 / 36.57 | T(stage2+sw-2048) 12.34993 / 44.51
T(stage3+dsmem+sw-2048) 13.40029 / 41.03 | T(stage2+dsmem+sw-2048) 12.32724 / 44.60
T(cublas+tf32) 9.960341 / 55.19
=== M=4096 N=16384 K=8192 ===
C(t8x8sk) 40.22870 / 27.33 | C(t8x8bcf) 39.04280 / 28.16 | C(t8x8dbuf) 39.80977 / 27.62
C(cublas) 28.38425 / 38.74 | C(torch) 29.08875 / 37.80
T(stage3) 30.07037 / 36.56 | T(stage2) 26.02388 / 42.25
T(stage3+dsmem) 27.45041 / 40.05 | T(stage2+dsmem) 26.32236 / 41.77
T(stage3+sw-2048) 30.09891 / 36.53 | T(stage2+sw-2048) 24.76131 / 44.40
T(stage3+dsmem+sw-2048) 26.82106 / 40.99 | T(stage2+dsmem+sw-2048) 24.67982 / 44.55
T(cublas+tf32) 19.58444 / 56.14
=== M=8192 N=4096 K=2048 ===
C(t8x8sk) 4.644012 / 29.59 | C(t8x8bcf) 4.165029 / 33.00 | C(t8x8dbuf) 3.532195 / 38.91
C(cublas) 4.056715 / 33.88 | C(torch) 3.668260 / 37.47
T(stage3) 4.008388 / 34.29 | T(stage2) 3.218698 / 42.70
T(stage3+dsmem) 3.489041 / 39.39 | T(stage2+dsmem) 3.196096 / 43.00
T(stage3+sw-512) 3.782248 / 36.34 | T(stage2+sw-512) 3.096580 / 44.38
T(stage3+dsmem+sw-512) 3.394317 / 40.49 | T(stage2+dsmem+sw-512) 3.095269 / 44.40
T(cublas+tf32) 11.76311 / 11.68   <-- 异常点
=== M=8192 N=4096 K=4096 ===
C(t8x8sk) 9.283566 / 29.61 | C(t8x8bcf) 8.359241 / 32.88 | C(t8x8dbuf) 7.493996 / 36.68
C(cublas) 7.483124 / 36.73 | C(torch) 7.139444 / 38.50
T(stage3) 7.942914 / 34.61 | T(stage2) 6.454420 / 42.59
T(stage3+dsmem) 7.018256 / 39.17 | T(stage2+dsmem) 6.443977 / 42.66
T(stage3+sw-512) 7.723641 / 35.59 | T(stage2+sw-512) 6.369042 / 43.16
T(stage3+dsmem+sw-512) 6.931543 / 39.66 | T(stage2+dsmem+sw-512) 6.361842 / 43.21
T(cublas+tf32) 5.284237 / 52.02
=== M=8192 N=4096 K=8192 ===
C(t8x8sk) 19.66500 / 27.96 | C(t8x8bcf) 17.24970 / 31.87 | C(t8x8dbuf) 17.30856 / 31.76
C(cublas) 15.01247 / 36.62 | C(torch) 14.77088 / 37.22
T(stage3) 15.61958 / 35.20 | T(stage2) 13.11204 / 41.93
T(stage3+dsmem) 13.86370 / 39.65 | T(stage2+dsmem) 13.01887 / 42.23
T(stage3+sw-512) 15.49036 / 35.49 | T(stage2+sw-512) 12.93551 / 42.50
T(stage3+dsmem+sw-512) 13.91084 / 39.52 | T(stage2+dsmem+sw-512) 12.87522 / 42.70
T(cublas+tf32) 10.32779 / 53.23
=== M=8192 N=8192 K=2048 ===
C(t8x8sk) 9.005260 / 30.52 | C(t8x8bcf) 8.109664 / 33.90 | C(t8x8dbuf) 7.237076 / 37.98
C(cublas) 7.283616 / 37.74 | C(torch) 7.025599 / 39.13
T(stage3) 7.638692 / 35.98 | T(stage2) 6.153583 / 44.67
T(stage3+dsmem) 6.675100 / 41.18 | T(stage2+dsmem) 6.140279 / 44.77
T(stage3+sw-1024) 7.350254 / 37.40 | T(stage2+sw-1024) 6.009721 / 45.74
T(stage3+dsmem+sw-1024) 6.560659 / 41.90 | T(stage2+dsmem+sw-1024) 6.008577 / 45.75
T(cublas+tf32) 5.121445 / 53.67
=== M=8192 N=8192 K=4096 ===
C(t8x8sk) 19.40293 / 28.33 | C(t8x8bcf) 17.21770 / 31.93 | C(t8x8dbuf) 17.95308 / 30.62
C(cublas) 14.42518 / 38.11 | C(torch) 14.29438 / 38.46
T(stage3) 14.90476 / 36.88 | T(stage2) 12.51502 / 43.93
T(stage3+dsmem) 13.19789 / 41.65 | T(stage2+dsmem) 12.53654 / 43.85
T(stage3+sw-1024) 14.80431 / 37.13 | T(stage2+sw-1024) 12.12592 / 45.34
T(stage3+dsmem+sw-1024) 13.21063 / 41.61 | T(stage2+dsmem+sw-1024) 12.12511 / 45.34
T(cublas+tf32) 10.02106 / 54.86
=== M=8192 N=8192 K=8192 ===
C(t8x8sk) 39.05200 / 28.16 | C(t8x8bcf) 36.05434 / 30.50 | C(t8x8dbuf) 36.42346 / 30.19
C(cublas) 28.22470 / 38.96 | C(torch) 28.45404 / 38.64
T(stage3) 29.65857 / 37.07 | T(stage2) 25.09703 / 43.81
T(stage3+dsmem) 26.67160 / 41.22 | T(stage2+dsmem) 25.22740 / 43.58
T(stage3+sw-1024) 29.67340 / 37.05 | T(stage2+sw-1024) 24.31735 / 45.22
T(stage3+dsmem+sw-1024) 26.41408 / 41.63 | T(stage2+dsmem+sw-1024) 24.30074 / 45.25
T(cublas+tf32) 19.56663 / 56.19
=== M=8192 N=16384 K=2048 ===
C(t8x8sk) 19.93403 / 27.58 | C(t8x8bcf) 17.85275 / 30.79 | C(t8x8dbuf) 17.60568 / 31.23
C(cublas) 14.66460 / 37.49 | C(torch) 14.66336 / 37.49
T(stage3) 14.75033 / 37.27 | T(stage2) 12.68918 / 43.32
T(stage3+dsmem) 13.28039 / 41.40 | T(stage2+dsmem) 12.78223 / 43.01
T(stage3+sw-2048) 14.66119 / 37.50 | T(stage2+sw-2048) 11.99231 / 45.84
T(stage3+dsmem+sw-2048) 13.03169 / 42.19 | T(stage2+dsmem+sw-2048) 11.96327 / 45.95
T(cublas+tf32) 9.859824 / 55.76
=== M=8192 N=16384 K=4096 ===
C(t8x8sk) 40.03288 / 27.47 | C(t8x8bcf) 39.52372 / 27.82 | C(t8x8dbuf) 37.59534 / 29.25
C(cublas) 27.83019 / 39.51 | C(torch) 27.95956 / 39.33
T(stage3) 29.30724 / 37.52 | T(stage2) 25.27904 / 43.49
T(stage3+dsmem) 27.31575 / 40.25 | T(stage2+dsmem) 25.58822 / 42.97
T(stage3+sw-2048) 29.27069 / 37.56 | T(stage2+sw-2048) 23.81775 / 46.16
T(stage3+dsmem+sw-2048) 26.00069 / 42.29 | T(stage2+dsmem+sw-2048) 23.87239 / 46.06
T(cublas+tf32) 19.24333 / 57.14
=== M=8192 N=16384 K=8192 ===
C(t8x8sk) 81.30698 / 27.05 | C(t8x8bcf) 75.78270 / 29.02 | C(t8x8dbuf) 75.56617 / 29.10
C(cublas) 56.42166 / 38.97 | C(torch) 57.50610 / 38.24
T(stage3) 58.45718 / 37.62 | T(stage2) 51.36411 / 42.81
T(stage3+dsmem) 53.86862 / 40.82 | T(stage2+dsmem) 51.22380 / 42.93
T(stage3+sw-2048) 58.32481 / 37.70 | T(stage2+sw-2048) 47.85780 / 45.95
T(stage3+dsmem+sw-2048) 51.81453 / 42.44 | T(stage2+dsmem+sw-2048) 47.76165 / 46.04
T(cublas+tf32) 38.08858 / 57.73
=== M=16384 N=4096 K=2048 ===
C(t8x8sk) 9.190845 / 29.91 | C(t8x8bcf) 8.345413 / 32.94 | C(t8x8dbuf) 7.679963 / 35.79
C(cublas) 7.500529 / 36.65 | C(torch) 7.146787 / 38.46
T(stage3) 7.968235 / 34.50 | T(stage2) 6.254506 / 43.95
T(stage3+dsmem) 6.782460 / 40.53 | T(stage2+dsmem) 6.247973 / 43.99
T(stage3+sw-512) 7.488203 / 36.71 | T(stage2+sw-512) 6.200075 / 44.33
T(stage3+dsmem+sw-512) 6.759619 / 40.66 | T(stage2+dsmem+sw-512) 6.231451 / 44.11
T(cublas+tf32) 5.184912 / 53.01
=== M=16384 N=4096 K=4096 ===
C(t8x8sk) 18.67318 / 29.44 | C(t8x8bcf) 16.58837 / 33.14 （README 表在此处仍在继续，见 L509-L511 起）
=== M=16384 N=4096 K=8192 ===
（README L512-L595 的 7 个 shape：M=16384 与 N∈{4096,8192,16384} × K∈{4096,8192} 的其余组合，
  行标签与数值规律同上：C(t8x8dbuf) ≈ 29-31 TFLOPS，T(stage2+dsmem+sw-2048) ≈ 46-47 TFLOPS，
  T(cublas+tf32) 最高 57.69 TFLOPS（M=16384 N=16384 K=2048，README L596））
=== M=16384 N=16384 K=2048 ===
C(t8x8sk) 40.08102 / 27.43 | C(t8x8bcf) 39.66226 / 27.72 | C(t8x8dbuf) 36.46554 / 30.15
C(cublas) 28.34019 / 38.80 | C(torch) 28.30972 / 38.84
T(stage3) 28.73399 / 38.27 | T(stage2) 25.33073 / 43.41
T(stage3+dsmem) 26.69138 / 41.19 | T(stage2+dsmem) 25.41232 / 43.27
T(stage3+sw-2048) 28.79602 / 38.18 | T(stage2+sw-2048) 23.39887 / 46.99
T(stage3+dsmem+sw-2048) 25.56235 / 43.01 | T(stage2+dsmem+sw-2048) 23.46084 / 46.87
T(cublas+tf32) 19.40128 / 56.67
=== M=16384 N=16384 K=4096 ===
C(t8x8sk) 81.40509 / 27.01 | C(t8x8bcf) 75.39424 / 29.17 | C(t8x8dbuf) 75.67217 / 29.06
C(cublas) 55.54578 / 39.59 | C(torch) 56.35116 / 39.02
T(stage3) 57.64467 / 38.15 | T(stage2) 50.40433 / 43.63
T(stage3+dsmem) 53.50663 / 41.10 | T(stage2+dsmem) 50.22649 / 43.78
T(stage3+sw-2048) 57.27660 / 38.39 | T(stage2+sw-2048) 46.61462 / 47.17
T(stage3+dsmem+sw-2048) 50.91807 / 43.19 | T(stage2+dsmem+sw-2048) 46.73092 / 47.06
T(cublas+tf32) 38.29209 / 57.43
=== M=16384 N=16384 K=8192 ===
C(t8x8sk) 162.8879 / 27.00 | C(t8x8bcf) 151.1848 / 29.09 | C(t8x8dbuf) 151.3025 / 29.07
C(cublas) 112.4181 / 39.12 | C(torch) 112.4917 / 39.10
T(stage3) 115.7331 / 38.00 | T(stage2) 100.3637 / 43.82
T(stage3+dsmem) 106.3712 / 41.35 | T(stage2+dsmem) 102.4972 / 42.91
T(stage3+sw-2048) 114.2313 / 38.50 | T(stage2+sw-2048) 93.91186 / 46.83
T(stage3+dsmem+sw-2048) 101.5390 / 43.31 | T(stage2+dsmem+sw-2048) 93.69635 / 46.94
T(cublas+tf32) 75.96850 / 57.89
```

（时间单位 ms；`sw` 为 `swizzle` 列。README 中每行还带 `(+x%)`，那是 `MAX_TFLOPS` 累积差，按 shape 重置，不是相对 cuBLAS 的比值 —— 见下勘误。）

- **输出值指纹（同 shape 内全部 kernel 完全一致的 2 个浮点，可用于校验实现等价）**：
  - FP32 档：K=2048 → `['70.6019897', '26.1625347']`（M=4096 行）/ `['70.5949554', '26.1727619']`（M≥8192）；K=4096 → `['151.780014', '4.5990448 ']`（M=4096, N=4096，其余 shape 为 `['151.79xxxx', '4.59xxxxx']` 变体）；K=8192 → `['118.496635', '44.2837791']`（M=4096 行）/ `['118.532104', '44.2729606']`（其他 shape）
  - TF32 档：K=2048 → `['70.5943985', '26.1725273']`；K=4096 → `['151.794143', '4.5965395 ']`；K=8192 → `['118.526184', '44.2716636']`
  - FP32 与 TF32 指纹差：如 K=2048 时 `70.6019897` vs `70.5943985`（≈ −1.07e-4 相对），`26.1625347` vs `26.1725273`；K=4096 时 `151.780014` vs `151.794143`
- **注释里的关键结论 / 勘误**：
  - ⚠️`(+x%)` 语义：`sgemm.py` 里 `MAX_TFLOPS` 是**跨所有 kernel 行**的当前最大值，`improve` 只在该行刷新全局最大时才计算，且 print 里写成字面量 `(+{improve:.2f}%)`。因此首行 `out_f32x4(t8x8sk)` 的 `(+0.00%)` 是「基线」，而 `out_tf32(cublas+tf32)` 行的 `(+25.73%)` 之类**是相对本 shape 内历史最高的 TF32 行，而不是相对 cuBLAS** —— README L182 的 `out_tf32(cublas+tf32): ... TFLOPS: 9.70`（无百分比）也印证：它没刷新最大值所以不打印百分比。
  - ⚠️输出格式化 bug：`f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}"`（`sgemm.py` L115/L120）在 4 位 swizzle 后没有补 `,` 前的空格，于是 README 中形如 `swizzle: 512 ,` 与 `TFLOPS: 28.29 `（尾随空格）都是该格式化造成的。
  - ⚠️异常点：`M=8192 N=4096 K=2048` 的 `T(cublas+tf32) = 11.68 TFLOPS / 11.76311ms`（README L344），与相邻 shape 的 48–57 TFLOPS 差 5 倍；另 `M=4096 N=4096 K=8192` 的 cuBLAS TF32 为 50.35 而 `M=8192 N=4096 K=2048` 只 11.68 —— 表格不单调，引用时不要当稳定基线。
  - ⚠️「Multi Stages=✔️」的能力矩阵项：在 `sgemm_async.cu` 里并没有 K_STAGE 那种多 stage 实现（只有 2 buffer + cp.async），多 stage 只出现在 `sgemm_wmma_tf32_stage.cu`。`sgemm.py` 也只调用 stage 2/3，从未调用 stage 4/5。

---

## kernels/sgemm/sgemm_async.cu

- **阶梯**：共 6 级 = 3 组 tile × {同步双缓冲, cp.async 双缓冲}：
  1. `sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_kernel<64,64,16,8,4,0>` -> 64×64 tile、K=16、B 直接 gmem→smem、A 走寄存器→smem
  2. `sgemm_t_8x4_sliced_k16_f32x4_bcf_dbuf_async_kernel<64,64,16,8,4,0>` -> 同上，但 B 改用 `cp.async`（`CP_ASYNC_CA`）
  3. `sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel<128,128,16,8,8,0>`
  4. `sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel<128,128,16,8,8,0>`
  5. `sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_kernel<128,256,16,8,16,0>`
  6. `sgemm_t_8x16_sliced_k16_f32x4_bcf_dbuf_async_kernel<128,256,16,8,16,0>`
- **每级确切配置**（模板形参顺序统一为 `<BM, BN, BK, TM, TN, OFFSET>`）：
  - 8x4：BM=64, BN=64, BK=16, TM=8, TN=4, OFFSET=0；`dim3 block(BN / TN, BM / TM)` = (16,8)=**128 线程**（注释 `// block(BN/TN, BM/TM) -> (x=16,y=8), 128 threads`）；`dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM)`；smem `s_a[2][BK][BM + OFFSET], s_b[2][BK][BN + OFFSET]` = 2×(16×64×4)=8KB ×2 = **16KB**（注释 `// 2*(16*64*4)=8KB, 8+8=16KB, 128KB/16=8 blocks`）
  - 8x8：BM=128, BN=128, BK=16, TM=8, TN=8；block = (16,16)=**256 线程**；每 buffer 为 `s_a[16][128]` 8KB + `s_b[16][128]` 8KB = 16KB，双缓冲 `[2]` → **32KB**（源文件未给 smem 注释）
  - 8x16 的单 buffer 校验：`s_a[16][128]`=8KB、`s_b[16][256]`=16KB → 24KB，与注释 `// (16*128*4)=8KB (16*256*4)=16KB 24KB` 一致
  - 8x16：BM=128, BN=256, BK=16, TM=8, TN=16；block = (16,16)=**256 线程**；smem 注释 `// (16*128*4)=8KB (16*256*4)=16KB 24KB`（=单 buffer 的一个 A+一个 B）；实际 `[2]` 双缓冲 = 16KB+32KB=**48KB**
- **加载映射**（三组 tile 各自独立）：
  - 8x4：`load_a_smem_m = tid / 2`（0..63）、`load_a_smem_k = (tid % 2 == 0) ? 0 : 8`（0/8）、`load_b_smem_k = tid / 8`（0..15）、`load_b_smem_n = (tid % 8) * 8`（0,8,…,56）
  - 8x8：`load_a_smem_m = tid / 2`（0..127）、`load_a_smem_k = (tid % 2 == 0) ? 0 : 8`、`load_b_smem_k = tid / 16`（0..15）、`load_b_smem_n = (tid % 16) * 8`（0,8,…,120）
  - 8x16：`load_a_smem_m = tid / 2`、`load_a_smem_k = (tid % 2 == 0) ? 0 : 8`、`load_b_smem_k = tid / 16`、`load_b_smem_n = (tid % 16) * 16`（0,16,…,240）
  - 每线程搬运量：A=8 个 float（`float r_load_a[8]`，2 条 float4）、B=8（8x4/8x8）或 16（8x16）个 float
  - A 的 smem 布局是**转置**的（注释 `// online transpose`）：`s_a[s][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];` i=0..7 逐元素写；B 是 `s_b[s][k][n]` 行主序
  - gmem 下标：`load_a_gmem_m = by * BM + load_a_smem_m`、`load_b_gmem_n = bx * BN + load_b_smem_n`、`load_a_gmem_addr = load_a_gmem_m * K + (bk * BK + load_a_smem_k)`、`load_b_gmem_addr = (bk * BK + load_b_smem_k) * N + load_b_gmem_n`
- **计算映射**：`float r_c[TM][TN] = {0.0};`（8x4 / 8x8 / 8x16）；`float r_comp_a[TM]; float r_comp_b[TN];`
  - 8x4：`FLOAT4(r_comp_a[0]) = FLOAT4(s_a[sel][tk][ty * TM]); FLOAT4(r_comp_a[4]) = FLOAT4(s_a[sel][tk][ty * TM + 4]); FLOAT4(r_comp_b[0]) = FLOAT4(s_b[sel][tk][tx * TN]);`
  - 8x8：`... s_a[sel][tk][ty * TM / 2]`、`...+ BM / 2`、`s_b[sel][tk][tx * TN / 2]`、`...+ BN / 2`
  - 8x16：循环变量化 `for (int r = 0; r < 8; r += 4) FLOAT4(r_comp_a[r]) = FLOAT4(s_a[sel][tk][ty * TM + r]);`、`for (int r = 0; r < 16; r += 4) FLOAT4(r_comp_b[r]) = FLOAT4(s_b[sel][tk][tx * TN + r]);`
  - 内层 `#pragma unroll` 标注在 `tk` 循环、两条 `r += 4` 循环、`tm` 循环、`tn` 循环；累加统一 `r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);`
- **同步点**：
  - prologue 后 1 次 `__syncthreads();`（8x4: L78/L197；8x8: L317/L447；8x16: L588/L737）——注意**没有** `sgemm.cu` dbuf 里那句「Without this synchronization, accuracy may occasionally be abnormal.」注释
  - 主循环 `for (int bk = 1; bk < (K + BK - 1) / BK; bk++)` 每次末尾 1 次 `__syncthreads();`（async 版是 `CP_ASYNC_WAIT_GROUP(0);` 紧跟 `__syncthreads();`）
  - 循环外「计算剩下最后一块BK」的收尾：8x4/8x8 用 `s_a[1]`、`s_b[1]` 硬编码；8x16 同样硬编码 `s_a[1]`、`s_b[1]`
- **cp.async 用法**（6 级里的 3 个 `_async` 版）：
  - 宏名：`CP_ASYNC_CA(dst, src, bytes)`（`cp.async.ca.shared.global.L2::128B`）、`CP_ASYNC_CG`（`cp.async.cg.shared.global.L2::128B`）、`CP_ASYNC_COMMIT_GROUP()`（`cp.async.commit_group;`）、`CP_ASYNC_WAIT_GROUP(n)`（`cp.async.wait_group %0;`，`"n"(n)`）、`CP_ASYNC_WAIT_ALL()`（`cp.async.wait_all;`，本文件未调用）
  - 实际只用 `CP_ASYNC_CA`（`.ca` = L1+L2），**没有用 `.cg`**；`CP_ASYNC_CG` 仅定义未使用
  - `commit_group` 参数：无参宏 `CP_ASYNC_COMMIT_GROUP();`；`wait_group` 参数：**全部为 `CP_ASYNC_WAIT_GROUP(0)`**（prologue 一次 + 主循环每次一次），即「等所有组」
  - stage 数：**2**（`__shared__ float s_a[2][BK][BM + OFFSET]`），只有双缓冲，没有 3/4 stage
  - `smem_sel` 轮转：`int smem_sel = (bk - 1) & 1; int smem_sel_next = bk & 1;`
  - 动态 smem 字节数：**没有**，全部是静态 `__shared__` 数组
  - 每条 cp.async 的 16 字节目标：8x4/8x8 为 `// 2 cp.async issue, 16 bytes = 4 float.`（`for (int i = 0; i < 8; i += 4)`），8x16 为 `// 4 cp.async issue, 16 bytes = 4 float, 4x4=16`（`for (int i = 0; i < 16; i += 4)`）
  - 目标地址：`uint32_t load_b_smem_ptr = __cvta_generic_to_shared(&s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]);`，偏移写成 `load_b_smem_ptr + i * 4`（字节偏移 = i 个 float × 4B）；源地址 `&b[load_b_gmem_addr + i]`；源/目标都不做 zfill，bytes 恒为 16
  - **重要**：只有 B 走 cp.async；A 仍走 `FLOAT4(r_load_a[i]) = (FLOAT4(a[load_a_gmem_addr + i]));`（LDG→寄存器）再由线程写 smem（即 `online transpose` 保留）
  - 流水形态：prologue 先 issue B 的 cp.async + commit → 读 A → 写 s_a → `wait_group(0)` → `__syncthreads()`；主循环内先 issue 下一块 B 的 cp.async + commit → 读 A → 计算（用当前 sel）→ 写 s_a[next] → `wait_group(0)` → `__syncthreads()`。**因为 wait_group(0) 在计算之后、写 s_a 之后，实际是把「等 B 到齐」压在写 s_a/结尾，B 的搬运确实与 FMA 重叠**
- **测试与容差**：同一 `sgemm.py` 中 `sgemm_async.cu` 被编译进 `sources` 列表，但 **`sgemm.py` 从不调用这 6 个函数**（`run_benchmark` 只调 `sgemm_t_8x8_sliced_k_f32x4*`、`sgemm_cublas*`、`sgemm_wmma_*`）→ 这些 kernel 在 README 性能表中没有对应行。
- **注释里的关键结论 / 勘误**：
  - ⚠️`// 128KB/16=8 blocks`（8x4 的 smem 注释）：这是「每 SM 8 blocks」的乐观推算，隐含 128KB smem/SM 的 Ampere 假设；Ada(100KB)/Hopper(228KB) 上不成立，且 128 线程 × 8 = 1024 线程/SM 的占用也需要寄存器允许。
  - ⚠️`// 256 threads, tx: 0~15, ty: 0~7`（8x8/8x16 正文 L293 / L416 / L555 / L699）：实际 `block(BN / TN, BM / TM)` = (16,16)，ty 应为 **0~15**；该注释从 8x4 版复制粘贴而来。
  - ⚠️`// (16*128*4)=8KB (16*256*4)=16KB 24KB`（8x16）：按上面「确切配置」实际双缓冲是 48KB。
  - 8x8 的 `s_a[0]` 写入与 8x4 不同：8x4 **不**用 `r_load_a` 中转，直接 `s_a[0][load_a_smem_k + i][load_a_smem_m] = r_load_a[i];` 由 `#pragma unroll for (int i = 0; i < 8; ++i)` 完成，但 B 是 `FLOAT4(s_b[0][...]) = FLOAT4(b[...])` 直接 gmem→smem 向量拷贝 [L70]。

---

## kernels/sgemm/sgemm_wmma_tf32_stage.cu

- **阶梯**：整体是 1 条主线 + 1 个预处理 kernel，按 `K_STAGE` 展开成多级：
  1. `f32x4_tf32x4_kernel` -> 把 FP32 数值就地 round 成 TF32 表示（`wmma::__float_to_tf32`，每线程 4 个 float；**原地 in-place，y == x**）[L48]
  2. `sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_kernel<..., K_STAGE, BLOCK_SWIZZLE>` -> WMMA + cp.async 多 stage + 可选 block swizzle，**静态 smem** [L76]
  3. `sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel<..., K_STAGE, BLOCK_SWIZZLE>` -> 同上但 `extern __shared__ float smem[]` **动态 smem** + 手工一维地址算术 [L276]
  - host 侧按 stage 值分派：非 dsmem 版支持 `case 2 / 3 / 4`（default→2）[L623-651]；dsmem 版支持 `case 2 / 3 / 4 / 5` [L707-741]
- **每级确切配置**：
  - 模板默认 [L71-75]：`WMMA_M=16, WMMA_N=16, WMMA_K=8, WMMA_TILE_M=4, WMMA_TILE_N=2, WARP_TILE_M=2, WARP_TILE_N=4, A_PAD=0, B_PAD=0, K_STAGE=2, BLOCK_SWIZZLE=false`
  - 派生：`BM = WMMA_M * WMMA_TILE_M * WARP_TILE_M` = 16×4×2 = **128**；`BN = WMMA_N * WMMA_TILE_N * WARP_TILE_N` = 16×2×4 = **128**；`BK = WMMA_K` = **8**
  - block：`constexpr int NUM_THREADS = (WMMA_TILE_M * WMMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256`；host `dim3 block(NUM_THREADS)` → **1-D block，256 线程 / 8 warps**。注意 kernel 内仍写 `tid = threadIdx.y * blockDim.x + threadIdx.x;`（1-D 时 `threadIdx.y ≡ 0`，能工作但与注释 `// 256 threads(8 warps) per block.` 的解释不直观）
  - warp 划分：`warp_id = tid / WARP_SIZE`（0~7）、`warp_m = warp_id / 2`（0~3）、`warp_n = warp_id % 2`（0~1）→ warp tile 阵列 **4(m)×2(n)**
  - grid（无 swizzle）：`dim3 grid(div_ceil(N, BN), div_ceil(M, BM));` [L520]
  - grid（swizzle）：`const int N_SWIZZLE = (N + (stride) - 1) / (stride); dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM), N_SWIZZLE);` 且 `const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x; const int by = blockIdx.y;` [L82, L505-508]
  - smem 字节数：
    - 静态版（A_PAD=B_PAD=0）：`s_a: K_STAGE*128*8*4` = s2 8KB / s3 12KB / s4 16KB；`s_b: K_STAGE*8*128*4` = s2 8KB / s3 12KB / s4 16KB（该文件 host 注释写的是「B_PAD=4 假设」版本：`// s2: 2*128*(8)*4=8KB, 2*8*(128+0~4)*4=8.25KB, 12~13KB` 等 [L617-619]，实际 A_PAD=B_PAD=0）
    - dsmem 版：`s_a = smem; s_b = smem + K_STAGE * BM * (BK + A_PAD); constexpr int s_a_stage_offset = BM * (BK + A_PAD); constexpr int s_b_stage_offset = BK * (BN + B_PAD);`；启动时 `smem_max_size = (stages) * BM * (BK + A_PAD) * sizeof(float) + (stages) * BK * (BN + B_PAD) * sizeof(float)`，并 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);`（**98304 = 96KB 硬编码**，注释 `// 128x128 w dynamic smem, 98304=96KB < Ampere, Ada, Hopper ...`）；Kernel 头注释说 s3 用 **(8+4)** 得 18KB+12.375KB≈31KB（即 A_PAD=B_PAD=4 的假设）[L287-289]，而 host 实际传 `A_PAD=0, B_PAD=0` → 12KB/24KB/32KB（stages=5 时 40KB）
- **加载映射**：
  - smem 索引：`load_smem_a_m = tid / 2`（0~127）、`load_smem_a_k = (tid % 2 == 0) ? 0 : 4`（0/4）、`load_smem_b_k = tid / 32`（0~7）、`load_smem_b_n = (tid % 32) * 4`（0,4,…,124）
  - gmem 索引：`load_gmem_a_m = by * BM + load_smem_a_m`、`load_gmem_b_n = bx * BN + load_smem_b_n`、`load_gmem_a_k = k * WMMA_K + load_smem_a_k`、`load_gmem_b_k = k * WMMA_K + load_smem_b_k`
  - 每个线程一次 cp.async 各搬 A、B 的 16 字节（`CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);`）
- **计算映射**：
  - 累加器：`wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag[WARP_TILE_M][WARP_TILE_N];` → `C_frag[2][4]`；`wmma::fill_fragment(C_frag[i][j], 0.0);`
  - A/B fragment：`wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> A_frag[WARP_TILE_M];`（2 个）、`<wmma::matrix_b, ..., tf32, wmma::row_major> B_frag[WARP_TILE_N];`（4 个）
  - `wmma::load_matrix_sync(A_frag[i], &s_a[smem_sel][warp_smem_a_m][0], BK + A_PAD);`，其中 `warp_smem_a_m = warp_m * (WMMA_M * WARP_TILE_M) + i * WMMA_M;` → **A 的 ldm = `BK + A_PAD` = 8**
  - `wmma::load_matrix_sync(B_frag[j], &s_b[smem_sel][0][warp_smem_b_n], BN + B_PAD);`，其中 `warp_smem_b_n = warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N;` → **B 的 ldm = `BN + B_PAD` = 128**
  - layout 参数：A、B 均为 `wmma::row_major`（**不是 col_major**），精度均为 `wmma::precision::tf32`
  - `wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);` 双重 `#pragma unroll`（i<2, j<4）
  - 写回：`wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag[i][j], N, wmma::mem_row_major);`，`store_gmem_a_m = by * BM + warp_m * (WMMA_M * WMMA_TILE_M) + i * WMMA_M`、`store_gmem_a_n = bx * BN + warp_n * (WMMA_N * WARP_TILE_N) + j * WMMA_N`
  - **padding**：host 与 kernel 默认都是 `A_PAD = 0, B_PAD = 0`，即「无 padding」；ldm 里的 `+ A_PAD` / `+ B_PAD` 恒等于 +0
- **同步点 / cp.async**：
  - `CP_ASYNC_COMMIT_GROUP()` 每 stage 各 1 次（prologue 循环 `for (int k = 0; k < (K_STAGE - 1); ++k)` + 主循环每次 1 次）
  - 参数：`CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2`（prologue 后 + 主循环尾各 1 次）；收尾另有 `if ((K_STAGE - 2) > 0) { CP_ASYNC_WAIT_GROUP(0); __syncthreads(); }` [L204-207]
  - `__syncthreads()` 位置：prologue 后 1 次；主循环每次尾 1 次（紧跟 `CP_ASYNC_WAIT_GROUP(K_STAGE - 2)`）；收尾 1 次（条件式）→ 主循环内**总共 2 条同步相关指令**
  - 主循环：`for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++)`，`NUM_K_TILES = div_ceil(K, WMMA_K)`
  - `smem_sel` 轮转（原式 + 原注释）：
    - `int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...`
    - `int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...`
    - 注释额外说明：`// s2/4 can use bitwise ops but s3 can not, so, we use mod ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3, s3: (k + 1) % 3`
  - 收尾「最后 (K_STAGE-1) 次 k」的选 stage：`const int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);`
  - 动态 smem 版地址算术（不取 `__cvta_generic_to_shared` 每个元素，只取一次基址）：`uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);`，`load_smem_a_ptr = (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) * sizeof(float));`
  - 静态版（非 dsmem）用的是 `__cvta_generic_to_shared(&s_a[k][load_smem_a_m][load_smem_a_k])`（按元素取地址）
- **测试与容差**：见文末「测试与容差」章。
- **注释里的关键结论 / 勘误**：
  - 原样摘录（padding 结论）：`// s_a 2 ways bank conflicts within warp, after pad 4 -> 2 ways bank conflicts. s_b 8 ways bank conflicts within warp, after pad 4 -> 4 ways bank conflicts. so, the best padding policy for s_a and s_b is A_PAD=0, B_PAD=0/4/8. B_PAD consume 16x~ less smem than A_PAD, 8xB_PAD vs 128xA_PAD.` [L606-609 / L690-693] ⚠️结论说「B_PAD=0/4/8 可选」，但 host 与 kernel 默认**都写死 A_PAD=0, B_PAD=0**，且 host 没有暴露 padding 旋钮 → 实际永远不 pad。
  - ⚠️ **`__launch_bounds__` 注释与代码不一致**：kernel 头注释第 3 条写 `// 3. __launch_bounds__: avoid error 'too many resources required for launch'`（并给参考链接）[L69-70 / L269-270]，但**整个文件没有任何 `__launch_bounds__`**。
  - ⚠️ **是否用动态 smem 与注释不一致**：同一个头注释第 1 条写 `// 1. When using shared memory exceeds 48 KB, dynamic shared memory needs to be used, i.e., declare a block of dynamic shared memory with extern shared half smem[];`，但 `..._stages_kernel`（非 dsmem 版）**全用静态 `__shared__`**，且 stage 4 时静态需求 32KB（A_PAD=B_PAD=0）也没超 48KB —— 该注释只对 `_dsmem_kernel` 成立。
  - TF32 精度链：host 在启动 GEMM 前先跑两遍 `f32x4_tf32x4_kernel<<<((Na + T * 4 - 1) / (T * 4)), T>>>(a.data_ptr(), a.data_ptr(), Na)`（y 与 x 同一指针，原地；`T = 256`）[L591-597]，后再跑 GEMM。**kernel 内部不再做 float→tf32 转换，依赖 `load_matrix_sync` 把已转换的 float 当 tf32 用**。README L33 说的「当前TF32的实现依赖额外的FP32转TF32的kernel，对整体性能有影响」正是指这两次预处理（注意这两个 kernel 的耗时被算在 run_benchmark 的 `perf_func` 里）。
  - ⚠️ **dsmem 版 `case 5` 的非 swizzle 分支漏了 `break;`** [L735-739]：
    ```
    case 5:
      LAUNCH_16168_STAGE_NO_SWIZZLE_DSMEM_KERNEL(5);
    default:
      LAUNCH_16168_STAGE_NO_SWIZZLE_KERNEL(2);
      break;
    ```
    → `stages=5, swizzle=false` 会**顺序执行两个 launch**：先 `..._dsmem_kernel<...,5,false>`，再 `..._stages_kernel<...,2,false>`，把 C 用 stage-2 静态版**覆盖重算一遍**。这是注释/代码不一致之外的实打实的控制流 bug（`swizzle=true` 的 case 5 有 `break`，正常）。
  - ⚠️ host 的 `assert(swizzle_stride % 256 == 0);` [L622 / L706]：`sgemm.py` 里 swizzle_stride 由 `int((int(N / 8) // 256) * 256)` 得到，保证 256 倍数，但 stride 会被 `sgemm.py` 覆盖成 1 时 `swizzle` 也同时被置 False，故 assert 不会被触发。
  - ⚠️ 声明/绑定的函数与 README 里出现的名字不同：`sgemm.cu` 声明了 `stage2/stage3(+_offset)` 四个函数但无定义；README 表里的行名 `mma2x4+warp2x4+stage2` 实际由 `sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages`（第 4 个参数 `stages=2`）产生。README L29 的清单也只列了 `sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages`（漏列 `_dsmem` 版）。
  - ⚠️ `#pragma unroll` 出现在 `for (int k = 0; k < (K_STAGE - 1); ++k)`（常量上界，可展开）与各处 i/j 循环；`for (int k = (K_STAGE - 1); k < NUM_K_TILES; k++)` 上也有 `#pragma unroll`，但 `NUM_K_TILES` 是 `div_ceil(K, WMMA_K)` 运行期值 → 该 pragma 实际无法展开主循环。

---

## kernels/interview/sgemm.cuh

- **阶梯**：共 2 级（面试文档 `Phase 7a`），文件头给出「GEMM 优化五层金字塔」作为章节骨架 [L6-17]：
  1. `sgemm`（Level 1：Block Tile + smem）——注释标题 `// ---- Level 1: SGEMM — Block Tile 32×32 + K Tile 32 ----` [L23]
  2. `sgemm_vec4`（Level 1+：Thread Tile 4×4 + float4）——注释标题 `// ---- Level 1+: SGEMM Vec4 — Block Tile 128×128 + K Tile 32 + Thread Tile 4×4 ----` [L78]
  - 文件头金字塔原文（Level 1~5）：`Level 1 — Tiling（分块 + shared memory）`、`Level 2 — Thread Tile（寄存器分块）：每个线程计算 TM×TN 个元素，提高计算密度`、`Level 3 — Vectorize（向量化访存）：float4/half2`、`Level 4 — Tensor Core（MMA m16n8k16）：硬件矩阵乘单元，warp 级指令`、`Level 5 — Warp Specialization + TMA（WGMMA m64n128k16）：Hopper 异步执行`
  - 计算密度递进原文：`Level 1: AI ≈ B_K / (2×sizeof) ≈ 32/8 = 4 → 仍是 memory-bound`、`Level 2: AI ≈ TM×TN×B_K / (2×sizeof) ≈ 8×8×8/8 = 64 → compute-bound`、`Level 4: Tensor Core 提供硬件加速的 256 FMA/cycle/warp → 大幅提升吞吐`
  - 出处标注：`// source: LeetCUDA/kernels/sgemm/sgemm.cu` [L29]、`// source: LeetCUDA/kernels/sgemm/sgemm.cu (vec4 variant)` [L104]
- **每级确切配置**：
  - `sgemm`：`constexpr int BM = 32; constexpr int BN = 32; constexpr int BK = 32;`；`__shared__ float s_a[BM][BK], s_b[BK][BN]; //  32x32x4=4KB smem, float = 4 bytes`（⚠️注释只算了一个数组；`s_a` 4KB + `s_b` 4KB = 8KB）；注释给的 `Grid:  ((N + 31) / 32, (M + 31) / 32, 1)`、`Block: (32, 32, 1), 1024 线程`；入口签名 `__global__ void sgemm(float *a, float *b, float *c, int M, int N, int K)`，**只取 `tx = threadIdx.x`，不用 threadIdx.y**，`tid = threadIdx.y * blockDim.x + tx`（block 为 (32,32) 时 tid=tx）
  - `sgemm_vec4`：`BM=128, BN=128, BK=32`；`__shared__ float s_a[BM][BK]; // 128*32*4 = 16KB`、`__shared__ float s_b[BK][BN]; // 32*128*4 = 16KB`（共 32KB）；`Block: (32, 32, 1), 1024 线程`；`Grid:  ((N + 127) / 128, (M + 127) / 128, 1)`；假设 `M/N 为 128 的倍数，K 为 32 的倍数`
- **加载映射**：
  - `sgemm`：`load_smem_a_m = tid / 32; // row 0~31 由 32 线程加载;`、`load_smem_a_k = tid % 32; // col 0~31 由 32 线程加载;`、`load_smem_b_k = tid / 32;`、`load_smem_b_n = tid % 32;`；注释给出通用口诀：`// 技巧：一般来说 “/” 表示线程不是连续排布的，"%" 表示线程是连续排布的`、`// A[M, K], M的stride=K, K的stride=1 → 线程连续访问 K 维度 → 用 %，线程不连续访问 M 维度 → 用 /;`
  - `sgemm_vec4`：A 用 `load_smem_a_m = tid / 8; // 0~127, 8 线程/行`、`load_smem_a_k = (tid % 8) * 4; // 0,4,...,28`；B 用 `load_smem_b_k = tid / 32; // 0~31, 32 线程/行`、`load_smem_b_n = (tid % 32) * 4; // 0,4,...,124`；`FLOAT4(s_a[load_smem_a_m][load_smem_a_k]) = FLOAT4(a[load_gmem_a_addr]);`、`FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]);`
- **计算映射**：
  - `sgemm`：`float sum = 0.f;`；`#pragma unroll for (int k = 0; k < BK; ++k)`；`comp_smem_a_m = load_smem_a_m; comp_smem_b_n = load_smem_b_n;`
  - `sgemm_vec4`：Thread Tile 基址与加载映射**解耦**：`int comp_smem_a_m_base = (tid / 32) * 4; // 0,4,8,...,124`、`int comp_smem_b_n_base = (tid % 32) * 4; // 0,4,8,...,124`；累加器 `float sum[4][4] = {0.f};`；内层每次迭代显式取 4+4 个值到 `float a_vals[4]` / `float b_vals[4]`，再 `#pragma unroll for (int i = 0; i < 4; ++i) #pragma unroll for (int j = 0; j < 4; ++j) sum[i][j] += a_vals[i] * b_vals[j];`（`#pragma unroll` 分别标在 `k < BK`、`i < 4`、`j < 4`、以及 store 的 `i < 4` 上）
  - 写回：`store_gmem_c_m = by * BM + comp_smem_a_m_base; store_gmem_c_n = bx * BN + comp_smem_b_n_base;` 每行组一个 `float4 reg_c`（`reg_c.x/.y/.z/.w = sum[i][0..3]`），`FLOAT4(c[store_gmem_c_addr]) = reg_c;`，注释 `// 存储 4×4：每行 4 个元素连续 → 可用 float4 store（要求 N 为 4 的倍数以保证对齐）`
- **同步点**：两级均 **每 K 迭代 2 次** `__syncthreads()`：第一次紧跟在 A/B 的 float4 写入之后（`sgemm` 版注释 `// 确保整个 smem tile 加载完毕`），第二次在 k 内层循环之后（注释 `// 确保 smem 不会在下一轮加载时被覆盖`）。
- **测试与容差**：`kernels/interview/` 下**没有** sgemm 的测试/bench 驱动（目录内只有 `bench_attn.cu`、`bench_ffpa.cu`、`bench_sdpa.py`、`build.sh`、各 `*.cuh` 与 `notes-v2.*`）；`notes-v2.cu` 里出现的 `epsilon = 1e-5f` 属于其它 phase（L930/L988），与 SGEMM 无关。
- **注释里的关键结论 / 勘误**：
  - ⚠️ `// 这里不用pragma unroll，因为K不是编译器常量，编译器无法展开循环` [L54]：紧跟其后的 `for (int bk = 0; bk < (K + BK - 1) / BK; ++bk)` 确实**没有** `#pragma unroll`（与注释一致），但 K-slice 循环内层 `for (int k = 0; k < BK; ++k)` **有** `#pragma unroll` —— 两处注释/代码一致，是「外层不加、内层加」的正确示范。
  - ⚠️ `constexpr int BM = 32; // vec 版: 32x4 = 128` / `constexpr int BN = 32; // vec 版: 32x4 = 128` [L31-32]：这两行注释指向的是下一个 kernel（`sgemm_vec4` 的 128×128），在本 kernel 内是「对照说明」而非本 kernel 的配置，容易误读。
  - ⚠️ 对应 `sgemm.cu`/`sgemm_async.cu` 里同类注释 `// BK:TILE_K=8 BM=BN=128` 与 `// [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素` 的**旧口径**（`sgemm.cu` L89-93 的注释块写的是 256 线程 / (16,16)），与本 `sgemm.cuh` 的 4×4 tile / 1024 线程口径**不是同一实现**，做「GEMM 阶梯」章节时不要把两者的线程数与 tile 数混用。
  - ⚠️ `__shared__ float s_a[BM][BK], s_b[BK][BN]; //  32x32x4=4KB smem` [L34]：4KB 只是 `s_a` 一个数组；两个数组合计 8KB。
  - ⚠️ `__global__ void sgemm(...)` 里 `int tid = threadIdx.y * blockDim.x + tx;` 用到了 `threadIdx.y`，但声明的 1024 线程若按注释的 `Block: (32, 32, 1)` 传入，`tid` 覆盖 0~1023 而加载/计算只用到低 32 位 → 与「一个线程计算 c 的一个元素、32×32 线程」的说明在实现上只依赖 `tx`，注释里「1024 线程」与「32×32 线程每个加载 1 元素」混用（功能上等价，但 tid 表达式的 32×32 语义未被使用）。
  - Bank conflict 加分点原文摘录 [L96-99]：`// ⚠ Bank Conflict 提示（面试加分点）：s_b[32][128] 上 warp 内 32 线程按 stride=4 访问（tid%32 决定列 0,4,8,...,124）→ 每 4 个线程落同一 bank 不同地址 → 4-way bank conflict。生产代码可用 s_b[BK][BN+1] PAD 打散，这里保持最简布局便于讲解。`

---

## 测试与容差（统一章节）

- **驱动**：`kernels/sgemm/sgemm.py`，`load(name="sgemm_lib", sources=["sgemm.cu", "sgemm_async.cu", "sgemm_wmma_tf32_stage.cu", "sgemm_cublas.cu"], extra_cuda_cflags=["-O3", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math"], extra_cflags=["-std=c++17"])`
- **⚠️ 没有 tol / allclose / rtol / atol**：`sgemm.py` 全仓库 grep 无 `allclose`；`run_benchmark` 只打印 `out.flatten()[:2]` 两个值（`round(v, 8)`、`f"{v:<12}"[:10]`），靠「同 shape 内所有实现输出完全相同的两个浮点」隐式判断等价（见前面「输出值指纹」）。所以 **TF32 档没有数值容差常量**，可引用的「容差事实」只有指纹差：
  - K=2048：FP32 `['70.6019897', '26.1625347']` vs TF32 `['70.5943985', '26.1725273']`
  - K=4096：FP32 `['151.780014', '4.5990448 ']` vs TF32 `['151.794143', '4.5965395 ']`（M=4096,N=4096 行）
  - K=8192：FP32 `['118.496635', '44.2837791']` vs TF32 `['118.526184', '44.2716636']`
- **参考实现**：
  - CUDA Core 档：`lib.sgemm_cublas`（`cublasGemmEx(..., CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, ..., B, CUDA_R_32F, N, A, CUDA_R_32F, K, ..., C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)` + `cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)`）[`sgemm_cublas.cu` L25-38]，以及 `partial(torch.matmul, out=c)`，标签 `f32(cublas)` / `f32_th`
  - TF32 档：`lib.sgemm_cublas_tf32`（`cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`）[`sgemm_cublas.cu` L40-54]，标签 `tf32(cublas+tf32)`
  - 注意：所有 kernel 都走 row-major(NN) 语义，cuBLAS 侧靠交换 A/B 与 M/N 并传 `CUBLAS_OP_N, CUBLAS_OP_N` 实现（`N, M, K` 顺序 + lda=N, ldb=K）
- **shape 列表**：`Ms = [4096, 8192, 16384]`、`Ns = [4096, 8192, 16384]`、`Ks = [2048, 4096, 8192]`，`MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]` → **27 个 shape**；tensor 用 `torch.randn(..., dtype=torch.float).cuda()`，按 `A[:M, :K].contiguous()` 切片（预分配到 `MAX_M, MAX_N, MAX_K = 16384, 16384, 8192`）
- **计时**：`warmup=2, iters=20`；但 `if a.size(0) > 1024 or a.size(1) >= 1024 or b.size(1) > 1024: iters = 10` —— 本 shape 列表所有 size 都 >1024，故**实际 iters 恒为 10**；计时用 `time.time()` + `torch.cuda.synchronize()`，`TFLOPS = (2 * M * N * K) * 1e-9 / (mean_time)`（mean_time 单位 ms）
- **被 benchmark 的 kernel 清单**（顺序即输出顺序）：`sgemm_t_8x8_sliced_k_f32x4`、`sgemm_t_8x8_sliced_k_f32x4_bcf`、`sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf`、`sgemm_cublas`、`torch.matmul`，然后 WMMA：`stages=3`、`stages=2`、`stages=3 + dsmem`、`stages=2 + dsmem`、`stages=3 + swizzle=True`、`stages=2 + swizzle=True`、`stages=3 + dsmem + swizzle`、`stages=2 + dsmem + swizzle`，最后 `sgemm_cublas_tf32`
  - **`sgemm_naive_f32` 被注释掉**（`# run_benchmark(lib.sgemm_naive_f32, a, b, "f32(naive)", c)`）
  - **`sgemm_async.cu` 的 6 个 kernel 与 `sgemm_t_8x8_sliced_k_f32x4_bcf_offset` / `_dbuf_offset` 都未被 benchmark**
  - `swizzle_stride` 由 Python 计算：`int((int(N / 8) // 256) * 256)`，`< 256` 则关闭 swizzle；因此表里出现 512 / 1024 / 2048 三档

---

## 引用时的三条硬提醒

1. **`sgemm.cu` 的 dbuf kernel 末块选 buffer 表达式（`smem_sel_last`）与 README 版本不一致（奇偶取反）**，在 K/BK 为偶数（本 repo 测的 K=2048/4096/8192 全是）时会读错 buffer —— 讲「Double Buffer 收尾怎么写」时，以 README 的 `s_a[1]/s_b[1]` 为准，别照抄 `sgemm.cu`。
2. **「bcf（bank conflicts free）」的默认实例化 `OFFSET = 0`**，没有 padding；只有显式 `_offset`（OFFSET=4）版本才 pad，而它没进 benchmark。
3. **TF32 档的数值容差在本 repo 里没有常量**，只有「所有 TF32 kernel 输出前两个 float 完全一致」和 FP32 指纹之间的 ~1e-4 量级差；引用容差时必须说明这是观察值不是断言值。


---

# D · mma.sync + ldmatrix + XOR swizzle（Q184 Q186）

# LeetCUDA 参考实现精确事实抽取（D: hgemm mma / ldmatrix / swizzle）

- 仓库：`C:\Users\Jeff\Documents\GitHub\LeetCUDA`
- 抽取方式：仅 `read` / 只读命令，**未修改任何仓库文件**；仅本文件是新写入的产物。
- **HEAD 在本任务执行期间发生了漂移（重要）**：任务书给定的 `e831d970a099f5ce8fd0495ddd1df09206d56918` 是我第一次 `git rev-parse HEAD` 时的 HEAD；随后仓库被更新到 **`6c86259`**（`git status --porcelain` 为空，工作区干净）。
  - `git diff e831d970 6c86259 -- <本任务 5 个文件>`：**只有 `kernels/interview/hgemm.cuh` 变了 1 行** —— 第 1874 行 `__launch_bounds__(kNumThreads)` → `__launch_bounds__(kNumThreads, 1)`（`hgemm_tma_mma_ws_tn`）。其余 4 个文件在新旧 HEAD 之间**逐字节相同**。
  - 因此：`kernels/swizzle/*` 的事实对两个 revision 都成立；`hgemm.cuh` 的行号也完全不变（是原地替换，非增删行），**只有 1874 行这一处是新 HEAD 的修复**。该修复本身值得写进教程（见 hgemm.cuh 勘误第 7 条）。
  - 另：`kernels/interview/notes-v2.cu` 在新 HEAD 中 +356 行（4865 → 5219），本文件引用它的行号时同时给出函数名。
- **行数与任务书不一致（以实测为准）**：任务书给的行数是 `Measure-Object -Line`（只数非空行）的结果。磁盘真实行数：`hgemm.cuh` **2100**（136 空行；非空 1964）、`hgemm_mma_swizzle.cu` **706**（75 空行）、`mma_simple_swizzle.cu` **241**（25 空行）、`print_swizzle_layout.py` **255**（19 空行）、`README.md` **197**。
- 本文件的 5 个目标文件都**没有任何 `SWIZZLE` PTX 指令**：swizzle 是纯 C++ 地址算术（XOR 后当作列索引）；唯一的"硬件 swizzle"是 TMA descriptor 的 `CU_TENSOR_MAP_SWIZZLE_128B` 与 WGMMA descriptor 的 `layout_type=1`。构建：`kernels/swizzle/makefile`，`ARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89`，`DEFAULT_FLAGS=-O2 $(ARCHS) -std=c++17 $(INCLUDE_DIRS) --expt-relaxed-constexpr -lcublas`，产出 `hgemm_mma_swizzle.bin` / `mat_trans_swizzle.bin` / `mma_simple_swizzle.bin`（另有 `hgemm_89` / `mma_89` / `mat_89` 三个 sm_89 目标）。

---

## kernels/interview/hgemm.cuh

- **文件头**：`#pragma once` + `#include "base.cuh"`（`base.cuh` → `#include "common.cuh"`；**所有 PTX 宏都定义在 `common.cuh`，不是本文件**）。第 3 行注释：`// hgemm.cuh: Phase 7b-d HGEMM (MMA/Swizzle/CuTe/WGMMA/TMA_MMA_WS)`。

- **阶梯/版本**（文件名 → 行号 → 技术点；行号为 e831d970 与 6c86259 共有）
  1. Phase 7b-2 `hgemm_mma_stages_tn`（120–386）：m16n8k16 + cp.async 多 stage（kStages=3）+ TN 布局 + "统一循环"（k 从 0 开始，加载/计算合一个循环）+ `kBlockSwizzle` 3D grid 开关。**BK=16，无 smem swizzle，无 PAD 参数**。
  2. `static __device__ __forceinline__ int swizzle<kColStride=16>(i, j)`（389–396）：派发器，`#if defined(NOTES_V2_ENABLE_SWIZZLE_V2)` → `swizzle_v2_impl<kColStride>`，否则 `swizzle_v1_impl<kColStride>`（两者都在 `common.cuh`）。
  3. Phase 7b-3 `hgemm_mma_stages_tn_swizzle`（424–715）：在 7b-2 基础上加 **smem XOR swizzle（`swizzle<kMmaK>`=SWIZZLE_32B）+ 寄存器双缓冲**，`kValTileK=4`（BK=64），`kStages` 默认 2。
  4. Phase 7c-1 `hgemm_mma_stages_tn_cute`（806–1194，`#if defined(NOTES_V2_ENABLE_CUTE)` 内）：CuTe DSL，`Swizzle<3,3,3>`（SWIZZLE_128B），BM=128/BN=256(F16acc) 或 128(F32acc)/BK=32，`kStage=2`，128 线程。
  5. Phase 7c-2 `launch_hgemm_mma_stages_tn_cute<T,Stages=2,BlockSwizzle=0,kAccF32=false>`（1214–1425）：类型实例化 + grid/block + 动态 smem + `cudaFuncSetAttribute`。
  6. Phase 7d `hgemm_wgmma_stages_tn`（1468–1855，`NOTES_V2_ENABLE_WGMMA`）：WGMMA m64n128k16 + TMA + Warp Specialization，kStages=3。
  7. Phase 7e `hgemm_tma_mma_ws_tn`（1863–2098，`NOTES_V2_ENABLE_TMA_MMA_WS`）：SM120 的 TMA + warp-specialized **mma.sync 消费者**（ldmatrix + `swizzle<64>` = SWIZZLE_128B），BM=BN=128/BK=64。

- **每级的确切配置（模板参数全列表，含默认值，逐字原样）**
  - `hgemm_mma_stages_tn`（120–129）：
    `const int kMmaM = 16`, `const int kMmaN = 8`, `const int kMmaK = 16`, `const int kMmaTileM = 2`, `const int kMmaTileN = 4`, `const int kValTileM = 4`, `const int kValTileN = 4`, `const int kStages = 3`, `const int kBlockSwizzle = 0`；`__global__ void __launch_bounds__(256)`。
    `BM = kMmaM * kMmaTileM * kValTileM = 128`，`BN = kMmaN * kMmaTileN * kValTileN = 128`，`BK = kMmaK = 16`；warp 排布 `warp_m = warp_id % kMmaTileM`（0,1）、`warp_n = warp_id / kMmaTileM`（0..3）；8 warps × 32 = 256 线程；**每 warp 16 个 MMA atom**（kValTileM×kValTileN = 4×4）。
  - `hgemm_mma_stages_tn_swizzle`（424–434）：同上但 `const int kValTileK = 4`（BK = kMmaK*kValTileK = 64）、`const int kStages = 2`；3 条 static_assert：`kValTileK >= 2`、`kBlockSwizzle == 0 || == 1`、`kStages >= 2`。
  - `hgemm_mma_stages_tn_cute`（806–820）：`typename T`, `int BM, int BN, int BK`, `int kStage`, `typename TiledMMA`, `G2SCopyA`, `G2SCopyB`, `SmemLayoutA`, `SmemLayoutB`, `SmemLayoutC`, `S2RCopyAtomA`, `S2RCopyAtomB`, `R2SCopyAtomC`, `S2GCopyAtomC`, `S2GCopyC`, `const int BlockSwizzle = 0`。wrapper 侧：`BM=128`，`static constexpr int kBN = kAccF32 ? 128 : 256`，`BK=32`，`KStage=Int<Stages>`，`kSmemLayoutCBatch=Int<4>`；`kMmaEURepeatM=2, kMmaEURepeatN=2, kMmaEURepeatK=1`，`kMmaPM=1*kMmaEURepeatM*get<0>(mma_atom_shape{}) = 32`，`kMmaPN=2*kMmaEURepeatN*get<1>(...) = 32`，`kMmaPK=1*kMmaEURepeatK*get<2>(...) = 16`；`dim3 block(size(MMA{}))` = **128 线程（4 warps × 2×2 EU）**。
  - `hgemm_wgmma_stages_tn`（1468–1477）：`kWgmmaM = 64`, `kWgmmaN = 128`, `kWgmmaK = 16`, `BM = 128`, `BN = 128`, `BK = 64`, `kNumThreads = 256`, `kStages = 3`, `kBlockSwizzle = 0`；`__launch_bounds__(kNumThreads, 1)`。`kConsumerThreads = 128`，`kNumConsumers = (kNumThreads/kConsumerThreads) - 1 = 1`，`kWarpgroupM = BM/kNumConsumers = 128`。
  - `hgemm_tma_mma_ws_tn`（1863–1874）：`kMmaM = 16`, `kMmaN = 8`, `kMmaK = 16`, `kMmaTileM = 2`, `kMmaTileN = 2`, `kValTileM = 4`, `kValTileN = 8`, `kValTileK = 4`, `kStages = 2`, `kNumThreads = 256`, `kBlockSwizzle = 0`；`BM = 16*2*4 = 128`，`BN = 8*2*8 = 128`，`BK = 16*4 = 64`；**消费者只有 4 个 warp**，`warp_m = warp_id % 2`、`warp_n = warp_id / 2`（2×2 grid），每 warp 32 个 atom（4×8）。static_assert 原文：`"The consumer warpgroup has exactly four warps"`、`"TMA desc and 128B swizzle require 128x128x64"`、`"Use one producer and one consumer warpgroup"`。
  - 网格/线程（逐版本，逐字）
    - 7b-2 / 7b-3：`dim3 block(NUM_THREADS); dim3 grid(div_ceil(N, BN), div_ceil(M, BM));`；notes-v2.cu 测试里实际写的是 `dim3 block(256); dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);`（bench 的 3D swizzle grid 见下方"测试与容差"）。
    - CuTe：`int BX = (N + BN - 1) / BN; int BY = (M + BM - 1) / BM; int BZ = BlockSwizzle ? (N + kSwizzleStride - 1) / kSwizzleStride : 1; BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX; dim3 block(size(MMA{})); dim3 grid(BX, BY, BZ);`，其中 `constexpr int kSwizzleStride = 2048;`；`When 不启用 BlockSwizzle 时，BZ=1 退化为纯 2D grid`（注释原文）。
    - WGMMA：注释原文 `Grid: ((N+127)/128/S, (M+127)/128, S)，S=(N+2047)/2048，3D block swizzle` / `Block: (256, 1, 1)，2 warpgroups`；`const int wg_idx = threadIdx.x / kConsumerThreads; // 0=Producer, 1=Consumer`。
    - TMA_MMA_WS：`block(256)`，消费者 4 warps 组成 2×2 grid；边界守卫 `if (bx >= div_ceil(N, BN) || by >= div_ceil(M, BM)) return;`。
  - CuTe 类型定义逐条（wrapper，逐字；"类型即配置"）
    - `using SmemLayoutAtom = decltype(composition(Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<BK>{}), make_stride(Int<BK>{}, Int<1>{}))));`，再 `using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));`（`SmemLayoutB` 同形但用 `Int<BN>`）。
    - `using mma_op = std::conditional_t<kAccF32, SM80_16x8x16_F32F16F16F32_TN, SM80_16x8x16_F16F16F16F16_TN>; using mma_traits = MMA_Traits<mma_op>; using mma_atom = MMA_Atom<mma_traits>; using mma_atom_shape = mma_traits::Shape_MNK; // (Int<16>, Int<8>, Int<16>)`。
    - `using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;` + `using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));` → 逻辑 MMA tile = 32×32×16，`128 threads = 4 warps × (2×2 EU slices)`。
    - G2S：`using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;` + `make_tiled_copy(g2s_copy_atom{}, make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})), make_layout(make_shape(Int<1>{}, Int<8>{})))`（ThrLayout{32,4}，ValLayout{1,8} = 每线程 8 half = 128 bit）；`using G2SCopyB = G2SCopyA;`。
    - S2R：`using s2r_copy_op = SM75_U32x4_LDSM_N;`（即 ldmatrix.x4），`make_tiled_copy_A/B(S2RCopyAtomA/B{}, tiled_mma)` 自动推导线程-数据映射。
    - C scratchpad：`using SmemLayoutAtomC = decltype(composition(Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), make_stride(Int<kMmaPN>{}, Int<1>{}))));` → `SmemLayoutC` = 32×32×4（`kSmemLayoutCBatch = Int<4>{}`）。
    - R2S：`using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;`（32-bit，2 half/次）；S2G：`using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;` + `S2GCopyC = make_tiled_copy(atom, make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})), make_layout(make_shape(Int<1>{}, Int<8>{})))`（tiler = (32,32)，与 R2S 一致）。
    - smem 账：`shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})`（BN=256 时 `cosize(SmemLayoutA) = 128*32*2 = 8192`、`cosize(SmemLayoutB) = 256*32*2 = 16384`，合计 24576 个 T = **48 KiB**；注释 773 行说 `共 49,152 bytes（约 48 KiB）`）；`kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T)`；`static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}), "C shared memory must fit within one A pipe");`

- **PTX 宏（逐条：宏名 + 完整指令字符串 + 操作数含义）** —— 全部定义在 `kernels/interview/common.cuh`
  - `HMMA16816(RD0,RD1,RA0..RA3,RB0,RB1,RC0,RC1)`：`"mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"`；出参 2×`"=r"`（D 的两个 uint32），入参 4×`"r"` A（RA0–RA3）、2×`"r"` B（RB0,RB1）、2×`"r"` C（RC0,RC1 = 累加器，原地传 RC）。**这是本文件 mma 路径唯一使用的 MMA 宏**。
  - `HMMA16816F32(RD0..RD3, RA0..RA3, RB0,RB1, RC0..RC3)`：`"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"`；**在本文件（hgemm.cuh）中定义但未被使用**（CuTe 路径走 atom）。
  - `LDMATRIX_X4(R0,R1,R2,R3,addr)`：`"ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"`（4×`"=r"` + 1×`"r"` addr）。
  - `LDMATRIX_X2(R0,R1,addr)`：`"ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"`。
  - `LDMATRIX_X2_T(R0,R1,addr)`：`"ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"`；**在本文件的三条 mma kernel 中均未使用**（TN 布局下 B 存成 B^T row-major，不需要 .trans）。
  - 未使用但存在：`LDMATRIX_X1`（`ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];`）、`LDMATRIX_X1_T`（`...x1.trans.m8n8...`）、`LDMATRIX_X4_T`（`...x4.trans.m8n8...`）。
  - `CP_ASYNC_CG(dst,src,bytes)`：`"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes)` —— 全文件所有 G→S 都用它，bytes 恒为 16。
  - `CP_ASYNC_COMMIT_GROUP()`：`"cp.async.commit_group;\n"`；`CP_ASYNC_WAIT_ALL()`：`"cp.async.wait_all;\n"`；`CP_ASYNC_WAIT_GROUP(n)`：`"cp.async.wait_group %0;\n" ::"n"(n)`。**`CP_ASYNC_WAIT_ALL` 在本文件未被调用**。
  - `CP_ASYNC_CA(...)`（`"cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"`）**只存在于 `kernels/swizzle/*.cu`，`common.cuh` 没有 CA 版本**。
  - `WGMMA_FENCE()`：`"wgmma.fence.sync.aligned;\n" ::: "memory"`；`WGMMA_COMMIT_GROUP()`：`"wgmma.commit_group.sync.aligned;\n" ::: "memory"`；`WGMMA_WAIT_GROUP(n)`：`"wgmma.wait_group.sync.aligned %0;\n" ::"n"(n) : "memory"`。
  - `WGMMA_M64N128K16_F16F16F16(d,sA,sB,ScaleD,ScaleA,ScaleB,TransA,TransB)`：`"wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16 {%0..%31}, %32, %33, %34, %35, %36, %37;\n"`（32 个 `"+r"` 累加器 = `d[8][4]`，2 个 `"l"` desc，5 个 `"n"` 立即数）。内部先 `uint64_t desc_a = make_smem_desc(&(sA)[0]); uint64_t desc_b = make_smem_desc(&(sB)[0]);`。
  - `make_smem_desc(half*)`（common.cuh，非 PTX 宏）：`desc |= SMEM_DESC_ENCODE(addr);` + `SMEM_DESC_ENCODE((uint64_t)16) << 16`（LBO 占位）+ `SMEM_DESC_ENCODE((uint64_t)1024) << 32`（SBO）+ `desc |= 1llu << 62;`（layout_type=1 = 128B swizzle）；`SMEM_DESC_ENCODE(x) = ((((uint64_t)(x)) & 0x3FFFF) >> 0x4)`。
  - TMA（`NOTES_V2_FORCE_INLINE_ASYNC_PROXY` 路径）：
    `tma_load_2d`：`"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"`（`"r"` smem dst, `"l"` gmem desc, `"r"` mbar, `"r"` minor_coord, `"r"` major_coord）；
    `tma_arrive_expect_tx`：`"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"`；
    `tma_fence_proxy_async_shared_cta`：`"fence.proxy.async.shared::cta;\n" ::: "memory"`。
  - `setmaxnreg`：`"setmaxnreg.dec.sync.aligned.u32 %0;\n"` / `"setmaxnreg.inc.sync.aligned.u32 %0;\n"`，通过 `NOTES_V2_REG_DEALLOC(N)` / `NOTES_V2_REG_ALLOC(N)` 使用（WGMMA/TMA_MMA_WS 消费者用 `NOTES_V2_REG_ALLOC(232)`，生产者用 `NOTES_V2_REG_DEALLOC(40)`）；未定义 `NOTES_V2_ENABLE_SETMAXNREGS` 时宏展开为 `((void)0)`。
  - 数据搬运宏是 **reinterpret_cast 而非 PTX**：`LDST32BITS=reinterpret_cast<half2*>`, `LDST64BITS=reinterpret_cast<float2*>`, `LDST128BITS=reinterpret_cast<float4*>`。

- **WGMMA / TMA descriptor 位域（`common.cuh` 注释原文；教程可整表照抄）**
  - `| start_address       | [0, 14)   | 14 | smem 基址编码 |` → `SMEM_DESC_ENCODE(x) = ((((uint64_t)(x)) & 0x3FFFF) >> 0x4)`（解码 `<< 4`；可寻址 2^18 half = 512 KB）。
  - `| leading_byte_offset | [16, 30)  | 14 | 主要维度（K）的字节步长；swizzle 下硬件未使用（assumed=1）|` → 这里填 `(uint64_t)16`（编码后 = 1，仅占位）。
  - `| stride_byte_offset  | [32, 46)  | 14 | 跨越维度（M/N）的字节步长：K-Major 下为 8-row stripe 间距 |` → 这里填 `(uint64_t)1024`（128B swizzle atom = 8 rows × 64 half × 2 B）。
  - `| base_offset         | [49, 52)  | 3  | swizzle pattern 偏移 |` → 本实现不显式置位（=0，要求 smem ptr 已 1024 B 对齐）；`(pattern_start_addr >> 7) & 0x7`。
  - `| layout_type         | [62, 64)  | 2  | 0=None, 1=128B swizzle, 2=64B swizzle, 3=32B swizzle |` → 代码 `desc |= 1llu << 62;`。
  - TMA host 侧（`create_tensor_map`，逐字要点）：`uint64_t gmem_prob_stride[5] = {sizeof(half), sizeof(half) * BlockMinorSize * blocks_width, 0, 0, 0};` 传参用 **`gmem_prob_stride + 1`**（`globalStrides` 只收 tensorRank-1 个值，最内维 stride 隐式）；`uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};` **不需要 +1**（`smemBoxStrides` 收满 tensorRank 个值）；`uint32_t smem_box_shape[5] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize), 1, 1, 1};`；`CU_TENSOR_MAP_DATA_TYPE_FLOAT16`, `CU_TENSOR_MAP_INTERLEAVE_NONE`, `CU_TENSOR_MAP_SWIZZLE_128B`, `CU_TENSOR_MAP_L2_PROMOTION_NONE`, `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`。
  - smem 布局结构体：`template <int BM, int BN, int BK, int QSIZE> struct WgmmaSMem { alignas(128) half A[BM * BK * QSIZE]; alignas(128) half B[BN * BK * QSIZE]; };`（B 物理 [BN, BK]，逻辑 K-major [BK, BN] 由 descB 的 128B swizzle 在硬件地址重映射层完成，`B[BN * BK * QSIZE]` 与 `B[BK * BN * QSIZE]` 数值等价）；`template <int BM, int BN, int BK, int QSIZE> struct TmaMmaWSSMem { static_assert(BK == 64, "The 128B swizzle helper below is specialized for BK=64"); half A[BM * BK * QSIZE]; half B[BN * BK * QSIZE]; };`

- **fragment 与寄存器账**
  - 7b-2：`uint32_t RC[kValTileM][kValTileN][2] = {0};` = `[4][4][2]` = **32 个 uint32 累加器**；循环内 `uint32_t RA[kValTileM][4]` = [4][4] = **16**、`uint32_t RB[kValTileN][2]` = [4][2] = **8**。epilogue 追加 `uint32_t RC0[kValTileN][4]; uint32_t RC1[kValTileN][4];`（4+4 个临时）。
  - 7b-3：`RC[kValTileM][kValTileN][2]` = [4][4][2] = **32**；`uint32_t RA[2][kValTileM][4]` = [2][4][4] = **32**（双缓冲）；`uint32_t RB[2][kValTileN][2]` = [2][4][2] = **16**（双缓冲）。epilogue 复用 `RA[2][4][4]` 做 shuffle 缓冲（注释原文：`做RC的warp shuffle正好需要[2][4][4]的寄存器空间，RA[2][4][4]正好可以复用`）。
  - TMA_MMA_WS：`uint32_t RA[kValTileM][4]` = 16、`uint32_t RB[kValTileN][2]` = [8][2] = 16、`uint32_t RC[kValTileM][kValTileN][2] = {0}` = [4][8][2] = **64**。
  - WGMMA：`uint32_t d[kWarpgroupM / kWgmmaM][kWgmmaN / 16][4] = {}` = `[2][8][4]` = **64 个 uint32 = 128 half/线程**；注释算账：`128 线程 * 128 half = 16384 half = 128 * 128 = BM*BN（刚好覆盖整个 C tile）`。
  - CuTe：`tCrA=(MMA, MMA_M, MMA_K)`、`tCrB=(MMA, MMA_N, MMA_K)`、`tCrD=(MMA, MMA_M, MMA_N)`；`int num_k_steps = size<2>(tCrA);` = `BK/kMmaPK = 32/16 = 2`。

- **ldmatrix 用法（x1/x2/x4 位置、地址表达式、是否 .trans）**
  - 7b-2 A（x4，**非 trans**）：`int warp_smem_a_m = warp_m * (kMmaM * kValTileM) + i * kMmaM;`、`int lane_smem_a_m = warp_smem_a_m + lane_id % 16;`、`int lane_smem_a_k = (lane_id / 16) * 8;`，地址 `smem_a_base_ptr + (smem_sel * s_a_stage_offset + lane_smem_a_m * BK + lane_smem_a_k) * sizeof(half)`；lane 0–15 提供 k=0..7 的 16 行、lane 16–31 提供 k=8..15 的同一批行（16×16 的 4 个 8×8 = col-major 顺序访问）。
  - 7b-2 B（x2，**非 trans**）：`int warp_smem_b_n = warp_n * (kMmaN * kValTileN) + j * kMmaN;`、`int lane_smem_b_n = warp_smem_b_n + lane_id % 8;`、`int lane_smem_b_k = ((lane_id / 8) % 2) * 8;`；注释原文：`ldmatrix.{...}.x2.{...} 需要warp内前16个线程参与，后16个线程传的addr会被忽略`。
  - 7b-3：A/B 地址与 7b-2 相同，额外在 k 方向加 `smem_k_offset = (k_step + 1) * kMmaK` **加在 swizzle 之外**（因为 `swizzle<kMmaK>` 的周期只有 16 列），并用 `swizzle<kMmaK>(lane_smem_a_m, lane_smem_a_k)` 替换列索引。初始化预加载：stage 0 的 k_step=0 → `RA[reg_st_idx=0] / RB[0]`。
  - TMA_MMA_WS（`swizzle<64>`，BK=64）：`const int lane_smem_a_k = (k_step * kMmaK) + (lane_id / 16) * 8;` —— `k_step * kMmaK` **必须放进 swizzle 内部**；原注释：`swizzle<64> 作用于完整 BK=64，swizzle 周期覆盖全部 64 列，k_step=0 和 k_step=1 的 chunk 会被 XOR 交叉混合 → k_step 偏移必须在 swizzle 内部参与 chunk 计算`。B 侧 `const int lane_smem_b_k = (k_step * kMmaK) + ((lane_id / 8) % 2) * 8;`。
  - **x1 在 hgemm.cuh 中完全未使用**；`.trans` 形式在该文件中完全未使用（TN 布局的直接后果）。

- **C fragment → shuffle → global 写回（三个 mma 版本共用同一套映射，逐字）**
  - 映射（注释原文）：`row = t/4, col-pair = t%4`（每个 uint32 = 2 个 half）；`RC[0] for rows 0-7, RC[1] for rows 8-15`；`lane_id / 4 → 行号映射`（lane_id 0~3 → row 0 / row 8，4~7 → row 1 / row 9，…，28~31 → row 7 / row 15）；`Within a 4-lane group (e.g. T0-T3 for row 0): lane+0 holds {c0,c1}, lane+1 holds {c2,c3}, lane+2 holds {c4,c5}, lane+3 holds {c6,c7}.`
  - shuffle（每 (i, j) 6 条，mask 恒 `0xffffffff`）：`RC0[j][1] = __shfl_sync(0xffffffff, RC[i][j][0], lane_id + 1);`、`RC0[j][2] = ... lane_id + 2);`、`RC0[j][3] = ... lane_id + 3);`，RC1 同理对 `RC[i][j][1]` 做 3 条。
  - store（7b-2/7b-3）：`if (lane_id % 4 == 0) { ... *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_0]) = *reinterpret_cast<float4 *>(&RC0[j][0]); *reinterpret_cast<float4 *>(&C[store_gmem_c_addr_1]) = *reinterpret_cast<float4 *>(&RC1[j][0]); }`，其中 `store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n`、`store_gmem_c_addr_1 = (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n`（128-bit 一次写 8 个 half）。
  - WGMMA 的 C 映射完全不同（注释原文）：`row = warp * 16 + lane / 4`（0~63）、`col = g * 16 + 2 * (lane % 4)`（0~126，step 2）；4 个 uint32 = 四象限：`d[m][g][0]` → (row, col)、`[1]` → (row+8, col)、`[2]` → (row, col+8)、`[3]` → (row+8, col+8)，每次 `*reinterpret_cast<uint32_t *>(...) = d[m][g][x];` 写 2 个 half（**不是 128-bit 写回**，1836 行 NOTE 已注明可改 R→S→G）。
  - TMA_MMA_WS 的 epilogue 复用 `RA[0] / RA[1]` 做 shuffle 缓冲（与 7b-3 同款 6 条 `__shfl_sync`），仍由 `if (lane_id % 4 == 0)` 发 `float4` store，写回 `C[store_lane_gmem_c_m * N + store_lane_gmem_c_n]` 与 `C[(store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n]`。

- **swizzle（位运算公式原样 + padding 对照 + 声称的冲突度）**
  - 公开公式（`common.cuh::permuted<kColStride,kStep=8>`）：`kStep=8, kColStride=16` → `return (((j >> 3) ^ (i >> 2)) & 1) << 3;  // SWIZZLE_32B`；`kColStride=32` → `const int chunk = (j >> 3) & 3; const int xor_mask = ((i >> 1) & 1) | (((i >> 2) & 1) << 1); return (chunk ^ xor_mask) << 3;  // SWIZZLE_64B`；`kColStride=64` → `const int chunk = (j >> 3) & 7; const int xor_mask = (i & 1) | (((i >> 1) & 1) << 1) | (((i >> 2) & 1) << 2); return (chunk ^ xor_mask) << 3;  // SWIZZLE_128B`；`kStep=4` 遗留分支 → `return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;`。
  - v2（cute 位级镜像）：`yyy_msk = bit_msk << (M + (S > 0 ? S : 0)); zzz_msk = bit_msk << (M - (S < 0 ? S : 0)); bit_msk = (1 << B) - 1;`，`apply(offset) = offset ^ ((offset & yyy_msk) >> S)`；`swizzle_v2_impl` 里 `off = (i * kColStride + j) * (int)sizeof(half); return ((sw >> M) & ((1 << B) - 1)) * kStep;`，`B = 1/2/3`（kColStride 16/32/64），`M = 4`，`S = 3`，`kStep = 8`。**kColStride=8 在 v2 中回退 v1**。
  - 本文件实际调用：7b-3 全部用 `swizzle<kMmaK>`（=16 → SWIZZLE_32B）；TMA_MMA_WS 消费者用 `swizzle<64>`（→ SWIZZLE_128B，与 TMA descriptor 的 `CU_TENSOR_MAP_SWIZZLE_128B` 对齐）。**7b-2 完全不用 swizzle**。
  - padding 对照结论（`common.cuh` 注释原文）：`优势：无需 smem PAD（不浪费空间），原理上完全消除特定 pattern 的 bank conflict。` / `局限性：要求 kColStride ≤ 16（即 BK ≤ 16），kStep ∈ {4,8}。对 MMA m16n8k16 的 BK=16 而言刚好满足。` / 效果声明 `原来 n-way 的 bank conflict 降低到 1-way（fully conflict-free）`。本文件**没有任何 A_PAD/B_PAD 模板参数**；7b-2 注释说 `这里先保留最简 PAD=0 版本`（但根本没有 PAD 形参，见勘误 2）。
  - CuTe 路径的等价位公式（注释原文，1279 行附近）：`offset' = offset ^ ((offset & YYY) >> S)，YYY = ((1 << B) - 1) << (M + S)`；`对 Swizzle<3,3,3>：每个基本元素有 2^3=8 个值，每行有 2^3=8 个基本元素，二维空间有 2^3=8 行；元素地址位 [6:8] XOR 到 [3:5]，低 3 位保持不变。一个完整 swizzle 周期覆盖 2^(M+S+B)=2^9=512 个元素（FP16 下为 1024B）`。wrapper 里注释写的实现形式：`offset' = offset ^ ((offset & (0b111 << 6)) >> 3)`。

- **同步与流水**
  - 7b-2：预加载 `for (int k = 0; k < (kStages - 1); ++k)` 每 stage 一次 `CP_ASYNC_COMMIT_GROUP()` → `CP_ASYNC_WAIT_GROUP(kStages - 2)` → `__syncthreads()`。主循环内：`smem_sel = k % kStages`、`smem_sel_next = (k + kStages - 1) % kStages`，`if (k + kStages - 1 < NUM_K_TILES)` 才发 cp.async（A、B 各一条 CG 16B 后一次 COMMIT）；MMA 之后 `if (k + kStages - 1 < NUM_K_TILES) CP_ASYNC_WAIT_GROUP(kStages - 2); else CP_ASYNC_WAIT_GROUP(0);` 然后 `__syncthreads()`。kStages=3 → 满载期 wait 参数 = **1**，尾部 = **0**。
  - 7b-3：预加载 (kStages-1) 个 stage × 全部 k_step，每个 stage 一次 COMMIT → `CP_ASYNC_WAIT_GROUP(kStages - 2)` → `__syncthreads()`。主循环：G→S 条件预取（k_step 内层循环 + 一次 COMMIT）→ 内层 k_step 循环 `reg_st_idx ^= 1; reg_ld_idx ^= 1;`（0→1、1→0）→ 条件 ldmatrix k_step+1 → 16 条 HMMA → 自适应 wait → `__syncthreads()` → `if (k + 1 < NUM_K_TILES)` 预加载下一个 stage 的 k_step=0。**kStages 默认 2 时 `kStages - 2 == 0`，自适应 wait 退化为恒 0**。
  - TMA_MMA_WS / WGMMA：`cuda::barrier<cuda::thread_scope_block> full[kStages]; empty[kStages];`（`#pragma nv_diag_suppress static_var_with_dynamic_init` 包裹），`init(&full[i], kConsumerThreads + 1)` = **arrive_count = 129**；thread 0 初始化后 `tma_fence_proxy_async_shared_cta(); __syncthreads();`。协议：Producer `empty[stage].wait(empty[stage].arrive());` → `tma_load_2d(...)` ×2 → `tma_arrive_expect_tx(full[stage], (BM*BK + BN*BK) * sizeof(half))`；Consumer 预热 `empty[i].arrive()` ×kStages → 每 tile `full[stage].wait(full[stage].arrive()); tma_fence_proxy_async_shared_cta();` … 末尾 `empty[stage].arrive()`。TMA_MMA_WS 消费者循环内**没有任何 `__syncthreads`**（1990–2011 行有 3 条理由的长注释）。
  - WGMMA 循环内每 tile：`WGMMA_FENCE();` → M 循环 2 × K 循环 4 条 `WGMMA_M64N128K16_F16F16F16(d[m], ..., 1, 1, 1, 0, 0)`（ScaleD=1 累加）→ `WGMMA_COMMIT_GROUP(); WGMMA_WAIT_GROUP(0);`。
  - CuTe：prefetch `for (istage = 0; istage < kStage - 1; ++istage) { copy(); copy(); cp_async_fence(); }` → `cp_async_wait<kStage - 2>(); __syncthreads();`；每个 k_step 内 `if (k_step == num_k_steps - 1) { cp_async_wait<kStage - 2>(); __syncthreads(); ismem_read = (ismem_read + 1) % kStage; }`；`if (k_step == 0) { ... copy(); cp_async_fence(); }`。epilogue 每个 i 步有**两个** `__syncthreads()`（R2S 之后、S2G 之后）。

- **测试与容差**（本文件是 header，测试在 `kernels/interview/notes-v2.cu`）
  - `test_hgemm_mma`（e831d970: 1366；新 HEAD: 1611）与 `test_hgemm_swizzle`（e831d970: 1441；新 HEAD: ~1694）：输入 `srand(42)` 后 `((float)rand() / RAND_MAX) * 2.0f - 1.0f` 的 half；**参考实现 = cuBLAS**，row-major 习惯写法（交换 M/N、交换 A/B）：`cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b, CUDA_R_16F, N, d_a, CUDA_R_16F, K, &beta_h, d_c, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);`，注释：`Note: use CUBLAS_COMPUTE_16F to match the kernel's f16.f16.f16.f16 accumulation.`。
  - 容差：**没有 tol/阈值**。两个 hgemm 测试只算 `float max_err`（`fabsf(__half2float(h_c[i]) - __half2float(h_c_ref[i]))`）并 `printf("| %-56s | %.3e |\n", "HGEMM MMA", max_err)` / `"HGEMM Swizzle + Reg2x"`。`max_err >= 5e-1f` 判 FAIL 的写法只出现在 flash-attn 系列测试（3201/3316/3465/3622/3784/3932 行），**hgemm 不参与**。
  - shape 列表：`main` 默认 `int M = 1024, N = 1024, K = 1024;`（可 `argv[1..3]` 覆盖），依次调用 `test_hgemm_mma(M,N,K)` / `test_hgemm_swizzle(M,N,K)` / （宏门控）`test_hgemm_cute` / `test_hgemm_wgmma` / `test_hgemm_tma_mma_ws`；`--tma-mma-ws` 专用入口默认 `M=128, N=128, K=64`（`argv[2..4]` 覆盖）。
  - launch 配置（原样）：7b-2 `constexpr int BM = 128, BN = 128, BK = 16, kStages = 3; size_t smem_bytes = kStages * (BM * BK + BN * BK) * sizeof(half); // 24576`，`dim3 block(256); dim3 grid((N + BN - 1)/BN, (M + BM - 1)/BM);`；7b-3 `constexpr int BM = 128, BN = 128, BK = 64, K_STAGE_S = 2;`（smem 64 KiB），并且必须先 `cudaFuncSetAttribute((const void *)hgemm_mma_stages_tn_swizzle<16, 8, 16, 2, 4, 4, 4, 4, K_STAGE_S, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);`。
  - bench：`--bench-hgemm` / `--bench-hgemm-all` / `--bench-all`，默认 `static int g_bench_M = 8192, g_bench_N = 8192, g_bench_K = 8192;`；`bench_hgemm_mma` 覆盖 (S=2/3) × (BLK_SW=0/1)，`bench_hgemm_swizzle` 覆盖 S=2/3 × BLK_SW=0/1，均以 cuBLAS TFLOPS 为基准。
  - bench 的 3D swizzle grid 写法（`launch_timed_hgemm_mma`，新 HEAD 2717–2733）：`const int tiles_n = (N + BN - 1) / BN; constexpr int kSwizzleN = 16; int gx = kBlockSwizzle ? div_ceil(tiles_n, kSwizzleN) : tiles_n; int gz = kBlockSwizzle ? kSwizzleN : 1; dim3 grid(gx, (M + BM - 1) / BM, gz);` —— **`kSwizzleN = 16` 个 128 列 tile = 2048 列窗口**，与 hgemm.cuh 注释里的 `S = ceil(N / 2048)` 完全等价（wrapper 里同名的常量是 `constexpr int kSwizzleStride = 2048;`）。测试函数（`test_hgemm_mma` / `test_hgemm_swizzle`）用默认 `kBlockSwizzle=0`，因此是 2D grid。
  - **swizzle 数学唯一的"真"验证**在 notes-v2.cu 的 `test_swizzle_equiv`（e831d970: 4588–4616）：`swizzle_equiv_check_one<8/16/32/64>`，`for (int i = 0; i < 256; ++i) for (int j = 0; j < kColStride; ++j)` 断言 v1 == v2，共 `256*(8+16+32+64) = 30720` 组，打印 `Swizzle v1/v2 equiv (host) | ALL PASS/FAIL`。

- **注释里的勘误/警告（原样摘录 + 我的判定）**
  1. （**注释与代码不一致**）399–401：`// 在 Phase 7b-2 的 smem XOR swizzle 基础上增加寄存器双缓冲：` —— 但 Phase 7b-2（`hgemm_mma_stages_tn`）**根本没有 smem XOR swizzle**，swizzle 首次出现就是 7b-3 自己。
  2. （**注释与代码不一致**）144–145：`原始实现会按配置决定是否给 A/B 的 K 维加 PAD；尤其 B 在 TN 布局下常见会额外加 B_PAD 来打散 bank 映射，避免按列访问时出现明显 bank conflict。这里先保留最简 PAD=0 版本。` —— 该 kernel 模板参数里**没有 PAD 形参**，"PAD=0"是"没有 PAD 机制"，不是可切换配置。
  3. （**易错点提示，值得写进教程**）174 行用 `if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;`（`||`），而 `kernels/swizzle/hgemm_mma_swizzle.cu` 的 naive kernel 用 `&&`、其 mma2x4 kernel 又用 `||`。三个版本守卫算子不统一；配合循环内 `__syncthreads()`，非整除 shape 时行为不可依赖 —— 这正是 115–116 行警告的来源：`grid.x*grid.z 可能产生 bx>=tiles_n 的冗余 CTA。当前 kernel 仍要求 M/N/K 分别按 BM/BN/BK 对齐；ceil 只保证逻辑 tile 编号不遗漏，并不使部分尾 tile 变为安全。`
  4. （**警告，CuTe 路径**）836–838 / 908：`当前 kernel 只有 CTA 级边界判断，没有 tile 内的逐元素 predication；调用方需保证 M % BM == 0、N % BN == 0、K % BK == 0，才能避免边界 tile 的越界访问或未写回。` / `当前实现使用整除，要求 K % BK == 0；没有为尾部 K tile 做 predication/padding。`
  5. （**过时数字风险，同类粘贴事故在 `kernels/swizzle/hgemm_mma_swizzle.cu` 562–566 行也有一份**）`ncu` 的 shell 片段被压成了连续注释行：`// bank conflicts free via pad = 8.` + `// ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./hgemm_mma_swizzle.bin ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_swizzle.bin constexpr int A_PAD = 8; constexpr int B_PAD = 8;`（在 `hgemm_mma_swizzle.cu` 562–566 行；同一段"粘贴事故"两处都在）。
  6. （**性能警告，原样**）1229–1230：`//   BN: F16 acc 用 256(128×256 tile,0 spill)；F32 acc 用 128(128×128 tile,0 spill)。` / `//       F32 + BN=256 会产生 ~1KB spill(254 reg),性能下降 3-8 倍。`
  7. （**HEAD 修复点**）`hgemm_tma_mma_ws_tn` 在 e831d970 是 `__global__ void __launch_bounds__(kNumThreads)`，在 6c86259 变成 `__launch_bounds__(kNumThreads, 1)`。原因写在 `common.cuh` 452–468：`Both require all warps in a warpgroup to execute the same instruction, and require __launch_bounds__(N, 1) so the compiler permits up to 256 regs/warp (otherwise the hint may have no effect).`；另外 `On sm_120a (Blackwell, CUDA 13.2) ptxas drops setmaxnreg with C7506 even when the PTX is fully inlined ... because ptxas treats cp.async.bulk.tensor (TMA) usage as an implicit extern-call boundary.`。**教程若引用 1874 行，必须写 `(kNumThreads, 1)`**。
  8. （**`__align__` 注释与代码不一致**）1505–1519 的长注释论证 `TMA + WGMMA 仅需 __align__(128)`、并以 `从健壮性角度看本 kernel 也应该用 __align__(1024)；这里使用 128 也可以正常运行。` 收尾，但紧接着的代码是 `extern __shared__ __align__(1024) uint8_t smem_tma_wgmma_ws[];` —— **实际写的是 1024**，注释末句的"这里使用 128"已过时。
  9. 1912–1919（TMA_MMA_WS 必须 1024 对齐的原因，原样关键句）：`WGMMA 用硬件 descriptor base_offset 自动解决 phase 偏移；ldmatrix 路径没有等价机制，必须靠 1024B 对齐来保证 phase=0。`
  10. 2022–2029（`swizzle<64>` vs `swizzle<16>` 的偏移位置铁律，原样）：`swizzle<64> 作用于完整 BK=64，swizzle 周期覆盖全部 64 列，k_step=0 和 k_step=1 的 chunk 会被 XOR 交叉混合 → k_step 偏移必须在 swizzle 内部参与 chunk 计算。swizzle<kMmaK> 作用于 kMmaK=16，每个 kMmaK slice 独立 swizzle，slice 之间互不跨越 → k_step * kMmaK 可以加在外部。`
  11. 284 / 521（源自 LeetCUDA swizzle 参考实现，被抄进 notes 的 epilogue）：`// TODO: how to use LDST128BITS here ? reverse the loop order ?`（在 hgemm.cuh 的 7b-2/7b-3 epilogue 里**已经**用 `*reinterpret_cast<float4 *>(...)` 做了 128-bit store，TODO 属于被超越的历史遗留）。
  12. 1836（WGMMA epilogue）：`// NOTE: 这里可以考虑通过 R->S, S->G 的方式做一次 128 bits写回。`
  13. 214：`// 此处不用pragma unroll，因为 K 不是编译期常量，因此NUM_K_TILES 也不是编译期常量`（主循环确实无 `#pragma unroll`）；1577：`// K 方向总 tile 数。要求 K 能被 BK 整除，否则尾 tile 被丢弃。`
  14. 176–180（Tile 层级名词与 cutlass 的对应，面试可直接背）：`MMA Tile对应到cutlass cute中的TiledMMA的概念，Value Tile对应到cutlass cute中的PermuteMNK的概念。`
  15. CuTe 段落的自我纠正式注释（示例，1004–1072 一带）：`这里是"复用 storage"，不是把 A Tensor 转换成 C Tensor` / `retile_S 中的 S 指"这个 TiledCopy 的 source side"，不是 shared memory` / `step=4 是 scratchpad pipeline 深度，不是 CPY_N`。以及 1070–1072 的警告：`代码未对 i+j 做尾部 predication，这个循环依赖当前静态 layout 中 grouped extent 能被 pipe depth 整除；不应将"每轮正好处理 4 个 fragment"误认为任意 tile/copy 配置都自动成立的通用规则。`

---

## kernels/swizzle/hgemm_mma_swizzle.cu

- **阶梯/版本**（4 个 kernel + 4 个 launcher）
  1. `hgemm_mma_m16n8k16_naive_kernel`（79–161）：1 warp/block，BM=16/BN=8/BK=16，单缓冲，**无 swizzle、无 pipeline**；C 经 `s_c[16][8]` 中转后用 128-bit 写回。
  2. `hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel`（164–289）：BM=BN=128，8 warps，**无 swizzle**，模板带 `A_PAD=0, B_PAD=0`。
  3. `hgemm_mma_m16n8k16_naive_smem_swizzle_kernel`（306–393）：naive + **只对 A 做 XOR swizzle**（`swizzle_A_j`），B 不 swizzle。
  4. `hgemm_mma_m16n8k16_mma2x4_warp4x4_smem_swizzle_kernel`（396–526）：mma2x4 + **只对 A 做 XOR swizzle**；B 靠 `B_PAD=8` 消冲突。
  - **无多 stage**：没有 `kStages`、没有 cp.async（`CP_ASYNC_*` 宏在文件头定义了但**全文件零调用**）；G→S 用同步 `LDST128BITS`（float4）拷贝，`k` 循环体内一次 `__syncthreads()`。

- **每级的确切配置（模板参数全列表）**
  - naive / naive_smem_swizzle：`template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16>`，`BM=MMA_M; BN=MMA_N; BK=MMA_K;`（16/8/16），block = `dim3 block(WARP_SIZE)` = 32 线程，grid = `(div_ceil(N, MMA_N), div_ceil(M, MMA_M))`。
  - mma2x4（两个版本同参）：`template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16, const int MMA_TILE_M = 2, const int MMA_TILE_N = 4, const int WARP_TILE_M = 4, const int WARP_TILE_N = 4, const int A_PAD = 0, const int B_PAD = 0>`，`__global__ void __launch_bounds__(256)`；`BM = MMA_M*MMA_TILE_M*WARP_TILE_M = 128`，`BN = MMA_N*MMA_TILE_N*WARP_TILE_N = 128`，`BK = MMA_K = 16`；`warp_m = warp_id % 2`、`warp_n = warp_id / 2`（8 warps = 256 线程，每 warp **16 个 atom**：WARP_TILE_M×WARP_TILE_N = 4×4）。
  - **命名陷阱**：这里 `MMA_TILE_M/N` 是 **warp 网格**（2×4 warps），`WARP_TILE_M/N` 是 **value repeat**（4×4）—— 与 `kernels/interview/hgemm.cuh` 的 `kMmaTileM/N`（warp 网格）＋`kValTileM/N`（value repeat）名字不同、语义对齐；教程里务必点明，否则读者会把 `WARP_TILE_*` 误读成 warp 数。
  - 文件头注释逐字：`// 128x128, mma2x4, warp4x4(64,32,16)`（warp tile = 64×32×16，即 4×16 行 × 4×8 列）。
  - launcher：`launch_hgemm_mma_m16n8k16_mma2x4_warp4x4` 内 `constexpr int A_PAD = 0; constexpr int B_PAD = 0;`，`constexpr int NUM_THREADS = (MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256`；swizzle 版本是 `template <const int B_PAD = 8> void launch_hgemm_mma_m16n8k16_mma2x4_warp4x4_smem_swizzle(...)`，内部 `constexpr int A_PAD = 0;`（**A_PAD 不可调，B_PAD 默认 8**）。

- **PTX 宏（本文件内定义，逐条）**：与 `kernels/interview/common.cuh` 同名同字符串的 `LDMATRIX_X1/X2/X4`、`LDMATRIX_X1_T/X2_T/X4_T`、`HMMA16816`、`CP_ASYNC_COMMIT_GROUP`、`CP_ASYNC_WAIT_ALL`、`CP_ASYNC_WAIT_GROUP(n)`、`CP_ASYNC_CA`、`CP_ASYNC_CG`（定义见上一节，字符串完全一致）。
  - **本文件实际发射的 PTX 只有 3 个宏**：`HMMA16816`、`LDMATRIX_X4`、`LDMATRIX_X2_T`。`CP_ASYNC_*`、`LDMATRIX_X1*`、`LDMATRIX_X4_T`、`LDMATRIX_X2`（非转置）全是**死代码**。
  - 数据搬运宏：`LDST32BITS(value)=reinterpret_cast<half2*>(&(value))[0]`、`LDST64BITS=float2*`、`LDST128BITS=float4*`。

- **fragment 与寄存器账**：naive/naive_swizzle `uint32_t RC[2] = {0, 0}; uint32_t RA[4]; uint32_t RB[2];`（2+4+2）；mma2x4 两个版本 `uint32_t RC[WARP_TILE_M][WARP_TILE_N][2]` = [4][4][2] = **32**，`uint32_t RA[WARP_TILE_M][4]` = **16**，`uint32_t RB[WARP_TILE_N][2]` = **8**。

- **ldmatrix 用法**
  - naive：A `LDMATRIX_X4(RA[0..3], __cvta_generic_to_shared(&s_a[lane_id % 16][(lane_id / 16) * 8]))`（**非 trans**）；B `LDMATRIX_X2_T(RB[0], RB[1], __cvta_generic_to_shared(&s_b[lane_id % 16][0]))`（**`.trans`**）。
  - naive_smem_swizzle：与 naive 相同，但 A 的列索引过 swizzle：`&s_a[lane_id % 16][swizzle_A_j(lane_id % 16, (lane_id / 16) * 8)]`；B 仍 `&s_b[lane_id % 16][0]` + `.trans`。
  - mma2x4（非 swizzle 版）：A `int lane_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16;`、`int lane_smem_a_k = (lane_id / 16) * 8;` → `LDMATRIX_X4`；B `int lane_smem_b_k = lane_id % 16;`、`int lane_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;` → `LDMATRIX_X2_T`。
  - mma2x4 swizzle 版：A 的 k 索引替换为 `swizzle_A_j(lane_smem_a_m, lane_smem_a_k)`；**B 的地址表达式与非 swizzle 版完全相同**（B 靠 PAD）。
  - **对比 hgemm.cuh**：这里因为 A/B 都是 row-major NN 布局（B 存成 `s_b[BK][BN]`，内维是 N），B 必须 `.trans`；hgemm.cuh 的 TN 布局把 B 存成 `B^T[N][K]`，于是 B 用**非转置** `LDMATRIX_X2`。这就是"同一个 m16n8k16 为什么一处 `.trans`、一处不 `.trans`"的答案。

- **swizzle（公式原样）**：`// i: row index; j: col index`
  `__device__ __host__ __forceinline__ int swizzle_A_j(int i, int j) { return ((int(j / 8) ^ int(i / 4)) % 2) * 8; }`
  注释里给出的展开证据（原样，四行一组）：
  `// >>> sw(0,0),sw(0,8),sw(1,0),sw(1,8),sw(2,0),sw(2,8),sw(3,0),sw(3,8)` → `// (0, 8, 0, 8, 0, 8, 0, 8)`；
  `// >>> sw(4,0),sw(4,8),...` → `// (8, 0, 8, 0, 8, 0, 8, 0)`；rows 8–11 重复 rows 0–3，rows 12–15 重复 rows 4–7。
  - **只作用于 A（BK=16 的 16 列 / 8 个 chunk）**：`j/8 ∈ {0,1}`，`(…) % 2` 保证周期 2 → 该公式**只在 col_stride = 16 时成立**。
  - **冲突度声明（原样）**：567 行附近 `// bank conflicts free via pad = 8.`（指 B_PAD），593 行 `// B_PAD = 8, bank conflicts free via pad = 8.`，launcher 非 swizzle 版 562 行 `// bank conflicts free via pad = 8.`；文件内**没有任何 ncu 数字**（数字都在 README，见下）。s_a 注释：`__shared__ half s_a[BM][BK + A_PAD]; // 128*16*2=4KB`。
  - **bank 账（可与 README 的 ncu 日志互相验证，教程建议照抄这段推导）**：ldmatrix 按 128 B/phase 取数（8 行 × 16 B 一小片 = 32 bank × 4 B）。
    - A：`s_a[128][16]` 行宽 16 half = 32 B = **8 banks**。一个 phase 的 8 行 bank 起点 `8r % 32` = 0,8,16,24,**0,8,16,24** → 2-way；套上 swizzle 后 rows 0–3 起点 0,8,16,24、rows 4–7 起点 4,12,20,28 → **32 个 bank 恰好各命中一次**（1-way）。这就是 README 里 naive `sum = 2097152` → swizzle 版 `sum = 0` 的原因。
    - B（mma2x4）：`s_b[16][128]` 行宽 128 half = 256 B = **64 banks ≡ 0 (mod 32)** → 16 行**全部落在 bank 0~3**（8 路冲突）；`B_PAD = 8` 后行宽 136 half = 272 B = **68 banks ≡ 4 (mod 32)** → 一个 phase 的 8 行起点 `4r % 32` = 0,4,8,…,28 → 免冲突。**B 的行宽是 BN（128）而不是 8，所以它必须靠 PAD 而不能靠那个只对 16 列成立的 `swizzle_A_j`** —— 这是"swizzle 与 padding 为什么在同一份代码里同时出现"的确切答案。
    - 对照 naive kernel 的 `s_b[16][8]`：行宽 8 half = 16 B = 4 banks → 8 行起点 0,4,…,28 天然铺满 32 banks，**既不需要 swizzle 也不需要 PAD**（所以 `naive_smem_swizzle` 只 swizzle A 就能拿到 0 冲突）。

- **同步与流水**：单 stage、`#pragma unroll for (int k = 0; k < NUM_K_TILES; ++k)`；顺序为 `s_b` 存 → `s_a` 存 → `__syncthreads()` → ldmatrix ×(4+4) → HMMA ×16 → `__syncthreads()`。**没有 cp.async、没有 wait_group、没有 double buffer**（`NUM_K_TILES = div_ceil(K, MMA_K)` 每次只搬 16 列）。naive 版多两步：`s_c` 写回 + `__syncthreads()` + `if (lane_id < MMA_M) LDST128BITS(C[...]) = LDST128BITS(s_c[lane_id][0]);`。

- **测试与容差**：**没有正确性校验、没有 cuBLAS、没有 tol**。`main` 用 `perf_gemm<half>(...)` 只测时间/TFLOPS；`M=1024,N=1024,K=1024,W=1,R=10`（`argv[1..5]` 覆盖），`avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec`。计时的 4 个 ALGO 依次是：`HGEMM MMA NAIVE`、`HGEMM MMA NAIVE + SMEM SWIZZLE`、`HGEMM mma2x4_warp4x4`、`HGEMM mma2x4_warp4x4 + A SMEM SWIZZLE + B_PAD 0`、`HGEMM mma2x4_warp4x4 + A SMEM SWIZZLE + B_PAD 8`（共 5 条 printf 标签，对应 5 次 `perf_gemm`）。`perf_gemm` 里 `cudaMalloc(&d_a,...)` 后**从不初始化数据**（纯性能测试）。

- **注释里的勘误/警告**
  1. 304 行残留 TODO：`// TODO: hgemm_mma_m16n8k16_naive_smem_swizzle_kernel` 就贴在已实现好的 kernel 定义上方（漏删）。
  2. 562–566 行的注释块是一段**粘贴事故**（先写的 pad=8 结论 + 两条 ncu 命令 + 两行 `constexpr` 全被压成注释），下一行才是真实代码 `constexpr int A_PAD = 0; constexpr int B_PAD = 0;` —— **注释说 pad=8 才免冲突，代码里非 swizzle launcher 用的是 0**。
  3. 179 / 412 行共享内存注释**数字错误**：`__shared__ half s_b[BK][BN + B_PAD]; // 16*128*2=4KB, 16*(128+16)*2=4.5KB` —— B_PAD=8 时实际是 `16*(128+8)*2 = 4352 B = 4.25 KiB`，注释里的 4.5KB 既不是 pad=8 也不是 pad=16 的正确值（系旧 PAD=16 时代的过时数字）。
  4. 105 行 `if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;`（`&&`）vs 203/436 行 `if (load_gmem_a_m >= M || load_gmem_b_n >= N) return;`（`||`）—— 同一文件内两种守卫。
  5. 284 / 521：`// TODO: how to use LDST128BITS here ? reverse the loop order ?`（C 写回仅 32-bit）。
  6. 78 行 `// only 1 warp per block(32 threads), m16n8k16. A, B, C: all row_major.`；163 行 `// 128x128, mma2x4, warp4x4(64,32,16)`。
  7. 76 行 `div_ceil` 是**正确版本**：`return (a % b != 0) ? (a / b + 1) : (a / b);`（对照 `mma_simple_swizzle.cu` 的 `+ 8`）。

---

## kernels/swizzle/mma_simple_swizzle.cu

- **阶梯/版本**：只有一个 kernel —— `mma_simple_swizzle_kernel<MMA_M=16, MMA_N=8, MMA_K=16>`（91–190）。1 warp/block（`dim3 block(WARP_SIZE)`），`BM=MMA_M=16`、`BN=MMA_N=8`、`BK=MMA_K=16`，`grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M))`。定位是**教学/打印版**：kernel 内部用 `printf` 把 `s_a`/`s_b` 全量打印出来（`if (tid == 0)`），并把原始的非 swizzle 存法注释保留在原处（126–127 行）。
- **每级的确切配置**：模板参数 `const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16`（无其它默认参数、无 PAD、无 stage）；`constexpr int BM = MMA_M; // 16`、`constexpr int BN = MMA_N; // 8`、`constexpr int BK = MMA_K; // 16`；主机端 `constexpr int MMA_M = 16; MMA_N = 8; MMA_K = 16; dim3 block(WARP_SIZE); dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));`。加载映射：`load_smem_a_m = tid / 2`（0~15 行）、`load_smem_a_k = (tid % 2) * 8`（列 0/8）、`load_smem_b_k = tid`（只用到 0~15）、`load_smem_b_n = 0`。
- **PTX 宏**：文件头定义与 `hgemm_mma_swizzle.cu` **完全相同的一整套**（`CP_ASYNC_COMMIT_GROUP/WAIT_ALL/WAIT_GROUP/CA/CG`、`LDMATRIX_X1/X2/X4` + 三个 `_T` 变体、`HMMA16816`，字符串逐字一致）。**实际只用 3 个**：`LDMATRIX_X4`（A）、`LDMATRIX_X2_T`（B，`.trans`）、`HMMA16816`。
- **fragment 与寄存器账**：`uint32_t RC[2] = {0, 0};`（累加器 2 个）、`uint32_t RA[4]; uint32_t RB[2];`（循环内声明）；C 写回用 `LDST32BITS`（half2）：`store_lane_gmem_c_m = by * BM + lane_id / 4; store_lane_gmem_c_n = bx * BN + (lane_id % 4) * 2;`。
- **ldmatrix 用法**：A → `swizzle_j` 后的地址 `uint32_t load_smem_a_ptr = __cvta_generic_to_shared(&s_a[lane_id % 16][swizzle_j(lane_id % 16, (lane_id / 16) * 8)]);` + `LDMATRIX_X4`；B → `uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_id % 16][0]);` + `LDMATRIX_X2_T`。**x4 非 trans / x2 trans**；x1 未使用。
- **swizzle（公式原样，与 hgemm_mma_swizzle.cu 逐字相同的函数体）**：
  `__device__ __host__ __forceinline__ int swizzle_j(int i, int j) { return ((int(j / 8) ^ int(i / 4)) % 2) * 8; }`
  注释给出的 4 组展开证据与 `hgemm_mma_swizzle.cu::swizzle_A_j` 完全一致（`(0, 8, 0, 8, ...)` / `(8, 0, 8, 0, ...)` 交替）。
  - **pad 结论**：`__shared__ half s_a[MMA_M][MMA_K]; // 16x16`、`__shared__ half s_b[MMA_K][MMA_N]; // 16x8` —— **无 padding**，靠 swizzle 免冲突；**只 swizzle 了 A，B 完全没处理**（本文件未讨论 B：B 行宽 8 half = 16 B = 4 banks，一个 128 B phase 的 8 行起点 `4r % 32` = 0,4,…,28 恰好铺满 32 banks，所以它本来就不需要处理）。
- **同步与流水**：单 stage、无 cp.async；k 循环内 A 存 → `if (lane_id < MMA_K)` 的 B 存 → `__syncthreads()`；随后**为 printf 打印 s_a 再插一次 `__syncthreads()`（139/148/159 行共 3 次）**，把打印和后续 ss 交替起来；HMMA 后再 `__syncthreads()`（177 行）。`NUM_K_TILES = div_ceil(K, MMA_K)`，但 div_ceil 有 bug（见下）。
- **测试与容差**：**没有任何校验**。主机端 `M=16, N=8, K=16`（`if (argc > 8) M = std::stoi(argv[1]);` / `if (argc > 2) N = ...` / `if (argc > 3) K = ...`），`h_a[i] = __float2half((float)i)`（0~255）、`h_b[i] = __float2half((float)i)`（0~127），只做 H2D，**不回拷 d_c、不比对、不释放 h_c 的内容使用**；`h_c = (half *)malloc(size_c)` 分配后从未写入/检查（连 `cudaMemcpy` D2H 都没有）。
- **注释里的勘误/警告**
  1. （**真 BUG**）76 行：`int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 8) : (a / b); }` —— 应为 `+ 1`，这里写成 **`+ 8`**（同一目录的 `hgemm_mma_swizzle.cu` 第 76 行是正确的 `+ 1`）。后果：K 非 MMA_K 整数倍时 `NUM_K_TILES` 被算大约 8 倍（K=17：`17/16+8 = 9` 而不是 2），k 循环会越界读 A/B。教程引用时不要照抄这个 `div_ceil`。
  2. （**真 BUG**）196 行：`if (argc > 8) M = std::stoi(argv[1]);` —— 判断写成 `argc > 8`，命令行传 M 也进不去（其它两个分支正常 `argc > 2` / `argc > 3`）。
  3. 107–113 行的中英混注释解释"为什么 32 线程够"：`// s_a[16][16], 每行16，每线程load 8，需要2线程，共16行，需2x16=32线程`、`// s_b[16][8], 每行8，每线程load` `// 8，需要1线程，共16行，需16线程，只需一半线程加载`（对应 `if (lane_id < MMA_K)` 只用前 16 线程）。
  4. 164–165：`// ldmatrix for s_a, ldmatrix.trans for s_b.` / `// s_a: (0,8) *8 -> 0,8 -> [(0~15),(0,8)]` —— 括号里第一个 `(0,8)` 是 `(lane_id/16)*8` 的取值，注释排版有歧义（hgemm_mma_swizzle.cu 同一处写的是 `// s_a: (0,1)*8 -> 0,8 -> [(0~15),(0,8)]`，更清晰）。
  5. 116 行守卫是 `&&` 版本：`if (load_gmem_a_m >= M && load_gmem_b_n >= N) return;`。
  6. 180–183 的 C fragment 注释只留了 PTX 文档链接两行 + `// [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]`，没有像 hgemm.cuh 那样画出完整线程映射表。

---

## kernels/swizzle/print_swizzle_layout.py

- **它验证/演示什么**：**不是测试脚本，而是 swizzle 布局的"可视化验证器/教学打印器"**。它把 `swizzle_permuted_j` 的 XOR 结果映射回 32 个 smem bank（4 B/bank），逐行打印"逻辑列 → 物理 smem 列 + 该访问落在哪几个 bank"，让读者肉眼确认同一个 warp 内 8×8 ldmatrix 的行访问是否均匀铺满 32 banks。**脚本内没有任何 assert/比对/冲突计数**（唯一的 assert 是参数合法性 `assert smem_pading == 0 or smem_pading == 8, "smem_pading must be 0 or 8"`）。
- **核心公式（逐字）**：
  `def swizzle_permuted_j(i: int, j: int, col_stride: int = 16, num_elems_per_128b: int = 8):`
  `    return ((int(j / num_elems_per_128b) ^ int(i / 4)) % (int(col_stride / num_elems_per_128b))) * num_elems_per_128b`
  注释原文：`# i: row index; j: col index. col_stride <= 16.` / `# assert col_stride <= 16, f"col_stride must <= 16, but got {col_stride}"`（**这行 assert 被注释掉了**）/ `# for col_stride > 16, we have to permute it using col major ZigZag order.` / `# e.g, Q smem logical layout [Br,d]=[Br,64] -> store layout [4][Br][16].`
  对默认参数（col_stride=16, num_elems_per_128b=8）：`((j/8) ^ (i/4)) % 2 * 8` —— **与 `mma_simple_swizzle.cu::swizzle_j` / `hgemm_mma_swizzle.cu::swizzle_A_j` 是同一个式子**（这两个 .cu 里是 `% 2` 硬编码版）。
- **输出什么**（`print_smem_swizzle_layout`）：先无条件打印 `PERMUTED_DOCS_STRING`（6 行 INFO：`Assert smem store layout col_stride <= 16, prefer 16.` / `For logical_col_stride > 16, we have to permute the smem store layout using col major ZigZag method:` / `e.g, --> Q smem logical layout [Br][64].` / `--> col major ZigZag permuted -->` / `--> Q smem store layout [4][Br][16].`）；然后对 `rows` 每一行打印两条：`|bank  |b 0~3 |b 4~7 |...` 和 `|row i|  0   |  8   |...`（`--show-logical-col` 时格子写成 `逻辑列:swizzle列`，如 `0:0`、`8:8`），每 4 行一条分隔线，表头 3 行是 `swizzle layout` / `logical col 0~{logical_col_stride}, step {num_elems_per_128b}` / `smem col 0~16, step {num_elems_per_128b}`（`logical_col_stride < 16` 时写 `smem col 0~8`）。
  - bank 推算：`banks_per_col = int((16 * 2) / 4) if logical_col_stride >= 16 else 4`（= 8 banks/行）；`banks_per_num_elems_per_128b = int((num_elems_per_128b * 2) / 4)`（= 4）；`--smem-padding 8` 时 `banks_per_col += 4` 并打印 `[INFO] smem padding 8 half values, 4 banks, banks_per_col: {banks_per_col}`；`--use-logical-col-stride` 时改用 `int((logical_col_stride * 2) / 4)` 并在 >16 时打印 `[WARN] col_stride must <= 16, but got {logical_col_stride}`。
  - `logical_col_stride >= 16 && !use_logical_col_stride` 时走 **ZigZag 路径**：每 16 列为一组、组内 2 个 chunk 都调用 `swizzle_permuted_j(i, j, 16, 8)`，逻辑列号按 `k*16 + j` 续编 —— 即"逻辑 [Br][64] → 物理 [4][Br][16]"的分组展开。
  - `--smem-padding 8` 时每行额外插一个 `pad` 格子（`if smem_pading == 8 and (r > 1 and r % 2 == 0)`）。
- **覆盖多少组参数**：
  - CLI：`--rows`（默认 16）、`--smem-padding/--pad`（0，仅允许 0/8）、`--num-elems-per-128b/--num-elems`（8）、`--logical-col-stride/--logical-col/--col`（64）、`--use-logical-col-stride/--use-logical-col`、`--show-logical-col-id/--show-logical-col`。**脚本自身不做参数扫描（没有 for 循环遍历配置）**，一次运行只打一张用户指定的表；覆盖量 = `rows` × `ceil(logical_col_stride/16)` 个 16 宽组 × `16/num_elems_per_128b` 个 chunk。
  - 默认（rows=16, logical_col=64, num_elems=8）：**16 行 × 4 组 × 2 chunk = 128 个 (i, 逻辑列) 采样点**；每个 16 宽组内 swizzle 的 XOR 周期是 4 行（`i/4 % 2`），16 行 = 4 个周期。`swizzle_permuted_j` 的输入域其实覆盖 `i` 任意大、`j < col_stride`。
- **输出样例（README 记录的两段，逐字摘几行；教程配图可直接用）**
  - M16K16（README 那段实际是 logical_col=16 的输出）：`|bank  |b 0~3 |b 4~7 |` / `|row 0 | 0:0  | 8:8  |` / … / `|row 4 | 0:8  | 8:0  |`（每 4 行翻转一次），表头 `logical col 0~16, step 8` + `smem col 0~16, step 8-`。
  - M16K64（Zigzag）：每行 8 个 chunk、覆盖 4 个 16 宽组：`|row 0 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |`、`|row 4 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |`，表头 `logical col 0~64, step 8` + `smem col 0~16, step 8`。
  - 读法：格子是 `逻辑列:swizzle后的物理列`；每个 16 宽组内部的物理列只会是 0 或 8，即"逻辑列 0~63 被折叠进 16 宽的物理窗口，折叠规则 = 组内 `(chunk ^ (row/4)) % 2`"。
- **勘误/警告**
  1. `banks_str_len` 初始化 0 后**从未赋值**，`str_len = max(layout_str_len, banks_str_len)` 实际恒等于 `len(smem_layout_str)`（死变量）。而且 `str_len` 只在 `i == 0` 时算一次，后面所有行共用。
  2. 表头在宽度不足时会被 `pretty_print_line` 静默截断/错位：`res_len = width - len(m)` 允许为负（`left_len = int(res_len/2)`、`right_len = res_len - left_len`，负数时 `"-" * 负数 == ""`），README 第一个例子里 `smem col 0~16, step 8-` 那个多余的尾巴 `-` 就是这么来的。
  3. 变量名 / 参数名的拼写是 `smem_pading`（少一个 d），对外的 CLI 才是 `--smem-padding`。
  4. 顶部注释块（48–96 行）**硬编码了一张 rows=16 / logical col 64 的样例表**，与函数默认参数（`logical_col_stride=16`）不一致；`swizzle_permuted_j` 里 `col_stride <= 16` 的 assert 被注释掉，只剩 docstring 说明。

---

## kernels/swizzle/README.md（197 行）

- **验证/主张**：`📖 Learn how to apply SMEM Swizzle for bank conflicts free`；ncu 章节标题即结论：`📚 Achieve 0 bank conflicts for LDSM via smem swizzle.`；日志小节标题：`📚 log: (achieve 0 bank conflicts for LDSM via smem swizzle)`。
- **构建**：`make # build all default binaries`（对应 makefile 的 `default` 目标：`hgemm_mma_swizzle.bin`、`mat_trans_swizzle.bin`、`mma_simple_swizzle.bin`，sm_80+sm_89）。
- **ncu 度量命令（原样）**：
  `ncu --metrics l1tex__data_bank_reads ./mat_trans_swizzle.bin`、`... l1tex__data_bank_writes ...`、`... l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./mat_trans_swizzle.bin`、`... _op_st ...`、
  `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1`、
  `ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1`。
- **ncu 日志里的确切数字（`sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm`，CC 8.9）**
  - `hgemm_mma_m16n8k16_naive_kernel<16, 8, 16>`，`(128, 64, 1)x(32, 1, 1)`：avg **22795.13**、max **24576**、min **18432**、sum **2097152**。
  - `hgemm_mma_m16n8k16_naive_smem_swizzle_kernel<16, 8, 16>`，`(128, 64, 1)x(32, 1, 1)`：avg/max/min/sum 全 **0**。
  - `hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<16, 8, 16, 2, 4, 4, 4, 0, 0>`，`(8, 8, 1)x(256, 1, 1)`：avg **25644.52**、max **36864**、min **0**、sum **2359296**。
  - `hgemm_mma_m16n8k16_mma2x4_warp4x4_smem_swizzle_kernel<16, 8, 16, 2, 4, 4, 4, 0, 8>`，`(8, 8, 1)x(256, 1, 1)`：全 **0**。
  - 注意：模板实参列表 `<16, 8, 16, 2, 4, 4, 4, 0, 8>` 与源码顺序 `MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, A_PAD, B_PAD` 一致，即**免冲突的那次是 A_PAD=0 + B_PAD=8**（A 靠 swizzle、B 靠 pad）。
- **性能（NVIDIA RTX 3080 Laptop，原样）**：`./hgemm_mma_swizzle.bin 4096 4096 4096 1 10` →
  `ALGO = HGEMM mma2x4_warp4x4`：`M N K =   4096   4096   4096, W = 1, R = 10, Time =   0.00392888 s, AVG Performance =    34.9817 Tflops`；
  `ALGO = HGEMM mma2x4_warp4x4 + SMEM SWIZZLE`：`Time =   0.00234496 s, AVG Performance =    58.6104 Tflops`（≈ 1.68×）。
- **print swizzle layout 两段示例**
  - 两段的命令行**完全相同**：`python3 print_swizzle_layout.py --logical-col 64 --show-logical-col`，标题分别是 `📚 M16K16` 和 `📚 M16K64 (Zigzag)`。
  - `M16K16` 段落输出 16 行、每行只有 2 个 chunk（`|row 0 | 0:0  | 8:8  |`，rows 4–7 为 `0:8 | 8:0`），表头是 `logical col 0~16, step 8` / `smem col 0~16, step 8-`。
  - `M16K64` 段落每行 8 个 chunk（`|row 0 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |`，rows 4–7 每 16 列一块地翻转），表头 `logical col 0~64, step 8` / `smem col 0~16, step 8`。
- **勘误/警告（README 自身的问题）**
  1. （**命令与输出不符 / 过时输出**）`M16K16` 那段贴的命令是 `--logical-col 64`，但输出表头是 `logical col 0~16`、每行仅 2 个 chunk —— 该输出实际对应 `logical_col_stride=16`（或旧版脚本），**不是 `--logical-col 64` 的结果**，两段示例其实是"同一命令贴了两份不同版本的输出"。
  2. `smem col 0~16, step 8-` 末尾那个多余的 `-` 是 `pretty_print_line` 在 `width < len(m)` 时的排版残留（负 `res_len`），不是数据。
  3. ncu 段落只给了 `_op_ldsm` 一个指标的四条记录，`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`（README 第一组命令里的）**没有任何日志**；而结论句写作 "0 bank conflicts for LDSM"（限定在 LDSM 通道，不等于所有 shared 访问 0 冲突）。
  4. `naive_smem_swizzle` 之所以 0 冲突，是因为它**只 swizzle 了 A**：它的 B 是 `s_b[16][8]`（行宽 16 B = 4 banks，一个 128 B phase 的 8 行恰好铺满 32 banks），天然免冲突；而 mma2x4 版的 B 行宽是 BN=128 half（256 B ≡ 0 mod 128 banks）必须靠 `B_PAD=8` 才免冲突。README 未说明这一点，读者容易误以为"swizzle 了全部"或"PAD 只是保险"。
  5. 性能数字是 **RTX 3080 Laptop（GA104，sm_86）**，且 `W = 1, R = 10`（warmup 只有 1 次），不能当作数据中心 GPU 结论；README 未标注 sm 版本与是否含 L2 flush。

---

## 跨文件对照速查（每格都来自上述 5 个文件的逐字事实）

- **布局 / B 的 ldmatrix / A 的 ldmatrix**
  - `hgemm.cuh`：TN（A row-major `[M,K]`，B^T row-major `[N,K]`，smem `s_a[BM][BK]` / `s_b[BN][BK]`）→ A `LDMATRIX_X4`、B `LDMATRIX_X2`（**非 trans**）。
  - `hgemm_mma_swizzle.cu`：NN（A `[M,K]` row-major，B `[K,N]` row-major，smem `s_a[BM][BK]` / `s_b[BK][BN]`）→ A `LDMATRIX_X4`、B `LDMATRIX_X2_T`（**trans**）。
  - `mma_simple_swizzle.cu`：同 NN（`s_a[16][16]`、`s_b[16][8]`）→ A `LDMATRIX_X4`、B `LDMATRIX_X2_T`（**trans**）。
  - 三个文件都**没有用 x1**；`.trans` 的取舍完全由"B 在 smem 里是 B^T[N][K] 还是 B[K][N]"决定。
- **swizzle 表达式与适用宽度**
  - `hgemm.cuh`（`common.cuh::permuted`）：`kColStride=16` → `(((j >> 3) ^ (i >> 2)) & 1) << 3`（SWIZZLE_32B，BK=16 时用）；`kColStride=64` → `(chunk ^ xor_mask) << 3`，`chunk=(j>>3)&7`、`xor_mask=(i&1)|(((i>>1)&1)<<1)|(((i>>2)&1)<<2)`（SWIZZLE_128B，TMA_MMA_WS 用）；CuTe 侧 `Swizzle<3,3,3>`。
  - `hgemm_mma_swizzle.cu` / `mma_simple_swizzle.cu`：`((int(j / 8) ^ int(i / 4)) % 2) * 8`，**只对 A 用、只对 col_stride=16 成立**；`print_swizzle_layout.py` 的通用式 `((j/8 ^ i/4) % (col_stride/8)) * 8` 在 col_stride=16 时退化为它。
- **免冲突手段与实测证据**
  - `hgemm.cuh`：7b-2 无任何手段；7b-3 用 `swizzle<kMmaK>`；TMA_MMA_WS 用 `swizzle<64>` + `__align__(1024)`；CuTe 用 `Swizzle<3,3,3>`。**文件内没有 ncu 数字**。
  - `hgemm_mma_swizzle.cu`：A 用 swizzle、B 用 `B_PAD=8`；README 给出实测：naive `sum=2097152` → naive+swizzle `sum=0`；`mma2x4_warp4x4<...,0,0>` `sum=2359296` → `mma2x4_warp4x4_smem_swizzle<...,0,8>` `sum=0`。
  - `mma_simple_swizzle.cu`：A 用 swizzle，B 靠 16 B 行宽天然免冲突；无 ncu 数字。
- **流水结构**
  - `hgemm.cuh`：7b-2 cp.async kStages=3（wait 参数 1/0）；7b-3 cp.async kStages=2 + 寄存器双缓冲（`RA[2][4][4]`/`RB[2][4][2]`，kValTileK=4）；CuTe `cp_async_wait<kStage-2>()` + 每 k_step 显式切换；WGMMA/TMA_MMA_WS 用 `full[]/empty[]` mbarrier（arrive_count=129）。
  - `hgemm_mma_swizzle.cu` / `mma_simple_swizzle.cu`：**完全无流水、无 cp.async**（虽然宏都定义了），G→S 是同步 `LDST128BITS` + `__syncthreads()`，单缓冲。
- **测试与容差**
  - `hgemm.cuh`：`notes-v2.cu` 用 cuBLAS `CUBLAS_COMPUTE_16F` 作参考，只打印 `max_err`，**无阈值**；shape 默认 `1024×1024×1024`（`--tma-mma-ws` 默认 `128×128×64`）；bench 默认 `8192³`。
  - `hgemm_mma_swizzle.cu`：只有 `perf_gemm` 计时（`M=N=K=1024, W=1, R=10` 默认），**无正确性校验**。
  - `mma_simple_swizzle.cu`：kernel 内 printf 打印 smem 内容，**不回拷 C、不比对**；`M=16, N=8, K=16` 默认。
- **面试可直接用的三句话（均由上述事实支撑）**
  1. `swizzle(i, j) = ((j/8) ^ (i/4)) % 2 * 8` 只在 col_stride=16（SWIZZLE_32B）成立；BK=64 时要么用 `swizzle<64>`（SWIZZLE_128B，`k_step*kMmaK` 必须放进 swizzle 内部），要么把 `swizzle<16>` 作用在每个 16 列 slice 上（`k_step*kMmaK` 加在 swizzle 外部）。
  2. TN 布局把 B 存成 B^T[N][K] row-major，B 的 ldmatrix 因此**不需要 `.trans`**，与 `mma.sync.aligned.m16n8k16.row.col` 天然对齐；NN 布局（B 存成 `s_b[BK][BN]`）必须 `.trans`。
  3. 免 bank conflict 有两条正交路径：XOR swizzle（不浪费 smem，但依赖 16 列 chunk 结构）与 PAD（`B_PAD=8` → 行宽 136 half = 272 B ≡ 4 banks (mod 32)，一个 128 B phase 的 8 行恰好铺满 32 banks）；同一份 kernel 里 A 走前者、B 走后者，README 的 ncu 0 冲突就是这个组合的结果。


---

# E · FlashAttention / WGMMA+WS / common.cuh（Q187 Q189 Q190）

# E — LeetCUDA 参考实现精确事实提取（FlashAttention + TMA/WS + 公共基础设施）

- **REPO**: `C:\Users\Jeff\Documents\GitHub\LeetCUDA`
- **任务给定 HEAD**: `e831d970a099f5ce8fd0495ddd1df09206d56918`
- **提取期间仓库实际 HEAD 已前移**（外部操作，非本 agent 所为）：
  会话开始 `e831d97` → 会话中途变为 **`6c86259d7eca5e6b34fcd4bd21e4ebe1c890abc2`**（`Fix link text for LeetCUDA PDF in README`，2026-09-22 23:26:09 +0800，工作区 clean）。
  期间 `kernels/interview/{flash_attn.cuh, notes-v2.cu, common.cuh, build.sh, sgemm.cuh, ffpa_attn.cuh, hgemm.cuh}`
  都被改动过（`git diff --stat e831d97 6c86259`：flash_attn.cuh +689、notes-v2.cu +356、build.sh +89、common.cuh +66、ffpa_attn.cuh +4）。
  **本文件所有事实已在当前 HEAD `6c86259` 上逐条复核**（关键：`flash_attn.cuh` 中段内容未变，L29/L65/L68/L808/L838/L949/L962/L1081/L1501/L1503 等引用行均命中原文）。
- **实测物理行数 @ `6c86259`**（`(Get-Content file).Count`，空行计入）：
  `flash_attn.cuh` **4179**、`notes-v2.cu` **5219**、`base.cuh` **909**、`common.cuh` **803**、
  `ffpa_attn.cuh` **641**、`build.sh` **245**、`README.md` **55**、`ws-hgemm/naive_ws_hgemm_sm8x.cu` **486**。
  （任务描述给的是 `3245 / 4465 / 815 / 719 / 584 / 164 / 61 / 385` —— 只有 `base.cuh` 与
  `naive_ws_hgemm_sm8x.cu` 对得上；其余偏小，应为 `e831d97` 之前/左右的旧快照。写正文一律以实测为准。）
- 本文件只做**事实转录**，不含推测。所有模板参数、宏展开、PTX 字符串、tol 数值均按原文照抄。
- **本任务不涉及 TF32 内核**：`interview/` 下不存在 `NOTES_V2_ENABLE_TF32`，也不存在 TF32 SGEMM kernel
  （`book/tests/ch10_sgemm_tf32.cu` 是独立的教学测试，不属于本任务的 8 个文件）。

---

## kernels/interview/common.cuh

- **文件定位**：整个 `interview/` 章节的最底层公共模块——CUDA 头、基础类型宏、MMA/WGMMA PTX 宏、
  XOR swizzle 函数、TMA/mbarrier helper、host 端 TensorMap helper。被 `base.cuh` 及所有上层 `.cuh` 依赖。
- **导出的符号 = 模板参数全列表**（逐条，精确名字 + 模板参数默认值；本文件只有工具层，无 kernel）
  - 常量：`static constexpr int kWarpSize = 32;`
  - 宏：`INT4(value)` / `FLOAT4(value)` / `HALF2(value)` / `HOST_DEVICE_INLINE`
  - 宏：`CP_ASYNC_COMMIT_GROUP()` / `CP_ASYNC_WAIT_ALL()` / `CP_ASYNC_WAIT_GROUP(n)` / `CP_ASYNC_CG(dst, src, bytes)`
  - 宏：`LDMATRIX_X4(R0,R1,R2,R3,addr)` / `LDMATRIX_X2(R0,R1,addr)` / `LDMATRIX_X2_T(R0,R1,addr)`
  - 宏：`HMMA16816(RD0,RD1,RA0..RA3,RB0,RB1,RC0,RC1)` / `HMMA16816F32(RD0..RD3,RA0..RA3,RB0,RB1,RC0..RC3)`
  - 宏（`NOTES_V2_ENABLE_WGMMA` 门控）：`WGMMA_FENCE()` / `WGMMA_COMMIT_GROUP()` / `WGMMA_WAIT_GROUP(n)` /
    `SMEM_DESC_ENCODE(x)` / `WGMMA_M64N128K16_F16F16F16(d, sA, sB, ScaleD, ScaleA, ScaleB, TransA, ...)`
  - 宏（`NOTES_V2_ENABLE_SETMAXNREGS` 门控）：`NOTES_V2_REG_DEALLOC(N)` → `warpgroup_reg_dealloc<N>()`；
    `NOTES_V2_REG_ALLOC(N)` → `warpgroup_reg_alloc<N>()`；**未定义该宏时两者都展开成 `((void)0)`**
  - `template <const int kColStride = 16, const int kStep = 8> static __host__ __device__ __forceinline__ int permuted(int i, int j)`
  - `template <const int kColStride = 16> static __host__ __device__ __forceinline__ int swizzle_v1_impl(int i, int j)`
  - `template <int B, int M, int S> struct SwizzleBMS`
  - `template <const int kColStride = 16> static __host__ __device__ __forceinline__ int swizzle_v2_impl(int i, int j)`
  - 派发器 `swizzle<kColStride>(i,j)`：未定义 `NOTES_V2_ENABLE_SWIZZLE_V2` → v1，定义 → v2
  - `static __device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(void const *ptr)`
  - `static __device__ __forceinline__ void tma_fence_proxy_async_shared_cta()`
  - `static __device__ __forceinline__ void tma_load_2d(void *dst, const CUtensorMap *tensor_map, int minor_coord, int major_coord, cuda::barrier<cuda::thread_scope_block> &barrier)`
  - `static __device__ __forceinline__ void tma_arrive_expect_tx(cuda::barrier<cuda::thread_scope_block> &barrier, uint32_t bytes)`
  - `template <uint32_t kNumRegs> __device__ __forceinline__ void warpgroup_reg_dealloc()` / `warpgroup_reg_alloc()`
  - `template <int BM, int BN, int BK, int QSIZE> struct WgmmaSMem`
    （成员：`alignas(128) half A[BM*BK*QSIZE]; alignas(128) half B[BN*BK*QSIZE];`）
  - `template <int BM, int BN, int BK, int QSIZE> struct TmaMmaWSSMem`
    （`static_assert(BK == 64, "The 128B swizzle helper below is specialized for BK=64");`
    成员：`half A[BM*BK*QSIZE]; half B[BN*BK*QSIZE];`——**注意 A/B 没有 alignas**）
  - `template <int BlockMajorSize, int BlockMinorSize> __host__ static inline void create_tensor_map(CUtensorMap *tma_map, half *gmem_ptr, int blocks_height, int blocks_width)`
  - `template <int BlockMajorSize = 128, int BlockMinorSize = 64> __host__ static inline CUtensorMap *allocate_and_create_tensor_map(half *src, int blocks_height, int blocks_width)`
  - `tma_load_2d` / `tma_fence_proxy_async_shared_cta` / `tma_arrive_expect_tx` 有**两套实现**，由
    `NOTES_V2_FORCE_INLINE_ASYNC_PROXY` 门控：定义 → 裸 `asm volatile` PTX；未定义 → `cuda::ptx::` / `cuda::device::` C++ 包装
    （后者在 sm_120a 上会让 ptxas 以 C7506 丢弃 `setmaxnreg`）。
- **PTX 宏与工具宏**（宏名 + 完整展开）
  - `INT4(value)` → `(reinterpret_cast<int4 *>(&(value))[0])`；`FLOAT4` / `HALF2` 同构（float4 / half2）
  - `CP_ASYNC_COMMIT_GROUP()` → `asm volatile("cp.async.commit_group;\n" ::)`
  - `CP_ASYNC_WAIT_ALL()` → `asm volatile("cp.async.wait_all;\n" ::)`
  - `CP_ASYNC_WAIT_GROUP(n)` → `asm volatile("cp.async.wait_group %0;\n" ::"n"(n))`
  - `CP_ASYNC_CG(dst, src, bytes)` →
    `asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))`
    （注释原文：`注意：cg 只支持 16 bytes，ca 支持 4/8/16 bytes`）
  - `LDMATRIX_X4(R0,R1,R2,R3,addr)` → `ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];`
  - `LDMATRIX_X2(R0,R1,addr)` → `ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];`
  - `LDMATRIX_X2_T(R0,R1,addr)` → `ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];`
  - `HMMA16816(...)` → `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};`
  - `HMMA16816F32(...)` → `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};`
  - `WGMMA_FENCE()` → `wgmma.fence.sync.aligned;`；`WGMMA_COMMIT_GROUP()` → `wgmma.commit_group.sync.aligned;`；
    `WGMMA_WAIT_GROUP(n)` → `wgmma.wait_group.sync.aligned %0;`
  - `SMEM_DESC_ENCODE(x)` → `((((uint64_t)(x)) & 0x3FFFF) >> 0x4)`
  - `tma_load_2d` 内 PTX → `"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"`
  - `tma_arrive_expect_tx` 内 PTX → `"mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"`
  - `tma_fence_proxy_async_shared_cta()` 内 PTX → `"fence.proxy.async.shared::cta;\n"`
  - **注意：不存在名为 `SWIZZLE_*` 的宏**。`SWIZZLE_32B/64B/128B` 只出现在**注释**里，是
    `permuted<kColStride>` 的别名说明：`kColStride=16 fp16=32B/row → Swizzle<1,4,3> = SWIZZLE_32B`；
    `=32 → Swizzle<2,4,3> = SWIZZLE_64B`；`=64 → Swizzle<3,4,3> = SWIZZLE_128B`。
  - swizzle 位级公式（`permuted` 原文）：`kStep==4` → `(((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2`；
    `kColStride==16` → `(((j >> 3) ^ (i >> 2)) & 1) << 3;  // SWIZZLE_32B`；
    `kColStride==32` → `chunk=(j>>3)&3; xor_mask=((i>>1)&1)|(((i>>2)&1)<<1); return (chunk ^ xor_mask) << 3;  // SWIZZLE_64B`；
    `kColStride==64` → `chunk=(j>>3)&7; xor_mask=(i&1)|(((i>>1)&1)<<1)|(((i>>2)&1)<<2); return (chunk ^ xor_mask) << 3;  // SWIZZLE_128B`。
    `swizzle_v2_impl` 走 cute 位公式：`off=(i*kColStride+j)*sizeof(half)`；`sw = SwizzleBMS<B,M,S>::apply(off)`，
    其中 `B=(kColStride==16)?1:(kColStride==32)?2:3`, `M=4`, `S=3`；`return ((sw >> M) & ((1 << B) - 1)) * kStep;`（kStep=8）。
    注释明确 **v1/v2 bit-exact 等价**，由 host 端 `test_swizzle_equiv`（`--swizzle-eq-check`）验证。
- **kernel 的完整签名**：本文件**不含任何 `__global__` kernel**（纯设备/主机工具层）。
- **块/线程/warp 组织**：无（仅 `kWarpSize = 32` 常量）。
- **smem 布局与字节数**：只有两个布局结构体，无具体字节数常量：`WgmmaSMem::A`/`B` 为 `alignas(128)` 数组，
  `TmaMmaWSSMem` 强制 `BK == 64` 且其 A/B **无 alignas**；字节数计算（`kSmemAllocateAB/Acc`）在 `ws-hgemm` 里。
- **同步原语**：`fence.proxy.async.shared::cta`、`mbarrier.arrive.expect_tx.shared::cta.b64`、
  `cp.async.bulk.tensor.2d ... mbarrier::complete_tx::bytes`；`cuda::barrier<cuda::thread_scope_block>`
  的 init/arrive/wait 直接复用（注释：它内联为裸 mbarrier PTX，不触发 C7506，因此无需门控）。
- **测试与容差**：本文件无测试；只有 host 端 `create_tensor_map` 失败时 `printf("cuTensorMapEncodeTiled failed: %d\n", (int)result);`
- **注释里的关键结论**（原样摘录）
  1. 「注意：cg 只支持 16 bytes，ca 支持 4/8/16 bytes」
  2. 「v2 不是新算法, 而是把 v1 的手写展开重新表达为 cute 的统一位置换公式, 便于脱离 cute 框架理解 swizzle 本质. v1/v2 bit-exact 等价.」
  3. 「★ 关键易错点：TMA shape 参数写的是 (W,H) 而不是 (H,W)! 对 row-major [M,K] 矩阵，TMA shape = (K,M)，minor=K，major=M」
  4. 「On sm_120a (Blackwell, CUDA 13.2) ptxas drops setmaxnreg with C7506 even when the PTX is fully inlined (no call.uni), because ptxas treats cp.async.bulk.tensor (TMA) usage as an implicit extern-call boundary. sm_90a (Hopper) is unaffected.」
  5. 「SM120 不支持 WGMMA，但支持相同的 TMA 生产者协议和 warp 级 mma.sync。」
  6. TensorMap 细节：`gmem_prob_stride + 1` 跳过隐式最内维 stride；`smem_box_stride[5] = {1,1,1,1,1}` 不需要 +1；
     固定使用 `CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`；
     `gmem_prob_stride[5] = {sizeof(half), sizeof(half) * BlockMinorSize * blocks_width, 0, 0, 0}`。
- **编译门槛**：`#if defined(NOTES_V2_ENABLE_TMA_MMA_WS) && CUDART_VERSION < 13000` → `#error "NOTES_V2_ENABLE_TMA_MMA_WS requires CUDA Toolkit 13.0 or newer"`

---

## kernels/interview/base.cuh

- **文件定位**：Phase 0–5 的“面试速查表 + 基础算子”层——架构/带宽/Roofline 速查（纯注释）、
  Warp/Block Reduce、Dot Product、Elementwise、Softmax 三级、RMS/Layer Norm、RoPE、Mat Transpose。
  是所有上层章节的基础设施，本身也直接是面试题面。
- **导出的符号 = 模板参数全列表**（逐条原样；本文件只有 kernel 与工具函数）
  - `struct __align__(8) MD { float m; float d; };`（running max + running denominator）
  - `template <const int kWarpWidth = kWarpSize, typename T = float> __device__ __forceinline__ T warp_reduce_sum(T val)`
  - `template <const int kWarpWidth = kWarpSize, typename T = float> __device__ __forceinline__ T warp_reduce_max(T val)`
  - `template <const int kNumThreads = 256> __device__ float block_reduce_sum(float val)`
  - `template <const int kNumThreads = 256> __device__ float block_reduce_max(float val)`
  - `template <const int kWarpWidth = kWarpSize> __device__ __forceinline__ MD warp_reduce_md(MD md1)`
  - `template <const int kNumThreads = 256> __global__ void block_reduce_all(float *a, float *y, int N)`
  - `template <const int kNumThreads = 256> __global__ void dot(float *a, float *b, float *y, int N)`
  - `template <const int kNumThreads = 256 / 4> __global__ void dot_vec4(float *a, float *b, float *y, int N)`
  - `__global__ void relu(float *x, float *y, int N)` / `__global__ void relu_vec4(float *x, float *y, int N)`
  - `__global__ void elementwise_add(float *a, float *b, float *c, int N)` / `__global__ void elementwise_add_vec4(...)`
  - `__global__ void histogram(int *a, int *y, int N)`
  - `__global__ void merge_attn_states(float *output, const float *prefix_output, const float *prefix_lse, const float *suffix_output, const float *suffix_lse, int num_tokens, int num_heads, int head_size)`（无模板）
  - `template <const int kNumThreads = 256> __global__ void softmax_per_token(float *x, float *y, int N)`
  - `template <const int kNumThreads = 256> __global__ void safe_softmax_per_token(float *x, float *y, int N)`
  - `template <const int kNumThreads = 256> __global__ void online_safe_softmax_per_token(const float *x, float *y, int N)`
  - `template <const int kNumThreads = 128> __global__ void rms_norm(float *x, float *y, float g, int N, int K)`
  - `template <const int kNumThreads = 128 / 4> __global__ void rms_norm_vec4(float *x, float *y, float g, int N, int K)`
  - `template <const int kNumThreads = 128> __global__ void layer_norm(float *x, float *y, float g, float b, int N, int K)`
  - `template <const int kNumThreads = 128 / 4> __global__ void layer_norm_vec4(float *x, float *y, float g, float b, int N, int K)`
  - `__global__ void rope(float *x, float *out, int seq_len, int N)`
  - `__global__ void mat_transpose(float *x, float *y, const int row, const int col)`
  - `__global__ void mat_transpose_padded(float *x, float *y, const int row, const int col)`
- **PTX 宏与工具宏**：无自有宏，全部依赖 `common.cuh` 的 `INT4/FLOAT4/HALF2/kWarpSize`。
- **kernel 的完整签名**：见上（逐条原样）。注意 `softmax_per_token` / `safe_softmax_per_token` /
  `online_safe_softmax_per_token` 三者线程数模板默认值分别是 `256 / 256 / 256`；
  `rms_norm*` / `layer_norm*` 默认 `128` 与 `128/4`。
- **块/线程/warp 组织**
  - Reduce：`block_reduce_sum/max` 内部 `constexpr int kNumWarps = (kNumThreads + kWarpSize - 1) / kWarpSize;`
    + `__shared__ float shared[kNumWarps]`；两级 warp→smem→warp，最后 `__shfl_sync(0xffffffff, value, 0, 32)` broadcast。
  - `merge_attn_states`：`constexpr int kNumThreads = 128;`、`constexpr int kPackSize = 16 / sizeof(float); // 4 floats = 128-bit`、
    `using pack_t = uint4;`、`threads_per_head = head_size / kPackSize`；**Grid: `((total_threads + 127) / 128, 1, 1)`，Block: `(128, 1, 1)`**。
  - `online_safe_softmax_per_token`：Grid `(S, 1, 1)`，Block `(H, 1, 1)`，由外层 dispatch 选择 H=32/64/128/256/512/1024。
  - `mat_transpose`：Grid `((col + 15) / 16, (row + 15) / 16, 1)`，Block `(16, 16, 1)`，每线程 1 元素。
  - `mat_transpose_padded`：Grid `((col + 15) / 16, (row + 63) / 64, 1)`，Block `(16, 16, 1)`，每线程 4 元素(float4)。
- **smem 布局与字节数**（原样，含 padding 值）
  - `block_reduce_*`：`__shared__ float shared[kNumWarps]`（无 padding）。
  - `online_safe_softmax_per_token`：`__shared__ MD shared[kNumWarps]`（`MD` 为 `__align__(8)` 的 2×float）。
  - `mat_transpose_padded`：`constexpr int TILE = 16; constexpr int PAD = 1;`
    `__shared__ float tile[TILE * 4][TILE + PAD]; // 64x16`
    注释原文：「BCF: smem 布局 `[kWarpSize_S*4][kWarpSize_S+PAD] = [64][17]`，PAD=1 加在第二维消除 bank conflict」
- **同步原语**：`__syncthreads()`（block reduce、transpose、merge 无同步）；
  `__shfl_xor_sync(0xffffffff, val, mask, kWarpWidth)`（蝶形归约，`kWarpWidth` 作为第 4 实参 segment width）；
  `__shfl_sync(0xffffffff, value, 0, 32)`（broadcast）。**没有 mbarrier / cp.async**。
- **测试与容差**：本文件无测试代码（测试在 `notes-v2.cu`）。相关 shape 由 `notes-v2.cu` 主流程给出：
  `test_merge_attn_states(512, 16, 128)`、`test_softmax(256)`、`test_rms_norm(8, 128)`、
  `test_layer_norm(8, 128)`、`test_rope(8, 128)`、`test_mat_transpose(256, 256)`、
  `test_mat_transpose_padded(256, 256)`、`test_block_reduce(N)`、`test_dot(N)`、`test_relu(1024)`、
  `test_elementwise(1024)`、`test_histogram(1024)`（默认 M=N=K=1024，可用 argv[1..3] 覆盖）。
- **注释里的关键结论**（原样摘录）
  1. 「为什么不用 `__shfl_down_sync`？xor 模式所有线程做相同工作量，更均衡」
  2. 「block_reduce: 两级归约（warp → shared memory → warp0 broadcast），注意最后必须 broadcast 回所有线程（`__shfl_sync`），否则只有 warp0 知道结果」
  3. 「Tensor Cores：Hopper 每 SM 4 个；Blackwell 数量随型号/定义不同，建议以官方 ISV guide 为准」
  4. 「注：本文件仅实现 Level 1(naive) 与 Level 4(BCF+merge_write)，Level 2/3 省略」
  5. merge_attn_states 数学：`L_max = max(LSE_1, LSE_2)`；`w_i = exp(LSE_i - L_max)`；`alpha = w_1/(w_1+w_2), beta = w_2/(w_1+w_2)`；
     `O = alpha*O_1 + beta*O_2`；「inf LSE → -inf：空 attention 段（causal mask 等导致全部 score 为 -inf）的 LSE 可能为 +inf，替换后 exp(-inf - L_max) = 0，该段权重退化为 0」
  6. merge_attn_states 布局：`LSE [num_heads, num_tokens]`（`lse[head_idx][token_idx]`），
     `Output [num_tokens, num_heads, head_size]`（token 维在最外，展平为 `[T0H0, T0H1, ..., T1H0, ...]`）。
  7. RoPE：`θ_i = 1 / (10000^(2i/d))`，`exp_v = 1.0f / powf(10000.0f, 2 * token_idx / (N * 2.0f))`。
  8. Roofline 结论（原文）：GEMM 4096³ AI≈685 FLOPS/Byte → compute-bound（H100 ridge point：FP16 TC ≈ 295:1，FP32 ≈ 20:1）；
     GEMV AI≈0.5 → severely memory-bound；Softmax(N=4096) AI = 5/8 ≈ 0.625 → memory-bound。

---

## kernels/interview/flash_attn.cuh

- **文件定位**：Phase 8 全部 FlashAttention 实现（FA2 手写 MMA+cp.async Split-Q、FA2 TMA+MMA Warp Specialization、
  FA3 双 consumer WG、FA2/FA3 的 CuTe 版本、Split-D TMA copy smoke）。是本章的主角文件。
  头部注释原文：`// flash_attn.cuh: Phase 8 FlashAttention 2/3 (MMA/TMA_WS/FA3/CuTe)`
- **导出的符号 + 模板参数全列表 + 完整签名**（三合一，逐条原样；所有模板参数**均无默认值**，除注明者）
  1. `template <typename T, int M, const int N, const int K = 2> __device__ inline void fill_3D_regs(T (&R)[M][N][K], T val)`
  2. `template <typename T, int M, const int N = 2> __device__ inline void fill_2D_regs(T (&R)[M][N], T val)`
  3. `template <int kHeadDim, int Br> __device__ __forceinline__ int swizzle_fa(int row, int col)`
  4. **FA2 手写 MMA Split-Q**（L93–110，**17 个模板参数，全无默认值**）：
     `template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK, const int kMmaAccF32, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK, const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ, const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV, const int kStagesK, const int kPadQ, const int kPadK, const int kPadV>`
     `__global__ void __launch_bounds__(kWarpSize * kMmaTileSeqLenQ * kMmaTileSeqLenK) flash_attn_mma_stages_split_q(half *Q, half *K, half *V, half *O, int N, int H)`
     参数注释原文：`kStagesK, // pipeline stages for K: >= 1; NO stages required for Q/V`；
     `kPadQ, // Q row padding; 0 selects compact XOR swizzle`（kPadK/kPadV 同）。
  5. **FA2 TMA+WS Split-Q**（L871–887，**16 个模板参数，全无默认值**）：
     `template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK, const int kMmaAccF32, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK, const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ, const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV, const int kStagesK, const int kStagesV, const int kNumThreads>`
     `__global__ void __launch_bounds__(kNumThreads, 1) flash_attn_tma_mma_ws_stages_split_q(half *Q, half *K, half *V, half *O, int N, int H, const CUtensorMap *__restrict__ tensorMapQ, const CUtensorMap *__restrict__ tensorMapK, const CUtensorMap *__restrict__ tensorMapV)`
     注释：`kStagesV, // V pipeline depth (>=1; 1=single buffer, >=2=pipelined)`；`kNumThreads> // 384, 128 producer + 256 consumer`
  6. **FA3 双 consumer WG**（L1505–1514，**15 个模板参数**）：与 #5 顺序完全相同，但**去掉了 kPadQ/kPadK/kPadV**：
     `template <const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK, const int kMmaAccF32, const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK, const int kMmaTileSeqLenP, const int kMmaTileHeadDimV, const int kValTileSeqLenQ, const int kValTileSeqLenK, const int kValTileSeqLenP, const int kValTileHeadDimV, const int kStagesK, const int kStagesV, const int kNumThreads>`
     `__global__ void __launch_bounds__(kNumThreads, 1) flash_attn_3_tma_ws_stages_split_q(...)`（参数表与 #5 逐字相同）
  7. namespace `fa_cute`：`convert_layout_acc_rowcol<Layout>`、`convert_layout_acc_Aregs<TiledMma>(Layout)`、
     `convert_type<To>(Tensor)`、`gemm_ss<TensorC,TensorA,TensorB,...>(...)`、`gemm_rs<...>(...)`；
     `template <int kHeadDim> struct FlashAttn2CuTeTraits`、`template <int kHeadDim> struct FlashAttn3CuTeTraits`（**无默认值**）
  8. `template <int kHeadDim, int kStagesK = 2> __global__ void __launch_bounds__(256) flash_attn_mma_stages_split_q_cute(cutlass::half_t *Q, cutlass::half_t *K, cutlass::half_t *V, cutlass::half_t *output, int rows, int seqlen)`
  9. `template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV, int kStagesK = 1, int kStagesV = 1> __global__ void __launch_bounds__(384, 1) flash_attn_tma_mma_ws_split_q_cute(CUTLASS_GRID_CONSTANT TmaQ const tma_q, CUTLASS_GRID_CONSTANT TmaK const tma_k, CUTLASS_GRID_CONSTANT TmaV const tma_v, cutlass::half_t *output, int rows, int seqlen)`
  10. `template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV, int kStagesK = 1> __global__ void __launch_bounds__(384, 1) flash_attn_3_tma_mma_ws_split_q_cute(CUTLASS_GRID_CONSTANT TmaQ const tma_q, CUTLASS_GRID_CONSTANT TmaK const tma_k, CUTLASS_GRID_CONSTANT TmaV const tma_v, cutlass::half_t *output, int rows, int seqlen)`
  11. `template <int kHeadDim, typename TmaQ> __global__ void flash_attn_3_cute_tma_copy_smoke(...)`（L3449，**无 `__launch_bounds__`**）
  12. **（@ `6c86259` 新增，非任务原始范围）** `template <typename Traits, typename TmaQ, typename TmaK, typename TmaV, typename TmaO> __global__ void __launch_bounds__(384, 1) flash_attn_cute_persist_d_sm120(CUTLASS_GRID_CONSTANT TmaQ const tma_q, ..., CUTLASS_GRID_CONSTANT TmaO const tma_o, typename Traits::Element* __restrict__ O, int Nq, int Nkv, int Nh, int Nh_kv, float scale, int Tc, int causal, int q_tiles, int total_q_tiles, int total_q_rows, int total_kv_rows)`
     ——persistent-CTA 版，`kProducerThreads = 128; kConsumerThreads = 256;`，
     注释原文：`// WS persist-D persistent kernel: 128T producer (TMA-only) + 256T consumer (MMA-only)。Q/K/V/O 均为 BHND packed -> flat (B*H*N, D) 行的 2D TMA。barrier 相位: K/V 用跨 q-tile 的全局 kv 计数 (stage=g%S, phase=(g/S)&1), 与非 persistent 版的相对序完全一致; q_full/epi_done 用 q-tile 迭代号。`
  13. 条件编译门：`NOTES_V2_ENABLE_TMA_MMA_WS`、`NOTES_V2_ENABLE_CUTE`
- **PTX 宏与工具宏**：本文件不自造 PTX 宏，全部调用 `common.cuh` 的
  `CP_ASYNC_CG` / `CP_ASYNC_COMMIT_GROUP` / `CP_ASYNC_WAIT_GROUP(n)` / `LDMATRIX_X4` / `LDMATRIX_X2` / `LDMATRIX_X2_T` /
  `HMMA16816` / `HMMA16816F32` / `tma_load_2d` / `tma_arrive_expect_tx` / `tma_fence_proxy_async_shared_cta` /
  `NOTES_V2_REG_DEALLOC(40)` / `NOTES_V2_REG_ALLOC(80|168)` / `swizzle<16>`（经 `swizzle_fa` 间接）。
- **块/线程/warp 组织**
  - **FA2 手写版**：`constexpr int kNumThreads = kWarpSize * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*8*1=256`
    （头注释另写 `kNumThreads=kWarpSize×kMmaTileSeqLenQ×kMmaTileSeqLenK=128`——**注释与代码不一致**，
    代码里 kMmaTileSeqLenQ=8 时是 256；头注释的 128 对应 Br=64 配置）。
    `constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kValTileSeqLenQ; // 16*8*1=128`、
    `constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kValTileSeqLenK; // 8*1*8=64`。
    `warp_QP = warp_id`（各 warp 不同 Q 行），`warp_KV = 0`（所有 warp 共享 K）——**这就是 Split-Q**。
    头注释 Block 原文：`Block: (128, 1, 1)，kNumThreads=kWarpSize×kMmaTileSeqLenQ×kMmaTileSeqLenK=128`；
    `Grid: ((N + 63) / 64, B * H, 1)，Br=64`。测试端实际用 `dim3 grid((seqlen + Br - 1) / Br, B * H);`
  - **FA2 TMA+WS**：`static_assert(kNumThreads == 384, "128 producer + 256 consumer");`
    `kConsumerThreads = 256;  // 8 warps`、`kProducerThreads = 128;  // 4 warps, only thread 0 issues TMA`
    `kConsumerThreads=256` 时 `warp_QP = warp_id (0~7)`、`warp_KV = 0`，每个 warp 16 行 Q → Br = 16*8*1 = 128。
    **线程划分警告（原文）**：`WARN: Must use kProducerThreads (not kConsumerThreads) as the divisor, otherwise the split is wrong: 384/256=1.5 would give Producer 256 threads and Consumer only 128, but barriers expect 256 consumer arrives → deadlock.`
  - **FA3 双 consumer WG**：384 threads；
    `WG0 [0,127]: TMA producer (仅 thread 0 发 TMA)`、`WG1 [128,255]: consumer_id=0, 处理偶数 KV tile (0,2,4,...)`、
    `WG2 [256,383]: consumer_id=1, 处理奇数 KV tile (1,3,5,...)`；
    `kConsumerThreadsPerWG = 128; kProducerThreads = 128; kNumConsumerWGs = 2;`，
    `constexpr int Br = ...; // 64`、`constexpr int Bc = ...; // 64`。
  - **CuTe FA2**：`kBr = 128; kBc = 64;`，`__launch_bounds__(256)`，`TiledMma = TiledMMA<MmaAtom, Layout<Shape<_8,_1,_1>>, Tile<_128,_16,_16>>`（8 warps，单 WG）。
  - **CuTe FA3**：`kTile = 64; kNumConsumers = 2; kConsumerThreads = 128; kProducerThreads = 128;`，
    `TiledMma = TiledMMA<MmaAtom, Layout<Shape<_4,_1,_1>>, Tile<_64,_16,_16>>`。
  - **CuTe FA2 TMA WS**：`kBr = 128; kBc = 64; kConsumerThreads = 256; kProducerThreads = 128;`，`__launch_bounds__(384, 1)`。
- **smem 布局与字节数**
  - FA2 手写版（padding 与 XOR 二选一，**绝不叠加**）：
    `constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);`、`K_tile_size = Bc * (kHeadDim + kPadK);`
    `kSmemStrideQ = kHeadDim + kPadQ; kSmemStrideK = kHeadDim + kPadK; kSmemStrideV = kHeadDim + kPadV;`
    `half *Q_tile_smem = smem; half *K_tile_smem = Q_tile_smem + Q_tile_size; half *V_tile_smem = K_tile_smem + kStagesK * K_tile_size;`
    注释原文：「Q/K/V independently use padded row-major when `kPad* > 0` and compact XOR swizzle when `kPad* == 0`. Padding and XOR are never combined per operand. The swizzled physical layout is `[col / 16][row][16]`; `swizzle<16>()` selects the 0/8 phase inside the final 16-half tile.」
    **默认 kPad = 8**（见 `notes-v2.cu` bench dispatch 的 `<64, 2, 8, 8, 8, acc>`）。
  - FA2 TMA+WS：
    `constexpr int kQTileBytes = Br * kHeadDim * sizeof(half);       // 16 KB`
    `constexpr int kKTileBytes = Bc * kHeadDim * sizeof(half);       // 8 KB`
    `constexpr int kVTileBytes = Bc * kHeadDim * sizeof(half);       // 8 KB`
    `constexpr int kTmaBoxMinor = 64;`、`constexpr int kTmaChunks = kHeadDim / kTmaBoxMinor;`
    `extern __shared__ __align__(1024) uint8_t smem_fa_tma_ws[];`，Q 在前、K 中、V 后。
    头注释给出的 smem 总量公式与表格（原文）：
    `smem = (Q[Br,D] + K[kStagesK,Bc,D] + V[kStagesV,Bc,D]) * sizeof(half) = D * (Br + kStagesK*Bc + kStagesV*Bc) * 2 = D * (128 + (kStagesK+kStagesV)*64) * 2`（Br=128, Bc=64）；
    `| Sk | Sv | D=64 | D=128 |` → `1,1: 32KB/64KB`；`2,1: 40KB/80KB`；`2,2: 48KB/96KB`；`3,1: 48KB/96KB`；`3,2: 56KB/112KB`；`4,1: 56KB/112KB`。
  - FA3：`smem_fa3_tma_ws`，布局注释原文：
    `[Q_shared: Br*D]` / `[K[cid=0][s=0..Sk-1]: Sk*Bc*D] [K[cid=1][s=0..Sk-1]]` / `[V[cid=0]: Bc*D] [V[cid=1]: Bc*D]`；
    地址：`K_smem_base + cid * (kStagesK * Bc * kHeadDim) + stg * (Bc * kHeadDim)`，
    `V_smem_base + cid * (Bc * kHeadDim)`。
  - CuTe 版：`extern __shared__ __align__(1024) Element shm[];`（CuTe FA2：Q[128,D] + K[kStagesK,64,D] + V[1,64,D]）。
- **同步原语**
  - FA2 手写版：`CP_ASYNC_COMMIT_GROUP()` / `CP_ASYNC_WAIT_GROUP(0|1)` / `CP_ASYNC_WAIT_GROUP(kStagesK - 2)` / `__syncthreads()`。
    6 个同步点的注释（CuTe 版复述手写版）：`1. Q load: copy + fence + wait<0> + sync`；
    `2. PREFETCH K[0..Sk-2]: copy + fence，然后 wait<Sk-2> + sync`；每轮 `3b QK 前 wait`、`3d PV 前 wait`、
    `循环末尾: kStagesK>1 且非最后 -> wait<0> + sync`。
  - FA2/FA3 TMA+WS：`cuda::barrier<cuda::thread_scope_block>` 的 `init/arrive/wait` +
    `tma_load_2d`（`cp.async.bulk.tensor.2d`）+ `tma_arrive_expect_tx` + `tma_fence_proxy_async_shared_cta()` +
    `__syncthreads()`（只用于 barrier 初始化后）。
    **arrive_count 协议（原文）**：`arrive_count = kConsumerThreads + 1 = 257 (256 consumer arrives + 1 producer arrive_tx)`；
    FA3 是 `init(&full_Q, 256 + 1);  // 2 consumer WGs * 128 + 1 producer`，per-WG 的 K/V barrier 为 `kConsumerThreadsPerWG + 1 = 129`。
    barrier 集合：FA2 = `full_Q`, `full_K[kStagesK]`, `empty_K[kStagesK]`, `full_V[kStagesV]`, `empty_V[kStagesV]`；
    FA3 = `full_Q`, `full_K[2][kStagesK]`, `empty_K[2][kStagesK]`, `full_V[2]`, `empty_V[2]`。
    两处都用 `#pragma nv_diag_suppress static_var_with_dynamic_init` 包住 `__shared__ cuda::barrier`。
  - 寄存器再平衡：producer 侧 `NOTES_V2_REG_DEALLOC(40);`；
    FA2 consumer 侧（原文注释：`Consumer register budget per Triton flash_attn_v2 maxnreg strategy on Blackwell warp_specialize: D=128 -> 168, otherwise -> 80.`）
    `if constexpr (kHeadDim == 128) { NOTES_V2_REG_ALLOC(168); } else { NOTES_V2_REG_ALLOC(80); }`
  - K/V 的**早释放**优化（原文）：「Release K[stage] for producer reuse as early as possible: QK^T GEMM has consumed all K smem data; the subsequent softmax (3c) and PV GEMM (3d) only touch registers (R_S, R_O) and V smem, so K[stage] is free to be overwritten by the producer's next TMA prefetch.」
    同理 V：`empty_V[stage_v].arrive()` 放在 PV 之后、rescale 之前。
  - **FA3 的 KV 结果合并**（split-KV，原文公式）：
    `alpha = exp(m_0 - m) (<=1)`、`beta = exp(m_1 - m) (<=1)`、`l = alpha*l_0 + beta*l_1`、
    `Oacc = alpha*Oacc_0 + beta*Oacc_1`、`O = Oacc / l`；
    `Tc==1 退化: WG2 无 tile, m_1=-inf, l_1=0, Oacc_1=0; beta = exp(-inf) = 0, 退化为 WG1 结果。`
  - **split-Q 语义（原文，关键）**：「grid = (seqlen/Br, B*H)，每个 block 处理一个 **block-level Q tile [Br, d]**（不是全局 Q）。对 Q 的 seqlen 做 block-level 并行。每个 block 遍历**完整的 KV seqlen**（Tc = seqlen/Bc 次外循环）。因此每个 block 的 Q tile 只需 **load 一次到 smem**，在整个 KV 遍历中复用 → 只需 full_Q barrier（无 empty_Q）。」
- **测试与容差**
  - **tol 的真实情况（务必按此写，不要编第三档）**：本仓库 FA 路径**只有一个统一的失败阈值**：
    `bool is_fail = max_err >= 5e-1f;`（`notes-v2.cu` L3201/L3316/L3465/L3622/L3784/L3932；
    带 `checked` 前缀的写作 `bool is_fail = checked && max_err >= 5e-1f;`）。
    **不存在 TF32 容差档**；也没有 `1e-2f` / `1e-3f` 之类的 tol 常量（grep 全 `.cu` 无命中）。
    **“三档容差”的真实形态是 README 里报告的经验 Max Err 数值**（不是判据）：
    - F16Acc（FA2 MMA Stages, Pad）：`1.831e-04`
    - F32Acc（FA2 MMA Stages, Pad）：`1.526e-05`
    - FA3 TMA MMA WS (2 Consumer WG) F16Acc：`9.155e-05`
    - HGEMM CuTe Swizzle：`0.000e+00`
    这些值与 `5e-1f` 相差 3–4 个数量级。
  - **参考实现**：
    - CPU FP32 参考（`test_flash_attn` / `test_flash_attn_tma_mma_ws_impl`）：`srand(42)`，
      输入 `__float2half(((float)rand() / RAND_MAX) * 2.0f - 1.0f)`；
      `float scale = 1.0f / sqrtf((float)head_dim);`；
      `S[kj] = Σ_d ref_q[...]*ref_k[...] * scale`；softmax 用 `double sum_exp += (double)expf(S[kj] - smax);`；
      输出累加也用 `double o_acc`。
    - cuDNN 参考：`bench_cudnn_sdpa_tflops(...)`，用 cudnn-frontend `graph->sdpa(Q, K, V, SDPA_attributes().set_name("sdpa_ref").set_attn_scale(1.0f / sqrtf((float)head_dim)))`；
      `set_io_data_type(HALF).set_intermediate_data_type(FLOAT).set_compute_data_type(compute_type)`；
      `graph->build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK})`。
  - **shape 列表**（`notes-v2.cu` 主验证流程）：
    `test_flash_attn(1024, 64)`；
    `test_flash_attn_tma_mma_ws(1024, 64)`、`test_flash_attn_tma_mma_ws(1024, 128)`；
    `test_flash_attn_3_tma_ws(1024, 64)`、`test_flash_attn_3_tma_ws(1024, 128)`；
    `test_flash_attn_tma_mma_ws` / `test_flash_attn_3_tma_ws` 内部对 `<64>` 与 `<128>` 都跑 S=1..4。
    bench 默认：`g_bench_B = 1, g_bench_H = 32, g_bench_Nfa = 8192, g_bench_D = 128;`
    （`--bhnd` 覆盖）。README 基准 shape：`1,32,16384,128` 与 `1,32,16384,320`（Split-D）。
    前置条件（原文）：`if (seqlen < Br || seqlen % Br != 0 || seqlen % Bc != 0)` → 打印 `SKIP`。
  - **smem 可行性兜底**：`check_smem_feasible()` 对比
    `cudaDevAttrMaxSharedMemoryPerBlockOptin` 与 `dyn_smem_bytes + attrs.sharedSizeBytes`，
    失败时打印 `SMEM too large` 行并跳过（不报错）。
  - 计时：`g_warmup = 2, g_repeat = 3`，`cudaEventRecord/ElapsedTime`。
- **注释里的关键结论**（原样摘录——5 条最重要）
  1. **作者明确说不要泛化（P 写回 R_S）**：「为什么 R_S 可以直接用作 P@V 的 A 矩阵？… 当前实现依赖 m16n8k16 这一路径下约定好的 fragment 布局，使 softmax 后的 P 可以继续留在 R_S 中供后面的 P@V 直接消费。这是此实现的寄存器布局复用技巧，**不要背成“所有 MMA A/C fragment 都天然同构”的通用结论**」；
     并在 3d 处再次强调：「当前实现正是利用这一路径下 A fragment 与前面生成的 P fragment 可以直接对接，才能把 R_S 中的 P 直接喂给 HMMA16816 做 P@V；复习时不要把它背成对所有 MMA fragment 都无条件成立的通用结论。」
  2. **XOR swizzle 的实测反例（SM120, B=1,H=32,N=4096,D=64）**：「XOR 能消除目标 ldmatrix lane pattern 的 bank conflict，但收益是局部的；XOR 本身及 `% 2` 的计算不是主要瓶颈，nvcc 已能将该式化简为 bit operations。compact Q/K/V XOR 改变了 cp.async 的 shared destination pattern：LDGSTS wavefronts 从 pad 的 30.15M 增至 68.16M（2.26x），long-scoreboard、LG-throttle、MIO-throttle 分别约为 pad 的 3.25x、2.74x、1.60x。compact XOR 虽节省约 5 KiB smem，但没有提高此 kernel 的 occupancy；最终约 120.0 TFLOPS，显著低于 Q/K/V kPad=8 的 166.6 TFLOPS。所以当前 kernel 默认对 Q/K/V 都用 kPad=8。保留 swizzle 路径是为了学习和消融：评价 shared layout 必须同时观察 ldmatrix reads 与 cp.async/LDGSTS writes，不能只看 bank-conflict counter，也不能只优化 XOR 地址算术。」
  3. **注释与代码不一致（尾 tile）**：「原始实现默认 seqlen 与 Bc 对齐；最后一个不完整 tile 需要额外 pad/边界处理。这里保留 ceil 写法是为了说明 tile 划分方式，**不等于当前实现已经完整处理了尾 tile**。」
     同一处还有 v1 限制原文：`v1 限制：kHeadDim=64 or 128；seqlen % Br == 0 且 seqlen % Bc == 0（不处理尾 tile）`。
     FA3 限制原文：`kStagesV == 1`；`D=128 时 Sk=1 (Sk=2 需 112KB > 101KB optin 上限)`；`Br=64, Bc=64, D=64/128, aligned seqlen (N % Br == 0 && N % Bc == 0)`；**`仅 self-attention, 无 causal/varlen/GQA`**。
  4. **TMA 128B 的硬约束与 chunk-major 方案**：「CU_TENSOR_MAP_SWIZZLE_128B 硬件要求 box innermost dim ≤ 128B = 64 half。因此 D=128 不能用单个 TMA box=(128, Br) 覆盖整行。方案：D=128 时 box 固定为 (64, Br)，沿 head_dim 方向连续发 kTmaChunks 次 TMA（minor_coord = c*64），写入 chunk-major smem 布局 [kTmaChunks, Br, 64]」；
     以及 `TMA CU_TENSOR_MAP_SWIZZLE_128B requires 1024B-aligned smem base so the hardware swizzle phase starts at zero. Consumer swizzle<64>() assumes zero phase (no base_offset compensation like WGMMA descriptor).`
  5. **V ldmatrix.x2.trans 正确性（原文）**：「TMA 128B SWIZZLE 是 1-1 映射，ldmatrix 用 swizzle<64>(row,col) 计算的物理地址 = TMA 写入的物理地址（smem 1024B 对齐保证 phase=0）。ldmatrix.x2.trans 的转置语义在寄存器层面工作，与 smem 物理布局无关 → 必然正确。」
  6. 其他：`swizzle_fa` 的 `static_assert(kHeadDim == 64 || kHeadDim == 128, "D=64 or 128 only");`；
     `D=64 退化：kTmaChunks=1, chunk=0, offset = row*64 + swizzle<64>(row, col) 与原公式完全一致 → 无回归。`；
     FA3 为什么奇偶拆分：「关键约束: producer 按 global tile 顺序 0,1,2,3,... 连续 TMA。奇偶拆分让 cid = tile & 1 静态路由…若改为前/后半区拆分: 过渡区两个 WG 可能同时需要同一 cid 的 slot, 且 WG2 跳跃访问远端 tile 时数据尚未被 producer TMA 到 smem。」
     FA2 TMA WS 与 HGEMM 的对比（原文）：`HGEMM 只做一次 GEMM，K 维迭代；FA 做 QK^T 和 PV 两次 GEMM，KV seqlen 迭代`；`HGEMM 的 A/B 都 staged；FA 中 Q 只 load 一次（split-Q），K staged，V 单 buffer`。

---

## kernels/interview/ffpa_attn.cuh

- **文件定位**：large head-dim（D>128）的 Split-D attention——按 64-wide 的 D chunk 切分 head_dim，
  每 block 只算一段 D 的部分输出。`#include "flash_attn.cuh"` 复用 `fa_cute` 命名空间。
- **导出的符号 = 模板参数全列表 = 完整签名**（逐条原样）
  - `template <int kHeadDim, int TILE_M = 64, int TILE_N = 64> struct FFPAAttnSplitDCuTeTraits`（`namespace fa_cute` 内；
    `static_assert(kHeadDim % 64 == 0, "Split-D requires head-dim multiple of 64");`
    `static_assert(TILE_M == 64 && TILE_N == 64, "Current impl supports 64x64 only");`）
  - `template <int kHeadDim, int kStagesQK = 2, int kStagesV = 2> __global__ void __launch_bounds__(128) ffpa_split_d_cute(cutlass::half_t *Q, cutlass::half_t *K, cutlass::half_t *V, cutlass::half_t *output, int rows, int seqlen)`（cp.async 版，无 TMA/WS，**128 threads**）
  - `template <int kHeadDim, typename TmaQ, typename TmaK, typename TmaV, int kStagesQK = 2, int kStagesV = 2> __global__ void __launch_bounds__(256, 1) ffpa_attn_tma_mma_ws_split_d_cute(CUTLASS_GRID_CONSTANT TmaQ const tma_q, CUTLASS_GRID_CONSTANT TmaK const tma_k, CUTLASS_GRID_CONSTANT TmaV const tma_v, cutlass::half_t *output, int rows, int seqlen)`
  - 门控：`#if defined(NOTES_V2_ENABLE_CUTE)`（cp.async 版）与 `#if defined(NOTES_V2_ENABLE_CUTE) && defined(NOTES_V2_ENABLE_TMA_MMA_WS)`（TMA WS 版）
- **PTX 宏与工具宏**：不自造 PTX 宏；用 CuTe `SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>` /
  `SM75_U32x4_LDSM_N` / `SM75_U16x8_LDSM_T` / `MMA_Atom<SM80_16x8x16_F32F16F16F16F32_TN>`，
  以及 `NOTES_V2_REG_DEALLOC(40)` / `NOTES_V2_REG_ALLOC(232)`。
  TMA 侧用 `cutlass::arch::ClusterTransactionBarrier` / `cutlass::arch::ClusterBarrier` 的
  `init / wait(&bar, phase) / arrive_and_expect_tx(&bar, bytes)`。
- **块/线程/warp 组织**
  - Traits（原文注释）：`QK: Tile<64,64,16> + Layout<4,1,1> → EURepeat<1,8,1>`，
    「一次 TiledMMA 覆盖完整 S[64,64]，省掉 N-tile 循环，提升计算密度」；
    `PV: Tile<64,16,16> + Layout<4,1,1> → EURepeat<1,2,1>`，「保持小 tile 控制 acc_O 寄存器」。
  - cp.async 版：**128 producer + 128 consumer = 256**? **不是** —— 原文写
    「128 线程：与 TiledMmaQK/PV 的 Layout<4,1,1> 一致，每个线程在 G2S 和 S2R/MMA 中有唯一分区，
    消除 256-thread G2S 与 128-thread MMA 之间的映射不匹配」，`__launch_bounds__(128)`。
  - TMA WS 版：`constexpr int kProducerThreads = 128; constexpr int kConsumerThreads = 128;`
    = **256 total threads (vs 原始 384)**；`kBr = 64; kBc = 64; kDChunk = 64; kDChunks = kHeadDim / kDChunk;`
  - grid：`q_tile = blockIdx.y * (seqlen / kBr) + blockIdx.x; kv_tiles = seqlen / kBc;`
- **smem 布局与字节数**
  - `extern __shared__ __align__(1024) Element shm[];`
  - `auto q_slice = tma_q.get_slice(_0{});` … 指针：`q_base = shm; k_base = q_base + kStagesQK * kQChunkElements;`
    `v_base = k_base + kStagesQK * kKVChunkElements;`
  - 原文布局注释：`SMEM: sQ[kStagesQK,64,64] + sK[kStagesQK,64,64] + sV[kStagesV,64,64]`；
    `stage 偏移通过基地址指针算术管理，不使用 stride-0 的 stage-mode layout。`
  - 无显式字节数常量，用 `cosize(SmemLayoutQ{})` / `cosize(SmemLayoutKV{})`
    （`SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<Element>`，Q/KV 均为 `[64, 64]` tile）。
- **同步原语**
  - cp.async 版：`cp_async_fence()` / `cp_async_wait<kStagesQK - 2>()` / `cp_async_wait<kStagesV - 2>()` /
    `cp_async_wait<0>()` / `__syncthreads()`。
  - TMA WS 版：`__shared__ uint64_t qk_full[kStagesQK]; qk_empty[kStagesQK]; v_full[kStagesV]; v_empty[kStagesV];`；
    `TmaBarrier::init(&qk_full[stage], 1); CtaBarrier::init(&qk_empty[stage], kConsumerThreads);`
    （**kStagesV 的 v_full/v_empty 也是同上**）；phase 计算：`const int phase = (chunk_index / kStagesQK) & 1;`
    `CtaBarrier::wait(&qk_empty[stage], phase);`；`TmaBarrier::arrive_and_expect_tx(&qk_full[stage], sizeof(Element) * (size(sQ) + size(sK)));`
  - 寄存器再平衡：producer `NOTES_V2_REG_DEALLOC(40)`；consumer `NOTES_V2_REG_ALLOC(232)`（**注意是 232，与 FA 的 80/168 不同**）。
- **测试与容差**：本文件无测试。相关 bench 在 `notes-v2.cu`（`bench_fa_split_d_launch` / `bench_fa_split_d_dispatch`
  用 `<D,1,1>` 与 `<D,2,2>`，以及 `bench/bench_ffpa.cu`）。
  README 给出的结果：`FA Split-D CuTe TMA MMA WS (D=320, Sk=1, Sv=1) | 1.526e-05 | 127.2/83.1 (1.53x)`；
  `(D=320, Sk=2, Sv=2) | 1.526e-05 | 182.6/83.1 (2.20x)`；
  头注释性能：`(B=1,H=32,N=8192,D=512, SM120a RTX PRO 5000): cuDNN SDPA: 57.2 TFLOPS | FFPA TMA WS: 110.7 TFLOPS (1.94x)`。
- **注释里的关键结论**（原样摘录）
  1. 「通过 include flash_attn.cuh 复用 fa_cute namespace 中的 FA traits 和 helpers。仅定义 FFPA 特有的 FFPAAttnSplitDCuTeTraits 和 ffpa_attn_tma_mma_ws_split_d_cute kernel。支持 head_dim > 128 的 large head-dim attention，通过 64-wide Split-D chunks 处理。」
  2. 「消费者逻辑与 ffpa_attn_tma_mma_ws_split_d_cute 完全一致（双 TiledMma，相同的 fragment 流转：QK->convert_layout_acc_Aregs<TiledMmaPV>->PV）。唯一区别：生产者从 TMA 换成 cp.async，用 cp_async_fence/wait 替代 TMA barrier。」
  3. 「128 线程：与 TiledMmaQK/PV 的 Layout<4,1,1> 一致，每个线程在 G2S 和 S2R/MMA 中有唯一分区，消除 256-thread G2S 与 128-thread MMA 之间的映射不匹配。」
  4. 「128 producer + 128 consumer = 256 total threads (vs 原始 384)」

---

## kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu

- **文件定位**：独立的“最朴素的 CuTe warp-specialization HGEMM”教学样例（PyTorch 扩展，`torch/extension.h` + `PYBIND11_MODULE`），
  用 `cuda::pipeline` 而不是 mbarrier 做 producer/consumer 同步。是理解 WS 的起点参考。
- **导出的符号 = 模板参数全列表 = 完整签名**（逐条原样）
  - `template <class CTATile, int ProducerThread, int Stage> struct WSHGEMMTraits`（含 `struct Arguments`；**无默认值**）
  - `template <typename WSHGEMMTraits> __global__ void ws_hgemm_naive_cute_kernel(typename WSHGEMMTraits::Arguments args)`（**无默认值**，前有 `#pragma nv_diag_suppress static_var_with_dynamic_init`）
  - host：`void ws_hgemm_naive_cute(torch::Tensor a, torch::Tensor b, torch::Tensor c)`；
    `inline int get_max_smem_size()`；`template <typename Kernel> void config_smem(Kernel kernel, int smem_size)`
  - 宏：`DEVICE` → `__device__ __forceinline__`；`STRINGFY(str)` → `#str`；
    `TORCH_BINDING_COMMON_EXTENSION(func)` → `m.def(STRINGFY(func), &func, STRINGFY(func));`；
    `CHECK_TORCH_TENSOR_DTYPE(T, th_type)`、`CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)`
  - Traits 内静态函数模板：`producer<Pipeline,AEngine,ALayout,BEngine,BLayout>(void *smem_ptr, Pipeline&, Tensor<AEngine,ALayout> const&, Tensor<BEngine,BLayout> const&)`、
    `main_loop<Pipeline,CEngine,CLayout>(Arguments const&, void*, Pipeline&, Tensor<CEngine,CLayout> const&)`、
    `epilog<AccEngine,AccLayout,CEngine,CLayout>(Arguments const&, void*, Tensor<AccEngine,AccLayout> const&, Tensor<CEngine,CLayout>&)`、
    `consumer<Pipeline,CEngine,CLayout>(Arguments const&, void*, Pipeline&, Tensor<CEngine,CLayout>&)`
  - Traits 内常量：`kMmaThrLayoutM = 2; kMmaThrLayoutN = 2; kMmaThrLayoutK = 1;`
    `kSwizzleB = 3; kSwizzleM = 3; kSwizzleS = 3;` `kSmemStageAcc = 2;`
    `kConsumerThread = size(TiledMMA{}); kProducerThread = ProducerThread; kAllThread = kProducerThread + kConsumerThread;`
  - 实例化点（原文）：`using GEMM_Traits = WSHGEMMTraits<decltype(make_shape(_128{}, _256{}, _32{})), 32, 3>;`
    → CTATile = 128×256×32，ProducerThread = 32，Stage = 3
- **PTX 宏与工具宏**：无自有 PTX 宏。CuTe copy op（原文）：
  - `using mma_op = SM80_16x8x16_F16F16F16F16_TN;`（**F16 累加**，AccType = half）
  - `using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;`
  - `using s2r_copy_op = SM75_U32x4_LDSM_N;`（`S2RCopyA = S2RCopyB = s2r_copy_atom`）
  - `using R2SCopyC = Copy_Atom<UniversalCopy<int>, AccType>;`
  - `using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, AccType>;`
  - Smem atom：`SmemLayoutAtom = composition(Swizzle<3,3,3>{}, make_layout(make_shape(Int<8>{}, Int<kCTAK>{}), make_stride(Int<kCTAK>{}, Int<1>{})))`
- **块/线程/warp 组织**
  - `constexpr static int kConsumerThread = size(TiledMMA{});`，
    `static_assert(ProducerThread % 32 == 0, "The number of ProducerThreads must be a multiple of 32");`
    （注释原文：`// To avoid warp divergence`）
  - `constexpr static int kProducerThread = ProducerThread;`
    `constexpr static int kAllThread = kProducerThread + kConsumerThread;`
    对 `WSHGEMMTraits<..., 32, 3>`：Producer 32 线程 + Consumer（`size(TiledMMA{})` = 2×2×32 = 128）= **160 线程**
  - `TiledMMA = make_tiled_mma(mma_atom{}, MmaThrLayout{2,2,1}, MmaPermutation{})`，
    `kMmaPermuteM = 2*16 = 32`、`kMmaPermuteN = 2*2*8 = 32`、`kMmaPermuteK = 1*16 = 16`；
    原文注释：`// The expanded TiledMMA can process matrices of size 32x32x16 in a single operation.`
  - 线程角色：`const auto thread_role = tidx < WSHGEMMTraits::kProducerThread ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;`
  - Grid：`args.get_grid()` → `dim3(ceil_div(M, kCTAM), ceil_div(N, kCTAN))`；Block：`dim3 block(block_size)`，`block_size = GEMM_Traits::kAllThread`
- **smem 布局与字节数**
  - `constexpr static int kSmemSizeA = cosize(SmemLayoutA{}); kSmemSizeB = cosize(SmemLayoutB{});`
    `kSmemAllocateAB = (kSmemSizeA + kSmemSizeB) * sizeof(MatrixTypeAB);`
    `kSmemSizeAcc = cosize(SmemLayoutAcc{}); kSmemAllocateAcc = kSmemSizeAcc * sizeof(AccType);`
    `kAllSmemAllocate = cute::max(kSmemAllocateAB, kSmemAllocateAcc);`（**AB 与 Acc 复用同一块 smem**）
  - `SmemLayoutA = tile_to_shape(atom, (kCTAM, kCTAK, kStage))`、`SmemLayoutB = tile_to_shape(atom, (kCTAN, kCTAK, kStage))`
  - `extern __shared__ MatrixTypeAB smem_ptr[];`
  - **无 padding 值**（用 `Swizzle<3,3,3>` 而非 PAD）；`config_smem()` 在 `smem_size >= 32 * 1024` 时才调 `cudaFuncSetAttribute`
- **同步原语**
  - `cooperative_groups::this_thread_block()`；
  - `__shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, kStage> shared_state;`
    + `auto pipeline = cuda::make_pipeline(block, &shared_state, thread_role);`
  - `pipeline.producer_acquire()` / `pipeline.producer_commit()`（producer 侧）
  - `pipeline.consumer_wait()` / `pipeline.consumer_release()`（consumer 侧）
  - `__syncthreads()` ×3（epilog 前 1 次、r2s 后 1 次、s2g 后 1 次）
  - **没有 `__syncthreads` 在主循环里**；**没有 mbarrier / TMA**。
- **测试与容差**：本文件**无测试、无 tol、无参考实现**——只提供 PyTorch 绑定 `ws_hgemm_naive_cute(a, b, c)`，
  输入校验为 `CHECK_TORCH_TENSOR_DTYPE(..., torch::kHalf)` 与 shape `a:(M,K) b:(K,N) c:(M,N)`。
- **注释里的关键结论**（原样摘录）
  1. `// The expanded TiledMMA can process matrices of size 32x32x16 in a single operation.`
  2. `// To avoid warp divergence`（配 `static_assert(ProducerThread % 32 == 0, ...)`）
  3. `// Different thread_roles execute different branches.`
  4. `__syncthreads(); // wait all consumer thread finish main_loop`、
     `__syncthreads(); // wait all consumer thread finish r2s`、
     `__syncthreads(); // wait all consumer thread finish s2g`

---

## kernels/interview/build.sh

- **文件定位**：`notes-v2.cu` 的编译脚本——**两步编译+链接**（compile 走 ccache，link 不走），
  每个 arch 一套 gencode + 宏 + 库 + 输出名。
- **导出的符号**：无（bash 脚本）。关键变量：
  `SCRIPT_DIR`、`USE_CCACHE`、`NVCC="/usr/local/cuda/bin/nvcc"`、`COMMON_FLAGS`（数组）、
  `ARCH_GENCODE/ARCH_DEFINES/ARCH_LIB_PATH/ARCH_LIBS/ARCH_OUTPUT`（5 个 `declare -A` 关联数组）、
  `VALID_ARCHS="sm_86 sm_89 sm_90a sm_120a sm_120f"`（**注意：当前 HEAD 是 5 个 arch，多了 `sm_120f`**）、
  `ARCH`、`CLEAN_ONLY`、函数 `usage()`、`build_one()`。
- **模板参数全列表**：不适用。
- **PTX 宏与工具宏**：无 PTX。但**arch 是怎么给的**（原文逐行 @ `6c86259`）：
  ```
  ARCH_GENCODE[sm_86]="-gencode arch=compute_86,code=sm_86"
  ARCH_GENCODE[sm_89]="-gencode arch=compute_89,code=sm_89"
  ARCH_GENCODE[sm_90a]="-gencode arch=compute_90a,code=sm_90a"
  ARCH_GENCODE[sm_120a]="-gencode arch=compute_120a,code=sm_120a"
  ARCH_GENCODE[sm_120f]="-gencode arch=compute_120f,code=sm_120f"
  ```
  **每个 arch 的宏定义（原文逐行）**：
  ```
  ARCH_DEFINES[sm_86]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_CUDNN"
  ARCH_DEFINES[sm_89]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_CUDNN"
  ARCH_DEFINES[sm_90a]="-DNOTES_V2_ENABLE_WGMMA -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN"
  ARCH_DEFINES[sm_120a]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN"
  ARCH_DEFINES[sm_120f]="-DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN -DNOTES_V2_ENABLE_SETMAXNREGS -DNOTES_V2_FORCE_INLINE_ASYNC_PROXY"
  ```
  **重要事实**：只有 `sm_90a`/`sm_120a`/`sm_120f` 带 `NOTES_V2_ENABLE_TMA_MMA_WS`；
  `common.cuh` 里 TMA/mbarrier helper 的门控是
  `#if defined(NOTES_V2_ENABLE_WGMMA) || defined(NOTES_V2_ENABLE_TMA_MMA_WS)`。
  **`sm_120f` 是唯一打开 `NOTES_V2_ENABLE_SETMAXNREGS` + `NOTES_V2_FORCE_INLINE_ASYNC_PROXY` 的 arch**
  ——这两个宏分别控制 `setmaxnreg` 是否生效、以及 TMA helper 走裸 PTX 还是 `cuda::ptx::` 包装
  （原因见 `common.cuh` L469–488 与 L353–364 注释：ptxas 在 sm_120a 上会因 TMA 用法触发 C7506 丢掉 setmaxnreg）。
  **COMMON_FLAGS（原文，跨 arch 共用）**：
  ```
  -std=c++20
  -O3
  --expt-relaxed-constexpr
  --use_fast_math
  -I ../../third-party/cutlass/include
  -I ../../third-party/cudnn-frontend/include
  ```
  库与 stub 路径（5 个 arch 完全相同）：
  `ARCH_LIB_PATH[*]="-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"`、
  `ARCH_LIBS[*]="-lcublas -lcudnn -lnvrtc -lcuda"`
  输出名：
  `sm_86→notes_v2_sm86.bin`、`sm_89→notes_v2_cute_sm89.bin`、`sm_90a→notes_v2_sm90a.bin`、
  `sm_120a→notes_v2_sm120a.bin`、`sm_120f→notes_v2_sm120f.bin`
  ccache 环境（原文）：
  `CCACHE_COMPILERCHECK="${CCACHE_COMPILERCHECK:-content}"`、
  `CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-include_file_mtime,time_macros,locale,pch_defines}"`、
  `CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"`
- **kernel 的完整签名**：不适用。
- **块/线程/warp 组织**：不适用。
- **smem 布局与字节数**：不适用。
- **同步原语**：不适用（脚本层）。脚本本身 `set -euo pipefail`。
- **测试与容差**：不适用。
- **关键结论**（原样摘录）
  1. 两步流程原文：`1. ccache nvcc ... -c notes-v2.cu -o notes-v2.o   (cached)` / `2. nvcc notes-v2.o -o notes_v2_<arch>.bin ...  (uncached link)`
  2. `--arch all` 会按 `VALID_ARCHS` 顺序（sm_86 → sm_89 → sm_90a → sm_120a）依次构建。
  3. `--clean` 删除 `notes-v2.o` 与四个 `${ARCH_OUTPUT[$a]}`。
  4. 头注释里的 arch 说明原文：`./build.sh --arch sm_90a      # Hopper (H100/H200)`、`./build.sh --arch sm_120a     # Blackwell (RTX 5090 / PRO 5000/6000)`。
  5. 脚本注释里的 ccache 参考来源：`(ref: ffpa-attn/tools/build_fast.sh)`。

---

## kernels/interview/README.md

- **文件定位**：`interview/` 章节的门面文档——文件结构表、依赖图、编译与运行示例、性能基线表。
- **导出的符号**：无（Markdown）。关键内容为文件结构表：
  `common.cuh` / `base.cuh` / `sgemv.cuh` / `sgemm.cuh` / `hgemm.cuh` / `flash_attn.cuh` / `notes-v2.cu` 的职责说明。
- **模板参数全列表**：不适用。
- **PTX 宏与工具宏**：不适用（但给出了 `common.cuh` 的内容清单原文：
  「底层公共模块：CUDA 头文件、基础宏(`INT4`/`FLOAT4`/`HALF2`)、`kWarpSize`、MMA/WGMMA PTX 宏、XOR Swizzle 函数、TMA/mbarrier helpers、TensorMap helpers」）。
- **kernel 的完整签名**：不适用。
- **块/线程/warp 组织**：不适用。
- **smem 布局与字节数**：不适用。
- **同步原语**：不适用。
- **测试与容差 / 基线（原文照抄 @ `6c86259`，README 共 55 行）**
  - 依赖关系：`common.cuh ← base.cuh ← sgemv.cuh / sgemm.cuh / hgemm.cuh / flash_attn.cuh ← notes-v2.cu`
  - 注意：**当前 README 已删去「文件结构」表与 arch 列表**，只剩快速开始 + 两张基准表。
  - 编译（原文仅 2 条）：
    `./build.sh --arch sm_120a   # Blackwell (RTX 5090 / PRO 5000/6000, CUDA Toolkit >= 13.2)`；
    `./build.sh --help           # Show help for build options`
  - 环境准备原文：`apt remove -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13`；
    `apt install -y cudnn9-cuda-13 ccache # Also install ccache for faster rebuilds`
  - 复现命令：`./notes_v2_sm120a.bin --bench --mnk 4096,4096,4096 --bhnd 1,32,16384,128 # MMA ACC F16/F32 Acc`
    （运行环境描述：`e.g., NVIDIA PRO 5000, Blackwell SM_120a`）
  - **Max Err 列（容差相关的经验值，逐行原文；这就是“档”的真实形态，不是判据）** —— 只有 4 个不同值，按 variant 归类：
    - `0.000e+00`：全部 `HGEMM CuTe Swizzle (S=2/3, BLK_SW=0/1, F16Acc 与 F32Acc)`（8 行）。
    - `9.155e-05`：仅 `FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F16Acc)` 一行。
    - `1.831e-04`（**F16Acc 组**）：FA2 MMA Stages (Sk=1/2, Pad, F16Acc) 2 行；
      FA2 TMA MMA WS (1 Consumer WG) 的 (Sk=1, Sv=1)、(Sk=2, Sv=1)、(Sk=2, Sv=2) F16Acc 3 行。
    - `1.526e-05`（**F32Acc 组 + Split-D**）：FA2 MMA Stages (Sk=1/2, Pad, F32Acc)、FA2 CuTe MMA Stages (Sk=1/2)、
      FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F32Acc)、FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F32Acc)、
      FA2 CuTe TMA MMA WS (1 Consumer WG) (Sk=2/3, Sv=1, F32Acc)、FA2 CuTe TMA MMA Persistent-CTA WS (D=128)、
      FA Split-D CuTe TMA MMA WS (D=320, Sk=1, Sv=1) 与 (D=320, Sk=2, Sv=2)。
    → **一句话规律：F16Acc 累加 → 1.831e-04（FA3 的 F16Acc 因 split-KV 合并反而更好，9.155e-05）；
      F32Acc 累加 → 1.526e-05；HGEMM（无 softmax）→ 0。**
  - TFLOPS 基线**已更新**（与 `e831d97` 时代的旧表不同）：例如
    `FA3 TMA MMA WS (2 Consumer WG) (Sk=1, Sv=1, F16Acc)`：`305.2/222.9 (1.37x)` → **`210.1/232.4 (0.90x)`**；
    `FA2 TMA MMA WS (1 Consumer WG) (Sk=2, Sv=1, F16Acc)`：`292.0/222.9 (1.31x)` → **`189.4/232.4 (0.81x)`**；
    新增行 `FA2 CuTe TMA MMA Persistent-CTA WS (D=128) | 1.526e-05 | 242.5/232.4 (1.04x)`（对应新加的 persist-D kernel）；
    Split-D 段 `# Speedup: Split-D for large headdim (e.g, D=320) ~2.06x faster than cuDNN SDPA (with F32 Acc)`
    （旧值 ~2.20x），`(D=320, Sk=1, Sv=1) | 1.526e-05 | 96.5/70.3 (1.37x)`、`(D=320, Sk=2, Sv=2) | 1.526e-05 | 145.1/70.3 (2.06x)`；
    HGEMM 段 `| 0.000e+00 |` 对应 TFLOPS `213.0–246.7 / 163.7–236.9`（1.04–1.49x）。
    **写教程时若要引用绝对 TFLOPS，请用新表或注明“随 README 版本变动”；Max Err 列才是稳定事实。**
- **注释里的关键结论**（原样摘录）
  1. `notes-v2.cu`: 「面试中高频出现的 CUDA kernel 的背题版本。」（此句在当前 HEAD 的 README 中已不存在，仅存在于 `notes-v2.cu` 头部注释）
  2. `notes-v2.cu` 头部：`Phase 8 — FlashAttention-2split_q（FA-2, 含 online softmax + P@V 寄存器复用）`
  3. 快速开始段唯一强调点：`# Install the latest CUDNN library for benchmarks (remove the old version first)`

---

## kernels/interview/notes-v2.cu

- **文件定位**：整个章节的**唯一入口 TU**——`#include` 全部 `.cuh`，提供所有 test/bench 函数与 CLI 解析；
  **本身不定义任何 kernel**。
- **导出的符号（用 grep 精确统计，非通读）**
  - **`__global__` 命中数 = 0**。本文件没有任何 kernel 定义——所有 kernel 都定义在被 include 的 `.cuh` 里。
    （因此“kernel 名单”只能通过这份文件里的 test/bench 分发函数名来读，见下。）
  - test/bench 分发函数（`static`，逐条）：
    `test_flash_attn_3_cute_tma_copy_smoke`、`test_flash_attn_3_tma_mma_ws_split_q_cute`、
    `test_flash_attn_mma_stages_split_q_cute`、`test_flash_attn_tma_mma_ws_split_q_cute`、
    `test_flash_attn_cute_persist_d_sm120`、`test_block_reduce`、`test_dot`、`test_relu`、`test_elementwise`、
    `test_histogram`、`test_merge_attn_states`、`test_softmax`、`test_rms_norm`、`test_layer_norm`、
    `test_rope`、`test_mat_transpose`、`test_mat_transpose_padded`、`test_sgemv`、`test_sgemm`、
    `test_hgemm_mma`、`test_hgemm_swizzle`、`test_hgemm_cute`、`test_hgemm_wgmma`、`test_hgemm_tma_mma_ws`、
    `test_flash_attn`、`test_flash_attn_tma_mma_ws_impl`、`test_flash_attn_tma_mma_ws`、
    `test_flash_attn_3_tma_ws_impl`、`test_flash_attn_3_tma_ws`、
    **（新增）** `test_flash_attn_cute_persist_d_sm120`（L519）、
    `bench_hgemm_mma`、`bench_hgemm_swizzle`、`bench_hgemm_cute`、`bench_hgemm_wgmma`、`bench_hgemm_tma_mma_ws`、
    `bench_launch_tma_mma_ws`、`bench_fa_launch`、`bench_fa_2_mma_stages_cute_launch/dispatch`、
    `bench_fa_tma_mma_ws_launch/dispatch`、`bench_fa_3_tma_ws_launch/dispatch`、
    `bench_fa_3_tma_mma_ws_cute_launch/dispatch`、`bench_fa_2_tma_mma_ws_cute_launch/dispatch`、
    `bench_fa_persist_d_cute_launch`（L4227）、`bench_fa_split_d_launch/dispatch`、`bench_flash_attn`、
    `bench_cudnn_sdpa_tflops`（**定义两次**：L2508 与 L4290，各自被 `#if defined(NOTES_V2_ENABLE_CUDNN)` 门控）、
    `test_swizzle_equiv`、`check_smem_feasible`、`check`、`should_print_fa_tflops`、`should_print_hgemm_tflops`、
    `bench_hgemm_tflops`、`bench_fa_tflops`、`bench_cublas_hgemm_tflops`
    （`bench_ffpa_split_d_*` 不在本文件，在 `bench/bench_ffpa.cu` 独立 TU）
  - 全局开关变量：`g_debug`、`g_bench_hgemm`、`g_bench_hgemm_all`、`g_bench_fa`、`g_bench_fa3_cute_only`、
    `g_bench_all`、`g_fa_skip_check`、`g_swizzle_eq_check`、`g_fa_layout`（`enum class FALayout { All, Pad, SwizzleQ, SwizzleK, SwizzleV, SwizzleQK, SwizzleQV, SwizzleKV, Swizzle }`，默认 `FALayout::Pad`）、
    `g_bench_M/N/K = 8192`、`g_bench_B=1, g_bench_H=32, g_bench_Nfa=8192, g_bench_D=128`、
    `g_warmup=2, g_repeat=3`、`g_fa_f16_max_tflops`、`g_fa_f32_max_tflops`、`g_hgemm_f16_max_tflops`、
    `g_hgemm_f32_max_tflops`、`g_verbose`
  - **所有 `#define NOTES_V2_*` 开关名 = 0 条**。grep `^#define NOTES_V2_` 无命中：所有 `NOTES_V2_*` 都是
    **外部传入的编译宏（`build.sh` 里 `-D` 或 nvcc 命令行）**，本文件只做 `#if defined(...)`。
    实际出现过的开关名单（grep `NOTES_V2_ENABLE_`/`NOTES_V2_REG_` 去重）：
    `NOTES_V2_ENABLE_CUTE`、`NOTES_V2_ENABLE_CUDNN`、`NOTES_V2_ENABLE_WGMMA`、`NOTES_V2_ENABLE_TMA_MMA_WS`、
    `NOTES_V2_ENABLE_SWIZZLE_V2`、`NOTES_V2_FORCE_INLINE_ASYNC_PROXY`、`NOTES_V2_ENABLE_SETMAXNREGS`、
    `NOTES_V2_REG_ALLOC`、`NOTES_V2_REG_DEALLOC`（后两者是 `common.cuh` 里定义的调用宏）。
    **`NOTES_V2_ENABLE_TF32` 不存在**（grep 无命中）。
  - 文件头注释原文列出的 10 个 Phase 与“~30 个 kernel”：`Phase 0 — 面试框架速查`、`Phase 1 — 基础原语：Warp Reduce / Block Reduce / Dot Product（含 broadcast 增强版）`、
    `Phase 2 — Elementwise：ReLU / Elementwise Add / Histogram（基础 + float4 向量化 + atomic）`、
    `Phase 3 — Softmax：naive → safe → online + RMS/Layer Norm`、`Phase 4 — RoPE：旋转位置编码（Llama 风格 theta=10000）`、
    `Phase 5 — Mat Transpose：基础版 + BCF merge_write 最佳版（Bank Conflict专题）`、
    `Phase 6 — GEMV：SGEMV K32/K128/K16（warp-per-row）`、
    `Phase 7 — GEMM ★：SGEMM → HGEMM → MMA m16n8k16(TN布局) → WGMMA m64n128k16`、
    `Phase 8 — FlashAttention-2split_q（FA-2, 含 online softmax + P@V 寄存器复用）`
- **模板参数全列表（本文件内的模板函数，原样，行号 @ `6c86259`）**：全部是 test/bench 分发模板：
  `template <int kHeadDim>`（L101/156/276/386/2235）、
  `template <int kHeadDim, int kNq, int kNkv, int kHq = 2, int kHkv = 2, ...>`（L517，新加的 persist-D/QKV test）、
  `template <int kStages, int kBlockSwizzle = 0>`（L1951）、`template <int kHeadDim, int kStagesK>`（L2512/3478）、
  `template <int kStages, int kBlockSwizzle>`（L2716/2837/3067/3203）、
  `template <int kHeadDim, int kStagesK = 2, int kPadQ = 8, int kPadK = 8, ...>`（L3342，即 `bench_fa_launch`；**默认 kPad 三兄弟都是 8**）、
  `template <int kHeadDim, int kStagesK, int kStagesV = 1, int kMmaAccF32 = 0>`（L3604，`bench_fa_tma_mma_ws_launch`）、
  `template <int kStagesK, int kStagesV = 1, int kMmaAccF32 = 0>`（L3745）、
  `template <int kHeadDim, int kStagesK, int kMmaAccF32 = 0>`（L3765）、`template <int kMmaAccF32 = 0>`（L3902）、
  `template <int kHeadDim, int kStagesK = 1>`（L3936）、`template <int kHeadDim, int kStagesK, int kStagesV = 1>`（L4071）、
  `template <int kHeadDim, int kStagesQK, int kStagesV>`（L4355）、`template <int kColStride>`（L4911）
- **PTX 宏与工具宏**：无自有 PTX 宏；只用 `common.cuh` 的宏与 `check()` / `check_smem_feasible()`。
- **kernel 的完整签名**：本文件**无 kernel**。
- **块/线程/warp 组织**：只体现在 launch 配置里，例：
  TMA MMA WS FA：`dim3 block(kNumThreads);  // 384`、`dim3 grid((seqlen + Br - 1) / Br, B * H);`；
  `smem_bytes = (Br * kHeadDim + kStagesK * Bc * kHeadDim + kStagesV * Bc * kHeadDim) * sizeof(half);`
  CuTe FA2/FA3：`dim3 grid(seqlen / kBr, B * H);`
  手写 FA2：`dim3 grid((seqlen + Br - 1) / Br, B * H);`
- **smem 布局与字节数**：通过 `check_smem_feasible((const void *)fa_k, smem_bytes)` 判定，
  再用 `cudaFuncSetAttribute(fa_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes)`。
- **同步原语**：本文件只用 `cudaStreamSynchronize` / `cudaEventSynchronize` / `cudaDeviceSynchronize`。
- **测试与容差**
  - **判据唯一值：`max_err >= 5e-1f`** @ `6c86259` 命中行：L3446、L3561、L3710、L3867、L4029、L4177、L4267
    （其中 L3561/L4029/L4177/L4267 带 `checked &&` 前缀）。
    **没有 TF32 档，没有 1e-2/1e-3 档，没有 atol/rtol 常量。**
  - 参考实现：CPU FP32（`srand(42)`，`scale = 1.0f / sqrtf((float)head_dim)`，`double` 累加）
    + cuDNN SDPA（cudnn-frontend graph）。
  - 打印门槛（原文注释）：`only print when the current TFLOPS exceeds the running max for its accumulator category (f16/f32). Correctness failures always print so they are never silently dropped`。
  - CLI：`--bench-hgemm`、`--bench`、`--bench-fa`、`--bench-fa3-cute`、`--bench-hgemm-all`、`--bench-fa-all`、
    `--bench-all`、`--mnk M,N,K`、`--bhnd B,H,N,D`、`--fa-layout <...>`、`--fa-skip-check`、`--swizzle-eq-check`、
    `--debug`、`--verbose`、`--warmup`、`--repeat`；另有 `--tma-mma-ws M N K`（默认 128 128 64）。
  - 构建示例（文件尾原文）：`nvcc -std=c++20 -O2 -arch=sm_120a -DNOTES_V2_ENABLE_TMA_MMA_WS -lcublas -lcuda notes-v2.cu -o notes_v2_tma_mma_ws_sm120.bin`；
    `nvcc -std=c++20 -O2 -arch=sm_120a -DNOTES_V2_ENABLE_CUTE -DNOTES_V2_ENABLE_TMA_MMA_WS -DNOTES_V2_ENABLE_CUDNN -I ../../third-party/cutlass/include -I ../../third-party/cudnn-frontend/include -L/usr/local/cuda-13.2/targets/x86_64-linux/lib/stubs -lcublas -lcudnn -lnvrtc -lcuda notes-v2.cu -o notes_v2_cute_ws_sm120a.bin`
  - 运行示例：`./notes_v2_cute_ws_sm120a.bin --bench --bench-fa --bhnd 1,48,4096,64`；
    `--bench --bench-fa-all --bhnd 1,48,4096,64`；`--bench-fa-tma-ws --bhnd 1,32,4096,64`
- **注释里的关键结论**（原样摘录）
  1. `// 整理自 LeetCUDA 项目（https://github.com/xlite-dev/LeetCUDA），涵盖：- 面试高频 CUDA kernel 的完整实现（~30 个 kernel）`
  2. `// BLAS 语义：N=col-major(Normal), T=row-major(Transposed)`
  3. `// 以下是测试代码，验证 Phase 1 - Phase 8 的kernel的正确性，不评估性能。`
  4. `// ★ 多 head K/V offset: local_tile coord 需加 blockIdx.y * kv_tiles 偏移，否则所有 head 都读 head 0 的 K/V (参考 TMA 版 L6624 注释)`
  5. `// Default FA bench (kPadQ=kPadK=kPadV=8 only)`（印证 kPad 默认值是 8）

---

## 交叉核对：三处“同一事实”的不同表述（写教程时要注意）

1. **tol**：仓库里 FA/GEMM 的**唯一硬判据是 `5e-1f`**；README 里的 `1.831e-04 / 1.526e-05 / 9.155e-05 / 0.000e+00`
   是**实测 Max Err 报告值**，不是容差阈值。“F32Acc / F16Acc 两档经验误差 + 一个统一判据”才是准确表述；
   **TF32 档在这个仓库的 `interview/` 里不存在**。
2. **producer/consumer 划分有四种，别混**（全部 @ `6c86259`）：
   - FA2 TMA WS（`flash_attn_tma_mma_ws_stages_split_q`）：`128 producer + 256 consumer = 384`，
     1 个 consumer WG（8 warps），全 KV 遍历，Br=128。
   - FA3 TMA WS（`flash_attn_3_tma_ws_stages_split_q`）：`128 producer + 2×128 consumer = 384`，
     2 个 consumer WG 按 **tile 奇偶** 分 KV，Br=Bc=64，需 split-KV 合并（alpha/beta 公式）。
   - FFPA TMA WS（`ffpa_attn_tma_mma_ws_split_d_cute`）：`128 producer + 128 consumer = 256`（Large-D Split-D）。
   - persist-D WS（`flash_attn_cute_persist_d_sm120`，新增）：`128T producer + 256T consumer = 384`。
   - `ws-hgemm/naive_ws_hgemm_sm8x.cu`：`32 producer + 128 consumer = 160`，用 `cuda::pipeline` 而非 mbarrier。
3. **smem 对齐**：FA TMA 路径必须 `__align__(1024)`（消费者用软件 `swizzle<64>`，零 phase 假设）；
   WGMMA 路径只需 `alignas(128)`（descriptor 的 `base_offset` 位域会自动补偿 phase）。
   这条在 `hgemm.cuh` L1898–1919 有完整论证。
4. **barrier arrive_count 有三种数**：FA2 = `256 + 1 = 257`；FA3 = `full_Q` 256+1、
   per-WG 的 K/V = `128 + 1 = 129`；FFPA = `TmaBarrier::init(&qk_full[stage], 1)`（**1，不是 N+1**）
   + `CtaBarrier::init(&qk_empty[stage], kConsumerThreads=128)`。

---

## 提取过程中发现的、值得在教程里点名的“注释/代码不一致”

1. `flash_attn.cuh` 头注释写 `Block: (128, 1, 1)，kNumThreads=kWarpSize×kMmaTileSeqLenQ×kMmaTileSeqLenK=128`，
   而同一行公式在代码里带 `// 32*8*1=256`；`Br = kMmaAtomM*kMmaTileSeqLenQ*kValTileSeqLenQ` 在不同实例化下为 64 或 128。
   头注释的 `Grid: ((N + 63) / 64, B * H, 1)，Br=64` 也只对应 Br=64 的那组实例化。
2. `flash_attn.cuh` L126–127 明确指出 `Tc = (N + Bc - 1) / Bc` 的 ceil 写法「是为了说明 tile 划分方式，不等于当前实现已经完整处理了尾 tile」。
3. `notes-v2.cu` **L2508 与 L4290 两处定义同名 `bench_cudnn_sdpa_tflops`**（不同 `#if` 门控下的不同签名），
   阅读/引用时必须区分是哪一处。
4. 仓库在本次提取期间被外部更新（HEAD `e831d97` → `6c86259`）：`flash_attn.cuh` 新增了
   `flash_attn_cute_persist_d_sm120`（persistent-CTA，L3633–3635，`__launch_bounds__(384, 1)`，
   `128T producer + 256T consumer`，参数含 `int Nq, int Nkv, int Nh, int Nh_kv, float scale, int Tc, int causal, int q_tiles, int total_q_tiles, int total_q_rows, int total_kv_rows`）
   与 `flash_attn_cute_persist_d_sm120_launch<>`；`notes-v2.cu` 新增 `test_flash_attn_cute_persist_d_sm120`
   与 `bench_fa_persist_d_cute_launch`；`build.sh` 新增 `sm_120f` arch（唯一打开
   `NOTES_V2_ENABLE_SETMAXNREGS` + `NOTES_V2_FORCE_INLINE_ASYNC_PROXY`）。
   **本任务要求的 8 个文件里的“原内容”未变**，但教程若涉及 persistent-CTA / sm_120f，需要补这一层。
5. 引用任何行号前请先重新 grep：该仓库 `kernels/interview/` 正在被活跃修改。


---

