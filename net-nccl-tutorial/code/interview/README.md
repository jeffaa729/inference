# interview-code —— §7 手撕题的完整代码

> **`interview.md` 的 §7 是从这里的 `.cu` 文件自动生成的**（`build_kernels.py`），
> 所以你在 Markdown / PDF 里看到的代码，就是编译器看到的那份，不会漂移。

## 目录

| 文件 | 内容 |
|---|---|
| `kernels/lc_common.cuh` | 共享头：**宏与工具函数对齐 LeetCUDA `kernels/interview/common.cuh`**（`FLOAT4`/`HALF2`、`CP_ASYNC_*`、`LDMATRIX_*`、`HMMA16816`、`permuted`/`SwizzleBMS`、`make_smem_desc`/`tma_*`、`warpgroup_reg_*`）+ 测试脚手架（计时、对拍、容差、设备自检） |
| `kernels/q176_vecadd.cu` … `q190_merge_attn.cu` | 15 道手撕题，每题**一个可独立编译的完整程序** |
| `build.sh` / `build.ps1` | 编译运行（Linux/macOS 与 Windows），自动找 nvcc、自动取本机 arch |
| `build_kernels.py` | 把 `.cu` 嵌进 `interview.md` §7（含 **LeetCUDA 溯源表**；`--check` 可验证是否同步） |

## 参考实现：LeetCUDA

这 15 道题的**参考实现在 [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA)**
（本地 checkout `C:\Users\Jeff\Documents\GitHub\LeetCUDA`，**引用锚点 HEAD `6c86259`**（2026-09-22））。
`interview.md` §7.0 有一张逐题溯源表，告诉你每题去看哪份实现。

**本套件的宏命名与工具函数刻意与 LeetCUDA 的 `kernels/interview/common.cuh` 逐字对齐**，
所以在两边看到的写法是同一套；Q181 的 swizzle 版转置更是与
`kernels/swizzle/mat_trans_swizzle.cu` 的 `mat_trans_smem_swizzle_kernel` 同构。

### 与 LeetCUDA 的 7 处【有意差异】（别以为是抄错）

1. **容差**：LeetCUDA 的 `.py` 里**没有 tol、没有 `allclose`、没有任何断言** —— 它们是纯 benchmark，
   正确性靠人眼看打印出来的前 3 个值；它的 `notes-v2.cu` 对 attention 也只有
   一个 `max_err >= 5e-1f` 的失败判据。**「LeetCUDA 有三档容差」是个常见误解** ——
   那是它 README 里的**实测 Max Err 报告值**（F16Acc ~1.83e-4、F32Acc ~1.53e-5、FA3 双 WG ~9.16e-5）。
   **本套件自定义三档判据（1e-3 / 5e-2 / 1e-2）+ CPU 对拍**，判据硬得多；
   同时 `lc_common.cuh` 里保留了 `TOL_LEETCUDA_FAIL` 与三个 `OBS_*` 常量方便对照。
2. LeetCUDA 的 `build.sh` 带 `--use_fast_math`；**本套件没带**，所以容差是保守的。
3. **越界读守卫**：LeetCUDA 的 `LDST128BITS`/`FLOAT4` 载入一律无守卫
   （`*_x16_pack` 连累加循环都没有 `(idx+i) < N`）；**本套件统一二段式守卫**，
   所以能安全地跑非 4/8 倍数的 N —— 你抄去生产时这一点很关键。
4. **partial warp 哨兵**：LeetCUDA 的 softmax `case 32` 会拿 8 线程的 block 去跑
   全掩码 `__shfl_xor_sync(0xffffffff, …)`（**UB**）；本套件要么保证 block 是 32 的整数倍，
   要么让边界线程携带单位元。
5. **参考实现的口径**：LeetCUDA `layer_norm.py` 的 `naive_layer_norm` 用无偏 `std`（除 K-1）
   而 kernel 用有偏 `variance/K`；`rms_norm.py` 的参考实现没有 eps 而 kernel 有 `1e-5`。
   **本套件的 CPU 参考与 kernel 口径一致**，所以对拍是干净的。
6. **不依赖 cuBLAS/cuDNN**（LeetCUDA 的 bench 会跟它们比）。手撕题不需要基线库；
   想比就把 `kernels/sgemm/sgemm_cublas.cu` 那类基线自己接上。
7. **不复制 LeetCUDA 的拼写与已知的注释-代码不一致**：`LANUCH_*`（应为 LAUNCH）、
   `STRINGFY`、`DISPATCH_SATE_*`；`sgemm.cu` 有一行以 `+` 开头当空语句；
   bf16 reduce 注释说"跨 warp 用 fp32"但代码是 bf16；`cp.async` 那版的
   `wait_group` 参数恒为 0（所以流水深度其实没拉开）。**引用时会如实指出，不照抄。**

## 每个 `.cu` 文件的结构

```
// qNNN_xxx.cu — QNNN：题目
// 编译：./build.sh qNNN
// ─────────────────────────────────────────────
// 解析要点        ← 面试官在看的那些点（为什么这么写、口径、坑）
// ─────────────────────────────────────────────
#include "common.cuh"
__global__ void 参考实现 / 优化实现 ...   ← 2~4 个版本，对照着看
int main() { ... }                        ← 对拍 + 边界用例 + 带宽/TFLOPS + 结论打印
```

**每个 `main()` 都会做四件事**：① 与 CPU 参考对拍（带容差口径）；
② 跑极端/边界形状（非 4 倍数、全负数、不整除……）；③ 计时并打印带宽或 TFLOPS；
④ 打印这段代码要说明的结论。

## 快速开始

```bash
# Linux / macOS / WSL
cd interview-code
./build.sh --run all          # 编译 + 跑全部 15 个
./build.sh --run q183         # 只跑 tiled SGEMM
./build.sh --arch sm_90       # 指定架构
./build.sh --clean
```

```powershell
# Windows
cd interview-code
.\build.ps1 -Run all
.\build.ps1 -Run q183
powershell -ExecutionPolicy Bypass -File .\build.ps1 -Run all   # 被策略拦住时
```

`nvcc` 的查找顺序：`$CUDA_HOME/bin` → `$CUDA_PATH/bin` → `PATH` → 常见安装目录。
**找不到时脚本会直接报错并给出两条获取途径，不会假装编译成功。**

## 没有 CUDA 也能做的两件事

```bash
python check_kernel_sources.py    # 静态检查（在仓库上一级目录）
python ../interview-code/build_kernels.py --check   # §7 是否与 .cu 同步
```

`check_kernel_sources.py` 检查 10 类问题：花括号/括号配平、块注释未闭合、
CUDA 调用漏错误检查、kernel 定义了但没 launch、`__shared__` 写后读缺
`__syncthreads()`、`__shfl_*_sync` 的 mask 与提前 return 冲突、sm_90 指令缺
架构守卫、浮点用 `==` 比较、指针参数缺 `__restrict__`。

**它不是编译器** —— 只能挡住"结构性错误"，正确性必须靠 `./build.sh --run all`
在你自己的卡上验证。

## 关于本机的编译验证（诚实说明）

本仓库的写作环境**没有装 CUDA Toolkit**，`nvcc` 需要管理员权限才能装。
所以这 15 个文件的状态是：

| 项目 | 状态 |
|---|---|
| 结构检查（配平 / 同步 / 错误检查 / launch 覆盖 …） | ✅ `check_kernel_sources.py` 无 finding |
| 与 §7 同步 | ✅ `build_kernels.py --check` 通过 |
| **编译** | ⚠️ **未在本机验证** —— 需要在有 CUDA 的机器上跑 `./build.sh` |
| **运行 / 数值 / 性能数字** | ⚠️ **未在本机验证** —— 需要 `./build.sh --run all` |

本机只有一张 RTX 4060 Laptop 8 GB（`sm_89`），
**TMA / WGMMA / TMEM 的路径（q187）在它上面本来也跑不了**，
q187 出厂就带 `SKIP_TMA_IMPL` 守卫，在 sm_90 以下会打印说明并跳过。

代码里引用的所有**性能数字都标了口径**（来自那本 459 页 kernel 专著的实测，
测试机 RTX PRO 5000 72GB / `sm_120a`），**不是本机测的，也不可直接迁移**。

## 免管理员装一个 nvcc（可选）

如果只想做类型检查而不装整个 Toolkit，可以解包 CUDA 安装器自己拼一个便携套件：

```powershell
# 1) 下载 CUDA local installer（本质是 7z 自解压包）
#    https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_561.17_windows.exe
# 2) 用 7zr.exe（588 KB，免安装）解包
.\7zr.exe x cuda_12.6.3_installer.exe -o<cuda_extract>
# 3) 拼装：把下面这些目录拷进同一个 <kit>\v12.6
#      cuda_nvcc\nvcc\bin          -> <kit>\v12.6\bin        (nvcc, ptxas, fatbinary...)
#      cuda_nvcc\nvcc\nvvm          -> <kit>\v12.6\nvvm
#      cuda_nvcc\nvcc\include       -> <kit>\v12.6\include    (crt/*.h)
#      cuda_cudart\cudart\include   -> <kit>\v12.6\include    (cuda_runtime.h, cuda_fp16.h...)
#      cuda_cccl\thrust\include     -> <kit>\v12.6\include
#      cuda_cudart\cudart\lib\x64   -> <kit>\v12.6\lib\x64
#      cuda_cudart\cudart\bin       -> <kit>\v12.6\bin        (cudart64_12.dll)
# 4) 还需要 MSVC 的 cl.exe 与 Windows SDK（nvcc 的 host 编译器）
$env:CUDA_HOME = "<kit>\v12.6"
```

**一个已知的坑**：CUDA 12.6 的 `cudafe++` 与 VS 2022 v17.14+（MSVC 14.44+）的 STL
不兼容 —— 会报 `error STL1002: Unexpected compiler version, expected CUDA 13.2 or newer`，
或直接 `cudafe++ died with status 0xC0000005`。
**要么装 CUDA 13.2+，要么装一个更老的 MSVC toolset**（两者都需要管理员）。
这也是本仓库没能在本机完成编译验证的原因。
