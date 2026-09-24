<#
.SYNOPSIS
  Build & run the hand-written kernel exercises (Q176-Q190) on Windows.

.EXAMPLE
  .\build.ps1                      # build all
  .\build.ps1 q183                 # build one
  .\build.ps1 -Run q183            # build + run
  .\build.ps1 -Run all             # build + run everything
  .\build.ps1 -Arch sm_89
  .\build.ps1 -Clean

  # 一般用法（Windows PowerShell 里若被策略拦住，改用 bash 版 build.sh）
  powershell -ExecutionPolicy Bypass -File .\build.ps1 -Run all

.NOTES
  nvcc 的查找顺序：$env:CUDA_HOME\bin、$env:CUDA_PATH\bin、PATH、常见安装目录。
  找不到时脚本会明确报错并给出两条获取途径（管理员安装 / 免安装便携套件），
  **不会假装编译成功**。
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
  [string[]]$Targets = @(),
  [switch]$Run,
  [switch]$Clean,
  [string]$Arch = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$KernelDir = "kernels"
$BuildDir = "build"

if ($Clean) {
  if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
  Write-Host "cleaned $BuildDir"
  exit 0
}

# ---------------------------------------------------------------- find nvcc
function Find-Nvcc {
  $cands = @()
  if ($env:CUDA_HOME) { $cands += (Join-Path $env:CUDA_HOME "bin\nvcc.exe") }
  if ($env:CUDA_PATH) { $cands += (Join-Path $env:CUDA_PATH "bin\nvcc.exe") }
  $cmd = Get-Command nvcc -ErrorAction SilentlyContinue
  if ($cmd) { $cands += $cmd.Source }
  $cands += (Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nvcc.exe" -ErrorAction SilentlyContinue | ForEach-Object FullName)
  $cands += (Get-ChildItem "$env:LOCALAPPDATA\Programs\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nvcc.exe" -ErrorAction SilentlyContinue | ForEach-Object FullName)
  foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
  return $null
}

$nvcc = Find-Nvcc
if (-not $nvcc) {
  Write-Host "ERROR: 找不到 nvcc。" -ForegroundColor Red
  Write-Host @"

两条途径：
  A) 标准安装（需要管理员）：
       winget install Nvidia.CUDA
     或下载 CUDA Toolkit 安装包直接装。

  B) 免管理员便携套件（本仓库用来做类型检查的方式）：
     1. 下载 CUDA local installer（本质是 7z 自解压包）
     2. 7z x cuda_<ver>_windows.exe -o<cuda_extract>
     3. 把 cuda_nvcc\nvcc\{bin,nvvm}、cuda_cudart\cudart\{include,lib} 以及
        CCCL 头文件拷进同一个 <kit>\vX.Y 目录树（README.md 里有完整清单）
     4. `$env:CUDA_HOME = "<kit>\vX.Y"

装好后重跑 .\build.ps1
"@
  exit 2
}
Write-Host "nvcc: $nvcc"
& $nvcc --version | Select-Object -Last 2

# ---------------------------------------------------------------- arch
if (-not $Arch) {
  try {
    $cc = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($cc) { $Arch = "sm_" + ($cc -replace '[^0-9]', '') }
  } catch { }
  if (-not $Arch) { $Arch = "sm_80" }
}
Write-Host "arch: $Arch"
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# ---------------------------------------------------------------- sources
$all = Get-ChildItem "$KernelDir\q*.cu" | Sort-Object Name
$srcs = @()
if ($Targets.Count -eq 0 -or ($Targets -contains "all")) {
  $srcs = $all
} else {
  foreach ($t in $Targets) {
    $hit = $all | Where-Object { $_.BaseName -eq $t -or $_.BaseName -like "$t*" } | Select-Object -First 1
    if ($hit) { $srcs += $hit } else { Write-Host "skip: no source for '$t'" -ForegroundColor Yellow }
  }
}
if ($srcs.Count -eq 0) { Write-Host "nothing to build" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- build
$baseFlags = @("-std=c++17", "-O3", "-lineinfo", "-Xcompiler", "/W3", "--expt-relaxed-constexpr")
$failed = 0
foreach ($s in $srcs) {
  Write-Host "`n=== $($s.BaseName) ==="
  $out = Join-Path $BuildDir "$($s.BaseName).exe"
  $log = Join-Path $BuildDir "$($s.BaseName).build.log"
  $args = $baseFlags + @("-I$KernelDir", "-arch=$Arch", "-o", $out, $s.FullName)
  $p = Start-Process -FilePath $nvcc -ArgumentList $args -Wait -PassThru -NoNewWindow `
       -RedirectStandardError $log
  if ($p.ExitCode -ne 0 -or -not (Test-Path $out)) {
    Write-Host "BUILD FAILED — see $log" -ForegroundColor Red
    Get-Content $log -Tail 25
    $failed++
    continue
  }
  Write-Host "built -> $out"
  if ($Run) {
    Push-Location $BuildDir
    try { & ".\$($s.BaseName).exe" } finally { Pop-Location }
  }
}

Write-Host ""
if ($failed -gt 0) { Write-Host "$failed target(s) failed to build" -ForegroundColor Red; exit 1 }
Write-Host "all targets built (arch=$Arch)"
if (-not $Run) { Write-Host "run them with:  .\build.ps1 -Run all" }
exit 0
