#!/usr/bin/env bash
# build.sh — build & run the hand-written kernel exercises (Q176-Q190).
#
#   ./build.sh                 # build all
#   ./build.sh q183            # build one
#   ./build.sh --run q183      # build + run
#   ./build.sh --run all       # build + run everything
#   ./build.sh --clean
#   ./build.sh --arch sm_90    # override the target architecture
#
# nvcc is located in this order: $CUDA_HOME/bin, $CUDA_PATH/bin, `which nvcc`,
# then the usual toolkit install dirs. If none is found the script prints the
# two ways to get one (admin install, or the no-admin portable kit described in
# README.md) and exits non-zero — it never pretends to have compiled.
set -uo pipefail

cd "$(dirname "$0")"
KERNEL_DIR="kernels"
BUILD_DIR="build"
ARCH="${ARCH:-}"
RUN=0
TARGETS=()

# ---------------------------------------------------------------- args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run|-r) RUN=1 ;;
    --clean) rm -rf "$BUILD_DIR"; echo "cleaned $BUILD_DIR"; exit 0 ;;
    --arch) shift; ARCH="$1" ;;
    --arch=*) ARCH="${1#--arch=}" ;;
    all) TARGETS=() ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) TARGETS+=("$1") ;;
  esac
  shift
done

# ---------------------------------------------------------------- find nvcc
find_nvcc() {
  local c
  for c in "${CUDA_HOME:-}/bin/nvcc" "${CUDA_PATH:-}/bin/nvcc"; do
    [[ -n "$c" && -x "$c" ]] && { echo "$c"; return; }
  done
  if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return; fi
  for c in /usr/local/cuda/bin/nvcc \
           "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"/*/bin/nvcc.exe \
           "$LOCALAPPDATA/Programs/NVIDIA GPU Computing Toolkit/CUDA"/*/bin/nvcc.exe; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
}
NVCC="$(find_nvcc)"

if [[ -z "$NVCC" ]]; then
  cat >&2 <<'EOF'
ERROR: nvcc not found.

Two ways to get one:
  A) Standard install (needs admin):
       winget install Nvidia.CUDA      # or download the toolkit and run the installer
  B) No-admin portable kit (what this repo used to type-check the kernels):
       1. download the CUDA local installer (a 7z self-extracting archive)
       2. 7z x cuda_<ver>_windows.exe -o<cuda_extract>
       3. copy cuda_nvcc/nvcc/{bin,nvvm}, cuda_cudart/cudart/{include,lib}, and the
          CCCL headers into one <kit>/vX.Y tree  (see README.md for the exact list)
       4. export CUDA_HOME=<kit>/vX.Y

Once nvcc is on PATH (or CUDA_HOME is set), re-run ./build.sh
EOF
  exit 2
fi

echo "nvcc: $NVCC"
"$NVCC" --version | tail -2

# ---------------------------------------------------------------- pick arch
if [[ -z "$ARCH" ]]; then
  # Prefer the real device; fall back to a common compute capability.
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    [[ -n "$CC" ]] && ARCH="sm_${CC}"
  fi
  ARCH="${ARCH:-sm_80}"
fi
echo "arch: $ARCH"
mkdir -p "$BUILD_DIR"

# ---------------------------------------------------------------- collect
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  mapfile -t SRCS < <(ls "$KERNEL_DIR"/q*.cu 2>/dev/null | sort)
else
  SRCS=()
  for t in "${TARGETS[@]}"; do
    [[ "$t" == all ]] && { mapfile -t SRCS < <(ls "$KERNEL_DIR"/q*.cu | sort); break; }
    [[ "$t" == *.cu ]] && f="$KERNEL_DIR/$t" || f="$KERNEL_DIR/${t}.cu"
    [[ -f "$f" ]] || f="$(ls "$KERNEL_DIR"/${t}*.cu 2>/dev/null | head -1)"
    [[ -f "$f" ]] && SRCS+=("$f") || echo "skip: no source for '$t'" >&2
  done
fi
[[ ${#SRCS[@]} -eq 0 ]] && { echo "no sources to build" >&2; exit 1; }

# ---------------------------------------------------------------- build
NVCC_FLAGS=(-std=c++17 -O3 -lineinfo -Xcompiler -Wall --expt-relaxed-constexpr)
[[ "$ARCH" == sm_90* ]] && NVCC_FLAGS+=(-gencode "arch=compute_90a,code=${ARCH}a")
if [[ -n "${EXTRA_NVCC_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  NVCC_FLAGS+=($EXTRA_NVCC_FLAGS)
fi

FAILED=0
for src in "${SRCS[@]}"; do
  name="$(basename "${src%.cu}")"
  out="$BUILD_DIR/$name"
  printf '\n=== %s ===\n' "$name"
  # shellcheck disable=SC2086
  if ! "$NVCC" "${NVCC_FLAGS[@]}" -I"$KERNEL_DIR" -arch="$ARCH" -o "$out" "$src" \
       2> "$BUILD_DIR/$name.build.log"; then
    echo "BUILD FAILED — see $BUILD_DIR/$name.build.log"
    tail -25 "$BUILD_DIR/$name.build.log"
    FAILED=$((FAILED + 1))
    continue
  fi
  echo "built -> $out"
  if [[ $RUN -eq 1 ]]; then
    ( cd "$BUILD_DIR" && "./$name" ) || echo "run failed: $name"
  fi
done

echo
if [[ $FAILED -gt 0 ]]; then
  echo "$FAILED target(s) failed to build"
  exit 1
fi
echo "all targets built (arch=$ARCH)"
[[ $RUN -eq 0 ]] && echo "run them with:  ./build.sh --run all"
exit 0
