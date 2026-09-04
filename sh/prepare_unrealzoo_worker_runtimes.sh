#!/usr/bin/env bash
set -euo pipefail

# 创建每个 worker 独立的最小 UE runtime。仅复制可执行文件和 unrealcv.ini；
# 72 GB 的 Engine 与游戏 Content 继续使用只读软链接共享。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${WORKERS:-4}"
SOURCE_RUNTIME="${SOURCE_RUNTIME:-${PROJECT_ROOT}/unrealzoo/Linux/UnrealZoo_UE5_6_Linux_v3.0.0}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT}/output/runtime/unreal_env_workers}"

if [[ ! "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 2
fi

source_linux="${SOURCE_RUNTIME}/Linux"
source_game="${source_linux}/UnrealZoo_UE5_6"
source_bin="${source_game}/Binaries/Linux/UnrealZoo_UE5_6"
source_ini="${source_game}/Binaries/Linux/unrealcv.ini"

for path in "${source_linux}/Engine" "${source_game}/Content" "${source_bin}" "${source_ini}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required UE runtime path is missing: ${path}" >&2
    exit 1
  fi
done

for ((slot = 0; slot < WORKERS; slot++)); do
  game_root="${RUNTIME_ROOT}/worker${slot}/Linux/UnrealZoo_UE5_6"
  bin_root="${game_root}/Binaries/Linux"
  bin="${bin_root}/UnrealZoo_UE5_6"
  mkdir -p "${bin_root}" "${game_root}/Saved" "${game_root}/Intermediate"
  ln -sfn "${source_linux}/Engine" "${RUNTIME_ROOT}/worker${slot}/Linux/Engine"
  ln -sfn "${source_game}/Content" "${game_root}/Content"
  if [[ ! -e "${bin}" ]]; then
    cp --reflink=auto "${source_bin}" "${bin}"
    chmod u+x "${bin}"
  fi
  if [[ ! -e "${bin_root}/unrealcv.ini" ]]; then
    cp "${source_ini}" "${bin_root}/unrealcv.ini"
  fi
  printf '[runtime] worker=%s binary=%s\n' "${slot}" "${bin}"
done
