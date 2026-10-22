#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_dir/build/rsqrt_precision"
mkdir -p "$build_dir"
g++ -std=c++20 -O3 "$repo_dir/sim/cmodel/vxm_rsqrt_precision.cpp" \
  -o "$build_dir/vxm_rsqrt_precision"
"$build_dir/vxm_rsqrt_precision"
