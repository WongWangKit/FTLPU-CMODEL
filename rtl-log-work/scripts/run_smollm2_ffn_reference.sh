#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cmodel_dir=${FTLPU_CMODEL_DIR:-"$repo_dir/../FTLPU-CMODEL"}
build_dir="$repo_dir/build/cmodel_smollm2"
repo_tmp_dir="$repo_dir/build/tmp"

mkdir -p "$build_dir" "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"

cmake -S "$cmodel_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" \
  --target smollm2_block8_dequant_ffn_test -j"${JOBS:-2}"
"$build_dir/smollm2_block8_dequant_ffn_test"
