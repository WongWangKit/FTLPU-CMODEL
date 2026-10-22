#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cmodel_dir=${FTLPU_CMODEL_DIR:-"$repo_dir/../FTLPU-CMODEL"}
build_dir="$repo_dir/build/smollm2_ffn"
vector_build_dir="$build_dir/cmodel"
vcs_build_dir="$build_dir/vcs"
repo_tmp_dir="$repo_dir/build/tmp"
generator="$vector_build_dir/smollm2_ffn"
license_spec=${LM_LICENSE_FILE:-27000@andromeda}

mkdir -p "$vector_build_dir" "$vcs_build_dir/csrc" "$repo_tmp_dir"
export LM_LICENSE_FILE="$license_spec"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" -I "$repo_dir/sim/cmodel" \
  "$repo_dir/sim/cmodel/smollm2_ffn.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$generator"

"$generator" \
  "$repo_dir/sim/vectors/smollm2_ffn_init.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_golden.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_gate_schedule.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_swiglu_schedule.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_down_schedule.hex"

source_files=()
while IFS= read -r source_file; do
  if [[ -n "$source_file" && "$source_file" != \#* ]]; then
    source_files+=("$repo_dir/$source_file")
  fi
done < "$repo_dir/sim/filelist.f"

ln -sfn "$repo_dir/sim" "$vcs_build_dir/sim"
(
  cd "$vcs_build_dir"
  vcs -full64 -sverilog -timescale=1ns/1ps \
    -LDFLAGS -Wl,--no-as-needed \
    -top lpu_smollm2_ffn_tb "${source_files[@]}" \
    -o "$vcs_build_dir/simv" -Mdir="$vcs_build_dir/csrc"
  "$vcs_build_dir/simv"
)
