#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_tmp_dir="$repo_dir/build/tmp"
mkdir -p "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"
export LM_LICENSE_FILE=${LM_LICENSE_FILE:-27000@andromeda}

source_files=()
while IFS= read -r source_file; do
  if [[ -n "$source_file" && "$source_file" != \#* ]]; then
    source_files+=("$repo_dir/$source_file")
  fi
done < "$repo_dir/sim/filelist.f"

run_test() {
  local top=$1
  local name=$2
  local test_dir="$repo_dir/build/vcs/$name"
  mkdir -p "$test_dir"
  cd "$test_dir"
  vcs -full64 -sverilog -timescale=1ns/1ps \
    -top "$top" "${source_files[@]}" \
    -o "$test_dir/simv" -Mdir="$test_dir/csrc" \
    -l "$test_dir/compile.log"
  "$test_dir/simv" -no_save -l "$test_dir/run.log"
}

run_test lpu_vxm_lane_chain_tb vxm_lane_chain
run_test lpu_vxm_superlane_lockstep_tb vxm_superlane_lockstep
