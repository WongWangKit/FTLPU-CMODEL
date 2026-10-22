#!/usr/bin/env bash
set -euo pipefail

# Focused native-format BYPASS/NEGATE semantics checks.
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tmp_dir="$repo_dir/build/tmp"
mkdir -p "$tmp_dir"
export TMPDIR="$tmp_dir"
export TEMP="$tmp_dir"
export TMP="$tmp_dir"
export LM_LICENSE_FILE="${LM_LICENSE_FILE:-27000@andromeda}"

source_files=()
while IFS= read -r source_file; do
  if [[ -n "$source_file" && "$source_file" != \#* ]]; then
    source_files+=("$repo_dir/$source_file")
  fi
done < "$repo_dir/sim/filelist.f"

for top in lpu_vxm_input_converter_tb lpu_vxm_execution_stage_tb; do
  test_dir="$repo_dir/build/vcs/raw_ops_${top}"
  mkdir -p "$test_dir"
  (
    cd "$test_dir"
    vcs -full64 -sverilog -timescale=1ns/1ps \
      -LDFLAGS -Wl,--no-as-needed \
      -top "$top" "${source_files[@]}" \
      -o "$test_dir/simv" -Mdir="$test_dir/csrc" \
      2>&1 | tee "$test_dir/compile.log"
    "$test_dir/simv" 2>&1 | tee "$test_dir/run.log"
  )
done
