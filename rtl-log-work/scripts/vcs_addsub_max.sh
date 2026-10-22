#!/usr/bin/env bash
set -euo pipefail

# Focused port-level ADD/SUB/MAX/MUL regression; no other VXM tests are run.
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
test_dir="$repo_dir/build/vcs/vxm_addsub_max_instruction"
tmp_dir="$repo_dir/build/tmp"
mkdir -p "$test_dir" "$tmp_dir"
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

(
  cd "$test_dir"
  vcs -full64 -sverilog -timescale=1ns/1ps \
    -LDFLAGS -Wl,--no-as-needed \
    -top lpu_vxm_addsub_max_instruction_tb \
    "${source_files[@]}" \
    -o "$test_dir/simv" -Mdir="$test_dir/csrc" \
    2>&1 | tee "$test_dir/compile.log"
  "$test_dir/simv" 2>&1 | tee "$test_dir/run.log"
)
