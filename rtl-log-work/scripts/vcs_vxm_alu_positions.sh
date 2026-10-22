#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_root="$repo_dir/build/vcs/vxm_alu_positions"
repo_tmp_dir="$repo_dir/build/tmp"
coverage_kind="line+cond+branch+fsm+tgl+assert"
license_spec=${LM_LICENSE_FILE:-27000@andromeda}
export LM_LICENSE_FILE="$license_spec"
mkdir -p "$build_root" "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"

source_files=()
while IFS= read -r source_file; do
  if [[ -n "$source_file" && "$source_file" != \#* ]]; then
    source_files+=("$repo_dir/$source_file")
  fi
done < "$repo_dir/sim/filelist.f"

for stage in $(seq 0 15); do
  top="lpu_vxm_alu${stage}_tb"
  test_dir="$build_root/alu${stage}"
  coverage_dir="$test_dir/coverage.vdb"
  mkdir -p "$test_dir"
  echo "[VXM ALU$stage] compiling $top"
  (
    cd "$test_dir"
    vcs -full64 -sverilog -timescale=1ns/1ps \
      -cm "$coverage_kind" -cm_dir "$coverage_dir" \
      -top "$top" "${source_files[@]}" \
      -o "$test_dir/simv" -Mdir="$test_dir/csrc" \
      -l "$test_dir/compile.log"
    echo "[VXM ALU$stage] running $top"
    "$test_dir/simv" -no_save \
      -cm "$coverage_kind" -cm_dir "$coverage_dir" \
      -l "$test_dir/run.log"
  )
done

echo "VXM_ALU_POSITION_REGRESSION_PASS"
