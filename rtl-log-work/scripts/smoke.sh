#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_dir/build"
repo_tmp_dir="$build_dir/tmp"
mkdir -p "$build_dir" "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"
cd "$repo_dir"

iverilog -g2012 -s lpu_smoke_tb -o "$build_dir/lpu_smoke.vvp" -f sim/filelist.f
vvp "$build_dir/lpu_smoke.vvp"
iverilog -g2012 -s lpu_arch_tb -o "$build_dir/lpu_arch.vvp" -f sim/filelist.f
vvp "$build_dir/lpu_arch.vvp"
