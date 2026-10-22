#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$repo_dir/build/dc_full_macro"
repo_tmp_dir="$repo_dir/build/tmp"
mkdir -p "$build_dir" "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir" TEMP="$repo_tmp_dir" TMP="$repo_tmp_dir"
cd "$repo_dir"
dc_shell -64bit -f "$repo_dir/scripts/dc_full_macro_synth.tcl" \
  -output_log_file "$build_dir/dc.log"
