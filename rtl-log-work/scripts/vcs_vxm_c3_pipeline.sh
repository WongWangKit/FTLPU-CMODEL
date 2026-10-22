#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
python3 sim/cmodel/generate_log_vectors.py
python3 sim/cmodel/generate_c3_pipeline_vectors.py
for stage in 3 7 11 15; do
  build_dir="build/vcs/vxm_c3_pipeline_${stage}"
  mkdir -p "$build_dir"
  vcs -full64 -sverilog -timescale=1ns/1ps -top lpu_vxm_c3_pipeline_tb \
    -pvalue+lpu_vxm_c3_pipeline_tb.PHYSICAL_STAGE="$stage" \
    -f sim/filelist_vxm_c3_pipeline.f -Mdir="$build_dir/csrc" \
    -o "$build_dir/simv" -l "$build_dir/compile.log"
  "$build_dir/simv" -no_save -l "$build_dir/run.log"
done
