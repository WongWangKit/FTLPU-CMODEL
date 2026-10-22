#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
python3 sim/cmodel/generate_log_vectors.py
python3 sim/cmodel/generate_c3_pipeline_vectors.py
for stage in 3 7 11 15; do
  build_dir="build/verilator/vxm_c3_pipeline_${stage}"
  mkdir -p "$build_dir"
  verilator --binary --timing --assert -Wno-fatal --timescale 1ns/1ps \
    --top-module lpu_vxm_c3_pipeline_tb -GPHYSICAL_STAGE="$stage" \
    -f sim/filelist_vxm_c3_pipeline.f --Mdir "$build_dir" \
    >"$build_dir/compile.log" 2>&1
  "$build_dir/Vlpu_vxm_c3_pipeline_tb" | tee "$build_dir/run.log"
done
