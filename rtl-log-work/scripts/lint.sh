#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_tmp_dir="$repo_dir/build/tmp"
mkdir -p "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"
cd "$repo_dir"

verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_smoke_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_arch_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_mem_stream_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_alu_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_input_converter_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_fp16_stream_groups_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_datapath_mux_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_execution_stage_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_significand_multiplier_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_special_coefficient_multiplier_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_recip_linear_fixed_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_rsqrt_linear_fixed_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_fp32_rsqrt_correction_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_segmented_addsub_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_grouped_lzc_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_shared_left_shift_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_basic_frontend_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_shared_float_compare_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_addsub_max_instruction_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_tile_pair_lut_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_16_alu_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_instruction_controller_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_tile_execution_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_tile_fp32_phase_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_tile_special_paths_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_slice_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_slice_fp32_special_tb
verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist.f --top-module lpu_vxm_slice_bf16_special_tb

verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal \
  -f sim/filelist_vxm_log.f --top-module lpu_vxm_log_tb

verilator --lint-only --Wall -Wno-DECLFILENAME -Wno-fatal --timing \
  -f sim/filelist_vxm_c3_pipeline.f --top-module lpu_vxm_c3_pipeline_tb
