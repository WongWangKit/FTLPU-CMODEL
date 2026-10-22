#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
vcs_build_dir="$repo_dir/build/vcs"
repo_tmp_dir="$repo_dir/build/tmp"
license_spec=${LM_LICENSE_FILE:-27000@andromeda}
export LM_LICENSE_FILE="$license_spec"
mkdir -p "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"

# The legacy license client can briefly refuse a second checkout immediately
# after a simulation exits. Retry only the compiler checkout; test failures
# from a generated simv still stop the regression immediately.
vcs_compile() {
  local attempt
  for attempt in 1 2 3; do
    if vcs "$@"; then
      return 0
    fi
    if (( attempt < 3 )); then
      sleep 2
    fi
  done
  return 1
}

source_files=()
while IFS= read -r source_file; do
  if [[ -n "$source_file" && "$source_file" != \#* ]]; then
    source_files+=("$repo_dir/$source_file")
  fi
done < "$repo_dir/sim/filelist.f"

vcs_test() {
  local top=$1
  local name=$2
  local test_dir="$vcs_build_dir/$name"
  mkdir -p "$test_dir"
  ln -sfn "$repo_dir/sim" "$test_dir/sim"
  (
    cd "$test_dir"
    vcs_compile -full64 -sverilog -timescale=1ns/1ps \
      -LDFLAGS -Wl,--no-as-needed \
      -top "$top" "${source_files[@]}" \
      -o "$test_dir/simv" -Mdir="$test_dir/csrc"
    "$test_dir/simv"
  )
}

"$repo_dir/scripts/generate_cmodel_vectors.sh"

vcs_test lpu_smoke_tb smoke
vcs_test lpu_arch_tb arch
vcs_test lpu_mem_stream_tb mem_stream
vcs_test lpu_sxm_transpose_tb sxm_transpose
vcs_test lpu_sxm_wavefront_tb sxm_wavefront
vcs_test lpu_vxm_control_tb vxm_control
vcs_test lpu_vxm_icu_map_tb vxm_icu_map
vcs_test lpu_vxm_alu_tb vxm_alu
vcs_test lpu_vxm_input_converter_tb vxm_input_converter
vcs_test lpu_vxm_fp16_stream_groups_tb vxm_fp16_stream_groups
vcs_test lpu_vxm_datapath_mux_tb vxm_datapath_mux
vcs_test lpu_vxm_execution_stage_tb vxm_execution_stage
vcs_test lpu_vxm_significand_multiplier_tb vxm_significand_multiplier
vcs_test lpu_vxm_special_coefficient_multiplier_tb vxm_special_coefficient_multiplier
vcs_test lpu_vxm_recip_linear_fixed_tb vxm_recip_linear_fixed
vcs_test lpu_vxm_rsqrt_linear_fixed_tb vxm_rsqrt_linear_fixed
vcs_test lpu_vxm_fp32_rsqrt_correction_tb vxm_fp32_rsqrt_correction
vcs_test lpu_vxm_segmented_addsub_tb vxm_segmented_addsub
vcs_test lpu_vxm_grouped_lzc_tb vxm_grouped_lzc
vcs_test lpu_vxm_shared_left_shift_tb vxm_shared_left_shift
vcs_test lpu_vxm_basic_frontend_tb vxm_basic_frontend
vcs_test lpu_vxm_shared_float_compare_tb vxm_shared_float_compare
vcs_test lpu_vxm_addsub_max_instruction_tb vxm_addsub_max_instruction
vcs_test lpu_vxm_tile_pair_lut_tb vxm_tile_pair_lut
vcs_test lpu_vxm_16_alu_tb vxm_16_alu
vcs_test lpu_vxm_instruction_controller_tb vxm_instruction_controller
vcs_test lpu_vxm_tile_execution_tb vxm_tile_execution
vcs_test lpu_vxm_tile_fp32_phase_tb vxm_tile_fp32_phase
vcs_test lpu_vxm_tile_special_paths_tb vxm_tile_special_paths
vcs_test lpu_vxm_lane_chain_tb vxm_lane_chain
vcs_test lpu_vxm_superlane_lockstep_tb vxm_superlane_lockstep
vcs_test lpu_vxm_slice_tb vxm_slice
vcs_test lpu_vxm_slice_fp32_special_tb vxm_slice_fp32_special
vcs_test lpu_vxm_slice_bf16_special_tb vxm_slice_bf16_special
vcs_test lpu_mxm_vector_tb mxm_vector
vcs_test lpu_mxm_fp16_tb mxm_fp16
vcs_test lpu_mxm_fault_tb mxm_fault
vcs_test lpu_mxm_accumulator_tb mxm_accumulator
vcs_test lpu_mxm_column_tb mxm_column
vcs_test lpu_mxm_int8_dequant_tb mxm_int8_dequant
vcs_test lpu_mxm_block8_tb mxm_block8
vcs_test lpu_mxm_weight_stream_tb mxm_weight_stream
