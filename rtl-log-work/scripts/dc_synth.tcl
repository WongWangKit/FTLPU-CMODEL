set repo_dir [file normalize [file join [file dirname [info script]] ..]]
set build_dir [file join $repo_dir build dc]
set report_dir [file join $build_dir reports]
set output_dir [file join $build_dir output]
set work_dir [file join $build_dir work]
file mkdir $report_dir $output_dir $work_dir

define_design_lib WORK -path $work_dir

set default_stdcell_db /home/public/t28/t28/stdcell/TSMCHOME/digital/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp12t30p140_180a/tcbn28hpcplusbwp12t30p140tt0p9v25c_ccs.db
if {[info exists ::env(FTLPU_STDCELL_DB)]} {
  set stdcell_db [file normalize $::env(FTLPU_STDCELL_DB)]
} else {
  set stdcell_db $default_stdcell_db
}
if {![file readable $stdcell_db]} {
  error "Standard-cell library is not readable: $stdcell_db"
}

set macro_libraries [list]
foreach macro_db [list \
  [file join $repo_dir build tech arm28_sram sram_64_2048 sram_64_2048_tt_ctypical_0p90v_0p90v_25c.db] \
  [file join $repo_dir build tech arm28_sram sram_32_4096 sram_32_4096_tt_ctypical_0p90v_0p90v_25c.db]] {
  if {[file readable $macro_db]} {
    lappend macro_libraries $macro_db
  }
}

set_app_var target_library [list $stdcell_db]
set_app_var link_library [concat [list * $stdcell_db] $macro_libraries \
  [list dw_foundation.sldb]]
set_app_var hdlin_enable_hier_map true

set rtl_files [list \
  rtl/core/lpu_pkg.sv \
  rtl/core/lpu_isa_decode.sv \
  rtl/core/lpu_stream_stage.sv \
  rtl/core/lpu_control_pipeline.sv \
  rtl/icu/lpu_icu_queue.sv \
  rtl/icu/lpu_icu.sv \
  rtl/mem/lpu_sram_1rw_banked.sv \
  rtl/mem/lpu_mem_tile_slice.sv \
  rtl/mem/lpu_mem_tile_slice_sram.sv \
  rtl/mem/lpu_mem_tile_slice_select.sv \
  rtl/mem/lpu_mem_column.sv \
  rtl/mem/lpu_mem_hemisphere.sv \
  rtl/vxm/lpu_vxm_math_pkg.sv \
  rtl/vxm/lpu_vxm_fp16_pkg.sv \
  rtl/vxm/lpu_vxm_input_converter.sv \
  rtl/mxm/lpu_mxm_control.sv \
  rtl/mxm/lpu_mxm_dequantizer.sv \
  rtl/mxm/lpu_mxm_weight_buffer.sv \
  rtl/mxm/lpu_mxm_dot_bank.sv \
  rtl/mxm/lpu_mxm_compute.sv \
  rtl/mxm/lpu_mxm_accumulator_sram.sv \
  rtl/mxm/lpu_mxm_shared_accumulator.sv \
  rtl/mxm/lpu_mxm_accumulator.sv \
  rtl/mxm/lpu_mxm_block_accumulator.sv \
  rtl/mxm/lpu_mxm_slice.sv \
  rtl/sxm/lpu_sxm_control.sv \
  rtl/sxm/lpu_sxm_slice.sv \
  rtl/vxm/lpu_vxm_global_config.sv \
  rtl/vxm/lpu_vxm_control.sv \
  rtl/vxm/lpu_vxm_fp16_stream_groups.sv \
  rtl/vxm/lpu_vxm_local_decoder.sv \
  rtl/vxm/lpu_vxm_datapath_mux.sv \
  rtl/vxm/lpu_vxm_datapath_stage.sv \
  rtl/vxm/lpu_vxm_instruction_controller.sv \
  rtl/vxm/lpu_vxm_mul8x8.sv \
  rtl/vxm/lpu_vxm_csa3_2.sv \
  rtl/vxm/lpu_vxm_significand_multiplier.sv \
  rtl/vxm/lpu_vxm_shared_float_multiplier.sv \
  rtl/vxm/lpu_vxm_special_coefficient_multiplier.sv \
  rtl/vxm/lpu_vxm_recip_linear_fixed.sv \
  rtl/vxm/lpu_vxm_fp32_newton_multiplier.sv \
  rtl/vxm/lpu_vxm_rsqrt_linear_fixed.sv \
  rtl/vxm/lpu_vxm_fp32_rsqrt_correction.sv \
  rtl/vxm/lpu_vxm_float_unpack.sv \
  rtl/vxm/lpu_vxm_basic_opcode_decode.sv \
  rtl/vxm/lpu_vxm_shared_float_compare.sv \
  rtl/vxm/lpu_vxm_segmented_addsub.sv \
  rtl/vxm/lpu_vxm_grouped_lzc.sv \
  rtl/vxm/lpu_vxm_shared_left_shift.sv \
  rtl/vxm/lpu_vxm_shared_float_adder.sv \
  rtl/vxm/lpu_vxm_basic_alu.sv \
  rtl/vxm/lpu_vxm_lut_sram.sv \
  rtl/vxm/lpu_vxm_tile_pair_lut.sv \
  rtl/vxm/lpu_vxm_log_math_pkg.sv \
  rtl/vxm/lpu_vxm_c3_math_pkg.sv \
  rtl/vxm/lpu_vxm_c3_pipeline.sv \
  rtl/vxm/lpu_vxm_special_alu.sv \
  rtl/vxm/lpu_vxm_alu.sv \
  rtl/vxm/lpu_vxm_execution_stage.sv \
  rtl/vxm/lpu_vxm_tile_execution.sv \
  rtl/vxm/lpu_vxm_stream_bridge.sv \
  rtl/vxm/lpu_vxm_slice.sv \
  rtl/lpu_top.sv]

set absolute_rtl_files [list]
foreach rtl_file $rtl_files {
  lappend absolute_rtl_files [file join $repo_dir $rtl_file]
}

analyze -format sverilog $absolute_rtl_files
# Keep the architectural datapaths intact while using a small SRAM/ICU
# configuration for a practical library-mapped smoke synthesis. The full-size
# SRAM arrays are intended to be replaced by technology memory macros.
elaborate lpu_top -parameters "ICU_QUEUE_DEPTH=4,MEM_DEPTH_ROWS=64,ACTIVE_MEM_COLUMNS=52"
current_design lpu_top
link

create_clock -name clk -period 2.0 [get_ports clk_i]
set_clock_uncertainty 0.1 [get_clocks clk]
set_input_delay 0.2 -clock clk [remove_from_collection [all_inputs] [get_ports clk_i]]
set_output_delay 0.2 -clock clk [all_outputs]
set_false_path -from [get_ports rst_ni]

redirect -file [file join $report_dir check_design.rpt] {check_design}
redirect -file [file join $report_dir precompile_area.rpt] {report_area -hierarchy}
compile -map_effort low -area_effort low

redirect -file [file join $report_dir area.rpt] {report_area -hierarchy}
redirect -file [file join $report_dir timing.rpt] {report_timing -max_paths 20}
redirect -file [file join $report_dir qor.rpt] {report_qor}
redirect -file [file join $report_dir references.rpt] {report_reference -hierarchy}

write -format ddc -hierarchy -output [file join $output_dir lpu_top.ddc]
write -format verilog -hierarchy -output [file join $output_dir lpu_top_mapped.v]
write_sdc [file join $output_dir lpu_top.sdc]
quit
