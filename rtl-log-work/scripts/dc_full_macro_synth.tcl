set repo_dir [file normalize [file join [file dirname [info script]] ..]]
set build_dir [file join $repo_dir build dc_full_macro]
set report_dir [file join $build_dir reports]
set output_dir [file join $build_dir output]
set work_dir [file join $build_dir work]
file mkdir $report_dir $output_dir $work_dir
define_design_lib WORK -path $work_dir

set stdcell_db /home/public/t28/t28/stdcell/TSMCHOME/digital/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp12t30p140_180a/tcbn28hpcplusbwp12t30p140tt0p9v25c_ccs.db
set mem_db [file join $repo_dir build tech arm28_sram sram_64_2048 sram_64_2048_tt_ctypical_0p90v_0p90v_25c.db]
set acc_db [file join $repo_dir build tech arm28_sram sram_128_512 sram_128_512_tt_ctypical_0p90v_0p90v_25c.db]
foreach db_file [list $stdcell_db $mem_db $acc_db] {
  if {![file readable $db_file]} { error "Required library is not readable: $db_file" }
}
set_app_var target_library [list $stdcell_db]
set_app_var link_library [list * $stdcell_db $mem_db $acc_db dw_foundation.sldb]

set relative_rtl_files [list \
  rtl/core/lpu_pkg.sv rtl/core/lpu_isa_decode.sv \
  rtl/core/lpu_stream_stage.sv rtl/core/lpu_control_pipeline.sv \
  rtl/icu/lpu_icu_queue.sv rtl/icu/lpu_icu.sv \
  rtl/mem/lpu_sram_1rw_banked.sv rtl/mem/lpu_mem_tile_slice.sv \
  rtl/mem/lpu_mem_tile_slice_sram.sv rtl/mem/lpu_mem_tile_slice_select.sv \
  rtl/mem/lpu_mem_column.sv rtl/mem/lpu_mem_hemisphere.sv \
  rtl/vxm/lpu_vxm_math_pkg.sv rtl/vxm/lpu_vxm_fp16_pkg.sv \
  rtl/vxm/lpu_vxm_input_converter.sv \
  rtl/mxm/lpu_mxm_control.sv \
  rtl/mxm/lpu_mxm_dequantizer.sv rtl/mxm/lpu_mxm_weight_buffer.sv \
  rtl/mxm/lpu_mxm_dot_bank.sv rtl/mxm/lpu_mxm_compute.sv \
  rtl/mxm/lpu_mxm_accumulator_sram.sv \
  rtl/mxm/lpu_mxm_accumulator.sv rtl/mxm/lpu_mxm_block_accumulator.sv \
  rtl/mxm/lpu_mxm_shared_accumulator.sv rtl/mxm/lpu_mxm_slice.sv \
  rtl/sxm/lpu_sxm_control.sv rtl/sxm/lpu_sxm_slice.sv \
  rtl/vxm/lpu_vxm_global_config.sv rtl/vxm/lpu_vxm_control.sv \
  rtl/vxm/lpu_vxm_fp16_stream_groups.sv \
  rtl/vxm/lpu_vxm_local_decoder.sv rtl/vxm/lpu_vxm_datapath_mux.sv \
  rtl/vxm/lpu_vxm_datapath_stage.sv \
  rtl/vxm/lpu_vxm_instruction_controller.sv \
  rtl/vxm/lpu_vxm_mul8x8.sv rtl/vxm/lpu_vxm_csa3_2.sv \
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
  rtl/vxm/lpu_vxm_lut_sram.sv rtl/vxm/lpu_vxm_tile_pair_lut.sv \
  rtl/vxm/lpu_vxm_log_math_pkg.sv \
  rtl/vxm/lpu_vxm_c3_math_pkg.sv \
  rtl/vxm/lpu_vxm_c3_pipeline.sv \
  rtl/vxm/lpu_vxm_special_alu.sv \
  rtl/vxm/lpu_vxm_alu.sv \
  rtl/vxm/lpu_vxm_execution_stage.sv rtl/vxm/lpu_vxm_tile_execution.sv \
  rtl/vxm/lpu_vxm_stream_bridge.sv rtl/vxm/lpu_vxm_slice.sv rtl/lpu_top.sv]
set rtl_files [list]
foreach rtl_file $relative_rtl_files {
  lappend rtl_files [file join $repo_dir $rtl_file]
}

analyze -format sverilog \
  -define {FTLPU_USE_SRAM64X2048_MACRO FTLPU_USE_SRAM128X512_MACRO FTLPU_DISABLE_BLOCK8} \
  $rtl_files
elaborate lpu_top -parameters \
  "ICU_QUEUE_DEPTH=16,MEM_DEPTH_ROWS=65536,ACTIVE_MEM_COLUMNS=52,MXM_ACCUMULATOR_BLOCK_COUNT=32,USE_SRAM_MACRO=1"
current_design lpu_top
link

set mem_macros [get_cells -hierarchical -filter "ref_name == sram_64_2048"]
set acc_macros [get_cells -hierarchical -filter "ref_name == sram_128_512"]
if {[sizeof_collection $mem_macros] != 13312} {
  error "Expected 13312 MEM macros, found [sizeof_collection $mem_macros]"
}
if {[sizeof_collection $acc_macros] != 64} {
  error "Expected 64 ACC macros, found [sizeof_collection $acc_macros]"
}
set_dont_touch [add_to_collection $mem_macros $acc_macros]

create_clock -name clk -period 3.0 [get_ports clk_i]
set_clock_uncertainty 0.1 [get_clocks clk]
set_input_delay 0.2 -clock clk [remove_from_collection [all_inputs] [get_ports clk_i]]
set_output_delay 0.2 -clock clk [all_outputs]
set_false_path -from [get_ports rst_ni]
redirect -file [file join $report_dir check_design.rpt] {check_design}
redirect -file [file join $report_dir precompile_references.rpt] {report_reference -hierarchy}
compile -map_effort low -area_effort low

set mapped_mem_macros [get_cells -hierarchical -filter "ref_name == sram_64_2048"]
set mapped_acc_macros [get_cells -hierarchical -filter "ref_name == sram_128_512"]
if {[sizeof_collection $mapped_mem_macros] != 13312 ||
    [sizeof_collection $mapped_acc_macros] != 64} {
  error "Mapped design lost SRAM macros"
}
redirect -file [file join $report_dir area.rpt] {report_area -hierarchy}
redirect -file [file join $report_dir timing.rpt] {report_timing -max_paths 20}
redirect -file [file join $report_dir qor.rpt] {report_qor}
redirect -file [file join $report_dir references.rpt] {report_reference -hierarchy}
redirect -file [file join $report_dir macro_instances.rpt] {
  puts "sram_64_2048 instances: [sizeof_collection $mapped_mem_macros]"
  puts "sram_128_512 instances: [sizeof_collection $mapped_acc_macros]"
}
write -format ddc -hierarchy -output [file join $output_dir lpu_top_full_macro.ddc]
write -format verilog -hierarchy -output [file join $output_dir lpu_top_full_macro_mapped.v]
write_sdc [file join $output_dir lpu_top_full_macro.sdc]
puts "FULL_MEMORY_MACRO_SYNTHESIS_PASS"
quit
