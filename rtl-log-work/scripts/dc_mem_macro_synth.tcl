set repo_dir [file normalize [file join [file dirname [info script]] ..]]
set build_dir [file join $repo_dir build dc_mem_macro]
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
set sram_db [file join $repo_dir build tech arm28_sram sram_64_2048 sram_64_2048_tt_ctypical_0p90v_0p90v_25c.db]

foreach db_file [list $stdcell_db $sram_db] {
  if {![file readable $db_file]} {
    error "Required library is not readable: $db_file"
  }
}

set_app_var target_library [list $stdcell_db]
set_app_var link_library [list * $stdcell_db $sram_db dw_foundation.sldb]

set rtl_files [list \
  [file join $repo_dir rtl core lpu_isa_decode.sv] \
  [file join $repo_dir rtl mem lpu_sram_1rw_banked.sv] \
  [file join $repo_dir rtl mem lpu_mem_tile_slice.sv] \
  [file join $repo_dir rtl mem lpu_mem_tile_slice_sram.sv] \
  [file join $repo_dir rtl mem lpu_mem_tile_slice_select.sv]]

analyze -format sverilog -define FTLPU_USE_SRAM64X2048_MACRO $rtl_files
elaborate lpu_mem_tile_slice -parameters \
  "DEPTH_ROWS=65536,LANES=8,USE_SRAM_MACRO=1"
link

set macro_cells [get_cells -hierarchical -filter "ref_name == sram_64_2048"]
if {[sizeof_collection $macro_cells] != 32} {
  error "Expected 32 sram_64_2048 instances, found [sizeof_collection $macro_cells]"
}
set_dont_touch $macro_cells

create_clock -name clk -period 3.0 [get_ports clk_i]
set_clock_uncertainty 0.1 [get_clocks clk]
set_input_delay 0.2 -clock clk \
  [remove_from_collection [all_inputs] [get_ports clk_i]]
set_output_delay 0.2 -clock clk [all_outputs]
set_false_path -from [get_ports rst_ni]

redirect -file [file join $report_dir check_design.rpt] {check_design}
redirect -file [file join $report_dir precompile_references.rpt] {
  report_reference -hierarchy
}

compile -map_effort medium -area_effort medium

set mapped_macro_cells \
  [get_cells -hierarchical -filter "ref_name == sram_64_2048"]
if {[sizeof_collection $mapped_macro_cells] != 32} {
  error "Mapped design lost SRAM macros: found [sizeof_collection $mapped_macro_cells]"
}

redirect -file [file join $report_dir area.rpt] {report_area -hierarchy}
redirect -file [file join $report_dir timing.rpt] {
  report_timing -max_paths 20
}
redirect -file [file join $report_dir qor.rpt] {report_qor}
redirect -file [file join $report_dir references.rpt] {
  report_reference -hierarchy
}
redirect -file [file join $report_dir macro_instances.rpt] {
  puts "sram_64_2048 instances: [sizeof_collection $mapped_macro_cells]"
  foreach_in_collection macro_cell $mapped_macro_cells {
    puts [get_object_name $macro_cell]
  }
}

write -format ddc -hierarchy \
  -output [file join $output_dir lpu_mem_tile_slice_macro.ddc]
write -format verilog -hierarchy \
  -output [file join $output_dir lpu_mem_tile_slice_macro_mapped.v]
write_sdc [file join $output_dir lpu_mem_tile_slice_macro.sdc]
puts "MEMORY_MACRO_SYNTHESIS_PASS"
quit
