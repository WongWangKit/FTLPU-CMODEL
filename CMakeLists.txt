cmake_minimum_required(VERSION 3.20)

project(FTLPU_CMODEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The header-only ICU/system models instantiate enough Debug COMDAT sections
# to exceed COFF's default section count. /bigobj changes only the object-file
# container limit; it does not alter modeled timing or runtime behavior.
if(MSVC)
    add_compile_options(/bigobj)
endif()

set(FTLPU_HARDWARE_CONFIG
    "${CMAKE_CURRENT_SOURCE_DIR}/config/ftlpu-lpu32.json"
    CACHE FILEPATH
    "Shared FTLPU hardware target JSON used by the CModel and software")
set(FTLPU_GENERATED_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
include(cmake/FTLPUHardwareConfig.cmake)
ftlpu_generate_hardware_config(
    "${FTLPU_HARDWARE_CONFIG}"
    "${FTLPU_GENERATED_INCLUDE_DIR}/ftlpu/core/hardware_config.hpp")

add_library(ftlpu_cmodel STATIC
    src/mem/sram.cpp
    src/mxm/accumulator.cpp
    src/system/stream_topology_builder.cpp
)
target_include_directories(ftlpu_cmodel PUBLIC
    "${FTLPU_GENERATED_INCLUDE_DIR}"
    include)

enable_testing()

function(ftlpu_add_test target source test_name)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ftlpu_cmodel)
    # VXM/SXM and MEM timing tests use assert for cycle and numeric checks.
    # Keep those checks active in the normal Release regression build too.
    if(target MATCHES "^(vxm_|sxm_|icu_sxm_|mem_|vector_roundtrip)")
        if(MSVC)
            target_compile_options(${target} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${target} PRIVATE -UNDEBUG)
        endif()
    endif()
    target_include_directories(
        ${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${target}>)
    set_tests_properties(${test_name} PROPERTIES LABELS "${ARGN}")
endfunction()

function(ftlpu_add_demo target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ftlpu_cmodel)
endfunction()

# Unit tests: one class or functional block, without a whole-chip route.
ftlpu_add_test(hardware_profile_test tests/integration/hardware_profile_test.cpp
    integration.hardware_profile integration core fast)
ftlpu_add_test(instruction_codec_test tests/unit/core/instruction_codec_test.cpp
    unit.core.instruction_codec unit core fast)
# Header-only MEM port/SXM codec coverage also builds when another FU's ISA
# explicitly rejects the experimental target geometry.
add_executable(geometry_encoding_test tests/unit/core/geometry_encoding_test.cpp)
target_include_directories(geometry_encoding_test PRIVATE
    "${FTLPU_GENERATED_INCLUDE_DIR}" include)
add_test(NAME unit.core.geometry_encoding COMMAND $<TARGET_FILE:geometry_encoding_test>)
set_tests_properties(unit.core.geometry_encoding PROPERTIES LABELS "unit;core;sxm;fast")
ftlpu_add_test(bf16_test tests/unit/core/bf16_test.cpp
    unit.core.bf16 unit core fast)
ftlpu_add_test(wide_transport_codec_test tests/unit/core/wide_transport_codec_test.cpp
    unit.core.wide_transport_codec unit core mxm c2c fast)
ftlpu_add_test(timing_test tests/unit/core/timing_test.cpp
    unit.core.timing unit core timing fast)
ftlpu_add_test(sram_1rw_test tests/unit/core/sram_1rw_test.cpp
    unit.core.sram_1rw unit core timing fast)
ftlpu_add_test(sram_fifo_test tests/unit/core/sram_fifo_test.cpp
    unit.core.sram_fifo unit core timing fast)

ftlpu_add_test(stream_fabric_test tests/unit/stream/stream_fabric_test.cpp
    unit.stream.fabric unit stream fast)
ftlpu_add_test(stream_port_test tests/unit/stream/stream_port_test.cpp
    unit.stream.port unit stream fast)
ftlpu_add_test(stream_fabric_clock_test
    tests/unit/stream/stream_fabric_clock_test.cpp
    unit.stream.clocked_fabric unit stream timing fast)
ftlpu_add_test(stream_fabric_cycle_trace_test
    tests/unit/stream/stream_fabric_cycle_trace_test.cpp
    unit.stream.cycle_trace unit stream timing)

ftlpu_add_test(mem_slice_test tests/unit/mem/slice_test.cpp
    unit.mem.slice unit mem fast)
ftlpu_add_test(mem_array_separated_test tests/unit/mem/mem_array_test.cpp
    unit.mem.array unit mem fast)
ftlpu_add_test(mem_replay_copy_test tests/unit/mem/replay_copy_test.cpp
    unit.mem.replay_copy unit mem fast)
ftlpu_add_test(mem_sync_read_test tests/unit/mem/mem_sync_read_test.cpp
    unit.mem.sync_read unit mem fast)
ftlpu_add_test(mem_array_r1_test tests/unit/mem/mem_array_test.cpp
    unit.mem.array_r1 unit mem fast)
target_compile_definitions(mem_array_r1_test PRIVATE FTLPU_TEST_MEM_R1=1)
ftlpu_add_test(mem_read_mode_test tests/unit/mem/mem_read_mode_test.cpp
    unit.mem.read_mode unit mem fast)
ftlpu_add_test(mem_sr_cycle_trace_test tests/unit/mem/mem_sr_cycle_trace_test.cpp
    unit.mem.sr_cycle_trace unit mem stream timing)
ftlpu_add_test(mem_clocked_test tests/unit/mem/mem_clocked_test.cpp
    unit.mem.clocked unit mem timing fast)
ftlpu_add_test(vector_roundtrip_test tests/unit/mem/vector_roundtrip_test.cpp
    unit.mem.vector_roundtrip unit mem fast)

ftlpu_add_test(mxm_supercell_test tests/unit/mxm/supercell_test.cpp
    unit.mxm.supercell unit mxm fast)
ftlpu_add_test(mxm_array_test tests/unit/mxm/array_test.cpp
    unit.mxm.array unit mxm fast)
ftlpu_add_test(mxm_control_slice_test tests/unit/mxm/control_slice_test.cpp
    unit.mxm.control_slice unit mxm fast)
ftlpu_add_test(mxm_accumulator_test tests/unit/mxm/accumulator_test.cpp
    unit.mxm.accumulator unit mxm fast)
ftlpu_add_test(mxm_clocked_test tests/unit/mxm/mxm_clocked_test.cpp
    unit.mxm.clocked unit mxm timing fast)
ftlpu_add_test(mxm_weight_dequantizer_test tests/unit/mxm/weight_dequantizer_test.cpp
    unit.mxm.weight_dequantizer unit mxm fast)
ftlpu_add_test(mxm_bf16_numeric_alignment_test
    tests/unit/mxm/bf16_numeric_alignment_test.cpp
    unit.mxm.bf16_numeric_alignment unit mxm fast)
ftlpu_add_test(mxm_core_clocked_test tests/unit/mxm/core_clocked_test.cpp
    unit.mxm.core_clocked unit mxm timing fast)
add_executable(mxm_datapath_c1_cycle_trace_test
    tests/unit/mxm/datapath_c1_cycle_trace_test.cpp)
target_link_libraries(mxm_datapath_c1_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME unit.mxm.datapath_c1_cycle_trace
    COMMAND $<TARGET_FILE:mxm_datapath_c1_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_c1_cycle")
set_tests_properties(unit.mxm.datapath_c1_cycle_trace PROPERTIES
    LABELS "unit;mxm;timing;cycle_trace")
add_executable(mxm_datapath_c2_cycle_trace_test
    tests/unit/mxm/datapath_c2_cycle_trace_test.cpp)
target_link_libraries(mxm_datapath_c2_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME unit.mxm.datapath_c2_cycle_trace
    COMMAND $<TARGET_FILE:mxm_datapath_c2_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_c2_cycle")
set_tests_properties(unit.mxm.datapath_c2_cycle_trace PROPERTIES
    LABELS "unit;mxm;timing;cycle_trace")
add_executable(mxm_datapath_c3_cycle_trace_test
    tests/unit/mxm/datapath_c3_cycle_trace_test.cpp)
target_link_libraries(mxm_datapath_c3_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME unit.mxm.datapath_c3_cycle_trace
    COMMAND $<TARGET_FILE:mxm_datapath_c3_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_c3_cycle")
set_tests_properties(unit.mxm.datapath_c3_cycle_trace PROPERTIES
    LABELS "unit;mxm;timing;cycle_trace")
add_executable(mxm_datapath_c4_cycle_trace_test
    tests/unit/mxm/datapath_c4_cycle_trace_test.cpp)
target_link_libraries(mxm_datapath_c4_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME unit.mxm.datapath_c4_cycle_trace
    COMMAND $<TARGET_FILE:mxm_datapath_c4_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_c4_cycle")
set_tests_properties(unit.mxm.datapath_c4_cycle_trace PROPERTIES
    LABELS "unit;mxm;timing;cycle_trace")
add_executable(mxm_datapath_c5_cycle_trace_test
    tests/unit/mxm/datapath_c5_cycle_trace_test.cpp)
target_link_libraries(mxm_datapath_c5_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME unit.mxm.datapath_c5_cycle_trace
    COMMAND $<TARGET_FILE:mxm_datapath_c5_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_c5_cycle")
set_tests_properties(unit.mxm.datapath_c5_cycle_trace PROPERTIES
    LABELS "unit;mxm;timing;cycle_trace")
add_executable(mxm_datapath_c6_cycle_trace_test
    tests/unit/mxm/datapath_c6_cycle_trace_test.cpp)
target_link_libraries(mxm_datapath_c6_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME unit.mxm.datapath_c6_cycle_trace
    COMMAND $<TARGET_FILE:mxm_datapath_c6_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_c6_cycle")
set_tests_properties(unit.mxm.datapath_c6_cycle_trace PROPERTIES
    LABELS "unit;mxm;timing;cycle_trace")

ftlpu_add_test(vxm_alu_test tests/unit/vxm/alu_test.cpp
    unit.vxm.alu unit vxm fast)
ftlpu_add_test(vxm_lane_test tests/unit/vxm/lane_test.cpp
    unit.vxm.lane unit vxm fast)
ftlpu_add_test(vxm_superlane_test tests/unit/vxm/superlane_test.cpp
    unit.vxm.superlane unit vxm fast)
ftlpu_add_test(vxm_slice_test tests/unit/vxm/slice_test.cpp
    unit.vxm.slice unit vxm fast)
ftlpu_add_test(vxm_clocked_test tests/unit/vxm/vxm_clocked_test.cpp
    unit.vxm.clocked unit vxm timing fast)
ftlpu_add_test(vxm_fp16_16alu_test tests/unit/vxm/fp16_16alu_test.cpp
    unit.vxm.fp16_16alu unit vxm fast)
ftlpu_add_test(vxm_input_buffer_test tests/unit/vxm/input_buffer_test.cpp
    unit.vxm.input_buffer unit vxm fast)
ftlpu_add_test(vxm_operator_throughput_test tests/unit/vxm/operator_throughput_test.cpp
    unit.vxm.operator_throughput unit vxm fast)
ftlpu_add_test(vxm_special_alu_test tests/unit/vxm/special_alu_test.cpp
    unit.vxm.special_alu unit vxm fast)
ftlpu_add_test(vxm_static_quantizer_test tests/unit/vxm/static_quantizer_test.cpp
    unit.vxm.static_quantizer unit vxm fast)

ftlpu_add_test(vxm_log_test tests/unit/vxm/log_test.cpp
    unit.vxm.log unit vxm fast)

ftlpu_add_test(sxm_test tests/unit/sxm/slice_test.cpp
    unit.sxm.slice unit sxm fast)
ftlpu_add_test(sxm_wide_transpose_test tests/unit/sxm/wide_transpose_test.cpp
    unit.sxm.wide_transpose unit sxm fast)
ftlpu_add_test(sxm_attention_transpose_test tests/unit/sxm/attention_transpose_test.cpp
    unit.sxm.attention_transpose unit sxm fast)
ftlpu_add_test(sxm_local_blocks_timing_test tests/unit/sxm/local_blocks_timing_test.cpp
    unit.sxm.local_blocks_timing unit sxm fast)
ftlpu_add_test(sxm_clocked_test tests/unit/sxm/sxm_clocked_test.cpp
    unit.sxm.clocked unit sxm fast)
ftlpu_add_test(icu_barrier_test tests/unit/icu/barrier_test.cpp
    unit.icu.barrier unit icu fast)
ftlpu_add_test(icu_common_barrier_test tests/unit/icu/common_barrier_test.cpp
    unit.icu.common_barrier unit icu fast)
ftlpu_add_test(icu_queue_control_test tests/unit/icu/queue_control_test.cpp
    unit.icu.queue_control unit icu fast)
ftlpu_add_test(icu_qctl_runtime_test tests/unit/icu/qctl_runtime_test.cpp
    unit.icu.qctl_runtime unit icu timing fast)
ftlpu_add_test(icu_qctl_phase2b_trace_test
    tests/unit/icu/qctl_phase2b_trace_test.cpp
    unit.icu.qctl_phase2b_trace unit icu timing fast)
ftlpu_add_test(icu_macro_schedule_test tests/unit/icu/macro_schedule_test.cpp
    unit.icu.macro_schedule unit icu fast)
ftlpu_add_test(icu_raw_macro_decoder_test tests/unit/icu/raw_macro_decoder_test.cpp
    unit.icu.raw_macro_decoder unit icu fast)
ftlpu_add_test(icu_mxm_local_imem_fetch_test
    tests/unit/icu/mxm_local_imem_fetch_test.cpp
    unit.icu.mxm_local_imem_fetch unit icu mxm timing fast)
ftlpu_add_test(icu_mxm_five_stream_test
    tests/unit/icu/mxm_five_stream_icu_test.cpp
    unit.icu.mxm_five_stream unit icu mxm timing fast)
add_executable(raw_macro_level1_trace_test
    tests/unit/icu/raw_macro_level1_trace_test.cpp)
target_link_libraries(raw_macro_level1_trace_test PRIVATE ftlpu_cmodel)
target_include_directories(raw_macro_level1_trace_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
add_test(NAME unit.icu.raw_macro_level1_trace
    COMMAND $<TARGET_FILE:raw_macro_level1_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/raw_macro_level1/cmodel_trace.jsonl")
set_tests_properties(unit.icu.raw_macro_level1_trace PROPERTIES
    LABELS "unit;icu;timing;trace")
add_executable(raw_macro_p2_1_fetch_trace_test
    tests/unit/icu/raw_macro_p2_1_fetch_trace_test.cpp)
target_link_libraries(raw_macro_p2_1_fetch_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME unit.icu.raw_macro_p2_1_fetch_trace
    COMMAND $<TARGET_FILE:raw_macro_p2_1_fetch_trace_test>
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/raw_macro_p2_1/machine_words.txt"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/support/raw_macro_p2_1/directed_cases.csv"
        "${CMAKE_CURRENT_BINARY_DIR}/raw_macro_p2_1/cmodel_trace.jsonl")
set_tests_properties(unit.icu.raw_macro_p2_1_fetch_trace PROPERTIES
    LABELS "unit;icu;timing;trace;p2_1")
add_executable(raw_macro_level2_trace_test
    tests/unit/mem/raw_macro_level2_trace_test.cpp)
target_link_libraries(raw_macro_level2_trace_test PRIVATE ftlpu_cmodel)
target_include_directories(raw_macro_level2_trace_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
if(MSVC)
    target_compile_options(raw_macro_level2_trace_test PRIVATE /UNDEBUG)
else()
    target_compile_options(raw_macro_level2_trace_test PRIVATE -UNDEBUG)
endif()
add_test(NAME unit.mem.raw_macro_level2_trace
    COMMAND $<TARGET_FILE:raw_macro_level2_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/raw_macro_level2/cmodel_trace.jsonl"
        "${CMAKE_CURRENT_BINARY_DIR}/raw_macro_level2/shared_input_manifest.json")
set_tests_properties(unit.mem.raw_macro_level2_trace PROPERTIES
    LABELS "unit;mem;icu;stream;timing;trace")
add_executable(raw_macro_phase1_space_trace_test
    tests/unit/mem/raw_macro_phase1_space_trace_test.cpp)
target_link_libraries(raw_macro_phase1_space_trace_test PRIVATE ftlpu_cmodel)
target_include_directories(raw_macro_phase1_space_trace_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
if(MSVC)
    target_compile_options(raw_macro_phase1_space_trace_test PRIVATE /UNDEBUG)
else()
    target_compile_options(raw_macro_phase1_space_trace_test PRIVATE -UNDEBUG)
endif()
add_test(NAME unit.mem.raw_macro_phase1_space_trace
    COMMAND $<TARGET_FILE:raw_macro_phase1_space_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/raw_macro_phase1_space")
set_tests_properties(unit.mem.raw_macro_phase1_space_trace PROPERTIES
    LABELS "unit;mem;icu;stream;timing;trace;phase1_space")
ftlpu_add_test(icu_vxm_sequential_instruction_test
    tests/unit/icu/vxm_sequential_instruction_test.cpp
    unit.icu.vxm_sequential unit icu fast)
ftlpu_add_test(icu_c2c_packet_codec_test
    tests/unit/icu/c2c_packet_codec_test.cpp
    unit.icu.c2c_packet_codec unit icu c2c fast)
ftlpu_add_test(icu_clocked_test tests/unit/icu/icu_clocked_test.cpp
    unit.icu.clocked unit icu timing fast)
ftlpu_add_test(icu_rtl_fifo_test tests/unit/icu/rtl_fifo_test.cpp
    unit.icu.rtl_fifo unit icu timing fast)
ftlpu_add_test(instruction_fetch_dma_test
    tests/unit/icu/instruction_fetch_dma_test.cpp
    unit.icu.instruction_fetch_dma unit icu dma timing fast)
ftlpu_add_demo(icu_rtl_fifo_golden_trace
    tests/unit/icu/rtl_fifo_golden_trace.cpp)
ftlpu_add_demo(instruction_dma_golden_trace
    tests/unit/icu/instruction_dma_golden_trace.cpp)
ftlpu_add_test(program_load_command_controller_test
    tests/unit/icu/program_load_command_controller_test.cpp
    unit.icu.program_load_command_controller unit icu program timing fast)
ftlpu_add_test(program_distribution_transport_test
    tests/unit/icu/program_distribution_transport_test.cpp
    unit.icu.program_distribution_transport unit icu program timing fast)
add_executable(phase3_program_loading_trace_test
    tests/subsystem/system/phase3_program_loading_trace_test.cpp)
target_link_libraries(phase3_program_loading_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME subsystem.program.phase3_r2_trace
    COMMAND $<TARGET_FILE:phase3_program_loading_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/phase3_r2_dma_reassembler_word_fifo"
        r2)
set_tests_properties(subsystem.program.phase3_r2_trace PROPERTIES
    LABELS "subsystem;program;dma;icu;phase3")
add_test(NAME subsystem.program.phase3_r3_trace
    COMMAND $<TARGET_FILE:phase3_program_loading_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/phase3_r3_distributor_relay_elastic"
        r3)
set_tests_properties(subsystem.program.phase3_r3_trace PROPERTIES
    LABELS "subsystem;program;dma;icu;phase3")

foreach(endpoint mem mxm)
    add_test(NAME subsystem.program.phase3_r4_${endpoint}_trace
        COMMAND $<TARGET_FILE:phase3_program_loading_trace_test>
            "${CMAKE_CURRENT_BINARY_DIR}/phase3_r4_program_endpoint"
            r4_${endpoint})
    set_tests_properties(subsystem.program.phase3_r4_${endpoint}_trace PROPERTIES
        LABELS "subsystem;program;dma;icu;phase3")
endforeach()

# Subsystem tests: multiple hardware blocks connected by physical SR routes.
add_test(NAME subsystem.program.phase3_r4_mxm_endpoint_direct_trace
    COMMAND $<TARGET_FILE:phase3_program_loading_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/phase3_r4_mxm_endpoint_direct"
        r4_mxm_endpoint_direct)
set_tests_properties(subsystem.program.phase3_r4_mxm_endpoint_direct_trace PROPERTIES
    LABELS "subsystem;program;icu;phase3")
add_executable(phase3_qctl_program_load_trace_test
    tests/subsystem/system/phase3_qctl_program_load_trace_test.cpp)
target_link_libraries(phase3_qctl_program_load_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME subsystem.program.phase3_r5_qctl_direct_trace
    COMMAND $<TARGET_FILE:phase3_qctl_program_load_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/phase3_r5_qctl_direct_load")
set_tests_properties(subsystem.program.phase3_r5_qctl_direct_trace PROPERTIES
    LABELS "subsystem;program;icu;phase3")
add_executable(phase3_stream_interconnect_trace_test
    tests/subsystem/system/phase3_stream_interconnect_trace_test.cpp)
target_link_libraries(phase3_stream_interconnect_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME subsystem.program.phase3_r7_stream_interconnect_trace
    COMMAND $<TARGET_FILE:phase3_stream_interconnect_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/phase3_r7_stream_interconnect")
set_tests_properties(subsystem.program.phase3_r7_stream_interconnect_trace PROPERTIES
    LABELS "subsystem;program;stream;phase3")
ftlpu_add_test(c2c_test tests/subsystem/c2c/dual_chip_test.cpp
    subsystem.c2c.dual_chip subsystem c2c fast)
ftlpu_add_test(c2c_clocked_test tests/subsystem/c2c/c2c_clocked_test.cpp
    subsystem.c2c.clocked subsystem c2c timing fast)
ftlpu_add_test(c2c_dma_ddr4_test tests/subsystem/c2c/dma_ddr4_test.cpp
    subsystem.c2c.dma_ddr4 subsystem c2c dma ddr4 fast)
ftlpu_add_test(c2c_dma_clocked_test
    tests/subsystem/c2c/c2c_dma_clocked_test.cpp
    subsystem.c2c.dma_clocked subsystem c2c dma ddr4 timing fast)
ftlpu_add_test(stream_topology_test tests/subsystem/system/stream_topology_test.cpp
    subsystem.topology.stream subsystem topology fast)
ftlpu_add_test(full_chip_topology_test tests/subsystem/system/full_chip_topology_test.cpp
    subsystem.topology.full_chip subsystem topology fast)
ftlpu_add_test(passive_vxm_bridge_test tests/subsystem/system/passive_vxm_bridge_test.cpp
    subsystem.topology.passive_vxm_bridge subsystem topology vxm fast)
ftlpu_add_test(distributed16_passive_copy_test
    tests/subsystem/system/distributed16_passive_copy_test.cpp
    subsystem.topology.distributed16_passive_copy subsystem topology vxm fast)
ftlpu_add_test(vxm_repeated_distributed_input_test
    tests/subsystem/system/vxm_repeated_distributed_input_test.cpp
    subsystem.topology.vxm_repeated_distributed_input subsystem topology vxm fast)
ftlpu_add_test(hardware_configuration_test tests/subsystem/system/hardware_configuration_test.cpp
    subsystem.hardware.configuration subsystem hardware fast)
ftlpu_add_test(tsp_timing_integration_test
    tests/subsystem/system/tsp_timing_integration_test.cpp
    subsystem.system.timing subsystem system timing fast)
ftlpu_add_test(mem_sr_sxm_clock_integration_test
    tests/subsystem/system/mem_sr_sxm_clock_integration_test.cpp
    subsystem.system.mem_sr_sxm_clock subsystem system mem stream sxm timing fast)
ftlpu_add_test(program_loader_test tests/subsystem/system/program_loader_test.cpp
    subsystem.program.loader subsystem program dma fast)
ftlpu_add_test(program_loader_relay_mapping_test
    tests/subsystem/system/program_loader_relay_mapping_test.cpp
    subsystem.program.relay_mapping subsystem program dma fast)
ftlpu_add_test(mxm_full_instruction_path_test
    tests/subsystem/system/mxm_full_instruction_path_test.cpp
    subsystem.system.mxm_full_instruction_path subsystem system icu mxm timing)
add_executable(mxm_five_stream_cycle_trace_test
    tests/subsystem/system/mxm_five_stream_cycle_trace_test.cpp)
target_link_libraries(mxm_five_stream_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME subsystem.system.mxm_five_stream_cycle_trace
    COMMAND $<TARGET_FILE:mxm_five_stream_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_full_path_cycle/c7")
set_tests_properties(subsystem.system.mxm_five_stream_cycle_trace PROPERTIES
    LABELS "subsystem;system;icu;mxm;timing")
add_executable(mxm_full_path_cycle_trace_test
    tests/subsystem/system/mxm_full_path_cycle_trace_test.cpp)
target_link_libraries(mxm_full_path_cycle_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME subsystem.system.mxm_full_path_cycle_trace
    COMMAND $<TARGET_FILE:mxm_full_path_cycle_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/mxm_full_path_cycle/c8")
set_tests_properties(subsystem.system.mxm_full_path_cycle_trace PROPERTIES
    LABELS "subsystem;system;icu;mxm;timing")
add_executable(qctl_runtime_issue_joint_trace_test
    tests/subsystem/system/qctl_runtime_issue_joint_trace_test.cpp)
target_link_libraries(qctl_runtime_issue_joint_trace_test PRIVATE ftlpu_cmodel)
add_test(NAME subsystem.system.qctl_runtime_issue_joint_trace
    COMMAND $<TARGET_FILE:qctl_runtime_issue_joint_trace_test>
        "${CMAKE_CURRENT_BINARY_DIR}/qctl_runtime_issue_joint/cmodel")
set_tests_properties(subsystem.system.qctl_runtime_issue_joint_trace PROPERTIES
    LABELS "subsystem;system;icu;qctl;mem;mxm;timing")
add_test(NAME subsystem.program.configuration COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/subsystem/system/program_load_configuration_test.cmake")
add_test(NAME subsystem.icu.configuration COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/subsystem/system/icu_configuration_test.cmake")

ftlpu_add_test(mxm_bf16_test tests/subsystem/mxm/bf16_test.cpp
    subsystem.mxm.bf16 subsystem mxm fast)
ftlpu_add_test(mxm_decode_streamed_weight_test
    tests/subsystem/mxm/decode_streamed_weight_test.cpp
    subsystem.mxm.decode_streamed_weight subsystem mxm fast)
ftlpu_add_test(sxm_mem_transpose_test tests/subsystem/sxm/mem_transpose_test.cpp
    subsystem.sxm.mem_transpose subsystem sxm fast)
add_test(NAME subsystem.sxm.mem_transpose_smoke COMMAND sxm_mem_transpose_test 2)
set_tests_properties(subsystem.sxm.mem_transpose_smoke PROPERTIES LABELS "subsystem;sxm;fast")

# Kernel tests: one neural-network operator or fused operator pipeline.
ftlpu_add_test(w8a16_projection_test tests/kernel/w8a16_projection_test.cpp
    kernel.projection.w8a16 kernel projection slow serial)
ftlpu_add_test(prefill_weight_streams_test tests/kernel/prefill_weight_streams_test.cpp
    kernel.projection.prefill_weight_streams kernel projection fast)
ftlpu_add_test(w8a16_swiglu_test tests/kernel/w8a16_swiglu_test.cpp
    kernel.swiglu.w8a16 kernel swiglu slow serial)
ftlpu_add_test(rmsnorm_test tests/kernel/rmsnorm_test.cpp
    kernel.rmsnorm kernel rmsnorm fast)

# Model tests: model-shaped phases and complete layer schedules.
ftlpu_add_test(smollm2_attention_test
    tests/model/smollm2/prefill_attention_test.cpp
    model.smollm2.prefill_attention model smollm2 slow serial)
ftlpu_add_test(decoder_layer_decode_test
    tests/model/decoder/decode_layer_test.cpp
    model.decoder.decode_layer model decoder slow serial)
ftlpu_add_test(smollm2_decode_ffn_test
    tests/model/smollm2/decode_ffn_test.cpp
    model.smollm2.decode_ffn model smollm2 slow serial)
ftlpu_add_test(smollm2_decode_attention_test
    tests/model/smollm2/decode_attention_test.cpp
    model.smollm2.decode_attention model smollm2 slow serial)

# CPU- and memory-heavy schedules run one at a time.
set_tests_properties(
    kernel.projection.w8a16
    kernel.swiglu.w8a16
    model.smollm2.prefill_attention
    model.decoder.decode_layer
    model.smollm2.decode_ffn
    model.smollm2.decode_attention
    PROPERTIES RUN_SERIAL TRUE
)

ftlpu_add_demo(mxm_control_trace_demo examples/mxm/control_trace_demo.cpp)
ftlpu_add_demo(sxm_chain_demo examples/sxm/chain_demo.cpp)
ftlpu_add_demo(stream_topology_api_demo examples/system/stream_topology_api_demo.cpp)

ftlpu_add_test(vxm_fp32_input_test tests/unit/vxm/fp32_input_test.cpp vxm_fp32_input)

# Opt-in SXM typed matrix reference. Avoid linking unrelated MXM/System ISA
# objects so the bounded T=2/4/8 MEM/SXM reference can be tested independently.
foreach(pair IN ITEMS "sxm_matrix_test|tests/unit/sxm/matrix_test.cpp|unit.sxm.matrix"
                      "sxm_matrix_mem_vxm_test|tests/subsystem/sxm/matrix_mem_vxm_test.cpp|subsystem.sxm.matrix_mem_vxm")
    string(REPLACE "|" ";" fields "${pair}")
    list(GET fields 0 target)
    list(GET fields 1 source)
    list(GET fields 2 test_name)
    add_executable(${target} ${source} src/mem/sram.cpp)
    target_include_directories(${target} PRIVATE "${FTLPU_GENERATED_INCLUDE_DIR}" include tests/support)
    if(MSVC)
        target_compile_options(${target} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${target} PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${target}>)
    set_tests_properties(${test_name} PROPERTIES LABELS "sxm;matrix;fast")
endforeach()
