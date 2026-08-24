cmake_minimum_required(VERSION 3.20)

project(FTLPU_CMODEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT 256 CACHE STRING
    "Number of complete 32x32 FP32 partial-sum blocks per MXM accumulator")
if(NOT FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT MATCHES "^[1-9][0-9]*$" OR
   FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT GREATER 256)
    message(FATAL_ERROR
        "FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT must be an integer in [1, 256]")
endif()

add_library(ftlpu_cmodel STATIC
    src/mem/sram.cpp
    src/mxm/accumulator.cpp
    src/mxm/block_accumulator.cpp
)
target_include_directories(ftlpu_cmodel PUBLIC include)
target_compile_definitions(ftlpu_cmodel PUBLIC
    FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT=${FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT})

enable_testing()

function(ftlpu_add_test target source test_name)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ftlpu_cmodel)
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
ftlpu_add_test(instruction_codec_test tests/unit/core/instruction_codec_test.cpp
    unit.core.instruction_codec unit core fast)
ftlpu_add_test(bf16_test tests/unit/core/bf16_test.cpp
    unit.core.bf16 unit core fast)

ftlpu_add_test(stream_fabric_test tests/unit/stream/stream_fabric_test.cpp
    unit.stream.fabric unit stream fast)
ftlpu_add_test(stream_port_test tests/unit/stream/stream_port_test.cpp
    unit.stream.port unit stream fast)

ftlpu_add_test(mem_slice_test tests/unit/mem/slice_test.cpp
    unit.mem.slice unit mem fast)
ftlpu_add_test(mem_array_separated_test tests/unit/mem/mem_array_test.cpp
    unit.mem.array unit mem fast)
ftlpu_add_test(tile_array_test tests/unit/mem/tile_array_test.cpp
    unit.mem.tile_array unit mem fast)
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
ftlpu_add_test(mxm_block_accumulator_test tests/unit/mxm/block_accumulator_test.cpp
    unit.mxm.block_accumulator unit mxm fast)
ftlpu_add_test(mxm_weight_dequantizer_test tests/unit/mxm/weight_dequantizer_test.cpp
    unit.mxm.weight_dequantizer unit mxm fast)

ftlpu_add_test(vxm_alu_test tests/unit/vxm/alu_test.cpp
    unit.vxm.alu unit vxm fast)
ftlpu_add_test(vxm_lane_test tests/unit/vxm/lane_test.cpp
    unit.vxm.lane unit vxm fast)
ftlpu_add_test(vxm_superlane_test tests/unit/vxm/superlane_test.cpp
    unit.vxm.superlane unit vxm fast)
ftlpu_add_test(vxm_slice_test tests/unit/vxm/slice_test.cpp
    unit.vxm.slice unit vxm fast)
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

ftlpu_add_test(sxm_test tests/unit/sxm/slice_test.cpp
    unit.sxm.slice unit sxm fast)
ftlpu_add_test(sxm_attention_transpose_test tests/unit/sxm/attention_transpose_test.cpp
    unit.sxm.attention_transpose unit sxm fast)
ftlpu_add_test(icu_barrier_test tests/unit/icu/barrier_test.cpp
    unit.icu.barrier unit icu fast)
ftlpu_add_test(icu_loop_test tests/unit/icu/loop_test.cpp
    unit.icu.loop unit icu fast)
ftlpu_add_test(icu_repeat_2d_test tests/unit/icu/repeat_2d_test.cpp
    unit.icu.repeat_2d unit icu fast)
ftlpu_add_test(icu_macro_schedule_test tests/unit/icu/macro_schedule_test.cpp
    unit.icu.macro_schedule unit icu fast)

# Subsystem tests: multiple hardware blocks connected by physical SR routes.
ftlpu_add_test(c2c_test tests/subsystem/c2c/dual_chip_test.cpp
    subsystem.c2c.dual_chip subsystem c2c fast)
ftlpu_add_test(c2c_dma_ddr4_test tests/subsystem/c2c/dma_ddr4_test.cpp
    subsystem.c2c.dma_ddr4 subsystem c2c dma ddr4 fast)
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

ftlpu_add_test(mxm_bf16_test tests/subsystem/mxm/bf16_test.cpp
    subsystem.mxm.bf16 subsystem mxm fast)
ftlpu_add_test(mxm_decode_streamed_weight_test
    tests/subsystem/mxm/decode_streamed_weight_test.cpp
    subsystem.mxm.decode_streamed_weight subsystem mxm fast)
ftlpu_add_test(mxm_decode_layout_comparison_test
    tests/subsystem/mxm/decode_layout_comparison_test.cpp
    subsystem.mxm.decode_layout_comparison subsystem mxm fast)
ftlpu_add_test(sxm_mem_transpose_test tests/subsystem/sxm/mem_transpose_test.cpp
    subsystem.sxm.mem_transpose subsystem sxm fast)

# Kernel tests: one neural-network operator or fused operator pipeline.
ftlpu_add_test(w8a16_projection_test tests/kernel/w8a16_projection_test.cpp
    kernel.projection.w8a16 kernel projection slow serial)
ftlpu_add_test(w8a16_swiglu_test tests/kernel/w8a16_swiglu_test.cpp
    kernel.swiglu.w8a16 kernel swiglu slow serial)
ftlpu_add_test(rmsnorm_test tests/kernel/rmsnorm_test.cpp
    kernel.rmsnorm kernel rmsnorm fast)

# Model tests: model-shaped phases and complete layer schedules.
ftlpu_add_test(smollm2_decode_layout_comparison_test
    tests/model/smollm2/decode_layout_comparison_test.cpp
    model.smollm2.decode_layout_comparison model smollm2 slow serial)
ftlpu_add_test(smollm2_block8_dequant_ffn_test
    tests/model/smollm2/prefill_ffn_block8_test.cpp
    model.smollm2.prefill_ffn_block8 model smollm2 slow serial)
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

add_library(smollm2_layer_phases STATIC
    tests/model/smollm2/prefill_attention_test.cpp
    tests/model/smollm2/prefill_ffn_block8_test.cpp
    tests/model/smollm2/decode_attention_test.cpp
    tests/model/smollm2/decode_ffn_test.cpp
)
target_compile_definitions(
    smollm2_layer_phases PRIVATE FTLPU_SMOLLM2_LAYER_PHASE_ONLY)
target_link_libraries(smollm2_layer_phases PRIVATE ftlpu_cmodel)
target_include_directories(
    smollm2_layer_phases
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)

add_executable(
    smollm2_prefill_decode_layer_test
    tests/model/smollm2/prefill_decode_layer_test.cpp)
target_link_libraries(
    smollm2_prefill_decode_layer_test
    PRIVATE ftlpu_cmodel smollm2_layer_phases)
target_include_directories(
    smollm2_prefill_decode_layer_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/support)
add_test(
    NAME model.smollm2.prefill_decode_layer
    COMMAND $<TARGET_FILE:smollm2_prefill_decode_layer_test>)
set_tests_properties(
    model.smollm2.prefill_decode_layer
    PROPERTIES LABELS "model;smollm2;slow;serial")

# CPU- and memory-heavy schedules run one at a time.
set_tests_properties(
    kernel.projection.w8a16
    kernel.swiglu.w8a16
    model.smollm2.decode_layout_comparison
    model.smollm2.prefill_ffn_block8
    model.smollm2.prefill_attention
    model.decoder.decode_layer
    model.smollm2.decode_ffn
    model.smollm2.decode_attention
    model.smollm2.prefill_decode_layer
    PROPERTIES RUN_SERIAL TRUE
)

ftlpu_add_demo(northbound_pipeline_demo examples/core/northbound_pipeline_demo.cpp)
ftlpu_add_demo(tile_array_trace_demo examples/mem/tile_array_trace_demo.cpp)
ftlpu_add_demo(vector_roundtrip_demo examples/mem/vector_roundtrip_demo.cpp)
ftlpu_add_demo(mxm_control_trace_demo examples/mxm/control_trace_demo.cpp)
ftlpu_add_demo(mem_mxm_trace_demo examples/mxm/mem_mxm_trace_demo.cpp)
ftlpu_add_demo(sxm_chain_demo examples/sxm/chain_demo.cpp)
ftlpu_add_demo(stream_topology_api_demo examples/system/stream_topology_api_demo.cpp)
