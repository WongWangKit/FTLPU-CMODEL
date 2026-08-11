cmake_minimum_required(VERSION 3.20)

project(FTLPU_CMODEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FTLPU_MXM_ACCUMULATOR_BLOCK_COUNT 32 CACHE STRING
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

function(ftlpu_add_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ftlpu_cmodel)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

function(ftlpu_add_demo target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ftlpu_cmodel)
endfunction()

ftlpu_add_test(instruction_codec_test tests/core/instruction_codec_test.cpp)
ftlpu_add_test(bf16_test tests/core/bf16_test.cpp)

ftlpu_add_test(stream_fabric_test tests/stream/stream_fabric_test.cpp)
ftlpu_add_test(stream_port_test tests/stream/stream_port_test.cpp)

ftlpu_add_test(mem_slice_test tests/mem/slice_test.cpp)

ftlpu_add_test(c2c_test tests/c2c/c2c_test.cpp)
ftlpu_add_test(mem_array_separated_test tests/mem/mem_array_separated_test.cpp)
ftlpu_add_test(tile_array_test tests/mem/tile_array_test.cpp)
ftlpu_add_test(vector_roundtrip_test tests/mem/vector_roundtrip_test.cpp)

ftlpu_add_test(stream_topology_test tests/system/stream_topology_test.cpp)
ftlpu_add_test(full_chip_topology_test tests/system/full_chip_topology_test.cpp)
ftlpu_add_test(hardware_configuration_test tests/system/hardware_configuration_test.cpp)

ftlpu_add_test(mxm_supercell_test tests/mxm/supercell_test.cpp)
ftlpu_add_test(mxm_array_test tests/mxm/array_test.cpp)
ftlpu_add_test(mxm_control_slice_test tests/mxm/control_slice_test.cpp)
ftlpu_add_test(mxm_accumulator_test tests/mxm/accumulator_test.cpp)
ftlpu_add_test(mxm_block_accumulator_test tests/mxm/block_accumulator_test.cpp)
ftlpu_add_test(mxm_weight_dequantizer_test tests/mxm/weight_dequantizer_test.cpp)
ftlpu_add_test(mxm_bf16_test tests/integration/mxm_bf16_test.cpp)
ftlpu_add_test(mxm_decode_streamed_weight_test tests/integration/mxm_decode_streamed_weight_test.cpp)
ftlpu_add_test(mxm_decode_layout_comparison_test tests/integration/mxm_decode_layout_comparison_test.cpp)
ftlpu_add_test(smollm2_decode_layout_comparison_test tests/integration/smollm2_decode_layout_comparison_test.cpp)
ftlpu_add_test(w8a16_projection_test tests/integration/w8a16_projection_test.cpp)
ftlpu_add_test(w8a16_swiglu_test tests/integration/w8a16_swiglu_test.cpp)
ftlpu_add_test(dual_hemisphere_w8a16_swiglu_test tests/integration/dual_hemisphere_w8a16_swiglu_test.cpp)
ftlpu_add_test(smollm2_block8_dequant_ffn_test tests/integration/smollm2_block8_dequant_ffn_test.cpp)
ftlpu_add_test(rmsnorm_test tests/integration/rmsnorm_test.cpp)
ftlpu_add_test(smollm2_attention_test tests/integration/smollm2_attention_test.cpp)
ftlpu_add_test(decoder_layer_decode_test tests/integration/decoder_layer_decode_test.cpp)
ftlpu_add_test(smollm2_decode_ffn_test tests/integration/smollm2_decode_ffn_test.cpp)
ftlpu_add_test(smollm2_decode_attention_test tests/integration/smollm2_decode_attention_test.cpp)

add_library(smollm2_layer_phases STATIC
    tests/integration/smollm2_attention_test.cpp
    tests/integration/smollm2_block8_dequant_ffn_test.cpp
    tests/integration/smollm2_decode_attention_test.cpp
    tests/integration/smollm2_decode_ffn_test.cpp
)
target_compile_definitions(
    smollm2_layer_phases PRIVATE FTLPU_SMOLLM2_LAYER_PHASE_ONLY)
target_link_libraries(smollm2_layer_phases PRIVATE ftlpu_cmodel)

add_executable(
    smollm2_prefill_decode_layer_test
    tests/integration/smollm2_prefill_decode_layer_test.cpp)
target_link_libraries(
    smollm2_prefill_decode_layer_test
    PRIVATE ftlpu_cmodel smollm2_layer_phases)
add_test(
    NAME smollm2_prefill_decode_layer_test
    COMMAND smollm2_prefill_decode_layer_test)

# These workloads saturate the simulator's CPU and memory paths. Running them
# concurrently is substantially slower than scheduling them one at a time.
set_tests_properties(
    w8a16_projection_test
    w8a16_swiglu_test
    dual_hemisphere_w8a16_swiglu_test
    smollm2_block8_dequant_ffn_test
    smollm2_attention_test
    decoder_layer_decode_test
    smollm2_decode_ffn_test
    smollm2_decode_attention_test
    smollm2_decode_layout_comparison_test
    smollm2_prefill_decode_layer_test
    PROPERTIES RUN_SERIAL TRUE
)

ftlpu_add_test(vxm_alu_test tests/vxm/alu_test.cpp)
ftlpu_add_test(vxm_lane_test tests/vxm/lane_test.cpp)
ftlpu_add_test(vxm_superlane_test tests/vxm/superlane_test.cpp)
ftlpu_add_test(vxm_slice_test tests/vxm/slice_test.cpp)
ftlpu_add_test(vxm_dual_backend_test tests/vxm/dual_backend_test.cpp)

ftlpu_add_test(sxm_test tests/sxm/sxm_test.cpp)
ftlpu_add_test(sxm_attention_transpose_test tests/sxm/attention_transpose_test.cpp)
ftlpu_add_test(sxm_mem_transpose_test tests/integration/sxm_mem_transpose_test.cpp)

ftlpu_add_test(icu_test tests/icu/icu_test.cpp)
ftlpu_add_test(icu_barrier_test tests/icu/barrier_test.cpp)

ftlpu_add_demo(northbound_pipeline_demo examples/core/northbound_pipeline_demo.cpp)
ftlpu_add_demo(tile_array_trace_demo examples/mem/tile_array_trace_demo.cpp)
ftlpu_add_demo(vector_roundtrip_demo examples/mem/vector_roundtrip_demo.cpp)
ftlpu_add_demo(mxm_control_trace_demo examples/mxm/control_trace_demo.cpp)
ftlpu_add_demo(mem_mxm_trace_demo examples/mxm/mem_mxm_trace_demo.cpp)
ftlpu_add_demo(icu_mem_demo examples/icu/icu_mem_demo.cpp)
ftlpu_add_demo(sxm_chain_demo examples/sxm/chain_demo.cpp)
ftlpu_add_demo(stream_topology_api_demo examples/system/stream_topology_api_demo.cpp)
