cmake_minimum_required(VERSION 3.20)

project(FTLPU_CMODEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(ftlpu_cmodel INTERFACE)
target_include_directories(ftlpu_cmodel INTERFACE include)

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

ftlpu_add_test(stream_fabric_test tests/stream/stream_fabric_test.cpp)
ftlpu_add_test(stream_port_test tests/stream/stream_port_test.cpp)

ftlpu_add_test(mem_slice_test tests/mem/slice_test.cpp)
ftlpu_add_test(mem_array_separated_test tests/mem/mem_array_separated_test.cpp)
ftlpu_add_test(mem_accumulator_test tests/mem/accumulator_test.cpp)
ftlpu_add_test(tile_array_test tests/mem/tile_array_test.cpp)
ftlpu_add_test(vector_roundtrip_test tests/mem/vector_roundtrip_test.cpp)

ftlpu_add_test(stream_topology_test tests/system/stream_topology_test.cpp)
ftlpu_add_test(full_chip_topology_test tests/system/full_chip_topology_test.cpp)

ftlpu_add_test(mxm_supercell_test tests/mxm/supercell_test.cpp)
ftlpu_add_test(mxm_array_test tests/mxm/array_test.cpp)
ftlpu_add_test(mxm_control_slice_test tests/mxm/control_slice_test.cpp)
ftlpu_add_test(w8a16_projection_test tests/integration/w8a16_projection_test.cpp)
ftlpu_add_test(w8a16_swiglu_test tests/integration/w8a16_swiglu_test.cpp)
ftlpu_add_test(dual_hemisphere_w8a16_swiglu_test tests/integration/dual_hemisphere_w8a16_swiglu_test.cpp)
ftlpu_add_test(rmsnorm_test tests/integration/rmsnorm_test.cpp)
ftlpu_add_test(smollm2_attention_test tests/integration/smollm2_attention_test.cpp)

ftlpu_add_test(vxm_alu_test tests/vxm/alu_test.cpp)
ftlpu_add_test(vxm_lane_test tests/vxm/lane_test.cpp)
ftlpu_add_test(vxm_superlane_test tests/vxm/superlane_test.cpp)
ftlpu_add_test(vxm_slice_test tests/vxm/slice_test.cpp)

ftlpu_add_test(sxm_test tests/sxm/sxm_test.cpp)
ftlpu_add_test(sxm_attention_transpose_test tests/sxm/attention_transpose_test.cpp)
ftlpu_add_test(sxm_mem_transpose_test tests/integration/sxm_mem_transpose_test.cpp)

ftlpu_add_test(icu_test tests/icu/icu_test.cpp)

ftlpu_add_demo(northbound_pipeline_demo examples/core/northbound_pipeline_demo.cpp)
ftlpu_add_demo(tile_array_trace_demo examples/mem/tile_array_trace_demo.cpp)
ftlpu_add_demo(vector_roundtrip_demo examples/mem/vector_roundtrip_demo.cpp)
ftlpu_add_demo(mxm_control_trace_demo examples/mxm/control_trace_demo.cpp)
ftlpu_add_demo(mem_mxm_trace_demo examples/mxm/mem_mxm_trace_demo.cpp)
ftlpu_add_demo(icu_mem_demo examples/icu/icu_mem_demo.cpp)
ftlpu_add_demo(sxm_chain_demo examples/sxm/chain_demo.cpp)
ftlpu_add_demo(stream_topology_api_demo examples/system/stream_topology_api_demo.cpp)
