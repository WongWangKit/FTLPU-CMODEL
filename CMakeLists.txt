cmake_minimum_required(VERSION 3.20)

project(FTLPU_CMODEL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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

ftlpu_add_test(stream_register_test tests/core/stream_register_test.cpp)
ftlpu_add_test(topology_test tests/core/topology_test.cpp)
ftlpu_add_test(instruction_pipeline_test tests/core/instruction_pipeline_test.cpp)
ftlpu_add_test(instruction_codec_test tests/core/instruction_codec_test.cpp)

ftlpu_add_test(mem_slice_test tests/mem/slice_test.cpp)
ftlpu_add_test(tile_array_test tests/mem/tile_array_test.cpp)
ftlpu_add_test(vector_roundtrip_test tests/mem/vector_roundtrip_test.cpp)

ftlpu_add_test(mxm_supercell_test tests/mxm/supercell_test.cpp)
ftlpu_add_test(mxm_array_test tests/mxm/array_test.cpp)
ftlpu_add_test(mxm_control_slice_test tests/mxm/control_slice_test.cpp)
ftlpu_add_test(gemm_320_test tests/mxm/gemm_320_test.cpp)

ftlpu_add_test(mem_mxm_integration_test tests/integration/mem_mxm_test.cpp)
ftlpu_add_test(mem_vxm_integration_test tests/integration/mem_vxm_test.cpp)
ftlpu_add_test(mem_dual_mxm_swiglu_test tests/integration/mem_dual_mxm_swiglu_test.cpp)
ftlpu_add_test(mem_dual_mxm_swiglu_offline_icu_test tests/integration/mem_dual_mxm_swiglu_test.cpp)
target_compile_definitions(mem_dual_mxm_swiglu_offline_icu_test PRIVATE FTLPU_OFFLINE_ICU_FFN_TEST)
ftlpu_add_test(icu_stream_scheduled_test tests/integration/icu_stream_scheduled_test.cpp)

ftlpu_add_test(vxm_alu_test tests/vxm/alu_test.cpp)
ftlpu_add_test(vxm_lane_test tests/vxm/lane_test.cpp)
ftlpu_add_test(vxm_superlane_test tests/vxm/superlane_test.cpp)
ftlpu_add_test(vxm_slice_test tests/vxm/slice_test.cpp)

ftlpu_add_test(sxm_test tests/sxm/sxm_test.cpp)

ftlpu_add_test(icu_test tests/icu/icu_test.cpp)

ftlpu_add_demo(northbound_pipeline_demo examples/core/northbound_pipeline_demo.cpp)
ftlpu_add_demo(tile_array_trace_demo examples/mem/tile_array_trace_demo.cpp)
ftlpu_add_demo(vector_roundtrip_demo examples/mem/vector_roundtrip_demo.cpp)
ftlpu_add_demo(mxm_control_trace_demo examples/mxm/control_trace_demo.cpp)
ftlpu_add_demo(mem_mxm_trace_demo examples/mxm/mem_mxm_trace_demo.cpp)
ftlpu_add_demo(gemm_320_trace_demo examples/mxm/gemm_320_trace_demo.cpp)
ftlpu_add_demo(icu_mem_demo examples/icu/icu_mem_demo.cpp)
ftlpu_add_demo(sxm_chain_demo examples/sxm/chain_demo.cpp)
ftlpu_add_demo(vxm_lane_trace_demo examples/vxm/lane_trace_demo.cpp)
