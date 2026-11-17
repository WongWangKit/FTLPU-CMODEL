cmake_minimum_required(VERSION 3.24)
project(ftlpu_b1_interface LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FTLPU_SOFTWARE_ROOT "D:/LPU/FTLPU-SOFTWARE-main" CACHE PATH "FTLPU-SOFTWARE source")
set(FTLPU_CMODEL_ROOT "D:/LPU/FTLPU-CMODEL-clock" CACHE PATH "FTLPU-CMODEL-clock source")
set(FTLPU_RUNTIME_BUILD "D:/LPU/FTLPU-CMODEL-clock/out/b1_interface_integration/build_runtime" CACHE PATH "Configured SOFTWARE runtime build")

add_executable(b1_program_load_diagnostic b1_program_load_diagnostic.cpp)
add_executable(b1_barr_transport_diagnostic b1_barr_transport_diagnostic.cpp)
add_executable(b1_mxm_timing_diagnostic b1_mxm_timing_diagnostic.cpp)
add_executable(b1_binary_delivery_adapter b1_binary_delivery_adapter.cpp)
foreach(target b1_program_load_diagnostic b1_barr_transport_diagnostic b1_mxm_timing_diagnostic b1_binary_delivery_adapter)
target_include_directories(${target} PRIVATE
    "${FTLPU_SOFTWARE_ROOT}/runtime/include"
    "${FTLPU_RUNTIME_BUILD}/generated/include"
    "${FTLPU_CMODEL_ROOT}/include")
target_link_libraries(${target} PRIVATE
    "${FTLPU_RUNTIME_BUILD}/runtime/Release/ftlpu_software.lib"
    "${FTLPU_RUNTIME_BUILD}/runtime/Release/ftlpu_cmodel_static.lib")
endforeach()


