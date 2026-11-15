cmake_minimum_required(VERSION 3.20)

project(ftlpu_stage4_combinational_chain LANGUAGES CXX)

add_executable(stage4_cmodel cmodel/main.cpp)
target_compile_features(stage4_cmodel PRIVATE cxx_std_20)
target_include_directories(stage4_cmodel PRIVATE ..)

add_executable(timing_reference_compile_check cmodel/reference_compile_check.cpp)
target_compile_features(timing_reference_compile_check PRIVATE cxx_std_20)
target_include_directories(timing_reference_compile_check PRIVATE ..)

if(MSVC)
  target_compile_options(stage4_cmodel PRIVATE /W4 /permissive-)
  target_compile_options(timing_reference_compile_check PRIVATE /W4 /permissive-)
else()
  target_compile_options(stage4_cmodel PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(timing_reference_compile_check PRIVATE -Wall -Wextra -Wpedantic)
endif()
