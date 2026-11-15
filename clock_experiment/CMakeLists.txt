cmake_minimum_required(VERSION 3.20)

project(ftlpu_clock_experiment LANGUAGES CXX)

add_executable(clock_cmodel cmodel/main.cpp)
target_compile_features(clock_cmodel PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(clock_cmodel PRIVATE /W4 /permissive-)
else()
  target_compile_options(clock_cmodel PRIVATE -Wall -Wextra -Wpedantic)
endif()
