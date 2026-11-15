cmake_minimum_required(VERSION 3.20)

project(ftlpu_two_fifo_interaction_experiment LANGUAGES CXX)

add_executable(two_fifo_cmodel cmodel/main.cpp)
target_compile_features(two_fifo_cmodel PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(two_fifo_cmodel PRIVATE /W4 /permissive-)
else()
  target_compile_options(two_fifo_cmodel PRIVATE -Wall -Wextra -Wpedantic)
endif()
