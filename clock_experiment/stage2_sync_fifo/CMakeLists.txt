cmake_minimum_required(VERSION 3.20)

project(ftlpu_sync_fifo_experiment LANGUAGES CXX)

add_executable(sync_fifo_cmodel cmodel/main.cpp)
target_compile_features(sync_fifo_cmodel PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(sync_fifo_cmodel PRIVATE /W4 /permissive-)
else()
  target_compile_options(sync_fifo_cmodel PRIVATE -Wall -Wextra -Wpedantic)
endif()
