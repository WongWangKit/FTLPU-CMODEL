cmake_minimum_required(VERSION 3.20)
project(clocked_module_framework LANGUAGES CXX)

add_executable(clocked_module_framework_cmodel
    cmodel/main.cpp
    cmodel/framework_test.cpp
)
target_compile_features(clocked_module_framework_cmodel PRIVATE cxx_std_20)
target_include_directories(clocked_module_framework_cmodel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/cmodel
)

if(MSVC)
    target_compile_options(clocked_module_framework_cmodel PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(clocked_module_framework_cmodel PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

