cmake_minimum_required(VERSION 3.20)
include("${SOURCE_DIR}/cmake/FTLPUHardwareConfig.cmake")
if(DEFINED INPUT_JSON)
    ftlpu_generate_hardware_config("${INPUT_JSON}" "${OUTPUT_HEADER}")
    return()
endif()

file(MAKE_DIRECTORY "${BINARY_DIR}/program-load-config-test")
set(test_dir "${BINARY_DIR}/program-load-config-test")
file(READ "${SOURCE_DIR}/config/ftlpu-lpu32.json" original)
foreach(profile IN ITEMS "ftlpu-lpu16-decode;80;128;224"
                         "ftlpu-lpu32;80;128;136"
                         "ftlpu-lpu64-tiles8;64;128;160")
    list(GET profile 0 name)
    list(GET profile 1 p)
    list(GET profile 2 rt)
    list(GET profile 3 q)
    ftlpu_generate_hardware_config("${SOURCE_DIR}/config/${name}.json" "${test_dir}/${name}.hpp")
    file(READ "${test_dir}/${name}.hpp" generated)
    foreach(pattern IN ITEMS "kProgramLoadResponseFifoDepth = ${p};"
                             "kDmaFifoDepthVectors = ${rt};"
                             "kProgramLoadOutstandingDepth = 4;"
                             "kDmaFifoDepthVectors, ${q}};")
        string(FIND "${generated}" "${pattern}" position)
        if(position LESS 0)
            message(FATAL_ERROR "${name} missing sizing: ${pattern}")
        endif()
    endforeach()
endforeach()
set(fields dma_fifo_depth_vectors bus_bytes_per_cycle route_latency_cycles
    imem_write_ii_cycles imem_write_commit_latency_cycles
    mem_groups_per_relay response_fifo_depth ddr_request_queue_depth)
set(constants kDmaFifoDepthVectors kProgramLoadBusBytesPerCycle
    kProgramLoadRouteLatencyCycles kProgramLoadImemWriteIiCycles
    kProgramLoadImemWriteCommitLatencyCycles
    kProgramLoadMemGroupsPerRelay kProgramLoadResponseFifoDepth _)
foreach(index RANGE 0 7)
    list(GET fields ${index} field)
    list(GET constants ${index} constant)
    if(field STREQUAL "dma_fifo_depth_vectors" OR field STREQUAL "ddr_request_queue_depth")
        set(section external_memory)
    else()
        set(section program_load)
    endif()
    string(JSON changed SET "${original}" ${section} ${field} 4)
    if(field STREQUAL "mem_groups_per_relay")
        # Relay count is derived, not a second independently tunable geometry.
        string(JSON changed REMOVE "${changed}" program_load relays_per_hemisphere)
    endif()
    file(WRITE "${test_dir}/changed.json" "${changed}")
    ftlpu_generate_hardware_config("${test_dir}/changed.json" "${test_dir}/changed.hpp")
    file(READ "${test_dir}/changed.hpp" generated)
    set(expected "${constant} = 4;")
    if(field STREQUAL "ddr_request_queue_depth")
        set(expected "kDmaFifoDepthVectors, 4};")
    endif()
    string(FIND "${generated}" "${expected}" position)
    if(position LESS 0)
        message(FATAL_ERROR "${section}.${field} was not connected to CModel initialization")
    endif()
    foreach(invalid IN ITEMS 0 -1 1.5 null true 2147483648 "\"4\"" "[]" "{}")
        string(JSON changed SET "${original}" ${section} ${field} "${invalid}")
        file(WRITE "${test_dir}/invalid.json" "${changed}")
        execute_process(COMMAND "${CMAKE_COMMAND}"
            "-DSOURCE_DIR=${SOURCE_DIR}" "-DINPUT_JSON=${test_dir}/invalid.json"
            "-DOUTPUT_HEADER=${test_dir}/invalid.hpp" -P "${CMAKE_CURRENT_LIST_FILE}"
            RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
        if(result EQUAL 0)
            message(FATAL_ERROR "${section}.${field} accepted ${invalid}")
        endif()
    endforeach()
endforeach()
# Explicit stale Relay geometry must be rejected; omitting it derives ceil.
string(JSON changed SET "${original}" program_load mem_groups_per_relay 4)
file(WRITE "${test_dir}/mismatched-relays.json" "${changed}")
execute_process(COMMAND "${CMAKE_COMMAND}"
    "-DSOURCE_DIR=${SOURCE_DIR}" "-DINPUT_JSON=${test_dir}/mismatched-relays.json"
    "-DOUTPUT_HEADER=${test_dir}/mismatched.hpp" -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
if(result EQUAL 0)
    message(FATAL_ERROR "explicit stale Relay count was accepted")
endif()
string(JSON changed REMOVE "${changed}" program_load relays_per_hemisphere)
file(WRITE "${test_dir}/derived-relays.json" "${changed}")
ftlpu_generate_hardware_config("${test_dir}/derived-relays.json" "${test_dir}/derived.hpp")
file(READ "${test_dir}/derived.hpp" generated)
if(NOT generated MATCHES "kProgramLoadRelaysPerHemisphere = 4;")
    message(FATAL_ERROR "N52/G4/groups4 did not derive four Relays")
endif()
string(JSON omitted REMOVE "${original}" program_load)
string(JSON omitted REMOVE "${omitted}" external_memory dma_fifo_depth_vectors)
file(WRITE "${test_dir}/omitted.json" "${omitted}")
ftlpu_generate_hardware_config("${test_dir}/omitted.json" "${test_dir}/omitted.hpp")
file(READ "${test_dir}/omitted.hpp" generated)
foreach(expected IN ITEMS "kDmaFifoDepthVectors = 16;"
    "kProgramLoadBusBytesPerCycle = 32;" "kProgramLoadRouteLatencyCycles = 1;"
    "kProgramLoadImemWriteIiCycles = 1;" "kProgramLoadImemWriteCommitLatencyCycles = 1;"
    "kProgramLoadMemGroupsPerRelay = 2;")
    string(FIND "${generated}" "${expected}" position)
    if(position LESS 0)
        message(FATAL_ERROR "missing optional-field default: ${expected}")
    endif()
endforeach()
message(STATUS "program load configuration parsing passed")
