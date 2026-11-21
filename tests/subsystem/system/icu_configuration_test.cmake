cmake_minimum_required(VERSION 3.20)
include("${SOURCE_DIR}/cmake/FTLPUHardwareConfig.cmake")
if(DEFINED INPUT_JSON)
    ftlpu_generate_hardware_config("${INPUT_JSON}" "${OUTPUT_HEADER}")
    return()
endif()

set(test_dir "${BINARY_DIR}/icu-config-test")
file(MAKE_DIRECTORY "${test_dir}")
file(READ "${SOURCE_DIR}/config/ftlpu-lpu32.json" original)
ftlpu_generate_hardware_config("${SOURCE_DIR}/config/ftlpu-lpu32.json" "${test_dir}/default.hpp")
file(READ "${test_dir}/default.hpp" expected)

# Pre-profile JSON omitted every storage/frontend field. Its complete generated
# parameter set must remain identical to the explicit default profile. The
# raw configuration fingerprint must differ: it identifies the actual input.
set(legacy "${original}")
string(JSON count LENGTH "${original}" icu)
math(EXPR last "${count} - 1")
foreach(index RANGE 0 ${last})
    string(JSON key MEMBER "${original}" icu ${index})
    if(NOT key MATCHES "^(common_|queue_control_)")
        string(JSON legacy REMOVE "${legacy}" icu ${key})
    endif()
endforeach()
file(WRITE "${test_dir}/legacy.json" "${legacy}")
ftlpu_generate_hardware_config("${test_dir}/legacy.json" "${test_dir}/legacy.hpp")
file(READ "${test_dir}/legacy.hpp" actual)
file(SHA256 "${SOURCE_DIR}/config/ftlpu-lpu32.json" original_hash)
file(SHA256 "${test_dir}/legacy.json" legacy_hash)
string(FIND "${expected}" "${original_hash}" original_hash_position)
string(FIND "${actual}" "${legacy_hash}" legacy_hash_position)
if(original_hash_position LESS 0 OR legacy_hash_position LESS 0 OR original_hash STREQUAL legacy_hash)
    message(FATAL_ERROR "configuration input fingerprint is missing or aliased")
endif()
string(REPLACE "${original_hash}" "CONFIG_INPUT_HASH" expected "${expected}")
string(REPLACE "${legacy_hash}" "CONFIG_INPUT_HASH" actual "${actual}")
if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "explicit ICU defaults changed the legacy hardware profile")
endif()

set(changed "${original}")
foreach(pair IN ITEMS "mem_imem_depth|512" "mxm_imem_depth|256"
    "c2c_imem_depth|256" "mem_iq_depth|8" "mxm_iq_depth|8"
    "mem_macro_contexts|16" "mxm_macro_contexts|24" "fetch_latency_cycles|2"
    "macro_imem_request_ii_cycles|2" "macro_reservoir_words|4"
    "macro_decode_window_bits|32" "macro_ddb_depth|4"
    "macro_admission_lookahead|6" "macro_ddb_run_capacity|4")
    string(REPLACE "|" ";" parts "${pair}")
    list(GET parts 0 key)
    list(GET parts 1 value)
    string(JSON changed SET "${changed}" icu ${key} ${value})
endforeach()
file(WRITE "${test_dir}/changed.json" "${changed}")
ftlpu_generate_hardware_config("${test_dir}/changed.json" "${test_dir}/include/ftlpu/core/hardware_config.hpp")
file(READ "${test_dir}/include/ftlpu/core/hardware_config.hpp" actual)
foreach(expected IN ITEMS "kIcuMemImemDepth = 512;" "kIcuMxmImemDepth = 256;"
    "kIcuC2cImemDepth = 256;" "kIcuMemIqDepth = 8;" "kIcuMxmIqDepth = 8;"
    "kIcuMemMacroContextDepth = 16;" "kIcuMxmMacroContextDepth = 24;"
    "kIcuFetchLatencyCycles = 2;" "kIcuMacroImemRequestInitiationIntervalCycles = 2;"
    "kIcuMacroReservoirWords = 4;" "kIcuMacroDecodeWindowBits = 32;"
    "kIcuMacroDdbDepth = 4;" "kIcuMacroAdmissionLookahead = 6;" "kIcuMacroDdbRunCapacity = 4;")
    string(FIND "${actual}" "${expected}" position)
    if(position LESS 0)
        message(FATAL_ERROR "ICU override did not reach generated header: ${expected}")
    endif()
endforeach()

# A native target can have explicit iMEM sizing while keeping BARR absent.
file(READ "${SOURCE_DIR}/config/ftlpu-lpu16-decode.json" native)
string(JSON native SET "${native}" icu "{\"common_barrier_encoding_version\":0,\"common_barrier_capacity\":0,\"mem_imem_depth\":32}")
file(WRITE "${test_dir}/native.json" "${native}")
ftlpu_generate_hardware_config("${test_dir}/native.json" "${test_dir}/native.hpp")
file(READ "${test_dir}/native.hpp" header)
foreach(s IN ITEMS "kCommonBarrierEncodingVersion = 0;" "kCommonBarrierCapacity = 0;" "kIcuMemImemDepth = 32;")
    string(FIND "${header}" "${s}" pos)
    if(pos LESS 0)
        message(FATAL_ERROR "native geometry changed capability: ${s}")
    endif()
endforeach()

# Invalid types, representability, capacities and ISA layouts must fail during
# configuration instead of quietly falling back or building a different model.
foreach(pair IN ITEMS "mem_iq_depth|0" "mem_imem_depth|-1"
    "mxm_imem_depth|2147483648" "fetch_latency_cycles|1.5"
    "macro_ddb_depth|true" "macro_ddb_depth|\"8\"" "mem_macro_contexts|null"
    "macro_reservoir_words|1" "macro_decode_window_bits|65"
    "mem_instruction_bits|128" "mem_macro_context_bits|128" "mem_iq_depth|131073"
    "common_barrier_encoding_version|0" "common_barrier_capacity|0")
    string(REPLACE "|" ";" parts "${pair}")
    list(GET parts 0 key)
    list(GET parts 1 value)
    string(JSON invalid SET "${original}" icu ${key} ${value})
    file(WRITE "${test_dir}/invalid.json" "${invalid}")
    execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${SOURCE_DIR}"
        "-DINPUT_JSON=${test_dir}/invalid.json" "-DOUTPUT_HEADER=${test_dir}/invalid.hpp"
        -P "${CMAKE_CURRENT_LIST_FILE}" RESULT_VARIABLE result OUTPUT_QUIET ERROR_QUIET)
    if(result EQUAL 0)
        message(FATAL_ERROR "icu.${key} accepted ${value}")
    endif()
endforeach()
message(STATUS "ICU configuration passed: legacy defaults, custom profile and invalid geometry")
