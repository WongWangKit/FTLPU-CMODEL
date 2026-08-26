function(ftlpu_generate_hardware_config input_json output_header)
    if(NOT EXISTS "${input_json}")
        message(FATAL_ERROR
            "FTLPU hardware configuration does not exist: ${input_json}")
    endif()

    get_filename_component(input_json "${input_json}" ABSOLUTE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input_json}")
    file(READ "${input_json}" FTLPU_HW_JSON)

    macro(ftlpu_json_get output)
        string(JSON ${output} ERROR_VARIABLE FTLPU_JSON_ERROR
            GET "${FTLPU_HW_JSON}" ${ARGN})
        if(FTLPU_JSON_ERROR)
            string(JOIN "." FTLPU_JSON_FIELD ${ARGN})
            message(FATAL_ERROR
                "Invalid FTLPU hardware configuration field '${FTLPU_JSON_FIELD}': "
                "${FTLPU_JSON_ERROR}")
        endif()
    endmacro()

    ftlpu_json_get(FTLPU_HW_SCHEMA_VERSION schema_version)
    if(NOT FTLPU_HW_SCHEMA_VERSION EQUAL 1)
        message(FATAL_ERROR
            "Unsupported FTLPU hardware schema_version: ${FTLPU_HW_SCHEMA_VERSION}")
    endif()

    ftlpu_json_get(FTLPU_HW_TARGET_NAME target name)
    if(NOT FTLPU_HW_TARGET_NAME MATCHES "^[A-Za-z0-9_.-]+$")
        message(FATAL_ERROR
            "target.name must contain only letters, digits, '.', '_' or '-'")
    endif()

    ftlpu_json_get(FTLPU_HW_HEMISPHERES topology hemispheres)
    ftlpu_json_get(FTLPU_HW_TILES_PER_SLICE topology tiles_per_slice)
    ftlpu_json_get(FTLPU_HW_LANES_PER_TILE topology lanes_per_tile)
    ftlpu_json_get(FTLPU_HW_MEM_SLICES mem slices_per_hemisphere)
    ftlpu_json_get(FTLPU_HW_MEM_BANKS mem banks_per_slice)
    ftlpu_json_get(FTLPU_HW_MEM_ROWS_PER_BANK mem rows_per_bank)
    ftlpu_json_get(FTLPU_HW_MEM_BYTES_PER_LANE mem bytes_per_lane)
    ftlpu_json_get(FTLPU_HW_SR_REGISTERS sr registers_per_lane)
    ftlpu_json_get(FTLPU_HW_SR_BYTES_PER_STREAM sr bytes_per_stream_per_lane)
    ftlpu_json_get(FTLPU_HW_MXM_ACCUM_CONTEXTS mxm accum_contexts)
    ftlpu_json_get(FTLPU_HW_VXM_ALUS_PER_DIRECTION vxm alus)

    set(FTLPU_POSITIVE_INTEGER_FIELDS
        FTLPU_HW_HEMISPHERES
        FTLPU_HW_TILES_PER_SLICE
        FTLPU_HW_LANES_PER_TILE
        FTLPU_HW_MEM_SLICES
        FTLPU_HW_MEM_BANKS
        FTLPU_HW_MEM_ROWS_PER_BANK
        FTLPU_HW_MEM_BYTES_PER_LANE
        FTLPU_HW_SR_REGISTERS
        FTLPU_HW_SR_BYTES_PER_STREAM
        FTLPU_HW_MXM_ACCUM_CONTEXTS
        FTLPU_HW_VXM_ALUS_PER_DIRECTION)
    foreach(field IN LISTS FTLPU_POSITIVE_INTEGER_FIELDS)
        if(NOT "${${field}}" MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "${field} must be a positive integer")
        endif()
    endforeach()

    if(NOT FTLPU_HW_SR_REGISTERS EQUAL 64)
        message(FATAL_ERROR
            "sr.registers_per_lane must be 64 for the current 6-bit stream ISA")
    endif()
    math(EXPR FTLPU_HW_SR_REMAINDER "${FTLPU_HW_SR_REGISTERS} % 2")
    if(NOT FTLPU_HW_SR_REMAINDER EQUAL 0)
        message(FATAL_ERROR "sr.registers_per_lane must be even")
    endif()
    if(NOT FTLPU_HW_MEM_BYTES_PER_LANE EQUAL 1 OR
       NOT FTLPU_HW_SR_BYTES_PER_STREAM EQUAL 1)
        message(FATAL_ERROR
            "the current byte-stream datapath requires both byte-width fields to be 1")
    endif()
    if(FTLPU_HW_MXM_ACCUM_CONTEXTS GREATER 256)
        message(FATAL_ERROR
            "mxm.accum_contexts exceeds the 13-bit accumulator address capacity")
    endif()

    # MEM addresses occupy a contiguous power-of-two field in the ISA.
    set(FTLPU_ROWS_POWER_OF_TWO 1)
    while(FTLPU_ROWS_POWER_OF_TWO LESS FTLPU_HW_MEM_ROWS_PER_BANK)
        math(EXPR FTLPU_ROWS_POWER_OF_TWO "${FTLPU_ROWS_POWER_OF_TWO} * 2")
    endwhile()
    if(FTLPU_HW_MEM_ROWS_PER_BANK LESS 64 OR
       NOT FTLPU_ROWS_POWER_OF_TWO EQUAL FTLPU_HW_MEM_ROWS_PER_BANK OR
       FTLPU_HW_MEM_ROWS_PER_BANK GREATER 32768)
        message(FATAL_ERROR
            "mem.rows_per_bank must be a power of two in [64, 32768]")
    endif()

    string(JSON FTLPU_HW_MXM_MODE_COUNT ERROR_VARIABLE FTLPU_JSON_ERROR
        LENGTH "${FTLPU_HW_JSON}" mxm supported_modes)
    if(FTLPU_JSON_ERROR OR FTLPU_HW_MXM_MODE_COUNT EQUAL 0)
        message(FATAL_ERROR "mxm.supported_modes must be a non-empty array")
    endif()
    set(FTLPU_HW_MXM_SUPPORTS_NATIVE4X4 false)
    set(FTLPU_HW_MXM_SUPPORTS_LINEAR1X16 false)
    math(EXPR FTLPU_HW_MXM_MODE_LAST "${FTLPU_HW_MXM_MODE_COUNT} - 1")
    foreach(index RANGE 0 ${FTLPU_HW_MXM_MODE_LAST})
        ftlpu_json_get(FTLPU_HW_MXM_MODE mxm supported_modes ${index})
        if(FTLPU_HW_MXM_MODE STREQUAL "native4x4")
            if(FTLPU_HW_MXM_SUPPORTS_NATIVE4X4)
                message(FATAL_ERROR "mxm.supported_modes contains native4x4 twice")
            endif()
            set(FTLPU_HW_MXM_SUPPORTS_NATIVE4X4 true)
        elseif(FTLPU_HW_MXM_MODE STREQUAL "linear1x16")
            if(FTLPU_HW_MXM_SUPPORTS_LINEAR1X16)
                message(FATAL_ERROR "mxm.supported_modes contains linear1x16 twice")
            endif()
            set(FTLPU_HW_MXM_SUPPORTS_LINEAR1X16 true)
        else()
            message(FATAL_ERROR
                "Unsupported MXM mode '${FTLPU_HW_MXM_MODE}'")
        endif()
    endforeach()

    get_filename_component(output_directory "${output_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_directory}")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/hardware_config.hpp.in"
        "${output_header}"
        @ONLY)

    message(STATUS
        "FTLPU hardware target '${FTLPU_HW_TARGET_NAME}' from ${input_json}")
endfunction()
