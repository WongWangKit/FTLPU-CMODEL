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
    if(NOT FTLPU_HW_SCHEMA_VERSION EQUAL 2)
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

    # Schema v2 makes the physical stream network part of the shared target
    # description.  Paths are only JSON authoring sugar: generated C++ always
    # contains a flat, constexpr list of directed routes.
    macro(ftlpu_json_length output)
        string(JSON ${output} ERROR_VARIABLE FTLPU_JSON_ERROR
            LENGTH "${FTLPU_HW_JSON}" ${ARGN})
        if(FTLPU_JSON_ERROR)
            string(JOIN "." FTLPU_JSON_FIELD ${ARGN})
            message(FATAL_ERROR
                "Invalid FTLPU hardware configuration array '${FTLPU_JSON_FIELD}': "
                "${FTLPU_JSON_ERROR}")
        endif()
    endmacro()

    macro(ftlpu_stream_name_is_valid value context)
        if(NOT "${value}" MATCHES "^[A-Za-z0-9_.-]+$")
            message(FATAL_ERROR
                "${context} must contain only letters, digits, '.', '_' or '-'")
        endif()
    endmacro()

    macro(ftlpu_stream_column_id output column_name context)
        list(FIND FTLPU_STREAM_COLUMN_NAMES "${column_name}" ${output})
        if(${output} EQUAL -1)
            message(FATAL_ERROR
                "${context} references unknown stream column '${column_name}'")
        endif()
    endmacro()

    macro(ftlpu_stream_fabric_id output fabric_name context)
        list(FIND FTLPU_STREAM_FABRIC_NAMES "${fabric_name}" ${output})
        if(${output} EQUAL -1)
            message(FATAL_ERROR
                "${context} references unknown stream fabric '${fabric_name}'")
        endif()
    endmacro()

    macro(ftlpu_stream_direction output value context)
        if("${value}" STREQUAL "east")
            set(${output} "::ftlpu::target::StreamDirection::East")
        elseif("${value}" STREQUAL "west")
            set(${output} "::ftlpu::target::StreamDirection::West")
        else()
            message(FATAL_ERROR "${context} has invalid direction '${value}'")
        endif()
    endmacro()

    macro(ftlpu_stream_route_kind output value context)
        if("${value}" STREQUAL "normal")
            set(${output} "::ftlpu::target::StreamRouteKind::Normal")
        elseif("${value}" STREQUAL "bypass")
            set(${output} "::ftlpu::target::StreamRouteKind::Bypass")
        else()
            message(FATAL_ERROR "${context} has invalid route kind '${value}'")
        endif()
    endmacro()

    macro(ftlpu_stream_cpp_bool output value)
        if(${value})
            set(${output} true)
        else()
            set(${output} false)
        endif()
    endmacro()

    macro(ftlpu_stream_append_route name source destination direction kind enabled multicast latency context)
        ftlpu_stream_name_is_valid("${name}" "${context}.name")
        list(FIND FTLPU_STREAM_ROUTE_NAMES "${name}" FTLPU_STREAM_ROUTE_DUPLICATE)
        if(NOT FTLPU_STREAM_ROUTE_DUPLICATE EQUAL -1)
            message(FATAL_ERROR "duplicate stream route name '${name}'")
        endif()
        if(NOT "${latency}" MATCHES "^[1-9][0-9]*$")
            message(FATAL_ERROR "${context}.latency_cycles must be a positive integer")
        endif()
        list(APPEND FTLPU_STREAM_ROUTE_NAMES "${name}")
        string(APPEND FTLPU_STREAM_ROUTE_INITIALIZERS
            "    ::ftlpu::target::StreamRouteDescriptor{\"${name}\", ${source}, ${destination}, ${direction}, ${kind}, ${enabled}, ${multicast}, ${latency}},\n")
        math(EXPR FTLPU_STREAM_ROUTE_COUNT "${FTLPU_STREAM_ROUTE_COUNT} + 1")
    endmacro()

    macro(ftlpu_stream_read_port output)
        ftlpu_json_get(FTLPU_STREAM_PORT_COLUMN ${ARGN} column)
        ftlpu_json_get(FTLPU_STREAM_PORT_DIRECTION ${ARGN} direction)
        string(JOIN "." FTLPU_STREAM_PORT_CONTEXT ${ARGN})
        ftlpu_stream_column_id(FTLPU_STREAM_PORT_COLUMN_ID
            "${FTLPU_STREAM_PORT_COLUMN}" "${FTLPU_STREAM_PORT_CONTEXT}")
        ftlpu_stream_direction(FTLPU_STREAM_PORT_DIRECTION_ENUM
            "${FTLPU_STREAM_PORT_DIRECTION}" "${FTLPU_STREAM_PORT_CONTEXT}")
        set(${output}
            "::ftlpu::target::StreamPortDescriptor{${FTLPU_STREAM_PORT_COLUMN_ID}, ${FTLPU_STREAM_PORT_DIRECTION_ENUM}}")
    endmacro()

    ftlpu_json_length(FTLPU_STREAM_FABRIC_COUNT
        stream_topology fabric_template replicas)
    if(FTLPU_STREAM_FABRIC_COUNT EQUAL 0)
        message(FATAL_ERROR "stream_topology.fabric_template.replicas must not be empty")
    endif()
    if(NOT FTLPU_STREAM_FABRIC_COUNT EQUAL FTLPU_HW_HEMISPHERES)
        message(FATAL_ERROR
            "stream fabric replica count must equal topology.hemispheres")
    endif()
    set(FTLPU_STREAM_FABRIC_NAMES)
    set(FTLPU_STREAM_FABRIC_INITIALIZERS "")
    math(EXPR FTLPU_STREAM_FABRIC_LAST "${FTLPU_STREAM_FABRIC_COUNT} - 1")
    foreach(index RANGE 0 ${FTLPU_STREAM_FABRIC_LAST})
        ftlpu_json_get(FTLPU_STREAM_FABRIC_NAME
            stream_topology fabric_template replicas ${index})
        ftlpu_stream_name_is_valid("${FTLPU_STREAM_FABRIC_NAME}"
            "stream_topology.fabric_template.replicas[${index}]")
        list(FIND FTLPU_STREAM_FABRIC_NAMES "${FTLPU_STREAM_FABRIC_NAME}"
            FTLPU_STREAM_FABRIC_DUPLICATE)
        if(NOT FTLPU_STREAM_FABRIC_DUPLICATE EQUAL -1)
            message(FATAL_ERROR "duplicate stream fabric '${FTLPU_STREAM_FABRIC_NAME}'")
        endif()
        list(APPEND FTLPU_STREAM_FABRIC_NAMES "${FTLPU_STREAM_FABRIC_NAME}")
        string(APPEND FTLPU_STREAM_FABRIC_INITIALIZERS
            "    std::string_view{\"${FTLPU_STREAM_FABRIC_NAME}\"},\n")
    endforeach()

    ftlpu_json_length(FTLPU_STREAM_COLUMN_COUNT
        stream_topology fabric_template columns)
    if(FTLPU_STREAM_COLUMN_COUNT EQUAL 0)
        message(FATAL_ERROR "stream_topology.fabric_template.columns must not be empty")
    endif()
    set(FTLPU_STREAM_COLUMN_NAMES)
    set(FTLPU_STREAM_COLUMN_INITIALIZERS "")
    math(EXPR FTLPU_STREAM_COLUMN_LAST "${FTLPU_STREAM_COLUMN_COUNT} - 1")
    foreach(index RANGE 0 ${FTLPU_STREAM_COLUMN_LAST})
        ftlpu_json_get(FTLPU_STREAM_COLUMN_NAME
            stream_topology fabric_template columns ${index})
        ftlpu_stream_name_is_valid("${FTLPU_STREAM_COLUMN_NAME}"
            "stream_topology.fabric_template.columns[${index}]")
        list(FIND FTLPU_STREAM_COLUMN_NAMES "${FTLPU_STREAM_COLUMN_NAME}"
            FTLPU_STREAM_COLUMN_DUPLICATE)
        if(NOT FTLPU_STREAM_COLUMN_DUPLICATE EQUAL -1)
            message(FATAL_ERROR "duplicate stream column '${FTLPU_STREAM_COLUMN_NAME}'")
        endif()
        list(APPEND FTLPU_STREAM_COLUMN_NAMES "${FTLPU_STREAM_COLUMN_NAME}")
        string(APPEND FTLPU_STREAM_COLUMN_INITIALIZERS
            "    ::ftlpu::target::StreamColumnDescriptor{\"${FTLPU_STREAM_COLUMN_NAME}\"},\n")
    endforeach()

    set(FTLPU_STREAM_ROUTE_NAMES)
    set(FTLPU_STREAM_ROUTE_INITIALIZERS "")
    set(FTLPU_STREAM_ROUTE_COUNT 0)
    ftlpu_json_length(FTLPU_STREAM_PATH_COUNT
        stream_topology fabric_template paths)
    if(FTLPU_STREAM_PATH_COUNT GREATER 0)
        math(EXPR FTLPU_STREAM_PATH_LAST "${FTLPU_STREAM_PATH_COUNT} - 1")
        foreach(path_index RANGE 0 ${FTLPU_STREAM_PATH_LAST})
            set(FTLPU_STREAM_PATH_CONTEXT
                "stream_topology.fabric_template.paths[${path_index}]")
            ftlpu_json_get(FTLPU_STREAM_PATH_NAME
                stream_topology fabric_template paths ${path_index} name)
            ftlpu_stream_name_is_valid("${FTLPU_STREAM_PATH_NAME}"
                "${FTLPU_STREAM_PATH_CONTEXT}.name")
            ftlpu_json_get(FTLPU_STREAM_PATH_FORWARD
                stream_topology fabric_template paths ${path_index} forward_direction)
            ftlpu_json_get(FTLPU_STREAM_PATH_REVERSE
                stream_topology fabric_template paths ${path_index} reverse_direction)
            ftlpu_json_get(FTLPU_STREAM_PATH_BIDIRECTIONAL
                stream_topology fabric_template paths ${path_index} bidirectional)
            ftlpu_json_get(FTLPU_STREAM_PATH_KIND
                stream_topology fabric_template paths ${path_index} kind)
            ftlpu_json_get(FTLPU_STREAM_PATH_ENABLED
                stream_topology fabric_template paths ${path_index} enabled_by_default)
            ftlpu_json_get(FTLPU_STREAM_PATH_MULTICAST
                stream_topology fabric_template paths ${path_index} multicast_allowed)
            ftlpu_json_get(FTLPU_STREAM_PATH_LATENCY
                stream_topology fabric_template paths ${path_index} hop_latency_cycles)
            ftlpu_stream_direction(FTLPU_STREAM_PATH_FORWARD_ENUM
                "${FTLPU_STREAM_PATH_FORWARD}" "${FTLPU_STREAM_PATH_CONTEXT}")
            ftlpu_stream_direction(FTLPU_STREAM_PATH_REVERSE_ENUM
                "${FTLPU_STREAM_PATH_REVERSE}" "${FTLPU_STREAM_PATH_CONTEXT}")
            ftlpu_stream_route_kind(FTLPU_STREAM_PATH_KIND_ENUM
                "${FTLPU_STREAM_PATH_KIND}" "${FTLPU_STREAM_PATH_CONTEXT}")
            ftlpu_stream_cpp_bool(FTLPU_STREAM_PATH_BIDIRECTIONAL_CPP
                "${FTLPU_STREAM_PATH_BIDIRECTIONAL}")
            ftlpu_stream_cpp_bool(FTLPU_STREAM_PATH_ENABLED_CPP
                "${FTLPU_STREAM_PATH_ENABLED}")
            ftlpu_stream_cpp_bool(FTLPU_STREAM_PATH_MULTICAST_CPP
                "${FTLPU_STREAM_PATH_MULTICAST}")
            ftlpu_json_length(FTLPU_STREAM_PATH_COLUMN_COUNT
                stream_topology fabric_template paths ${path_index} columns)
            if(FTLPU_STREAM_PATH_COLUMN_COUNT LESS 2)
                message(FATAL_ERROR "${FTLPU_STREAM_PATH_CONTEXT}.columns needs at least two columns")
            endif()
            math(EXPR FTLPU_STREAM_PATH_EDGE_LAST
                "${FTLPU_STREAM_PATH_COLUMN_COUNT} - 2")
            foreach(edge_index RANGE 0 ${FTLPU_STREAM_PATH_EDGE_LAST})
                math(EXPR next_index "${edge_index} + 1")
                ftlpu_json_get(FTLPU_STREAM_PATH_SOURCE_NAME
                    stream_topology fabric_template paths ${path_index} columns ${edge_index})
                ftlpu_json_get(FTLPU_STREAM_PATH_DESTINATION_NAME
                    stream_topology fabric_template paths ${path_index} columns ${next_index})
                ftlpu_stream_column_id(FTLPU_STREAM_PATH_SOURCE
                    "${FTLPU_STREAM_PATH_SOURCE_NAME}" "${FTLPU_STREAM_PATH_CONTEXT}")
                ftlpu_stream_column_id(FTLPU_STREAM_PATH_DESTINATION
                    "${FTLPU_STREAM_PATH_DESTINATION_NAME}" "${FTLPU_STREAM_PATH_CONTEXT}")
                set(FTLPU_STREAM_FORWARD_ROUTE_NAME
                    "${FTLPU_STREAM_PATH_NAME}.forward.${edge_index}")
                ftlpu_stream_append_route(
                    "${FTLPU_STREAM_FORWARD_ROUTE_NAME}"
                    "${FTLPU_STREAM_PATH_SOURCE}" "${FTLPU_STREAM_PATH_DESTINATION}"
                    "${FTLPU_STREAM_PATH_FORWARD_ENUM}" "${FTLPU_STREAM_PATH_KIND_ENUM}"
                    "${FTLPU_STREAM_PATH_ENABLED_CPP}" "${FTLPU_STREAM_PATH_MULTICAST_CPP}"
                    "${FTLPU_STREAM_PATH_LATENCY}" "${FTLPU_STREAM_PATH_CONTEXT}")
                if(FTLPU_STREAM_PATH_BIDIRECTIONAL_CPP)
                    set(FTLPU_STREAM_REVERSE_ROUTE_NAME
                        "${FTLPU_STREAM_PATH_NAME}.reverse.${edge_index}")
                    ftlpu_stream_append_route(
                        "${FTLPU_STREAM_REVERSE_ROUTE_NAME}"
                        "${FTLPU_STREAM_PATH_DESTINATION}" "${FTLPU_STREAM_PATH_SOURCE}"
                        "${FTLPU_STREAM_PATH_REVERSE_ENUM}" "${FTLPU_STREAM_PATH_KIND_ENUM}"
                        "${FTLPU_STREAM_PATH_ENABLED_CPP}" "${FTLPU_STREAM_PATH_MULTICAST_CPP}"
                        "${FTLPU_STREAM_PATH_LATENCY}" "${FTLPU_STREAM_PATH_CONTEXT}")
                endif()
            endforeach()
        endforeach()
    endif()

    ftlpu_json_length(FTLPU_STREAM_EXPLICIT_ROUTE_COUNT
        stream_topology fabric_template routes)
    if(FTLPU_STREAM_EXPLICIT_ROUTE_COUNT GREATER 0)
        math(EXPR FTLPU_STREAM_EXPLICIT_ROUTE_LAST
            "${FTLPU_STREAM_EXPLICIT_ROUTE_COUNT} - 1")
        foreach(route_index RANGE 0 ${FTLPU_STREAM_EXPLICIT_ROUTE_LAST})
            set(FTLPU_STREAM_ROUTE_CONTEXT
                "stream_topology.fabric_template.routes[${route_index}]")
            ftlpu_json_get(FTLPU_STREAM_ROUTE_NAME
                stream_topology fabric_template routes ${route_index} name)
            ftlpu_json_get(FTLPU_STREAM_ROUTE_SOURCE_NAME
                stream_topology fabric_template routes ${route_index} source)
            ftlpu_json_get(FTLPU_STREAM_ROUTE_DESTINATION_NAME
                stream_topology fabric_template routes ${route_index} destination)
            ftlpu_json_get(FTLPU_STREAM_ROUTE_DIRECTION
                stream_topology fabric_template routes ${route_index} direction)
            ftlpu_json_get(FTLPU_STREAM_ROUTE_KIND
                stream_topology fabric_template routes ${route_index} kind)
            ftlpu_json_get(FTLPU_STREAM_ROUTE_ENABLED
                stream_topology fabric_template routes ${route_index} enabled_by_default)
            ftlpu_json_get(FTLPU_STREAM_ROUTE_MULTICAST
                stream_topology fabric_template routes ${route_index} multicast_allowed)
            ftlpu_json_get(FTLPU_STREAM_ROUTE_LATENCY
                stream_topology fabric_template routes ${route_index} latency_cycles)
            ftlpu_stream_column_id(FTLPU_STREAM_ROUTE_SOURCE
                "${FTLPU_STREAM_ROUTE_SOURCE_NAME}" "${FTLPU_STREAM_ROUTE_CONTEXT}")
            ftlpu_stream_column_id(FTLPU_STREAM_ROUTE_DESTINATION
                "${FTLPU_STREAM_ROUTE_DESTINATION_NAME}" "${FTLPU_STREAM_ROUTE_CONTEXT}")
            ftlpu_stream_direction(FTLPU_STREAM_ROUTE_DIRECTION_ENUM
                "${FTLPU_STREAM_ROUTE_DIRECTION}" "${FTLPU_STREAM_ROUTE_CONTEXT}")
            ftlpu_stream_route_kind(FTLPU_STREAM_ROUTE_KIND_ENUM
                "${FTLPU_STREAM_ROUTE_KIND}" "${FTLPU_STREAM_ROUTE_CONTEXT}")
            ftlpu_stream_cpp_bool(FTLPU_STREAM_ROUTE_ENABLED_CPP
                "${FTLPU_STREAM_ROUTE_ENABLED}")
            ftlpu_stream_cpp_bool(FTLPU_STREAM_ROUTE_MULTICAST_CPP
                "${FTLPU_STREAM_ROUTE_MULTICAST}")
            ftlpu_stream_append_route(
                "${FTLPU_STREAM_ROUTE_NAME}"
                "${FTLPU_STREAM_ROUTE_SOURCE}" "${FTLPU_STREAM_ROUTE_DESTINATION}"
                "${FTLPU_STREAM_ROUTE_DIRECTION_ENUM}" "${FTLPU_STREAM_ROUTE_KIND_ENUM}"
                "${FTLPU_STREAM_ROUTE_ENABLED_CPP}" "${FTLPU_STREAM_ROUTE_MULTICAST_CPP}"
                "${FTLPU_STREAM_ROUTE_LATENCY}" "${FTLPU_STREAM_ROUTE_CONTEXT}")
        endforeach()
    endif()

    ftlpu_json_get(FTLPU_STREAM_MEM_SLICES_PER_GROUP
        stream_topology bindings mem slices_per_group)
    if(NOT FTLPU_STREAM_MEM_SLICES_PER_GROUP MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "stream_topology.bindings.mem.slices_per_group must be positive")
    endif()
    if(NOT FTLPU_STREAM_MEM_SLICES_PER_GROUP EQUAL FTLPU_HW_TILES_PER_SLICE)
        message(FATAL_ERROR
            "the current CModel requires MEM slices_per_group to equal topology.tiles_per_slice")
    endif()
    math(EXPR FTLPU_STREAM_MEM_GROUP_REMAINDER
        "${FTLPU_HW_MEM_SLICES} % ${FTLPU_STREAM_MEM_SLICES_PER_GROUP}")
    if(NOT FTLPU_STREAM_MEM_GROUP_REMAINDER EQUAL 0)
        message(FATAL_ERROR "mem.slices_per_hemisphere must divide evenly into stream MEM groups")
    endif()
    math(EXPR FTLPU_STREAM_EXPECTED_MEM_BOUNDARIES
        "${FTLPU_HW_MEM_SLICES} / ${FTLPU_STREAM_MEM_SLICES_PER_GROUP} + 1")
    ftlpu_json_length(FTLPU_STREAM_MEM_BOUNDARY_COUNT
        stream_topology bindings mem boundary_columns)
    if(NOT FTLPU_STREAM_MEM_BOUNDARY_COUNT EQUAL FTLPU_STREAM_EXPECTED_MEM_BOUNDARIES)
        message(FATAL_ERROR
            "stream_topology.bindings.mem.boundary_columns has the wrong size")
    endif()
    set(FTLPU_STREAM_MEM_BOUNDARY_INITIALIZERS "")
    math(EXPR FTLPU_STREAM_MEM_BOUNDARY_LAST
        "${FTLPU_STREAM_MEM_BOUNDARY_COUNT} - 1")
    foreach(index RANGE 0 ${FTLPU_STREAM_MEM_BOUNDARY_LAST})
        ftlpu_json_get(FTLPU_STREAM_MEM_BOUNDARY_NAME
            stream_topology bindings mem boundary_columns ${index})
        ftlpu_stream_column_id(FTLPU_STREAM_MEM_BOUNDARY_ID
            "${FTLPU_STREAM_MEM_BOUNDARY_NAME}"
            "stream_topology.bindings.mem.boundary_columns[${index}]")
        string(APPEND FTLPU_STREAM_MEM_BOUNDARY_INITIALIZERS
            "    std::size_t{${FTLPU_STREAM_MEM_BOUNDARY_ID}},\n")
    endforeach()

    ftlpu_stream_read_port(FTLPU_STREAM_SXM_EAST_INPUT
        stream_topology bindings sxm east_input)
    ftlpu_stream_read_port(FTLPU_STREAM_SXM_EAST_OUTPUT
        stream_topology bindings sxm east_output)
    ftlpu_stream_read_port(FTLPU_STREAM_SXM_WEST_INPUT
        stream_topology bindings sxm west_input)
    ftlpu_stream_read_port(FTLPU_STREAM_SXM_WEST_OUTPUT
        stream_topology bindings sxm west_output)
    ftlpu_stream_read_port(FTLPU_STREAM_MXM_WEIGHT_INPUT
        stream_topology bindings mxm weight_input)
    ftlpu_stream_read_port(FTLPU_STREAM_MXM_ACTIVATION_INPUT
        stream_topology bindings mxm activation_input)
    ftlpu_stream_read_port(FTLPU_STREAM_MXM_RESULT_OUTPUT
        stream_topology bindings mxm result_output)
    ftlpu_stream_read_port(FTLPU_STREAM_VXM_INPUT
        stream_topology bindings vxm input)
    ftlpu_stream_read_port(FTLPU_STREAM_VXM_OUTPUT
        stream_topology bindings vxm output)
    ftlpu_stream_read_port(FTLPU_STREAM_C2C_TX_INPUT
        stream_topology bindings c2c tx_input)
    ftlpu_stream_read_port(FTLPU_STREAM_C2C_RX_OUTPUT
        stream_topology bindings c2c rx_output)

    set(FTLPU_STREAM_TRANSFER_INITIALIZERS "")
    set(FTLPU_STREAM_TRANSFER_NAMES)
    ftlpu_json_length(FTLPU_STREAM_TRANSFER_COUNT
        stream_topology system_transfers)
    if(FTLPU_STREAM_TRANSFER_COUNT GREATER 0)
        math(EXPR FTLPU_STREAM_TRANSFER_LAST "${FTLPU_STREAM_TRANSFER_COUNT} - 1")
        foreach(index RANGE 0 ${FTLPU_STREAM_TRANSFER_LAST})
            set(FTLPU_STREAM_TRANSFER_CONTEXT "stream_topology.system_transfers[${index}]")
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_NAME
                stream_topology system_transfers ${index} name)
            ftlpu_stream_name_is_valid("${FTLPU_STREAM_TRANSFER_NAME}"
                "${FTLPU_STREAM_TRANSFER_CONTEXT}.name")
            list(FIND FTLPU_STREAM_TRANSFER_NAMES "${FTLPU_STREAM_TRANSFER_NAME}"
                FTLPU_STREAM_TRANSFER_DUPLICATE)
            if(NOT FTLPU_STREAM_TRANSFER_DUPLICATE EQUAL -1)
                message(FATAL_ERROR "duplicate stream system transfer '${FTLPU_STREAM_TRANSFER_NAME}'")
            endif()
            list(APPEND FTLPU_STREAM_TRANSFER_NAMES "${FTLPU_STREAM_TRANSFER_NAME}")
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_KIND
                stream_topology system_transfers ${index} kind)
            if(NOT FTLPU_STREAM_TRANSFER_KIND STREQUAL "passive_bridge")
                message(FATAL_ERROR
                    "${FTLPU_STREAM_TRANSFER_CONTEXT} has unsupported kind '${FTLPU_STREAM_TRANSFER_KIND}'")
            endif()
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_SOURCE_FABRIC_NAME
                stream_topology system_transfers ${index} source fabric)
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_SOURCE_COLUMN_NAME
                stream_topology system_transfers ${index} source column)
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_SOURCE_DIRECTION
                stream_topology system_transfers ${index} source direction)
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_DESTINATION_FABRIC_NAME
                stream_topology system_transfers ${index} destination fabric)
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_DESTINATION_COLUMN_NAME
                stream_topology system_transfers ${index} destination column)
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_DESTINATION_DIRECTION
                stream_topology system_transfers ${index} destination direction)
            ftlpu_json_get(FTLPU_STREAM_TRANSFER_LATENCY
                stream_topology system_transfers ${index} latency_cycles)
            if(NOT FTLPU_STREAM_TRANSFER_LATENCY MATCHES "^[1-9][0-9]*$")
                message(FATAL_ERROR
                    "${FTLPU_STREAM_TRANSFER_CONTEXT}.latency_cycles must be positive")
            endif()
            ftlpu_stream_fabric_id(FTLPU_STREAM_TRANSFER_SOURCE_FABRIC
                "${FTLPU_STREAM_TRANSFER_SOURCE_FABRIC_NAME}" "${FTLPU_STREAM_TRANSFER_CONTEXT}")
            ftlpu_stream_fabric_id(FTLPU_STREAM_TRANSFER_DESTINATION_FABRIC
                "${FTLPU_STREAM_TRANSFER_DESTINATION_FABRIC_NAME}" "${FTLPU_STREAM_TRANSFER_CONTEXT}")
            ftlpu_stream_column_id(FTLPU_STREAM_TRANSFER_SOURCE_COLUMN
                "${FTLPU_STREAM_TRANSFER_SOURCE_COLUMN_NAME}" "${FTLPU_STREAM_TRANSFER_CONTEXT}")
            ftlpu_stream_column_id(FTLPU_STREAM_TRANSFER_DESTINATION_COLUMN
                "${FTLPU_STREAM_TRANSFER_DESTINATION_COLUMN_NAME}" "${FTLPU_STREAM_TRANSFER_CONTEXT}")
            ftlpu_stream_direction(FTLPU_STREAM_TRANSFER_SOURCE_DIRECTION_ENUM
                "${FTLPU_STREAM_TRANSFER_SOURCE_DIRECTION}" "${FTLPU_STREAM_TRANSFER_CONTEXT}")
            ftlpu_stream_direction(FTLPU_STREAM_TRANSFER_DESTINATION_DIRECTION_ENUM
                "${FTLPU_STREAM_TRANSFER_DESTINATION_DIRECTION}" "${FTLPU_STREAM_TRANSFER_CONTEXT}")
            string(APPEND FTLPU_STREAM_TRANSFER_INITIALIZERS
                "    ::ftlpu::target::StreamSystemTransferDescriptor{\"${FTLPU_STREAM_TRANSFER_NAME}\", ::ftlpu::target::StreamSystemTransferKind::PassiveBridge, ${FTLPU_STREAM_TRANSFER_SOURCE_FABRIC}, {${FTLPU_STREAM_TRANSFER_SOURCE_COLUMN}, ${FTLPU_STREAM_TRANSFER_SOURCE_DIRECTION_ENUM}}, ${FTLPU_STREAM_TRANSFER_DESTINATION_FABRIC}, {${FTLPU_STREAM_TRANSFER_DESTINATION_COLUMN}, ${FTLPU_STREAM_TRANSFER_DESTINATION_DIRECTION_ENUM}}, ${FTLPU_STREAM_TRANSFER_LATENCY}},\n")
        endforeach()
    endif()

    get_filename_component(output_directory "${output_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_directory}")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/hardware_config.hpp.in"
        "${output_header}"
        @ONLY)

    message(STATUS
        "FTLPU hardware target '${FTLPU_HW_TARGET_NAME}' from ${input_json}")
endfunction()
