# Included inside ftlpu_generate_hardware_config after the optional integer
# reader is defined. Keep omitted fields compatible with pre-ICU-profile JSON.
# The final column distinguishes existing container fields from build-only
# frontend settings; the compiler rejects overlays of the latter.
set(FTLPU_ICU_PARAMETERS
    "mem_instruction_bits|MemInstructionBits|96|abi"
    "mem_imem_depth|MemImemDepth|131072|abi"
    "mem_iq_depth|MemIqDepth|16|build"
    "mem_macro_contexts|MemMacroContextDepth|32|abi"
    "mxm_instruction_bits|MxmInstructionBits|128|abi"
    "mxm_imem_depth|MxmImemDepth|65536|abi"
    "mxm_iq_depth|MxmIqDepth|16|build"
    "mxm_macro_contexts|MxmMacroContextDepth|40|abi"
    "vxm_instruction_bits|VxmInstructionBits|96|abi"
    "vxm_imem_depth|VxmImemDepth|2048|abi"
    "vxm_iq_depth|VxmIqDepth|16|build"
    "vxm_macro_contexts|VxmMacroContextDepth|4|build"
    "sxm_instruction_bits|SxmInstructionBits|96|abi"
    "sxm_imem_depth|SxmImemDepth|2048|abi"
    "sxm_iq_depth|SxmIqDepth|16|build"
    "sxm_macro_contexts|SxmMacroContextDepth|4|build"
    "c2c_instruction_bits|C2cInstructionBits|96|build"
    "c2c_imem_depth|C2cImemDepth|32768|build"
    "c2c_iq_depth|C2cIqDepth|16|build"
    "c2c_macro_contexts|C2cMacroContextDepth|4|build"
    "mem_macro_context_bits|MemMacroContextBits|256|abi"
    "mxm_macro_context_bits|MxmMacroContextBits|256|abi"
    "fetch_latency_cycles|FetchLatencyCycles|1|build"
    "macro_imem_request_ii_cycles|MacroImemRequestInitiationIntervalCycles|1|build"
    "barrier_latency_cycles|BarrierLatencyCycles|35|build"
    "macro_reservoir_words|MacroReservoirWords|3|build"
    "macro_decode_window_bits|MacroDecodeWindowBits|64|build"
    "macro_ddb_depth|MacroDdbDepth|8|build"
    "macro_admission_lookahead|MacroAdmissionLookahead|8|build"
    "macro_ddb_run_capacity|MacroDdbRunCapacity|8|build")
set(FTLPU_ICU_GEOMETRY_DECLARATIONS "")
set(FTLPU_ICU_BUILD_PARAMETER_ENTRIES "")
set(FTLPU_ICU_BUILD_PARAMETER_COUNT 0)
foreach(spec IN LISTS FTLPU_ICU_PARAMETERS)
    string(REPLACE "|" ";" parts "${spec}")
    list(GET parts 0 key)
    list(GET parts 1 suffix)
    list(GET parts 2 default_value)
    list(GET parts 3 storage)
    ftlpu_timing_parameter(FTLPU_ICU_${key} ${default_value} icu ${key})
    string(APPEND FTLPU_ICU_GEOMETRY_DECLARATIONS
        "inline constexpr std::size_t kIcu${suffix} = ${FTLPU_ICU_${key}};\n")
    if(storage STREQUAL "build")
        string(APPEND FTLPU_ICU_BUILD_PARAMETER_ENTRIES
            "    IcuBuildParameter{\"${key}\", kIcu${suffix}},\n")
        math(EXPR FTLPU_ICU_BUILD_PARAMETER_COUNT "${FTLPU_ICU_BUILD_PARAMETER_COUNT} + 1")
    endif()
endforeach()

# Slot widths and packed Context layout are defined by the current ISA.
foreach(family mem mxm vxm sxm c2c)
    set(width 96)
    if(family STREQUAL "mxm")
        set(width 128)
    endif()
    if(NOT FTLPU_ICU_${family}_instruction_bits EQUAL width)
        message(FATAL_ERROR "icu.${family}_instruction_bits must be ${width} for the current ISA")
    endif()
    if(FTLPU_ICU_${family}_iq_depth GREATER FTLPU_ICU_${family}_imem_depth)
        message(FATAL_ERROR "icu.${family}_iq_depth exceeds its i-MEM depth")
    endif()
endforeach()
foreach(family mem mxm)
    if(NOT FTLPU_ICU_${family}_macro_context_bits EQUAL 256)
        message(FATAL_ERROR "icu.${family}_macro_context_bits must be 256 for packed-v1")
    endif()
endforeach()
if(FTLPU_ICU_macro_reservoir_words LESS 2)
    message(FATAL_ERROR "icu.macro_reservoir_words must be at least 2")
endif()
if(FTLPU_ICU_macro_decode_window_bits GREATER 64)
    message(FATAL_ERROR "icu.macro_decode_window_bits must be at most 64")
endif()
# The C++ queue also checks IQ size against the actual geometry-dependent
# packet length. Keep that check beside the codec, not a duplicated estimate.
