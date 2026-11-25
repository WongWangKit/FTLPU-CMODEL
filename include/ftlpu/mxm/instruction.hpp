#pragma once

#include <cstdint>
#include <stdexcept>

namespace ftlpu {

struct MxmComputeCommand {
    std::uint8_t activation_stream{}; // RTL [6:1].
    bool weight_buffer{};             // RTL [0].

    friend bool operator==(const MxmComputeCommand&,
        const MxmComputeCommand&) = default;
};

struct MxmLoadCommand {
    bool weight_buffer{}; // RTL [1].
    bool column_mode{};   // RTL [0], MATRIX=0 / COLUMN=1.

    friend bool operator==(const MxmLoadCommand&,
        const MxmLoadCommand&) = default;
};

struct MxmParamLoadCommand {
    bool param_buffer{}; // RTL [0].

    friend bool operator==(const MxmParamLoadCommand&,
        const MxmParamLoadCommand&) = default;
};

struct MxmDequantCommand {
    bool param_buffer{}; // RTL [0].

    friend bool operator==(const MxmDequantCommand&,
        const MxmDequantCommand&) = default;
};

enum class MxmBackendOperation : std::uint8_t {
    DirectOut = 0,
    Store = 1,
    AccStore = 2,
    AccOut = 3,
};

struct MxmBackendCommand {
    bool out_bf16{}; // RTL [12].
    std::uint16_t address{}; // RTL [11:2] for the current 1024-row array.
    MxmBackendOperation opcode{MxmBackendOperation::DirectOut}; // RTL [1:0].

    friend bool operator==(const MxmBackendCommand&,
        const MxmBackendCommand&) = default;
};

namespace isa {

inline std::uint64_t encode_mxm_command(MxmLoadCommand command)
{
    return (static_cast<std::uint64_t>(command.weight_buffer) << 1)
        | static_cast<std::uint64_t>(command.column_mode);
}

inline std::uint64_t encode_mxm_command(MxmComputeCommand command)
{
    if (command.activation_stream > 30)
        throw std::logic_error(
            "MXM compute activation stream is outside the RTL domain");
    return (static_cast<std::uint64_t>(command.activation_stream) << 1)
        | static_cast<std::uint64_t>(command.weight_buffer);
}

inline std::uint64_t encode_mxm_command(MxmDequantCommand command)
{
    return static_cast<std::uint64_t>(command.param_buffer);
}

inline std::uint64_t encode_mxm_command(MxmParamLoadCommand command)
{
    return static_cast<std::uint64_t>(command.param_buffer);
}

inline std::uint64_t encode_mxm_command(MxmBackendCommand command)
{
    if (command.address >= 1024)
        throw std::logic_error("MXM Backend address is outside the RTL domain");
    const auto opcode = static_cast<std::uint8_t>(command.opcode);
    if (opcode > 3)
        throw std::logic_error("MXM Backend opcode is outside the RTL domain");
    if (command.out_bf16
        && command.opcode != MxmBackendOperation::DirectOut
        && command.opcode != MxmBackendOperation::AccOut)
        throw std::logic_error(
            "MXM Backend BF16 output is illegal for a store operation");
    return (static_cast<std::uint64_t>(command.out_bf16) << 12)
        | (static_cast<std::uint64_t>(command.address) << 2)
        | opcode;
}

inline MxmLoadCommand decode_mxm_load_command(std::uint64_t bits)
{
    if ((bits >> 2) != 0)
        throw std::logic_error("MXM Load command has non-zero reserved bits");
    return MxmLoadCommand{
        static_cast<bool>((bits >> 1) & 1u),
        static_cast<bool>(bits & 1u)};
}

inline MxmComputeCommand decode_mxm_compute_command(std::uint64_t bits)
{
    if ((bits >> 7) != 0)
        throw std::logic_error(
            "MXM Compute command has non-zero reserved bits");
    MxmComputeCommand result{
        static_cast<std::uint8_t>((bits >> 1) & 0x3fu),
        static_cast<bool>(bits & 1u)};
    static_cast<void>(encode_mxm_command(result));
    return result;
}

inline MxmDequantCommand decode_mxm_dequant_command(std::uint64_t bits)
{
    if ((bits >> 1) != 0)
        throw std::logic_error(
            "MXM Dequant command has non-zero reserved bits");
    return MxmDequantCommand{static_cast<bool>(bits & 1u)};
}

inline MxmParamLoadCommand decode_mxm_param_load_command(std::uint64_t bits)
{
    if ((bits >> 1) != 0)
        throw std::logic_error(
            "MXM Param Load command has non-zero reserved bits");
    return MxmParamLoadCommand{static_cast<bool>(bits & 1u)};
}

inline MxmBackendCommand decode_mxm_backend_command(std::uint64_t bits)
{
    if ((bits >> 13) != 0)
        throw std::logic_error(
            "MXM Backend command has non-zero reserved bits");
    MxmBackendCommand result{
        static_cast<bool>((bits >> 12) & 1u),
        static_cast<std::uint16_t>((bits >> 2) & 0x3ffu),
        static_cast<MxmBackendOperation>(bits & 0x3u)};
    static_cast<void>(encode_mxm_command(result));
    return result;
}

} // namespace isa
} // namespace ftlpu
