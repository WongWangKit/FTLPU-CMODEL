#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/lane.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ftlpu {

namespace isa {

// FTLPU hardware ISA encoding for the modeled slice.
//
// MEM 32b:
//   [1:0] opcode, [7:2] stream, [13:8] map stream, [26:14] SRAM address.
// MXM control 32b:
//   IW     [1:0] opcode, [6:2] supercell column, [7] weight buffer.
//   Compute [1:0] opcode, [2] weight buffer.
//   Output [1:0] opcode, [12:7] stream base.
// VXM ALU 3x32b:
//   word0 [4:0] opcode, [6:5] lhs kind, [12:7] lhs index,
//         [14:13] rhs kind, [20:15] rhs index, [21] cast target,
//         [22] output valid, [28:23] output stream.
//   word1 lhs immediate literal, word2 rhs immediate literal.
// ICU queue command 32b:
//   NOP    [1:0] opcode, [31:2] cycle count.
//   Repeat [1:0] opcode, [11:2] count, [19:12] interval,
//          [31:20] signed MEM address stride.
using EncodedMemInstruction = std::uint32_t;
using EncodedMxmInstruction = std::uint32_t;
using EncodedIcuCommand = std::uint32_t;

struct EncodedVxmInstruction {
    std::array<std::uint32_t, 3> words{};
};

enum class IcuCommandOpcode : std::uint8_t {
    Instruction = 0,
    Nop = 1,
    Repeat = 2,
};

namespace detail {

inline std::uint32_t float_to_bits(float value)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float bits_to_float(std::uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void require_unsigned_fit(std::uint64_t value, std::uint64_t max, const char* field)
{
    if (value > max) {
        throw std::out_of_range(field);
    }
}

inline void require_signed_fit(std::int64_t value, std::int64_t min, std::int64_t max, const char* field)
{
    if (value < min || value > max) {
        throw std::out_of_range(field);
    }
}

inline std::uint64_t low_bits(std::uint64_t word, unsigned shift, std::uint64_t mask)
{
    return (word >> shift) & mask;
}

inline void require_reserved_zero(std::uint64_t word, std::uint64_t used_mask, const char* instruction)
{
    if ((word & ~used_mask) != 0) {
        throw std::logic_error(instruction);
    }
}

inline void require_default_float(float value, const char* field)
{
    if (value != 1.0f) {
        throw std::logic_error(field);
    }
}

inline void require_zero_float(float value, const char* field)
{
    if (value != 0.0f) {
        throw std::logic_error(field);
    }
}

inline void require_operand_index_fits(const VxmLaneOperand& operand, const char* field)
{
    switch (operand.kind) {
    case VxmLaneOperandKind::AluOutput:
        require_unsigned_fit(operand.index, VxmLane::kAluCount - 1, field);
        return;
    case VxmLaneOperandKind::StreamInt32:
        require_unsigned_fit(operand.index, hw::kStreams - 4, field);
        return;
    case VxmLaneOperandKind::Immediate:
        require_unsigned_fit(operand.index, 0, field);
        return;
    }
    throw std::logic_error("unknown VXM operand kind");
}

inline void require_operand_hardware_encodable(const VxmLaneOperand& operand, const char* field)
{
    require_operand_index_fits(operand, field);
    require_default_float(operand.scale, "VXM operand scale is model metadata, not a hardware ISA field");
    if (operand.kind != VxmLaneOperandKind::Immediate) {
        require_zero_float(operand.immediate, "VXM non-immediate operand carries a literal value");
    }
}

} // namespace detail

inline EncodedMemInstruction encode_mem_instruction(const MemInstruction& instruction)
{
    constexpr std::uint64_t kOpcodeMask = 0x3;
    constexpr std::uint64_t kStreamMask = 0x3f;
    constexpr std::uint64_t kAddressMask = hw::kSramDepthWords - 1;

    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.opcode),
        kOpcodeMask,
        "MEM opcode does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.stream),
        kStreamMask,
        "MEM stream does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.map_stream),
        kStreamMask,
        "MEM map stream does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.address),
        kAddressMask,
        "MEM address does not fit encoded instruction");

    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(instruction.opcode)
        | (static_cast<std::uint64_t>(instruction.stream) << 2)
        | (static_cast<std::uint64_t>(instruction.map_stream) << 8)
        | (static_cast<std::uint64_t>(instruction.address) << 14));
}

inline MemInstruction decode_mem_instruction(EncodedMemInstruction word)
{
    constexpr std::uint64_t kUsedMask = 0x07ffffffull;
    detail::require_reserved_zero(word, kUsedMask, "encoded MEM instruction has non-zero reserved bits");
    const auto opcode = static_cast<MemOpcode>(detail::low_bits(word, 0, 0x3));
    const auto stream = static_cast<std::size_t>(detail::low_bits(word, 2, 0x3f));
    const auto map_stream = static_cast<std::size_t>(detail::low_bits(word, 8, 0x3f));
    const auto address = static_cast<std::size_t>(detail::low_bits(word, 14, 0x1fff));

    switch (opcode) {
    case MemOpcode::Read:
        return MemInstruction::Read(address, stream);
    case MemOpcode::Write:
        return MemInstruction::Write(address, stream);
    case MemOpcode::Gather:
        return MemInstruction::Gather(stream, map_stream);
    case MemOpcode::Scatter:
        return MemInstruction::Scatter(stream, map_stream);
    }
    throw std::logic_error("unknown encoded MEM opcode");
}

inline EncodedMxmInstruction encode_mxm_instruction(const MxmControlInstruction& instruction)
{
    constexpr std::uint32_t kOpcodeMask = 0x3;
    constexpr std::uint32_t kSupercellColumnMask = 0x1f;
    constexpr std::uint32_t kWeightBufferMask = 0x1;
    constexpr std::uint32_t kStreamBaseMask = 0x3f;
    constexpr std::uint32_t kActivationStreamMask = 0x3f;

    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.opcode),
        kOpcodeMask,
        "MXM opcode does not fit encoded instruction");
    const auto opcode = static_cast<std::uint32_t>(instruction.opcode);
    switch (instruction.opcode) {
    case MxmControlOpcode::IW:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.supercell_column),
            kSupercellColumnMask,
            "MXM supercell column does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint32_t>(instruction.supercell_column) << 2)
            | (static_cast<std::uint32_t>(instruction.weight_buffer) << 7);
    case MxmControlOpcode::Compute:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.activation_stream_base),
            kActivationStreamMask,
            "MXM activation stream base does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint32_t>(instruction.weight_buffer) << 2)
            | (static_cast<std::uint32_t>(instruction.activation_stream_base) << 3);
    case MxmControlOpcode::Output:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.stream_base),
            kStreamBaseMask,
            "MXM stream base does not fit encoded instruction");
        return opcode | (static_cast<std::uint32_t>(instruction.stream_base) << 7);
    }
    throw std::logic_error("unknown MXM opcode");
}

inline MxmControlInstruction decode_mxm_instruction(EncodedMxmInstruction word)
{
    const auto opcode = static_cast<MxmControlOpcode>(word & 0x3u);
    const auto supercell_column = static_cast<std::size_t>((word >> 2) & 0x1fu);
    const auto iw_weight_buffer = static_cast<std::size_t>((word >> 7) & 0x1u);
    const auto compute_weight_buffer = static_cast<std::size_t>((word >> 2) & 0x1u);
    const auto compute_activation_stream_base = static_cast<std::size_t>((word >> 3) & 0x3fu);
    const auto stream_base = static_cast<std::size_t>((word >> 7) & 0x3fu);

    switch (opcode) {
    case MxmControlOpcode::IW:
        detail::require_reserved_zero(word, 0x000000ffu, "encoded MXM IW instruction has non-zero reserved bits");
        return MxmControlInstruction::IW(supercell_column, iw_weight_buffer);
    case MxmControlOpcode::Compute:
        detail::require_reserved_zero(word, 0x000001ffu, "encoded MXM Compute instruction has non-zero reserved bits");
        return MxmControlInstruction::Compute(compute_weight_buffer, compute_activation_stream_base);
    case MxmControlOpcode::Output:
        detail::require_reserved_zero(word, 0x00001f83u, "encoded MXM Output instruction has non-zero reserved bits");
        return MxmControlInstruction::Output(stream_base);
    }
    throw std::logic_error("unknown encoded MXM opcode");
}

inline EncodedVxmInstruction encode_vxm_instruction(const VxmLaneAluInstruction& instruction)
{
    constexpr std::uint32_t kOpcodeMask = 0x1f;
    constexpr std::uint32_t kOperandKindMask = 0x3;
    constexpr std::uint32_t kOutputStreamMask = 0x3f;

    detail::require_default_float(instruction.scale, "VXM instruction scale is model metadata, not a hardware ISA field");
    if (instruction.output_zero_point != 0) {
        throw std::logic_error("VXM output zero point is synthesized with ALU ops, not an encoded ISA field");
    }
    detail::require_operand_hardware_encodable(instruction.lhs, "VXM lhs operand index does not fit encoded instruction");
    detail::require_operand_hardware_encodable(instruction.rhs, "VXM rhs operand index does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.opcode),
        kOpcodeMask,
        "VXM opcode does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.lhs.kind),
        kOperandKindMask,
        "VXM lhs operand kind does not fit encoded instruction");
    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.rhs.kind),
        kOperandKindMask,
        "VXM rhs operand kind does not fit encoded instruction");
    if (instruction.output_stream.has_value()) {
        detail::require_unsigned_fit(
            *instruction.output_stream,
            kOutputStreamMask,
            "VXM output stream does not fit encoded instruction");
    }

    auto control = static_cast<std::uint32_t>(instruction.opcode)
        | (static_cast<std::uint32_t>(instruction.lhs.kind) << 5)
        | (static_cast<std::uint32_t>(instruction.lhs.index) << 7)
        | (static_cast<std::uint32_t>(instruction.rhs.kind) << 13)
        | (static_cast<std::uint32_t>(instruction.rhs.index) << 15)
        | (static_cast<std::uint32_t>(instruction.cast_target) << 21);
    if (instruction.output_stream.has_value()) {
        control |= 1u << 22;
        control |= static_cast<std::uint32_t>(*instruction.output_stream) << 23;
    }

    return EncodedVxmInstruction {
        std::array<std::uint32_t, 3> {
            control,
            detail::float_to_bits(instruction.lhs.immediate),
            detail::float_to_bits(instruction.rhs.immediate),
        },
    };
}

inline VxmLaneAluInstruction decode_vxm_instruction(const EncodedVxmInstruction& encoded)
{
    const auto control = encoded.words[0];
    constexpr std::uint32_t kUsedMask = 0x1fffffffu;
    detail::require_reserved_zero(control, kUsedMask, "encoded VXM control word has non-zero reserved bits");
    auto instruction = VxmLaneAluInstruction {};
    instruction.opcode = static_cast<VxmAluOpcode>(control & 0x1fu);
    instruction.lhs.kind = static_cast<VxmLaneOperandKind>((control >> 5) & 0x3u);
    instruction.lhs.index = static_cast<std::size_t>((control >> 7) & 0x3fu);
    instruction.rhs.kind = static_cast<VxmLaneOperandKind>((control >> 13) & 0x3u);
    instruction.rhs.index = static_cast<std::size_t>((control >> 15) & 0x3fu);
    instruction.cast_target = static_cast<VxmCastTarget>((control >> 21) & 0x1u);
    if (((control >> 22) & 0x1u) != 0u) {
        instruction.output_stream = static_cast<std::size_t>((control >> 23) & 0x3fu);
    } else {
        instruction.output_stream.reset();
    }
    instruction.lhs.immediate = detail::bits_to_float(encoded.words[1]);
    instruction.rhs.immediate = detail::bits_to_float(encoded.words[2]);
    instruction.scale = 1.0f;
    instruction.output_zero_point = 0;
    instruction.lhs.scale = 1.0f;
    instruction.rhs.scale = 1.0f;
    if (instruction.lhs.kind != VxmLaneOperandKind::Immediate) {
        instruction.lhs.immediate = 0.0f;
    }
    if (instruction.rhs.kind != VxmLaneOperandKind::Immediate) {
        instruction.rhs.immediate = 0.0f;
    }
    return instruction;
}

inline EncodedIcuCommand encode_icu_nop(std::size_t cycles)
{
    constexpr std::uint64_t kCountMask = 0x3fffffffull;
    detail::require_unsigned_fit(cycles, kCountMask, "ICU NOP cycle count does not fit encoded command");
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(IcuCommandOpcode::Nop)
        | (static_cast<std::uint64_t>(cycles) << 2));
}

inline EncodedIcuCommand encode_icu_repeat(const InstructionControlUnit::Repeat& repeat)
{
    constexpr std::uint64_t kCountMask = 0x3ffull;
    constexpr std::uint64_t kIntervalMask = 0xffull;
    constexpr auto kStrideMin = static_cast<std::int64_t>(-2048);
    constexpr auto kStrideMax = static_cast<std::int64_t>(2047);
    detail::require_unsigned_fit(repeat.count, kCountMask, "ICU Repeat count does not fit encoded command");
    detail::require_unsigned_fit(repeat.interval, kIntervalMask, "ICU Repeat interval does not fit encoded command");
    detail::require_signed_fit(
        repeat.address_stride,
        kStrideMin,
        kStrideMax,
        "ICU Repeat address stride does not fit encoded command");

    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(IcuCommandOpcode::Repeat)
        | (static_cast<std::uint64_t>(repeat.count) << 2)
        | (static_cast<std::uint64_t>(repeat.interval) << 12)
        | (static_cast<std::uint64_t>(static_cast<std::uint16_t>(repeat.address_stride) & 0x0fffu) << 20));
}

inline IcuCommandOpcode decode_icu_command_opcode(EncodedIcuCommand command)
{
    return static_cast<IcuCommandOpcode>(command & 0x3u);
}

inline std::size_t decode_icu_nop_cycles(EncodedIcuCommand command)
{
    if (decode_icu_command_opcode(command) != IcuCommandOpcode::Nop) {
        throw std::logic_error("encoded ICU command is not NOP");
    }
    return static_cast<std::size_t>((command >> 2) & 0x3fffffffull);
}

inline InstructionControlUnit::Repeat decode_icu_repeat(EncodedIcuCommand command)
{
    if (decode_icu_command_opcode(command) != IcuCommandOpcode::Repeat) {
        throw std::logic_error("encoded ICU command is not Repeat");
    }
    auto stride = static_cast<std::int32_t>((command >> 20) & 0x0fffu);
    if ((stride & 0x800) != 0) {
        stride |= ~0x0fff;
    }
    return InstructionControlUnit::Repeat {
        static_cast<std::size_t>((command >> 2) & 0x3ffull),
        static_cast<std::size_t>((command >> 12) & 0xffull),
        stride,
    };
}

} // namespace isa

} // namespace ftlpu
