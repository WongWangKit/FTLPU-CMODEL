#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mxm/control_slice.hpp"

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
//   [1:0] opcode, [7:2] stream, [13:8] map stream,
//   [31:14] slice-local SRAM word address. GroqLike uses the low 13 bits
//           exactly as before; narrower-vector configurations may use the
//           formerly reserved high bits.
// MEM extended 3x32b (ReadWrite only):
//   word0 [2:0] opcode, [8:3] primary stream, [14:9] secondary stream,
//   word1 primary SRAM word address;
//   word2 secondary SRAM word address.
// MXM control 32b:
//   IW      [1:0] opcode, [2] weight buffer, [4:3] weight load mode.
//   Compute [1:0] opcode, [2] weight buffer, [8:3] reserved/fixed input,
//           [14:9] output stream base, [17:15] accumulator mode,
//           [18] MXM pair mode, [19] explicit K-block start,
//           [25:20] old-partial stream base.
// ICU queue command 32b:
//   NOP    [2:0] opcode, [18:3] 16-bit cycle count.
//   Repeat [2:0] opcode, [11:3] count, [19:12] interval,
//          [31:20] signed MEM address stride.
using EncodedMemInstruction = std::uint32_t;
struct EncodedExtendedMemInstruction {
    std::array<std::uint32_t, 3> words{};
};
using EncodedMxmInstruction = std::uint32_t;
using EncodedIcuCommand = std::uint32_t;

enum class IcuCommandOpcode : std::uint8_t {
    Reserved = 0,
    Nop = 1,
    Repeat = 2,
    Sync = 3,
    Notify = 4,
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

} // namespace detail

inline EncodedMemInstruction encode_mem_instruction(const MemInstruction& instruction)
{
    constexpr std::uint64_t kOpcodeMask = 0x3;
    constexpr std::uint64_t kStreamMask = 0x3f;
    constexpr std::uint64_t kAddressMask = MemLocalWordAddress13::kLimit - 1;
    static_assert(
        MemLocalWordAddress13::kBits <= 18,
        "configured SRAM depth does not fit the 18-bit MEM instruction address field");

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
        static_cast<std::uint64_t>(instruction.address.encoded()),
        kAddressMask,
        "MEM local word address does not fit encoded instruction");

    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(instruction.opcode)
        | (static_cast<std::uint64_t>(instruction.stream) << 2)
        | (static_cast<std::uint64_t>(instruction.map_stream) << 8)
        | (static_cast<std::uint64_t>(instruction.address.encoded()) << 14));
}

inline MemInstruction decode_mem_instruction(EncodedMemInstruction word)
{
    constexpr auto kAddressMask = MemLocalWordAddress13::kLimit - 1;
    constexpr std::uint64_t kUsedMask =
        0x3fffull | (static_cast<std::uint64_t>(kAddressMask) << 14);
    detail::require_reserved_zero(word, kUsedMask, "encoded MEM instruction has non-zero reserved bits");
    const auto opcode = static_cast<MemOpcode>(detail::low_bits(word, 0, 0x3));
    const auto stream = static_cast<std::size_t>(detail::low_bits(word, 2, 0x3f));
    const auto map_stream = static_cast<std::size_t>(detail::low_bits(word, 8, 0x3f));
    const auto address = static_cast<std::size_t>(
        detail::low_bits(word, 14, kAddressMask));

    switch (opcode) {
    case MemOpcode::Read:
        return MemInstruction::Read(address, stream);
    case MemOpcode::Write:
        return MemInstruction::Write(address, stream);
    case MemOpcode::Gather:
        return MemInstruction::Gather(stream, map_stream);
    case MemOpcode::Scatter:
        return MemInstruction::Scatter(stream, map_stream);
    case MemOpcode::ReadWrite:
        break;
    }
    throw std::logic_error("unknown encoded MEM opcode");
}

inline EncodedExtendedMemInstruction encode_extended_mem_instruction(
    const MemInstruction& instruction)
{
    constexpr std::uint32_t kStreamMask = 0x3f;
    constexpr std::uint32_t kAddressMask =
        static_cast<std::uint32_t>(MemLocalWordAddress13::kLimit - 1);
    static_assert(
        MemLocalWordAddress13::kBits <= 18,
        "configured SRAM depth does not fit the extended MEM address word");

    if (instruction.opcode != MemOpcode::ReadWrite) {
        throw std::logic_error(
            "extended MEM encoding is reserved for ReadWrite");
    }
    detail::require_unsigned_fit(
        instruction.stream, kStreamMask,
        "extended MEM primary stream does not fit");
    detail::require_unsigned_fit(
        instruction.address.encoded(), kAddressMask,
        "extended MEM primary address does not fit");

    EncodedExtendedMemInstruction encoded{};
    encoded.words[0] =
        static_cast<std::uint32_t>(instruction.opcode)
        | (static_cast<std::uint32_t>(instruction.stream) << 3);
    encoded.words[1] =
        static_cast<std::uint32_t>(instruction.address.encoded());

    detail::require_unsigned_fit(
        instruction.write_stream, kStreamMask,
        "MEM ReadWrite destination stream does not fit");
    detail::require_unsigned_fit(
        instruction.write_address.encoded(), kAddressMask,
        "MEM ReadWrite destination address does not fit");
    if (instruction.address == instruction.write_address) {
        throw std::logic_error(
            "MEM ReadWrite encodes identical read and write addresses");
    }
    encoded.words[0] |=
        static_cast<std::uint32_t>(instruction.write_stream) << 9;
    encoded.words[2] =
        static_cast<std::uint32_t>(instruction.write_address.encoded());
    return encoded;
}

inline MemInstruction decode_extended_mem_instruction(
    const EncodedExtendedMemInstruction& encoded)
{
    constexpr std::uint32_t kControlMask = 0x7fffu;
    constexpr std::uint32_t kAddressMask =
        static_cast<std::uint32_t>(MemLocalWordAddress13::kLimit - 1);
    detail::require_reserved_zero(
        encoded.words[0], kControlMask,
        "extended MEM control word has non-zero reserved bits");
    detail::require_reserved_zero(
        encoded.words[1], kAddressMask,
        "extended MEM primary address word has non-zero reserved bits");

    const auto opcode =
        static_cast<MemOpcode>(encoded.words[0] & 0x7u);
    const auto stream =
        static_cast<std::size_t>((encoded.words[0] >> 3) & 0x3fu);
    const auto secondary_stream =
        static_cast<std::size_t>((encoded.words[0] >> 9) & 0x3fu);
    const auto address =
        static_cast<std::size_t>(encoded.words[1] & kAddressMask);
    switch (opcode) {
    case MemOpcode::ReadWrite: {
        detail::require_reserved_zero(
            encoded.words[2], kAddressMask,
            "MEM ReadWrite destination address has non-zero reserved bits");
        return MemInstruction::ReadWrite(
            address,
            stream,
            static_cast<std::size_t>(encoded.words[2] & kAddressMask),
            secondary_stream);
    }
    case MemOpcode::Read:
    case MemOpcode::Write:
    case MemOpcode::Gather:
    case MemOpcode::Scatter:
        break;
    }
    throw std::logic_error("unknown extended MEM opcode");
}

inline EncodedMxmInstruction encode_mxm_instruction(const MxmControlInstruction& instruction)
{
    constexpr std::uint32_t kOpcodeMask = 0x3;
    constexpr std::uint32_t kWeightBufferMask = 0x1;
    constexpr std::uint32_t kStreamBaseMask = 0x3f;
    constexpr std::uint32_t kWeightLoadModeMask = 0x3;
    constexpr std::uint32_t kAccumulatorModeMask = 0x7;
    constexpr std::uint32_t kPartialStreamMask = 0x3f;

    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.opcode),
        kOpcodeMask,
        "MXM opcode does not fit encoded instruction");
    const auto opcode = static_cast<std::uint32_t>(instruction.opcode);
    switch (instruction.opcode) {
    case MxmControlOpcode::IW:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_load_mode),
            kWeightLoadModeMask,
            "MXM weight load mode does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint32_t>(instruction.weight_buffer) << 2)
            | (static_cast<std::uint32_t>(instruction.weight_load_mode) << 3);
    case MxmControlOpcode::LoadScales:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        return opcode | (static_cast<std::uint32_t>(instruction.weight_buffer) << 2);
    case MxmControlOpcode::ActivationDequantize:
        return opcode;
    case MxmControlOpcode::Compute:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.stream_base),
            kStreamBaseMask,
            "MXM output stream base does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.accumulator_mode),
            kAccumulatorModeMask,
            "MXM accumulator mode does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.partial_stream_base),
            kPartialStreamMask,
            "MXM partial stream base does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint32_t>(instruction.weight_buffer) << 2)
            | (static_cast<std::uint32_t>(instruction.stream_base) << 9)
            | (static_cast<std::uint32_t>(instruction.accumulator_mode) << 15)
            | (static_cast<std::uint32_t>(instruction.pair_mode) << 18)
            | (static_cast<std::uint32_t>(instruction.start_of_k_block) << 19)
            | (static_cast<std::uint32_t>(instruction.partial_stream_base) << 20);
    }
    throw std::logic_error("unknown MXM opcode");
}

inline MxmControlInstruction decode_mxm_instruction(EncodedMxmInstruction word)
{
    const auto opcode = static_cast<MxmControlOpcode>(word & 0x3u);
    const auto iw_weight_buffer = static_cast<std::size_t>((word >> 2) & 0x1u);
    const auto iw_load_mode = static_cast<MxmWeightLoadMode>(
        (word >> 3) & 0x3u);
    const auto compute_weight_buffer = static_cast<std::size_t>((word >> 2) & 0x1u);
    const auto stream_base = static_cast<std::size_t>((word >> 9) & 0x3fu);
    const auto accumulator_mode =
        static_cast<MxmAccumulatorMode>((word >> 15) & 0x7u);
    const auto pair_mode =
        static_cast<MxmPairMode>((word >> 18) & 0x1u);
    const auto start_of_k_block = ((word >> 19) & 0x1u) != 0;
    const auto partial_stream_base =
        static_cast<std::size_t>((word >> 20) & 0x3fu);

    switch (opcode) {
    case MxmControlOpcode::IW:
        detail::require_reserved_zero(word, 0x0000001fu, "encoded MXM IW instruction has non-zero reserved bits");
        MxmControlInstruction::check_weight_load_mode(iw_load_mode);
        return MxmControlInstruction::IW(
            iw_weight_buffer, iw_load_mode);
    case MxmControlOpcode::LoadScales:
        detail::require_reserved_zero(
            word,
            0x00000007u,
            "encoded MXM LoadScales instruction has non-zero reserved bits");
        return MxmControlInstruction::LoadScales(
            iw_weight_buffer);
    case MxmControlOpcode::Compute:
        detail::require_reserved_zero(word, 0x03fffe07u, "encoded MXM Compute instruction has non-zero reserved bits");
        if (static_cast<std::uint8_t>(accumulator_mode)
            > static_cast<std::uint8_t>(MxmAccumulatorMode::MemoryFinalize)) {
            throw std::logic_error("encoded MXM instruction has an invalid accumulator mode");
        }
        return MxmControlInstruction::ComputeAccumulating(
            compute_weight_buffer,
            stream_base,
            accumulator_mode,
            partial_stream_base,
            pair_mode,
            start_of_k_block);
    case MxmControlOpcode::ActivationDequantize:
        detail::require_reserved_zero(
            word,
            0x00000003u,
            "encoded MXM ActivationDequantize has non-zero reserved bits");
        return MxmControlInstruction::ActivationDequantize();
    }
    throw std::logic_error("unknown encoded MXM opcode");
}

inline EncodedIcuCommand encode_icu_control_instruction(
    const IcuControlInstruction& instruction)
{
    constexpr std::uint64_t kNopCountMask = 0xffffull;
    constexpr std::uint64_t kRepeatCountMask = 0x1ffull;
    constexpr std::uint64_t kIntervalMask = 0xffull;
    constexpr auto kStrideMin = static_cast<std::int64_t>(-2048);
    constexpr auto kStrideMax = static_cast<std::int64_t>(2047);
    const auto opcode = static_cast<std::uint32_t>(instruction.opcode);

    switch (instruction.opcode) {
    case IcuControlOpcode::Nop:
        detail::require_unsigned_fit(
            instruction.count, kNopCountMask,
            "ICU NOP cycle count does not fit 16-bit encoded command");
        return opcode | (static_cast<std::uint32_t>(instruction.count) << 3);
    case IcuControlOpcode::Repeat:
        detail::require_unsigned_fit(
            instruction.count, kRepeatCountMask,
            "ICU Repeat count does not fit encoded command");
        detail::require_unsigned_fit(
            instruction.interval, kIntervalMask,
            "ICU Repeat interval does not fit encoded command");
        detail::require_signed_fit(
            instruction.address_stride, kStrideMin, kStrideMax,
            "ICU Repeat address stride does not fit encoded command");
        return opcode
            | (static_cast<std::uint32_t>(instruction.count) << 3)
            | (static_cast<std::uint32_t>(instruction.interval) << 12)
            | ((static_cast<std::uint32_t>(instruction.address_stride) & 0x0fffu) << 20);
    case IcuControlOpcode::Sync:
    case IcuControlOpcode::Notify:
        return opcode;
    }
    throw std::logic_error("unknown ICU control opcode");
}

inline IcuControlInstruction decode_icu_control_instruction(EncodedIcuCommand command)
{
    const auto opcode = static_cast<IcuControlOpcode>(command & 0x7u);
    switch (opcode) {
    case IcuControlOpcode::Nop:
        detail::require_reserved_zero(command, 0x0007ffffu, "encoded ICU NOP has non-zero reserved bits");
        return IcuControlInstruction::Nop(
            static_cast<std::size_t>((command >> 3) & 0xffffu));
    case IcuControlOpcode::Repeat: {
        auto stride = static_cast<std::int32_t>((command >> 20) & 0x0fffu);
        if ((stride & 0x800) != 0) {
            stride |= ~0x0fff;
        }
        return IcuControlInstruction::Repeat(
            static_cast<std::size_t>((command >> 3) & 0x1ffu),
            static_cast<std::size_t>((command >> 12) & 0xffu),
            stride);
    }
    case IcuControlOpcode::Sync:
        detail::require_reserved_zero(command, 0x7u, "encoded ICU Sync has non-zero reserved bits");
        return IcuControlInstruction::Sync();
    case IcuControlOpcode::Notify:
        detail::require_reserved_zero(command, 0x7u, "encoded ICU Notify has non-zero reserved bits");
        return IcuControlInstruction::Notify();
    }
    throw std::logic_error("unknown encoded ICU control opcode");
}

inline EncodedIcuCommand encode_icu_nop(std::size_t cycles)
{
    return encode_icu_control_instruction(IcuControlInstruction::Nop(cycles));
}

inline EncodedIcuCommand encode_icu_repeat(const IcuRepeat& repeat)
{
    return encode_icu_control_instruction(
        IcuControlInstruction::Repeat(repeat.count, repeat.interval, repeat.address_stride));
}

inline IcuCommandOpcode decode_icu_command_opcode(EncodedIcuCommand command)
{
    return static_cast<IcuCommandOpcode>(command & 0x7u);
}

inline std::size_t decode_icu_nop_cycles(EncodedIcuCommand command)
{
    if (decode_icu_command_opcode(command) != IcuCommandOpcode::Nop) {
        throw std::logic_error("encoded ICU command is not NOP");
    }
    return decode_icu_control_instruction(command).count;
}

inline IcuRepeat decode_icu_repeat(EncodedIcuCommand command)
{
    if (decode_icu_command_opcode(command) != IcuCommandOpcode::Repeat) {
        throw std::logic_error("encoded ICU command is not Repeat");
    }
    const auto decoded = decode_icu_control_instruction(command);
    return IcuRepeat {decoded.count, decoded.interval, decoded.address_stride};
}

} // namespace isa

} // namespace ftlpu
