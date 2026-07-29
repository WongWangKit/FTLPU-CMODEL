#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/mem/slice.hpp"
#include "ftlpu/mxm/control_slice.hpp"
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
//   [1:0] opcode, [7:2] stream, [13:8] map stream,
//   [31:14] slice-local SRAM word address. GroqLike uses the low 13 bits
//           exactly as before; narrower-vector configurations may use the
//           formerly reserved high bits.
// MEM extended 3x32b (ReadWrite/Accumulate only):
//   word0 [2:0] opcode, [8:3] primary stream, [14:9] secondary stream,
//         [15] accumulator destination;
//   word1 primary SRAM word address;
//   word2 secondary SRAM word address (ReadWrite only).
// MXM control 32b:
//   IW      [1:0] opcode, [2] weight buffer.
//   Compute [1:0] opcode, [2] weight buffer, [8:3] activation stream base,
//           [14:9] output stream base, [15] accumulator bank,
//           [16] accumulate existing, [17] reduce/output,
//           [18] explicit K-block start, [19] accumulator-control present.
// VXM ALU 3x32b:
//   word0 [2:0] opcode, [3] special-ALU selector,
//         [6:4] lhs kind, [9:7] rhs kind, [10] precision,
//         [12:11] output type, [13] output valid,
//         [15:14] fixed output block, [18:16] accumulator control,
//         [19] input hemisphere, [20] output hemisphere,
//         [31:21] repeat-count minus one.
//   word1 [31:16] lhs FP16 literal/scale.
//   word2 [15:0] rhs FP16 literal/scale,
//         [31:16] output FP16 scale.
// Stream selectors are intentionally absent: each chain head and tail has a
// fixed physical stream-group binding in the 8-ALU VXM.
// ICU queue command 32b:
//   Fetch  [2:0] opcode, [8:3] packed source StreamId.
//   NOP    [2:0] opcode, [18:3] 16-bit cycle count.
//   Repeat [2:0] opcode, [11:3] count, [19:12] interval,
//          [31:20] signed MEM address stride.
using EncodedMemInstruction = std::uint32_t;
struct EncodedExtendedMemInstruction {
    std::array<std::uint32_t, 3> words{};
};
using EncodedMxmInstruction = std::uint32_t;
using EncodedIcuCommand = std::uint32_t;

struct EncodedVxmInstruction {
    std::array<std::uint32_t, 3> words{};
};

enum class IcuCommandOpcode : std::uint8_t {
    Fetch = 0,
    Instruction = Fetch,
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

inline void require_operand_hardware_encodable(const VxmLaneOperand& operand, const char* field)
{
    switch (operand.kind) {
    case VxmLaneOperandKind::PreviousValue:
    case VxmLaneOperandKind::OriginalValue:
    case VxmLaneOperandKind::AuxiliaryValue:
    case VxmLaneOperandKind::AccumulatorValue:
        require_zero_float(
            operand.immediate,
            "VXM local operand carries an immediate");
        require_default_float(
            operand.scale,
            "VXM local operand carries a scale");
        break;
    case VxmLaneOperandKind::StreamInt32:
    case VxmLaneOperandKind::StreamFloat32:
    case VxmLaneOperandKind::StreamFloat16:
        require_zero_float(
            operand.immediate,
            "VXM stream operand carries an immediate");
        break;
    case VxmLaneOperandKind::ImmediateValue:
        require_default_float(
            operand.scale,
            "VXM immediate operand carries a scale");
        break;
    default:
        throw std::logic_error(field);
    }
    if (operand.zero_point != 0) {
        throw std::logic_error(
            "VXM packet supports symmetric stream conversion only");
    }
}

inline std::uint16_t encode_vxm_operand_scalar(
    const VxmLaneOperand& operand)
{
    const auto value =
        operand.kind == VxmLaneOperandKind::ImmediateValue
        ? operand.immediate
        : operand.scale;
    return VxmDataFormat::float_to_fp16_bits(value);
}

inline void decode_vxm_operand_scalar(
    VxmLaneOperand& operand,
    std::uint16_t encoded)
{
    const auto value =
        VxmDataFormat::fp16_bits_to_float(encoded);
    if (operand.kind == VxmLaneOperandKind::ImmediateValue) {
        operand.immediate = value;
        operand.scale = 1.0f;
    } else if (operand.kind == VxmLaneOperandKind::StreamInt32
               || operand.kind
                    == VxmLaneOperandKind::StreamFloat32
               || operand.kind
                    == VxmLaneOperandKind::StreamFloat16) {
        operand.immediate = 0.0f;
        operand.scale = value;
    } else {
        operand.immediate = 0.0f;
        operand.scale = 1.0f;
    }
    operand.zero_point = 0;
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

    if (instruction.opcode != MemOpcode::ReadWrite
        && instruction.opcode != MemOpcode::Accumulate) {
        throw std::logic_error(
            "extended MEM encoding is reserved for ReadWrite/Accumulate");
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

    if (instruction.opcode == MemOpcode::ReadWrite) {
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

    if (instruction.accumulator_destination
        == MemAccumulatorDestination::Stream) {
        encoded.words[0] |= 1u << 15;
    }
    return encoded;
}

inline MemInstruction decode_extended_mem_instruction(
    const EncodedExtendedMemInstruction& encoded)
{
    constexpr std::uint32_t kControlMask = 0xffffu;
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
        if ((encoded.words[0] & (1u << 15)) != 0) {
            throw std::logic_error(
                "MEM ReadWrite sets the accumulator destination bit");
        }
        detail::require_reserved_zero(
            encoded.words[2], kAddressMask,
            "MEM ReadWrite destination address has non-zero reserved bits");
        return MemInstruction::ReadWrite(
            address,
            stream,
            static_cast<std::size_t>(encoded.words[2] & kAddressMask),
            secondary_stream);
    }
    case MemOpcode::Accumulate:
        if (secondary_stream != 0 || encoded.words[2] != 0) {
            throw std::logic_error(
                "MEM Accumulate has non-zero secondary fields");
        }
        return MemInstruction::Accumulate(
            address,
            stream,
            (encoded.words[0] & (1u << 15)) != 0
                ? MemAccumulatorDestination::Stream
                : MemAccumulatorDestination::Sram);
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
    constexpr std::uint32_t kActivationStreamMask = 0x3f;
    constexpr std::uint32_t kAccumulatorBankMask = 0x1;
    constexpr std::uint32_t kAccumulatorControlPresent = 1u << 19;

    detail::require_unsigned_fit(
        static_cast<std::uint64_t>(instruction.opcode),
        kOpcodeMask,
        "MXM opcode does not fit encoded instruction");
    const auto opcode = static_cast<std::uint32_t>(instruction.opcode);
    switch (instruction.opcode) {
    case MxmControlOpcode::IW:
    case MxmControlOpcode::LoadScales:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        return opcode | (static_cast<std::uint32_t>(instruction.weight_buffer) << 2);
    case MxmControlOpcode::Compute:
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.weight_buffer),
            kWeightBufferMask,
            "MXM weight buffer does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.activation_stream_base),
            kActivationStreamMask,
            "MXM activation stream base does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.stream_base),
            kStreamBaseMask,
            "MXM output stream base does not fit encoded instruction");
        detail::require_unsigned_fit(
            static_cast<std::uint64_t>(instruction.accumulator_bank),
            kAccumulatorBankMask,
            "MXM accumulator bank does not fit encoded instruction");
        return opcode
            | (static_cast<std::uint32_t>(instruction.weight_buffer) << 2)
            | (static_cast<std::uint32_t>(instruction.activation_stream_base) << 3)
            | (static_cast<std::uint32_t>(instruction.stream_base) << 9)
            | (static_cast<std::uint32_t>(instruction.accumulator_bank) << 15)
            | (static_cast<std::uint32_t>(instruction.accumulate) << 16)
            | (static_cast<std::uint32_t>(instruction.reduce) << 17)
            | (static_cast<std::uint32_t>(instruction.start_of_k_block) << 18)
            | kAccumulatorControlPresent;
    }
    throw std::logic_error("unknown MXM opcode");
}

inline MxmControlInstruction decode_mxm_instruction(EncodedMxmInstruction word)
{
    const auto opcode = static_cast<MxmControlOpcode>(word & 0x3u);
    const auto iw_weight_buffer = static_cast<std::size_t>((word >> 2) & 0x1u);
    const auto compute_weight_buffer = static_cast<std::size_t>((word >> 2) & 0x1u);
    const auto compute_activation_stream_base = static_cast<std::size_t>((word >> 3) & 0x3fu);
    const auto stream_base = static_cast<std::size_t>((word >> 9) & 0x3fu);
    const auto accumulator_bank = static_cast<std::size_t>((word >> 15) & 0x1u);
    const auto accumulate = ((word >> 16) & 0x1u) != 0;
    const auto reduce = ((word >> 17) & 0x1u) != 0;
    const auto start_of_k_block = ((word >> 18) & 0x1u) != 0;
    const auto has_accumulator_control = ((word >> 19) & 0x1u) != 0;

    switch (opcode) {
    case MxmControlOpcode::IW:
        detail::require_reserved_zero(word, 0x00000007u, "encoded MXM IW instruction has non-zero reserved bits");
        return MxmControlInstruction::IW(iw_weight_buffer);
    case MxmControlOpcode::LoadScales:
        detail::require_reserved_zero(
            word,
            0x00000007u,
            "encoded MXM LoadScales instruction has non-zero reserved bits");
        return MxmControlInstruction::LoadScales(
            iw_weight_buffer);
    case MxmControlOpcode::Compute:
        detail::require_reserved_zero(word, 0x000fffffu, "encoded MXM Compute instruction has non-zero reserved bits");
        if (!has_accumulator_control) {
            return MxmControlInstruction::Compute(
                compute_weight_buffer,
                compute_activation_stream_base,
                stream_base);
        }
        return MxmControlInstruction::ComputeToAccumulator(
            compute_weight_buffer,
            accumulator_bank,
            compute_activation_stream_base,
            stream_base,
            accumulate,
            reduce,
            start_of_k_block);
    }
    throw std::logic_error("unknown encoded MXM opcode");
}

inline EncodedVxmInstruction encode_vxm_instruction(const VxmLaneAluInstruction& instruction)
{
    constexpr std::uint32_t kOpcodeMask = 0x7;
    constexpr std::uint32_t kOperandKindMask = 0x7;
    constexpr std::uint32_t kRepeatMask = 0x7ff;

    if (instruction.output_zero_point != 0) {
        throw std::logic_error(
            "VXM packet supports symmetric output conversion only");
    }
    if (instruction.repeat_count == 0
        || instruction.repeat_count > kRepeatMask + 1) {
        throw std::out_of_range(
            "VXM repeat count does not fit the 11-bit packet field");
    }
    detail::require_operand_hardware_encodable(
        instruction.lhs,
        "VXM lhs operand is not encodable");
    detail::require_operand_hardware_encodable(
        instruction.rhs,
        "VXM rhs operand is not encodable");

    const auto special =
        std::holds_alternative<VxmSpecialAluOpcode>(
            instruction.operation);
    const auto opcode = special
        ? static_cast<std::uint32_t>(
            std::get<VxmSpecialAluOpcode>(
                instruction.operation))
        : static_cast<std::uint32_t>(
            std::get<VxmAluOpcode>(instruction.operation));
    detail::require_unsigned_fit(
        opcode,
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

    std::uint32_t output_block = 0;
    if (instruction.output_stream.has_value()) {
        if (*instruction.output_stream
                % VxmLane::kStreamGroupBytes
                != 0
            || *instruction.output_stream
                >= VxmLane::kBlockCount
                    * VxmLane::kStreamGroupBytes) {
            throw std::logic_error(
                "VXM output is not one of the four fixed stream groups");
        }
        output_block = static_cast<std::uint32_t>(
            *instruction.output_stream
            / VxmLane::kStreamGroupBytes);
    }

    auto control = opcode
        | (static_cast<std::uint32_t>(special) << 3)
        | (static_cast<std::uint32_t>(
               instruction.lhs.kind)
            << 4)
        | (static_cast<std::uint32_t>(
               instruction.rhs.kind)
            << 7)
        | (static_cast<std::uint32_t>(
               instruction.precision)
            << 10)
        | (static_cast<std::uint32_t>(
               instruction.output_type)
            << 11)
        | (output_block << 14)
        | (static_cast<std::uint32_t>(
               instruction.accumulator_reset)
            << 16)
        | (static_cast<std::uint32_t>(
               instruction.accumulator_write)
            << 17)
        | (static_cast<std::uint32_t>(
               instruction.accumulator_emit)
            << 18)
        | (static_cast<std::uint32_t>(
               hemisphere_index(
                   instruction.input_hemisphere))
            << 19)
        | (static_cast<std::uint32_t>(
               hemisphere_index(
                   instruction.output_hemisphere))
            << 20)
        | (static_cast<std::uint32_t>(
               instruction.repeat_count - 1)
            << 21);
    if (instruction.output_stream.has_value()) {
        control |= 1u << 13;
    }

    return EncodedVxmInstruction {
        std::array<std::uint32_t, 3> {
            control,
            static_cast<std::uint32_t>(
                detail::encode_vxm_operand_scalar(
                    instruction.lhs))
                << 16,
            static_cast<std::uint32_t>(
                detail::encode_vxm_operand_scalar(
                    instruction.rhs))
                | (static_cast<std::uint32_t>(
                       VxmDataFormat::float_to_fp16_bits(
                           instruction.output_scale))
                    << 16),
        },
    };
}

inline VxmLaneAluInstruction decode_vxm_instruction(const EncodedVxmInstruction& encoded)
{
    const auto control = encoded.words[0];
    auto instruction = VxmLaneAluInstruction {};

    const auto opcode = control & 0x7u;
    if (((control >> 3) & 0x1u) != 0) {
        instruction.operation =
            static_cast<VxmSpecialAluOpcode>(opcode);
    } else {
        instruction.operation =
            static_cast<VxmAluOpcode>(opcode);
    }
    instruction.lhs.kind =
        static_cast<VxmLaneOperandKind>(
            (control >> 4) & 0x7u);
    instruction.rhs.kind =
        static_cast<VxmLaneOperandKind>(
            (control >> 7) & 0x7u);
    instruction.precision =
        static_cast<VxmAluPrecision>(
            (control >> 10) & 0x1u);
    instruction.output_type =
        static_cast<VxmCastTarget>(
            (control >> 11) & 0x3u);
    if (((control >> 13) & 0x1u) != 0) {
        instruction.output_stream =
            static_cast<std::size_t>(
                (control >> 14) & 0x3u)
            * VxmLane::kStreamGroupBytes;
    }
    instruction.accumulator_reset =
        ((control >> 16) & 0x1u) != 0;
    instruction.accumulator_write =
        ((control >> 17) & 0x1u) != 0;
    instruction.accumulator_emit =
        ((control >> 18) & 0x1u) != 0;
    instruction.input_hemisphere =
        static_cast<Hemisphere>(
            (control >> 19) & 0x1u);
    instruction.output_hemisphere =
        static_cast<Hemisphere>(
            (control >> 20) & 0x1u);
    instruction.repeat_count =
        static_cast<std::size_t>(
            (control >> 21) & 0x7ffu)
        + 1;

    detail::decode_vxm_operand_scalar(
        instruction.lhs,
        static_cast<std::uint16_t>(
            encoded.words[1] >> 16));
    detail::decode_vxm_operand_scalar(
        instruction.rhs,
        static_cast<std::uint16_t>(
            encoded.words[2]));
    instruction.output_scale =
        VxmDataFormat::fp16_bits_to_float(
            static_cast<std::uint16_t>(
                encoded.words[2] >> 16));
    instruction.output_zero_point = 0;

    detail::require_operand_hardware_encodable(
        instruction.lhs,
        "decoded VXM lhs operand is invalid");
    detail::require_operand_hardware_encodable(
        instruction.rhs,
        "decoded VXM rhs operand is invalid");
    return instruction;
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
    case IcuControlOpcode::Fetch:
        detail::require_unsigned_fit(
            instruction.source_stream.packed(), hw::kStreams - 1,
            "ICU Fetch stream does not fit encoded command");
        return opcode
            | (static_cast<std::uint32_t>(instruction.source_stream.packed()) << 3);
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
    case IcuControlOpcode::Fetch:
        detail::require_reserved_zero(command, 0x000001ffu, "encoded ICU Fetch has non-zero reserved bits");
        return IcuControlInstruction::Fetch(
            StreamId::from_packed(static_cast<std::size_t>((command >> 3) & 0x3fu)));
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
