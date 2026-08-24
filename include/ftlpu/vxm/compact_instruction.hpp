#pragma once

#include "ftlpu/vxm/lane.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <variant>

namespace ftlpu {

// The Slice transports this 96-bit packet. The physical instruction channel
// identifies the ALU stage, so no stage or Stream Register index is encoded.
// Wide mux controls are reconstructed by the local Superlane decoder.
struct VxmCompactInstruction {
    std::uint64_t control{0};
    std::uint32_t immediate_bits{0};

    friend bool operator==(
        const VxmCompactInstruction&, const VxmCompactInstruction&) = default;
};

struct VxmDecodedInstruction {
    VxmChainDepth chain_depth{VxmChainDepth::Eight};
    VxmLaneAluInstruction instruction{};
};

class VxmCompactInstructionCodec {
public:
    static constexpr std::size_t kEncodedBits = 96;
    static constexpr std::uint64_t kOpcodeMask = 0xfull;
    static constexpr std::uint64_t kOperandMask = 0x7ull;
    static constexpr std::uint64_t kOutputTypeMask = 0x3ull;
    static constexpr std::uint64_t kDepthMask = 0x3ull;
    static constexpr std::uint64_t kRepeatMask = 0xffffffffull;
    static constexpr std::uint64_t kStreamGroupMask = 0x1full;

    static VxmCompactInstruction encode(
        std::size_t stage, VxmChainDepth depth,
        const VxmLaneAluInstruction& instruction)
    {
        if (instruction.repeat_count == 0
            || instruction.repeat_count
                > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "VXM compact repeat_count is outside 1..2^32-1");
        }
        validate_compact_operand(instruction.lhs);
        validate_compact_operand(instruction.rhs);
        validate_immediate_pair(instruction.lhs, instruction.rhs);
        validate_scales(instruction);

        auto validator = VxmLane{};
        validator.validate_broadcast_instruction(depth, stage, instruction);

        auto control = std::uint64_t{0};
        insert(control, encode_opcode(instruction.operation),
               kOpcodeShift, kOpcodeMask);
        insert(control, encode_operand(instruction.lhs.kind),
               kLhsShift, kOperandMask);
        insert(control, encode_operand(instruction.rhs.kind),
               kRhsShift, kOperandMask);
        insert(control,
               instruction.precision == VxmAluPrecision::Float32 ? 1 : 0,
               kPrecisionShift, 0x1);
        insert(control, encode_output_type(instruction.output_type),
               kOutputTypeShift, kOutputTypeMask);
        insert(control, instruction.output_stream.has_value() ? 1 : 0,
               kOutputEnableShift, 0x1);
        insert(control, instruction.accumulator_reset ? 1 : 0,
               kAccumulatorResetShift, 0x1);
        insert(control, instruction.accumulator_write ? 1 : 0,
               kAccumulatorWriteShift, 0x1);
        insert(control, instruction.accumulator_emit ? 1 : 0,
               kAccumulatorEmitShift, 0x1);
        insert(control, encode_depth(depth), kDepthShift, kDepthMask);
        insert(control, instruction.repeat_count,
               kRepeatShift, kRepeatMask);
        insert(control, instruction.local_scalar_write ? 1 : 0,
               kLocalScalarWriteShift, 0x1);
        insert(control, encode_stream_group(instruction.lhs),
               kLhsStreamGroupShift, kStreamGroupMask);
        insert(control, encode_stream_group(instruction.rhs),
               kRhsStreamGroupShift, kStreamGroupMask);

        auto immediate = 0.0f;
        if (instruction.lhs.kind == VxmLaneOperandKind::Immediate) {
            immediate = instruction.lhs.immediate;
        } else if (instruction.rhs.kind == VxmLaneOperandKind::Immediate) {
            immediate = instruction.rhs.immediate;
        }
        auto immediate_bits = std::uint32_t{0};
        std::memcpy(&immediate_bits, &immediate, sizeof(immediate_bits));
        return {control, immediate_bits};
    }

    static VxmDecodedInstruction decode(
        std::size_t stage, const VxmCompactInstruction& packet)
    {
        auto immediate = 0.0f;
        std::memcpy(&immediate, &packet.immediate_bits, sizeof(immediate));

        auto decoded = VxmDecodedInstruction{};
        decoded.chain_depth = decode_depth(extract(
            packet.control, kDepthShift, kDepthMask));
        auto& instruction = decoded.instruction;
        instruction.operation = decode_opcode(extract(
            packet.control, kOpcodeShift, kOpcodeMask));
        instruction.lhs = decode_operand(extract(
            packet.control, kLhsShift, kOperandMask), immediate);
        instruction.rhs = decode_operand(extract(
            packet.control, kRhsShift, kOperandMask), immediate);
        decode_stream_group(instruction.lhs, extract(packet.control,
            kLhsStreamGroupShift, kStreamGroupMask));
        decode_stream_group(instruction.rhs, extract(packet.control,
            kRhsStreamGroupShift, kStreamGroupMask));
        instruction.precision = extract(
            packet.control, kPrecisionShift, 0x1)
            ? VxmAluPrecision::Float32 : VxmAluPrecision::Float16;
        instruction.output_type = decode_output_type(extract(
            packet.control, kOutputTypeShift, kOutputTypeMask));
        instruction.accumulator_reset = extract(
            packet.control, kAccumulatorResetShift, 0x1) != 0;
        instruction.accumulator_write = extract(
            packet.control, kAccumulatorWriteShift, 0x1) != 0;
        instruction.accumulator_emit = extract(
            packet.control, kAccumulatorEmitShift, 0x1) != 0;
        instruction.repeat_count = static_cast<std::size_t>(extract(
            packet.control, kRepeatShift, kRepeatMask));
        instruction.local_scalar_write = extract(
            packet.control, kLocalScalarWriteShift, 0x1) != 0;
        if (instruction.repeat_count == 0) {
            throw std::invalid_argument(
                "VXM compact instruction decoded a zero repeat_count");
        }
        if (extract(packet.control, kOutputEnableShift, 0x1)) {
            instruction.output_stream =
                VxmLane::fixed_output_stream_for_block(
                    VxmLane::block_for_stage(stage));
        }

        auto validator = VxmLane{};
        validator.validate_broadcast_instruction(
            decoded.chain_depth, stage, instruction);
        return decoded;
    }

private:
    static constexpr unsigned kOpcodeShift = 0;
    static constexpr unsigned kLhsShift = 4;
    static constexpr unsigned kRhsShift = 7;
    static constexpr unsigned kPrecisionShift = 10;
    static constexpr unsigned kOutputTypeShift = 11;
    static constexpr unsigned kOutputEnableShift = 13;
    static constexpr unsigned kAccumulatorResetShift = 14;
    static constexpr unsigned kAccumulatorWriteShift = 15;
    static constexpr unsigned kAccumulatorEmitShift = 16;
    static constexpr unsigned kDepthShift = 17;
    static constexpr unsigned kRepeatShift = 19;
    static constexpr unsigned kLocalScalarWriteShift = 51;
    static constexpr unsigned kLhsStreamGroupShift = 52;
    static constexpr unsigned kRhsStreamGroupShift = 57;

    static void insert(std::uint64_t& word, std::uint64_t value,
                       unsigned shift, std::uint64_t mask)
    {
        if ((value & ~mask) != 0) {
            throw std::invalid_argument(
                "VXM compact instruction field does not fit");
        }
        word |= (value & mask) << shift;
    }

    static std::uint64_t extract(
        std::uint64_t word, unsigned shift, std::uint64_t mask)
    {
        return (word >> shift) & mask;
    }

    static std::uint64_t encode_opcode(const VxmLaneOperation& operation)
    {
        if (const auto* basic = std::get_if<VxmAluOpcode>(&operation)) {
            switch (*basic) {
            case VxmAluOpcode::Bypass: return 0;
            case VxmAluOpcode::Add: return 1;
            case VxmAluOpcode::Subtract: return 2;
            case VxmAluOpcode::Multiply: return 3;
            case VxmAluOpcode::Negate: return 4;
            case VxmAluOpcode::Max: return 5;
            }
        }
        switch (std::get<VxmSpecialAluOpcode>(operation)) {
        case VxmSpecialAluOpcode::Exp: return 6;
        case VxmSpecialAluOpcode::Reciprocal: return 7;
        case VxmSpecialAluOpcode::Rsqrt: return 8;
        }
        throw std::invalid_argument("unsupported VXM compact opcode");
    }

    static VxmLaneOperation decode_opcode(std::uint64_t opcode)
    {
        switch (opcode) {
        case 0: return VxmAluOpcode::Bypass;
        case 1: return VxmAluOpcode::Add;
        case 2: return VxmAluOpcode::Subtract;
        case 3: return VxmAluOpcode::Multiply;
        case 4: return VxmAluOpcode::Negate;
        case 5: return VxmAluOpcode::Max;
        case 6: return VxmSpecialAluOpcode::Exp;
        case 7: return VxmSpecialAluOpcode::Reciprocal;
        case 8: return VxmSpecialAluOpcode::Rsqrt;
        default:
            throw std::invalid_argument(
                "VXM compact instruction has an invalid opcode");
        }
    }

    static std::uint64_t encode_operand(VxmLaneOperandKind kind)
    {
        return static_cast<std::uint64_t>(kind);
    }

    static std::uint64_t encode_stream_group(
        const VxmLaneOperand& operand)
    {
        if (operand.stream_group < 0) return 0;
        if (operand.kind != VxmLaneOperandKind::StreamFloat16
            && operand.kind != VxmLaneOperandKind::StreamBFloat16)
            throw std::invalid_argument(
                "only VXM stream operands may select a stream group");
        if (operand.stream_group >= 8)
            throw std::invalid_argument(
                "VXM stream-group selector is outside 0..7");
        return static_cast<std::uint64_t>(operand.stream_group + 1);
    }

    static void decode_stream_group(
        VxmLaneOperand& operand, std::uint64_t encoded)
    {
        if (encoded == 0) return;
        if (encoded > 8
            || (operand.kind != VxmLaneOperandKind::StreamFloat16
                && operand.kind != VxmLaneOperandKind::StreamBFloat16))
            throw std::invalid_argument(
                "VXM compact instruction has an invalid stream-group selector");
        operand.stream_group = static_cast<std::int32_t>(encoded - 1);
    }

    static VxmLaneOperand decode_operand(
        std::uint64_t encoded, float immediate)
    {
        if (encoded > static_cast<std::uint64_t>(
                          VxmLaneOperandKind::Feedback)) {
            throw std::invalid_argument(
                "VXM compact instruction has an invalid operand selector");
        }
        const auto kind = static_cast<VxmLaneOperandKind>(encoded);
        switch (kind) {
        case VxmLaneOperandKind::Previous: return VxmLaneOperand::Previous();
        case VxmLaneOperandKind::Original: return VxmLaneOperand::Original();
        case VxmLaneOperandKind::Auxiliary: return VxmLaneOperand::Aux();
        case VxmLaneOperandKind::Accumulator: return VxmLaneOperand::Acc();
        case VxmLaneOperandKind::StreamFloat16:
            return VxmLaneOperand::StreamFloat16();
        case VxmLaneOperandKind::StreamBFloat16:
            return VxmLaneOperand::StreamBFloat16();
        case VxmLaneOperandKind::Immediate:
            return VxmLaneOperand::Imm(immediate);
        case VxmLaneOperandKind::Feedback:
            return VxmLaneOperand::Feedback();
        }
        throw std::invalid_argument("invalid VXM compact operand");
    }

    static std::uint64_t encode_output_type(VxmCastTarget type)
    {
        return static_cast<std::uint64_t>(type);
    }

    static VxmCastTarget decode_output_type(std::uint64_t encoded)
    {
        if (encoded > static_cast<std::uint64_t>(VxmCastTarget::BFloat16)) {
            throw std::invalid_argument(
                "VXM compact instruction has an invalid output type");
        }
        return static_cast<VxmCastTarget>(encoded);
    }

    static std::uint64_t encode_depth(VxmChainDepth depth)
    {
        switch (depth) {
        case VxmChainDepth::Two: return 0;
        case VxmChainDepth::Four: return 1;
        case VxmChainDepth::Eight: return 2;
        }
        throw std::invalid_argument("unsupported VXM compact chain depth");
    }

    static VxmChainDepth decode_depth(std::uint64_t encoded)
    {
        switch (encoded) {
        case 0: return VxmChainDepth::Two;
        case 1: return VxmChainDepth::Four;
        case 2: return VxmChainDepth::Eight;
        default:
            throw std::invalid_argument(
                "VXM compact instruction has an invalid chain depth");
        }
    }

    static void validate_compact_operand(const VxmLaneOperand& operand)
    {
        if ((operand.kind == VxmLaneOperandKind::StreamFloat16
             || operand.kind == VxmLaneOperandKind::StreamBFloat16)
            && (operand.scale != 1.0f || operand.zero_point != 0)) {
            throw std::invalid_argument(
                "VXM compact packet requires separately configured "
                "non-default input scaling");
        }
    }

    static void validate_immediate_pair(
        const VxmLaneOperand& lhs, const VxmLaneOperand& rhs)
    {
        if (lhs.kind == VxmLaneOperandKind::Immediate
            && rhs.kind == VxmLaneOperandKind::Immediate
            && lhs.immediate != rhs.immediate) {
            throw std::invalid_argument(
                "VXM compact packet carries only one immediate value");
        }
    }

    static void validate_scales(const VxmLaneAluInstruction& instruction)
    {
        if (instruction.output_scale != 1.0f
            || instruction.output_zero_point != 0) {
            throw std::invalid_argument(
                "VXM compact packet requires separately configured "
                "non-default output quantization");
        }
    }
};

} // namespace ftlpu
