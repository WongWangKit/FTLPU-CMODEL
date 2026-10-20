#pragma once

#include "ftlpu/core/stream.hpp"
#include "ftlpu/vxm/alu.hpp"
#include "ftlpu/vxm/lane.hpp"
#include "ftlpu/vxm/special_alu.hpp"
#include "ftlpu/vxm/static_quantizer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace ftlpu {

// Hardware-visible VXM operands. Lowering resolves previous/original/feedback
// dependencies to ordinary SR numbers before this representation is created.
enum class VxmSrOperandKind : std::uint8_t {
    StreamFloat16 = 0,
    StreamBFloat16 = 1,
    StreamFloat32 = 2,
    Immediate = 3,
    Accumulator = 4,
    StreamInt8 = 5,
};

struct VxmSrOperand {
    VxmSrOperandKind kind {VxmSrOperandKind::Immediate};
    std::uint8_t sr {0}; // packed E0..E31,W0..W31 byte-SR selector
    float immediate {0.0f};

    static VxmSrOperand StreamFloat16(std::size_t sr)
    {
        return {VxmSrOperandKind::StreamFloat16, checked(sr), 0.0f};
    }
    static VxmSrOperand StreamBFloat16(std::size_t sr)
    {
        return {VxmSrOperandKind::StreamBFloat16, checked(sr), 0.0f};
    }
    static VxmSrOperand StreamFloat32(std::size_t sr)
    {
        return {VxmSrOperandKind::StreamFloat32, checked(sr), 0.0f};
    }
    static VxmSrOperand StreamInt8(std::size_t sr)
    {
        return {VxmSrOperandKind::StreamInt8, checked(sr), 0.0f};
    }
    static VxmSrOperand Immediate(float value)
    {
        return {VxmSrOperandKind::Immediate, 0, value};
    }
    static VxmSrOperand Accumulator()
    {
        return {VxmSrOperandKind::Accumulator, 0, 0.0f};
    }

private:
    static std::uint8_t checked(std::size_t sr)
    {
        if (sr >= hw::kStreams)
            throw std::out_of_range("VXM operand SR is outside E0..W31");
        return static_cast<std::uint8_t>(sr);
    }
};

using VxmSrOperation = std::variant<VxmAluOpcode, VxmSpecialAluOpcode>;

struct VxmSrInstruction {
    VxmSrOperation operation {VxmAluOpcode::Bypass};
    VxmSrOperand lhs {VxmSrOperand::Immediate(0.0f)};
    VxmSrOperand rhs {VxmSrOperand::Immediate(0.0f)};
    std::uint8_t result_sr {0};
    VxmAluPrecision precision {VxmAluPrecision::Float16};
    VxmCastTarget result_type {VxmCastTarget::Float16};
    bool accumulator_reset {false};
    bool accumulator_write {false};
    bool accumulator_emit {true};
    bool accumulator_reset_first {false};
    bool accumulator_emit_last {false};
    // A tapped stream operand feeds the ALU and keeps forwarding to the next
    // SR column. The final reader consumes it in the usual way.
    bool lhs_stream_tap {false};
    bool rhs_stream_tap {false};
    std::uint32_t repeat_count {1};
};

struct VxmSrCompactInstruction {
    std::uint64_t control {0};
    std::uint32_t immediate_bits {0};
    friend bool operator==(const VxmSrCompactInstruction&,
        const VxmSrCompactInstruction&) = default;
};

// 96-bit physical format:
// [3:0] op, [6:4] lhs kind, [12:7] lhs SR, [15:13] rhs kind,
// [21:16] rhs SR, [27:22] result SR, [28] precision,
// [30:29] result type, [33:31] accumulator flags, [57:34] repeat,
// [58] reset accumulator only on the first repeat, [59] emit only on the
// final repeat, [60] lhs stream tap, [61] rhs stream tap.
// [63:62] is reserved and must be zero.
// The other word is the shared FP32 immediate. There is no chain-depth field.
class VxmSrCompactInstructionCodec {
public:
    static constexpr std::size_t kEncodedBits = 96;

    static VxmSrCompactInstruction encode(const VxmSrInstruction& instruction)
    {
        validate(instruction);
        std::uint64_t control = 0;
        put(control, encode_opcode(instruction.operation), 0, 0xf);
        put(control, static_cast<std::uint64_t>(instruction.lhs.kind), 4, 0x7);
        put(control, instruction.lhs.sr, 7, 0x3f);
        put(control, static_cast<std::uint64_t>(instruction.rhs.kind), 13, 0x7);
        put(control, instruction.rhs.sr, 16, 0x3f);
        put(control, instruction.result_sr, 22, 0x3f);
        put(control, instruction.precision == VxmAluPrecision::Float32, 28, 0x1);
        put(control, static_cast<std::uint64_t>(instruction.result_type), 29, 0x3);
        put(control, instruction.accumulator_reset, 31, 0x1);
        put(control, instruction.accumulator_write, 32, 0x1);
        put(control, instruction.accumulator_emit, 33, 0x1);
        put(control, instruction.repeat_count, 34, 0xffffffull);
        put(control, instruction.accumulator_reset_first, 58, 0x1);
        put(control, instruction.accumulator_emit_last, 59, 0x1);
        put(control, instruction.lhs_stream_tap, 60, 0x1);
        put(control, instruction.rhs_stream_tap, 61, 0x1);

        float immediate = 0.0f;
        bool immediateSeen = false;
        const auto selectImmediate = [&](const VxmSrOperand& operand) {
            if (!immediateSeen
                && operand.kind == VxmSrOperandKind::Immediate) {
                immediate = operand.immediate;
                immediateSeen = true;
            }
        };
        selectImmediate(instruction.lhs);
        if (uses_rhs(instruction.operation)) selectImmediate(instruction.rhs);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &immediate, sizeof(bits));
        return {control, bits};
    }

    static VxmSrInstruction decode(const VxmSrCompactInstruction& packet)
    {
        VxmSrInstruction result {};
        result.operation = decode_opcode(get(packet.control, 0, 0xf));
        result.lhs.kind = static_cast<VxmSrOperandKind>(get(packet.control, 4, 0x7));
        result.lhs.sr = static_cast<std::uint8_t>(get(packet.control, 7, 0x3f));
        result.rhs.kind = static_cast<VxmSrOperandKind>(get(packet.control, 13, 0x7));
        result.rhs.sr = static_cast<std::uint8_t>(get(packet.control, 16, 0x3f));
        result.result_sr = static_cast<std::uint8_t>(get(packet.control, 22, 0x3f));
        result.precision = get(packet.control, 28, 0x1)
            ? VxmAluPrecision::Float32 : VxmAluPrecision::Float16;
        result.result_type = static_cast<VxmCastTarget>(get(packet.control, 29, 0x3));
        result.accumulator_reset = get(packet.control, 31, 0x1) != 0;
        result.accumulator_write = get(packet.control, 32, 0x1) != 0;
        result.accumulator_emit = get(packet.control, 33, 0x1) != 0;
        result.repeat_count = static_cast<std::uint32_t>(
            get(packet.control, 34, 0xffffffull));
        result.accumulator_reset_first = get(packet.control, 58, 0x1) != 0;
        result.accumulator_emit_last = get(packet.control, 59, 0x1) != 0;
        result.lhs_stream_tap = get(packet.control, 60, 0x1) != 0;
        result.rhs_stream_tap = get(packet.control, 61, 0x1) != 0;
        float immediate = 0.0f;
        std::memcpy(&immediate, &packet.immediate_bits, sizeof(immediate));
        if (result.lhs.kind == VxmSrOperandKind::Immediate)
            result.lhs.immediate = immediate;
        if (result.rhs.kind == VxmSrOperandKind::Immediate)
            result.rhs.immediate = immediate;
        validate(result);
        return result;
    }

private:
    static bool is_stream(const VxmSrOperand& operand)
    {
        return operand.kind == VxmSrOperandKind::StreamFloat16
            || operand.kind == VxmSrOperandKind::StreamBFloat16
            || operand.kind == VxmSrOperandKind::StreamFloat32
            || operand.kind == VxmSrOperandKind::StreamInt8;
    }

    static void validate(const VxmSrInstruction& instruction)
    {
        if (instruction.repeat_count == 0)
            throw std::invalid_argument("VXM repeat count must be non-zero");
        if (instruction.repeat_count > 0xffffffu)
            throw std::out_of_range("VXM repeat count exceeds 24 bits");
        if (instruction.accumulator_reset_first
            && (!instruction.accumulator_reset
                || !instruction.accumulator_write))
            throw std::invalid_argument(
                "first-repeat accumulator reset requires reset and write");
        if (instruction.accumulator_emit_last
            && (!instruction.accumulator_emit
                || !instruction.accumulator_write))
            throw std::invalid_argument(
                "last-repeat accumulator emit requires emit and write");
        validate_operand(instruction.lhs);
        if (uses_rhs(instruction.operation)) validate_operand(instruction.rhs);
        if (instruction.lhs_stream_tap && !is_stream(instruction.lhs))
            throw std::invalid_argument("VXM lhs tap requires a stream operand");
        if (instruction.rhs_stream_tap
            && (!uses_rhs(instruction.operation) || !is_stream(instruction.rhs)))
            throw std::invalid_argument("VXM rhs tap requires a used stream operand");
        const auto check_immediate = [&](const VxmSrOperand& operand,
                                         float& shared, bool& seen) {
            if (operand.kind != VxmSrOperandKind::Immediate) return;
            if (seen && operand.immediate != shared)
                throw std::invalid_argument(
                    "VXM packet carries one shared FP32 immediate");
            shared = operand.immediate;
            seen = true;
        };
        float shared = 0.0f;
        bool seen = false;
        check_immediate(instruction.lhs, shared, seen);
        if (uses_rhs(instruction.operation))
            check_immediate(instruction.rhs, shared, seen);
        validate_span(instruction.result_sr,
            result_width(instruction.result_type), "VXM result SR span");
    }

    static void validate_operand(const VxmSrOperand& operand)
    {
        if (operand.kind == VxmSrOperandKind::StreamFloat16
            || operand.kind == VxmSrOperandKind::StreamBFloat16)
            validate_span(operand.sr, 2, "VXM 16-bit operand SR span");
        else if (operand.kind == VxmSrOperandKind::StreamFloat32)
            validate_span(operand.sr, 4, "VXM FP32 operand SR span");
        else if (operand.kind == VxmSrOperandKind::StreamInt8)
            validate_span(operand.sr, 1, "VXM int8 operand SR span");
        else if (operand.kind != VxmSrOperandKind::Immediate
            && operand.kind != VxmSrOperandKind::Accumulator)
            throw std::invalid_argument("invalid VXM SR operand kind");
    }

    static void validate_span(std::size_t sr, std::size_t width,
        const char* error)
    {
        if (sr >= hw::kStreams) throw std::out_of_range(error);
        const auto id = StreamId::from_packed(sr);
        if (id.index() + width > hw::kStreamsPerDirection)
            throw std::out_of_range(error);
    }

    static std::size_t result_width(VxmCastTarget type)
    {
        switch (type) {
        case VxmCastTarget::Int8: return 1;
        case VxmCastTarget::Float16:
        case VxmCastTarget::BFloat16: return 2;
        case VxmCastTarget::Float32: return 4;
        }
        throw std::invalid_argument("invalid VXM result type");
    }

    static bool uses_rhs(const VxmSrOperation& operation)
    {
        const auto* opcode = std::get_if<VxmAluOpcode>(&operation);
        if (!opcode) return false;
        return *opcode != VxmAluOpcode::Bypass
            && *opcode != VxmAluOpcode::Negate;
    }

    static void put(std::uint64_t& word, std::uint64_t value,
        unsigned shift, std::uint64_t mask)
    {
        if ((value & ~mask) != 0)
            throw std::out_of_range("VXM SR instruction field does not fit");
        word |= (value & mask) << shift;
    }
    static std::uint64_t get(std::uint64_t word, unsigned shift,
        std::uint64_t mask)
    {
        return (word >> shift) & mask;
    }

    static std::uint64_t encode_opcode(const VxmSrOperation& operation)
    {
        if (const auto* opcode = std::get_if<VxmAluOpcode>(&operation)) {
            switch (*opcode) {
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
        throw std::invalid_argument("unsupported VXM opcode");
    }

    static VxmSrOperation decode_opcode(std::uint64_t opcode)
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
        default: throw std::invalid_argument("invalid VXM opcode");
        }
    }
};

// Temporary compiler-side legalization helper for Schedule/Command producers
// that still describe chain-relative operands.  The returned instruction is
// already the physical, explicit-SR form; no FU instruction is generated and
// compressed afterwards.  New lowering should construct VxmSrInstruction
// directly and eventually remove this adapter.
inline VxmSrInstruction lower_legacy_vxm_to_explicit_sr(
    std::size_t column, VxmChainDepth depth,
    const VxmLaneAluInstruction& legacy,
    StreamDirection local_direction = StreamDirection::East,
    StreamDirection result_direction = StreamDirection::East)
{
    if (column >= hw::kVxmAluCount)
        throw std::out_of_range("VXM column exceeds the configured physical column count");
    const auto chain = static_cast<std::size_t>(depth);
    if (chain == 0 || chain > hw::kVxmAluCount
        || hw::kVxmAluCount % chain != 0)
        throw std::invalid_argument("invalid VXM chain depth");
    const auto head = column - column % chain;

    const auto direction_for = [&](VxmStreamSource source) {
        switch (source) {
        case VxmStreamSource::Local: return local_direction;
        case VxmStreamSource::East: return StreamDirection::East;
        case VxmStreamSource::West: return StreamDirection::West;
        }
        throw std::invalid_argument("invalid VXM stream source");
    };
    const auto stream_at = [](StreamDirection direction, std::size_t index) {
        return direction == StreamDirection::East
            ? StreamId::East(index) : StreamId::West(index);
    };
    const auto fixed_stream = [&](std::size_t stage, bool rhs,
                                  StreamDirection direction) {
        const auto group = VxmLane::fixed_input_group_for_stage(stage, rhs);
        return stream_at(direction, group * VxmLane::kStreamGroupBytes);
    };
    const auto chain_carrier = fixed_stream(head, false, local_direction);
    const auto original = fixed_stream(head, false, local_direction);
    const auto auxiliary = fixed_stream(head, true, local_direction);

    const auto lower_operand = [&](const VxmLaneOperand& operand,
                                   bool rhs_port) -> VxmSrOperand {
        switch (operand.kind) {
        case VxmLaneOperandKind::Previous:
        case VxmLaneOperandKind::Feedback:
            return VxmSrOperand::StreamFloat16(chain_carrier.packed());
        case VxmLaneOperandKind::Original:
            return VxmSrOperand::StreamFloat16(original.packed());
        case VxmLaneOperandKind::Auxiliary:
            return VxmSrOperand::StreamFloat16(auxiliary.packed());
        case VxmLaneOperandKind::Accumulator:
            return VxmSrOperand::Accumulator();
        case VxmLaneOperandKind::Immediate:
            return VxmSrOperand::Immediate(operand.immediate);
        case VxmLaneOperandKind::StreamFloat16:
        case VxmLaneOperandKind::StreamBFloat16: {
            if (operand.scale != 1.0f || operand.zero_point != 0)
                throw std::invalid_argument(
                    "explicit-SR VXM operand requires separately lowered scaling");
            if (operand.stream_group >= 8)
                throw std::out_of_range(
                    "legacy VXM stream group is outside 0..7");
            const auto direction = direction_for(operand.stream_source);
            const auto group = operand.stream_group >= 0
                ? static_cast<std::size_t>(operand.stream_group)
                : VxmLane::fixed_input_group_for_stage(column, rhs_port);
            const auto sr = stream_at(direction,
                group * VxmLane::kStreamGroupBytes).packed();
            return operand.kind == VxmLaneOperandKind::StreamFloat16
                ? VxmSrOperand::StreamFloat16(sr)
                : VxmSrOperand::StreamBFloat16(sr);
        }
        }
        throw std::invalid_argument("invalid legacy VXM operand");
    };

    VxmSrInstruction result {};
    result.operation = legacy.operation;
    result.lhs = lower_operand(legacy.lhs, false);
    result.rhs = lower_operand(legacy.rhs, true);
    const auto output_index = legacy.output_stream.value_or(
        chain_carrier.index());
    result.result_sr = static_cast<std::uint8_t>(
        stream_at(legacy.output_stream ? result_direction
                                       : chain_carrier.direction(),
            output_index).packed());
    result.precision = legacy.precision;
    result.result_type = legacy.output_type;
    result.accumulator_reset = legacy.accumulator_reset;
    // A legacy local-scalar write is a one-shot write to the same physical
    // per-column scalar register used by the explicit-SR accumulator operand.
    // It must not also emit a stream result.
    result.accumulator_write =
        legacy.accumulator_write || legacy.local_scalar_write;
    result.accumulator_emit =
        legacy.local_scalar_write ? false : legacy.accumulator_emit;
    result.accumulator_reset_first = legacy.accumulator_reset_first;
    result.accumulator_emit_last = legacy.accumulator_emit_last;
    if (legacy.repeat_count > 0xffffffu)
        throw std::out_of_range(
            "explicit-SR VXM repeat count exceeds 24 bits");
    result.repeat_count = static_cast<std::uint32_t>(legacy.repeat_count);
    return result;
}

} // namespace ftlpu
