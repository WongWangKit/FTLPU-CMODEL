#pragma once

#include "ftlpu/system/icu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>

namespace ftlpu::test {

struct DequantSpec {
    std::size_t input_stream_base{0};
    std::size_t output_stream_base{0};
    std::array<float, hw::kLanesPerTile> scales{};
    Hemisphere input_hemisphere{Hemisphere::East};
    Hemisphere output_hemisphere{Hemisphere::East};
};

struct SwishSpec {
    std::size_t gate_stream_base{0};
    std::size_t up_stream_base{4};
    std::size_t output_stream_base{0};
    Hemisphere input_hemisphere{Hemisphere::East};
    Hemisphere output_hemisphere{Hemisphere::East};
};

inline void enqueue_alu_at(
    InstructionControlUnit& icu,
    std::array<std::size_t, VxmLane::kAluCount>& cursors,
    std::size_t alu,
    std::size_t cycle,
    VxmLaneAluInstruction instruction)
{
    if (cycle < cursors[alu]) {
        throw std::logic_error(
            "offline VXM ALU" + std::to_string(alu)
            + " requests cycle " + std::to_string(cycle)
            + " but its cursor is " + std::to_string(cursors[alu]));
    }
    icu.enqueue_vxm_nop(alu, cycle - cursors[alu]);
    icu.enqueue_vxm(alu, instruction);
    cursors[alu] = cycle + 1;
}

inline void enqueue_dequant(
    InstructionControlUnit& icu,
    std::array<std::size_t, VxmLane::kAluCount>& cursors,
    std::size_t cycle,
    const DequantSpec& spec)
{
    for (std::size_t element = 0; element < hw::kLanesPerTile; ++element) {
        enqueue_alu_at(icu, cursors, element, cycle, {
            VxmAluOpcode::Multiply,
            VxmLaneOperand::StreamInt8(spec.input_stream_base + element),
            VxmLaneOperand::Imm(spec.scales[element]),
            1.0f, 0, VxmCastTarget::Float32, std::nullopt,
            spec.input_hemisphere, spec.output_hemisphere});
        enqueue_alu_at(icu, cursors, hw::kLanesPerTile + element, cycle + 1, {
            VxmAluOpcode::Cast,
            VxmLaneOperand::Alu(element),
            VxmLaneOperand::Imm(0.0f),
            1.0f, 0, VxmCastTarget::Float16, spec.output_stream_base + element * 2,
            spec.input_hemisphere, spec.output_hemisphere});
    }
}

inline void enqueue_swish(
    InstructionControlUnit& icu,
    std::array<std::size_t, VxmLane::kAluCount>& cursors,
    std::size_t cycle,
    const SwishSpec& spec)
{
    const auto inst = [&](VxmAluOpcode op, VxmLaneOperand lhs, VxmLaneOperand rhs,
                          VxmCastTarget cast = VxmCastTarget::Float32,
                          std::optional<std::size_t> output = std::nullopt) {
        return VxmLaneAluInstruction {op, lhs, rhs, 1.0f, 0, cast, output,
            spec.input_hemisphere, spec.output_hemisphere};
    };
    enqueue_alu_at(icu, cursors, 0, cycle, inst(VxmAluOpcode::Negate,
        VxmLaneOperand::StreamFloat32(spec.gate_stream_base), VxmLaneOperand::Imm(0.0f)));
    enqueue_alu_at(icu, cursors, 1, cycle, inst(VxmAluOpcode::Multiply,
        VxmLaneOperand::StreamFloat32(spec.gate_stream_base),
        VxmLaneOperand::StreamFloat32(spec.up_stream_base)));
    enqueue_alu_at(icu, cursors, 2, cycle + 1, inst(VxmAluOpcode::Exp,
        VxmLaneOperand::Alu(0), VxmLaneOperand::Imm(0.0f)));
    enqueue_alu_at(icu, cursors, 5, cycle + 1, inst(VxmAluOpcode::Pass,
        VxmLaneOperand::Alu(1), VxmLaneOperand::Imm(0.0f)));
    enqueue_alu_at(icu, cursors, 3, cycle + 2, inst(VxmAluOpcode::Add,
        VxmLaneOperand::Alu(2), VxmLaneOperand::Imm(1.0f)));
    enqueue_alu_at(icu, cursors, 6, cycle + 2, inst(VxmAluOpcode::Pass,
        VxmLaneOperand::Alu(5), VxmLaneOperand::Imm(0.0f)));
    enqueue_alu_at(icu, cursors, 4, cycle + 3, inst(VxmAluOpcode::Divide,
        VxmLaneOperand::Imm(1.0f), VxmLaneOperand::Alu(3)));
    enqueue_alu_at(icu, cursors, 7, cycle + 3, inst(VxmAluOpcode::Pass,
        VxmLaneOperand::Alu(6), VxmLaneOperand::Imm(0.0f)));
    enqueue_alu_at(icu, cursors, 8, cycle + 4, inst(VxmAluOpcode::Multiply,
        VxmLaneOperand::Alu(7), VxmLaneOperand::Alu(4)));
    enqueue_alu_at(icu, cursors, 9, cycle + 5, inst(VxmAluOpcode::Cast,
        VxmLaneOperand::Alu(8), VxmLaneOperand::Imm(0.0f),
        VxmCastTarget::Float16, spec.output_stream_base));
}

} // namespace ftlpu::test
