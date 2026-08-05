#pragma once

#include "ftlpu/vxm/compiler/lowering.hpp"
#include "ftlpu/vxm/compiler/feedback_planner.hpp"
#include "ftlpu/vxm/compiler/stream_plan.hpp"
#include "ftlpu/vxm/compact_instruction.hpp"
#include "ftlpu/vxm/superlane.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::vxm::compiler {

struct VxmConfigCommand {
    std::size_t phase_id{0};
    std::size_t arrival_cycle{0};
    std::size_t stage{0};
    VxmCompactInstruction packet{};
};

struct VxmCompiledPhase {
    std::size_t phase_id{0};
    std::string name{};
    VxmChainDepth chain_depth{VxmChainDepth::Eight};
    std::size_t data_start_cycle{0};
    std::size_t end_cycle{0};
    std::size_t config_deadline_cycle{0};
    bool feedback_from_previous{false};
    std::vector<VxmConfigCommand> config_commands{};
    std::vector<VxmStreamRequirement> stream_requirements{};
    std::vector<VxmLocalScalarLoad> local_scalar_loads{};
};

struct VxmCompiledProgram {
    VxmKernel kernel{};
    VxmSchedule schedule{};
    VxmStreamPlan stream_plan{};
    std::vector<VxmCompiledPhase> phases{};
};

namespace detail {

inline VxmLaneOperand encode_operand(const VxmScheduledOperand& operand)
{
    switch (operand.kind) {
    case VxmScheduledOperandKind::StreamValue:
        return VxmLaneOperand::StreamFloat16();
    case VxmScheduledOperandKind::Previous:
        return VxmLaneOperand::Previous();
    case VxmScheduledOperandKind::Original:
        return VxmLaneOperand::Original();
    case VxmScheduledOperandKind::Auxiliary:
        return VxmLaneOperand::Aux();
    case VxmScheduledOperandKind::Immediate:
        return VxmLaneOperand::Imm(operand.immediate);
    case VxmScheduledOperandKind::LocalScalar:
        return VxmLaneOperand::Acc();
    case VxmScheduledOperandKind::Feedback:
        return VxmLaneOperand::Feedback();
    }
    throw std::invalid_argument("unknown VXM scheduled operand");
}

inline VxmLaneAluInstruction encode_instruction(
    const VxmScheduledInstruction& scheduled, VxmChainDepth depth)
{
    auto encoded = VxmLaneAluInstruction{};
    encoded.operation = scheduled.operation;
    encoded.lhs = encode_operand(scheduled.lhs);
    encoded.rhs = encode_operand(scheduled.rhs);
    encoded.precision = scheduled.precision;
    encoded.output_type = VxmCastTarget::Float16;
    encoded.accumulator_reset = scheduled.accumulator_reset;
    encoded.accumulator_write = scheduled.accumulator_write;
    encoded.accumulator_emit = scheduled.accumulator_emit;
    encoded.repeat_count = scheduled.repeat_count;
    if (scheduled.output && scheduled.stream_output) {
        encoded.output_stream =
            VxmLane::fixed_output_stream_for_block(
                VxmLane::block_for_stage(scheduled.stage));
    }

    auto validator = VxmLane{};
    validator.validate_broadcast_instruction(
        depth, scheduled.stage, encoded);
    return encoded;
}

} // namespace detail

inline VxmCompiledProgram compile_schedule(
    VxmKernel kernel, VxmSchedule schedule)
{
    kernel.validate();
    plan_multi_chain_feedback_merges(schedule);
    schedule.validate();
    auto stream_plan = make_stream_plan(schedule);

    auto program = VxmCompiledProgram{};
    program.kernel = std::move(kernel);
    program.schedule = schedule;
    program.stream_plan = stream_plan;

    for (const auto& phase : program.schedule.phases) {
        auto compiled = VxmCompiledPhase{};
        compiled.phase_id = phase.id;
        compiled.name = phase.name;
        compiled.chain_depth = phase.chain_depth;
        compiled.data_start_cycle = phase.data_start_cycle;
        compiled.end_cycle = phase.end_cycle;
        compiled.feedback_from_previous =
            phase.feedback_from_previous;
        compiled.config_deadline_cycle =
            phase.data_start_cycle > phase.config_lead_cycles
            ? phase.data_start_cycle - phase.config_lead_cycles
            : 0;

        // The Slice owns eight south-to-north instruction lanes. Logical Ci
        // configures both physical Ci and C(i+8); the two physical datapaths
        // carry different tokens under the same operation. A Schedule may
        // describe both physical copies for Stream planning, but only the
        // canonical C0..C7 instruction sequence is transported.
        auto stage_arrival_offset =
            std::array<std::size_t,
                       VxmSuperlaneInstructionControl::kStageCount>{};
        auto canonical_packets = std::array<
            std::vector<VxmCompactInstruction>,
            VxmSuperlaneInstructionControl::kStageCount>{};
        for (const auto& scheduled : phase.instructions) {
            if (scheduled.stage
                >= VxmSuperlaneInstructionControl::kStageCount) {
                continue;
            }
            auto& offset = stage_arrival_offset.at(scheduled.stage);
            if (offset >= VxmSuperlaneInstructionControl::kFifoDepth) {
                throw std::overflow_error(
                    "VXM phase requires more than three Config FIFO entries "
                    "for one ALU stage");
            }
            const auto packet = VxmCompactInstructionCodec::encode(
                scheduled.stage,
                phase.chain_depth,
                detail::encode_instruction(
                    scheduled, phase.chain_depth));
            canonical_packets[scheduled.stage].push_back(packet);
            compiled.config_commands.push_back({
                phase.id,
                compiled.config_deadline_cycle + offset,
                scheduled.stage,
                packet,
            });
            ++offset;
        }
        auto mirrored_offsets = std::array<
            std::size_t,
            VxmSuperlaneInstructionControl::kStageCount>{};
        for (const auto& scheduled : phase.instructions) {
            if (scheduled.stage
                < VxmSuperlaneInstructionControl::kStageCount) {
                continue;
            }
            const auto logical_stage = scheduled.stage
                - VxmSuperlaneInstructionControl::kMirroredStageOffset;
            auto& mirrored_offset = mirrored_offsets.at(logical_stage);
            const auto packet = VxmCompactInstructionCodec::encode(
                scheduled.stage,
                phase.chain_depth,
                detail::encode_instruction(
                    scheduled, phase.chain_depth));
            if (mirrored_offset >= canonical_packets[logical_stage].size()
                || packet
                    != canonical_packets[logical_stage][mirrored_offset]) {
                throw std::invalid_argument(
                    "VXM physical stages Ci/C(i+8) require identical "
                    "instruction sequences under shared control");
            }
            ++mirrored_offset;
        }
        for (const auto& requirement : program.stream_plan.requirements) {
            if (requirement.phase_id == phase.id) {
                compiled.stream_requirements.push_back(requirement);
            }
        }
        for (const auto& load : program.stream_plan.local_scalar_loads) {
            if (load.phase_id == phase.id) {
                compiled.local_scalar_loads.push_back(load);
            }
        }
        program.phases.push_back(std::move(compiled));
    }
    return program;
}

inline VxmCompiledProgram compile_kernel(VxmKernel kernel)
{
    kernel.validate();
    kernel.eliminate_dead_code();
    kernel.validate();
    auto schedule = lower_kernel(kernel);
    return compile_schedule(std::move(kernel), std::move(schedule));
}

} // namespace ftlpu::vxm::compiler
