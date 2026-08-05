#pragma once

#include "ftlpu/vxm/compiler/compiler.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace vxm_compiler_test::feedback {

using namespace ftlpu;
using namespace ftlpu::vxm::compiler;

struct ReductionKernel {
    VxmKernel kernel{};
    std::array<ValueId, 4> inputs{};
    ValueId pair0{kInvalidValue};
    ValueId pair1{kInvalidValue};
    ValueId total{kInvalidValue};
};

ReductionKernel make_neural_reduction_kernel()
{
    auto builder = VxmKernelBuilder{"neural_partial_sum"};
    auto tensors = std::array<VxmTensor, 4>{};
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        tensors[index] = builder.input(
            "partial_" + std::to_string(index), {{1}});
    }
    const auto pair0 =
        builder.add(tensors[0], tensors[1], "pair_sum_0");
    const auto pair1 =
        builder.add(tensors[2], tensors[3], "pair_sum_1");
    const auto total =
        builder.add(pair0, pair1, "denominator");
    builder.output(total);

    return {
        builder.finish(),
        {tensors[0].id, tensors[1].id,
         tensors[2].id, tensors[3].id},
        pair0.id,
        pair1.id,
        total.id,
    };
}

VxmScheduledInstruction basic_instruction(
    std::size_t stage, VxmAluOpcode opcode,
    VxmScheduledOperand lhs, VxmScheduledOperand rhs,
    std::size_t repeat_count,
    std::optional<ValueId> output = std::nullopt)
{
    auto instruction = VxmScheduledInstruction{};
    instruction.stage = stage;
    instruction.operation = opcode;
    instruction.lhs = lhs;
    instruction.rhs = rhs;
    instruction.precision = VxmAluPrecision::Float32;
    instruction.repeat_count = repeat_count;
    instruction.output = output;
    return instruction;
}

VxmPhaseInput phase_input(
    ValueId value, std::size_t head, const VxmKernel& kernel)
{
    return {
        value,
        head,
        VxmInputPort::Lhs,
        VxmInputAccess::StreamEachCycle,
        kernel.type_of(value),
    };
}

VxmPhaseOutput phase_output(
    ValueId value, std::size_t tail, const VxmKernel& kernel)
{
    return {value, tail, kernel.type_of(value), true, true};
}

VxmScheduledPhase make_four_partial_phase(
    const ReductionKernel& reduction)
{
    auto phase = VxmScheduledPhase{};
    phase.id = 0;
    phase.name = "four_parallel_partial_values";
    phase.chain_depth = VxmChainDepth::Two;
    phase.element_count = 1;
    phase.data_start_cycle = 1;
    phase.end_cycle = 3;

    for (std::size_t chain = 0; chain < 4; ++chain) {
        const auto head = chain * 2;
        const auto tail = head + 1;
        const auto value = reduction.inputs[chain];
        phase.inputs.push_back(
            phase_input(value, head, reduction.kernel));
        phase.instructions.push_back(basic_instruction(
            head, VxmAluOpcode::Bypass,
            VxmScheduledOperand::Stream(value),
            VxmScheduledOperand::Immediate(0.0f), 1));
        phase.instructions.push_back(basic_instruction(
            tail, VxmAluOpcode::Bypass,
            VxmScheduledOperand::Previous(),
            VxmScheduledOperand::Immediate(0.0f), 1, value));
        phase.outputs.push_back(
            phase_output(value, tail, reduction.kernel));
    }
    return phase;
}

void append_pair_reduction_chain(
    VxmScheduledPhase& phase, std::size_t head,
    ValueId first, ValueId second, ValueId output,
    const VxmKernel& kernel)
{
    phase.inputs.push_back(phase_input(first, head, kernel));
    phase.inputs.push_back(phase_input(second, head, kernel));
    phase.instructions.push_back(basic_instruction(
        head, VxmAluOpcode::Bypass,
        VxmScheduledOperand::Stream(first),
        VxmScheduledOperand::Immediate(0.0f), 2));
    phase.instructions.push_back(basic_instruction(
        head + 1, VxmAluOpcode::Bypass,
        VxmScheduledOperand::Previous(),
        VxmScheduledOperand::Immediate(0.0f), 2));
    phase.instructions.push_back(basic_instruction(
        head + 2, VxmAluOpcode::Bypass,
        VxmScheduledOperand::Previous(),
        VxmScheduledOperand::Immediate(0.0f), 2));

    auto first_add = basic_instruction(
        head + 3, VxmAluOpcode::Add,
        VxmScheduledOperand::Previous(),
        VxmScheduledOperand::LocalScalar(output), 1);
    first_add.accumulator_reset = true;
    first_add.accumulator_write = true;
    first_add.accumulator_emit = false;
    phase.instructions.push_back(first_add);

    auto final_add = basic_instruction(
        head + 3, VxmAluOpcode::Add,
        VxmScheduledOperand::Previous(),
        VxmScheduledOperand::LocalScalar(output), 1, output);
    final_add.accumulator_write = true;
    phase.instructions.push_back(final_add);
    phase.outputs.push_back(phase_output(
        output, head + 3, kernel));
}

VxmScheduledPhase make_two_pair_phase(
    const ReductionKernel& reduction)
{
    auto phase = VxmScheduledPhase{};
    phase.id = 1;
    phase.name = "two_parallel_pair_sums";
    phase.chain_depth = VxmChainDepth::Four;
    phase.element_count = 2;
    phase.data_start_cycle = 3;
    phase.end_cycle = 8;

    append_pair_reduction_chain(
        phase, 0,
        reduction.inputs[0], reduction.inputs[1],
        reduction.pair0, reduction.kernel);
    append_pair_reduction_chain(
        phase, 4,
        reduction.inputs[2], reduction.inputs[3],
        reduction.pair1, reduction.kernel);
    return phase;
}

VxmScheduledPhase make_final_phase(
    const ReductionKernel& reduction, std::size_t id,
    std::size_t start_cycle, bool direct_from_four)
{
    auto phase = VxmScheduledPhase{};
    phase.id = id;
    phase.name = direct_from_four
        ? "direct_four_to_one_sum"
        : "two_to_one_final_sum";
    phase.chain_depth = VxmChainDepth::Eight;
    phase.element_count = direct_from_four ? 4 : 2;
    phase.data_start_cycle = start_cycle;
    phase.end_cycle =
        start_cycle + phase.element_count + 8 - 1;

    const auto sources = direct_from_four
        ? std::vector<ValueId>{
            reduction.inputs[0], reduction.inputs[1],
            reduction.inputs[2], reduction.inputs[3]}
        : std::vector<ValueId>{reduction.pair0, reduction.pair1};
    for (const auto source : sources) {
        phase.inputs.push_back(
            phase_input(source, 0, reduction.kernel));
    }
    for (std::size_t stage = 0; stage < 7; ++stage) {
        phase.instructions.push_back(basic_instruction(
            stage, VxmAluOpcode::Bypass,
            stage == 0
                ? VxmScheduledOperand::Stream(sources.front())
                : VxmScheduledOperand::Previous(),
            VxmScheduledOperand::Immediate(0.0f),
            phase.element_count));
    }

    auto first_add = basic_instruction(
        7, VxmAluOpcode::Add,
        VxmScheduledOperand::Previous(),
        VxmScheduledOperand::LocalScalar(reduction.total), 1);
    first_add.accumulator_reset = true;
    first_add.accumulator_write = true;
    first_add.accumulator_emit = false;
    phase.instructions.push_back(first_add);
    if (phase.element_count > 2) {
        auto middle_add = basic_instruction(
            7, VxmAluOpcode::Add,
            VxmScheduledOperand::Previous(),
            VxmScheduledOperand::LocalScalar(reduction.total),
            phase.element_count - 2);
        middle_add.accumulator_write = true;
        middle_add.accumulator_emit = false;
        phase.instructions.push_back(middle_add);
    }
    auto final_add = basic_instruction(
        7, VxmAluOpcode::Add,
        VxmScheduledOperand::Previous(),
        VxmScheduledOperand::LocalScalar(reduction.total),
        1, reduction.total);
    final_add.accumulator_write = true;
    phase.instructions.push_back(final_add);
    phase.outputs.push_back(
        phase_output(reduction.total, 7, reduction.kernel));
    return phase;
}

VxmSchedule make_staged_schedule(const ReductionKernel& reduction)
{
    auto schedule = VxmSchedule{};
    schedule.kernel_name = reduction.kernel.name;
    schedule.phases.push_back(make_four_partial_phase(reduction));
    schedule.phases.push_back(make_two_pair_phase(reduction));
    schedule.phases.push_back(
        make_final_phase(reduction, 2, 8, false));

    schedule.total_cycles = schedule.phases.back().end_cycle;
    schedule.validate();
    return schedule;
}

VxmSchedule make_direct_schedule(const ReductionKernel& reduction)
{
    auto schedule = VxmSchedule{};
    schedule.kernel_name = reduction.kernel.name;
    schedule.phases.push_back(make_four_partial_phase(reduction));
    schedule.phases.push_back(
        make_final_phase(reduction, 1, 3, true));
    schedule.total_cycles = schedule.phases.back().end_cycle;
    schedule.validate();
    return schedule;
}

void check_routes(
    const VxmScheduledPhase& phase,
    VxmChainDepth expected_depth,
    std::size_t expected_routes,
    std::size_t expected_holding)
{
    assert(phase.chain_depth == expected_depth);
    assert(phase.feedback_from_previous);
    assert(phase.feedback_routes.size() == expected_routes);
    assert(std::count_if(
        phase.feedback_routes.begin(),
        phase.feedback_routes.end(),
        [](const auto& route) {
            return route.uses_holding_register;
        }) == static_cast<std::ptrdiff_t>(expected_holding));
}

VxmCModelRunResult execute_and_check(
    const ReductionKernel& reduction,
    const VxmCompiledProgram& program)
{
    auto values = VxmHostValueStore{};
    for (std::size_t tile = 0;
         tile < VxmSlice::kTileCount; ++tile) {
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            for (std::size_t input = 0; input < 4; ++input) {
                values.set(
                    reduction.inputs[input], tile, lane,
                    {static_cast<float>(
                        1 + input + lane + tile)});
            }
        }
    }

    auto slice = VxmSlice{};
    auto result =
        VxmSliceCModelAdapter{VxmSramTiming{1, 1}}.run(
            slice, program, values);
    for (std::size_t tile = 0;
         tile < VxmSlice::kTileCount; ++tile) {
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            const auto& output =
                values.get(reduction.total, tile, lane);
            assert(output.size() == 1);
            const auto expected =
                10.0f + 4.0f * static_cast<float>(lane + tile);
            assert(std::fabs(output.front() - expected) < 1.0e-6f);
        }
    }
    return result;
}

void check_no_bubble(
    const VxmCModelRunResult& result,
    std::size_t boundary_cycle,
    std::initializer_list<std::size_t> old_tails,
    std::initializer_list<std::size_t> new_heads,
    VxmChainDepth new_depth)
{
    const auto& boundary = result.timeline.at(boundary_cycle);
    for (const auto tail : old_tails) {
        assert(boundary.alu_states[0][tail]
               == VxmLaneAluTraceState::Executed);
    }
    assert(std::any_of(
        boundary.depth_changes.begin(),
        boundary.depth_changes.end(),
        [new_depth](const auto& event) {
            return event.superlane == 0
                && event.feedback_transition
                && event.depth == new_depth;
        }));
    const auto& next = result.timeline.at(boundary_cycle + 1);
    for (const auto head : new_heads) {
        assert(next.alu_states[0][head]
               == VxmLaneAluTraceState::Executed);
    }
}

inline void check()
{
    const auto reduction = make_neural_reduction_kernel();

    const auto staged = make_staged_schedule(reduction);
    const auto staged_program =
        compile_schedule(reduction.kernel, staged);
    check_routes(
        staged_program.schedule.phases[1],
        VxmChainDepth::Four, 4, 2);
    check_routes(
        staged_program.schedule.phases[2],
        VxmChainDepth::Eight, 2, 1);
    const auto staged_result =
        execute_and_check(reduction, staged_program);
    check_no_bubble(
        staged_result,
        staged_program.phases[1].data_start_cycle
            + staged_result.phase_shifts[1] - 1,
        {1, 3, 5, 7}, {0, 4},
        VxmChainDepth::Four);
    check_no_bubble(
        staged_result,
        staged_program.phases[2].data_start_cycle
            + staged_result.phase_shifts[2] - 1,
        {3, 7}, {0},
        VxmChainDepth::Eight);

    const auto direct = make_direct_schedule(reduction);
    const auto direct_program =
        compile_schedule(reduction.kernel, direct);
    check_routes(
        direct_program.schedule.phases[1],
        VxmChainDepth::Eight, 4, 3);
    const auto direct_result =
        execute_and_check(reduction, direct_program);
    check_no_bubble(
        direct_result,
        direct_program.phases[1].data_start_cycle
            + direct_result.phase_shifts[1] - 1,
        {1, 3, 5, 7}, {0},
        VxmChainDepth::Eight);

    std::cout
        << "VXM compiler multi-chain feedback passed: "
        << "2->4, 4->8, 2->8 neural reduction\n";
}

} // namespace vxm_compiler_test::feedback
