#pragma once

#include "ftlpu/vxm/compiler/schedule_ir.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu::vxm::compiler {

enum class VxmStreamDirection {
    Input,
    Output,
};

// This is the VXM-visible contract exported to the global static scheduler.
// It contains absolute Stream Register byte indices; no VXM chain-head name is
// exposed outside the VXM compiler.
struct VxmStreamRequirement {
    std::size_t phase_id{0};
    std::string phase_name{};
    ValueId value{kInvalidValue};
    VxmStreamDirection direction{VxmStreamDirection::Input};
    std::size_t stream_base{0};
    std::size_t byte_count{VxmLane::kStreamGroupBytes};
    std::size_t first_cycle{0};
    std::size_t transfer_count{0};
    std::size_t period{1};

    // A held value is written once and reused from the same fixed Stream
    // Register group for reuse_count VXM input cycles.
    bool hold{false};
    std::size_t reuse_count{1};
    std::size_t element_base{0};
};

struct VxmLocalScalarLoad {
    std::size_t phase_id{0};
    ValueId value{kInvalidValue};
    std::size_t source_stream_base{0};
    std::size_t destination_stage{0};
    std::size_t load_cycle{0};
    std::size_t element_index{0};
};

struct VxmStreamPlan {
    std::vector<VxmStreamRequirement> requirements{};
    std::vector<VxmLocalScalarLoad> local_scalar_loads{};
};

namespace detail {

inline std::size_t output_stream_for_value(
    const VxmSchedule& schedule, std::size_t before_phase, ValueId value,
    std::size_t token_index)
{
    for (std::size_t phase_index = before_phase; phase_index-- > 0;) {
        for (const auto& output : schedule.phases[phase_index].outputs) {
            if (output.value == value
                && output.token_index == token_index) {
                return VxmLane::fixed_output_stream_for_block(
                    VxmLane::block_for_stage(output.chain_tail));
            }
        }
    }
    throw std::invalid_argument(
        "VXM local scalar has no earlier fixed Stream output");
}

inline std::size_t row_element_base(
    const VxmTensorType& type, std::size_t token_index)
{
    if (type.rank() < 2) return 0;
    return token_index * type.shape.back();
}

} // namespace detail

inline VxmStreamPlan make_stream_plan(const VxmSchedule& schedule)
{
    schedule.validate();
    auto plan = VxmStreamPlan{};

    for (const auto& phase : schedule.phases) {
        for (const auto& input : phase.inputs) {
            const auto rhs = input.port == VxmInputPort::Rhs;
            const auto stream_base =
                VxmLane::fixed_input_group_for_stage(input.chain_head, rhs)
                * VxmLane::kStreamGroupBytes;
            const auto hold =
                input.access == VxmInputAccess::HoldForPhase;
            plan.requirements.push_back({
                phase.id,
                phase.name,
                input.value,
                VxmStreamDirection::Input,
                stream_base,
                VxmLane::kStreamGroupBytes,
                phase.data_start_cycle,
                hold ? std::size_t{1} : phase.element_count,
                hold ? std::size_t{0} : std::size_t{1},
                hold,
                phase.element_count,
                detail::row_element_base(input.type, input.token_index),
            });
        }

        for (const auto& output : phase.outputs) {
            if (!output.stream_write) continue;
            const auto stream_base =
                VxmLane::fixed_output_stream_for_block(
                    VxmLane::block_for_stage(output.chain_tail));
            const auto count =
                output.scalar ? std::size_t{1} : phase.element_count;
            plan.requirements.push_back({
                phase.id,
                phase.name,
                output.value,
                VxmStreamDirection::Output,
                stream_base,
                VxmLane::kStreamGroupBytes,
                phase.end_cycle - count,
                count,
                1,
                false,
                1,
                detail::row_element_base(output.type, output.token_index),
            });
        }

        for (const auto& scalar : phase.local_scalars) {
            if (phase.data_start_cycle == 0) {
                throw std::invalid_argument(
                    "VXM local scalar has no cycle available for loading");
            }
            plan.local_scalar_loads.push_back({
                phase.id,
                scalar.value,
                detail::output_stream_for_value(
                    schedule, phase.id, scalar.value,
                    scalar.token_index),
                scalar.stage,
                phase.data_start_cycle - 1,
                detail::row_element_base(
                    scalar.type, scalar.token_index),
            });
        }
    }
    return plan;
}

} // namespace ftlpu::vxm::compiler
