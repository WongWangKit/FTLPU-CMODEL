#pragma once

#include "ftlpu/vxm/compiler/kernel_ir.hpp"
#include "ftlpu/vxm/lane.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::vxm::compiler {

enum class VxmScheduledOperandKind {
    StreamValue,
    Previous,
    Original,
    Auxiliary,
    Immediate,
    LocalScalar,
    Feedback,
};

struct VxmScheduledOperand {
    VxmScheduledOperandKind kind{VxmScheduledOperandKind::Immediate};
    ValueId value{kInvalidValue};
    float immediate{0.0f};

    static VxmScheduledOperand Stream(ValueId value)
    {
        return {VxmScheduledOperandKind::StreamValue, value, 0.0f};
    }
    static VxmScheduledOperand Previous()
    {
        return {VxmScheduledOperandKind::Previous};
    }
    static VxmScheduledOperand Original()
    {
        return {VxmScheduledOperandKind::Original};
    }
    static VxmScheduledOperand Auxiliary()
    {
        return {VxmScheduledOperandKind::Auxiliary};
    }
    static VxmScheduledOperand Immediate(float immediate)
    {
        return {VxmScheduledOperandKind::Immediate,
                kInvalidValue, immediate};
    }
    static VxmScheduledOperand LocalScalar(ValueId value)
    {
        return {VxmScheduledOperandKind::LocalScalar, value, 0.0f};
    }
    static VxmScheduledOperand Feedback(ValueId value)
    {
        return {VxmScheduledOperandKind::Feedback, value, 0.0f};
    }
};

enum class VxmInputPort {
    Lhs,
    Rhs,
};

enum class VxmInputAccess {
    StreamEachCycle,
    HoldForPhase,
};

struct VxmPhaseInput {
    ValueId value{kInvalidValue};
    std::size_t chain_head{0};
    VxmInputPort port{VxmInputPort::Lhs};
    VxmInputAccess access{VxmInputAccess::StreamEachCycle};
    VxmTensorType type{};
    std::size_t token_index{0};
};

struct VxmLocalScalarBinding {
    ValueId value{kInvalidValue};
    std::size_t stage{0};
    VxmTensorType type{};
    std::size_t token_index{0};
};

struct VxmPhaseOutput {
    ValueId value{kInvalidValue};
    std::size_t chain_tail{0};
    VxmTensorType type{};
    bool scalar{false};
    bool stream_write{true};
    std::size_t token_index{0};
};

struct VxmScheduledInstruction {
    std::size_t stage{0};
    VxmLaneOperation operation{VxmAluOpcode::Bypass};
    VxmScheduledOperand lhs{VxmScheduledOperand::Previous()};
    VxmScheduledOperand rhs{VxmScheduledOperand::Immediate(0.0f)};
    VxmAluPrecision precision{VxmAluPrecision::Float16};

    std::size_t repeat_count{1};
    bool accumulator_reset{false};
    bool accumulator_write{false};
    bool accumulator_emit{true};
    std::optional<ValueId> output{};
    bool stream_output{true};
};

struct VxmFeedbackRoute {
    ValueId value{kInvalidValue};
    std::size_t source_tail{0};
    std::size_t destination_head{0};
    bool uses_holding_register{false};
};

struct VxmScheduledPhase {
    std::size_t id{0};
    std::string name{};
    VxmChainDepth chain_depth{VxmChainDepth::Eight};

    // element_count is the number accepted by each active chain.
    std::size_t element_count{0};
    std::size_t token_begin{0};
    std::size_t parallel_chain_count{1};
    std::size_t data_start_cycle{0};
    std::size_t end_cycle{0}; // exclusive
    std::size_t config_lead_cycles{1};
    bool feedback_from_previous{false};

    std::vector<VxmFeedbackRoute> feedback_routes{};
    std::vector<VxmPhaseInput> inputs{};
    std::vector<VxmLocalScalarBinding> local_scalars{};
    std::vector<VxmPhaseOutput> outputs{};
    std::vector<VxmScheduledInstruction> instructions{};
};

struct VxmSchedule {
    std::string kernel_name{};
    std::vector<VxmScheduledPhase> phases{};
    std::size_t total_cycles{0};

    void validate() const
    {
        std::size_t previous_end = 0;
        for (std::size_t phase_index = 0;
             phase_index < phases.size(); ++phase_index) {
            const auto& phase = phases[phase_index];
            if (phase.id != phase_index) {
                throw std::invalid_argument(
                    "VXM Schedule phase IDs must be contiguous");
            }
            if (phase.name.empty() || phase.element_count == 0) {
                throw std::invalid_argument(
                    "VXM Schedule phase requires a name and element count");
            }
            if (phase.data_start_cycle < previous_end
                || phase.end_cycle <= phase.data_start_cycle) {
                throw std::invalid_argument(
                    "VXM Schedule phases overlap or have invalid timing");
            }
            previous_end = phase.end_cycle;

            const auto depth =
                static_cast<std::size_t>(phase.chain_depth);
            const auto physical_chains = VxmLane::kAluCount / depth;
            if (phase.parallel_chain_count == 0
                || phase.parallel_chain_count > physical_chains) {
                throw std::invalid_argument(
                    "VXM Schedule phase has an invalid parallel chain count");
            }
            if (phase.feedback_from_previous) {
                if (phase.id == 0 || phase.feedback_routes.empty()) {
                    throw std::invalid_argument(
                        "VXM feedback phase must follow another phase and "
                        "contain at least one feedback route");
                }
                const auto& previous = phases[phase_index - 1];
                const auto previous_depth =
                    static_cast<std::size_t>(previous.chain_depth);
                auto routes_per_head =
                    std::vector<std::size_t>(VxmLane::kAluCount);
                for (const auto& route : phase.feedback_routes) {
                    if (route.source_tail >= VxmLane::kAluCount
                        || route.source_tail % previous_depth
                            != previous_depth - 1
                        || route.destination_head >= VxmLane::kAluCount
                        || route.destination_head % depth != 0
                        || !VxmLane::has_fixed_feedback_path(
                            route.source_tail,
                            route.destination_head)) {
                        throw std::invalid_argument(
                            "VXM feedback route is outside the fixed "
                            "tail-to-head network");
                    }
                    const auto source = std::find_if(
                        previous.outputs.begin(),
                        previous.outputs.end(),
                        [&route](const auto& output) {
                            return output.value == route.value
                                && output.chain_tail
                                    == route.source_tail
                                && !output.stream_write;
                        });
                    if (source == previous.outputs.end()) {
                        throw std::invalid_argument(
                            "VXM feedback route has no matching "
                            "feedback-only source output");
                    }
                    ++routes_per_head[route.destination_head];
                }
                for (std::size_t head = 0;
                     head < VxmLane::kAluCount; head += depth) {
                    if (routes_per_head[head] == 0) continue;
                    if (routes_per_head[head] != phase.element_count) {
                        throw std::invalid_argument(
                            "VXM feedback route count does not match the "
                            "destination chain repeat count");
                    }
                    const auto feedback_head = std::find_if(
                        phase.instructions.begin(),
                        phase.instructions.end(),
                        [head](const auto& instruction) {
                            return instruction.stage == head
                                && instruction.lhs.kind
                                    == VxmScheduledOperandKind::Feedback;
                        });
                    if (feedback_head == phase.instructions.end()) {
                        throw std::invalid_argument(
                            "VXM feedback destination head is not "
                            "configured for Feedback");
                    }
                }
            }
            for (const auto& instruction : phase.instructions) {
                if (instruction.stage >= VxmLane::kAluCount) {
                    throw std::out_of_range(
                        "VXM Schedule instruction has invalid ALU stage");
                }
            }
            for (const auto& input : phase.inputs) {
                if (input.chain_head >= VxmLane::kAluCount
                    || input.chain_head % depth != 0) {
                    throw std::invalid_argument(
                        "VXM Schedule input is not bound to a chain head");
                }
            }
            for (const auto& output : phase.outputs) {
                if (output.chain_tail >= VxmLane::kAluCount
                    || output.chain_tail % depth != depth - 1) {
                    throw std::invalid_argument(
                        "VXM Schedule output is not bound to a chain tail");
                }
            }
        }
        if (total_cycles < previous_end) {
            throw std::invalid_argument(
                "VXM Schedule total cycle count is too small");
        }
    }
};

} // namespace ftlpu::vxm::compiler
