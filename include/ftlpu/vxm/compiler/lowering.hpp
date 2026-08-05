#pragma once

#include "ftlpu/vxm/compiler/dependency_graph.hpp"
#include "ftlpu/vxm/compiler/feedback_planner.hpp"
#include "ftlpu/vxm/compiler/schedule_ir.hpp"
#include "ftlpu/vxm/special_alu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace ftlpu::vxm::compiler {

namespace detail {

inline std::size_t operation_latency(const VxmLaneOperation& operation)
{
    if (const auto* basic = std::get_if<VxmAluOpcode>(&operation)) {
        return VxmAlu::latency({*basic, VxmAluPrecision::Float32});
    }
    return VxmSpecialAlu::kPipelineLatency;
}

inline std::size_t chain_latency(const VxmScheduledPhase& phase)
{
    const auto depth = static_cast<std::size_t>(phase.chain_depth);
    auto latency = std::size_t{0};
    for (std::size_t stage = 0; stage < depth; ++stage) {
        const auto found = std::find_if(
            phase.instructions.begin(), phase.instructions.end(),
            [stage](const auto& instruction) {
                return instruction.stage == stage;
            });
        if (found == phase.instructions.end()) {
            throw std::invalid_argument(
                "VXM phase does not configure every active chain stage");
        }
        latency += operation_latency(found->operation);
    }
    return latency;
}

inline void finish_phase(VxmScheduledPhase& phase, std::size_t start_cycle)
{
    phase.data_start_cycle = start_cycle;
    phase.end_cycle =
        start_cycle + phase.element_count + chain_latency(phase) - 1;
}

inline VxmScheduledInstruction instruction(
    std::size_t stage, VxmLaneOperation operation,
    VxmScheduledOperand lhs, VxmScheduledOperand rhs,
    std::size_t repeat_count, std::optional<ValueId> output = std::nullopt)
{
    return {
        stage, std::move(operation), lhs, rhs,
        VxmAluPrecision::Float32, repeat_count,
        false, false, true, output, true};
}

inline void append_reduction_instructions(
    VxmScheduledPhase& phase, std::size_t stage, VxmAluOpcode opcode,
    ValueId output, std::size_t count)
{
    if (count == 0) {
        throw std::invalid_argument(
            "VXM reduction requires at least one value");
    }

    auto first = instruction(
        stage, opcode, VxmScheduledOperand::Previous(),
        VxmScheduledOperand::LocalScalar(output), 1,
        count == 1 ? std::optional<ValueId>{output} : std::nullopt);
    first.accumulator_reset = true;
    first.accumulator_write = true;
    first.accumulator_emit = count == 1;
    phase.instructions.push_back(first);

    if (count > 2) {
        auto middle = instruction(
            stage, opcode, VxmScheduledOperand::Previous(),
            VxmScheduledOperand::LocalScalar(output), count - 2);
        middle.accumulator_write = true;
        middle.accumulator_emit = false;
        phase.instructions.push_back(middle);
    }

    if (count > 1) {
        auto last = instruction(
            stage, opcode, VxmScheduledOperand::Previous(),
            VxmScheduledOperand::LocalScalar(output), 1, output);
        last.accumulator_write = true;
        last.accumulator_emit = true;
        phase.instructions.push_back(last);
    }
}

inline VxmPhaseInput stream_input(
    ValueId value, std::size_t chain_head, VxmInputPort port,
    const VxmKernel& kernel,
    VxmInputAccess access = VxmInputAccess::StreamEachCycle,
    std::size_t token_index = 0)
{
    return {
        value, chain_head, port, access,
        kernel.type_of(value), token_index};
}

inline VxmPhaseOutput phase_output(
    ValueId value, std::size_t tail, const VxmKernel& kernel,
    bool scalar, std::size_t token_index = 0)
{
    return {value, tail, kernel.type_of(value), scalar, true,
            token_index};
}

inline std::size_t elements_per_row(const VxmTensorType& type)
{
    if (type.rank() == 0) return 1;
    if (type.rank() == 1) return type.shape.front();
    return type.shape.back();
}

inline std::size_t tensor_row_count(const VxmTensorType& type)
{
    if (type.rank() < 2) return 1;
    auto rows = std::size_t{1};
    for (std::size_t index = 0; index + 1 < type.rank(); ++index) {
        rows *= type.shape[index];
    }
    return rows;
}

inline bool is_compute_opcode(VxmIrOpcode opcode)
{
    return opcode != VxmIrOpcode::Input
        && opcode != VxmIrOpcode::Constant
        && opcode != VxmIrOpcode::Broadcast;
}

inline ValueId unwrap_broadcast(
    const VxmDependencyGraph& graph, ValueId value)
{
    const auto& node = graph.producer(value);
    return node.opcode == VxmIrOpcode::Broadcast
        ? node.inputs.at(0) : value;
}

inline VxmLaneOperation lane_operation(VxmIrOpcode opcode)
{
    switch (opcode) {
    case VxmIrOpcode::Add:
    case VxmIrOpcode::ReduceAdd:
        return VxmAluOpcode::Add;
    case VxmIrOpcode::Subtract:
        return VxmAluOpcode::Subtract;
    case VxmIrOpcode::Multiply:
        return VxmAluOpcode::Multiply;
    case VxmIrOpcode::Negate:
        return VxmAluOpcode::Negate;
    case VxmIrOpcode::Maximum:
    case VxmIrOpcode::ReduceMax:
        return VxmAluOpcode::Max;
    case VxmIrOpcode::Exp:
        return VxmSpecialAluOpcode::Exp;
    case VxmIrOpcode::Reciprocal:
        return VxmSpecialAluOpcode::Reciprocal;
    case VxmIrOpcode::Rsqrt:
        return VxmSpecialAluOpcode::Rsqrt;
    case VxmIrOpcode::Input:
    case VxmIrOpcode::Constant:
    case VxmIrOpcode::Broadcast:
        break;
    }
    throw std::invalid_argument(
        "VXM non-compute IR node cannot be mapped to an ALU");
}

inline bool is_commutative(VxmIrOpcode opcode)
{
    return opcode == VxmIrOpcode::Add
        || opcode == VxmIrOpcode::Multiply
        || opcode == VxmIrOpcode::Maximum;
}

struct ChainOperation {
    VxmLaneOperation operation{VxmAluOpcode::Bypass};
    bool reduction{false};
    bool local_scalar_rhs{false};
};

struct ChainPlacement {
    VxmChainDepth depth{VxmChainDepth::Eight};
    std::vector<std::size_t> stages{};
    std::size_t bypass_count{0};
};

inline bool stage_supports(
    std::size_t stage, const ChainOperation& operation)
{
    const auto column = stage % 4;
    if ((operation.reduction || operation.local_scalar_rhs)
        && column != 1 && column != 3) {
        return false;
    }
    if (std::holds_alternative<VxmAluOpcode>(operation.operation)) {
        return true;
    }
    const auto special =
        std::get<VxmSpecialAluOpcode>(operation.operation);
    return (column == 1 && special == VxmSpecialAluOpcode::Exp)
        || (column == 3
            && (special == VxmSpecialAluOpcode::Reciprocal
                || special == VxmSpecialAluOpcode::Rsqrt));
}

inline std::optional<ChainPlacement> try_place_chain(
    const std::vector<ChainOperation>& operations, VxmChainDepth depth)
{
    const auto length = static_cast<std::size_t>(depth);
    auto placement = ChainPlacement{depth};
    auto cursor = std::size_t{0};
    for (const auto& operation : operations) {
        auto found = length;
        if (operation.reduction) {
            const auto tail = length - 1;
            if (tail >= cursor && stage_supports(tail, operation)) {
                found = tail;
            }
        } else {
            for (auto stage = cursor; stage < length; ++stage) {
                if (stage_supports(stage, operation)) {
                    found = stage;
                    break;
                }
            }
        }
        if (found == length) return std::nullopt;
        placement.stages.push_back(found);
        cursor = found + 1;
        if (operation.reduction && cursor != length) {
            return std::nullopt;
        }
    }
    placement.bypass_count = length - placement.stages.size();
    return placement;
}

inline ChainPlacement place_chain(
    const std::vector<ChainOperation>& operations)
{
    if (operations.empty()) {
        throw std::invalid_argument(
            "VXM lowering cannot place an empty dependency chain");
    }
    constexpr auto depths = std::array{
        VxmChainDepth::Two,
        VxmChainDepth::Four,
        VxmChainDepth::Eight,
    };
    for (const auto depth : depths) {
        if (auto placement = try_place_chain(operations, depth)) {
            return *placement;
        }
    }
    throw std::invalid_argument(
        "VXM dependency chain requires another VXM pass");
}

enum class PlannedOperationKind {
    IrNode,
    RsqrtEpsilonAdd,
};

struct PlannedOperation {
    PlannedOperationKind kind{PlannedOperationKind::IrNode};
    const VxmIrNode* node{nullptr};
    ChainOperation placement{};
};

class GenericLowering {
public:
    explicit GenericLowering(const VxmKernel& kernel)
        : kernel_(kernel), graph_(kernel), schedule_{kernel.name}
    {
        for (const auto& node : kernel_.nodes) {
            if (node.opcode == VxmIrOpcode::Input
                || node.opcode == VxmIrOpcode::Constant) {
                materialized_.insert(node.output);
            }
        }
    }

    VxmSchedule run()
    {
        for (const auto target : phase_targets()) {
            plan_value(target);
        }
        if (schedule_.phases.empty()) {
            throw std::invalid_argument(
                "VXM Kernel contains no executable output");
        }
        apply_feedback_boundaries();
        schedule_.total_cycles = schedule_.phases.back().end_cycle;
        schedule_.validate();
        return schedule_;
    }

private:
    std::vector<ValueId> phase_targets() const
    {
        auto targets = std::unordered_set<ValueId>{};
        auto recompute_boundaries = std::vector<ValueId>{};
        for (const auto input : kernel_.inputs) {
            recompute_boundaries.push_back(input);
        }
        for (const auto& node : kernel_.nodes) {
            if (VxmDependencyGraph::is_reduction(node)) {
                targets.insert(node.output);
                recompute_boundaries.push_back(node.output);
            }
        }

        for (const auto& node : kernel_.nodes) {
            if (!is_compute_opcode(node.opcode)) continue;
            for (const auto consumer : graph_.consumers(node.output)) {
                const auto& consumer_node = graph_.producer(consumer);
                if (consumer_node.opcode == VxmIrOpcode::Broadcast
                    && consumer_node.type.element_count()
                        > node.type.element_count()) {
                    targets.insert(node.output);
                }
            }
            const auto fanout = graph_.decide_fanout(
                node.output, recompute_boundaries);
            if (fanout.action == VxmFanoutAction::Materialize) {
                targets.insert(node.output);
            }
        }
        for (const auto output : kernel_.outputs) {
            targets.insert(unwrap_broadcast(graph_, output));
        }

        auto ordered = std::vector<ValueId>{};
        for (const auto& node : kernel_.nodes) {
            if (targets.contains(node.output)) {
                ordered.push_back(node.output);
            }
        }
        return ordered;
    }

    void plan_value(ValueId value)
    {
        value = unwrap_broadcast(graph_, value);
        if (materialized_.contains(value)) return;
        const auto& target = graph_.producer(value);
        if (!is_compute_opcode(target.opcode)) {
            materialized_.insert(value);
            return;
        }
        if (!planning_.insert(value).second) {
            throw std::logic_error(
                "VXM dependency graph unexpectedly contains a cycle");
        }

        auto chain = extract_chain(value);
        auto chain_values = std::unordered_set<ValueId>{};
        for (const auto* node : chain) {
            chain_values.insert(node->output);
        }
        for (const auto* node : chain) {
            for (const auto input : node->inputs) {
                const auto source = unwrap_broadcast(graph_, input);
                const auto& producer = graph_.producer(source);
                if (is_compute_opcode(producer.opcode)
                    && !materialized_.contains(source)
                    && !chain_values.contains(source)) {
                    plan_value(source);
                }
            }
        }

        chain = extract_chain(value);
        emit_chain(chain);
        planning_.erase(value);
    }

    std::vector<const VxmIrNode*> extract_chain(ValueId target) const
    {
        auto reverse = std::vector<const VxmIrNode*>{};
        auto seen = std::unordered_set<ValueId>{};
        auto current = target;
        while (!materialized_.contains(current)) {
            const auto& node = graph_.producer(current);
            if (!is_compute_opcode(node.opcode)
                || !seen.insert(current).second) {
                break;
            }
            reverse.push_back(&node);

            std::optional<ValueId> main{};
            for (const auto input : node.inputs) {
                const auto source = unwrap_broadcast(graph_, input);
                const auto& producer = graph_.producer(source);
                if (!materialized_.contains(source)
                    && is_compute_opcode(producer.opcode)) {
                    main = source;
                    break;
                }
            }
            if (!main) break;
            current = *main;
        }
        std::reverse(reverse.begin(), reverse.end());
        return reverse;
    }

    std::vector<PlannedOperation> expand_operations(
        const std::vector<const VxmIrNode*>& nodes,
        std::size_t phase_count) const
    {
        auto result = std::vector<PlannedOperation>{};
        auto earlier = std::unordered_set<ValueId>{};
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const auto* node = nodes[index];
            if (node->opcode == VxmIrOpcode::Rsqrt) {
                result.push_back({
                    PlannedOperationKind::RsqrtEpsilonAdd,
                    node,
                    {VxmAluOpcode::Add, false, false},
                });
            }

            auto local_scalar_rhs = false;
            if (node->inputs.size() == 2 && phase_count > 1 && index > 0) {
                for (const auto input : node->inputs) {
                    const auto source = unwrap_broadcast(graph_, input);
                    const auto& producer = graph_.producer(source);
                    if (!earlier.contains(source)
                        && producer.opcode != VxmIrOpcode::Constant
                        && elements_per_row(producer.type) == 1) {
                        local_scalar_rhs = true;
                    }
                }
            }
            const auto reduction =
                VxmDependencyGraph::is_reduction(*node);
            result.push_back({
                PlannedOperationKind::IrNode,
                node,
                {lane_operation(node->opcode),
                 reduction, reduction || local_scalar_rhs},
            });
            earlier.insert(node->output);
        }
        return result;
    }

    struct PhaseShape {
        std::size_t rows{1};
        std::size_t elements_per_chain{1};
    };

    PhaseShape phase_shape(
        const std::vector<const VxmIrNode*>& nodes) const
    {
        const auto& last = *nodes.back();
        if (!VxmDependencyGraph::is_reduction(last)) {
            return {
                tensor_row_count(last.type),
                elements_per_row(last.type)};
        }
        const auto& input_type =
            kernel_.type_of(last.inputs.at(0));
        const auto axis = *last.axis;
        if (axis + 1 != input_type.rank()) {
            throw std::invalid_argument(
                "VXM row-parallel reduction currently requires the last axis");
        }
        auto outer = std::size_t{1};
        for (std::size_t index = 0; index < input_type.rank(); ++index) {
            if (index != axis) outer *= input_type.shape[index];
        }
        return {outer, input_type.shape[axis]};
    }

    void emit_chain(const std::vector<const VxmIrNode*>& nodes)
    {
        if (nodes.empty()) return;
        auto offset = std::size_t{0};
        while (offset < nodes.size()) {
            std::size_t end = nodes.size();
            std::optional<ChainPlacement> selected{};
            std::vector<PlannedOperation> selected_operations{};
            while (end > offset) {
                auto segment = std::vector<const VxmIrNode*>{
                    nodes.begin() + static_cast<std::ptrdiff_t>(offset),
                    nodes.begin() + static_cast<std::ptrdiff_t>(end)};
                const auto shape = phase_shape(segment);
                auto operations = expand_operations(
                    segment, shape.elements_per_chain);
                auto constraints = std::vector<ChainOperation>{};
                for (const auto& operation : operations) {
                    constraints.push_back(operation.placement);
                }
                try {
                    selected = place_chain(constraints);
                    selected_operations = std::move(operations);
                    break;
                } catch (const std::invalid_argument&) {
                    --end;
                }
            }
            if (!selected || end == offset) {
                throw std::invalid_argument(
                    "one VXM IR operation cannot be placed in 2/4/8 mode");
            }

            auto segment = std::vector<const VxmIrNode*>{
                nodes.begin() + static_cast<std::ptrdiff_t>(offset),
                nodes.begin() + static_cast<std::ptrdiff_t>(end)};
            emit_phase(segment, selected_operations, *selected);
            materialized_.insert(segment.back()->output);
            offset = end;
        }
    }

    struct Bindings {
        ValueId primary{kInvalidValue};
        ValueId auxiliary{kInvalidValue};
        std::vector<std::pair<ValueId, std::size_t>> local_scalars{};
    };

    Bindings bind_inputs(
        const std::vector<const VxmIrNode*>& nodes,
        const std::vector<PlannedOperation>& operations,
        const ChainPlacement& placement, std::size_t count) const
    {
        auto bindings = Bindings{};
        const auto first = nodes.front();
        auto assign_head = [&bindings](ValueId value, bool prefer_rhs) {
            if (bindings.primary == kInvalidValue && !prefer_rhs) {
                bindings.primary = value;
            } else if (bindings.auxiliary == kInvalidValue) {
                bindings.auxiliary = value;
            } else if (bindings.primary == kInvalidValue) {
                bindings.primary = value;
            } else if (bindings.primary != value
                       && bindings.auxiliary != value) {
                throw std::invalid_argument(
                    "VXM chain requires more than two head Stream values");
            }
        };

        for (std::size_t index = 0; index < first->inputs.size(); ++index) {
            const auto value = unwrap_broadcast(graph_, first->inputs[index]);
            if (graph_.producer(value).opcode != VxmIrOpcode::Constant) {
                assign_head(value, index == 1);
            }
        }

        auto operation_index = std::size_t{0};
        for (const auto* node : nodes) {
            while (operations[operation_index].kind
                   == PlannedOperationKind::RsqrtEpsilonAdd) {
                ++operation_index;
            }
            const auto stage = placement.stages[operation_index];
            for (const auto input : node->inputs) {
                const auto value = unwrap_broadcast(graph_, input);
                const auto& producer = graph_.producer(value);
                if (producer.opcode == VxmIrOpcode::Constant) continue;
                const auto produced_in_chain = std::any_of(
                    nodes.begin(), nodes.end(),
                    [value](const auto* candidate) {
                        return candidate->output == value;
                    });
                if (produced_in_chain) continue;
                if (count > 1 && elements_per_row(producer.type) == 1
                    && node != first && producer.opcode != VxmIrOpcode::Input) {
                    const auto binding = std::pair{value, stage};
                    if (std::find(
                            bindings.local_scalars.begin(),
                            bindings.local_scalars.end(),
                            binding) == bindings.local_scalars.end()) {
                        bindings.local_scalars.push_back(binding);
                    }
                } else if (bindings.primary != value
                           && bindings.auxiliary != value) {
                    assign_head(value, true);
                }
            }
            ++operation_index;
        }
        if (bindings.primary == kInvalidValue) {
            throw std::invalid_argument(
                "VXM chain has no value for its fixed head input");
        }
        return bindings;
    }

    VxmScheduledOperand external_operand(
        ValueId value, std::size_t stage, bool lhs,
        const Bindings& bindings, bool first_node) const
    {
        value = unwrap_broadcast(graph_, value);
        const auto& producer = graph_.producer(value);
        if (producer.opcode == VxmIrOpcode::Constant) {
            return VxmScheduledOperand::Immediate(*producer.constant);
        }
        for (const auto& [scalar, scalar_stage] : bindings.local_scalars) {
            if (scalar == value && scalar_stage == stage) {
                return VxmScheduledOperand::LocalScalar(value);
            }
        }
        if (stage == 0) return VxmScheduledOperand::Stream(value);
        if (first_node && lhs && value == bindings.primary) {
            return VxmScheduledOperand::Previous();
        }
        if (value == bindings.primary) {
            return VxmScheduledOperand::Original();
        }
        if (value == bindings.auxiliary) {
            return VxmScheduledOperand::Auxiliary();
        }
        throw std::invalid_argument(
            "VXM internal operand is not Previous/Original/Aux/Immediate/"
            "LocalScalar");
    }

    void emit_phase(
        const std::vector<const VxmIrNode*>& nodes,
        const std::vector<PlannedOperation>& operations,
        const ChainPlacement& placement)
    {
        const auto shape = phase_shape(nodes);
        const auto count = shape.elements_per_chain;
        const auto bindings =
            bind_inputs(nodes, operations, placement, count);
        const auto depth = static_cast<std::size_t>(placement.depth);
        const auto tail = depth - 1;
        const auto output = nodes.back()->output;
        const auto scalar_output =
            VxmDependencyGraph::is_reduction(*nodes.back())
            || elements_per_row(kernel_.type_of(output)) == 1;
        const auto access_for = [this, count](ValueId value) {
            return count > 1
                    && elements_per_row(kernel_.type_of(value)) == 1
                ? VxmInputAccess::HoldForPhase
                : VxmInputAccess::StreamEachCycle;
        };

        auto operation_at_stage =
            std::vector<std::optional<std::size_t>>(depth);
        for (std::size_t index = 0; index < operations.size(); ++index) {
            operation_at_stage[placement.stages[index]] = index;
        }

        const auto chain_capacity = VxmLane::kAluCount / depth;
        for (std::size_t token_begin = 0;
             token_begin < shape.rows;
             token_begin += chain_capacity) {
            const auto active_chains = std::min(
                chain_capacity, shape.rows - token_begin);
            auto phase = VxmScheduledPhase{};
            phase.id = schedule_.phases.size();
            phase.name = "phase_" + std::to_string(phase.id)
                + "_v" + std::to_string(output)
                + "_rows" + std::to_string(token_begin)
                + '_' + std::to_string(token_begin + active_chains);
            phase.chain_depth = placement.depth;
            phase.element_count = count;
            phase.token_begin = token_begin;
            phase.parallel_chain_count = active_chains;

            for (std::size_t chain = 0;
                 chain < active_chains; ++chain) {
                const auto base = chain * depth;
                const auto token_index = token_begin + chain;
                phase.inputs.push_back(stream_input(
                    bindings.primary, base, VxmInputPort::Lhs,
                    kernel_, access_for(bindings.primary), token_index));
                if (bindings.auxiliary != kInvalidValue) {
                    phase.inputs.push_back(stream_input(
                        bindings.auxiliary, base, VxmInputPort::Rhs,
                        kernel_, access_for(bindings.auxiliary),
                        token_index));
                }
                for (const auto& [value, stage] :
                     bindings.local_scalars) {
                    phase.local_scalars.push_back({
                        value, base + stage, kernel_.type_of(value),
                        token_index});
                }
                phase.outputs.push_back(phase_output(
                    output, base + tail, kernel_, scalar_output,
                    token_index));

                const VxmIrNode* previous_node = nullptr;
                for (std::size_t stage = 0; stage < depth; ++stage) {
                    const auto physical_stage = base + stage;
                    if (!operation_at_stage[stage]) {
                        const auto lhs = stage == 0
                            ? VxmScheduledOperand::Stream(bindings.primary)
                            : VxmScheduledOperand::Previous();
                        const auto rhs = stage == 0
                                && bindings.auxiliary != kInvalidValue
                            ? VxmScheduledOperand::Stream(bindings.auxiliary)
                            : VxmScheduledOperand::Immediate(0.0f);
                        phase.instructions.push_back(instruction(
                            physical_stage, VxmAluOpcode::Bypass,
                            lhs, rhs, count,
                            stage == tail
                                ? std::optional<ValueId>{output}
                                : std::nullopt));
                        continue;
                    }

                    const auto operation_index = *operation_at_stage[stage];
                    const auto& planned = operations[operation_index];
                    if (planned.kind
                        == PlannedOperationKind::RsqrtEpsilonAdd) {
                        phase.instructions.push_back(instruction(
                            physical_stage, VxmAluOpcode::Add,
                            VxmScheduledOperand::Previous(),
                            VxmScheduledOperand::Immediate(
                                *planned.node->epsilon),
                            count));
                        previous_node = nullptr;
                        continue;
                    }

                    const auto& node = *planned.node;
                    if (VxmDependencyGraph::is_reduction(node)) {
                        append_reduction_instructions(
                            phase, physical_stage,
                            node.opcode == VxmIrOpcode::ReduceAdd
                                ? VxmAluOpcode::Add
                                : VxmAluOpcode::Max,
                            node.output, count);
                        previous_node = &node;
                        continue;
                    }

                    auto lhs = VxmScheduledOperand::Immediate(0.0f);
                    auto rhs = VxmScheduledOperand::Immediate(0.0f);
                    if (node.inputs.size() == 1) {
                        const auto input =
                            unwrap_broadcast(graph_, node.inputs[0]);
                        if ((previous_node
                             && previous_node->output == input)
                            || (operation_index > 0
                                && operations[operation_index - 1].kind
                                    == PlannedOperationKind::RsqrtEpsilonAdd)) {
                            lhs = VxmScheduledOperand::Previous();
                        } else {
                            lhs = external_operand(
                                input, stage, true, bindings,
                                &node == nodes.front());
                        }
                        if (stage == 0
                            && bindings.auxiliary != kInvalidValue) {
                            rhs = VxmScheduledOperand::Stream(
                                bindings.auxiliary);
                        }
                    } else {
                        const auto lhs_value =
                            unwrap_broadcast(graph_, node.inputs[0]);
                        const auto rhs_value =
                            unwrap_broadcast(graph_, node.inputs[1]);
                        const auto lhs_previous = previous_node
                            && previous_node->output == lhs_value;
                        const auto rhs_previous = previous_node
                            && previous_node->output == rhs_value;
                        if (lhs_previous) {
                            lhs = VxmScheduledOperand::Previous();
                            rhs = external_operand(
                                rhs_value, stage, false, bindings,
                                &node == nodes.front());
                        } else if (rhs_previous
                                   && is_commutative(node.opcode)) {
                            lhs = VxmScheduledOperand::Previous();
                            rhs = external_operand(
                                lhs_value, stage, false, bindings,
                                &node == nodes.front());
                        } else {
                            lhs = external_operand(
                                lhs_value, stage, true, bindings,
                                &node == nodes.front());
                            rhs = external_operand(
                                rhs_value, stage, false, bindings,
                                &node == nodes.front());
                        }
                    }

                    phase.instructions.push_back(instruction(
                        physical_stage, lane_operation(node.opcode),
                        lhs, rhs, count,
                        stage == tail
                            ? std::optional<ValueId>{output}
                            : std::nullopt));
                    previous_node = &node;
                }
            }

            finish_phase(phase, next_cycle_);
            next_cycle_ = phase.end_cycle;
            schedule_.phases.push_back(std::move(phase));
        }
    }

    void apply_feedback_boundaries()
    {
        for (std::size_t index = 1;
             index < schedule_.phases.size(); ++index) {
            auto& previous = schedule_.phases[index - 1];
            auto& current = schedule_.phases[index];
            if (current.element_count != 1
                || previous.outputs.size() != 1
                || !previous.outputs.front().scalar) {
                continue;
            }

            const auto value = previous.outputs.front().value;
            if (graph_.use_count(value) != 1) continue;

            const auto input = std::find_if(
                current.inputs.begin(), current.inputs.end(),
                [value](const auto& candidate) {
                    return candidate.value == value
                        && candidate.chain_head == 0
                        && candidate.port == VxmInputPort::Lhs;
                });
            const auto head = std::find_if(
                current.instructions.begin(), current.instructions.end(),
                [value](const auto& candidate) {
                    return candidate.stage == 0
                        && candidate.lhs.kind
                            == VxmScheduledOperandKind::StreamValue
                        && candidate.lhs.value == value;
                });
            if (input == current.inputs.end()
                || head == current.instructions.end()) {
                continue;
            }

            const auto old_tail = previous.outputs.front().chain_tail;
            if (!VxmLane::has_fixed_feedback_path(old_tail, 0)) {
                continue;
            }

            apply_feedback_routes(
                previous, current,
                {{value, old_tail, 0, false}});
        }
    }

    const VxmKernel& kernel_;
    VxmDependencyGraph graph_;
    VxmSchedule schedule_{};
    std::unordered_set<ValueId> materialized_{};
    std::unordered_set<ValueId> planning_{};
    std::size_t next_cycle_{1};
};

} // namespace detail

inline VxmSchedule lower_kernel(const VxmKernel& kernel)
{
    kernel.validate();
    return detail::GenericLowering{kernel}.run();
}

} // namespace ftlpu::vxm::compiler
