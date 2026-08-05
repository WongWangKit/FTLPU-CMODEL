#pragma once

#include "ftlpu/vxm/compiler/kernel_ir.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ftlpu::vxm::compiler {

enum class VxmFanoutAction {
    None,
    CarryChainInput,
    Recompute,
    Materialize,
};

struct VxmFanoutDecision {
    VxmFanoutAction action{VxmFanoutAction::None};
    std::size_t recompute_operations{0};
    bool directly_feeds_reduction{false};
};

// Kernel IR stores producer edges in each node. Lowering also needs the
// reverse consumer edges to detect linear chains, fan-out and Reduce barriers.
class VxmDependencyGraph {
public:
    explicit VxmDependencyGraph(const VxmKernel& kernel)
        : kernel_(kernel)
    {
        kernel_.validate();
        for (const auto& node : kernel_.nodes) {
            producers_.emplace(node.output, &node);
            consumers_.try_emplace(node.output);
        }
        for (const auto& node : kernel_.nodes) {
            for (const auto input : node.inputs) {
                consumers_.at(input).push_back(node.output);
            }
        }
    }

    const VxmIrNode& producer(ValueId value) const
    {
        const auto found = producers_.find(value);
        if (found == producers_.end()) {
            throw std::out_of_range(
                "VXM dependency value has no producer");
        }
        return *found->second;
    }

    const std::vector<ValueId>& consumers(ValueId value) const
    {
        const auto found = consumers_.find(value);
        if (found == consumers_.end()) {
            throw std::out_of_range(
                "VXM dependency value is unknown");
        }
        return found->second;
    }

    std::size_t use_count(ValueId value) const
    {
        return consumers(value).size();
    }

    bool is_kernel_input(ValueId value) const
    {
        return producer(value).opcode == VxmIrOpcode::Input;
    }

    bool is_kernel_output(ValueId value) const
    {
        for (const auto output : kernel_.outputs) {
            if (output == value) return true;
        }
        return false;
    }

    static bool is_reduction(const VxmIrNode& node)
    {
        return node.opcode == VxmIrOpcode::ReduceAdd
            || node.opcode == VxmIrOpcode::ReduceMax;
    }

    // First-version fan-out policy:
    // - a value already present at the chain head may travel as Original/Aux;
    // - a value feeding a Reduce may be recomputed when its backward slice
    //   contains at most recompute_limit operations;
    // - every other intermediate fan-out is materialized and read again.
    VxmFanoutDecision decide_fanout(
        ValueId value, const std::vector<ValueId>& chain_inputs,
        std::size_t recompute_limit = 2) const
    {
        if (use_count(value) <= 1) return {};
        const auto available = std::unordered_set<ValueId>{
            chain_inputs.begin(), chain_inputs.end()};
        if (available.contains(value)) {
            return {VxmFanoutAction::CarryChainInput, 0, false};
        }

        const auto feeds_reduction = std::any_of(
            consumers(value).begin(), consumers(value).end(),
            [this](ValueId consumer) {
                return is_reduction(producer(consumer));
            });
        if (feeds_reduction) {
            auto visited = std::unordered_set<ValueId>{};
            const auto cost =
                recompute_cost(value, available, visited, recompute_limit);
            if (cost <= recompute_limit) {
                return {
                    VxmFanoutAction::Recompute, cost, true};
            }
            return {
                VxmFanoutAction::Materialize, cost, true};
        }
        return {VxmFanoutAction::Materialize, 0, false};
    }

private:
    std::size_t recompute_cost(
        ValueId value, const std::unordered_set<ValueId>& available,
        std::unordered_set<ValueId>& visited,
        std::size_t limit) const
    {
        if (available.contains(value) || !visited.emplace(value).second) {
            return 0;
        }
        const auto& node = producer(value);
        if (node.opcode == VxmIrOpcode::Input
            || node.opcode == VxmIrOpcode::Constant) {
            return 0;
        }
        if (is_reduction(node)) {
            return limit + 1;
        }

        auto cost = node.opcode == VxmIrOpcode::Broadcast
            ? std::size_t{0} : std::size_t{1};
        for (const auto input : node.inputs) {
            cost += recompute_cost(input, available, visited, limit);
            if (cost > limit) return cost;
        }
        return cost;
    }

    const VxmKernel& kernel_;
    std::unordered_map<ValueId, const VxmIrNode*> producers_{};
    std::unordered_map<ValueId, std::vector<ValueId>> consumers_{};
};

} // namespace ftlpu::vxm::compiler
