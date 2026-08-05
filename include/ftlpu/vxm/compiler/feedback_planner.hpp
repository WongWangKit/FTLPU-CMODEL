#pragma once

#include "ftlpu/vxm/compiler/schedule_ir.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ftlpu::vxm::compiler {

inline void apply_feedback_routes(
    VxmScheduledPhase& previous,
    VxmScheduledPhase& current,
    std::vector<VxmFeedbackRoute> routes)
{
    if (routes.empty()) {
        throw std::invalid_argument(
            "VXM feedback planning requires at least one route");
    }

    auto routed_values = std::unordered_set<ValueId>{};
    auto destination_heads = std::unordered_set<std::size_t>{};
    for (const auto& route : routes) {
        if (!VxmLane::has_fixed_feedback_path(
                route.source_tail, route.destination_head)) {
            throw std::invalid_argument(
                "VXM feedback route is not physically implemented");
        }
        routed_values.insert(route.value);
        destination_heads.insert(route.destination_head);

        auto source_found = false;
        for (auto& output : previous.outputs) {
            if (output.value == route.value
                && output.chain_tail == route.source_tail) {
                output.stream_write = false;
                source_found = true;
            }
        }
        if (!source_found) {
            throw std::invalid_argument(
                "VXM feedback route does not match a previous phase output");
        }
        for (auto& instruction : previous.instructions) {
            if (instruction.stage == route.source_tail
                && instruction.output
                && *instruction.output == route.value) {
                instruction.stream_output = false;
            }
        }
    }

    for (const auto head : destination_heads) {
        auto configured = false;
        const auto first_route = std::find_if(
            routes.begin(), routes.end(),
            [head](const auto& route) {
                return route.destination_head == head;
            });
        for (auto& instruction : current.instructions) {
            if (instruction.stage != head) continue;
            instruction.lhs =
                VxmScheduledOperand::Feedback(first_route->value);
            configured = true;
        }
        if (!configured) {
            throw std::invalid_argument(
                "VXM feedback destination has no chain-head instruction");
        }
    }

    current.inputs.erase(
        std::remove_if(
            current.inputs.begin(), current.inputs.end(),
            [&routed_values, &destination_heads](const auto& input) {
                return routed_values.contains(input.value)
                    && destination_heads.contains(input.chain_head);
            }),
        current.inputs.end());
    current.feedback_from_previous = true;
    current.feedback_routes = std::move(routes);

    // One cycle expands the compact instruction in the local decoder. The
    // extra lead cycle leaves the decoded control in Next Config before the
    // old chain tails retire.
    current.config_lead_cycles = 2;
}

inline void apply_multi_chain_feedback_merge(
    VxmScheduledPhase& previous,
    VxmScheduledPhase& current)
{
    const auto old_depth =
        static_cast<std::size_t>(previous.chain_depth);
    const auto new_depth =
        static_cast<std::size_t>(current.chain_depth);
    if (old_depth >= new_depth || new_depth % old_depth != 0) {
        throw std::invalid_argument(
            "VXM multi-chain feedback requires a 2->4, 4->8, or 2->8 "
            "chain-depth merge");
    }

    const auto results_per_new_chain = new_depth / old_depth;
    if (current.element_count != results_per_new_chain) {
        throw std::invalid_argument(
            "VXM merge phase element_count must equal the number of old "
            "chain results consumed by each new chain");
    }

    auto routes = std::vector<VxmFeedbackRoute>{};
    auto route_count =
        std::unordered_map<std::size_t, std::size_t>{};
    for (const auto& candidate : previous.outputs) {
        if (!candidate.stream_write
            || candidate.chain_tail % old_depth != old_depth - 1) {
            continue;
        }
        const auto* output = &candidate;
        const auto tail = output->chain_tail;
        const auto destination_head =
            (tail / new_depth) * new_depth;
        const auto ordinal = route_count[destination_head]++;
        routes.push_back({
            output->value,
            tail,
            destination_head,
            ordinal != 0,
        });
    }

    apply_feedback_routes(previous, current, std::move(routes));
}

inline bool can_apply_multi_chain_feedback_merge(
    const VxmScheduledPhase& previous,
    const VxmScheduledPhase& current)
{
    if (current.feedback_from_previous) return false;
    const auto old_depth =
        static_cast<std::size_t>(previous.chain_depth);
    const auto new_depth =
        static_cast<std::size_t>(current.chain_depth);
    if (old_depth >= new_depth || new_depth % old_depth != 0
        || current.element_count != new_depth / old_depth) {
        return false;
    }
    const auto results_per_new_chain = new_depth / old_depth;

    auto active_outputs = std::size_t{0};
    auto results_per_head =
        std::unordered_map<std::size_t, std::size_t>{};
    for (const auto& output : previous.outputs) {
        if (!output.stream_write
            || output.chain_tail % old_depth != old_depth - 1) {
            continue;
        }
        ++active_outputs;
        const auto head =
            (output.chain_tail / new_depth) * new_depth;
        ++results_per_head[head];
        const auto input = std::find_if(
            current.inputs.begin(), current.inputs.end(),
            [&output, head](const auto& candidate) {
                return candidate.value == output.value
                    && candidate.chain_head == head
                    && candidate.port == VxmInputPort::Lhs;
            });
        if (input == current.inputs.end()) return false;
        const auto head_instruction = std::find_if(
            current.instructions.begin(), current.instructions.end(),
            [head](const auto& instruction) {
                return instruction.stage == head;
            });
        if (head_instruction == current.instructions.end()) return false;
    }
    if (active_outputs == 0) return false;
    return std::all_of(
        results_per_head.begin(), results_per_head.end(),
        [results_per_new_chain](const auto& entry) {
            return entry.second == results_per_new_chain;
        });
}

inline void plan_multi_chain_feedback_merges(VxmSchedule& schedule)
{
    for (std::size_t index = 1;
         index < schedule.phases.size(); ++index) {
        auto& previous = schedule.phases[index - 1];
        auto& current = schedule.phases[index];
        if (can_apply_multi_chain_feedback_merge(previous, current)) {
            apply_multi_chain_feedback_merge(previous, current);
        }
    }
}

} // namespace ftlpu::vxm::compiler
