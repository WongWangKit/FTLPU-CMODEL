#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ftlpu::vxm::compiler {

using ValueId = std::uint32_t;
inline constexpr ValueId kInvalidValue = static_cast<ValueId>(-1);

// Kernel IR operations describe mathematical data flow. Input, Constant and
// Broadcast are compiler concepts, not VXM ALU opcodes.
enum class VxmIrOpcode {
    Input,
    Constant,
    Add,
    Subtract,
    Multiply,
    Negate,
    Maximum,
    Exp,
    Reciprocal,
    Rsqrt,
    ReduceAdd,
    ReduceMax,
    Broadcast,
};

// Kernel IR describes the VXM-visible tensor interface. The current hardware
// transfers FP16 values; wider accumulators and Special ALU intermediates are
// fixed implementation details and therefore do not appear in Kernel IR.
enum class VxmIrElementType {
    Float16,
};

struct VxmTensorType {
    std::vector<std::size_t> shape{};
    VxmIrElementType element_type{VxmIrElementType::Float16};

    std::size_t rank() const { return shape.size(); }

    std::size_t element_count() const
    {
        return std::accumulate(
            shape.begin(), shape.end(), std::size_t{1},
            std::multiplies<std::size_t>{});
    }

    bool scalar() const { return shape.empty(); }

    friend bool operator==(const VxmTensorType&, const VxmTensorType&) = default;
};

struct VxmIrNode {
    ValueId output{kInvalidValue};
    VxmIrOpcode opcode{VxmIrOpcode::Input};
    std::vector<ValueId> inputs{};
    VxmTensorType type{};

    // axis is normalized to a non-negative index by KernelBuilder.
    std::optional<std::size_t> axis{};
    std::optional<float> constant{};
    // Rsqrt adds epsilon in its widened internal computation. It is an FP32
    // configuration value, not an FP16 Tensor operand.
    std::optional<float> epsilon{};
    std::string name{};
};

struct VxmKernel {
    std::string name{};
    std::vector<ValueId> inputs{};
    std::vector<ValueId> outputs{};
    std::vector<VxmIrNode> nodes{};

    const VxmIrNode& producer(ValueId value) const
    {
        const auto found = std::find_if(
            nodes.begin(), nodes.end(),
            [value](const auto& node) { return node.output == value; });
        if (found == nodes.end()) {
            throw std::out_of_range("VXM Kernel IR ValueId has no producer");
        }
        return *found;
    }

    const VxmIrNode& named(const std::string& node_name) const
    {
        const auto found = std::find_if(
            nodes.begin(), nodes.end(),
            [&node_name](const auto& node) { return node.name == node_name; });
        if (found == nodes.end()) {
            throw std::out_of_range(
                "VXM Kernel IR has no node named " + node_name);
        }
        return *found;
    }

    const VxmTensorType& type_of(ValueId value) const
    {
        return producer(value).type;
    }

    // Remove nodes that cannot reach a Kernel output. ValueIds remain stable.
    void eliminate_dead_code()
    {
        auto definitions = std::unordered_map<ValueId, std::size_t>{};
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            if (!definitions.emplace(nodes[index].output, index).second) {
                throw std::invalid_argument(
                    "VXM Kernel IR contains a duplicate ValueId");
            }
        }

        auto live = std::unordered_set<ValueId>{};
        auto worklist = outputs;
        while (!worklist.empty()) {
            const auto value = worklist.back();
            worklist.pop_back();
            if (!live.emplace(value).second) continue;
            const auto found = definitions.find(value);
            if (found == definitions.end()) {
                throw std::invalid_argument(
                    "VXM Kernel output or operand has no producer");
            }
            for (const auto input : nodes[found->second].inputs) {
                worklist.push_back(input);
            }
        }

        nodes.erase(
            std::remove_if(
                nodes.begin(), nodes.end(),
                [&live](const auto& node) {
                    return !live.contains(node.output);
                }),
            nodes.end());
        inputs.erase(
            std::remove_if(
                inputs.begin(), inputs.end(),
                [&live](ValueId input) { return !live.contains(input); }),
            inputs.end());
    }

    void validate() const
    {
        if (name.empty()) {
            throw std::invalid_argument("VXM Kernel IR requires a name");
        }
        if (outputs.empty()) {
            throw std::invalid_argument("VXM Kernel IR requires an output");
        }

        auto definitions = std::unordered_map<ValueId, std::size_t>{};
        auto names = std::unordered_map<std::string, ValueId>{};
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const auto& node = nodes[index];
            const auto node_label = "VXM Kernel IR node v"
                + std::to_string(node.output);
            if (node.output == kInvalidValue
                || !definitions.emplace(node.output, index).second) {
                throw std::invalid_argument(
                    "VXM Kernel IR contains an invalid or duplicate ValueId");
            }
            if (!node.name.empty()
                && !names.emplace(node.name, node.output).second) {
                throw std::invalid_argument(
                    "VXM Kernel IR contains a duplicate node name");
            }
            for (const auto dimension : node.type.shape) {
                if (dimension == 0) {
                    throw std::invalid_argument(
                        node_label + " contains a zero-sized dimension");
                }
            }

            auto input_nodes = std::vector<const VxmIrNode*>{};
            input_nodes.reserve(node.inputs.size());
            for (const auto input : node.inputs) {
                const auto producer = definitions.find(input);
                if (producer == definitions.end() || producer->second >= index) {
                    throw std::invalid_argument(
                        "VXM Kernel IR is not in topological order");
                }
                input_nodes.push_back(&nodes[producer->second]);
            }

            const auto expected_operands = [&node]() -> std::size_t {
                switch (node.opcode) {
                case VxmIrOpcode::Input:
                case VxmIrOpcode::Constant:
                    return 0;
                case VxmIrOpcode::Negate:
                case VxmIrOpcode::Exp:
                case VxmIrOpcode::Reciprocal:
                case VxmIrOpcode::Rsqrt:
                case VxmIrOpcode::ReduceAdd:
                case VxmIrOpcode::ReduceMax:
                case VxmIrOpcode::Broadcast:
                    return 1;
                case VxmIrOpcode::Add:
                case VxmIrOpcode::Subtract:
                case VxmIrOpcode::Multiply:
                case VxmIrOpcode::Maximum:
                    return 2;
                }
                throw std::logic_error("unknown VXM Kernel IR opcode");
            }();
            if (node.inputs.size() != expected_operands) {
                throw std::invalid_argument(
                    node_label + " has an invalid operand count");
            }

            const auto is_reduction =
                node.opcode == VxmIrOpcode::ReduceAdd
                || node.opcode == VxmIrOpcode::ReduceMax;
            if (node.axis.has_value() != is_reduction) {
                throw std::invalid_argument(
                    node_label + " has invalid reduction-axis metadata");
            }
            if (node.constant.has_value()
                != (node.opcode == VxmIrOpcode::Constant)) {
                throw std::invalid_argument(
                    node_label + " has invalid constant metadata");
            }
            if (node.epsilon.has_value()
                != (node.opcode == VxmIrOpcode::Rsqrt)) {
                throw std::invalid_argument(
                    node_label + " has invalid Rsqrt epsilon metadata");
            }
            if (node.epsilon
                && (!std::isfinite(*node.epsilon) || *node.epsilon < 0.0f)) {
                throw std::invalid_argument(
                    node_label + " requires a finite non-negative epsilon");
            }

            switch (node.opcode) {
            case VxmIrOpcode::Input:
                break;
            case VxmIrOpcode::Constant:
                if (!node.type.scalar()) {
                    throw std::invalid_argument(
                        node_label + " must be a scalar constant");
                }
                break;
            case VxmIrOpcode::Negate:
            case VxmIrOpcode::Exp:
            case VxmIrOpcode::Reciprocal:
            case VxmIrOpcode::Rsqrt:
                if (node.type != input_nodes[0]->type) {
                    throw std::invalid_argument(
                        node_label
                        + " unary result type does not match its operand");
                }
                break;
            case VxmIrOpcode::Add:
            case VxmIrOpcode::Subtract:
            case VxmIrOpcode::Multiply:
            case VxmIrOpcode::Maximum:
                if (input_nodes[0]->type != input_nodes[1]->type
                    || node.type != input_nodes[0]->type) {
                    throw std::invalid_argument(
                        node_label
                        + " binary operand/result types do not match");
                }
                break;
            case VxmIrOpcode::ReduceAdd:
            case VxmIrOpcode::ReduceMax: {
                const auto& input_type = input_nodes[0]->type;
                if (input_type.scalar() || *node.axis >= input_type.rank()) {
                    throw std::invalid_argument(
                        node_label + " has an invalid reduction axis");
                }
                auto expected_type = input_type;
                expected_type.shape[*node.axis] = 1;
                if (node.type != expected_type) {
                    throw std::invalid_argument(
                        node_label + " has an invalid reduction result type");
                }
                break;
            }
            case VxmIrOpcode::Broadcast:
                if (node.type.element_type
                        != input_nodes[0]->type.element_type
                    || !broadcast_compatible(
                        input_nodes[0]->type.shape, node.type.shape)) {
                    throw std::invalid_argument(
                        node_label + " has an invalid broadcast result type");
                }
                break;
            }
        }

        auto declared_inputs = std::unordered_map<ValueId, bool>{};
        for (const auto input : inputs) {
            if (!declared_inputs.emplace(input, true).second) {
                throw std::invalid_argument(
                    "VXM Kernel IR contains a duplicate kernel input");
            }
            if (producer(input).opcode != VxmIrOpcode::Input) {
                throw std::invalid_argument(
                    "VXM Kernel input does not refer to an Input node");
            }
        }
        for (const auto& node : nodes) {
            if (node.opcode == VxmIrOpcode::Input
                && !declared_inputs.contains(node.output)) {
                throw std::invalid_argument(
                    "VXM Kernel IR contains an undeclared Input node");
            }
        }

        auto declared_outputs = std::unordered_map<ValueId, bool>{};
        for (const auto output : outputs) {
            (void)producer(output);
            if (!declared_outputs.emplace(output, true).second) {
                throw std::invalid_argument(
                    "VXM Kernel IR contains a duplicate kernel output");
            }
        }
    }

private:
    static bool broadcast_compatible(
        const std::vector<std::size_t>& source,
        const std::vector<std::size_t>& target)
    {
        if (source.size() > target.size()) return false;
        const auto offset = target.size() - source.size();
        for (std::size_t index = 0; index < source.size(); ++index) {
            if (source[index] != 1
                && source[index] != target[offset + index]) {
                return false;
            }
        }
        return true;
    }
};

} // namespace ftlpu::vxm::compiler
