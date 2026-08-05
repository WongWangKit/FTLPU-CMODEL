#pragma once

#include "ftlpu/vxm/compiler/kernel_ir.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::vxm::compiler {

class VxmKernelBuilder;

struct VxmTensor {
    ValueId id{kInvalidValue};
    VxmKernelBuilder* builder{nullptr};
};

class VxmKernelBuilder {
public:
    explicit VxmKernelBuilder(std::string kernel_name)
    {
        kernel_.name = std::move(kernel_name);
    }

    VxmTensor input(std::string name, VxmTensorType type)
    {
        const auto tensor = append(
            VxmIrOpcode::Input, {}, std::move(type), std::move(name));
        kernel_.inputs.push_back(tensor.id);
        return tensor;
    }

    VxmTensor constant(float value, std::string name = {})
    {
        auto tensor = append(
            VxmIrOpcode::Constant, {}, VxmTensorType{}, std::move(name));
        node(tensor.id).constant = value;
        return tensor;
    }

    VxmTensor add(VxmTensor lhs, VxmTensor rhs, std::string name = {})
    {
        return binary(VxmIrOpcode::Add, lhs, rhs, std::move(name));
    }

    VxmTensor subtract(VxmTensor lhs, VxmTensor rhs, std::string name = {})
    {
        return binary(VxmIrOpcode::Subtract, lhs, rhs, std::move(name));
    }

    VxmTensor multiply(VxmTensor lhs, VxmTensor rhs, std::string name = {})
    {
        return binary(VxmIrOpcode::Multiply, lhs, rhs, std::move(name));
    }

    // Divide is convenient Kernel syntax rather than a Kernel IR opcode.
    // The target VXM implements lhs / rhs as lhs * reciprocal(rhs).
    VxmTensor divide(VxmTensor lhs, VxmTensor rhs, std::string name = {})
    {
        check_tensor(lhs);
        check_tensor(rhs);
        const auto inverse = reciprocal(rhs);
        return multiply(lhs, inverse, std::move(name));
    }

    VxmTensor maximum(VxmTensor lhs, VxmTensor rhs, std::string name = {})
    {
        return binary(VxmIrOpcode::Maximum, lhs, rhs, std::move(name));
    }

    VxmTensor negate(VxmTensor input, std::string name = {})
    {
        return unary(VxmIrOpcode::Negate, input, std::move(name));
    }

    VxmTensor exp(VxmTensor input, std::string name = {})
    {
        return unary(VxmIrOpcode::Exp, input, std::move(name));
    }

    VxmTensor reciprocal(VxmTensor input, std::string name = {})
    {
        return unary(VxmIrOpcode::Reciprocal, input, std::move(name));
    }

    VxmTensor rsqrt(VxmTensor input, float epsilon,
                    std::string name = {})
    {
        auto result = unary(VxmIrOpcode::Rsqrt, input, std::move(name));
        node(result.id).epsilon = epsilon;
        return result;
    }

    VxmTensor reduce_add(VxmTensor input, int axis, std::string name = {})
    {
        return reduction(
            VxmIrOpcode::ReduceAdd, input, axis, std::move(name));
    }

    VxmTensor reduce_max(VxmTensor input, int axis, std::string name = {})
    {
        return reduction(
            VxmIrOpcode::ReduceMax, input, axis, std::move(name));
    }

    VxmTensor broadcast(VxmTensor input,
                        std::vector<std::size_t> target_shape,
                        std::string name = {})
    {
        check_tensor(input);
        const auto& source = kernel_.type_of(input.id);
        if (!broadcast_compatible(source.shape, target_shape)) {
            throw std::invalid_argument(
                "VXM Kernel broadcast shapes are incompatible");
        }
        return append(
            VxmIrOpcode::Broadcast, {input.id},
            {std::move(target_shape), source.element_type}, std::move(name));
    }

    void output(VxmTensor tensor)
    {
        check_tensor(tensor);
        kernel_.outputs.push_back(tensor.id);
    }

    const VxmTensorType& type_of(VxmTensor tensor) const
    {
        check_tensor(tensor);
        return kernel_.type_of(tensor.id);
    }

    VxmKernel finish()
    {
        kernel_.validate();
        kernel_.eliminate_dead_code();
        kernel_.validate();
        return std::move(kernel_);
    }

private:
    VxmTensor append(VxmIrOpcode opcode, std::vector<ValueId> inputs,
                     VxmTensorType type, std::string name)
    {
        const auto id = next_value_++;
        kernel_.nodes.push_back(
            {id, opcode, std::move(inputs), std::move(type),
             std::nullopt, std::nullopt, std::nullopt, std::move(name)});
        return {id, this};
    }

    VxmIrNode& node(ValueId value)
    {
        return const_cast<VxmIrNode&>(kernel_.producer(value));
    }

    VxmTensor unary(VxmIrOpcode opcode, VxmTensor input, std::string name)
    {
        check_tensor(input);
        return append(
            opcode, {input.id}, kernel_.type_of(input.id), std::move(name));
    }

    VxmTensor binary(VxmIrOpcode opcode, VxmTensor lhs, VxmTensor rhs,
                     std::string name)
    {
        check_tensor(lhs);
        check_tensor(rhs);
        const auto& lhs_type = kernel_.type_of(lhs.id);
        const auto& rhs_type = kernel_.type_of(rhs.id);
        if (lhs_type != rhs_type) {
            throw std::invalid_argument(
                "VXM Kernel binary operands require identical tensor types; "
                "insert an explicit Broadcast or Cast");
        }
        return append(
            opcode, {lhs.id, rhs.id}, lhs_type, std::move(name));
    }

    VxmTensor reduction(VxmIrOpcode opcode, VxmTensor input, int axis,
                        std::string name)
    {
        check_tensor(input);
        const auto& input_type = kernel_.type_of(input.id);
        if (input_type.scalar()) {
            throw std::invalid_argument("VXM Kernel cannot reduce a scalar");
        }
        auto normalized = axis;
        if (normalized < 0) {
            normalized += static_cast<int>(input_type.rank());
        }
        if (normalized < 0
            || normalized >= static_cast<int>(input_type.rank())) {
            throw std::out_of_range("VXM Kernel reduction axis is invalid");
        }

        // Keep the reduced dimension as one. This makes a later explicit
        // Broadcast back to the input shape unambiguous.
        auto output_shape = input_type.shape;
        output_shape[static_cast<std::size_t>(normalized)] = 1;
        auto result = append(
            opcode, {input.id},
            {std::move(output_shape), input_type.element_type}, std::move(name));
        node(result.id).axis = static_cast<std::size_t>(normalized);
        return result;
    }

    void check_tensor(VxmTensor tensor) const
    {
        if (tensor.builder != this || tensor.id == kInvalidValue) {
            throw std::invalid_argument(
                "VXM Tensor belongs to a different KernelBuilder");
        }
    }

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

    VxmKernel kernel_{};
    ValueId next_value_{0};
};

} // namespace ftlpu::vxm::compiler
