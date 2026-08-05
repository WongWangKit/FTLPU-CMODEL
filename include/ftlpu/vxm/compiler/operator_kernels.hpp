#pragma once

#include "ftlpu/vxm/compiler/kernel_builder.hpp"

#include <cstddef>
#include <vector>

namespace ftlpu::vxm::compiler {

inline VxmKernel make_softmax_kernel(
    std::size_t rows, std::size_t row_length)
{
    VxmKernelBuilder builder{"softmax"};
    const auto tensor_type = VxmTensorType{{rows, row_length}};
    const auto x = builder.input("x", tensor_type);
    const auto maximum = builder.reduce_max(x, -1, "maximum");
    const auto maximum_vector =
        builder.broadcast(maximum, tensor_type.shape, "maximum_vector");
    const auto shifted =
        builder.subtract(x, maximum_vector, "shifted");
    const auto exponent = builder.exp(shifted, "exponent");
    const auto denominator =
        builder.reduce_add(exponent, -1, "denominator");
    const auto inverse =
        builder.reciprocal(denominator, "inverse_denominator");
    const auto inverse_vector =
        builder.broadcast(inverse, tensor_type.shape, "inverse_vector");
    const auto output =
        builder.multiply(exponent, inverse_vector, "output");
    builder.output(output);
    return builder.finish();
}

inline VxmKernel make_rmsnorm_kernel(
    std::size_t rows, std::size_t row_length, float epsilon)
{
    VxmKernelBuilder builder{"rmsnorm"};
    const auto tensor_type = VxmTensorType{{rows, row_length}};
    const auto row_scalar_type =
        VxmTensorType{{rows, 1}};

    const auto x = builder.input("x", tensor_type);
    const auto gamma =
        builder.input("gamma", {{row_length}});
    const auto gamma_vector =
        builder.broadcast(gamma, tensor_type.shape, "gamma_vector");
    const auto square = builder.multiply(x, x, "square");
    const auto square_sum =
        builder.reduce_add(square, -1, "square_sum");

    const auto inverse_count =
        builder.constant(1.0f / static_cast<float>(row_length),
                         "inverse_count");
    const auto inverse_count_rows = builder.broadcast(
        inverse_count, row_scalar_type.shape, "inverse_count_rows");
    const auto mean_square = builder.multiply(
        square_sum, inverse_count_rows, "mean_square");

    const auto inverse_rms =
        builder.rsqrt(mean_square, epsilon, "inverse_rms");
    const auto inverse_rms_vector = builder.broadcast(
        inverse_rms, tensor_type.shape, "inverse_rms_vector");

    const auto scaled =
        builder.multiply(x, gamma_vector, "scaled");
    const auto output =
        builder.multiply(scaled, inverse_rms_vector, "output");
    builder.output(output);
    return builder.finish();
}

inline VxmKernel make_swiglu_kernel(
    std::size_t element_count)
{
    VxmKernelBuilder builder{"swiglu"};
    const auto tensor_type =
        VxmTensorType{{element_count}};
    const auto gate = builder.input("gate", tensor_type);
    const auto up = builder.input("up", tensor_type);
    const auto negated = builder.negate(gate, "negated_gate");
    const auto exponent = builder.exp(negated, "exponent");
    const auto one =
        builder.constant(1.0f, "one");
    const auto one_vector =
        builder.broadcast(one, tensor_type.shape, "one_vector");
    const auto denominator =
        builder.add(exponent, one_vector, "denominator");
    const auto sigmoid =
        builder.reciprocal(denominator, "sigmoid");
    const auto silu =
        builder.multiply(gate, sigmoid, "silu");
    const auto output =
        builder.multiply(silu, up, "output");
    builder.output(output);
    return builder.finish();
}

inline VxmKernel make_swiglu_kernel(
    std::size_t rows, std::size_t row_length)
{
    VxmKernelBuilder builder{"swiglu"};
    const auto tensor_type = VxmTensorType{{rows, row_length}};
    const auto gate = builder.input("gate", tensor_type);
    const auto up = builder.input("up", tensor_type);
    const auto negated = builder.negate(gate, "negated_gate");
    const auto exponent = builder.exp(negated, "exponent");
    const auto one = builder.constant(1.0f, "one");
    const auto one_vector =
        builder.broadcast(one, tensor_type.shape, "one_vector");
    const auto denominator =
        builder.add(exponent, one_vector, "denominator");
    const auto sigmoid =
        builder.reciprocal(denominator, "sigmoid");
    const auto silu = builder.multiply(gate, sigmoid, "silu");
    const auto output = builder.multiply(silu, up, "output");
    builder.output(output);
    return builder.finish();
}

} // namespace ftlpu::vxm::compiler
