#include "ftlpu/core/bf16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "smollm2_norm_residual_harness.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() try
{
    constexpr auto hidden =
        ftlpu::test::smollm2_layer::kHidden;
    auto input = std::vector<float>(hidden);
    auto second = std::vector<float>(hidden);
    for (std::size_t column = 0; column < hidden; ++column) {
        input[column] = ftlpu::Bf16::from_float(
            1.0f + static_cast<float>(column % 17) * 0.015625f)
                            .to_float();
        second[column] = ftlpu::Bf16::from_float(
            static_cast<float>(static_cast<int>(column % 11) - 5)
            * 0.03125f).to_float();
    }
    auto system = ftlpu::TspSliceSystem {};
    const auto norm =
        ftlpu::test::smollm2_norm_residual::run_rmsnorm(
            system, input, {}, "directed norm");
    const auto residual =
        ftlpu::test::smollm2_norm_residual::run_residual(
            system, norm.output, second, {}, "directed residual");
    for (const auto value : residual.output) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite norm/residual output");
        }
    }
    std::cout
        << "SmolLM2 fixed-chain RMSNorm+Residual passed: hidden="
        << hidden << ", cycles="
        << norm.cycles + residual.cycles << '\n';
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "SmolLM2 fixed-chain RMSNorm+Residual failed: "
              << error.what() << '\n';
    return 1;
}
