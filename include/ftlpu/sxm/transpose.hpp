#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>

namespace ftlpu {

class Transpose16x16 {
public:
    template <typename T>
    using Matrix = std::array<std::array<T, hw::kLanesPerTile>, hw::kLanesPerTile>;

    template <typename T>
    static Matrix<T> apply(const Matrix<T>& input)
    {
        Matrix<T> output{};
        for (std::size_t stream = 0; stream < hw::kLanesPerTile; ++stream) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                output[stream][lane] = input[lane][stream];
            }
        }
        return output;
    }
};

} // namespace ftlpu
