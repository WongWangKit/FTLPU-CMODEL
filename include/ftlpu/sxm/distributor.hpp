#pragma once

#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace ftlpu {

class Distribute16 {
public:
    static constexpr std::size_t kZeroFill = SxmInstruction::kZeroFill;
    using Map = SxmInstruction::LaneMap;

    template <typename T>
    using Vector = std::array<T, hw::kLanesPerTile>;

    static Map identity_map()
    {
        Map map{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            map[lane] = lane;
        }
        return map;
    }

    template <typename T>
    static Vector<T> apply(const Vector<T>& input, const Map& map, T zero = T{})
    {
        Vector<T> output{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto source = map[lane];
            if (source == kZeroFill) {
                output[lane] = zero;
                continue;
            }
            if (source >= hw::kLanesPerTile) {
                throw std::out_of_range("SXM distributor lane source is outside the 16-lane vector");
            }
            output[lane] = input[source];
        }
        return output;
    }
};

} // namespace ftlpu
