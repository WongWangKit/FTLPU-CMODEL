#pragma once

#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace ftlpu {

class Permute320 {
public:
    static constexpr std::size_t kTotalLanes = SxmInstruction::kTotalLanes;
    using Map = SxmInstruction::PermuteMap;

    template <typename T>
    using Vector = std::array<T, hw::kLanesPerTile>;

    template <typename T>
    using Plane = std::array<Vector<T>, hw::kTileRows>;

    static Map identity_map()
    {
        Map map{};
        for (std::size_t lane = 0; lane < kTotalLanes; ++lane) {
            map[lane] = lane;
        }
        return map;
    }

    static void validate_bijection(const Map& map)
    {
        std::array<bool, kTotalLanes> seen{};
        for (const auto source : map) {
            if (source >= kTotalLanes) {
                throw std::out_of_range("SXM permutation source is outside the 320-lane plane");
            }
            if (seen[source]) {
                throw std::logic_error("SXM permutation map is not bijective");
            }
            seen[source] = true;
        }
    }

    template <typename T>
    static Plane<T> apply(const Plane<T>& input, const Map& map)
    {
        validate_bijection(map);

        Plane<T> output{};
        for (std::size_t out_index = 0; out_index < kTotalLanes; ++out_index) {
            const auto in_index = map[out_index];
            output[row_of(out_index)][lane_of(out_index)] = input[row_of(in_index)][lane_of(in_index)];
        }
        return output;
    }

private:
    static constexpr std::size_t row_of(std::size_t index)
    {
        return index / hw::kLanesPerTile;
    }

    static constexpr std::size_t lane_of(std::size_t index)
    {
        return index % hw::kLanesPerTile;
    }
};

} // namespace ftlpu
