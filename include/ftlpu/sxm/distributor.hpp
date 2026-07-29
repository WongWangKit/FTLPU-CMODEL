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
    using TileVector = std::array<T, hw::kLanesPerTile>;

    static Map identity_map()
    {
        Map map{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            map[lane] = lane;
        }
        return map;
    }

    template <typename T>
    static TileVector<T> apply(const TileVector<T>& input, const Map& map, T zero = T{})
    {
        TileVector<T> output{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto source = map[lane];
            if (source == kZeroFill) {
                output[lane] = zero;
                continue;
            }
            if (source >= hw::kLanesPerTile) {
                throw std::out_of_range("SXM distributor lane source is outside the configured vector");
            }
            output[lane] = input[source];
        }
        return output;
    }
};

} // namespace ftlpu
