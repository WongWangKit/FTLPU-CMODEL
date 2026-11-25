#pragma once

#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace ftlpu {

class ShiftSelect {
public:
    template <typename T>
    using TileVector = std::array<T, hw::kLanesPerTile>;

    template <typename T>
    using StreamVector = std::array<TileVector<T>, hw::kTileRows>;

    template <typename T>
    static StreamVector<T> apply(
        const StreamVector<T>& input,
        SxmShiftSource source,
        std::size_t distance = 1,
        T zero = T{})
    {
        StreamVector<T> output{};
        for (std::size_t index = 0; index < SxmInstruction::kTotalLanes; ++index) {
            const auto source_index = select_source_lane(index, source, distance);
            if (source_index >= SxmInstruction::kTotalLanes) {
                output[tile_of(index)][lane_of(index)] = zero;
                continue;
            }
            output[tile_of(index)][lane_of(index)] = input[tile_of(source_index)][lane_of(source_index)];
        }
        return output;
    }

private:
    static std::size_t select_source_lane(std::size_t lane, SxmShiftSource source, std::size_t distance)
    {
        switch (source) {
        case SxmShiftSource::Unshifted:
            return lane;
        case SxmShiftSource::NorthShifted:
            return lane + distance;
        case SxmShiftSource::SouthShifted:
            if (lane < distance) {
                return SxmInstruction::kTotalLanes;
            }
            return lane - distance;
        }
        throw std::logic_error("unsupported SXM shift source");
    }

    static constexpr std::size_t tile_of(std::size_t lane)
    {
        return lane / hw::kLanesPerTile;
    }

    static constexpr std::size_t lane_of(std::size_t lane)
    {
        return lane % hw::kLanesPerTile;
    }
};

} // namespace ftlpu
