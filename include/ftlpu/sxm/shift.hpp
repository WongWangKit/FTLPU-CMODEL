#pragma once

#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace ftlpu {

class ShiftSelect {
public:
    template <typename T>
    using Vector = std::array<T, hw::kLanesPerTile>;

    template <typename T>
    using Plane = std::array<Vector<T>, hw::kTileRows>;

    template <typename T>
    static Plane<T> apply(
        const Plane<T>& input,
        SxmShiftSource source,
        std::size_t distance = 1,
        T zero = T{})
    {
        Plane<T> output{};
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            const auto source_row = select_source_row(row, source, distance);
            if (source_row >= hw::kTileRows) {
                output[row].fill(zero);
                continue;
            }
            output[row] = input[source_row];
        }
        return output;
    }

private:
    static std::size_t select_source_row(std::size_t row, SxmShiftSource source, std::size_t distance)
    {
        switch (source) {
        case SxmShiftSource::Unshifted:
            return row;
        case SxmShiftSource::NorthShifted:
            return row + distance;
        case SxmShiftSource::SouthShifted:
            if (row < distance) {
                return hw::kTileRows;
            }
            return row - distance;
        }
        throw std::logic_error("unsupported SXM shift source");
    }
};

} // namespace ftlpu
