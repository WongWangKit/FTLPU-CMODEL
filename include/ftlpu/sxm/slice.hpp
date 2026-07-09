#pragma once

#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/permute.hpp"
#include "ftlpu/sxm/shift.hpp"
#include "ftlpu/sxm/transpose.hpp"

namespace ftlpu {

class SxmSlice {
public:
    template <typename T>
    using Vector = std::array<T, hw::kLanesPerTile>;

    template <typename T>
    using Plane = std::array<Vector<T>, hw::kTileRows>;

    template <typename T>
    using Matrix16 = std::array<Vector<T>, hw::kLanesPerTile>;

    template <typename T>
    static Vector<T> distribute(const Vector<T>& input, const Distribute16::Map& map, T zero = T{})
    {
        return Distribute16::apply(input, map, zero);
    }

    template <typename T>
    static Matrix16<T> transpose(const Matrix16<T>& input)
    {
        return Transpose16x16::apply(input);
    }

    template <typename T>
    static Plane<T> shift_select(
        const Plane<T>& input,
        SxmShiftSource source,
        std::size_t distance = 1,
        T zero = T{})
    {
        return ShiftSelect::apply(input, source, distance, zero);
    }

    template <typename T>
    static Plane<T> permute(const Plane<T>& input, const Permute320::Map& map)
    {
        return Permute320::apply(input, map);
    }
};

} // namespace ftlpu
