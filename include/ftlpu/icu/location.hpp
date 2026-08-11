#pragma once

#include "ftlpu/core/hemisphere.hpp"

#include <cstddef>
#include <cstdint>

namespace ftlpu {

enum class IcuLocationKind : std::uint8_t {
    Mem,
    Vxm,
    MxmLoad,
    MxmCompute,
    MxmDequant,
    Sxm,
};

struct IcuLocation {
    IcuLocationKind kind{IcuLocationKind::Mem};
    std::size_t unit{0};
    std::size_t index{0};

    static constexpr IcuLocation Mem(std::size_t slice) noexcept
    {
        return Mem(Hemisphere::East, slice);
    }

    static constexpr IcuLocation Mem(
        Hemisphere hemisphere, std::size_t slice) noexcept
    {
        return {IcuLocationKind::Mem, hemisphere_index(hemisphere), slice};
    }

    static constexpr IcuLocation Vxm(std::size_t alu) noexcept
    {
        return {IcuLocationKind::Vxm, 0, alu};
    }

    static constexpr IcuLocation MxmLoad(std::size_t mxm) noexcept
    {
        return {IcuLocationKind::MxmLoad, mxm, 0};
    }

    static constexpr IcuLocation MxmCompute(std::size_t mxm) noexcept
    {
        return {IcuLocationKind::MxmCompute, mxm, 0};
    }

    static constexpr IcuLocation MxmDequant(std::size_t mxm) noexcept
    {
        return {IcuLocationKind::MxmDequant, mxm, 0};
    }

    static constexpr IcuLocation Sxm(
        Hemisphere hemisphere, std::size_t port = 0) noexcept
    {
        return {IcuLocationKind::Sxm, hemisphere_index(hemisphere), port};
    }

    friend bool operator==(const IcuLocation&, const IcuLocation&) = default;
};

} // namespace ftlpu
