#pragma once

#include <cstddef>

namespace ftlpu {

enum class Hemisphere : std::size_t {
    East = 0,
    West = 1,
};

constexpr std::size_t hemisphere_index(Hemisphere hemisphere)
{
    return static_cast<std::size_t>(hemisphere);
}

constexpr const char* hemisphere_name(Hemisphere hemisphere)
{
    return hemisphere == Hemisphere::East ? "east" : "west";
}

constexpr const char* hemisphere_short_name(Hemisphere hemisphere)
{
    return hemisphere == Hemisphere::East ? "E" : "W";
}

} // namespace ftlpu
