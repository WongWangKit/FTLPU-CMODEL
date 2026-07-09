#pragma once

#include <cstdint>

namespace half_float {
struct half {
    std::uint16_t storage{0};

    half() = default;
    constexpr half(std::uint16_t bits) : storage(bits) {}
    half(float value) : storage(0) { (void)value; }
    operator float() const { return 0.0f; }
};
}  // namespace half_float

using half = float;
