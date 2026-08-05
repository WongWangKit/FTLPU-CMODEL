#pragma once

#include <bit>
#include <cstdint>

namespace ftlpu {

// IEEE 754 bfloat16 storage with round-to-nearest-even conversion.
class Bf16 {
public:
    constexpr Bf16() = default;

    static constexpr Bf16 from_bits(std::uint16_t bits) noexcept
    {
        return Bf16(bits);
    }

    static Bf16 from_float(float value) noexcept
    {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        const auto exponent = bits & 0x7f800000u;
        const auto mantissa = bits & 0x007fffffu;

        if (exponent == 0x7f800000u && mantissa != 0) {
            auto result = static_cast<std::uint16_t>(bits >> 16);
            if ((result & 0x007fu) == 0) {
                result |= 1u;
            }
            return Bf16(result);
        }

        const auto rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
        return Bf16(static_cast<std::uint16_t>((bits + rounding_bias) >> 16));
    }

    constexpr std::uint16_t bits() const noexcept
    {
        return bits_;
    }

    float to_float() const noexcept
    {
        return std::bit_cast<float>(static_cast<std::uint32_t>(bits_) << 16);
    }

private:
    explicit constexpr Bf16(std::uint16_t bits) noexcept
        : bits_(bits)
    {
    }

    std::uint16_t bits_{0};
};

} // namespace ftlpu
